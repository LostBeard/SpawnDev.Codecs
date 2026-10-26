// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Snappy compressor on the accelerator, byte-identical with the Rust snap crate (python-snappy / cramjam's backend) in
// both container formats:
//   Raw:    the Snappy block format (format_description.txt): the uncompressed length as a varint, then the elements
//           (literals, copies). snap cuts the input into 64 KB blocks and compresses each with a fresh hash table, so no
//           copy ever crosses a block: every block is independent.
//   Framed: the Snappy framing format (framing_format.txt): the stream identifier, then one chunk per 64 KB block, each
//           with the masked CRC-32C of its input; a block whose raw stream (its own varint + elements) is not below
//           7/8 of its size is stored uncompressed (snap's rule).
// An empty input is the single byte 0 (raw) or nothing at all (framed), as snap writes it.
//
// One thread per block walks snap's Block::compress as a resumable state machine: a launch stops once it has advanced
// Budget input positions and the next launch continues from the saved state (search position, skip counter, next hash,
// pending candidate), so no launch grows with the block. The hash table holds 256 .. 16384 entries (the smallest power
// of two >= the block, as snap sizes it); blocks run in waves of waveBlocks. After a wave, one thread lays the wave's
// blocks out after the previous ones (running output position in device memory), one thread per slot byte copies them
// (compressed bytes from the wave's slots, stored chunks straight from the input), and one thread per block writes its
// chunk header and CRC-32C (framed). Nothing but the final length crosses to the host.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression.Snappy;

/// <summary>The two Snappy container formats.</summary>
public enum SnappyFormat
{
    /// <summary>The raw block format: varint length + elements (python-snappy <c>snappy.compress</c>, Parquet, Kafka).</summary>
    Raw = 0,
    /// <summary>The framing format: stream identifier + checksummed chunks (<c>snappy.stream_compress</c>, <c>.sz</c> files).</summary>
    Framed = 1,
}

/// <summary>Snappy compressor (raw or framed) on the accelerator, byte-identical with the Rust snap crate.</summary>
public sealed class SnappyCompressorGpu : IDisposable
{
    /// <summary>Block size (snap's MAX_BLOCK_SIZE; also the framing format's chunk limit).</summary>
    public const int BlockSize = 64 * 1024;
    /// <summary>Default blocks compressed at once (each holds a 16384-entry hash table and a ~75 KB slot).</summary>
    public const int DefaultWaveBlocks = 512;
    /// <summary>Input positions one block advances per launch.</summary>
    public const int Budget = 8192;

    internal const int MaxTable = 1 << 14;
    private const int InputMargin = 15, MinNonLiteralBlock = 1 + 1 + InputMargin;
    private const uint HashMul = 0x1E35A7BDu;

    // Per-block state ints.
    internal const int SPhase = 0, SS = 1, SNextEmit = 2, SOp = 3, SNextHash = 4, SSkip = 5, SSNext = 6, SCand = 7, SResult = 8;
    internal const int StateInts = 9;
    internal const int PhInit = 0, PhTop = 1, PhSearch = 2, PhMatch = 3, PhLast = 4, PhDone = 5;

    // prm ints: input length, blocks, format, running output position, unfinished blocks, final length.
    internal const int PLength = 0, PBlocks = 1, PFormat = 2, PPos = 3, PUnfinished = 4, PTotal = 5;
    internal const int PrmInts = 8;

    /// <summary>Bytes of a wave slot: the largest block body snap can write (32 + n + n / 6 for n = 64 KB) + slack.</summary>
    private const int SlotBytes = 32 + BlockSize + BlockSize / 6 + 16;

    private readonly Accelerator _accelerator;
    private readonly int _waveBlocks;
    private readonly LaunchPacer _pacer;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _clearTables;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _step;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _copy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, int> _chunkHeaders;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _header;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm, _state, _table, _offsets;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _crcTable;
    // _empty: the input view of an empty input (WebGPU refuses one buffer bound to two read_write slots).
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _slots, _empty;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>
    /// Compiles the kernels and allocates one wave's working memory (~141 KB per block of <paramref name="waveBlocks"/>:
    /// ~72 MB at the default 512).
    /// </summary>
    public SnappyCompressorGpu(Accelerator accelerator, int waveBlocks = DefaultWaveBlocks)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (waveBlocks < 1 || waveBlocks > 16384) throw new ArgumentOutOfRangeException(nameof(waveBlocks));
        _accelerator = accelerator;
        _waveBlocks = waveBlocks;
        _pacer = new LaunchPacer(accelerator);
        _clearTables = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(ClearKernel);
        _step = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(StepKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(LayoutKernel);
        _copy = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(CopyKernel);
        _chunkHeaders = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>, int>(ChunkHeaderKernel);
        _header = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(HeaderKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _state = accelerator.Allocate1D<int>((long)_waveBlocks * StateInts);
        _table = accelerator.Allocate1D<int>((long)_waveBlocks * MaxTable);
        _offsets = accelerator.Allocate1D<int>(_waveBlocks);
        _slots = accelerator.Allocate1D<byte>((long)_waveBlocks * SlotBytes);
        _empty = accelerator.Allocate1D<byte>(1);
        _crcTable = accelerator.Allocate1D(Crc32Gpu.BuildTable(Crc32Gpu.CastagnoliPolynomial));
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    /// <summary>Largest stream <paramref name="length"/> input bytes can produce in <paramref name="format"/>.</summary>
    public static long GetMaxCompressedSize(long length, SnappyFormat format = SnappyFormat.Raw)
    {
        if (format == SnappyFormat.Raw) return 32 + length + length / 6;
        long blocks = (length + BlockSize - 1) / BlockSize;
        return 10 + blocks * (8 + BlockSize);
    }

    // ------------------------------------------------------------------
    // Overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Compresses the first <paramref name="length"/> bytes of <paramref name="input"/> into a Snappy stream at the start
    /// of <paramref name="output"/> (at least <see cref="GetMaxCompressedSize"/> bytes); returns the stream's length.
    /// </summary>
    public async Task<int> CompressAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, SnappyFormat format = SnappyFormat.Raw)
    {
        if (length < 0 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length}.");
        if (format != SnappyFormat.Raw && format != SnappyFormat.Framed) throw new ArgumentOutOfRangeException(nameof(format));
        long cap = GetMaxCompressedSize(length, format);
        if (cap > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "The stream must stay under 2 GiB per call.");
        if (output.Length < cap) throw new ArgumentException($"The output holds {output.Length} bytes; {cap} are needed (GetMaxCompressedSize).", nameof(output));

        int blocks = (int)((length + (long)BlockSize - 1) / BlockSize);
        var prm = new int[PrmInts];
        prm[PLength] = length;
        prm[PBlocks] = blocks;
        prm[PFormat] = (int)format;
        _prm.View.CopyFromCPU(prm);

        // The kernels read the input's length from this view (the copy kernel bounds the last block with it).
        var data = length > 0 ? input.SubView(0, length) : _empty.View;
        _header(1, output, _prm.View);
        for (int waveBase = 0; waveBase < blocks; waveBase += _waveBlocks)
        {
            int count = Math.Min(_waveBlocks, blocks - waveBase);
            _clearTables(count * MaxTable, _table.View, _state.View);
            int steps = BlockSize / Budget + 2;
            for (int s = 0; s < steps; s++)
            {
                _step(count, data, _slots.View, _table.View, _state.View, _prm.View, waveBase);
                await _pacer.LaunchedAsync();
            }
            _layout(1, _state.View, _offsets.View, _prm.View, count, waveBase);
            _copy(count * SlotBytes, data, _slots.View, output, _state.View, _offsets.View, _prm.View, waveBase);
            if (format == SnappyFormat.Framed) _chunkHeaders(count, data, output, _state.View, _offsets.View, _prm.View, _crcTable.View, waveBase);
            await _pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
        var result = await _prm.View.SubView(PPos, 3).CopyToHostAsync();
        if (result[PUnfinished - PPos] != 0) throw new InvalidOperationException($"Snappy compressor: {result[PUnfinished - PPos]} blocks did not finish in their launch budget (bug).");
        return result[PTotal - PPos];
    }

    /// <summary>Compresses <paramref name="data"/> into a new Snappy stream.</summary>
    public async Task<byte[]> CompressAsync(ReadOnlyMemory<byte> data, SnappyFormat format = SnappyFormat.Raw)
    {
        var src = _stageIn.Ensure(data.Length);
        var dst = _stageOut.Ensure(GetMaxCompressedSize(data.Length, format));
        if (data.Length > 0) DeviceTransfer.Upload(src, data.Span);
        int n = await CompressAsync(src, data.Length, dst, format);
        return await DeviceTransfer.DownloadAsync(dst, n);
    }

    /// <summary>Compresses the bytes of <paramref name="data"/> into a new Snappy stream (the bytes never enter the .NET heap); the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> CompressAsync(TypedArray data, SnappyFormat format = SnappyFormat.Raw)
    {
        ArgumentNullException.ThrowIfNull(data);
        long byteLength = data.ByteLength;
        if (byteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(data), "At most 2^31 - 1 bytes per call.");
        int length = (int)byteLength;
        var src = _stageIn.Ensure(length);
        var dst = _stageOut.Ensure(GetMaxCompressedSize(length, format));
        if (length > 0) DeviceTransfer.Upload(src, data);
        int n = await CompressAsync(src, length, dst, format);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, n);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    private static uint Read32(ArrayView<byte> d, int at) =>
        (uint)d[at] | ((uint)d[at + 1] << 8) | ((uint)d[at + 2] << 16) | ((uint)d[at + 3] << 24);

    private static int Hash(ArrayView<byte> d, int at, int shift) => (int)((Read32(d, at) * HashMul) >> shift);

    /// <summary>Bytes of the varint of <paramref name="v"/> (v &gt;= 0).</summary>
    internal static int VarintLength(int v)
    {
        int n = 1;
        uint x = (uint)v;
        while (x >= 0x80)
        {
            x >>= 7;
            n++;
        }
        return n;
    }

    /// <summary>Writes the varint of <paramref name="v"/> at <paramref name="at"/>; returns its length.</summary>
    private static int WriteVarint(ArrayView<byte> o, int at, int v)
    {
        uint x = (uint)v;
        int p = at;
        while (x >= 0x80)
        {
            o[p] = (byte)(x | 0x80);
            p++;
            x >>= 7;
        }
        o[p] = (byte)x;
        return p + 1 - at;
    }

    private static int BlockLength(int length, int block)
    {
        int size = length - block * BlockSize;
        if (size > BlockSize) size = BlockSize;
        return size;
    }

    /// <summary>Zeroes the wave's hash tables and resets every block's state (thread per table entry).</summary>
    private static void ClearKernel(Index1D i, ArrayView<int> table, ArrayView<int> state)
    {
        table[i] = 0;
        int e = i.X;
        if ((e & (MaxTable - 1)) == 0)
        {
            int at = (e >> 14) * StateInts;
            for (int k = 0; k < StateInts; k++) state[at + k] = 0;
        }
    }

    /// <summary>
    /// One bounded launch of block waveBase + idx: snap's Block::compress as a state machine (a block under 17 bytes is
    /// one literal). Positions are relative to the block start; the block's elements go to its slot.
    /// </summary>
    private static void StepKernel(Index1D idx, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> table, ArrayView<int> state,
        ArrayView<int> prm, int waveBase)
    {
        int b = idx.X;
        int sa = b * StateInts;
        int phase = state[sa + SPhase];
        if (phase == PhDone) return;
        int block = waveBase + b;
        int start = block * BlockSize;
        int n = BlockLength(prm[PLength], block);
        int tb = b * MaxTable;
        int so = b * SlotBytes;
        int sLimit = n - InputMargin;
        // snap's table: the smallest power of two >= the block, 256 .. 16384 entries.
        int shift = 24;
        int tableSize = 256;
        while ((tableSize < MaxTable) & (tableSize < n))
        {
            shift--;
            tableSize *= 2;
        }

        int s = state[sa + SS];
        int nextEmit = state[sa + SNextEmit];
        int op = state[sa + SOp];
        int nextHash = state[sa + SNextHash];
        int skip = state[sa + SSkip];
        int sNext = state[sa + SSNext];
        int cand = state[sa + SCand];

        if (phase == PhInit)
        {
            nextEmit = 0;
            op = 0;
            phase = PhLast;
            if (n >= MinNonLiteralBlock)
            {
                s = 1;
                nextHash = Hash(data, start + 1, shift);
                phase = PhTop;
            }
        }

        int launchStart = s;
        bool running = true;
        while (running)
        {
            if (phase == PhTop)
            {
                skip = 32;
                sNext = s;
                phase = PhSearch;
            }
            else if (phase == PhSearch)
            {
                bool searching = true;
                while (searching)
                {
                    s = sNext;
                    int between = skip >> 5;
                    sNext = s + between;
                    skip += between;
                    if (sNext > sLimit)
                    {
                        phase = PhLast;
                        searching = false;
                    }
                    else
                    {
                        cand = table[tb + nextHash];
                        table[tb + nextHash] = s;
                        nextHash = Hash(data, start + sNext, shift);
                        if (Read32(data, start + s) == Read32(data, start + cand))
                        {
                            phase = PhMatch;
                            searching = false;
                        }
                        else if (s - launchStart >= Budget)
                        {
                            searching = false;
                            running = false;
                        }
                    }
                }
                if (phase == PhMatch) op = EmitLiteral(data, start, nextEmit, s, slots, so, op);
            }
            else if (phase == PhMatch)
            {
                // Extend the match, emit it, then test the position after it.
                int baseS = s;
                s += 4;
                int c = cand + 4;
                bool extending = s < n;
                while (extending)
                {
                    if (data[start + s] == data[start + c])
                    {
                        s++;
                        c++;
                        extending = s < n;
                    }
                    else extending = false;
                }
                op = EmitCopy(baseS - cand, s - baseS, slots, so, op);
                nextEmit = s;
                if (s >= sLimit) phase = PhLast;
                else
                {
                    table[tb + Hash(data, start + s - 1, shift)] = s - 1;
                    int h = Hash(data, start + s, shift);
                    cand = table[tb + h];
                    table[tb + h] = s;
                    if (Read32(data, start + s) != Read32(data, start + cand))
                    {
                        nextHash = Hash(data, start + s + 1, shift);
                        s++;
                        phase = PhTop;
                    }
                    if (s - launchStart >= Budget) running = false;
                }
            }
            else if (phase == PhLast)
            {
                if (nextEmit < n) op = EmitLiteral(data, start, nextEmit, n, slots, so, op);
                state[sa + SResult] = op;
                phase = PhDone;
                running = false;
            }
            else running = false;
        }

        state[sa + SPhase] = phase;
        state[sa + SS] = s;
        state[sa + SNextEmit] = nextEmit;
        state[sa + SOp] = op;
        state[sa + SNextHash] = nextHash;
        state[sa + SSkip] = skip;
        state[sa + SSNext] = sNext;
        state[sa + SCand] = cand;
    }

    /// <summary>A literal of block bytes [from, to) at slot position op (snap's length forms: inline, 1 or 2 bytes); returns the new op.</summary>
    private static int EmitLiteral(ArrayView<byte> data, int start, int from, int to, ArrayView<byte> slots, int so, int op)
    {
        int len = to - from;
        int m = len - 1;
        int p = so + op;
        if (m <= 59)
        {
            slots[p] = (byte)(m << 2);
            p++;
        }
        else if (m < 256)
        {
            slots[p] = 60 << 2;
            slots[p + 1] = (byte)m;
            p += 2;
        }
        else
        {
            slots[p] = 61 << 2;
            slots[p + 1] = (byte)m;
            slots[p + 2] = (byte)(m >> 8);
            p += 3;
        }
        for (int k = 0; k < len; k++) slots[p + k] = data[start + from + k];
        return p + len - so;
    }

    /// <summary>snap's emit_copy: copy-2 elements of 64 while 68 or more remain, one of 60 above 64, then copy-1 when it fits; returns the new op.</summary>
    private static int EmitCopy(int offset, int len, ArrayView<byte> slots, int so, int op)
    {
        int p = so + op;
        while (len >= 68)
        {
            slots[p] = (63 << 2) | 2;
            slots[p + 1] = (byte)offset;
            slots[p + 2] = (byte)(offset >> 8);
            p += 3;
            len -= 64;
        }
        if (len > 64)
        {
            slots[p] = (59 << 2) | 2;
            slots[p + 1] = (byte)offset;
            slots[p + 2] = (byte)(offset >> 8);
            p += 3;
            len -= 60;
        }
        if ((len <= 11) & (offset <= 2047))
        {
            slots[p] = (byte)(((offset >> 8) << 5) | ((len - 4) << 2) | 1);
            slots[p + 1] = (byte)offset;
            p += 2;
        }
        else
        {
            slots[p] = (byte)(((len - 1) << 2) | 2);
            slots[p + 1] = (byte)offset;
            slots[p + 2] = (byte)(offset >> 8);
            p += 3;
        }
        return p - so;
    }

    /// <summary>Framed: whether block b is stored uncompressed (its raw stream, varint included, is not below 7/8 of it).</summary>
    private static bool IsStored(ArrayView<int> state, int b, int n) =>
        VarintLength(n) + state[b * StateInts + SResult] >= n - n / 8;

    /// <summary>Bytes block b occupies in the output.</summary>
    private static int ChunkBytes(ArrayView<int> state, int b, int n, int format)
    {
        int body = state[b * StateInts + SResult];
        int bytes = body;
        if (format == (int)SnappyFormat.Framed)
        {
            bytes = 8 + VarintLength(n) + body;
            if (IsStored(state, b, n)) bytes = 8 + n;
        }
        return bytes;
    }

    /// <summary>One thread: the raw stream's length varint, or the framed stream identifier (none for empty input); starts the running position.</summary>
    private static void HeaderKernel(Index1D idx, ArrayView<byte> output, ArrayView<int> prm)
    {
        int length = prm[PLength];
        int p = 0;
        if (prm[PFormat] == (int)SnappyFormat.Raw) p = WriteVarint(output, 0, length);
        else if (length > 0)
        {
            output[0] = 0xFF;
            output[1] = 0x06;
            output[2] = 0x00;
            output[3] = 0x00;
            output[4] = (byte)'s';
            output[5] = (byte)'N';
            output[6] = (byte)'a';
            output[7] = (byte)'P';
            output[8] = (byte)'p';
            output[9] = (byte)'Y';
            p = 10;
        }
        prm[PPos] = p;
        prm[PTotal] = p;
    }

    /// <summary>One thread: output offsets of the wave's blocks after the running position; counts unfinished blocks (a bug guard).</summary>
    private static void LayoutKernel(Index1D idx, ArrayView<int> state, ArrayView<int> offsets, ArrayView<int> prm, int count, int waveBase)
    {
        int pos = prm[PPos];
        int length = prm[PLength];
        int format = prm[PFormat];
        int unfinished = 0;
        for (int b = 0; b < count; b++)
        {
            offsets[b] = pos;
            if (state[b * StateInts + SPhase] != PhDone) unfinished++;
            pos += ChunkBytes(state, b, BlockLength(length, waveBase + b), format);
        }
        prm[PUnfinished] += unfinished;
        prm[PPos] = pos;
        prm[PTotal] = pos;
    }

    /// <summary>
    /// One thread per slot byte of the wave: raw, the block's elements; framed, after the 8-byte chunk header, the varint
    /// and elements of a compressed chunk or the input bytes of a stored one.
    /// </summary>
    private static void CopyKernel(Index1D i, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<byte> output, ArrayView<int> state,
        ArrayView<int> offsets, ArrayView<int> prm, int waveBase)
    {
        int b = i.X / SlotBytes;
        int k = i.X - b * SlotBytes;
        int block = waveBase + b;
        // data is exactly the input (SubView of the input length), so its length bounds the last block.
        int n = BlockLength(data.IntLength, block);
        int body = state[b * StateInts + SResult];
        int at = offsets[b];
        if (prm[PFormat] == (int)SnappyFormat.Raw)
        {
            if (k < body) output[at + k] = slots[b * SlotBytes + k];
        }
        else if (IsStored(state, b, n))
        {
            if (k < n) output[at + 8 + k] = data[block * BlockSize + k];
        }
        else if (k < body) output[at + 8 + VarintLength(n) + k] = slots[b * SlotBytes + k];
    }

    /// <summary>One thread per block of the wave (framed): chunk type, length, masked CRC-32C of the block's input, and a compressed chunk's varint.</summary>
    private static void ChunkHeaderKernel(Index1D idx, ArrayView<byte> data, ArrayView<byte> output, ArrayView<int> state, ArrayView<int> offsets,
        ArrayView<int> prm, ArrayView<uint> crcTable, int waveBase)
    {
        int b = idx.X;
        int block = waveBase + b;
        int n = BlockLength(data.IntLength, block);
        int at = offsets[b];
        bool stored = IsStored(state, b, n);
        int chunkLen = 4 + n;
        if (!stored) chunkLen = 4 + VarintLength(n) + state[b * StateInts + SResult];
        output[at] = (byte)(stored ? 1 : 0);
        output[at + 1] = (byte)chunkLen;
        output[at + 2] = (byte)(chunkLen >> 8);
        output[at + 3] = (byte)(chunkLen >> 16);
        uint crc = Crc32Gpu.Finish(Crc32Gpu.Update(Crc32Gpu.Seed, data, block * BlockSize, n, crcTable));
        uint masked = ((crc >> 15) | (crc << 17)) + 0xA282EAD8u;
        output[at + 4] = (byte)masked;
        output[at + 5] = (byte)(masked >> 8);
        output[at + 6] = (byte)(masked >> 16);
        output[at + 7] = (byte)(masked >> 24);
        if (!stored) WriteVarint(output, at + 8, n);
    }

    public void Dispose()
    {
        _prm.Dispose();
        _state.Dispose();
        _table.Dispose();
        _offsets.Dispose();
        _slots.Dispose();
        _empty.Dispose();
        _crcTable.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
