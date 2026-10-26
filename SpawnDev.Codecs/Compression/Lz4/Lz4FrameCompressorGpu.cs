// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// LZ4 frame compressor on the accelerator, byte-identical with liblz4 1.9.4 (LZ4F_compressFrame, compression level 0
// = acceleration 1, or a negative level's acceleration) for frames of INDEPENDENT 64 KB blocks (block_linked=False):
// the frame header (optional content size), each block as liblz4's fast compressor writes it or stored raw when it does
// not shrink, optional block checksums, the end mark, an optional content checksum. Any LZ4 decoder reads it.
//
// Why exactly this configuration: a 64 KB block takes liblz4's byU16 path (input < LZ4_64Klimit), and before every
// independent block LZ4_prepareTable clears the hash table (currentOffset + size >= 0xFFFF after a full block, or the
// table is fresh), so each block compresses exactly as a fresh LZ4_compress_fast_extState with dstCapacity = size - 1
// (limitedOutput: liblz4 gives up as soon as its conservative bound says the block would not shrink, and the frame
// then stores it raw). Every block is independent, so every block is one GPU thread.
//
// One thread per block walks LZ4_compress_generic_validated (byU16, noDict, limitedOutput) as a resumable state
// machine: a launch stops once it has advanced Budget input positions and the next launch continues from the saved
// state (search position, step, skip counter, pending token), so no launch grows with the block. Blocks run in waves of
// waveBlocks (each needs an 8192-entry hash table); after a wave, one thread lays the wave's blocks out after the
// previous ones (running output position in device memory), then one thread per output byte copies them (compressed
// bytes from the wave's slots, raw blocks straight from the input) and one thread per block writes its header and
// checksum. The content checksum (XXH32 of the whole input, serial by definition) runs as a chain of bounded launches.
// Nothing but the final length crosses to the host.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression.Lz4;

/// <summary>Options of <see cref="Lz4FrameCompressorGpu"/> (the LZ4 frame descriptor flags we write).</summary>
public sealed record Lz4FrameOptions
{
    /// <summary>Write the uncompressed size into the frame header (liblz4 / python-lz4 default: true; never for empty input).</summary>
    public bool StoreContentSize { get; init; } = true;
    /// <summary>XXH32 of each stored block after it.</summary>
    public bool BlockChecksum { get; init; }
    /// <summary>XXH32 of the whole input after the end mark (a serial pass over the input).</summary>
    public bool ContentChecksum { get; init; }
    /// <summary>liblz4 acceleration (1 = compression level 0; level -n = acceleration n + 1). 1..65537.</summary>
    public int Acceleration { get; init; } = 1;
}

/// <summary>LZ4 frame compressor (independent 64 KB blocks) on the accelerator, byte-identical with liblz4.</summary>
public sealed class Lz4FrameCompressorGpu : IDisposable
{
    /// <summary>Block size (the frame's max block size 64 KB, block size ID 4).</summary>
    public const int BlockSize = 64 * 1024;
    /// <summary>Default blocks compressed at once (each holds an 8192-entry hash table and a 64 KB slot).</summary>
    public const int DefaultWaveBlocks = 512;
    /// <summary>Input positions one block advances per launch.</summary>
    public const int Budget = 8192;
    /// <summary>Content checksum stripes (16 bytes) per launch.</summary>
    public const int ChecksumStripes = 65536;

    internal const int HashLog = 13, HashSize = 1 << HashLog;
    internal const int MinMatch = 4, LastLiterals = 5, MfLimit = 12, MinLength = MfLimit + 1, SkipTrigger = 6;
    private const int Magic = 0x184D2204;

    // Per-block state ints.
    internal const int SPhase = 0, SIp = 1, SAnchor = 2, SOp = 3, SForwardH = 4, SForwardIp = 5, SStep = 6, SSearchNb = 7,
        SMatch = 8, STokenPos = 9, SToken = 10, SResult = 11;
    internal const int StateInts = 12;
    internal const int PhInit = 0, PhTop = 1, PhSearch = 2, PhMatch = 3, PhLast = 4, PhDone = 5;

    // prm ints: input length, blocks, flags, acceleration, running output position, unfinished blocks, header length,
    // final length, then the XXH32 state of the content checksum.
    internal const int PLength = 0, PBlocks = 1, PFlags = 2, PAccel = 3, PPos = 4, PUnfinished = 5, PHeader = 6, PTotal = 7, PXxh = 8;
    internal const int PrmInts = 16;
    internal const int FlagContentSize = 1, FlagBlockChecksum = 2, FlagContentChecksum = 4;

    /// <summary>Bytes of a wave slot: the block (at most its size) + slack.</summary>
    private const int SlotBytes = BlockSize + 16;

    private readonly Accelerator _accelerator;
    private readonly int _waveBlocks;
    private readonly LaunchPacer _pacer;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>> _clearTables;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _step;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _copy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _blockHeaders;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _header;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>> _tail;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, int> _xxhStripes;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm, _state, _table, _offsets;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _slots;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>
    /// Compiles the kernels and allocates one wave's working memory (~82 KB per block of <paramref name="waveBlocks"/>:
    /// ~42 MB at the default 512).
    /// </summary>
    public Lz4FrameCompressorGpu(Accelerator accelerator, int waveBlocks = DefaultWaveBlocks)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (waveBlocks < 1 || waveBlocks > 32768) throw new ArgumentOutOfRangeException(nameof(waveBlocks));
        _accelerator = accelerator;
        _waveBlocks = waveBlocks;
        _pacer = new LaunchPacer(accelerator);
        _clearTables = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(ClearKernel);
        _step = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(StepKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(LayoutKernel);
        _copy = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(CopyKernel);
        _blockHeaders = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(BlockHeaderKernel);
        _header = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(HeaderKernel);
        _tail = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>>(TailKernel);
        _xxhStripes = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, int>(ContentStripesKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _state = accelerator.Allocate1D<int>((long)_waveBlocks * StateInts);
        _table = accelerator.Allocate1D<int>((long)_waveBlocks * HashSize);
        _offsets = accelerator.Allocate1D<int>(_waveBlocks);
        _slots = accelerator.Allocate1D<byte>((long)_waveBlocks * SlotBytes);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    /// <summary>Largest frame <paramref name="length"/> input bytes can produce with any options.</summary>
    public static long GetMaxCompressedSize(long length)
    {
        long blocks = (length + BlockSize - 1) / BlockSize;
        return 4 + 2 + 8 + 1 + blocks * (4 + BlockSize + 4) + 4 + 4;
    }

    // ------------------------------------------------------------------
    // Overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Compresses the first <paramref name="length"/> bytes of <paramref name="input"/> into an LZ4 frame at the start of
    /// <paramref name="output"/> (at least <see cref="GetMaxCompressedSize"/> bytes); returns the frame's length.
    /// </summary>
    public async Task<int> CompressAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, Lz4FrameOptions? options = null)
    {
        options ??= new Lz4FrameOptions();
        if (length < 0 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length}.");
        if (options.Acceleration < 1 || options.Acceleration > 65537) throw new ArgumentOutOfRangeException(nameof(options), "Acceleration must be 1..65537.");
        long cap = GetMaxCompressedSize(length);
        if (cap > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "The frame must stay under 2 GiB per call.");
        if (output.Length < cap) throw new ArgumentException($"The output holds {output.Length} bytes; {cap} are needed (GetMaxCompressedSize).", nameof(output));

        int blocks = (int)((length + (long)BlockSize - 1) / BlockSize);
        int flags = (options.StoreContentSize && length > 0 ? FlagContentSize : 0)
                  | (options.BlockChecksum ? FlagBlockChecksum : 0)
                  | (options.ContentChecksum ? FlagContentChecksum : 0);
        var prm = new int[PrmInts];
        prm[PLength] = length;
        prm[PBlocks] = blocks;
        prm[PFlags] = flags;
        prm[PAccel] = options.Acceleration;
        _prm.View.CopyFromCPU(prm);

        // The kernels read the input's length from this view (the copy kernel bounds the last block with it).
        var data = length > 0 ? input.SubView(0, length) : output.SubView(0, 1);
        _header(1, output, _prm.View);
        for (int waveBase = 0; waveBase < blocks; waveBase += _waveBlocks)
        {
            int count = Math.Min(_waveBlocks, blocks - waveBase);
            _clearTables(count * HashSize, _table.View, _state.View);
            int steps = BlockSize / Budget + 2;
            for (int s = 0; s < steps; s++)
            {
                _step(count, data, _slots.View, _table.View, _state.View, _prm.View, waveBase);
                await _pacer.LaunchedAsync();
            }
            _layout(1, _state.View, _offsets.View, _prm.View, count, waveBase);
            _copy(count * SlotBytes, data, _slots.View, output, _state.View, _offsets.View, waveBase);
            _blockHeaders(count, output, _state.View, _offsets.View, _prm.View, waveBase);
            await _pacer.LaunchedAsync();
        }
        if (options.ContentChecksum)
        {
            int stripes = length / 16;
            for (int done = 0; done < stripes; done += ChecksumStripes)
            {
                _xxhStripes(1, data, _prm.View, done);
                await _pacer.LaunchedAsync();
            }
        }
        _tail(1, data, output, _prm.View);
        await _accelerator.SynchronizeAsync();
        var result = await _prm.View.SubView(PUnfinished, 3).CopyToHostAsync();
        if (result[0] != 0) throw new InvalidOperationException($"LZ4 compressor: {result[0]} blocks did not finish in their launch budget (bug).");
        return result[2];
    }

    /// <summary>Compresses <paramref name="data"/> into a new LZ4 frame.</summary>
    public async Task<byte[]> CompressAsync(ReadOnlyMemory<byte> data, Lz4FrameOptions? options = null)
    {
        var src = _stageIn.Ensure(data.Length);
        var dst = _stageOut.Ensure(GetMaxCompressedSize(data.Length));
        if (data.Length > 0) DeviceTransfer.Upload(src, data.Span);
        int n = await CompressAsync(src, data.Length, dst, options);
        return await DeviceTransfer.DownloadAsync(dst, n);
    }

    /// <summary>Compresses the bytes of <paramref name="data"/> into a new LZ4 frame (the bytes never enter the .NET heap); the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> CompressAsync(TypedArray data, Lz4FrameOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        long byteLength = data.ByteLength;
        if (byteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(data), "At most 2^31 - 1 bytes per call.");
        int length = (int)byteLength;
        var src = _stageIn.Ensure(length);
        var dst = _stageOut.Ensure(GetMaxCompressedSize(length));
        if (length > 0) DeviceTransfer.Upload(src, data);
        int n = await CompressAsync(src, length, dst, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, n);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    private static uint Read32(ArrayView<byte> d, int at) =>
        (uint)d[at] | ((uint)d[at + 1] << 8) | ((uint)d[at + 2] << 16) | ((uint)d[at + 3] << 24);

    /// <summary>LZ4_hash4 for byU16 (hash log 13).</summary>
    private static int Hash(ArrayView<byte> d, int at) => (int)((Read32(d, at) * 2654435761u) >> (32 - HashLog));

    /// <summary>Zeroes the wave's hash tables and resets every block's state (thread per table entry).</summary>
    private static void ClearKernel(Index1D i, ArrayView<int> table, ArrayView<int> state)
    {
        table[i] = 0;
        int e = i.X;
        if ((e & (HashSize - 1)) == 0)
        {
            int at = (e >> HashLog) * StateInts;
            for (int k = 0; k < StateInts; k++) state[at + k] = 0;
        }
    }

    /// <summary>
    /// One bounded launch of block waveBase + idx: LZ4_compress_generic_validated (byU16, noDict, limitedOutput,
    /// dstCapacity = size - 1) as a state machine. Positions are relative to the block start.
    /// </summary>
    private static void StepKernel(Index1D idx, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> table, ArrayView<int> state,
        ArrayView<int> prm, int waveBase)
    {
        int b = idx.X;
        int sa = b * StateInts;
        int phase = state[sa + SPhase];
        if (phase == PhDone) return;
        int block = waveBase + b;
        int length = prm[PLength];
        int start = block * BlockSize;
        int size = length - start;
        if (size > BlockSize) size = BlockSize;
        int tb = b * HashSize;
        int so = b * SlotBytes;
        int olimit = size - 1;
        int mflimitPlusOne = size - MfLimit + 1;
        int matchlimit = size - LastLiterals;
        int accel = prm[PAccel];

        int ip = state[sa + SIp];
        int anchor = state[sa + SAnchor];
        int op = state[sa + SOp];
        int forwardH = state[sa + SForwardH];
        int forwardIp = state[sa + SForwardIp];
        int step = state[sa + SStep];
        int searchNb = state[sa + SSearchNb];
        int match = state[sa + SMatch];
        int tokenPos = state[sa + STokenPos];
        int token = state[sa + SToken];
        int result = 0;

        if (phase == PhInit)
        {
            anchor = 0;
            op = 0;
            phase = PhLast;
            if (size >= MinLength)
            {
                table[tb + Hash(data, start)] = 0;
                ip = 1;
                forwardH = Hash(data, start + 1);
                phase = PhTop;
            }
        }

        int launchStart = ip;
        bool running = true;
        while (running)
        {
            if (phase == PhTop)
            {
                forwardIp = ip;
                step = 1;
                searchNb = accel << SkipTrigger;
                phase = PhSearch;
            }
            else if (phase == PhSearch)
            {
                // Find a match (byU16: the 64 KB distance limit can never be exceeded).
                bool searching = true;
                while (searching)
                {
                    int h = forwardH;
                    int current = forwardIp;
                    int matchIndex = table[tb + h];
                    ip = forwardIp;
                    forwardIp += step;
                    step = searchNb >> SkipTrigger;
                    searchNb++;
                    bool past = forwardIp > mflimitPlusOne;
                    if (past)
                    {
                        phase = PhLast;
                        searching = false;
                    }
                    else
                    {
                        forwardH = Hash(data, start + forwardIp);
                        table[tb + h] = current;
                        bool same = Read32(data, start + matchIndex) == Read32(data, start + ip);
                        if (same)
                        {
                            match = matchIndex;
                            phase = PhMatch;
                            searching = false;
                        }
                        else if (forwardIp - launchStart >= Budget)
                        {
                            searching = false;
                            running = false;
                        }
                    }
                }
                if (phase == PhMatch)
                {
                    // Catch up.
                    bool back = true;
                    while (back)
                    {
                        bool can = (ip > anchor) & (match > 0);
                        if (can) can = data[start + ip - 1] == data[start + match - 1];
                        if (can) { ip--; match--; }
                        else back = false;
                    }
                    // Literals.
                    int litLength = ip - anchor;
                    tokenPos = op;
                    op++;
                    bool overflow = op + litLength + (2 + 1 + LastLiterals) + litLength / 255 > olimit;
                    if (overflow)
                    {
                        result = -1;
                        phase = PhDone;
                        running = false;
                    }
                    else
                    {
                        if (litLength >= 15)
                        {
                            token = 15 << 4;
                            int len = litLength - 15;
                            while (len >= 255)
                            {
                                slots[so + op] = 255;
                                op++;
                                len -= 255;
                            }
                            slots[so + op] = (byte)len;
                            op++;
                        }
                        else token = litLength << 4;
                        for (int k = 0; k < litLength; k++) slots[so + op + k] = data[start + anchor + k];
                        op += litLength;
                    }
                }
            }
            else if (phase == PhMatch)
            {
                // Offset, match length, then test the next position.
                int distance = ip - match;
                slots[so + op] = (byte)distance;
                slots[so + op + 1] = (byte)(distance >> 8);
                op += 2;
                int matchCode = 0;
                int a = ip + MinMatch, m = match + MinMatch;
                bool counting = a < matchlimit;
                while (counting)
                {
                    if (data[start + a] == data[start + m])
                    {
                        a++;
                        m++;
                        matchCode++;
                        counting = a < matchlimit;
                    }
                    else counting = false;
                }
                ip += matchCode + MinMatch;
                bool overflow = op + (1 + LastLiterals) + (matchCode + 240) / 255 > olimit;
                if (overflow)
                {
                    result = -1;
                    phase = PhDone;
                    running = false;
                }
                else
                {
                    if (matchCode >= 15)
                    {
                        token += 15;
                        matchCode -= 15;
                        while (matchCode >= 255)
                        {
                            slots[so + op] = 255;
                            op++;
                            matchCode -= 255;
                        }
                        slots[so + op] = (byte)matchCode;
                        op++;
                    }
                    else token += matchCode;
                    slots[so + tokenPos] = (byte)token;
                    anchor = ip;
                    if (ip >= mflimitPlusOne) phase = PhLast;
                    else
                    {
                        table[tb + Hash(data, start + ip - 2)] = ip - 2;
                        int h = Hash(data, start + ip);
                        int matchIndex = table[tb + h];
                        table[tb + h] = ip;
                        bool same = Read32(data, start + matchIndex) == Read32(data, start + ip);
                        if (same)
                        {
                            tokenPos = op;
                            token = 0;
                            op++;
                            match = matchIndex;
                        }
                        else
                        {
                            ip++;
                            forwardH = Hash(data, start + ip);
                            phase = PhTop;
                        }
                        if (ip - launchStart >= Budget) running = false;
                    }
                }
            }
            else if (phase == PhLast)
            {
                int lastRun = size - anchor;
                bool overflow = op + lastRun + 1 + (lastRun + 255 - 15) / 255 > olimit;
                if (overflow) result = -1;
                else
                {
                    if (lastRun >= 15)
                    {
                        slots[so + op] = 15 << 4;
                        op++;
                        int acc = lastRun - 15;
                        while (acc >= 255)
                        {
                            slots[so + op] = 255;
                            op++;
                            acc -= 255;
                        }
                        slots[so + op] = (byte)acc;
                        op++;
                    }
                    else
                    {
                        slots[so + op] = (byte)(lastRun << 4);
                        op++;
                    }
                    for (int k = 0; k < lastRun; k++) slots[so + op + k] = data[start + anchor + k];
                    op += lastRun;
                    result = op;
                }
                phase = PhDone;
                running = false;
            }
            else running = false;
        }

        state[sa + SPhase] = phase;
        state[sa + SIp] = ip;
        state[sa + SAnchor] = anchor;
        state[sa + SOp] = op;
        state[sa + SForwardH] = forwardH;
        state[sa + SForwardIp] = forwardIp;
        state[sa + SStep] = step;
        state[sa + SSearchNb] = searchNb;
        state[sa + SMatch] = match;
        state[sa + STokenPos] = tokenPos;
        state[sa + SToken] = token;
        if (phase == PhDone) state[sa + SResult] = result;
    }

    private static int BlockLength(ArrayView<int> prm, int block)
    {
        int size = prm[PLength] - block * BlockSize;
        if (size > BlockSize) size = BlockSize;
        return size;
    }

    /// <summary>Bytes a finished block stores: its compressed size, or its raw size when liblz4 would store it raw (result -1).</summary>
    private static int StoredLength(ArrayView<int> state, int b, int size)
    {
        int r = state[b * StateInts + SResult];
        int stored = size;
        if (r > 0) stored = r;
        return stored;
    }

    /// <summary>One thread: the frame header (magic, FLG, BD, optional content size, header checksum); starts the running position.</summary>
    private static void HeaderKernel(Index1D idx, ArrayView<byte> output, ArrayView<int> prm)
    {
        int flags = prm[PFlags];
        output[0] = 0x04;
        output[1] = 0x22;
        output[2] = 0x4D;
        output[3] = 0x18;
        int flg = 0x40 | 0x20; // version 01, independent blocks
        if ((flags & FlagBlockChecksum) != 0) flg |= 0x10;
        if ((flags & FlagContentSize) != 0) flg |= 0x08;
        if ((flags & FlagContentChecksum) != 0) flg |= 0x04;
        output[4] = (byte)flg;
        output[5] = 4 << 4; // block max size ID 4 = 64 KB
        int p = 6;
        if ((flags & FlagContentSize) != 0)
        {
            int len = prm[PLength];
            output[6] = (byte)len;
            output[7] = (byte)(len >> 8);
            output[8] = (byte)(len >> 16);
            output[9] = (byte)(len >> 24);
            output[10] = 0;
            output[11] = 0;
            output[12] = 0;
            output[13] = 0;
            p = 14;
        }
        output[p] = (byte)(Xxh32Gpu.Hash(output, 4, p - 4, 0) >> 8);
        p++;
        prm[PHeader] = p;
        prm[PPos] = p;
        if ((flags & FlagContentChecksum) != 0) Xxh32Gpu.Begin(prm, PXxh, 0);
    }

    /// <summary>One thread: output offsets of the wave's blocks after the running position; counts unfinished blocks (a bug guard).</summary>
    private static void LayoutKernel(Index1D idx, ArrayView<int> state, ArrayView<int> offsets, ArrayView<int> prm, int count, int waveBase)
    {
        int pos = prm[PPos];
        int extra = 4;
        if ((prm[PFlags] & FlagBlockChecksum) != 0) extra = 8;
        int unfinished = 0;
        for (int b = 0; b < count; b++)
        {
            offsets[b] = pos;
            if (state[b * StateInts + SPhase] != PhDone) unfinished++;
            pos += extra + StoredLength(state, b, BlockLength(prm, waveBase + b));
        }
        prm[PUnfinished] += unfinished;
        prm[PPos] = pos;
    }

    /// <summary>One thread per slot byte of the wave: the stored bytes of each block after its 4-byte header (compressed from the slot, raw from the input).</summary>
    private static void CopyKernel(Index1D i, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<byte> output, ArrayView<int> state,
        ArrayView<int> offsets, int waveBase)
    {
        int b = i.X / SlotBytes;
        int k = i.X - b * SlotBytes;
        int block = waveBase + b;
        // data is exactly the input (SubView of the input length), so its length bounds the last block.
        int size = data.IntLength - block * BlockSize;
        if (size > BlockSize) size = BlockSize;
        int r = state[b * StateInts + SResult];
        int stored = StoredLength(state, b, size);
        if (k >= stored) return;
        byte v;
        if (r > 0) v = slots[b * SlotBytes + k];
        else v = data[block * BlockSize + k];
        output[offsets[b] + 4 + k] = v;
    }

    /// <summary>One thread per block of the wave: the block header (size, raw flag) and the optional block checksum.</summary>
    private static void BlockHeaderKernel(Index1D idx, ArrayView<byte> output, ArrayView<int> state, ArrayView<int> offsets, ArrayView<int> prm, int waveBase)
    {
        int b = idx.X;
        int size = BlockLength(prm, waveBase + b);
        int r = state[b * StateInts + SResult];
        int stored = StoredLength(state, b, size);
        uint word = (uint)stored;
        if (r <= 0) word |= 0x80000000u;
        int at = offsets[b];
        output[at] = (byte)word;
        output[at + 1] = (byte)(word >> 8);
        output[at + 2] = (byte)(word >> 16);
        output[at + 3] = (byte)(word >> 24);
        if ((prm[PFlags] & FlagBlockChecksum) != 0)
        {
            uint c = Xxh32Gpu.Hash(output, at + 4, stored, 0);
            int ca = at + 4 + stored;
            output[ca] = (byte)c;
            output[ca + 1] = (byte)(c >> 8);
            output[ca + 2] = (byte)(c >> 16);
            output[ca + 3] = (byte)(c >> 24);
        }
    }

    /// <summary>One thread: folds the next ChecksumStripes 16-byte stripes of the input into the content checksum.</summary>
    private static void ContentStripesKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> prm, int doneStripes)
    {
        int total = prm[PLength] / 16;
        int n = total - doneStripes;
        if (n > ChecksumStripes) n = ChecksumStripes;
        if (n > 0) Xxh32Gpu.Stripes(prm, PXxh, data, doneStripes * 16, n);
    }

    /// <summary>One thread: the end mark and the optional content checksum; the frame length into prm[PTotal].</summary>
    private static void TailKernel(Index1D idx, ArrayView<byte> data, ArrayView<byte> output, ArrayView<int> prm)
    {
        int pos = prm[PPos];
        output[pos] = 0;
        output[pos + 1] = 0;
        output[pos + 2] = 0;
        output[pos + 3] = 0;
        pos += 4;
        if ((prm[PFlags] & FlagContentChecksum) != 0)
        {
            int length = prm[PLength];
            int whole = length / 16 * 16;
            uint c = Xxh32Gpu.Finish(prm, PXxh, data, whole, length - whole, 0);
            output[pos] = (byte)c;
            output[pos + 1] = (byte)(c >> 8);
            output[pos + 2] = (byte)(c >> 16);
            output[pos + 3] = (byte)(c >> 24);
            pos += 4;
        }
        prm[PTotal] = pos;
    }

    public void Dispose()
    {
        _prm.Dispose();
        _state.Dispose();
        _table.Dispose();
        _offsets.Dispose();
        _slots.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
