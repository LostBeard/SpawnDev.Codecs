// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GDeflate decompressor on the accelerator (Microsoft DirectStorage's GDeflate, Apache 2.0; format in its
// GDeflate/README.md and TileStream.h). A GDeflate stream is a tile stream: an 8-byte header, a table of tile offsets,
// then tiles of 64 KB of output each, every tile one GDeflate bitstream: DEFLATE blocks (stored / static / dynamic
// Huffman) whose bits are dealt over 32 sub-streams so a GPU can decode 32 symbols at once. The symbol alphabet is
// Deflate64's (the reference library is built with DEFLATE64): distance symbols 30 / 31 reach 64 KB back, and length
// symbol 285 is 3 + 16 extra bits (stored blocks carry LEN only, no NLEN).
//
// Decode model (the reference decoder, libdeflate's gdeflate_decompress_template.h): 32 sub-streams, each a 64-bit
// bit buffer, served round-robin from one input cursor; after a sub-stream's slot it is refilled with the next 32-bit
// packet when it holds fewer than 32 bits. A literal/length symbol takes one slot; a match's distance takes the same
// sub-stream's NEXT slot, and the copy runs then (its output range was reserved when the length decoded). Stored
// blocks hand one byte per slot. Every symbol fits the 32 bits a refill guarantees, so a symbol decodes from a peek.
//
// Here one thread decodes one tile (64 KB, so a launch is bounded whatever the stream's size), tiles in waves. The
// header and tile table are read on the accelerator; the host reads back the total length and an error word.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression.GDeflate;

/// <summary>GDeflate (DirectStorage tile stream) decompressor on the accelerator.</summary>
public sealed class GDeflateDecompressorGpu : IDisposable
{
    /// <summary>Output bytes of every tile but the last.</summary>
    public const int TileSize = 64 * 1024;
    /// <summary>Tiles decoded at once by default (each needs ScratchInts ints of working memory).</summary>
    public const int DefaultWave = 4096;
    /// <summary>The GDeflate codec id in the tile stream header.</summary>
    public const int CodecId = 4;

    private const int Streams = 32;
    private const int MaxBits = 15;
    private const int LitLutBits = 10, DistLutBits = 8;

    // Per-tile scratch (ints).
    private const int BitLoOff = 0, BitHiOff = BitLoOff + Streams, BitsLeftOff = BitHiOff + Streams,
        CopyLenOff = BitsLeftOff + Streams, CopyOutOff = CopyLenOff + Streams;
    private const int LensOff = CopyOutOff + Streams;          // [320 + 138 overrun]
    private const int LitCountOff = LensOff + 460;              // [16]
    private const int LitSymOff = LitCountOff + 16;             // [288]
    private const int DistCountOff = LitSymOff + 288;           // [16]
    private const int DistSymOff = DistCountOff + 16;           // [32]
    private const int PreCountOff = DistSymOff + 32;            // [16]
    private const int PreSymOff = PreCountOff + 16;             // [24]
    private const int OffsOff = PreSymOff + 24;                 // [16]
    private const int LitLutOff = OffsOff + 16;                 // [1024]
    private const int DistLutOff = LitLutOff + (1 << LitLutBits); // [256]
    private const int PreLutOff = DistLutOff + (1 << DistLutBits); // [128]
    /// <summary>Ints of working memory per tile.</summary>
    public const int ScratchInts = PreLutOff + 128;

    // prm ints.
    internal const int PLength = 0, PTiles = 1, PTotal = 2, PLastSize = 3, PData = 4, PError = 5, PErrorAt = 6, PDone = 7;
    internal const int PrmInts = 8;

    /// <summary>Why a stream was refused (the error word the kernels write).</summary>
    public enum GDeflateError
    {
        None = 0, Truncated = 1, BadHeader = 2, BadTileTable = 3, BadBlockType = 4, BadCodeLengths = 5, BadHuffmanCode = 6,
        BadSymbol = 7, BadDistance = 8, TileLength = 9, OutputTooSmall = 10, Empty = 11,
    }

    /// <summary>Thrown for a stream the decoder refuses; <see cref="Reason"/> says why, <see cref="Offset"/> where (input byte, or tile index).</summary>
    public sealed class GDeflateFormatException : FormatException
    {
        public GDeflateError Reason { get; }
        public long Offset { get; }
        public GDeflateFormatException(GDeflateError reason, long offset) : base($"GDeflate: {reason} at {offset}.")
        {
            Reason = reason;
            Offset = offset;
        }
    }

    private readonly Accelerator _accelerator;
    private readonly int _wave;
    private readonly LaunchPacer _pacer;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _header;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _decode;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm, _scratch;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _noOut;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>Compiles the kernels and allocates one wave's working memory (~7 KB per tile of <paramref name="wave"/>).</summary>
    public GDeflateDecompressorGpu(Accelerator accelerator, int wave = DefaultWave)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (wave < 1 || wave > 65535) throw new ArgumentOutOfRangeException(nameof(wave));
        _accelerator = accelerator;
        _wave = wave;
        _pacer = new LaunchPacer(accelerator);
        _header = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(HeaderKernel);
        _decode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(DecodeKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _scratch = accelerator.Allocate1D<int>((long)wave * ScratchInts);
        _noOut = accelerator.Allocate1D<byte>(1);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Decompresses the GDeflate tile stream in the first <paramref name="length"/> bytes of <paramref name="input"/> into
    /// <paramref name="output"/>; returns the decompressed length. Throws <see cref="GDeflateFormatException"/> for a bad
    /// stream (including output too small).
    /// </summary>
    public async Task<int> DecompressAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output)
    {
        int total = await PrepareAsync(input, length);
        if (total > output.Length) throw new GDeflateFormatException(GDeflateError.OutputTooSmall, total);
        await DecodeAsync(input, length, output, total);
        return total;
    }

    /// <summary>The decompressed length the stream's header declares (header and tile table validated; nothing decoded).</summary>
    public Task<int> GetDecompressedLengthAsync(ArrayView1D<byte, Stride1D.Dense> input, int length) => PrepareAsync(input, length);

    /// <summary>Decompresses <paramref name="stream"/> into a new array.</summary>
    public async Task<byte[]> DecompressAsync(ReadOnlyMemory<byte> stream)
    {
        var src = _stageIn.Ensure(stream.Length);
        if (stream.Length > 0) DeviceTransfer.Upload(src, stream.Span);
        int total = await PrepareAsync(src, stream.Length);
        var dst = _stageOut.Ensure(total);
        await DecodeAsync(src, stream.Length, dst, total);
        return await DeviceTransfer.DownloadAsync(dst, total);
    }

    /// <summary>Decompresses the bytes of <paramref name="stream"/> into a new Uint8Array (never on the .NET heap); the caller disposes it.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> DecompressAsync(TypedArray stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (stream.ByteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(stream), "At most 2^31 - 1 bytes per call.");
        int length = (int)stream.ByteLength;
        var src = _stageIn.Ensure(length);
        if (length > 0) DeviceTransfer.Upload(src, stream);
        int total = await PrepareAsync(src, length);
        var dst = _stageOut.Ensure(total);
        await DecodeAsync(src, length, dst, total);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, total);
    }

    // ------------------------------------------------------------------
    // Host coordination
    // ------------------------------------------------------------------

    private int _tiles;

    private async Task<int> PrepareAsync(ArrayView1D<byte, Stride1D.Dense> input, int length)
    {
        if (length < 0 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length}.");
        if (length == 0) throw new GDeflateFormatException(GDeflateError.Empty, 0);
        var prm = new int[PrmInts];
        prm[PLength] = length;
        _prm.View.CopyFromCPU(prm);
        _header(1, input.SubView(0, length), _prm.View);
        await _accelerator.SynchronizeAsync();
        var head = await _prm.View.CopyToHostAsync();
        ThrowIfError(head);
        _tiles = head[PTiles];
        return head[PTotal];
    }

    private async Task DecodeAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, int total)
    {
        if (total > output.Length) throw new GDeflateFormatException(GDeflateError.OutputTooSmall, total);
        var data = input.SubView(0, length);
        var outView = total > 0 ? output.SubView(0, total) : _noOut.View;
        // One launch per wave: a tile is at most 64 KB of output, so the launch is bounded.
        for (int first = 0; first < _tiles; first += _wave)
        {
            int count = Math.Min(_wave, _tiles - first);
            _decode(count, data, outView, _scratch.View, _prm.View, first);
            await _pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
        var end = await _prm.View.CopyToHostAsync();
        ThrowIfError(end);
        if (end[PDone] != _tiles) throw new InvalidOperationException($"GDeflate decoder: {_tiles - end[PDone]} tiles did not finish (bug).");
    }

    private static void ThrowIfError(int[] prm)
    {
        if (prm[PError] != 0) throw new GDeflateFormatException((GDeflateError)prm[PError], prm[PErrorAt]);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    private static uint Read32(ArrayView<byte> d, int at) =>
        (uint)d[at] | ((uint)d[at + 1] << 8) | ((uint)d[at + 2] << 16) | ((uint)d[at + 3] << 24);

    private static void Fail(ArrayView<int> prm, GDeflateError e, int at)
    {
        if (prm[PError] == 0)
        {
            prm[PError] = (int)e;
            prm[PErrorAt] = at;
        }
    }

    /// <summary>
    /// One thread: the tile stream header (id, magic, tile size index, tile count, last tile size) and the tile table
    /// (offsets increasing, every tile inside the input); the total length into prm.
    /// </summary>
    private static void HeaderKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> prm)
    {
        int length = prm[PLength];
        if (length < 8) { Fail(prm, GDeflateError.Truncated, 0); return; }
        int id = d[0];
        int magic = d[1];
        int tiles = d[2] | (d[3] << 8);
        uint bits = Read32(d, 4);
        int sizeIdx = (int)(bits & 3u);
        int last = (int)((bits >> 2) & 0x3FFFFu);
        bool bad = (id != CodecId) | (magic != (id ^ 0xFF)) | (sizeIdx != 1) | (tiles == 0) | (last > TileSize);
        if (bad) { Fail(prm, GDeflateError.BadHeader, 0); return; }
        int data = 8 + 4 * tiles;
        if (data > length) { Fail(prm, GDeflateError.Truncated, 8); return; }
        // Entry 0 is the last tile's compressed size; entry i > 0 is tile i's offset from the data start.
        long prev = 0;
        bool tableBad = false;
        for (int t = 1; t < tiles; t++)
        {
            long off = Read32(d, 8 + 4 * t);
            if (off < prev) tableBad = true;
            prev = off;
        }
        long lastOff = 0;
        if (tiles > 1) lastOff = Read32(d, 8 + 4 * (tiles - 1));
        long end = data + lastOff + Read32(d, 8);
        if (tableBad | (end > length)) { Fail(prm, GDeflateError.BadTileTable, 8); return; }
        prm[PTiles] = tiles;
        prm[PLastSize] = last;
        prm[PData] = data;
        prm[PTotal] = tiles * TileSize - (last == 0 ? 0 : TileSize - last);
        prm[PDone] = 0;
    }

    /// <summary>One thread per tile of the wave: the whole tile (at most 64 KB of output).</summary>
    private static void DecodeKernel(Index1D idx, ArrayView<byte> d, ArrayView<byte> o, ArrayView<int> sc, ArrayView<int> prm, int first)
    {
        if (prm[PError] != 0) return;
        int tile = first + idx.X;
        int tiles = prm[PTiles];
        int data = prm[PData];
        int sb = idx.X * ScratchInts;
        int inPos = data;
        if (tile > 0) inPos = data + (int)Read32(d, 8 + 4 * tile);
        int inEnd = data + (int)Read32(d, 8);
        if (tile < tiles - 1) inEnd = data + (int)Read32(d, 8 + 4 * (tile + 1));
        else if (tiles > 1) inEnd = data + (int)Read32(d, 8 + 4 * (tiles - 1)) + (int)Read32(d, 8);
        int outBase = tile * TileSize;
        int outEnd = outBase + TileSize;
        int last = prm[PLastSize];
        if ((tile == tiles - 1) & (last != 0)) outEnd = outBase + last;

        var st = new TileState { InPos = inPos, InEnd = inEnd, OutPos = outBase, OutBase = outBase, OutEnd = outEnd, Idx = 0, IsCopy = 0, Err = 0 };
        for (int n = 0; n < Streams; n++)
        {
            sc[sb + BitLoOff + n] = 0;
            sc[sb + BitHiOff + n] = 0;
            sc[sb + BitsLeftOff + n] = 0;
            sc[sb + CopyLenOff + n] = 0;
            Advance(ref st, d, sc, sb);
        }

        bool final = false;
        bool going = st.Err == 0;
        while (going)
        {
            st.Idx = 0;
            final = PopBits(ref st, sc, sb, 1) != 0;
            int type = PopBits(ref st, sc, sb, 2);
            Ensure(ref st, d, sc, sb);
            if (type == 2) DynamicHeader(ref st, d, sc, sb);
            else if (type == 1) StaticHeader(sc, sb);
            else if (type != 0) st.Err = (int)GDeflateError.BadBlockType;

            if ((type == 0) & (st.Err == 0))
            {
                // Stored: a 16-bit length, then one byte per slot.
                int len = PopBits(ref st, sc, sb, 16);
                if (st.OutPos + len > st.OutEnd) st.Err = (int)GDeflateError.TileLength;
                int k = 0;
                bool copying = (k < len) & (st.Err == 0);
                while (copying)
                {
                    o[st.OutPos] = (byte)PopBits(ref st, sc, sb, 8);
                    st.OutPos++;
                    Advance(ref st, d, sc, sb);
                    k++;
                    copying = (k < len) & (st.Err == 0);
                }
            }
            else if (st.Err == 0)
            {
                st.Idx = 0;
                bool symbols = true;
                while (symbols)
                {
                    if ((st.IsCopy & 1u) == 0)
                    {
                        int sym = DecodeSymbol(ref st, sc, sb, LitLutOff, LitLutBits, LitCountOff, LitSymOff);
                        if (st.Err != 0) symbols = false;
                        else if (sym < 256)
                        {
                            if (st.OutPos >= st.OutEnd) { st.Err = (int)GDeflateError.TileLength; symbols = false; }
                            else
                            {
                                o[st.OutPos] = (byte)sym;
                                st.OutPos++;
                                Advance(ref st, d, sc, sb);
                            }
                        }
                        else if (sym == 256) symbols = false; // end of block: straight to the deferred copies
                        else
                        {
                            int ls = sym - 257;
                            int len = 0;
                            if (ls >= 29) st.Err = (int)GDeflateError.BadSymbol;
                            else if (ls < 8) len = 3 + ls;
                            else if (ls == 28) len = 3 + PopBits(ref st, sc, sb, 16); // Deflate64: 3 + 16 extra bits
                            else
                            {
                                int e = ((ls - 8) >> 2) + 1;
                                len = 3 + ((4 + ((ls - 8) & 3)) << e) + PopBits(ref st, sc, sb, e);
                            }
                            if ((st.Err == 0) & (st.OutPos + len > st.OutEnd)) st.Err = (int)GDeflateError.TileLength;
                            if (st.Err != 0) symbols = false;
                            else
                            {
                                sc[sb + CopyLenOff + st.Idx] = len;
                                sc[sb + CopyOutOff + st.Idx] = st.OutPos;
                                st.IsCopy |= 1u;
                                st.OutPos += len;
                                Advance(ref st, d, sc, sb);
                            }
                        }
                    }
                    else
                    {
                        DoCopy(ref st, o, sc, sb);
                        if (st.Err != 0) symbols = false;
                        else Advance(ref st, d, sc, sb);
                    }
                    if (st.Err != 0) symbols = false;
                }
            }

            // Block done: the outstanding deferred copies, one slot per sub-stream.
            int m = 0;
            bool flushing = st.Err == 0;
            while (flushing)
            {
                if ((st.IsCopy & 1u) != 0) DoCopy(ref st, o, sc, sb);
                if (st.Err == 0) Advance(ref st, d, sc, sb);
                m++;
                flushing = (m < Streams) & (st.Err == 0);
            }
            going = (!final) & (st.Err == 0);
        }

        if ((st.Err == 0) & (st.OutPos != st.OutEnd)) st.Err = (int)GDeflateError.TileLength;
        if (st.Err != 0) Fail(prm, (GDeflateError)st.Err, tile);
        else Atomic.Add(ref prm[PDone], 1);
    }

    private struct TileState
    {
        public int InPos, InEnd, OutPos, OutBase, OutEnd, Idx, Err;
        public uint IsCopy;
    }

    /// <summary>Refills the current sub-stream with the next 32-bit packet when it holds fewer than 32 bits.</summary>
    private static void Ensure(ref TileState st, ArrayView<byte> d, ArrayView<int> sc, int sb)
    {
        int s = st.Idx;
        int left = sc[sb + BitsLeftOff + s];
        if ((left < 32) & (st.Err == 0))
        {
            if (st.InPos + 4 > st.InEnd) st.Err = (int)GDeflateError.Truncated;
            else
            {
                uint v = Read32(d, st.InPos);
                st.InPos += 4;
                uint lo = (uint)sc[sb + BitLoOff + s];
                uint hi = (uint)sc[sb + BitHiOff + s];
                if (left == 0) lo = v;
                else
                {
                    lo |= v << left;
                    hi |= v >> (32 - left);
                }
                sc[sb + BitLoOff + s] = (int)lo;
                sc[sb + BitHiOff + s] = (int)hi;
                sc[sb + BitsLeftOff + s] = left + 32;
            }
        }
    }

    /// <summary>Ensure the current sub-stream, then move to the next one (the copy mask rotates with it).</summary>
    private static void Advance(ref TileState st, ArrayView<byte> d, ArrayView<int> sc, int sb)
    {
        Ensure(ref st, d, sc, sb);
        st.Idx = (st.Idx + 1) & (Streams - 1);
        st.IsCopy = (st.IsCopy >> 1) | (st.IsCopy << 31);
    }

    /// <summary>The next 32 bits of the current sub-stream (fewer are valid when it holds fewer).</summary>
    private static uint Peek(ref TileState st, ArrayView<int> sc, int sb) => (uint)sc[sb + BitLoOff + st.Idx];

    /// <summary>Drops <paramref name="n"/> (0..32) bits of the current sub-stream.</summary>
    private static void Remove(ref TileState st, ArrayView<int> sc, int sb, int n)
    {
        int s = st.Idx;
        int left = sc[sb + BitsLeftOff + s];
        if (n > left) st.Err = (int)GDeflateError.Truncated;
        else if (n > 0)
        {
            uint lo = (uint)sc[sb + BitLoOff + s];
            uint hi = (uint)sc[sb + BitHiOff + s];
            if (n == 32)
            {
                lo = hi;
                hi = 0;
            }
            else
            {
                lo = (lo >> n) | (hi << (32 - n));
                hi >>= n;
            }
            sc[sb + BitLoOff + s] = (int)lo;
            sc[sb + BitHiOff + s] = (int)hi;
            sc[sb + BitsLeftOff + s] = left - n;
        }
    }

    private static int PopBits(ref TileState st, ArrayView<int> sc, int sb, int n)
    {
        uint v = Peek(ref st, sc, sb);
        if (n < 32) v &= (1u << n) - 1u;
        Remove(ref st, sc, sb, n);
        return (int)v;
    }

    /// <summary>Dynamic block header: HLIT / HDIST / HCLEN, the precode lengths (one per slot), the code lengths.</summary>
    private static void DynamicHeader(ref TileState st, ArrayView<byte> d, ArrayView<int> sc, int sb)
    {
        int nlit = PopBits(ref st, sc, sb, 5) + 257;
        int ndist = PopBits(ref st, sc, sb, 5) + 1;
        int ncode = PopBits(ref st, sc, sb, 4) + 4;
        Ensure(ref st, d, sc, sb);
        int lens = sb + LensOff;
        for (int i = 0; i < 19; i++) sc[lens + i] = 0;
        int k = 0;
        bool reading = (k < ncode) & (st.Err == 0);
        while (reading)
        {
            sc[lens + Compression.InflateGpu.CodeLengthOrder(k)] = PopBits(ref st, sc, sb, 3);
            Advance(ref st, d, sc, sb);
            k++;
            reading = (k < ncode) & (st.Err == 0);
        }
        if (st.Err != 0) return;
        int left = Construct(sc, sb, lens, 19, sb + PreCountOff, sb + PreSymOff);
        if (left != 0) { st.Err = (int)GDeflateError.BadCodeLengths; return; }
        BuildLut(sc, sb + PreCountOff, sb + PreSymOff, sb + PreLutOff, 7);

        st.Idx = 0;
        int total = nlit + ndist;
        int index = 0;
        bool going = (index < total) & (st.Err == 0);
        while (going)
        {
            int sym = DecodeSymbol(ref st, sc, sb, PreLutOff, 7, PreCountOff, PreSymOff);
            if (st.Err == 0)
            {
                if (sym < 16)
                {
                    sc[lens + index] = sym;
                    index++;
                }
                else
                {
                    int rep = 0;
                    int val = 0;
                    if (sym == 16)
                    {
                        if (index == 0) st.Err = (int)GDeflateError.BadCodeLengths;
                        else
                        {
                            val = sc[lens + index - 1];
                            rep = 3 + PopBits(ref st, sc, sb, 2);
                        }
                    }
                    else if (sym == 17) rep = 3 + PopBits(ref st, sc, sb, 3);
                    else rep = 11 + PopBits(ref st, sc, sb, 7);
                    // The reference lets a run overrun the lengths (into slack it ignores); so do we.
                    for (int r = 0; r < rep; r++) sc[lens + index + r] = val;
                    index += rep;
                }
                if (st.Err == 0) Advance(ref st, d, sc, sb);
            }
            going = (index < total) & (st.Err == 0);
        }
        if (st.Err != 0) return;
        BuildCodes(ref st, sc, sb, nlit, ndist);
    }

    /// <summary>Static block: the RFC 1951 fixed lengths (288 literal/length, 32 distance).</summary>
    private static void StaticHeader(ArrayView<int> sc, int sb)
    {
        int lens = sb + LensOff;
        for (int i = 0; i < 144; i++) sc[lens + i] = 8;
        for (int i = 144; i < 256; i++) sc[lens + i] = 9;
        for (int i = 256; i < 280; i++) sc[lens + i] = 7;
        for (int i = 280; i < 288; i++) sc[lens + i] = 8;
        for (int i = 288; i < 320; i++) sc[lens + i] = 5;
        var dummy = new TileState();
        BuildCodes(ref dummy, sc, sb, 288, 32);
    }

    /// <summary>Literal/length and distance codes from the lengths: complete, or incomplete with a single code of length 1.</summary>
    private static void BuildCodes(ref TileState st, ArrayView<int> sc, int sb, int nlit, int ndist)
    {
        int lens = sb + LensOff;
        int left = Construct(sc, sb, lens, nlit, sb + LitCountOff, sb + LitSymOff);
        bool litBad = (left < 0) | ((left > 0) & (nlit != sc[sb + LitCountOff] + sc[sb + LitCountOff + 1]));
        left = Construct(sc, sb, lens + nlit, ndist, sb + DistCountOff, sb + DistSymOff);
        bool distBad = (left < 0) | ((left > 0) & (ndist != sc[sb + DistCountOff] + sc[sb + DistCountOff + 1]));
        if (litBad | distBad) st.Err = (int)GDeflateError.BadHuffmanCode;
        else
        {
            BuildLut(sc, sb + LitCountOff, sb + LitSymOff, sb + LitLutOff, LitLutBits);
            BuildLut(sc, sb + DistCountOff, sb + DistSymOff, sb + DistLutOff, DistLutBits);
        }
    }

    /// <summary>The deferred copy of the current sub-stream: its distance symbol, then the bytes (overlap-safe).</summary>
    private static void DoCopy(ref TileState st, ArrayView<byte> o, ArrayView<int> sc, int sb)
    {
        int len = sc[sb + CopyLenOff + st.Idx];
        int at = sc[sb + CopyOutOff + st.Idx];
        int dsym = DecodeSymbol(ref st, sc, sb, DistLutOff, DistLutBits, DistCountOff, DistSymOff);
        if (st.Err != 0) return;
        // Deflate64: symbols 30 and 31 (bases 32769 and 49153, 14 extra bits) reach across the 64 KB tile.
        if (dsym >= 32) { st.Err = (int)GDeflateError.BadDistance; return; }
        int dist = dsym + 1;
        if (dsym >= 4)
        {
            int de = (dsym >> 1) - 1;
            dist = ((2 + (dsym & 1)) << de) + 1 + PopBits(ref st, sc, sb, de);
        }
        if (dist > at - st.OutBase) { st.Err = (int)GDeflateError.BadDistance; return; }
        for (int i = 0; i < len; i++) o[at + i] = o[at + i - dist];
        st.IsCopy &= ~1u;
    }

    /// <summary>
    /// One symbol of the current sub-stream from its peeked bits: the first-level table, else the canonical walk (codes
    /// longer than the table). The bits are taken only when the symbol decoded.
    /// </summary>
    private static int DecodeSymbol(ref TileState st, ArrayView<int> sc, int sb, int lutOff, int lutBits, int countOff, int symOff)
    {
        uint bits = Peek(ref st, sc, sb);
        int entry = sc[sb + lutOff + (int)(bits & ((1u << lutBits) - 1u))];
        int len = entry & 15;
        int result = -1;
        if (len != 0)
        {
            result = entry >> 4;
            Remove(ref st, sc, sb, len);
        }
        else
        {
            int code = 0, first = 0, index = 0;
            int l = 1;
            bool walking = true;
            while (walking)
            {
                code |= (int)((bits >> (l - 1)) & 1u);
                int count = sc[sb + countOff + l];
                if (code - count < first)
                {
                    result = sc[sb + symOff + index + (code - first)];
                    walking = false;
                }
                else
                {
                    index += count;
                    first += count;
                    first <<= 1;
                    code <<= 1;
                    l++;
                    walking = l <= MaxBits;
                }
            }
            if (result < 0) st.Err = (int)GDeflateError.BadSymbol;
            else Remove(ref st, sc, sb, l);
        }
        return result;
    }

    /// <summary>Canonical decoding tables (puff's construct): 0 complete, &gt; 0 incomplete, &lt; 0 over-subscribed.</summary>
    private static int Construct(ArrayView<int> sc, int sb, int lengthsAt, int n, int countAt, int symAt)
    {
        int offsAt = sb + OffsOff;
        for (int len = 0; len <= MaxBits; len++) sc[countAt + len] = 0;
        for (int sym = 0; sym < n; sym++)
        {
            int slot = countAt + sc[lengthsAt + sym];
            sc[slot] = sc[slot] + 1;
        }
        int left = 1;
        if (sc[countAt] == n) left = 0;
        else
        {
            for (int len = 1; len <= MaxBits; len++)
            {
                left <<= 1;
                left -= sc[countAt + len];
            }
        }
        bool build = (left >= 0) & (sc[countAt] != n);
        if (build)
        {
            sc[offsAt + 1] = 0;
            for (int len = 1; len < MaxBits; len++) sc[offsAt + len + 1] = sc[offsAt + len] + sc[countAt + len];
            for (int sym = 0; sym < n; sym++)
            {
                int l = sc[lengthsAt + sym];
                if (l != 0)
                {
                    int at = sc[offsAt + l];
                    sc[symAt + at] = sym;
                    sc[offsAt + l] = at + 1;
                }
            }
        }
        return left;
    }

    /// <summary>First-level table: (symbol &lt;&lt; 4) | length for every code of at most lutBits bits, bit-reversed index; 0 elsewhere.</summary>
    private static void BuildLut(ArrayView<int> sc, int countAt, int symAt, int lutAt, int lutBits)
    {
        int size = 1 << lutBits;
        for (int i = 0; i < size; i++) sc[lutAt + i] = 0;
        int code = 0;
        int index = 0;
        for (int len = 1; len <= lutBits; len++)
        {
            int count = sc[countAt + len];
            for (int k = 0; k < count; k++)
            {
                int rev = 0;
                for (int b = 0; b < len; b++) rev |= ((code >> b) & 1) << (len - 1 - b);
                int entry = (sc[symAt + index + k] << 4) | len;
                for (int fill = rev; fill < size; fill += 1 << len) sc[lutAt + fill] = entry;
                code++;
            }
            index += count;
            code <<= 1;
        }
    }

    public void Dispose()
    {
        _prm.Dispose();
        _scratch.Dispose();
        _noOut.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
