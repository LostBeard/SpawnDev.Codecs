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

/// <summary>How <see cref="GDeflateDecompressorGpu"/> spreads a tile over the accelerator.</summary>
public enum GDeflateDecodeMode
{
    /// <summary>32 threads per tile (the format's design): one symbol per sub-stream per round. The default.</summary>
    LanesPerTile = 0,
    /// <summary>One thread per tile walking the sub-streams in turn (the reference decoder's order).</summary>
    ThreadPerTile = 1,
}

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
    private const int LitLutBits = 9, DistLutBits = 8;

    // Per-tile scratch (ints). The lanes kernel keeps everything before CopyLenOff in shared memory, and the smaller it
    // is the more tiles an SM runs at once (decode is occupancy-bound: measured): symbol lists and first-level tables
    // hold 16-bit entries two per int (Get16 / Set16), code lengths 4-bit entries eight per int (Get4 / Set4). Only
    // one thread writes them (the block header is serial), so the read-modify-write of a packed int never races.
    private const int BitLoOff = 0, BitHiOff = BitLoOff + Streams, BitsLeftOff = BitHiOff + Streams;
    private const int LensOff = BitsLeftOff + Streams;         // [320 + 138 overrun] x 4 bits
    private const int LitCountOff = LensOff + 58;               // [16]
    private const int LitSymOff = LitCountOff + 16;             // [288] x 16 bits
    private const int DistCountOff = LitSymOff + 144;           // [16]
    private const int DistSymOff = DistCountOff + 16;           // [32] x 16 bits
    private const int PreCountOff = DistSymOff + 16;            // [16]
    private const int PreSymOff = PreCountOff + 16;             // [24] x 16 bits
    private const int OffsOff = PreSymOff + 12;                 // [16]
    private const int LitLutOff = OffsOff + 16;                 // [512] x 16 bits
    private const int DistLutOff = LitLutOff + (1 << LitLutBits) / 2; // [256] x 16 bits
    private const int PreLutOff = DistLutOff + (1 << DistLutBits) / 2; // [128] x 16 bits
    /// <summary>Ints of the lanes kernel's shared memory per tile (the thread-per-tile copy slots follow).</summary>
    private const int LanesScratchInts = PreLutOff + 64;
    // Thread-per-tile only: each sub-stream's deferred copy.
    private const int CopyLenOff = LanesScratchInts, CopyOutOff = CopyLenOff + Streams;
    /// <summary>Ints of working memory per tile.</summary>
    public const int ScratchInts = CopyOutOff + Streams;

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
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int> _decodeLanes;
    private readonly GDeflateDecodeMode _mode;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm, _scratch;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _noOut;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>
    /// Compiles the kernels; <see cref="GDeflateDecodeMode.ThreadPerTile"/> also allocates one wave's working memory (~9 KB
    /// per tile of <paramref name="wave"/>; the lanes mode keeps it in shared memory).
    /// </summary>
    public GDeflateDecompressorGpu(Accelerator accelerator, int wave = DefaultWave, GDeflateDecodeMode mode = GDeflateDecodeMode.LanesPerTile)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (wave < 1 || wave > 65535) throw new ArgumentOutOfRangeException(nameof(wave));
        _accelerator = accelerator;
        _wave = wave;
        _mode = mode;
        _pacer = new LaunchPacer(accelerator);
        _header = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(HeaderKernel);
        _decode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(DecodeKernel);
        _decodeLanes = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int>(DecodeLanesKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _scratch = accelerator.Allocate1D<int>(mode == GDeflateDecodeMode.ThreadPerTile ? (long)wave * ScratchInts : 1);
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
            if (_mode == GDeflateDecodeMode.ThreadPerTile) _decode(count, data, outView, _scratch.View, _prm.View, first);
            else _decodeLanes(new KernelConfig(count, Streams), data, outView, _prm.View, first);
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

    // ------------------------------------------------------------------
    // 32 lanes per tile (GDeflateDecodeMode.LanesPerTile)
    // ------------------------------------------------------------------

    // Shared control ints: input cursor and end, error, final-block flag, block type, stored length; per-lane areas for
    // votes, prefix sums and the round's copy parameters.
    private const int CCursor = 0, CInEnd = 1, CErr = 2, CFinal = 3, CType = 4, CStoredLen = 5;
    // Votes and prefix sums share one per-lane area: each writes it, barriers, reads it and barriers again.
    private const int CVote = 8, CScan = CVote, CCopyDist = CVote + Streams, CCopyLen = CCopyDist + Streams,
        CCopyPos = CCopyLen + Streams;
    private const int ControlInts = CCopyPos + Streams;

    /// <summary>
    /// One group of 32 lanes per tile (the HLSL reference decoder's portable path). Every lane's bit reader, the decode
    /// tables and the input cursor live in shared memory. Lane 0 reads each block header (and a dynamic block's code
    /// lengths) serially in the reference's order; Huffman blocks then decode in rounds of one symbol per lane: a ballot
    /// refills the lanes that fell under 32 bits in lane order, a prefix sum places the round's output, literals are
    /// stored and the copies run in lane order with all 32 lanes copying. Stored blocks hand one byte per lane per round.
    /// Every loop that holds a barrier is steered by Group.BarrierOr / BarrierPopCount (uniform on every backend).
    /// </summary>
    private static void DecodeLanesKernel(ArrayView<byte> d, ArrayView<byte> o, ArrayView<int> prm, int first)
    {
        var sc = SharedMemory.Allocate<int>(LanesScratchInts);
        var ctl = SharedMemory.Allocate<int>(ControlInts);
        int lane = Group.IdxX;
        int tile = first + Grid.IdxX;
        int tiles = prm[PTiles];
        int data = prm[PData];
        int inPos = data;
        if (tile > 0) inPos = data + (int)Read32(d, 8 + 4 * tile);
        int inEnd = data + (int)Read32(d, 8);
        if (tile < tiles - 1) inEnd = data + (int)Read32(d, 8 + 4 * (tile + 1));
        else if (tiles > 1) inEnd = data + (int)Read32(d, 8 + 4 * (tiles - 1)) + (int)Read32(d, 8);
        int outBase = tile * TileSize;
        int outEnd = outBase + TileSize;
        int last = prm[PLastSize];
        if ((tile == tiles - 1) & (last != 0)) outEnd = outBase + last;

        if (lane == 0)
        {
            ctl[CCursor] = inPos + 4 * Streams;
            ctl[CInEnd] = inEnd;
            ctl[CErr] = inPos + 4 * Streams > inEnd ? (int)GDeflateError.Truncated : 0;
        }
        // Every sub-stream starts with one 32-bit packet, in lane order.
        uint firstPacket = 0;
        if (inPos + 4 * lane + 4 <= inEnd) firstPacket = Read32(d, inPos + 4 * lane);
        sc[BitLoOff + lane] = (int)firstPacket;
        sc[BitHiOff + lane] = 0;
        sc[BitsLeftOff + lane] = 32;
        Group.Barrier();

        int dst = outBase;
        bool going = Group.BarrierAnd(ctl[CErr] == 0);
        while (going)
        {
            if (lane == 0) HeaderSerial(d, sc, ctl);
            Group.Barrier();
            bool err = Group.BarrierOr(ctl[CErr] != 0);
            bool stored = Group.BarrierOr(ctl[CType] == 0);
            if (!err)
            {
                if (stored) dst = StoredLanes(d, o, sc, ctl, lane, dst, outEnd);
                else dst = HuffmanLanes(d, o, sc, ctl, lane, dst, outBase, outEnd);
            }
            bool final = Group.BarrierOr(ctl[CFinal] != 0);
            going = Group.BarrierAnd((!final) & (ctl[CErr] == 0));
        }

        if (lane == 0)
        {
            int e = ctl[CErr];
            if ((e == 0) & (dst != outEnd)) e = (int)GDeflateError.TileLength;
            if (e != 0) Fail(prm, (GDeflateError)e, tile);
            else Atomic.Add(ref prm[PDone], 1);
        }
    }

    /// <summary>Lane 0: the block header in the reference's order (a dynamic block's code lengths and tables too).</summary>
    private static void HeaderSerial(ArrayView<byte> d, ArrayView<int> sc, ArrayView<int> ctl)
    {
        var st = new TileState { InPos = ctl[CCursor], InEnd = ctl[CInEnd], Idx = 0, IsCopy = 0, Err = 0 };
        int final = PopBits(ref st, sc, 0, 1);
        int type = PopBits(ref st, sc, 0, 2);
        Ensure(ref st, d, sc, 0);
        int storedLen = 0;
        if (type == 2) DynamicHeader(ref st, d, sc, 0);
        else if (type == 1) StaticHeader(sc, 0);
        else if (type == 0) storedLen = PopBits(ref st, sc, 0, 16);
        else st.Err = (int)GDeflateError.BadBlockType;
        ctl[CCursor] = st.InPos;
        if (st.Err != 0) ctl[CErr] = st.Err;
        ctl[CFinal] = final;
        ctl[CType] = type;
        ctl[CStoredLen] = storedLen;
    }

    /// <summary>A stored block: one byte per lane per round (byte k from sub-stream k mod 32).</summary>
    private static int StoredLanes(ArrayView<byte> d, ArrayView<byte> o, ArrayView<int> sc, ArrayView<int> ctl, int lane, int dst, int outEnd)
    {
        int size = ctl[CStoredLen];
        int done = 0;
        bool more = Group.BarrierOr(done < size);
        while (more)
        {
            bool active = done + lane < size;
            int b = 0;
            if (active) b = LanePop(sc, ctl, lane, 8);
            LaneRefill(d, sc, ctl, lane, active);
            if (active)
            {
                int at = dst + done + lane;
                if (at < outEnd) o[at] = (byte)b;
                else ctl[CErr] = (int)GDeflateError.TileLength;
            }
            done += Streams;
            more = Group.BarrierOr((done < size) & (ctl[CErr] == 0));
        }
        return dst + size;
    }

    /// <summary>A Huffman block in rounds of one symbol per lane (the HLSL reference's CompressedBlock).</summary>
    private static int HuffmanLanes(ArrayView<byte> d, ArrayView<byte> o, ArrayView<int> sc, ArrayView<int> ctl, int lane,
        int dst, int outBase, int outEnd)
    {
        uint below = (1u << lane) - 1u;
        // Round 0: literal/length symbols only.
        int clen = 0;
        int sym = LaneDecode(sc, ctl, lane, false, ref clen);
        bool eob = sym == 256;
        int eobMask = Vote(ctl, lane, eob);
        bool stop = Group.BarrierOr(eob | (ctl[CErr] != 0));
        bool oob = ((uint)eobMask & below) != 0;
        int value = 0;
        if (!oob) value = LaneTranslate(sc, ctl, lane, sym, clen, false);
        LaneRefill(d, sc, ctl, lane, !oob);
        int length = oob ? 0 : value;
        int total = 0;
        int offset = Scan(ctl, lane, length, ref total);
        bool isCopy = (!oob) & (sym > 256);
        int lit = sym;

        bool more = !stop;
        while (more)
        {
            bool isDist = isCopy;
            sym = LaneDecode(sc, ctl, lane, isDist, ref clen);
            eob = (!isDist) & (sym == 256);
            eobMask = Vote(ctl, lane, eob);
            stop = Group.BarrierOr(eob | (ctl[CErr] != 0));
            oob = ((uint)eobMask & below) != 0;
            bool consume = isDist | (!oob);
            value = 0;
            if (consume) value = LaneTranslate(sc, ctl, lane, sym, clen, isDist);
            LaneRefill(d, sc, ctl, lane, consume);
            // The previous round's literals and copies (this round decoded the copies' distances).
            WriteRound(o, ctl, lane, dst + offset, value, length, lit, isDist, outBase, outEnd);
            dst += total;
            length = (isDist | oob) ? 0 : value;
            offset = Scan(ctl, lane, length, ref total);
            isCopy = (!isDist) & (!oob) & (sym > 256);
            lit = sym;
            more = !stop;
        }

        // Last round: the distances of the copies decoded with the end-of-block round.
        int dist = 0;
        if (isCopy)
        {
            sym = LaneDecode(sc, ctl, lane, true, ref clen);
            dist = LaneTranslate(sc, ctl, lane, sym, clen, true);
        }
        LaneRefill(d, sc, ctl, lane, isCopy);
        WriteRound(o, ctl, lane, dst + offset, dist, length, lit, isCopy, outBase, outEnd);
        return dst + total;
    }

    /// <summary>
    /// Stores the round's literals at their places, then runs its copies in lane order with all lanes copying (overlap
    /// safe: byte i of a copy reads the byte dist back modulo dist, which is never part of the copy itself).
    /// </summary>
    private static void WriteRound(ArrayView<byte> o, ArrayView<int> ctl, int lane, int pos, int dist, int length, int lit, bool isCopy,
        int outBase, int outEnd)
    {
        bool bad = (length > 0) & (pos + length > outEnd);
        if (bad) ctl[CErr] = (int)GDeflateError.TileLength;
        if ((!isCopy) & (length != 0) & (!bad)) o[pos] = (byte)lit;
        bool copy = isCopy & (length > 0) & (!bad);
        if (copy & ((dist <= 0) | (dist > pos - outBase)))
        {
            ctl[CErr] = (int)GDeflateError.BadDistance;
            copy = false;
        }
        ctl[CCopyDist + lane] = dist;
        ctl[CCopyLen + lane] = copy ? length : 0;
        ctl[CCopyPos + lane] = pos;
        int copyMask = Vote(ctl, lane, copy);
        int copies = Group.BarrierPopCount(copy);
        for (int k = 0; k < copies; k++)
        {
            int l = NthSetBit((uint)copyMask, k);
            int cdist = ctl[CCopyDist + l];
            int clen = ctl[CCopyLen + l];
            int cpos = ctl[CCopyPos + l];
            for (int i = lane; i < clen; i += Streams) o[cpos + i] = o[cpos - cdist + i % cdist];
            Group.Barrier();
        }
    }

    /// <summary>Ballot: bit i set when lane i's predicate holds (two barriers).</summary>
    private static int Vote(ArrayView<int> ctl, int lane, bool p)
    {
        ctl[CVote + lane] = p ? 1 : 0;
        Group.Barrier();
        int mask = 0;
        for (int i = 0; i < Streams; i++) mask |= ctl[CVote + i] << i;
        Group.Barrier();
        return mask;
    }

    /// <summary>Exclusive prefix sum over the lanes; <paramref name="total"/> = the sum of all.</summary>
    private static int Scan(ArrayView<int> ctl, int lane, int v, ref int total)
    {
        ctl[CScan + lane] = v;
        Group.Barrier();
        int s = 0, t = 0;
        for (int i = 0; i < Streams; i++)
        {
            int x = ctl[CScan + i];
            t += x;
            if (i < lane) s += x;
        }
        Group.Barrier();
        total = t;
        return s;
    }

    /// <summary>
    /// Refills, in lane order, the lanes (with <paramref name="p"/>) holding fewer than 32 bits: each reads the 32-bit
    /// packet at the cursor plus 4 x the number of refilling lanes below it; then the cursor moves past them all.
    /// </summary>
    private static void LaneRefill(ArrayView<byte> d, ArrayView<int> sc, ArrayView<int> ctl, int lane, bool p)
    {
        int left = sc[BitsLeftOff + lane];
        bool need = p & (left < 32);
        int mask = Vote(ctl, lane, need);
        int cursor = ctl[CCursor];
        if (need)
        {
            int at = cursor + 4 * PopCount((uint)mask & ((1u << lane) - 1u));
            if (at + 4 > ctl[CInEnd]) ctl[CErr] = (int)GDeflateError.Truncated;
            else
            {
                uint v = Read32(d, at);
                uint lo = (uint)sc[BitLoOff + lane];
                uint hi = (uint)sc[BitHiOff + lane];
                if (left == 0) lo = v;
                else
                {
                    lo |= v << left;
                    hi |= v >> (32 - left);
                }
                sc[BitLoOff + lane] = (int)lo;
                sc[BitHiOff + lane] = (int)hi;
                sc[BitsLeftOff + lane] = left + 32;
            }
        }
        Group.Barrier();
        if (lane == 0) ctl[CCursor] = cursor + 4 * PopCount((uint)mask);
        Group.Barrier();
    }

    private static int LanePop(ArrayView<int> sc, ArrayView<int> ctl, int lane, int n)
    {
        var st = new TileState { Idx = lane };
        int v = PopBits(ref st, sc, 0, n);
        if (st.Err != 0) ctl[CErr] = st.Err;
        return v;
    }

    /// <summary>The lane's next symbol from its peeked bits (nothing consumed); -1 (and the error) when none decodes.</summary>
    private static int LaneDecode(ArrayView<int> sc, ArrayView<int> ctl, int lane, bool isDist, ref int len)
    {
        uint bits = (uint)sc[BitLoOff + lane];
        int sym;
        if (isDist) sym = PeekSymbol(sc, 0, bits, DistLutOff, DistLutBits, DistCountOff, DistSymOff, ref len);
        else sym = PeekSymbol(sc, 0, bits, LitLutOff, LitLutBits, LitCountOff, LitSymOff, ref len);
        if (sym < 0) ctl[CErr] = (int)GDeflateError.BadSymbol;
        return sym;
    }

    /// <summary>
    /// Consumes the lane's symbol (<paramref name="clen"/> bits) and its extra bits; returns the distance, or for a
    /// literal/length symbol the output bytes it stands for (literal 1, match its length, end of block 0).
    /// </summary>
    private static int LaneTranslate(ArrayView<int> sc, ArrayView<int> ctl, int lane, int sym, int clen, bool isDist)
    {
        var st = new TileState { Idx = lane };
        int value = 0;
        if (sym < 0) st.Err = (int)GDeflateError.BadSymbol;
        else
        {
            Remove(ref st, sc, 0, clen);
            if (isDist)
            {
                if (sym >= 32) st.Err = (int)GDeflateError.BadDistance;
                else
                {
                    value = sym + 1;
                    if (sym >= 4)
                    {
                        int de = (sym >> 1) - 1;
                        value = ((2 + (sym & 1)) << de) + 1 + PopBits(ref st, sc, 0, de);
                    }
                }
            }
            else if (sym < 256) value = 1;
            else if (sym > 256)
            {
                int ls = sym - 257;
                if (ls >= 29) st.Err = (int)GDeflateError.BadSymbol;
                else if (ls < 8) value = 3 + ls;
                else if (ls == 28) value = 3 + PopBits(ref st, sc, 0, 16); // Deflate64
                else
                {
                    int e = ((ls - 8) >> 2) + 1;
                    value = 3 + ((4 + ((ls - 8) & 3)) << e) + PopBits(ref st, sc, 0, e);
                }
            }
        }
        if (st.Err != 0) ctl[CErr] = st.Err;
        return value;
    }

    private static int PopCount(uint v)
    {
        v -= (v >> 1) & 0x55555555u;
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        return (int)((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
    }

    /// <summary>Index of the k-th (0-based) set bit of <paramref name="mask"/>.</summary>
    private static int NthSetBit(uint mask, int k)
    {
        int result = 0;
        int seen = 0;
        for (int b = 0; b < Streams; b++)
        {
            bool set = ((mask >> b) & 1u) != 0;
            if (set & (seen == k)) result = b;
            if (set) seen++;
        }
        return result;
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
        for (int i = 0; i < 19; i++) Set4(sc, lens, i, 0);
        int k = 0;
        bool reading = (k < ncode) & (st.Err == 0);
        while (reading)
        {
            Set4(sc, lens, Compression.InflateGpu.CodeLengthOrder(k), PopBits(ref st, sc, sb, 3));
            Advance(ref st, d, sc, sb);
            k++;
            reading = (k < ncode) & (st.Err == 0);
        }
        if (st.Err != 0) return;
        int left = Construct(sc, sb, lens, 0, 19, sb + PreCountOff, sb + PreSymOff);
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
                    Set4(sc, lens, index, sym);
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
                            val = Get4(sc, lens, index - 1);
                            rep = 3 + PopBits(ref st, sc, sb, 2);
                        }
                    }
                    else if (sym == 17) rep = 3 + PopBits(ref st, sc, sb, 3);
                    else rep = 11 + PopBits(ref st, sc, sb, 7);
                    // The reference lets a run overrun the lengths (into slack it ignores); so do we.
                    for (int r = 0; r < rep; r++) Set4(sc, lens, index + r, val);
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
        for (int i = 0; i < 144; i++) Set4(sc, lens, i, 8);
        for (int i = 144; i < 256; i++) Set4(sc, lens, i, 9);
        for (int i = 256; i < 280; i++) Set4(sc, lens, i, 7);
        for (int i = 280; i < 288; i++) Set4(sc, lens, i, 8);
        for (int i = 288; i < 320; i++) Set4(sc, lens, i, 5);
        var dummy = new TileState();
        BuildCodes(ref dummy, sc, sb, 288, 32);
    }

    /// <summary>Literal/length and distance codes from the lengths: complete, or incomplete with a single code of length 1.</summary>
    private static void BuildCodes(ref TileState st, ArrayView<int> sc, int sb, int nlit, int ndist)
    {
        int lens = sb + LensOff;
        int left = Construct(sc, sb, lens, 0, nlit, sb + LitCountOff, sb + LitSymOff);
        bool litBad = (left < 0) | ((left > 0) & (nlit != sc[sb + LitCountOff] + sc[sb + LitCountOff + 1]));
        left = Construct(sc, sb, lens, nlit, ndist, sb + DistCountOff, sb + DistSymOff);
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
        int len = 0;
        int result = PeekSymbol(sc, sb, Peek(ref st, sc, sb), lutOff, lutBits, countOff, symOff, ref len);
        if (result < 0) st.Err = (int)GDeflateError.BadSymbol;
        else Remove(ref st, sc, sb, len);
        return result;
    }

    /// <summary>The symbol <paramref name="bits"/> start with and its code length (-1 when no code matches); nothing consumed.</summary>
    private static int PeekSymbol(ArrayView<int> sc, int sb, uint bits, int lutOff, int lutBits, int countOff, int symOff, ref int len)
    {
        int entry = Get16(sc, sb + lutOff, (int)(bits & ((1u << lutBits) - 1u)));
        len = entry & 15;
        int result = -1;
        if (len != 0) result = entry >> 4;
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
                    result = Get16(sc, sb + symOff, index + (code - first));
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
            len = l;
        }
        return result;
    }

    /// <summary>
    /// Canonical decoding tables (puff's construct) from the <paramref name="n"/> 4-bit lengths from entry
    /// <paramref name="first"/> at <paramref name="lengthsAt"/>: 0 complete, &gt; 0 incomplete, &lt; 0 over-subscribed.
    /// </summary>
    private static int Construct(ArrayView<int> sc, int sb, int lengthsAt, int first, int n, int countAt, int symAt)
    {
        int offsAt = sb + OffsOff;
        for (int len = 0; len <= MaxBits; len++) sc[countAt + len] = 0;
        for (int sym = 0; sym < n; sym++)
        {
            int slot = countAt + Get4(sc, lengthsAt, first + sym);
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
                int l = Get4(sc, lengthsAt, first + sym);
                if (l != 0)
                {
                    int at = sc[offsAt + l];
                    Set16(sc, symAt, at, sym);
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
        for (int i = 0; i < size / 2; i++) sc[lutAt + i] = 0;
        int code = 0;
        int index = 0;
        for (int len = 1; len <= lutBits; len++)
        {
            int count = sc[countAt + len];
            for (int k = 0; k < count; k++)
            {
                int rev = 0;
                for (int b = 0; b < len; b++) rev |= ((code >> b) & 1) << (len - 1 - b);
                int entry = (Get16(sc, symAt, index + k) << 4) | len;
                for (int fill = rev; fill < size; fill += 1 << len) Set16(sc, lutAt, fill, entry);
                code++;
            }
            index += count;
            code <<= 1;
        }
    }

    /// <summary>The 16-bit entry <paramref name="i"/> of the packed array at <paramref name="at"/>.</summary>
    private static int Get16(ArrayView<int> sc, int at, int i) => (sc[at + (i >> 1)] >> ((i & 1) << 4)) & 0xFFFF;

    /// <summary>Sets the 16-bit entry <paramref name="i"/> of the packed array at <paramref name="at"/> (one writer at a time).</summary>
    private static void Set16(ArrayView<int> sc, int at, int i, int v)
    {
        int w = at + (i >> 1);
        int sh = (i & 1) << 4;
        sc[w] = (sc[w] & ~(0xFFFF << sh)) | ((v & 0xFFFF) << sh);
    }

    /// <summary>The 4-bit entry <paramref name="i"/> of the packed array at <paramref name="at"/>.</summary>
    private static int Get4(ArrayView<int> sc, int at, int i) => (sc[at + (i >> 3)] >> ((i & 7) << 2)) & 15;

    /// <summary>Sets the 4-bit entry <paramref name="i"/> of the packed array at <paramref name="at"/> (one writer at a time).</summary>
    private static void Set4(ArrayView<int> sc, int at, int i, int v)
    {
        int w = at + (i >> 3);
        int sh = (i & 7) << 2;
        sc[w] = (sc[w] & ~(15 << sh)) | ((v & 15) << sh);
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
