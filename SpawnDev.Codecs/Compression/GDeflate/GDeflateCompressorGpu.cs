// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GDeflate compressor on the accelerator (Microsoft DirectStorage tile streams; GDeflateDecompressorGpu and the
// reference library read what it writes). The input is cut into 64 KB tiles; each tile is parsed by DeflateGpu's LZ77
// matcher (levels 1-2 greedy, 3 lazy; 4 = Huffman only; 0 = stored), its codes are built with the Deflate64 alphabet
// GDeflate uses, and it is written as one block (dynamic Huffman, or stored when that is smaller) swizzled over the 32
// sub-streams (DeflateGpu.EmitGDeflateLanes, 32 lanes per tile). Tiles go in waves: the parse in parallel steps
// (DeflateGpu.ParallelParse: hash chains by 32 lanes, a match search per position, a resumable walk per tile), the plan
// by 256 lanes per tile, the writer by 32; a layout launch places the tiles after the 8-byte header and the tile table,
// then one thread per byte copies them. Nothing but the final length crosses to the host.
//
// The output decodes with Microsoft's reference; it is not byte-identical with it (a different matcher), the format
// does not require that.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression.GDeflate;

/// <summary>GDeflate (DirectStorage tile stream) compressor on the accelerator.</summary>
public sealed class GDeflateCompressorGpu : IDisposable
{
    /// <summary>Input bytes of every tile but the last.</summary>
    public const int TileSize = GDeflateDecompressorGpu.TileSize;
    /// <summary>
    /// Tiles compressed at once when the constructor's wave is 0: 512 on desktop accelerators (~290 MB of working memory),
    /// 64 on browser ones (buffer limits). One thread parses a tile, so parse throughput grows with the wave: measured on
    /// an RTX 4070 with the serial writer, 64 MB of BF16 weights at level 1: 0.008 GB/s at 64 tiles, 0.034 at 512.
    /// </summary>
    public const int DesktopWave = 512, BrowserWave = 64;
    /// <summary>Most tiles a stream can hold (16-bit tile count).</summary>
    public const int MaxTiles = 65535;
    /// <summary>
    /// Positions per tile one match launch searches, and one walk launch advances (a search walks at most 64 chain links
    /// of at most 258 compares).
    /// </summary>
    public const int ParseChunk = 8192;

    // prm ints: input length, tiles, level, scratch ints per tile, slot bytes per tile, running output position, failed
    // flag.
    internal const int PLength = 0, PTiles = 1, PLevel = 2, PScratchPer = 3, PSlot = 4, PPos = 5, PFailed = 6;
    internal const int PrmInts = 8;

    private readonly Accelerator _accelerator;
    private readonly int _wave;
    private readonly LaunchPacer _pacer;
    private readonly int _scratchPer, _slotBytes;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _init;
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _chains;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int> _match;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _walk;
    private readonly Action<KernelConfig, ArrayView<int>, ArrayView<int>, int> _plan;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _literals;
    private readonly int _planLanes;
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _emit;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _copy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _header;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm, _scratch, _offsets;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _slots;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>
    /// Compiles the kernels and allocates one wave's working memory (~570 KB per tile of <paramref name="wave"/>; 0 picks
    /// <see cref="DesktopWave"/> or <see cref="BrowserWave"/> by accelerator).
    /// </summary>
    public GDeflateCompressorGpu(Accelerator accelerator, int wave = 0)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (wave == 0)
        {
            bool browser = accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm;
            wave = browser ? BrowserWave : DesktopWave;
        }
        if (wave < 1 || wave > 4096) throw new ArgumentOutOfRangeException(nameof(wave));
        _accelerator = accelerator;
        _wave = wave;
        _pacer = new LaunchPacer(accelerator);
        _scratchPer = DeflateGpu.ScratchInts(TileSize) + DeflateGpu.GDeflateWriterInts;
        // A multiple of 4: every packet of a tile is then one aligned word that only its lane writes.
        _slotBytes = (DeflateGpu.MaxGDeflateTileOutput(TileSize) + 3) & ~3;
        _init = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(InitKernel);
        _chains = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(ChainsKernel);
        _match = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(MatchKernel);
        _walk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(WalkKernel);
        _plan = accelerator.LoadStreamKernel<ArrayView<int>, ArrayView<int>, int>(PlanKernel);
        _planLanes = Math.Min(PlanLanes, accelerator.MaxNumThreadsPerGroup);
        _literals = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(LiteralsKernel);
        _emit = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(EmitKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(LayoutKernel);
        _copy = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(CopyKernel);
        _header = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(HeaderKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _scratch = accelerator.Allocate1D<int>((long)wave * _scratchPer);
        _offsets = accelerator.Allocate1D<int>(wave);
        _slots = accelerator.Allocate1D<byte>((long)wave * _slotBytes);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    /// <summary>Largest stream <paramref name="length"/> input bytes can produce.</summary>
    public static long GetMaxCompressedSize(long length)
    {
        long tiles = Math.Max(1, (length + TileSize - 1) / TileSize);
        return 8 + 4 * tiles + tiles * DeflateGpu.MaxGDeflateTileOutput(TileSize);
    }

    // ------------------------------------------------------------------
    // Overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Compresses the first <paramref name="length"/> bytes (at least 1) of <paramref name="input"/> into a GDeflate tile
    /// stream at the start of <paramref name="output"/> (at least <see cref="GetMaxCompressedSize"/> bytes); returns the
    /// stream's length. <paramref name="level"/>: 0 stored, 1-2 greedy, 3 lazy, 4 Huffman only.
    /// </summary>
    public async Task<int> CompressAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, int level = 2)
    {
        if (length < 1 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length} (GDeflate streams hold at least one byte).");
        if (level < 0 || level > DeflateGpu.LevelHuffmanOnly) throw new ArgumentOutOfRangeException(nameof(level));
        int tiles = (int)((length + (long)TileSize - 1) / TileSize);
        if (tiles > MaxTiles) throw new ArgumentOutOfRangeException(nameof(length), $"At most {MaxTiles} tiles ({(long)MaxTiles * TileSize} bytes) per stream.");
        long cap = GetMaxCompressedSize(length);
        if (cap > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "The stream must stay under 2 GiB per call.");
        if (output.Length < cap) throw new ArgumentException($"The output holds {output.Length} bytes; {cap} are needed (GetMaxCompressedSize).", nameof(output));

        var prm = new int[PrmInts];
        prm[PLength] = length;
        prm[PTiles] = tiles;
        prm[PLevel] = level;
        prm[PScratchPer] = _scratchPer;
        prm[PSlot] = _slotBytes;
        prm[PPos] = 8 + 4 * tiles;
        _prm.View.CopyFromCPU(prm);

        var data = input.SubView(0, length);
        _header(1, output, _prm.View);
        int parseSteps = (TileSize + ParseChunk - 1) / ParseChunk;
        for (int first = 0; first < tiles; first += _wave)
        {
            int count = Math.Min(_wave, tiles - first);
            _init(count * DeflateGpu.HashSize, _scratch.View, _prm.View, first);
            if (level == DeflateGpu.LevelHuffmanOnly) _literals(count * TileSize, data, _scratch.View, _prm.View, first);
            else if (level > 0)
            {
                // The parse in parallel steps (DeflateGpu.ParallelParse): hash chains, every position's match, the walk.
                _chains(new KernelConfig(count, DeflateGpu.ChainLanes), data, _scratch.View, _prm.View, first);
                for (int from = 0; from < TileSize; from += ParseChunk)
                {
                    _match(count * ParseChunk, data, _scratch.View, _prm.View, first, from);
                    await _pacer.LaunchedAsync();
                }
                for (int i = 0; i < parseSteps; i++)
                {
                    _walk(count, data, _scratch.View, _prm.View, first);
                    await _pacer.LaunchedAsync();
                }
            }
            _plan(new KernelConfig(count, _planLanes), _scratch.View, _prm.View, first);
            _emit(new KernelConfig(count, EmitLanes), data, _slots.View, _scratch.View, _prm.View, first);
            _layout(1, output, _scratch.View, _offsets.View, _prm.View, first, count);
            _copy(count * _slotBytes, _slots.View, output, _scratch.View, _offsets.View, _prm.View, first);
            await _pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
        var result = await _prm.View.SubView(PPos, 2).CopyToHostAsync();
        if (result[1] != 0) throw new InvalidOperationException("GDeflate compressor: a tile overflowed its slot (bug).");
        return result[0];
    }

    /// <summary>Compresses <paramref name="data"/> (at least 1 byte) into a new GDeflate stream.</summary>
    public async Task<byte[]> CompressAsync(ReadOnlyMemory<byte> data, int level = 2)
    {
        var src = _stageIn.Ensure(Math.Max(1, data.Length));
        var dst = _stageOut.Ensure(GetMaxCompressedSize(data.Length));
        if (data.Length > 0) DeviceTransfer.Upload(src, data.Span);
        int n = await CompressAsync(src, data.Length, dst, level);
        return await DeviceTransfer.DownloadAsync(dst, n);
    }

    /// <summary>Compresses the bytes of <paramref name="data"/> into a new GDeflate stream (never on the .NET heap); the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> CompressAsync(TypedArray data, int level = 2)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.ByteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(data), "At most 2^31 - 1 bytes per call.");
        int length = (int)data.ByteLength;
        var src = _stageIn.Ensure(Math.Max(1, length));
        var dst = _stageOut.Ensure(GetMaxCompressedSize(length));
        if (length > 0) DeviceTransfer.Upload(src, data);
        int n = await CompressAsync(src, length, dst, level);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, n);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    private static int TileLength(ArrayView<int> prm, int tile)
    {
        int len = prm[PLength] - tile * TileSize;
        if (len > TileSize) len = TileSize;
        return len;
    }

    private static void InitKernel(Index1D idx, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        int b = idx.X / DeflateGpu.HashSize;
        int k = idx.X - b * DeflateGpu.HashSize;
        DeflateGpu.InitBlockThread(k, TileLength(prm, first + b), prm[PLevel], scratch, b * prm[PScratchPer], TileSize);
    }

    /// <summary>One group of DeflateGpu.ChainLanes lanes per tile of the wave: its hash chains.</summary>
    private static void ChainsKernel(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        var hs = SharedMemory.Allocate<int>(DeflateGpu.ChainLanes);
        int b = Grid.IdxX;
        int tile = first + b;
        DeflateGpu.BuildChainsLanes(data, tile * TileSize, TileLength(prm, tile), scratch, b * prm[PScratchPer], TileSize, hs, Group.IdxX);
    }

    /// <summary>One thread per position of positions [from, from + ParseChunk) of the wave's tiles: its best match.</summary>
    private static void MatchKernel(Index1D i, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int first, int from)
    {
        int b = i.X / ParseChunk;
        int k = from + (i.X - b * ParseChunk);
        int tile = first + b;
        DeflateGpu.FindMatchThread(data, tile * TileSize, TileLength(prm, tile), k, prm[PLevel], scratch, b * prm[PScratchPer], TileSize);
    }

    /// <summary>One thread per tile of the wave: the next ParseChunk positions of its greedy or lazy parse.</summary>
    private static void WalkKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        int tile = first + idx.X;
        DeflateGpu.WalkParseStep(data, tile * TileSize, TileLength(prm, tile), prm[PLevel], scratch, idx.X * prm[PScratchPer], TileSize,
            ParseChunk);
    }

    /// <summary>Threads per tile when planning (they count its symbols).</summary>
    private const int PlanLanes = 256;

    /// <summary>One group per tile of the wave: its codes and the stored / static / dynamic decision.</summary>
    private static void PlanKernel(ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        var h = SharedMemory.Allocate<int>(DeflateGpu.GDeflatePlanSharedInts);
        int b = Grid.IdxX;
        DeflateGpu.PlanGDeflateLanes(TileLength(prm, first + b), prm[PLevel], scratch, b * prm[PScratchPer], TileSize, h,
            Group.IdxX, Group.DimX);
    }

    /// <summary>Level 4: one thread per byte of the wave's tiles writes its literal token (no parse).</summary>
    private static void LiteralsKernel(Index1D i, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        int b = i.X / TileSize;
        int k = i.X - b * TileSize;
        int tile = first + b;
        DeflateGpu.LiteralTokensThread(data, tile * TileSize, TileLength(prm, tile), k, scratch, b * prm[PScratchPer], TileSize);
    }

    /// <summary>Threads per tile when writing: one per GDeflate sub-stream.</summary>
    private const int EmitLanes = 32;

    /// <summary>One group of 32 lanes per tile of the wave: the whole tile as one GDeflate stream into its slot.</summary>
    private static void EmitKernel(ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        var w = SharedMemory.Allocate<int>(DeflateGpu.GDeflateLanesSharedInts);
        int b = Grid.IdxX;
        int tile = first + b;
        int len = TileLength(prm, tile);
        DeflateGpu.EmitGDeflateLanes(data, tile * TileSize, len, slots, b * prm[PSlot], prm[PSlot], scratch, b * prm[PScratchPer], TileSize,
            w, Group.IdxX);
    }

    /// <summary>
    /// One thread: places the wave's tiles after the running position and writes their tile table entries (entry 0 = the
    /// last tile's size, entry i &gt; 0 = tile i's offset from the data start).
    /// </summary>
    private static void LayoutKernel(Index1D _, ArrayView<byte> output, ArrayView<int> scratch, ArrayView<int> offsets, ArrayView<int> prm,
        int first, int count)
    {
        int tiles = prm[PTiles];
        int dataStart = 8 + 4 * tiles;
        int pos = prm[PPos];
        int failed = prm[PFailed];
        for (int b = 0; b < count; b++)
        {
            int tile = first + b;
            int size = DeflateGpu.BlockOutput(scratch, b * prm[PScratchPer], TileSize);
            if (size < 0)
            {
                failed = 1;
                size = 0;
            }
            offsets[b] = pos;
            if (tile > 0) PutLe32(output, 8 + 4 * tile, pos - dataStart);
            if (tile == tiles - 1) PutLe32(output, 8, size);
            pos += size;
        }
        prm[PPos] = pos;
        prm[PFailed] = failed;
    }

    private static void PutLe32(ArrayView<byte> o, int at, int v)
    {
        o[at] = (byte)v;
        o[at + 1] = (byte)(v >> 8);
        o[at + 2] = (byte)(v >> 16);
        o[at + 3] = (byte)(v >> 24);
    }

    /// <summary>One thread per slot byte of the wave: the tile's bytes at its place.</summary>
    private static void CopyKernel(Index1D i, ArrayView<byte> slots, ArrayView<byte> output, ArrayView<int> scratch, ArrayView<int> offsets,
        ArrayView<int> prm, int first)
    {
        int slot = prm[PSlot];
        int b = i.X / slot;
        int k = i.X - b * slot;
        int size = DeflateGpu.BlockOutput(scratch, b * prm[PScratchPer], TileSize);
        if (k < size) output[offsets[b] + k] = slots[i];
    }

    /// <summary>One thread: the 8-byte tile stream header (codec id 4, its magic, tile count, tile size index 1, last tile size).</summary>
    private static void HeaderKernel(Index1D _, ArrayView<byte> output, ArrayView<int> prm)
    {
        int tiles = prm[PTiles];
        int last = prm[PLength] - (tiles - 1) * TileSize;
        if (last == TileSize) last = 0;
        output[0] = (byte)GDeflateDecompressorGpu.CodecId;
        output[1] = (byte)(GDeflateDecompressorGpu.CodecId ^ 0xFF);
        output[2] = (byte)tiles;
        output[3] = (byte)(tiles >> 8);
        uint bits = 1u | ((uint)last << 2);
        output[4] = (byte)bits;
        output[5] = (byte)(bits >> 8);
        output[6] = (byte)(bits >> 16);
        output[7] = (byte)(bits >> 24);
    }

    public void Dispose()
    {
        _prm.Dispose();
        _scratch.Dispose();
        _offsets.Dispose();
        _slots.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
