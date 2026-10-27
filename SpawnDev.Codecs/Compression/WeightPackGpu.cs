// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Lossless packing of numeric tensors (model weights: BF16 / FP16 / FP32 / any fixed-size element) on the accelerator.
// The redundancy of float weights is in the BYTE DISTRIBUTION of each byte position, not in repeats: the exponent byte
// of a BF16 weight carries ~3 bits, the low mantissa bytes of upcast weights a few more. So:
//   1. byte grouping (ShuffleFilterGpu): plane j holds byte j of every element;
//   2. each plane is cut into ChunkBytes chunks, and each chunk is ONE complete raw DEFLATE stream at the Huffman-only
//      level (DeflateGpu.LevelHuffmanOnly: the chunk's own dynamic Huffman code, or a stored block when that is smaller);
//   3. a small header and a table of chunk lengths.
// Measured on our own models (Plans/PLAN-Data-Compression.md C4): BF16 0.677, FP16 0.662, FP32 0.833 of the original,
// against ZipNN 0.719 / 0.675 / 0.771 and zstd -3 0.788 / 0.772 / 0.853 on the same samples (FP32: ZipNN's zstd stage
// finds long-range repeats this coder does not; recorded, not hidden).
//
// Every chunk compresses and decompresses independently. Compression is parallel inside each chunk too: 256 threads
// count its bytes, one thread builds its Huffman code (DeflateGpu.PlanHuffmanOnly: the code construction of the
// Huffman-only level, so the bytes are exactly a DeflateGpu Huffman-only block), and 256 threads write its codes at bit
// offsets from a prefix sum of the code lengths. Decompression is one thread per chunk (InflateGpu raw, one launch per
// chunk) in waves, with the chunk offsets computed on the accelerator from the table. Every chunk is a standard DEFLATE stream, so any zlib
// (raw inflate, wbits -15) reads the format with the table.
//
// Format "SDWP" version 1, little-endian:
//   0  "SDWP"    4  version (1)    5  element size (1..16)    6  flags (0)    7  reserved (0)
//   8  element count (u64)    16  chunk bytes (u32)    20  chunks per plane (u32)
//   24 table: elementSize * chunksPerPlane u32 = compressed length of each chunk, plane-major
//   then the chunks, plane-major, back to back.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression;

/// <summary>Lossless tensor (model weight) packer: byte planes + chunked Huffman-only DEFLATE, on the accelerator.</summary>
public sealed class WeightPackGpu : IDisposable
{
    /// <summary>Bytes of one plane chunk (one independent DEFLATE stream).</summary>
    public const int ChunkBytes = 65536;
    /// <summary>Header bytes before the chunk table.</summary>
    public const int HeaderBytes = 24;
    /// <summary>Chunks compressed at once (~84 KB of working memory each: ~86 MB).</summary>
    public const int CompressWave = 1024;
    /// <summary>Most threads per chunk in the counting and emitting kernels (fewer where the device's groups are smaller).</summary>
    private const int MaxPackLanes = 256;
    /// <summary>Default chunks decompressed at once (each needs InflateGpu.ScratchInts ints: ~75 MB at 8192).</summary>
    public const int DecompressWave = 8192;
    /// <summary>
    /// Output budget of one decompress launch per chunk: a whole chunk and its end (a chunk is at most ChunkBytes, so a
    /// launch stays bounded however big the tensor; one launch per wave instead of one per 2 KB of input).
    /// </summary>
    private const int DecodeOutputBudget = ChunkBytes + 1;
    /// <summary>Table entries one layout launch walks.</summary>
    public const int LayoutChunk = 65536;

    private const uint Magic = 0x50574453; // "SDWP"
    private const int SlotBytes = ChunkBytes + 64;
    private const int SlotWords = SlotBytes / 4;

    // prm ints: element size, elements (plane length), chunks per plane, total chunks, running output position (compress)
    // or running input position (decompress), the input length, error flag + where.
    private const int PElem = 0, PPlane = 1, PPerPlane = 2, PChunks = 3, PPos = 4, PLength = 5, PError = 6, PErrorAt = 7, PLayoutAt = 8;
    private const int PrmInts = 12;

    private readonly Accelerator _accelerator;
    private readonly int _decompressWave;
    private readonly LaunchPacer _pacer;
    private readonly ShuffleFilterGpu _shuffle;
    private readonly int _packLanes;
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<int>, int, int> _cCount;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int> _cPlan;
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<uint>, ArrayView<int>, int, int> _cEmit;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _cLayout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _cCopy;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _cHeader;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _dLayout;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _dInit;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _dStep;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _dCheck;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm, _counts, _tails, _dScratch;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _hdr;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _words;
    private readonly StageBuffer<byte> _planes, _stageIn, _stageOut;
    private readonly StageBuffer<int> _offsets;

    /// <summary>
    /// Compiles the kernels and allocates the working memory: ~86 MB to compress, ~9.5 KB per chunk of
    /// <paramref name="decompressWave"/> to decompress (~78 MB at the default 8192).
    /// </summary>
    public WeightPackGpu(Accelerator accelerator, int decompressWave = DecompressWave)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (decompressWave < 1 || decompressWave > 65536) throw new ArgumentOutOfRangeException(nameof(decompressWave));
        _accelerator = accelerator;
        _decompressWave = decompressWave;
        _pacer = new LaunchPacer(accelerator);
        _shuffle = new ShuffleFilterGpu(accelerator);
        // A power of two that divides ChunkBytes: 256, or the device's largest group (64 on the CPU accelerator).
        _packLanes = MaxPackLanes;
        while (_packLanes > accelerator.MaxNumThreadsPerGroup) _packLanes >>= 1;
        _cCount = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<int>, int, int>(CompressCountKernel);
        _cPlan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, int>(CompressPlanKernel);
        _cEmit = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<int>, ArrayView<byte>, ArrayView<uint>, ArrayView<int>, int, int>(CompressEmitKernel);
        _cLayout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(CompressLayoutKernel);
        _cCopy = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<uint>, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(CompressCopyKernel);
        _cHeader = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(CompressHeaderKernel);
        _dLayout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(DecompressLayoutKernel);
        _dInit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(DecompressInitKernel);
        _dStep = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(DecompressStepKernel);
        _dCheck = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(DecompressCheckKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _counts = accelerator.Allocate1D<int>((long)CompressWave * 256);
        _tails = accelerator.Allocate1D<int>((long)CompressWave * DeflateGpu.PlanTailInts);
        _hdr = accelerator.Allocate1D<byte>((long)CompressWave * DeflateGpu.MaxDynamicHeaderBytes);
        _words = accelerator.Allocate1D<uint>((long)CompressWave * SlotWords);
        _dScratch = accelerator.Allocate1D<int>((long)_decompressWave * InflateGpu.ScratchInts);
        _planes = new StageBuffer<byte>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _offsets = new StageBuffer<int>(accelerator);
    }

    /// <summary>Chunks per plane of <paramref name="elements"/> elements.</summary>
    public static int ChunksPerPlane(long elements) => (int)Math.Max(1, (elements + ChunkBytes - 1) / ChunkBytes);

    /// <summary>Largest packed size of <paramref name="length"/> bytes of <paramref name="elementSize"/>-byte elements.</summary>
    public static long GetMaxPackedSize(long length, int elementSize)
    {
        long elements = length / elementSize;
        long chunks = (long)elementSize * ChunksPerPlane(elements);
        return HeaderBytes + 4 * chunks + length + chunks * 16;
    }

    // ------------------------------------------------------------------
    // Pack
    // ------------------------------------------------------------------

    /// <summary>
    /// Packs the first <paramref name="length"/> bytes of <paramref name="input"/> (whole <paramref name="elementSize"/>-byte
    /// elements) into <paramref name="output"/> (at least <see cref="GetMaxPackedSize"/> bytes); returns the packed length.
    /// </summary>
    public async Task<int> PackAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, int elementSize, ArrayView1D<byte, Stride1D.Dense> output)
    {
        if (elementSize < 1 || elementSize > 16) throw new ArgumentOutOfRangeException(nameof(elementSize), "Element size must be 1..16 bytes.");
        if (length < 0 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length}.");
        if (length % elementSize != 0) throw new ArgumentException($"{length} bytes is not a whole number of {elementSize}-byte elements.", nameof(length));
        long cap = GetMaxPackedSize(length, elementSize);
        if (cap > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "The packed form must stay under 2 GiB per call.");
        if (output.Length < cap) throw new ArgumentException($"The output holds {output.Length} bytes; {cap} are needed (GetMaxPackedSize).", nameof(output));

        int elements = length / elementSize;
        int perPlane = ChunksPerPlane(elements);
        int chunks = elementSize * perPlane;
        var planes = _planes.Ensure(Math.Max(1, length));
        if (length > 0) await _shuffle.ShuffleAsync(input, planes, length, elementSize);

        var prm = new int[PrmInts];
        prm[PElem] = elementSize;
        prm[PPlane] = elements;
        prm[PPerPlane] = perPlane;
        prm[PChunks] = chunks;
        prm[PPos] = HeaderBytes + 4 * chunks;
        prm[PLength] = length;
        _prm.View.CopyFromCPU(prm);

        for (int first = 0; first < chunks; first += CompressWave)
        {
            int count = Math.Min(CompressWave, chunks - first);
            var group = new KernelConfig(count, _packLanes);
            _cCount(group, planes, _counts.View, _words.View, _prm.View, first, _packLanes);
            _cPlan(count, _counts.View, _tails.View, _hdr.View, _prm.View, first);
            _cEmit(group, planes, _tails.View, _hdr.View, _words.View, _prm.View, first, _packLanes);
            _cLayout(1, output, _tails.View, _prm.View, OffsetsFor(count), first);
            _cCopy(count * SlotBytes, planes, _words.View, _hdr.View, output, _tails.View, OffsetsFor(count), _prm.View, first);
            await _pacer.LaunchedAsync();
        }
        _cHeader(1, output, _prm.View);
        await _accelerator.SynchronizeAsync();
        var r = await _prm.View.CopyToHostAsync();
        if (r[PError] != 0) throw new InvalidOperationException($"WeightPack: chunk {r[PErrorAt]} did not fit its slot (bug).");
        return r[PPos];
    }

    private ArrayView<int> OffsetsFor(int count) => _offsets.Ensure(CompressWave).SubView(0, CompressWave);

    /// <summary>Packs <paramref name="data"/> (whole <paramref name="elementSize"/>-byte elements).</summary>
    public async Task<byte[]> PackAsync(ReadOnlyMemory<byte> data, int elementSize)
    {
        var src = _stageIn.Ensure(Math.Max(1, data.Length));
        var dst = _stageOut.Ensure(GetMaxPackedSize(data.Length, Math.Max(1, elementSize)));
        if (data.Length > 0) DeviceTransfer.Upload(src, data.Span);
        int n = await PackAsync(src, data.Length, elementSize, dst);
        return await DeviceTransfer.DownloadAsync(dst, n);
    }

    /// <summary>Packs the bytes of <paramref name="data"/> (never on the .NET heap); the caller disposes the result.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> PackAsync(TypedArray data, int elementSize)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.ByteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(data), "At most 2^31 - 1 bytes per call.");
        int length = (int)data.ByteLength;
        var src = _stageIn.Ensure(Math.Max(1, length));
        var dst = _stageOut.Ensure(GetMaxPackedSize(length, Math.Max(1, elementSize)));
        if (length > 0) DeviceTransfer.Upload(src, data);
        int n = await PackAsync(src, length, elementSize, dst);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, n);
    }

    // ------------------------------------------------------------------
    // Unpack
    // ------------------------------------------------------------------

    /// <summary>The unpacked byte length of a packed blob (reads its 24-byte header).</summary>
    public static long UnpackedLength(ReadOnlySpan<byte> header)
    {
        ParseHeader(header, out int elementSize, out long elements, out _, out _);
        return elements * elementSize;
    }

    private static void ParseHeader(ReadOnlySpan<byte> h, out int elementSize, out long elements, out int chunkBytes, out int perPlane)
    {
        if (h.Length < HeaderBytes) throw new InvalidDataException($"WeightPack: {h.Length} bytes is shorter than the header.");
        if (BitConverter.ToUInt32(h.Slice(0, 4)) != Magic) throw new InvalidDataException("WeightPack: bad magic (not SDWP).");
        if (h[4] != 1) throw new InvalidDataException($"WeightPack: unsupported version {h[4]}.");
        elementSize = h[5];
        if (elementSize < 1 || elementSize > 16) throw new InvalidDataException($"WeightPack: bad element size {elementSize}.");
        if (h[6] != 0 || h[7] != 0) throw new InvalidDataException("WeightPack: unknown flags.");
        elements = BitConverter.ToInt64(h.Slice(8, 8));
        chunkBytes = BitConverter.ToInt32(h.Slice(16, 4));
        perPlane = BitConverter.ToInt32(h.Slice(20, 4));
        if (elements < 0 || elements * elementSize > int.MaxValue) throw new InvalidDataException($"WeightPack: {elements} elements is out of range.");
        if (chunkBytes != ChunkBytes) throw new InvalidDataException($"WeightPack: chunk size {chunkBytes} is not {ChunkBytes}.");
        if (perPlane != ChunksPerPlane(elements)) throw new InvalidDataException($"WeightPack: {perPlane} chunks per plane for {elements} elements.");
    }

    /// <summary>
    /// Unpacks the packed blob in the first <paramref name="length"/> bytes of <paramref name="input"/> into
    /// <paramref name="output"/>; returns the unpacked length. Throws <see cref="InvalidDataException"/> for a damaged blob.
    /// </summary>
    public async Task<int> UnpackAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output)
    {
        if (length < HeaderBytes || length > input.Length) throw new InvalidDataException($"WeightPack: {length} bytes is shorter than the header.");
        var header = await input.SubView(0, HeaderBytes).CopyToHostAsync();
        ParseHeader(header, out int elementSize, out long elementsL, out _, out int perPlane);
        int elements = (int)elementsL;
        int total = elements * elementSize;
        if (output.Length < total) throw new ArgumentException($"The output holds {output.Length} bytes; {total} are needed.", nameof(output));
        int chunks = elementSize * perPlane;
        if (HeaderBytes + 4L * chunks > length) throw new InvalidDataException("WeightPack: the chunk table is truncated.");
        if (total == 0) return 0;

        var prm = new int[PrmInts];
        prm[PElem] = elementSize;
        prm[PPlane] = elements;
        prm[PPerPlane] = perPlane;
        prm[PChunks] = chunks;
        prm[PPos] = HeaderBytes + 4 * chunks;
        prm[PLength] = length;
        _prm.View.CopyFromCPU(prm);
        var offsets = _offsets.Ensure(chunks);
        var data = input.SubView(0, length);
        for (int s = 0; s * LayoutChunk < chunks; s++)
        {
            _dLayout(1, data, offsets, _prm.View);
            await _pacer.LaunchedAsync();
        }

        var planes = _planes.Ensure(total);
        for (int first = 0; first < chunks; first += _decompressWave)
        {
            int count = Math.Min(_decompressWave, chunks - first);
            _dInit(count, offsets, _dScratch.View, _prm.View, first);
            // One launch inflates every chunk of the wave to its end (DecodeOutputBudget); the check refuses a chunk
            // that did not finish.
            _dStep(count, data, planes, _dScratch.View, _prm.View, first);
            await _pacer.LaunchedAsync();
            _dCheck(count, offsets, _dScratch.View, _prm.View, first);
        }
        await _shuffle.UnshuffleAsync(planes, output, total, elementSize);
        await _accelerator.SynchronizeAsync();
        var r = await _prm.View.CopyToHostAsync();
        if (r[PError] != 0) throw new InvalidDataException($"WeightPack: chunk {r[PErrorAt]} is damaged ({(r[PError] == 1 ? "the table runs past the data" : "it does not inflate to its size")}).");
        return total;
    }

    /// <summary>Unpacks <paramref name="packed"/> into a new array.</summary>
    public async Task<byte[]> UnpackAsync(ReadOnlyMemory<byte> packed)
    {
        long total = UnpackedLength(packed.Span);
        var src = _stageIn.Ensure(Math.Max(1, packed.Length));
        DeviceTransfer.Upload(src, packed.Span);
        var dst = _stageOut.Ensure(Math.Max(1, total));
        int n = await UnpackAsync(src, packed.Length, dst);
        return await DeviceTransfer.DownloadAsync(dst, n);
    }

    /// <summary>Unpacks the bytes of <paramref name="packed"/> into a new Uint8Array (never on the .NET heap); the caller disposes it.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> UnpackAsync(TypedArray packed)
    {
        ArgumentNullException.ThrowIfNull(packed);
        if (packed.ByteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(packed), "At most 2^31 - 1 bytes per call.");
        int length = (int)packed.ByteLength;
        var src = _stageIn.Ensure(Math.Max(1, length));
        DeviceTransfer.Upload(src, packed);
        var header = await src.SubView(0, Math.Min(length, HeaderBytes)).CopyToHostAsync();
        long total = UnpackedLength(header);
        var dst = _stageOut.Ensure(Math.Max(1, total));
        int n = await UnpackAsync(src, length, dst);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, n);
    }

    // ------------------------------------------------------------------
    // Kernels: compression (one thread per chunk of the wave; chunk g = plane g / perPlane, chunk g % perPlane)
    // ------------------------------------------------------------------

    private static int ChunkStart(ArrayView<int> prm, int g)
    {
        int perPlane = prm[PPerPlane];
        int plane = g / perPlane;
        int c = g - plane * perPlane;
        return plane * prm[PPlane] + c * ChunkBytes;
    }

    private static int ChunkLength(ArrayView<int> prm, int g)
    {
        int perPlane = prm[PPerPlane];
        int c = g - g / perPlane * perPlane;
        int len = prm[PPlane] - c * ChunkBytes;
        if (len > ChunkBytes) len = ChunkBytes;
        if (len < 0) len = 0;
        return len;
    }

    /// <summary>
    /// <paramref name="lanes"/> threads per chunk of the wave: the chunk's byte counts (shared-memory histogram) into counts,
    /// and its output words cleared for the emit's ORs.
    /// </summary>
    private static void CompressCountKernel(ArrayView<byte> planes, ArrayView<int> counts, ArrayView<uint> words, ArrayView<int> prm, int first, int lanes)
    {
        var hist = SharedMemory.Allocate<int>(256);
        int lane = Group.IdxX;
        int b = Grid.IdxX;
        int g = first + b;
        for (int v = lane; v < 256; v += lanes) hist[v] = 0;
        for (int w = lane; w < SlotWords; w += lanes) words[(long)b * SlotWords + w] = 0;
        Group.Barrier();
        int start = ChunkStart(prm, g);
        int len = ChunkLength(prm, g);
        for (int i = lane; i < len; i += lanes) Atomic.Add(ref hist[(int)planes[start + i]], 1);
        Group.Barrier();
        for (int v = lane; v < 256; v += lanes) counts[b * 256 + v] = hist[v];
    }

    /// <summary>One thread per chunk of the wave: its Huffman code, the stored-or-dynamic decision and the block header.</summary>
    private static void CompressPlanKernel(Index1D i, ArrayView<int> counts, ArrayView<int> tails, ArrayView<byte> hdr, ArrayView<int> prm, int first)
    {
        int g = first + i.X;
        DeflateGpu.PlanHuffmanOnly(counts, i.X * 256, ChunkLength(prm, g), tails, i.X * DeflateGpu.PlanTailInts,
            hdr, i.X * DeflateGpu.MaxDynamicHeaderBytes);
    }

    /// <summary>
    /// <paramref name="lanes"/> threads per chunk of the wave (dynamic blocks): each sums the code lengths of its ChunkBytes / lanes bytes, a
    /// prefix sum over the threads gives its first bit, and it ORs its codes into the chunk's words; thread 0 adds the
    /// header bits, the last thread the end-of-block code.
    /// </summary>
    private static void CompressEmitKernel(ArrayView<byte> planes, ArrayView<int> tails, ArrayView<byte> hdr, ArrayView<uint> words,
        ArrayView<int> prm, int first, int lanes)
    {
        var codes = SharedMemory.Allocate<int>(256);
        var lens = SharedMemory.Allocate<int>(256);
        var part = SharedMemory.Allocate<int>(MaxPackLanes);
        int lane = Group.IdxX;
        int b = Grid.IdxX;
        int g = first + b;
        int tail = b * DeflateGpu.PlanTailInts;
        for (int v = lane; v < 256; v += lanes)
        {
            codes[v] = tails[tail + DeflateGpu.PlanLitCodeOff + v];
            lens[v] = tails[tail + DeflateGpu.PlanLitLenOff + v];
        }
        Group.Barrier();
        int start = ChunkStart(prm, g);
        int len = ChunkLength(prm, g);
        int per = ChunkBytes / lanes;
        int from = lane * per;
        int to = from + per;
        if (to > len) to = len;
        int sum = 0;
        for (int k = from; k < to; k++) sum += lens[(int)planes[start + k]];
        part[lane] = sum;
        Group.Barrier();
        int before = 0;
        int all = 0;
        for (int j = 0; j < lanes; j++)
        {
            int x = part[j];
            all += x;
            if (j < lane) before += x;
        }
        bool dynamic = (len > 0) & (DeflateGpu.PlannedKind(tails, tail) == 0);
        if (dynamic)
        {
            long wbase = (long)b * SlotWords;
            int headerBits = DeflateGpu.PlannedHeaderBits(tails, tail);
            int bit = headerBits + before;
            for (int k = from; k < to; k++)
            {
                int v = planes[start + k];
                OrBits(words, wbase, bit, (uint)codes[v], lens[v]);
                bit += lens[v];
            }
            if (lane == 0)
            {
                int hb = b * DeflateGpu.MaxDynamicHeaderBytes;
                for (int k = 0; k * 8 < headerBits; k++)
                {
                    int n = headerBits - k * 8;
                    if (n > 8) n = 8;
                    OrBits(words, wbase, k * 8, (uint)hdr[hb + k] & ((1u << n) - 1u), n);
                }
            }
            if (lane == lanes - 1)
                OrBits(words, wbase, headerBits + all, (uint)tails[tail + DeflateGpu.PlanLitCodeOff + 256], tails[tail + DeflateGpu.PlanLitLenOff + 256]);
        }
    }

    /// <summary>ORs the <paramref name="n"/> (at most 25) low bits of <paramref name="code"/> in at bit <paramref name="bit"/> (LSB first).</summary>
    private static void OrBits(ArrayView<uint> words, long wbase, int bit, uint code, int n)
    {
        int w = bit >> 5;
        int o = bit & 31;
        Atomic.Or(ref words[wbase + w], code << o);
        if (o + n > 32) Atomic.Or(ref words[wbase + w + 1], code >> (32 - o));
    }

    /// <summary>One thread: the wave's chunk lengths into the table and their output offsets after the running position.</summary>
    private static void CompressLayoutKernel(Index1D idx, ArrayView<byte> output, ArrayView<int> tails, ArrayView<int> prm, ArrayView<int> offsets, int first)
    {
        int chunks = prm[PChunks];
        int count = chunks - first;
        if (count > CompressWave) count = CompressWave;
        int pos = prm[PPos];
        for (int b = 0; b < count; b++)
        {
            int n = DeflateGpu.PlannedBytes(tails, b * DeflateGpu.PlanTailInts);
            if (n > SlotBytes)
            {
                prm[PError] = 1;
                prm[PErrorAt] = first + b;
                n = 0;
            }
            offsets[b] = pos;
            int t = HeaderBytes + 4 * (first + b);
            output[t] = (byte)n;
            output[t + 1] = (byte)(n >> 8);
            output[t + 2] = (byte)(n >> 16);
            output[t + 3] = (byte)(n >> 24);
            pos += n;
        }
        prm[PPos] = pos;
    }

    /// <summary>
    /// One thread per slot byte of the wave: each chunk's DEFLATE bytes to its place: a dynamic block's from its words,
    /// a stored block's headers (every 65535 bytes) and bytes from the plane, an empty block's 2 bytes.
    /// </summary>
    private static void CompressCopyKernel(Index1D i, ArrayView<byte> planes, ArrayView<uint> words, ArrayView<byte> hdr, ArrayView<byte> output,
        ArrayView<int> tails, ArrayView<int> offsets, ArrayView<int> prm, int first)
    {
        int b = i.X / SlotBytes;
        int k = i.X - b * SlotBytes;
        int tail = b * DeflateGpu.PlanTailInts;
        int n = DeflateGpu.PlannedBytes(tails, tail);
        if (k >= n) return;
        int kind = DeflateGpu.PlannedKind(tails, tail);
        int v;
        if (kind == 0) v = (int)((words[(long)b * SlotWords + (k >> 2)] >> (8 * (k & 3))) & 0xFFu);
        else if (kind == 2) v = hdr[b * DeflateGpu.MaxDynamicHeaderBytes + k];
        else
        {
            // Stored sub-blocks of at most 65535 bytes, each after a 5-byte header (BFINAL on the last, LEN, NLEN).
            int g = first + b;
            int len = ChunkLength(prm, g);
            int sub = k / 65540;
            int r = k - sub * 65540;
            int subStart = sub * 65535;
            int cnt = len - subStart;
            if (cnt > 65535) cnt = 65535;
            if (r == 0) v = subStart + cnt == len ? 1 : 0;
            else if (r == 1) v = cnt & 0xFF;
            else if (r == 2) v = (cnt >> 8) & 0xFF;
            else if (r == 3) v = ~cnt & 0xFF;
            else if (r == 4) v = (~cnt >> 8) & 0xFF;
            else v = planes[ChunkStart(prm, g) + subStart + r - 5];
        }
        output[offsets[b] + k] = (byte)v;
    }

    /// <summary>One thread: the header.</summary>
    private static void CompressHeaderKernel(Index1D idx, ArrayView<byte> output, ArrayView<int> prm)
    {
        output[0] = (byte)'S';
        output[1] = (byte)'D';
        output[2] = (byte)'W';
        output[3] = (byte)'P';
        output[4] = 1;
        output[5] = (byte)prm[PElem];
        output[6] = 0;
        output[7] = 0;
        int e = prm[PPlane];
        output[8] = (byte)e;
        output[9] = (byte)(e >> 8);
        output[10] = (byte)(e >> 16);
        output[11] = (byte)(e >> 24);
        for (int k = 12; k < 16; k++) output[k] = 0;
        output[16] = ChunkBytes & 0xFF;
        output[17] = (ChunkBytes >> 8) & 0xFF;
        output[18] = (ChunkBytes >> 16) & 0xFF;
        output[19] = (ChunkBytes >> 24) & 0xFF;
        int p = prm[PPerPlane];
        output[20] = (byte)p;
        output[21] = (byte)(p >> 8);
        output[22] = (byte)(p >> 16);
        output[23] = (byte)(p >> 24);
    }

    // ------------------------------------------------------------------
    // Kernels: decompression
    // ------------------------------------------------------------------

    /// <summary>One thread, LayoutChunk table entries per launch: each chunk's input offset; flags a table that runs past the data.</summary>
    private static void DecompressLayoutKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> offsets, ArrayView<int> prm)
    {
        int chunks = prm[PChunks];
        int g0 = prm[PLayoutAt];
        int g1 = g0 + LayoutChunk;
        if (g1 > chunks) g1 = chunks;
        int pos = prm[PPos];
        int length = prm[PLength];
        for (int g = g0; g < g1; g++)
        {
            int t = HeaderBytes + 4 * g;
            int n = data[t] | (data[t + 1] << 8) | (data[t + 2] << 16) | (data[t + 3] << 24);
            offsets[g] = pos;
            long next = (long)pos + n;
            bool bad = (n < 0) | (next > length);
            if (bad)
            {
                if (prm[PError] == 0)
                {
                    prm[PError] = 1;
                    prm[PErrorAt] = g;
                }
                next = pos;
            }
            pos = (int)next;
        }
        prm[PLayoutAt] = g1;
        prm[PPos] = pos;
    }

    /// <summary>Chunk g's compressed length = the next offset minus its own (the last one: the running end).</summary>
    private static int CompressedLength(ArrayView<int> offsets, ArrayView<int> prm, int g)
    {
        int end = prm[PPos];
        if (g + 1 < prm[PChunks]) end = offsets[g + 1];
        return end - offsets[g];
    }

    private static void DecompressInitKernel(Index1D i, ArrayView<int> offsets, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        int g = first + i.X;
        InflateGpu.InitRaw(offsets[g], CompressedLength(offsets, prm, g), ChunkStart(prm, g), ChunkLength(prm, g), scratch, i.X * InflateGpu.ScratchInts);
    }

    private static void DecompressStepKernel(Index1D i, ArrayView<byte> data, ArrayView<byte> planes, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        if (prm[PError] != 0) return;
        InflateGpu.Step(data, planes, scratch, i.X * InflateGpu.ScratchInts, int.MaxValue, DecodeOutputBudget);
    }

    /// <summary>One thread per chunk of the wave: the chunk must have inflated to exactly its size.</summary>
    private static void DecompressCheckKernel(Index1D i, ArrayView<int> offsets, ArrayView<int> scratch, ArrayView<int> prm, int first)
    {
        int g = first + i.X;
        int r = InflateGpu.Result(scratch, i.X * InflateGpu.ScratchInts);
        if (r != ChunkLength(prm, g))
        {
            if (prm[PError] == 0)
            {
                prm[PError] = 2;
                prm[PErrorAt] = g;
            }
        }
    }

    public void Dispose()
    {
        _shuffle.Dispose();
        _prm.Dispose();
        _counts.Dispose();
        _tails.Dispose();
        _hdr.Dispose();
        _words.Dispose();
        _dScratch.Dispose();
        _planes.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
        _offsets.Dispose();
    }
}
