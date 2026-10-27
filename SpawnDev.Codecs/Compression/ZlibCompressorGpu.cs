// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// zlib (RFC 1950) compressor on the accelerator. The input is cut into independent blocks; each block is
// compressed (DeflateGpu) into a fixed-size slot, then small kernels lay the slots out back to back (a byte
// copy: every non-final block ends byte-aligned) and write the 2-byte zlib header and the Adler-32 trailer.
// Neither the input nor the output visits the host; the host reads back one int (the compressed size).
//
// No launch does work that grows with the input (a long launch stalls the desktop compositor and trips the OS
// GPU watchdog): a wave of blocks is compressed one thread per block through DeflateGpu's resumable phases
// (set-up, ParseChunk-position parse launches, code planning, EmitChunk-token emit launches); the Adler-32 is
// summed in parallel AdlerPieceBytes pieces and combined per block; the layout walks LayoutChunk blocks per
// launch. The launch counts are known on the host from the sizes; only level 3 (lazy matching, which may stop a
// parse launch on its compare budget) reads back one int per wave to see whether its parse needs more launches.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression;

/// <summary>Launches the block-parallel DEFLATE compressor kernels.</summary>
internal sealed class DeflateKernels : IDisposable
{
    // prm slots: input length, block size, level, slot capacity, block count, scratch ints per thread, Adler pieces
    // per block, pending parses (level 3 readback), the resumable layout state (stream position, Adler a / b, failed
    // flag; shared by the zlib and PNG layouts), the lazy parse's compare budget per launch.
    public const int PrmLength = 0, PrmBlockSize = 1, PrmLevel = 2, PrmSlotCap = 3, PrmBlocks = 4, PrmScratchPer = 5,
        PrmAdlerPieces = 6, PrmPending = 7, PrmLayoutPos = 8, PrmLayoutA = 9, PrmLayoutB = 10, PrmLayoutFailed = 11,
        PrmCompareBudget = 12;
    public const int PrmInts = 16;

    /// <summary>Input bytes one thread sums for the Adler-32 (pieces are combined per block).</summary>
    public const int AdlerPieceBytes = 4096;

    /// <summary>Blocks one layout launch walks.</summary>
    public const int LayoutChunk = 8192;

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _adlerPiece;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> _adlerBlock;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _init;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _parse;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _pending;
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _chains;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int> _match;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _walk;
    private readonly Action<KernelConfig, ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<int>, int> _emitLanes;
    private readonly int _emitLaneCount;
    private readonly Action<KernelConfig, ArrayView<int>, ArrayView<int>, int> _count;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _literals;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _plan, _emit;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _result;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _copy;

    public DeflateKernels(Accelerator accelerator)
    {
        _adlerPiece = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(AdlerPieceKernel);
        _adlerBlock = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(AdlerBlockKernel);
        _init = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(InitKernel);
        _parse = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(ParseKernel);
        _pending = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(PendingKernel);
        _chains = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(ChainsKernel);
        _match = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(MatchKernel);
        _walk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(WalkKernel);
        _emitLanes = accelerator.LoadStreamKernel<ArrayView<byte>, ArrayView<int>, ArrayView<uint>, ArrayView<int>, int>(EmitLanesKernel);
        _emitLaneCount = Math.Min(EmitLanes, accelerator.MaxNumThreadsPerGroup);
        _count = accelerator.LoadStreamKernel<ArrayView<int>, ArrayView<int>, int>(CountKernel);
        _literals = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(LiteralsKernel);
        _plan = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(PlanKernel);
        _emit = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(EmitKernel);
        _result = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(ResultKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int>(LayoutKernel);
        _copy = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(CopyKernel);
    }

    /// <summary>Adler-32 pieces per block of <paramref name="blockSize"/> bytes.</summary>
    public static int AdlerPiecesPerBlock(int blockSize) => (blockSize + AdlerPieceBytes - 1) / AdlerPieceBytes;

    /// <summary>Per-block Adler-32 sums into res[block * 4 + 1 / + 2]: parallel pieces into <paramref name="adler"/>, then one thread per block.</summary>
    public void RunAdler(ArrayView<byte> data, ArrayView<int> adler, ArrayView<int> res, ArrayView<int> prm, int blocks, int piecesPerBlock)
    {
        _adlerPiece(blocks * piecesPerBlock, data, adler, prm);
        _adlerBlock(blocks, adler, res, prm);
    }

    /// <summary>Set-up of the blocks [waveBase, waveBase + count): one thread per hash slot of each.</summary>
    public void RunInit(ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _init(count * DeflateGpu.HashSize, scratch, prm, waveBase);

    /// <summary>One bounded parse launch of the blocks [waveBase, waveBase + count).</summary>
    public void RunParse(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _parse(count, data, scratch, prm, waveBase);

    /// <summary>The wave's hash chains (DeflateGpu.ParallelParse step 1): one group of DeflateGpu.ChainLanes per block.</summary>
    public void RunChains(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _chains(new KernelConfig(count, DeflateGpu.ChainLanes), data, scratch, prm, waveBase);

    /// <summary>Every position's match in [from, from + ParallelParseChunk) of the wave's blocks (step 2).</summary>
    public void RunMatch(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count, int from) =>
        _match(count * DeflateGpu.ParallelParseChunk, data, scratch, prm, waveBase, from);

    /// <summary>One bounded walk launch of the wave (step 3).</summary>
    public void RunWalk(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _walk(count, data, scratch, prm, waveBase);

    /// <summary>Level 4 without a parse: one thread per byte of the wave's blocks writes its literal token.</summary>
    public void RunLiterals(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count, int blockSize) =>
        _literals(count * blockSize, data, scratch, prm, waveBase);

    /// <summary>The wave's frequencies and extra bits after a parse that did not count them (groups of EmitLanes lanes).</summary>
    public void RunCount(ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _count(new KernelConfig(count, _emitLaneCount), scratch, prm, waveBase);

    /// <summary>Threads per block writing a dynamic block's tokens (DeflateGpu.EmitBlockLanes).</summary>
    public const int EmitLanes = 256;

    /// <summary>The wave's dynamic blocks written by groups of lanes (stored blocks are left to RunEmit).</summary>
    public void RunEmitLanes(ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<uint> words, ArrayView<int> prm, int waveBase, int count) =>
        _emitLanes(new KernelConfig(count, _emitLaneCount), slots, scratch, words, prm, waveBase);

    /// <summary>Counts the wave's unfinished parses into prm[PrmPending].</summary>
    public void RunPending(ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _pending(1, scratch, prm, waveBase, count);

    /// <summary>Codes + block headers of the wave.</summary>
    public void RunPlan(ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _plan(count, data, slots, scratch, prm, waveBase);

    /// <summary>One bounded emit launch of the wave.</summary>
    public void RunEmit(ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count) =>
        _emit(count, data, slots, scratch, prm, waveBase);

    /// <summary>Output length and input length of the wave's blocks into res.</summary>
    public void RunResult(ArrayView<int> scratch, ArrayView<int> res, ArrayView<int> prm, int waveBase, int count) =>
        _result(count, scratch, res, prm, waveBase);

    /// <summary>Per-block offsets, zlib header, combined Adler-32 trailer, total size in layout[blockCount]: LayoutChunk blocks per launch.</summary>
    public void RunLayout(ArrayView<int> res, ArrayView<byte> stream, ArrayView<int> prm, ArrayView<int> layout, int blocks)
    {
        for (int step = 0; step * LayoutChunk < blocks; step++) _layout(1, res, stream, prm, layout, step);
    }

    /// <summary>Copy every block's bytes from its slot to its place in the stream (one thread per slot byte).</summary>
    public void RunCopy(
        ArrayView<byte> slots, ArrayView<byte> stream, ArrayView<int> res, ArrayView<int> layout, ArrayView<int> prm,
        int slotBytes)
    {
        _copy(slotBytes, slots, stream, res, layout, prm);
    }

    private static int BlockLength(ArrayView<int> prm, int block)
    {
        int bs = prm[PrmBlockSize];
        int len = prm[PrmLength] - block * bs;
        if (len > bs) len = bs;
        return len;
    }

    private static void AdlerPieceKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> adler, ArrayView<int> prm)
    {
        int i = idx.X;
        int perBlock = prm[PrmAdlerPieces];
        int block = i / perBlock;
        int j = i - block * perBlock;
        if (block >= prm[PrmBlocks]) return;
        int len = BlockLength(prm, block);
        int from = j * AdlerPieceBytes;
        if (from >= len) return;
        int n = len - from;
        if (n > AdlerPieceBytes) n = AdlerPieceBytes;
        // (b << 16) | a of the piece, summed from the initial state a = 1, b = 0.
        adler[i] = (int)DeflateGpu.AdlerOfRange(data, block * prm[PrmBlockSize] + from, n);
    }

    private static void AdlerBlockKernel(Index1D idx, ArrayView<int> adler, ArrayView<int> res, ArrayView<int> prm)
    {
        int block = idx.X;
        if (block >= prm[PrmBlocks]) return;
        int len = BlockLength(prm, block);
        int perBlock = prm[PrmAdlerPieces];
        const uint Base = 65521u;
        uint a = 1;
        uint b = 0;
        for (int j = 0; j * AdlerPieceBytes < len; j++)
        {
            // Adler-32 combine: pieces were summed from the initial state a = 1, b = 0.
            uint packed = (uint)adler[block * perBlock + j];
            uint a2 = packed & 0xFFFFu;
            uint b2 = packed >> 16;
            int n = len - j * AdlerPieceBytes;
            if (n > AdlerPieceBytes) n = AdlerPieceBytes;
            uint len2 = (uint)n % Base;
            b = (b + b2 + len2 * ((a + Base - 1u) % Base)) % Base;
            a = (a + a2 + Base - 1u) % Base;
        }
        res[block * 4 + 1] = (int)a;
        res[block * 4 + 2] = (int)b;
    }

    private static void InitKernel(Index1D idx, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        int b = idx.X / DeflateGpu.HashSize;
        int k = idx.X - b * DeflateGpu.HashSize;
        int block = waveBase + b;
        if (block >= prm[PrmBlocks]) return;
        DeflateGpu.InitBlockThread(k, BlockLength(prm, block), prm[PrmLevel], scratch, b * prm[PrmScratchPer], prm[PrmBlockSize]);
    }

    private static void ParseKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        int block = waveBase + idx.X;
        if (block >= prm[PrmBlocks]) return;
        int bs = prm[PrmBlockSize];
        int level = prm[PrmLevel];
        // Greedy parses are bounded by their positions alone (at most 4 * chain compares each): only the lazy one
        // stops on compares, and only its host loop checks for unfinished parses.
        int compareBudget = level == 3 ? prm[PrmCompareBudget] : int.MaxValue;
        DeflateGpu.ParseStep(data, block * bs, BlockLength(prm, block), level, scratch, idx.X * prm[PrmScratchPer], bs,
            DeflateGpu.ParseChunk, compareBudget);
    }

    private static void ChainsKernel(ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        var hs = SharedMemory.Allocate<int>(DeflateGpu.ChainLanes);
        int b = Grid.IdxX;
        int block = waveBase + b;
        int bs = prm[PrmBlockSize];
        int len = 0;
        if (block < prm[PrmBlocks]) len = BlockLength(prm, block);
        // A block past the end has length 0: its group still meets every barrier.
        DeflateGpu.BuildChainsLanes(data, block * bs, len, scratch, b * prm[PrmScratchPer], bs, hs, Group.IdxX);
    }

    private static void MatchKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int from)
    {
        int b = idx.X / DeflateGpu.ParallelParseChunk;
        int k = from + (idx.X - b * DeflateGpu.ParallelParseChunk);
        int block = waveBase + b;
        if (block >= prm[PrmBlocks]) return;
        int bs = prm[PrmBlockSize];
        DeflateGpu.FindMatchThread(data, block * bs, BlockLength(prm, block), k, prm[PrmLevel], scratch, b * prm[PrmScratchPer], bs);
    }

    private static void WalkKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        int block = waveBase + idx.X;
        if (block >= prm[PrmBlocks]) return;
        int bs = prm[PrmBlockSize];
        DeflateGpu.WalkParseStep(data, block * bs, BlockLength(prm, block), prm[PrmLevel], scratch, idx.X * prm[PrmScratchPer], bs,
            DeflateGpu.ParallelParseChunk, false);
    }

    private static void PendingKernel(Index1D _, ArrayView<int> scratch, ArrayView<int> prm, int waveBase, int count)
    {
        int pending = 0;
        for (int b = 0; b < count; b++)
        {
            if (waveBase + b < prm[PrmBlocks])
                pending += DeflateGpu.ParsePending(scratch, b * prm[PrmScratchPer], prm[PrmBlockSize]);
        }
        prm[PrmPending] = pending;
    }

    private static void PlanKernel(
        Index1D idx, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        int block = waveBase + idx.X;
        int blocks = prm[PrmBlocks];
        if (block >= blocks) return;
        int bs = prm[PrmBlockSize];
        int slotCap = prm[PrmSlotCap];
        DeflateGpu.PlanBlock(data, block * bs, BlockLength(prm, block), block == blocks - 1, prm[PrmLevel],
            slots, block * slotCap, slotCap, scratch, idx.X * prm[PrmScratchPer], bs);
    }

    private static void EmitKernel(
        Index1D idx, ArrayView<byte> data, ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        int block = waveBase + idx.X;
        int blocks = prm[PrmBlocks];
        if (block >= blocks) return;
        int bs = prm[PrmBlockSize];
        DeflateGpu.EmitStep(data, block * bs, BlockLength(prm, block), block == blocks - 1,
            slots, block * prm[PrmSlotCap], scratch, idx.X * prm[PrmScratchPer], bs, DeflateGpu.EmitChunk);
    }

    private static void LiteralsKernel(Index1D idx, ArrayView<byte> data, ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        int bs = prm[PrmBlockSize];
        int b = idx.X / bs;
        int k = idx.X - b * bs;
        int block = waveBase + b;
        if (block >= prm[PrmBlocks]) return;
        DeflateGpu.LiteralTokensThread(data, block * bs, BlockLength(prm, block), k, scratch, b * prm[PrmScratchPer], bs);
    }

    private static void CountKernel(ArrayView<int> scratch, ArrayView<int> prm, int waveBase)
    {
        var h = SharedMemory.Allocate<int>(DeflateGpu.CountSharedInts);
        int b = Grid.IdxX;
        DeflateGpu.CountBlockLanes(scratch, b * prm[PrmScratchPer], prm[PrmBlockSize], h, Group.IdxX, Group.DimX);
    }

    private static void EmitLanesKernel(ArrayView<byte> slots, ArrayView<int> scratch, ArrayView<uint> words, ArrayView<int> prm, int waveBase)
    {
        var part = SharedMemory.Allocate<int>(EmitLanes);
        int b = Grid.IdxX;
        int block = waveBase + b;
        int blocks = prm[PrmBlocks];
        int bs = prm[PrmBlockSize];
        // A group past the last block finds no planned block (its scratch is the wave's, phase not EmitTokens) and still
        // meets every barrier.
        bool real = block < blocks;
        int sb = b * prm[PrmScratchPer];
        DeflateGpu.EmitBlockLanes(block == blocks - 1, slots, real ? block * prm[PrmSlotCap] : 0, scratch, sb, bs, words,
            (long)b * DeflateGpu.EmitWordsPerBlock(bs), part, Group.IdxX, Group.DimX);
    }

    private static void ResultKernel(Index1D idx, ArrayView<int> scratch, ArrayView<int> res, ArrayView<int> prm, int waveBase)
    {
        int block = waveBase + idx.X;
        if (block >= prm[PrmBlocks]) return;
        res[block * 4 + 0] = DeflateGpu.BlockOutput(scratch, idx.X * prm[PrmScratchPer], prm[PrmBlockSize]);
        res[block * 4 + 3] = BlockLength(prm, block);
    }

    private static void LayoutKernel(
        Index1D _, ArrayView<int> res, ArrayView<byte> stream, ArrayView<int> prm, ArrayView<int> layout, int step)
    {
        int blockCount = prm[PrmBlocks];
        int lo = step * LayoutChunk;
        int hi = lo + LayoutChunk;
        if (hi > blockCount) hi = blockCount;
        const uint Base = 65521u;
        uint a = 1;
        uint b = 0;
        int pos = 2;
        int failed = 0;
        if (step > 0)
        {
            pos = prm[PrmLayoutPos];
            a = (uint)prm[PrmLayoutA];
            b = (uint)prm[PrmLayoutB];
            failed = prm[PrmLayoutFailed];
        }
        for (int i = lo; i < hi; i++)
        {
            int outLen = res[i * 4 + 0];
            if (outLen < 0) failed = 1;
            layout[i] = pos;
            if (outLen > 0) pos += outLen;
            // Adler-32 combine: blocks were summed from the initial state a = 1, b = 0.
            uint a2 = (uint)res[i * 4 + 1];
            uint b2 = (uint)res[i * 4 + 2];
            uint len2 = (uint)res[i * 4 + 3] % Base;
            b = (b + b2 + len2 * ((a + Base - 1u) % Base)) % Base;
            a = (a + a2 + Base - 1u) % Base;
        }
        prm[PrmLayoutPos] = pos;
        prm[PrmLayoutA] = (int)a;
        prm[PrmLayoutB] = (int)b;
        prm[PrmLayoutFailed] = failed;
        if (hi == blockCount)
        {
            stream[0] = 0x78;
            stream[1] = 0x01;
            uint adler = (b << 16) | a;
            stream[pos + 0] = (byte)(adler >> 24);
            stream[pos + 1] = (byte)(adler >> 16);
            stream[pos + 2] = (byte)(adler >> 8);
            stream[pos + 3] = (byte)adler;
            layout[blockCount] = failed != 0 ? -1 : pos + 4;
        }
    }

    private static void CopyKernel(
        Index1D idx, ArrayView<byte> slots, ArrayView<byte> stream, ArrayView<int> res,
        ArrayView<int> layout, ArrayView<int> prm)
    {
        int slotCap = prm[PrmSlotCap];
        int i = idx.X;
        int block = i / slotCap;
        int off = i - block * slotCap;
        if (block >= prm[PrmBlocks]) return;
        if (off < res[block * 4 + 0])
            stream[layout[block] + off] = slots[i];
    }

    public void Dispose() { }
}

/// <summary>
/// Owns the block-compression working memory and enqueues the per-block kernels. Shared by the zlib compressor
/// and the PNG encoder (which frames the same blocks as IDAT chunks instead of one zlib stream).
/// </summary>
internal sealed class DeflateBlockRunner : IDisposable
{
    /// <summary>Parse launches a level-3 wave adds between two checks for unfinished parses.</summary>
    private const int LazyParseBatch = 4;

    private readonly Accelerator _accelerator;
    private readonly int _wave;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm;
    private readonly StageBuffer<byte> _slots;
    private readonly StageBuffer<int> _res;
    private readonly StageBuffer<int> _scratch;
    private readonly StageBuffer<int> _adler;
    private readonly StageBuffer<uint> _words;

    public DeflateKernels Kernels { get; }
    public int BlockSize { get; }
    public int BlockCount { get; private set; }
    public int SlotCapacity { get; private set; }
    public long SlotBytes { get; private set; }
    public ArrayView1D<byte, Stride1D.Dense> Slots => _slots.View;
    public ArrayView1D<int, Stride1D.Dense> Res => _res.View;
    public ArrayView1D<int, Stride1D.Dense> Prm => _prm.View;

    /// <summary>
    /// Tests only: parse with the serial DeflateGpu.ParseStep and write every block with the serial EmitStep (one thread
    /// per block, resumable launches) instead of DeflateGpu.ParallelParse, the lanes counting and EmitBlockLanes. It is
    /// the oracle of the parallel path, which must write the same bytes.
    /// </summary>
    internal bool SerialParse { get; set; }

    /// <summary>
    /// Byte compares after which a serial lazy (level 3) parse launch stops (<see cref="DeflateGpu.ParseCompareBudget"/>).
    /// Tests lower it to drive the unfinished-parse loop; the output does not depend on it.
    /// </summary>
    internal int ParseCompareBudget { get; set; } = DeflateGpu.ParseCompareBudget;

    public DeflateBlockRunner(Accelerator accelerator, int blockSize, int threadsPerWave)
    {
        if (blockSize < 1024 || blockSize > DeflateGpu.MaxBlockSize) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (threadsPerWave < 0) throw new ArgumentOutOfRangeException(nameof(threadsPerWave));
        _accelerator = accelerator;
        BlockSize = blockSize;
        _wave = threadsPerWave > 0 ? threadsPerWave : AutoWave(accelerator, blockSize);
        Kernels = new DeflateKernels(accelerator);
        _prm = accelerator.Allocate1D<int>(DeflateKernels.PrmInts);
        _slots = new StageBuffer<byte>(accelerator);
        _res = new StageBuffer<int>(accelerator);
        _scratch = new StageBuffer<int>(accelerator);
        _adler = new StageBuffer<int>(accelerator);
        _words = new StageBuffer<uint>(accelerator);
    }

    public static int BlocksFor(int length, int blockSize) => Math.Max(1, (length + blockSize - 1) / blockSize);

    /// <summary>Working memory a wave may take when the caller leaves the wave to the runner: desktop / browser accelerators.</summary>
    public const long DesktopWaveBytes = 256L << 20, BrowserWaveBytes = 32L << 20;

    /// <summary>
    /// Blocks per wave when the caller passes 0: as many as fit in <see cref="DesktopWaveBytes"/> (browser accelerators:
    /// <see cref="BrowserWaveBytes"/>) of working memory. One thread or group compresses a block in each step, so throughput
    /// grows with the wave (RTX 4070, 64 MB of BF16 weights at level 1: 0.006 GB/s at 32 blocks, 0.053 at 512).
    /// </summary>
    public static int AutoWave(Accelerator accelerator, int blockSize)
    {
        bool browser = accelerator.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm;
        long budget = browser ? BrowserWaveBytes : DesktopWaveBytes;
        long perBlock = 4L * DeflateGpu.ScratchInts(blockSize);
        return (int)Math.Clamp(budget / perBlock, 1, 4096);
    }

    /// <summary>
    /// Enqueue compression of the first <paramref name="length"/> bytes of <paramref name="data"/>: block outputs in
    /// <see cref="Slots"/>, per-block results (output length, Adler a, Adler b, input length) in <see cref="Res"/>.
    /// Levels 0-2 only enqueue; level 3 synchronizes once per wave to read its unfinished-parse count.
    /// </summary>
    public async Task RunAsync(ArrayView<byte> data, int length, int level)
    {
        int blocks = BlocksFor(length, BlockSize);
        int slotCap = DeflateGpu.MaxBlockOutput(BlockSize);
        int scratchPer = DeflateGpu.ScratchInts(BlockSize);
        int piecesPerBlock = DeflateKernels.AdlerPiecesPerBlock(BlockSize);
        long slotBytes = (long)blocks * slotCap;
        if (slotBytes > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "input too large for one call");
        BlockCount = blocks;
        SlotCapacity = slotCap;
        SlotBytes = slotBytes;

        _slots.Ensure(slotBytes);
        _res.Ensure((long)blocks * 4);
        _adler.Ensure((long)blocks * piecesPerBlock);
        int wave = Math.Min(_wave, blocks);
        _scratch.Ensure((long)wave * scratchPer);
        _words.Ensure((long)wave * DeflateGpu.EmitWordsPerBlock(BlockSize));
        var prm = new int[DeflateKernels.PrmInts];
        prm[DeflateKernels.PrmLength] = length;
        prm[DeflateKernels.PrmBlockSize] = BlockSize;
        prm[DeflateKernels.PrmLevel] = level;
        prm[DeflateKernels.PrmSlotCap] = slotCap;
        prm[DeflateKernels.PrmBlocks] = blocks;
        prm[DeflateKernels.PrmScratchPer] = scratchPer;
        prm[DeflateKernels.PrmAdlerPieces] = piecesPerBlock;
        prm[DeflateKernels.PrmCompareBudget] = ParseCompareBudget;
        _prm.View.CopyFromCPU(ref prm[0], prm.Length);

        Kernels.RunAdler(data, _adler.View, _res.View, _prm.View, blocks, piecesPerBlock);
        int parseSteps = DeflateGpu.ParseSteps(BlockSize);
        int walkSteps = DeflateGpu.WalkSteps(BlockSize);
        int emitSteps = DeflateGpu.EmitSteps(BlockSize);
        var pacer = new LaunchPacer(_accelerator);
        for (int first = 0; first < blocks; first += wave)
        {
            int count = Math.Min(wave, blocks - first);
            Kernels.RunInit(_scratch.View, _prm.View, first, count);
            if (SerialParse & (level > 0))
            {
                for (int i = 0; i < parseSteps; i++)
                {
                    Kernels.RunParse(data, _scratch.View, _prm.View, first, count);
                    await pacer.LaunchedAsync();
                }
                if (level == 3)
                {
                    // Lazy matching can stop a parse launch on its compare budget before ParseChunk positions.
                    bool pending = true;
                    while (pending)
                    {
                        Kernels.RunPending(_scratch.View, _prm.View, first, count);
                        await _accelerator.SynchronizeAsync();
                        pending = (await _prm.View.SubView(DeflateKernels.PrmPending, 1).CopyToHostAsync())[0] != 0;
                        if (pending)
                            for (int i = 0; i < LazyParseBatch; i++) Kernels.RunParse(data, _scratch.View, _prm.View, first, count);
                    }
                }
            }
            else if (level == DeflateGpu.LevelHuffmanOnly)
            {
                Kernels.RunLiterals(data, _scratch.View, _prm.View, first, count, BlockSize);
                Kernels.RunCount(_scratch.View, _prm.View, first, count);
            }
            else if (level > 0)
            {
                // The parse in parallel steps (DeflateGpu.ParallelParse), the same tokens as ParseStep, then its counts.
                Kernels.RunChains(data, _scratch.View, _prm.View, first, count);
                for (int from = 0; from < BlockSize; from += DeflateGpu.ParallelParseChunk)
                {
                    Kernels.RunMatch(data, _scratch.View, _prm.View, first, count, from);
                    await pacer.LaunchedAsync();
                }
                for (int i = 0; i < walkSteps; i++)
                {
                    Kernels.RunWalk(data, _scratch.View, _prm.View, first, count);
                    await pacer.LaunchedAsync();
                }
                Kernels.RunCount(_scratch.View, _prm.View, first, count);
            }
            Kernels.RunPlan(data, _slots.View, _scratch.View, _prm.View, first, count);
            // Dynamic blocks by lanes; the bounded serial launches then write the stored blocks (rare), or every block on
            // the serial oracle path.
            if (!SerialParse) Kernels.RunEmitLanes(_slots.View, _scratch.View, _words.View, _prm.View, first, count);
            for (int i = 0; i < emitSteps; i++)
            {
                Kernels.RunEmit(data, _slots.View, _scratch.View, _prm.View, first, count);
                await pacer.LaunchedAsync();
            }
            Kernels.RunResult(_scratch.View, _res.View, _prm.View, first, count);
        }
    }

    public void Dispose()
    {
        Kernels.Dispose();
        _prm.Dispose();
        _slots.Dispose();
        _res.Dispose();
        _scratch.Dispose();
        _adler.Dispose();
        _words.Dispose();
    }
}

/// <summary>zlib compressor running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class ZlibCompressorGpu : IDisposable
{
    /// <summary>Default input bytes per independent block.</summary>
    public const int DefaultBlockSize = 65536;

    private readonly Accelerator _accelerator;
    private readonly int _blockSize;
    private readonly DeflateBlockRunner _runner;
    private readonly StageBuffer<int> _layout;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels.</summary>
    /// <param name="accelerator">The accelerator.</param>
    /// <param name="blockSize">Input bytes per independent block (1 KiB .. 1 MiB). Larger blocks compress better, smaller ones parallelize more.</param>
    /// <param name="threadsPerWave">Blocks compressed concurrently (scratch memory scales with this); 0 = as many as fit in
    /// DeflateBlockRunner.DesktopWaveBytes (browser accelerators: BrowserWaveBytes) of working memory.</param>
    public ZlibCompressorGpu(Accelerator accelerator, int blockSize = DefaultBlockSize, int threadsPerWave = 0)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (blockSize < 1024 || blockSize > DeflateGpu.MaxBlockSize) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (threadsPerWave < 0) throw new ArgumentOutOfRangeException(nameof(threadsPerWave));
        _accelerator = accelerator;
        _blockSize = blockSize;
        _runner = new DeflateBlockRunner(accelerator, blockSize, threadsPerWave);
        _layout = new StageBuffer<int>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    private static int BlockCount(int length, int blockSize) => Math.Max(1, (length + blockSize - 1) / blockSize);

    /// <summary>Tests only: the serial-parse oracle (see DeflateBlockRunner.SerialParse).</summary>
    internal bool SerialParse { get => _runner.SerialParse; set => _runner.SerialParse = value; }

    /// <summary>Compare budget of a serial lazy parse launch (tests lower it; the output does not depend on it).</summary>
    internal int ParseCompareBudget { get => _runner.ParseCompareBudget; set => _runner.ParseCompareBudget = value; }

    /// <summary>Upper bound of the compressed size of <paramref name="inputLength"/> bytes (the output view must be at least this long).</summary>
    public static long GetMaxCompressedSize(int inputLength, int blockSize = DefaultBlockSize) =>
        6L + (long)BlockCount(inputLength, blockSize) * DeflateGpu.MaxBlockOutput(blockSize);

    /// <summary>
    /// Compress <paramref name="length"/> bytes that are on the accelerator into a zlib stream in
    /// <paramref name="output"/> (at least <see cref="GetMaxCompressedSize"/> bytes). Returns the stream length.
    /// </summary>
    /// <param name="level">0 = stored, 1 = fast, 2 = default, 3 = best (lazy matching), 4 = Huffman only (<see cref="DeflateGpu.LevelHuffmanOnly"/>).</param>
    public async Task<int> CompressAsync(
        ArrayView1D<byte, Stride1D.Dense> data, int length,
        ArrayView1D<byte, Stride1D.Dense> output, int level = 2)
    {
        if (length < 0 || length > data.Length) throw new ArgumentOutOfRangeException(nameof(length));
        if (level < 0 || level > DeflateGpu.LevelHuffmanOnly) throw new ArgumentOutOfRangeException(nameof(level));
        long maxOut = GetMaxCompressedSize(length, _blockSize);
        if (output.Length < maxOut)
            throw new ArgumentException($"output must hold {maxOut} bytes.", nameof(output));

        int blocks = BlockCount(length, _blockSize);
        _layout.Ensure(blocks + 1);
        await _runner.RunAsync(data, length, level);
        _runner.Kernels.RunLayout(_runner.Res, output, _runner.Prm, _layout.View, blocks);
        _runner.Kernels.RunCopy(_runner.Slots, output, _runner.Res, _layout.View, _runner.Prm, (int)_runner.SlotBytes);
        await _accelerator.SynchronizeAsync();

        int total = (await _layout.View.SubView(blocks, 1).CopyToHostAsync())[0];
        if (total < 0)
            throw new InvalidOperationException("zlib compression overflowed a block slot (internal error).");
        return total;
    }

    /// <summary>Compress managed bytes; returns the zlib stream as a managed array.</summary>
    public async Task<byte[]> CompressAsync(ReadOnlyMemory<byte> data, int level = 2)
    {
        _stageIn.Ensure(Math.Max(1, data.Length));
        if (data.Length > 0) DeviceTransfer.Upload(_stageIn.View, data.Span);
        _stageOut.Ensure(GetMaxCompressedSize(data.Length, _blockSize));
        int n = await CompressAsync(_stageIn.View, data.Length, _stageOut.View, level);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, n);
    }

    /// <summary>Compress a JS typed array (browser accelerators only); returns the zlib stream as a JS <see cref="Uint8Array"/>.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> CompressAsync(TypedArray data, int level = 2)
    {
        ArgumentNullException.ThrowIfNull(data);
        long length = data.ByteLength;
        _stageIn.Ensure(Math.Max(1, length));
        DeviceTransfer.Upload(_stageIn.View, data);
        _stageOut.Ensure(GetMaxCompressedSize(checked((int)length), _blockSize));
        int n = await CompressAsync(_stageIn.View, (int)length, _stageOut.View, level);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, n);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _runner.Dispose();
        _layout.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
