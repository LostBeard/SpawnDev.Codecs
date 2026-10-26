// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// LZ4 decompressor on the accelerator for the LZ4 frame format (lz4_Frame_format.md 1.6.x): any block size (64 KB ..
// 4 MB), independent or linked blocks, block / content checksums, content size, skippable frames, concatenated frames,
// and the legacy frame format (lz4 -l). Dictionary frames (dictID) are refused. Every check fails with the reason.
//
// Stages, none of which grows a launch with the input:
//   1. Walk (one thread, WalkHops headers per launch): frame headers (header checksum verified) and block headers, into a
//      block table (source offset, stored size, raw / checksum / linked flags, frame) and a frame table. Run twice: once to
//      count (the host sizes the tables from two ints), once to record.
//   2. Scan (one thread per block, ScanBudget input bytes per launch): the decoded size of every block from its sequence
//      headers alone, validating literal runs and the end-of-block rules.
//   3. Layout (one thread, LayoutChunk blocks per launch): output offset of every block, totals per frame; the host reads
//      back the total and checks it against the output.
//   4. Decode: independent blocks one thread each, DecodeBudget output bytes per launch (match copies resume mid-match);
//      linked blocks (which may copy from the previous blocks' output) by one thread in block order, same budget. Offsets
//      are checked against the data decoded so far (and, linked, the frame's earlier output, up to 64 KB back).
//   5. Checks: block checksums (XXH32 of the stored bytes, one thread per block, resumable), content size, content
//      checksum (XXH32 of the frame's output, one thread per frame, resumable).
// The host reads back a handful of ints (counts, the total length, one error word) and nothing else.
//
// API shape: the same three overloads as the codecs (ArrayView1D / managed / browser TypedArray).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Compression.Lz4;

/// <summary>LZ4 frame (and legacy frame) decompressor on the accelerator.</summary>
public sealed class Lz4FrameDecompressorGpu : IDisposable
{
    /// <summary>Frame / block headers one walk launch reads.</summary>
    public const int WalkHops = 4096;
    /// <summary>Compressed bytes one scan launch reads per block.</summary>
    public const int ScanBudget = 65536;
    /// <summary>Output bytes one decode launch writes per block.</summary>
    public const int DecodeBudget = 65536;
    /// <summary>Blocks one layout launch walks.</summary>
    public const int LayoutChunk = 65536;
    /// <summary>Bytes one checksum launch hashes per block or frame (a multiple of 16).</summary>
    public const int HashBudget = 1 << 20;

    private const uint MagicFrame = 0x184D2204u, MagicLegacy = 0x184C2102u, MagicSkipLow = 0x184D2A50u;
    private const int LegacyBlock = 8 << 20;

    // Block table (ints per block).
    // BAux: the scan's running size, then the checksum's hashed byte count (-1 = checked). BSeq / BLitLeft / BTok: where
    // the decoder stopped inside a sequence (0 at a token, 1 in its literals, 2 at its match header).
    internal const int BSrc = 0, BStored = 1, BFlags = 2, BFrame = 3, BSize = 4, BDst = 5, BMax = 6, BCheck = 7,
        BIp = 8, BOp = 9, BMatchLeft = 10, BMatchOff = 11, BPhase = 12, BAux = 13, BSeq = 14, BLitLeft = 15, BTok = 16, BXxh = 17;
    internal const int BlockInts = BXxh + Xxh32Gpu.StateInts;
    internal const int BRaw = 1, BHasCheck = 2, BLinked = 4, BLegacy = 8;

    // Frame table (ints per frame).
    internal const int FFlags = 0, FSizeLo = 1, FSizeHi = 2, FCheck = 3, FFirst = 4, FCount = 5, FDst = 6, FTotal = 7, FHashed = 8, FXxh = 9;
    internal const int FrameInts = FXxh + Xxh32Gpu.StateInts;
    internal const int FHasSize = 1, FHasCheck = 2, FLegacy = 4, FLinked = 8;

    // prm ints.
    internal const int PLength = 0, PPos = 1, PState = 2, PBlocks = 3, PFrames = 4, PRecord = 5, PError = 6, PErrorAt = 7,
        PTotal = 8, PLayoutBlock = 9, PLayoutPos = 10, PCurFrame = 11, PCurFlags = 12, PCurMax = 13, PCapBlocks = 14, PCapFrames = 15,
        PLinkedBlock = 16, PLinkedNext = 17, PMaxStored = 18, PMaxSeen = 19;
    internal const int PrmInts = 24;
    // Walk states.
    internal const int WMagic = 0, WBlocks = 1, WLegacy = 2, WDone = 3;

    /// <summary>Why a frame was refused (the error word the kernels write).</summary>
    public enum Lz4Error
    {
        None = 0, Truncated = 1, BadMagic = 2, BadVersion = 3, ReservedBits = 4, BadBlockSizeId = 5, HeaderChecksum = 6,
        DictionaryNotSupported = 7, BlockTooLarge = 8, BadSequence = 9, BadOffset = 10, OutputTooSmall = 11,
        BlockChecksum = 12, ContentChecksum = 13, ContentSize = 14, TooLarge = 15, Empty = 16,
    }

    private readonly Accelerator _accelerator;
    private readonly LaunchPacer _pacer;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _walk;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _scan;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _decode;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _blockCheck;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _frameCheck;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm;
    private MemoryBuffer1D<int, Stride1D.Dense>? _blocks, _frames;
    // Stand-in tables of the counting walk (never written; WebGPU refuses one buffer bound to two read_write slots).
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _noBlocks, _noFrames;
    private readonly StageBuffer<byte> _stageIn, _stageOut;

    /// <summary>Compiles the kernels.</summary>
    public Lz4FrameDecompressorGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _pacer = new LaunchPacer(accelerator);
        _walk = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(WalkKernel);
        _scan = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(ScanKernel);
        _layout = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(LayoutKernel);
        _decode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(DecodeKernel);
        _blockCheck = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(BlockCheckKernel);
        _frameCheck = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(FrameCheckKernel);
        _prm = accelerator.Allocate1D<int>(PrmInts);
        _noBlocks = accelerator.Allocate1D<int>(1);
        _noFrames = accelerator.Allocate1D<int>(1);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    /// <summary>Thrown for a frame the decoder refuses; <see cref="Reason"/> says why, <see cref="Offset"/> where (input byte, or block index for block-level checks).</summary>
    public sealed class Lz4FormatException : FormatException
    {
        public Lz4Error Reason { get; }
        public long Offset { get; }
        public Lz4FormatException(Lz4Error reason, long offset) : base($"LZ4: {reason} at {offset}.")
        {
            Reason = reason;
            Offset = offset;
        }
    }

    // ------------------------------------------------------------------
    // Overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Decompresses the frames in the first <paramref name="length"/> bytes of <paramref name="input"/> into
    /// <paramref name="output"/>; returns the decompressed length. Throws <see cref="Lz4FormatException"/> for a bad frame
    /// (including output too small).
    /// </summary>
    public async Task<int> DecompressAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output)
    {
        int total = await PrepareAsync(input, length);
        if (total > output.Length) throw new Lz4FormatException(Lz4Error.OutputTooSmall, total);
        await DecodeAsync(input, length, output, total);
        return total;
    }

    /// <summary>The decompressed length of the frames in <paramref name="input"/> (walks and scans them; no output written).</summary>
    public Task<int> GetDecompressedLengthAsync(ArrayView1D<byte, Stride1D.Dense> input, int length) => PrepareAsync(input, length);

    /// <summary>Decompresses <paramref name="frame"/> into a new array.</summary>
    public async Task<byte[]> DecompressAsync(ReadOnlyMemory<byte> frame)
    {
        var src = _stageIn.Ensure(frame.Length);
        if (frame.Length > 0) DeviceTransfer.Upload(src, frame.Span);
        int total = await PrepareAsync(src, frame.Length);
        var dst = _stageOut.Ensure(total);
        await DecodeAsync(src, frame.Length, dst, total);
        return await DeviceTransfer.DownloadAsync(dst, total);
    }

    /// <summary>Decompresses the bytes of <paramref name="frame"/> into a new Uint8Array (never on the .NET heap); the caller disposes it.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> DecompressAsync(TypedArray frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (frame.ByteLength > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(frame), "At most 2^31 - 1 bytes per call.");
        int length = (int)frame.ByteLength;
        var src = _stageIn.Ensure(length);
        if (length > 0) DeviceTransfer.Upload(src, frame);
        int total = await PrepareAsync(src, length);
        var dst = _stageOut.Ensure(total);
        await DecodeAsync(src, length, dst, total);
        return await DeviceTransfer.DownloadUint8ArrayAsync(dst, total);
    }

    // ------------------------------------------------------------------
    // Host coordination
    // ------------------------------------------------------------------

    private int _blockCount, _frameCount, _maxStored, _maxBlock;
    private bool _anyLinked;

    /// <summary>Walk (count, then record), scan, layout. Returns the total decompressed length.</summary>
    private async Task<int> PrepareAsync(ArrayView1D<byte, Stride1D.Dense> input, int length)
    {
        if (length < 0 || length > input.Length) throw new ArgumentOutOfRangeException(nameof(length), $"{length} bytes; the input holds {input.Length}.");
        if (length == 0) throw new Lz4FormatException(Lz4Error.Empty, 0);
        var data = input.SubView(0, length);

        // Pass 1: count blocks and frames.
        await RunWalkAsync(data, length, record: false, dummy: true);
        var counts = await _prm.View.CopyToHostAsync();
        ThrowIfError(counts);
        _blockCount = counts[PBlocks];
        _frameCount = counts[PFrames];
        EnsureTables(Math.Max(1, _blockCount), Math.Max(1, _frameCount));

        // Pass 2: record.
        await RunWalkAsync(data, length, record: true, dummy: false);
        var walked = await _prm.View.CopyToHostAsync();
        ThrowIfError(walked);
        _maxStored = walked[PMaxStored];
        _maxBlock = walked[PMaxSeen];
        _anyLinked = walked[PLinkedBlock] != 0;

        if (_blockCount > 0)
        {
            int scanSteps = _maxStored / ScanBudget + 2;
            for (int s = 0; s < scanSteps; s++)
            {
                _scan(_blockCount, data, _blocks!.View, _prm.View);
                await _pacer.LaunchedAsync();
            }
            for (int s = 0; s * LayoutChunk < _blockCount; s++)
            {
                _layout(1, _blocks!.View, _frames!.View, _prm.View);
                await _pacer.LaunchedAsync();
            }
        }
        await _accelerator.SynchronizeAsync();
        var laid = await _prm.View.CopyToHostAsync();
        ThrowIfError(laid);
        return laid[PTotal];
    }

    private async Task RunWalkAsync(ArrayView1D<byte, Stride1D.Dense> data, int length, bool record, bool dummy)
    {
        var prm = new int[PrmInts];
        prm[PLength] = length;
        prm[PRecord] = record ? 1 : 0;
        prm[PCapBlocks] = record ? _blockCount : 0;
        prm[PCapFrames] = record ? _frameCount : 0;
        _prm.View.CopyFromCPU(prm);
        var blocks = dummy || _blocks is null ? _noBlocks.View : _blocks.View;
        var frames = dummy || _frames is null ? _noFrames.View : _frames.View;
        // Every hop but a frame's closing one reads at least 4 bytes (a legacy frame closes without reading), so
        // length / 2 / WalkHops + 3 launches always reach the end; check the state every 64 launches so short
        // inputs stop early.
        int maxLaunches = length / 2 / WalkHops + 3;
        for (int l = 0; l < maxLaunches; l++)
        {
            _walk(1, data, blocks, frames, _prm.View);
            await _pacer.LaunchedAsync();
            if ((l & 63) == 63)
            {
                await _accelerator.SynchronizeAsync();
                var st = await _prm.View.SubView(PState, 1).CopyToHostAsync();
                if (st[0] == WDone) break;
            }
        }
        await _accelerator.SynchronizeAsync();
        var end = await _prm.View.CopyToHostAsync();
        ThrowIfError(end);
        if (end[PState] != WDone) throw new InvalidOperationException("LZ4 walk did not reach the end of the input in its launch bound (bug).");
    }

    private void EnsureTables(int blocks, int frames)
    {
        if (_blocks is null || _blocks.Length < (long)blocks * BlockInts)
        {
            _blocks?.Dispose();
            _blocks = _accelerator.Allocate1D<int>((long)blocks * BlockInts);
        }
        if (_frames is null || _frames.Length < (long)frames * FrameInts)
        {
            _frames?.Dispose();
            _frames = _accelerator.Allocate1D<int>((long)frames * FrameInts);
        }
    }

    private async Task DecodeAsync(ArrayView1D<byte, Stride1D.Dense> input, int length, ArrayView1D<byte, Stride1D.Dense> output, int total)
    {
        if (total > output.Length) throw new Lz4FormatException(Lz4Error.OutputTooSmall, total);
        if (_blockCount == 0) return;
        var data = input.SubView(0, length);
        var outView = output.SubView(0, Math.Max(1, total));
        // Independent blocks: one thread each. Linked blocks: one thread walks them in order (mode 1) after the
        // independent pass skipped them.
        int steps = _maxBlock / DecodeBudget + 2;
        for (int s = 0; s < steps; s++)
        {
            _decode(_blockCount, data, outView, _blocks!.View, _frames!.View, _prm.View, 0);
            await _pacer.LaunchedAsync();
        }
        if (_anyLinked)
        {
            long linkedSteps = (long)total / DecodeBudget + 2L * _blockCount + 2;
            for (long s = 0; s < linkedSteps; s++)
            {
                _decode(1, data, outView, _blocks!.View, _frames!.View, _prm.View, 1);
                await _pacer.LaunchedAsync();
            }
        }
        int hashSteps = _maxStored / HashBudget + 2;
        for (int s = 0; s < hashSteps; s++)
        {
            _blockCheck(_blockCount, data, _blocks!.View, _prm.View);
            await _pacer.LaunchedAsync();
        }
        long frameHashSteps = (long)total / HashBudget + 2;
        for (long s = 0; s < frameHashSteps; s++)
        {
            _frameCheck(_frameCount, outView, _blocks!.View, _frames!.View, _prm.View);
            await _pacer.LaunchedAsync();
        }
        await _accelerator.SynchronizeAsync();
        ThrowIfError(await _prm.View.CopyToHostAsync());
    }

    private static void ThrowIfError(int[] prm)
    {
        if (prm[PError] != 0) throw new Lz4FormatException((Lz4Error)prm[PError], prm[PErrorAt]);
    }

    // ------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------

    private static uint Read32(ArrayView<byte> d, int at) =>
        (uint)d[at] | ((uint)d[at + 1] << 8) | ((uint)d[at + 2] << 16) | ((uint)d[at + 3] << 24);

    /// <summary>First error wins (single-thread kernels and per-block kernels both report through here).</summary>
    private static void Fail(ArrayView<int> prm, Lz4Error e, int at)
    {
        if (prm[PError] == 0)
        {
            prm[PError] = (int)e;
            prm[PErrorAt] = at;
        }
    }

    /// <summary>
    /// One thread: up to WalkHops frame / block headers. Count pass (prm[PRecord] = 0): counts blocks and frames.
    /// Record pass: fills the tables, and leaves the largest stored size in prm[PCapBlocks] and the largest block size in
    /// prm[PCapFrames], and 1 in prm[PLinkedBlock] when any frame has linked blocks.
    /// </summary>
    private static void WalkKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> blocks, ArrayView<int> frames, ArrayView<int> prm)
    {
        int state = prm[PState];
        if (state == WDone) return;
        if (prm[PError] != 0) return;
        int length = prm[PLength];
        int pos = prm[PPos];
        bool record = prm[PRecord] != 0;
        int nb = prm[PBlocks], nf = prm[PFrames];
        int frame = prm[PCurFrame];
        int fflags = prm[PCurFlags];
        int maxBlock = prm[PCurMax];
        int maxStored = 0, maxSeen = 0;
        int linked = prm[PLinkedBlock];
        if (record)
        {
            maxStored = prm[PMaxStored];
            maxSeen = prm[PMaxSeen];
        }

        int hops = 0;
        bool go = true;
        while (go)
        {
            if (hops >= WalkHops) { go = false; continue; }
            hops++;
            if (state == WMagic)
            {
                if (pos == length)
                {
                    state = WDone;
                    go = false;
                    continue;
                }
                if (pos + 4 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                uint magic = Read32(d, pos);
                if ((magic & 0xFFFFFFF0u) == MagicSkipLow)
                {
                    if (pos + 8 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    uint skip = Read32(d, pos + 4);
                    long next = (long)pos + 8 + skip;
                    if (next > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    pos = (int)next;
                }
                else if (magic == MagicLegacy)
                {
                    frame = nf;
                    if (record)
                    {
                        int fa = nf * FrameInts;
                        for (int k = 0; k < FrameInts; k++) frames[fa + k] = 0;
                        frames[fa + FFlags] = FLegacy;
                        frames[fa + FFirst] = nb;
                    }
                    nf++;
                    fflags = FLegacy;
                    maxBlock = LegacyBlock;
                    pos += 4;
                    state = WLegacy;
                }
                else if (magic == MagicFrame)
                {
                    if (pos + 7 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    int flg = d[pos + 4];
                    int bd = d[pos + 5];
                    bool bad = (flg >> 6) != 1;
                    if (bad) { Fail(prm, Lz4Error.BadVersion, pos + 4); go = false; continue; }
                    bool reserved = ((flg & 0x02) != 0) | ((bd & 0x8F) != 0);
                    if (reserved) { Fail(prm, Lz4Error.ReservedBits, pos + 4); go = false; continue; }
                    int bsid = (bd >> 4) & 7;
                    if (bsid < 4) { Fail(prm, Lz4Error.BadBlockSizeId, pos + 5); go = false; continue; }
                    if ((flg & 0x01) != 0) { Fail(prm, Lz4Error.DictionaryNotSupported, pos + 4); go = false; continue; }
                    int hlen = 2;
                    bool hasSize = (flg & 0x08) != 0;
                    if (hasSize) hlen += 8;
                    if (pos + 4 + hlen + 1 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    uint hc = (Xxh32Gpu.Hash(d, pos + 4, hlen, 0) >> 8) & 0xFF;
                    if (hc != d[pos + 4 + hlen]) { Fail(prm, Lz4Error.HeaderChecksum, pos + 4 + hlen); go = false; continue; }
                    maxBlock = 1 << (8 + 2 * bsid);
                    fflags = 0;
                    if (hasSize) fflags |= FHasSize;
                    if ((flg & 0x04) != 0) fflags |= FHasCheck;
                    if ((flg & 0x20) == 0) fflags |= FLinked;
                    // Block checksum flag rides in the high bits of the current flags.
                    if ((flg & 0x10) != 0) fflags |= 0x100;
                    frame = nf;
                    if (record)
                    {
                        int fa = nf * FrameInts;
                        for (int k = 0; k < FrameInts; k++) frames[fa + k] = 0;
                        frames[fa + FFlags] = fflags & 0xFF;
                        frames[fa + FFirst] = nb;
                        if (hasSize)
                        {
                            frames[fa + FSizeLo] = (int)Read32(d, pos + 6);
                            frames[fa + FSizeHi] = (int)Read32(d, pos + 10);
                        }
                    }
                    nf++;
                    pos += 4 + hlen + 1;
                    state = WBlocks;
                }
                else { Fail(prm, Lz4Error.BadMagic, pos); go = false; continue; }
            }
            else if (state == WBlocks)
            {
                if (pos + 4 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                uint word = Read32(d, pos);
                if (word == 0)
                {
                    pos += 4;
                    if ((fflags & FHasCheck) != 0)
                    {
                        if (pos + 4 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                        if (record) frames[frame * FrameInts + FCheck] = (int)Read32(d, pos);
                        pos += 4;
                    }
                    if (record) frames[frame * FrameInts + FCount] = nb - frames[frame * FrameInts + FFirst];
                    state = WMagic;
                }
                else
                {
                    int stored = (int)(word & 0x7FFFFFFFu);
                    bool raw = (word & 0x80000000u) != 0;
                    bool hasCheck = (fflags & 0x100) != 0;
                    if (stored > maxBlock) { Fail(prm, Lz4Error.BlockTooLarge, pos); go = false; continue; }
                    long end = (long)pos + 4 + stored + (hasCheck ? 4 : 0);
                    if (end > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    if (record)
                    {
                        int ba = nb * BlockInts;
                        for (int k = 0; k < BlockInts; k++) blocks[ba + k] = 0;
                        blocks[ba + BSrc] = pos + 4;
                        blocks[ba + BStored] = stored;
                        int bf = 0;
                        if (raw) bf |= BRaw;
                        if (hasCheck) bf |= BHasCheck;
                        if ((fflags & FLinked) != 0) bf |= BLinked;
                        blocks[ba + BFlags] = bf;
                        blocks[ba + BFrame] = frame;
                        blocks[ba + BMax] = maxBlock;
                        if (hasCheck) blocks[ba + BCheck] = (int)Read32(d, pos + 4 + stored);
                        if (stored > maxStored) maxStored = stored;
                        if (maxBlock > maxSeen) maxSeen = maxBlock;
                        if ((fflags & FLinked) != 0) linked = 1;
                    }
                    nb++;
                    pos = (int)end;
                }
            }
            else if (state == WLegacy)
            {
                // Legacy frame: blocks until the input ends or the next magic number.
                bool atEnd = pos == length;
                bool nextMagic = false;
                if (!atEnd)
                {
                    if (pos + 4 > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    uint word = Read32(d, pos);
                    nextMagic = (word == MagicLegacy) | (word == MagicFrame) | ((word & 0xFFFFFFF0u) == MagicSkipLow);
                }
                if (atEnd | nextMagic)
                {
                    if (record) frames[frame * FrameInts + FCount] = nb - frames[frame * FrameInts + FFirst];
                    state = WMagic;
                }
                else
                {
                    int stored = (int)Read32(d, pos);
                    if ((stored <= 0) | (stored > LegacyBlock + LegacyBlock / 255 + 16)) { Fail(prm, Lz4Error.BlockTooLarge, pos); go = false; continue; }
                    long end = (long)pos + 4 + stored;
                    if (end > length) { Fail(prm, Lz4Error.Truncated, pos); go = false; continue; }
                    if (record)
                    {
                        int ba = nb * BlockInts;
                        for (int k = 0; k < BlockInts; k++) blocks[ba + k] = 0;
                        blocks[ba + BSrc] = pos + 4;
                        blocks[ba + BStored] = stored;
                        blocks[ba + BFlags] = BLegacy;
                        blocks[ba + BFrame] = frame;
                        blocks[ba + BMax] = LegacyBlock;
                        if (stored > maxStored) maxStored = stored;
                        if (LegacyBlock > maxSeen) maxSeen = LegacyBlock;
                    }
                    nb++;
                    pos = (int)end;
                }
            }
            else go = false;
        }

        prm[PState] = state;
        prm[PPos] = pos;
        prm[PBlocks] = nb;
        prm[PFrames] = nf;
        prm[PCurFrame] = frame;
        prm[PCurFlags] = fflags;
        prm[PCurMax] = maxBlock;
        prm[PLinkedBlock] = linked;
        if (record)
        {
            prm[PMaxStored] = maxStored;
            prm[PMaxSeen] = maxSeen;
        }
        if ((state == WDone) & (nf == 0)) Fail(prm, Lz4Error.Empty, 0);
    }

    /// <summary>
    /// One thread per block, ScanBudget input bytes per launch: the decoded size from the sequence headers (raw blocks:
    /// the stored size), with the literal / end-of-block rules checked. Result in BSize, BPhase = 2 when done.
    /// </summary>
    private static void ScanKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> blocks, ArrayView<int> prm)
    {
        int ba = idx.X * BlockInts;
        int phase = blocks[ba + BPhase];
        if (phase == 2) return;
        int src = blocks[ba + BSrc];
        int stored = blocks[ba + BStored];
        int flags = blocks[ba + BFlags];
        if ((flags & BRaw) != 0)
        {
            blocks[ba + BSize] = stored;
            blocks[ba + BPhase] = 2;
            return;
        }
        int ip = blocks[ba + BIp];
        int size = blocks[ba + BAux];
        int end = src + stored;
        if (phase == 0) ip = src;
        int max = blocks[ba + BMax];
        int budgetEnd = ip + ScanBudget;
        bool go = true;
        bool done = false;
        while (go)
        {
            bool stop = ip >= budgetEnd;
            if (stop) { go = false; continue; }
            if (ip >= end) { Fail(prm, Lz4Error.BadSequence, src); go = false; done = true; continue; }
            int token = d[ip];
            ip++;
            int lit = token >> 4;
            if (lit == 15)
            {
                bool more = true;
                while (more)
                {
                    if (ip >= end) { more = false; lit = -1; }
                    else
                    {
                        int b = d[ip];
                        ip++;
                        lit += b;
                        more = b == 255;
                    }
                }
            }
            bool litBad = (lit < 0) | (ip + lit > end);
            if (litBad) { Fail(prm, Lz4Error.BadSequence, src); go = false; done = true; continue; }
            ip += lit;
            size += lit;
            if (ip == end)
            {
                // Last sequence: literals only.
                done = true;
                go = false;
                continue;
            }
            if (ip + 2 > end) { Fail(prm, Lz4Error.BadSequence, src); go = false; done = true; continue; }
            int offset = d[ip] | (d[ip + 1] << 8);
            ip += 2;
            if (offset == 0) { Fail(prm, Lz4Error.BadOffset, src); go = false; done = true; continue; }
            int ml = token & 15;
            if (ml == 15)
            {
                bool more = true;
                while (more)
                {
                    if (ip >= end) { more = false; ml = -1; }
                    else
                    {
                        int b = d[ip];
                        ip++;
                        ml += b;
                        more = b == 255;
                    }
                }
            }
            if (ml < 0) { Fail(prm, Lz4Error.BadSequence, src); go = false; done = true; continue; }
            size += ml + 4;
            if (size > max) { Fail(prm, Lz4Error.BlockTooLarge, src); go = false; done = true; continue; }
        }
        if (size > max) Fail(prm, Lz4Error.BlockTooLarge, src);
        blocks[ba + BIp] = ip;
        blocks[ba + BAux] = size;
        if (done)
        {
            blocks[ba + BSize] = size;
            blocks[ba + BPhase] = 2;
        }
        else blocks[ba + BPhase] = 1;
    }

    /// <summary>One thread, LayoutChunk blocks per launch: output offsets (and frame starts / totals); prm[PTotal] = running end.</summary>
    private static void LayoutKernel(Index1D idx, ArrayView<int> blocks, ArrayView<int> frames, ArrayView<int> prm)
    {
        int b0 = prm[PLayoutBlock];
        int count = prm[PBlocks];
        int pos = prm[PLayoutPos];
        int b1 = b0 + LayoutChunk;
        if (b1 > count) b1 = count;
        for (int b = b0; b < b1; b++)
        {
            int ba = b * BlockInts;
            if (blocks[ba + BPhase] != 2) Fail(prm, Lz4Error.BadSequence, blocks[ba + BSrc]);
            int f = blocks[ba + BFrame];
            int fa = f * FrameInts;
            if (frames[fa + FFirst] == b) frames[fa + FDst] = pos;
            blocks[ba + BDst] = pos;
            // Decode state starts fresh.
            blocks[ba + BIp] = blocks[ba + BSrc];
            blocks[ba + BOp] = 0;
            blocks[ba + BMatchLeft] = 0;
            blocks[ba + BSeq] = 0;
            blocks[ba + BLitLeft] = 0;
            blocks[ba + BAux] = 0;
            blocks[ba + BPhase] = 0;
            long next = (long)pos + blocks[ba + BSize];
            if (next > int.MaxValue) { Fail(prm, Lz4Error.TooLarge, b); next = pos; }
            pos = (int)next;
            frames[fa + FTotal] = pos - frames[fa + FDst];
        }
        prm[PLayoutBlock] = b1;
        prm[PLayoutPos] = pos;
        prm[PTotal] = pos;
    }

    /// <summary>
    /// Decode, DecodeBudget output bytes per launch and block (a long literal run or match resumes in the next launch).
    /// mode 0: one thread per block, independent blocks only (a match must stay inside its block). mode 1: one thread walks
    /// the linked blocks in order (prm[PLinkedNext] = the next one), each match reaching back at most to its frame's start.
    /// The structure was validated by the scan; the decoder re-checks only what the scan cannot know (match targets).
    /// </summary>
    private static void DecodeKernel(Index1D idx, ArrayView<byte> d, ArrayView<byte> o, ArrayView<int> blocks, ArrayView<int> frames,
        ArrayView<int> prm, int mode)
    {
        if (prm[PError] != 0) return;
        int count = prm[PBlocks];
        int b = idx.X;
        if (mode == 1)
        {
            b = prm[PLinkedNext];
            bool find = true;
            while (find)
            {
                if (b >= count) find = false;
                else
                {
                    int f = blocks[b * BlockInts + BFlags];
                    bool skip = ((f & BLinked) == 0) | (blocks[b * BlockInts + BPhase] == 2);
                    if (skip) b++;
                    else find = false;
                }
            }
            prm[PLinkedNext] = b;
            if (b >= count) return;
        }
        int ba = b * BlockInts;
        int flags = blocks[ba + BFlags];
        bool isLinked = (flags & BLinked) != 0;
        if ((mode == 0) & isLinked) return;
        if (blocks[ba + BPhase] == 2) return;
        int src = blocks[ba + BSrc];
        int stored = blocks[ba + BStored];
        int dst = blocks[ba + BDst];
        int size = blocks[ba + BSize];
        int op = blocks[ba + BOp];
        int ip = blocks[ba + BIp];
        int matchLeft = blocks[ba + BMatchLeft];
        int matchOff = blocks[ba + BMatchOff];
        int seq = blocks[ba + BSeq];
        int litLeft = blocks[ba + BLitLeft];
        int tok = blocks[ba + BTok];
        int end = src + stored;
        int budgetEnd = op + DecodeBudget;
        int low = dst;
        if (isLinked) low = frames[blocks[ba + BFrame] * FrameInts + FDst];
        bool done = false;
        if ((flags & BRaw) != 0)
        {
            int n = stored - op;
            if (n > DecodeBudget) n = DecodeBudget;
            for (int k = 0; k < n; k++) o[dst + op + k] = d[src + op + k];
            op += n;
            done = op >= stored;
        }
        else
        {
            bool go = true;
            while (go)
            {
                if (matchLeft > 0)
                {
                    int n = matchLeft;
                    if (n > budgetEnd - op) n = budgetEnd - op;
                    int from = dst + op - matchOff;
                    for (int k = 0; k < n; k++) o[dst + op + k] = o[from + k];
                    op += n;
                    matchLeft -= n;
                }
                bool stop = (matchLeft > 0) | (op >= budgetEnd);
                if (stop) { go = false; continue; }
                if (seq == 0)
                {
                    if (ip >= end) { done = true; go = false; continue; }
                    int token = d[ip];
                    ip++;
                    int lit = token >> 4;
                    if (lit == 15)
                    {
                        bool more = true;
                        while (more)
                        {
                            int bb = d[ip];
                            ip++;
                            lit += bb;
                            more = bb == 255;
                        }
                    }
                    litLeft = lit;
                    tok = token & 15;
                    seq = 1;
                }
                if (seq == 1)
                {
                    int n = litLeft;
                    if (n > budgetEnd - op) n = budgetEnd - op;
                    for (int k = 0; k < n; k++) o[dst + op + k] = d[ip + k];
                    ip += n;
                    op += n;
                    litLeft -= n;
                    if (litLeft > 0) { go = false; continue; }
                    seq = 2;
                }
                if (seq == 2)
                {
                    // Last sequence: literals only.
                    if (ip >= end) { done = true; go = false; continue; }
                    int offset = d[ip] | (d[ip + 1] << 8);
                    ip += 2;
                    int ml = tok;
                    if (ml == 15)
                    {
                        bool more = true;
                        while (more)
                        {
                            int bb = d[ip];
                            ip++;
                            ml += bb;
                            more = bb == 255;
                        }
                    }
                    ml += 4;
                    bool bad = dst + op - offset < low;
                    if (bad) { Fail(prm, Lz4Error.BadOffset, ip); go = false; done = true; continue; }
                    matchOff = offset;
                    matchLeft = ml;
                    seq = 0;
                }
            }
        }
        blocks[ba + BOp] = op;
        blocks[ba + BIp] = ip;
        blocks[ba + BMatchLeft] = matchLeft;
        blocks[ba + BMatchOff] = matchOff;
        blocks[ba + BSeq] = seq;
        blocks[ba + BLitLeft] = litLeft;
        blocks[ba + BTok] = tok;
        if (done)
        {
            if (op != size) Fail(prm, Lz4Error.BadSequence, src);
            blocks[ba + BPhase] = 2;
        }
        else blocks[ba + BPhase] = 1;
    }

    /// <summary>One thread per block: XXH32 of the stored bytes against the block checksum, HashBudget bytes per launch.</summary>
    private static void BlockCheckKernel(Index1D idx, ArrayView<byte> d, ArrayView<int> blocks, ArrayView<int> prm)
    {
        int ba = idx.X * BlockInts;
        int flags = blocks[ba + BFlags];
        if ((flags & BHasCheck) == 0) return;
        int x = ba + BXxh;
        int hashed = blocks[ba + BAux];
        int stored = blocks[ba + BStored];
        int src = blocks[ba + BSrc];
        if (hashed < 0) return; // done
        if (hashed == 0) Xxh32Gpu.Begin(blocks, x, 0);
        int whole = stored / 16;
        int doneStripes = hashed / 16;
        int n = whole - doneStripes;
        if (n > HashBudget / 16) n = HashBudget / 16;
        if (n > 0) Xxh32Gpu.Stripes(blocks, x, d, src + doneStripes * 16, n);
        hashed += n * 16;
        if (hashed >= whole * 16)
        {
            uint h = Xxh32Gpu.Finish(blocks, x, d, src + whole * 16, stored - whole * 16, 0);
            if (h != (uint)blocks[ba + BCheck]) Fail(prm, Lz4Error.BlockChecksum, idx.X);
            hashed = -1;
        }
        blocks[ba + BAux] = hashed;
    }

    /// <summary>One thread per frame: content size, and XXH32 of the frame's output against the content checksum, HashBudget bytes per launch.</summary>
    private static void FrameCheckKernel(Index1D idx, ArrayView<byte> o, ArrayView<int> blocks, ArrayView<int> frames, ArrayView<int> prm)
    {
        int fa = idx.X * FrameInts;
        int flags = frames[fa + FFlags];
        int hashed = frames[fa + FHashed];
        if (hashed < 0) return;
        int total = frames[fa + FTotal];
        int dst = frames[fa + FDst];
        if (hashed == 0)
        {
            bool sizeBad = ((flags & FHasSize) != 0) & ((frames[fa + FSizeHi] != 0) | (frames[fa + FSizeLo] != total));
            if (sizeBad) Fail(prm, Lz4Error.ContentSize, idx.X);
            if ((flags & FHasCheck) == 0)
            {
                frames[fa + FHashed] = -1;
                return;
            }
            Xxh32Gpu.Begin(frames, fa + FXxh, 0);
        }
        int whole = total / 16;
        int doneStripes = hashed / 16;
        int n = whole - doneStripes;
        if (n > HashBudget / 16) n = HashBudget / 16;
        if (n > 0) Xxh32Gpu.Stripes(frames, fa + FXxh, o, dst + doneStripes * 16, n);
        hashed += n * 16;
        if (hashed >= whole * 16)
        {
            uint h = Xxh32Gpu.Finish(frames, fa + FXxh, o, dst + whole * 16, total - whole * 16, 0);
            if (h != (uint)frames[fa + FCheck]) Fail(prm, Lz4Error.ContentChecksum, idx.X);
            hashed = -1;
        }
        frames[fa + FHashed] = hashed;
    }

    public void Dispose()
    {
        _prm.Dispose();
        _noBlocks.Dispose();
        _noFrames.Dispose();
        _blocks?.Dispose();
        _frames?.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
