// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe entropy coding kernel. Runs the per-MB tile-data
// entropy stage entirely GPU-resident: walks the partition tree in
// VP9's normative z-order, emits per-MB skip / Y mode / UV mode /
// per-plane coef tokens via Vp9BlockCoefEncoderGpu, and tracks the
// above + left context arrays the partition / mode / coef contexts
// read.
//
// One thread per frame for v1, sequential within the thread, but RESUMABLE: a launch codes at most
// LeavesPerLaunch leaf slots (16x16 positions of the superblock z-order walk) and stops, so no launch
// grows with the picture (a whole-frame launch held the GPU for tens of milliseconds at test sizes and
// would be seconds at 4K). The above / left context arrays live in a per-frame device region (they were
// per-thread LocalMemory), the bool encoder and the slot cursor in a small int state region; the host
// launches ceil(slots / LeavesPerLaunch) steps (no readback). The step that codes the last slot also
// stops the coder and writes the length; a step after that does nothing.
//
// V1 simplifications (mirror Vp9KeyframeEncoder.EncodeKeyFrame):
//   - Width + height multiples of 64 (single tile, integer-SB grid -
//     no boundary partition-forcing).
//   - Profile 0, YUV 4:2:0.
//   - Every leaf is Block16x16 + PARTITION_NONE.
//   - Every leaf encodes Y mode = DC_PRED, UV mode = DC_PRED.
//   - Skip flag hardcoded to 0 (matches CPU encoder v1).
//   - tx_mode = Allow32x32 (no per-block tx_size signalling).
//   - Default coef probs (no compressed-header updates).
//
// Walk order (depth-first z-order, mirrors libvpx encode_sb):
//   for sbRow:
//     reset left arrays
//     for sbCol:
//       Block64x64 -> SPLIT
//         topLeft 32x32 -> SPLIT
//           topLeft 16x16 -> NONE leaf
//           topRight 16x16 -> NONE leaf
//           botLeft 16x16  -> NONE leaf
//           botRight 16x16 -> NONE leaf
//         topRight 32x32 -> SPLIT (4 leaves)
//         botLeft 32x32  -> SPLIT (4 leaves)
//         botRight 32x32 -> SPLIT (4 leaves)
//
// Per leaf: skip_flag bit, intra_y_mode tree (1 bit for DC_PRED),
// intra_uv_mode tree (1 bit for DC_PRED), then 3 coef-token streams
// (Y Tx16x16, U Tx8x8, V Tx8x8) via Vp9BlockCoefEncoderGpu.
//
// Constant tables for prob lookups + scan/neighbor + cat probs +
// pareto8 etc. all come pre-packed via Vp9KeyframeConstantsGpu.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-frame slot strides for the batch entropy kernel.</summary>
public struct Vp9FrameEntropyBatchStrides
{
    /// <summary>Y coefs per frame (mbCount*256).</summary>
    public int YCoefStride;
    /// <summary>UV coefs per frame (mbCount*64).</summary>
    public int UvCoefStride;
    /// <summary>tile output bytes per frame.</summary>
    public int OutBufStride;
    /// <summary>mbCols (working dim).</summary>
    public int MbCols;
    /// <summary>mbRows (working dim).</summary>
    public int MbRows;
    /// <summary>
    /// Display-dim mi cols = (FrameWidth+7)>>3. Used for boundary
    /// forced-partition handling at SBs that straddle the right edge.
    /// </summary>
    public int FrameMiCols;
    /// <summary>Display-dim mi rows = (FrameHeight+7)>>3. Bottom-edge boundary.</summary>
    public int FrameMiRows;
}

/// <summary>
/// VP9 v1 keyframe entropy coding kernel. One thread per frame in bounded resumable launches of
/// <see cref="LeavesPerLaunch"/> leaf slots; emits the bool-coded tile bitstream from already-quantized
/// per-MB coefs.
/// </summary>
/// <remarks>Not thread-safe: the context and resumable-launch state buffers are owned by the instance.</remarks>
public sealed class Vp9FrameEntropyKernel : IDisposable
{
    /// <summary>
    /// V1 cap on miColsAligned (frame width / 8). At 512, frame width
    /// supports up to 4096px (covers 4K UHD 3840 wide). Throws at
    /// <see cref="RunAsync(ArrayView{short}, ArrayView{short}, ArrayView{short}, ArrayView{byte}, ArrayView{long}, ArrayView{byte}, ArrayView{ushort}, int, int, int, int)"/>
    /// time if exceeded so the limit surfaces early. The context arrays (~12 x MaxMiColsAligned bytes per
    /// frame) live in a device buffer owned by the kernel instance.
    /// </summary>
    public const int MaxMiColsAligned = 512;

    /// <summary>
    /// Leaf slots (16x16 positions of the superblock z-order walk, 16 per superblock) one launch codes per
    /// frame. A leaf is up to 384 coefficient tokens of serial bool coding (well under a millisecond on a
    /// desktop GPU), so a launch stays at a few milliseconds whatever the picture size.
    /// </summary>
    public const int LeavesPerLaunch = 8;

    // Per-frame context region (bytes): the above / left arrays of the walk, the same sizes the per-thread
    // LocalMemory arrays had (their lengths bound the context-cell indexing, so they must not change).
    internal const int CtxAboveYMode = 0;
    internal const int CtxAboveSkip = CtxAboveYMode + MaxMiColsAligned * 2;
    internal const int CtxAbovePart = CtxAboveSkip + MaxMiColsAligned;
    internal const int CtxAboveTx = CtxAbovePart + MaxMiColsAligned;
    internal const int CtxAboveYEnt = CtxAboveTx + MaxMiColsAligned;
    internal const int CtxAboveUEnt = CtxAboveYEnt + MaxMiColsAligned * 2;
    internal const int CtxAboveVEnt = CtxAboveUEnt + MaxMiColsAligned;
    internal const int CtxLeftYMode = CtxAboveVEnt + MaxMiColsAligned;
    internal const int CtxLeftSkip = CtxLeftYMode + 16;
    internal const int CtxLeftPart = CtxLeftSkip + 8;
    internal const int CtxLeftTx = CtxLeftPart + 8;
    internal const int CtxLeftYEnt = CtxLeftTx + 8;
    internal const int CtxLeftUEnt = CtxLeftYEnt + 16;
    internal const int CtxLeftVEnt = CtxLeftUEnt + 8;
    /// <summary>Context bytes per frame.</summary>
    internal const int CtxBytes = CtxLeftVEnt + 8;

    // Per-frame resumable state (ints).
    private const int StStarted = 0;
    private const int StDone = 1;
    private const int StNextSlot = 2;
    private const int StEnc = 3;
    private const int StateInts = StEnc + Vp8BoolEncoderGpu.StateInts;

    private readonly Accelerator _accelerator;
    private readonly StageBuffer<byte> _ctx;
    private readonly StageBuffer<int> _state;

    // Compile-time pressure note: the entropy walker (5K inlined call sites
    // / 52K Wasm locals per kernel module pre-2026-05-06) is heavy enough
    // on Wasm that emitting two near-identical modules (single + batch)
    // doubled cold-compile cost. Collapsed to ONE batch kernel; single-frame
    // dispatch goes through extent=1 + full-buffer strides. Saves one full
    // Wasm module's compile pass per backend at no runtime cost (the per-
    // dispatch SubView is a pointer-arithmetic no-op when stride == buffer
    // length). Fix at the codegen level (helper extraction so the inlined
    // body becomes a fn-def call) is Geordi's lane - tracked via DevComms
    // tuvok-to-geordi-codecs-wasm-monolithic-inlining-2026-05-06.md.
    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<ushort>,
        ArrayView<byte>, ArrayView<int>,
        Vp9FrameEntropyBatchStrides, int> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp9FrameEntropyKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _ctx = new StageBuffer<byte>(accelerator);
        _state = new StageBuffer<int>(accelerator);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>,
            ArrayView<byte>, ArrayView<int>,
            Vp9FrameEntropyBatchStrides, int>(Vp9EntropyStepKernel);
    }

    /// <summary>
    /// Run the entropy kernel (bounded launches, paced with <see cref="LaunchPacer"/>; the work is queued,
    /// not awaited to completion).
    /// </summary>
    /// <param name="yCoefs">Per-MB Y quantized coefs (mbCount * 256 shorts).</param>
    /// <param name="uCoefs">Per-MB U quantized coefs (mbCount * 64 shorts).</param>
    /// <param name="vCoefs">Per-MB V quantized coefs (mbCount * 64 shorts).</param>
    /// <param name="outBuf">Tile bytes output (worst-case sized).</param>
    /// <param name="outLen">1 long: actual byte count written.</param>
    /// <param name="byteConsts"><see cref="Vp9KeyframeConstantsGpu.BuildByteConstsBuffer"/> output.</param>
    /// <param name="ushortConsts"><see cref="Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer"/> output.</param>
    /// <param name="mbCols">Macroblock columns. Must be a multiple of 4 (SB-aligned) and <see cref="MaxMiColsAligned"/>/2.</param>
    /// <param name="mbRows">Macroblock rows. Must be a multiple of 4 (SB-aligned).</param>
    public Task RunAsync(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int mbCols, int mbRows)
    {
        // Single-arg form: assumes display dims == working dims (no boundary).
        return RunAsync(yCoefs, uCoefs, vCoefs, outBuf, outLen, byteConsts, ushortConsts,
            mbCols, mbRows, mbCols * 2, mbRows * 2);
    }

    /// <summary>
    /// Run with explicit display mi dims for spec-compliant boundary
    /// forced-partition handling at SBs that straddle the right/bottom edge.
    /// <paramref name="frameMiCols"/> = (DisplayWidth+7)>>3, similarly rows.
    /// </summary>
    public async Task RunAsync(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int mbCols, int mbRows,
        int frameMiCols, int frameMiRows)
    {
        if (mbCols <= 0 || (mbCols & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(mbCols), "mbCols must be a positive multiple of 4 (SB-aligned).");
        if (mbRows <= 0 || (mbRows & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(mbRows), "mbRows must be a positive multiple of 4 (SB-aligned).");
        if (mbCols * 2 > MaxMiColsAligned)
            throw new ArgumentOutOfRangeException(nameof(mbCols),
                $"v1 entropy kernel caps mbCols at {MaxMiColsAligned / 2}; got {mbCols}. Lift MaxMiColsAligned to grow.");
        if (frameMiCols <= 0 || frameMiRows <= 0
            || frameMiCols > mbCols * 2 || frameMiRows > mbRows * 2)
            throw new ArgumentOutOfRangeException(nameof(frameMiCols),
                "Display mi dims must be in (0, working*2].");
        if (outLen.Length < 1)
            throw new ArgumentException("outLen must hold 1 long.", nameof(outLen));

        // Single-frame dispatch goes through the batch kernel at extent=1
        // with full-buffer strides; the inner SubView(0, fullLength) is a
        // no-op pointer reset. Avoids compiling a second near-identical
        // entropy walker module per backend.
        var strides = new Vp9FrameEntropyBatchStrides
        {
            YCoefStride = checked((int)yCoefs.Length),
            UvCoefStride = checked((int)uCoefs.Length),
            OutBufStride = checked((int)outBuf.Length),
            MbCols = mbCols,
            MbRows = mbRows,
            FrameMiCols = frameMiCols,
            FrameMiRows = frameMiRows,
        };
        await RunStepsAsync(1, yCoefs, uCoefs, vCoefs, outBuf, outLen, byteConsts, ushortConsts, strides);
    }

    /// <summary>
    /// Batch entropy: extent=N, each thread walks one frame's MBs, <see cref="LeavesPerLaunch"/> leaf slots
    /// per launch.
    /// </summary>
    public async Task RunBatchAsync(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int frameCount, Vp9FrameEntropyBatchStrides strides)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        await RunStepsAsync(frameCount, yCoefs, uCoefs, vCoefs, outBuf, outLen, byteConsts, ushortConsts, strides);
    }

    private async Task RunStepsAsync(
        int frameCount,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        Vp9FrameEntropyBatchStrides strides)
    {
        long ctxLen = (long)frameCount * CtxBytes;
        long stateLen = (long)frameCount * StateInts;
        var ctx = (await _ctx.EnsureDrainedAsync(ctxLen)).SubView(0, ctxLen);
        var state = (await _state.EnsureDrainedAsync(stateLen)).SubView(0, stateLen);
        // Every context starts at zero (mode DC_PRED = 0, no skip, no partition / coefficient context).
        ctx.MemSetToZero();
        state.MemSetToZero();
        int slots = (strides.MbCols >> 2) * (strides.MbRows >> 2) * 16;
        int steps = (slots + LeavesPerLaunch - 1) / LeavesPerLaunch;
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < steps; s++)
        {
            _batchKernel(frameCount,
                yCoefs, uCoefs, vCoefs,
                outBuf, outLen,
                byteConsts, ushortConsts,
                ctx, state, strides, LeavesPerLaunch);
            await pacer.LaunchedAsync();
            await Vp9KeyframeEncoderGpu.TraceLaunchAsync(_accelerator, $"entropy step {s} of {steps} ({frameCount} frames)");
        }
    }

    /// <summary>One bounded step of the entropy walk: thread = one frame, the next leaf slots of its walk.</summary>
    private static void Vp9EntropyStepKernel(
        Index1D idx,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> ctx, ArrayView<int> stepState,
        Vp9FrameEntropyBatchStrides s, int leavesPerLaunch)
    {
        int f = idx.X;
        var fY = yCoefs.SubView((long)f * s.YCoefStride, s.YCoefStride);
        var fU = uCoefs.SubView((long)f * s.UvCoefStride, s.UvCoefStride);
        var fV = vCoefs.SubView((long)f * s.UvCoefStride, s.UvCoefStride);
        var fOut = outBuf.SubView((long)f * s.OutBufStride, s.OutBufStride);
        var fOutLen = outLen.SubView(f, 1);
        var fCtx = ctx.SubView((long)f * CtxBytes, CtxBytes);
        var fState = stepState.SubView((long)f * StateInts, StateInts);
        int frameMi = (s.FrameMiCols & 0xFFFF) | (s.FrameMiRows << 16);
        EncodeFrameStep(fY, fU, fV, fOut, fOutLen, byteConsts, ushortConsts, fCtx, fState,
            s.MbCols, s.MbRows, frameMi, leavesPerLaunch);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void EncodeFrameStep(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> ctx, ArrayView<int> st,
        int mbCols, int mbRows, int frameMi, int leavesPerLaunch)
    {
        if (st[StDone] != 0) return;

        // Context arrays (device memory, carried between launches). Sizes matter: above arrays are
        // MaxMiColsAligned * 2 for b4-cell granular ones (Y entropy + Y mode), MaxMiColsAligned for
        // mi-granular ones. Chroma above arrays are subsampled by 2 in the X dimension.
        var aboveYMode = ctx.SubView(CtxAboveYMode, MaxMiColsAligned * 2);
        var aboveSkip = ctx.SubView(CtxAboveSkip, MaxMiColsAligned);
        var abovePartCtx = ctx.SubView(CtxAbovePart, MaxMiColsAligned);
        var aboveTxSize = ctx.SubView(CtxAboveTx, MaxMiColsAligned);
        var aboveYEntropyCtx = ctx.SubView(CtxAboveYEnt, MaxMiColsAligned * 2);
        var aboveUEntropyCtx = ctx.SubView(CtxAboveUEnt, MaxMiColsAligned);
        var aboveVEntropyCtx = ctx.SubView(CtxAboveVEnt, MaxMiColsAligned);

        var leftYMode = ctx.SubView(CtxLeftYMode, 16);
        var leftSkip = ctx.SubView(CtxLeftSkip, 8);
        var leftPartCtx = ctx.SubView(CtxLeftPart, 8);
        var leftTxSize = ctx.SubView(CtxLeftTx, 8);
        var leftYEntropyCtx = ctx.SubView(CtxLeftYEnt, 16);
        var leftUEntropyCtx = ctx.SubView(CtxLeftUEnt, 8);
        var leftVEntropyCtx = ctx.SubView(CtxLeftVEnt, 8);

        // Per-block scratch (not carried between blocks).
        var tokenCache = LocalMemory.Allocate<byte>(256);

        Vp8BoolEncoderGpuState state;
        if (st[StStarted] == 0)
        {
            // Initialize bool encoder + emit VP9 marker bit.
            state = Vp8BoolEncoderGpu.Init();
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);
            st[StStarted] = 1;
        }
        else
        {
            state = Vp8BoolEncoderGpu.Load(st, StEnc);
        }

        // Walk SBs row-major, 16 leaf slots (z-order 16x16 positions) per SB. mbCols/mbRows are
        // multiples of 4 by contract so SB grid is exact.
        int sbCols = mbCols >> 2;
        int slotCount = (mbRows >> 2) * sbCols * 16;
        int slot = st[StNextSlot];
        int slotEnd = slot + leavesPerLaunch;
        if (slotEnd > slotCount) slotEnd = slotCount;
        bool go = slot < slotEnd;
        while (go)
        {
            EncodeLeafSlot(
                ref state, outBuf,
                yCoefs, uCoefs, vCoefs,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftPartCtx, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                tokenCache,
                slot, sbCols, mbCols, frameMi);
            slot++;
            go = slot < slotEnd;
        }

        if (slot >= slotCount)
        {
            Vp8BoolEncoderGpu.Stop(ref state, outBuf);
            outLen[0] = state.OutLen;
            st[StDone] = 1;
        }
        st[StNextSlot] = slot;
        Vp8BoolEncoderGpu.Save(st, StEnc, ref state);
    }

    /// <summary>
    /// One 16x16 position of the superblock z-order walk (slot = sb * 16 + q32 * 4 + q16): the partition
    /// symbols that precede it (the 64x64 one at the superblock's first slot, the 32x32 one at a 32x32 block's
    /// first slot) and the 16x16 block itself. Out-of-frame blocks are skipped exactly as the recursive walk
    /// skipped them.
    /// </summary>
    private static void EncodeLeafSlot(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int slot, int sbCols, int mbCols, int frameMi)
    {
        int frameMiCols = frameMi & 0xFFFF;
        int frameMiRows = frameMi >> 16;
        int sb = slot >> 4;
        int q = slot & 15;
        int sbRow = sb / sbCols;
        int sbCol = sb - sbRow * sbCols;
        int miRow64 = sbRow * 8;
        int miCol64 = sbCol * 8;

        if (q == 0)
        {
            if (sbCol == 0)
            {
                // Reset left arrays at the start of each SB row.
                for (int i = 0; i < 16; i++) { leftYMode[i] = 0; leftYEntropyCtx[i] = 0; }
                for (int i = 0; i < 8; i++) { leftSkip[i] = 0; leftPartCtx[i] = 0; leftTxSize[i] = 0; leftUEntropyCtx[i] = 0; leftVEntropyCtx[i] = 0; }
            }

            // SB64 boundary check: bsl=3, half=4 mi.
            bool hasRows64 = (miRow64 + 4) < frameMiRows;
            bool hasCols64 = (miCol64 + 4) < frameMiCols;
            if (hasRows64 & hasCols64)
            {
                // Full 4-way CDF, emit SPLIT.
                EmitPartitionSplit(ref state, outBuf, byteConsts,
                    sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64,
                    abovePartCtx, leftPartCtx);
            }
            else if (hasCols64)
            {
                // Bottom edge: 1-bit at probs[1] = SPLIT (1) vs HORZ (0).
                EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                    sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64,
                    abovePartCtx, leftPartCtx, probIdx: 1, value: 1);
            }
            else if (hasRows64)
            {
                // Right edge: 1-bit at probs[2] = SPLIT (1) vs VERT (0).
                EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                    sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64,
                    abovePartCtx, leftPartCtx, probIdx: 2, value: 1);
            }
            // else corner: forced split, no emit.
        }

        // The 32x32 child (z-order TL, TR, BL, BR); out-of-frame children (top-left at or past the frame
        // boundary) are skipped with everything under them.
        int q32 = q >> 2;
        int q16 = q & 3;
        int miRow32 = miRow64 + ((q32 & 2) >> 1) * 4;
        int miCol32 = miCol64 + (q32 & 1) * 4;
        bool in32 = (miRow32 < frameMiRows) & (miCol32 < frameMiCols);
        if (!in32) return;

        if (q16 == 0)
        {
            // SB32 boundary check: bsl=2, half=2 mi.
            bool hasRows32 = (miRow32 + 2) < frameMiRows;
            bool hasCols32 = (miCol32 + 2) < frameMiCols;
            if (hasRows32 & hasCols32)
            {
                EmitPartitionSplit(ref state, outBuf, byteConsts,
                    sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32,
                    abovePartCtx, leftPartCtx);
            }
            else if (hasCols32)
            {
                EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                    sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32,
                    abovePartCtx, leftPartCtx, probIdx: 1, value: 1);
            }
            else if (hasRows32)
            {
                EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                    sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32,
                    abovePartCtx, leftPartCtx, probIdx: 2, value: 1);
            }
            // else corner: forced split, no emit.
        }

        // The 16x16 child, skipping an out-of-frame one.
        int miRow16 = miRow32 + ((q16 & 2) >> 1) * 2;
        int miCol16 = miCol32 + (q16 & 1) * 2;
        bool in16 = (miRow16 < frameMiRows) & (miCol16 < frameMiCols);
        if (!in16) return;
        EncodeBlock16x16(
            ref state, outBuf,
            yCoefs, uCoefs, vCoefs,
            byteConsts, ushortConsts,
            aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
            aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
            leftYMode, leftSkip, leftPartCtx, leftTxSize,
            leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
            tokenCache,
            miRow16, miCol16, mbCols, frameMi);
    }

    private static void EncodeBlock16x16(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int miRow16, int miCol16, int mbCols, int frameMi)
    {
        int frameMiCols = frameMi & 0xFFFF;
        int frameMiRows = frameMi >> 16;

        // SB16 boundary check: bsl=1, half=1 mi.
        bool hasRows = (miRow16 + 1) < frameMiRows;
        bool hasCols = (miCol16 + 1) < frameMiCols;

        bool leaf = false;
        int is16x8 = 0;
        if (hasRows && hasCols)
        {
            // Standard path: NONE + BLOCK_16X16 leaf.
            EmitPartitionNone(ref state, outBuf, byteConsts,
                sizeIdx: 1, bsl: 1, miRow: miRow16, miCol: miCol16,
                abovePartCtx, leftPartCtx);
            leaf = true;
        }
        else if (hasCols)
        {
            // Bottom-edge boundary: emit PARTITION_HORZ (1-bit value 0 at probs[1]) and encode the top BLOCK_16X8
            // leaf only; the bottom sub-block's top-left is at miRow16+1 >= frameMiRows, so the decoder skips it.
            EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                sizeIdx: 1, bsl: 1, miRow: miRow16, miCol: miCol16,
                abovePartCtx, leftPartCtx, probIdx: 1, value: 0);
            leaf = true;
            is16x8 = 1;
        }
        // else hasRows-only or corner: not implemented in v1 (1920x1080
        // doesn't hit these cases). Tracked as task #23 for full 4K + arbitrary
        // dim support.

        if (leaf)
        {
            EncodeLeaf(
                ref state, outBuf,
                yCoefs, uCoefs, vCoefs,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                tokenCache,
                miRow16 >> 1, miCol16 >> 1, miRow16, miCol16, mbCols, is16x8);
        }

        // UpdatePartitionContext: same for 16x16+NONE and 16x16+HORZ in our path.
        for (int i = 0; i < 2; i++)
        {
            int c = miCol16 + i;
            int r = (miRow16 + i) & 7;
            if (c < MaxMiColsAligned) abovePartCtx[c] = 12;
            leftPartCtx[r] = 12;
        }
    }

    private static void EmitPartitionSplit(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol,
        ArrayView<byte> abovePartCtx, ArrayView<byte> leftPartCtx)
    {
        int leftIdx = miRow & 7;
        int aboveIdx = miCol;
        int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
        int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
        int splitState = leftBit * 2 + aboveBit;
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset
                       + ((long)sizeIdx * 4 + splitState) * 3;
        // SPLIT walks the partition tree as bits 1, 1, 1.
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, byteConsts[probsBase + 0]);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, byteConsts[probsBase + 1]);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, byteConsts[probsBase + 2]);
    }

    /// <summary>
    /// Emit the 1-bit partition decision used at SB-grid boundary edges.
    /// At the bottom-only boundary (!hasRows &amp;&amp; hasCols) the decoder reads
    /// probs[1] selecting between PARTITION_HORZ (bit 0) and PARTITION_SPLIT (bit 1).
    /// At the right-only boundary (hasRows &amp;&amp; !hasCols), probs[2] selects
    /// between PARTITION_VERT (0) and PARTITION_SPLIT (1).
    /// At the corner (!hasRows &amp;&amp; !hasCols) no bits are read; partition is
    /// implicitly SPLIT - this helper isn't called for that case.
    /// </summary>
    private static void EmitPartitionBoundary1Bit(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol,
        ArrayView<byte> abovePartCtx, ArrayView<byte> leftPartCtx,
        int probIdx, int value)
    {
        int leftIdx = miRow & 7;
        int aboveIdx = miCol;
        int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
        int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
        int splitState = leftBit * 2 + aboveBit;
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset
                       + ((long)sizeIdx * 4 + splitState) * 3;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, value, byteConsts[probsBase + probIdx]);
    }

    private static void EmitPartitionNone(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol,
        ArrayView<byte> abovePartCtx, ArrayView<byte> leftPartCtx)
    {
        int leftIdx = miRow & 7;
        int aboveIdx = miCol;
        int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
        int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
        int splitState = leftBit * 2 + aboveBit;
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset
                       + ((long)sizeIdx * 4 + splitState) * 3;
        // NONE walks the partition tree as bit 0 at probs[0].
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[probsBase + 0]);
    }

    /// <summary>
    /// One leaf: BLOCK_16X16 (<paramref name="is16x8"/> 0: one Tx16x16 Y + one Tx8x8 U + one Tx8x8 V) or the top-half
    /// BLOCK_16X8 at the bottom frame edge (1: two Tx8x8 Y + two Tx4x4 U + two Tx4x4 V, per the VP9 spec for BLOCK_16X8
    /// with TX_MODE=ALLOW_32X32). Skip flag, Y mode and UV mode are the same for both (DC_PRED, skip=0 in v1).
    /// The transforms are coded in ONE loop with one block-coder call site: written out per transform (3 + 6 call
    /// sites) the block coder was inlined 9 times, and the walker's WGSL ran Chrome's shader compiler (Tint) out of
    /// memory.
    /// </summary>
    private static void EncodeLeaf(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int mbR, int mbC, int miRow, int miCol, int mbCols, int is16x8)
    {
        // ---- Skip flag ----
        // skip_context = above + left (each is 0 or 1).
        int leftIdxMi = miRow & 7;
        int leftSkipBit = leftSkip[leftIdxMi];
        int aboveSkipBit = miCol < MaxMiColsAligned ? aboveSkip[miCol] : 0;
        int skipContext = aboveSkipBit + leftSkipBit;
        int skipFlag = 0; // v1: hardcoded 0.
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, skipFlag,
            byteConsts[Vp9KeyframeConstantsGpu.SkipProbsOffset + skipContext]);

        // ---- Y mode (DC_PRED, value 0) ----
        // Probs row indexed by (above_y_mode, left_y_mode). For DC_PRED the tree's first decision is at probs[0];
        // bit 0 means leaf Vp9IntraMode.DcPred.
        int b4Col = miCol * 2;
        int leftB4Idx = (miRow & 7) * 2;
        int aboveYCell = b4Col < MaxMiColsAligned * 2 ? aboveYMode[b4Col] : 0;
        int leftYCell = leftYMode[leftB4Idx];
        long yProbBase = Vp9KeyframeConstantsGpu.KfYModeProbsOffset
                       + (long)(aboveYCell * 10 + leftYCell) * 9;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[yProbBase + 0]);

        // ---- UV mode (DC_PRED) ----
        // Probs row indexed by Y mode. yMode = DcPred = 0.
        long uvProbBase = Vp9KeyframeConstantsGpu.KfUvModeProbsOffset; // + 0 * 9
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[uvProbBase + 0]);

        // ---- Update mode-info contexts ----
        // BLOCK_16X16: b4Wide = b4High = 4, miWide = miHigh = 2, Tx16x16. BLOCK_16X8: b4High = 2, miHigh = 1, Tx8x8.
        int b4High = is16x8 != 0 ? 2 : 4;
        int miHigh = is16x8 != 0 ? 1 : 2;
        byte txSize = (byte)(is16x8 != 0 ? (int)Vp9TxSize.Tx8x8 : (int)Vp9TxSize.Tx16x16);
        for (int i = 0; i < 4; i++)
        {
            int c = b4Col + i;
            if (c < MaxMiColsAligned * 2) aboveYMode[c] = 0; // DcPred
        }
        for (int i = 0; i < b4High; i++)
        {
            int r = (leftB4Idx + i) & 15;
            leftYMode[r] = 0; // DcPred
        }
        for (int i = 0; i < 2; i++)
        {
            int c = miCol + i;
            if (c < MaxMiColsAligned) { aboveSkip[c] = (byte)skipFlag; aboveTxSize[c] = txSize; }
        }
        for (int i = 0; i < miHigh; i++)
        {
            int r = (leftIdxMi + i) & 7;
            leftSkip[r] = (byte)skipFlag;
            leftTxSize[r] = txSize;
        }

        // ---- Per-transform coef emission, one call site ----
        // Y: 16x16 at (mbR*16, mbC*16): above cell mbC*4, left cell (mbR & 3)*4 (cells of 4 px); the 16x8 leaf's two
        // Tx8x8 are stored at yCoefs[mbIdx*256 + 0..63] and [+64..127]. U / V: 8x8 at (mbR*8, mbC*8): above cell mbC*2,
        // left cell (mbR & 3)*2; the 16x8 leaf's two Tx4x4 are at [mbIdx*64 + 0..15] and [+16..31].
        long mbIdx = (long)mbR * mbCols + mbC;
        int jobs = is16x8 != 0 ? 6 : 3;
        for (int j = 0; j < jobs; j++)
        {
            int plane = is16x8 != 0 ? (j >> 1) : j;
            int half = is16x8 != 0 ? (j & 1) : 0;
            long coefBase;
            int coefCount, aboveCell, leftCell, cellsPerTx, isTx4x4, tx, planeType;
            if (plane == 0)
            {
                coefBase = mbIdx * 256 + half * 64;
                coefCount = is16x8 != 0 ? 64 : 256;
                aboveCell = mbC * 4 + half * 2;
                leftCell = (mbR & 3) * 4;
                cellsPerTx = is16x8 != 0 ? 2 : 4;
                isTx4x4 = 0;
                tx = is16x8 != 0 ? (int)Vp9TxSize.Tx8x8 : (int)Vp9TxSize.Tx16x16;
                planeType = (int)Vp9BlockCoefEnums.PlaneType.Y;
            }
            else
            {
                coefBase = mbIdx * 64 + half * 16;
                coefCount = is16x8 != 0 ? 16 : 64;
                aboveCell = mbC * 2 + half;
                leftCell = (mbR & 3) * 2;
                cellsPerTx = is16x8 != 0 ? 1 : 2;
                isTx4x4 = is16x8;
                tx = is16x8 != 0 ? (int)Vp9TxSize.Tx4x4 : (int)Vp9TxSize.Tx8x8;
                planeType = (int)Vp9BlockCoefEnums.PlaneType.Uv;
            }
            EncodePlaneCoefs(
                ref state, outBuf, plane,
                yCoefs, uCoefs, vCoefs, coefBase, coefCount,
                byteConsts, ushortConsts,
                aboveCell, leftCell, cellsPerTx,
                isTx4x4, tx, planeType,
                tokenCache,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx);
        }
    }

    /// <summary>Entropy-context cell <paramref name="idx"/> of plane <paramref name="plane"/> (0 Y, 1 U, 2 V); 0 past the array.</summary>
    private static int ReadCtx(int plane, ArrayView<byte> y, ArrayView<byte> u, ArrayView<byte> v, int idx)
    {
        int r = 0;
        if (plane == 0) { if (idx < y.Length) r = y[idx]; }
        else if (plane == 1) { if (idx < u.Length) r = u[idx]; }
        else { if (idx < v.Length) r = v[idx]; }
        return r;
    }

    /// <summary>Writes entropy-context cell <paramref name="idx"/> of plane <paramref name="plane"/> (ignored past the array).</summary>
    private static void WriteCtx(int plane, ArrayView<byte> y, ArrayView<byte> u, ArrayView<byte> v, int idx, byte value)
    {
        if (plane == 0) { if (idx < y.Length) y[idx] = value; }
        else if (plane == 1) { if (idx < u.Length) u[idx] = value; }
        else { if (idx < v.Length) v[idx] = value; }
    }

    private static void EncodePlaneCoefs(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf, int plane,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs, long coefBase, int coefCount,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int aboveCellOff, int leftCellOff, int cellsPerTx,
        int isTx4x4, int txSizeForCoefProbs, int planeType,
        ArrayView<byte> tokenCache,
        ArrayView<byte> aboveY, ArrayView<byte> aboveU, ArrayView<byte> aboveV,
        ArrayView<byte> leftY, ArrayView<byte> leftU, ArrayView<byte> leftV)
    {
        // Initial coef context = (aboveAgg != 0) + (leftAgg != 0).
        int aboveAgg = 0;
        int leftAgg = 0;
        for (int i = 0; i < cellsPerTx; i++)
        {
            aboveAgg |= ReadCtx(plane, aboveY, aboveU, aboveV, aboveCellOff + i);
            leftAgg |= ReadCtx(plane, leftY, leftU, leftV, leftCellOff + i);
        }
        int initialCtx = (aboveAgg != 0 ? 1 : 0) + (leftAgg != 0 ? 1 : 0);

        // Pick scan + neighbors based on tx size.
        long scanBase, neighborsBase, coefProbsBase;
        if (txSizeForCoefProbs == (int)Vp9TxSize.Tx4x4)
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan4x4Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors4x4Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs4x4Offset;
        }
        else if (txSizeForCoefProbs == (int)Vp9TxSize.Tx8x8)
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan8x8Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors8x8Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs8x8Offset;
        }
        else // Tx16x16
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan16x16Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors16x16Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs16x16Offset;
        }

        // The transform's coefficients, copied from their plane (one small loop per plane: a loop is never
        // if-converted, so no plane is read out of range).
        var block = LocalMemory.Allocate<short>(256);
        if (plane == 0) { for (int i = 0; i < coefCount; i++) block[i] = yCoefs[coefBase + i]; }
        else if (plane == 1) { for (int i = 0; i < coefCount; i++) block[i] = uCoefs[coefBase + i]; }
        else { for (int i = 0; i < coefCount; i++) block[i] = vCoefs[coefBase + i]; }

        var scanView = ushortConsts.SubView(scanBase, coefCount);
        var neighborsView = ushortConsts.SubView(neighborsBase, (long)coefCount * 2);
        var coefProbsView = byteConsts.SubView(coefProbsBase, 432);
        var coefsView = block.SubView(0, coefCount);
        var coefConstsView = byteConsts.SubView(
            Vp9KeyframeConstantsGpu.CoefConstsOffset,
            Vp9KeyframeConstantsGpu.CoefConstsLength);

        int eob = Vp9BlockCoefEncoderGpu.EncodeBlock(
            ref state, outBuf,
            coefsView,
            scanView, neighborsView,
            coefProbsView, coefConstsView, tokenCache,
            coefCount,
            planeType: planeType,
            refType: (int)Vp9BlockCoefEnums.RefType.Intra,
            initialCtx: initialCtx,
            isHighBitDepth: 0,
            isTx4x4: isTx4x4);

        // Update entropy context cells: each cell becomes (eob > 0 ? 1 : 0).
        byte ec = (byte)(eob > 0 ? 1 : 0);
        for (int i = 0; i < cellsPerTx; i++)
        {
            WriteCtx(plane, aboveY, aboveU, aboveV, aboveCellOff + i, ec);
            WriteCtx(plane, leftY, leftU, leftV, leftCellOff + i, ec);
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose()
    {
        _ctx.Dispose();
        _state.Dispose();
    }
}
