// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame-level entropy coding kernel. Runs the per-MB entropy
// stage entirely GPU-resident: writes mode bits to partition0 and
// coef tokens to a single token partition (v1 = npart=1).
//
// One thread per frame for v1, sequential within the thread, but RESUMABLE: a launch codes at most
// MbsPerLaunch macroblocks and stops, so no launch grows with the picture (a single launch over a whole
// frame held the GPU for tens of milliseconds at test sizes and seconds at 4K, long enough to stall the
// desktop compositor and trip the OS GPU watchdog). Everything carried from one macroblock to the next -
// both bool encoders, the left non-zero contexts and the macroblock cursor - lives in a small per-frame
// int state region (StateInts) between launches; the above contexts were already in device memory. The
// host launches ceil(mbCount / MbsPerLaunch) steps (a bound it knows, no readback); the step that codes
// the last macroblock also stops both coders and writes the lengths, and a step after that does nothing.
//
// What this kernel does NOT do (caller responsibility):
// - Frame header writing (caller pre-populates partition0 with the
//   frame header bits via the CPU Vp8FrameHeaderWriter, then runs
//   this kernel which APPENDS the per-MB modes after the header).
//   For the v1 unit test we skip the header entirely - both CPU and
//   GPU encoders skip it for an apples-to-apples comparison.
// - npart > 1 token partitions.
// - Inter modes / B_PRED / 4x4 mode trees (v1 assumes all DC_PRED).
//
// Constants buffer layout (constsExtended) - 62 bytes total:
//   Bytes  0..55  : Vp8CoefBlockEncoderGpu standard 56-byte layout
//                   (zigzag + bands + cat3-6).
//   Bytes 56..58  : kfYModeProbs (3 bytes - DefaultKfYModeProb[0..2]).
//   Bytes 59..61  : kfUvModeProbs (3 bytes - DefaultKfUvModeProb[0..2]).
// Caller materializes via ExtendConstsBuffer().

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>Per-frame slot strides for the batch entropy kernel.</summary>
public struct Vp8FrameEntropyBatchStrides
{
    /// <summary>Y4 coefs per frame (mbCount * 256).</summary>
    public int Y4Stride;
    /// <summary>Y2 coefs per frame (mbCount * 16).</summary>
    public int Y2Stride;
    /// <summary>UV coefs per frame (mbCount * 64).</summary>
    public int UvStride;
    /// <summary>partition0Out bytes per frame.</summary>
    public int P0Stride;
    /// <summary>tokenP0Out bytes per frame.</summary>
    public int TpStride;
    /// <summary>aboveCtx bytes per frame (mbCols * 9).</summary>
    public int AboveStride;
    /// <summary>mbCols.</summary>
    public int MbCols;
    /// <summary>mbRows.</summary>
    public int MbRows;
}

/// <summary>
/// VP8 frame-level entropy coding kernel. v1: one thread per frame, all MBs DC_PRED, npart=1, run as
/// bounded resumable launches of <see cref="MbsPerLaunch"/> macroblocks. Writes per-MB modes to partition0
/// and coefs to a single token partition - all GPU-resident.
/// </summary>
/// <remarks>Not thread-safe: the resumable-launch state buffer is owned by the instance.</remarks>
public sealed class Vp8FrameEntropyKernel : IDisposable
{
    /// <summary>Total bytes in the extended constants buffer.</summary>
    public const int ConstsExtendedTotalBytes = 62;
    /// <summary>kfYModeProbs offset within constsExtended.</summary>
    public const int KfYModeProbsOffset = 56;
    /// <summary>kfUvModeProbs offset within constsExtended.</summary>
    public const int KfUvModeProbsOffset = 59;

    /// <summary>
    /// Macroblocks one launch codes per frame. A macroblock is up to 25 coefficient blocks of 16 tokens (~0.2 ms
    /// of serial bool coding for a busy one on a desktop GPU), so a launch stays at a few milliseconds whatever
    /// the picture size.
    /// </summary>
    public const int MbsPerLaunch = 16;

    // Per-frame resumable state (ints).
    private const int StStarted = 0;    // 0 until the first step initializes the coders
    private const int StDone = 1;       // 1 once both coders are stopped and the lengths written
    private const int StNextMb = 2;     // next macroblock (raster index)
    private const int StP0 = 3;         // partition0 bool encoder (Vp8BoolEncoderGpu.StateInts)
    private const int StTp = StP0 + Vp8BoolEncoderGpu.StateInts;   // token partition bool encoder
    private const int StLeft = StTp + Vp8BoolEncoderGpu.StateInts; // 9 left non-zero contexts
    /// <summary>Ints of resumable state per frame.</summary>
    public const int StateInts = StLeft + 9;

    private readonly Accelerator _accelerator;
    private readonly StageBuffer<int> _state;

    // Batch kernel handles both single-frame and N-frame dispatches; single-
    // frame goes through extent=1 with full-buffer strides. See VP9 entropy
    // kernel for the same pattern - collapsed 2026-05-06 to halve cold-
    // compile cost on Wasm where the inlined entropy walker body is the
    // dominant compile-time pressure (Wasm structural analysis: 5K inlined
    // call sites + 52K locals per kernel module pre-collapse).
    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<int>, ArrayView<int>,
        Vp8FrameEntropyBatchStrides, int> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp8FrameEntropyKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _state = new StageBuffer<int>(accelerator);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<int>, ArrayView<int>,
            Vp8FrameEntropyBatchStrides, int>(Vp8EntropyStepKernel);
    }

    /// <summary>
    /// Build the extended consts buffer (56-byte standard + 6 mode-prob bytes).
    /// </summary>
    public static byte[] BuildExtendedConstsBuffer()
    {
        var buf = new byte[ConstsExtendedTotalBytes];
        var standard = Vp8CoefBlockEncoderGpu.BuildConstsBuffer();
        Array.Copy(standard, buf, standard.Length);
        var kfY = Vp8ModeProbTables.DefaultKfYModeProb;
        var kfUv = Vp8ModeProbTables.DefaultKfUvModeProb;
        Array.Copy(kfY, 0, buf, KfYModeProbsOffset, 3);
        Array.Copy(kfUv, 0, buf, KfUvModeProbsOffset, 3);
        return buf;
    }

    /// <summary>
    /// Encode one frame's per-MB entropy (bounded launches of <see cref="MbsPerLaunch"/> macroblocks, paced with
    /// <see cref="LaunchPacer"/>). The work is queued, not awaited to completion.
    /// </summary>
    /// <param name="y4Coefs">mbCount * 16 * 16 shorts.</param>
    /// <param name="y2Coefs">mbCount * 16 shorts.</param>
    /// <param name="uCoefs">mbCount * 4 * 16 shorts.</param>
    /// <param name="vCoefs">mbCount * 4 * 16 shorts.</param>
    /// <param name="coefProbsByType">4 block types * 264 bytes (Y_no_DC=0, Y2=1, UV=2, Y_with_DC=3).</param>
    /// <param name="constsExtended">62-byte buffer per <see cref="BuildExtendedConstsBuffer"/>.</param>
    /// <param name="partition0Out">Per-MB modes output bool stream (worst-case sized).</param>
    /// <param name="tokenP0Out">Coef tokens output bool stream.</param>
    /// <param name="outLens">2 longs: [partition0Len, tokenP0Len].</param>
    /// <param name="aboveCtx">Frame-wide above-context buffer (mbCols * 9 bytes). Caller zero-initializes.</param>
    /// <param name="initialP0State">Initial partition0 bool encoder state (5 ints): [LowValue, Range, Count, OutLen, _]. Encoded as int because of ILGPU type-handling rules; LowValue/Range fit in 32 bits unsigned; Count is signed (starts at -24); OutLen is the byte offset into partition0Out where partition0 currently ends. The 5th int is reserved/unused. Pass [0, 255, -24, 0, 0] to start fresh.</param>
    /// <param name="mbCols">Macroblock columns.</param>
    /// <param name="mbRows">Macroblock rows.</param>
    public async Task RunAsync(
        ArrayView<short> y4Coefs,
        ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs,
        ArrayView<short> vCoefs,
        ArrayView<byte> coefProbsByType,
        ArrayView<byte> constsExtended,
        ArrayView<byte> partition0Out,
        ArrayView<byte> tokenP0Out,
        ArrayView<long> outLens,
        ArrayView<byte> aboveCtx,
        ArrayView<int> initialP0State,
        int mbCols, int mbRows)
    {
        if (mbCols <= 0 || mbRows <= 0) throw new ArgumentOutOfRangeException();
        if (outLens.Length < 2) throw new ArgumentException("outLens must hold 2 longs.", nameof(outLens));
        if (aboveCtx.Length < mbCols * 9L) throw new ArgumentException("aboveCtx must hold mbCols*9 bytes.", nameof(aboveCtx));
        if (constsExtended.Length < ConstsExtendedTotalBytes)
            throw new ArgumentException("constsExtended must be at least ConstsExtendedTotalBytes.", nameof(constsExtended));
        if (initialP0State.Length < 5)
            throw new ArgumentException("initialP0State must hold 5 ints.", nameof(initialP0State));
        // Single-frame goes through batch kernel at extent=1 with full-buffer
        // strides; SubView(0, fullLength) is a pointer-arithmetic no-op.
        var strides = new Vp8FrameEntropyBatchStrides
        {
            Y4Stride = checked((int)y4Coefs.Length),
            Y2Stride = checked((int)y2Coefs.Length),
            UvStride = checked((int)uCoefs.Length),
            P0Stride = checked((int)partition0Out.Length),
            TpStride = checked((int)tokenP0Out.Length),
            AboveStride = checked((int)aboveCtx.Length),
            MbCols = mbCols,
            MbRows = mbRows,
        };
        await RunStepsAsync(1,
            y4Coefs, y2Coefs, uCoefs, vCoefs,
            coefProbsByType, constsExtended,
            partition0Out, tokenP0Out, outLens,
            aboveCtx, initialP0State, strides);
    }

    /// <summary>
    /// Batch entropy: encode <paramref name="frameCount"/> frames in parallel
    /// (extent = frameCount, one thread per frame, each launch codes <see cref="MbsPerLaunch"/>
    /// macroblocks of every frame). Each per-frame buffer is laid out as N concatenated slots, sliced via
    /// SubView in the kernel using the per-frame strides supplied in <paramref name="strides"/>.
    /// outLens is layered as 2 longs per frame: [F*2+0]=p0Len, [F*2+1]=tpLen.
    /// </summary>
    public async Task RunBatchAsync(
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> coefProbsByType, ArrayView<byte> constsExtended,
        ArrayView<byte> partition0Out, ArrayView<byte> tokenP0Out,
        ArrayView<long> outLens,
        ArrayView<byte> aboveCtx, ArrayView<int> initialP0State,
        int frameCount, Vp8FrameEntropyBatchStrides strides)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        await RunStepsAsync(frameCount,
            y4Coefs, y2Coefs, uCoefs, vCoefs,
            coefProbsByType, constsExtended,
            partition0Out, tokenP0Out, outLens,
            aboveCtx, initialP0State, strides);
    }

    private async Task RunStepsAsync(
        int frameCount,
        ArrayView<short> y4Coefs, ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> coefProbsByType, ArrayView<byte> constsExtended,
        ArrayView<byte> partition0Out, ArrayView<byte> tokenP0Out,
        ArrayView<long> outLens,
        ArrayView<byte> aboveCtx, ArrayView<int> initialP0State,
        Vp8FrameEntropyBatchStrides strides)
    {
        long stateLen = (long)frameCount * StateInts;
        var state = (await _state.EnsureDrainedAsync(stateLen)).SubView(0, stateLen);
        state.MemSetToZero();
        int mbCount = strides.MbCols * strides.MbRows;
        int steps = (mbCount + MbsPerLaunch - 1) / MbsPerLaunch;
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < steps; s++)
        {
            _batchKernel(frameCount,
                y4Coefs, y2Coefs, uCoefs, vCoefs,
                coefProbsByType, constsExtended,
                partition0Out, tokenP0Out, outLens,
                aboveCtx, initialP0State, state, strides, MbsPerLaunch);
            await pacer.LaunchedAsync();
        }
    }

    /// <summary>
    /// One bounded step of the entropy walk: each thread picks its frame slot via SubView and codes the next
    /// <paramref name="mbsPerLaunch"/> macroblocks of that frame. Single-frame dispatch goes through extent=1
    /// with full-buffer strides (SubView(0, fullLen) is a no-op).
    /// </summary>
    private static void Vp8EntropyStepKernel(
        Index1D idx,
        ArrayView<short> y4Coefs,
        ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs,
        ArrayView<short> vCoefs,
        ArrayView<byte> coefProbsByType,
        ArrayView<byte> constsExtended,
        ArrayView<byte> partition0Out,
        ArrayView<byte> tokenP0Out,
        ArrayView<long> outLens,
        ArrayView<byte> aboveCtx,
        ArrayView<int> initialP0State,
        ArrayView<int> stepState,
        Vp8FrameEntropyBatchStrides s,
        int mbsPerLaunch)
    {
        int f = idx.X;
        var fy4 = y4Coefs.SubView((long)f * s.Y4Stride, s.Y4Stride);
        var fy2 = y2Coefs.SubView((long)f * s.Y2Stride, s.Y2Stride);
        var fu = uCoefs.SubView((long)f * s.UvStride, s.UvStride);
        var fv = vCoefs.SubView((long)f * s.UvStride, s.UvStride);
        var fp0 = partition0Out.SubView((long)f * s.P0Stride, s.P0Stride);
        var ftp = tokenP0Out.SubView((long)f * s.TpStride, s.TpStride);
        var fOutLens = outLens.SubView((long)f * 2, 2);
        var fAbove = aboveCtx.SubView((long)f * s.AboveStride, s.AboveStride);
        var fInitState = initialP0State.SubView((long)f * 5, 5);
        var fStep = stepState.SubView((long)f * StateInts, StateInts);
        EncodeFrameStep(
            fy4, fy2, fu, fv,
            coefProbsByType, constsExtended,
            fp0, ftp, fOutLens,
            fAbove, fInitState, fStep,
            s.MbCols, s.MbRows, mbsPerLaunch);
    }

    /// <summary>
    /// Static body of one step. Takes already-frame-scoped views (callers SubView for batch).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void EncodeFrameStep(
        ArrayView<short> y4Coefs,
        ArrayView<short> y2Coefs,
        ArrayView<short> uCoefs,
        ArrayView<short> vCoefs,
        ArrayView<byte> coefProbsByType,
        ArrayView<byte> constsExtended,
        ArrayView<byte> partition0Out,
        ArrayView<byte> tokenP0Out,
        ArrayView<long> outLens,
        ArrayView<byte> aboveCtx,
        ArrayView<int> initialP0State,
        ArrayView<int> st,
        int mbCols, int mbRows, int mbsPerLaunch)
    {
        if (st[StDone] != 0) return;

        // Probe slices in coefProbsByType. Block types: 0=Y_no_DC,
        // 1=Y2, 2=UV, 3=Y_with_DC.
        const int probsPerType = 8 * 33;
        var probsY4 = coefProbsByType.SubView(0L * probsPerType, probsPerType);
        var probsY2 = coefProbsByType.SubView(1L * probsPerType, probsPerType);
        var probsUv = coefProbsByType.SubView(2L * probsPerType, probsPerType);

        Vp8BoolEncoderGpuState p0State;
        Vp8BoolEncoderGpuState tpState;
        if (st[StStarted] == 0)
        {
            // First step: partition0 continues from the snapshot the setup kernel took after writing the
            // frame header; tokenP0 always starts fresh. Left contexts start at zero (state was cleared).
            p0State = new Vp8BoolEncoderGpuState
            {
                LowValue = (uint)initialP0State[0],
                Range = (uint)initialP0State[1],
                Count = initialP0State[2],
                OutLen = (long)initialP0State[3],
            };
            tpState = Vp8BoolEncoderGpu.Init();
            st[StStarted] = 1;
        }
        else
        {
            p0State = Vp8BoolEncoderGpu.Load(st, StP0);
            tpState = Vp8BoolEncoderGpu.Load(st, StTp);
        }

        // Per-row left context (9 cells); reset per row, carried across steps in st[StLeft ..].
        int leftY4_0 = st[StLeft + 0], leftY4_1 = st[StLeft + 1], leftY4_2 = st[StLeft + 2], leftY4_3 = st[StLeft + 3];
        int leftU_0 = st[StLeft + 4], leftU_1 = st[StLeft + 5];
        int leftV_0 = st[StLeft + 6], leftV_1 = st[StLeft + 7];
        int leftY2 = st[StLeft + 8];

        // Mode probs read once into registers.
        byte kfYProb0 = constsExtended[56];
        byte kfYProb1 = constsExtended[57];
        byte kfYProb2 = constsExtended[58];
        byte kfUvProb0 = constsExtended[59];

        int mbCount = mbCols * mbRows;
        int mbIdx = st[StNextMb];
        int mbEnd = mbIdx + mbsPerLaunch;
        if (mbEnd > mbCount) mbEnd = mbCount;
        bool go = mbIdx < mbEnd;
        while (go)
        {
            int mbRow = mbIdx / mbCols;
            int mbCol = mbIdx - mbRow * mbCols;
            if (mbCol == 0)
            {
                leftY4_0 = 0; leftY4_1 = 0; leftY4_2 = 0; leftY4_3 = 0;
                leftU_0 = 0; leftU_1 = 0;
                leftV_0 = 0; leftV_1 = 0;
                leftY2 = 0;
            }

            long aboveBase = (long)mbCol * 9;

            // 1. Encode Y mode = DcPred (path: bit 1, bit 0, bit 0).
            Vp8BoolEncoderGpu.EncodeBool(ref p0State, partition0Out, 1, kfYProb0);
            Vp8BoolEncoderGpu.EncodeBool(ref p0State, partition0Out, 0, kfYProb1);
            Vp8BoolEncoderGpu.EncodeBool(ref p0State, partition0Out, 0, kfYProb2);

            // 2. Encode UV mode = DcPred (single bit 0).
            Vp8BoolEncoderGpu.EncodeBool(ref p0State, partition0Out, 0, kfUvProb0);

            // 3. Y2 (block type 1, firstCoef=0).
            int y2Ctx = aboveCtx[aboveBase + 8] + leftY2;
            long y2Base = (long)mbIdx * 16;
            int y2Eob = Vp8CoefBlockEncoderGpu.Encode(
                ref tpState, tokenP0Out, probsY2, constsExtended,
                y2Ctx, 0, y2Coefs.SubView(y2Base, 16));
            int y2HasCoef = y2Eob > 0 ? 1 : 0;
            aboveCtx[aboveBase + 8] = (byte)y2HasCoef;
            leftY2 = y2HasCoef;

            // 4. Y4 16 blocks (block type 0, firstCoef=1).
            long y4McBase = (long)mbIdx * 256;
            for (int by = 0; by < 4; by++)
            {
                for (int bx = 0; bx < 4; bx++)
                {
                    int blockIdxInMb = by * 4 + bx;
                    int aboveVal;
                    if (bx == 0) aboveVal = aboveCtx[aboveBase + 0];
                    else if (bx == 1) aboveVal = aboveCtx[aboveBase + 1];
                    else if (bx == 2) aboveVal = aboveCtx[aboveBase + 2];
                    else aboveVal = aboveCtx[aboveBase + 3];

                    int leftVal;
                    if (by == 0) leftVal = leftY4_0;
                    else if (by == 1) leftVal = leftY4_1;
                    else if (by == 2) leftVal = leftY4_2;
                    else leftVal = leftY4_3;

                    int ctx = aboveVal + leftVal;
                    int eob = Vp8CoefBlockEncoderGpu.Encode(
                        ref tpState, tokenP0Out, probsY4, constsExtended,
                        ctx, 1,
                        y4Coefs.SubView(y4McBase + (long)blockIdxInMb * 16, 16));
                    int hasCoef = eob > 0 ? 1 : 0;

                    if (bx == 0) aboveCtx[aboveBase + 0] = (byte)hasCoef;
                    else if (bx == 1) aboveCtx[aboveBase + 1] = (byte)hasCoef;
                    else if (bx == 2) aboveCtx[aboveBase + 2] = (byte)hasCoef;
                    else aboveCtx[aboveBase + 3] = (byte)hasCoef;

                    if (by == 0) leftY4_0 = hasCoef;
                    else if (by == 1) leftY4_1 = hasCoef;
                    else if (by == 2) leftY4_2 = hasCoef;
                    else leftY4_3 = hasCoef;
                }
            }

            // 5. U 4 blocks (block type 2, firstCoef=0).
            long uMcBase = (long)mbIdx * 4 * 16;
            for (int by = 0; by < 2; by++)
            {
                for (int bx = 0; bx < 2; bx++)
                {
                    int blockIdx = by * 2 + bx;
                    int aboveVal = (bx == 0)
                        ? (int)aboveCtx[aboveBase + 4]
                        : (int)aboveCtx[aboveBase + 5];
                    int leftVal = (by == 0) ? leftU_0 : leftU_1;
                    int ctx = aboveVal + leftVal;
                    int eob = Vp8CoefBlockEncoderGpu.Encode(
                        ref tpState, tokenP0Out, probsUv, constsExtended,
                        ctx, 0,
                        uCoefs.SubView(uMcBase + (long)blockIdx * 16, 16));
                    int hasCoef = eob > 0 ? 1 : 0;
                    if (bx == 0) aboveCtx[aboveBase + 4] = (byte)hasCoef;
                    else aboveCtx[aboveBase + 5] = (byte)hasCoef;
                    if (by == 0) leftU_0 = hasCoef;
                    else leftU_1 = hasCoef;
                }
            }

            // 6. V 4 blocks (block type 2, firstCoef=0).
            long vMcBase = (long)mbIdx * 4 * 16;
            for (int by = 0; by < 2; by++)
            {
                for (int bx = 0; bx < 2; bx++)
                {
                    int blockIdx = by * 2 + bx;
                    int aboveVal = (bx == 0)
                        ? (int)aboveCtx[aboveBase + 6]
                        : (int)aboveCtx[aboveBase + 7];
                    int leftVal = (by == 0) ? leftV_0 : leftV_1;
                    int ctx = aboveVal + leftVal;
                    int eob = Vp8CoefBlockEncoderGpu.Encode(
                        ref tpState, tokenP0Out, probsUv, constsExtended,
                        ctx, 0,
                        vCoefs.SubView(vMcBase + (long)blockIdx * 16, 16));
                    int hasCoef = eob > 0 ? 1 : 0;
                    if (bx == 0) aboveCtx[aboveBase + 6] = (byte)hasCoef;
                    else aboveCtx[aboveBase + 7] = (byte)hasCoef;
                    if (by == 0) leftV_0 = hasCoef;
                    else leftV_1 = hasCoef;
                }
            }

            mbIdx++;
            go = mbIdx < mbEnd;
        }

        if (mbIdx >= mbCount)
        {
            Vp8BoolEncoderGpu.Stop(ref p0State, partition0Out);
            Vp8BoolEncoderGpu.Stop(ref tpState, tokenP0Out);
            outLens[0] = p0State.OutLen;
            outLens[1] = tpState.OutLen;
            st[StDone] = 1;
        }

        st[StNextMb] = mbIdx;
        Vp8BoolEncoderGpu.Save(st, StP0, ref p0State);
        Vp8BoolEncoderGpu.Save(st, StTp, ref tpState);
        st[StLeft + 0] = leftY4_0; st[StLeft + 1] = leftY4_1; st[StLeft + 2] = leftY4_2; st[StLeft + 3] = leftY4_3;
        st[StLeft + 4] = leftU_0; st[StLeft + 5] = leftU_1;
        st[StLeft + 6] = leftV_0; st[StLeft + 7] = leftV_1;
        st[StLeft + 8] = leftY2;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() => _state.Dispose();
}
