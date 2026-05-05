// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame setup kernel. Single-thread-per-frame GPU kernel that
// computes the per-MB dequantizer values from the base Q index +
// delta values + writes the VP8 frame header bool stream to
// partition0Out using Vp8BoolEncoderGpu.
//
// Output:
//   - dequantizers[6]: [Y1Dc, Y1Ac, Y2Dc, Y2Ac, UvDc, UvAc]
//   - partition0Out: filled with the frame header bytes
//   - initialP0State[5]: [LowValue, Range, Count, OutLen, _]
//     (The existing Vp8FrameEntropyKernel picks this up and
//     continues writing the per-MB modes from this state.)
//
// v1 simplifications (matches Vp8KeyframeEncoder defaults):
//   - ColorSpace = 0, ClampingType = 0
//   - Segmentation disabled
//   - Loop filter disabled (filterLevel = 0)
//   - Log2NumPartitions = 0
//   - All deltaQ = 0
//   - RefreshEntropyProbs = true
//   - All coef probs match defaults (no updates emitted)
//   - MbNoSkipCoeffEnabled = false

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 frame setup kernel. Computes dequantizers + writes frame header
/// to partition0Out. Single thread per frame.
/// </summary>
public sealed class Vp8FrameSetupKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<int>, ArrayView<int>,
        ArrayView<byte>, ArrayView<byte>,
        ArrayView<int>, ArrayView<byte>, ArrayView<int>,
        int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<int>, ArrayView<int>,
        ArrayView<byte>, ArrayView<byte>,
        ArrayView<int>, ArrayView<byte>, ArrayView<int>,
        int, int, int> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp8FrameSetupKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>, ArrayView<int>,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<int>, ArrayView<byte>, ArrayView<int>,
            int>(SetupFrameKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<int>, ArrayView<int>,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<int>, ArrayView<byte>, ArrayView<int>,
            int, int, int>(SetupFrameBatchKernel);
    }

    /// <summary>Batch setup: extent=N, each thread sets up one frame's slot.</summary>
    public void RunBatch(
        ArrayView<int> dcQLookup, ArrayView<int> acQLookup,
        ArrayView<byte> defaultCoefProbs, ArrayView<byte> updateCoefProbs,
        ArrayView<int> dequantOut, ArrayView<byte> partition0Out, ArrayView<int> initialP0StateOut,
        int baseQIndex, int frameCount, int dequantStride, int p0Stride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        _batchKernel(frameCount, dcQLookup, acQLookup, defaultCoefProbs, updateCoefProbs,
            dequantOut, partition0Out, initialP0StateOut, baseQIndex, dequantStride, p0Stride);
    }

    private static void SetupFrameBatchKernel(
        Index1D idx,
        ArrayView<int> dcQLookup, ArrayView<int> acQLookup,
        ArrayView<byte> defaultCoefProbs, ArrayView<byte> updateCoefProbs,
        ArrayView<int> dequantOut, ArrayView<byte> partition0Out, ArrayView<int> initialP0StateOut,
        int baseQIndex, int dequantStride, int p0Stride)
    {
        int f = idx.X;
        var fDQ = dequantOut.SubView((long)f * dequantStride, dequantStride);
        var fP0 = partition0Out.SubView((long)f * p0Stride, p0Stride);
        var fInit = initialP0StateOut.SubView((long)f * 5, 5);
        SetupFrameBody(dcQLookup, acQLookup, defaultCoefProbs, updateCoefProbs,
            fDQ, fP0, fInit, baseQIndex);
    }

    /// <summary>
    /// Run the frame setup. dcQLookup + acQLookup are the 128-entry VP8
    /// quantizer tables (caller materializes via BuildDcQLookup /
    /// BuildAcQLookup). defaultCoefProbs + updateCoefProbs are the
    /// 1056-byte 4D tables flat-packed.
    /// </summary>
    public void Run(
        ArrayView<int> dcQLookup,
        ArrayView<int> acQLookup,
        ArrayView<byte> defaultCoefProbs,
        ArrayView<byte> updateCoefProbs,
        ArrayView<int> dequantOut,
        ArrayView<byte> partition0Out,
        ArrayView<int> initialP0StateOut,
        int baseQIndex)
    {
        if (dcQLookup.Length < 128)
            throw new ArgumentException("dcQLookup must hold 128 ints.", nameof(dcQLookup));
        if (acQLookup.Length < 128)
            throw new ArgumentException("acQLookup must hold 128 ints.", nameof(acQLookup));
        if (defaultCoefProbs.Length < 1056)
            throw new ArgumentException("defaultCoefProbs must hold 1056 bytes.", nameof(defaultCoefProbs));
        if (updateCoefProbs.Length < 1056)
            throw new ArgumentException("updateCoefProbs must hold 1056 bytes.", nameof(updateCoefProbs));
        if (dequantOut.Length < 6)
            throw new ArgumentException("dequantOut must hold 6 ints.", nameof(dequantOut));
        if (initialP0StateOut.Length < 5)
            throw new ArgumentException("initialP0StateOut must hold 5 ints.", nameof(initialP0StateOut));
        _kernel(1, dcQLookup, acQLookup, defaultCoefProbs, updateCoefProbs,
            dequantOut, partition0Out, initialP0StateOut, baseQIndex);
    }

    /// <summary>Build the 128-int DC quantizer lookup buffer to upload once per accelerator.</summary>
    public static int[] BuildDcQLookup() => (int[])Vp8Quantizer.DcQLookup.Clone();

    /// <summary>Build the 128-int AC quantizer lookup buffer.</summary>
    public static int[] BuildAcQLookup() => (int[])Vp8Quantizer.AcQLookup.Clone();

    /// <summary>Build the 4D default coef probs flattened to 1056 bytes (block_type * 264 + band * 33 + ctx * 11 + node).</summary>
    public static byte[] BuildDefaultCoefProbs()
    {
        var buf = new byte[4 * 264];
        var src = Vp8DefaultCoefProbs.DefaultProbs;
        for (int t = 0; t < 4; t++)
            for (int band = 0; band < 8; band++)
                for (int c = 0; c < 3; c++)
                    for (int n = 0; n < 11; n++)
                        buf[t * 264 + band * 33 + c * 11 + n] = src[t, band, c, n];
        return buf;
    }

    /// <summary>Build the 4D update coef probs flattened to 1056 bytes.</summary>
    public static byte[] BuildUpdateCoefProbs()
    {
        var buf = new byte[4 * 264];
        var src = Vp8CoefUpdateProbs.UpdateProbs;
        for (int t = 0; t < 4; t++)
            for (int band = 0; band < 8; band++)
                for (int c = 0; c < 3; c++)
                    for (int n = 0; n < 11; n++)
                        buf[t * 264 + band * 33 + c * 11 + n] = src[t, band, c, n];
        return buf;
    }

    private static void SetupFrameKernel(
        Index1D _,
        ArrayView<int> dcQLookup,
        ArrayView<int> acQLookup,
        ArrayView<byte> defaultCoefProbs,
        ArrayView<byte> updateCoefProbs,
        ArrayView<int> dequantOut,
        ArrayView<byte> partition0Out,
        ArrayView<int> initialP0StateOut,
        int baseQIndex)
    {
        SetupFrameBody(dcQLookup, acQLookup, defaultCoefProbs, updateCoefProbs,
            dequantOut, partition0Out, initialP0StateOut, baseQIndex);
    }

    private static void SetupFrameBody(
        ArrayView<int> dcQLookup,
        ArrayView<int> acQLookup,
        ArrayView<byte> defaultCoefProbs,
        ArrayView<byte> updateCoefProbs,
        ArrayView<int> dequantOut,
        ArrayView<byte> partition0Out,
        ArrayView<int> initialP0StateOut,
        int baseQIndex)
    {
        // 1. Compute dequantizers. v1: no deltas, no segmentation -
        // direct lookups + Y2/UV adjustments.
        int qClamped = baseQIndex < 0 ? 0 : baseQIndex > 127 ? 127 : baseQIndex;
        int y1Dc = dcQLookup[qClamped];
        int y1Ac = acQLookup[qClamped];
        int y2Dc = y1Dc * 2;
        int y2AcRaw = (acQLookup[qClamped] * 101581) >> 16;
        int y2Ac = y2AcRaw < 8 ? 8 : y2AcRaw;
        int uvDc = y1Dc > 132 ? 132 : y1Dc;
        int uvAc = y1Ac;

        dequantOut[0] = y1Dc;
        dequantOut[1] = y1Ac;
        dequantOut[2] = y2Dc;
        dequantOut[3] = y2Ac;
        dequantOut[4] = uvDc;
        dequantOut[5] = uvAc;

        // 2. Init bool encoder + write frame header.
        var state = Vp8BoolEncoderGpu.Init();

        // colorspace = 0 (1 bit)
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1);
        // clamping_type = 0 (1 bit)
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1);

        // segmentation: enabled = false (1 bit; rest skipped)
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1);

        // loop filter: type=0 (1), level=0 (6), sharpness=0 (3), modeRefLfDeltaEnabled=false (1).
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1);
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 6);
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 3);
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1);

        // log2NumPartitions = 0 (2 bits)
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 2);

        // quantizer: baseQIndex (7), then 5 deltaQ markers (1 bit each "no delta").
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, qClamped, 7);
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1); // y1DcDeltaQ
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1); // y2DcDeltaQ
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1); // y2AcDeltaQ
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1); // uvDcDeltaQ
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1); // uvAcDeltaQ

        // refreshEntropyProbs = true (1 bit)
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 1, 1);

        // coef probs updates: 4 * 8 * 3 * 11 = 1056 emits in libvpx
        // iteration order. For v1 (default probs unchanged), every
        // emit is bool 0 with the update prob from updateCoefProbs.
        // Single-counter loop with runtime bound - keeps the CUDA JIT
        // from trying to unroll 1056 inline iterations.
        int totalEntries = (int)updateCoefProbs.Length;
        for (int i = 0; i < totalEntries; i++)
        {
            byte updateProb = updateCoefProbs[i];
            byte currentProb = defaultCoefProbs[i];
            byte defaultProb = defaultCoefProbs[i];
            if (currentProb != defaultProb)
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, partition0Out, 1, updateProb);
                Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, currentProb, 8);
            }
            else
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, partition0Out, 0, updateProb);
            }
        }

        // mb_no_skip_coeff_enabled = false (1 bit; no further emit if false)
        Vp8BoolEncoderGpu.EncodeValue(ref state, partition0Out, 0, 1);

        // 3. Save state for the entropy kernel to continue from.
        initialP0StateOut[0] = (int)state.LowValue;
        initialP0StateOut[1] = (int)state.Range;
        initialP0StateOut[2] = state.Count;
        initialP0StateOut[3] = (int)state.OutLen;
        initialP0StateOut[4] = 0;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
