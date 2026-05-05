// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 dequantizer compute kernel. Single-thread per dispatch; reads
// baseQIndex + per-plane delta values + the 256-entry DC/AC quantizer
// lookup tables and writes 4 dequantizer values: Y_DC, Y_AC, UV_DC,
// UV_AC. Foundational primitive for the future Vp9KeyframeEncoderGpu /
// Vp9KeyframeDecoderGpu integration classes.
//
// Mirror of Vp9Dequantizer.PlaneQuantizer for both Y and UV planes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 dequantizer compute kernel. Single thread per dispatch. Looks
/// up Y / UV DC / AC dequantizers from the 256-entry quantizer
/// tables and writes them to a 4-int output buffer.
/// </summary>
public sealed class Vp9DequantizerComputeKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<int>,
        int, int, int, int, int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<int>,
        int, int, int, int, int, int> _batchKernel;

    /// <summary>Compile.</summary>
    public Vp9DequantizerComputeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<int>,
            int, int, int, int, int>(ComputeKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<int>,
            int, int, int, int, int, int>(ComputeBatchKernel);
    }

    /// <summary>Batch dequantizer: extent=N, each thread fills one frame's slot.</summary>
    public void RunBatch(
        ArrayView<short> dcQLookup, ArrayView<short> acQLookup,
        ArrayView<int> dequantOut,
        int baseQIndex, int yDcDelta, int yAcDelta, int uvDcDelta, int uvAcDelta,
        int frameCount, int dequantStride)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        _batchKernel(frameCount, dcQLookup, acQLookup, dequantOut,
            baseQIndex, yDcDelta, yAcDelta, uvDcDelta, uvAcDelta, dequantStride);
    }

    private static void ComputeBatchKernel(
        Index1D idx,
        ArrayView<short> dcQLookup, ArrayView<short> acQLookup,
        ArrayView<int> dequantOut,
        int baseQIndex, int yDcDelta, int yAcDelta, int uvDcDelta, int uvAcDelta,
        int dequantStride)
    {
        int f = idx.X;
        var fDQ = dequantOut.SubView((long)f * dequantStride, dequantStride);
        fDQ[0] = LookupClamped(dcQLookup, baseQIndex + yDcDelta);
        fDQ[1] = LookupClamped(acQLookup, baseQIndex + yAcDelta);
        fDQ[2] = LookupClamped(dcQLookup, baseQIndex + uvDcDelta);
        fDQ[3] = LookupClamped(acQLookup, baseQIndex + uvAcDelta);
    }

    /// <summary>
    /// Compute the 4 plane dequantizers. dequantOut layout: [Y_DC, Y_AC, UV_DC, UV_AC].
    /// </summary>
    public void Run(
        ArrayView<short> dcQLookup,
        ArrayView<short> acQLookup,
        ArrayView<int> dequantOut,
        int baseQIndex,
        int yDcDelta, int yAcDelta,
        int uvDcDelta, int uvAcDelta)
    {
        if (dcQLookup.Length < 256) throw new ArgumentException("dcQLookup must hold 256 shorts.", nameof(dcQLookup));
        if (acQLookup.Length < 256) throw new ArgumentException("acQLookup must hold 256 shorts.", nameof(acQLookup));
        if (dequantOut.Length < 4) throw new ArgumentException("dequantOut must hold 4 ints.", nameof(dequantOut));
        _kernel(1, dcQLookup, acQLookup, dequantOut,
            baseQIndex, yDcDelta, yAcDelta, uvDcDelta, uvAcDelta);
    }

    /// <summary>Build the 256-short DC quantizer lookup buffer for upload.</summary>
    public static short[] BuildDcQLookup() => (short[])Vp9Dequantizer.DcQLookup8.Clone();

    /// <summary>Build the 256-short AC quantizer lookup buffer for upload.</summary>
    public static short[] BuildAcQLookup() => (short[])Vp9Dequantizer.AcQLookup8.Clone();

    private static void ComputeKernel(
        Index1D _,
        ArrayView<short> dcQLookup,
        ArrayView<short> acQLookup,
        ArrayView<int> dequantOut,
        int baseQIndex,
        int yDcDelta, int yAcDelta,
        int uvDcDelta, int uvAcDelta)
    {
        dequantOut[0] = LookupClamped(dcQLookup, baseQIndex + yDcDelta);
        dequantOut[1] = LookupClamped(acQLookup, baseQIndex + yAcDelta);
        dequantOut[2] = LookupClamped(dcQLookup, baseQIndex + uvDcDelta);
        dequantOut[3] = LookupClamped(acQLookup, baseQIndex + uvAcDelta);
    }

    private static int LookupClamped(ArrayView<short> table, int idx)
    {
        if (idx < 0) idx = 0;
        else if (idx > 255) idx = 255;
        return table[idx];
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
