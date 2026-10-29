// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltBandsGpuTestKernel"/>: up to eight integer arguments (their meaning depends on the op).</summary>
public struct CeltBandsCase
{
    /// <summary>Argument 0.</summary>
    public int A;
    /// <summary>Argument 1.</summary>
    public int B;
    /// <summary>Argument 2.</summary>
    public int C;
    /// <summary>Argument 3.</summary>
    public int D;
    /// <summary>Argument 4.</summary>
    public int E;
    /// <summary>Argument 5.</summary>
    public int F;
    /// <summary>Argument 6.</summary>
    public int G;
    /// <summary>Argument 7.</summary>
    public int H;
}

/// <summary>
/// Drives <see cref="CeltBandsGpu"/> on the accelerator, one thread per <see cref="CeltBandsCase"/>. Vectors at
/// <c>case * stride</c> in X and Y; the per-case integer block (energies, masks) at <c>case * stride</c> in Aux; one scalar
/// result per case.
/// </summary>
public sealed class CeltBandsGpuTestKernel : IDisposable
{
    /// <summary>Which <see cref="CeltBandsGpu"/> function a launch tests; the case fields each one reads are listed.</summary>
    public enum Op
    {
        /// <summary>Scalar = BitexactCos(A).</summary>
        BitexactCos = 0,
        /// <summary>Scalar = BitexactLog2tan(A, B).</summary>
        BitexactLog2tan = 1,
        /// <summary>Scalar = ComputeQn(N = A, b = B, offset = C, pulse_cap = D, stereo = E).</summary>
        ComputeQn = 2,
        /// <summary>IntensityStereo(X, Y, bandE = Aux[0..1], N = A).</summary>
        IntensityStereo = 3,
        /// <summary>StereoSplit(X, Y, N = A).</summary>
        StereoSplit = 4,
        /// <summary>StereoMerge(X, Y, mid = B, N = A).</summary>
        StereoMerge = 5,
        /// <summary>Haar1(X, N0 = A, stride = B).</summary>
        Haar1 = 6,
        /// <summary>DeinterleaveHadamard(X, N0 = A, stride = B, hadamard = C).</summary>
        Deinterleave = 7,
        /// <summary>InterleaveHadamard(X, N0 = A, stride = B, hadamard = C).</summary>
        Interleave = 8,
        /// <summary>
        /// AntiCollapse(X (channel stride = F), LM = A, C = B, start = C, end = D, seed = (uint)E); Aux holds masks (2 * NbEBands),
        /// logE, prev1logE, prev2logE (2 * NbEBands each) and pulses (NbEBands), in that order.
        /// </summary>
        AntiCollapse = 9,
        /// <summary>
        /// CeltCombFilterGpu.CombFilter(T0 = A, T1 = B, N = C, g0 = D, g1 = E, tapset0 = F, tapset1 = G, overlap 120): X holds
        /// <see cref="CombHistory"/> samples of history, then the frame; H = 1 filters X in place, 0 writes Y.
        /// </summary>
        CombFilter = 10,
    }

    /// <summary>History samples before the frame in the CombFilter op (libopus COMBFILTER_MAXPERIOD + 2).</summary>
    public const int CombHistory = 1026;

    private readonly Action<Index1D, int, ArrayView<CeltBandsCase>, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<int>,
        ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile.</summary>
    public CeltBandsGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, ArrayView<CeltBandsCase>, CeltModeGpu, ArrayView<int>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(TestKernel);
    }

    /// <summary>Run <paramref name="op"/> over every case (X, Y, Aux and Scratch hold <paramref name="stride"/> ints per case).</summary>
    public void Run(Op op, ArrayView<CeltBandsCase> cases, CeltModeGpu mode, ArrayView<int> x, ArrayView<int> y, ArrayView<int> aux,
        ArrayView<int> scratch, ArrayView<int> scalar, int stride)
        => _kernel((int)cases.Length, (int)op, cases, mode, x, y, aux, scratch, scalar, stride);

    private static void TestKernel(Index1D c, int op, ArrayView<CeltBandsCase> cases, CeltModeGpu mode, ArrayView<int> x,
        ArrayView<int> y, ArrayView<int> aux, ArrayView<int> scratch, ArrayView<int> scalar, int stride)
    {
        CeltBandsCase k = cases[c];
        int at = c * stride;
        const int nb = CeltModeTables.NbEBands;
        if (op == (int)Op.BitexactCos) scalar[c] = CeltBandsGpu.BitexactCos(k.A);
        else if (op == (int)Op.BitexactLog2tan) scalar[c] = CeltBandsGpu.BitexactLog2tan(k.A, k.B);
        else if (op == (int)Op.ComputeQn) scalar[c] = CeltBandsGpu.ComputeQn(k.A, k.B, k.C, k.D, k.E);
        else if (op == (int)Op.IntensityStereo) CeltBandsGpu.IntensityStereo(x, at, y, at, aux, at, 1, 0, k.A);
        else if (op == (int)Op.StereoSplit) CeltBandsGpu.StereoSplit(x, at, y, at, k.A);
        else if (op == (int)Op.StereoMerge) CeltBandsGpu.StereoMerge(x, at, y, at, k.B, k.A);
        else if (op == (int)Op.Haar1) CeltBandsGpu.Haar1(x, at, k.A, k.B);
        else if (op == (int)Op.Deinterleave) CeltBandsGpu.DeinterleaveHadamard(mode, x, at, k.A, k.B, k.C, scratch, at);
        else if (op == (int)Op.Interleave) CeltBandsGpu.InterleaveHadamard(mode, x, at, k.A, k.B, k.C, scratch, at);
        else if (op == (int)Op.AntiCollapse)
            CeltBandsGpu.AntiCollapse(mode, x, at, k.F, aux, at, k.A, k.B, k.C, k.D,
                aux, at + 2 * nb, aux, at + 4 * nb, aux, at + 6 * nb, aux, at + 8 * nb, (uint)k.E);
        else if (op == (int)Op.CombFilter)
        {
            if (k.H != 0)
                CeltCombFilterGpu.CombFilter(mode, x, at + CombHistory, x, at + CombHistory, k.A, k.B, k.C, k.D, k.E, k.F, k.G, 120);
            else
                CeltCombFilterGpu.CombFilter(mode, y, at + CombHistory, x, at + CombHistory, k.A, k.B, k.C, k.D, k.E, k.F, k.G, 120);
        }
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
