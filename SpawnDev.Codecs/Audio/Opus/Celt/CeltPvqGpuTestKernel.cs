// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltPvqGpuTestKernel"/>: the band size and PVQ parameters of one call.</summary>
public struct CeltPvqCase
{
    /// <summary>Band size N (the vector length).</summary>
    public int N;
    /// <summary>Pulse count K.</summary>
    public int K;
    /// <summary>Spread decision 0..3 (exp_rotation).</summary>
    public int Spread;
    /// <summary>Interleaved block count B (exp_rotation stride / collapse mask blocks).</summary>
    public int B;
    /// <summary>Gain (Q15) for normalise_residual / renormalise_vector, or the stereo flag for stereo_itheta.</summary>
    public int Gain;
    /// <summary>exp_rotation direction (1 before quantization, -1 after), or the codebook index for Cwrsi.</summary>
    public int Dir;
    /// <summary>Byte offset of this case's range-coded packet (AlgUnquant).</summary>
    public int PacketAt;
    /// <summary>Byte length of this case's packet (AlgUnquant).</summary>
    public int PacketLength;
}

/// <summary>
/// Drives the <see cref="CeltPvqGpu"/> functions on the accelerator, one thread per <see cref="CeltPvqCase"/>. Vectors live at
/// <c>case * stride</c> in the vector views; per-case scalar results (squared norm, codebook index, collapse mask, angle) at
/// <c>case</c>.
/// </summary>
public sealed class CeltPvqGpuTestKernel : IDisposable
{
    /// <summary>Which <see cref="CeltPvqGpu"/> function a launch tests.</summary>
    public enum Op
    {
        /// <summary>Cwrsi(N, K, index = Dir) into Y; squared norm into Scalar.</summary>
        Cwrsi = 0,
        /// <summary>Icwrs(N, Y) into Scalar.</summary>
        Icwrs = 1,
        /// <summary>ExpRotation(X, N, Dir, B, K, Spread) in place.</summary>
        ExpRotation = 2,
        /// <summary>NormaliseResidual(Y, X, N, ryy = Scalar, Gain) into X.</summary>
        NormaliseResidual = 3,
        /// <summary>AlgUnquant(X, N, K, Spread, B, packet, Gain) into X (Y = scratch); collapse mask into Scalar.</summary>
        AlgUnquant = 4,
        /// <summary>AlgQuantSearch(X, N, K, Spread, B) in place; pulses into Y; collapse mask into Scalar.</summary>
        AlgQuantSearch = 5,
        /// <summary>RenormaliseVector(X, N, Gain) in place.</summary>
        RenormaliseVector = 6,
        /// <summary>StereoItheta(X, Y, stereo = Gain, N) into Scalar.</summary>
        StereoItheta = 7,
        /// <summary>ExtractCollapseMask(Y, N, B) into Scalar.</summary>
        ExtractCollapseMask = 8,
    }

    private readonly Action<Index1D, int, ArrayView<CeltPvqCase>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<uint>,
        ArrayView<byte>, ArrayView<uint>, int> _kernel;

    /// <summary>Compile.</summary>
    public CeltPvqGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, ArrayView<CeltPvqCase>, ArrayView<int>, ArrayView<int>,
            ArrayView<int>, ArrayView<uint>, ArrayView<byte>, ArrayView<uint>, int>(TestKernel);
    }

    /// <summary>
    /// Run <paramref name="op"/> over every case. <paramref name="x"/> and <paramref name="y"/> hold <paramref name="stride"/>
    /// ints per case, <paramref name="scratch"/> 2 * <paramref name="stride"/>; <paramref name="scalar"/> one uint per case;
    /// <paramref name="packets"/> the range-coded packets; <paramref name="pvqU"/> <see cref="CeltPvqTables.Data"/>.
    /// </summary>
    public void Run(Op op, ArrayView<CeltPvqCase> cases, ArrayView<int> x, ArrayView<int> y, ArrayView<int> scratch,
        ArrayView<uint> scalar, ArrayView<byte> packets, ArrayView<uint> pvqU, int stride)
        => _kernel((int)cases.Length, (int)op, cases, x, y, scratch, scalar, packets, pvqU, stride);

    private static void TestKernel(Index1D c, int op, ArrayView<CeltPvqCase> cases, ArrayView<int> x, ArrayView<int> y,
        ArrayView<int> scratch, ArrayView<uint> scalar, ArrayView<byte> packets, ArrayView<uint> pvqU, int stride)
    {
        CeltPvqCase k = cases[c];
        int at = c * stride;
        if (op == (int)Op.Cwrsi)
            scalar[c] = (uint)CeltPvqGpu.Cwrsi(k.N, k.K, (uint)k.Dir, y, at, pvqU, 0);
        else if (op == (int)Op.Icwrs)
            scalar[c] = CeltPvqGpu.Icwrs(k.N, y, at, pvqU, 0);
        else if (op == (int)Op.ExpRotation)
            CeltPvqGpu.ExpRotation(x, at, k.N, k.Dir, k.B, k.K, k.Spread);
        else if (op == (int)Op.NormaliseResidual)
            CeltPvqGpu.NormaliseResidual(y, at, x, at, k.N, (int)scalar[c], k.Gain);
        else if (op == (int)Op.AlgUnquant)
        {
            var dec = OpusRangeDecoderGpu.Init(packets, k.PacketAt, (uint)k.PacketLength);
            scalar[c] = CeltPvqGpu.AlgUnquant(x, at, k.N, k.K, k.Spread, k.B, ref dec, packets, k.PacketAt, (uint)k.PacketLength,
                k.Gain, y, at, pvqU, 0);
        }
        else if (op == (int)Op.AlgQuantSearch)
            scalar[c] = CeltPvqGpu.AlgQuantSearch(x, at, k.N, k.K, k.Spread, k.B, y, at, scratch, 2 * at, scratch, 2 * at + stride);
        else if (op == (int)Op.RenormaliseVector)
            CeltPvqGpu.RenormaliseVector(x, at, k.N, k.Gain);
        else if (op == (int)Op.StereoItheta)
            scalar[c] = (uint)CeltPvqGpu.StereoItheta(x, at, y, at, k.Gain, k.N);
        else if (op == (int)Op.ExtractCollapseMask)
            scalar[c] = CeltPvqGpu.ExtractCollapseMask(y, at, k.N, k.B);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
