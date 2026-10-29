// Cross-backend tests for the CELT GPU primitives (Audio/Opus/Celt) against Concentus 2.2.2, the BSD-3 C# port of libopus that
// SpawnDev.Codecs.References uses as its CELT oracle. Concentus keeps its CELT functions internal: array-argument methods are
// called through reflection, Span-argument ones through delegates bound to the internal methods, and the ones that also take
// Concentus' internal EntropyCoder through small DynamicMethods (castclass + call). Every comparison is bit-exact.

using System.Reflection;
using System.Reflection.Emit;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Concentus' CELT internals as callable oracles.</summary>
    private static class ConcentusCeltOracle
    {
        public delegate void ExpRotationFn(Span<int> x, int xPtr, int len, int dir, int stride, int k, int spread);
        public delegate void NormaliseResidualFn(int[] iy, Span<int> x, int xPtr, int n, int ryy, int gain);
        public delegate void RenormaliseVectorFn(Span<int> x, int n, int gain);
        public delegate int StereoIthetaFn(Span<int> x, Span<int> y, int stereo, int n);
        public delegate uint AlgQuantFn(Span<int> x, int xPtr, int n, int k, int spread, int b, object enc, Span<byte> buf);
        public delegate void EncDoneFn(object enc, Span<byte> buf);

        private static readonly Assembly A = typeof(Concentus.OpusCodecFactory).Assembly;
        private const BindingFlags Any = BindingFlags.Static | BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        private static readonly Type Ec = A.GetType("Concentus.Common.EntropyCoder", true)!;

        private static MethodInfo M(string type, string name, params Type[] args) =>
            A.GetType("Concentus.Celt." + type, true)!.GetMethod(name, Any, null, args, null)
            ?? throw new MissingMethodException("Concentus.Celt." + type, name);

        private static Delegate Emit(Type delegateType, MethodInfo target, int entropyCoderArg)
        {
            var parameters = delegateType.GetMethod("Invoke")!.GetParameters().Select(p => p.ParameterType).ToArray();
            var dm = new DynamicMethod("oracle_" + target.Name, delegateType.GetMethod("Invoke")!.ReturnType, parameters,
                typeof(ConcentusCeltOracle).Module, true);
            var il = dm.GetILGenerator();
            for (int i = 0; i < parameters.Length; i++)
            {
                il.Emit(OpCodes.Ldarg, (short)i);
                if (i == entropyCoderArg) il.Emit(OpCodes.Castclass, Ec);
            }
            il.Emit(target.IsStatic ? OpCodes.Call : OpCodes.Callvirt, target);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate(delegateType);
        }

        private static readonly MethodInfo CwrsiM = M("CWRS", "cwrsi", typeof(int), typeof(int), typeof(uint), typeof(int[]));
        private static readonly MethodInfo IcwrsM = M("CWRS", "icwrs", typeof(int), typeof(int[]));
        private static readonly MethodInfo PvqVM = M("CWRS", "CELT_PVQ_V", typeof(int), typeof(int));
        private static readonly MethodInfo CollapseM = M("VQ", "extract_collapse_mask", typeof(int[]), typeof(int), typeof(int));
        public static readonly ExpRotationFn ExpRotation = (ExpRotationFn)Delegate.CreateDelegate(typeof(ExpRotationFn),
            M("VQ", "exp_rotation", typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int)));
        public static readonly NormaliseResidualFn NormaliseResidual = (NormaliseResidualFn)Delegate.CreateDelegate(typeof(NormaliseResidualFn),
            M("VQ", "normalise_residual", typeof(int[]), typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int)));
        public static readonly RenormaliseVectorFn RenormaliseVector = (RenormaliseVectorFn)Delegate.CreateDelegate(typeof(RenormaliseVectorFn),
            M("VQ", "renormalise_vector", typeof(Span<int>), typeof(int), typeof(int)));
        public static readonly StereoIthetaFn StereoItheta = (StereoIthetaFn)Delegate.CreateDelegate(typeof(StereoIthetaFn),
            M("VQ", "stereo_itheta", typeof(Span<int>), typeof(Span<int>), typeof(int), typeof(int)));
        private static readonly AlgQuantFn AlgQuant = (AlgQuantFn)Emit(typeof(AlgQuantFn),
            M("VQ", "alg_quant", typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), Ec, typeof(Span<byte>)), 6);
        private static readonly EncDoneFn EncDone = (EncDoneFn)Emit(typeof(EncDoneFn),
            Ec.GetMethod("enc_done", Any, null, new[] { typeof(Span<byte>) }, null)!, 0);

        public static int Cwrsi(int n, int k, uint i, int[] y) => (int)CwrsiM.Invoke(null, new object[] { n, k, i, y })!;
        public static uint Icwrs(int n, int[] y) => (uint)IcwrsM.Invoke(null, new object[] { n, y })!;
        public static uint PvqV(int n, int k) => (uint)PvqVM.Invoke(null, new object[] { n, k })!;
        public static uint ExtractCollapseMask(int[] iy, int n, int b) => (uint)CollapseM.Invoke(null, new object[] { iy, n, b })!;

        /// <summary>Concentus alg_quant of <paramref name="x"/> (in place), range-coded: the collapse mask and the coded bytes.</summary>
        public static (uint Mask, byte[] Bytes) AlgQuantCoded(int[] x, int n, int k, int spread, int b)
        {
            object enc = Activator.CreateInstance(Ec, true)!;
            Ec.GetMethod("enc_init", Any)!.Invoke(enc, new object[] { 256u });
            var buf = new byte[256];
            uint mask = AlgQuant(x, 0, n, k, spread, b, enc, buf);
            EncDone(enc, buf);
            return (mask, buf);
        }
    }

    private const int CeltPvqStride = 256;

    /// <summary>
    /// (N, K) pairs CELT codes: the U(N, K) table holds exactly those, so both lookups of V(N, K) = U(N, K) + U(N, K + 1) must
    /// stay inside their table row (outside it a lookup runs into the next row and V is garbage).
    /// </summary>
    private static (int N, int K)[] CeltPvqPairs(Random rng, int count, int maxN, int maxK)
    {
        static bool InRow(int a, int b)
        {
            int lo = Math.Min(a, b), hi = Math.Max(a, b);
            var rows = CeltPvqTables.RowStart;
            if (lo >= rows.Length) return false;
            uint end = lo + 1 < rows.Length ? rows[lo + 1] : (uint)CeltPvqTables.Data.Length;
            return rows[lo] + (uint)hi < end;
        }
        var list = new List<(int, int)>();
        while (list.Count < count)
        {
            int n = 2 + rng.Next(maxN - 1), k = 1 + rng.Next(maxK);
            if (InRow(n, k) && InRow(n, k + 1) && ConcentusCeltOracle.PvqV(n, k) > 1) list.Add((n, k));
        }
        return list.ToArray();
    }

    private static int[] CeltRandomPulses(Random rng, int n, int k)
    {
        var y = new int[n];
        for (int p = 0; p < k; p++) y[rng.Next(n)]++;
        for (int i = 0; i < n; i++) if (rng.Next(2) == 0) y[i] = -y[i];
        return y;
    }

    /// <summary>A band-shape-like Q14 vector, well inside int16.</summary>
    private static int[] CeltRandomShape(Random rng, int n)
    {
        var x = new int[n];
        for (int i = 0; i < n; i++) x[i] = rng.Next(-12000, 12001);
        return x;
    }

    /// <summary>Run one <see cref="CeltPvqGpuTestKernel"/> op over <paramref name="cases"/>; returns X, Y and the per-case scalars.</summary>
    private static async Task<(int[] X, int[] Y, uint[] Scalar)> CeltPvqLaunchAsync(Accelerator acc, CeltPvqGpuTestKernel.Op op,
        CeltPvqCase[] cases, int[] x, int[] y, uint[] scalar, byte[]? packets = null)
    {
        using var dU = acc.Allocate1D(CeltPvqTables.Data);
        using var dCases = acc.Allocate1D(cases);
        using var dX = acc.Allocate1D(x);
        using var dY = acc.Allocate1D(y);
        using var dScratch = acc.Allocate1D<int>(cases.Length * CeltPvqStride * 2);
        using var dS = acc.Allocate1D(scalar);
        using var dP = acc.Allocate1D(packets is { Length: > 0 } ? packets : new byte[4]);
        using var kernel = new CeltPvqGpuTestKernel(acc);
        kernel.Run(op, dCases.View, dX.View, dY.View, dScratch.View, dS.View, dP.View, dU.View, CeltPvqStride);
        await acc.SynchronizeAsync();
        return (await dX.CopyToHostAsync(), await dY.CopyToHostAsync(), await dS.CopyToHostAsync());
    }

    [TestMethod]
    public async Task CeltPvqGpu_Cwrsi_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC317);
            var pairs = CeltPvqPairs(rng, 120, 40, 40);
            var cases = new CeltPvqCase[pairs.Length];
            var want = new int[pairs.Length][]; var wantNorm = new int[pairs.Length];
            for (int c = 0; c < pairs.Length; c++)
            {
                var (n, k) = pairs[c];
                uint v = ConcentusCeltOracle.PvqV(n, k);
                uint i = c % 3 == 0 ? 0u : c % 3 == 1 ? v - 1 : (uint)rng.NextInt64(v);   // both ends of the codebook + random
                cases[c] = new CeltPvqCase { N = n, K = k, Dir = (int)i };
                want[c] = new int[n]; wantNorm[c] = ConcentusCeltOracle.Cwrsi(n, k, i, want[c]);
            }
            var (_, y, s) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.Cwrsi, cases,
                new int[pairs.Length * CeltPvqStride], new int[pairs.Length * CeltPvqStride], new uint[pairs.Length]);
            for (int c = 0; c < pairs.Length; c++)
            {
                if ((int)s[c] != wantNorm[c]) throw new Exception($"cwrsi N={pairs[c].N} K={pairs[c].K}: squared norm {(int)s[c]}, Concentus {wantNorm[c]}");
                for (int j = 0; j < pairs[c].N; j++)
                    if (y[c * CeltPvqStride + j] != want[c][j])
                        throw new Exception($"cwrsi N={pairs[c].N} K={pairs[c].K} index {(uint)cases[c].Dir}: y[{j}] {y[c * CeltPvqStride + j]}, Concentus {want[c][j]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltPvqGpu_Icwrs_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC318);
            var pairs = CeltPvqPairs(rng, 120, 40, 40);
            var cases = new CeltPvqCase[pairs.Length];
            var yIn = new int[pairs.Length * CeltPvqStride]; var want = new uint[pairs.Length];
            for (int c = 0; c < pairs.Length; c++)
            {
                var (n, k) = pairs[c];
                var p = CeltRandomPulses(rng, n, k);
                cases[c] = new CeltPvqCase { N = n, K = k };
                Array.Copy(p, 0, yIn, c * CeltPvqStride, n);
                want[c] = ConcentusCeltOracle.Icwrs(n, p);
            }
            var (_, _, s) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.Icwrs, cases, new int[yIn.Length], yIn, new uint[pairs.Length]);
            for (int c = 0; c < pairs.Length; c++)
                if (s[c] != want[c]) throw new Exception($"icwrs N={pairs[c].N} K={pairs[c].K}: {s[c]}, Concentus {want[c]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltPvqGpu_ExpRotation_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC319);
            const int count = 100;
            var cases = new CeltPvqCase[count]; var xIn = new int[count * CeltPvqStride]; var want = new int[count][];
            for (int c = 0; c < count; c++)
            {
                int b = new[] { 1, 2, 4, 8 }[rng.Next(4)];
                int n = b * (1 + rng.Next(CeltPvqStride / b - 1));
                int k = 1 + rng.Next(Math.Max(1, n / 2));
                int spread = rng.Next(4), dir = rng.Next(2) == 0 ? 1 : -1;
                var x = CeltRandomShape(rng, n);
                Array.Copy(x, 0, xIn, c * CeltPvqStride, n);
                cases[c] = new CeltPvqCase { N = n, K = k, Spread = spread, B = b, Dir = dir };
                ConcentusCeltOracle.ExpRotation(x, 0, n, dir, b, k, spread);
                want[c] = x;
            }
            var (x2, _, _) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.ExpRotation, cases, xIn, new int[xIn.Length], new uint[count]);
            for (int c = 0; c < count; c++)
                for (int j = 0; j < cases[c].N; j++)
                    if (x2[c * CeltPvqStride + j] != want[c][j])
                        throw new Exception($"exp_rotation N={cases[c].N} B={cases[c].B} K={cases[c].K} spread={cases[c].Spread} dir={cases[c].Dir}: x[{j}] {x2[c * CeltPvqStride + j]}, Concentus {want[c][j]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltPvqGpu_AlgUnquant_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC31A);
            var pairs = CeltPvqPairs(rng, 100, 40, 30);
            var cases = new CeltPvqCase[pairs.Length];
            var want = new int[pairs.Length][]; var wantMask = new uint[pairs.Length];
            var packets = new List<byte>();
            for (int c = 0; c < pairs.Length; c++)
            {
                var (n, k) = pairs[c];
                var p = CeltRandomPulses(rng, n, k);
                int b = new[] { 1, 2, 4 }[rng.Next(3)]; if (n % b != 0) b = 1;
                int spread = rng.Next(4), gain = new[] { 32767, 23170, 16384, 9000 }[rng.Next(4)];
                // the band's pulse vector as CELT codes it: ec_enc_uint(icwrs(y), V(N, K))
                var enc = new OpusRangeEncoder(256);
                enc.EncodeUint(ConcentusCeltOracle.Icwrs(n, p), ConcentusCeltOracle.PvqV(n, k));
                enc.Done();
                var bytes = enc.ToArray();
                cases[c] = new CeltPvqCase { N = n, K = k, Spread = spread, B = b, Gain = gain, PacketAt = packets.Count, PacketLength = bytes.Length };
                packets.AddRange(bytes);
                while (packets.Count % 4 != 0) packets.Add(0);
                // Concentus alg_unquant once the pulses are decoded: normalise_residual, exp_rotation(-1), extract_collapse_mask
                int ryy = 0;
                for (int i = 0; i < n; i++) ryy += p[i] * p[i];
                want[c] = new int[n];
                ConcentusCeltOracle.NormaliseResidual(p, want[c], 0, n, ryy, gain);
                ConcentusCeltOracle.ExpRotation(want[c], 0, n, -1, b, k, spread);
                wantMask[c] = ConcentusCeltOracle.ExtractCollapseMask(p, n, b);
            }
            var (x2, _, s) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.AlgUnquant, cases,
                new int[pairs.Length * CeltPvqStride], new int[pairs.Length * CeltPvqStride], new uint[pairs.Length], packets.ToArray());
            for (int c = 0; c < pairs.Length; c++)
            {
                if (s[c] != wantMask[c]) throw new Exception($"alg_unquant N={cases[c].N} K={cases[c].K} B={cases[c].B}: collapse mask {s[c]}, Concentus {wantMask[c]}");
                for (int j = 0; j < cases[c].N; j++)
                    if (x2[c * CeltPvqStride + j] != want[c][j])
                        throw new Exception($"alg_unquant N={cases[c].N} K={cases[c].K} spread={cases[c].Spread}: x[{j}] {x2[c * CeltPvqStride + j]}, Concentus {want[c][j]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltPvqGpu_AlgQuantSearch_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC31B);
            var pairs = CeltPvqPairs(rng, 100, 40, 30);
            var cases = new CeltPvqCase[pairs.Length]; var xIn = new int[pairs.Length * CeltPvqStride];
            var wantX = new int[pairs.Length][]; var wantY = new int[pairs.Length][]; var wantMask = new uint[pairs.Length];
            for (int c = 0; c < pairs.Length; c++)
            {
                var (n, k) = pairs[c];
                int b = new[] { 1, 2, 4 }[rng.Next(3)]; if (n % b != 0) b = 1;
                int spread = rng.Next(4);
                var x = CeltRandomShape(rng, n);
                ConcentusCeltOracle.RenormaliseVector(x, n, 32767);   // a unit-norm band, as alg_quant receives
                Array.Copy(x, 0, xIn, c * CeltPvqStride, n);
                cases[c] = new CeltPvqCase { N = n, K = k, Spread = spread, B = b };
                var (mask, bytes) = ConcentusCeltOracle.AlgQuantCoded(x, n, k, spread, b);
                // the pulses Concentus chose, read back from its coded index
                uint index = new OpusRangeDecoder(bytes).DecodeUint(ConcentusCeltOracle.PvqV(n, k));
                wantY[c] = new int[n];
                ConcentusCeltOracle.Cwrsi(n, k, index, wantY[c]);
                wantX[c] = x; wantMask[c] = mask;
            }
            var (x2, y2, s) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.AlgQuantSearch, cases, xIn, new int[xIn.Length], new uint[pairs.Length]);
            for (int c = 0; c < pairs.Length; c++)
            {
                if (s[c] != wantMask[c]) throw new Exception($"alg_quant N={cases[c].N} K={cases[c].K} B={cases[c].B}: collapse mask {s[c]}, Concentus {wantMask[c]}");
                for (int j = 0; j < cases[c].N; j++)
                {
                    if (y2[c * CeltPvqStride + j] != wantY[c][j])
                        throw new Exception($"alg_quant N={cases[c].N} K={cases[c].K}: pulse[{j}] {y2[c * CeltPvqStride + j]}, Concentus {wantY[c][j]}");
                    if (x2[c * CeltPvqStride + j] != wantX[c][j])
                        throw new Exception($"alg_quant N={cases[c].N} K={cases[c].K}: x[{j}] {x2[c * CeltPvqStride + j]}, Concentus {wantX[c][j]}");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltPvqGpu_RenormaliseVector_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC31C);
            const int count = 100;
            var cases = new CeltPvqCase[count]; var xIn = new int[count * CeltPvqStride]; var want = new int[count][];
            for (int c = 0; c < count; c++)
            {
                int n = 1 + rng.Next(CeltPvqStride - 1), gain = new[] { 32767, 23170, 16384 }[rng.Next(3)];
                var x = CeltRandomShape(rng, n);
                Array.Copy(x, 0, xIn, c * CeltPvqStride, n);
                cases[c] = new CeltPvqCase { N = n, Gain = gain };
                ConcentusCeltOracle.RenormaliseVector(x, n, gain);
                want[c] = x;
            }
            var (x2, _, _) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.RenormaliseVector, cases, xIn, new int[xIn.Length], new uint[count]);
            for (int c = 0; c < count; c++)
                for (int j = 0; j < cases[c].N; j++)
                    if (x2[c * CeltPvqStride + j] != want[c][j])
                        throw new Exception($"renormalise_vector N={cases[c].N} gain={cases[c].Gain}: x[{j}] {x2[c * CeltPvqStride + j]}, Concentus {want[c][j]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltPvqGpu_StereoItheta_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC31D);
            const int count = 100;
            var cases = new CeltPvqCase[count];
            var xIn = new int[count * CeltPvqStride]; var yIn = new int[count * CeltPvqStride]; var want = new int[count];
            for (int c = 0; c < count; c++)
            {
                int n = 1 + rng.Next(CeltPvqStride - 1), stereo = rng.Next(2);
                var x = CeltRandomShape(rng, n); var y = CeltRandomShape(rng, n);
                Array.Copy(x, 0, xIn, c * CeltPvqStride, n);
                Array.Copy(y, 0, yIn, c * CeltPvqStride, n);
                cases[c] = new CeltPvqCase { N = n, Gain = stereo };
                want[c] = ConcentusCeltOracle.StereoItheta(x, y, stereo, n);
            }
            var (_, _, s) = await CeltPvqLaunchAsync(acc, CeltPvqGpuTestKernel.Op.StereoItheta, cases, xIn, yIn, new uint[count]);
            for (int c = 0; c < count; c++)
                if ((int)s[c] != want[c]) throw new Exception($"stereo_itheta N={cases[c].N} stereo={cases[c].Gain}: {(int)s[c]}, Concentus {want[c]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
