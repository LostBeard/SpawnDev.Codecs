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

        private static readonly Type ModeT = A.GetType("Concentus.Celt.Structs.CeltMode", true)!;
        /// <summary>Concentus' static 48 kHz / 960 mode (CeltMode.mode48000_960_120).</summary>
        public static readonly object Mode = ModeT.GetField("mode48000_960_120", Any)!.GetValue(null)!;

        /// <summary>A DynamicMethod calling <paramref name="target"/> with the arguments at the <paramref name="casts"/> indices cast
        /// to Concentus' internal types (the delegate passes them as object).</summary>
        private static Delegate Emit(Type delegateType, MethodInfo target, Dictionary<int, Type> casts)
        {
            var parameters = delegateType.GetMethod("Invoke")!.GetParameters().Select(p => p.ParameterType).ToArray();
            var dm = new DynamicMethod("oracle_" + target.Name, delegateType.GetMethod("Invoke")!.ReturnType, parameters,
                typeof(ConcentusCeltOracle).Module, true);
            var il = dm.GetILGenerator();
            for (int i = 0; i < parameters.Length; i++)
            {
                il.Emit(OpCodes.Ldarg, (short)i);
                if (casts.TryGetValue(i, out var castTo)) il.Emit(OpCodes.Castclass, castTo);
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
            M("VQ", "alg_quant", typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), Ec, typeof(Span<byte>)),
            new Dictionary<int, Type> { [6] = Ec });
        private static readonly EncDoneFn EncDone = (EncDoneFn)Emit(typeof(EncDoneFn),
            Ec.GetMethod("enc_done", Any, null, new[] { typeof(Span<byte>) }, null)!, new Dictionary<int, Type> { [0] = Ec });

        public delegate int ComputeAllocationDecodeFn(object m, int start, int end, int[] offsets, int[] cap, int allocTrim,
            ref int intensity, ref int dualStereo, int total, out int balance, int[] pulses, int[] ebits, int[] finePriority,
            int c, int lm, object ec, ReadOnlySpan<byte> data, int prev, int signalBandwidth);
        public delegate void DecInitFn(object ec, ReadOnlySpan<byte> buf, uint storage);
        private static readonly Type IntRef = typeof(int).MakeByRefType();
        public static readonly ComputeAllocationDecodeFn ComputeAllocationDecode = (ComputeAllocationDecodeFn)Emit(typeof(ComputeAllocationDecodeFn),
            M("Rate", "compute_allocation_decode", ModeT, typeof(int), typeof(int), typeof(int[]), typeof(int[]), typeof(int), IntRef, IntRef,
                typeof(int), IntRef, typeof(int[]), typeof(int[]), typeof(int[]), typeof(int), typeof(int), Ec, typeof(ReadOnlySpan<byte>),
                typeof(int), typeof(int)),
            new Dictionary<int, Type> { [0] = ModeT, [15] = Ec });
        private static readonly DecInitFn DecInit = (DecInitFn)Emit(typeof(DecInitFn),
            Ec.GetMethod("dec_init", Any, null, new[] { typeof(ReadOnlySpan<byte>), typeof(uint) }, null)!, new Dictionary<int, Type> { [0] = Ec });
        private static readonly MethodInfo InitCapsM = M("CeltCommon", "init_caps", ModeT, typeof(int[]), typeof(int), typeof(int));

        /// <summary>Concentus init_caps for the static mode.</summary>
        public static int[] InitCaps(int lm, int c)
        {
            var cap = new int[CeltModeTables.NbEBands];
            InitCapsM.Invoke(null, new object[] { Mode, cap, lm, c });
            return cap;
        }

        // bands.c
        public delegate void IntensityStereoFn(object m, Span<int> x, int xPtr, Span<int> y, int yPtr, int[][] bandE, int bandID, int n);
        public delegate void StereoSplitFn(Span<int> x, int xPtr, Span<int> y, int yPtr, int n);
        public delegate void StereoMergeFn(Span<int> x, Span<int> y, int mid, int n);
        public delegate void Haar1Fn(Span<int> x, int xPtr, int n0, int stride);
        public delegate void HadamardFn(Span<int> x, int xPtr, int n0, int stride, int hadamard);
        private static readonly MethodInfo BitexactCosM = M("Bands", "bitexact_cos", typeof(int));
        private static readonly MethodInfo BitexactLog2tanM = M("Bands", "bitexact_log2tan", typeof(int), typeof(int));
        private static readonly MethodInfo ComputeQnM = M("Bands", "compute_qn", typeof(int), typeof(int), typeof(int), typeof(int), typeof(int));
        private static readonly MethodInfo AntiCollapseM = M("Bands", "anti_collapse", ModeT, typeof(int[][]), typeof(byte[]), typeof(int),
            typeof(int), typeof(int), typeof(int), typeof(int), typeof(int[]), typeof(int[]), typeof(int[]), typeof(int[]), typeof(uint));
        public static readonly IntensityStereoFn IntensityStereo = (IntensityStereoFn)Emit(typeof(IntensityStereoFn),
            M("Bands", "intensity_stereo", ModeT, typeof(Span<int>), typeof(int), typeof(Span<int>), typeof(int), typeof(int[][]), typeof(int), typeof(int)),
            new Dictionary<int, Type> { [0] = ModeT });
        public static readonly StereoSplitFn StereoSplit = (StereoSplitFn)Delegate.CreateDelegate(typeof(StereoSplitFn),
            M("Bands", "stereo_split", typeof(Span<int>), typeof(int), typeof(Span<int>), typeof(int), typeof(int)));
        public static readonly StereoMergeFn StereoMerge = (StereoMergeFn)Delegate.CreateDelegate(typeof(StereoMergeFn),
            M("Bands", "stereo_merge", typeof(Span<int>), typeof(Span<int>), typeof(int), typeof(int)));
        public static readonly Haar1Fn Haar1 = (Haar1Fn)Delegate.CreateDelegate(typeof(Haar1Fn),
            M("Bands", "haar1", typeof(Span<int>), typeof(int), typeof(int), typeof(int)));
        public static readonly HadamardFn DeinterleaveHadamard = (HadamardFn)Delegate.CreateDelegate(typeof(HadamardFn),
            M("Bands", "deinterleave_hadamard", typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int)));
        public static readonly HadamardFn InterleaveHadamard = (HadamardFn)Delegate.CreateDelegate(typeof(HadamardFn),
            M("Bands", "interleave_hadamard", typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int)));
        public static int BitexactCos(int x) => (int)BitexactCosM.Invoke(null, new object[] { x })!;
        public static int BitexactLog2tan(int s, int c) => (int)BitexactLog2tanM.Invoke(null, new object[] { s, c })!;
        public static int ComputeQn(int n, int b, int offset, int pulseCap, int stereo) =>
            (int)ComputeQnM.Invoke(null, new object[] { n, b, offset, pulseCap, stereo })!;
        public static void AntiCollapse(int[][] x, byte[] masks, int lm, int c, int size, int start, int end, int[] logE, int[] prev1,
            int[] prev2, int[] pulses, uint seed) =>
            AntiCollapseM.Invoke(null, new object[] { Mode, x, masks, lm, c, size, start, end, logE, prev1, prev2, pulses, seed });

        // celt.c comb_filter
        public delegate void CombFilterFn(Span<int> y, int yPtr, Span<int> x, int xPtr, int t0, int t1, int n, int g0, int g1,
            int tapset0, int tapset1, int[] window, int overlap);
        public static readonly CombFilterFn CombFilter = (CombFilterFn)Delegate.CreateDelegate(typeof(CombFilterFn),
            M("CeltCommon", "comb_filter", typeof(Span<int>), typeof(int), typeof(Span<int>), typeof(int), typeof(int), typeof(int), typeof(int),
                typeof(int), typeof(int), typeof(int), typeof(int), typeof(int[]), typeof(int)));
        /// <summary>Concentus' 120-sample overlap window (Tables.window120).</summary>
        public static readonly int[] Window120 = (int[])A.GetType("Concentus.Celt.Tables", true)!.GetField("window120", Any)!.GetValue(null)!;

        // quant_bands.c, celt.c tf_decode, bands.c denormalise_bands, celt_decoder.c deemphasis
        public delegate void UnquantCoarseEnergyFn(object m, int start, int end, int[] oldEBands, int intra, object dec, ReadOnlySpan<byte> data,
            int c, int lm);
        public delegate void TfDecodeFn(int start, int end, int isTransient, int[] tfRes, int lm, object dec, ReadOnlySpan<byte> data);
        public delegate void UnquantFineEnergyFn(object m, int start, int end, int[] oldEBands, int[] fineQuant, object dec, ReadOnlySpan<byte> data,
            int c);
        public delegate void UnquantEnergyFinaliseFn(object m, int start, int end, int[] oldEBands, int[] fineQuant, int[] finePriority,
            int bitsLeft, object dec, ReadOnlySpan<byte> data, int c);
        public delegate void DenormaliseBandsFn(object m, int[] x, Span<int> freq, int freqPtr, Span<int> bandLogE, int bandLogEPtr, int start,
            int end, int bigM, int downsample, int silence);
        public delegate void DeemphasisFn(int[][] input, int[] inputPtrs, Span<short> pcm, int pcmPtr, int n, int c, int downsample, int[] coef,
            int[] mem, int accum);
        public static readonly UnquantCoarseEnergyFn UnquantCoarseEnergy = (UnquantCoarseEnergyFn)Emit(typeof(UnquantCoarseEnergyFn),
            M("QuantizeBands", "unquant_coarse_energy", ModeT, typeof(int), typeof(int), typeof(int[]), typeof(int), Ec, typeof(ReadOnlySpan<byte>),
                typeof(int), typeof(int)),
            new Dictionary<int, Type> { [0] = ModeT, [5] = Ec });
        public static readonly TfDecodeFn TfDecode = (TfDecodeFn)Emit(typeof(TfDecodeFn),
            M("CeltCommon", "tf_decode", typeof(int), typeof(int), typeof(int), typeof(int[]), typeof(int), Ec, typeof(ReadOnlySpan<byte>)),
            new Dictionary<int, Type> { [5] = Ec });
        public static readonly UnquantFineEnergyFn UnquantFineEnergy = (UnquantFineEnergyFn)Emit(typeof(UnquantFineEnergyFn),
            M("QuantizeBands", "unquant_fine_energy", ModeT, typeof(int), typeof(int), typeof(int[]), typeof(int[]), Ec, typeof(ReadOnlySpan<byte>),
                typeof(int)),
            new Dictionary<int, Type> { [0] = ModeT, [5] = Ec });
        public static readonly UnquantEnergyFinaliseFn UnquantEnergyFinalise = (UnquantEnergyFinaliseFn)Emit(typeof(UnquantEnergyFinaliseFn),
            M("QuantizeBands", "unquant_energy_finalise", ModeT, typeof(int), typeof(int), typeof(int[]), typeof(int[]), typeof(int[]), typeof(int),
                Ec, typeof(ReadOnlySpan<byte>), typeof(int)),
            new Dictionary<int, Type> { [0] = ModeT, [7] = Ec });
        public static readonly DenormaliseBandsFn DenormaliseBands = (DenormaliseBandsFn)Emit(typeof(DenormaliseBandsFn),
            M("Bands", "denormalise_bands", ModeT, typeof(int[]), typeof(Span<int>), typeof(int), typeof(Span<int>), typeof(int), typeof(int),
                typeof(int), typeof(int), typeof(int), typeof(int)),
            new Dictionary<int, Type> { [0] = ModeT });
        public static readonly DeemphasisFn Deemphasis = (DeemphasisFn)Delegate.CreateDelegate(typeof(DeemphasisFn),
            M("CeltCommon", "deemphasis", typeof(int[][]), typeof(int[]), typeof(Span<short>), typeof(int), typeof(int), typeof(int), typeof(int),
                typeof(int[]), typeof(int[]), typeof(int)));
        public delegate void QuantAllBandsDecodeFn(int encode, object m, int start, int end, int[] x, int[] y, byte[] masks, int[][] bandE,
            int[] pulses, int shortBlocks, int spread, int dualStereo, int intensity, int[] tfRes, int totalBits, int balance, object ec,
            ReadOnlySpan<byte> data, int lm, int codedBands, ref uint seed);
        public static readonly QuantAllBandsDecodeFn QuantAllBandsDecode = (QuantAllBandsDecodeFn)Emit(typeof(QuantAllBandsDecodeFn),
            M("Bands", "quant_all_bands_decode", typeof(int), ModeT, typeof(int), typeof(int), typeof(int[]), typeof(int[]), typeof(byte[]),
                typeof(int[][]), typeof(int[]), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int[]), typeof(int), typeof(int), Ec,
                typeof(ReadOnlySpan<byte>), typeof(int), typeof(int), typeof(uint).MakeByRefType()),
            new Dictionary<int, Type> { [1] = ModeT, [16] = Ec });
        // mdct.c
        public delegate void MdctFn(object l, Span<int> input, int inputPtr, Span<int> output, int outputPtr, int[] window, int overlap,
            int shift, int stride);
        private static readonly Type MdctT = A.GetType("Concentus.Celt.Structs.MDCTLookup", true)!;
        /// <summary>The static mode's MDCT lookup (CeltMode.mdct).</summary>
        public static readonly object Mdct = ModeT.GetField("mdct", Any)!.GetValue(Mode)!;
        public static readonly MdctFn MdctBackward = (MdctFn)Emit(typeof(MdctFn),
            M("MDCT", "clt_mdct_backward", MdctT, typeof(Span<int>), typeof(int), typeof(Span<int>), typeof(int), typeof(int[]), typeof(int),
                typeof(int), typeof(int)),
            new Dictionary<int, Type> { [0] = MdctT });
        public static readonly MdctFn MdctForward = (MdctFn)Emit(typeof(MdctFn),
            M("MDCT", "clt_mdct_forward", MdctT, typeof(Span<int>), typeof(int), typeof(Span<int>), typeof(int), typeof(int[]), typeof(int),
                typeof(int), typeof(int)),
            new Dictionary<int, Type> { [0] = MdctT });
        private static readonly MethodInfo TellFracM = Ec.GetMethod("tell_frac", Any, null, Type.EmptyTypes, null)!;
        private static readonly FieldInfo RngF = Ec.GetField("rng", Any)!;
        /// <summary>The decoder's position in 1/8 bits (ec_tell_frac).</summary>
        public static uint TellFrac(object ec) => (uint)TellFracM.Invoke(ec, null)!;
        /// <summary>The decoder's range (the final range once a frame is done).</summary>
        public static uint Rng(object ec) => (uint)RngF.GetValue(ec)!;

        /// <summary>A Concentus range decoder over <paramref name="packet"/>.</summary>
        public static object Decoder(byte[] packet)
        {
            object ec = Activator.CreateInstance(Ec, true)!;
            DecInit(ec, packet, (uint)packet.Length);
            return ec;
        }

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

    /// <summary>
    /// A band-shape-like Q14 vector: inside int16, and small enough that its energy (sum of squares) fits int32, as CELT's
    /// normalised bands always do. Larger vectors overflow the int accumulators of renormalise_vector / stereo_merge /
    /// stereo_itheta: C# wraps, OpenCL C leaves signed overflow undefined, and the results differ (outside CELT's domain).
    /// </summary>
    private static int[] CeltRandomShape(Random rng, int n)
    {
        int amp = (int)Math.Min(12000, Math.Sqrt((1 << 30) / (double)Math.Max(1, n)));
        var x = new int[n];
        for (int i = 0; i < n; i++) x[i] = rng.Next(-amp, amp + 1);
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

    [TestMethod]
    public async Task CeltRateGpu_ComputeAllocationDecode_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int nb = CeltModeTables.NbEBands, count = 150;
            var rng = new Random(0xC320);
            var cases = new CeltRateCase[count]; var blocks = new int[count * CeltRateGpuTestKernel.OutInts];
            var packets = new List<byte>();
            var want = new (int[] Cap, int[] Pulses, int[] Ebits, int[] Fine, int Coded, int Balance, int Intensity, int Dual)[count];
            for (int c = 0; c < count; c++)
            {
                // mono / stereo, every frame size, full band and the hybrid start band, every trim, small and large budgets;
                // the allocation decisions (skip, intensity, dual stereo) decode from random bytes, the same bytes on both sides
                int C = 1 + rng.Next(2), lm = rng.Next(4);
                int start = rng.Next(4) == 0 ? 17 : 0;
                int end = new[] { 13, 17, 19, 21 }[rng.Next(4)]; if (end <= start) end = nb;
                int trim = rng.Next(11), total = rng.Next(rng.Next(2) == 0 ? 2000 : 16000);
                int prev = rng.Next(nb + 1), signalBandwidth = rng.Next(end);
                var offsets = new int[nb];
                for (int i = start; i < end; i++) if (rng.Next(5) == 0) offsets[i] = 8 * (1 + rng.Next(6)) * C;
                var packet = new byte[64]; rng.NextBytes(packet);
                cases[c] = new CeltRateCase { Start = start, End = end, AllocTrim = trim, Total = total, C = C, LM = lm, Prev = prev,
                    SignalBandwidth = signalBandwidth, PacketAt = packets.Count, PacketLength = packet.Length };
                packets.AddRange(packet);
                Array.Copy(offsets, 0, blocks, c * CeltRateGpuTestKernel.OutInts, nb);
                var cap = ConcentusCeltOracle.InitCaps(lm, C);
                var pulses = new int[nb]; var ebits = new int[nb]; var fine = new int[nb];
                int intensity = 0, dual = 0;
                int coded = ConcentusCeltOracle.ComputeAllocationDecode(ConcentusCeltOracle.Mode, start, end, offsets, (int[])cap.Clone(), trim,
                    ref intensity, ref dual, total, out int balance, pulses, ebits, fine, C, lm, ConcentusCeltOracle.Decoder(packet), packet,
                    prev, signalBandwidth);
                want[c] = (cap, pulses, ebits, fine, coded, balance, intensity, dual);
            }
            var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
            using var _ = modeBuffer;
            using var dCases = acc.Allocate1D(cases);
            using var dBlocks = acc.Allocate1D(blocks);
            using var dScratch = acc.Allocate1D<int>(count * 4 * nb);
            using var dPackets = acc.Allocate1D(packets.ToArray());
            using var kernel = new CeltRateGpuTestKernel(acc);
            kernel.Run(dCases.View, mode, dBlocks.View, dScratch.View, dPackets.View);
            await acc.SynchronizeAsync();
            var got = await dBlocks.CopyToHostAsync();
            for (int c = 0; c < count; c++)
            {
                int o = c * CeltRateGpuTestKernel.OutInts;
                var w = want[c];
                string where = $"C={cases[c].C} LM={cases[c].LM} bands {cases[c].Start}..{cases[c].End} trim={cases[c].AllocTrim} total={cases[c].Total}";
                for (int i = 0; i < nb; i++)
                {
                    if (got[o + nb + i] != w.Cap[i]) throw new Exception($"init_caps {where}: cap[{i}] {got[o + nb + i]}, Concentus {w.Cap[i]}");
                    if (got[o + 2 * nb + i] != w.Pulses[i]) throw new Exception($"compute_allocation {where}: pulses[{i}] {got[o + 2 * nb + i]}, Concentus {w.Pulses[i]}");
                    if (got[o + 3 * nb + i] != w.Ebits[i]) throw new Exception($"compute_allocation {where}: ebits[{i}] {got[o + 3 * nb + i]}, Concentus {w.Ebits[i]}");
                    if (got[o + 4 * nb + i] != w.Fine[i]) throw new Exception($"compute_allocation {where}: fine_priority[{i}] {got[o + 4 * nb + i]}, Concentus {w.Fine[i]}");
                }
                int r = o + 5 * nb;
                if (got[r] != w.Coded || got[r + 1] != w.Balance || got[r + 2] != w.Intensity || got[r + 3] != w.Dual)
                    throw new Exception($"compute_allocation {where}: coded/balance/intensity/dual {got[r]}/{got[r + 1]}/{got[r + 2]}/{got[r + 3]}, " +
                        $"Concentus {w.Coded}/{w.Balance}/{w.Intensity}/{w.Dual}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private const int CeltBandsStride = 2048;

    /// <summary>Run one <see cref="CeltBandsGpuTestKernel"/> op over <paramref name="cases"/>; returns X, Y and the per-case scalars.</summary>
    private static async Task<(int[] X, int[] Y, int[] Scalar)> CeltBandsLaunchAsync(Accelerator acc, CeltBandsGpuTestKernel.Op op,
        CeltBandsCase[] cases, int[] x, int[] y, int[] aux)
    {
        var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
        using var _ = modeBuffer;
        using var dCases = acc.Allocate1D(cases);
        using var dX = acc.Allocate1D(x);
        using var dY = acc.Allocate1D(y);
        using var dAux = acc.Allocate1D(aux);
        using var dScratch = acc.Allocate1D<int>(x.Length);
        using var dS = acc.Allocate1D<int>(cases.Length);
        using var kernel = new CeltBandsGpuTestKernel(acc);
        kernel.Run(op, dCases.View, mode, dX.View, dY.View, dAux.View, dScratch.View, dS.View, CeltBandsStride);
        await acc.SynchronizeAsync();
        return (await dX.CopyToHostAsync(), await dY.CopyToHostAsync(), await dS.CopyToHostAsync());
    }

    [TestMethod]
    public async Task CeltBandsGpu_SplitHelpers_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC330);
            const int n = 200;
            var cos = new CeltBandsCase[n]; var log2tan = new CeltBandsCase[n]; var qn = new CeltBandsCase[n];
            var wantCos = new int[n]; var wantLog2tan = new int[n]; var wantQn = new int[n];
            for (int c = 0; c < n; c++)
            {
                int a = rng.Next(0, 16385);
                cos[c] = new CeltBandsCase { A = a }; wantCos[c] = ConcentusCeltOracle.BitexactCos(a);
                int s = rng.Next(1, 32768), k = rng.Next(1, 32768);
                log2tan[c] = new CeltBandsCase { A = s, B = k }; wantLog2tan[c] = ConcentusCeltOracle.BitexactLog2tan(s, k);
                int bandN = 1 + rng.Next(176), b = rng.Next(-100, 3000), offset = rng.Next(-30, 60), cap = rng.Next(0, 600), stereo = rng.Next(2);
                qn[c] = new CeltBandsCase { A = bandN, B = b, C = offset, D = cap, E = stereo };
                wantQn[c] = ConcentusCeltOracle.ComputeQn(bandN, b, offset, cap, stereo);
            }
            var empty = new int[n * CeltBandsStride];
            var (_, _, s1) = await CeltBandsLaunchAsync(acc, CeltBandsGpuTestKernel.Op.BitexactCos, cos, empty, empty, empty);
            var (_, _, s2) = await CeltBandsLaunchAsync(acc, CeltBandsGpuTestKernel.Op.BitexactLog2tan, log2tan, empty, empty, empty);
            var (_, _, s3) = await CeltBandsLaunchAsync(acc, CeltBandsGpuTestKernel.Op.ComputeQn, qn, empty, empty, empty);
            for (int c = 0; c < n; c++)
            {
                if (s1[c] != wantCos[c]) throw new Exception($"bitexact_cos({cos[c].A}): {s1[c]}, Concentus {wantCos[c]}");
                if (s2[c] != wantLog2tan[c]) throw new Exception($"bitexact_log2tan({log2tan[c].A}, {log2tan[c].B}): {s2[c]}, Concentus {wantLog2tan[c]}");
                if (s3[c] != wantQn[c]) throw new Exception($"compute_qn(N={qn[c].A}, b={qn[c].B}, offset={qn[c].C}, cap={qn[c].D}, stereo={qn[c].E}): {s3[c]}, Concentus {wantQn[c]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltBandsGpu_StereoOps_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC331);
            foreach (var op in new[] { CeltBandsGpuTestKernel.Op.IntensityStereo, CeltBandsGpuTestKernel.Op.StereoSplit, CeltBandsGpuTestKernel.Op.StereoMerge })
            {
                const int n = 80;
                var cases = new CeltBandsCase[n];
                var xIn = new int[n * CeltBandsStride]; var yIn = new int[n * CeltBandsStride]; var aux = new int[n * CeltBandsStride];
                var wantX = new int[n][]; var wantY = new int[n][];
                for (int c = 0; c < n; c++)
                {
                    int bandN = 1 + rng.Next(176);
                    var x = CeltRandomShape(rng, bandN); var y = CeltRandomShape(rng, bandN);
                    Array.Copy(x, 0, xIn, c * CeltBandsStride, bandN);
                    Array.Copy(y, 0, yIn, c * CeltBandsStride, bandN);
                    int mid = rng.Next(0, 32768);
                    cases[c] = new CeltBandsCase { A = bandN, B = mid };
                    if (op == CeltBandsGpuTestKernel.Op.IntensityStereo)
                    {
                        int e0 = rng.Next(1, 1 << rng.Next(1, 30)), e1 = rng.Next(1, 1 << rng.Next(1, 30));
                        aux[c * CeltBandsStride] = e0; aux[c * CeltBandsStride + 1] = e1;
                        ConcentusCeltOracle.IntensityStereo(ConcentusCeltOracle.Mode, x, 0, y, 0, new[] { new[] { e0 }, new[] { e1 } }, 0, bandN);
                    }
                    else if (op == CeltBandsGpuTestKernel.Op.StereoSplit) ConcentusCeltOracle.StereoSplit(x, 0, y, 0, bandN);
                    else ConcentusCeltOracle.StereoMerge(x, y, mid, bandN);
                    wantX[c] = x; wantY[c] = y;
                }
                var (x2, y2, _) = await CeltBandsLaunchAsync(acc, op, cases, xIn, yIn, aux);
                for (int c = 0; c < n; c++)
                    for (int j = 0; j < cases[c].A; j++)
                        if (x2[c * CeltBandsStride + j] != wantX[c][j] || y2[c * CeltBandsStride + j] != wantY[c][j])
                            throw new Exception($"{op} N={cases[c].A}: [{j}] x {x2[c * CeltBandsStride + j]} y {y2[c * CeltBandsStride + j]}, Concentus x {wantX[c][j]} y {wantY[c][j]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltBandsGpu_HaarAndInterleave_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(0xC332);
            foreach (var op in new[] { CeltBandsGpuTestKernel.Op.Haar1, CeltBandsGpuTestKernel.Op.Deinterleave, CeltBandsGpuTestKernel.Op.Interleave })
            {
                const int n = 80;
                var cases = new CeltBandsCase[n]; var xIn = new int[n * CeltBandsStride]; var want = new int[n][]; var len = new int[n];
                for (int c = 0; c < n; c++)
                {
                    // strides 2..16 (the Hadamard order table covers exactly those), N0 * stride inside the per-case window
                    int stride = 1 << rng.Next(1, 5), hadamard = op == CeltBandsGpuTestKernel.Op.Haar1 ? 0 : rng.Next(2);
                    int n0 = op == CeltBandsGpuTestKernel.Op.Haar1 ? 2 * (1 + rng.Next(60)) : 1 + rng.Next(CeltBandsStride / stride / 2);
                    len[c] = n0 * stride;
                    var x = CeltRandomShape(rng, len[c]);
                    Array.Copy(x, 0, xIn, c * CeltBandsStride, len[c]);
                    cases[c] = new CeltBandsCase { A = n0, B = stride, C = hadamard };
                    if (op == CeltBandsGpuTestKernel.Op.Haar1) ConcentusCeltOracle.Haar1(x, 0, n0, stride);
                    else if (op == CeltBandsGpuTestKernel.Op.Deinterleave) ConcentusCeltOracle.DeinterleaveHadamard(x, 0, n0, stride, hadamard);
                    else ConcentusCeltOracle.InterleaveHadamard(x, 0, n0, stride, hadamard);
                    want[c] = x;
                }
                var (x2, _, _) = await CeltBandsLaunchAsync(acc, op, cases, xIn, new int[xIn.Length], new int[xIn.Length]);
                for (int c = 0; c < n; c++)
                    for (int j = 0; j < len[c]; j++)
                        if (x2[c * CeltBandsStride + j] != want[c][j])
                            throw new Exception($"{op} N0={cases[c].A} stride={cases[c].B} hadamard={cases[c].C}: x[{j}] {x2[c * CeltBandsStride + j]}, Concentus {want[c][j]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltBandsGpu_AntiCollapse_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int nb = CeltModeTables.NbEBands, n = 60;
            var rng = new Random(0xC333);
            var cases = new CeltBandsCase[n]; var xIn = new int[n * CeltBandsStride]; var aux = new int[n * CeltBandsStride];
            var want = new int[n][];
            for (int c = 0; c < n; c++)
            {
                // a frame of C channels x (120 << LM) bins, random collapse masks (one bit per short block), energy history, pulses
                int lm = rng.Next(4), C = 1 + rng.Next(2), size = 120 << lm;
                int start = rng.Next(3) == 0 ? 17 : 0, end = new[] { 13, 17, 19, 21 }[rng.Next(4)];
                if (end <= start) end = nb;
                uint seed = (uint)rng.NextInt64(uint.MaxValue);
                var x = new int[C][];
                for (int ch = 0; ch < C; ch++) x[ch] = CeltRandomShape(rng, size);
                var masks = new byte[nb * C];
                for (int i = 0; i < masks.Length; i++) masks[i] = (byte)rng.Next(Math.Min(256, 1 << (1 << lm)));
                int[] Energies() { var e = new int[2 * nb]; for (int i = 0; i < e.Length; i++) e[i] = rng.Next(-10 << 10, 20 << 10); return e; }
                var logE = Energies(); var prev1 = Energies(); var prev2 = Energies();
                var pulses = new int[nb];
                for (int i = 0; i < nb; i++) pulses[i] = rng.Next(0, 300);
                int o = c * CeltBandsStride;
                for (int ch = 0; ch < C; ch++) Array.Copy(x[ch], 0, xIn, o + ch * size, size);
                for (int i = 0; i < masks.Length; i++) aux[o + i] = masks[i];
                Array.Copy(logE, 0, aux, o + 2 * nb, 2 * nb);
                Array.Copy(prev1, 0, aux, o + 4 * nb, 2 * nb);
                Array.Copy(prev2, 0, aux, o + 6 * nb, 2 * nb);
                Array.Copy(pulses, 0, aux, o + 8 * nb, nb);
                cases[c] = new CeltBandsCase { A = lm, B = C, C = start, D = end, E = (int)seed, F = size };
                ConcentusCeltOracle.AntiCollapse(x, masks, lm, C, size, start, end, logE, prev1, prev2, pulses, seed);
                want[c] = x.SelectMany(v => v).ToArray();
            }
            var (x2, _, _) = await CeltBandsLaunchAsync(acc, CeltBandsGpuTestKernel.Op.AntiCollapse, cases, xIn, new int[xIn.Length], aux);
            for (int c = 0; c < n; c++)
                for (int j = 0; j < want[c].Length; j++)
                    if (x2[c * CeltBandsStride + j] != want[c][j])
                        throw new Exception($"anti_collapse LM={cases[c].A} C={cases[c].B} bands {cases[c].C}..{cases[c].D}: x[{j}] {x2[c * CeltBandsStride + j]}, Concentus {want[c][j]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltCombFilterGpu_PostFilter_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 80, history = CeltBandsGpuTestKernel.CombHistory;
            var rng = new Random(0xC334);
            var cases = new CeltBandsCase[n];
            var xIn = new int[n * CeltBandsStride]; var yIn = new int[n * CeltBandsStride];
            var want = new int[n][]; var inPlace = new bool[n];
            for (int c = 0; c < n; c++)
            {
                // Q12 signal, every frame size, periods 15..1023, gains up to 0.75, every tapset; some frames repeat the previous
                // frame's parameters (no cross-fade), some have a zero gain; half run in place as the decoder does
                int frame = 120 << rng.Next(4), t0 = rng.Next(15, 1024), t1 = rng.Next(4) == 0 ? t0 : rng.Next(15, 1024);
                int g0 = rng.Next(5) == 0 ? 0 : rng.Next(0, 24577), g1 = rng.Next(5) == 0 ? 0 : rng.Next(0, 24577);
                int tapset0 = rng.Next(3), tapset1 = rng.Next(3);
                if (rng.Next(6) == 0) { g1 = g0; t1 = t0; tapset1 = tapset0; }
                inPlace[c] = rng.Next(2) == 0;
                var x = new int[history + frame]; for (int i = 0; i < x.Length; i++) x[i] = rng.Next(-(1 << 27), 1 << 27);
                var y = new int[history + frame]; for (int i = 0; i < y.Length; i++) y[i] = rng.Next(-(1 << 27), 1 << 27);
                Array.Copy(x, 0, xIn, c * CeltBandsStride, x.Length);
                Array.Copy(y, 0, yIn, c * CeltBandsStride, y.Length);
                cases[c] = new CeltBandsCase { A = t0, B = t1, C = frame, D = g0, E = g1, F = tapset0, G = tapset1, H = inPlace[c] ? 1 : 0 };
                if (inPlace[c])
                {
                    ConcentusCeltOracle.CombFilter(x, history, x, history, t0, t1, frame, g0, g1, tapset0, tapset1, ConcentusCeltOracle.Window120, 120);
                    want[c] = x;
                }
                else
                {
                    ConcentusCeltOracle.CombFilter(y, history, x, history, t0, t1, frame, g0, g1, tapset0, tapset1, ConcentusCeltOracle.Window120, 120);
                    want[c] = y;
                }
            }
            var (x2, y2, _) = await CeltBandsLaunchAsync(acc, CeltBandsGpuTestKernel.Op.CombFilter, cases, xIn, yIn, new int[xIn.Length]);
            for (int c = 0; c < n; c++)
            {
                var got = inPlace[c] ? x2 : y2;
                for (int j = 0; j < want[c].Length; j++)
                    if (got[c * CeltBandsStride + j] != want[c][j])
                        throw new Exception($"comb_filter T0={cases[c].A} T1={cases[c].B} N={cases[c].C} g0={cases[c].D} g1={cases[c].E} " +
                            $"tapsets {cases[c].F}/{cases[c].G} in place={inPlace[c]}: y[{j - history}] {got[c * CeltBandsStride + j]}, Concentus {want[c][j]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // quant_partition's recursion runs as an explicit-stack loop that leaves from both arms of an if; before SpawnDev.ILGPU
    // 5.4.12 that hung Velocity (block order) and ran the wrong arm's code on WebGPU (a back edge without `continue`).
    [TestMethod]
    public async Task CeltQuantBandsGpu_QuantAllBandsDecode_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int nb = CeltModeTables.NbEBands, count = 200, B = CeltQuantBandsGpuTestKernel.BlockInts;
            var rng = new Random(0xC343);
            var cases = new CeltQuantBandsCase[count]; var blocks = new int[count * B];
            var packets = new List<byte>();
            var want = new (int[] X, int[] Y, byte[] Masks, int Coded, int Balance, int Intensity, int Dual, uint Seed, int Tell)[count];
            for (int c = 0; c < count; c++)
            {
                // the allocation then every band of a frame on one decoder, as celt_decode_with_ec runs them: mono / stereo, every
                // frame size, long and short blocks, every spread and trim, full band and the hybrid start band, 20..219-byte
                // packets of random bytes (the same on both sides), random dynalloc offsets and tf_res from tf_select_table
                int C = 1 + rng.Next(2), lm = rng.Next(4), bigM = 1 << lm;
                int start = rng.Next(4) == 0 ? 17 : 0, end = new[] { 13, 17, 19, 21 }[rng.Next(4)]; if (end <= start) end = nb;
                int trim = rng.Next(11), spread = rng.Next(4);
                int isTransient = lm > 0 && rng.Next(3) == 0 ? 1 : 0, shortBlocks = isTransient != 0 ? bigM : 0, tfSel = rng.Next(2);
                int len = 20 + rng.Next(200);
                var packet = new byte[len]; rng.NextBytes(packet);
                int prev = rng.Next(nb + 1), signalBandwidth = rng.Next(end);
                uint seed = (uint)rng.NextInt64(uint.MaxValue);
                var offsets = new int[nb]; for (int i = start; i < end; i++) if (rng.Next(6) == 0) offsets[i] = 8 * (1 + rng.Next(4)) * C;
                var tf = new int[nb];
                for (int i = start; i < end; i++) tf[i] = CeltModeTables.TfSelectTable[lm * 8 + 4 * isTransient + 2 * tfSel + rng.Next(2)];
                int totalBits = len * 8 << 3;
                object dec = ConcentusCeltOracle.Decoder(packet);
                int allocTotal = totalBits - (int)ConcentusCeltOracle.TellFrac(dec) - 1;
                var cap = ConcentusCeltOracle.InitCaps(lm, C);
                var pulses = new int[nb]; var ebits = new int[nb]; var fine = new int[nb];
                int intensity = 0, dual = 0;
                int coded = ConcentusCeltOracle.ComputeAllocationDecode(ConcentusCeltOracle.Mode, start, end, offsets, cap, trim, ref intensity,
                    ref dual, allocTotal, out int balance, pulses, ebits, fine, C, lm, dec, packet, prev, signalBandwidth);
                var x = new int[bigM * 100]; var y = C == 2 ? new int[bigM * 100] : null; var masks = new byte[nb * C];
                uint seedOut = seed;
                ConcentusCeltOracle.QuantAllBandsDecode(0, ConcentusCeltOracle.Mode, start, end, x, y, masks, new[] { new int[nb], new int[nb] },
                    pulses, shortBlocks, spread, dual, intensity, tf, totalBits, balance, dec, packet, lm, coded, ref seedOut);
                want[c] = (x, y, masks, coded, balance, intensity, dual, seedOut, (int)ConcentusCeltOracle.TellFrac(dec));
                cases[c] = new CeltQuantBandsCase { C = C, LM = lm, Start = start, End = end, AllocTrim = trim, Spread = spread,
                    ShortBlocks = shortBlocks, Seed = seed, Total = allocTotal, TotalBits = totalBits, Prev = prev,
                    SignalBandwidth = signalBandwidth, PacketAt = packets.Count, PacketLength = len };
                packets.AddRange(packet); while (packets.Count % 4 != 0) packets.Add(0);
                Array.Copy(offsets, 0, blocks, c * B + CeltQuantBandsGpuTestKernel.OffsetsAt, nb);
                Array.Copy(tf, 0, blocks, c * B + CeltQuantBandsGpuTestKernel.TfAt, nb);
            }
            var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
            using var _ = modeBuffer;
            using var dU = acc.Allocate1D(CeltPvqTables.Data);
            using var dCases = acc.Allocate1D(cases);
            using var dBlocks = acc.Allocate1D(blocks);
            using var dScratch = acc.Allocate1D<int>(count * CeltQuantBandsGpuTestKernel.ScratchPerCase);
            using var dPackets = acc.Allocate1D(packets.ToArray());
            using var kernel = new CeltQuantBandsGpuTestKernel(acc);
            kernel.Run(dCases.View, mode, dBlocks.View, dScratch.View, dPackets.View, dU.View);
            await acc.SynchronizeAsync();
            var got = await dBlocks.CopyToHostAsync();
            for (int c = 0; c < count; c++)
            {
                int o = c * B, r = o + CeltQuantBandsGpuTestKernel.ResultsAt;
                var w = want[c];
                var k = cases[c];
                string where = $"C={k.C} LM={k.LM} bands {k.Start}..{k.End} short={k.ShortBlocks} spread={k.Spread} trim={k.AllocTrim} {k.PacketLength} bytes";
                if (got[r] != w.Coded || got[r + 1] != w.Balance || got[r + 2] != w.Intensity || got[r + 3] != w.Dual)
                    throw new Exception($"compute_allocation {where}: coded/balance/intensity/dual {got[r]}/{got[r + 1]}/{got[r + 2]}/{got[r + 3]}, " +
                        $"Concentus {w.Coded}/{w.Balance}/{w.Intensity}/{w.Dual}");
                for (int j = 0; j < w.X.Length; j++)
                    if (got[o + CeltQuantBandsGpuTestKernel.XAt + j] != w.X[j])
                        throw new Exception($"quant_all_bands {where}: X[{j}] {got[o + CeltQuantBandsGpuTestKernel.XAt + j]}, Concentus {w.X[j]}");
                for (int j = 0; w.Y != null && j < w.Y.Length; j++)
                    if (got[o + CeltQuantBandsGpuTestKernel.YAt + j] != w.Y[j])
                        throw new Exception($"quant_all_bands {where}: Y[{j}] {got[o + CeltQuantBandsGpuTestKernel.YAt + j]}, Concentus {w.Y[j]}");
                for (int j = 0; j < w.Masks.Length; j++)
                    if (got[o + CeltQuantBandsGpuTestKernel.MasksAt + j] != w.Masks[j])
                        throw new Exception($"quant_all_bands {where}: collapse mask[{j}] {got[o + CeltQuantBandsGpuTestKernel.MasksAt + j]}, Concentus {w.Masks[j]}");
                if ((uint)got[r + 4] != w.Seed || got[r + 5] != w.Tell)
                    throw new Exception($"quant_all_bands {where}: seed / tell_frac {(uint)got[r + 4]} / {got[r + 5]}, Concentus {w.Seed} / {w.Tell}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltMdctGpu_BackwardAndForward_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int count = 64, B = CeltMdctGpuTestKernel.BlockInts, overlap = CeltMdctGpuTestKernel.Overlap;
            var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
            using var _ = modeBuffer;
            using var kernel = new CeltMdctGpuTestKernel(acc);
            foreach (bool backward in new[] { true, false })
            {
                var rng = new Random(backward ? 0xC344 : 0xC345);
                var cases = new CeltMdctCase[count]; var blocks = new int[count * B];
                var want = new int[count][];
                for (int c = 0; c < count; c++)
                {
                    // every frame size, one long block or 1 << LM short blocks (every FFT size: 480, 240, 120, 60 points);
                    // coefficients of up to 2^18 and Q12 samples of up to 2^26, which no butterfly can push past int32; the
                    // inverse also reads the previous frame's overlap already in its output (random here, the same on both sides)
                    int lm = rng.Next(4), transient = lm > 0 && rng.Next(2) == 0 ? 1 : 0;
                    int shift = transient != 0 ? 3 : 3 - lm, blocksN = transient != 0 ? 1 << lm : 1;
                    int half = (CeltMdctGpu.MdctN >> shift) >> 1, frame = half * blocksN;
                    int amp = backward ? (rng.Next(2) == 0 ? 1 << 18 : 1 << 12) : (rng.Next(2) == 0 ? 1 << 26 : 1 << 16);
                    var input = new int[backward ? frame : frame + overlap];
                    for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-amp, amp);
                    var output = new int[backward ? frame + overlap : frame];
                    if (backward) for (int i = 0; i < output.Length; i++) output[i] = rng.Next(-(1 << 26), 1 << 26);
                    cases[c] = new CeltMdctCase { Shift = shift, Blocks = blocksN };
                    int o = c * B;
                    Array.Copy(input, 0, blocks, o + CeltMdctGpuTestKernel.InAt, input.Length);
                    Array.Copy(output, 0, blocks, o + CeltMdctGpuTestKernel.OutAt, output.Length);
                    for (int b = 0; b < blocksN; b++)
                    {
                        if (backward)
                            ConcentusCeltOracle.MdctBackward(ConcentusCeltOracle.Mdct, input, b, output, half * b, ConcentusCeltOracle.Window120,
                                overlap, shift, blocksN);
                        else
                            ConcentusCeltOracle.MdctForward(ConcentusCeltOracle.Mdct, input, half * b, output, b, ConcentusCeltOracle.Window120,
                                overlap, shift, blocksN);
                    }
                    want[c] = output;
                }
                using var dCases = acc.Allocate1D(cases);
                using var dBlocks = acc.Allocate1D(blocks);
                kernel.Run(backward, dCases.View, mode, dBlocks.View);
                await acc.SynchronizeAsync();
                var got = await dBlocks.CopyToHostAsync();
                for (int c = 0; c < count; c++)
                {
                    int o = c * B + CeltMdctGpuTestKernel.OutAt;
                    for (int i = 0; i < want[c].Length; i++)
                        if (got[o + i] != want[c][i])
                            throw new Exception($"clt_mdct_{(backward ? "backward" : "forward")} shift={cases[c].Shift} blocks={cases[c].Blocks}: " +
                                $"out[{i}] {got[o + i]}, Concentus {want[c][i]}");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltEnergyGpu_CoarseTfFineFinalise_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int nb = CeltModeTables.NbEBands, count = 200, B = CeltEnergyGpuTestKernel.BlockInts;
            var rng = new Random(0xC340);
            var cases = new CeltEnergyCase[count]; var blocks = new int[count * B];
            var packets = new List<byte>();
            var want = new (int[] OldE, int[] Tf, uint[] Tell, uint Rng)[count];
            for (int c = 0; c < count; c++)
            {
                // mono / stereo, every frame size, full band and the hybrid start band, intra and inter, transient or not; packets
                // from 2 to 80 bytes so the coarse energy runs out of bits (the 2-bit icdf and 1-bit fallbacks) as well as Laplace-
                // decodes; the previous frame's energies, fine_quant (0..8) and fine_priority at random, the same on both sides
                int C = 1 + rng.Next(2), lm = rng.Next(4);
                int start = rng.Next(4) == 0 ? 17 : 0;
                int end = new[] { 13, 17, 19, 21 }[rng.Next(4)]; if (end <= start) end = nb;
                int intra = rng.Next(2), transient = rng.Next(2), bitsLeft = rng.Next(48);
                var oldE = new int[2 * nb]; for (int i = 0; i < oldE.Length; i++) oldE[i] = rng.Next(-12000, 20000);
                var fine = new int[nb]; for (int i = 0; i < nb; i++) fine[i] = rng.Next(3) == 0 ? 0 : rng.Next(9);
                var prio = new int[nb]; for (int i = 0; i < nb; i++) prio[i] = rng.Next(2);
                var packet = new byte[2 + rng.Next(79)]; rng.NextBytes(packet);
                cases[c] = new CeltEnergyCase { C = C, LM = lm, Start = start, End = end, Intra = intra, IsTransient = transient,
                    BitsLeft = bitsLeft, PacketAt = packets.Count, PacketLength = packet.Length };
                packets.AddRange(packet);
                int o = c * B;
                Array.Copy(oldE, 0, blocks, o + CeltEnergyGpuTestKernel.OldEAt, oldE.Length);
                Array.Copy(fine, 0, blocks, o + CeltEnergyGpuTestKernel.FineAt, nb);
                Array.Copy(prio, 0, blocks, o + CeltEnergyGpuTestKernel.PrioAt, nb);
                var dec = ConcentusCeltOracle.Decoder(packet);
                var tell = new uint[4];
                var tf = new int[nb];
                ConcentusCeltOracle.UnquantCoarseEnergy(ConcentusCeltOracle.Mode, start, end, oldE, intra, dec, packet, C, lm);
                tell[0] = ConcentusCeltOracle.TellFrac(dec);
                ConcentusCeltOracle.TfDecode(start, end, transient, tf, lm, dec, packet);
                tell[1] = ConcentusCeltOracle.TellFrac(dec);
                ConcentusCeltOracle.UnquantFineEnergy(ConcentusCeltOracle.Mode, start, end, oldE, fine, dec, packet, C);
                tell[2] = ConcentusCeltOracle.TellFrac(dec);
                ConcentusCeltOracle.UnquantEnergyFinalise(ConcentusCeltOracle.Mode, start, end, oldE, fine, prio, bitsLeft, dec, packet, C);
                tell[3] = ConcentusCeltOracle.TellFrac(dec);
                want[c] = (oldE, tf, tell, ConcentusCeltOracle.Rng(dec));
            }
            var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
            using var _ = modeBuffer;
            using var dCases = acc.Allocate1D(cases);
            using var dBlocks = acc.Allocate1D(blocks);
            using var dPackets = acc.Allocate1D(packets.ToArray());
            using var kernel = new CeltEnergyGpuTestKernel(acc);
            kernel.Run(dCases.View, mode, dBlocks.View, dPackets.View);
            await acc.SynchronizeAsync();
            var got = await dBlocks.CopyToHostAsync();
            string[] stage = { "unquant_coarse_energy", "tf_decode", "unquant_fine_energy", "unquant_energy_finalise" };
            for (int c = 0; c < count; c++)
            {
                int o = c * B;
                var w = want[c];
                var k = cases[c];
                string where = $"C={k.C} LM={k.LM} bands {k.Start}..{k.End} intra={k.Intra} transient={k.IsTransient} {k.PacketLength} bytes";
                for (int s = 0; s < 4; s++)
                    if ((uint)got[o + CeltEnergyGpuTestKernel.ResultsAt + s] != w.Tell[s])
                        throw new Exception($"{stage[s]} {where}: tell_frac {(uint)got[o + CeltEnergyGpuTestKernel.ResultsAt + s]}, Concentus {w.Tell[s]}");
                for (int i = k.Start; i < k.End; i++)
                    if (got[o + CeltEnergyGpuTestKernel.TfAt + i] != w.Tf[i])
                        throw new Exception($"tf_decode {where}: tf_res[{i}] {got[o + CeltEnergyGpuTestKernel.TfAt + i]}, Concentus {w.Tf[i]}");
                for (int i = 0; i < 2 * nb; i++)
                    if (got[o + CeltEnergyGpuTestKernel.OldEAt + i] != w.OldE[i])
                        throw new Exception($"band energies {where}: oldEBands[{i}] {got[o + CeltEnergyGpuTestKernel.OldEAt + i]}, Concentus {w.OldE[i]}");
                if ((uint)got[o + CeltEnergyGpuTestKernel.ResultsAt + 4] != w.Rng)
                    throw new Exception($"energy decode {where}: rng {(uint)got[o + CeltEnergyGpuTestKernel.ResultsAt + 4]:X8}, Concentus {w.Rng:X8}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltSynthesisGpu_DenormaliseBands_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int nb = CeltModeTables.NbEBands, count = 120, B = CeltSynthesisGpuTestKernel.BlockInts;
            var rng = new Random(0xC341);
            var cases = new CeltSynthesisCase[count]; var blocks = new int[count * B];
            var want = new int[count][];
            int[] downsamples = { 1, 1, 1, 2, 3, 4, 6 };
            for (int c = 0; c < count; c++)
            {
                // every frame size, full band and the hybrid start band, every decimation, some silence frames; Q15 shapes and band
                // energies from below the zero-gain cut (shift > 31) up to the largest gain whose shift stays >= 0 (above it
                // Concentus 2.2.2 and libopus differ, see CeltSynthesisGpu.DenormaliseBands)
                int lm = rng.Next(4), bigM = 1 << lm;
                int start = rng.Next(4) == 0 ? 17 : 0;
                int end = new[] { 13, 17, 19, 21 }[rng.Next(4)]; if (end <= start) end = nb;
                int downsample = downsamples[rng.Next(downsamples.Length)], silence = rng.Next(8) == 0 ? 1 : 0;
                var x = new int[120 * bigM]; for (int i = 0; i < x.Length; i++) x[i] = rng.Next(-32768, 32768);
                var e = new int[nb];
                for (int i = 0; i < nb; i++) e[i] = rng.Next(-20000, 17408 - (CeltModeTables.EMeans[i] << 6));
                cases[c] = new CeltSynthesisCase { LM = lm, Start = start, End = end, Downsample = downsample, Silence = silence };
                int o = c * B;
                Array.Copy(x, 0, blocks, o + CeltSynthesisGpuTestKernel.XAt, x.Length);
                Array.Copy(e, 0, blocks, o + CeltSynthesisGpuTestKernel.EAt, nb);
                var freq = new int[120 * bigM]; for (int i = 0; i < freq.Length; i++) freq[i] = 0x5A5A5A5;
                for (int i = 0; i < freq.Length; i++) blocks[o + CeltSynthesisGpuTestKernel.FreqAt + i] = freq[i];
                ConcentusCeltOracle.DenormaliseBands(ConcentusCeltOracle.Mode, x, freq, 0, e, 0, start, end, bigM, downsample, silence);
                want[c] = freq;
            }
            var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
            using var _ = modeBuffer;
            using var dCases = acc.Allocate1D(cases);
            using var dBlocks = acc.Allocate1D(blocks);
            using var dPcm = acc.Allocate1D<short>(count * CeltSynthesisGpuTestKernel.PcmShorts);
            using var kernel = new CeltSynthesisGpuTestKernel(acc);
            kernel.Run(CeltSynthesisGpuTestKernel.Op.Denormalise, dCases.View, mode, dBlocks.View, dPcm.View);
            await acc.SynchronizeAsync();
            var got = await dBlocks.CopyToHostAsync();
            for (int c = 0; c < count; c++)
            {
                int o = c * B + CeltSynthesisGpuTestKernel.FreqAt;
                var k = cases[c];
                for (int i = 0; i < want[c].Length; i++)
                    if (got[o + i] != want[c][i])
                        throw new Exception($"denormalise_bands LM={k.LM} bands {k.Start}..{k.End} downsample={k.Downsample} silence={k.Silence}: " +
                            $"freq[{i}] {got[o + i]}, Concentus {want[c][i]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task CeltSynthesisGpu_Deemphasis_BitExactVsConcentus()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int count = 120, B = CeltSynthesisGpuTestKernel.BlockInts, P = CeltSynthesisGpuTestKernel.PcmShorts;
            var rng = new Random(0xC342);
            var cases = new CeltSynthesisCase[count]; var blocks = new int[count * B]; var pcm = new short[count * P];
            var want = new (short[] Pcm, int[] Mem)[count];
            int[] downsamples = { 1, 1, 1, 2, 3, 4, 6 };
            for (int c = 0; c < count; c++)
            {
                // mono / stereo (the stereo fast path: downsample 1, no accum), every frame size and decimation, accumulating into
                // the PCM there (hybrid) or not; the real coefficient (0.85 in Q15) or a random one up to 0.9; a signal of up to
                // 2^26 (Q12, so samples saturate to int16 too) and filter memory of up to 2^27: with a gain of at most 1 / (1 - 0.9)
                // the sums stay inside int32 (an overflow wraps in C# and is undefined in OpenCL C)
                int lm = rng.Next(4), n = 120 << lm, C = 1 + rng.Next(2);
                int downsample = downsamples[rng.Next(downsamples.Length)], accum = rng.Next(3) == 0 ? 1 : 0;
                int coef0 = rng.Next(2) == 0 ? 27853 : rng.Next(0, 29492);
                var input = new int[2][];
                for (int ch = 0; ch < 2; ch++)
                {
                    input[ch] = new int[n];
                    int amp = rng.Next(2) == 0 ? 1 << 26 : 1 << 16;
                    for (int i = 0; i < n; i++) input[ch][i] = rng.Next(-amp, amp);
                }
                var mem = new[] { rng.Next(-(1 << 27), 1 << 27), rng.Next(-(1 << 27), 1 << 27) };
                var outPcm = new short[P]; for (int i = 0; i < P; i++) outPcm[i] = (short)rng.Next(-32768, 32768);
                cases[c] = new CeltSynthesisCase { LM = lm, C = C, Downsample = downsample, Accum = accum, Coef0 = coef0 };
                int o = c * B;
                for (int ch = 0; ch < 2; ch++)
                    Array.Copy(input[ch], 0, blocks, o + CeltSynthesisGpuTestKernel.InAt + ch * CeltSynthesisGpuTestKernel.InStride, n);
                blocks[o + CeltSynthesisGpuTestKernel.MemAt] = mem[0];
                blocks[o + CeltSynthesisGpuTestKernel.MemAt + 1] = mem[1];
                Array.Copy(outPcm, 0, pcm, c * P, P);
                ConcentusCeltOracle.Deemphasis(input, new int[2], outPcm, 0, n, C, downsample, new[] { coef0, 0, 0, 0 }, mem, accum);
                want[c] = (outPcm, mem);
            }
            var (mode, modeBuffer) = CeltModeGpu.Upload(acc);
            using var _ = modeBuffer;
            using var dCases = acc.Allocate1D(cases);
            using var dBlocks = acc.Allocate1D(blocks);
            using var dPcm = acc.Allocate1D(pcm);
            using var kernel = new CeltSynthesisGpuTestKernel(acc);
            kernel.Run(CeltSynthesisGpuTestKernel.Op.Deemphasis, dCases.View, mode, dBlocks.View, dPcm.View);
            await acc.SynchronizeAsync();
            var gotPcm = await dPcm.CopyToHostAsync();
            var gotBlocks = await dBlocks.CopyToHostAsync();
            for (int c = 0; c < count; c++)
            {
                var k = cases[c];
                string where = $"C={k.C} N={120 << k.LM} downsample={k.Downsample} accum={k.Accum} coef={k.Coef0}";
                for (int i = 0; i < P; i++)
                    if (gotPcm[c * P + i] != want[c].Pcm[i])
                        throw new Exception($"deemphasis {where}: pcm[{i}] {gotPcm[c * P + i]}, Concentus {want[c].Pcm[i]}");
                for (int ch = 0; ch < k.C; ch++)
                    if (gotBlocks[c * B + CeltSynthesisGpuTestKernel.MemAt + ch] != want[c].Mem[ch])
                        throw new Exception($"deemphasis {where}: mem[{ch}] {gotBlocks[c * B + CeltSynthesisGpuTestKernel.MemAt + ch]}, Concentus {want[c].Mem[ch]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
