// Tests for Vp9InterPred (SpawnDev.Codecs/Video/Vp9/Decoder/Vp9InterPredict.cs, port of libvpx v1.17.0's VP9 inter
// prediction: vpx_convolve.c, vp9_filter.c, vp9_scale.c, vp9_reconinter.c MV helpers, vp9_decodeframe.c
// dec_build_inter_predictors_sb).
//
// Fixture (scripts/video/libvpx_ref/make_inter_fixture.py + inter_ref.c, TestData/Vp9Dsp/inter.gz): libvpx's own
// dec_build_inter_predictors_sb (copied verbatim from vp9_decodeframe.c, linked against the compiled libvpx) on
// pseudo-random blocks: every BLOCK_SIZE incl. 4x4 / 4x8 / 8x4 with four sub-block MVs, every interpolation filter,
// single and compound, MVs from zero to far outside the frame, frame-edge blocks at legal partition positions, crop sizes
// that are and are not multiples of 8, 4:2:0 / 4:4:4 / 4:2:2 / 4:4:0, references of the same size and scaled (2x, 1.5x,
// 0.75x, one axis, 16x upscale). The reference pixels are regenerated here from (seed, kind) with inter_ref.c's LCG and
// stored WITHOUT borders (garbage right of / below the crop rectangle and between planes); libvpx's outputs were
// recorded with its border extended, and the harness proved that its visible pixels are the same with a garbage border
// (what the decoder really has).
//
// What must hold: every predicted pixel of every plane of every block equals libvpx's (integer arithmetic: exact on every
// backend), the bytes of the destination outside the block are untouched, and SetupScaleFactors equals
// vp9_setup_scale_factors_for_frame. Both paths are checked: the whole block in one call (PredictBlockPlane) and every
// 4x4 unit on its own (PredictBlockPlaneRegion, the decoder's InterPredictKernel path).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Vp9DspInterFixture = "SpawnDev.Codecs.Demo.Shared.TestData.Vp9Dsp.inter.gz";

    private sealed record Vp9InterScenario(int SsX, int SsY, int CurW, int CurH, int[] W, int[] H, int[] Seed, int[] Kind,
        Vp9ScaleFactors[] Sf);

    private sealed record Vp9InterCase(int Scenario, int MiRow, int MiCol, int BSize, int Filter, bool Compound, int[] Mv,
        int[] Bmi0, int[] Bmi1, byte[][] Want);

    // case record of the test kernel (ints)
    private const int InterCaseInts = 64, InterCaseRefs = 6, InterCaseDst = 54, InterDstStride = 64, InterDstPlane = 64 * 64;

    private static int Vp9InterN8W(int bsize) => new[] { 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8 }[bsize];
    private static int Vp9InterN8H(int bsize) => new[] { 1, 1, 1, 1, 2, 1, 2, 4, 2, 4, 8, 4, 8 }[bsize];

    private static (List<Vp9InterScenario>, List<Vp9InterCase>) Vp9InterFixture()
    {
        var raw = Gunzip(ReadEmbedded(Vp9DspInterFixture));
        int p = 0;
        int I() { int v = BitConverter.ToInt32(raw, p); p += 4; return v; }
        if (raw[0] != 'V' || raw[1] != 'P' || raw[2] != '9' || raw[3] != 'I') throw new Exception("inter.gz: bad magic");
        p = 4;
        if (I() != 1) throw new Exception("inter.gz: unknown version");
        int ns = I(), nc = I();
        var scen = new List<Vp9InterScenario>();
        for (int s = 0; s < ns; s++)
        {
            int ssx = I(), ssy = I(), cw = I(), ch = I();
            var w = new int[2]; var h = new int[2]; var seed = new int[2]; var kind = new int[2]; var sf = new Vp9ScaleFactors[2];
            for (int r = 0; r < 2; r++)
            {
                w[r] = I(); h[r] = I(); seed[r] = I(); kind[r] = I();
                sf[r].XScaleFp = I(); sf[r].YScaleFp = I(); sf[r].XStepQ4 = I(); sf[r].YStepQ4 = I();
            }
            scen.Add(new Vp9InterScenario(ssx, ssy, cw, ch, w, h, seed, kind, sf));
        }
        var cases = new List<Vp9InterCase>();
        for (int c = 0; c < nc; c++)
        {
            int si = I(), miRow = I(), miCol = I(), bsize = I(), filter = I(), compound = I();
            int[] mv = { I(), I() };
            int[] b0 = { I(), I(), I(), I() }, b1 = { I(), I(), I(), I() };
            var s = scen[si];
            var want = new byte[3][];
            for (int plane = 0; plane < 3; plane++)
            {
                int ssx = plane == 0 ? 0 : s.SsX, ssy = plane == 0 ? 0 : s.SsY;
                int n = ((Vp9InterN8W(bsize) << 1) >> ssx) * 4 * (((Vp9InterN8H(bsize) << 1) >> ssy) * 4);
                want[plane] = raw.AsSpan(p, n).ToArray();
                p += n;
            }
            cases.Add(new Vp9InterCase(si, miRow, miCol, bsize, filter, compound != 0, mv, b0, b1, want));
        }
        if (p != raw.Length) throw new Exception($"inter.gz: {raw.Length - p} trailing bytes");
        return (scen, cases);
    }

    /// <summary>inter_ref.c fill_plane: the crop rectangle of one plane from the running LCG.</summary>
    private static void Vp9InterFillPlane(byte[] buf, int at, int stride, int cw, int ch, int kind, ref uint rng)
    {
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++)
            {
                rng = rng * 1664525u + 1013904223u;
                int r = (int)(rng >> 24), v;
                if (kind == 0) v = r;
                else if (kind == 1)
                {
                    v = (x * 255 / (cw > 1 ? cw - 1 : 1) + y * 255 / (ch > 1 ? ch - 1 : 1)) / 2 + (r & 31) - 16;
                    v = v < 0 ? 0 : v > 255 ? 255 : v;
                }
                else v = (r & 1) != 0 ? 255 : 0;
                buf[at + y * stride + x] = (byte)v;
            }
    }

    private static void Vp9InterPredTestKernel(Index1D i, ArrayView<int> cases, ArrayView<int> mi, ArrayView<byte> refs,
        ArrayView<byte> dst, ArrayView<int> scratch)
    {
        long c = (long)i.X * InterCaseInts;
        for (int plane = 0; plane < 3; plane++)
        {
            int ssX = plane == 0 ? 0 : cases[c], ssY = plane == 0 ? 0 : cases[c + 1];
            long a0 = c + InterCaseRefs + plane * 8, a1 = c + InterCaseRefs + (3 + plane) * 8;
            Vp9InterRefPlane r0, r1;
            r0.Offset = cases[a0]; r0.Stride = cases[a0 + 1]; r0.CropWidth = cases[a0 + 2]; r0.CropHeight = cases[a0 + 3];
            r0.XScaleFp = cases[a0 + 4]; r0.YScaleFp = cases[a0 + 5]; r0.XStepQ4 = cases[a0 + 6]; r0.YStepQ4 = cases[a0 + 7];
            r1.Offset = cases[a1]; r1.Stride = cases[a1 + 1]; r1.CropWidth = cases[a1 + 2]; r1.CropHeight = cases[a1 + 3];
            r1.XScaleFp = cases[a1 + 4]; r1.YScaleFp = cases[a1 + 5]; r1.XStepQ4 = cases[a1 + 6]; r1.YStepQ4 = cases[a1 + 7];
            Vp9InterPred.PredictBlockPlane(ssX, ssY, cases[c + 2], cases[c + 3], cases[c + 4], cases[c + 5],
                mi, (long)i.X * Vp9Mi.Stride, refs, r0, refs, r1,
                dst, cases[c + InterCaseDst + plane], InterDstStride, scratch, (long)i.X * Vp9InterPred.ScratchInts);
        }
    }

    /// <summary>The decoder's per-unit path: thread (case, plane, unit row) predicts each 4x4 unit of its row of the plane
    /// block on its own with <see cref="Vp9InterPred.PredictBlockPlaneRegion"/>, as Vp9ReconKernels.InterPredictKernel does.</summary>
    private static void Vp9InterPredUnitTestKernel(Index1D i, ArrayView<int> cases, ArrayView<int> mi, ArrayView<byte> refs,
        ArrayView<byte> dst, ArrayView<int> scratch)
    {
        int t = i, caseIdx = t / 48, plane = (t % 48) / 16, uy = t % 16;
        long c = (long)caseIdx * InterCaseInts;
        int ssX = plane == 0 ? 0 : cases[c], ssY = plane == 0 ? 0 : cases[c + 1];
        long miAt = (long)caseIdx * Vp9Mi.Stride;
        int bsize = mi[miAt + Vp9Mi.SbType];
        int n4w = (Vp9InterPred.Num8x8Wide(bsize) << 1) >> ssX, n4h = (Vp9InterPred.Num8x8High(bsize) << 1) >> ssY;
        if (uy >= n4h) return;
        long a0 = c + InterCaseRefs + plane * 8, a1 = c + InterCaseRefs + (3 + plane) * 8;
        Vp9InterRefPlane r0, r1;
        r0.Offset = cases[a0]; r0.Stride = cases[a0 + 1]; r0.CropWidth = cases[a0 + 2]; r0.CropHeight = cases[a0 + 3];
        r0.XScaleFp = cases[a0 + 4]; r0.YScaleFp = cases[a0 + 5]; r0.XStepQ4 = cases[a0 + 6]; r0.YStepQ4 = cases[a0 + 7];
        r1.Offset = cases[a1]; r1.Stride = cases[a1 + 1]; r1.CropWidth = cases[a1 + 2]; r1.CropHeight = cases[a1 + 3];
        r1.XScaleFp = cases[a1 + 4]; r1.YScaleFp = cases[a1 + 5]; r1.XStepQ4 = cases[a1 + 6]; r1.YStepQ4 = cases[a1 + 7];
        for (int ux = 0; ux < n4w; ux++)
            Vp9InterPred.PredictBlockPlaneRegion(ssX, ssY, cases[c + 2], cases[c + 3], cases[c + 4], cases[c + 5],
                mi, miAt, refs, r0, refs, r1, dst, cases[c + InterCaseDst + plane], InterDstStride, ux * 4, uy * 4, 4, 4,
                scratch, (long)t * Vp9InterPred.RegionScratchInts);
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Vp9InterPred_AllCases_MatchLibvpx()
    {
        var (scen, cases) = Vp9InterFixture();
        var failures = new List<string>();

        // the scale-factor helper against vp9_setup_scale_factors_for_frame
        foreach (var s in scen)
            for (int r = 0; r < 2; r++)
            {
                var sf = Vp9InterPred.SetupScaleFactors(s.W[r], s.H[r], s.CurW, s.CurH);
                var want = s.Sf[r];
                if (sf.XScaleFp != want.XScaleFp || sf.YScaleFp != want.YScaleFp || sf.XStepQ4 != want.XStepQ4 || sf.YStepQ4 != want.YStepQ4)
                    failures.Add($"scale factors {s.W[r]}x{s.H[r]} -> {s.CurW}x{s.CurH}: ours {sf.XScaleFp}/{sf.YScaleFp}/{sf.XStepQ4}/{sf.YStepQ4}, " +
                        $"libvpx {want.XScaleFp}/{want.YScaleFp}/{want.XStepQ4}/{want.YStepQ4}");
            }

        // reference planes without borders: garbage everywhere outside the crop rectangles
        const int guard = 256;
        var planeAt = new int[scen.Count, 2, 3];
        var planeStride = new int[scen.Count, 2, 3];
        var planeCrop = new (int W, int H)[scen.Count, 2, 3];
        int total = 0;
        for (int si = 0; si < scen.Count; si++)
            for (int r = 0; r < 2; r++)
                for (int plane = 0; plane < 3; plane++)
                {
                    var s = scen[si];
                    int ssx = plane == 0 ? 0 : s.SsX, ssy = plane == 0 ? 0 : s.SsY;
                    int alignedW = (s.W[r] + 7) & ~7, alignedH = (s.H[r] + 7) & ~7;
                    planeStride[si, r, plane] = alignedW >> ssx;
                    planeCrop[si, r, plane] = ((s.W[r] + ssx) >> ssx, (s.H[r] + ssy) >> ssy);
                    planeAt[si, r, plane] = total + guard;
                    total += guard + (alignedW >> ssx) * (alignedH >> ssy);
                }
        total += guard;
        var refs = new byte[total];
        for (int k = 0; k < total; k++) refs[k] = (byte)(k * 167 + 91);
        for (int si = 0; si < scen.Count; si++)
            for (int r = 0; r < 2; r++)
            {
                uint rng = (uint)scen[si].Seed[r];
                for (int plane = 0; plane < 3; plane++)
                    Vp9InterFillPlane(refs, planeAt[si, r, plane], planeStride[si, r, plane], planeCrop[si, r, plane].W,
                        planeCrop[si, r, plane].H, scen[si].Kind[r], ref rng);
            }

        int n = cases.Count;
        var caseRec = new int[n * InterCaseInts];
        var miRec = new int[n * Vp9Mi.Stride];
        for (int i = 0; i < n; i++)
        {
            var c = cases[i];
            var s = scen[c.Scenario];
            int b = i * InterCaseInts;
            caseRec[b] = s.SsX; caseRec[b + 1] = s.SsY; caseRec[b + 2] = c.MiRow; caseRec[b + 3] = c.MiCol;
            caseRec[b + 4] = (s.CurH + 7) >> 3; caseRec[b + 5] = (s.CurW + 7) >> 3;
            for (int r = 0; r < 2; r++)
            {
                var sf = Vp9InterPred.SetupScaleFactors(s.W[r], s.H[r], s.CurW, s.CurH);
                for (int plane = 0; plane < 3; plane++)
                {
                    int a = b + InterCaseRefs + (r * 3 + plane) * 8;
                    caseRec[a] = planeAt[c.Scenario, r, plane];
                    caseRec[a + 1] = planeStride[c.Scenario, r, plane];
                    caseRec[a + 2] = planeCrop[c.Scenario, r, plane].W;
                    caseRec[a + 3] = planeCrop[c.Scenario, r, plane].H;
                    caseRec[a + 4] = sf.XScaleFp; caseRec[a + 5] = sf.YScaleFp; caseRec[a + 6] = sf.XStepQ4; caseRec[a + 7] = sf.YStepQ4;
                }
            }
            for (int plane = 0; plane < 3; plane++) caseRec[b + InterCaseDst + plane] = (i * 3 + plane) * InterDstPlane;
            int m = i * Vp9Mi.Stride;
            miRec[m + Vp9Mi.SbType] = c.BSize;
            miRec[m + Vp9Mi.Interp] = c.Filter;
            miRec[m + Vp9Mi.Ref0] = Vp9Mi.LastFrame;
            miRec[m + Vp9Mi.Ref1] = c.Compound ? Vp9Mi.GoldenFrame : Vp9Mi.NoRefFrame;
            miRec[m + Vp9Mi.Mv0] = c.Mv[0];
            miRec[m + Vp9Mi.Mv1] = c.Mv[1];
            for (int k = 0; k < 4; k++)
            {
                miRec[m + Vp9Mi.BmiMv0 + k] = c.Bmi0[k];
                miRec[m + Vp9Mi.BmiMv1 + k] = c.Bmi1[k];
            }
        }
        var dstInit = new byte[n * 3 * InterDstPlane];
        for (int k = 0; k < dstInit.Length; k++) dstInit[k] = (byte)(k * 37 + 11);

        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dCases = acc.Allocate1D(caseRec);
            using var dMi = acc.Allocate1D(miRec);
            using var dRefs = acc.Allocate1D(refs);
            using var dDst = acc.Allocate1D(dstInit);
            using var dScratch = acc.Allocate1D<int>((long)n * Vp9InterPred.ScratchInts);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>,
                ArrayView<int>>(Vp9InterPredTestKernel);
            kernel(new Index1D(n), dCases.View, dMi.View, dRefs.View, dDst.View, dScratch.View);
            await acc.SynchronizeAsync();
            var gotBlock = await dDst.CopyToHostAsync();
            // the same blocks again, each 4x4 unit predicted on its own (the decoder's InterPredictKernel path)
            using var dDstUnits = acc.Allocate1D(dstInit);
            using var dScratchUnits = acc.Allocate1D<int>((long)n * 48 * Vp9InterPred.RegionScratchInts);
            var unitKernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>,
                ArrayView<int>>(Vp9InterPredUnitTestKernel);
            unitKernel(new Index1D(n * 48), dCases.View, dMi.View, dRefs.View, dDstUnits.View, dScratchUnits.View);
            await acc.SynchronizeAsync();
            var gotUnits = await dDstUnits.CopyToHostAsync();

            foreach (var (path, got) in new[] { ("whole block", gotBlock), ("per 4x4 unit", gotUnits) })
            {
                int badCases = 0, pixels = 0, scaled = 0, sub8 = 0, compound = 0;
                var perFilter = new int[4];
                for (int i = 0; i < n; i++)
                {
                    var c = cases[i];
                    var s = scen[c.Scenario];
                    bool isScaled = Vp9InterPred.IsScaled(s.Sf[0].XScaleFp, s.Sf[0].YScaleFp) ||
                                    (c.Compound && Vp9InterPred.IsScaled(s.Sf[1].XScaleFp, s.Sf[1].YScaleFp));
                    if (isScaled) scaled++;
                    if (c.BSize < Vp9InterPred.Block8x8) sub8++;
                    if (c.Compound) compound++;
                    perFilter[c.Filter]++;
                    string? first = null;
                    int bad = 0;
                    for (int plane = 0; plane < 3; plane++)
                    {
                        int ssx = plane == 0 ? 0 : s.SsX, ssy = plane == 0 ? 0 : s.SsY;
                        int w = ((Vp9InterN8W(c.BSize) << 1) >> ssx) * 4, h = ((Vp9InterN8H(c.BSize) << 1) >> ssy) * 4;
                        int at = (i * 3 + plane) * InterDstPlane;
                        for (int y = 0; y < InterDstStride; y++)
                            for (int x = 0; x < InterDstStride; x++)
                            {
                                int k = at + y * InterDstStride + x;
                                bool inBlock = x < w && y < h;
                                byte want = inBlock ? c.Want[plane][y * w + x] : dstInit[k];
                                if (inBlock) pixels++;
                                if (got[k] == want) continue;
                                bad++;
                                first ??= $"plane {plane} ({x},{y}) got {got[k]} want {want}" + (inBlock ? "" : " (outside the block: must be untouched)");
                            }
                    }
                    if (bad == 0) continue;
                    badCases++;
                    if (failures.Count < 12)
                        failures.Add($"{path}: case {i}: scenario {c.Scenario} (ss {s.SsX},{s.SsY} cur {s.CurW}x{s.CurH} ref0 {s.W[0]}x{s.H[0]} " +
                            $"ref1 {s.W[1]}x{s.H[1]}) bsize {c.BSize} mi ({c.MiRow},{c.MiCol}) filter {c.Filter} compound {c.Compound} " +
                            $"mv {Vp9InterMvText(c.Mv[0])} {Vp9InterMvText(c.Mv[1])} bmi0 {string.Join(" ", c.Bmi0.Select(Vp9InterMvText))} " +
                            $"bmi1 {string.Join(" ", c.Bmi1.Select(Vp9InterMvText))}: {bad} bytes differ, first {first}");
                }
                Console.WriteLine($"{path}: {n} blocks ({scen.Count} frame setups, {pixels} predicted pixels): {scaled} with a scaled reference, " +
                    $"{sub8} below 8x8, {compound} compound; filters {string.Join("/", perFilter)}; {badCases} blocks differ from libvpx");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static string Vp9InterMvText(int mv) => $"({(short)(mv & 0xFFFF)},{mv >> 16})";
}
