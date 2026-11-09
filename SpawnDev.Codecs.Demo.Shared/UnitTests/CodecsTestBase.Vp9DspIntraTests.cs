// Tests for the VP9 decoder's intra prediction (Vp9IntraPred, port of libvpx v1.17.0 vp9_reconintra.c +
// vpx_dsp/intrapred.c).
//
// Fixture (scripts/video/libvpx_ref/make_intra_fixture.py, TestData/Vp9Dsp/intra.gz): libvpx's own C
// vp9_predict_intra_block on 3840 decoder-shaped cases (every mode x transform size x plane, 4:2:0 / 4:2:2 / 4:4:0 /
// 4:4:4 chroma, random frame sizes with blocks at the right / bottom edges, above / left mi present or not, transform
// blocks inside larger prediction blocks), with every pixel it can read outside the block and its prediction.
//
// What must hold: each case predicts libvpx's pixels exactly (on every backend: integer only), and nothing outside the
// transform block is written.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // per case in the kernel's int buffer: the fixture's 16 header ints, then the window offset and stride
    private const int Vp9IntraCaseInts = 18;

    private static readonly string[] Vp9IntraModeNames = { "DC", "V", "H", "D45", "D135", "D117", "D153", "D207", "D63", "TM" };

    private static void Vp9IntraPredTestKernel(Index1D i, ArrayView<int> prm, ArrayView<byte> win, ArrayView<int> scratch)
    {
        long p = (long)i.X * Vp9IntraCaseInts;
        int stride = prm[p + 17];
        Vp9IntraPred.Predict(prm[p + 2], prm[p + 0], prm[p + 1], prm[p + 3], prm[p + 4], prm[p + 5] != 0, prm[p + 6] != 0,
            prm[p + 7], prm[p + 8], prm[p + 9], prm[p + 10], prm[p + 11], prm[p + 12], prm[p + 13], prm[p + 14],
            win, prm[p + 16] + stride + 1, stride, scratch, (long)i.X * Vp9IntraPred.ScratchInts);
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Vp9IntraPred_AllCases_MatchLibvpx()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            byte[] d = Gunzip(ReadEmbedded("SpawnDev.Codecs.Demo.Shared.TestData.Vp9Dsp.intra.gz"));
            True(BitConverter.ToInt32(d, 0) == 0x49395056 && BitConverter.ToInt32(d, 4) == 1, "intra.gz: bad magic / version");
            int n = BitConverter.ToInt32(d, 8);
            True(n > 0, "intra.gz: no cases");
            // Each case gets a window of the plane: row 0 = the row above the block from x0 - 1 (2 * bs + 1 pixels),
            // rows 1 .. bs = the left pixel then the block; the rest of the window holds a pattern that must survive.
            var prm = new int[n * Vp9IntraCaseInts];
            var wantOffsets = new int[n];
            int pos = 12, winSize = 0;
            for (int c = 0; c < n; c++)
            {
                for (int k = 0; k < 16; k++) prm[c * Vp9IntraCaseInts + k] = BitConverter.ToInt32(d, pos + 4 * k);
                pos += 64;
                int bs = 4 << prm[c * Vp9IntraCaseInts];
                prm[c * Vp9IntraCaseInts + 16] = winSize;
                prm[c * Vp9IntraCaseInts + 17] = 2 * bs + 1;
                winSize += (2 * bs + 1) * (bs + 1);
                wantOffsets[c] = pos;
                pos += 2 * bs + 1 + bs + bs * bs;
            }
            True(pos == d.Length, $"intra.gz: {d.Length - pos} trailing bytes");
            var win = new byte[winSize];
            for (int k = 0; k < winSize; k++) win[k] = (byte)(k * 37 + 11);
            for (int c = 0; c < n; c++)
            {
                int bs = 4 << prm[c * Vp9IntraCaseInts], at = prm[c * Vp9IntraCaseInts + 16], stride = 2 * bs + 1, src = wantOffsets[c];
                for (int k = 0; k < 2 * bs + 1; k++) win[at + k] = d[src + k];
                for (int r = 0; r < bs; r++) win[at + (r + 1) * stride] = d[src + 2 * bs + 1 + r];
            }
            var before = (byte[])win.Clone();

            using var dPrm = acc.Allocate1D(prm);
            using var dWin = acc.Allocate1D(win);
            using var dScratch = acc.Allocate1D<int>((long)n * Vp9IntraPred.ScratchInts);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(Vp9IntraPredTestKernel);
            kernel(new Index1D(n), dPrm.View, dWin.View, dScratch.View);
            await acc.SynchronizeAsync();
            byte[] got = await dWin.CopyToHostAsync();

            var failures = new List<string>();
            int badCases = 0;
            var perMode = new int[10];
            for (int c = 0; c < n; c++)
            {
                int q = c * Vp9IntraCaseInts;
                int tx = prm[q], mode = prm[q + 1], bs = 4 << tx, at = prm[q + 16], stride = 2 * bs + 1;
                int want = wantOffsets[c] + 2 * bs + 1 + bs;
                string first = "";
                int bad = 0;
                for (int r = 0; r <= bs; r++)
                    for (int x = 0; x < stride; x++)
                    {
                        int idx = at + r * stride + x;
                        bool inBlock = r >= 1 && x >= 1 && x <= bs;
                        int expect = inBlock ? d[want + (r - 1) * bs + (x - 1)] : before[idx];
                        if (got[idx] == expect) continue;
                        if (bad++ == 0)
                            first = inBlock ? $"pixel ({x - 1},{r - 1}) = {got[idx]}, libvpx {expect}"
                                : $"wrote outside the block at window ({x - 1},{r - 1}): {got[idx]}, was {expect}";
                    }
                if (bad == 0) continue;
                badCases++;
                perMode[mode]++;
                if (failures.Count < 12)
                    failures.Add($"case {c}: {Vp9IntraModeNames[mode]} {bs}x{bs} plane {prm[q + 15]} bwl {prm[q + 2]} aoff {prm[q + 3]} loff {prm[q + 4]} " +
                        $"aboveMi {prm[q + 5]} leftMi {prm[q + 6]} mbEdges L{prm[q + 7]} T{prm[q + 8]} R{prm[q + 9]} B{prm[q + 10]} " +
                        $"ss {prm[q + 11]},{prm[q + 12]} frame {prm[q + 13]}x{prm[q + 14]}: {bad} pixels differ, first {first}");
            }
            Console.WriteLine($"{n} cases vs libvpx vp9_predict_intra_block: {n - badCases} exact, {badCases} differ" +
                (badCases > 0 ? " (per mode: " + string.Join(", ", Enumerable.Range(0, 10).Select(m => $"{Vp9IntraModeNames[m]} {perMode[m]}")) + ")" : ""));
            True(badCases == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
