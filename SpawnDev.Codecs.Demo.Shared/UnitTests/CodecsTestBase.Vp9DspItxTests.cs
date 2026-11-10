// Tests for the VP9 inverse transforms (Vp9Itx, port of libvpx v1.17.0 vpx_dsp/inv_txfm.c + vp9/common/vp9_idct.c).
//
// Fixture (scripts/video/libvpx_ref/itx_ref.c + make_itx_fixture.py, TestData/Vp9Dsp/itx.gz): libvpx's own C (the
// functions its decoder calls, 8-bit build) on deterministic pseudo-random blocks: every tx size x tx type (32x32: DCT),
// the lossless WHT, eobs on both sides of every eob-variant threshold, coefficients only at the first eob positions of
// the scan the decoder uses, magnitudes from +-1 to the int16 extremes, random prediction pixels in a block with stride
// n + 3 (pad bytes included in the comparison: a write outside the block fails).
//
// What must hold: every byte of every case equals libvpx's, all cases in one kernel launch, one thread per case.

using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // per case in the parameter buffer: txSize, txType, lossless, eob, stride, coefAt, dstAt
    private const int Vp9ItxParams = 7;

    private static void Vp9ItxFixtureKernel(Index1D i, ArrayView<int> p, ArrayView<int> coef, ArrayView<byte> dst, ArrayView<int> scratch)
    {
        int b = i.X * Vp9ItxParams;
        Vp9Itx.InverseTransformAdd(coef, p[b + 5], p[b], p[b + 1], p[b + 2] != 0, p[b + 3], dst, p[b + 6], p[b + 4],
            scratch, i.X * Vp9Itx.ScratchInts);
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Vp9Itx_AllCases_MatchLibvpx()
    {
        byte[] raw;
        using (var gz = new GZipStream(new MemoryStream(ReadEmbedded("SpawnDev.Codecs.Demo.Shared.TestData.Vp9Dsp.itx.gz")), CompressionMode.Decompress))
        using (var ms = new MemoryStream())
        {
            gz.CopyTo(ms);
            raw = ms.ToArray();
        }
        var rd = new BinaryReader(new MemoryStream(raw));
        int count = rd.ReadInt32();
        var prm = new int[count * Vp9ItxParams];
        var coefs = new List<int>();
        var dstIn = new List<byte>();
        var want = new List<byte>();
        int unchanged = 0;
        var perKind = new SortedDictionary<string, int>();
        for (int c = 0; c < count; c++)
        {
            int tx = rd.ReadInt32(), type = rd.ReadInt32(), lossless = rd.ReadInt32(), eob = rd.ReadInt32(), stride = rd.ReadInt32();
            int n = 4 << tx;
            int b = c * Vp9ItxParams;
            prm[b] = tx; prm[b + 1] = type; prm[b + 2] = lossless; prm[b + 3] = eob; prm[b + 4] = stride;
            prm[b + 5] = coefs.Count; prm[b + 6] = dstIn.Count;
            for (int k = 0; k < n * n; k++) coefs.Add(rd.ReadInt16());
            byte[] din = rd.ReadBytes(n * stride), dout = rd.ReadBytes(n * stride);
            dstIn.AddRange(din);
            want.AddRange(dout);
            if (din.AsSpan().SequenceEqual(dout)) unchanged++;
            string kind = lossless != 0 ? "WHT 4x4" : $"{n}x{n} type {type}";
            perKind[kind] = perKind.GetValueOrDefault(kind) + 1;
        }
        True(rd.BaseStream.Position == raw.Length, $"fixture has {raw.Length - rd.BaseStream.Position} trailing bytes");
        // the fixture is not vacuous: libvpx changed the prediction in most cases (tiny coefficients may round to 0)
        True(count > 3000 && unchanged * 4 < count, $"{count} cases, {unchanged} leave the prediction unchanged");

        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dP = acc.Allocate1D(prm);
            using var dCoef = acc.Allocate1D(coefs.ToArray());
            using var dDst = acc.Allocate1D(dstIn.ToArray());
            using var dScratch = acc.Allocate1D<int>((long)count * Vp9Itx.ScratchInts);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(Vp9ItxFixtureKernel);
            kernel(new Index1D(count), dP.View, dCoef.View, dDst.View, dScratch.View);
            await acc.SynchronizeAsync();
            byte[] got = await dDst.CopyToHostAsync();

            var failures = new List<string>();
            int badCases = 0;
            for (int c = 0; c < count; c++)
            {
                int b = c * Vp9ItxParams, n = 4 << prm[b], stride = prm[b + 4], at = prm[b + 6];
                for (int k = 0; k < n * stride; k++)
                {
                    if (got[at + k] == want[at + k]) continue;
                    badCases++;
                    if (failures.Count < 12)
                        failures.Add($"case {c} ({(prm[b + 2] != 0 ? "lossless WHT" : $"{n}x{n} tx_type {prm[b + 1]}")}, eob {prm[b + 3]}): " +
                            $"row {k / stride} col {k % stride}{(k % stride >= n ? " (pad)" : "")} got {got[at + k]} want {want[at + k]} (prediction {dstIn[at + k]})");
                    break;
                }
            }
            Console.WriteLine($"{count} cases ({string.Join(", ", perKind.Select(kv => $"{kv.Key}: {kv.Value}"))}); " +
                $"{unchanged} unchanged by libvpx; {badCases} differ");
            True(badCases == 0, $"{badCases} of {count} cases differ from libvpx:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
