// TransformRounding (the 32-bit form of the transform butterflies' rounded products) against the 64-bit formula it
// replaces in the GPU transforms.
//
// The 32-bit form is exact when |w0| + |w1| <= 32767 and 1 <= bit <= 16. This test:
//  - takes the weights from the real tables: every AV1 cosine (cos_bit 10..13, all 64 entries, both signs, paired
//    with its rotation partner, with cospi[32] and with itself) and every VP9 cosine (cospi_k_64, k = 1..31, paired
//    with cospi_(32-k)_64 as libvpx's rotations do, with cospi_16_64, and alone), and FAILS if a pair breaks the bound;
//  - compares RoundedDot / RoundedProduct with the 64-bit formula on the inputs that stress the split (the int
//    limits, values around +-65536 and +-32768, all-ones low halves) and on random full-range ints.
// What it can fail on: a wrong shift or mask in the 32-bit form (every pair fails), a weight pair over the bound (the
// bound check, and with it the int-limit inputs), an intermediate that overflows 32 bits (the int-limit inputs).
// It runs on the host only (no accelerator): the kernels are checked against the CPU reference transforms, which keep
// the 64-bit formula, by the Gpu-vs-Cpu transform tests and the encoder byte comparisons.

using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "CpuOnly")]
    public Task TransformRounding_EqualsThe64BitFormula()
    {
        CpuOnlyTest();

        static int Dot64(int w0, int a, int w1, int b, int bit) => (int)(((long)w0 * a + (long)w1 * b + (1L << (bit - 1))) >> bit);
        static int Product64(int w, int x, int bit) => (int)(((long)w * x + (1L << (bit - 1))) >> bit);

        var inputs = new List<int>
        {
            int.MinValue, int.MinValue + 1, int.MaxValue, int.MaxValue - 1, 0, 1, -1, 2, -2,
            65535, 65536, 65537, -65535, -65536, -65537, 32767, 32768, 32769, -32767, -32768, -32769,
            0x7FFF0000, 0x7FFF0001, unchecked((int)0x80000000), unchecked((int)0x8000FFFF), unchecked((int)0xFFFF0000),
            0x0000FFFF, 0x00010000, 0x7FFFFFFF, 255, -255, 1020, -1020, 4080, -4080,
        };
        var rng = new Random(0x7A11);
        var random = new int[3000];
        for (int i = 0; i < random.Length; i++) random[i] = rng.Next(int.MinValue, int.MaxValue);

        // (w0, w1, bit, label)
        var pairs = new List<(int w0, int w1, int bit, string what)>();
        for (int bit = 10; bit <= 13; bit++)
        {
            int[] cospi = Av1ForwardDct4.CospiArrData[bit - 10];
            Equal(64, cospi.Length, $"AV1 cospi table of cos_bit {bit}");
            for (int i = 0; i < 64; i++)
                foreach (int j in new[] { i, 64 - i, 32, 16, 48 })
                {
                    if (j < 0 || j > 63) continue;
                    pairs.Add((cospi[i], cospi[j], bit, $"AV1 cos_bit {bit} cospi[{i}], cospi[{j}]"));
                    pairs.Add((-cospi[i], cospi[j], bit, $"AV1 cos_bit {bit} -cospi[{i}], cospi[{j}]"));
                    pairs.Add((cospi[i], -cospi[j], bit, $"AV1 cos_bit {bit} cospi[{i}], -cospi[{j}]"));
                    pairs.Add((-cospi[i], -cospi[j], bit, $"AV1 cos_bit {bit} -cospi[{i}], -cospi[{j}]"));
                }
        }
        // VP9: cospi_k_64 = round(16384 * cos(k * pi / 64)), DCT_CONST_BITS = 14 (two spot values pin the formula).
        var vp9 = new int[33];
        for (int k = 0; k <= 32; k++) vp9[k] = (int)Math.Round(16384.0 * Math.Cos(k * Math.PI / 64.0));
        Equal(16364, vp9[1], "VP9 cospi_1_64");
        Equal(11585, vp9[16], "VP9 cospi_16_64");
        Equal(804, vp9[31], "VP9 cospi_31_64");
        for (int k = 1; k <= 31; k++)
        {
            foreach (int j in new[] { 32 - k, 16 })
            {
                pairs.Add((vp9[k], vp9[j], 14, $"VP9 cospi_{k}_64, cospi_{j}_64"));
                pairs.Add((-vp9[k], vp9[j], 14, $"VP9 -cospi_{k}_64, cospi_{j}_64"));
                pairs.Add((vp9[k], -vp9[j], 14, $"VP9 cospi_{k}_64, -cospi_{j}_64"));
                pairs.Add((-vp9[k], -vp9[j], 14, $"VP9 -cospi_{k}_64, -cospi_{j}_64"));
            }
        }
        // VP9 ADST4 sines (sinpi_k_9 at 14 bits), each alone and the largest two together are over the bound on
        // purpose NOT listed as a pair: 13377 + 15212 = 28589 fits, 15212 + 15212 = 30424 fits.
        int[] sinpi = { 5283, 9929, 13377, 15212 };
        foreach (int a in sinpi)
            foreach (int b in sinpi)
                pairs.Add((a, -b, 14, $"VP9 sinpi {a}, -{b}"));
        // The stated limit itself.
        pairs.Add((16384, 16383, 16, "limit 16384 + 16383, bit 16"));
        pairs.Add((-32767, 0, 1, "limit -32767 alone, bit 1"));
        pairs.Add((32767, 0, 14, "limit 32767 alone, bit 14"));

        var failures = new List<string>();
        long compared = 0;
        int largestSum = 0;
        foreach (var (w0, w1, bit, what) in pairs)
        {
            int sum = Math.Abs(w0) + Math.Abs(w1);
            if (sum > largestSum) largestSum = sum;
            if (sum > TransformRounding.MaxWeightSum)
            {
                failures.Add($"{what}: |w0| + |w1| = {sum} is over TransformRounding.MaxWeightSum ({TransformRounding.MaxWeightSum})");
                continue;
            }
            // The intermediates of the 32-bit form at their extreme, in 64-bit arithmetic: they must fit an int.
            long worstLow = (long)sum * 65535 + (1L << (bit - 1)), worstHigh = (long)sum * 32768;
            if (worstLow > int.MaxValue || worstHigh > int.MaxValue)
                failures.Add($"{what}: an intermediate can reach {Math.Max(worstLow, worstHigh)}, over int.MaxValue");
            bool pairFailed = false;
            for (int i = 0; i < inputs.Count && !pairFailed; i++)
                for (int j = 0; j < inputs.Count; j++)
                {
                    compared++;
                    int want = Dot64(w0, inputs[i], w1, inputs[j], bit), got = TransformRounding.RoundedDot(w0, inputs[i], w1, inputs[j], bit);
                    if (want != got)
                    {
                        failures.Add($"{what}: RoundedDot({w0}, {inputs[i]}, {w1}, {inputs[j]}, {bit}) = {got}, 64-bit formula {want}");
                        pairFailed = true;
                        break;
                    }
                }
            for (int i = 0; i + 1 < random.Length && !pairFailed; i += 2)
            {
                compared++;
                int want = Dot64(w0, random[i], w1, random[i + 1], bit), got = TransformRounding.RoundedDot(w0, random[i], w1, random[i + 1], bit);
                if (want != got)
                {
                    failures.Add($"{what}: RoundedDot({w0}, {random[i]}, {w1}, {random[i + 1]}, {bit}) = {got}, 64-bit formula {want}");
                    pairFailed = true;
                }
            }
            for (int i = 0; i < inputs.Count && !pairFailed; i++)
            {
                compared++;
                int want = Product64(w0, inputs[i], bit), got = TransformRounding.RoundedProduct(w0, inputs[i], bit);
                if (want != got)
                {
                    failures.Add($"{what}: RoundedProduct({w0}, {inputs[i]}, {bit}) = {got}, 64-bit formula {want}");
                    pairFailed = true;
                }
            }
            if (failures.Count > 12) break;
        }
        Console.WriteLine($"TransformRounding: {pairs.Count} weight pairs (largest |w0| + |w1| = {largestSum}), {compared} comparisons with the 64-bit formula");
        True(failures.Count == 0, string.Join("\n", failures));
        True(compared > 5_000_000, $"only {compared} comparisons ran");
        return Task.CompletedTask;
    }
}
