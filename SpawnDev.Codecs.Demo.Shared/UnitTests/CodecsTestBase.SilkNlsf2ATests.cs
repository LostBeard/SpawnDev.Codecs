using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkNlsf2A"/> - converting normalized LSFs to Q12 LPC coefficients
/// with stability guarantee via iterative bandwidth expansion.
///
/// The exit contract is simple and strong: for any valid NLSF input,
/// <see cref="SilkLpcInvPredGain.Compute"/> on the returned coefficients must be non-zero
/// (i.e. the filter is stable). We verify this property across many synthesized NLSF
/// fixtures for both supported orders (10 and 16).
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Validation --------

    [TestMethod]
    public void Nlsf2A_InvalidOrder_Throws()
    {
        var nlsf = new short[12];
        var aQ12 = new short[12];
        Throws<ArgumentException>(() => SilkNlsf2A.Compute(aQ12, nlsf, 12));
    }

    [TestMethod]
    public void Nlsf2A_OutputTooSmall_Throws()
    {
        var nlsf = new short[10];
        var aQ12 = new short[9];
        Throws<ArgumentException>(() => SilkNlsf2A.Compute(aQ12, nlsf, 10));
    }

    [TestMethod]
    public void Nlsf2A_InputTooSmall_Throws()
    {
        var nlsf = new short[9];
        var aQ12 = new short[10];
        Throws<ArgumentException>(() => SilkNlsf2A.Compute(aQ12, nlsf, 10));
    }

    // -------- Evenly spaced NLSFs (stability check) --------

    [TestMethod]
    public void Nlsf2A_EvenlySpacedOrder10_ProducesStableFilter()
    {
        // NLSF[k] = 32768 * (k+1) / (d+1) for k in [0, d).
        // This gives d distinct LSFs distributed over [0, pi] with uniform spacing,
        // which maps to a mild, clearly-stable filter.
        var nlsf = new short[10];
        for (int k = 0; k < 10; k++)
        {
            nlsf[k] = (short)(32768 * (k + 1) / 11);
        }

        var aQ12 = new short[10];
        SilkNlsf2A.Compute(aQ12, nlsf, 10);

        int invGain = SilkLpcInvPredGain.Compute(aQ12, 10);
        True(invGain > 0, $"Expected stable filter for evenly-spaced order-10 NLSFs; got invGain=0. A_Q12=[{string.Join(",", aQ12)}]");
        True(invGain <= (1 << 30), "invGain should be <= 2^30");
    }

    [TestMethod]
    public void Nlsf2A_EvenlySpacedOrder16_ProducesStableFilter()
    {
        var nlsf = new short[16];
        for (int k = 0; k < 16; k++)
        {
            nlsf[k] = (short)(32768 * (k + 1) / 17);
        }

        var aQ12 = new short[16];
        SilkNlsf2A.Compute(aQ12, nlsf, 16);

        int invGain = SilkLpcInvPredGain.Compute(aQ12, 16);
        True(invGain > 0, $"Expected stable filter for evenly-spaced order-16 NLSFs; got invGain=0. A_Q12=[{string.Join(",", aQ12)}]");
        True(invGain <= (1 << 30), "invGain should be <= 2^30");
    }

    // -------- Always-stable contract (generator test) --------

    [TestMethod]
    public void Nlsf2A_AlwaysProducesStableFilter_Order10()
    {
        // Generate 500 random sorted-ascending NLSF vectors and verify the output
        // filter is always stable. The stability loop MUST converge within
        // MAX_LPC_STABILIZE_ITERATIONS regardless of input.
        var rng = new Random(0xA1B2C3);
        var aQ12 = new short[10];
        var nlsf = new short[10];

        int stableCount = 0;
        for (int trial = 0; trial < 500; trial++)
        {
            // Generate strictly increasing NLSFs in [1, 32767]. Ensure minimum
            // separation so we exercise realistic decoder inputs.
            GenerateSortedNlsf(rng, nlsf, minSeparation: 32);

            SilkNlsf2A.Compute(aQ12, nlsf, 10);
            int invGain = SilkLpcInvPredGain.Compute(aQ12, 10);
            if (invGain == 0)
            {
                throw new Exception(
                    $"Trial {trial}: NLSF2A produced an unstable filter after the 16-iteration " +
                    $"bandwidth-expansion loop. NLSF=[{string.Join(",", nlsf)}], " +
                    $"A_Q12=[{string.Join(",", aQ12)}]");
            }
            stableCount++;
        }

        Equal(500, stableCount);
    }

    [TestMethod]
    public void Nlsf2A_AlwaysProducesStableFilter_Order16()
    {
        var rng = new Random(0x5A6B7C);
        var aQ12 = new short[16];
        var nlsf = new short[16];

        int stableCount = 0;
        for (int trial = 0; trial < 500; trial++)
        {
            GenerateSortedNlsf(rng, nlsf, minSeparation: 32);

            SilkNlsf2A.Compute(aQ12, nlsf, 16);
            int invGain = SilkLpcInvPredGain.Compute(aQ12, 16);
            if (invGain == 0)
            {
                throw new Exception(
                    $"Trial {trial}: NLSF2A produced an unstable filter. " +
                    $"NLSF=[{string.Join(",", nlsf)}], A_Q12=[{string.Join(",", aQ12)}]");
            }
            stableCount++;
        }

        Equal(500, stableCount);
    }

    // -------- Boundary NLSFs --------

    [TestMethod]
    public void Nlsf2A_MinimallySeparatedNlsfs_StillStable()
    {
        // Tightly packed NLSFs stress the bandwidth-expansion loop. Pack NLSFs
        // near the middle with min separation to exercise the stability iterations.
        var nlsf = new short[10];
        int start = 15000;
        for (int k = 0; k < 10; k++)
        {
            nlsf[k] = (short)(start + k * 64);
        }

        var aQ12 = new short[10];
        SilkNlsf2A.Compute(aQ12, nlsf, 10);

        int invGain = SilkLpcInvPredGain.Compute(aQ12, 10);
        True(invGain > 0, $"Expected stable filter after stabilization loop for tightly-packed NLSFs. A_Q12=[{string.Join(",", aQ12)}]");
    }

    [TestMethod]
    public void Nlsf2A_NlsfsSpanningFullRange_Stable()
    {
        // NLSFs near the extremes (small and large) produce filters with poles close to
        // unit circle in one direction. Stability loop must still converge.
        var nlsf = new short[10] { 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 32700 };
        var aQ12 = new short[10];
        SilkNlsf2A.Compute(aQ12, nlsf, 10);

        int invGain = SilkLpcInvPredGain.Compute(aQ12, 10);
        True(invGain > 0, $"Expected stable filter for full-range NLSFs. A_Q12=[{string.Join(",", aQ12)}]");
    }

    // -------- Helpers --------

    /// <summary>
    /// Generate strictly-increasing NLSFs in <c>[1, 32767]</c> with the given
    /// minimum separation between consecutive values. Matches the constraints
    /// that SILK's stabilization guarantees on decoder output.
    /// </summary>
    private static void GenerateSortedNlsf(Random rng, Span<short> nlsf, int minSeparation)
    {
        int d = nlsf.Length;
        int maxVal = 32767;

        int budget = maxVal - (d + 1) * minSeparation;
        if (budget < 0) budget = 0;

        int[] gaps = new int[d + 1];
        int gapSum = 0;
        for (int i = 0; i <= d; i++)
        {
            gaps[i] = rng.Next(0, 1000);
            gapSum += gaps[i];
        }

        double scale = gapSum > 0 ? budget / (double)gapSum : 0;
        int running = 0;
        for (int k = 0; k < d; k++)
        {
            running += minSeparation + (int)(gaps[k] * scale);
            if (running > maxVal) running = maxVal;
            nlsf[k] = (short)running;
        }
    }
}
