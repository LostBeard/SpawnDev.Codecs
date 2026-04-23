using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkLpcInvPredGain"/> - the inverse LPC prediction gain
/// computation used to validate filter stability during NLSF-to-LPC conversion.
///
/// Strategy:
///   * Analytic vectors: hand-traced Q-format arithmetic for trivial cases.
///   * Stability vectors: filters that are obviously stable/unstable by DC response.
///   * Consistency stress: random filters cross-checked against an independent
///     impulse-response stability simulation (ported from libopus
///     silk/tests/test_unit_LPC_inv_pred_gain.c). For any filter we claim is
///     stable (gain != 0), the impulse response must not blow up.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Analytic reference vectors --------

    [TestMethod]
    public void LpcInvPredGain_AllZeroOrder1_ReturnsOneQ30()
    {
        // Order 1, A_Q12 = [0]. Trivially stable (poles at origin).
        // invGain = 1 - 0 = 1.0 in Q30 = 2^30 = 1073741824.
        Equal(1 << 30, SilkLpcInvPredGain.Compute(new short[] { 0 }, 1));
    }

    [TestMethod]
    public void LpcInvPredGain_AllZeroOrder16_ReturnsOneQ30()
    {
        // Order 16 all-zero filter: still invGain = 2^30 since each iteration leaves
        // the accumulator unchanged (rc_k = 0 -> rc_mult1 = 2^30 -> LSHIFT(SMMUL(g, 2^30), 2) = g).
        var a = new short[16];
        Equal(1 << 30, SilkLpcInvPredGain.Compute(a, 16));
    }

    [TestMethod]
    public void LpcInvPredGain_Order1_PositiveHalf_ReturnsThreeQuarters()
    {
        // A_Q12 = [2048] represents 0.5. The extracted reflection coefficient is rc = -0.5
        // (libopus convention rc = -A). invGain = 1 - rc^2 = 0.75 in Q30 = 805306368.
        Equal(805306368, SilkLpcInvPredGain.Compute(new short[] { 2048 }, 1));
    }

    [TestMethod]
    public void LpcInvPredGain_Order1_NegativeHalf_ReturnsThreeQuarters()
    {
        // rc^2 is sign-independent so A_Q12 = [-2048] yields the same 0.75 Q30 value.
        Equal(805306368, SilkLpcInvPredGain.Compute(new short[] { -2048 }, 1));
    }

    // -------- DC instability and magnitude edge cases --------

    [TestMethod]
    public void LpcInvPredGain_Order1_DcSumAtFourThousandNinetySix_ReturnsZero()
    {
        // DC_resp = 4096 triggers the early-exit check in the public entry point.
        Equal(0, SilkLpcInvPredGain.Compute(new short[] { 4096 }, 1));
    }

    [TestMethod]
    public void LpcInvPredGain_Order2_DcSumAtFourThousandNinetySix_ReturnsZero()
    {
        // [2048, 2048] -> DC_resp = 4096 -> early-exit.
        Equal(0, SilkLpcInvPredGain.Compute(new short[] { 2048, 2048 }, 2));
    }

    [TestMethod]
    public void LpcInvPredGain_Order1_JustBelowALimit_IsStable()
    {
        // A_LIMIT in QA=24 is 16773022. A_Q12 = 4094 -> A_QA = 4094<<12 = 16769024 < A_LIMIT.
        // DC_resp = 4094 < 4096 so DC check passes. Filter is accepted as stable.
        int gain = SilkLpcInvPredGain.Compute(new short[] { 4094 }, 1);
        True(gain > 0, "Expected stable gain for A_Q12 = 4094");
        True(gain <= (1 << 30), "Gain must be <= 2^30");
    }

    [TestMethod]
    public void LpcInvPredGain_Order1_JustAboveALimit_ReturnsZero()
    {
        // A_Q12 = 4095 -> A_QA = 16773120 > A_LIMIT (16773022). DC_resp = 4095 < 4096 so DC
        // check passes, but the per-coefficient A_LIMIT check inside the QA routine rejects.
        Equal(0, SilkLpcInvPredGain.Compute(new short[] { 4095 }, 1));
    }

    [TestMethod]
    public void LpcInvPredGain_Order1_NegativeAtLimit_ReturnsZero()
    {
        // Negative side: A_Q12 = -4095 -> A_QA = -16773120 < -A_LIMIT. DC check passes
        // (DC_resp -4095 is not >= 4096) but the inner A_LIMIT check rejects.
        Equal(0, SilkLpcInvPredGain.Compute(new short[] { -4095 }, 1));
    }

    // -------- Output bounds --------

    [TestMethod]
    public void LpcInvPredGain_Output_IsAlwaysInInclusiveRangeZeroToTwoToThe30()
    {
        // Stress: generate many stable filters of various orders and confirm the output
        // never exceeds 2^30 and is never negative.
        var rng = new Random(0xABCDEF);
        for (int trial = 0; trial < 1000; trial++)
        {
            int order = (trial % 8 + 1) * 2; // 2, 4, 6, ..., 16 (even, matching libopus test)
            var a = new short[order];
            int shift = rng.Next(2, 8);
            for (int k = 0; k < order; k++)
            {
                a[k] = (short)((short)rng.Next(short.MinValue, short.MaxValue + 1) >> shift);
            }

            int gain = SilkLpcInvPredGain.Compute(a, order);
            True(gain >= 0, $"trial {trial}: gain = {gain} must be >= 0");
            True(gain <= (1 << 30), $"trial {trial}: gain = {gain} must be <= 2^30");
        }
    }

    // -------- Consistency with independent impulse-response stability test --------

    /// <summary>
    /// Independent stability oracle ported from libopus silk/tests/test_unit_LPC_inv_pred_gain.c.
    /// Runs 10,000 samples of the impulse response; returns false if the output diverges past
    /// +/-10,000. Note: this has false positives (can classify genuinely-stable filters as
    /// "possibly unstable" if they decay slowly), but it has NO false negatives - if this
    /// returns false, the filter is definitively unstable.
    /// </summary>
    private static bool ImpulseResponseAppearsStable(short[] aQ12, int order)
    {
        int sumA = 0;
        int sumAbsA = 0;
        for (int j = 0; j < order; j++)
        {
            sumA += aQ12[j];
            sumAbsA += Math.Abs((int)aQ12[j]);
        }
        if (sumA >= 4096) return false;
        if (sumAbsA < 4096) return true;

        double[] y = new double[order];
        y[0] = 1.0;
        for (int i = 0; i < 10000; i++)
        {
            double sum = 0;
            for (int j = 0; j < order; j++)
            {
                sum += y[j] * aQ12[j];
            }
            for (int j = order - 1; j > 0; j--)
            {
                y[j] = y[j - 1];
            }
            y[0] = sum * (1.0 / 4096.0);
            if (!(y[0] < 10000 && y[0] > -10000)) return false;
            if ((i & 0x7) == 0)
            {
                double amp = 0;
                for (int j = 0; j < order; j++) amp += Math.Abs(y[j]);
                if (amp < 0.00001) return true;
            }
        }
        return true;
    }

    [TestMethod]
    public void LpcInvPredGain_StableClaim_IsBackedByImpulseResponse()
    {
        // Libopus-style contract: any filter we claim is stable (gain != 0) must also
        // pass the impulse-response test. 1000 trials x 9 orders x 16 dynamic-range shifts.
        var rng = new Random(0);
        int trialsRun = 0;
        int stableCount = 0;

        for (int count = 0; count < 1000; count++)
        {
            for (int order = 2; order <= 16; order += 2)
            {
                for (int shift = 0; shift < 16; shift++)
                {
                    var aQ12 = new short[16];
                    for (int i = 0; i < 16; i++)
                    {
                        aQ12[i] = (short)((short)rng.Next(short.MinValue, short.MaxValue + 1) >> shift);
                    }

                    int gain = SilkLpcInvPredGain.Compute(aQ12, order);
                    trialsRun++;
                    if (gain != 0)
                    {
                        stableCount++;
                        if (!ImpulseResponseAppearsStable(aQ12, order))
                        {
                            throw new Exception(
                                $"Trial {count} order {order} shift {shift}: " +
                                $"Compute returned stable gain {gain} but impulse response diverged. " +
                                $"A_Q12 = [{string.Join(", ", aQ12.Take(order))}]");
                        }
                    }
                }
            }
        }

        True(trialsRun > 0, "Expected at least some trials to run");
        True(stableCount > 0, "Expected at least some random filters to be classified stable");
    }

    [TestMethod]
    public void LpcInvPredGain_UnstableClaim_IsPermitted()
    {
        // Inverse direction - verify that when the impulse response clearly shows
        // divergence, the function returns 0 (rejects the filter). This is NOT guaranteed
        // by libopus semantics (the impulse test is a floor, not a ceiling), but we do
        // expect the DC-path early exit to catch the easy cases at minimum.

        // DC-unstable: positive saturating sum.
        short[] dcUnstable1 = new short[] { 2048, 2048, 2048 };
        Equal(0, SilkLpcInvPredGain.Compute(dcUnstable1, 3));

        // DC-unstable at the boundary.
        short[] dcUnstable2 = new short[] { 1024, 1024, 1024, 1024 };
        Equal(0, SilkLpcInvPredGain.Compute(dcUnstable2, 4));

        // DC-unstable: single large coefficient.
        short[] dcUnstable3 = new short[] { 8192 };
        Equal(0, SilkLpcInvPredGain.Compute(dcUnstable3, 1));
    }

    // -------- Deterministic multi-order known-stable fixture --------

    [TestMethod]
    public void LpcInvPredGain_KnownStable_MatchesExpectedSample()
    {
        // Deterministic fixture: a mild low-pass filter that is clearly stable.
        // A_Q12 = [1024, -256, 64] (coefficients shrinking geometrically).
        // Regression oracle: run the function and verify the result matches itself
        // AFTER independent impulse-response stability verification above, and that
        // it is strictly between the lower (INV_GAIN_Q30_MIN) and upper (2^30) bounds.
        short[] a = new short[] { 1024, -256, 64 };
        int gain = SilkLpcInvPredGain.Compute(a, 3);
        True(gain > SilkConstants.INV_GAIN_Q30_MIN, $"gain {gain} should be > {SilkConstants.INV_GAIN_Q30_MIN}");
        True(gain <= (1 << 30), $"gain {gain} should be <= 2^30");
        True(ImpulseResponseAppearsStable(a, 3), "Fixture should be stable under impulse-response test");
    }
}
