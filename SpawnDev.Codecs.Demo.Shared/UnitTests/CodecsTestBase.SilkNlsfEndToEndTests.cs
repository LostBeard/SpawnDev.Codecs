using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end NLSF -&gt; LPC integration tests that exercise the real NB/MB and WB
/// codebooks together with the full decode path:
///
///     SilkNlsfDecoder.Decode(NbMb or Wb) -&gt; SilkNlsf2A.Compute -&gt; SilkLpcInvPredGain.Compute
///
/// Contract: for every first-stage codebook index (0..31) the decode -&gt; LPC -&gt; stability
/// loop must produce a stable filter (invGain != 0). This is the guarantee the SILK
/// decoder relies on - a real bitstream will only ever select indices in [0, 31], so
/// every one must round-trip cleanly.
///
/// Zero-residual variants test the bare codebook entries. Small-perturbation variants
/// cover the residual-adjust path without stressing the stabilizer beyond its spec.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Zero-residual (bare codebook vector) tests --------

    [TestMethod]
    public void NlsfPipeline_NbMb_AllFirstStageIndices_ProduceStableLpc()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        int order = cb.Order;
        Span<short> nlsfQ15 = stackalloc short[order];
        Span<short> aQ12 = stackalloc short[order];
        Span<sbyte> indices = stackalloc sbyte[order + 1];

        for (int cb1 = 0; cb1 < cb.NVectors; cb1++)
        {
            indices.Clear();
            indices[0] = (sbyte)cb1;

            SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);

            // Post-decode invariants: ordered, within Q15 range, delta-min spacing.
            for (int i = 0; i < order; i++)
            {
                True(nlsfQ15[i] >= 0, $"cb1={cb1}: nlsf[{i}]={nlsfQ15[i]} should be >= 0");
                True(nlsfQ15[i] <= 32767, $"cb1={cb1}: nlsf[{i}]={nlsfQ15[i]} should be <= 32767");
            }

            SilkNlsf2A.Compute(aQ12, nlsfQ15, order);
            int invGain = SilkLpcInvPredGain.Compute(aQ12, order);
            True(invGain > 0, $"cb1={cb1}: NbMb codebook should always produce a stable filter; got invGain=0");
        }
    }

    [TestMethod]
    public void NlsfPipeline_Wb_AllFirstStageIndices_ProduceStableLpc()
    {
        var cb = SilkNlsfCodebookTables.Wb;
        int order = cb.Order;
        Span<short> nlsfQ15 = stackalloc short[order];
        Span<short> aQ12 = stackalloc short[order];
        Span<sbyte> indices = stackalloc sbyte[order + 1];

        for (int cb1 = 0; cb1 < cb.NVectors; cb1++)
        {
            indices.Clear();
            indices[0] = (sbyte)cb1;

            SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);

            for (int i = 0; i < order; i++)
            {
                True(nlsfQ15[i] >= 0, $"cb1={cb1}: nlsf[{i}]={nlsfQ15[i]} should be >= 0");
                True(nlsfQ15[i] <= 32767, $"cb1={cb1}: nlsf[{i}]={nlsfQ15[i]} should be <= 32767");
            }

            SilkNlsf2A.Compute(aQ12, nlsfQ15, order);
            int invGain = SilkLpcInvPredGain.Compute(aQ12, order);
            True(invGain > 0, $"cb1={cb1}: Wb codebook should always produce a stable filter; got invGain=0");
        }
    }

    // -------- Small-perturbation stress --------

    [TestMethod]
    public void NlsfPipeline_NbMb_WithSmallResiduals_ProducesStableLpc()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        int order = cb.Order;
        Span<short> nlsfQ15 = stackalloc short[order];
        Span<short> aQ12 = stackalloc short[order];
        Span<sbyte> indices = stackalloc sbyte[order + 1];

        var rng = new Random(0x1234);
        int stable = 0;
        int trials = 200;
        for (int trial = 0; trial < trials; trial++)
        {
            indices.Clear();
            indices[0] = (sbyte)rng.Next(0, cb.NVectors);
            // Small non-zero residuals in [-NLSF_QUANT_MAX_AMPLITUDE, +NLSF_QUANT_MAX_AMPLITUDE].
            for (int i = 1; i <= order; i++)
            {
                indices[i] = (sbyte)rng.Next(-SilkConstants.NLSF_QUANT_MAX_AMPLITUDE, SilkConstants.NLSF_QUANT_MAX_AMPLITUDE + 1);
            }

            SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);
            SilkNlsf2A.Compute(aQ12, nlsfQ15, order);
            int invGain = SilkLpcInvPredGain.Compute(aQ12, order);
            if (invGain == 0)
            {
                throw new Exception(
                    $"Trial {trial} (cb1={indices[0]}): " +
                    $"unstable LPC after NbMb decode+Nlsf2A. " +
                    $"NLSF=[{string.Join(",", nlsfQ15.ToArray().Take(order))}], " +
                    $"A_Q12=[{string.Join(",", aQ12.ToArray().Take(order))}]");
            }
            stable++;
        }
        Equal(trials, stable);
    }

    [TestMethod]
    public void NlsfPipeline_Wb_WithSmallResiduals_ProducesStableLpc()
    {
        var cb = SilkNlsfCodebookTables.Wb;
        int order = cb.Order;
        Span<short> nlsfQ15 = stackalloc short[order];
        Span<short> aQ12 = stackalloc short[order];
        Span<sbyte> indices = stackalloc sbyte[order + 1];

        var rng = new Random(0x5678);
        int stable = 0;
        int trials = 200;
        for (int trial = 0; trial < trials; trial++)
        {
            indices.Clear();
            indices[0] = (sbyte)rng.Next(0, cb.NVectors);
            for (int i = 1; i <= order; i++)
            {
                indices[i] = (sbyte)rng.Next(-SilkConstants.NLSF_QUANT_MAX_AMPLITUDE, SilkConstants.NLSF_QUANT_MAX_AMPLITUDE + 1);
            }

            SilkNlsfDecoder.Decode(nlsfQ15, indices, cb);
            SilkNlsf2A.Compute(aQ12, nlsfQ15, order);
            int invGain = SilkLpcInvPredGain.Compute(aQ12, order);
            if (invGain == 0)
            {
                throw new Exception(
                    $"Trial {trial} (cb1={indices[0]}): " +
                    $"unstable LPC after Wb decode+Nlsf2A. " +
                    $"NLSF=[{string.Join(",", nlsfQ15.ToArray().Take(order))}], " +
                    $"A_Q12=[{string.Join(",", aQ12.ToArray().Take(order))}]");
            }
            stable++;
        }
        Equal(trials, stable);
    }
}
