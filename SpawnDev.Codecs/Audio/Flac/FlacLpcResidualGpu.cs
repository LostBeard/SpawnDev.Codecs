// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable FLAC LPC encoder-side residual computation. Mirror of
// FlacLpcSubframeEncoder.ComputeResidualWithQuantizedCoefs. Computes
// residual[n] = samples[order + n] - (predictor_sum >> quantLevel)
// where predictor_sum = sum_{i=0..order-1} coefs[i] * samples[order + n - 1 - i].
//
// Per-output-sample parallel because each residual reads samples
// without writing back to the same buffer (this is the encoder-side
// pass). One thread per output residual sample - true parallel-per-
// element across all 6 ILGPU backends.
//
// Pairs with FlacLpcReconstructGpu (decoder-side) - now both encode +
// decode of FLAC LPC subframes have GPU primitives. This complements
// FlacFixedResidualGpu (encoder-side FIXED) which already shipped.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC LPC encoder-side residual computation. Mirror of
/// the residual-compute helper in <see cref="FlacLpcSubframeEncoder"/>.
/// </summary>
public static class FlacLpcResidualGpu
{
    /// <summary>
    /// Compute one residual sample at output index <paramref name="n"/>.
    /// Output residual length = samples.Length - order; thread n in
    /// [0, residualLength) computes residual[n] = samples[order + n] -
    /// (sum_{i=0..order-1} coefs[i] * samples[order + n - 1 - i]) &gt;&gt; quantLevel.
    /// </summary>
    /// <param name="samples">Input PCM samples (length &gt;= order + residualLength).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="coefs">QLP coefficients MSB-first as applied (length &gt;= order).</param>
    /// <param name="coefsBase">Base offset.</param>
    /// <param name="residual">Output residuals (length residualLength).</param>
    /// <param name="residualBase">Base offset.</param>
    /// <param name="order">LPC predictor order.</param>
    /// <param name="quantLevel">Right-shift amount applied to the prediction sum.</param>
    /// <param name="n">Residual index in [0, residualLength).</param>
    public static void ComputeAt(
        ArrayView<int> samples, long samplesBase,
        ArrayView<int> coefs, long coefsBase,
        ArrayView<int> residual, long residualBase,
        int order, int quantLevel, int n)
    {
        long inputN = samplesBase + order + n;
        long pred = 0;
        for (int i = 0; i < order; i++)
        {
            pred += (long)coefs[coefsBase + i] * samples[inputN - 1 - i];
        }
        residual[residualBase + n] = samples[inputN] - (int)(pred >> quantLevel);
    }
}
