// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable FLAC LPC-subframe reconstruction. Mirror of the
// reconstruction loop inside FlacSubframeDecoder.DecodeLpc (RFC 9639
// Section 8.1.4 LPC predictor). Adds the quantized LPC predictor's
// prediction to each residual sample to recover the original signal.
//
// Sequential per-stream because each output sample depends on the
// previous order samples in the same buffer. One-thread-per-stream on
// the GPU. Multiple independent FLAC channels parallelize cleanly
// across threads.
//
// Pairs with FlacFixedReconstructGpu - completes the FLAC subframe
// decode side: now both FIXED and LPC predictor types have GPU mirrors.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC LPC subframe reconstructor. Mirror of the
/// reconstruction loop in <see cref="FlacSubframeDecoder"/>.DecodeLpc.
/// </summary>
public static class FlacLpcReconstructGpu
{
    /// <summary>
    /// Reconstruct one FLAC LPC subframe in place. The input buffer
    /// must contain the <paramref name="order"/> warm-up samples at indices
    /// [0..order) and the residuals at [order..length). On return, all
    /// indices contain the reconstructed signal samples. Bit-exact vs the
    /// CPU FlacSubframeDecoder.DecodeLpc reconstruction loop.
    /// </summary>
    /// <param name="samples">In/out signal buffer (length &gt;= length).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="coefs">QLP coefficients MSB-first as applied (length &gt;= order).</param>
    /// <param name="coefsBase">Base offset.</param>
    /// <param name="length">Frame block size.</param>
    /// <param name="order">LPC predictor order (1..32 per FLAC spec).</param>
    /// <param name="quantLevel">Signed right-shift amount applied to the
    /// integer prediction sum before adding to the residual (0..15).</param>
    public static void ReconstructAt(
        ArrayView<int> samples, long samplesBase,
        ArrayView<int> coefs, long coefsBase,
        int length, int order, int quantLevel)
    {
        for (int n = order; n < length; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++)
            {
                pred += (long)coefs[coefsBase + i] * samples[samplesBase + n - 1 - i];
            }
            samples[samplesBase + n] = (int)(samples[samplesBase + n] + (pred >> quantLevel));
        }
    }
}
