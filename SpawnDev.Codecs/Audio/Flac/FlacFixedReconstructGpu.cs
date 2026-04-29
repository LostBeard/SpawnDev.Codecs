// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable FLAC FIXED-subframe reconstruction. Mirror of the
// reconstruction loop inside FlacSubframeDecoder.DecodeFixed (RFC 9639
// Section 8.1.3 fixed predictor). Adds the fixed predictor's prediction
// to each residual sample to recover the original signal.
//
// Sequential per-stream because each output sample depends on the
// previous order samples in the same buffer. One-thread-per-stream on
// the GPU. Multiple independent FLAC channels parallelize cleanly
// across threads.
//
// Order 0 -> no prediction (samples already equal residuals).
// Order 1 -> pred = samples[n-1].
// Order 2 -> pred = 2*samples[n-1] - samples[n-2].
// Order 3 -> pred = 3*samples[n-1] - 3*samples[n-2] + samples[n-3].
// Order 4 -> pred = 4*samples[n-1] - 6*samples[n-2] + 4*samples[n-3] - samples[n-4].

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC FIXED subframe reconstructor. Mirror of the
/// reconstruction loop in <see cref="FlacSubframeDecoder"/>.DecodeFixed.
/// </summary>
public static class FlacFixedReconstructGpu
{
    /// <summary>
    /// Reconstruct one FLAC FIXED subframe in place. The input buffer
    /// must contain the <paramref name="order"/> warm-up samples at indices
    /// [0..order) and the residuals at [order..length). On return, all
    /// indices contain the reconstructed signal samples.
    /// Bit-exact vs the CPU FlacSubframeDecoder.DecodeFixed reconstruction
    /// loop.
    /// </summary>
    /// <param name="samples">In/out signal buffer (length &gt;= length).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="length">Frame block size.</param>
    /// <param name="order">FIXED predictor order (0..4).</param>
    public static void ReconstructAt(
        ArrayView<int> samples, long samplesBase,
        int length, int order)
    {
        for (int n = order; n < length; n++)
        {
            long pred = 0;

            // FixedCoefs by order:
            //   1 -> [1]
            //   2 -> [2, -1]
            //   3 -> [3, -3, 1]
            //   4 -> [4, -6, 4, -1]
            // Unrolled per order to keep the kernel simple.
            if (order == 1)
            {
                pred = samples[samplesBase + n - 1];
            }
            else if (order == 2)
            {
                pred = 2L * samples[samplesBase + n - 1]
                     - 1L * samples[samplesBase + n - 2];
            }
            else if (order == 3)
            {
                pred = 3L * samples[samplesBase + n - 1]
                     - 3L * samples[samplesBase + n - 2]
                     + 1L * samples[samplesBase + n - 3];
            }
            else if (order == 4)
            {
                pred = 4L * samples[samplesBase + n - 1]
                     - 6L * samples[samplesBase + n - 2]
                     + 4L * samples[samplesBase + n - 3]
                     - 1L * samples[samplesBase + n - 4];
            }

            samples[samplesBase + n] = (int)(samples[samplesBase + n] + pred);
        }
    }
}
