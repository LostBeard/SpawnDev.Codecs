// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC FIXED-predictor residual computation, GPU-callable. Per-sample
// independent forward-difference math (orders 1..4) - one thread per
// residual sample. Used by the FlacFixedSubframeEncoder GPU pipeline
// (Rice coding side stays sequential per subframe due to the bit
// writer state).
//
// Predictor coefficients (libFLAC fixed_compute_residual_):
//   order 1: r[n] = s[n] - s[n-1]
//   order 2: r[n] = s[n] - 2*s[n-1] + s[n-2]
//   order 3: r[n] = s[n] - 3*s[n-1] + 3*s[n-2] - s[n-3]
//   order 4: r[n] = s[n] - 4*s[n-1] + 6*s[n-2] - 4*s[n-3] + s[n-4]
//
// The residual array length is samples.Length - order. For sample
// index n in [order, samples.Length), residual index is n - order.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC FIXED residual computation. Per-sample helper.
/// </summary>
public static class FlacFixedResidualGpu
{
    /// <summary>
    /// Compute one residual sample at residual index <paramref name="ri"/>
    /// for FIXED order <paramref name="order"/> in [1, 4]. Reads
    /// <paramref name="order"/>+1 input samples starting at
    /// <paramref name="samples"/>[<paramref name="samplesBase"/> + ri];
    /// returns <c>samples[ri+order] - predictor</c>.
    /// </summary>
    public static int Residual(
        ArrayView<int> samples, long samplesBase, int order, int ri)
    {
        long n = samplesBase + ri + order;
        if (order == 1)
        {
            return samples[n] - samples[n - 1];
        }
        if (order == 2)
        {
            return samples[n] - 2 * samples[n - 1] + samples[n - 2];
        }
        if (order == 3)
        {
            return samples[n] - 3 * samples[n - 1] + 3 * samples[n - 2] - samples[n - 3];
        }
        // order == 4
        return samples[n]
            - 4 * samples[n - 1]
            + 6 * samples[n - 2]
            - 4 * samples[n - 3]
            +     samples[n - 4];
    }
}
