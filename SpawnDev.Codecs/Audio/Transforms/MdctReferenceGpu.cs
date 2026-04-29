// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable forward MDCT (O(N^2) reference). Bit-exact mirror of
// MdctReference for in-kernel use by audio codec encoder pipelines.
//
// Each output coefficient X[k] is computed independently:
//   X[k] = sum_{n=0}^{2N-1} x[n] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
// so the per-output computation maps cleanly to one thread per k. The
// kernel takes 2N time-domain floats and produces N frequency floats.

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// GPU-callable O(N^2) forward MDCT. Single-coefficient helper - one
/// thread per output coefficient.
/// </summary>
public static class MdctReferenceGpu
{
    /// <summary>
    /// Compute one MDCT output coefficient X[k] for an N-point MDCT
    /// (input length 2N). Reads time-domain samples from
    /// <paramref name="timeDomain"/> starting at <paramref name="inBase"/>.
    /// </summary>
    public static float Coefficient(
        ArrayView<float> timeDomain, long inBase, int n, int k)
    {
        float factor = (float)(Math.PI / n);
        float halfN = 0.5f * n;
        float acc = 0;
        int twoN = 2 * n;
        for (int idx = 0; idx < twoN; idx++)
        {
            float theta = factor * (idx + 0.5f + halfN) * (k + 0.5f);
            acc += timeDomain[inBase + idx] * XMath.Cos(theta);
        }
        return acc;
    }
}
