// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable inverse MDCT (O(N^2) reference). Bit-exact mirror of
// ImdctReference for in-kernel use by audio codec decoder pipelines
// (Vorbis, Opus/CELT, FLAC if it ever needs an MDCT path, etc.).
//
// Each output time-domain sample y[n] is computed independently:
//   y[n] = sum_{k=0}^{N-1} X[k] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
// One thread per n maps cleanly across all backends.

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// GPU-callable O(N^2) inverse MDCT. Single-sample helper - one
/// thread per output time-domain sample.
/// </summary>
public static class ImdctReferenceGpu
{
    /// <summary>
    /// Compute one IMDCT output sample y[idx] for an N-point IMDCT
    /// (2N output samples). Reads frequency coefficients from
    /// <paramref name="frequency"/> starting at <paramref name="inBase"/>.
    /// </summary>
    public static float Sample(
        ArrayView<float> frequency, long inBase, int n, int idx)
    {
        float factor = (float)(Math.PI / n);
        float halfN = 0.5f * n;
        float acc = 0;
        for (int k = 0; k < n; k++)
        {
            float theta = factor * (idx + 0.5f + halfN) * (k + 0.5f);
            acc += frequency[inBase + k] * XMath.Cos(theta);
        }
        return acc;
    }
}
