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
        // theta = pi / (4N) * (2 idx + 1 + N) * (2k + 1). The phase is kept as the exact integer
        // p = (2 idx + 1 + N)(2k + 1) mod 8N (one period), so the cosine's argument stays in [-pi, pi]:
        // a float theta grows to ~2.5 pi N (~8000 rad for N = 1024), where one float ulp is ~1e-3 rad
        // and every term of the sum carries that phase error (measured 4.5e-5 vs libvorbis on 2048
        // blocks, against 3.6e-7 for the double CPU reference), and fast GPU cosines lose more there.
        int period = 8 * n;
        int a = (2 * idx + 1 + n) % period;
        int stride = (2 * a) % period;
        float factor = (float)(Math.PI / (4.0 * n));
        int p = a;
        float acc = 0;
        for (int k = 0; k < n; k++)
        {
            int q = p >= 4 * n ? p - period : p;
            acc += frequency[inBase + k] * XMath.Cos(q * factor);
            p += stride;
            if (p >= period) p -= period;
        }
        return acc;
    }
}
