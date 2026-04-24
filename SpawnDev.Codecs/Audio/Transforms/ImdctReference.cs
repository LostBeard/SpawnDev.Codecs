// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// CPU reference implementation of the Inverse Modified Discrete Cosine
// Transform. Used by any codec that synthesises time-domain audio from an
// MDCT frequency-domain block - Vorbis, Opus/CELT, AAC, AC-3, MP3, etc.
//
// This is the O(N^2) direct formula. It's accurate to float precision and
// easy to reason about - perfect as a correctness reference. A FFT-based
// O(N log N) CPU implementation and an ILGPU-accelerated kernel will follow
// in later slices; both will be validated against this reference.

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// Reference (naive O(N^2)) IMDCT. Produces <c>2N</c> time-domain samples from
/// <c>N</c> frequency-domain coefficients using the formula
///   y[n] = sum_{k=0}^{N-1} X[k] * cos(pi / N * (n + 0.5 + N/2) * (k + 0.5))
/// for <c>n = 0..2N-1</c>. This matches the Vorbis I and Opus/CELT
/// definitions (pre-window, pre-overlap).
/// </summary>
public static class ImdctReference
{
    /// <summary>
    /// Transform <paramref name="frequencyCoefficients"/> (length <c>N</c>)
    /// into <paramref name="timeDomainOut"/> (length <c>2N</c>).
    /// </summary>
    public static void Transform(ReadOnlySpan<float> frequencyCoefficients, Span<float> timeDomainOut)
    {
        int n = frequencyCoefficients.Length;
        if (n == 0) throw new ArgumentException("IMDCT input must have at least 1 coefficient.", nameof(frequencyCoefficients));
        if (timeDomainOut.Length != 2 * n)
            throw new ArgumentException(
                $"IMDCT output length {timeDomainOut.Length} must be 2 * {n} = {2 * n}.",
                nameof(timeDomainOut));

        double factor = Math.PI / n;
        double halfN = 0.5 * n;
        for (int idx = 0; idx < 2 * n; idx++)
        {
            double acc = 0;
            for (int k = 0; k < n; k++)
            {
                double theta = factor * (idx + 0.5 + halfN) * (k + 0.5);
                acc += frequencyCoefficients[k] * Math.Cos(theta);
            }
            timeDomainOut[idx] = (float)acc;
        }
    }

    /// <summary>
    /// Overload that allocates its own output buffer. Convenience for tests.
    /// </summary>
    public static float[] Transform(ReadOnlySpan<float> frequencyCoefficients)
    {
        var outArr = new float[2 * frequencyCoefficients.Length];
        Transform(frequencyCoefficients, outArr);
        return outArr;
    }
}
