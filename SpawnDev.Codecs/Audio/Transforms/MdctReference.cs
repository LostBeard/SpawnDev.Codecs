// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// CPU reference implementation of the forward Modified Discrete Cosine
// Transform. Mirrors <see cref="ImdctReference"/>.

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// Forward MDCT (O(N^2) reference). Takes <c>2N</c> time-domain samples and
/// produces <c>N</c> frequency coefficients using
///   X[k] = sum_{n=0}^{2N-1} x[n] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))
/// for <c>k = 0..N-1</c>. Inverse of <see cref="ImdctReference"/>.
/// </summary>
public static class MdctReference
{
    /// <summary>
    /// Transform <paramref name="timeDomain"/> (length <c>2N</c>) into
    /// <paramref name="frequencyOut"/> (length <c>N</c>).
    /// </summary>
    public static void Transform(ReadOnlySpan<float> timeDomain, Span<float> frequencyOut)
    {
        if (timeDomain.Length == 0)
            throw new ArgumentException("MDCT input must have at least 2 samples.", nameof(timeDomain));
        if ((timeDomain.Length & 1) != 0)
            throw new ArgumentException("MDCT input length must be even.", nameof(timeDomain));
        int n = timeDomain.Length / 2;
        if (frequencyOut.Length != n)
            throw new ArgumentException(
                $"MDCT output length {frequencyOut.Length} must be input length / 2 = {n}.",
                nameof(frequencyOut));

        double factor = Math.PI / n;
        double halfN = 0.5 * n;
        for (int k = 0; k < n; k++)
        {
            double acc = 0;
            for (int idx = 0; idx < 2 * n; idx++)
            {
                double theta = factor * (idx + 0.5 + halfN) * (k + 0.5);
                acc += timeDomain[idx] * Math.Cos(theta);
            }
            frequencyOut[k] = (float)acc;
        }
    }

    /// <summary>Convenience overload that allocates its own output buffer.</summary>
    public static float[] Transform(ReadOnlySpan<float> timeDomain)
    {
        var outArr = new float[timeDomain.Length / 2];
        Transform(timeDomain, outArr);
        return outArr;
    }
}
