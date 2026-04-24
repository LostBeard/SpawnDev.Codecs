// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis synthesis window + overlap-add per Vorbis I Section 1.3.2 and
// Section 4.3.8.3 / 4.3.8.4. The window is a sin-flipped
// sin^2 shape w[i] = sin(pi/2 * sin^2(pi/n * (i + 0.5))). Long blocks use
// this shape at the long length; short blocks at the short length. Adjacent
// blocks overlap by 50% and their pointwise sum reconstructs the time domain
// losslessly for bit-exact identical inputs (MDCT TDAC property).

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Vorbis synthesis window helpers.</summary>
public static class VorbisWindow
{
    /// <summary>
    /// Generate a canonical Vorbis synthesis window of length <paramref name="n"/>.
    /// Used for both long and short blocks by simply passing the matching block
    /// size; transition windows (when a long block abuts a short one) are
    /// composed separately in <see cref="GenerateTransition"/>.
    /// </summary>
    public static float[] GenerateCanonical(int n)
    {
        if (n < 2) throw new ArgumentException("Window length must be >= 2.", nameof(n));
        var w = new float[n];
        double factor = Math.PI / n;
        for (int i = 0; i < n; i++)
        {
            double s = Math.Sin(factor * (i + 0.5));
            w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
        }
        return w;
    }

    /// <summary>
    /// Generate a Vorbis transition window for a long block whose left and/or
    /// right half overlaps with short-sized neighbours. Per Vorbis I Section
    /// 1.3.2, when a long block is adjacent to a short block, the long block's
    /// overlap region uses the short-block window shape within its first or
    /// last <c>shortBlockSize/2</c> samples; the rest is either zeros (before
    /// the rise) or ones (after the ramp completes).
    /// </summary>
    /// <param name="longSize">Long block size.</param>
    /// <param name="shortSize">Short block size.</param>
    /// <param name="prevLong">True if the previous block was long (no left transition needed).</param>
    /// <param name="nextLong">True if the next block is long (no right transition needed).</param>
    public static float[] GenerateTransition(int longSize, int shortSize, bool prevLong, bool nextLong)
    {
        if (shortSize > longSize) throw new ArgumentException("shortSize must be <= longSize.");
        if ((longSize & 1) != 0 || (shortSize & 1) != 0)
            throw new ArgumentException("Block sizes must be even.");
        var w = new float[longSize];
        int halfLong = longSize / 2;
        int halfShort = shortSize / 2;
        double shortFactor = Math.PI / shortSize;
        double longFactor = Math.PI / longSize;

        // Left half.
        if (prevLong)
        {
            for (int i = 0; i < halfLong; i++)
            {
                double s = Math.Sin(longFactor * (i + 0.5));
                w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
            }
        }
        else
        {
            // Zeros until the short window would start rising, then the short
            // window's rising half, then ones up to the center.
            int riseStart = halfLong - halfShort;
            for (int i = 0; i < riseStart; i++) w[i] = 0f;
            for (int j = 0; j < halfShort; j++)
            {
                int i = riseStart + j;
                double s = Math.Sin(shortFactor * (j + 0.5));
                w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
            }
        }

        // Right half (mirror logic).
        if (nextLong)
        {
            for (int i = halfLong; i < longSize; i++)
            {
                double s = Math.Sin(longFactor * (i + 0.5));
                w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
            }
        }
        else
        {
            int fallEnd = halfLong + halfShort;
            for (int j = 0; j < halfShort; j++)
            {
                int i = halfLong + j;
                double s = Math.Sin(shortFactor * (halfShort + j + 0.5));
                w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
            }
            for (int i = fallEnd; i < longSize; i++) w[i] = 0f;
        }
        return w;
    }

    /// <summary>
    /// Overlap-add the previous block's right half into the current block's
    /// left half and emit the matching count of finalised samples. The
    /// inputs are windowed time-domain samples; this helper just sums them
    /// pointwise.
    /// </summary>
    /// <param name="previousRightHalf">Previous block's windowed right-half samples (length halfBlockSize).</param>
    /// <param name="currentLeftHalf">Current block's windowed left-half samples (length halfBlockSize).</param>
    /// <param name="output">Destination span of length halfBlockSize.</param>
    public static void OverlapAdd(
        ReadOnlySpan<float> previousRightHalf,
        ReadOnlySpan<float> currentLeftHalf,
        Span<float> output)
    {
        int half = output.Length;
        if (previousRightHalf.Length != half || currentLeftHalf.Length != half)
            throw new ArgumentException("Overlap-add requires all three spans to be the same length.");
        for (int i = 0; i < half; i++)
            output[i] = previousRightHalf[i] + currentLeftHalf[i];
    }
}
