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
    /// right half overlaps with short-sized neighbours (Vorbis I Section 4.3.1,
    /// see <see cref="Generate"/>): a short neighbour's slope is centred on the
    /// long block's quarter point, with zeros outside it and ones between it and
    /// the block centre.
    /// </summary>
    /// <param name="longSize">Long block size.</param>
    /// <param name="shortSize">Short block size.</param>
    /// <param name="prevLong">True if the previous block was long (no left transition needed).</param>
    /// <param name="nextLong">True if the next block is long (no right transition needed).</param>
    public static float[] GenerateTransition(int longSize, int shortSize, bool prevLong, bool nextLong)
    {
        if (shortSize > longSize) throw new ArgumentException("shortSize must be <= longSize.");
        return Generate(longSize, prevLong ? longSize : shortSize, nextLong ? longSize : shortSize);
    }

    /// <summary>
    /// The synthesis window of a block of size <paramref name="n"/> per Vorbis I Section 4.3.1: the left slope
    /// spans <paramref name="leftN"/>/2 samples centred on n/4 (leftN = the size of the block it overlaps:
    /// the short size when a long block follows a short one, else n), the right slope spans
    /// <paramref name="rightN"/>/2 samples centred on 3n/4, zeros outside the slopes and ones between them.
    /// <c>Generate(n, n, n)</c> equals <see cref="GenerateCanonical"/>.
    /// </summary>
    public static float[] Generate(int n, int leftN, int rightN)
    {
        if (n < 2) throw new ArgumentException("Window length must be >= 2.", nameof(n));
        if ((n & 3) != 0 || (leftN & 3) != 0 || (rightN & 3) != 0 || leftN > n || rightN > n || leftN < 4 || rightN < 4)
            throw new ArgumentException("Block sizes must be multiples of 4 and the neighbours no larger than the block.");
        var w = new float[n];
        int leftStart = n / 4 - leftN / 4, leftEnd = n / 4 + leftN / 4, leftHalf = leftN / 2;
        int rightStart = 3 * n / 4 - rightN / 4, rightEnd = 3 * n / 4 + rightN / 4, rightHalf = rightN / 2;
        for (int i = leftStart; i < leftEnd; i++)
        {
            double s = Math.Sin((i - leftStart + 0.5) / leftHalf * (Math.PI / 2));
            w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
        }
        for (int i = leftEnd; i < rightStart; i++) w[i] = 1f;
        for (int i = rightStart; i < rightEnd; i++)
        {
            double s = Math.Sin((i - rightStart + 0.5) / rightHalf * (Math.PI / 2) + Math.PI / 2);
            w[i] = (float)Math.Sin(0.5 * Math.PI * s * s);
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
