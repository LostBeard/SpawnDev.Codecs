// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 category-token magnitude decoder. After the constrained tree
// walk (slice 145) lands on Category1..Category6, the decoder reads
// N residual bits MSB-first against the cat<N>_prob arrays from
// slice 140 and adds the result to the per-category MIN_VAL to
// recover the integer magnitude.
//
// libvpx reference: vp9/decoder/vp9_detokenize.c, the cat case of
// the read_coef_count loop. Bit ordering matches libvpx exactly:
// cat<N>_prob[0] is the most significant bit of the residual,
// cat<N>_prob[N-1] is the least.
//
// Sign handling is the caller's responsibility - libvpx reads the
// sign bit AFTER the category magnitude using a flat probability of
// 128, and applies it to negate the magnitude.

namespace SpawnDev.Codecs.Video.Vp9;

public static partial class Vp9CoefProbs
{
    /// <summary>VP9 category MIN_VAL constants (libvpx CAT<N>_MIN_VAL).</summary>
    public static class CatMinVal
    {
        /// <summary>Cat1 magnitude in [5, 6].</summary>
        public const int Cat1 = 5;
        /// <summary>Cat2 magnitude in [7, 10].</summary>
        public const int Cat2 = 7;
        /// <summary>Cat3 magnitude in [11, 18].</summary>
        public const int Cat3 = 11;
        /// <summary>Cat4 magnitude in [19, 34].</summary>
        public const int Cat4 = 19;
        /// <summary>Cat5 magnitude in [35, 66].</summary>
        public const int Cat5 = 35;
        /// <summary>Cat6 magnitude in [67, ...].</summary>
        public const int Cat6 = 67;
    }

    /// <summary>
    /// Decode the magnitude of a Cat1..Cat6 token. Reads N residual
    /// bits MSB-first against the corresponding cat&lt;N&gt;_prob
    /// table via <paramref name="readBit"/> and adds CatMinVal.Cat&lt;N&gt;.
    /// </summary>
    /// <param name="readBit">
    /// Caller-supplied bit reader (typically a closure over a
    /// <see cref="Vp9BoolDecoder"/>: <c>b =&gt; reader.Read(b)</c>).
    /// </param>
    /// <param name="cat">Token returned by <see cref="Vp9CoefTrees.DecodeConToken"/>.</param>
    /// <param name="isHighBitDepth">
    /// When true, Cat6 reads 18 residual bits against
    /// <see cref="Cat6ProbHigh12"/> (12-bit profile); when false,
    /// 14 bits against <see cref="Cat6Prob"/> (8-bit profile).
    /// </param>
    /// <returns>Unsigned coefficient magnitude (sign is the caller's job).</returns>
    public static int DecodeCategoryMagnitude(Func<byte, int> readBit, Vp9CoefToken cat, bool isHighBitDepth = false)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        return cat switch
        {
            Vp9CoefToken.Category1 => CatMinVal.Cat1 + ReadResidualMsbFirst(readBit, Cat1Prob),
            Vp9CoefToken.Category2 => CatMinVal.Cat2 + ReadResidualMsbFirst(readBit, Cat2Prob),
            Vp9CoefToken.Category3 => CatMinVal.Cat3 + ReadResidualMsbFirst(readBit, Cat3Prob),
            Vp9CoefToken.Category4 => CatMinVal.Cat4 + ReadResidualMsbFirst(readBit, Cat4Prob),
            Vp9CoefToken.Category5 => CatMinVal.Cat5 + ReadResidualMsbFirst(readBit, Cat5Prob),
            Vp9CoefToken.Category6 => CatMinVal.Cat6 + ReadResidualMsbFirst(
                readBit, isHighBitDepth ? Cat6ProbHigh12 : Cat6Prob),
            _ => throw new ArgumentOutOfRangeException(
                nameof(cat),
                $"Token {cat} is not a category token; only Category1..Category6 carry residual magnitude bits."),
        };
    }

    /// <summary>
    /// Read N bits MSB-first against the supplied probability vector.
    /// probs[0] decodes the highest-significance bit; probs[N-1] the
    /// lowest.
    /// </summary>
    private static int ReadResidualMsbFirst(Func<byte, int> readBit, byte[] probs)
    {
        int value = 0;
        for (int i = 0; i < probs.Length; i++)
            value = (value << 1) | readBit(probs[i]);
        return value;
    }
}
