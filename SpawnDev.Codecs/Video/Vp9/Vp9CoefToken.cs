// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 coefficient token alphabet + the binary tree topology that
// drives the entropy decoder. Bit-exact with libvpx
// vp9/common/vp9_entropy.h (token defines) and vp9_entropy.c
// (vp9_coef_con_tree).
//
// Each non-zero coefficient is decoded as a token from this 12-token
// alphabet. The first 5 tokens (ZERO through FOUR) carry their
// magnitude implicitly. CATEGORY1..CATEGORY6 represent magnitude
// ranges and require additional residual bits decoded against the
// cat<N>_prob arrays from slice 140. EOB_TOKEN ends the block.
//
// Token magnitudes (for reference):
//   ZERO_TOKEN      = 0  (no extra bits)
//   ONE_TOKEN       = 1
//   TWO_TOKEN       = 2
//   THREE_TOKEN     = 3
//   FOUR_TOKEN      = 4
//   CATEGORY1_TOKEN = 5..6      (1 extra bit)
//   CATEGORY2_TOKEN = 7..10     (2 extra bits)
//   CATEGORY3_TOKEN = 11..18    (3 extra bits)
//   CATEGORY4_TOKEN = 19..34    (4 extra bits)
//   CATEGORY5_TOKEN = 35..66    (5 extra bits)
//   CATEGORY6_TOKEN = 67..      (14 extra bits, 18 in 12-bit profiles)
//   EOB_TOKEN       = end-of-block marker
//
// Decoding flow (libvpx):
//   1. Read EOB? bit using prob[0] - if 1, return EOB.
//   2. Read ZERO? bit using prob[1] - if 0, return ZERO.
//   3. Read ONE? bit using prob[2] - if 0, return ONE.
//   4. Walk CoefConTree with probs[3..10] to land on TWO..CAT6.
//   5. For category tokens, decode the residual bits + sign.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 coefficient token alphabet (libvpx vp9_entropy.h ZERO_TOKEN
/// through EOB_TOKEN). Values match libvpx defines exactly.
/// </summary>
public enum Vp9CoefToken : byte
{
    /// <summary>Coefficient is zero.</summary>
    Zero = 0,
    /// <summary>Coefficient magnitude = 1.</summary>
    One = 1,
    /// <summary>Coefficient magnitude = 2.</summary>
    Two = 2,
    /// <summary>Coefficient magnitude = 3.</summary>
    Three = 3,
    /// <summary>Coefficient magnitude = 4.</summary>
    Four = 4,
    /// <summary>Magnitude in [5, 6]; 1 residual bit.</summary>
    Category1 = 5,
    /// <summary>Magnitude in [7, 10]; 2 residual bits.</summary>
    Category2 = 6,
    /// <summary>Magnitude in [11, 18]; 3 residual bits.</summary>
    Category3 = 7,
    /// <summary>Magnitude in [19, 34]; 4 residual bits.</summary>
    Category4 = 8,
    /// <summary>Magnitude in [35, 66]; 5 residual bits.</summary>
    Category5 = 9,
    /// <summary>Magnitude in [67, ...]; 14 residual bits (18 in 12-bit profiles).</summary>
    Category6 = 10,
    /// <summary>End-of-block marker - no further coefficients in this block.</summary>
    Eob = 11,
}

/// <summary>
/// VP9 binary tree topology for entropy decoding the constrained
/// coefficient sub-tree (libvpx vp9_coef_con_tree). The decoder
/// walks this tree after the EOB / ZERO / ONE pre-checks have
/// determined the coefficient is in TWO..CAT6.
/// </summary>
public static class Vp9CoefTrees
{
    /// <summary>
    /// libvpx vp9_coef_con_tree, 16 entries laid out as 8 internal
    /// nodes of (left, right) byte-index branches. Leaf entries
    /// are stored as the negative token value; non-leaf entries are
    /// the byte index of the next node within this same array.
    ///
    /// Tree structure (libvpx label / pair-index / array slot pair):
    ///   0/0..1   LOW_VAL          ->   2 = TWO,            6 = HIGH_LOW
    ///   1/2..3   TWO              -> -Two,                 4 = THREE
    ///   2/4..5   THREE            -> -Three,              -Four
    ///   3/6..7   HIGH_LOW         ->   8 = CAT_ONE,       10 = CAT_THREEFOUR
    ///   4/8..9   CAT_ONE          -> -Cat1,               -Cat2
    ///   5/10..11 CAT_THREEFOUR    ->  12 = CAT_THREE,     14 = CAT_FIVE
    ///   6/12..13 CAT_THREE        -> -Cat3,               -Cat4
    ///   7/14..15 CAT_FIVE         -> -Cat5,               -Cat6
    /// </summary>
    public static readonly sbyte[] CoefConTree = new sbyte[]
    {
        2, 6,                                                                     // 0 = LOW_VAL
        -(sbyte)Vp9CoefToken.Two, 4,                                              // 1 = TWO
        -(sbyte)Vp9CoefToken.Three, -(sbyte)Vp9CoefToken.Four,                    // 2 = THREE
        8, 10,                                                                    // 3 = HIGH_LOW
        -(sbyte)Vp9CoefToken.Category1, -(sbyte)Vp9CoefToken.Category2,           // 4 = CAT_ONE
        12, 14,                                                                   // 5 = CAT_THREEFOUR
        -(sbyte)Vp9CoefToken.Category3, -(sbyte)Vp9CoefToken.Category4,           // 6 = CAT_THREE
        -(sbyte)Vp9CoefToken.Category5, -(sbyte)Vp9CoefToken.Category6,           // 7 = CAT_FIVE
    };

    /// <summary>
    /// Walk the constrained coefficient tree given a probability vector.
    /// Mirrors libvpx <c>treed_read</c>: starting at byte index 0,
    /// each iteration reads a bit at <c>probs[i &gt;&gt; 1]</c> via
    /// <paramref name="readBit"/> and advances to
    /// <c>tree[i + bit]</c>. Stops when the resulting value is
    /// non-positive (a leaf), returning the corresponding token.
    /// </summary>
    /// <param name="readBit">
    /// Caller-supplied bit reader (typically a closure over a
    /// <see cref="Vp9BoolDecoder"/>). Takes a probability and
    /// returns 0 or 1.
    /// </param>
    /// <param name="probs">
    /// Probability vector indexed by tree node pair index (i.e.
    /// position 0 corresponds to byte slots 0/1, position 1 to
    /// 2/3, ... position 7 to 14/15 - the 8 internal nodes of the
    /// constrained sub-tree).
    /// </param>
    public static Vp9CoefToken DecodeConToken(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        if (readBit is null) throw new ArgumentNullException(nameof(readBit));
        if (probs.Length < 8)
            throw new ArgumentException("probs must hold >= 8 entries for the constrained tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = CoefConTree[i + bit];
            if (next <= 0)
                return (Vp9CoefToken)(-next);
            i = next;
        }
    }
}
