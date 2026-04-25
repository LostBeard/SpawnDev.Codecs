// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 sub-block MV averaging helpers. Used to derive chroma MVs
// from luma sub-block MVs when a sub-8x8 block has different MVs
// for each of its 4 4x4 sub-blocks (libvpx mi_mv_pred_q4 in
// vp9_reconinter.c).
//
// 4:2:0 chroma is half-resolution, so a chroma sample maps to a
// 2x2 luma region. When that 2x2 region spans 2 or 4 different
// luma sub-block MVs (because the parent luma block was partitioned
// into 4x4 / 4x8 / 8x4 sub-blocks), the chroma MV is the rounded
// average of the participating luma MVs.
//
// Rounding semantics (libvpx round_mv_comp_q4):
//   (value &lt; 0 ? value - 2 : value + 2) / 4
// i.e. round AWAY from zero on the +2 / -2 nudge, then truncate
// toward zero with integer divide. Net result: round-half-away-
// from-zero on the average.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 sub-block MV averaging helpers.</summary>
public static class Vp9MvAverage
{
    /// <summary>
    /// Average 2 MVs with round-half-away-from-zero on each component.
    /// Used by 4:2:0 chroma MV derivation when a 2x2 chroma region
    /// spans exactly 2 different luma 4x4 sub-block MVs.
    /// </summary>
    public static Vp9Mv Average2(Vp9Mv a, Vp9Mv b)
    {
        return new Vp9Mv(
            RoundComp(a.Row + b.Row, 2),
            RoundComp(a.Col + b.Col, 2));
    }

    /// <summary>
    /// Average 4 MVs with round-half-away-from-zero on each component.
    /// Used by 4:2:0 chroma MV derivation when a 2x2 chroma region
    /// spans all 4 luma 4x4 sub-block MVs (the libvpx
    /// <c>mi_mv_pred_q4</c> case).
    /// </summary>
    public static Vp9Mv Average4(Vp9Mv a, Vp9Mv b, Vp9Mv c, Vp9Mv d)
    {
        return new Vp9Mv(
            RoundComp4(a.Row + b.Row + c.Row + d.Row),
            RoundComp4(a.Col + b.Col + c.Col + d.Col));
    }

    /// <summary>
    /// libvpx-style 2-way component average:
    /// <c>(value &lt; 0 ? value - 1 : value + 1) / 2</c>. Equivalent
    /// to round-half-away-from-zero divide.
    /// </summary>
    private static int RoundComp(int sumOfTwo, int twoNudge)
    {
        // For 2 values summed, the round-half-away nudge is +1/-1;
        // for 4 values summed, it's +2/-2. twoNudge of 2 == 4-way path.
        // This helper handles the 2-way case exclusively.
        int nudge = twoNudge == 2 ? 1 : 2;
        int divisor = twoNudge == 2 ? 2 : 4;
        return (sumOfTwo < 0 ? sumOfTwo - nudge : sumOfTwo + nudge) / divisor;
    }

    /// <summary>
    /// libvpx <c>round_mv_comp_q4</c>: rounds the sum of 4 component
    /// values to a 4-way average with round-half-away-from-zero.
    /// </summary>
    public static int RoundComp4(int sumOfFour)
    {
        return (sumOfFour < 0 ? sumOfFour - 2 : sumOfFour + 2) / 4;
    }
}
