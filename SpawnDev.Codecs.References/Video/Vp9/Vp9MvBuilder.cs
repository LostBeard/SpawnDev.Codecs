// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector builder. Combines a reference MV with a
// decoded diff (from <see cref="Vp9MvPairReader"/>) and optionally
// rounds away the 1/8-pel bit when the frame disallows high
// precision motion. Mirror of libvpx vp9/decoder/vp9_decodemv.c
// read_mv (the post-component-decode portion) plus
// <c>lower_mv_precision</c> from vp9_mv_common.h.
//
// libvpx lower_mv_precision:
//   if (!allow_hp) {
//     if (mv->row & 1) mv->row += (mv->row > 0 ? -1 : 1);
//     if (mv->col & 1) mv->col += (mv->col > 0 ? -1 : 1);
//   }
// I.e. round TOWARD zero on the LSB to land on a 1/4-pel grid.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector builder.</summary>
public static class Vp9MvBuilder
{
    /// <summary>
    /// Round an MV component toward zero on its LSB if
    /// <paramref name="allowHighPrecision"/> is false. No-op otherwise.
    /// Mirror of libvpx <c>lower_mv_precision</c>.
    /// </summary>
    public static int LowerMvPrecisionComponent(int component, bool allowHighPrecision)
    {
        if (allowHighPrecision) return component;
        if ((component & 1) == 0) return component;
        // Round TOWARD zero: positive odd subtracts 1, negative odd adds 1.
        return component > 0 ? component - 1 : component + 1;
    }

    /// <summary>
    /// Round both components of <paramref name="mv"/> per
    /// <see cref="LowerMvPrecisionComponent"/>.
    /// </summary>
    public static Vp9Mv LowerMvPrecision(Vp9Mv mv, bool allowHighPrecision)
    {
        if (allowHighPrecision) return mv;
        return new Vp9Mv(
            LowerMvPrecisionComponent(mv.Row, allowHighPrecision),
            LowerMvPrecisionComponent(mv.Col, allowHighPrecision));
    }

    /// <summary>
    /// Add an MV diff (<paramref name="vertDiff"/>, <paramref name="horizDiff"/>)
    /// to <paramref name="referenceMv"/>, then apply
    /// <see cref="LowerMvPrecision"/> and a final
    /// <see cref="Vp9Mv.Clamp(int, int, int, int)"/> to land on the legal range.
    /// </summary>
    public static Vp9Mv ApplyDiff(
        Vp9Mv referenceMv,
        int vertDiff,
        int horizDiff,
        bool allowHighPrecision)
    {
        int row = referenceMv.Row + vertDiff;
        int col = referenceMv.Col + horizDiff;
        var combined = new Vp9Mv(row, col);
        return LowerMvPrecision(combined, allowHighPrecision).Clamp();
    }
}
