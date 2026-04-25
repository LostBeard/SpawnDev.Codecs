// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 MV-to-sub-pel split helpers. Inter prediction operates in
// Q4 fixed-point (1/16-pel resolution): the convolve walker needs
// (integer pixel, sub-pel index 0..15) to dispatch the correct
// filter row.
//
// VP9 stores MV components in 1/8-pel resolution (or 1/4-pel when
// allow_high_precision_mv is false; bit 0 always carries a half-pel
// flag, so even a "1/8-pel" MV with bit 0 = 0 is essentially
// 1/4-pel). Convert to Q4 by left-shifting one bit (1/8 -> 1/16).
//
// libvpx reference: inline conversions in vp9/common/vp9_reconinter.c
// (e.g. <c>const MV mv_q4 = ...</c> or the equivalent in
// <c>vpx_convolve8</c> dispatch sites).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 MV component to (integer pixel, sub-pel) split.</summary>
public static class Vp9MvSubPel
{
    /// <summary>
    /// Convert an MV component in 1/8-pel units to Q4 (1/16-pel) by
    /// left-shifting one bit. The libvpx convolve dispatch operates
    /// in Q4.
    /// </summary>
    public static int OneEighthPelToQ4(int mvComponentEighth) => mvComponentEighth << 1;

    /// <summary>
    /// Split a Q4 fixed-point position (1/16-pel resolution) into
    /// the integer pixel offset and the 0..15 sub-pel index.
    ///
    /// Arithmetic shift right preserves negative integer offsets:
    /// <c>(-1) &gt;&gt; 4 = -1</c>, <c>(-1) &amp; 15 = 15</c>, so a Q4
    /// of -1 splits as (-1 pel, sub-pel 15).
    /// </summary>
    public static (int IntegerPel, int SubPel) Split(int positionQ4)
    {
        int integerPel = positionQ4 >> Vp9SubPelFilters.SubPelBits;
        int subPel = positionQ4 & (Vp9SubPelFilters.SubPelShifts - 1);
        return (integerPel, subPel);
    }

    /// <summary>
    /// Combine an integer pixel offset and a sub-pel index into a
    /// Q4 fixed-point position. Inverse of <see cref="Split"/>.
    /// </summary>
    public static int Combine(int integerPel, int subPel)
    {
        if ((uint)subPel >= (uint)Vp9SubPelFilters.SubPelShifts)
            throw new ArgumentOutOfRangeException(nameof(subPel), subPel,
                "subPel must be in [0, 16).");
        return (integerPel << Vp9SubPelFilters.SubPelBits) | subPel;
    }
}
