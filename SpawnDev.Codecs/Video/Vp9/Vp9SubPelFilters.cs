// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 sub-pel interpolation filter tap tables. Bit-exact copy of
// libvpx vp9/common/vp9_filter.c. Used by inter-prediction motion
// compensation to produce 1/8-pel and 1/4-pel reference samples.
//
// Layout: each filter set is a 16 x 8 table:
//   - 16 sub-pel positions (SUBPEL_BITS = 4 -> SUBPEL_SHIFTS = 16)
//   - 8 taps per position (SUBPEL_TAPS = 8)
//
// Stored row-major as a flat int16[128]. Position i has taps at
//   indices [i*8 .. i*8 + 7], multiplying source samples
//   [src[-3], src[-2], src[-1], src[0], src[+1], src[+2], src[+3], src[+4]].
// The filter sums to 128 (FILTER_BITS = 7); after multiplication
// the integer convolution is rounded to the nearest 8-bit value.
//
// Filter indices follow libvpx INTERP_FILTER:
//   0 EightTap        (regular)  -> sub_pel_filters_8
//   1 EightTapSmooth  (low-pass) -> sub_pel_filters_8lp
//   2 EightTapSharp              -> sub_pel_filters_8s
//   3 Bilinear                   -> sub_pel_filters_4 (4-tap zero-padded
//                                   into 8-tap storage; outer two taps
//                                   are zero)

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 sub-pel interpolation filter tap tables.</summary>
public static class Vp9SubPelFilters
{
    /// <summary>libvpx <c>SUBPEL_BITS</c> = 4.</summary>
    public const int SubPelBits = 4;

    /// <summary>libvpx <c>SUBPEL_SHIFTS</c> = 16 (2^SUBPEL_BITS).</summary>
    public const int SubPelShifts = 1 << SubPelBits;

    /// <summary>libvpx <c>SUBPEL_TAPS</c> = 8.</summary>
    public const int SubPelTaps = 8;

    /// <summary>libvpx <c>FILTER_BITS</c> = 7. Filter sum / rounding.</summary>
    public const int FilterBits = 7;

    /// <summary>
    /// EightTap regular filter. libvpx <c>sub_pel_filters_8</c>.
    /// </summary>
    public static readonly short[] EightTap = new short[SubPelShifts * SubPelTaps]
    {
         0,  0,   0, 128,   0,   0,  0,  0,
         0,  1,  -5, 126,   8,  -3,  1,  0,
        -1,  3, -10, 122,  18,  -6,  2,  0,
        -1,  4, -13, 118,  27,  -9,  3, -1,
        -1,  4, -16, 112,  37, -11,  4, -1,
        -1,  5, -18, 105,  48, -14,  4, -1,
        -1,  5, -19,  97,  58, -16,  5, -1,
        -1,  6, -19,  88,  68, -18,  5, -1,
        -1,  6, -19,  78,  78, -19,  6, -1,
        -1,  5, -18,  68,  88, -19,  6, -1,
        -1,  5, -16,  58,  97, -19,  5, -1,
        -1,  4, -14,  48, 105, -18,  5, -1,
        -1,  4, -11,  37, 112, -16,  4, -1,
        -1,  3,  -9,  27, 118, -13,  4, -1,
         0,  2,  -6,  18, 122, -10,  3, -1,
         0,  1,  -3,   8, 126,  -5,  1,  0,
    };

    /// <summary>
    /// EightTap smooth (low-pass) filter. libvpx
    /// <c>sub_pel_filters_8lp</c>.
    /// </summary>
    public static readonly short[] EightTapSmooth = new short[SubPelShifts * SubPelTaps]
    {
         0,  0,  0, 128,   0,  0,  0,  0,
        -3, -1, 32,  64,  38,  1, -3,  0,
        -2, -2, 29,  63,  41,  2, -3,  0,
        -2, -2, 26,  63,  43,  4, -4,  0,
        -2, -3, 24,  62,  46,  5, -4,  0,
        -2, -3, 21,  60,  49,  7, -4,  0,
        -1, -4, 18,  59,  51,  9, -4,  0,
        -1, -4, 16,  57,  53, 12, -4, -1,
        -1, -4, 14,  55,  55, 14, -4, -1,
        -1, -4, 12,  53,  57, 16, -4, -1,
         0, -4,  9,  51,  59, 18, -4, -1,
         0, -4,  7,  49,  60, 21, -3, -2,
         0, -4,  5,  46,  62, 24, -3, -2,
         0, -4,  4,  43,  63, 26, -2, -2,
         0, -3,  2,  41,  63, 29, -2, -2,
         0, -3,  1,  38,  64, 32, -1, -3,
    };

    /// <summary>
    /// EightTap sharp filter. libvpx <c>sub_pel_filters_8s</c>.
    /// </summary>
    public static readonly short[] EightTapSharp = new short[SubPelShifts * SubPelTaps]
    {
         0,  0,   0, 128,   0,   0,  0,  0,
        -1,  3,  -7, 127,   8,  -3,  1,  0,
        -2,  5, -13, 125,  17,  -6,  3, -1,
        -3,  7, -17, 121,  27, -10,  5, -2,
        -4,  9, -20, 115,  37, -13,  6, -2,
        -4, 10, -23, 108,  48, -16,  8, -3,
        -4, 10, -24, 100,  59, -19,  9, -3,
        -4, 11, -24,  90,  70, -21, 10, -4,
        -4, 11, -23,  80,  80, -23, 11, -4,
        -4, 10, -21,  70,  90, -24, 11, -4,
        -3,  9, -19,  59, 100, -24, 10, -4,
        -3,  8, -16,  48, 108, -23, 10, -4,
        -2,  6, -13,  37, 115, -20,  9, -4,
        -2,  5, -10,  27, 121, -17,  7, -3,
        -1,  3,  -6,  17, 125, -13,  5, -2,
         0,  1,  -3,   8, 127,  -7,  3, -1,
    };

    /// <summary>
    /// Bilinear filter (4-tap zero-padded into 8-tap storage).
    /// libvpx <c>sub_pel_filters_4</c>.
    /// </summary>
    public static readonly short[] Bilinear = new short[SubPelShifts * SubPelTaps]
    {
        0, 0,   0, 128,   0,   0, 0, 0,
        0, 0,  -4, 126,   8,  -2, 0, 0,
        0, 0,  -6, 120,  18,  -4, 0, 0,
        0, 0,  -8, 114,  28,  -6, 0, 0,
        0, 0, -10, 108,  36,  -6, 0, 0,
        0, 0, -12, 102,  46,  -8, 0, 0,
        0, 0, -12,  94,  56, -10, 0, 0,
        0, 0, -12,  84,  66, -10, 0, 0,
        0, 0, -12,  76,  76, -12, 0, 0,
        0, 0, -10,  66,  84, -12, 0, 0,
        0, 0, -10,  56,  94, -12, 0, 0,
        0, 0,  -8,  46, 102, -12, 0, 0,
        0, 0,  -6,  36, 108, -10, 0, 0,
        0, 0,  -6,  28, 114,  -8, 0, 0,
        0, 0,  -4,  18, 120,  -6, 0, 0,
        0, 0,  -2,   8, 126,  -4, 0, 0,
    };

    /// <summary>
    /// Look up the filter table for a given <see cref="Vp9InterpFilter"/>.
    /// </summary>
    /// <remarks>
    /// <see cref="Vp9InterpFilter.Switchable"/> is a per-block selector
    /// at the frame level - it does not have its own tap table; pass
    /// the resolved per-block filter (one of EightTap / EightTapSmooth
    /// / EightTapSharp / Bilinear) instead.
    /// </remarks>
    public static short[] GetFilter(Vp9InterpFilter filter) => filter switch
    {
        Vp9InterpFilter.EightTap       => EightTap,
        Vp9InterpFilter.EightTapSmooth => EightTapSmooth,
        Vp9InterpFilter.EightTapSharp  => EightTapSharp,
        Vp9InterpFilter.Bilinear       => Bilinear,
        _ => throw new ArgumentOutOfRangeException(nameof(filter), filter,
            "Vp9SubPelFilters.GetFilter cannot resolve Switchable - pass the per-block resolved filter."),
    };

    /// <summary>
    /// Get the 8-tap row for a given filter and sub-pel position.
    /// </summary>
    public static ReadOnlySpan<short> GetRow(Vp9InterpFilter filter, int subPel)
    {
        if ((uint)subPel >= (uint)SubPelShifts)
            throw new ArgumentOutOfRangeException(nameof(subPel), subPel,
                "Sub-pel position must be in [0, SubPelShifts).");
        return GetFilter(filter).AsSpan(subPel * SubPelTaps, SubPelTaps);
    }
}
