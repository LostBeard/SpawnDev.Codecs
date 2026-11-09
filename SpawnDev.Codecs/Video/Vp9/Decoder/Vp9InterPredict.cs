// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inter prediction of one block of one plane from 8-bit reference planes, kernel-callable. Port of libvpx v1.17.0
// (BSD-3-Clause): vpx_dsp/vpx_convolve.c (convolve_horiz / convolve_vert / convolve_avg_*, vpx_convolve8 and _avg,
// _horiz, _vert, vpx_convolve_copy / _avg, vpx_scaled_2d / _horiz / _vert / _avg_*), vp9/common/vp9_filter.c (the
// EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP and BILINEAR kernels), vp9/common/vp9_scale.c
// (vp9_setup_scale_factors_for_frame, scaled_x / scaled_y, vp9_scale_mv, the predict[horiz][vert][avg] choice),
// vp9/common/vp9_reconinter.c (average_split_mvs, mi_mv_pred_q4 / _q2, round_mv_comp_q4 / _q2,
// clamp_mv_to_umv_border_sb) and vp9/decoder/vp9_decodeframe.c (dec_build_inter_predictors,
// dec_build_inter_predictors_sb, build_mc_border).
//
// Reference planes are stored WITHOUT borders. libvpx's decoder frame buffers have a 32 px border that the decoder
// never fills (it never calls vpx_extend_frame_borders); dec_build_inter_predictors instead copies the reference block
// through build_mc_border, which replicates the crop edge, whenever the filter window leaves the reference's crop
// rectangle and there is motion, scaling or a crop size that is not a multiple of 8. Every pixel build_mc_border
// produces is the reference pixel at the coordinate clamped to [0, crop - 1], and when libvpx does not extend, all
// reads of a block that influence a visible pixel lie inside the crop rectangle (proof in Fetch's comment). So this
// port reads every reference pixel at its clamped coordinate whenever the window leaves the crop rectangle, and
// directly otherwise. The one place libvpx differs: no motion, unscaled, crop sizes multiples of 8 and a block that
// hangs over the frame edge read libvpx's unfilled border, which only feeds predicted pixels outside the visible
// frame (scripts/video/libvpx_ref/inter_ref.c checks both facts against libvpx with a garbage-filled border).
//
// The 2-D convolve (vpx_convolve8_c) runs its two passes column by column: libvpx filters every intermediate row
// horizontally into a 64 x 135 byte buffer, then filters each column of it vertically; the intermediate value of row r,
// column x depends only on (r, x), so producing one column's intermediate_height values, filtering that column, then
// moving on gives the same bytes with a 135-entry scratch instead of 64 x 135. The _avg variants (convolve then
// ROUND_POWER_OF_TWO(dst + result, 1)) are fused per pixel, which is the same arithmetic as libvpx's temp + average.

using System.Runtime.CompilerServices;
using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>libvpx <c>struct scale_factors</c> without its function pointers (vp9_scale.h): a reference frame's size
/// relative to the frame being decoded. Computed once per reference per frame from the LUMA crop sizes and shared by all
/// planes.</summary>
internal struct Vp9ScaleFactors
{
    /// <summary>x_scale_fp: (ref crop width &lt;&lt; 14) / current width; <see cref="Vp9InterPred.RefNoScale"/> when equal,
    /// <see cref="Vp9InterPred.RefInvalidScale"/> when the sizes break valid_ref_frame_size (libvpx then fails the frame:
    /// "Reference frame has invalid dimensions").</summary>
    public int XScaleFp;
    /// <summary>y_scale_fp, as <see cref="XScaleFp"/> for heights.</summary>
    public int YScaleFp;
    /// <summary>x_step_q4 = scaled_x(16): horizontal step of the filter position per output pixel, 1/16 pel (16 unscaled,
    /// 32 for a reference twice as wide, 1 for one 16 times narrower).</summary>
    public int XStepQ4;
    /// <summary>y_step_q4 = scaled_y(16).</summary>
    public int YStepQ4;
}

/// <summary>One plane of a reference frame as the prediction reads it: a borderless 8-bit plane plus that reference's
/// scale factors.</summary>
internal struct Vp9InterRefPlane
{
    /// <summary>Index of the plane's pixel (0, 0) in the reference view (libvpx buf.y_buffer / u_buffer / v_buffer).</summary>
    public long Offset;
    /// <summary>Row pitch in bytes (libvpx y_stride / uv_stride); at least the crop width. Pixels right of / below the crop
    /// rectangle are never read.</summary>
    public int Stride;
    /// <summary>Crop width of THIS plane (libvpx y_crop_width, or uv_crop_width = (width + ss_x) &gt;&gt; ss_x): the
    /// frame_width of dec_build_inter_predictors, where border extension clamps.</summary>
    public int CropWidth;
    /// <summary>Crop height of this plane (y_crop_height / uv_crop_height).</summary>
    public int CropHeight;
    /// <summary>The reference's <see cref="Vp9ScaleFactors.XScaleFp"/> (the same value for all three planes).</summary>
    public int XScaleFp;
    /// <summary>The reference's <see cref="Vp9ScaleFactors.YScaleFp"/>.</summary>
    public int YScaleFp;
    /// <summary>The reference's <see cref="Vp9ScaleFactors.XStepQ4"/>.</summary>
    public int XStepQ4;
    /// <summary>The reference's <see cref="Vp9ScaleFactors.YStepQ4"/>.</summary>
    public int YStepQ4;
}

/// <summary>A VP9 filter kernel (libvpx InterpKernel): 8 taps summing to 128.</summary>
internal struct Vp9Kernel8
{
    public int T0, T1, T2, T3, T4, T5, T6, T7;
}

/// <summary>VP9 inter prediction (libvpx dec_build_inter_predictors_sb for one plane), 8-bit, kernel-callable: static
/// methods, no allocation, no recursion, no static arrays.</summary>
internal static class Vp9InterPred
{
    /// <summary>vpx_filter.h.</summary>
    public const int FilterBits = 7, SubpelBits = 4, SubpelMask = (1 << SubpelBits) - 1, SubpelShifts = 1 << SubpelBits, SubpelTaps = 8;
    /// <summary>vp9_scale.h.</summary>
    public const int RefScaleShift = 14, RefNoScale = 1 << RefScaleShift, RefInvalidScale = -1;
    /// <summary>VP9_INTERP_EXTEND (yv12config.h).</summary>
    public const int InterpExtend = 4;
    /// <summary>Scratch ints one <see cref="PredictBlockPlane"/> call needs: the largest intermediate_height of the 2-D
    /// convolve, ((64 - 1) * 32 + 15 &gt;&gt; 4) + 8 = 134, rounded to libvpx's 135 rows.</summary>
    public const int ScratchInts = 135;
    /// <summary>libvpx BLOCK_8X8: smaller sizes (BLOCK_4X4 = 0, BLOCK_4X8 = 1, BLOCK_8X4 = 2) predict per 4x4 sub-block.</summary>
    public const int Block8x8 = 3;

    // ---------------------------------------------------------------- vp9_scale.c

    /// <summary>scaled_x / scaled_y: (int)((int64_t)val * scaleFp &gt;&gt; REF_SCALE_SHIFT), in 32-bit arithmetic. With
    /// val = hi * 2^14 + lo (hi = val &gt;&gt; 14, 0 &lt;= lo &lt; 2^14), floor(val * fp / 2^14) = hi * fp + floor(lo * fp / 2^14)
    /// exactly, and lo * fp &lt; 2^14 * 2^17 fits an int for every valid scale (fp &lt;= 2 &lt;&lt; 14).</summary>
    public static int ScaleValue(int val, int scaleFp) => (val >> RefScaleShift) * scaleFp + (((val & (RefNoScale - 1)) * scaleFp) >> RefScaleShift);

    /// <summary>vp9_setup_scale_factors_for_frame(sf, other_w, other_h, this_w, this_h) without the function pointers:
    /// otherW/otherH = the reference's luma crop size, thisW/thisH = the current frame's (cm->width / cm->height).</summary>
    public static Vp9ScaleFactors SetupScaleFactors(int otherW, int otherH, int thisW, int thisH)
    {
        Vp9ScaleFactors sf;
        // valid_ref_frame_size
        if (!(2 * thisW >= otherW && 2 * thisH >= otherH && thisW <= 16 * otherW && thisH <= 16 * otherH))
        {
            sf.XScaleFp = RefInvalidScale;
            sf.YScaleFp = RefInvalidScale;
            sf.XStepQ4 = 0;
            sf.YStepQ4 = 0;
            return sf;
        }
        // get_fixed_point_scale_factor
        sf.XScaleFp = (otherW << RefScaleShift) / thisW;
        sf.YScaleFp = (otherH << RefScaleShift) / thisH;
        sf.XStepQ4 = ScaleValue(16, sf.XScaleFp);
        sf.YStepQ4 = ScaleValue(16, sf.YScaleFp);
        return sf;
    }

    /// <summary>vp9_is_valid_scale.</summary>
    public static bool IsValidScale(int xScaleFp, int yScaleFp) => xScaleFp != RefInvalidScale && yScaleFp != RefInvalidScale;

    /// <summary>vp9_is_scaled.</summary>
    public static bool IsScaled(int xScaleFp, int yScaleFp) =>
        IsValidScale(xScaleFp, yScaleFp) && (xScaleFp != RefNoScale || yScaleFp != RefNoScale);

    // ---------------------------------------------------------------- vp9_filter.c

    /// <summary>vp9_filter_kernels[filter][subpel] (filter: 0 EIGHTTAP, 1 EIGHTTAP_SMOOTH, 2 EIGHTTAP_SHARP, 3 BILINEAR).
    /// The tables live in two switches of packed bytes (tap + 64), so no static array reaches the kernel.</summary>
    public static Vp9Kernel8 Kernel(int filter, int subpel)
    {
        int index = (filter << SubpelBits) | subpel;
        uint lo = TapsLo(index), hi = TapsHi(index);
        Vp9Kernel8 k;
        k.T0 = (int)(lo & 0xFF) - 64;
        k.T1 = (int)((lo >> 8) & 0xFF) - 64;
        k.T2 = (int)((lo >> 16) & 0xFF) - 64;
        k.T3 = (int)(lo >> 24) - 64;
        k.T4 = (int)(hi & 0xFF) - 64;
        k.T5 = (int)((hi >> 8) & 0xFF) - 64;
        k.T6 = (int)((hi >> 16) & 0xFF) - 64;
        k.T7 = (int)(hi >> 24) - 64;
        return k;
    }

    // Generated from libvpx v1.17.0 vp9/common/vp9_filter.c: sub_pel_filters_8 (EIGHTTAP), sub_pel_filters_8lp
    // (EIGHTTAP_SMOOTH), sub_pel_filters_8s (EIGHTTAP_SHARP), bilinear_filters; index = filter * 16 + subpel, each byte
    // = tap + 64, tap 0 in the low byte.
    private static uint TapsLo(int index)
    {
        switch (index)
        {
            case 1: return 0xBE3B4140;   // sub_pel_filters_8[1] taps 0..3: 0, 1, -5, 126
            case 2: return 0xBA36433F;   // sub_pel_filters_8[2] taps 0..3: -1, 3, -10, 122
            case 3: return 0xB633443F;   // sub_pel_filters_8[3] taps 0..3: -1, 4, -13, 118
            case 4: return 0xB030443F;   // sub_pel_filters_8[4] taps 0..3: -1, 4, -16, 112
            case 5: return 0xA92E453F;   // sub_pel_filters_8[5] taps 0..3: -1, 5, -18, 105
            case 6: return 0xA12D453F;   // sub_pel_filters_8[6] taps 0..3: -1, 5, -19, 97
            case 7: return 0x982D463F;   // sub_pel_filters_8[7] taps 0..3: -1, 6, -19, 88
            case 8: return 0x8E2D463F;   // sub_pel_filters_8[8] taps 0..3: -1, 6, -19, 78
            case 9: return 0x842E453F;   // sub_pel_filters_8[9] taps 0..3: -1, 5, -18, 68
            case 10: return 0x7A30453F;   // sub_pel_filters_8[10] taps 0..3: -1, 5, -16, 58
            case 11: return 0x7032443F;   // sub_pel_filters_8[11] taps 0..3: -1, 4, -14, 48
            case 12: return 0x6535443F;   // sub_pel_filters_8[12] taps 0..3: -1, 4, -11, 37
            case 13: return 0x5B37433F;   // sub_pel_filters_8[13] taps 0..3: -1, 3, -9, 27
            case 14: return 0x523A4240;   // sub_pel_filters_8[14] taps 0..3: 0, 2, -6, 18
            case 15: return 0x483D4140;   // sub_pel_filters_8[15] taps 0..3: 0, 1, -3, 8
            case 17: return 0x80603F3D;   // sub_pel_filters_8lp[1] taps 0..3: -3, -1, 32, 64
            case 18: return 0x7F5D3E3E;   // sub_pel_filters_8lp[2] taps 0..3: -2, -2, 29, 63
            case 19: return 0x7F5A3E3E;   // sub_pel_filters_8lp[3] taps 0..3: -2, -2, 26, 63
            case 20: return 0x7E583D3E;   // sub_pel_filters_8lp[4] taps 0..3: -2, -3, 24, 62
            case 21: return 0x7C553D3E;   // sub_pel_filters_8lp[5] taps 0..3: -2, -3, 21, 60
            case 22: return 0x7B523C3F;   // sub_pel_filters_8lp[6] taps 0..3: -1, -4, 18, 59
            case 23: return 0x79503C3F;   // sub_pel_filters_8lp[7] taps 0..3: -1, -4, 16, 57
            case 24: return 0x774E3C3F;   // sub_pel_filters_8lp[8] taps 0..3: -1, -4, 14, 55
            case 25: return 0x754C3C3F;   // sub_pel_filters_8lp[9] taps 0..3: -1, -4, 12, 53
            case 26: return 0x73493C40;   // sub_pel_filters_8lp[10] taps 0..3: 0, -4, 9, 51
            case 27: return 0x71473C40;   // sub_pel_filters_8lp[11] taps 0..3: 0, -4, 7, 49
            case 28: return 0x6E453C40;   // sub_pel_filters_8lp[12] taps 0..3: 0, -4, 5, 46
            case 29: return 0x6B443C40;   // sub_pel_filters_8lp[13] taps 0..3: 0, -4, 4, 43
            case 30: return 0x69423D40;   // sub_pel_filters_8lp[14] taps 0..3: 0, -3, 2, 41
            case 31: return 0x66413D40;   // sub_pel_filters_8lp[15] taps 0..3: 0, -3, 1, 38
            case 33: return 0xBF39433F;   // sub_pel_filters_8s[1] taps 0..3: -1, 3, -7, 127
            case 34: return 0xBD33453E;   // sub_pel_filters_8s[2] taps 0..3: -2, 5, -13, 125
            case 35: return 0xB92F473D;   // sub_pel_filters_8s[3] taps 0..3: -3, 7, -17, 121
            case 36: return 0xB32C493C;   // sub_pel_filters_8s[4] taps 0..3: -4, 9, -20, 115
            case 37: return 0xAC294A3C;   // sub_pel_filters_8s[5] taps 0..3: -4, 10, -23, 108
            case 38: return 0xA4284A3C;   // sub_pel_filters_8s[6] taps 0..3: -4, 10, -24, 100
            case 39: return 0x9A284B3C;   // sub_pel_filters_8s[7] taps 0..3: -4, 11, -24, 90
            case 40: return 0x90294B3C;   // sub_pel_filters_8s[8] taps 0..3: -4, 11, -23, 80
            case 41: return 0x862B4A3C;   // sub_pel_filters_8s[9] taps 0..3: -4, 10, -21, 70
            case 42: return 0x7B2D493D;   // sub_pel_filters_8s[10] taps 0..3: -3, 9, -19, 59
            case 43: return 0x7030483D;   // sub_pel_filters_8s[11] taps 0..3: -3, 8, -16, 48
            case 44: return 0x6533463E;   // sub_pel_filters_8s[12] taps 0..3: -2, 6, -13, 37
            case 45: return 0x5B36453E;   // sub_pel_filters_8s[13] taps 0..3: -2, 5, -10, 27
            case 46: return 0x513A433F;   // sub_pel_filters_8s[14] taps 0..3: -1, 3, -6, 17
            case 47: return 0x483D4140;   // sub_pel_filters_8s[15] taps 0..3: 0, 1, -3, 8
            case 49: return 0xB8404040;   // bilinear_filters[1] taps 0..3: 0, 0, 0, 120
            case 50: return 0xB0404040;   // bilinear_filters[2] taps 0..3: 0, 0, 0, 112
            case 51: return 0xA8404040;   // bilinear_filters[3] taps 0..3: 0, 0, 0, 104
            case 52: return 0xA0404040;   // bilinear_filters[4] taps 0..3: 0, 0, 0, 96
            case 53: return 0x98404040;   // bilinear_filters[5] taps 0..3: 0, 0, 0, 88
            case 54: return 0x90404040;   // bilinear_filters[6] taps 0..3: 0, 0, 0, 80
            case 55: return 0x88404040;   // bilinear_filters[7] taps 0..3: 0, 0, 0, 72
            case 56: return 0x80404040;   // bilinear_filters[8] taps 0..3: 0, 0, 0, 64
            case 57: return 0x78404040;   // bilinear_filters[9] taps 0..3: 0, 0, 0, 56
            case 58: return 0x70404040;   // bilinear_filters[10] taps 0..3: 0, 0, 0, 48
            case 59: return 0x68404040;   // bilinear_filters[11] taps 0..3: 0, 0, 0, 40
            case 60: return 0x60404040;   // bilinear_filters[12] taps 0..3: 0, 0, 0, 32
            case 61: return 0x58404040;   // bilinear_filters[13] taps 0..3: 0, 0, 0, 24
            case 62: return 0x50404040;   // bilinear_filters[14] taps 0..3: 0, 0, 0, 16
            case 63: return 0x48404040;   // bilinear_filters[15] taps 0..3: 0, 0, 0, 8
            default: return 0xC0404040;   // sub-pel 0 of every filter: { 0, 0, 0, 128, 0, 0, 0, 0 }
        }
    }

    private static uint TapsHi(int index)
    {
        switch (index)
        {
            case 1: return 0x40413D48;   // sub_pel_filters_8[1] taps 4..7: 8, -3, 1, 0
            case 2: return 0x40423A52;   // sub_pel_filters_8[2] taps 4..7: 18, -6, 2, 0
            case 3: return 0x3F43375B;   // sub_pel_filters_8[3] taps 4..7: 27, -9, 3, -1
            case 4: return 0x3F443565;   // sub_pel_filters_8[4] taps 4..7: 37, -11, 4, -1
            case 5: return 0x3F443270;   // sub_pel_filters_8[5] taps 4..7: 48, -14, 4, -1
            case 6: return 0x3F45307A;   // sub_pel_filters_8[6] taps 4..7: 58, -16, 5, -1
            case 7: return 0x3F452E84;   // sub_pel_filters_8[7] taps 4..7: 68, -18, 5, -1
            case 8: return 0x3F462D8E;   // sub_pel_filters_8[8] taps 4..7: 78, -19, 6, -1
            case 9: return 0x3F462D98;   // sub_pel_filters_8[9] taps 4..7: 88, -19, 6, -1
            case 10: return 0x3F452DA1;   // sub_pel_filters_8[10] taps 4..7: 97, -19, 5, -1
            case 11: return 0x3F452EA9;   // sub_pel_filters_8[11] taps 4..7: 105, -18, 5, -1
            case 12: return 0x3F4430B0;   // sub_pel_filters_8[12] taps 4..7: 112, -16, 4, -1
            case 13: return 0x3F4433B6;   // sub_pel_filters_8[13] taps 4..7: 118, -13, 4, -1
            case 14: return 0x3F4336BA;   // sub_pel_filters_8[14] taps 4..7: 122, -10, 3, -1
            case 15: return 0x40413BBE;   // sub_pel_filters_8[15] taps 4..7: 126, -5, 1, 0
            case 17: return 0x403D4166;   // sub_pel_filters_8lp[1] taps 4..7: 38, 1, -3, 0
            case 18: return 0x403D4269;   // sub_pel_filters_8lp[2] taps 4..7: 41, 2, -3, 0
            case 19: return 0x403C446B;   // sub_pel_filters_8lp[3] taps 4..7: 43, 4, -4, 0
            case 20: return 0x403C456E;   // sub_pel_filters_8lp[4] taps 4..7: 46, 5, -4, 0
            case 21: return 0x403C4771;   // sub_pel_filters_8lp[5] taps 4..7: 49, 7, -4, 0
            case 22: return 0x403C4973;   // sub_pel_filters_8lp[6] taps 4..7: 51, 9, -4, 0
            case 23: return 0x3F3C4C75;   // sub_pel_filters_8lp[7] taps 4..7: 53, 12, -4, -1
            case 24: return 0x3F3C4E77;   // sub_pel_filters_8lp[8] taps 4..7: 55, 14, -4, -1
            case 25: return 0x3F3C5079;   // sub_pel_filters_8lp[9] taps 4..7: 57, 16, -4, -1
            case 26: return 0x3F3C527B;   // sub_pel_filters_8lp[10] taps 4..7: 59, 18, -4, -1
            case 27: return 0x3E3D557C;   // sub_pel_filters_8lp[11] taps 4..7: 60, 21, -3, -2
            case 28: return 0x3E3D587E;   // sub_pel_filters_8lp[12] taps 4..7: 62, 24, -3, -2
            case 29: return 0x3E3E5A7F;   // sub_pel_filters_8lp[13] taps 4..7: 63, 26, -2, -2
            case 30: return 0x3E3E5D7F;   // sub_pel_filters_8lp[14] taps 4..7: 63, 29, -2, -2
            case 31: return 0x3D3F6080;   // sub_pel_filters_8lp[15] taps 4..7: 64, 32, -1, -3
            case 33: return 0x40413D48;   // sub_pel_filters_8s[1] taps 4..7: 8, -3, 1, 0
            case 34: return 0x3F433A51;   // sub_pel_filters_8s[2] taps 4..7: 17, -6, 3, -1
            case 35: return 0x3E45365B;   // sub_pel_filters_8s[3] taps 4..7: 27, -10, 5, -2
            case 36: return 0x3E463365;   // sub_pel_filters_8s[4] taps 4..7: 37, -13, 6, -2
            case 37: return 0x3D483070;   // sub_pel_filters_8s[5] taps 4..7: 48, -16, 8, -3
            case 38: return 0x3D492D7B;   // sub_pel_filters_8s[6] taps 4..7: 59, -19, 9, -3
            case 39: return 0x3C4A2B86;   // sub_pel_filters_8s[7] taps 4..7: 70, -21, 10, -4
            case 40: return 0x3C4B2990;   // sub_pel_filters_8s[8] taps 4..7: 80, -23, 11, -4
            case 41: return 0x3C4B289A;   // sub_pel_filters_8s[9] taps 4..7: 90, -24, 11, -4
            case 42: return 0x3C4A28A4;   // sub_pel_filters_8s[10] taps 4..7: 100, -24, 10, -4
            case 43: return 0x3C4A29AC;   // sub_pel_filters_8s[11] taps 4..7: 108, -23, 10, -4
            case 44: return 0x3C492CB3;   // sub_pel_filters_8s[12] taps 4..7: 115, -20, 9, -4
            case 45: return 0x3D472FB9;   // sub_pel_filters_8s[13] taps 4..7: 121, -17, 7, -3
            case 46: return 0x3E4533BD;   // sub_pel_filters_8s[14] taps 4..7: 125, -13, 5, -2
            case 47: return 0x3F4339BF;   // sub_pel_filters_8s[15] taps 4..7: 127, -7, 3, -1
            case 49: return 0x40404048;   // bilinear_filters[1] taps 4..7: 8, 0, 0, 0
            case 50: return 0x40404050;   // bilinear_filters[2] taps 4..7: 16, 0, 0, 0
            case 51: return 0x40404058;   // bilinear_filters[3] taps 4..7: 24, 0, 0, 0
            case 52: return 0x40404060;   // bilinear_filters[4] taps 4..7: 32, 0, 0, 0
            case 53: return 0x40404068;   // bilinear_filters[5] taps 4..7: 40, 0, 0, 0
            case 54: return 0x40404070;   // bilinear_filters[6] taps 4..7: 48, 0, 0, 0
            case 55: return 0x40404078;   // bilinear_filters[7] taps 4..7: 56, 0, 0, 0
            case 56: return 0x40404080;   // bilinear_filters[8] taps 4..7: 64, 0, 0, 0
            case 57: return 0x40404088;   // bilinear_filters[9] taps 4..7: 72, 0, 0, 0
            case 58: return 0x40404090;   // bilinear_filters[10] taps 4..7: 80, 0, 0, 0
            case 59: return 0x40404098;   // bilinear_filters[11] taps 4..7: 88, 0, 0, 0
            case 60: return 0x404040A0;   // bilinear_filters[12] taps 4..7: 96, 0, 0, 0
            case 61: return 0x404040A8;   // bilinear_filters[13] taps 4..7: 104, 0, 0, 0
            case 62: return 0x404040B0;   // bilinear_filters[14] taps 4..7: 112, 0, 0, 0
            case 63: return 0x404040B8;   // bilinear_filters[15] taps 4..7: 120, 0, 0, 0
            default: return 0x40404040;   // sub-pel 0 of every filter: { 0, 0, 0, 128, 0, 0, 0, 0 }
        }
    }

    // ---------------------------------------------------------------- vpx_convolve.c

    /// <summary>clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)).</summary>
    private static int RoundClip(int sum)
    {
        int v = (sum + (1 << (FilterBits - 1))) >> FilterBits;
        return v < 0 ? 0 : v > 255 ? 255 : v;
    }

    /// <summary>Stores a predicted pixel: the plain convolves write it, the _avg ones write ROUND_POWER_OF_TWO(dst + v, 1).</summary>
    private static void Store(ArrayView<byte> dst, long at, int v, bool avg)
    {
        dst[at] = (byte)(avg ? (dst[at] + v + 1) >> 1 : v);
    }

    /// <summary>Reference pixel (x, y) of a plane. With <paramref name="edge"/> the coordinate is clamped to the crop
    /// rectangle: exactly the bytes build_mc_border copies (left / right runs of ref_row[0] / ref_row[w - 1], rows above 0
    /// read row 0, rows from h on read row h - 1). Without it the caller has proven the window inside the crop
    /// rectangle: libvpx's own test (x0 &gt;= 0, x1 &lt;= frame_width - 1 on the padded window, x1 computed from the
    /// 1/16-pel position x0_16 &gt;= 16 * x0 + subpel_x, so the last tap read, x0 + ((subpel_x + (w - 1) * xs) &gt;&gt; 4) + 4,
    /// is at most x1 - 1).</summary>
    private static int Fetch(ArrayView<byte> src, in Vp9InterRefPlane p, int x, int y, bool edge)
    {
        if (edge)
        {
            x = x < 0 ? 0 : x > p.CropWidth - 1 ? p.CropWidth - 1 : x;
            y = y < 0 ? 0 : y > p.CropHeight - 1 ? p.CropHeight - 1 : y;
        }
        return src[p.Offset + (long)y * p.Stride + x];
    }

    /// <summary>The 8-tap dot product of row <paramref name="y"/> from column <paramref name="x"/> (the first tap).</summary>
    private static int DotRow(ArrayView<byte> src, in Vp9InterRefPlane p, int x, int y, bool edge, in Vp9Kernel8 k) =>
        Fetch(src, p, x, y, edge) * k.T0 + Fetch(src, p, x + 1, y, edge) * k.T1 + Fetch(src, p, x + 2, y, edge) * k.T2 +
        Fetch(src, p, x + 3, y, edge) * k.T3 + Fetch(src, p, x + 4, y, edge) * k.T4 + Fetch(src, p, x + 5, y, edge) * k.T5 +
        Fetch(src, p, x + 6, y, edge) * k.T6 + Fetch(src, p, x + 7, y, edge) * k.T7;

    /// <summary>The 8-tap dot product of column <paramref name="x"/> from row <paramref name="y"/> (the first tap).</summary>
    private static int DotCol(ArrayView<byte> src, in Vp9InterRefPlane p, int x, int y, bool edge, in Vp9Kernel8 k) =>
        Fetch(src, p, x, y, edge) * k.T0 + Fetch(src, p, x, y + 1, edge) * k.T1 + Fetch(src, p, x, y + 2, edge) * k.T2 +
        Fetch(src, p, x, y + 3, edge) * k.T3 + Fetch(src, p, x, y + 4, edge) * k.T4 + Fetch(src, p, x, y + 5, edge) * k.T5 +
        Fetch(src, p, x, y + 6, edge) * k.T6 + Fetch(src, p, x, y + 7, edge) * k.T7;

    /// <summary>vpx_convolve_copy_c / vpx_convolve_avg_c: the w x h block at (x0, y0).</summary>
    private static void ConvolveCopy(ArrayView<byte> src, in Vp9InterRefPlane p, int x0, int y0, bool edge,
        ArrayView<byte> dst, long dstAt, int dstStride, int w, int h, bool avg)
    {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                Store(dst, dstAt + (long)y * dstStride + x, Fetch(src, p, x0 + x, y0 + y, edge), avg);
    }

    /// <summary>convolve_horiz / convolve_avg_horiz (vpx_convolve8_horiz, _avg_horiz, vpx_scaled_horiz, _avg_horiz):
    /// rows y0.., output x filtered at 1/16-pel position x0_q4 + x * x_step_q4 (src -= SUBPEL_TAPS / 2 - 1).</summary>
    private static void ConvolveHoriz(ArrayView<byte> src, in Vp9InterRefPlane p, int x0, int y0, bool edge, int filter,
        int x0Q4, int xStepQ4, ArrayView<byte> dst, long dstAt, int dstStride, int w, int h, bool avg)
    {
        for (int y = 0; y < h; y++)
        {
            int xQ4 = x0Q4;
            for (int x = 0; x < w; x++)
            {
                var k = Kernel(filter, xQ4 & SubpelMask);
                int sum = DotRow(src, p, x0 - (SubpelTaps / 2 - 1) + (xQ4 >> SubpelBits), y0 + y, edge, k);
                Store(dst, dstAt + (long)y * dstStride + x, RoundClip(sum), avg);
                xQ4 += xStepQ4;
            }
        }
    }

    /// <summary>convolve_vert / convolve_avg_vert (vpx_convolve8_vert, _avg_vert, vpx_scaled_vert, _avg_vert): columns
    /// x0.., output y filtered at 1/16-pel position y0_q4 + y * y_step_q4 (src -= src_stride * (SUBPEL_TAPS / 2 - 1)).</summary>
    private static void ConvolveVert(ArrayView<byte> src, in Vp9InterRefPlane p, int x0, int y0, bool edge, int filter,
        int y0Q4, int yStepQ4, ArrayView<byte> dst, long dstAt, int dstStride, int w, int h, bool avg)
    {
        for (int x = 0; x < w; x++)
        {
            int yQ4 = y0Q4;
            for (int y = 0; y < h; y++)
            {
                var k = Kernel(filter, yQ4 & SubpelMask);
                int sum = DotCol(src, p, x0 + x, y0 - (SubpelTaps / 2 - 1) + (yQ4 >> SubpelBits), edge, k);
                Store(dst, dstAt + (long)y * dstStride + x, RoundClip(sum), avg);
                yQ4 += yStepQ4;
            }
        }
    }

    /// <summary>vpx_convolve8_c / vpx_convolve8_avg_c (also vpx_scaled_2d / _avg_2d): horizontal pass over
    /// intermediate_height = (((h - 1) * y_step_q4 + y0_q4) &gt;&gt; SUBPEL_BITS) + SUBPEL_TAPS rows from y0 - 3 into byte
    /// values, then the vertical pass over them; done column by column (see the file header).</summary>
    private static void Convolve2D(ArrayView<byte> src, in Vp9InterRefPlane p, int x0, int y0, bool edge, int filter,
        int x0Q4, int xStepQ4, int y0Q4, int yStepQ4, ArrayView<byte> dst, long dstAt, int dstStride, int w, int h, bool avg,
        ArrayView<int> temp, long tempAt)
    {
        int intermediateHeight = (((h - 1) * yStepQ4 + y0Q4) >> SubpelBits) + SubpelTaps;
        int xQ4 = x0Q4;
        for (int x = 0; x < w; x++)
        {
            var kx = Kernel(filter, xQ4 & SubpelMask);
            int col = x0 - (SubpelTaps / 2 - 1) + (xQ4 >> SubpelBits);
            for (int r = 0; r < intermediateHeight; r++)
                temp[tempAt + r] = RoundClip(DotRow(src, p, col, y0 - (SubpelTaps / 2 - 1) + r, edge, kx));
            int yQ4 = y0Q4;
            for (int y = 0; y < h; y++)
            {
                var ky = Kernel(filter, yQ4 & SubpelMask);
                long t = tempAt + (yQ4 >> SubpelBits);
                int sum = temp[t] * ky.T0 + temp[t + 1] * ky.T1 + temp[t + 2] * ky.T2 + temp[t + 3] * ky.T3 +
                          temp[t + 4] * ky.T4 + temp[t + 5] * ky.T5 + temp[t + 6] * ky.T6 + temp[t + 7] * ky.T7;
                Store(dst, dstAt + (long)y * dstStride + x, RoundClip(sum), avg);
                yQ4 += yStepQ4;
            }
            xQ4 += xStepQ4;
        }
    }

    // ---------------------------------------------------------------- vp9_reconinter.c

    private static int MvRow(int mv) => (short)(mv & 0xFFFF);
    private static int MvCol(int mv) => mv >> 16;
    private static int PackMv(int row, int col) => (row & 0xFFFF) | (col << 16);

    /// <summary>round_mv_comp_q4 (C division truncates, as C# does).</summary>
    private static int RoundMvCompQ4(int value) => (value < 0 ? value - 2 : value + 2) / 4;

    /// <summary>round_mv_comp_q2.</summary>
    private static int RoundMvCompQ2(int value) => (value < 0 ? value - 1 : value + 1) / 2;

    /// <summary>average_split_mvs(pd, mi, ref, block): the MV of sub-8x8 block <paramref name="block"/> in a plane with
    /// subsampling (ssX, ssY); bmiAt = the block's bmi[0].as_mv[ref] (bmi[i] at bmiAt + i).</summary>
    public static int AverageSplitMv(ArrayView<int> mi, long bmiAt, int ssX, int ssY, int block)
    {
        int ssIdx = ((ssX > 0 ? 1 : 0) << 1) | (ssY > 0 ? 1 : 0);
        if (ssIdx == 0) return mi[bmiAt + block];
        if (ssIdx == 3)
        {
            // mi_mv_pred_q4
            int a = mi[bmiAt], b = mi[bmiAt + 1], c = mi[bmiAt + 2], d = mi[bmiAt + 3];
            return PackMv(RoundMvCompQ4(MvRow(a) + MvRow(b) + MvRow(c) + MvRow(d)),
                          RoundMvCompQ4(MvCol(a) + MvCol(b) + MvCol(c) + MvCol(d)));
        }
        // mi_mv_pred_q2(mi, ref, block, block + 2) for ss_idx 1 (4:4:0), (block, block + 1) for ss_idx 2 (4:2:2)
        int m0 = mi[bmiAt + block], m1 = mi[bmiAt + block + (ssIdx == 1 ? 2 : 1)];
        return PackMv(RoundMvCompQ2(MvRow(m0) + MvRow(m1)), RoundMvCompQ2(MvCol(m0) + MvCol(m1)));
    }

    // ---------------------------------------------------------------- vp9_decodeframe.c

    /// <summary>num_8x8_blocks_wide_lookup[bsize].</summary>
    private static int Num8x8Wide(int bsize) => bsize < 5 ? 1 : bsize < 8 ? 2 : bsize < 11 ? 4 : 8;

    /// <summary>num_8x8_blocks_high_lookup[bsize]: BLOCK_4X4 .. BLOCK_64X64 = 1,1,1,1,2,1,2,4,2,4,8,4,8.</summary>
    private static int Num8x8High(int bsize) =>
        bsize < 4 ? 1 : bsize == 4 ? 2 : bsize == 5 ? 1 : bsize == 6 ? 2 : bsize == 7 ? 4 : bsize == 8 ? 2 : bsize == 9 ? 4 : bsize == 10 ? 8 : bsize == 11 ? 4 : 8;

    /// <summary>
    /// dec_build_inter_predictors_sb for ONE plane of one inter block (the caller loops the planes): predicts every
    /// reference of the block into <paramref name="dst"/>, the second (compound) reference averaged in with the _avg
    /// convolves; blocks below 8x8 predict each 4x4 of the plane with its own MV (average_split_mvs in the chroma planes).
    /// </summary>
    /// <param name="ssX">This plane's subsampling_x (0 for luma; the frame's for chroma).</param>
    /// <param name="ssY">This plane's subsampling_y.</param>
    /// <param name="miRow">The block's mi_row (8x8 luma units).</param>
    /// <param name="miCol">The block's mi_col.</param>
    /// <param name="miRows">The current frame's cm->mi_rows ((height + 7) &gt;&gt; 3); with miCols they give
    /// set_mi_row_col's mb_to_top / bottom / left / right_edge (1/8 pel).</param>
    /// <param name="miCols">cm->mi_cols ((width + 7) &gt;&gt; 3).</param>
    /// <param name="mi">Mode-info records in the parse kernel's layout (<see cref="Vp9Mi"/>).</param>
    /// <param name="miAt">The block's record base (b * Vp9Mi.Stride). Read: SbType (libvpx BLOCK_SIZE; n4_w / n4_h of this
    /// plane = (num_8x8_wide &lt;&lt; 1) &gt;&gt; ssX, ...), Interp (0 EIGHTTAP, 1 EIGHTTAP_SMOOTH, 2 EIGHTTAP_SHARP,
    /// 3 BILINEAR), Ref1 (&gt; 0 = compound, has_second_ref), Mv0 / Mv1 (mi-&gt;mv[ref].as_int: row in the low 16 bits, col
    /// in the high 16, 1/8 luma pel) and, below BLOCK_8X8, BmiMv0 / BmiMv1 + 0..3 (bmi[i].as_mv[ref], duplicated for
    /// 4x8 / 8x4 as read_inter_block_mode_info stores them).</param>
    /// <param name="ref0">The first reference's pixels (mi-&gt;ref_frame[0]).</param>
    /// <param name="ref0Plane">This plane of the first reference: offset, stride, crop size, scale factors.</param>
    /// <param name="ref1">The second reference's pixels (mi-&gt;ref_frame[1]); not read unless compound.</param>
    /// <param name="ref1Plane">This plane of the second reference; not read unless compound.</param>
    /// <param name="dst">Destination plane (libvpx pd-&gt;dst).</param>
    /// <param name="dstAt">Index of the block's top-left pixel in <paramref name="dst"/>.</param>
    /// <param name="dstStride">Destination row pitch. The whole n4_w * 4 x n4_h * 4 block is written, also where it hangs
    /// over the frame edge (as libvpx writes it).</param>
    /// <param name="scratch">Per-call scratch of <see cref="ScratchInts"/> ints (the 2-D convolve's intermediate column).</param>
    /// <param name="scratchAt">Start of this call's scratch.</param>
    /// <remarks>The scale factors must be valid (libvpx stops the frame with VPX_CODEC_UNSUP_BITSTREAM otherwise).</remarks>
    // A real function on Wasm (and on the WebGPU / WebGL backends once ILGPU emits view-taking helpers as functions),
    // not inlined at every call site: keeps the kernels small enough for the browsers' shader and Wasm compilers.
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void PredictBlockPlane(int ssX, int ssY, int miRow, int miCol, int miRows, int miCols,
        ArrayView<int> mi, long miAt,
        ArrayView<byte> ref0, Vp9InterRefPlane ref0Plane, ArrayView<byte> ref1, Vp9InterRefPlane ref1Plane,
        ArrayView<byte> dst, long dstAt, int dstStride, ArrayView<int> scratch, long scratchAt)
    {
        int bsize = mi[miAt + Vp9Mi.SbType];
        int filter = mi[miAt + Vp9Mi.Interp];
        bool compound = mi[miAt + Vp9Mi.Ref1] > Vp9Mi.IntraFrame;
        int bw = Num8x8Wide(bsize), bh = Num8x8High(bsize);
        // set_mi_row_col
        int toTop = -((miRow * 8) * 8), toBottom = ((miRows - bh - miRow) * 8) * 8;
        int toLeft = -((miCol * 8) * 8), toRight = ((miCols - bw - miCol) * 8) * 8;
        PredictRef(ssX, ssY, miRow, miCol, bsize, bw, bh, toLeft, toRight, toTop, toBottom, filter, mi, miAt, 0,
            ref0, ref0Plane, dst, dstAt, dstStride, scratch, scratchAt);
        if (compound)
            PredictRef(ssX, ssY, miRow, miCol, bsize, bw, bh, toLeft, toRight, toTop, toBottom, filter, mi, miAt, 1,
                ref1, ref1Plane, dst, dstAt, dstStride, scratch, scratchAt);
    }

    /// <summary>The body of dec_build_inter_predictors_sb's loop for one reference and one plane.</summary>
    private static void PredictRef(int ssX, int ssY, int miRow, int miCol, int bsize, int bw, int bh,
        int toLeft, int toRight, int toTop, int toBottom, int filter, ArrayView<int> mi, long miAt, int refIdx,
        ArrayView<byte> src, Vp9InterRefPlane p, ArrayView<byte> dst, long dstAt, int dstStride, ArrayView<int> scratch, long scratchAt)
    {
        int n4w = (bw << 1) >> ssX, n4h = (bh << 1) >> ssY;   // set_plane_n4
        int n4wX4 = 4 * n4w, n4hX4 = 4 * n4h;
        int miX = miCol * 8, miY = miRow * 8;
        bool avg = refIdx == 1;
        if (bsize < Block8x8)
        {
            long bmiAt = miAt + Vp9Mi.BmiMv0 + 4 * refIdx;
            int i = 0;
            for (int y = 0; y < n4h; y++)
                for (int x = 0; x < n4w; x++)
                {
                    int mv = AverageSplitMv(mi, bmiAt, ssX, ssY, i++);
                    PredictOne(src, p, ssX, ssY, n4wX4, n4hX4, 4 * x, 4 * y, 4, 4, miX, miY, toLeft, toRight, toTop, toBottom,
                        filter, MvRow(mv), MvCol(mv), avg, dst, dstAt, dstStride, scratch, scratchAt);
                }
        }
        else
        {
            int mv = mi[miAt + Vp9Mi.Mv0 + refIdx];
            PredictOne(src, p, ssX, ssY, n4wX4, n4hX4, 0, 0, n4wX4, n4hX4, miX, miY, toLeft, toRight, toTop, toBottom,
                filter, MvRow(mv), MvCol(mv), avg, dst, dstAt, dstStride, scratch, scratchAt);
        }
    }

    /// <summary>dec_build_inter_predictors(plane, bw, bh, x, y, w, h, mi_x, mi_y, kernel, sf, mv, is_scaled, ref): the
    /// w x h block at (x, y) inside the bw x bh plane block, MV (mvRow, mvCol) in 1/8 luma pel.</summary>
    private static void PredictOne(ArrayView<byte> src, in Vp9InterRefPlane p, int ssX, int ssY, int bw, int bh,
        int x, int y, int w, int h, int miX, int miY, int toLeft, int toRight, int toTop, int toBottom, int filter,
        int mvRow, int mvCol, bool avg, ArrayView<byte> dst, long dstAt, int dstStride, ArrayView<int> scratch, long scratchAt)
    {
        int x0, y0, x0_16, y0_16, smvRow, smvCol, xs, ys;
        if (IsScaled(p.XScaleFp, p.YScaleFp))
        {
            // clamp_mv_to_umv_border_sb
            int spelLeft = (InterpExtend + bw) << SubpelBits, spelRight = spelLeft - SubpelShifts;
            int spelTop = (InterpExtend + bh) << SubpelBits, spelBottom = spelTop - SubpelShifts;
            int row = (short)(mvRow * (1 << (1 - ssY))), col = (short)(mvCol * (1 << (1 - ssX)));
            int minCol = toLeft * (1 << (1 - ssX)) - spelLeft, maxCol = toRight * (1 << (1 - ssX)) + spelRight;
            int minRow = toTop * (1 << (1 - ssY)) - spelTop, maxRow = toBottom * (1 << (1 - ssY)) + spelBottom;
            col = (short)(col < minCol ? minCol : col > maxCol ? maxCol : col);
            row = (short)(row < minRow ? minRow : row > maxRow ? maxRow : row);
            // co-ordinate of the containing block to pixel precision
            int xStart = -toLeft >> (3 + ssX), yStart = -toTop >> (3 + ssY);
            x0_16 = ScaleValue((xStart + x) << SubpelBits, p.XScaleFp);
            y0_16 = ScaleValue((yStart + y) << SubpelBits, p.YScaleFp);
            x0 = ScaleValue(xStart + x, p.XScaleFp);
            y0 = ScaleValue(yStart + y, p.YScaleFp);
            // vp9_scale_mv(&mv_q4, mi_x + x, mi_y + y, sf)
            int xOffQ4 = ScaleValue((miX + x) << SubpelBits, p.XScaleFp) & SubpelMask;
            int yOffQ4 = ScaleValue((miY + y) << SubpelBits, p.YScaleFp) & SubpelMask;
            smvRow = ScaleValue(row, p.YScaleFp) + yOffQ4;
            smvCol = ScaleValue(col, p.XScaleFp) + xOffQ4;
            xs = p.XStepQ4;
            ys = p.YStepQ4;
        }
        else
        {
            x0 = (-toLeft >> (3 + ssX)) + x;
            y0 = (-toTop >> (3 + ssY)) + y;
            x0_16 = x0 << SubpelBits;
            y0_16 = y0 << SubpelBits;
            smvRow = mvRow * (1 << (1 - ssY));
            smvCol = mvCol * (1 << (1 - ssX));
            xs = SubpelShifts;
            ys = SubpelShifts;
        }
        int subpelX = smvCol & SubpelMask, subpelY = smvRow & SubpelMask;
        x0 += smvCol >> SubpelBits;
        y0 += smvRow >> SubpelBits;
        x0_16 += smvCol;
        y0_16 += smvRow;

        // The window libvpx tests before build_mc_border (padded by the filter taps where it filters).
        int x1 = ((x0_16 + (w - 1) * xs) >> SubpelBits) + 1, y1 = ((y0_16 + (h - 1) * ys) >> SubpelBits) + 1;
        int wx0 = x0, wy0 = y0;
        if (subpelX != 0 || p.XStepQ4 != SubpelShifts) { wx0 -= InterpExtend - 1; x1 += InterpExtend; }
        if (subpelY != 0 || p.YStepQ4 != SubpelShifts) { wy0 -= InterpExtend - 1; y1 += InterpExtend; }
        bool edge = wx0 < 0 || wx0 > p.CropWidth - 1 || x1 < 0 || x1 > p.CropWidth - 1 ||
                    wy0 < 0 || wy0 > p.CropHeight - 1 || y1 < 0 || y1 > p.CropHeight - 1;

        long at = dstAt + (long)y * dstStride + x;
        // sf->predict[subpel_x != 0][subpel_y != 0][ref] as vp9_setup_scale_factors_for_frame fills it
        bool scaleX = p.XStepQ4 != SubpelShifts, scaleY = p.YStepQ4 != SubpelShifts;
        bool fx = subpelX != 0, fy = subpelY != 0;
        if (!scaleX && !scaleY)
        {
            if (!fx && !fy) ConvolveCopy(src, p, x0, y0, edge, dst, at, dstStride, w, h, avg);
            else if (!fy) ConvolveHoriz(src, p, x0, y0, edge, filter, subpelX, xs, dst, at, dstStride, w, h, avg);
            else if (!fx) ConvolveVert(src, p, x0, y0, edge, filter, subpelY, ys, dst, at, dstStride, w, h, avg);
            else Convolve2D(src, p, x0, y0, edge, filter, subpelX, xs, subpelY, ys, dst, at, dstStride, w, h, avg, scratch, scratchAt);
        }
        else if (!scaleX && !fx)   // x unscaled, y scaled: [0][*] = vpx_scaled_vert
            ConvolveVert(src, p, x0, y0, edge, filter, subpelY, ys, dst, at, dstStride, w, h, avg);
        else if (!scaleY && !fy)   // y unscaled, x scaled: [*][0] = vpx_scaled_horiz
            ConvolveHoriz(src, p, x0, y0, edge, filter, subpelX, xs, dst, at, dstStride, w, h, avg);
        else                       // vpx_scaled_2d
            Convolve2D(src, p, x0, y0, edge, filter, subpelX, xs, subpelY, ys, dst, at, dstStride, w, h, avg, scratch, scratchAt);
    }
}
