// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 default intra-mode CDF tables (y mode + uv mode + kf y mode +
// angle delta + filter intra + cfl).
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom av1/common/entropymode.c
//   default_kf_y_mode_cdf       (lines 21-73)
//   default_angle_delta_cdf     (lines 75-85)
//   default_if_y_mode_cdf       (lines 87-95)
//   default_uv_mode_cdf         (lines 97-152)
//   default_cfl_sign_cdf        (line  405)
//   default_filter_intra_mode_cdf (line 818)
//   default_filter_intra_cdfs   (lines 821-829)
//
// Constants (av1/common/enums.h + av1/common/entropymode.h):
//   INTRA_MODES         = 13   (DC_PRED .. PAETH_PRED)
//   UV_INTRA_MODES      = 14   (adds UV_CFL_PRED)
//   KF_MODE_CONTEXTS    = 5
//   BLOCK_SIZE_GROUPS   = 4
//   CFL_ALLOWED_TYPES   = 2    (DISALLOWED, ALLOWED)
//   DIRECTIONAL_MODES   = 8
//   MAX_ANGLE_DELTA     = 3    (so 7 angle delta symbols)
//   FILTER_INTRA_MODES  = 5
//   BLOCK_SIZES_ALL     = 22
//   CFL_JOINT_SIGNS     = 8
//
// Stored as inverse CDF (ICDF: CDF_PROB_TOP - cumprob), padded with zeros to
// the libaom CDF_SIZE() row width. Compatible with Av1RangeDecoder.DecodeCdfQ15.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 default CDFs for intra-mode entropy decode. Tables are immutable; if
/// adaptation is needed (e.g. mid-frame CDF updates per the spec), copy a
/// row before mutating.
/// </summary>
internal static class Av1DefaultIntraModeCdfs
{
    /// <summary>
    /// <c>default_kf_y_mode_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS][CDF_SIZE(INTRA_MODES)]</c>.
    /// Indexed by [aboveContext][leftContext]. Symbol space: <see cref="Av1IntraMode"/> (13).
    /// </summary>
    public static readonly ushort[][][] DefaultKfYModeCdf = new ushort[][][]
    {
        new ushort[][]
        {
            new ushort[] { 17180, 15741, 13430, 12550, 12086, 11658, 10943, 9524, 8579, 4603, 3675, 2302, 0, 0 },
            new ushort[] { 20752, 14702, 13252, 12465, 12049, 11324, 10880, 9736, 8334, 4110, 2596, 1359, 0, 0 },
            new ushort[] { 22716, 21997, 10472, 9980, 9713, 9529, 8635, 7148, 6608, 3432, 2839, 1201, 0, 0 },
            new ushort[] { 18677, 17362, 16326, 13960, 13632, 13222, 12770, 10672, 8022, 3183, 1810, 306, 0, 0 },
            new ushort[] { 20646, 19503, 17165, 16267, 14159, 12735, 10377, 7185, 6331, 2507, 1695, 293, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 22745, 13183, 11920, 11328, 10936, 10008, 9679, 8745, 7387, 3754, 2286, 1332, 0, 0 },
            new ushort[] { 26785, 8669, 8208, 7882, 7702, 6973, 6855, 6345, 5158, 2863, 1492, 974, 0, 0 },
            new ushort[] { 25324, 19987, 12591, 12040, 11691, 11161, 10598, 9363, 8299, 4853, 3678, 2276, 0, 0 },
            new ushort[] { 24231, 18079, 17336, 15681, 15360, 14596, 14360, 12943, 8119, 3615, 1672, 558, 0, 0 },
            new ushort[] { 25225, 18537, 17272, 16573, 14863, 12051, 10784, 8252, 6767, 3093, 1787, 774, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 20155, 19177, 11385, 10764, 10456, 10191, 9367, 7713, 7039, 3230, 2463, 691, 0, 0 },
            new ushort[] { 23081, 19298, 14262, 13538, 13164, 12621, 12073, 10706, 9549, 5025, 3557, 1861, 0, 0 },
            new ushort[] { 26585, 26263, 6744, 6516, 6402, 6334, 5686, 4414, 4213, 2301, 1974, 682, 0, 0 },
            new ushort[] { 22050, 21034, 17814, 15544, 15203, 14844, 14207, 11245, 8890, 3793, 2481, 516, 0, 0 },
            new ushort[] { 23574, 22910, 16267, 15505, 14344, 13597, 11205, 6807, 6207, 2696, 2031, 305, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 20166, 18369, 17280, 14387, 13990, 13453, 13044, 11349, 7708, 3072, 1851, 359, 0, 0 },
            new ushort[] { 24565, 18947, 18244, 15663, 15329, 14637, 14364, 13300, 7543, 3283, 1610, 426, 0, 0 },
            new ushort[] { 24317, 23037, 17764, 15125, 14756, 14343, 13698, 11230, 8163, 3650, 2690, 750, 0, 0 },
            new ushort[] { 25054, 23720, 23252, 16101, 15951, 15774, 15615, 14001, 6025, 2379, 1232, 240, 0, 0 },
            new ushort[] { 23925, 22488, 21272, 17451, 16116, 14825, 13660, 10050, 6999, 2815, 1785, 283, 0, 0 },
        },
        new ushort[][]
        {
            new ushort[] { 20190, 19097, 16789, 15934, 13693, 11855, 9779, 7319, 6549, 2554, 1618, 291, 0, 0 },
            new ushort[] { 23205, 19142, 17688, 16876, 15012, 11905, 10561, 8532, 7388, 3115, 1625, 491, 0, 0 },
            new ushort[] { 24412, 23867, 15152, 14512, 13418, 12662, 10170, 6821, 6302, 2868, 2245, 507, 0, 0 },
            new ushort[] { 21933, 20953, 19644, 16726, 15750, 14729, 13821, 10015, 8153, 3279, 1885, 286, 0, 0 },
            new ushort[] { 25150, 24480, 22909, 22259, 17382, 14111, 9865, 3992, 3588, 1413, 966, 175, 0, 0 },
        },
    };

    /// <summary>
    /// <c>default_if_y_mode_cdf[BLOCK_SIZE_GROUPS][CDF_SIZE(INTRA_MODES)]</c>.
    /// Used for non-keyframe luma intra mode (inter frame). Indexed by block size group (0..3).
    /// </summary>
    public static readonly ushort[][] DefaultYModeCdf = new ushort[][]
    {
        new ushort[] { 9967, 9279, 8475, 8012, 7167, 6645, 6162, 5350, 4823, 3540, 3083, 2419, 0, 0 },
        new ushort[] { 14095, 12923, 10137, 9450, 8818, 8119, 7241, 5404, 4616, 3067, 2784, 1916, 0, 0 },
        new ushort[] { 12998, 11789, 9372, 8829, 8527, 8114, 7632, 5695, 4938, 3408, 3038, 2109, 0, 0 },
        new ushort[] { 12613, 11467, 9930, 9590, 9507, 9235, 9065, 7964, 7416, 6193, 5752, 4719, 0, 0 },
    };

    /// <summary>
    /// <c>default_uv_mode_cdf[CFL_ALLOWED_TYPES][INTRA_MODES][CDF_SIZE(UV_INTRA_MODES)]</c>.
    /// Indexed by [cflAllowed][lumaMode]. Without CFL: 13 active symbols. With CFL: 14 active symbols.
    /// </summary>
    public static readonly ushort[][][] DefaultUvModeCdf = new ushort[][][]
    {
        // CFL_DISALLOWED: 13 symbols (CDF13 -> CDF_SIZE(13)=14, padded to 15)
        new ushort[][]
        {
            new ushort[] { 10137, 8616, 7390, 7107, 6782, 6248, 5713, 4845, 4524, 2709, 1827, 807, 0, 0, 0 },
            new ushort[] { 23255, 5887, 5795, 5722, 5650, 5104, 5029, 4944, 4409, 3263, 2968, 972, 0, 0, 0 },
            new ushort[] { 22923, 22853, 4105, 4064, 4011, 3988, 3570, 2946, 2914, 2004, 991, 739, 0, 0, 0 },
            new ushort[] { 19129, 18871, 18597, 7437, 7162, 7041, 6815, 5620, 4191, 2156, 1413, 275, 0, 0, 0 },
            new ushort[] { 23004, 22933, 22838, 22814, 7382, 5715, 4810, 4620, 4525, 1667, 1024, 405, 0, 0, 0 },
            new ushort[] { 20943, 19179, 19091, 19048, 17720, 3555, 3467, 3310, 3057, 1607, 1327, 218, 0, 0, 0 },
            new ushort[] { 18593, 18369, 16160, 15947, 15050, 14993, 4217, 2568, 2523, 931, 426, 101, 0, 0, 0 },
            new ushort[] { 19883, 19730, 17790, 17178, 17095, 17020, 16592, 3640, 3501, 2125, 807, 307, 0, 0, 0 },
            new ushort[] { 20742, 19107, 18894, 17463, 17278, 17042, 16773, 16495, 4325, 2380, 2001, 352, 0, 0, 0 },
            new ushort[] { 13716, 12928, 12189, 11852, 11618, 11301, 10883, 10049, 9594, 3907, 2389, 593, 0, 0, 0 },
            new ushort[] { 14141, 13119, 11794, 11549, 11276, 10952, 10569, 9649, 9241, 5715, 1371, 620, 0, 0, 0 },
            new ushort[] { 15742, 13764, 12771, 12429, 12182, 11665, 11419, 10861, 10286, 6872, 6227, 949, 0, 0, 0 },
            new ushort[] { 20644, 19009, 17809, 17776, 17761, 17717, 17690, 17602, 17513, 17015, 16729, 16162, 0, 0, 0 },
        },
        // CFL_ALLOWED: 14 symbols (CDF14 -> CDF_SIZE(14)=15)
        new ushort[][]
        {
            new ushort[] { 22361, 21560, 19868, 19587, 18945, 18593, 17869, 17112, 16782, 12682, 11773, 10313, 8556, 0, 0 },
            new ushort[] { 28236, 12988, 12711, 12553, 12340, 11697, 11569, 11317, 10669, 8540, 8075, 5736, 3296, 0, 0 },
            new ushort[] { 27495, 27389, 12591, 12498, 12383, 12329, 11819, 11073, 10994, 9630, 8512, 8065, 6089, 0, 0 },
            new ushort[] { 26028, 25601, 25106, 18616, 18232, 17983, 17734, 16027, 14397, 11248, 10562, 9379, 8586, 0, 0 },
            new ushort[] { 27781, 27400, 26840, 26700, 13654, 12453, 10911, 10515, 10357, 7857, 7388, 6741, 6392, 0, 0 },
            new ushort[] { 27398, 25879, 25521, 25375, 23270, 11654, 11366, 11015, 10787, 7988, 7382, 6251, 5592, 0, 0 },
            new ushort[] { 27952, 27807, 25564, 25442, 24003, 23838, 12599, 12086, 11965, 9580, 9005, 8313, 7828, 0, 0 },
            new ushort[] { 26160, 26028, 24239, 23719, 23511, 23412, 23033, 13941, 13709, 10432, 9564, 8804, 7975, 0, 0 },
            new ushort[] { 26770, 25349, 24987, 23835, 23513, 23219, 23015, 22351, 13870, 10274, 9629, 8004, 6779, 0, 0 },
            new ushort[] { 22108, 21470, 20218, 19811, 19446, 19144, 18728, 17764, 17234, 12054, 10979, 9325, 7907, 0, 0 },
            new ushort[] { 22246, 21238, 20216, 19805, 19390, 18989, 18523, 17533, 16866, 12666, 10072, 8994, 6930, 0, 0 },
            new ushort[] { 22669, 22077, 20129, 19719, 19382, 19103, 18643, 17605, 17132, 13092, 12294, 9249, 7560, 0, 0 },
            new ushort[] { 29624, 27681, 25386, 25264, 25175, 25078, 24967, 24704, 24536, 23520, 22893, 22247, 3720, 0, 0 },
        },
    };

    /// <summary>
    /// <c>default_angle_delta_cdf[DIRECTIONAL_MODES][CDF_SIZE(2 * MAX_ANGLE_DELTA + 1)]</c>.
    /// Indexed by directional mode index (0..7); 7 active symbols (deltas -3..+3).
    /// </summary>
    public static readonly ushort[][] DefaultAngleDeltaCdf = new ushort[][]
    {
        new ushort[] { 30588, 27736, 25201, 9992, 5779, 2551, 0, 0 },
        new ushort[] { 30467, 27160, 23967, 9281, 5794, 2438, 0, 0 },
        new ushort[] { 28988, 21750, 19069, 13414, 9685, 1482, 0, 0 },
        new ushort[] { 28187, 21542, 17621, 15630, 10934, 4371, 0, 0 },
        new ushort[] { 31031, 21841, 18259, 13180, 10023, 3945, 0, 0 },
        new ushort[] { 30104, 22592, 20283, 15118, 11168, 2273, 0, 0 },
        new ushort[] { 30528, 21672, 17315, 12427, 10207, 3851, 0, 0 },
        new ushort[] { 29163, 22340, 20309, 15092, 11524, 2113, 0, 0 },
    };

    /// <summary>
    /// <c>default_filter_intra_mode_cdf[CDF_SIZE(FILTER_INTRA_MODES)]</c>.
    /// Single row, 5 active symbols (FILTER_DC, FILTER_V, FILTER_H, FILTER_D157, FILTER_PAETH).
    /// </summary>
    public static readonly ushort[] DefaultFilterIntraModeCdf =
        new ushort[] { 23819, 19992, 15557, 3210, 0, 0 };

    /// <summary>
    /// <c>default_filter_intra_cdfs[BLOCK_SIZES_ALL][CDF_SIZE(2)]</c>.
    /// Per-block-size 2-symbol CDF for whether to enable filter intra.
    /// </summary>
    public static readonly ushort[][] DefaultFilterIntraCdfs = new ushort[][]
    {
        new ushort[] { 28147, 0, 0 },
        new ushort[] { 26025, 0, 0 },
        new ushort[] { 26875, 0, 0 },
        new ushort[] { 24902, 0, 0 },
        new ushort[] { 20217, 0, 0 },
        new ushort[] { 23374, 0, 0 },
        new ushort[] { 20360, 0, 0 },
        new ushort[] { 18467, 0, 0 },
        new ushort[] { 20012, 0, 0 },
        new ushort[] { 10425, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 19998, 0, 0 },
        new ushort[] { 22400, 0, 0 },
        new ushort[] { 12539, 0, 0 },
        new ushort[] { 14667, 0, 0 },
        new ushort[] { 16384, 0, 0 },
        new ushort[] { 16384, 0, 0 },
    };

    /// <summary>
    /// <c>default_cfl_sign_cdf[CDF_SIZE(CFL_JOINT_SIGNS)]</c>.
    /// Single row, 8 active symbols for CFL_SIGNS joint encoding.
    /// </summary>
    public static readonly ushort[] DefaultCflSignCdf =
        new ushort[] { 31350, 30645, 19428, 14363, 5796, 4425, 474, 0, 0 };

    /// <summary>
    /// <c>default_cfl_alpha_cdf[CFL_ALPHA_CONTEXTS][CDF_SIZE(CFL_ALPHABET_SIZE)]</c>.
    /// 6 contexts (one per non-zero joint sign combo for U or V), 16 magnitudes.
    /// Indexed by libaom's CFL_CONTEXT(joint_sign, plane).
    /// </summary>
    public static readonly ushort[][] DefaultCflAlphaCdf = new ushort[][]
    {
        new ushort[] { 25131, 12049, 1367, 287, 111, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 0, 0 },
        new ushort[] { 18403, 9165, 4633, 1600, 601, 373, 281, 195, 148, 121, 100, 96, 92, 88, 84, 0, 0 },
        new ushort[] { 21236, 10388, 4323, 1408, 419, 245, 184, 119, 95, 91, 87, 83, 79, 75, 71, 0, 0 },
        new ushort[] { 5778, 1366, 486, 197, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 0, 0 },
        new ushort[] { 15520, 6710, 3864, 2160, 1463, 891, 642, 447, 374, 304, 252, 208, 192, 175, 146, 0, 0 },
        new ushort[] { 18030, 11090, 6989, 4867, 3744, 2466, 1788, 925, 624, 355, 248, 174, 146, 112, 108, 0, 0 },
    };
}
