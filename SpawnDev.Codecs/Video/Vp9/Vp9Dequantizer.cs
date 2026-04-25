// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 dequantization - bit-exact CPU reference for libvpx
// vp9_dc_quant() / vp9_ac_quant() and the per-coefficient
// dequantize step that bridges the entropy decoder and the
// inverse transform kernels.
//
// VP9 uses two normative 256-entry lookup tables (one for the DC
// coefficient, one for the AC coefficients) indexed by a frame-
// level quantizer index plus optional per-plane / per-segment
// delta. Each block's dequantized coefficient is just
// `input_coeff * dequant_value`, where dequant_value comes from
// the DC table for position 0 of the coefficient scan and from
// the AC table for every other position.
//
// Spec: VP9 Bitstream Specification sec 8.6.1 "Quantization".
// libvpx reference: vp9/common/vp9_quant_common.c (dc_qlookup,
// ac_qlookup) and vp9_dequantize.c (dequantize_b_q1).
//
// This slice covers Profile 0 (8-bit) only. 10 / 12-bit profile
// dequant tables follow the same shape (just different values)
// and land alongside the high-bit-depth pipeline in a future slice.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Quantizer pair for a single plane - DC for the first scan
/// position, AC for every subsequent scan position.
/// </summary>
public readonly record struct Vp9PlaneQuantizer(short Dc, short Ac);

/// <summary>
/// CPU reference for VP9 dequantization. Bit-exact against
/// libvpx <c>vp9_dc_quant</c> / <c>vp9_ac_quant</c> and
/// <c>dequantize_b_q1</c>.
/// </summary>
public static class Vp9Dequantizer
{
    /// <summary>VP9 8-bit DC quantizer lookup table (sec 8.6.1).</summary>
    public static readonly short[] DcQLookup8 = new short[]
    {
        4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19,
        20, 21, 22, 23, 24, 25, 26, 26, 27, 28, 29, 30, 31, 32, 32, 33,
        34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47,
        48, 48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60,
        61, 62, 62, 63, 64, 65, 66, 66, 67, 68, 69, 70, 70, 71, 72, 73,
        74, 74, 75, 76, 77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85,
        87, 88, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110,
        111, 113, 114, 116, 117, 118, 120, 121, 123, 125, 127, 129, 131, 134, 136, 138,
        140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 161, 164, 166, 169, 172, 174,
        177, 180, 182, 185, 187, 190, 192, 195, 199, 202, 205, 208, 211, 214, 217, 220,
        223, 226, 230, 233, 237, 240, 243, 247, 250, 253, 257, 261, 265, 269, 272, 276,
        280, 284, 288, 292, 296, 300, 304, 309, 313, 317, 322, 326, 330, 335, 340, 344,
        349, 354, 359, 364, 369, 374, 379, 384, 389, 395, 400, 406, 411, 417, 423, 429,
        435, 441, 447, 454, 461, 467, 475, 482, 489, 497, 505, 513, 522, 530, 539, 549,
        559, 569, 579, 590, 602, 614, 626, 640, 654, 668, 684, 700, 717, 736, 755, 775,
        796, 819, 843, 869, 896, 925, 955, 988, 1022, 1058, 1098, 1139, 1184, 1232, 1282, 1336,
    };

    /// <summary>VP9 8-bit AC quantizer lookup table (sec 8.6.1).</summary>
    public static readonly short[] AcQLookup8 = new short[]
    {
        4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
        87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
        104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134,
        136, 138, 140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173,
        176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227,
        231, 235, 239, 243, 247, 251, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300,
        305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366, 373, 380, 387, 394, 401,
        408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520, 530, 540,
        550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729,
        743, 757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988,
        1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343,
        1369, 1396, 1423, 1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
    };

    /// <summary>
    /// Look up the DC quantizer at <paramref name="qIndex"/> + <paramref name="delta"/>,
    /// clamped to [0, 255]. Equivalent to libvpx <c>vp9_dc_quant</c> for 8-bit profiles.
    /// </summary>
    public static short DcQuant(int qIndex, int delta)
    {
        int idx = qIndex + delta;
        if (idx < 0) idx = 0;
        else if (idx > 255) idx = 255;
        return DcQLookup8[idx];
    }

    /// <summary>
    /// Look up the AC quantizer at <paramref name="qIndex"/> + <paramref name="delta"/>,
    /// clamped to [0, 255]. Equivalent to libvpx <c>vp9_ac_quant</c> for 8-bit profiles.
    /// </summary>
    public static short AcQuant(int qIndex, int delta)
    {
        int idx = qIndex + delta;
        if (idx < 0) idx = 0;
        else if (idx > 255) idx = 255;
        return AcQLookup8[idx];
    }

    /// <summary>
    /// Build the DC + AC quantizer pair for a plane from the frame
    /// base q_index and any per-plane delta.
    /// </summary>
    public static Vp9PlaneQuantizer PlaneQuantizer(int qIndex, int dcDelta, int acDelta)
    {
        return new Vp9PlaneQuantizer(
            Dc: DcQuant(qIndex, dcDelta),
            Ac: AcQuant(qIndex, acDelta));
    }

    /// <summary>
    /// Multiply each coefficient in <paramref name="coefficients"/> by its
    /// dequantizer value. The first coefficient (scan position 0) uses
    /// <paramref name="quant"/>.Dc; every subsequent coefficient uses
    /// <paramref name="quant"/>.Ac. Result is written in place.
    /// </summary>
    /// <remarks>
    /// VP9 stores coefficients in scan order (the same order produced by
    /// the boolean entropy decoder + scan-table indirection); the DC
    /// coefficient is always at scan-position 0 regardless of the 2D
    /// layout. The dequantization step is independent of the scan order
    /// because it indexes by scan position, not raster position.
    /// </remarks>
    public static void DequantizeInPlace(Span<short> coefficients, Vp9PlaneQuantizer quant)
    {
        if (coefficients.Length == 0) return;
        coefficients[0] = SaturatingMul(coefficients[0], quant.Dc);
        for (int i = 1; i < coefficients.Length; i++)
            coefficients[i] = SaturatingMul(coefficients[i], quant.Ac);
    }

    /// <summary>
    /// Multiply a coefficient by a dequantizer with int-domain math, then
    /// clamp to int16. VP9 quantized coefficients are int16 and dequantizer
    /// values are also int16; the product fits in int32 and the spec requires
    /// the result to be stored as int16 (libvpx <c>clip_pixel</c>-style on
    /// the coefficient domain).
    /// </summary>
    private static short SaturatingMul(short coeff, short dequant)
    {
        int product = coeff * dequant;
        if (product > short.MaxValue) return short.MaxValue;
        if (product < short.MinValue) return short.MinValue;
        return (short)product;
    }
}
