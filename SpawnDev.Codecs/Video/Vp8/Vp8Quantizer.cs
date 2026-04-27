// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 quantizer / dequantizer. RFC 6386 sec 9.7 + Annex A. Maps the
// 7-bit Q index (0..127) to per-plane DC + AC dequantization values used
// to scale decoded coefficients before the inverse transform.
//
// Six per-plane dequant values per macroblock:
//   Y1 DC   = dc_qlookup[Qy + y_dc_delta_q]                clamp at 132 N/A
//   Y1 AC   = ac_qlookup[Qy]
//   Y2 DC   = dc_qlookup[Qy + y2_dc_delta_q] * 2
//   Y2 AC   = ac_qlookup[Qy + y2_ac_delta_q] * 155/100,    floor at 8
//   UV DC   = dc_qlookup[Qy + uv_dc_delta_q],              clamp at 132
//   UV AC   = ac_qlookup[Qy + uv_ac_delta_q]
//
// All Q indices clamp to [0, 127] after delta application.
//
// Reference: libvpx vp8/common/quant_common.c (vp8_dc_quant, vp8_dc2quant,
// vp8_dc_uv_quant, vp8_ac_yquant, vp8_ac2quant, vp8_ac_uv_quant).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 dequantizer. Q index 0..127 -> dequant value lookup.</summary>
public static class Vp8Quantizer
{
    /// <summary>QIndex valid range (libvpx QINDEX_RANGE).</summary>
    public const int QIndexMin = 0;
    /// <summary>QIndex valid range max.</summary>
    public const int QIndexMax = 127;
    /// <summary>UV DC dequant clamp ceiling (libvpx vp8_dc_uv_quant).</summary>
    public const int UvDcClamp = 132;
    /// <summary>Y2 AC dequant floor (libvpx vp8_ac2quant).</summary>
    public const int Y2AcFloor = 8;

    /// <summary>DC quantizer lookup table (libvpx dc_qlookup, 128 entries).</summary>
    public static readonly int[] DcQLookup = new int[128]
    {
          4,   5,   6,   7,   8,   9,  10,  10,  11,  12,  13,  14,  15,  16,  17,
         17,  18,  19,  20,  20,  21,  21,  22,  22,  23,  23,  24,  25,  25,  26,
         27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  37,  38,  39,  40,
         41,  42,  43,  44,  45,  46,  46,  47,  48,  49,  50,  51,  52,  53,  54,
         55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  91,  93,  95,  96,  98, 100, 101, 102, 104,
        106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130, 132, 134, 136,
        138, 140, 143, 145, 148, 151, 154, 157,
    };

    /// <summary>AC quantizer lookup table (libvpx ac_qlookup, 128 entries).</summary>
    public static readonly int[] AcQLookup = new int[128]
    {
          4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,
         19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
         34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
         49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  60,  62,  64,  66,  68,
         70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
        100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134,
        137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181,
        185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 234, 239, 245,
        249, 254, 259, 264, 269, 274, 279, 284,
    };

    /// <summary>Y1 DC dequant value. <paramref name="qIndex"/> + <paramref name="delta"/> clamped to [0,127].</summary>
    public static int Y1Dc(int qIndex, int delta = 0) => DcQLookup[ClampQ(qIndex + delta)];

    /// <summary>Y1 AC dequant value (no delta - VP8 spec doesn't define one for Y1 AC).</summary>
    public static int Y1Ac(int qIndex) => AcQLookup[ClampQ(qIndex)];

    /// <summary>Y2 (second-order) DC dequant value: dc_qlookup[Q+delta] * 2.</summary>
    public static int Y2Dc(int qIndex, int delta = 0) => DcQLookup[ClampQ(qIndex + delta)] * 2;

    /// <summary>
    /// Y2 (second-order) AC dequant value: ac_qlookup[Q+delta] * 155/100, floor at 8.
    /// libvpx uses the bit-equivalent <c>(x * 101581) >> 16</c> shortcut.
    /// </summary>
    public static int Y2Ac(int qIndex, int delta = 0)
    {
        int v = (AcQLookup[ClampQ(qIndex + delta)] * 101581) >> 16;
        return v < Y2AcFloor ? Y2AcFloor : v;
    }

    /// <summary>UV DC dequant value: dc_qlookup[Q+delta], clamped at 132.</summary>
    public static int UvDc(int qIndex, int delta = 0)
    {
        int v = DcQLookup[ClampQ(qIndex + delta)];
        return v > UvDcClamp ? UvDcClamp : v;
    }

    /// <summary>UV AC dequant value: ac_qlookup[Q+delta].</summary>
    public static int UvAc(int qIndex, int delta = 0) => AcQLookup[ClampQ(qIndex + delta)];

    private static int ClampQ(int qIndex) =>
        qIndex < QIndexMin ? QIndexMin : qIndex > QIndexMax ? QIndexMax : qIndex;
}
