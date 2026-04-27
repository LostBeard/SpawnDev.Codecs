// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point forward DCT (1D). Bit-exact port of libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct4.
//
// AV1 forward transforms compose as: column 1D + row 1D + per-axis
// shift + interleave. The 1D fdct here is the building block for fdct4x4
// (apply twice: once on rows, once on cols, with 2 shifts in between).
//
// Cosine constants from libaom av1_txfm.c av1_cospi_arr_data[4][64]:
// indexed by [cos_bit - 10][i] gives cos(pi*i/64) * 2^cos_bit.
// AV1 fdct4 uses cos_bit = 13 by default in the AV1 forward txfm config.
//
// half_btf primitive: round_shift(w0*in0 + w1*in1, bit) - libaom inline.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 4-point forward DCT (1D building block).</summary>
public static class Av1ForwardDct4
{
    /// <summary>Default cosine-precision bits for fdct4 (libaom).</summary>
    public const int DefaultCosBit = 13;

    /// <summary>
    /// libaom <c>av1_cospi_arr_data</c> for cos_bit=10..13. Indexed
    /// [cos_bit - 10][i] gives cos(pi*i/64) * 2^cos_bit.
    /// </summary>
    public static readonly int[][] CospiArrData = new int[][]
    {
        // cos_bit = 10
        new int[] {
            1024, 1024, 1023, 1021, 1019, 1016, 1013, 1009, 1004, 999, 993, 987, 980,
            972,  964,  955,  946,  936,  926,  915,  903,  891,  878, 865, 851, 837,
            822,  807,  792,  775,  759,  742,  724,  706,  688,  669, 650, 630, 610,
            590,  569,  548,  526,  505,  483,  460,  438,  415,  392, 369, 345, 321,
            297,  273,  249,  224,  200,  175,  150,  125,  100,  75,  50,  25,
        },
        // cos_bit = 11
        new int[] {
            2048, 2047, 2046, 2042, 2038, 2033, 2026, 2018, 2009, 1998, 1987,
            1974, 1960, 1945, 1928, 1911, 1892, 1872, 1851, 1829, 1806, 1782,
            1757, 1730, 1703, 1674, 1645, 1615, 1583, 1551, 1517, 1483, 1448,
            1412, 1375, 1338, 1299, 1260, 1220, 1179, 1138, 1096, 1053, 1009,
             965,  921,  876,  830,  784,  737,  690,  642,  595,  546,  498,
             449,  400,  350,  301,  251,  201,  151,  100,   50,
        },
        // cos_bit = 12
        new int[] {
            4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
            3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
            3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
            2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
            1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092,  995,
             897,  799,  700,  601,  501,  401,  301,  201,  101,
        },
        // cos_bit = 13
        new int[] {
            8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946,
            7895, 7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128,
            7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793,
            5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038,
            3862, 3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990,
            1795, 1598, 1401, 1202, 1003,  803,  603,  402,  201,
        },
    };

    /// <summary>libaom <c>cospi_arr(cos_bit)</c> - returns the 64-element cospi vector.</summary>
    public static int[] CospiArr(int cosBit) => CospiArrData[cosBit - 10];

    /// <summary>
    /// libaom inline <c>half_btf(w0, in0, w1, in1, bit) =
    /// round_shift(w0*in0 + w1*in1, bit)</c>.
    /// </summary>
    public static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// 4-point forward DCT. Mirrors libaom <c>av1_fdct4</c>. Three stages
    /// of butterfly + cospi multiplications.
    /// </summary>
    public static void Transform(ReadOnlySpan<int> input, Span<int> output, int cosBit = DefaultCosBit)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");

        // Stage 1
        Span<int> stage1 = stackalloc int[4];
        stage1[0] = input[0] + input[3];
        stage1[1] = input[1] + input[2];
        stage1[2] = -input[2] + input[1];
        stage1[3] = -input[3] + input[0];

        // Stage 2
        var cospi = CospiArr(cosBit);
        Span<int> stage2 = stackalloc int[4];
        stage2[0] = HalfBtf(cospi[32], stage1[0], cospi[32], stage1[1], cosBit);
        stage2[1] = HalfBtf(-cospi[32], stage1[1], cospi[32], stage1[0], cosBit);
        stage2[2] = HalfBtf(cospi[48], stage1[2], cospi[16], stage1[3], cosBit);
        stage2[3] = HalfBtf(cospi[48], stage1[3], -cospi[16], stage1[2], cosBit);

        // Stage 3 (interleave)
        output[0] = stage2[0];
        output[1] = stage2[2];
        output[2] = stage2[1];
        output[3] = stage2[3];
    }
}
