// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 transform-size + transform-type enums + lookup tables.
// Mirrors libaom av1/common/enums.h (TX_SIZE / TX_TYPE) plus
// the per-tx-size width/height/log2 helper tables.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 6.4.1 Block size enums
//   sec 6.10.30 Transform type enums

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 TX_SIZE enum. Mirrors libaom <c>TX_SIZE</c> ordering.</summary>
public enum Av1TxSize : byte
{
    /// <summary>TX_4X4.</summary>
    Tx4x4 = 0,
    /// <summary>TX_8X8.</summary>
    Tx8x8 = 1,
    /// <summary>TX_16X16.</summary>
    Tx16x16 = 2,
    /// <summary>TX_32X32.</summary>
    Tx32x32 = 3,
    /// <summary>TX_64X64.</summary>
    Tx64x64 = 4,
    /// <summary>TX_4X8.</summary>
    Tx4x8 = 5,
    /// <summary>TX_8X4.</summary>
    Tx8x4 = 6,
    /// <summary>TX_8X16.</summary>
    Tx8x16 = 7,
    /// <summary>TX_16X8.</summary>
    Tx16x8 = 8,
    /// <summary>TX_16X32.</summary>
    Tx16x32 = 9,
    /// <summary>TX_32X16.</summary>
    Tx32x16 = 10,
    /// <summary>TX_32X64.</summary>
    Tx32x64 = 11,
    /// <summary>TX_64X32.</summary>
    Tx64x32 = 12,
    /// <summary>TX_4X16.</summary>
    Tx4x16 = 13,
    /// <summary>TX_16X4.</summary>
    Tx16x4 = 14,
    /// <summary>TX_8X32.</summary>
    Tx8x32 = 15,
    /// <summary>TX_32X8.</summary>
    Tx32x8 = 16,
    /// <summary>TX_16X64.</summary>
    Tx16x64 = 17,
    /// <summary>TX_64X16.</summary>
    Tx64x16 = 18,
    /// <summary>Sentinel for the number of TX sizes.</summary>
    TxSizesAll = 19,
    /// <summary>Number of square TX sizes.</summary>
    TxSizes = 5,
}

/// <summary>AV1 TX_TYPE enum. Mirrors libaom <c>TX_TYPE</c> ordering.</summary>
public enum Av1TxType : byte
{
    /// <summary>DCT  in both directions.</summary>
    DctDct = 0,
    /// <summary>ADST in vertical, DCT  in horizontal.</summary>
    AdstDct = 1,
    /// <summary>DCT  in vertical, ADST in horizontal.</summary>
    DctAdst = 2,
    /// <summary>ADST in both directions.</summary>
    AdstAdst = 3,
    /// <summary>FLIPADST in vertical, DCT  in horizontal.</summary>
    FlipAdstDct = 4,
    /// <summary>DCT      in vertical, FLIPADST in horizontal.</summary>
    DctFlipAdst = 5,
    /// <summary>FLIPADST in vertical, FLIPADST in horizontal.</summary>
    FlipAdstFlipAdst = 6,
    /// <summary>ADST     in vertical, FLIPADST in horizontal.</summary>
    AdstFlipAdst = 7,
    /// <summary>FLIPADST in vertical, ADST     in horizontal.</summary>
    FlipAdstAdst = 8,
    /// <summary>IDTX     in both directions.</summary>
    IdtxIdtx = 9,
    /// <summary>VERTICAL DCT, HORIZONTAL IDTX.</summary>
    VDct = 10,
    /// <summary>VERTICAL IDTX, HORIZONTAL DCT.</summary>
    HDct = 11,
    /// <summary>VERTICAL ADST, HORIZONTAL IDTX.</summary>
    VAdst = 12,
    /// <summary>VERTICAL IDTX, HORIZONTAL ADST.</summary>
    HAdst = 13,
    /// <summary>VERTICAL FLIPADST, HORIZONTAL IDTX.</summary>
    VFlipAdst = 14,
    /// <summary>VERTICAL IDTX, HORIZONTAL FLIPADST.</summary>
    HFlipAdst = 15,
    /// <summary>Sentinel for the number of TX types.</summary>
    TxTypes = 16,
}

/// <summary>
/// AV1 1D transform basis (DCT / ADST / FLIPADST / IDENTITY). Used to decompose
/// a 2D <see cref="Av1TxType"/> into its row + column 1D transforms.
/// </summary>
public enum Av1Tx1dType : byte
{
    /// <summary>Discrete Cosine Transform.</summary>
    Dct = 0,
    /// <summary>Asymmetric Discrete Sine Transform.</summary>
    Adst = 1,
    /// <summary>Flipped ADST (samples processed in reverse order).</summary>
    FlipAdst = 2,
    /// <summary>Identity transform (no transform).</summary>
    Identity = 3,
}

/// <summary>AV1 transform size + type lookup tables.</summary>
public static class Av1TxSizeInfo
{
    /// <summary>Transform width in pixels per <see cref="Av1TxSize"/>. Mirrors libaom <c>tx_size_wide</c>.</summary>
    public static readonly int[] TxWide = new int[]
    {
        4, 8, 16, 32, 64,        // square
        4, 8,                    // 4x8, 8x4
        8, 16,                   // 8x16, 16x8
        16, 32,                  // 16x32, 32x16
        32, 64,                  // 32x64, 64x32
        4, 16,                   // 4x16, 16x4
        8, 32,                   // 8x32, 32x8
        16, 64,                  // 16x64, 64x16
    };

    /// <summary>Transform height in pixels per <see cref="Av1TxSize"/>. Mirrors libaom <c>tx_size_high</c>.</summary>
    public static readonly int[] TxHigh = new int[]
    {
        4, 8, 16, 32, 64,
        8, 4,
        16, 8,
        32, 16,
        64, 32,
        16, 4,
        32, 8,
        64, 16,
    };

    /// <summary>log2(width). Mirrors libaom <c>tx_size_wide_log2</c>.</summary>
    public static readonly int[] TxWideLog2 = new int[]
    {
        2, 3, 4, 5, 6,
        2, 3,
        3, 4,
        4, 5,
        5, 6,
        2, 4,
        3, 5,
        4, 6,
    };

    /// <summary>log2(height). Mirrors libaom <c>tx_size_high_log2</c>.</summary>
    public static readonly int[] TxHighLog2 = new int[]
    {
        2, 3, 4, 5, 6,
        3, 2,
        4, 3,
        5, 4,
        6, 5,
        4, 2,
        5, 3,
        6, 4,
    };

    /// <summary>Largest square <see cref="Av1TxSize"/> per BLOCK_SIZE. Mirrors libaom <c>max_txsize_lookup</c>.</summary>
    public static readonly Av1TxSize[] MaxTxSize = new Av1TxSize[]
    {
        // 4x4, 4x8, 8x4, 8x8, 8x16, 16x8
        Av1TxSize.Tx4x4, Av1TxSize.Tx4x4, Av1TxSize.Tx4x4, Av1TxSize.Tx8x8, Av1TxSize.Tx8x8, Av1TxSize.Tx8x8,
        // 16x16, 16x32, 32x16, 32x32, 32x64, 64x32
        Av1TxSize.Tx16x16, Av1TxSize.Tx16x16, Av1TxSize.Tx16x16, Av1TxSize.Tx32x32, Av1TxSize.Tx32x32, Av1TxSize.Tx32x32,
        // 64x64, 64x128, 128x64, 128x128
        Av1TxSize.Tx64x64, Av1TxSize.Tx64x64, Av1TxSize.Tx64x64, Av1TxSize.Tx64x64,
        // 4x16, 16x4, 8x32, 32x8, 16x64, 64x16
        Av1TxSize.Tx4x4, Av1TxSize.Tx4x4, Av1TxSize.Tx8x8, Av1TxSize.Tx8x8, Av1TxSize.Tx16x16, Av1TxSize.Tx16x16,
    };

    /// <summary>Maximum rectangular <see cref="Av1TxSize"/> per BLOCK_SIZE. Mirrors libaom <c>max_txsize_rect_lookup</c>.</summary>
    public static readonly Av1TxSize[] MaxTxSizeRect = new Av1TxSize[]
    {
        // 4x4, 4x8, 8x4, 8x8, 8x16, 16x8
        Av1TxSize.Tx4x4, Av1TxSize.Tx4x8, Av1TxSize.Tx8x4, Av1TxSize.Tx8x8, Av1TxSize.Tx8x16, Av1TxSize.Tx16x8,
        // 16x16, 16x32, 32x16, 32x32, 32x64, 64x32
        Av1TxSize.Tx16x16, Av1TxSize.Tx16x32, Av1TxSize.Tx32x16, Av1TxSize.Tx32x32, Av1TxSize.Tx32x64, Av1TxSize.Tx64x32,
        // 64x64, 64x128, 128x64, 128x128
        Av1TxSize.Tx64x64, Av1TxSize.Tx64x64, Av1TxSize.Tx64x64, Av1TxSize.Tx64x64,
        // 4x16, 16x4, 8x32, 32x8, 16x64, 64x16
        Av1TxSize.Tx4x16, Av1TxSize.Tx16x4, Av1TxSize.Tx8x32, Av1TxSize.Tx32x8, Av1TxSize.Tx16x64, Av1TxSize.Tx64x16,
    };

    /// <summary>tx_size to square root of N (the 1D dimension assuming square). For non-square, returns max(w,h).</summary>
    public static int Get1dDim(Av1TxSize ts)
    {
        return Math.Max(TxWide[(int)ts], TxHigh[(int)ts]);
    }

    /// <summary>Number of coefficients in a transform block.</summary>
    public static int GetCoeffCount(Av1TxSize ts)
    {
        // TX_64X64 effectively only carries 32x32 coefs (high-freq zeroed by spec sec 7.7).
        int w = Math.Min(TxWide[(int)ts], 32);
        int h = Math.Min(TxHigh[(int)ts], 32);
        return w * h;
    }

    /// <summary>Decompose a 2D <see cref="Av1TxType"/> into its row 1D type. Mirrors libaom <c>tx_type_to_class</c> via vtx_tab.</summary>
    public static Av1Tx1dType GetRowType(Av1TxType tt) => RowTab[(int)tt];

    /// <summary>Decompose a 2D <see cref="Av1TxType"/> into its column 1D type.</summary>
    public static Av1Tx1dType GetColType(Av1TxType tt) => ColTab[(int)tt];

    /// <summary>libaom <c>vtx_tab</c>: column (vertical) basis per <see cref="Av1TxType"/>.</summary>
    private static readonly Av1Tx1dType[] ColTab = new Av1Tx1dType[]
    {
        Av1Tx1dType.Dct, Av1Tx1dType.Adst, Av1Tx1dType.Dct, Av1Tx1dType.Adst,
        Av1Tx1dType.FlipAdst, Av1Tx1dType.Dct, Av1Tx1dType.FlipAdst, Av1Tx1dType.Adst,
        Av1Tx1dType.FlipAdst, Av1Tx1dType.Identity, Av1Tx1dType.Dct, Av1Tx1dType.Identity,
        Av1Tx1dType.Adst, Av1Tx1dType.Identity, Av1Tx1dType.FlipAdst, Av1Tx1dType.Identity,
    };

    /// <summary>libaom <c>htx_tab</c>: row (horizontal) basis per <see cref="Av1TxType"/>.</summary>
    private static readonly Av1Tx1dType[] RowTab = new Av1Tx1dType[]
    {
        Av1Tx1dType.Dct, Av1Tx1dType.Dct, Av1Tx1dType.Adst, Av1Tx1dType.Adst,
        Av1Tx1dType.Dct, Av1Tx1dType.FlipAdst, Av1Tx1dType.FlipAdst, Av1Tx1dType.FlipAdst,
        Av1Tx1dType.Adst, Av1Tx1dType.Identity, Av1Tx1dType.Identity, Av1Tx1dType.Dct,
        Av1Tx1dType.Identity, Av1Tx1dType.Adst, Av1Tx1dType.Identity, Av1Tx1dType.FlipAdst,
    };
}
