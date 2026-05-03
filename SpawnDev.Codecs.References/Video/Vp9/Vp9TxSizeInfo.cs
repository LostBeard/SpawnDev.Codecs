// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 transform-size dimension helpers. Constant-time lookups for
// the four VP9 transform sizes:
//
//   Tx4x4   : side = 4,  coefs = 16
//   Tx8x8   : side = 8,  coefs = 64
//   Tx16x16 : side = 16, coefs = 256
//   Tx32x32 : side = 32, coefs = 1024
//
// libvpx reference: vp9/common/vp9_blockd.h <c>tx_size</c>-related
// macros (TX_SIZE_LOG2 conversions and num_coefs lookups).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 transform-size dimension helpers.</summary>
public static class Vp9TxSizeInfo
{
    /// <summary>Number of distinct transform sizes (libvpx <c>TX_SIZES</c>).</summary>
    public const int TxSizes = 4;

    /// <summary>
    /// Pixel side length per transform size.
    /// Indexed: 0=4, 1=8, 2=16, 3=32.
    /// </summary>
    public static readonly int[] SideLength = { 4, 8, 16, 32 };

    /// <summary>
    /// Coefficient count per transform block. Indexed:
    /// 0=16, 1=64, 2=256, 3=1024.
    /// </summary>
    public static readonly int[] CoefCounts = { 16, 64, 256, 1024 };

    /// <summary>Side length in pixels for <paramref name="size"/>.</summary>
    public static int Side(Vp9TxSize size)
    {
        int idx = (int)size;
        if ((uint)idx >= (uint)TxSizes)
            throw new ArgumentOutOfRangeException(nameof(size), size, "tx_size index out of range.");
        return SideLength[idx];
    }

    /// <summary>Number of coefficients (= side * side) for <paramref name="size"/>.</summary>
    public static int CoefCount(Vp9TxSize size)
    {
        int idx = (int)size;
        if ((uint)idx >= (uint)TxSizes)
            throw new ArgumentOutOfRangeException(nameof(size), size, "tx_size index out of range.");
        return CoefCounts[idx];
    }

    /// <summary>
    /// Log2 of side length. 4=2, 8=3, 16=4, 32=5. Useful when
    /// indexing log2-sized arrays (libvpx <c>tx_size_high_log2</c>).
    /// </summary>
    public static int Log2Side(Vp9TxSize size)
    {
        int idx = (int)size;
        if ((uint)idx >= (uint)TxSizes)
            throw new ArgumentOutOfRangeException(nameof(size), size, "tx_size index out of range.");
        return idx + 2;  // Tx4x4=0 -> log2(4)=2
    }
}
