// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 max-tx-size lookup. Maps each block size to the largest
// transform size allowed for that block (mirror of libvpx
// vp9/common/vp9_blockd.h max_txsize_lookup).
//
// The mapping is essentially "min(width, height) interpreted as
// tx size":
//   4x4 / 4x8 / 8x4               -> TX_4X4
//   8x8 / 8x16 / 16x8             -> TX_8X8
//   16x16 / 16x32 / 32x16         -> TX_16X16
//   32x32 / 32x64 / 64x32 / 64x64 -> TX_32X32

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 max transform size per block size.</summary>
public static class Vp9MaxTxSize
{
    /// <summary>
    /// Largest tx_size allowed per block size. Indexed by
    /// <see cref="Vp9BlockSize"/>. Mirror of libvpx
    /// <c>max_txsize_lookup</c>.
    /// </summary>
    public static readonly Vp9TxSize[] Lookup = new Vp9TxSize[Vp9BlockSizes.Count]
    {
        Vp9TxSize.Tx4x4,    // Block4x4
        Vp9TxSize.Tx4x4,    // Block4x8
        Vp9TxSize.Tx4x4,    // Block8x4
        Vp9TxSize.Tx8x8,    // Block8x8
        Vp9TxSize.Tx8x8,    // Block8x16
        Vp9TxSize.Tx8x8,    // Block16x8
        Vp9TxSize.Tx16x16,  // Block16x16
        Vp9TxSize.Tx16x16,  // Block16x32
        Vp9TxSize.Tx16x16,  // Block32x16
        Vp9TxSize.Tx32x32,  // Block32x32
        Vp9TxSize.Tx32x32,  // Block32x64
        Vp9TxSize.Tx32x32,  // Block64x32
        Vp9TxSize.Tx32x32,  // Block64x64
    };

    /// <summary>
    /// Largest tx_size for <paramref name="blockSize"/>.
    /// </summary>
    public static Vp9TxSize ForBlockSize(Vp9BlockSize blockSize)
    {
        int idx = (int)blockSize;
        if ((uint)idx >= (uint)Vp9BlockSizes.Count)
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize,
                "blockSize must be 0..12");
        return Lookup[idx];
    }
}
