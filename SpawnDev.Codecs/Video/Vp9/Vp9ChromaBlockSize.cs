// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 chroma block size lookup for 4:2:0 sampling. Mirror of
// libvpx vp9/common/vp9_common_data.c <c>ss_size_lookup</c>
// at (ss_x = 1, ss_y = 1).
//
// 4:2:0 chroma planes are half-resolution in both axes, so the
// chroma block for a luma block size is the corresponding
// half-size. Sub-8x8 luma blocks all collapse to the smallest
// chroma size (Block4x4) since chroma can't go below that.
//
// VP9 Profile 0 mandates 4:2:0; 4:2:2 (Profile 2) / 4:4:4 (Profile
// 3) require additional block sizes (BLOCK_4X16, BLOCK_16X4, etc.)
// that aren't in the 13-entry Vp9BlockSize enum. Those land in a
// future slice alongside the high-bit-depth pipeline.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 4:2:0 chroma block size lookup.</summary>
public static class Vp9ChromaBlockSize
{
    /// <summary>
    /// libvpx <c>ss_size_lookup[bsize][1][1]</c>: chroma block size
    /// for 4:2:0 sampling, indexed by luma block size.
    /// </summary>
    public static readonly Vp9BlockSize[] For420 = new Vp9BlockSize[Vp9BlockSizes.Count]
    {
        Vp9BlockSize.Block4x4,    // luma Block4x4    -> chroma Block4x4
        Vp9BlockSize.Block4x4,    // luma Block4x8    -> chroma Block4x4
        Vp9BlockSize.Block4x4,    // luma Block8x4    -> chroma Block4x4
        Vp9BlockSize.Block4x4,    // luma Block8x8    -> chroma Block4x4
        Vp9BlockSize.Block4x8,    // luma Block8x16   -> chroma Block4x8
        Vp9BlockSize.Block8x4,    // luma Block16x8   -> chroma Block8x4
        Vp9BlockSize.Block8x8,    // luma Block16x16  -> chroma Block8x8
        Vp9BlockSize.Block8x16,   // luma Block16x32  -> chroma Block8x16
        Vp9BlockSize.Block16x8,   // luma Block32x16  -> chroma Block16x8
        Vp9BlockSize.Block16x16,  // luma Block32x32  -> chroma Block16x16
        Vp9BlockSize.Block16x32,  // luma Block32x64  -> chroma Block16x32
        Vp9BlockSize.Block32x16,  // luma Block64x32  -> chroma Block32x16
        Vp9BlockSize.Block32x32,  // luma Block64x64  -> chroma Block32x32
    };

    /// <summary>
    /// Look up the 4:2:0 chroma block size for a luma block.
    /// </summary>
    public static Vp9BlockSize ForLumaBlock(Vp9BlockSize lumaBlockSize)
    {
        int idx = (int)lumaBlockSize;
        if ((uint)idx >= (uint)Vp9BlockSizes.Count)
            throw new ArgumentOutOfRangeException(nameof(lumaBlockSize), lumaBlockSize,
                "luma block size index out of range.");
        return For420[idx];
    }

    /// <summary>
    /// libvpx <c>get_uv_tx_size_impl</c> for 4:2:0 sampling. Given a
    /// luma tx_size and the luma block size, returns the chroma
    /// tx_size: clamped to the chroma block's max_tx_size, and forced
    /// to <see cref="Vp9TxSize.Tx4x4"/> for sub-8x8 luma blocks.
    /// </summary>
    public static Vp9TxSize GetChromaTxSize(Vp9TxSize lumaTxSize, Vp9BlockSize lumaBlockSize)
    {
        if (lumaBlockSize < Vp9BlockSize.Block8x8)
            return Vp9TxSize.Tx4x4;

        var chromaBs = ForLumaBlock(lumaBlockSize);
        var maxChromaTx = Vp9MaxTxSize.ForBlockSize(chromaBs);
        return (Vp9TxSize)Math.Min((int)lumaTxSize, (int)maxChromaTx);
    }
}
