// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block tx_size selection. Mirror of libvpx
// vp9/decoder/vp9_decodemv.c read_tx_size + read_selected_tx_size.
//
// The frame-level tx_mode field (Vp9TxMode) constrains what the
// per-block tx_size can be:
//
//   Only4x4         : every block is 4x4.
//   AllowOnly8x8    : MIN(max_tx_size, 8x8).
//   AllowOnly16x16  : MIN(max_tx_size, 16x16).
//   Allow32x32      : MIN(max_tx_size, 32x32) - i.e. just max_tx_size.
//   TxModeSelect    : tx_size is read from the bitstream per block,
//                     with a tx_size_probs context tree.
//
// max_tx_size = Vp9MaxTxSize.ForBlockSize(bsize). For TxModeSelect
// the bitstream may transmit smaller-than-max via a 1-3 bit prob
// tree; the tree shape depends on the max_tx_size:
//
//   max_tx_size == 8x8   : 1 bit  (4x4 vs 8x8)
//   max_tx_size == 16x16 : 1-2 bits (4x4 vs 8x8 vs 16x16)
//   max_tx_size == 32x32 : 1-3 bits (4x4 vs 8x8 vs 16x16 vs 32x32)
//
// At sub-8x8 block sizes (Block4x4 / Block4x8 / Block8x4) the
// bitstream never transmits tx_size; max_tx_size is forced to
// Tx4x4. Caller is responsible for passing the right max_tx_size
// (use Vp9MaxTxSize.ForBlockSize) and gating on bsize >= Block8x8
// before invoking the TxModeSelect bitstream-read path.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-block tx_size selection.</summary>
public static class Vp9TxSizeDecoder
{
    /// <summary>
    /// libvpx <c>tx_mode_to_biggest_tx_size</c>: maps a tx_mode to the
    /// largest tx_size it permits. TxModeSelect is mapped to Tx32x32
    /// because that mode permits any tx_size; the actual size is
    /// chosen from the bitstream rather than this lookup.
    /// </summary>
    public static readonly Vp9TxSize[] TxModeToBiggestTxSize = new Vp9TxSize[5]
    {
        Vp9TxSize.Tx4x4,    // Only4x4
        Vp9TxSize.Tx8x8,    // AllowOnly8x8
        Vp9TxSize.Tx16x16,  // AllowOnly16x16
        Vp9TxSize.Tx32x32,  // Allow32x32
        Vp9TxSize.Tx32x32,  // TxModeSelect (max permitted; per-block read overrides)
    };

    /// <summary>
    /// Read the per-block tx_size given the frame-level tx_mode and
    /// the block's max_tx_size. When <paramref name="txMode"/> is
    /// <see cref="Vp9TxMode.TxModeSelect"/> and the block is at least
    /// 8x8 (i.e. <paramref name="maxTxSize"/> &gt; <c>Tx4x4</c>) the
    /// caller must pass the per-context <paramref name="txSizeProbs"/>
    /// row and a non-null <paramref name="reader"/>; otherwise
    /// <paramref name="txSizeProbs"/> is unused.
    /// </summary>
    /// <param name="txMode">Frame-level tx_mode field.</param>
    /// <param name="maxTxSize">
    /// Block's maximum allowed tx_size from
    /// <see cref="Vp9MaxTxSize.ForBlockSize(Vp9BlockSize)"/>.
    /// </param>
    /// <param name="reader">
    /// Bool-coded reader. Required when <paramref name="txMode"/> is
    /// <see cref="Vp9TxMode.TxModeSelect"/> and
    /// <paramref name="maxTxSize"/> &gt; <c>Tx4x4</c>; ignored otherwise.
    /// </param>
    /// <param name="txSizeProbs">
    /// Per-context tx_size probability row (1, 2, or 3 entries
    /// depending on max_tx_size). Pulled from
    /// <see cref="Vp9TxModeProbs.P8x8"/> / <c>P16x16</c> / <c>P32x32</c>
    /// at the caller's tx_size_context.
    /// </param>
    public static Vp9TxSize ReadTxSize(
        Vp9TxMode txMode,
        Vp9TxSize maxTxSize,
        Vp9BoolDecoder? reader,
        ReadOnlySpan<byte> txSizeProbs)
    {
        if (txMode == Vp9TxMode.TxModeSelect && maxTxSize > Vp9TxSize.Tx4x4)
        {
            ArgumentNullException.ThrowIfNull(reader);
            return ReadSelectedTxSize(maxTxSize, reader, txSizeProbs);
        }

        // Forced mode: clamp to the smaller of (max permitted, max for block).
        var biggest = TxModeToBiggestTxSize[(int)txMode];
        return (Vp9TxSize)Math.Min((int)maxTxSize, (int)biggest);
    }

    /// <summary>
    /// Bitstream-read flavor for TxModeSelect frames. Reads 1-3 bits
    /// from <paramref name="reader"/> guided by
    /// <paramref name="txSizeProbs"/>, walking the tx_size tree until
    /// a 0 bit picks the current size.
    /// </summary>
    public static Vp9TxSize ReadSelectedTxSize(
        Vp9TxSize maxTxSize,
        Vp9BoolDecoder reader,
        ReadOnlySpan<byte> txSizeProbs)
    {
        ArgumentNullException.ThrowIfNull(reader);
        if (maxTxSize <= Vp9TxSize.Tx4x4)
            throw new ArgumentOutOfRangeException(nameof(maxTxSize), maxTxSize,
                "ReadSelectedTxSize requires max_tx_size > Tx4x4 (block must be >= 8x8).");

        int requiredProbs = (int)maxTxSize; // 1 for 8x8, 2 for 16x16, 3 for 32x32
        if (txSizeProbs.Length < requiredProbs)
            throw new ArgumentException(
                $"txSizeProbs must hold at least {requiredProbs} entries for max_tx_size={maxTxSize}.",
                nameof(txSizeProbs));

        int txSize = reader.Read(txSizeProbs[0]);
        if (txSize != (int)Vp9TxSize.Tx4x4 && maxTxSize >= Vp9TxSize.Tx16x16)
        {
            txSize += reader.Read(txSizeProbs[1]);
            if (txSize != (int)Vp9TxSize.Tx8x8 && maxTxSize >= Vp9TxSize.Tx32x32)
                txSize += reader.Read(txSizeProbs[2]);
        }
        return (Vp9TxSize)txSize;
    }
}
