// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 intra block decode composition. Bundles the three steps of an
// intra block reconstruction at block-level granularity:
//
//   1. Look up the inverse transform type from the intra mode
//      (libvpx intra_mode_to_tx_type_lookup; 32x32 forced to DctDct).
//   2. Run the per-mode intra predictor into the destination block.
//   3. Apply the inverse transform of the dequantized residual
//      coefficients onto the predicted block (which clipped-adds and
//      finishes the reconstruction).
//
// Edge buffer setup (real-sample copy + libvpx 127/129 fill for
// missing edges) is the caller's responsibility; this helper takes
// already-prepared above / left / topLeft buffers and the dequantized
// coefficient buffer.
//
// libvpx reference: the body of vp9/decoder/vp9_decodeframe.c
// predict_and_reconstruct_intra_block_inter_step (composed of the
// same three calls in the same order).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 intra block decode entry point. Composes
/// <see cref="Vp9IntraPredictor.Predict"/> and
/// <see cref="Vp9InverseTransform.Apply"/> into a single block-level
/// call. Bit-exact against the libvpx intra-block reconstruction
/// path for any (mode, txSize) combination supported by VP9.
/// </summary>
public static class Vp9IntraBlockDecode
{
    /// <summary>
    /// Decode a single intra block: predict + add inverse-transformed
    /// residual + clip to pixel range.
    /// </summary>
    /// <param name="mode">Intra prediction mode (0..9).</param>
    /// <param name="txSize">Transform block size (4, 8, 16, or 32 in N-form).</param>
    /// <param name="topLeft">
    /// Corner sample diagonally above-left of the block (libvpx
    /// <c>above[-1]</c>). Caller is responsible for filling with the
    /// libvpx default when the corner is out of frame; see
    /// <see cref="Vp9IntraEdgeFill.ResolveCorner"/>.
    /// </param>
    /// <param name="above">
    /// Above-row edge samples. Caller supplies at least N entries, or
    /// 2N for D45 / D63. Out-of-frame slots must be pre-filled per the
    /// libvpx 127 convention.
    /// </param>
    /// <param name="left">
    /// Left-column edge samples (N entries). Out-of-frame slots
    /// pre-filled per the libvpx 129 convention.
    /// </param>
    /// <param name="coeffs">
    /// N*N int16 dequantized residual coefficients in row-major
    /// order. The inverse transform consumes these and adds the
    /// resulting spatial residual to the predicted block.
    /// </param>
    /// <param name="dst">
    /// Destination block (n*stride bytes). Holds the reconstructed
    /// pixels on output.
    /// </param>
    /// <param name="stride">Stride in bytes for <paramref name="dst"/>.</param>
    /// <param name="haveAbove">
    /// True when the above row is in-frame. Drives only the DC-mode
    /// variant selection (top vs left vs 128).
    /// </param>
    /// <param name="haveLeft">
    /// True when the left column is in-frame. Drives only the DC-mode
    /// variant selection.
    /// </param>
    public static void Decode(
        Vp9IntraMode mode,
        Vp9TxSize txSize,
        byte topLeft,
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        ReadOnlySpan<short> coeffs,
        Span<byte> dst, int stride,
        bool haveAbove = true, bool haveLeft = true)
    {
        int n = TxSizeToN(txSize);

        // tx_type from intra mode at sub-32x32; libvpx forces 32x32 -> DctDct.
        Vp9TxType txType = txSize == Vp9TxSize.Tx32x32
            ? Vp9TxType.DctDct
            : Vp9IntraTxType.ForMode(mode);

        // Step 1+2 - run the predictor into dst.
        Vp9IntraPredictor.Predict(
            mode, topLeft, above, left,
            dst, n, stride,
            haveAbove, haveLeft);

        // Step 3 - inverse transform adds the residual and clips.
        Vp9InverseTransform.Apply(txType, txSize, coeffs, dst, stride);
    }

    /// <summary>
    /// Translate <see cref="Vp9TxSize"/> to its N-dimension.
    /// </summary>
    public static int TxSizeToN(Vp9TxSize txSize) => txSize switch
    {
        Vp9TxSize.Tx4x4 => 4,
        Vp9TxSize.Tx8x8 => 8,
        Vp9TxSize.Tx16x16 => 16,
        Vp9TxSize.Tx32x32 => 32,
        _ => throw new ArgumentOutOfRangeException(nameof(txSize), txSize, "Unknown tx_size"),
    };
}
