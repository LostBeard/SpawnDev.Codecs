// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 unified inverse transform dispatcher. Routes a (txType, txSize)
// pair to the existing per-size CPU references and applies the
// residual to the predicted block in place. Mirror of the libvpx
// inv_txfm_add() dispatch in vp9/common/vp9_idct.c.
//
// Per-size entry points already in this assembly:
//   4x4   - Vp9Iht4x4Reference.Iht4x4_16_Add (handles all 4 tx_types)
//   8x8   - Vp9Iht8x8Reference.Iht8x8_64_Add (handles all 4 tx_types)
//   16x16 - Vp9Iht16x16Reference.Iht16x16_256_Add (handles all 4 tx_types)
//   32x32 - Vp9Idct32x32Reference.Idct32x32_1024_Add (DCT_DCT only)
//
// libvpx hard-codes 32x32 to TX_TYPE = DCT_DCT regardless of intra
// mode; this dispatcher enforces the same constraint.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Unified VP9 inverse transform dispatcher. Selects the per-size
/// reference based on tx_size, applies the transform indicated by
/// tx_type, and adds the resulting residual to the predicted block.
/// </summary>
public static class Vp9InverseTransform
{
    /// <summary>
    /// Apply <c>input</c> (n*n int16 coefficients, row-major) as a
    /// residual to <c>dest</c> (n*n uint8 predicted block at
    /// <paramref name="stride"/> bytes per row), using the inverse
    /// transform indicated by (<paramref name="txType"/>,
    /// <paramref name="txSize"/>). Bit-exact against the libvpx
    /// per-size reference for each (txType, txSize) combination.
    /// </summary>
    /// <param name="txType">Inverse transform type.</param>
    /// <param name="txSize">Transform block size.</param>
    /// <param name="input">N*N int16 coefficients in row-major order.</param>
    /// <param name="dest">Predicted block; reconstructed block on output.</param>
    /// <param name="stride">Stride in bytes for <paramref name="dest"/>.</param>
    public static void Apply(
        Vp9TxType txType,
        Vp9TxSize txSize,
        ReadOnlySpan<short> input,
        Span<byte> dest,
        int stride)
    {
        switch (txSize)
        {
            case Vp9TxSize.Tx4x4:
                Vp9Iht4x4Reference.Iht4x4_16_Add(
                    (Vp9TxType4x4)(byte)txType, input, dest, stride);
                break;

            case Vp9TxSize.Tx8x8:
                Vp9Iht8x8Reference.Iht8x8_64_Add(
                    (Vp9TxType8x8)(byte)txType, input, dest, stride);
                break;

            case Vp9TxSize.Tx16x16:
                Vp9Iht16x16Reference.Iht16x16_256_Add(
                    (Vp9TxType16x16)(byte)txType, input, dest, stride);
                break;

            case Vp9TxSize.Tx32x32:
                if (txType != Vp9TxType.DctDct)
                    throw new ArgumentException(
                        "VP9 32x32 transforms must use DctDct (libvpx hard-codes this)",
                        nameof(txType));
                Vp9Idct32x32Reference.Idct32x32_1024_Add(input, dest, stride);
                break;

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(txSize), txSize, "Unknown VP9 transform size");
        }
    }
}
