// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 read_coef_probs / read_coef_probs_common parser. Walks the
// per-transform-size 5D coefficient probability table and applies
// per-entry vp9_diff_update_prob updates from the compressed
// frame header.
//
// 5D layout per tx_size: [PLANE_TYPES][REF_TYPES][COEF_BANDS][BAND_COEFF_CONTEXTS][UNCONSTRAINED_NODES]
//   PLANE_TYPES = 2 (Y, UV)
//   REF_TYPES = 2 (intra, inter)
//   COEF_BANDS = 6
//   BAND_COEFF_CONTEXTS = 3 for band 0, 6 for bands 1..5
//   UNCONSTRAINED_NODES = 3
//
// Existing flat layout in Vp9CoefProbs.DefaultCoefProbs* is row-
// major rectangular with band-0 contexts 3..5 zero-padded so the
// flat index arithmetic stays simple. The parser respects that:
// it only reads/writes the BAND_COEFF_CONTEXTS(band) "real" slots
// but the flat index includes the padding offset.
//
// Per-tx-size flat sizes (rectangular):
//   2 * 2 * 6 * 6 * 3 = 432 bytes (same across all tx sizes)
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c read_coef_probs +
// read_coef_probs_common.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 read_coef_probs parser.</summary>
public static class Vp9CoefProbsParser
{
    /// <summary>libvpx <c>PLANE_TYPES</c>.</summary>
    public const int PlaneTypes = 2;

    /// <summary>libvpx <c>REF_TYPES</c>.</summary>
    public const int RefTypes = 2;

    /// <summary>libvpx <c>COEF_BANDS</c>.</summary>
    public const int CoefBands = 6;

    /// <summary>libvpx <c>UNCONSTRAINED_NODES</c>.</summary>
    public const int UnconstrainedNodes = 3;

    /// <summary>
    /// Total context slots per band in the rectangular layout. Bands
    /// 1..5 use all 6 of these; band 0 uses only the first 3.
    /// </summary>
    public const int CoefContextsPerBand = 6;

    /// <summary>Flat size of one tx_size's coef-prob table.</summary>
    public const int FlatSize =
        PlaneTypes * RefTypes * CoefBands * CoefContextsPerBand * UnconstrainedNodes;

    /// <summary>Number of contexts actively used by a given band.</summary>
    public static int BandCoefContexts(int band) => band == 0 ? 3 : 6;

    /// <summary>
    /// Compute the flat index for (plane, ref, band, ctx, node) in a
    /// rectangular [2][2][6][6][3] layout.
    /// </summary>
    public static int FlatIndex(int plane, int refType, int band, int ctx, int node)
    {
        return ((((plane * RefTypes + refType) * CoefBands + band) * CoefContextsPerBand + ctx)
                * UnconstrainedNodes + node);
    }

    /// <summary>
    /// Decode one frame's coef-prob updates for a single tx_size.
    /// Reads the 1-bit update flag; if 0, returns without touching
    /// <paramref name="coefProbs"/>. If 1, walks the (plane, ref,
    /// band, ctx, node) loop and applies <c>vp9_diff_update_prob</c>
    /// to each active entry. Bit-exact against libvpx
    /// <c>read_coef_probs_common</c>.
    /// </summary>
    /// <param name="coefProbs">
    /// 432-byte flat coefficient probability table for one tx_size.
    /// Modified in place when an update is signalled.
    /// </param>
    /// <param name="reader">Compressed-header arithmetic reader.</param>
    public static void ReadCoefProbsCommon(byte[] coefProbs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(coefProbs);
        ArgumentNullException.ThrowIfNull(reader);
        if (coefProbs.Length < FlatSize)
            throw new ArgumentException(
                $"coefProbs must hold at least {FlatSize} bytes (got {coefProbs.Length})",
                nameof(coefProbs));

        if (reader.ReadBit() == 0) return;  // no update on this frame

        for (int plane = 0; plane < PlaneTypes; plane++)
        {
            for (int refType = 0; refType < RefTypes; refType++)
            {
                for (int band = 0; band < CoefBands; band++)
                {
                    int contexts = BandCoefContexts(band);
                    for (int ctx = 0; ctx < contexts; ctx++)
                    {
                        for (int node = 0; node < UnconstrainedNodes; node++)
                        {
                            int idx = FlatIndex(plane, refType, band, ctx, node);
                            coefProbs[idx] = Vp9DiffUpdateProb.Read(reader, coefProbs[idx]);
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Walk all tx_sizes from 4x4 up to the maximum permitted by
    /// <paramref name="txMode"/> and apply per-tx-size updates.
    /// Mirror of libvpx <c>read_coef_probs</c>.
    /// </summary>
    /// <param name="coefProbsPerTxSize">
    /// Array of 4 flat coefficient probability tables, one per
    /// tx_size 0..3 (Tx4x4 / Tx8x8 / Tx16x16 / Tx32x32). Each must
    /// be at least <see cref="FlatSize"/> bytes. Modified in place.
    /// </param>
    /// <param name="txMode">Frame tx_mode constraining max tx_size.</param>
    /// <param name="reader">Compressed-header arithmetic reader.</param>
    public static void ReadCoefProbs(
        byte[][] coefProbsPerTxSize,
        Vp9TxMode txMode,
        Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(coefProbsPerTxSize);
        ArgumentNullException.ThrowIfNull(reader);
        if (coefProbsPerTxSize.Length < 4)
            throw new ArgumentException(
                "coefProbsPerTxSize must have 4 entries (one per tx_size)",
                nameof(coefProbsPerTxSize));

        int maxTxSize = TxModeToBiggestTxSize(txMode);
        for (int i = 0; i <= maxTxSize; i++)
            ReadCoefProbsCommon(coefProbsPerTxSize[i], reader);
    }

    /// <summary>
    /// libvpx <c>tx_mode_to_biggest_tx_size</c> mapping. Returns the
    /// largest tx_size index 0..3 that can be used under
    /// <paramref name="txMode"/>.
    /// </summary>
    public static int TxModeToBiggestTxSize(Vp9TxMode txMode) => txMode switch
    {
        Vp9TxMode.Only4x4 => 0,
        Vp9TxMode.AllowOnly8x8 => 1,
        Vp9TxMode.AllowOnly16x16 => 2,
        Vp9TxMode.Allow32x32 => 3,
        Vp9TxMode.TxModeSelect => 3,
        _ => throw new ArgumentOutOfRangeException(nameof(txMode), txMode, "Unknown tx_mode"),
    };
}
