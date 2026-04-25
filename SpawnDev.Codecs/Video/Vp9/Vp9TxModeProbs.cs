// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 tx_mode probability storage + parser. The compressed header
// of a TxModeSelect frame carries diff_update_prob updates for the
// per-context tx_size selection trees:
//
//   p8x8 [TX_SIZE_CONTEXTS=2][TX_SIZES-3=1]   - choose 4x4 vs 8x8
//   p16x16[TX_SIZE_CONTEXTS=2][TX_SIZES-2=2]  - choose 4x4/8x8/16x16
//   p32x32[TX_SIZE_CONTEXTS=2][TX_SIZES-1=3]  - choose 4x4/8x8/16x16/32x32
//
// Total: 2*1 + 2*2 + 2*3 = 12 probability bytes.
//
// Mirror of libvpx vp9/decoder/vp9_decodeframe.c read_tx_mode_probs
// using the slice 210 vp9_diff_update_prob primitive.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 tx_size selection probabilities (libvpx struct tx_probs).</summary>
public sealed class Vp9TxModeProbs
{
    /// <summary>libvpx <c>TX_SIZE_CONTEXTS</c>.</summary>
    public const int TxSizeContexts = 2;

    /// <summary>libvpx <c>TX_SIZES</c>.</summary>
    public const int TxSizes = 4;

    /// <summary>p8x8: per-context 1-leaf tree choosing 4x4 vs 8x8.</summary>
    public byte[,] P8x8 { get; } = new byte[TxSizeContexts, TxSizes - 3];

    /// <summary>p16x16: per-context 2-leaf tree (4x4 / 8x8 / 16x16).</summary>
    public byte[,] P16x16 { get; } = new byte[TxSizeContexts, TxSizes - 2];

    /// <summary>p32x32: per-context 3-leaf tree (4x4 / 8x8 / 16x16 / 32x32).</summary>
    public byte[,] P32x32 { get; } = new byte[TxSizeContexts, TxSizes - 1];
}

/// <summary>Parser for the read_tx_mode_probs section of the compressed header.</summary>
public static class Vp9TxModeProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to every entry of the three tables in
    /// <paramref name="probs"/>. Mirror of libvpx
    /// <c>read_tx_mode_probs</c>.
    /// </summary>
    public static void Read(Vp9TxModeProbs probs, Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);

        for (int i = 0; i < Vp9TxModeProbs.TxSizeContexts; i++)
            for (int j = 0; j < Vp9TxModeProbs.TxSizes - 3; j++)
                probs.P8x8[i, j] = Vp9DiffUpdateProb.Read(reader, probs.P8x8[i, j]);

        for (int i = 0; i < Vp9TxModeProbs.TxSizeContexts; i++)
            for (int j = 0; j < Vp9TxModeProbs.TxSizes - 2; j++)
                probs.P16x16[i, j] = Vp9DiffUpdateProb.Read(reader, probs.P16x16[i, j]);

        for (int i = 0; i < Vp9TxModeProbs.TxSizeContexts; i++)
            for (int j = 0; j < Vp9TxModeProbs.TxSizes - 1; j++)
                probs.P32x32[i, j] = Vp9DiffUpdateProb.Read(reader, probs.P32x32[i, j]);
    }
}
