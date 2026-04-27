// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 above/left entropy context tracker for coefficient decode. Mirrors
// libaom MACROBLOCKD's <c>above_entropy_context</c> + <c>left_entropy_context</c>
// arrays used to provide the txb_skip / dc_sign contexts to
// av1_read_coeffs_txb.
//
// Two values per (plane, mi_unit):
//   - "cul_level" : cumulative absolute coef sum capped at COEFF_CONTEXT_MASK,
//                   plus DC sign packed in the top 2 bits.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 9.4.2 Initialization process for the coefficient decoder
//   sec 7.4   Coefficient decoder process

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 per-plane entropy context (above + left arrays) for txb_skip / dc_sign
/// CDF context computation.
/// </summary>
internal sealed class Av1EntropyContext
{
    private readonly byte[][] _above;  // [plane][miCol] -> packed cul_level + dcSign
    private readonly byte[][] _left;   // [plane][miRow & MIB_MASK]

    public Av1EntropyContext(int frameMiCols, int planes = 3)
    {
        _above = new byte[planes][];
        _left = new byte[planes][];
        for (int p = 0; p < planes; p++)
        {
            _above[p] = new byte[Math.Max(1, frameMiCols)];
            _left[p] = new byte[32]; // libaom MAX_MIB_SIZE
        }
    }

    /// <summary>Reset the left context array (call at the start of each tile row).</summary>
    public void ResetLeft()
    {
        for (int p = 0; p < _left.Length; p++) Array.Clear(_left[p]);
    }

    /// <summary>Reset the above context array (call at the start of each tile).</summary>
    public void ResetAbove()
    {
        for (int p = 0; p < _above.Length; p++) Array.Clear(_above[p]);
    }

    /// <summary>
    /// Compute the txb_skip CDF context for a tx block at (miRow, miCol)
    /// of size (txWMi, txHMi) (in mi units). Mirrors libaom <c>get_entropy_context</c>.
    /// </summary>
    public int GetTxbSkipContext(int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        // libaom: skip context = above_skip_count + left_skip_count, clamped to TXB_SKIP_CONTEXTS-1.
        // The above/left "skip" is determined by whether the cul_level is nonzero.
        int aboveSkip = 0;
        int leftSkip = 0;
        for (int i = 0; i < txWMi && miCol + i < _above[plane].Length; i++)
        {
            if (_above[plane][miCol + i] == 0) aboveSkip++;
        }
        for (int i = 0; i < txHMi && (miRow + i) < 32 * 100; i++) // bounded by left buffer + frame height
        {
            int idx = (miRow + i) & 31;
            if (_left[plane][idx] == 0) leftSkip++;
        }
        // Combine both into a 0..12 context.
        int ctx = (aboveSkip == txWMi ? 1 : 0) + (leftSkip == txHMi ? 1 : 0);
        return Math.Min(ctx + (txWMi > 1 ? 1 : 0) + (txHMi > 1 ? 1 : 0), Av1TxbCommon.TxbSkipContexts - 1);
    }

    /// <summary>
    /// Compute dc_sign CDF context for the given block. Mirrors libaom
    /// <c>get_dc_sign_context</c>: sums the 2-bit dc-sign halves of the
    /// above + left cul_level cells and maps to 0..2.
    /// </summary>
    public int GetDcSignContext(int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        int signAccum = 0;
        for (int i = 0; i < txWMi && miCol + i < _above[plane].Length; i++)
        {
            byte v = _above[plane][miCol + i];
            int sign = (v >> Av1TxbCommon.CoeffContextBits) & 0x3;
            if (sign == 1) signAccum--;
            else if (sign == 2) signAccum++;
        }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            byte v = _left[plane][idx];
            int sign = (v >> Av1TxbCommon.CoeffContextBits) & 0x3;
            if (sign == 1) signAccum--;
            else if (sign == 2) signAccum++;
        }
        return Av1TxbCommon.GetDcSignContext(signAccum);
    }

    /// <summary>Update the above + left arrays after decoding a tx block.</summary>
    public void Update(int plane, int miRow, int miCol, int txWMi, int txHMi, int culLevelWithSign)
    {
        byte v = (byte)(culLevelWithSign & 0xFF);
        for (int i = 0; i < txWMi && miCol + i < _above[plane].Length; i++)
        {
            _above[plane][miCol + i] = v;
        }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            _left[plane][idx] = v;
        }
    }
}
