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
    /// of size (txWMi, txHMi) (in mi units). Bit-exact port of libaom
    /// <c>get_txb_ctx</c> (av1/common/txb_common.h SPECIALIZE_GET_TXB_CTX).
    ///
    /// The context depends on whether the transform fills the entire plane
    /// block (planeBsizeIsTxsize) and the plane:
    ///   - Y, planeBsizeIsTxsize: returns 0
    ///   - Y, !planeBsizeIsTxsize: skip_contexts[top][left] table
    ///   - Chroma: get_entropy_context() + ctx_offset (7 or 10)
    /// </summary>
    public int GetTxbSkipContext(int plane, int miRow, int miCol, int txWMi, int txHMi,
        bool planeBsizeIsTxsize, bool planeBsizeLargerThanTxBsize)
    {
        if (plane == 0)
        {
            if (planeBsizeIsTxsize) return 0;

            // libaom: top |= a[k] for k in 0..txb_w_unit; same for left;
            // top &= COEFF_CONTEXT_MASK; top = min(top, 4); same for left.
            int top = 0;
            int left = 0;
            for (int i = 0; i < txWMi && miCol + i < _above[plane].Length; i++)
            {
                top |= _above[plane][miCol + i];
            }
            for (int i = 0; i < txHMi; i++)
            {
                int idx = (miRow + i) & 31;
                left |= _left[plane][idx];
            }
            top &= Av1TxbCommon.CoeffContextMask;
            top = Math.Min(top, 4);
            left &= Av1TxbCommon.CoeffContextMask;
            left = Math.Min(left, 4);
            return SkipContexts[top][left];
        }

        // Chroma: get_entropy_context combines (above_ec != 0) + (left_ec != 0)
        // for each of txWMi above + txHMi left mi-units, then we add ctx_offset.
        bool aboveNz = false;
        bool leftNz = false;
        for (int i = 0; i < txWMi && miCol + i < _above[plane].Length; i++)
        {
            if (_above[plane][miCol + i] != 0) { aboveNz = true; break; }
        }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            if (_left[plane][idx] != 0) { leftNz = true; break; }
        }
        int ctxBase = (aboveNz ? 1 : 0) + (leftNz ? 1 : 0);
        int ctxOffset = planeBsizeLargerThanTxBsize ? 10 : 7;
        return ctxBase + ctxOffset;
    }

    /// <summary>
    /// libaom <c>skip_contexts[5][5]</c> table from txb_common.h. Indexed by
    /// (top, left), each clipped to [0, 4]. Returns the txb_skip CDF context
    /// when planeBsize != txsize_to_bsize[tx_size].
    /// </summary>
    private static readonly byte[][] SkipContexts = new byte[][]
    {
        new byte[] { 1, 2, 2, 2, 3 },
        new byte[] { 2, 4, 4, 4, 5 },
        new byte[] { 2, 4, 4, 4, 5 },
        new byte[] { 2, 4, 4, 4, 5 },
        new byte[] { 3, 5, 5, 5, 6 },
    };

    /// <summary>
    /// Compute dc_sign CDF context for the given block. Bit-exact port of
    /// libaom <c>get_txb_ctx</c> (dc_sign portion). Sums signs (0/-1/+1) from
    /// above + left cul_level cells, then maps to 0..2 via dc_sign_contexts[].
    /// </summary>
    public int GetDcSignContext(int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        int dcSign = 0;
        for (int i = 0; i < txWMi && miCol + i < _above[plane].Length; i++)
        {
            int sign = ((int)_above[plane][miCol + i]) >> Av1TxbCommon.CoeffContextBits;
            dcSign += DcSigns[sign & 0x3];
        }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            int sign = ((int)_left[plane][idx]) >> Av1TxbCommon.CoeffContextBits;
            dcSign += DcSigns[sign & 0x3];
        }
        // libaom's dc_sign_contexts is sized 4*MAX_TX_SIZE_UNIT+1 = 65 with
        // MAX_TX_SIZE_UNIT = 16. We reproduce it inline; index = dc_sign + 32.
        int idxFinal = dcSign + 32;
        if (idxFinal < 0) idxFinal = 0;
        else if (idxFinal >= DcSignContextsTable.Length) idxFinal = DcSignContextsTable.Length - 1;
        return DcSignContextsTable[idxFinal];
    }

    /// <summary>libaom <c>signs[3]</c>: sign accumulator delta per packed sign bits.</summary>
    private static readonly sbyte[] DcSigns = new sbyte[] { 0, -1, 1, 0 };

    /// <summary>libaom <c>dc_sign_contexts[4 * 16 + 1]</c> from txb_common.h.</summary>
    private static readonly byte[] DcSignContextsTable = new byte[]
    {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    };

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
