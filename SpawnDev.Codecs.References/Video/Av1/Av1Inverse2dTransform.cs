// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 2D inverse transform dispatcher. Wraps the 1D Av1InverseDct/Adst/Identity
// primitives into a per-(tx_size, tx_type) 2D transform. Mirrors the libaom
// av1/common/av1_inv_txfm2d.c <c>inv_txfm2d_add_c</c> pipeline.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// libaom pipeline (per inv_txfm2d_add_c, lines 234-316 in av1_inv_txfm2d.c):
//   1. ROW pass: for each row r, gather temp_in[c] = input[c*H + r]
//      (libaom col-major aka bhl-stride layout); apply row 1D inverse
//      txfm_func_row over txfm_size_col entries; round-shift by -shift[0]
//      (negative -> shift right); store row-major into buf[r*W + c].
//   2. For rectangular tx (rect_type == +/-1, i.e. one dim is 2x the other):
//      pre-scale input[c*H + r] by NewInvSqrt2 / 2^NewSqrt2Bits BEFORE row pass.
//   3. COL pass: for each col c, gather temp_in[r] = buf[r*W + c]
//      (with optional left-right flip if tx_type has FlipAdst horizontal);
//      apply col 1D inverse txfm_func_col over txfm_size_row entries;
//      round-shift by -shift[1].
//   4. Write residual rows in raster order, applying optional up-down flip
//      if tx_type has FlipAdst vertical.
//
// The residual is what this dispatcher produces (writes int residuals).
// Caller is responsible for the predictor add + clip step.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 2D inverse transform dispatcher.</summary>
public static class Av1Inverse2dTransform
{
    /// <summary>libaom <c>NewSqrt2Bits</c>.</summary>
    private const int NewSqrt2Bits = 12;
    /// <summary>libaom <c>NewInvSqrt2</c> = round(2^12 / sqrt(2)).</summary>
    private const int NewInvSqrt2 = 2896;

    /// <summary>
    /// Apply the 2D inverse transform for (txSize, txType) to <paramref name="coeffs"/>
    /// (raster row-major, length = w*h) and write the residual into <paramref name="residual"/>.
    /// </summary>
    public static void Apply(Av1TxSize txSize, Av1TxType txType,
        ReadOnlySpan<int> coeffs, Span<int> residual)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        if (coeffs.Length < w * h) throw new ArgumentException($"coeffs too short: need {w * h}, got {coeffs.Length}", nameof(coeffs));
        if (residual.Length < w * h) throw new ArgumentException($"residual too short: need {w * h}, got {residual.Length}", nameof(residual));

        var (colShift, rowShift) = GetShifts(txSize);
        var rowType = Av1TxSizeInfo.GetRowType(txType);
        var colType = Av1TxSizeInfo.GetColType(txType);
        bool udFlip = colType == Av1Tx1dType.FlipAdst;
        bool lrFlip = rowType == Av1Tx1dType.FlipAdst;

        // libaom rect_type = log2(W/H), positive when wider.
        // |rect_type| == 1 needs the NewInvSqrt2 scale on the input pre-row-pass.
        int rectScale = 0;
        if ((w == 2 * h) || (h == 2 * w)) rectScale = 1;

        // 64-tall / 64-wide transforms only carry 32 actual coef rows / cols.
        // Caller's coefs are stored row-major in the FULL (w x h) layout,
        // but the high-frequency 32-rows / 32-cols are zero.
        // libaom inv_txfm2d_add_c reads input[c*H + r]: but the caller has
        // already converted coefs to ROW-MAJOR (row*W + col). Convert per-element.

        // Stage 1: ROW pass. Process each row of the transform block.
        var buf = new int[w * h];
        Span<int> rowIn = stackalloc int[64];
        Span<int> rowOut = stackalloc int[64];
        for (int r = 0; r < h; r++)
        {
            for (int c = 0; c < w; c++)
            {
                int v = coeffs[r * w + c];
                if (rectScale != 0)
                {
                    v = (int)(((long)v * NewInvSqrt2 + (1L << (NewSqrt2Bits - 1))) >> NewSqrt2Bits);
                }
                rowIn[c] = v;
            }
            Apply1D(rowType, w, rowIn.Slice(0, w), rowOut.Slice(0, w));
            for (int c = 0; c < w; c++)
            {
                buf[r * w + c] = RoundShiftRight(rowOut[c], colShift);
            }
        }

        // Stage 2: COL pass. Apply column inverse over each column,
        // applying lr_flip (reverse column order) when needed.
        Span<int> colIn = stackalloc int[64];
        Span<int> colOut = stackalloc int[64];
        for (int c = 0; c < w; c++)
        {
            int srcCol = lrFlip ? (w - 1 - c) : c;
            for (int r = 0; r < h; r++)
            {
                colIn[r] = buf[r * w + srcCol];
            }
            Apply1D(colType, h, colIn.Slice(0, h), colOut.Slice(0, h));
            for (int r = 0; r < h; r++)
            {
                int dstRow = udFlip ? (h - 1 - r) : r;
                residual[dstRow * w + c] = RoundShiftRight(colOut[r], rowShift);
            }
        }
    }

    /// <summary>Apply a 1D inverse transform of the requested type and length.</summary>
    private static void Apply1D(Av1Tx1dType type, int n, ReadOnlySpan<int> input, Span<int> output)
    {
        // ADST and FlipADST share the same 1D primitive; the flip is applied
        // by reversing the result axis post-pass (handled by the caller).
        switch (type)
        {
            case Av1Tx1dType.Dct:
                switch (n)
                {
                    case 4: Av1InverseDct4.Transform(input, output); return;
                    case 8: Av1InverseDct8.Transform(input, output); return;
                    case 16: Av1InverseDct16.Transform(input, output); return;
                    case 32: Av1InverseDct32.Transform(input, output); return;
                    case 64: Av1InverseDct64.Transform(input, output); return;
                    default:
                        throw new NotImplementedException($"Av1InverseDct{n} not implemented.");
                }
            case Av1Tx1dType.Adst:
            case Av1Tx1dType.FlipAdst:
                switch (n)
                {
                    case 4: Av1InverseAdst4.Transform(input, output); return;
                    case 8: Av1InverseAdst8.Transform(input, output); return;
                    case 16: Av1InverseAdst16.Transform(input, output); return;
                    // libaom has no iadst32/64. The 2D dispatch above never asks
                    // for ADST at 32 or 64 because EXT_TX_SET_TYPE for 32x32 is
                    // DCT-only (intra) or DCT_IDTX (inter). Throw if reached.
                    default:
                        throw new NotImplementedException($"Av1InverseAdst{n} not used by AV1 spec; reaching this is a decoder bug.");
                }
            case Av1Tx1dType.Identity:
                switch (n)
                {
                    case 4: Av1InverseIdentity.Transform4(input, output); return;
                    case 8: Av1InverseIdentity.Transform8(input, output); return;
                    case 16: Av1InverseIdentity.Transform16(input, output); return;
                    case 32: Av1InverseIdentity.Transform32(input, output); return;
                    // libaom has no iidentity64. AV1 doesn't use IDTX at 64-pt.
                    default:
                        throw new NotImplementedException($"Av1InverseIdentity{n} not used by AV1 spec.");
                }
            default:
                throw new ArgumentException($"Unknown 1D transform type {type}", nameof(type));
        }
    }

    /// <summary>libaom <c>round_shift</c>: arithmetic round-half-up by <paramref name="bit"/> bits.</summary>
    private static int RoundShiftRight(int value, int bit)
    {
        if (bit <= 0) return value << -bit;
        return (value + (1 << (bit - 1))) >> bit;
    }

    /// <summary>
    /// libaom <c>av1_inv_txfm_shift_ls</c>: per-tx-size {col_shift, row_shift}.
    /// Both are negative in libaom; we store the absolute right-shift amount.
    /// shift[0] applies after the ROW pass; shift[1] after the COL pass.
    /// </summary>
    private static (int rowPassShift, int colPassShift) GetShifts(Av1TxSize txSize)
    {
        // libaom inv_shift_NxM[2]: { -row_pass_shift, -col_pass_shift }, e.g.
        //   inv_shift_4x4   = { 0, -4 }   -> rowShift=0, colShift=4
        //   inv_shift_8x8   = { -1, -4 }  -> rowShift=1, colShift=4
        //   inv_shift_16x16 = { -2, -4 }
        //   inv_shift_32x32 = { -2, -4 }
        //   inv_shift_64x64 = { -2, -4 }
        //   inv_shift_4x8   = { 0, -4 }
        //   inv_shift_8x4   = { 0, -4 }
        //   inv_shift_8x16  = { -1, -4 }
        //   inv_shift_16x8  = { -1, -4 }
        //   inv_shift_16x32 = { -1, -4 }
        //   inv_shift_32x16 = { -1, -4 }
        //   inv_shift_32x64 = { -1, -4 }
        //   inv_shift_64x32 = { -1, -4 }
        //   inv_shift_4x16  = { -1, -4 }
        //   inv_shift_16x4  = { -1, -4 }
        //   inv_shift_8x32  = { -2, -4 }
        //   inv_shift_32x8  = { -2, -4 }
        //   inv_shift_16x64 = { -2, -4 }
        //   inv_shift_64x16 = { -2, -4 }
        return txSize switch
        {
            Av1TxSize.Tx4x4 => (0, 4),
            Av1TxSize.Tx8x8 => (1, 4),
            Av1TxSize.Tx16x16 => (2, 4),
            Av1TxSize.Tx32x32 => (2, 4),
            Av1TxSize.Tx64x64 => (2, 4),
            Av1TxSize.Tx4x8 => (0, 4),
            Av1TxSize.Tx8x4 => (0, 4),
            Av1TxSize.Tx8x16 => (1, 4),
            Av1TxSize.Tx16x8 => (1, 4),
            Av1TxSize.Tx16x32 => (1, 4),
            Av1TxSize.Tx32x16 => (1, 4),
            Av1TxSize.Tx32x64 => (1, 4),
            Av1TxSize.Tx64x32 => (1, 4),
            Av1TxSize.Tx4x16 => (1, 4),
            Av1TxSize.Tx16x4 => (1, 4),
            Av1TxSize.Tx8x32 => (2, 4),
            Av1TxSize.Tx32x8 => (2, 4),
            Av1TxSize.Tx16x64 => (2, 4),
            Av1TxSize.Tx64x16 => (2, 4),
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
    }
}
