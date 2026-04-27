// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 2D inverse transform dispatcher. Wraps the 1D Av1InverseDct/Adst/Identity
// primitives into a per-(tx_size, tx_type) 2D transform. Mirrors the libaom
// av1/common/av1_inv_txfm2d.c <c>av1_inv_txfm2d_add_*</c> family.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Pipeline (per libaom inv_txfm2d_add_facade):
//   1. Apply column 1D inverse over each column of the input
//   2. Round shift by shift[0] (size-dependent)
//   3. Apply row 1D inverse over each row
//   4. Round shift by shift[1] (size-dependent, signs flipped)
//   5. Caller adds residual to predictor + clips
//
// The residual is what this dispatcher produces (writes int residuals).
// Caller is responsible for the predictor add + clip step.
//
// Currently implemented for the square sizes 4x4, 8x8, 16x16 - covers the
// majority of BBB keyframe blocks. 32x32 / 64x64 + non-square sizes
// throw NotImplementedException pending Av1InverseDct32 / Adst32 / Adst32 ports.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 2D inverse transform dispatcher.</summary>
public static class Av1Inverse2dTransform
{
    /// <summary>
    /// Apply the 2D inverse transform for (txSize, txType) to <paramref name="coeffs"/>
    /// (raster, length = w*h) and write the residual into <paramref name="residual"/>.
    /// </summary>
    public static void Apply(Av1TxSize txSize, Av1TxType txType,
        ReadOnlySpan<int> coeffs, Span<int> residual)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        if (coeffs.Length < w * h) throw new ArgumentException($"coeffs too short: need {w * h}, got {coeffs.Length}", nameof(coeffs));
        if (residual.Length < w * h) throw new ArgumentException($"residual too short: need {w * h}, got {residual.Length}", nameof(residual));

        // Per libaom's inv_txfm_2d_add_facade: shift[0] is column-pass shift,
        // shift[1] is row-pass shift. Both are negative round-shifts in libaom.
        // For the square sizes we support, the shifts are:
        //   4x4:   col_shift=0, row_shift=4
        //   8x8:   col_shift=1, row_shift=4
        //   16x16: col_shift=2, row_shift=4
        //   32x32: col_shift=2, row_shift=4
        //   64x64: col_shift=2, row_shift=4
        var (colShift, rowShift) = GetShifts(txSize);

        var rowType = Av1TxSizeInfo.GetRowType(txType);
        var colType = Av1TxSizeInfo.GetColType(txType);

        // Stage 1: column 1D inverse over each column of input.
        // Library 1D impls operate on a length-N span. We process column by column.
        Span<int> colIn = stackalloc int[64];
        Span<int> colOut = stackalloc int[64];
        Span<int> intermediate = new int[w * h];
        for (int c = 0; c < w; c++)
        {
            for (int r = 0; r < h; r++) colIn[r] = coeffs[r * w + c];
            Apply1D(colType, h, colIn.Slice(0, h), colOut.Slice(0, h));
            // Round shift by colShift and store transposed into intermediate.
            for (int r = 0; r < h; r++)
            {
                int v = RoundShift(colOut[r], colShift);
                intermediate[r * w + c] = v;
            }
        }

        // Stage 2: row 1D inverse over each row of intermediate.
        Span<int> rowIn = stackalloc int[64];
        Span<int> rowOut = stackalloc int[64];
        for (int r = 0; r < h; r++)
        {
            for (int c = 0; c < w; c++) rowIn[c] = intermediate[r * w + c];
            Apply1D(rowType, w, rowIn.Slice(0, w), rowOut.Slice(0, w));
            for (int c = 0; c < w; c++)
            {
                residual[r * w + c] = RoundShift(rowOut[c], rowShift);
            }
        }

        // Handle FlipAdst by reversing the relevant axis after the transform.
        if (rowType == Av1Tx1dType.FlipAdst)
        {
            // Row was flipped: reverse each row.
            for (int r = 0; r < h; r++)
            {
                int rowBase = r * w;
                for (int c = 0; c < w / 2; c++)
                {
                    (residual[rowBase + c], residual[rowBase + w - 1 - c]) =
                        (residual[rowBase + w - 1 - c], residual[rowBase + c]);
                }
            }
        }
        if (colType == Av1Tx1dType.FlipAdst)
        {
            // Column was flipped: reverse each column.
            for (int c = 0; c < w; c++)
            {
                for (int r = 0; r < h / 2; r++)
                {
                    (residual[r * w + c], residual[(h - 1 - r) * w + c]) =
                        (residual[(h - 1 - r) * w + c], residual[r * w + c]);
                }
            }
        }
    }

    /// <summary>Apply a 1D inverse transform of the requested type and length.</summary>
    private static void Apply1D(Av1Tx1dType type, int n, ReadOnlySpan<int> input, Span<int> output)
    {
        // ADST and FlipADST share the same 1D primitive; the flip is applied
        // by reversing the result axis post-pass (handled by caller).
        switch (type)
        {
            case Av1Tx1dType.Dct:
                switch (n)
                {
                    case 4: Av1InverseDct4.Transform(input, output); return;
                    case 8: Av1InverseDct8.Transform(input, output); return;
                    case 16: Av1InverseDct16.Transform(input, output); return;
                    default:
                        throw new NotImplementedException($"Av1InverseDct{n} not yet implemented (needs Av1InverseDct{n}).");
                }
            case Av1Tx1dType.Adst:
            case Av1Tx1dType.FlipAdst:
                switch (n)
                {
                    case 4: Av1InverseAdst4.Transform(input, output); return;
                    case 8: Av1InverseAdst8.Transform(input, output); return;
                    case 16: Av1InverseAdst16.Transform(input, output); return;
                    default:
                        throw new NotImplementedException($"Av1InverseAdst{n} not yet implemented.");
                }
            case Av1Tx1dType.Identity:
                switch (n)
                {
                    case 4: Av1InverseIdentity.Transform4(input, output); return;
                    case 8: Av1InverseIdentity.Transform8(input, output); return;
                    case 16: Av1InverseIdentity.Transform16(input, output); return;
                    case 32: Av1InverseIdentity.Transform32(input, output); return;
                    default:
                        throw new NotImplementedException($"Av1InverseIdentity{n} not yet implemented.");
                }
            default:
                throw new ArgumentException($"Unknown 1D transform type {type}", nameof(type));
        }
    }

    /// <summary>libaom round-shift: arithmetic round-half-up by <paramref name="bit"/> bits.</summary>
    private static int RoundShift(int value, int bit)
    {
        if (bit <= 0) return value << -bit;
        return (value + (1 << (bit - 1))) >> bit;
    }

    /// <summary>libaom <c>inv_shift</c> table: per-tx-size column / row shifts.</summary>
    private static (int colShift, int rowShift) GetShifts(Av1TxSize txSize)
    {
        // libaom inv_txfm_shift_ls[]: each entry is { col_shift, row_shift }.
        // Negative means "shift left". Positive means "round-shift right".
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
