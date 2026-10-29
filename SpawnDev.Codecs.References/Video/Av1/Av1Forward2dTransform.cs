// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 2D forward transform dispatcher. Wraps the 1D
// Av1ForwardDct/Adst primitives into a per-(tx_size, tx_type) 2D
// transform. Mirrors the libaom av1/encoder/av1_fwd_txfm2d.c
// <c>av1_fwd_txfm2d_*</c> family.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Pipeline (per libaom fwd_txfm2d_c):
//   1. For each column c:
//      a. Load column from input
//      b. round_shift by -shift[0] (left-shift = pre-scale)
//      c. Apply column 1D forward transform (txfm_func_col, cos_bit_col)
//      d. round_shift by -shift[1] (between-pass)
//      e. Store transposed into buf (with optional lr_flip)
//   2. For each row r:
//      a. Apply row 1D forward transform on buf row (txfm_func_row, cos_bit_row)
//      b. round_shift by -shift[2] (final)
//      c. Store transposed back into output
//
// Currently implemented for the square sizes 4x4, 8x8, 16x16, 32x32 -
// covers all sizes for which both forward 1D Dct and Adst exist.
// 64x64 + non-square sizes will land alongside Dct64 / Adst64.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 2D forward transform dispatcher.</summary>
public static class Av1Forward2dTransform
{
    /// <summary>
    /// Apply the 2D forward transform for (txSize, txType) to
    /// <paramref name="input"/> (raster, length = w*h short residuals)
    /// and write coefficients into <paramref name="output"/> (length w*h ints).
    /// </summary>
    public static void Apply(Av1TxSize txSize, Av1TxType txType,
        ReadOnlySpan<short> input, Span<int> output)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        if (input.Length < w * h)
            throw new ArgumentException($"input too short: need {w * h}, got {input.Length}", nameof(input));
        if (output.Length < w * h)
            throw new ArgumentException($"output too short: need {w * h}, got {output.Length}", nameof(output));

        var (s0, s1, s2) = GetShifts(txSize);
        var (cosBitCol, cosBitRow) = GetCosBits(txSize);

        var rowType = Av1TxSizeInfo.GetRowType(txType);
        var colType = Av1TxSizeInfo.GetColType(txType);

        // Per libaom: ud_flip and lr_flip come from the FlipAdst variants of
        // colType / rowType. For the standard ADST_ADST etc. there's no flip.
        // FlipAdst rolls the input over the relevant axis pre-transform.
        bool udFlip = colType == Av1Tx1dType.FlipAdst;
        bool lrFlip = rowType == Av1Tx1dType.FlipAdst;

        Span<int> tempIn = stackalloc int[64];
        Span<int> tempOut = stackalloc int[64];
        var buf = new int[w * h];

        // === Column pass ===
        for (int c = 0; c < w; c++)
        {
            // Load column with optional ud_flip.
            if (udFlip)
            {
                for (int r = 0; r < h; r++) tempIn[r] = input[(h - 1 - r) * w + c];
            }
            else
            {
                for (int r = 0; r < h; r++) tempIn[r] = input[r * w + c];
            }

            // Pre-scale by -shift[0] bits.
            for (int r = 0; r < h; r++) tempIn[r] = RoundShift(tempIn[r], -s0);

            // Apply 1D forward over the column (treat ADST and FlipAdst the same;
            // the flip happens at the buffer-load step above).
            Apply1D(colType, h, tempIn.Slice(0, h), tempOut.Slice(0, h), cosBitCol);

            // Between-pass shift.
            for (int r = 0; r < h; r++) tempOut[r] = RoundShift(tempOut[r], -s1);

            // Store transposed into buf with optional lr_flip.
            int destCol = lrFlip ? (w - 1 - c) : c;
            for (int r = 0; r < h; r++) buf[r * w + destCol] = tempOut[r];
        }

        // === Row pass ===
        Span<int> rowOut = stackalloc int[64];
        for (int r = 0; r < h; r++)
        {
            // Apply 1D forward over the row.
            Apply1D(rowType, w, ((ReadOnlySpan<int>)buf.AsSpan(r * w, w)), rowOut.Slice(0, w), cosBitRow);

            // Final shift.
            for (int c = 0; c < w; c++) rowOut[c] = RoundShift(rowOut[c], -s2);

            // Store back into output as ROW-MAJOR (raster). Note: libaom's
            // own forward writes `output[c * h + r]` (column-major), which
            // pairs with libaom's inverse that reads `input[c * h + r]`.
            // Our existing Av1Inverse2dTransform reads row-major for
            // pipeline-internal consistency with the rest of our decoder, so
            // the forward writes row-major to pair with it.
            for (int c = 0; c < w; c++) output[r * w + c] = rowOut[c];
        }
    }

    /// <summary>Apply a 1D forward transform of the requested type and length.</summary>
    private static void Apply1D(Av1Tx1dType type, int n,
        ReadOnlySpan<int> input, Span<int> output, int cosBit)
    {
        switch (type)
        {
            case Av1Tx1dType.Dct:
                switch (n)
                {
                    case 4:  Av1ForwardDct4 .Transform(input, output, cosBit); return;
                    case 8:  Av1ForwardDct8 .Transform(input, output, cosBit); return;
                    case 16: Av1ForwardDct16.Transform(input, output, cosBit); return;
                    case 32: Av1ForwardDct32.Transform(input, output, cosBit); return;
                    default:
                        throw new NotImplementedException($"Av1ForwardDct{n} not yet implemented (64-point pending).");
                }
            case Av1Tx1dType.Adst:
            case Av1Tx1dType.FlipAdst:
                switch (n)
                {
                    case 4:  Av1ForwardAdst4 .Transform(input, output, cosBit); return;
                    case 8:  Av1ForwardAdst8 .Transform(input, output, cosBit); return;
                    case 16: Av1ForwardAdst16.Transform(input, output, cosBit); return;
                    default:
                        throw new NotImplementedException($"Av1ForwardAdst{n} not yet implemented.");
                }
            case Av1Tx1dType.Identity:
                // libaom av1_fidentity{4,8,16,32}_c: output = round_shift(input * NewSqrt2, NewSqrt2Bits)
                // for sizes 4 and 16; output = input * 2 for size 8; output = input * 4 for size 32.
                // Pure-integer, no cospi.
                ApplyIdentity(n, input, output);
                return;
            default:
                throw new ArgumentException($"Unknown 1D transform type {type}", nameof(type));
        }
    }

    /// <summary>libaom <c>av1_fidentity{4,8,16,32}_c</c>.</summary>
    private static void ApplyIdentity(int n, ReadOnlySpan<int> input, Span<int> output)
    {
        const int NewSqrt2 = 5793;       // sqrt(2) * 2^12 rounded
        const int NewSqrt2Bits = 12;
        switch (n)
        {
            case 4:
            case 16:
                for (int i = 0; i < n; i++)
                    output[i] = RoundShiftLong((long)input[i] * NewSqrt2, NewSqrt2Bits);
                return;
            case 8:
                for (int i = 0; i < n; i++) output[i] = input[i] * 2;
                return;
            case 32:
                for (int i = 0; i < n; i++) output[i] = input[i] * 4;
                return;
            default:
                throw new NotImplementedException($"Av1ForwardIdentity{n} not yet implemented.");
        }
    }

    /// <summary>libaom round-shift: arithmetic round-half-up by <paramref name="bit"/> bits.</summary>
    private static int RoundShift(int value, int bit)
    {
        if (bit == 0) return value;
        if (bit < 0) return value << -bit;
        return (value + (1 << (bit - 1))) >> bit;
    }

    /// <summary>libaom round_shift on a long: <c>(value + (1 &lt;&lt; (bit-1))) &gt;&gt; bit</c>.</summary>
    private static int RoundShiftLong(long value, int bit)
    {
        if (bit <= 0) throw new ArgumentOutOfRangeException(nameof(bit));
        return (int)((value + (1L << (bit - 1))) >> bit);
    }

    /// <summary>libaom <c>av1_fwd_txfm_shift_ls</c> per-tx-size 3-element shift array.</summary>
    private static (int s0, int s1, int s2) GetShifts(Av1TxSize txSize)
    {
        return txSize switch
        {
            Av1TxSize.Tx4x4   => (2,  0, 0),
            Av1TxSize.Tx8x8   => (2, -1, 0),
            Av1TxSize.Tx16x16 => (2, -2, 0),
            Av1TxSize.Tx32x32 => (2, -4, 0),
            Av1TxSize.Tx64x64 => (0, -2, -2),
            Av1TxSize.Tx4x8   => (2, -1, 0),
            Av1TxSize.Tx8x4   => (2, -1, 0),
            Av1TxSize.Tx8x16  => (2, -2, 0),
            Av1TxSize.Tx16x8  => (2, -2, 0),
            Av1TxSize.Tx16x32 => (2, -4, 0),
            Av1TxSize.Tx32x16 => (2, -4, 0),
            Av1TxSize.Tx32x64 => (0, -2, -2),
            Av1TxSize.Tx64x32 => (2, -4, -2),
            Av1TxSize.Tx4x16  => (2, -1, 0),
            Av1TxSize.Tx16x4  => (2, -1, 0),
            Av1TxSize.Tx8x32  => (2, -2, 0),
            Av1TxSize.Tx32x8  => (2, -2, 0),
            Av1TxSize.Tx16x64 => (0, -2, 0),
            Av1TxSize.Tx64x16 => (2, -4, 0),
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
    }

    /// <summary>libaom <c>av1_fwd_cos_bit_col</c> + <c>av1_fwd_cos_bit_row</c> for square sizes.</summary>
    private static (int cosBitCol, int cosBitRow) GetCosBits(Av1TxSize txSize)
    {
        return txSize switch
        {
            Av1TxSize.Tx4x4   => (13, 13),
            Av1TxSize.Tx8x8   => (13, 13),
            Av1TxSize.Tx16x16 => (13, 12),
            Av1TxSize.Tx32x32 => (12, 12),
            _ => throw new NotImplementedException($"GetCosBits for {txSize} not yet wired - non-square + 64x64 land later"),
        };
    }
}
