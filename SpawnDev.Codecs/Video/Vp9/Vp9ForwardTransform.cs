// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 forward-transform dispatcher. Picks the right forward transform
// based on transform size + type and runs it on a 1D vector.
//
// Per VP9 spec sec 8.7 (intra mode -> tx_type):
//   DC / V / H / TM / D45 / D63 / D117 / D135 / D153 / D207 -> DCT_DCT
//   ADST is reserved for future 4x4 DCT_ADST / ADST_DCT / ADST_ADST
//   on certain intra modes; but VP9 reference frames typically use
//   DCT_DCT or DCT_ADST/ADST_DCT/ADST_ADST per the intra mode lookup.
//
// Currently supported:
//   - Tx4x4 + DCT_DCT (Vp9ForwardDct4x4)
//   - Tx4x4 + ADST_ADST (Vp9ForwardAdst4 on rows + cols)
//   - Tx8x8 + DCT_DCT (Vp9ForwardDct8x8)
//   - Tx8x8 + ADST_ADST (Vp9ForwardAdst8 on rows + cols)
//   - Tx16x16 + DCT_DCT (Vp9ForwardDct16x16)
//
// Mixed DCT/ADST 1D combinations (DCT_ADST etc) require composing
// fdct + fadst across rows/cols separately - layer in when needed.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 forward-transform dispatcher.</summary>
public static class Vp9ForwardTransform
{
    /// <summary>
    /// Apply the appropriate forward transform for (txSize, txType) on
    /// a 2D residual block. Output is in raster order.
    /// </summary>
    /// <param name="txSize">Transform block size (Tx4x4 / Tx8x8 / Tx16x16).</param>
    /// <param name="txType">Transform type pair (DctDct / DctAdst / AdstDct / AdstAdst).</param>
    /// <param name="input">Input residual samples (rowStride * N entries).</param>
    /// <param name="rowStrideShorts">Row stride of input in shorts.</param>
    /// <param name="output">Output coefficients (N*N entries raster).</param>
    public static void Apply(Vp9TxSize txSize, Vp9TxType txType,
                             ReadOnlySpan<short> input, int rowStrideShorts,
                             Span<int> output)
    {
        switch (txSize)
        {
            case Vp9TxSize.Tx4x4:
                Apply4x4(txType, input, rowStrideShorts, output);
                break;
            case Vp9TxSize.Tx8x8:
                Apply8x8(txType, input, rowStrideShorts, output);
                break;
            case Vp9TxSize.Tx16x16:
                Apply16x16(txType, input, rowStrideShorts, output);
                break;
            case Vp9TxSize.Tx32x32:
                throw new NotImplementedException("Vp9 fdct32x32 not yet ported - use libvpx vpx_fdct32x32_c reference");
            default:
                throw new ArgumentOutOfRangeException(nameof(txSize), txSize, "Unknown VP9 transform size");
        }
    }

    private static void Apply4x4(Vp9TxType txType, ReadOnlySpan<short> input, int rowStride, Span<int> output)
    {
        switch (txType)
        {
            case Vp9TxType.DctDct:
                Vp9ForwardDct4x4.Transform(input, rowStride, output);
                break;
            case Vp9TxType.AdstAdst:
                ApplyAdstAdst4x4(input, rowStride, output);
                break;
            case Vp9TxType.DctAdst:
            case Vp9TxType.AdstDct:
                throw new NotImplementedException(
                    "VP9 mixed DCT/ADST 4x4 not yet ported - compose fdct4 + fadst4 across rows/cols");
            default:
                throw new ArgumentOutOfRangeException(nameof(txType));
        }
    }

    private static void Apply8x8(Vp9TxType txType, ReadOnlySpan<short> input, int rowStride, Span<int> output)
    {
        switch (txType)
        {
            case Vp9TxType.DctDct:
                Vp9ForwardDct8x8.Transform(input, rowStride, output);
                break;
            case Vp9TxType.AdstAdst:
                ApplyAdstAdst8x8(input, rowStride, output);
                break;
            case Vp9TxType.DctAdst:
            case Vp9TxType.AdstDct:
                throw new NotImplementedException(
                    "VP9 mixed DCT/ADST 8x8 not yet ported - compose fdct8 + fadst8 across rows/cols");
            default:
                throw new ArgumentOutOfRangeException(nameof(txType));
        }
    }

    private static void Apply16x16(Vp9TxType txType, ReadOnlySpan<short> input, int rowStride, Span<int> output)
    {
        switch (txType)
        {
            case Vp9TxType.DctDct:
                Vp9ForwardDct16x16.Transform(input, rowStride, output);
                break;
            case Vp9TxType.AdstAdst:
            case Vp9TxType.DctAdst:
            case Vp9TxType.AdstDct:
                throw new NotImplementedException(
                    "VP9 16x16 ADST not yet ported - libvpx fadst16 has 4 stages, port from vp9_dct.c");
            default:
                throw new ArgumentOutOfRangeException(nameof(txType));
        }
    }

    private static void ApplyAdstAdst4x4(ReadOnlySpan<short> input, int rowStride, Span<int> output)
    {
        // Pass 1: column ADST.
        Span<int> intermediate = stackalloc int[16];
        Span<int> col = stackalloc int[4];
        for (int c = 0; c < 4; c++)
        {
            for (int r = 0; r < 4; r++) col[r] = input[r * rowStride + c];
            Span<int> outCol = stackalloc int[4];
            Vp9ForwardAdst4.Transform(col, outCol);
            for (int r = 0; r < 4; r++) intermediate[r * 4 + c] = outCol[r];
        }

        // Pass 2: row ADST.
        Span<int> row = stackalloc int[4];
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++) row[c] = intermediate[r * 4 + c];
            Span<int> outRow = stackalloc int[4];
            Vp9ForwardAdst4.Transform(row, outRow);
            for (int c = 0; c < 4; c++) output[r * 4 + c] = outRow[c];
        }
    }

    private static void ApplyAdstAdst8x8(ReadOnlySpan<short> input, int rowStride, Span<int> output)
    {
        Span<int> intermediate = stackalloc int[64];
        Span<int> col = stackalloc int[8];
        for (int c = 0; c < 8; c++)
        {
            for (int r = 0; r < 8; r++) col[r] = input[r * rowStride + c];
            Span<int> outCol = stackalloc int[8];
            Vp9ForwardAdst8.Transform(col, outCol);
            for (int r = 0; r < 8; r++) intermediate[r * 8 + c] = outCol[r];
        }

        Span<int> row = stackalloc int[8];
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++) row[c] = intermediate[r * 8 + c];
            Span<int> outRow = stackalloc int[8];
            Vp9ForwardAdst8.Transform(row, outRow);
            for (int c = 0; c < 8; c++) output[r * 8 + c] = outRow[c];
        }
    }
}
