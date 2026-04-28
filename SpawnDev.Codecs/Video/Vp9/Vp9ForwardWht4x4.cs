// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4x4 forward Walsh-Hadamard transform (encoder side, lossless mode).
// Bit-exact port of libvpx vpx_dsp/fwd_txfm.c vpx_fwht4x4_c.
//
// In VP9 the WHT is used INSTEAD of the DCT when lossless is enabled
// (frame_header.lossless == 1). The libvpx encoder normalizes the input
// by 4 (input <<= 2) BEFORE calling fwht4x4 in the lossless path; the
// transform itself is unweighted Hadamard with a +1 rounding bias on
// the negative branch and a final UNIT_QUANT_FACTOR (= 4) multiplier
// applied during the second pass so the inverse (vpx_iwht4x4_16_add)
// can lossy-recover the input via simple right shifts.
//
// Two-pass: pass 1 rows -> intermediate, pass 2 cols -> output.
// Pass 2 multiplies by UNIT_QUANT_FACTOR (= 4) to give the encoder a
// scale factor that the lossless inverse undoes via (val + 1) >> 2
// (libvpx vpx_iwht4x4_16_add). Output type is int (tran_low_t).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 4x4 forward Walsh-Hadamard (encoder side, lossless mode).</summary>
public static class Vp9ForwardWht4x4
{
    /// <summary>libvpx UNIT_QUANT_FACTOR = 4 - applied during pass 2.</summary>
    public const int UnitQuantFactor = 4;

    /// <summary>
    /// Forward 4x4 Walsh-Hadamard. Mirrors libvpx <c>vpx_fwht4x4_c</c>.
    /// </summary>
    /// <param name="input">Input samples (rowStride * 4 entries).</param>
    /// <param name="rowStrideShorts">Row stride in shorts.</param>
    /// <param name="output">16 output coefficients (raster 4x4).</param>
    public static void Transform(ReadOnlySpan<short> input, int rowStrideShorts, Span<int> output)
    {
        if (input.Length < rowStrideShorts * 4)
            throw new ArgumentException($"input must have at least {rowStrideShorts * 4} entries", nameof(input));
        if (output.Length < 16)
            throw new ArgumentException("output must have 16 entries", nameof(output));

        // Pass 1: 4 rows -> output[col, row*4+col] (column-major write
        // back into output, libvpx writes op[0/4/8/12] per column).
        for (int i = 0; i < 4; i++)
        {
            long a1 = input[i + 0 * rowStrideShorts];
            long b1 = input[i + 1 * rowStrideShorts];
            long c1 = input[i + 2 * rowStrideShorts];
            long d1 = input[i + 3 * rowStrideShorts];

            a1 += b1;
            d1 = d1 - c1;
            long e1 = (a1 - d1) >> 1;
            b1 = e1 - b1;
            c1 = e1 - c1;
            a1 -= c1;
            d1 += b1;
            output[i + 0]  = (int)a1;
            output[i + 4]  = (int)c1;
            output[i + 8]  = (int)d1;
            output[i + 12] = (int)b1;
        }

        // Pass 2: 4 rows of intermediate (read back from output) -> final.
        // Stage 2 multiplies by UNIT_QUANT_FACTOR (= 4).
        for (int i = 0; i < 4; i++)
        {
            long a1 = output[i * 4 + 0];
            long b1 = output[i * 4 + 1];
            long c1 = output[i * 4 + 2];
            long d1 = output[i * 4 + 3];

            a1 += b1;
            d1 -= c1;
            long e1 = (a1 - d1) >> 1;
            b1 = e1 - b1;
            c1 = e1 - c1;
            a1 -= c1;
            d1 += b1;
            output[i * 4 + 0] = (int)(a1 * UnitQuantFactor);
            output[i * 4 + 1] = (int)(c1 * UnitQuantFactor);
            output[i * 4 + 2] = (int)(d1 * UnitQuantFactor);
            output[i * 4 + 3] = (int)(b1 * UnitQuantFactor);
        }
    }
}
