// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4x4 forward Walsh-Hadamard transform, GPU-callable form for
// in-kernel reuse. Bit-exact mirror of Vp9ForwardWht4x4.Transform
// (libvpx vpx_fwht4x4_c port).
//
// The WHT is VP9's lossless-mode transform, used instead of the DCT
// when frame_header.lossless == 1. Two-pass: pass 1 row WHT then pass
// 2 column WHT with UNIT_QUANT_FACTOR (= 4) multiplier so the lossless
// inverse can recover via simple right shifts.
//
// Single-thread per block.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 4x4 forward Walsh-Hadamard helper. Bit-exact mirror of
/// <see cref="Vp9ForwardWht4x4"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardWht4x4Gpu
{
    /// <summary>libvpx UNIT_QUANT_FACTOR = 4.</summary>
    public const int UnitQuantFactor = 4;

    /// <summary>
    /// Forward 4x4 Walsh-Hadamard. Reads <paramref name="input"/> at the
    /// given row stride from <paramref name="inBase"/>; writes 16 output
    /// coefficients (raster 4x4) to <paramref name="output"/> at
    /// <paramref name="outBase"/>.
    /// </summary>
    public static void Transform(
        ArrayView<short> input, long inBase, int rowStrideShorts,
        ArrayView<int> output, long outBase)
    {
        // Pass 1: 4 rows -> output (column-major write).
        for (int i = 0; i < 4; i++)
        {
            long a1 = input[inBase + i + 0 * rowStrideShorts];
            long b1 = input[inBase + i + 1 * rowStrideShorts];
            long c1 = input[inBase + i + 2 * rowStrideShorts];
            long d1 = input[inBase + i + 3 * rowStrideShorts];

            a1 += b1;
            d1 = d1 - c1;
            long e1 = (a1 - d1) >> 1;
            b1 = e1 - b1;
            c1 = e1 - c1;
            a1 -= c1;
            d1 += b1;
            output[outBase + i + 0]  = (int)a1;
            output[outBase + i + 4]  = (int)c1;
            output[outBase + i + 8]  = (int)d1;
            output[outBase + i + 12] = (int)b1;
        }

        // Pass 2: 4 rows of intermediate -> final (* UnitQuantFactor).
        for (int i = 0; i < 4; i++)
        {
            long a1 = output[outBase + i * 4 + 0];
            long b1 = output[outBase + i * 4 + 1];
            long c1 = output[outBase + i * 4 + 2];
            long d1 = output[outBase + i * 4 + 3];

            a1 += b1;
            d1 -= c1;
            long e1 = (a1 - d1) >> 1;
            b1 = e1 - b1;
            c1 = e1 - c1;
            a1 -= c1;
            d1 += b1;
            output[outBase + i * 4 + 0] = (int)(a1 * UnitQuantFactor);
            output[outBase + i * 4 + 1] = (int)(c1 * UnitQuantFactor);
            output[outBase + i * 4 + 2] = (int)(d1 * UnitQuantFactor);
            output[outBase + i * 4 + 3] = (int)(b1 * UnitQuantFactor);
        }
    }
}
