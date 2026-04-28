// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 per-block coefficient decoder, GPU-callable form. Bit-exact
// mirror of Vp8CoefBlockDecoder.Decode. Symmetric companion to
// Vp8CoefBlockEncoderGpu - reuses the same 56-byte constsFlat layout
// (zigzag + bands + cat3-6) and the same 264-byte probsFlat layout
// (8 bands * 3 ctx * 11 nodes per block type).

using ILGPU;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 per-block (4x4) coefficient decoder, GPU-callable. Bit-exact
/// mirror of <see cref="Vp8CoefBlockDecoder"/>. Returns the EOB
/// position written to the output buffer.
/// </summary>
public static class Vp8CoefBlockDecoderGpu
{
    private const int CtxPerBand = 3;
    private const int NodesPerCtx = 11;
    private const int ProbsRowSize = CtxPerBand * NodesPerCtx; // 33

    /// <summary>
    /// Decode one 16-element coefficient block from the bitstream.
    /// Returns EOB position (0 if all zero, else 1 + last non-zero scan slot).
    /// </summary>
    public static int Decode(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> bitstream,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        int ctx,
        int firstCoef,
        ArrayView<short> output,
        long outputBase)
    {
        // Zero output[outputBase + 0..15].
        for (int i = 0; i < 16; i++) output[outputBase + i] = 0;

        int n = firstCoef;
        int pBand = constsFlat[Vp8CoefBlockEncoderGpu.BandsOffset + n];
        int pCtx = ctx;

        // First emit: "block is empty?".
        if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
            probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 0]) == 0)
            return 0;

        while (true)
        {
            n++;
            // ZERO bit at the OLD (pBand, pCtx).
            if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 1]) == 0)
            {
                pBand = constsFlat[Vp8CoefBlockEncoderGpu.BandsOffset + n];
                pCtx = 0;
            }
            else
            {
                int v;
                if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                    probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 2]) == 0)
                {
                    v = 1;
                    pBand = constsFlat[Vp8CoefBlockEncoderGpu.BandsOffset + n];
                    pCtx = 1;
                }
                else
                {
                    long pRow = (long)pBand * ProbsRowSize + pCtx * NodesPerCtx;
                    if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                        probsFlat[pRow + 3]) == 0)
                    {
                        if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                            probsFlat[pRow + 4]) == 0)
                            v = 2;
                        else
                            v = 3 + Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                                probsFlat[pRow + 5]);
                    }
                    else
                    {
                        if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                            probsFlat[pRow + 6]) == 0)
                        {
                            if (Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                                probsFlat[pRow + 7]) == 0)
                                v = 5 + Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream, 159);
                            else
                            {
                                v = 7 + 2 * Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream, 165);
                                v += Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream, 145);
                            }
                        }
                        else
                        {
                            int bit1 = Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                                probsFlat[pRow + 8]);
                            int bit0 = Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                                probsFlat[pRow + 9 + bit1]);
                            int cat = 2 * bit1 + bit0;
                            int catOffset;
                            int width;
                            if (cat == 0) { catOffset = Vp8CoefBlockEncoderGpu.Cat3Offset; width = 3; }
                            else if (cat == 1) { catOffset = Vp8CoefBlockEncoderGpu.Cat4Offset; width = 4; }
                            else if (cat == 2) { catOffset = Vp8CoefBlockEncoderGpu.Cat5Offset; width = 5; }
                            else { catOffset = Vp8CoefBlockEncoderGpu.Cat6Offset; width = 11; }

                            v = 0;
                            for (int t = 0; t < width; t++)
                                v = v + v + Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                                    constsFlat[catOffset + t]);
                            v += 3 + (8 << cat);
                        }
                    }
                    pBand = constsFlat[Vp8CoefBlockEncoderGpu.BandsOffset + n];
                    pCtx = 2;
                }

                // Sign bit + write to output (zigzag-mapped).
                int sign = Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream, 128);
                int signed = sign != 0 ? -v : v;
                int rasterIdx = constsFlat[Vp8CoefBlockEncoderGpu.ZigzagOffset + (n - 1)];
                output[outputBase + rasterIdx] = (short)signed;

                // EOB check at NEW (pBand, pCtx).
                if (n == 16 || Vp8BoolDecoderGpu.DecodeBool(ref state, bitstream,
                    probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 0]) == 0)
                    return n;
            }
            if (n == 16) return 16;
        }
    }
}
