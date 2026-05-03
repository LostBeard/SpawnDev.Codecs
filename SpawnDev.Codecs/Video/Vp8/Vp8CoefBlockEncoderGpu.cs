// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 per-block coefficient encoder, GPU-callable form. Bit-exact
// mirror of Vp8CoefBlockEncoder.Encode. Written as static helpers
// taking the bool encoder state by ref + flat constant tables as
// ArrayView<byte> so it composes inside an ILGPU kernel.
//
// Probability table layout (probsFlat):
//   probsFlat[band * 33 + ctx * 11 + node] = probs[band, ctx, node]
//   - 8 bands (0..7)
//   - 3 prev-coef contexts (0..2)
//   - 11 entropy nodes (0..10)
//   Total 264 bytes per block type. Each block type has its own
//   probsFlat view; caller picks the right one (Y4-no-DC = 0,
//   Y2 = 1, UV = 2, Y_with_DC = 3) before calling.
//
// Constants buffer layout (constsFlat) - 56 bytes total:
//   [0..15]   zigzag scan (16 bytes)
//   [16..32]  coef bands (17 bytes including sentinel)
//   [33..35]  cat3 probs (3 bytes)
//   [36..39]  cat4 probs (4 bytes)
//   [40..44]  cat5 probs (5 bytes)
//   [45..55]  cat6 probs (11 bytes)
// Caller materializes this layout once per accelerator and reuses.

using ILGPU;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 per-block (4x4) coefficient encoder, GPU-callable. Bit-exact
/// port of <see cref="Vp8CoefBlockEncoder"/>.
/// </summary>
public static class Vp8CoefBlockEncoderGpu
{
    /// <summary>Layout offsets within the consolidated constsFlat buffer.</summary>
    public const int ZigzagOffset = 0;
    /// <summary>bands offset.</summary>
    public const int BandsOffset = 16;
    /// <summary>cat3 probs offset.</summary>
    public const int Cat3Offset = 33;
    /// <summary>cat4 probs offset.</summary>
    public const int Cat4Offset = 36;
    /// <summary>cat5 probs offset.</summary>
    public const int Cat5Offset = 40;
    /// <summary>cat6 probs offset.</summary>
    public const int Cat6Offset = 45;
    /// <summary>Total bytes in the constsFlat buffer.</summary>
    public const int ConstsTotalBytes = 56;

    private const int CtxPerBand = 3;
    private const int NodesPerCtx = 11;
    private const int ProbsRowSize = CtxPerBand * NodesPerCtx; // 33

    /// <summary>
    /// Build the consolidated constants buffer the GPU encoder reads.
    /// Layout matches the offset constants above. Caller uploads once
    /// per accelerator.
    /// </summary>
    public static byte[] BuildConstsBuffer()
    {
        var buf = new byte[ConstsTotalBytes];
        Array.Copy(Vp8CoefTables.ZigzagScan, 0, buf, ZigzagOffset, 16);
        Array.Copy(Vp8CoefTables.CoefBands, 0, buf, BandsOffset, 17);
        Array.Copy(Vp8CoefTables.Cat3Probs, 0, buf, Cat3Offset, 3);
        Array.Copy(Vp8CoefTables.Cat4Probs, 0, buf, Cat4Offset, 4);
        Array.Copy(Vp8CoefTables.Cat5Probs, 0, buf, Cat5Offset, 5);
        Array.Copy(Vp8CoefTables.Cat6Probs, 0, buf, Cat6Offset, 11);
        return buf;
    }

    /// <summary>
    /// Encode one 16-element coefficient block. Returns the EOB
    /// position (0 if all zero, else 1 + last non-zero scan slot).
    /// </summary>
    public static int Encode(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        int ctx,
        int firstCoef,
        ArrayView<short> coefs)
    {
        // Find EOB.
        int eob = 0;
        for (int scan = 15; scan >= firstCoef; scan--)
        {
            int raster = constsFlat[ZigzagOffset + scan];
            if (coefs[raster] != 0) { eob = scan + 1; break; }
        }

        int n = firstCoef;
        int pBand = constsFlat[BandsOffset + n];
        int pCtx = ctx;

        // First emit: "block is empty?"
        if (eob <= firstCoef)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0,
                probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 0]);
            return 0;
        }
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1,
            probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 0]);

        // Walk scan positions emitting per-position decisions.
        bool done = false;
        while (!done)
        {
            n++;
            int rasterPrev = constsFlat[ZigzagOffset + (n - 1)];
            int v = coefs[rasterPrev];

            if (v == 0)
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0,
                    probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 1]);
                pBand = constsFlat[BandsOffset + n];
                pCtx = 0;
            }
            else
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1,
                    probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 1]);
                int absV = v < 0 ? -v : v;
                if (absV == 1)
                {
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0,
                        probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 2]);
                    int newPBand = constsFlat[BandsOffset + n];
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, v < 0 ? 1 : 0, 128);
                    pBand = newPBand;
                    pCtx = 1;
                }
                else
                {
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1,
                        probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 2]);
                    EncodeAtLeastTwo(ref state, outBuf, probsFlat, constsFlat, pBand, pCtx, absV);
                    int newPBand = constsFlat[BandsOffset + n];
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, v < 0 ? 1 : 0, 128);
                    pBand = newPBand;
                    pCtx = 2;
                }

                if (n < 16)
                {
                    if (n == eob)
                    {
                        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0,
                            probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 0]);
                        return eob;
                    }
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1,
                        probsFlat[pBand * ProbsRowSize + pCtx * NodesPerCtx + 0]);
                }
            }
            if (n == 16) done = true;
        }
        return eob;
    }

    private static void EncodeAtLeastTwo(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        int band,
        int ctx,
        int absV)
    {
        long pRow = (long)band * ProbsRowSize + ctx * NodesPerCtx;

        if (absV == 2)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 4]);
        }
        else if (absV == 3)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, probsFlat[pRow + 4]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 5]);
        }
        else if (absV == 4)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, probsFlat[pRow + 4]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, probsFlat[pRow + 5]);
        }
        else
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, probsFlat[pRow + 3]);
            if (absV >= 5 && absV <= 10)
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 6]);
                if (absV >= 5 && absV <= 6)
                {
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, probsFlat[pRow + 7]);
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, absV - 5, 159);
                }
                else
                {
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, probsFlat[pRow + 7]);
                    int delta = absV - 7;
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, (delta >> 1) & 1, 165);
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, delta & 1, 145);
                }
            }
            else
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, probsFlat[pRow + 6]);
                int catOffset;
                int minVal;
                int width;
                if (absV <= 18) { catOffset = Cat3Offset; minVal = 11; width = 3; }
                else if (absV <= 34) { catOffset = Cat4Offset; minVal = 19; width = 4; }
                else if (absV <= 66) { catOffset = Cat5Offset; minVal = 35; width = 5; }
                else { catOffset = Cat6Offset; minVal = 67; width = 11; }

                int cat = (catOffset == Cat3Offset) ? 0
                       : (catOffset == Cat4Offset) ? 1
                       : (catOffset == Cat5Offset) ? 2
                       : 3;
                int bit1 = (cat >> 1) & 1;
                int bit0 = cat & 1;
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, bit1, probsFlat[pRow + 8]);
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, bit0, probsFlat[pRow + 9 + bit1]);

                int extra = absV - minVal;
                for (int i = 0; i < width; i++)
                {
                    int bitVal = (extra >> (width - 1 - i)) & 1;
                    Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, bitVal,
                        constsFlat[catOffset + i]);
                }
            }
        }
    }
}
