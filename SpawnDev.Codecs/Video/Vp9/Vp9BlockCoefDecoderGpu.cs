// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block coefficient decoder, GPU-callable form. Bit-exact
// mirror of Vp9BlockCoefDecoder.DecodeBlockCoefficients. Symmetric
// companion to Vp9BlockCoefEncoderGpu - reads the same wire format
// the encoder produces.
//
// Reuses Vp8BoolDecoderGpu for the underlying range coder (the bool
// math is identical between VP8 and VP9 per Vp9BoolDecoder.cs comment;
// VP9 only adds a leading marker bit consumed by the caller during
// init right after Init).
//
// The constants buffer layout, tree-walk strategy, and per-thread
// tokenCache scratch convention match Vp9BlockCoefEncoderGpu - both
// sides share BuildConstsBuffer() so a single upload supports both
// encode and decode kernels in one frame integration class.
//
// Constrained-tree walk: instead of reading the
// Vp9CoefTrees.CoefConTree array at runtime, the decoder inlines the
// fixed tree topology as a 4-level nested if. This avoids loading the
// tree into the consts buffer (which would only be 16 bytes anyway)
// and lets each backend's branch predictor see the leaf decisions
// directly.

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 per-block coefficient decoder, GPU-callable. Bit-exact port
/// of <see cref="Vp9BlockCoefDecoder"/>. Pairs with
/// <see cref="Vp8BoolDecoderGpu"/> for the underlying range coder
/// and shares the consts buffer with
/// <see cref="Vp9BlockCoefEncoderGpu"/>.
/// </summary>
public static class Vp9BlockCoefDecoderGpu
{
    /// <summary>
    /// Decode the coefficients of a single transform block in place.
    /// Returns the EOB position (count of decoded scan slots, 0..maxCoefs).
    /// Mirrors <see cref="Vp9BlockCoefDecoder.DecodeBlockCoefficients"/>
    /// bit-for-bit.
    /// </summary>
    /// <param name="state">Bool-decoder state, by ref.</param>
    /// <param name="inBuf">Input bitstream buffer.</param>
    /// <param name="block">
    /// Output coefficient block in raster layout. Pre-zeroed by the
    /// algorithm before decoding starts; entries past
    /// <paramref name="maxCoefs"/> are left untouched.
    /// </param>
    /// <param name="scan">Scan table for the active (txSize, scanType) pair.</param>
    /// <param name="neighbors">Neighbor table for the active (txSize, scanType) pair.</param>
    /// <param name="coefProbs">Default / per-frame coef-prob model bytes for this transform size.</param>
    /// <param name="consts">Packed consts buffer; see Vp9BlockCoefEncoderGpu.</param>
    /// <param name="tokenCache">
    /// Per-thread scratch for raster-position energy classes.
    /// Caller supplies a buffer of size <paramref name="maxCoefs"/>;
    /// the decoder zeros it before reading.
    /// </param>
    /// <param name="maxCoefs">Block size: 16, 64, 256, or 1024.</param>
    /// <param name="planeType">0 = Y, 1 = UV.</param>
    /// <param name="refType">0 = Intra, 1 = Inter.</param>
    /// <param name="initialCtx">Per-plane entropy context for scan position 0.</param>
    /// <param name="isHighBitDepth">1 = 12-bit profile, 0 = 8-bit profile (selects Cat6 prob table).</param>
    public static int DecodeBlock(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> inBuf,
        ArrayView<short> block,
        ArrayView<ushort> scan,
        ArrayView<ushort> neighbors,
        ArrayView<byte> coefProbs,
        ArrayView<byte> consts,
        ArrayView<byte> tokenCache,
        int maxCoefs,
        int planeType,
        int refType,
        int initialCtx,
        int isHighBitDepth)
    {
        // Pre-zero the output block we will write into.
        for (int i = 0; i < maxCoefs; i++) block[i] = 0;
        // Pre-zero the tokenCache (caller may pass uninitialised storage).
        for (int i = 0; i < maxCoefs; i++) tokenCache[i] = 0;

        int c = 0;
        int firstIter = 1;
        while (c < maxCoefs)
        {
            // Compute (band, ctx) for current scan position.
            int isTx4x4 = maxCoefs == 16 ? 1 : 0;
            int band = isTx4x4 != 0
                ? consts[Vp9BlockCoefEncoderGpu.Band4x4Offset + c]
                : consts[Vp9BlockCoefEncoderGpu.Band8x8PlusOffset + c];
            int ctx = firstIter != 0
                ? initialCtx
                : GetCoefContext(neighbors, tokenCache, c);
            firstIter = 0;

            int modelBase = Vp9CoefProbs.Index4x4(planeType, refType, band, ctx, 0);
            byte m0 = coefProbs[modelBase + 0];
            byte m1 = coefProbs[modelBase + 1];
            byte m2 = coefProbs[modelBase + 2];

            // EOB? bit (libvpx: read==0 means EOB).
            if (Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, m0) == 0) return c;

            // Inner ZERO loop.
            while (Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, m1) == 0)
            {
                // ZERO: block[scan[c]] stays 0 from pre-zero;
                // tokenCache[scan[c]] also 0 (PtEnergyClass[Zero] = 0
                // so no update needed).
                c++;
                if (c >= maxCoefs) return c;

                band = isTx4x4 != 0
                    ? consts[Vp9BlockCoefEncoderGpu.Band4x4Offset + c]
                    : consts[Vp9BlockCoefEncoderGpu.Band8x8PlusOffset + c];
                ctx = GetCoefContext(neighbors, tokenCache, c);
                modelBase = Vp9CoefProbs.Index4x4(planeType, refType, band, ctx, 0);
                m0 = coefProbs[modelBase + 0];
                m1 = coefProbs[modelBase + 1];
                m2 = coefProbs[modelBase + 2];
            }

            // Non-zero token. Decode magnitude.
            int magnitude;
            byte tokenByte;
            if (Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, m2) == 0)
            {
                tokenByte = (byte)Vp9CoefToken.One;
                magnitude = 1;
            }
            else
            {
                // Walk constrained tree via the pareto8 row driven by m2.
                long pareto8RowBase = (long)Vp9BlockCoefEncoderGpu.Pareto8FullOffset
                                    + (long)(m2 - 1) * 8;
                tokenByte = DecodeTreeWalk(ref state, inBuf, consts, pareto8RowBase);

                if (tokenByte == (byte)Vp9CoefToken.Two) magnitude = 2;
                else if (tokenByte == (byte)Vp9CoefToken.Three) magnitude = 3;
                else if (tokenByte == (byte)Vp9CoefToken.Four) magnitude = 4;
                else
                {
                    // Cat1..Cat6 -> read residual MSB-first + add minVal.
                    magnitude = DecodeCategoryMagnitude(
                        ref state, inBuf, consts, tokenByte, isHighBitDepth);
                }
            }

            // Sign bit at flat probability 128.
            int sign = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, 128);
            int value = sign != 0 ? -magnitude : magnitude;

            int rasterC = (int)scan[c];
            block[rasterC] = (short)value;
            tokenCache[rasterC] = consts[Vp9BlockCoefEncoderGpu.PtEnergyClassOffset + tokenByte];
            c++;
        }

        return c;
    }

    /// <summary>
    /// Compute the 0/1/2 entropy context value at scan position
    /// <paramref name="scanPos"/> from the two raster-neighbor energy
    /// classes. Mirrors <see cref="Vp9CoefContext.GetCoefContext"/>.
    /// </summary>
    private static int GetCoefContext(
        ArrayView<ushort> neighbors,
        ArrayView<byte> tokenCache,
        int scanPos)
    {
        int n0Index = 2 * scanPos;
        int n1Index = n0Index + 1;
        int n0 = neighbors[n0Index];
        int n1 = neighbors[n1Index];
        int e0 = tokenCache[n0];
        int e1 = tokenCache[n1];
        return (1 + e0 + e1) >> 1;
    }

    /// <summary>
    /// Decode the constrained coefficient tree starting at the LOW_VAL
    /// root and returning the resulting <see cref="Vp9CoefToken"/>.
    /// Mirrors <see cref="Vp9CoefTrees.DecodeConToken"/> with the tree
    /// topology unrolled as nested ifs - no array load of the tree
    /// itself, no kernel-side recursion.
    /// </summary>
    private static byte DecodeTreeWalk(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> inBuf,
        ArrayView<byte> consts,
        long pareto8RowBase)
    {
        int b0 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 0]);
        if (b0 == 0)
        {
            // Two/Three/Four path
            int b1 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 1]);
            if (b1 == 0) return (byte)Vp9CoefToken.Two;
            int b2 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 2]);
            return b2 == 0 ? (byte)Vp9CoefToken.Three : (byte)Vp9CoefToken.Four;
        }
        // Cat1..Cat6 path
        int b3 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 3]);
        if (b3 == 0)
        {
            int b4 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 4]);
            return b4 == 0 ? (byte)Vp9CoefToken.Category1 : (byte)Vp9CoefToken.Category2;
        }
        int b5 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 5]);
        if (b5 == 0)
        {
            int b6 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 6]);
            return b6 == 0 ? (byte)Vp9CoefToken.Category3 : (byte)Vp9CoefToken.Category4;
        }
        int b7 = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[pareto8RowBase + 7]);
        return b7 == 0 ? (byte)Vp9CoefToken.Category5 : (byte)Vp9CoefToken.Category6;
    }

    /// <summary>
    /// Decode the residual magnitude bits for a Cat1..Cat6 token,
    /// MSB-first, then add the per-category minVal. Mirrors
    /// <see cref="Vp9CoefProbs.DecodeCategoryMagnitude"/>.
    /// </summary>
    private static int DecodeCategoryMagnitude(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> inBuf,
        ArrayView<byte> consts,
        byte tokenByte,
        int isHighBitDepth)
    {
        int probOffset;
        int probLen;
        int minVal;
        if (tokenByte == (byte)Vp9CoefToken.Category1)
        { probOffset = Vp9BlockCoefEncoderGpu.Cat1ProbOffset; probLen = 1; minVal = Vp9CoefProbs.CatMinVal.Cat1; }
        else if (tokenByte == (byte)Vp9CoefToken.Category2)
        { probOffset = Vp9BlockCoefEncoderGpu.Cat2ProbOffset; probLen = 2; minVal = Vp9CoefProbs.CatMinVal.Cat2; }
        else if (tokenByte == (byte)Vp9CoefToken.Category3)
        { probOffset = Vp9BlockCoefEncoderGpu.Cat3ProbOffset; probLen = 3; minVal = Vp9CoefProbs.CatMinVal.Cat3; }
        else if (tokenByte == (byte)Vp9CoefToken.Category4)
        { probOffset = Vp9BlockCoefEncoderGpu.Cat4ProbOffset; probLen = 4; minVal = Vp9CoefProbs.CatMinVal.Cat4; }
        else if (tokenByte == (byte)Vp9CoefToken.Category5)
        { probOffset = Vp9BlockCoefEncoderGpu.Cat5ProbOffset; probLen = 5; minVal = Vp9CoefProbs.CatMinVal.Cat5; }
        else // Category6
        {
            minVal = Vp9CoefProbs.CatMinVal.Cat6;
            if (isHighBitDepth != 0) { probOffset = Vp9BlockCoefEncoderGpu.Cat6ProbHigh12Offset; probLen = 18; }
            else { probOffset = Vp9BlockCoefEncoderGpu.Cat6ProbOffset; probLen = 14; }
        }

        int value = 0;
        for (int i = 0; i < probLen; i++)
        {
            int bit = Vp8BoolDecoderGpu.DecodeBool(ref state, inBuf, consts[probOffset + i]);
            value = (value << 1) | bit;
        }
        return minVal + value;
    }
}
