// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block coefficient encoder, GPU-callable form. Bit-exact
// mirror of Vp9BlockCoefEncoder.EncodeBlockCoefficients. Written as
// static helpers operating on Vp8BoolEncoderGpuState (the bool-coder
// math is identical between VP8 and VP9 per Vp9BoolEncoder.cs:6-9;
// the only VP9-specific addition is a leading marker bit emitted by
// the caller right after Init).
//
// Why the constants buffer is organized this way:
// VP9 coef encoding pulls from many small tables (cat probs, energy
// classes, pareto8, two band tables) that are static per-accelerator.
// Packing them into a single ArrayView<byte> keeps the kernel
// signature within ILGPU's Action arg budget while still letting the
// host upload them once at frame setup time. Per-tx-size tables that
// vary by call (scan, neighbor, coef-probs) stay as separate views
// so the caller picks the right one for the current block.
//
// Layout of the consts buffer (3143 bytes total):
//   [0..1023]    bandTable8x8plus   (1024 bytes - shared across 8x8/16x16/32x32)
//   [1024..1039] bandTable4x4       (16 bytes)
//   [1040..1055] ptEnergyClass      (16 bytes - 12 used, 4 zero pad)
//   [1056..3095] pareto8Full        (255 rows x 8 cols = 2040 bytes)
//   [3096]       Cat1Prob           (1 byte)
//   [3097..3098] Cat2Prob           (2 bytes)
//   [3099..3101] Cat3Prob           (3 bytes)
//   [3102..3105] Cat4Prob           (4 bytes)
//   [3106..3110] Cat5Prob           (5 bytes)
//   [3111..3124] Cat6Prob           (14 bytes)
//   [3125..3142] Cat6ProbHigh12     (18 bytes)
//
// Tree walk: instead of porting the recursive SubtreeContains helper
// the CPU encoder uses, the GPU version unrolls Vp9CoefTrees.CoefConTree
// into a flat switch by token. Each path through the tree is a fixed
// sequence of 2-4 bool emits; tabulating them avoids any kernel-side
// recursion and lowers cleanly on every backend.

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 per-block coefficient encoder, GPU-callable. Bit-exact port
/// of <see cref="Vp9BlockCoefEncoder"/>. Pairs with
/// <see cref="Vp8BoolEncoderGpu"/> for the underlying range coder
/// (VP9 reuses VP8's bool coder math with a leading marker bit).
/// </summary>
public static class Vp9BlockCoefEncoderGpu
{
    /// <summary>Offset of the 8x8plus band lookup (1024 bytes) within the consts buffer.</summary>
    public const int Band8x8PlusOffset = 0;
    /// <summary>Offset of the 4x4 band lookup (16 bytes) within the consts buffer.</summary>
    public const int Band4x4Offset = 1024;
    /// <summary>Offset of the PtEnergyClass table (16 bytes) within the consts buffer.</summary>
    public const int PtEnergyClassOffset = 1040;
    /// <summary>Offset of the Pareto8Full table (255 rows x 8 cols = 2040 bytes) within the consts buffer.</summary>
    public const int Pareto8FullOffset = 1056;
    /// <summary>Offset of the Cat1Prob (1 byte) within the consts buffer.</summary>
    public const int Cat1ProbOffset = 3096;
    /// <summary>Offset of the Cat2Prob (2 bytes) within the consts buffer.</summary>
    public const int Cat2ProbOffset = 3097;
    /// <summary>Offset of the Cat3Prob (3 bytes) within the consts buffer.</summary>
    public const int Cat3ProbOffset = 3099;
    /// <summary>Offset of the Cat4Prob (4 bytes) within the consts buffer.</summary>
    public const int Cat4ProbOffset = 3102;
    /// <summary>Offset of the Cat5Prob (5 bytes) within the consts buffer.</summary>
    public const int Cat5ProbOffset = 3106;
    /// <summary>Offset of the Cat6Prob (14 bytes) within the consts buffer.</summary>
    public const int Cat6ProbOffset = 3111;
    /// <summary>Offset of the Cat6ProbHigh12 (18 bytes) within the consts buffer.</summary>
    public const int Cat6ProbHigh12Offset = 3125;
    /// <summary>Total size of the consts buffer in bytes.</summary>
    public const int ConstsTotalBytes = 3143;

    /// <summary>VP9 model probability size: 3 unconstrained nodes per (tx_size, plane, ref, band, ctx).</summary>
    public const int UnconstrainedNodes = 3;
    /// <summary>VP9 full probability size: 11 entropy nodes per scan position.</summary>
    public const int EntropyNodes = 11;

    /// <summary>
    /// Build the consolidated consts buffer for upload. Caller materialises
    /// once per accelerator and reuses across every coefficient block.
    /// </summary>
    public static byte[] BuildConstsBuffer()
    {
        var buf = new byte[ConstsTotalBytes];

        // bandTable8x8plus + bandTable4x4 - covers every tx size's band lookup.
        Array.Copy(Vp9CoefBands.CoefBandTrans8x8Plus, 0, buf, Band8x8PlusOffset, 1024);
        Array.Copy(Vp9CoefBands.CoefBand4x4, 0, buf, Band4x4Offset, 16);

        // PtEnergyClass: 12 entries (Vp9CoefToken values 0..11). The
        // remaining 4 byte slots stay zero - they are never indexed by
        // the algorithm because tokens are bounded by Vp9CoefToken.Eob = 11.
        Array.Copy(Vp9CoefContext.PtEnergyClass, 0, buf, PtEnergyClassOffset, Vp9CoefContext.PtEnergyClass.Length);

        // Pareto8Full: flatten the 255 x 8 byte[,] into a 2040-byte block
        // in row-major order. The kernel reads pareto[(pivot - 1) * 8 + i].
        int rows = Vp9CoefProbs.Pareto8Full.GetLength(0);
        for (int r = 0; r < rows; r++)
        for (int c = 0; c < 8; c++)
            buf[Pareto8FullOffset + r * 8 + c] = Vp9CoefProbs.Pareto8Full[r, c];

        Array.Copy(Vp9CoefProbs.Cat1Prob, 0, buf, Cat1ProbOffset, 1);
        Array.Copy(Vp9CoefProbs.Cat2Prob, 0, buf, Cat2ProbOffset, 2);
        Array.Copy(Vp9CoefProbs.Cat3Prob, 0, buf, Cat3ProbOffset, 3);
        Array.Copy(Vp9CoefProbs.Cat4Prob, 0, buf, Cat4ProbOffset, 4);
        Array.Copy(Vp9CoefProbs.Cat5Prob, 0, buf, Cat5ProbOffset, 5);
        Array.Copy(Vp9CoefProbs.Cat6Prob, 0, buf, Cat6ProbOffset, 14);
        Array.Copy(Vp9CoefProbs.Cat6ProbHigh12, 0, buf, Cat6ProbHigh12Offset, 18);

        return buf;
    }

    /// <summary>
    /// Encode the coefficients of a single transform block. Returns the
    /// EOB position (1 past the last non-zero scan slot, or 0 if the
    /// block is entirely zero). Mirrors
    /// <see cref="Vp9BlockCoefEncoder.EncodeBlockCoefficients"/> bit-for-bit.
    /// </summary>
    /// <param name="state">Bool-encoder state, by ref.</param>
    /// <param name="outBuf">Output bitstream buffer (caller pre-sizes).</param>
    /// <param name="coefs">
    /// Quantized coefficients in raster layout. Length must be at least
    /// <paramref name="maxCoefs"/>; entries past <paramref name="maxCoefs"/>
    /// are ignored.
    /// </param>
    /// <param name="scan">Scan table for the active (txSize, scanType) pair.</param>
    /// <param name="neighbors">Neighbor table for the active (txSize, scanType) pair.</param>
    /// <param name="coefProbs">
    /// Default coef-prob model bytes for this transform size (libvpx
    /// default_coef_probs_*). 432 bytes for 4x4, larger for the
    /// 8x8plus tables. Indexed via <see cref="Vp9CoefProbs.Index4x4"/>.
    /// </param>
    /// <param name="consts">Packed consts buffer; see class header for layout.</param>
    /// <param name="tokenCache">
    /// Per-thread scratch for raster-position energy classes. Caller
    /// supplies <c>LocalMemory.Allocate&lt;byte&gt;(maxCoefs)</c> and
    /// pre-zeros it (the algorithm overwrites every position it visits;
    /// scan slots that stay ZERO are never read again).
    /// </param>
    /// <param name="maxCoefs">Block size: 16, 64, 256, or 1024.</param>
    /// <param name="planeType">0 = Y, 1 = UV.</param>
    /// <param name="refType">0 = Intra, 1 = Inter.</param>
    /// <param name="initialCtx">Per-plane entropy context for scan position 0.</param>
    /// <param name="isHighBitDepth">1 = 12-bit profile, 0 = 8-bit profile (selects Cat6 prob table).</param>
    /// <param name="isTx4x4">1 = 4x4 transform (band lookup uses Band4x4 table), 0 = 8x8/16x16/32x32 (Band8x8Plus).</param>
    public static int EncodeBlock(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        ArrayView<short> coefs,
        ArrayView<ushort> scan,
        ArrayView<ushort> neighbors,
        ArrayView<byte> coefProbs,
        ArrayView<byte> consts,
        ArrayView<byte> tokenCache,
        int maxCoefs,
        int planeType,
        int refType,
        int initialCtx,
        int isHighBitDepth,
        int isTx4x4)
    {
        // Pre-zero the tokenCache region we may read from. Caller gets
        // LocalMemory back uninitialised on some backends.
        for (int i = 0; i < maxCoefs; i++) tokenCache[i] = 0;

        // Find EOB by scanning backward in scan order.
        int eob = 0;
        for (int i = maxCoefs - 1; i >= 0; i--)
        {
            int raster = (int)scan[i];
            if (coefs[raster] != 0) { eob = i + 1; break; }
        }

        int c = 0;
        int firstIter = 1;
        // Iterative outer loop matches CPU encoder's `while (c < maxCoefs)`.
        while (c < maxCoefs)
        {
            // ComputeProbs into a stack-allocated 11-byte vector. ILGPU
            // doesn't accept stackalloc inside helpers reliably, so we
            // expand inline using local variables for the 4 nodes the
            // algorithm actually reads (full[0], full[1], full[2], and
            // full[3..10] only for the constrained tree walk further down).
            int band = isTx4x4 != 0
                ? consts[Band4x4Offset + c]
                : consts[Band8x8PlusOffset + c];
            int ctx = firstIter != 0 ? initialCtx : GetCoefContext(neighbors, tokenCache, c);
            firstIter = 0;

            int modelBase = Vp9CoefProbs.Index4x4(planeType, refType, band, ctx, 0);
            byte m0 = coefProbs[modelBase + 0];
            byte m1 = coefProbs[modelBase + 1];
            byte m2 = coefProbs[modelBase + 2];

            if (c == eob)
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, m0); // EOB
                return eob;
            }
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, m0);     // !EOB

            // Inner ZERO loop.
            while (coefs[(int)scan[c]] == 0)
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, m1); // ZERO
                c++;
                if (c >= maxCoefs) return eob;

                // Recompute m0..m2 for the next position.
                band = isTx4x4 != 0
                    ? consts[Band4x4Offset + c]
                    : consts[Band8x8PlusOffset + c];
                ctx = GetCoefContext(neighbors, tokenCache, c);
                modelBase = Vp9CoefProbs.Index4x4(planeType, refType, band, ctx, 0);
                m0 = coefProbs[modelBase + 0];
                m1 = coefProbs[modelBase + 1];
                m2 = coefProbs[modelBase + 2];
            }

            // Non-zero token path.
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, m1);     // !ZERO

            int value = coefs[(int)scan[c]];
            int magnitude = value < 0 ? -value : value;

            // For tokens >= TWO we need the full 11-node prob vector. The
            // CPU encoder calls ModelToFullProbs which copies model[0..2]
            // verbatim then expands via Pareto8Full[m2 - 1, 0..7]. We
            // already have m0/m1/m2 (full[0..2]); the remaining 8 nodes
            // live in pareto8 row (m2 - 1).
            byte token; // Vp9CoefToken value as raw byte (0=Zero..11=Eob)
            if (magnitude == 1)
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, m2); // ONE
                token = (byte)Vp9CoefToken.One;
            }
            else
            {
                Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, m2); // !ONE

                token = MagnitudeToToken(magnitude, isHighBitDepth);

                // Walk the constrained tree using probs[3..10] from the
                // pareto8 row. Each token follows a fixed 2-4 bit path.
                long pareto8RowBase = (long)Pareto8FullOffset + (long)(m2 - 1) * 8;
                EmitTreeWalk(ref state, outBuf, consts, pareto8RowBase, token);

                // Category tokens emit residual magnitude bits MSB-first.
                if (token >= (byte)Vp9CoefToken.Category1 && token <= (byte)Vp9CoefToken.Category6)
                {
                    EmitCategoryMagnitude(ref state, outBuf, consts, token, magnitude, isHighBitDepth);
                }
            }

            // Sign bit at flat probability 128.
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, value < 0 ? 1 : 0, 128);

            // Update tokenCache for the just-emitted scan slot.
            int rasterC = (int)scan[c];
            tokenCache[rasterC] = consts[PtEnergyClassOffset + token];
            c++;
        }
        return eob;
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
    /// Map a coefficient magnitude to its <see cref="Vp9CoefToken"/>
    /// value. Bit-exact mirror of the CPU encoder's MagnitudeToToken.
    /// </summary>
    private static byte MagnitudeToToken(int magnitude, int isHighBitDepth)
    {
        // magnitude is guaranteed >= 2 by caller (the magnitude == 1
        // case is handled before this path).
        if (magnitude == 2) return (byte)Vp9CoefToken.Two;
        if (magnitude == 3) return (byte)Vp9CoefToken.Three;
        if (magnitude == 4) return (byte)Vp9CoefToken.Four;
        if (magnitude <= 6) return (byte)Vp9CoefToken.Category1;
        if (magnitude <= 10) return (byte)Vp9CoefToken.Category2;
        if (magnitude <= 18) return (byte)Vp9CoefToken.Category3;
        if (magnitude <= 34) return (byte)Vp9CoefToken.Category4;
        if (magnitude <= 66) return (byte)Vp9CoefToken.Category5;
        // Cat6 - upper bound enforced by caller via Vp9CoefProbs.CatMinVal.Cat6 ranges.
        return (byte)Vp9CoefToken.Category6;
    }

    /// <summary>
    /// Walk the VP9 constrained coefficient tree to <paramref name="token"/>,
    /// emitting one bit per internal node visited. Tree topology is
    /// from <see cref="Vp9CoefTrees.CoefConTree"/>; this routine
    /// unrolls the walk so the kernel never recurses.
    ///
    /// Probabilities come from pareto8Full[m2 - 1, 0..7]; the caller
    /// passes the byte offset of the row's first entry.
    /// </summary>
    private static void EmitTreeWalk(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        ArrayView<byte> consts,
        long pareto8RowBase,
        byte token)
    {
        // probs[i] = consts[pareto8RowBase + i] for i = 0..7.
        // Byte 0 (root LOW_VAL prob) decides Two/Three/Four vs Cat1..Cat6.
        // Byte 1 (TWO prob) decides Two vs (Three/Four).
        // Byte 2 (THREE prob) decides Three vs Four.
        // Byte 3 (HIGH_LOW prob) decides Cat1/2 vs Cat3/4/5/6.
        // Byte 4 (CAT_ONE prob) decides Cat1 vs Cat2.
        // Byte 5 (CAT_THREEFOUR prob) decides Cat3/4 vs Cat5/6.
        // Byte 6 (CAT_THREE prob) decides Cat3 vs Cat4.
        // Byte 7 (CAT_FIVE prob) decides Cat5 vs Cat6.
        if (token == (byte)Vp9CoefToken.Two)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 1]);
        }
        else if (token == (byte)Vp9CoefToken.Three)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 1]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 2]);
        }
        else if (token == (byte)Vp9CoefToken.Four)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 1]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 2]);
        }
        else if (token == (byte)Vp9CoefToken.Category1)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 4]);
        }
        else if (token == (byte)Vp9CoefToken.Category2)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 4]);
        }
        else if (token == (byte)Vp9CoefToken.Category3)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 5]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 6]);
        }
        else if (token == (byte)Vp9CoefToken.Category4)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 5]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 6]);
        }
        else if (token == (byte)Vp9CoefToken.Category5)
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 5]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, consts[pareto8RowBase + 7]);
        }
        else // Category6
        {
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 0]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 3]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 5]);
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, consts[pareto8RowBase + 7]);
        }
    }

    /// <summary>
    /// Emit the residual magnitude bits for a Cat1..Cat6 token,
    /// MSB-first. Mirrors the CPU encoder's WriteResidualMsbFirst.
    /// </summary>
    private static void EmitCategoryMagnitude(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        ArrayView<byte> consts,
        byte token,
        int magnitude,
        int isHighBitDepth)
    {
        // (probOffset, probLen, minVal) per category. Width == probLen
        // because Vp9 stores one prob per residual bit.
        int probOffset;
        int probLen;
        int minVal;
        if (token == (byte)Vp9CoefToken.Category1) { probOffset = Cat1ProbOffset; probLen = 1; minVal = Vp9CoefProbs.CatMinVal.Cat1; }
        else if (token == (byte)Vp9CoefToken.Category2) { probOffset = Cat2ProbOffset; probLen = 2; minVal = Vp9CoefProbs.CatMinVal.Cat2; }
        else if (token == (byte)Vp9CoefToken.Category3) { probOffset = Cat3ProbOffset; probLen = 3; minVal = Vp9CoefProbs.CatMinVal.Cat3; }
        else if (token == (byte)Vp9CoefToken.Category4) { probOffset = Cat4ProbOffset; probLen = 4; minVal = Vp9CoefProbs.CatMinVal.Cat4; }
        else if (token == (byte)Vp9CoefToken.Category5) { probOffset = Cat5ProbOffset; probLen = 5; minVal = Vp9CoefProbs.CatMinVal.Cat5; }
        else // Category6
        {
            minVal = Vp9CoefProbs.CatMinVal.Cat6;
            if (isHighBitDepth != 0) { probOffset = Cat6ProbHigh12Offset; probLen = 18; }
            else { probOffset = Cat6ProbOffset; probLen = 14; }
        }

        int extra = magnitude - minVal;
        for (int i = 0; i < probLen; i++)
        {
            int bit = (extra >> (probLen - 1 - i)) & 1;
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, bit, consts[probOffset + i]);
        }
    }
}
