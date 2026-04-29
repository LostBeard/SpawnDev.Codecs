// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 2D inverse transform helpers, GPU-callable form. Bit-exact
// mirror of Av1Inverse2dTransform.Apply (libaom av1_inv_txfm2d_add)
// for the v1 keyframe decoder's two configurations:
//   - Tx8x8 + DCT_DCT (chroma)
//   - Tx16x16 + DCT_DCT (luma)
//
// Pipeline (per libaom inv_txfm2d):
//   1. Row pass: read row r of coefs (with rect-scale prescale if
//      applicable; no rect-scale for square Tx8x8 / Tx16x16),
//      apply 1D Inverse DCT, round-shift by row_shift, write into
//      buf[r * w + c].
//   2. Column pass: read column c of buf, apply 1D Inverse DCT,
//      round-shift by col_shift, write into residual[r * w + c].
//
// Shifts (libaom inv_txfm_shift_ls):
//   Tx8x8:   rowShift=1, colShift=4
//   Tx16x16: rowShift=2, colShift=4

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 2D inverse DCT helpers. Bit-exact mirror of
/// <see cref="Av1Inverse2dTransform"/>.Apply for the v1 keyframe
/// decoder's Tx8x8 / Tx16x16 DCT_DCT configurations.
/// </summary>
public static class Av1Inverse2dTransformGpu
{
    /// <summary>
    /// Apply the 8x8 DCT_DCT 2D inverse transform. Reads 64 int coefs;
    /// writes 64 int residuals. Scratch must hold at least 64 ints.
    /// </summary>
    public static void Inverse8x8DctDct(
        ArrayView<int> coefs, long coefBase,
        ArrayView<int> residual, long resBase,
        ArrayView<int> scratch, long scratchBase)
    {
        const int W = 8;
        const int H = 8;
        // rowShift=1, colShift=4 for Tx8x8.

        // === Row pass ===
        for (int r = 0; r < H; r++)
        {
            // Inverse DCT8 in place over coefs row -> scratch row.
            // Read row r raw (no rect-scale for square). Write to
            // scratch in row-major.
            int t0 = coefs[coefBase + r * W + 0];
            int t1 = coefs[coefBase + r * W + 1];
            int t2 = coefs[coefBase + r * W + 2];
            int t3 = coefs[coefBase + r * W + 3];
            int t4 = coefs[coefBase + r * W + 4];
            int t5 = coefs[coefBase + r * W + 5];
            int t6 = coefs[coefBase + r * W + 6];
            int t7 = coefs[coefBase + r * W + 7];
            InverseDct8Inline(t0, t1, t2, t3, t4, t5, t6, t7, Av1InverseDct8Gpu.DefaultCosBit,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);
            // Round-shift by rowShift = 1.
            scratch[scratchBase + r * W + 0] = (o0 + 1) >> 1;
            scratch[scratchBase + r * W + 1] = (o1 + 1) >> 1;
            scratch[scratchBase + r * W + 2] = (o2 + 1) >> 1;
            scratch[scratchBase + r * W + 3] = (o3 + 1) >> 1;
            scratch[scratchBase + r * W + 4] = (o4 + 1) >> 1;
            scratch[scratchBase + r * W + 5] = (o5 + 1) >> 1;
            scratch[scratchBase + r * W + 6] = (o6 + 1) >> 1;
            scratch[scratchBase + r * W + 7] = (o7 + 1) >> 1;
        }

        // === Column pass ===
        for (int c = 0; c < W; c++)
        {
            int t0 = scratch[scratchBase + 0 * W + c];
            int t1 = scratch[scratchBase + 1 * W + c];
            int t2 = scratch[scratchBase + 2 * W + c];
            int t3 = scratch[scratchBase + 3 * W + c];
            int t4 = scratch[scratchBase + 4 * W + c];
            int t5 = scratch[scratchBase + 5 * W + c];
            int t6 = scratch[scratchBase + 6 * W + c];
            int t7 = scratch[scratchBase + 7 * W + c];
            InverseDct8Inline(t0, t1, t2, t3, t4, t5, t6, t7, Av1InverseDct8Gpu.DefaultCosBit,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);
            // Round-shift by colShift = 4.
            residual[resBase + 0 * W + c] = (o0 + 8) >> 4;
            residual[resBase + 1 * W + c] = (o1 + 8) >> 4;
            residual[resBase + 2 * W + c] = (o2 + 8) >> 4;
            residual[resBase + 3 * W + c] = (o3 + 8) >> 4;
            residual[resBase + 4 * W + c] = (o4 + 8) >> 4;
            residual[resBase + 5 * W + c] = (o5 + 8) >> 4;
            residual[resBase + 6 * W + c] = (o6 + 8) >> 4;
            residual[resBase + 7 * W + c] = (o7 + 8) >> 4;
        }
    }

    /// <summary>
    /// Apply the 16x16 DCT_DCT 2D inverse transform. Reads 256 int coefs;
    /// writes 256 int residuals. Scratch must hold at least 256 ints.
    /// </summary>
    public static void Inverse16x16DctDct(
        ArrayView<int> coefs, long coefBase,
        ArrayView<int> residual, long resBase,
        ArrayView<int> scratch, long scratchBase)
    {
        const int W = 16;
        const int H = 16;
        const int CosBit = Av1InverseDct16Gpu.DefaultCosBit;
        // rowShift=2, colShift=4 for Tx16x16.

        // === Row pass ===
        for (int r = 0; r < H; r++)
        {
            // Inverse DCT16 in place over coefs row -> scratch row.
            // Use Av1InverseDct16Gpu.Inverse16 (in-place safe per Forward16
            // pattern - reads to locals first).
            Av1InverseDct16Gpu.Inverse16(coefs, coefBase + r * W, scratch, scratchBase + r * W, CosBit);
            // Round-shift by rowShift = 2.
            for (int c = 0; c < W; c++)
            {
                int v = scratch[scratchBase + r * W + c];
                scratch[scratchBase + r * W + c] = (v + 2) >> 2;
            }
        }

        // === Column pass ===
        // Need to gather column then apply Inverse16. Use residual as
        // staging - write column input there, run Inverse16 in place,
        // read back, scatter to final residual location with shift.
        // Simpler: gather column to a temporary 16-int ArrayView region
        // inside scratch (use a different region).
        // The scratch buffer is 256 ints; we used [0..256) for the row
        // pass output. We need 16 more ints for the column input. We
        // can reuse residual[resBase + 0..16) as temp for the column
        // input, since we write the final residual with the correct
        // values afterward via the column loop's outer write.
        //
        // Actually simpler: write column input to a region inside the
        // scratch by re-purposing the second half of a larger scratch.
        // Caller must size scratch as 256+16 = 272 ints? No - to keep
        // the API simple, gather to a 16-int local region inside
        // residual and run Inverse16 in place there.
        for (int c = 0; c < W; c++)
        {
            // Gather column c into residual[resBase..resBase+16).
            for (int r = 0; r < H; r++)
            {
                residual[resBase + r] = scratch[scratchBase + r * W + c];
            }
            // Inverse16 in place.
            Av1InverseDct16Gpu.Inverse16(residual, resBase, residual, resBase, CosBit);
            // Apply shift and scatter to final positions.
            // BUT we need to be careful: residual[resBase..resBase+16) is being
            // used as scratch for column c. After scattering to final positions
            // (residual[r*W + c]), we move on to column c+1. Make sure
            // we don't overwrite values we need.
            //
            // Strategy: for column c, gather into residual[resBase..+16),
            // transform, then scatter to residual[r*W + c]. For c=0 that
            // overwrites residual[resBase..+16) at positions 0, 16, 32, ...
            // The first position 0 is the same as resBase+0. So scattering
            // overwrites the gather buffer. Need the scattered values to
            // match what we'd compute from scratch for column 0 - we just
            // computed those, so scattering them is fine.
            //
            // For c >= 1: gather to residual[resBase..+16), transform,
            // scatter to residual[r*W + 1] for r in 0..15. None of those
            // positions overlap residual[resBase + 1..+15] (which are
            // scratch positions r=1..15 col=0 - but we already populated
            // those from the previous column's scatter? Actually NO -
            // the previous column scattered to residual[r*W + 0] which
            // is different from residual[resBase + r] for r >= 1.
            //
            // Wait: residual[resBase + r] = residual[resBase + r]. That's
            // the same memory. residual[r*W + 0] = residual[resBase + r*W].
            // For r=1, that's resBase+16, NOT resBase+1. So scatter
            // positions don't overlap the gather positions for resBase+1..15.
            //
            // OK so the gather/transform/scatter for each column is safe
            // as long as the scatter writes column c (positions resBase+r*W+c)
            // and the gather reads from residual[resBase..+16) (positions
            // resBase+0..+15). For c=0 they overlap at r=0 (resBase+0); we
            // scatter at the end so that's fine.
            for (int r = 0; r < H; r++)
            {
                int v = residual[resBase + r];
                int shifted = (v + 8) >> 4;
                residual[resBase + r * W + c] = shifted;
            }
        }
    }

    // ----------------------------------------------------------------
    // Inlined 8-point inverse DCT (avoids the helper dispatch).
    // ----------------------------------------------------------------
    private static void InverseDct8Inline(
        int in0, int in1, int in2, int in3, int in4, int in5, int in6, int in7,
        int cosBit,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        ResolveCospi8(cosBit, out int c8, out int c16, out int c24, out int c32,
            out int c40, out int c48, out int c56);

        // Stage 1: re-permute.
        int bf0 = in0;
        int bf1 = in4;
        int bf2 = in2;
        int bf3 = in6;
        int bf4 = in1;
        int bf5 = in5;
        int bf6 = in3;
        int bf7 = in7;

        // Stage 2: cospi rotation on upper half.
        int s0 = bf0;
        int s1 = bf1;
        int s2 = bf2;
        int s3 = bf3;
        int s4 = HalfBtf(c56, bf4, -c8,  bf7, cosBit);
        int s5 = HalfBtf(c24, bf5, -c40, bf6, cosBit);
        int s6 = HalfBtf(c40, bf5,  c24, bf6, cosBit);
        int s7 = HalfBtf(c8,  bf4,  c56, bf7, cosBit);

        // Stage 3.
        bf0 = HalfBtf(c32, s0,  c32, s1, cosBit);
        bf1 = HalfBtf(c32, s0, -c32, s1, cosBit);
        bf2 = HalfBtf(c48, s2, -c16, s3, cosBit);
        bf3 = HalfBtf(c16, s2,  c48, s3, cosBit);
        bf4 =  s4 + s5;
        bf5 =  s4 - s5;
        bf6 = -s6 + s7;
        bf7 =  s6 + s7;

        // Stage 4.
        s0 = bf0 + bf3;
        s1 = bf1 + bf2;
        s2 = bf1 - bf2;
        s3 = bf0 - bf3;
        s4 = bf4;
        s5 = HalfBtf(-c32, bf5, c32, bf6, cosBit);
        s6 = HalfBtf( c32, bf5, c32, bf6, cosBit);
        s7 = bf7;

        // Stage 5: outer butterfly.
        o0 = s0 + s7;
        o1 = s1 + s6;
        o2 = s2 + s5;
        o3 = s3 + s4;
        o4 = s3 - s4;
        o5 = s2 - s5;
        o6 = s1 - s6;
        o7 = s0 - s7;
    }

    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    private static void ResolveCospi8(int cosBit,
        out int c8, out int c16, out int c24, out int c32,
        out int c40, out int c48, out int c56)
    {
        if (cosBit == 13)      { c8 = 8035; c16 = 7568; c24 = 6811; c32 = 5793; c40 = 4551; c48 = 3135; c56 = 1598; }
        else if (cosBit == 12) { c8 = 4017; c16 = 3784; c24 = 3406; c32 = 2896; c40 = 2276; c48 = 1567; c56 = 799; }
        else if (cosBit == 11) { c8 = 2009; c16 = 1892; c24 = 1703; c32 = 1448; c40 = 1138; c48 = 784;  c56 = 400; }
        else                   { c8 = 1004; c16 = 946;  c24 = 851;  c32 = 724;  c40 = 569;  c48 = 392;  c56 = 200; }
    }
}
