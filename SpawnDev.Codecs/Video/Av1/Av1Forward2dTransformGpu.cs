// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 2D forward transform helpers, GPU-callable form. Bit-exact
// mirror of Av1Forward2dTransform.Apply (libaom av1_fwd_txfm2d) for
// the v1 keyframe encoder's two configurations:
//   - Tx8x8 + DCT_DCT (chroma)
//   - Tx16x16 + DCT_DCT (luma)
//
// Pipeline (per libaom fwd_txfm2d):
//   1. Column pass: load column, pre-scale by 2 bits left-shift,
//      apply 1D Forward DCT, round-shift by between-pass amount,
//      store back into scratch buffer.
//   2. Row pass: apply 1D Forward DCT on each row, round-shift by
//      final amount (0 for both v1 sizes), write to output (raster).
//
// V1 shifts (libaom av1_fwd_txfm_shift_ls):
//   Tx8x8:   s0=2, s1=-1, s2=0; cosBitCol=13, cosBitRow=13.
//   Tx16x16: s0=2, s1=-2, s2=0; cosBitCol=13, cosBitRow=12.
//
// Caller pre-allocates a scratch ArrayView&lt;int&gt; sized for the column
// intermediate buffer (64 ints for Tx8x8, 256 ints for Tx16x16).

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 2D forward DCT helpers. Bit-exact mirror of
/// <see cref="Av1Forward2dTransform"/>.Apply for the v1 keyframe
/// encoder's Tx8x8 / Tx16x16 DCT_DCT configurations.
/// </summary>
public static class Av1Forward2dTransformGpu
{
    /// <summary>
    /// Apply the 8x8 DCT_DCT 2D forward transform. Reads 64 short
    /// residuals from <paramref name="input"/> starting at
    /// <paramref name="inBase"/> (raster, row-major); writes 64 int
    /// coefs to <paramref name="output"/> starting at
    /// <paramref name="outBase"/> (raster, row-major). <paramref name="scratch"/>
    /// must hold at least 64 ints starting at <paramref name="scratchBase"/>.
    /// </summary>
    public static void Forward8x8DctDct(
        ArrayView<short> input, long inBase,
        ArrayView<int> output, long outBase,
        ArrayView<int> scratch, long scratchBase)
    {
        const int W = 8;
        const int H = 8;
        const int CosBit = 13;
        // s0 = 2 (left-shift by 2 in column pass pre-scale).
        // s1 = -1 (round-shift right by 1 between passes).
        // s2 = 0 (no final shift).

        // === Column pass ===
        for (int c = 0; c < W; c++)
        {
            // Load column c into scratch[c*H .. (c+1)*H), pre-scaled
            // (left-shifted by 2 bits).
            long colBase = scratchBase + c * (long)H;
            for (int r = 0; r < H; r++)
            {
                int v = input[inBase + r * W + c];
                scratch[colBase + r] = v << 2;
            }
            // 1D forward DCT8 in place: read scratch[colBase..+H], write
            // back to scratch[colBase..+H]. Forward8 takes input/output
            // as separate views, so we use a temporary stash via the
            // output view as scratch. To avoid that complication, use a
            // local 8-int copy.
            int t0 = scratch[colBase + 0];
            int t1 = scratch[colBase + 1];
            int t2 = scratch[colBase + 2];
            int t3 = scratch[colBase + 3];
            int t4 = scratch[colBase + 4];
            int t5 = scratch[colBase + 5];
            int t6 = scratch[colBase + 6];
            int t7 = scratch[colBase + 7];
            ForwardDct8Inline(t0, t1, t2, t3, t4, t5, t6, t7, CosBit,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);
            // Round-shift right by 1 (between-pass shift, -s1 = 1).
            scratch[colBase + 0] = (o0 + 1) >> 1;
            scratch[colBase + 1] = (o1 + 1) >> 1;
            scratch[colBase + 2] = (o2 + 1) >> 1;
            scratch[colBase + 3] = (o3 + 1) >> 1;
            scratch[colBase + 4] = (o4 + 1) >> 1;
            scratch[colBase + 5] = (o5 + 1) >> 1;
            scratch[colBase + 6] = (o6 + 1) >> 1;
            scratch[colBase + 7] = (o7 + 1) >> 1;
        }

        // === Row pass ===
        // The buf layout from the column pass is buf[r * W + destCol]
        // (per the CPU code with destCol = c when no flip). But the
        // column pass above stored into scratch[c * H + r] — i.e. the
        // column-major stride. The row pass needs to read scratch[r * W + c]
        // back. Reformat via a transpose during the row pass: read
        // scratch[c * H + r] for column c at row r.
        for (int r = 0; r < H; r++)
        {
            int t0 = scratch[scratchBase + 0 * H + r];
            int t1 = scratch[scratchBase + 1 * H + r];
            int t2 = scratch[scratchBase + 2 * H + r];
            int t3 = scratch[scratchBase + 3 * H + r];
            int t4 = scratch[scratchBase + 4 * H + r];
            int t5 = scratch[scratchBase + 5 * H + r];
            int t6 = scratch[scratchBase + 6 * H + r];
            int t7 = scratch[scratchBase + 7 * H + r];
            ForwardDct8Inline(t0, t1, t2, t3, t4, t5, t6, t7, CosBit,
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);
            // Final shift = 0; raw store.
            output[outBase + r * W + 0] = o0;
            output[outBase + r * W + 1] = o1;
            output[outBase + r * W + 2] = o2;
            output[outBase + r * W + 3] = o3;
            output[outBase + r * W + 4] = o4;
            output[outBase + r * W + 5] = o5;
            output[outBase + r * W + 6] = o6;
            output[outBase + r * W + 7] = o7;
        }
    }

    /// <summary>
    /// Apply the 16x16 DCT_DCT 2D forward transform. Reads 256 short
    /// residuals; writes 256 int coefs. Scratch must hold at least
    /// 256 ints.
    /// </summary>
    public static void Forward16x16DctDct(
        ArrayView<short> input, long inBase,
        ArrayView<int> output, long outBase,
        ArrayView<int> scratch, long scratchBase)
    {
        const int W = 16;
        const int H = 16;
        const int CosBitCol = 13;
        const int CosBitRow = 12;
        // s0 = 2 (pre-scale left-shift), s1 = -2 (round-shift right by 2),
        // s2 = 0 (no final shift).

        // Column pass: load + pre-scale + DCT + round-shift, store column-major.
        for (int c = 0; c < W; c++)
        {
            long colBase = scratchBase + c * (long)H;
            for (int r = 0; r < H; r++)
            {
                int v = input[inBase + r * W + c];
                scratch[colBase + r] = v << 2;
            }
            // We need a Forward16 inline. Use the helper that exists
            // (Av1ForwardDct16Gpu.Forward16) but that requires
            // ArrayView<int> input + output. Use scratch as both.
            // Need a temp output region. We'll write to the column
            // and shift in place after.
            Av1ForwardDct16Gpu.Forward16(scratch, colBase, scratch, colBase, CosBitCol);
            // Round-shift right by 2.
            for (int r = 0; r < H; r++)
            {
                int v = scratch[colBase + r];
                scratch[colBase + r] = (v + 2) >> 2;
            }
        }

        // Row pass: read column-major (transposed) and apply Forward16
        // per row. Output is raster. We need a small temp scratch per
        // row for the input to Forward16.
        // Trick: reuse output[outBase + r * W + 0..15] as the temp row
        // input then overwrite with row pass output. Forward16 supports
        // in-place: same input/output views OK as long as the body
        // reads the input first into locals. Looking at Forward16: it
        // loads into 16 in0..in15 locals then computes - safe in place.
        for (int r = 0; r < H; r++)
        {
            // Gather row r from column-major scratch into output row.
            for (int c = 0; c < W; c++)
            {
                output[outBase + r * W + c] = scratch[scratchBase + c * H + r];
            }
            // Apply Forward16 in place over output row.
            Av1ForwardDct16Gpu.Forward16(output, outBase + r * W, output, outBase + r * W, CosBitRow);
            // s2 = 0, no final shift.
        }
    }

    // ----------------------------------------------------------------
    // Inlined 8-point forward DCT (avoids the
    // Av1ForwardDct8Gpu.Forward8 dispatch when we already have the 8
    // values as locals - one less ArrayView round-trip per column).
    // ----------------------------------------------------------------
    private static void ForwardDct8Inline(
        int in0, int in1, int in2, int in3, int in4, int in5, int in6, int in7,
        int cosBit,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        ResolveCospi8(cosBit, out int c8, out int c16, out int c24, out int c32,
            out int c40, out int c48, out int c56);

        // Stage 1
        int s10 =  in0 + in7;
        int s11 =  in1 + in6;
        int s12 =  in2 + in5;
        int s13 =  in3 + in4;
        int s14 = -in4 + in3;
        int s15 = -in5 + in2;
        int s16 = -in6 + in1;
        int s17 = -in7 + in0;

        // Stage 2
        int s20 = s10 + s13;
        int s21 = s11 + s12;
        int s22 = -s12 + s11;
        int s23 = -s13 + s10;
        int s24 = s14;
        int s25 = HalfBtf(-c32, s15,  c32, s16, cosBit);
        int s26 = HalfBtf( c32, s16,  c32, s15, cosBit);
        int s27 = s17;

        // Stage 3
        int s30 = HalfBtf( c32, s20,  c32, s21, cosBit);
        int s31 = HalfBtf(-c32, s21,  c32, s20, cosBit);
        int s32 = HalfBtf( c48, s22,  c16, s23, cosBit);
        int s33 = HalfBtf( c48, s23, -c16, s22, cosBit);
        int s34 = s24 + s25;
        int s35 = -s25 + s24;
        int s36 = -s26 + s27;
        int s37 = s27 + s26;

        // Stage 4
        int s40 = s30;
        int s41 = s31;
        int s42 = s32;
        int s43 = s33;
        int s44 = HalfBtf( c56, s34,  c8,  s37, cosBit);
        int s45 = HalfBtf( c24, s35,  c40, s36, cosBit);
        int s46 = HalfBtf( c24, s36, -c40, s35, cosBit);
        int s47 = HalfBtf( c56, s37, -c8,  s34, cosBit);

        // Stage 5 (interleave)
        o0 = s40;
        o1 = s44;
        o2 = s42;
        o3 = s46;
        o4 = s41;
        o5 = s45;
        o6 = s43;
        o7 = s47;
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
