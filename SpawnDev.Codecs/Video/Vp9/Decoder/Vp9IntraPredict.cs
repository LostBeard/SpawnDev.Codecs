// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 intra prediction (8-bit) for the decoder, kernel-callable. A port of libvpx v1.17.0 (BSD-3-Clause):
//   vp9/common/vp9_reconintra.c  vp9_predict_intra_block, build_intra_predictors (non-high-bitdepth), extend_modes,
//                                vp9_init_intra_predictors_internal (the pred / dc_pred dispatch)
//   vpx_dsp/intrapred.c          the C predictors that dispatch wires up: dc / dc_top / dc_left / dc_128 / v / h / tm
//                                (all sizes, generic), d207 / d63 / d45 / d117 / d135 / d153 (dedicated 4x4 functions,
//                                generic ones for 8x8..32x32). VP9 does not use d45e / d63e / he / ve.
// Bit-exact with libvpx's own C on every mode x transform size x plane, at frame edges and every neighbour
// availability (CodecsTestBase.Vp9DspIntraTests.cs, fixture from scripts/video/libvpx_ref/intra_ref.c).

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>VP9 intra prediction of one transform block, in place in a byte plane (libvpx vp9_predict_intra_block).</summary>
internal static class Vp9IntraPred
{
    /// <summary>libvpx PREDICTION_MODE values (intra).</summary>
    public const int DcPred = 0, VPred = 1, HPred = 2, D45Pred = 3, D135Pred = 4, D117Pred = 5, D153Pred = 6,
        D207Pred = 7, D63Pred = 8, TmPred = 9;

    // Scratch layout (ints holding pixels): left_col[32] at 0; above_row[-1] at 32, above_row[0..63] at 33..96
    // (libvpx's above_data[64 + 16] with above_row = above_data + 16: only [-1 .. 2 * bs - 1] is ever used).
    private const int LeftAt = 0, AboveAt = 33;

    /// <summary>Ints of scratch one <see cref="Predict"/> call uses (per thread): the left column and the above row.</summary>
    public const int ScratchInts = 97;

    // extend_modes[]: NEED_LEFT = 2, NEED_ABOVE = 4, NEED_ABOVERIGHT = 8
    private const int NeedLeft = 1 << 1, NeedAbove = 1 << 2, NeedAboveRight = 1 << 3;

    private static int ExtendModes(int mode)
    {
        switch (mode)
        {
            case DcPred: return NeedAbove | NeedLeft;
            case VPred: return NeedAbove;
            case HPred: return NeedLeft;
            case D45Pred: return NeedAboveRight;
            case D135Pred: return NeedLeft | NeedAbove;
            case D117Pred: return NeedLeft | NeedAbove;
            case D153Pred: return NeedLeft | NeedAbove;
            case D207Pred: return NeedLeft;
            case D63Pred: return NeedAboveRight;
            default: return NeedLeft | NeedAbove; // TM_PRED
        }
    }

    /// <summary>
    /// vp9_predict_intra_block + build_intra_predictors for one transform block (8-bit). libvpx's decoder predicts in
    /// place (ref == dst): the plane holds the already reconstructed neighbours and receives the prediction.
    /// </summary>
    /// <param name="bwlIn">bwl_in: pd->n4_wl, log2 of the prediction block's width in 4x4 units in this plane.</param>
    /// <param name="txSize">tx_size: 0..3 (TX_4X4..TX_32X32); the block is bs = 4 &lt;&lt; txSize pixels square.</param>
    /// <param name="mode">mode: PREDICTION_MODE 0..9 (DC, V, H, D45, D135, D117, D153, D207, D63, TM).</param>
    /// <param name="aoff">aoff: the transform block's column inside the prediction block, in 4x4 units (the decoder's col).</param>
    /// <param name="loff">loff: the transform block's row inside the prediction block, in 4x4 units (the decoder's row).</param>
    /// <param name="haveAboveMi">xd->above_mi != NULL (mi_row != 0).</param>
    /// <param name="haveLeftMi">xd->left_mi != NULL (mi_col &gt; the tile column's mi_col_start).</param>
    /// <param name="mbToLeftEdge">xd->mb_to_left_edge (1/8 luma pel, &lt;= 0).</param>
    /// <param name="mbToTopEdge">xd->mb_to_top_edge (1/8 luma pel, &lt;= 0).</param>
    /// <param name="mbToRightEdge">xd->mb_to_right_edge (1/8 luma pel; &lt; 0 when the prediction block crosses the right edge).</param>
    /// <param name="mbToBottomEdge">xd->mb_to_bottom_edge (1/8 luma pel; &lt; 0 when it crosses the bottom edge).</param>
    /// <param name="subsamplingX">pd->subsampling_x of this plane (0 for luma).</param>
    /// <param name="subsamplingY">pd->subsampling_y of this plane (0 for luma).</param>
    /// <param name="frameWidth">xd->cur_buf->y_width (plane 0) or uv_width: the 8-aligned decoded width, not the crop width.</param>
    /// <param name="frameHeight">xd->cur_buf->y_height (plane 0) or uv_height: the 8-aligned decoded height.</param>
    /// <param name="dst">The plane (ref and dst of libvpx).</param>
    /// <param name="dstAt">Index of the transform block's top-left pixel in <paramref name="dst"/>.</param>
    /// <param name="stride">The plane's stride (ref_stride == dst_stride).</param>
    /// <param name="scratch">Per-thread scratch of <see cref="ScratchInts"/> ints (libvpx's stack left_col / above_data).</param>
    /// <param name="scratchAt">Start of this thread's scratch.</param>
    /// <remarks>
    /// The block's pixel position x0 = (-mb_to_left_edge &gt;&gt; (3 + ss_x)) + 4 * aoff (y0 likewise) must lie inside the
    /// plane (x0 &lt; frameWidth, y0 &lt; frameHeight), as it always does for blocks libvpx's decoder predicts
    /// (predict_recon_intra stops at max_blocks_wide / max_blocks_high). For x0 &gt;= frameWidth libvpx leaves its above
    /// row uninitialised (and reads above_row[-1] before setting it when x0 == frameWidth); this port leaves the
    /// scratch as it was there too, so such input has no defined result in either.
    /// </remarks>
    public static void Predict(int bwlIn, int txSize, int mode, int aoff, int loff, bool haveAboveMi, bool haveLeftMi,
        int mbToLeftEdge, int mbToTopEdge, int mbToRightEdge, int mbToBottomEdge, int subsamplingX, int subsamplingY,
        int frameWidth, int frameHeight, ArrayView<byte> dst, long dstAt, int stride, ArrayView<int> scratch, long scratchAt)
    {
        // vp9_predict_intra_block
        int bw = 1 << bwlIn;
        int txw = 1 << txSize;
        bool upAvailable = loff != 0 || haveAboveMi;
        bool leftAvailable = aoff != 0 || haveLeftMi;
        bool rightAvailable = (aoff + txw) < bw;
        int x = aoff * 4;
        int y = loff * 4;

        // build_intra_predictors
        int bs = 4 << txSize;
        int x0 = (-mbToLeftEdge >> (3 + subsamplingX)) + x;
        int y0 = (-mbToTopEdge >> (3 + subsamplingY)) + y;
        long left = scratchAt + LeftAt;
        long above = scratchAt + AboveAt;
        long aboveRef = dstAt - stride;
        int ext = ExtendModes(mode);

        // NEED_LEFT
        if ((ext & NeedLeft) != 0)
        {
            if (leftAvailable)
            {
                if (mbToBottomEdge < 0 && y0 + bs > frameHeight)
                {
                    // slower path: the block needs border extension
                    int extendBottom = frameHeight - y0;
                    int i = 0;
                    for (; i < extendBottom; ++i) scratch[left + i] = dst[dstAt + (long)i * stride - 1];
                    for (; i < bs; ++i) scratch[left + i] = dst[dstAt + (long)(extendBottom - 1) * stride - 1];
                }
                else
                {
                    for (int i = 0; i < bs; ++i) scratch[left + i] = dst[dstAt + (long)i * stride - 1];
                }
            }
            else
            {
                for (int i = 0; i < bs; ++i) scratch[left + i] = 129;
            }
        }

        // NEED_ABOVE. libvpx's fast path for a 4x4 block with right and left available predicts straight from the
        // frame row (const_above_row = above_ref); the predictors of these modes read above[-1 .. 3] there, which the
        // copy below holds identically (above[-1] = above_ref[-1] since left is available).
        if ((ext & NeedAbove) != 0)
        {
            if (upAvailable)
            {
                if (mbToRightEdge < 0)
                {
                    if (x0 + bs <= frameWidth)
                    {
                        for (int i = 0; i < bs; ++i) scratch[above + i] = dst[aboveRef + i];
                    }
                    else if (x0 <= frameWidth)
                    {
                        int r = frameWidth - x0;
                        for (int i = 0; i < r; ++i) scratch[above + i] = dst[aboveRef + i];
                        int fill = scratch[above + r - 1];
                        for (int i = r; i < bs; ++i) scratch[above + i] = fill;
                    }
                }
                else
                {
                    for (int i = 0; i < bs; ++i) scratch[above + i] = dst[aboveRef + i];
                }
                scratch[above - 1] = leftAvailable ? dst[aboveRef - 1] : 129;
            }
            else
            {
                for (int i = 0; i < bs; ++i) scratch[above + i] = 127;
                scratch[above - 1] = 127;
            }
        }

        // NEED_ABOVERIGHT. The 4x4 right-and-left fast path (const_above_row = above_ref) reads above_ref[0 .. 7]: the
        // same pixels the copy below takes for bs == 4 with right available.
        if ((ext & NeedAboveRight) != 0)
        {
            if (upAvailable)
            {
                if (mbToRightEdge < 0)
                {
                    if (x0 + 2 * bs <= frameWidth)
                    {
                        if (rightAvailable && bs == 4)
                        {
                            for (int i = 0; i < 2 * bs; ++i) scratch[above + i] = dst[aboveRef + i];
                        }
                        else
                        {
                            for (int i = 0; i < bs; ++i) scratch[above + i] = dst[aboveRef + i];
                            int fill = scratch[above + bs - 1];
                            for (int i = bs; i < 2 * bs; ++i) scratch[above + i] = fill;
                        }
                    }
                    else if (x0 + bs <= frameWidth)
                    {
                        int r = frameWidth - x0;
                        if (rightAvailable && bs == 4)
                        {
                            for (int i = 0; i < r; ++i) scratch[above + i] = dst[aboveRef + i];
                            int fill = scratch[above + r - 1];
                            for (int i = r; i < 2 * bs; ++i) scratch[above + i] = fill;
                        }
                        else
                        {
                            for (int i = 0; i < bs; ++i) scratch[above + i] = dst[aboveRef + i];
                            int fill = scratch[above + bs - 1];
                            for (int i = bs; i < 2 * bs; ++i) scratch[above + i] = fill;
                        }
                    }
                    else if (x0 <= frameWidth)
                    {
                        int r = frameWidth - x0;
                        for (int i = 0; i < r; ++i) scratch[above + i] = dst[aboveRef + i];
                        int fill = scratch[above + r - 1];
                        for (int i = r; i < 2 * bs; ++i) scratch[above + i] = fill;
                    }
                }
                else
                {
                    for (int i = 0; i < bs; ++i) scratch[above + i] = dst[aboveRef + i];
                    if (bs == 4 && rightAvailable)
                    {
                        for (int i = bs; i < 2 * bs; ++i) scratch[above + i] = dst[aboveRef + i];
                    }
                    else
                    {
                        int fill = scratch[above + bs - 1];
                        for (int i = bs; i < 2 * bs; ++i) scratch[above + i] = fill;
                    }
                }
                scratch[above - 1] = leftAvailable ? dst[aboveRef - 1] : 129;
            }
            else
            {
                for (int i = 0; i < 2 * bs; ++i) scratch[above + i] = 127;
                scratch[above - 1] = 127;
            }
        }

        // predict: dc_pred[left_available][up_available][tx_size] or pred[mode][tx_size]
        switch (mode)
        {
            case DcPred:
                if (leftAvailable && upAvailable) Dc(dst, dstAt, stride, bs, scratch, above, left);
                else if (leftAvailable) DcEdge(dst, dstAt, stride, bs, scratch, left);  // dc_left
                else if (upAvailable) DcEdge(dst, dstAt, stride, bs, scratch, above);   // dc_top
                else Fill(dst, dstAt, stride, bs, 128);                                  // dc_128
                break;
            case VPred: V(dst, dstAt, stride, bs, scratch, above); break;
            case HPred: H(dst, dstAt, stride, bs, scratch, left); break;
            case TmPred: Tm(dst, dstAt, stride, bs, scratch, above, left); break;
            case D45Pred:
                if (bs == 4) D45x4(dst, dstAt, stride, scratch, above); else D45(dst, dstAt, stride, bs, scratch, above);
                break;
            case D63Pred:
                if (bs == 4) D63x4(dst, dstAt, stride, scratch, above); else D63(dst, dstAt, stride, bs, scratch, above);
                break;
            case D207Pred:
                if (bs == 4) D207x4(dst, dstAt, stride, scratch, left); else D207(dst, dstAt, stride, bs, scratch, left);
                break;
            case D117Pred:
                if (bs == 4) D117x4(dst, dstAt, stride, scratch, above, left); else D117(dst, dstAt, stride, bs, scratch, above, left);
                break;
            case D135Pred:
                if (bs == 4) D135x4(dst, dstAt, stride, scratch, above, left); else D135(dst, dstAt, stride, bs, scratch, above, left);
                break;
            default: // D153_PRED
                if (bs == 4) D153x4(dst, dstAt, stride, scratch, above, left); else D153(dst, dstAt, stride, bs, scratch, above, left);
                break;
        }
    }

    // AVG3(a, b, c) = (a + 2 * b + c + 2) >> 2, AVG2(a, b) = (a + b + 1) >> 1
    private static int Avg3(int a, int b, int c) => (a + 2 * b + c + 2) >> 2;
    private static int Avg2(int a, int b) => (a + b + 1) >> 1;

    private static void Put(ArrayView<byte> dst, long at, int stride, int c, int r, int v) => dst[at + (long)r * stride + c] = (byte)v;
    private static int Get(ArrayView<byte> dst, long at, int stride, int c, int r) => dst[at + (long)r * stride + c];

    private static void Fill(ArrayView<byte> dst, long at, int stride, int bs, int v)
    {
        for (int r = 0; r < bs; r++)
            for (int c = 0; c < bs; c++) Put(dst, at, stride, c, r, v);
    }

    // dc_predictor
    private static void Dc(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above, long left)
    {
        int sum = 0, count = 2 * bs;
        for (int i = 0; i < bs; i++) sum += s[above + i] + s[left + i];
        Fill(dst, at, stride, bs, (sum + (count >> 1)) / count);
    }

    // dc_top_predictor (edge = above) / dc_left_predictor (edge = left)
    private static void DcEdge(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long edge)
    {
        int sum = 0;
        for (int i = 0; i < bs; i++) sum += s[edge + i];
        Fill(dst, at, stride, bs, (sum + (bs >> 1)) / bs);
    }

    // v_predictor
    private static void V(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above)
    {
        for (int r = 0; r < bs; r++)
            for (int c = 0; c < bs; c++) Put(dst, at, stride, c, r, s[above + c]);
    }

    // h_predictor
    private static void H(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long left)
    {
        for (int r = 0; r < bs; r++)
        {
            int v = s[left + r];
            for (int c = 0; c < bs; c++) Put(dst, at, stride, c, r, v);
        }
    }

    // tm_predictor: clip_pixel(left[r] + above[c] - above[-1])
    private static void Tm(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above, long left)
    {
        int ytopLeft = s[above - 1];
        for (int r = 0; r < bs; r++)
            for (int c = 0; c < bs; c++)
            {
                int v = s[left + r] + s[above + c] - ytopLeft;
                Put(dst, at, stride, c, r, v < 0 ? 0 : v > 255 ? 255 : v);
            }
    }

    // d45_predictor (8x8 .. 32x32)
    private static void D45(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above)
    {
        int aboveRight = s[above + bs - 1];
        for (int x = 0; x < bs - 1; ++x) Put(dst, at, stride, x, 0, Avg3(s[above + x], s[above + x + 1], s[above + x + 2]));
        Put(dst, at, stride, bs - 1, 0, aboveRight);
        // row x: memcpy(dst, dst_row0 + x, size = bs - 1 - x); memset(dst + size, above_right, x + 1)
        for (int x = 1, size = bs - 2; x < bs; ++x, --size)
        {
            for (int c = 0; c < size; c++) Put(dst, at, stride, c, x, Get(dst, at, stride, x + c, 0));
            for (int c = size; c < bs; c++) Put(dst, at, stride, c, x, aboveRight);
        }
    }

    // vpx_d45_predictor_4x4_c
    private static void D45x4(ArrayView<byte> dst, long at, int stride, ArrayView<int> s, long above)
    {
        int A = s[above], B = s[above + 1], C = s[above + 2], D = s[above + 3];
        int E = s[above + 4], F = s[above + 5], G = s[above + 6], H = s[above + 7];
        Put(dst, at, stride, 0, 0, Avg3(A, B, C));
        int v = Avg3(B, C, D);
        Put(dst, at, stride, 1, 0, v); Put(dst, at, stride, 0, 1, v);
        v = Avg3(C, D, E);
        Put(dst, at, stride, 2, 0, v); Put(dst, at, stride, 1, 1, v); Put(dst, at, stride, 0, 2, v);
        v = Avg3(D, E, F);
        Put(dst, at, stride, 3, 0, v); Put(dst, at, stride, 2, 1, v); Put(dst, at, stride, 1, 2, v); Put(dst, at, stride, 0, 3, v);
        v = Avg3(E, F, G);
        Put(dst, at, stride, 3, 1, v); Put(dst, at, stride, 2, 2, v); Put(dst, at, stride, 1, 3, v);
        v = Avg3(F, G, H);
        Put(dst, at, stride, 3, 2, v); Put(dst, at, stride, 2, 3, v);
        Put(dst, at, stride, 3, 3, H);
    }

    // d63_predictor (8x8 .. 32x32)
    private static void D63(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above)
    {
        for (int c = 0; c < bs; ++c)
        {
            Put(dst, at, stride, c, 0, Avg2(s[above + c], s[above + c + 1]));
            Put(dst, at, stride, c, 1, Avg3(s[above + c], s[above + c + 1], s[above + c + 2]));
        }
        // above[bs - 1] as libvpx has it. For bs >= 8 the edge construction always makes above[bs .. 2 * bs - 1] copies
        // of above[bs - 1] (only 4x4 gets real above-right pixels), so above[bs] would give the same pixels: a
        // red-check of that swap cannot fail. Red-checked with above[bs - 2] instead.
        int last = s[above + bs - 1];
        for (int r = 2, size = bs - 2; r < bs; r += 2, --size)
        {
            for (int c = 0; c < size; c++)
            {
                Put(dst, at, stride, c, r, Get(dst, at, stride, (r >> 1) + c, 0));
                Put(dst, at, stride, c, r + 1, Get(dst, at, stride, (r >> 1) + c, 1));
            }
            for (int c = size; c < bs; c++)
            {
                Put(dst, at, stride, c, r, last);
                Put(dst, at, stride, c, r + 1, last);
            }
        }
    }

    // vpx_d63_predictor_4x4_c
    private static void D63x4(ArrayView<byte> dst, long at, int stride, ArrayView<int> s, long above)
    {
        int A = s[above], B = s[above + 1], C = s[above + 2], D = s[above + 3];
        int E = s[above + 4], F = s[above + 5], G = s[above + 6];
        Put(dst, at, stride, 0, 0, Avg2(A, B));
        int v = Avg2(B, C); Put(dst, at, stride, 1, 0, v); Put(dst, at, stride, 0, 2, v);
        v = Avg2(C, D); Put(dst, at, stride, 2, 0, v); Put(dst, at, stride, 1, 2, v);
        v = Avg2(D, E); Put(dst, at, stride, 3, 0, v); Put(dst, at, stride, 2, 2, v);
        Put(dst, at, stride, 3, 2, Avg2(E, F));
        Put(dst, at, stride, 0, 1, Avg3(A, B, C));
        v = Avg3(B, C, D); Put(dst, at, stride, 1, 1, v); Put(dst, at, stride, 0, 3, v);
        v = Avg3(C, D, E); Put(dst, at, stride, 2, 1, v); Put(dst, at, stride, 1, 3, v);
        v = Avg3(D, E, F); Put(dst, at, stride, 3, 1, v); Put(dst, at, stride, 2, 3, v);
        Put(dst, at, stride, 3, 3, Avg3(E, F, G));
    }

    // d207_predictor (8x8 .. 32x32)
    private static void D207(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long left)
    {
        // first column
        for (int r = 0; r < bs - 1; ++r) Put(dst, at, stride, 0, r, Avg2(s[left + r], s[left + r + 1]));
        int lastLeft = s[left + bs - 1];
        Put(dst, at, stride, 0, bs - 1, lastLeft);
        // second column
        for (int r = 0; r < bs - 2; ++r) Put(dst, at, stride, 1, r, Avg3(s[left + r], s[left + r + 1], s[left + r + 2]));
        Put(dst, at, stride, 1, bs - 2, Avg3(s[left + bs - 2], lastLeft, lastLeft));
        Put(dst, at, stride, 1, bs - 1, lastLeft);
        // rest of last row
        for (int c = 0; c < bs - 2; ++c) Put(dst, at, stride, 2 + c, bs - 1, lastLeft);
        // dst(2 + c, r) = dst(c, r + 1), bottom up
        for (int r = bs - 2; r >= 0; --r)
            for (int c = 0; c < bs - 2; ++c) Put(dst, at, stride, 2 + c, r, Get(dst, at, stride, c, r + 1));
    }

    // vpx_d207_predictor_4x4_c
    private static void D207x4(ArrayView<byte> dst, long at, int stride, ArrayView<int> s, long left)
    {
        int I = s[left], J = s[left + 1], K = s[left + 2], L = s[left + 3];
        Put(dst, at, stride, 0, 0, Avg2(I, J));
        int v = Avg2(J, K); Put(dst, at, stride, 2, 0, v); Put(dst, at, stride, 0, 1, v);
        v = Avg2(K, L); Put(dst, at, stride, 2, 1, v); Put(dst, at, stride, 0, 2, v);
        Put(dst, at, stride, 1, 0, Avg3(I, J, K));
        v = Avg3(J, K, L); Put(dst, at, stride, 3, 0, v); Put(dst, at, stride, 1, 1, v);
        v = Avg3(K, L, L); Put(dst, at, stride, 3, 1, v); Put(dst, at, stride, 1, 2, v);
        Put(dst, at, stride, 3, 2, L); Put(dst, at, stride, 2, 2, L); Put(dst, at, stride, 0, 3, L);
        Put(dst, at, stride, 1, 3, L); Put(dst, at, stride, 2, 3, L); Put(dst, at, stride, 3, 3, L);
    }

    // d117_predictor (8x8 .. 32x32)
    private static void D117(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above, long left)
    {
        // first row
        for (int c = 0; c < bs; c++) Put(dst, at, stride, c, 0, Avg2(s[above + c - 1], s[above + c]));
        // second row
        Put(dst, at, stride, 0, 1, Avg3(s[left], s[above - 1], s[above]));
        for (int c = 1; c < bs; c++) Put(dst, at, stride, c, 1, Avg3(s[above + c - 2], s[above + c - 1], s[above + c]));
        // the rest of first col
        Put(dst, at, stride, 0, 2, Avg3(s[above - 1], s[left], s[left + 1]));
        for (int r = 3; r < bs; ++r) Put(dst, at, stride, 0, r, Avg3(s[left + r - 3], s[left + r - 2], s[left + r - 1]));
        // the rest of the block: dst(c, r) = dst(c - 1, r - 2)
        for (int r = 2; r < bs; ++r)
            for (int c = 1; c < bs; c++) Put(dst, at, stride, c, r, Get(dst, at, stride, c - 1, r - 2));
    }

    // vpx_d117_predictor_4x4_c
    private static void D117x4(ArrayView<byte> dst, long at, int stride, ArrayView<int> s, long above, long left)
    {
        int I = s[left], J = s[left + 1], K = s[left + 2];
        int X = s[above - 1], A = s[above], B = s[above + 1], C = s[above + 2], D = s[above + 3];
        int v = Avg2(X, A); Put(dst, at, stride, 0, 0, v); Put(dst, at, stride, 1, 2, v);
        v = Avg2(A, B); Put(dst, at, stride, 1, 0, v); Put(dst, at, stride, 2, 2, v);
        v = Avg2(B, C); Put(dst, at, stride, 2, 0, v); Put(dst, at, stride, 3, 2, v);
        Put(dst, at, stride, 3, 0, Avg2(C, D));
        Put(dst, at, stride, 0, 3, Avg3(K, J, I));
        Put(dst, at, stride, 0, 2, Avg3(J, I, X));
        v = Avg3(I, X, A); Put(dst, at, stride, 0, 1, v); Put(dst, at, stride, 1, 3, v);
        v = Avg3(X, A, B); Put(dst, at, stride, 1, 1, v); Put(dst, at, stride, 2, 3, v);
        v = Avg3(A, B, C); Put(dst, at, stride, 2, 1, v); Put(dst, at, stride, 3, 3, v);
        Put(dst, at, stride, 3, 1, Avg3(B, C, D));
    }

    // d135_predictor border[i] (border runs from the bottom-left up the left edge and along the top):
    // dst(c, r) = border[bs - 1 - r + c]
    private static int D135Border(int i, int bs, ArrayView<int> s, long above, long left)
    {
        if (i < bs - 2) return Avg3(s[left + bs - 3 - i], s[left + bs - 2 - i], s[left + bs - 1 - i]);
        if (i == bs - 2) return Avg3(s[above - 1], s[left], s[left + 1]);
        if (i == bs - 1) return Avg3(s[left], s[above - 1], s[above]);
        if (i == bs) return Avg3(s[above - 1], s[above], s[above + 1]);
        int j = i - bs - 1;
        return Avg3(s[above + j], s[above + j + 1], s[above + j + 2]);
    }

    // d135_predictor (8x8 .. 32x32)
    private static void D135(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above, long left)
    {
        for (int r = 0; r < bs; ++r)
            for (int c = 0; c < bs; ++c) Put(dst, at, stride, c, r, D135Border(bs - 1 - r + c, bs, s, above, left));
    }

    // vpx_d135_predictor_4x4_c
    private static void D135x4(ArrayView<byte> dst, long at, int stride, ArrayView<int> s, long above, long left)
    {
        int I = s[left], J = s[left + 1], K = s[left + 2], L = s[left + 3];
        int X = s[above - 1], A = s[above], B = s[above + 1], C = s[above + 2], D = s[above + 3];
        Put(dst, at, stride, 0, 3, Avg3(J, K, L));
        int v = Avg3(I, J, K); Put(dst, at, stride, 1, 3, v); Put(dst, at, stride, 0, 2, v);
        v = Avg3(X, I, J); Put(dst, at, stride, 2, 3, v); Put(dst, at, stride, 1, 2, v); Put(dst, at, stride, 0, 1, v);
        v = Avg3(A, X, I);
        Put(dst, at, stride, 3, 3, v); Put(dst, at, stride, 2, 2, v); Put(dst, at, stride, 1, 1, v); Put(dst, at, stride, 0, 0, v);
        v = Avg3(B, A, X); Put(dst, at, stride, 3, 2, v); Put(dst, at, stride, 2, 1, v); Put(dst, at, stride, 1, 0, v);
        v = Avg3(C, B, A); Put(dst, at, stride, 3, 1, v); Put(dst, at, stride, 2, 0, v);
        Put(dst, at, stride, 3, 0, Avg3(D, C, B));
    }

    // d153_predictor (8x8 .. 32x32)
    private static void D153(ArrayView<byte> dst, long at, int stride, int bs, ArrayView<int> s, long above, long left)
    {
        // column 0
        Put(dst, at, stride, 0, 0, Avg2(s[above - 1], s[left]));
        for (int r = 1; r < bs; r++) Put(dst, at, stride, 0, r, Avg2(s[left + r - 1], s[left + r]));
        // column 1
        Put(dst, at, stride, 1, 0, Avg3(s[left], s[above - 1], s[above]));
        Put(dst, at, stride, 1, 1, Avg3(s[above - 1], s[left], s[left + 1]));
        for (int r = 2; r < bs; r++) Put(dst, at, stride, 1, r, Avg3(s[left + r - 2], s[left + r - 1], s[left + r]));
        // row 0 from column 2
        for (int c = 0; c < bs - 2; c++) Put(dst, at, stride, 2 + c, 0, Avg3(s[above + c - 1], s[above + c], s[above + c + 1]));
        // dst(2 + c, r) = dst(c, r - 1)
        for (int r = 1; r < bs; ++r)
            for (int c = 0; c < bs - 2; c++) Put(dst, at, stride, 2 + c, r, Get(dst, at, stride, c, r - 1));
    }

    // vpx_d153_predictor_4x4_c
    private static void D153x4(ArrayView<byte> dst, long at, int stride, ArrayView<int> s, long above, long left)
    {
        int I = s[left], J = s[left + 1], K = s[left + 2], L = s[left + 3];
        int X = s[above - 1], A = s[above], B = s[above + 1], C = s[above + 2];
        int v = Avg2(I, X); Put(dst, at, stride, 0, 0, v); Put(dst, at, stride, 2, 1, v);
        v = Avg2(J, I); Put(dst, at, stride, 0, 1, v); Put(dst, at, stride, 2, 2, v);
        v = Avg2(K, J); Put(dst, at, stride, 0, 2, v); Put(dst, at, stride, 2, 3, v);
        Put(dst, at, stride, 0, 3, Avg2(L, K));
        Put(dst, at, stride, 3, 0, Avg3(A, B, C));
        Put(dst, at, stride, 2, 0, Avg3(X, A, B));
        v = Avg3(I, X, A); Put(dst, at, stride, 1, 0, v); Put(dst, at, stride, 3, 1, v);
        v = Avg3(J, I, X); Put(dst, at, stride, 1, 1, v); Put(dst, at, stride, 3, 2, v);
        v = Avg3(K, J, I); Put(dst, at, stride, 1, 2, v); Put(dst, at, stride, 3, 3, v);
        Put(dst, at, stride, 1, 3, Avg3(L, K, J));
    }
}
