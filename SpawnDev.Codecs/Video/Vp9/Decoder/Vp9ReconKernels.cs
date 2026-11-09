// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 reconstruction on the accelerator: prediction + inverse transform of every block, from the parse kernel's
// output. A port of libvpx v1.17.0 (BSD-3-Clause) vp9/decoder/vp9_decodeframe.c recon_tile_row / process_partition
// (RECON) / recon_block / predict_and_reconstruct_intra_block_row_mt / reconstruct_inter_block_row_mt /
// predict_recon_intra / predict_recon_inter. One thread per tile column: tiles are independent for reconstruction
// (intra edges never cross a tile column, inter prediction reads only reference frames). The partitions are replayed
// from the parse kernel's per-superblock list; coefficients and eobs are consumed in the order the parse wrote them.
// The DSP is Vp9IntraPred (vp9_predict_intra_block), Vp9InterPred (dec_build_inter_predictors_sb, one plane) and Vp9Itx.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Per-frame int arguments of the reconstruction kernel.</summary>
internal static class Vp9RArg
{
    public const int MiCols = 0, MiRows = 1, Log2TileCols = 2, Log2TileRows = 3, Lossless = 4, SsX = 5, SsY = 6, SbCols = 7,
        GridOffset = 8, DqStride = 9, DqU = 10, DqV = 11;
    /// <summary>Current frame, per plane: absolute offset in the pool, stride, libvpx y_width/uv_width, y_height/uv_height.</summary>
    public const int Cur = 16;           // + plane * 4
    /// <summary>References LAST, GOLDEN, ALTREF, per plane: offset, stride, crop width, crop height.</summary>
    public const int Ref = 32;           // + (ref * 3 + plane) * 4
    /// <summary>Per reference: x_scale_fp, y_scale_fp, x_step_q4, y_step_q4.</summary>
    public const int Scale = 72;         // + ref * 4
    public const int Count = 88;
    /// <summary>Scratch ints per thread (the largest DSP need: Vp9Itx).</summary>
    public const int Scratch = 1056;
}

internal static class Vp9ReconKernels
{
    private const int Block8x8 = 3, Block64x64 = 12;
    private const int PartitionNone = 0, PartitionHorz = 1, PartitionVert = 2, PartitionSplit = 3;

    /// <summary>The inter prediction reference plane for MV reference <paramref name="refFrame"/> (1 LAST .. 3 ALTREF).</summary>
    private static Vp9InterRefPlane RefPlane(ArrayView<int> a, int refFrame, int plane)
    {
        int r = refFrame < 1 ? 0 : refFrame - 1;
        int at = Vp9RArg.Ref + (r * 3 + plane) * 4, sc = Vp9RArg.Scale + r * 4;
        Vp9InterRefPlane p;
        p.Offset = a[at];
        p.Stride = a[at + 1];
        p.CropWidth = a[at + 2];
        p.CropHeight = a[at + 3];
        p.XScaleFp = a[sc];
        p.YScaleFp = a[sc + 1];
        p.XStepQ4 = a[sc + 2];
        p.YStepQ4 = a[sc + 3];
        return p;
    }

    /// <summary>recon_block for one block (bwl / bhl as process_partition passes them).</summary>
    private static void ReconBlock(ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi, ArrayView<int> sbBuf,
        ArrayView<int> dq, ArrayView<int> scratch, long scr, int sb, int miRow, int miCol, int colStart, int bwl, int bhl,
        ref int dqY, ref int dqU, ref int dqV, ref int eobY, ref int eobU, ref int eobV)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int bw = 1 << (bwl - 1), bh = 1 << (bhl - 1);
        int toTop = -(miRow * 8 * 8), toBottom = (miRows - bh - miRow) * 8 * 8;
        int toLeft = -(miCol * 8 * 8), toRight = (miCols - bw - miCol) * 8 * 8;
        int ssX = a[Vp9RArg.SsX], ssY = a[Vp9RArg.SsY];
        bool inter = mi[b + Vp9Mi.Ref0] > Vp9Mi.IntraFrame;
        bool skip = mi[b + Vp9Mi.Skip] != 0;
        bool lossless = a[Vp9RArg.Lossless] != 0;
        int bsize = mi[b + Vp9Mi.SbType];
        int sbEob = sb * Vp9PLayout.SbStride;

        if (inter)
        {
            for (int plane = 0; plane < 3; plane++)
            {
                int sx = plane == 0 ? 0 : ssX, sy = plane == 0 ? 0 : ssY;
                int stride = a[Vp9RArg.Cur + plane * 4 + 1];
                long dstAt = a[Vp9RArg.Cur + plane * 4] + (long)((miRow * 8) >> sy) * stride + ((miCol * 8) >> sx);
                int ref0 = mi[b + Vp9Mi.Ref0], ref1 = mi[b + Vp9Mi.Ref1];
                Vp9InterPred.PredictBlockPlane(sx, sy, miRow, miCol, miRows, miCols, mi, b,
                    pool, RefPlane(a, ref0, plane), pool, RefPlane(a, ref1 > 0 ? ref1 : ref0, plane), pool, dstAt, stride, scratch, scr);
            }
            if (skip) return;
        }

        for (int plane = 0; plane < 3; plane++)
        {
            int sx = plane == 0 ? 0 : ssX, sy = plane == 0 ? 0 : ssY;
            int txSize = plane == 0 ? mi[b + Vp9Mi.TxSize]
                : t[Vp9T.UvTxsizeLookup + ((bsize * 4 + mi[b + Vp9Mi.TxSize]) * 2 + sx) * 2 + sy];
            int n4w = (bw << 1) >> sx, n4h = (bh << 1) >> sy;
            int step = 1 << txSize;
            int maxBlocksWide = n4w + (toRight >= 0 ? 0 : toRight >> (5 + sx));
            int maxBlocksHigh = n4h + (toBottom >= 0 ? 0 : toBottom >> (5 + sy));
            int stride = a[Vp9RArg.Cur + plane * 4 + 1];
            long planeAt = a[Vp9RArg.Cur + plane * 4];
            int px0 = (miCol * 8) >> sx, py0 = (miRow * 8) >> sy;
            for (int row = 0; row < maxBlocksHigh; row += step)
                for (int col = 0; col < maxBlocksWide; col += step)
                {
                    long dstAt = planeAt + (long)(py0 + row * 4) * stride + px0 + col * 4;
                    int txType = 0;
                    if (!inter)
                    {
                        int mode = plane == 0 ? mi[b + Vp9Mi.Mode] : mi[b + Vp9Mi.UvMode];
                        if (bsize < Block8x8 && plane == 0) mode = mi[b + Vp9Mi.BmiMode + (row << 1) + col];
                        Vp9IntraPred.Predict(bwl - sx, txSize, mode, col, row, miRow != 0, miCol > colStart,
                            toLeft, toTop, toRight, toBottom, sx, sy, a[Vp9RArg.Cur + plane * 4 + 2], a[Vp9RArg.Cur + plane * 4 + 3],
                            pool, dstAt, stride, scratch, scr);
                        if (skip) continue;
                        txType = plane != 0 || lossless ? 0 : t[Vp9T.IntraModeToTxTypeLookup + mode];
                    }
                    int eob, dqAt;
                    if (plane == 0) { eob = sbBuf[sbEob + Vp9PLayout.SbEobY + eobY]; eobY++; dqAt = dqY; dqY += 16 << (txSize << 1); }
                    else if (plane == 1) { eob = sbBuf[sbEob + Vp9PLayout.SbEobU + eobU]; eobU++; dqAt = dqU; dqU += 16 << (txSize << 1); }
                    else { eob = sbBuf[sbEob + Vp9PLayout.SbEobV + eobV]; eobV++; dqAt = dqV; dqV += 16 << (txSize << 1); }
                    if (eob > 0) Vp9Itx.InverseTransformAdd(dq, dqAt, txSize, txType, lossless, eob, pool, dstAt, stride, scratch, scr);
                }
        }
    }

    /// <summary>One thread per tile column: every superblock of the column, tile row after tile row, as libvpx RECON.</summary>
    public static void ReconKernel(Index1D index, ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi,
        ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> scratch)
    {
        int tileCol = index;
        int log2Cols = a[Vp9RArg.Log2TileCols], log2Rows = a[Vp9RArg.Log2TileRows];
        int tileRows = 1 << log2Rows;
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int sbCols = (miCols + 7) >> 3, sbRows = (miRows + 7) >> 3;
        int colStart = ((tileCol * sbCols) >> log2Cols) << 3, colEnd = (((tileCol + 1) * sbCols) >> log2Cols) << 3;
        if (colStart > miCols) colStart = miCols;
        if (colEnd > miCols) colEnd = miCols;
        long scr = (long)tileCol * (Vp9RArg.Scratch + 64);
        long stack = scr + Vp9RArg.Scratch;   // 16 entries x 4 ints after the DSP scratch
        int dqStride = a[Vp9RArg.DqStride];
        for (int tileRow = 0; tileRow < tileRows; tileRow++)
        {
            int rowStart = ((tileRow * sbRows) >> log2Rows) << 3, rowEnd = (((tileRow + 1) * sbRows) >> log2Rows) << 3;
            if (rowStart > miRows) rowStart = miRows;
            if (rowEnd > miRows) rowEnd = miRows;
            for (int miRow = rowStart; miRow < rowEnd; miRow += 8)
                for (int miCol = colStart; miCol < colEnd; miCol += 8)
                {
                    int sb = (miRow >> 3) * sbCols + (miCol >> 3);
                    int dqY = sb * dqStride, dqU = dqY + a[Vp9RArg.DqU], dqV = dqY + a[Vp9RArg.DqV];
                    int eobY = 0, eobU = 0, eobV = 0, parts = 0;
                    scratch[stack] = miRow; scratch[stack + 1] = miCol; scratch[stack + 2] = Block64x64; scratch[stack + 3] = 4;
                    int sp = 1;
                    while (sp > 0)
                    {
                        sp--;
                        int rr = scratch[stack + sp * 4], cc = scratch[stack + sp * 4 + 1], bsize = scratch[stack + sp * 4 + 2], n4l2 = scratch[stack + sp * 4 + 3];
                        int n8l2 = n4l2 - 1, hbs = (1 << n8l2) >> 1;
                        if (rr >= miRows || cc >= miCols) continue;
                        bool hasRows = rr + hbs < miRows, hasCols = cc + hbs < miCols;
                        int partition = sbBuf[sb * Vp9PLayout.SbStride + Vp9PLayout.SbPartitions + parts];
                        parts++;
                        int subsize = t[Vp9T.SubsizeLookup + partition * 13 + bsize];
                        if (hbs == 0)
                            ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, rr, cc, colStart, 1, 1, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        else if (partition == PartitionNone)
                            ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, rr, cc, colStart, n4l2, n4l2, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        else if (partition == PartitionHorz)
                        {
                            ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, rr, cc, colStart, n4l2, n8l2, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                            if (hasRows)
                                ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, rr + hbs, cc, colStart, n4l2, n8l2, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        }
                        else if (partition == PartitionVert)
                        {
                            ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, rr, cc, colStart, n8l2, n4l2, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                            if (hasCols)
                                ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, rr, cc + hbs, colStart, n8l2, n4l2, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        }
                        else
                        {
                            for (int k = 3; k >= 0; k--)
                            {
                                scratch[stack + sp * 4] = rr + ((k >> 1) != 0 ? hbs : 0);
                                scratch[stack + sp * 4 + 1] = cc + ((k & 1) != 0 ? hbs : 0);
                                scratch[stack + sp * 4 + 2] = subsize;
                                scratch[stack + sp * 4 + 3] = n8l2;
                                sp++;
                            }
                        }
                    }
                }
        }
    }
}
