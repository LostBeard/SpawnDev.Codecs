// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 reconstruction on the accelerator: prediction + inverse transform of every block, from the parse kernel's
// output. A port of libvpx v1.17.0 (BSD-3-Clause) vp9/decoder/vp9_decodeframe.c recon_tile_row / process_partition
// (RECON) / recon_block / predict_and_reconstruct_intra_block_row_mt / reconstruct_inter_block_row_mt /
// predict_recon_intra / predict_recon_inter, scheduled in three launches plus a wavefront:
//   1. inter prediction, one thread per 4x4 unit of each plane (InterPredictKernel);
//   2. inter residual, one thread per inter transform block (InterResidualKernel);
//   3. intra blocks, one thread per superblock on a superblock wavefront (IntraWaveKernel).
// Inter blocks read only reference frames, which are complete, so every inter pixel's prediction is independent of the
// others; each transform block's residual is added onto its own pixels only, after all prediction (launch order).
// Intra blocks read reconstructed pixels above and to the left, inter or intra, so they come last.
// Blocks are found from the parse kernel's output: the mode info grid gives the block at any mi position, and the
// per-superblock block table (record, size, where its coefficients and eobs start) gives every block exactly the data
// the parse wrote for it. The DSP is Vp9IntraPred (vp9_predict_intra_block), Vp9InterPred
// (dec_build_inter_predictors_sb, one plane) and Vp9Itx.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Per-frame int arguments of the reconstruction kernel.</summary>
internal static class Vp9RArg
{
    public const int MiCols = 0, MiRows = 1, Log2TileCols = 2, Log2TileRows = 3, Lossless = 4, SsX = 5, SsY = 6, SbCols = 7,
        GridOffset = 8, DqStride = 9, DqU = 10, DqV = 11,
        /// <summary>Diagnostics only (0 in normal decoding): bit 0 skips inverse transforms, bit 1 skips intra prediction.</summary>
        DebugFlags = 12;
    /// <summary>Current frame, per plane: absolute offset in the pool, stride, libvpx y_width/uv_width, y_height/uv_height.</summary>
    public const int Cur = 16;           // + plane * 4
    /// <summary>References LAST, GOLDEN, ALTREF, per plane: offset, stride, crop width, crop height.</summary>
    public const int Ref = 32;           // + (ref * 3 + plane) * 4
    /// <summary>Per reference: x_scale_fp, y_scale_fp, x_step_q4, y_step_q4.</summary>
    public const int Scale = 72;         // + ref * 4
    public const int Count = 88;
    /// <summary>Scratch ints per thread of the intra pass (the largest DSP need: Vp9Itx).</summary>
    public const int Scratch = 1056;
    /// <summary>
    /// Inter residual scratch: the transform block whose coefficients start at dq offset c (a multiple of 16: every
    /// plane region and every transform block's coefficient count, 16 &lt;&lt; (2 * tx_size), is one) owns the scratch
    /// ints [c / 4 * 5, (c + n * n) / 4 * 5). Coefficient ranges of different transform blocks do not overlap, so neither
    /// do these, and n * n * 5 / 4 &gt;= n * n + n = <see cref="Vp9Itx"/>'s need for n &gt;= 4.
    /// </summary>
    public static long ResidualScratchAt(int dqAt) => (long)(dqAt >> 2) * 5;
}

internal static class Vp9ReconKernels
{
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

    /// <summary>
    /// The plane and the 4x4 unit (column, row in that plane's 4x4 units) of thread <paramref name="index"/> when the
    /// threads cover a luma area of <paramref name="lumaW"/> x <paramref name="lumaH"/> 4x4 units, then U, then V (each
    /// (lumaW &gt;&gt; ssX) x (lumaH &gt;&gt; ssY)). False past the last unit.
    /// </summary>
    private static bool UnitOf(int index, int lumaW, int lumaH, int ssX, int ssY, out int plane, out int ux, out int uy)
    {
        int chromaW = lumaW >> ssX, chromaH = lumaH >> ssY;
        int lumaUnits = lumaW * lumaH, chromaUnits = chromaW * chromaH;
        int u = index, unitsW = lumaW;
        plane = 0;
        if (u >= lumaUnits)
        {
            u -= lumaUnits;
            plane = 1;
            unitsW = chromaW;
            if (u >= chromaUnits)
            {
                u -= chromaUnits;
                plane = 2;
            }
        }
        ux = u % unitsW;
        uy = u / unitsW;
        return u < (plane == 0 ? lumaUnits : chromaUnits);
    }

    /// <summary>
    /// Inter prediction (dec_build_inter_predictors_sb), one thread per 4x4 unit of each plane over the whole superblock
    /// grid: Y units, then U, then V (launch <c>16 sbCols x 16 sbRows</c> luma units plus both chroma planes). Each thread
    /// finds its block through the mode info grid; units past the frame's last mi row / column belong to the block
    /// holding the nearest mi position when that block reaches over the frame edge (libvpx predicts those pixels too).
    /// Units of intra blocks, and units no block covers, do nothing. Writes only the thread's own 4x4 pixels.
    /// </summary>
    public static void InterPredictKernel(Index1D index, ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> mi, ArrayView<int> scratch)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int ssX = a[Vp9RArg.SsX], ssY = a[Vp9RArg.SsY];
        int sbCols = a[Vp9RArg.SbCols], sbRows = (miRows + 7) >> 3;
        if (!UnitOf(index, sbCols * 16, sbRows * 16, ssX, ssY, out int plane, out int ux, out int uy)) return;
        int sx = plane == 0 ? 0 : ssX, sy = plane == 0 ? 0 : ssY;
        // the unit's mi position, then the grid position nearest to it
        int miC = (ux << sx) >> 1, miR = (uy << sy) >> 1;
        int gc = miC < miCols ? miC : miCols - 1, gr = miR < miRows ? miR : miRows - 1;
        int rec = mi[a[Vp9RArg.GridOffset] + gr * miCols + gc];
        int b = rec * Vp9Mi.Stride;
        if (mi[b + Vp9Mi.Ref0] <= Vp9Mi.IntraFrame) return;
        int bRow = rec / miCols, bCol = rec - bRow * miCols;
        int bsize = mi[b + Vp9Mi.SbType];
        int bw = Vp9InterPred.Num8x8Wide(bsize), bh = Vp9InterPred.Num8x8High(bsize);   // 1 below BLOCK_8X8
        // past the grid: only when this block reaches the unit (the block holding the nearest grid position contains
        // gc / gr, so miC >= bCol and miR >= bRow already hold)
        if (miC >= bCol + bw || miR >= bRow + bh) return;
        int stride = a[Vp9RArg.Cur + plane * 4 + 1];
        int px0 = (bCol * 8) >> sx, py0 = (bRow * 8) >> sy;
        long dstAt = a[Vp9RArg.Cur + plane * 4] + (long)py0 * stride + px0;
        int ref0 = mi[b + Vp9Mi.Ref0], ref1 = mi[b + Vp9Mi.Ref1];
        Vp9InterPred.PredictBlockPlaneRegion(sx, sy, bRow, bCol, miRows, miCols, mi, b,
            pool, RefPlane(a, ref0, plane), pool, RefPlane(a, ref1 > 0 ? ref1 : ref0, plane), pool, dstAt, stride,
            ux * 4 - px0, uy * 4 - py0, 4, 4, scratch, (long)(int)index * Vp9InterPred.RegionScratchInts);
    }

    /// <summary>
    /// Inter residual (the inverse transform loop of reconstruct_inter_block_row_mt), one thread per 4x4 unit of each
    /// plane inside the mi grid: Y units, then U, then V (launch <c>2 miCols x 2 miRows</c> luma units plus both chroma
    /// planes). The thread at the top-left unit of a transform block of a non-skipped inter block adds that transform
    /// block's residual; the others do nothing. Launch after <see cref="InterPredictKernel"/>.
    /// The transform blocks of one plane of a block all have the block's tx size and are coded in raster order over
    /// max_blocks_high x max_blocks_wide (step 1 &lt;&lt; tx_size), so the j-th one's coefficients start
    /// j * (16 &lt;&lt; 2 tx_size) after the block's first and its eob is the j-th after the block's first; the block's
    /// firsts are in its block table entry (the parse kernel's running dq / eob positions when it began the block).
    /// </summary>
    public static void InterResidualKernel(Index1D index, ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi,
        ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> scratch)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int ssX = a[Vp9RArg.SsX], ssY = a[Vp9RArg.SsY];
        if (!UnitOf(index, miCols * 2, miRows * 2, ssX, ssY, out int plane, out int ux, out int uy)) return;
        int sx = plane == 0 ? 0 : ssX, sy = plane == 0 ? 0 : ssY;
        int rec = mi[a[Vp9RArg.GridOffset] + ((uy << sy) >> 1) * miCols + ((ux << sx) >> 1)];
        int b = rec * Vp9Mi.Stride;
        // intra blocks: pass B; skipped inter blocks have no coefficients (the parse rewinds the dq / eob positions of an
        // inter block of 8x8 or more whose eobs are all 0 and marks it skipped)
        if (mi[b + Vp9Mi.Ref0] <= Vp9Mi.IntraFrame || mi[b + Vp9Mi.Skip] != 0) return;
        int bRow = rec / miCols, bCol = rec - bRow * miCols;
        int tb = ((bRow >> 3) * a[Vp9RArg.SbCols] + (bCol >> 3)) * Vp9PLayout.SbStride;
        int e = tb + Vp9PLayout.SbBlocks + mi[b + Vp9Mi.TableSlot] * Vp9PLayout.BlkStride;
        int bw = 1 << (sbBuf[e + Vp9PLayout.BlkBwl] - 1), bh = 1 << (sbBuf[e + Vp9PLayout.BlkBhl] - 1);
        int bsize = mi[b + Vp9Mi.SbType];
        int txSize = plane == 0 ? mi[b + Vp9Mi.TxSize]
            : t[Vp9T.UvTxsizeLookup + ((bsize * 4 + mi[b + Vp9Mi.TxSize]) * 2 + sx) * 2 + sy];
        int step = 1 << txSize;
        // the unit's position in the block, in this plane's 4x4 units
        int col = ux - ((bCol * 2) >> sx), row = uy - ((bRow * 2) >> sy);
        if (((col | row) & (step - 1)) != 0) return;
        int toBottom = (miRows - bh - bRow) * 8 * 8, toRight = (miCols - bw - bCol) * 8 * 8;
        int n4w = (bw << 1) >> sx, n4h = (bh << 1) >> sy;
        int maxBlocksWide = n4w + (toRight >= 0 ? 0 : toRight >> (5 + sx));
        int maxBlocksHigh = n4h + (toBottom >= 0 ? 0 : toBottom >> (5 + sy));
        if (col >= maxBlocksWide || row >= maxBlocksHigh) return;
        int j = (row >> txSize) * ((maxBlocksWide + step - 1) >> txSize) + (col >> txSize);
        int adv = 16 << (txSize << 1);
        int dqAt, eob;
        if (plane == 0) { dqAt = sbBuf[e + Vp9PLayout.BlkDqY] + j * adv; eob = sbBuf[tb + Vp9PLayout.SbEobY + sbBuf[e + Vp9PLayout.BlkEobY] + j]; }
        else if (plane == 1) { dqAt = sbBuf[e + Vp9PLayout.BlkDqU] + j * adv; eob = sbBuf[tb + Vp9PLayout.SbEobU + sbBuf[e + Vp9PLayout.BlkEobU] + j]; }
        else { dqAt = sbBuf[e + Vp9PLayout.BlkDqV] + j * adv; eob = sbBuf[tb + Vp9PLayout.SbEobV + sbBuf[e + Vp9PLayout.BlkEobV] + j]; }
        if (eob <= 0 || (a[Vp9RArg.DebugFlags] & 1) != 0) return;
        int stride = a[Vp9RArg.Cur + plane * 4 + 1];
        long dstAt = a[Vp9RArg.Cur + plane * 4] + (long)(uy * 4) * stride + ux * 4;
        Vp9Itx.InverseTransformAdd(dq, dqAt, txSize, 0, a[Vp9RArg.Lossless] != 0, eob, pool, dstAt, stride,
            scratch, Vp9RArg.ResidualScratchAt(dqAt));
    }

    /// <summary>recon_block for one intra block (bwl / bhl as process_partition passes them).</summary>
    private static void ReconIntraBlock(ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi, ArrayView<int> sbBuf,
        ArrayView<int> dq, ArrayView<int> scratch, long scr, int sb, int miRow, int miCol, int colStart, int bwl, int bhl,
        ref int dqY, ref int dqU, ref int dqV, ref int eobY, ref int eobU, ref int eobV)
    {
        int miCols = a[Vp9RArg.MiCols], miRows = a[Vp9RArg.MiRows];
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int bw = 1 << (bwl - 1), bh = 1 << (bhl - 1);
        int toTop = -(miRow * 8 * 8), toBottom = (miRows - bh - miRow) * 8 * 8;
        int toLeft = -(miCol * 8 * 8), toRight = (miCols - bw - miCol) * 8 * 8;
        int ssX = a[Vp9RArg.SsX], ssY = a[Vp9RArg.SsY];
        bool skip = mi[b + Vp9Mi.Skip] != 0;
        bool lossless = a[Vp9RArg.Lossless] != 0;
        int bsize = mi[b + Vp9Mi.SbType];
        int sbEob = sb * Vp9PLayout.SbStride;

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
                    int mode = plane == 0 ? mi[b + Vp9Mi.Mode] : mi[b + Vp9Mi.UvMode];
                    if (bsize < Vp9Mi.Block8x8 && plane == 0) mode = mi[b + Vp9Mi.BmiMode + (row << 1) + col];
                    if ((a[Vp9RArg.DebugFlags] & 2) == 0)
                        Vp9IntraPred.Predict(bwl - sx, txSize, mode, col, row, miRow != 0, miCol > colStart,
                            toLeft, toTop, toRight, toBottom, sx, sy, a[Vp9RArg.Cur + plane * 4 + 2], a[Vp9RArg.Cur + plane * 4 + 3],
                            pool, dstAt, stride, scratch, scr);
                    if (skip) continue;
                    int txType = plane != 0 || lossless ? 0 : t[Vp9T.IntraModeToTxTypeLookup + mode];
                    int eob, dqAt;
                    if (plane == 0) { eob = sbBuf[sbEob + Vp9PLayout.SbEobY + eobY]; eobY++; dqAt = dqY; dqY += 16 << (txSize << 1); }
                    else if (plane == 1) { eob = sbBuf[sbEob + Vp9PLayout.SbEobU + eobU]; eobU++; dqAt = dqU; dqU += 16 << (txSize << 1); }
                    else { eob = sbBuf[sbEob + Vp9PLayout.SbEobV + eobV]; eobV++; dqAt = dqV; dqV += 16 << (txSize << 1); }
                    if (eob > 0 && (a[Vp9RArg.DebugFlags] & 1) == 0) Vp9Itx.InverseTransformAdd(dq, dqAt, txSize, txType, lossless, eob, pool, dstAt, stride, scratch, scr);
                }
        }
    }

    /// <summary>The first mi column of the tile column holding <paramref name="miCol"/> (libvpx tile->mi_col_start).</summary>
    private static int TileColStart(int miCol, int miCols, int log2Cols)
    {
        int sbCols = (miCols + 7) >> 3, tiles = 1 << log2Cols;
        for (int t = tiles - 1; t >= 0; t--)
        {
            int start = ((t * sbCols) >> log2Cols) << 3;
            if (start > miCols) start = miCols;
            if (miCol >= start) return start;
        }
        return 0;
    }

    /// <summary>Reconstruct the intra blocks of superblock <paramref name="sb"/>, in parse order.</summary>
    private static void ReconIntraSuperblock(ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi, ArrayView<int> sbBuf,
        ArrayView<int> dq, ArrayView<int> scratch, long scr, int sb)
    {
        int miCols = a[Vp9RArg.MiCols];
        int tb = sb * Vp9PLayout.SbStride;
        int count = sbBuf[tb + Vp9PLayout.SbBlockCount];
        for (int slot = 0; slot < count; slot++)
        {
            int e = tb + Vp9PLayout.SbBlocks + slot * Vp9PLayout.BlkStride;
            int rec = sbBuf[e + Vp9PLayout.BlkRec];
            if (mi[rec * Vp9Mi.Stride + Vp9Mi.Ref0] > Vp9Mi.IntraFrame) continue;
            int miRow = rec / miCols, miCol = rec % miCols;
            int dqY = sbBuf[e + Vp9PLayout.BlkDqY], dqU = sbBuf[e + Vp9PLayout.BlkDqU], dqV = sbBuf[e + Vp9PLayout.BlkDqV];
            int eobY = sbBuf[e + Vp9PLayout.BlkEobY], eobU = sbBuf[e + Vp9PLayout.BlkEobU], eobV = sbBuf[e + Vp9PLayout.BlkEobV];
            int colStart = TileColStart(miCol, miCols, a[Vp9RArg.Log2TileCols]);
            ReconIntraBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, miRow, miCol, colStart,
                sbBuf[e + Vp9PLayout.BlkBwl], sbBuf[e + Vp9PLayout.BlkBhl], ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
        }
    }

    /// <summary>
    /// Pass B, one thread per superblock on wavefront step <paramref name="k"/> = row + col: the superblock's intra blocks
    /// in parse order. Intra prediction reads only pixels above and to the left (libvpx: above-right stays inside the
    /// block), all reconstructed by the inter launches or by earlier steps / earlier blocks of the same superblock.
    /// </summary>
    public static void IntraWaveKernel(Index1D index, ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi,
        ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> scratch, int k)
    {
        int r = index, c = k - r;
        int sbCols = (a[Vp9RArg.MiCols] + 7) >> 3;
        if (c < 0 || c >= sbCols) return;
        int sb = r * sbCols + c;
        ReconIntraSuperblock(pool, a, t, mi, sbBuf, dq, scratch, (long)sb * Vp9RArg.Scratch, sb);
    }
}
