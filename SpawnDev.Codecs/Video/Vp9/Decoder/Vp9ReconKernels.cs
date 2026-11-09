// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 reconstruction on the accelerator: prediction + inverse transform of every block, from the parse kernel's
// output. A port of libvpx v1.17.0 (BSD-3-Clause) vp9/decoder/vp9_decodeframe.c recon_tile_row / process_partition
// (RECON) / recon_block / predict_and_reconstruct_intra_block_row_mt / reconstruct_inter_block_row_mt /
// predict_recon_intra / predict_recon_inter, scheduled in two passes: inter blocks (all superblocks at once), then intra
// blocks on a superblock wavefront. Blocks are replayed from the parse kernel's per-superblock block table (record,
// size, and where its coefficients and eobs start), so every block reads exactly the data the parse wrote for it.
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

    /// <summary>Reconstruct the blocks of superblock <paramref name="sb"/> that are inter (pass A) or intra (pass B), in parse order.</summary>
    private static void ReconSuperblock(ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi, ArrayView<int> sbBuf,
        ArrayView<int> dq, ArrayView<int> scratch, long scr, int sb, bool interPass)
    {
        int miCols = a[Vp9RArg.MiCols];
        int tb = sb * Vp9PLayout.SbStride;
        int count = sbBuf[tb + Vp9PLayout.SbBlockCount];
        for (int slot = 0; slot < count; slot++)
        {
            int e = tb + Vp9PLayout.SbBlocks + slot * Vp9PLayout.BlkStride;
            int rec = sbBuf[e + Vp9PLayout.BlkRec];
            bool inter = mi[rec * Vp9Mi.Stride + Vp9Mi.Ref0] > Vp9Mi.IntraFrame;
            if (inter != interPass) continue;
            int miRow = rec / miCols, miCol = rec % miCols;
            int dqY = sbBuf[e + Vp9PLayout.BlkDqY], dqU = sbBuf[e + Vp9PLayout.BlkDqU], dqV = sbBuf[e + Vp9PLayout.BlkDqV];
            int eobY = sbBuf[e + Vp9PLayout.BlkEobY], eobU = sbBuf[e + Vp9PLayout.BlkEobU], eobV = sbBuf[e + Vp9PLayout.BlkEobV];
            int colStart = interPass ? 0 : TileColStart(miCol, miCols, a[Vp9RArg.Log2TileCols]);
            ReconBlock(pool, a, t, mi, sbBuf, dq, scratch, scr, sb, miRow, miCol, colStart,
                sbBuf[e + Vp9PLayout.BlkBwl], sbBuf[e + Vp9PLayout.BlkBhl], ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
        }
    }

    /// <summary>
    /// Pass A, one thread per superblock: every inter block (prediction from reference frames + residual). Inter blocks
    /// read only reference frames, so all superblocks run at once.
    /// </summary>
    public static void InterPassKernel(Index1D index, ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi,
        ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> scratch)
    {
        int sb = index;
        ReconSuperblock(pool, a, t, mi, sbBuf, dq, scratch, (long)sb * Vp9RArg.Scratch, sb, true);
    }

    /// <summary>
    /// Pass B, one thread per superblock on wavefront step <paramref name="k"/> = row + col: the superblock's intra blocks
    /// in parse order. Intra prediction reads only pixels above and to the left (libvpx: above-right stays inside the
    /// block), all reconstructed by pass A or by earlier steps / earlier blocks of the same superblock.
    /// </summary>
    public static void IntraWaveKernel(Index1D index, ArrayView<byte> pool, ArrayView<int> a, ArrayView<int> t, ArrayView<int> mi,
        ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> scratch, int k)
    {
        int r = index, c = k - r;
        int sbCols = (a[Vp9RArg.MiCols] + 7) >> 3;
        if (c < 0 || c >= sbCols) return;
        int sb = r * sbCols + c;
        ReconSuperblock(pool, a, t, mi, sbBuf, dq, scratch, (long)sb * Vp9RArg.Scratch, sb, false);
    }
}
