// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 parse kernel, tokens + blocks + partitions + the kernel entry. Port of libvpx v1.17.0 (BSD-3-Clause)
// vp9/decoder/vp9_detokenize.c (decode_coefs, vp9_decode_block_tokens) and vp9/decoder/vp9_decodeframe.c
// (parse_block, parse_*_block_row_mt, process_partition PARSE, read_partition, partition contexts, decode_tiles' tile
// loop). Coefficients are stored as libvpx's 8-bit build stores them: tran_low_t is int16_t there, so (short) values.

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

internal static partial class Vp9ParseKernels
{
    private const int CountsTokenEob = 3;

    /// <summary>decode_coefs; returns the eob.</summary>
    private static int DecodeCoefs(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> work, int tokenCache, ArrayView<int> dq, int dqAt, int tx, int type, int refType, int dqDc, int dqAc,
        int c0Ctx, int scan, int nb, ArrayView<int> counts, int cnt)
    {
        int maxEob = 16 << (tx << 1);
        int probsBase = fc + Vp9Ctx.Coef + ((tx * 2 + type) * 2 + refType) * 6 * 6 * 3;
        int bandTable = tx == 0 ? Vp9T.CoefbandTrans4x4 : Vp9T.CoefbandTrans8x8plus;
        int cntBase = cnt >= 0 ? cnt + Vp9Cnt.Coef + ((tx * 2 + type) * 2 + refType) * 6 * 6 * 4 : -1;
        int eobBase = cnt >= 0 ? cnt + Vp9Cnt.EobBranch + ((tx * 2 + type) * 2 + refType) * 6 * 6 : -1;
        int dqShift = tx == 3 ? 1 : 0;
        int dqv = dqDc;
        int c = 0, coefCtx = c0Ctx;
        while (c < maxEob)
        {
            int band = t[bandTable + c];
            int prob = probsBase + (band * 6 + coefCtx) * 3;
            if (cnt >= 0) counts[eobBase + band * 6 + coefCtx]++;
            if (Read(ref r, d, ctx[prob]) == 0)
            {
                if (cnt >= 0) counts[cntBase + (band * 6 + coefCtx) * 4 + CountsTokenEob]++;
                break;
            }
            while (Read(ref r, d, ctx[prob + 1]) == 0)
            {
                if (cnt >= 0) counts[cntBase + (band * 6 + coefCtx) * 4 + 0]++;
                dqv = dqAc;
                work[tokenCache + t[scan + c]] = 0;
                ++c;
                if (c >= maxEob) return c;   // zero tokens at the end (no eob token)
                coefCtx = CoefContext(t, work, tokenCache, nb, c);
                band = t[bandTable + c];
                prob = probsBase + (band * 6 + coefCtx) * 3;
            }
            int v;
            if (Read(ref r, d, ctx[prob + 2]) != 0)
            {
                int p = Vp9T.Pareto8Full + (ctx[prob + 2] - 1) * 8;
                if (cnt >= 0) counts[cntBase + (band * 6 + coefCtx) * 4 + 2]++;
                if (Read(ref r, d, t[p]) != 0)
                {
                    int val;
                    if (Read(ref r, d, t[p + 3]) != 0)
                    {
                        work[tokenCache + t[scan + c]] = 5;
                        if (Read(ref r, d, t[p + 5]) != 0)
                        {
                            if (Read(ref r, d, t[p + 7]) != 0) val = 67 + ReadCoeff(ref r, d, t, Vp9T.Cat6Prob, 14);
                            else val = 35 + ReadCoeff(ref r, d, t, Vp9T.Cat5Prob, 5);
                        }
                        else if (Read(ref r, d, t[p + 6]) != 0) val = 19 + ReadCoeff(ref r, d, t, Vp9T.Cat4Prob, 4);
                        else val = 11 + ReadCoeff(ref r, d, t, Vp9T.Cat3Prob, 3);
                    }
                    else
                    {
                        work[tokenCache + t[scan + c]] = 4;
                        if (Read(ref r, d, t[p + 4]) != 0) val = 7 + ReadCoeff(ref r, d, t, Vp9T.Cat2Prob, 2);
                        else val = 5 + ReadCoeff(ref r, d, t, Vp9T.Cat1Prob, 1);
                    }
                    v = (val * dqv) >> dqShift;
                }
                else if (Read(ref r, d, t[p + 1]) != 0)
                {
                    work[tokenCache + t[scan + c]] = 3;
                    v = ((3 + Read(ref r, d, t[p + 2])) * dqv) >> dqShift;
                }
                else
                {
                    work[tokenCache + t[scan + c]] = 2;
                    v = (2 * dqv) >> dqShift;
                }
            }
            else
            {
                if (cnt >= 0) counts[cntBase + (band * 6 + coefCtx) * 4 + 1]++;
                work[tokenCache + t[scan + c]] = 1;
                v = dqv >> dqShift;
            }
            int signed = Read(ref r, d, 128) != 0 ? -v : v;
            dq[dqAt + t[scan + c]] = (short)signed;
            ++c;
            coefCtx = CoefContext(t, work, tokenCache, nb, c);
            dqv = dqAc;
        }
        return c;
    }

    private static int ReadCoeff(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> t, int probs, int n)
    {
        int val = 0;
        for (int i = 0; i < n; i++) val = (val << 1) | Read(ref r, d, t[probs + i]);
        return val;
    }

    /// <summary>get_coef_context (MAX_NEIGHBORS = 2).</summary>
    private static int CoefContext(ArrayView<int> t, ArrayView<int> work, int tokenCache, int nb, int c) =>
        (1 + work[tokenCache + t[nb + 2 * c]] + work[tokenCache + t[nb + 2 * c + 1]]) >> 1;

    /// <summary>The scan / neighbors table offsets of vp9_scan_orders[tx][type].</summary>
    private static int ScanOffset(int tx, int type) => tx switch
    {
        0 => type switch { 0 => Vp9T.Scan00, 1 => Vp9T.Scan01, 2 => Vp9T.Scan02, _ => Vp9T.Scan03 },
        1 => type switch { 0 => Vp9T.Scan10, 1 => Vp9T.Scan11, 2 => Vp9T.Scan12, _ => Vp9T.Scan13 },
        2 => type switch { 0 => Vp9T.Scan20, 1 => Vp9T.Scan21, 2 => Vp9T.Scan22, _ => Vp9T.Scan23 },
        _ => type switch { 0 => Vp9T.Scan30, 1 => Vp9T.Scan31, 2 => Vp9T.Scan32, _ => Vp9T.Scan33 },
    };

    private static int NeighborsOffset(int tx, int type) => tx switch
    {
        0 => type switch { 0 => Vp9T.Neighbors00, 1 => Vp9T.Neighbors01, 2 => Vp9T.Neighbors02, _ => Vp9T.Neighbors03 },
        1 => type switch { 0 => Vp9T.Neighbors10, 1 => Vp9T.Neighbors11, 2 => Vp9T.Neighbors12, _ => Vp9T.Neighbors13 },
        2 => type switch { 0 => Vp9T.Neighbors20, 1 => Vp9T.Neighbors21, 2 => Vp9T.Neighbors22, _ => Vp9T.Neighbors23 },
        _ => type switch { 0 => Vp9T.Neighbors30, 1 => Vp9T.Neighbors31, 2 => Vp9T.Neighbors32, _ => Vp9T.Neighbors33 },
    };

    /// <summary>
    /// vp9_decode_block_tokens for one transform block at (x, y) (4x4 units in the plane, relative to the block's above /
    /// left context positions). maxW / maxH = xd->max_blocks_wide / high (0 = the block is inside the frame).
    /// </summary>
    private static int DecodeBlockTokens(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> a, ArrayView<int> work, int scratch, ArrayView<int> dq, int dqAt, int plane, int tx, int txType,
        int aboveAt, int leftAt, int x, int y, int maxW, int maxH, int refType, int segId, ArrayView<int> counts, int cnt)
    {
        int n = 1 << tx;
        int ac = 0, lc = 0;
        for (int k = 0; k < n; k++)
        {
            ac |= work[aboveAt + x + k];
            lc |= work[leftAt + y + k];
        }
        int coefCtx = (ac != 0 ? 1 : 0) + (lc != 0 ? 1 : 0);
        int dqDc = a[Vp9PArg.Dequant + segId * 4 + (plane == 0 ? 0 : 2)];
        int dqAc = a[Vp9PArg.Dequant + segId * 4 + (plane == 0 ? 1 : 3)];
        int eob = DecodeCoefs(ref r, d, ctx, fc, t, work, scratch + Vp9PLayout.ScTokenCache, dq, dqAt, tx, plane == 0 ? 0 : 1, refType,
            dqDc, dqAc, coefCtx, ScanOffset(tx, txType), NeighborsOffset(tx, txType), counts, cnt);
        // the context entries past the frame edge are cleared (libvpx's shifted little-endian store)
        int validA = n, validL = n;
        if (tx > 0)
        {
            if (maxW != 0 && n + x > maxW) validA = maxW - x;
            if (maxH != 0 && n + y > maxH) validL = maxH - y;
        }
        int nz = eob > 0 ? 1 : 0;
        for (int k = 0; k < n; k++)
        {
            work[aboveAt + x + k] = k < validA ? nz : 0;
            work[leftAt + y + k] = k < validL ? nz : 0;
        }
        return eob;
    }

    /// <summary>parse_block; false when the block is corrupt (libvpx xd->corrupted).</summary>
    private static bool ParseBlock(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> a, ArrayView<int> ctx, ArrayView<int> t,
        ArrayView<int> mi, int sb, ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> work, ArrayView<int> maps,
        ArrayView<int> counts, ArrayView<int> results, int scratch, int cnt, int miRow, int miCol, int colStart, int colEnd,
        int bsize, int bwl, int bhl, int bmodeWl, int bmodeHl, ref int dqY, ref int dqU, ref int dqV, ref int eobY, ref int eobU, ref int eobV)
    {
        int fc = Vp9Ctx.Current * Vp9Ctx.Bytes;
        int miCols = a[Vp9PArg.MiCols], miRows = a[Vp9PArg.MiRows];
        int bw = 1 << (bwl - 1), bh = 1 << (bhl - 1);
        int xMis = bw < miCols - miCol ? bw : miCols - miCol;
        int yMis = bh < miRows - miRow ? bh : miRows - miRow;
        int miOffset = miRow * miCols + miCol;
        int rec = miOffset;
        int grid = a[Vp9PArg.GridOffset];
        for (int y = 0; y < yMis; y++)
            for (int x = 0; x < xMis; x++) mi[grid + miOffset + y * miCols + x] = rec;
        int b = rec * Vp9Mi.Stride;
        mi[b + Vp9Mi.SbType] = bsize;
        // block table entry (reconstruction replays blocks from it)
        {
            int tb = sb * Vp9PLayout.SbStride;
            int slot = sbBuf[tb + Vp9PLayout.SbBlockCount];
            sbBuf[tb + Vp9PLayout.SbBlockCount] = slot + 1;
            int e = tb + Vp9PLayout.SbBlocks + slot * Vp9PLayout.BlkStride;
            sbBuf[e + Vp9PLayout.BlkRec] = rec;
            sbBuf[e + Vp9PLayout.BlkBwl] = bwl;
            sbBuf[e + Vp9PLayout.BlkBhl] = bhl;
            sbBuf[e + Vp9PLayout.BlkDqY] = dqY;
            sbBuf[e + Vp9PLayout.BlkDqU] = dqU;
            sbBuf[e + Vp9PLayout.BlkDqV] = dqV;
            sbBuf[e + Vp9PLayout.BlkEobY] = eobY;
            sbBuf[e + Vp9PLayout.BlkEobU] = eobU;
            sbBuf[e + Vp9PLayout.BlkEobV] = eobV;
        }

        var p = new BlockPos
        {
            MiRow = miRow, MiCol = miCol, MiColStart = colStart, MiColEnd = colEnd,
            ToTop = -(miRow * 8 * 8), ToBottom = (miRows - bh - miRow) * 8 * 8,
            ToLeft = -(miCol * 8 * 8), ToRight = (miCols - bw - miCol) * 8 * 8,
        };
        int above = miRow != 0 ? mi[grid + (miRow - 1) * miCols + miCol] : -1;
        int left = miCol > colStart ? mi[grid + miRow * miCols + miCol - 1] : -1;
        int txMode = results[Vp9CompArgs.RTxMode];
        bool ok = true;
        bool inter = false;
        if (a[Vp9PArg.IntraOnly] != 0)
            ReadIntraFrameModeInfo(ref r, d, ctx, fc, t, a, mi, maps, rec, above, left, bsize, txMode, miOffset, xMis, yMis, counts, cnt);
        else
        {
            int segId = ReadInterSegmentId(ref r, d, t, a, maps, mi, rec, above, left, miOffset, xMis, yMis);
            mi[b + Vp9Mi.SegmentId] = segId;
            int skip = ReadSkip(ref r, d, ctx, fc, a, mi, above, left, segId, counts, cnt);
            mi[b + Vp9Mi.Skip] = skip;
            // read_is_inter_block
            if (SegFeatureActive(a, segId, SegLvlRefFrame)) inter = SegData(a, segId, SegLvlRefFrame) != Vp9Mi.IntraFrame;
            else
            {
                int c = Vp9ParseContexts.IntraInterContext(mi, above, left);
                int bit = Read(ref r, d, ctx[fc + Vp9Ctx.IntraInter + c]);
                if (cnt >= 0) counts[cnt + Vp9Cnt.IntraInter + c * 2 + bit]++;
                inter = bit != 0;
            }
            mi[b + Vp9Mi.TxSize] = ReadTxSize(ref r, d, ctx, fc, t, mi, above, left, bsize, txMode, skip == 0 || !inter, counts, cnt);
            if (inter) ok = ReadInterBlockModeInfo(ref r, d, ctx, fc, t, a, mi, maps, results, p, rec, above, left, bsize, bmodeWl, bmodeHl, counts, cnt);
            else ReadIntraBlockModeInfo(ref r, d, ctx, fc, t, mi, rec, bsize, counts, cnt);
            // the frame's MV buffer (vp9_read_mode_info)
            int curMv = a[Vp9PArg.CurMvOffset];
            for (int h = 0; h < yMis; h++)
                for (int w = 0; w < xMis; w++)
                {
                    int at = curMv + (miOffset + h * miCols + w) * Vp9PLayout.MvStride;
                    maps[at] = mi[b + Vp9Mi.Ref0];
                    maps[at + 1] = mi[b + Vp9Mi.Ref1];
                    maps[at + 2] = mi[b + Vp9Mi.Mv0];
                    maps[at + 3] = mi[b + Vp9Mi.Mv1];
                }
        }
        if (!ok) return false;
        inter = Vp9Mi.IsInter(mi, rec);

        int ssX = a[Vp9PArg.SsX], ssY = a[Vp9PArg.SsY];
        int alignedCols = a[Vp9PArg.SbCols] * 8;
        int skipFlag = mi[b + Vp9Mi.Skip];
        if (skipFlag != 0)
        {
            // dec_reset_skip_context
            for (int plane = 0; plane < 3; plane++)
            {
                int sx = plane == 0 ? 0 : ssX, sy = plane == 0 ? 0 : ssY;
                int aboveAt = a[Vp9PArg.AboveCtxOffset] + plane * 2 * alignedCols + ((miCol * 2) >> sx);
                int leftAt = scratch + Vp9PLayout.ScLeftCtx + plane * 16 + (((miRow * 2) & 15) >> sy);
                int n4w = (bw << 1) >> sx, n4h = (bh << 1) >> sy;
                for (int k = 0; k < n4w; k++) work[aboveAt + k] = 0;
                for (int k = 0; k < n4h; k++) work[leftAt + k] = 0;
            }
        }
        if (inter && skipFlag != 0) return true;

        bool lossless = a[Vp9PArg.Lossless] != 0;
        int segIdB = mi[b + Vp9Mi.SegmentId];
        int saveDqY = dqY, saveDqU = dqU, saveDqV = dqV, saveEobY = eobY, saveEobU = eobU, saveEobV = eobV;
        int eobTotal = 0;
        int sbEob = sb * Vp9PLayout.SbStride;
        for (int plane = 0; plane < 3; plane++)
        {
            int sx = plane == 0 ? 0 : ssX, sy = plane == 0 ? 0 : ssY;
            int txSize = plane == 0 ? mi[b + Vp9Mi.TxSize]
                : t[Vp9T.UvTxsizeLookup + ((bsize * 4 + mi[b + Vp9Mi.TxSize]) * 2 + sx) * 2 + sy];
            int n4w = (bw << 1) >> sx, n4h = (bh << 1) >> sy;
            int step = 1 << txSize;
            int maxBlocksWide = n4w + (p.ToRight >= 0 ? 0 : p.ToRight >> (5 + sx));
            int maxBlocksHigh = n4h + (p.ToBottom >= 0 ? 0 : p.ToBottom >> (5 + sy));
            int ctxMaxW = p.ToRight >= 0 ? 0 : maxBlocksWide;
            int ctxMaxH = p.ToBottom >= 0 ? 0 : maxBlocksHigh;
            int aboveAt = a[Vp9PArg.AboveCtxOffset] + plane * 2 * alignedCols + ((miCol * 2) >> sx);
            int leftAt = scratch + Vp9PLayout.ScLeftCtx + plane * 16 + (((miRow * 2) & 15) >> sy);
            for (int row = 0; row < maxBlocksHigh; row += step)
                for (int col = 0; col < maxBlocksWide; col += step)
                {
                    int txType = 0;
                    if (!inter)
                    {
                        if (skipFlag != 0) continue;
                        int mode = plane == 0 ? mi[b + Vp9Mi.Mode] : mi[b + Vp9Mi.UvMode];
                        if (bsize < Block8x8 && plane == 0) mode = mi[b + Vp9Mi.BmiMode + (row << 1) + col];
                        txType = plane != 0 || lossless ? 0 : t[Vp9T.IntraModeToTxTypeLookup + mode];
                    }
                    int dqAt = plane == 0 ? dqY : plane == 1 ? dqU : dqV;
                    int eob = DecodeBlockTokens(ref r, d, ctx, fc, t, a, work, scratch, dq, dqAt, plane, txSize, txType,
                        aboveAt, leftAt, col, row, ctxMaxW, ctxMaxH, inter ? 1 : 0, segIdB, counts, cnt);
                    eobTotal += eob;
                    int adv = 16 << (txSize << 1);
                    if (plane == 0) { sbBuf[sbEob + Vp9PLayout.SbEobY + eobY] = eob; eobY++; dqY += adv; }
                    else if (plane == 1) { sbBuf[sbEob + Vp9PLayout.SbEobU + eobU] = eob; eobU++; dqU += adv; }
                    else { sbBuf[sbEob + Vp9PLayout.SbEobV + eobV] = eob; eobV++; dqV += adv; }
                }
        }
        if (inter && bsize >= Block8x8 && eobTotal == 0)
        {
            mi[b + Vp9Mi.Skip] = 1;   // skip loopfilter (and the neighbours' skip / tx contexts see it)
            dqY = saveDqY; dqU = saveDqU; dqV = saveDqV;
            eobY = saveEobY; eobU = saveEobU; eobV = saveEobV;
        }
        return true;
    }

    /// <summary>One thread per tile column: every superblock of the column, tile row after tile row.</summary>
    public static void ParseKernel(Index1D index, ArrayView<byte> d, ArrayView<int> a, ArrayView<int> ctx, ArrayView<int> t,
        ArrayView<int> mi, ArrayView<int> sbBuf, ArrayView<int> dq, ArrayView<int> work, ArrayView<int> maps,
        ArrayView<int> counts, ArrayView<int> results)
    {
        int tileCol = index;
        int log2Cols = a[Vp9PArg.Log2TileCols], log2Rows = a[Vp9PArg.Log2TileRows];
        int tileCols = 1 << log2Cols, tileRows = 1 << log2Rows;
        int miCols = a[Vp9PArg.MiCols], miRows = a[Vp9PArg.MiRows];
        int sbCols = (miCols + 7) >> 3, sbRows = (miRows + 7) >> 3;
        int colStart = ((tileCol * sbCols) >> log2Cols) << 3;
        int colEnd = (((tileCol + 1) * sbCols) >> log2Cols) << 3;
        if (colStart > miCols) colStart = miCols;
        if (colEnd > miCols) colEnd = miCols;
        int cnt = a[Vp9PArg.CountsOn] != 0 ? tileCol * Vp9Cnt.Size : -1;
        int scratch = a[Vp9PArg.ScratchOffset] + tileCol * Vp9PLayout.ScratchStride;
        int fc = Vp9Ctx.Current * Vp9Ctx.Bytes;
        int stack = scratch + Vp9PLayout.ScStack;
        int status = 0;
        int dqStride = a[Vp9PArg.DqStride];
        bool intraOnly = a[Vp9PArg.IntraOnly] != 0;

        for (int tileRow = 0; tileRow < tileRows && status == 0; tileRow++)
        {
            int rowStart = ((tileRow * sbRows) >> log2Rows) << 3;
            int rowEnd = (((tileRow + 1) * sbRows) >> log2Rows) << 3;
            if (rowStart > miRows) rowStart = miRows;
            if (rowEnd > miRows) rowEnd = miRows;
            int tb = Vp9PArg.TileBuf + (tileRow * tileCols + tileCol) * 2;
            var r = Vp8BoolDecoderGpu.Init(d, a[tb], a[tb + 1]);
            if (Read(ref r, d, 128) != 0) status = 1;   // marker bit
            for (int miRow = rowStart; miRow < rowEnd && status == 0; miRow += 8)
            {
                for (int k = 0; k < 48; k++) work[scratch + Vp9PLayout.ScLeftCtx + k] = 0;
                for (int k = 0; k < 8; k++) work[scratch + Vp9PLayout.ScLeftPart + k] = 0;
                for (int miCol = colStart; miCol < colEnd && status == 0; miCol += 8)
                {
                    int sb = (miRow >> 3) * sbCols + (miCol >> 3);
                    int dqY = sb * dqStride, dqU = dqY + a[Vp9PArg.DqU], dqV = dqY + a[Vp9PArg.DqV];
                    int eobY = 0, eobU = 0, eobV = 0, parts = 0;
                    sbBuf[sb * Vp9PLayout.SbStride + Vp9PLayout.SbBlockCount] = 0;
                    int sp = 0;
                    work[stack] = miRow; work[stack + 1] = miCol; work[stack + 2] = Block64x64; work[stack + 3] = 4;
                    sp = 1;
                    while (sp > 0 && status == 0)
                    {
                        sp--;
                        int rr = work[stack + sp * 4], cc = work[stack + sp * 4 + 1], bsize = work[stack + sp * 4 + 2], n4l2 = work[stack + sp * 4 + 3];
                        int n8l2 = n4l2 - 1, num8 = 1 << n8l2, hbs = num8 >> 1;
                        if (rr >= miRows || cc >= miCols) continue;
                        bool hasRows = rr + hbs < miRows, hasCols = cc + hbs < miCols;
                        // read_partition
                        int aboveP = (work[a[Vp9PArg.AbovePartOffset] + cc] >> n8l2) & 1;
                        int leftP = (work[scratch + Vp9PLayout.ScLeftPart + (rr & 7)] >> n8l2) & 1;
                        int pctx = leftP * 2 + aboveP + n8l2 * 4;
                        ArrayView<int> probView = intraOnly ? t : ctx;
                        int probAt = intraOnly ? Vp9T.KfPartitionProbs + pctx * 3 : fc + Vp9Ctx.Partition + pctx * 3;
                        int partition;
                        if (hasRows && hasCols) partition = ReadTree(ref r, d, t, Vp9T.PartitionTree, probView, probAt);
                        else if (!hasRows && hasCols) partition = Read(ref r, d, probView[probAt + 1]) != 0 ? PartitionSplit : PartitionHorz;
                        else if (hasRows) partition = Read(ref r, d, probView[probAt + 2]) != 0 ? PartitionSplit : PartitionVert;
                        else partition = PartitionSplit;
                        if (cnt >= 0) counts[cnt + Vp9Cnt.Partition + pctx * 4 + partition]++;
                        sbBuf[sb * Vp9PLayout.SbStride + Vp9PLayout.SbPartitions + parts] = partition;
                        parts++;
                        int subsize = t[Vp9T.SubsizeLookup + partition * 13 + bsize];
                        bool ok = true;
                        if (hbs == 0)
                        {
                            int wl = (partition & PartitionVert) != 0 ? 0 : 1;
                            int hl = (partition & PartitionHorz) != 0 ? 0 : 1;
                            ok = ParseBlock(ref r, d, a, ctx, t, mi, sb, sbBuf, dq, work, maps, counts, results, scratch, cnt, rr, cc, colStart, colEnd,
                                subsize, 1, 1, wl, hl, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        }
                        else if (partition == PartitionNone)
                            ok = ParseBlock(ref r, d, a, ctx, t, mi, sb, sbBuf, dq, work, maps, counts, results, scratch, cnt, rr, cc, colStart, colEnd,
                                subsize, n4l2, n4l2, 0, 0, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        else if (partition == PartitionHorz)
                        {
                            ok = ParseBlock(ref r, d, a, ctx, t, mi, sb, sbBuf, dq, work, maps, counts, results, scratch, cnt, rr, cc, colStart, colEnd,
                                subsize, n4l2, n8l2, 0, 0, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                            if (ok && hasRows)
                                ok = ParseBlock(ref r, d, a, ctx, t, mi, sb, sbBuf, dq, work, maps, counts, results, scratch, cnt, rr + hbs, cc, colStart, colEnd,
                                    subsize, n4l2, n8l2, 0, 0, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        }
                        else if (partition == PartitionVert)
                        {
                            ok = ParseBlock(ref r, d, a, ctx, t, mi, sb, sbBuf, dq, work, maps, counts, results, scratch, cnt, rr, cc, colStart, colEnd,
                                subsize, n8l2, n4l2, 0, 0, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                            if (ok && hasCols)
                                ok = ParseBlock(ref r, d, a, ctx, t, mi, sb, sbBuf, dq, work, maps, counts, results, scratch, cnt, rr, cc + hbs, colStart, colEnd,
                                    subsize, n8l2, n4l2, 0, 0, ref dqY, ref dqU, ref dqV, ref eobY, ref eobU, ref eobV);
                        }
                        else
                        {
                            // PARTITION_SPLIT above 8x8: children in reverse so (rr, cc) is visited first
                            for (int k = 3; k >= 0; k--)
                            {
                                work[stack + sp * 4] = rr + ((k >> 1) != 0 ? hbs : 0);
                                work[stack + sp * 4 + 1] = cc + ((k & 1) != 0 ? hbs : 0);
                                work[stack + sp * 4 + 2] = subsize;
                                work[stack + sp * 4 + 3] = n8l2;
                                sp++;
                            }
                        }
                        if (!ok) { status = 2; break; }
                        // dec_update_partition_context
                        if ((bsize == Block8x8 || partition != PartitionSplit) && bsize >= Block8x8)
                        {
                            int pa = t[Vp9T.PartitionContextLookup + subsize * 2], pl = t[Vp9T.PartitionContextLookup + subsize * 2 + 1];
                            for (int k = 0; k < num8; k++)
                            {
                                work[a[Vp9PArg.AbovePartOffset] + cc + k] = pa;
                                work[scratch + Vp9PLayout.ScLeftPart + ((rr & 7) + k)] = pl;
                            }
                        }
                    }
                }
            }
            // vpx_reader_has_error after the tile
            if (r.Count > Vp8BoolDecoderGpu.BdValueSize && r.Count < Vp8BoolDecoderGpu.LotsOfBits && status == 0) status = 3;
        }
        work[a[Vp9PArg.StatusOffset] + tileCol] = status;
    }
}
