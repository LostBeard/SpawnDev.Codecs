// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernel 3 as STAGE kernels: one macroblock's VP8EncTokenLoop body split at libwebp's own seams, each stage its own
// kernel (one thread), launched in order for every macroblock. Same statements, same order, same results as the single
// EncodeRowKernel walk; the split exists because that kernel's WGSL was ~61k lines (Chrome's shader compiler inlines
// everything and took minutes / killed the GPU process on kernels of that size). Every stage stays well under the browser
// budget (Plans/PLAN-Oversized-Kernels.md).
//
//   PrepareStage   InitTop / InitLeft at a row start, VP8IteratorImport, the I16 / UV predictions
//   Intra16Stage   PickBestIntra16
//   Intra4Stage    PickBestIntra4
//   UVStage        PickBestUV
//   RecordStage    skip flag, RecordTokens (into the row's buffer), the partition-0 cost, VP8IteratorSaveBoundary
//
// What crosses stages lives on the device: the ModeScore scalars and the output-buffer swap in the ROW's int scratch,
// and everything that already crossed macroblocks (see EncodeRowKernel).
//
// WAVEFRONT (2026-11-20): inside a probability window (the level costs change only at the refresh, every
// max(mbCount >> 3, 96) macroblocks in raster order) a macroblock (x, y) depends on earlier ones only through (x - 1, y)
// (left samples, left nz / derr, the output-buffer swap: the row's scratch) and (x, y - 1) / (x + 1, y - 1) (top /
// top-right samples, nz, derr, the 4x4 mode map). Each stage is launched once per diagonal d = x + 2y of the window, a
// group (RecordStage: a thread) per macroblock: group g of a launch at (row, x) is macroblock (x - 2g, row + g). Within
// one launch no two macroblocks touch the same column of the top state (their columns differ by 2 or more), and a row
// has one macroblock per diagonal. The statistics (RecordStats halves saturated counters: order-dependent) and the token
// stream are the window pass's (WindowKernel), in raster order: the same bytes as the serial walk.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static partial class WebPLossyEncodeRowKernel
{
    /// <summary>Stage kernels launched per macroblock, in this order.</summary>
    public const int StageCount = 5;

    private static void SaveScore(ArrayView<int> si, ref WebPModeScore rd)
    {
        si[IRdScalars + 0] = (int)rd.D; si[IRdScalars + 1] = (int)(rd.D >> 32);
        si[IRdScalars + 2] = (int)rd.SD; si[IRdScalars + 3] = (int)(rd.SD >> 32);
        si[IRdScalars + 4] = (int)rd.H; si[IRdScalars + 5] = (int)(rd.H >> 32);
        si[IRdScalars + 6] = (int)rd.R; si[IRdScalars + 7] = (int)(rd.R >> 32);
        si[IRdScalars + 8] = (int)rd.Score; si[IRdScalars + 9] = (int)(rd.Score >> 32);
        si[IRdScalars + 10] = (int)rd.Nz;
    }

    private static long Long(ArrayView<int> si, int o) => ((long)si[o + 1] << 32) | (uint)si[o];

    private static WebPModeScore LoadScore(ArrayView<int> si)
    {
        var rd = new WebPModeScore();
        rd.D = Long(si, IRdScalars + 0);
        rd.SD = Long(si, IRdScalars + 2);
        rd.H = Long(si, IRdScalars + 4);
        rd.R = Long(si, IRdScalars + 6);
        rd.Score = Long(si, IRdScalars + 8);
        rd.Nz = (uint)si[IRdScalars + 10];
        return rd;
    }

    private static RowCtx MakeCtx(ArrayView<int> meta, ArrayView<int> si, ArrayView<int> mbRecs, int row, int x)
    {
        int mbW = meta[M.MbW];
        int predsW = 4 * mbW + 1;
        return new RowCtx
        {
            Y = row, X = x, MbW = mbW, MbH = meta[M.MbH], Width = meta[M.Width], Height = meta[M.Height], PredsW = predsW,
            OutBuf = si[IRowOut], Out2Buf = si[IRowOut2], ErrDiff = meta[M.ErrorDiffusion] != 0,
            PredsIdx = 1 + predsW + row * 4 * predsW + x * 4,
            YTopOff = x * 16,
            UVTopOff = mbW * 16 + x * 16,
            Seg = mbRecs[(row * mbW + x) * M.MbStride + M.MbSegment],
        };
    }

    private static void SaveOut(ArrayView<int> si, ref RowCtx c)
    {
        si[IRowOut] = c.OutBuf;
        si[IRowOut2] = c.Out2Buf;
    }

    // The macroblock row's byte / int scratch (each row its own since the wavefront: a diagonal runs one macroblock per row,
    // and a row's macroblocks still run left to right, so the row-carried state - left samples, left nz / derr, the
    // output-buffer swap - is exactly the serial walk's).
    private static ArrayView<byte> RowBytes(ArrayView<byte> work, ArrayView<int> meta, int row) =>
        work.SubView(meta[M.WorkScratchOff] + row * ByteScratchSize, ByteScratchSize);

    private static ArrayView<int> RowInts(ArrayView<int> state, ArrayView<int> meta, int row) =>
        state.SubView(meta[M.StScratchOff] + row * IntScratchSize, IntScratchSize);

    /// <summary>Lanes of the prepare stage's thread group.</summary>
    public const int PrepareLanes = 64;

    /// <summary>
    /// Stage 1 of macroblock (<paramref name="x"/>, <paramref name="row"/>): row start, import, probability refresh, I16 / UV
    /// predictions - explicitly grouped, one group of <see cref="PrepareLanes"/> lanes: lane 0 does the row start and the
    /// bookkeeping; the import, the probability update, the level-cost tables and the prediction pixels are independent
    /// items shared out over the lanes (barriers between the steps that read each other's results).
    /// </summary>
    public static void PrepareStage(
        ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        // an early return before barriers must be uniform for WGSL: the group-wide OR of the flag is
        if (Group.BarrierOr(meta[M.Error] != 0)) return;
        row += Grid.IdxX; x -= 2 * Grid.IdxX;   // the diagonal's macroblock of this group
        int lane = Group.IdxX;
        int mbW = meta[M.MbW];
        int width = meta[M.Width], height = meta[M.Height];
        int cw = (width + 1) >> 1;
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = RowBytes(work, meta, row);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var topDerr = state.SubView(meta[M.StTopDerrOff], mbW * 4);
        var si = RowInts(state, meta, row);
        int uOff = meta[M.YuvUOff], vOff = meta[M.YuvVOff];

        if ((x == 0) & (lane == 0))
        {
            if (row == 0)
            {
                // InitTop
                for (int i = 0; i < 2 * mbW * 16; ++i) yTop[i] = 127;
                for (int i = 0; i < mbW + 1; ++i) nzRow[i] = 0;
                for (int i = 0; i < mbW * 4; ++i) topDerr[i] = 0;
            }
            // VP8IteratorSetRow -> InitLeft
            byte corner = (byte)(row > 0 ? 129 : 127);
            sb[BLeft + YLeft - 1] = corner; sb[BLeft + ULeft - 1] = corner; sb[BLeft + VLeft - 1] = corner;
            for (int i = 0; i < 16; ++i) sb[BLeft + YLeft + i] = 129;
            for (int i = 0; i < 8; ++i) { sb[BLeft + ULeft + i] = 129; sb[BLeft + VLeft + i] = 129; }
            si[ILeftNz + 8] = 0;
            for (int i = 0; i < 4; ++i) si[ILeftDerr + i] = 0;
            si[IRowOut] = BOutA;
            si[IRowOut2] = BOutB;
            si[IRowTokens] = 0;
        }
        Group.Barrier();   // the row start (left / top samples, the output buffers) is visible
        var c = MakeCtx(meta, si, mbRecs, row, x);

        // ---- VP8IteratorImport (no boundary): the 16x16 luma and two 8x8 chroma samples, one item each
        int w = Min(width - x * 16, 16), h = Min(height - row * 16, 16);
        int uvW = (w + 1) >> 1, uvH = (h + 1) >> 1;
        for (int k = lane; k < 256 + 64; k += PrepareLanes)
        {
            if (k < 256)
            {
                int py = k >> 4, px = k & 15;
                sb[BIn + py * BPS + px] = yuv[(row * 16 + Min(py, h - 1)) * width + x * 16 + Min(px, w - 1)];
            }
            else
            {
                int py = (k - 256) >> 3, px = (k - 256) & 7;
                int s = (row * 8 + Min(py, uvH - 1)) * cw + x * 8 + Min(px, uvW - 1);
                sb[BIn + 16 + py * BPS + px] = yuv[uOff + s];
                sb[BIn + 24 + py * BPS + px] = yuv[vOff + s];
            }
        }

        // (the probability refresh, `if (--cnt < 0)` in VP8EncTokenLoop, is RefreshKernel: the host launches it before the
        // first macroblock of each probability window, whose bounds it computes from the same countdown)

        // ---- VP8Decimate (RD_OPT_BASIC): the I16 / UV predictions (VP8MakeLuma16Preds + VP8MakeChroma8Preds) + rd init
        PredictAll(lane, PrepareLanes, sb, yTop,
            x != 0 ? BLeft + YLeft : -1, row != 0 ? c.YTopOff : -1,
            x != 0 ? BLeft + ULeft : -1, row != 0 ? c.UVTopOff : -1,
            x != 0 ? BLeft + VLeft : -1, row != 0 ? c.UVTopOff + 8 : -1);
        if (lane == 0)
        {
            var rd = new WebPModeScore();
            rd.Init();
            SaveScore(si, ref rd);
        }
    }

    /// <summary>Stage 2: PickBestIntra16, explicitly grouped: one group of <see cref="I16Lanes"/> lanes (a lane per mode).</summary>
    public static void Intra16Stage(
        ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        // explicitly grouped: one group of I16Lanes lanes (a lane per mode); the error check before the barrier in
        // PickBestIntra16 is made uniform with a group-wide OR (WGSL)
        if (Group.BarrierOr(meta[M.Error] != 0)) return;
        row += Grid.IdxX; x -= 2 * Grid.IdxX;   // the diagonal's macroblock of this group
        int lane = Group.IdxX;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        var preds = work.SubView(0, (4 * mbW + 1) * (4 * mbH + 1));
        var sb = RowBytes(work, meta, row);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = RowInts(state, meta, row);
        var c = MakeCtx(meta, si, mbRecs, row, x);
        var rd = LoadScore(si);
        int modeI16 = -1;
        PickBestIntra16(lane, sb, si, meta, t, coeffs, lc, nzRow, preds, mbRecs, ref c, ref rd, ref modeI16);
        if (lane == 0)
        {
            SaveScore(si, ref rd);
            SaveOut(si, ref c);
        }
    }

    /// <summary>Stage 3: PickBestIntra4, explicitly grouped: one group of <see cref="I4Lanes"/> lanes (a lane per mode).</summary>
    public static void Intra4Stage(
        ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        // an early return before the barriers in PickBestIntra4 must be uniform for WGSL's validator: a value read from
        // a storage buffer is not, the group-wide OR of it is
        if (Group.BarrierOr(meta[M.Error] != 0)) return;
        row += Grid.IdxX; x -= 2 * Grid.IdxX;   // the diagonal's macroblock of this group
        int lane = Group.IdxX;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        var preds = work.SubView(0, (4 * mbW + 1) * (4 * mbH + 1));
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = RowBytes(work, meta, row);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = RowInts(state, meta, row);
        var c = MakeCtx(meta, si, mbRecs, row, x);
        var rd = LoadScore(si);
        PickBestIntra4(lane, sb, si, meta, t, coeffs, lc, nzRow, preds, yTop, mbRecs, ref c, ref rd);
        if (lane == 0)
        {
            SaveScore(si, ref rd);
            SaveOut(si, ref c);
        }
    }

    /// <summary>Stage 4: PickBestUV, explicitly grouped: one group of <see cref="UVLanes"/> lanes (a lane per mode).</summary>
    public static void UVStage(
        ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        // the error check before the barrier in PickBestUV is made uniform with a group-wide OR (WGSL)
        if (Group.BarrierOr(meta[M.Error] != 0)) return;
        row += Grid.IdxX; x -= 2 * Grid.IdxX;   // the diagonal's macroblock of this group
        int lane = Group.IdxX;
        int mbW = meta[M.MbW];
        var sb = RowBytes(work, meta, row);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var topDerr = state.SubView(meta[M.StTopDerrOff], mbW * 4);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = RowInts(state, meta, row);
        var c = MakeCtx(meta, si, mbRecs, row, x);
        var rd = LoadScore(si);
        PickBestUV(lane, sb, si, meta, t, coeffs, lc, nzRow, topDerr, mbRecs, ref c, ref rd);
        if (lane == 0)
        {
            SaveScore(si, ref rd);
            SaveOut(si, ref c);
        }
    }

    /// <summary>Stage 5: skip flag, RecordTokens, the partition-0 size, VP8IteratorSaveBoundary.</summary>
    public static void RecordStage(
        Index1D g, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        if (meta[M.Error] != 0) return;
        row += g; x -= 2 * g;   // the diagonal's macroblock of this thread
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = RowBytes(work, meta, row);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var si = RowInts(state, meta, row);
        var c = MakeCtx(meta, si, mbRecs, row, x);
        var rd = LoadScore(si);
        int rec = (row * mbW + x) * M.MbStride;
        mbRecs[rec + M.MbSkip] = rd.Nz == 0 ? 1 : 0;

        // ---- RecordTokens, into this row's token buffer (the window pass counts the statistics and appends the rows'
        // tokens to the stream in raster order); the partition-0 cost is summed there too
        int cap = meta[M.RowTokenCap];
        RecordTokens(si, nzRow, x, coeffs, t, meta, tokens.SubView((long)row * cap, cap), mbRecs[rec + M.MbType] == 1);
        mbRecs[rec + M.MbP0Bits] = (int)rd.H;

        // ---- VP8IteratorSaveBoundary
        int o = c.OutBuf;
        if (x < mbW - 1)
        {
            for (int i = 0; i < 16; ++i) sb[BLeft + YLeft + i] = sb[o + 15 + i * BPS];
            for (int i = 0; i < 8; ++i)
            {
                sb[BLeft + ULeft + i] = sb[o + 16 + 7 + i * BPS];
                sb[BLeft + VLeft + i] = sb[o + 16 + 15 + i * BPS];
            }
            sb[BLeft + YLeft - 1] = yTop[c.YTopOff + 15];
            sb[BLeft + ULeft - 1] = yTop[c.UVTopOff + 0 + 7];
            sb[BLeft + VLeft - 1] = yTop[c.UVTopOff + 8 + 7];
        }
        if (row < mbH - 1)
        {
            for (int i = 0; i < 16; ++i) yTop[c.YTopOff + i] = sb[o + 15 * BPS + i];
            for (int i = 0; i < 16; ++i) yTop[c.UVTopOff + i] = sb[o + 16 + 7 * BPS + i];
        }
    }
}
