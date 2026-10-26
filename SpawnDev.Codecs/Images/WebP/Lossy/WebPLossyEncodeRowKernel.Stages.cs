// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernel 3 as STAGE kernels: one macroblock's VP8EncTokenLoop body split at libwebp's own seams, each stage its own
// kernel (one thread), launched in order for every macroblock. Same statements, same order, same results as the single
// EncodeRowKernel walk; the split exists because that kernel's WGSL was ~61k lines (Chrome's shader compiler inlines
// everything and took minutes / killed the GPU process on kernels of that size). Every stage stays well under the browser
// budget (Plans/PLAN-Oversized-Kernels.md).
//
//   PrepareStage   InitTop / InitLeft at a row start, VP8IteratorImport, the probability refresh, the I16 / UV predictions
//   Intra16Stage   PickBestIntra16
//   Intra4Stage    PickBestIntra4
//   UVStage        PickBestUV
//   RecordStage    skip flag, RecordTokens, the partition-0 size, VP8IteratorSaveBoundary
//
// What crosses stages lives on the device: the ModeScore scalars in the int scratch (IRdScalars), the output-buffer
// swap in meta[RowOutBuf / RowOut2Buf], and everything that already crossed macroblocks (see EncodeRowKernel).

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

    private static RowCtx MakeCtx(ArrayView<int> meta, ArrayView<int> mbRecs, int row, int x)
    {
        int mbW = meta[M.MbW];
        int predsW = 4 * mbW + 1;
        return new RowCtx
        {
            Y = row, X = x, MbW = mbW, MbH = meta[M.MbH], Width = meta[M.Width], Height = meta[M.Height], PredsW = predsW,
            OutBuf = meta[M.RowOutBuf], Out2Buf = meta[M.RowOut2Buf], ErrDiff = meta[M.ErrorDiffusion] != 0,
            PredsIdx = 1 + predsW + row * 4 * predsW + x * 4,
            YTopOff = x * 16,
            UVTopOff = mbW * 16 + x * 16,
            Seg = mbRecs[(row * mbW + x) * M.MbStride + M.MbSegment],
        };
    }

    private static void SaveOut(ArrayView<int> meta, ref RowCtx c)
    {
        meta[M.RowOutBuf] = c.OutBuf;
        meta[M.RowOut2Buf] = c.Out2Buf;
    }

    /// <summary>Stage 1 of macroblock (<paramref name="x"/>, <paramref name="row"/>): row start, import, probability refresh, I16 / UV predictions.</summary>
    public static void PrepareStage(
        Index1D _, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW];
        int width = meta[M.Width], height = meta[M.Height];
        int cw = (width + 1) >> 1;
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = work.SubView(meta[M.WorkScratchOff], ByteScratchSize);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var topDerr = state.SubView(meta[M.StTopDerrOff], mbW * 4);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var stats = state.SubView(meta[M.StStatsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = state.SubView(meta[M.StScratchOff], IntScratchSize);
        int uOff = meta[M.YuvUOff], vOff = meta[M.YuvVOff];

        if (x == 0)
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
            meta[M.RowOutBuf] = BOutA;
            meta[M.RowOut2Buf] = BOutB;
        }
        var c = MakeCtx(meta, mbRecs, row, x);

        // ---- VP8IteratorImport (no boundary)
        int w = Min(width - x * 16, 16), h = Min(height - row * 16, 16);
        int uvW = (w + 1) >> 1, uvH = (h + 1) >> 1;
        for (int py = 0; py < 16; ++py)
            for (int px = 0; px < 16; ++px)
                sb[BIn + py * BPS + px] = yuv[(row * 16 + Min(py, h - 1)) * width + x * 16 + Min(px, w - 1)];
        for (int py = 0; py < 8; ++py)
            for (int px = 0; px < 8; ++px)
            {
                int s = (row * 8 + Min(py, uvH - 1)) * cw + x * 8 + Min(px, uvW - 1);
                sb[BIn + 16 + py * BPS + px] = yuv[uOff + s];
                sb[BIn + 24 + py * BPS + px] = yuv[vOff + s];
            }

        // ---- probability refresh (`if (--cnt < 0)`)
        int cnt = meta[M.MbCountDown] - 1;
        if (cnt < 0)
        {
            FinalizeTokenProbas(coeffs, stats, t, meta);
            if (meta[M.ProbaDirty] != 0)
            {
                WebPLossySetupKernel.CalculateLevelCosts(coeffs, lc, t);
                meta[M.ProbaDirty] = 0;
            }
            cnt = meta[M.MaxCount];
        }
        meta[M.MbCountDown] = cnt;

        // ---- VP8Decimate (RD_OPT_BASIC): the I16 / UV predictions (VP8MakeLuma16Preds + VP8MakeChroma8Preds) + rd init
        PredictAll(sb, yTop,
            x != 0 ? BLeft + YLeft : -1, row != 0 ? c.YTopOff : -1,
            x != 0 ? BLeft + ULeft : -1, row != 0 ? c.UVTopOff : -1,
            x != 0 ? BLeft + VLeft : -1, row != 0 ? c.UVTopOff + 8 : -1);
        var rd = new WebPModeScore();
        rd.Init();
        SaveScore(si, ref rd);
    }

    /// <summary>Stage 2: PickBestIntra16.</summary>
    public static void Intra16Stage(
        Index1D _, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        var preds = work.SubView(0, (4 * mbW + 1) * (4 * mbH + 1));
        var sb = work.SubView(meta[M.WorkScratchOff], ByteScratchSize);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = state.SubView(meta[M.StScratchOff], IntScratchSize);
        var c = MakeCtx(meta, mbRecs, row, x);
        var rd = LoadScore(si);
        int modeI16 = -1;
        PickBestIntra16(sb, si, meta, t, coeffs, lc, nzRow, preds, mbRecs, ref c, ref rd, ref modeI16);
        SaveScore(si, ref rd);
        SaveOut(meta, ref c);
    }

    /// <summary>Stage 3: PickBestIntra4.</summary>
    public static void Intra4Stage(
        Index1D _, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        var preds = work.SubView(0, (4 * mbW + 1) * (4 * mbH + 1));
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = work.SubView(meta[M.WorkScratchOff], ByteScratchSize);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = state.SubView(meta[M.StScratchOff], IntScratchSize);
        var c = MakeCtx(meta, mbRecs, row, x);
        var rd = LoadScore(si);
        PickBestIntra4(sb, si, meta, t, coeffs, lc, nzRow, preds, yTop, mbRecs, ref c, ref rd);
        SaveScore(si, ref rd);
        SaveOut(meta, ref c);
    }

    /// <summary>Stage 4: PickBestUV.</summary>
    public static void UVStage(
        Index1D _, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW];
        var sb = work.SubView(meta[M.WorkScratchOff], ByteScratchSize);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var topDerr = state.SubView(meta[M.StTopDerrOff], mbW * 4);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        var si = state.SubView(meta[M.StScratchOff], IntScratchSize);
        var c = MakeCtx(meta, mbRecs, row, x);
        var rd = LoadScore(si);
        PickBestUV(sb, si, meta, t, coeffs, lc, nzRow, topDerr, mbRecs, ref c, ref rd);
        SaveScore(si, ref rd);
        SaveOut(meta, ref c);
    }

    /// <summary>Stage 5: skip flag, RecordTokens, the partition-0 size, VP8IteratorSaveBoundary.</summary>
    public static void RecordStage(
        Index1D _, ArrayView<byte> yuv, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state,
        ArrayView<int> meta, ArrayView<int> t, ArrayView<int> tokens, int row, int x)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        var yTop = work.SubView(meta[M.WorkYTopOff], 2 * mbW * 16);
        var sb = work.SubView(meta[M.WorkScratchOff], ByteScratchSize);
        var nzRow = state.SubView(meta[M.StNzOff], mbW + 1);
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var stats = state.SubView(meta[M.StStatsOff], WebPLossySetupKernel.ProbaCount);
        var si = state.SubView(meta[M.StScratchOff], IntScratchSize);
        var c = MakeCtx(meta, mbRecs, row, x);
        var rd = LoadScore(si);
        int rec = (row * mbW + x) * M.MbStride;
        mbRecs[rec + M.MbSkip] = rd.Nz == 0 ? 1 : 0;

        // ---- RecordTokens
        RecordTokens(si, nzRow, x, coeffs, stats, t, meta, tokens, mbRecs[rec + M.MbType] == 1);
        long sizeP0 = ((long)meta[M.SizeP0Hi] << 32) | (uint)meta[M.SizeP0Lo];
        sizeP0 += rd.H;
        meta[M.SizeP0Lo] = (int)sizeP0;
        meta[M.SizeP0Hi] = (int)(sizeP0 >> 32);

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
