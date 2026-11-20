// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The probability windows of the macroblock wavefront (see WebPLossyEncodeRowKernel.Stages.cs):
//
//   RefreshKernel      VP8EncTokenLoop's `if (--cnt < 0)`: FinalizeTokenProbas, then (when a probability changed)
//                      VP8CalculateLevelCosts. The host launches it before the first macroblock of every window but the first.
//   WindowKernel       after a window's last diagonal, a group per macroblock row of the window: the row's tokens appended
//                      to the stream at the row's raster-order offset, and counted per probability (count / ones, atomics).
//   WindowApplyKernel  a thread per probability: the window's counts added to the statistics. RecordStats halves a counter
//                      that reached 0xfffe0000 before an update, so ONLY a counter that can reach it inside the window
//                      depends on the order of its updates; such a counter (rare: 65534 updates of one probability) is
//                      replayed from the window's tokens in raster order instead. Thread 0 adds the window's partition-0
//                      cost to SizeP0 and the window's tokens to the token count (the next RefreshKernel empties
//                      the rows' buffers).

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static partial class WebPLossyEncodeRowKernel
{
    /// <summary>Lanes of a <see cref="WindowKernel"/> group (the ILGPU CPU backend allows 64).</summary>
    public const int WindowLanes = 64;

    /// <summary>The probability refresh between two windows; one group of <see cref="PrepareLanes"/> lanes.</summary>
    public static void RefreshKernel(ArrayView<int> state, ArrayView<int> meta, ArrayView<int> t)
    {
        if (Group.BarrierOr(meta[M.Error] != 0)) return;
        int lane = Group.IdxX;
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var stats = state.SubView(meta[M.StStatsOff], WebPLossySetupKernel.ProbaCount);
        var lc = state.SubView(meta[M.StLevelCostOff], WebPLossySetupKernel.LevelCostCount);
        bool changed = Group.BarrierOr(FinalizeTokenProbasLanes(lane, PrepareLanes, coeffs, stats, t));
        // (BarrierOr is a barrier: every lane's coefficients are written)
        if (changed) WebPLossySetupKernel.CalculateLevelCostsLanes(lane, PrepareLanes, coeffs, lc, t);
        if (lane == 0) meta[M.ProbaDirty] = 0;   // FinalizeTokenProbas sets it, the level costs consume it
        // the previous window's tokens are in the stream: every row's buffer starts empty for this window
        for (int r = lane; r < meta[M.MbH]; r += PrepareLanes) RowInts(state, meta, r)[IRowTokens] = 0;
    }

    private static int StatIndex(int v) => (v & 0x3fff) - ((v & StatBelowBit) != 0 ? 1 : 0);

    /// <summary>
    /// Window [<paramref name="first"/>, <paramref name="end"/>) (macroblocks, raster order), group g = row first / mbW + g:
    /// append the row's tokens to <paramref name="tokens"/> and count them per probability.
    /// </summary>
    public static void WindowKernel(ArrayView<int> mbRecs, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> rowTokens, ArrayView<int> tokens, int first, int end)
    {
        if (Group.BarrierOr(meta[M.Error] != 0)) return;
        int lane = Group.IdxX;
        int mbW = meta[M.MbW];
        int cap = meta[M.RowTokenCap];
        int y0 = first / mbW, y1 = (end - 1) / mbW, y = y0 + Grid.IdxX;
        var win = state.SubView(meta[M.StWinOff], 2 * WebPLossySetupKernel.ProbaCount);
        int baseN = meta[M.TokenCount];
        int before = 0, total = 0;
        for (int r = y0; r <= y1; ++r)
        {
            int n = RowInts(state, meta, r)[IRowTokens];
            if (r < y) before += n;
            total += n;
        }
        if (Group.BarrierOr(baseN + total > meta[M.TokenCapacity]))
        {
            if (lane == 0)
            {
                meta[M.Error] = (int)WebPLossyEncError.TokenBufferTooSmall;
                meta[M.TokenCount] = baseN + total;   // the host sizes the retry from it
            }
            return;
        }
        int count = RowInts(state, meta, y)[IRowTokens];
        long src = (long)y * cap;
        int dst = baseN + before;
        for (int i = lane; i < count; i += WindowLanes)
        {
            int v = rowTokens[src + i];
            tokens[dst + i] = v & ~StatBelowBit;
            if ((v & FixedProbaBit) == 0)
            {
                int idx = StatIndex(v);
                Atomic.Add(ref win[idx], 1);
                Atomic.Add(ref win[WebPLossySetupKernel.ProbaCount + idx], (v >> 15) & 1);
            }
        }
    }

    /// <summary>Window [<paramref name="first"/>, <paramref name="end"/>): thread i adds the window's statistics of probability i.</summary>
    public static void WindowApplyKernel(Index1D i, ArrayView<int> mbRecs, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> rowTokens, ArrayView<int> tokens, int first, int end)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW];
        int cap = meta[M.RowTokenCap];
        int y0 = first / mbW, y1 = (end - 1) / mbW;
        var stats = state.SubView(meta[M.StStatsOff], WebPLossySetupKernel.ProbaCount);
        var win = state.SubView(meta[M.StWinOff], 2 * WebPLossySetupKernel.ProbaCount);
        int n = win[i], ones = win[WebPLossySetupKernel.ProbaCount + i];
        uint p = (uint)stats[i];
        if (((p >> 16) + (uint)n <= 0xfffeu) & (meta[M.WinReplayAll] == 0))
        {
            // no update of this window finds the counter at 0xfffe0000 or above: no halving, the order does not matter
            stats[i] = (int)(p + ((uint)n << 16) + (uint)ones);
        }
        else
        {
            // the counter saturates inside the window: replay its updates in raster order
            for (int r = y0; r <= y1; ++r)
            {
                int count = RowInts(state, meta, r)[IRowTokens];
                long src = (long)r * cap;
                for (int k = 0; k < count; ++k)
                {
                    int v = rowTokens[src + k];
                    if (((v & FixedProbaBit) == 0) && StatIndex(v) == i) RecordStats((v >> 15) & 1, stats, i);
                }
            }
        }
        win[i] = 0;
        win[WebPLossySetupKernel.ProbaCount + i] = 0;
        if (i == 0)
        {
            int total = 0;
            for (int r = y0; r <= y1; ++r) total += RowInts(state, meta, r)[IRowTokens];
            long sizeP0 = ((long)meta[M.SizeP0Hi] << 32) | (uint)meta[M.SizeP0Lo];
            for (int k = first; k < end; ++k) sizeP0 += mbRecs[k * M.MbStride + M.MbP0Bits];
            meta[M.SizeP0Lo] = (int)sizeP0;
            meta[M.SizeP0Hi] = (int)(sizeP0 >> 32);
            meta[M.TokenCount] = meta[M.TokenCount] + total;
        }
    }
}
