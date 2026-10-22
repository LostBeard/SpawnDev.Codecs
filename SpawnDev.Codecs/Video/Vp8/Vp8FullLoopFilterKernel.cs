// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full-syntax VP8 keyframe decoder, kernel 5: the in-loop deblocking filter (RFC 6386 section 15, libvpx
// loopfilter.c), run after the whole frame is reconstructed. Bit-exact port of Vp8LoopFilter (the CPU reference).
//
// Schedule. The reference filters macroblocks in raster order, each macroblock reading pixels its left and top
// neighbours already filtered, and the top-edge filter of (r, c) reads bottom rows of (r-1, c) that the LEFT-edge filter
// of (r-1, c+1) modifies. So (r, c) must run after (r, c-1) and after (r-1, c+1) (and, transitively, after everything
// above). With the diagonal d = c + 2 * r both are on earlier diagonals, and two macroblocks on one diagonal touch
// disjoint pixels (their filter footprints are the macroblock plus 4 pixels left and 4 above, and diagonal neighbours are
// two columns apart). One launch per diagonal, one GPU thread per macroblock that does ALL of its edges (luma + chroma,
// left edge, inner vertical, top edge, inner horizontal). This is a correctness-first schedule: parallelism is the number
// of macroblocks on a diagonal, and each thread is serial over its ~100 edge-lines. Splitting the edges of one macroblock
// across threads is possible (the vertical-edge passes of different macroblocks in a row commute) but not done here.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp8;

internal static class Vp8FullLoopFilterKernel
{
    public static void Kernel(
        Index1D idx,
        ArrayView<byte> yP, ArrayView<byte> uP, ArrayView<byte> vP,
        ArrayView<byte> recs, ArrayView<int> meta,
        int diag, int rowLo)
    {
        if (meta[Vp8FullMeta.Error] != 0) return;
        int frameLevel = meta[Vp8FullMeta.FilterLevel];
        if (frameLevel == 0) return;             // frame level 0: no filtering at all, whatever the segments say

        int mbCols = meta[Vp8FullMeta.MbCols];
        int r = rowLo + idx.X;
        int c = diag - Vp8FullMeta.WaveRowWeight * r;
        long rec = ((long)r * mbCols + c) * Vp8MbRecord.Size;
        int yMode = recs[rec + Vp8MbRecord.YMode];
        int seg = recs[rec + Vp8MbRecord.Segment];
        int hasCoef = recs[rec + Vp8MbRecord.HasCoef];
        bool bpred = yMode == 4;

        int level = frameLevel;
        if (meta[Vp8FullMeta.SegEnabled] != 0)
        {
            int v = meta[Vp8FullMeta.SegLf + seg];
            level = meta[Vp8FullMeta.SegAbs] != 0 ? v : level + v;
            level = level < 0 ? 0 : (level > 63 ? 63 : level);
        }
        if (meta[Vp8FullMeta.LfDeltaEnabled] != 0)
        {
            level += meta[Vp8FullMeta.RefLfDelta];
            if (bpred) level += meta[Vp8FullMeta.ModeLfDelta];
            level = level < 0 ? 0 : (level > 63 ? 63 : level);
        }
        if (level == 0) return;

        int sharp = meta[Vp8FullMeta.Sharpness];
        int interior = level >> ((sharp > 0 ? 1 : 0) + (sharp > 4 ? 1 : 0));
        if (sharp > 0)
        {
            if (interior > 9 - sharp) interior = 9 - sharp;
        }
        if (interior < 1) interior = 1;
        int hev = level >= 40 ? 2 : (level >= 15 ? 1 : 0);
        int mbLimit = (level + 2) * 2 + interior;
        int subLimit = level * 2 + interior;
        bool inner = bpred | (hasCoef != 0);

        int ys = mbCols * 16;
        int uvs = mbCols * 8;
        int yOff = r * 16 * ys + c * 16;
        int uvOff = r * 8 * uvs + c * 8;

        if (meta[Vp8FullMeta.FilterType] == 1)
        {
            if (c > 0) SimpleEdge(yP, yOff, 1, ys, 16, mbLimit);
            if (inner)
                for (int x = 4; x < 16; x += 4) SimpleEdge(yP, yOff + x, 1, ys, 16, subLimit);
            if (r > 0) SimpleEdge(yP, yOff, ys, 1, 16, mbLimit);
            if (inner)
                for (int y = 4; y < 16; y += 4) SimpleEdge(yP, yOff + y * ys, ys, 1, 16, subLimit);
        }
        else
        {
            if (c > 0)
            {
                MbEdge(yP, yOff, 1, ys, 16, mbLimit, interior, hev);
                MbEdge(uP, uvOff, 1, uvs, 8, mbLimit, interior, hev);
                MbEdge(vP, uvOff, 1, uvs, 8, mbLimit, interior, hev);
            }
            if (inner)
            {
                for (int x = 4; x < 16; x += 4) SubEdge(yP, yOff + x, 1, ys, 16, subLimit, interior, hev);
                SubEdge(uP, uvOff + 4, 1, uvs, 8, subLimit, interior, hev);
                SubEdge(vP, uvOff + 4, 1, uvs, 8, subLimit, interior, hev);
            }
            if (r > 0)
            {
                MbEdge(yP, yOff, ys, 1, 16, mbLimit, interior, hev);
                MbEdge(uP, uvOff, uvs, 1, 8, mbLimit, interior, hev);
                MbEdge(vP, uvOff, uvs, 1, 8, mbLimit, interior, hev);
            }
            if (inner)
            {
                for (int y = 4; y < 16; y += 4) SubEdge(yP, yOff + y * ys, ys, 1, 16, subLimit, interior, hev);
                SubEdge(uP, uvOff + 4 * uvs, uvs, 1, 8, subLimit, interior, hev);
                SubEdge(vP, uvOff + 4 * uvs, uvs, 1, 8, subLimit, interior, hev);
            }
        }
    }

    private static int C(int v) => v < -128 ? -128 : (v > 127 ? 127 : v);   // signed char clamp
    private static int Abs(int v) => v < 0 ? -v : v;

    /// <summary>Simple filter along one edge; pos = first pixel on the q side, step = across the edge, line = along it.</summary>
    private static void SimpleEdge(ArrayView<byte> p, int pos, int step, int line, int count, int limit)
    {
        for (int i = 0; i < count; i++)
        {
            int q0i = pos + i * line;
            int p1 = p[q0i - 2 * step], p0 = p[q0i - step], q0 = p[q0i], q1 = p[q0i + step];
            bool pass = (Abs(p0 - q0) * 2 + (Abs(p1 - q1) >> 1)) <= limit;
            if (pass)
            {
                int ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128;
                int f = C(ps1 - qs1);
                f = C(f + 3 * (qs0 - ps0));
                int f1 = C(f + 4) >> 3;
                p[q0i] = (byte)(C(qs0 - f1) + 128);
                int f2 = C(f + 3) >> 3;
                p[q0i - step] = (byte)(C(ps0 + f2) + 128);
            }
        }
    }

    private static bool NormalMask(ArrayView<byte> p, int pos, int step, int limit, int blimit)
    {
        int p3 = p[pos - 4 * step], p2 = p[pos - 3 * step], p1 = p[pos - 2 * step], p0 = p[pos - step];
        int q0 = p[pos], q1 = p[pos + step], q2 = p[pos + 2 * step], q3 = p[pos + 3 * step];
        bool m = (Abs(p3 - p2) <= limit) & (Abs(p2 - p1) <= limit) & (Abs(p1 - p0) <= limit)
               & (Abs(q1 - q0) <= limit) & (Abs(q2 - q1) <= limit) & (Abs(q3 - q2) <= limit)
               & ((Abs(p0 - q0) * 2 + (Abs(p1 - q1) >> 1)) <= blimit);
        return m;
    }

    /// <summary>Normal filter, sub-block edge: modifies up to two pixels on each side.</summary>
    private static void SubEdge(ArrayView<byte> p, int pos, int step, int line, int count, int blimit, int limit, int hevThr)
    {
        for (int i = 0; i < count; i++)
        {
            int q0i = pos + i * line;
            if (NormalMask(p, q0i, step, limit, blimit))
            {
                int p1 = p[q0i - 2 * step], p0 = p[q0i - step], q0 = p[q0i], q1 = p[q0i + step];
                bool hev = (Abs(p1 - p0) > hevThr) | (Abs(q1 - q0) > hevThr);
                int ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128;
                int f = hev ? C(ps1 - qs1) : 0;
                f = C(f + 3 * (qs0 - ps0));
                int f1 = C(f + 4) >> 3;
                int f2 = C(f + 3) >> 3;
                p[q0i] = (byte)(C(qs0 - f1) + 128);
                p[q0i - step] = (byte)(C(ps0 + f2) + 128);
                if (!hev)
                {
                    int a = (f1 + 1) >> 1;
                    p[q0i + step] = (byte)(C(qs1 - a) + 128);
                    p[q0i - 2 * step] = (byte)(C(ps1 + a) + 128);
                }
            }
        }
    }

    /// <summary>Normal filter, macroblock edge: modifies up to three pixels on each side.</summary>
    private static void MbEdge(ArrayView<byte> p, int pos, int step, int line, int count, int blimit, int limit, int hevThr)
    {
        for (int i = 0; i < count; i++)
        {
            int q0i = pos + i * line;
            if (NormalMask(p, q0i, step, limit, blimit))
            {
                int p2 = p[q0i - 3 * step], p1 = p[q0i - 2 * step], p0 = p[q0i - step];
                int q0 = p[q0i], q1 = p[q0i + step], q2 = p[q0i + 2 * step];
                bool hev = (Abs(p1 - p0) > hevThr) | (Abs(q1 - q0) > hevThr);
                int ps2 = p2 - 128, ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128, qs2 = q2 - 128;
                int s = C(ps1 - qs1);
                s = C(s + 3 * (qs0 - ps0));
                if (hev)
                {
                    int f1 = C(s + 4) >> 3;
                    int f2 = C(s + 3) >> 3;
                    p[q0i] = (byte)(C(qs0 - f1) + 128);
                    p[q0i - step] = (byte)(C(ps0 + f2) + 128);
                }
                else
                {
                    int u = C((63 + s * 27) >> 7);
                    p[q0i] = (byte)(C(qs0 - u) + 128);
                    p[q0i - step] = (byte)(C(ps0 + u) + 128);
                    u = C((63 + s * 18) >> 7);
                    p[q0i + step] = (byte)(C(qs1 - u) + 128);
                    p[q0i - 2 * step] = (byte)(C(ps1 + u) + 128);
                    u = C((63 + s * 9) >> 7);
                    p[q0i + 2 * step] = (byte)(C(qs2 - u) + 128);
                    p[q0i - 3 * step] = (byte)(C(ps2 + u) + 128);
                }
            }
        }
    }
}
