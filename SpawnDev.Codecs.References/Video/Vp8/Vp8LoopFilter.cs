// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 in-loop deblocking filter, CPU reference (RFC 6386 section 15, libvpx vp8/common/loopfilter*.c).
// Runs over the WHOLE reconstructed key frame after every macroblock has been predicted + reconstructed: intra prediction
// reads UNFILTERED pixels, the filter then rewrites them in macroblock raster order, and each macroblock reads pixels
// its left and top neighbours already filtered (so the order below is part of the format, not an implementation choice).
//
// Per macroblock, in this order (libvpx vp8_loop_filter_frame):
//   1. left macroblock edge   (only when mbCol > 0)            limit = ((level + 2) * 2 + interior)
//   2. inner vertical edges   (x = 4, 8, 12; chroma x = 4)     limit = (level * 2 + interior)   (skipped when no inner filtering)
//   3. top macroblock edge    (only when mbRow > 0)
//   4. inner horizontal edges (y = 4, 8, 12; chroma y = 4)     (skipped when no inner filtering)
// The simple filter (header filter_type = 1) touches luma only and uses the same edge limits.
//
// Per macroblock level (key frame: ref frame = intra, so only the B_PRED mode delta can apply):
//   level = frame level, or the segment's (absolute, or frame level + delta, clamped 0..63)
//   if mode_ref_lf_delta_enabled: level += ref_lf_delta[intra]; if the MB is B_PRED: level += mode_lf_delta[0]; clamp 0..63
//   interior limit = level >> ((sharpness > 0) + (sharpness > 4)); with sharpness > 0 capped at 9 - sharpness; at least 1
//   high-edge-variance threshold (key frames): level >= 40 -> 2, level >= 15 -> 1, else 0
//   level 0 = macroblock not filtered; frame filter_level 0 = frame not filtered AT ALL (even if a segment has a level).
//   Inner edges are filtered unless the MB is not B_PRED and has no coefficients (skip flag set, or every block's eob is 0).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 loop filter, CPU reference.</summary>
public static class Vp8LoopFilter
{
    /// <summary>The loop filter level of one macroblock (0 = leave it alone).</summary>
    public static int MbLevel(Vp8FrameHeader header, int segmentId, bool isBPred)
    {
        var lf = header.LoopFilter;
        int level = Vp8SegmentationLookup.ResolveLoopFilterLevel(segmentId, lf.FilterLevel, header.Segmentation);
        if (lf.ModeRefLfDeltaEnabled)
        {
            level += lf.RefLfDeltas[0];
            if (isBPred) level += lf.ModeLfDeltas[0];
            level = level < 0 ? 0 : level > 63 ? 63 : level;
        }
        return level;
    }

    /// <summary>Interior limit for a level and sharpness.</summary>
    public static int InteriorLimit(int level, int sharpness)
    {
        int limit = level >> ((sharpness > 0 ? 1 : 0) + (sharpness > 4 ? 1 : 0));
        if (sharpness > 0 && limit > 9 - sharpness) limit = 9 - sharpness;
        return limit < 1 ? 1 : limit;
    }

    /// <summary>High edge variance threshold for key frames.</summary>
    public static int HevThreshold(int level) => level >= 40 ? 2 : level >= 15 ? 1 : 0;

    /// <summary>
    /// Filter the whole reconstructed frame in place.
    /// </summary>
    /// <param name="fb">The reconstructed (unfiltered) frame.</param>
    /// <param name="header">Frame header (filter type / level / sharpness / deltas / segmentation).</param>
    /// <param name="mbSegment">Segment id per macroblock (raster order).</param>
    /// <param name="mbIsBPred">True per macroblock that is B_PRED.</param>
    /// <param name="mbInnerFilter">True per macroblock whose inner (sub-block) edges are filtered.</param>
    public static void Apply(Vp8FrameBuffer fb, Vp8FrameHeader header, int[] mbSegment, bool[] mbIsBPred, bool[] mbInnerFilter)
    {
        var lf = header.LoopFilter;
        if (lf.FilterLevel == 0) return;
        bool simple = lf.FilterType == 1;
        int mbCols = fb.MbCols, mbRows = fb.MbRows;
        for (int mbRow = 0; mbRow < mbRows; mbRow++)
        {
            for (int mbCol = 0; mbCol < mbCols; mbCol++)
            {
                int idx = mbRow * mbCols + mbCol;
                int level = MbLevel(header, mbSegment[idx], mbIsBPred[idx]);
                if (level == 0) continue;
                int interior = InteriorLimit(level, lf.SharpnessLevel);
                int mbLimit = (level + 2) * 2 + interior;
                int subLimit = level * 2 + interior;
                int hev = HevThreshold(level);
                bool inner = mbInnerFilter[idx];

                int yOff = mbRow * 16 * fb.YStride + mbCol * 16;
                int uvOff = mbRow * 8 * fb.UvStride + mbCol * 8;
                int ys = fb.YStride, uvs = fb.UvStride;

                if (simple)
                {
                    if (mbCol > 0) SimpleEdge(fb.YPlane, yOff, 1, ys, 16, mbLimit);
                    if (inner)
                        for (int x = 4; x < 16; x += 4) SimpleEdge(fb.YPlane, yOff + x, 1, ys, 16, subLimit);
                    if (mbRow > 0) SimpleEdge(fb.YPlane, yOff, ys, 1, 16, mbLimit);
                    if (inner)
                        for (int y = 4; y < 16; y += 4) SimpleEdge(fb.YPlane, yOff + y * ys, ys, 1, 16, subLimit);
                    continue;
                }

                if (mbCol > 0)
                {
                    MbEdge(fb.YPlane, yOff, 1, ys, 16, mbLimit, interior, hev);
                    MbEdge(fb.UPlane, uvOff, 1, uvs, 8, mbLimit, interior, hev);
                    MbEdge(fb.VPlane, uvOff, 1, uvs, 8, mbLimit, interior, hev);
                }
                if (inner)
                {
                    for (int x = 4; x < 16; x += 4) SubEdge(fb.YPlane, yOff + x, 1, ys, 16, subLimit, interior, hev);
                    SubEdge(fb.UPlane, uvOff + 4, 1, uvs, 8, subLimit, interior, hev);
                    SubEdge(fb.VPlane, uvOff + 4, 1, uvs, 8, subLimit, interior, hev);
                }
                if (mbRow > 0)
                {
                    MbEdge(fb.YPlane, yOff, ys, 1, 16, mbLimit, interior, hev);
                    MbEdge(fb.UPlane, uvOff, uvs, 1, 8, mbLimit, interior, hev);
                    MbEdge(fb.VPlane, uvOff, uvs, 1, 8, mbLimit, interior, hev);
                }
                if (inner)
                {
                    for (int y = 4; y < 16; y += 4) SubEdge(fb.YPlane, yOff + y * ys, ys, 1, 16, subLimit, interior, hev);
                    SubEdge(fb.UPlane, uvOff + 4 * uvs, uvs, 1, 8, subLimit, interior, hev);
                    SubEdge(fb.VPlane, uvOff + 4 * uvs, uvs, 1, 8, subLimit, interior, hev);
                }
            }
        }
    }

    private static int C(int v) => v < -128 ? -128 : v > 127 ? 127 : v;   // signed char clamp
    private static int Abs(int v) => v < 0 ? -v : v;

    /// <summary>Simple filter across one edge: pos = first pixel on the q side, step = across the edge, line = along it.</summary>
    private static void SimpleEdge(byte[] p, int pos, int step, int line, int count, int limit)
    {
        for (int i = 0; i < count; i++, pos += line)
        {
            int p1 = p[pos - 2 * step], p0 = p[pos - step], q0 = p[pos], q1 = p[pos + step];
            if (Abs(p0 - q0) * 2 + (Abs(p1 - q1) >> 1) > limit) continue;
            int ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128;
            int f = C(ps1 - qs1);
            f = C(f + 3 * (qs0 - ps0));
            int f1 = C(f + 4) >> 3;
            p[pos] = (byte)(C(qs0 - f1) + 128);
            int f2 = C(f + 3) >> 3;
            p[pos - step] = (byte)(C(ps0 + f2) + 128);
        }
    }

    private static bool NormalMask(byte[] p, int pos, int step, int limit, int blimit)
    {
        int p3 = p[pos - 4 * step], p2 = p[pos - 3 * step], p1 = p[pos - 2 * step], p0 = p[pos - step];
        int q0 = p[pos], q1 = p[pos + step], q2 = p[pos + 2 * step], q3 = p[pos + 3 * step];
        return Abs(p3 - p2) <= limit && Abs(p2 - p1) <= limit && Abs(p1 - p0) <= limit
            && Abs(q1 - q0) <= limit && Abs(q2 - q1) <= limit && Abs(q3 - q2) <= limit
            && Abs(p0 - q0) * 2 + (Abs(p1 - q1) >> 1) <= blimit;
    }

    /// <summary>Normal filter, sub-block (inner) edge: modifies up to two pixels each side.</summary>
    private static void SubEdge(byte[] p, int pos, int step, int line, int count, int blimit, int limit, int hevThr)
    {
        for (int i = 0; i < count; i++, pos += line)
        {
            if (!NormalMask(p, pos, step, limit, blimit)) continue;
            int p1 = p[pos - 2 * step], p0 = p[pos - step], q0 = p[pos], q1 = p[pos + step];
            bool hev = Abs(p1 - p0) > hevThr || Abs(q1 - q0) > hevThr;
            int ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128;
            int f = hev ? C(ps1 - qs1) : 0;
            f = C(f + 3 * (qs0 - ps0));
            int f1 = C(f + 4) >> 3;
            int f2 = C(f + 3) >> 3;
            p[pos] = (byte)(C(qs0 - f1) + 128);
            p[pos - step] = (byte)(C(ps0 + f2) + 128);
            if (!hev)
            {
                int a = (f1 + 1) >> 1;
                p[pos + step] = (byte)(C(qs1 - a) + 128);
                p[pos - 2 * step] = (byte)(C(ps1 + a) + 128);
            }
        }
    }

    /// <summary>Normal filter, macroblock edge: modifies up to three pixels each side.</summary>
    private static void MbEdge(byte[] p, int pos, int step, int line, int count, int blimit, int limit, int hevThr)
    {
        for (int i = 0; i < count; i++, pos += line)
        {
            if (!NormalMask(p, pos, step, limit, blimit)) continue;
            int p2 = p[pos - 3 * step], p1 = p[pos - 2 * step], p0 = p[pos - step];
            int q0 = p[pos], q1 = p[pos + step], q2 = p[pos + 2 * step];
            bool hev = Abs(p1 - p0) > hevThr || Abs(q1 - q0) > hevThr;
            int ps2 = p2 - 128, ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128, qs2 = q2 - 128;
            int s = C(ps1 - qs1);
            s = C(s + 3 * (qs0 - ps0));
            if (hev)
            {
                int f1 = C(s + 4) >> 3;
                int f2 = C(s + 3) >> 3;
                p[pos] = (byte)(C(qs0 - f1) + 128);
                p[pos - step] = (byte)(C(ps0 + f2) + 128);
            }
            else
            {
                int u = C((63 + s * 27) >> 7);
                p[pos] = (byte)(C(qs0 - u) + 128);
                p[pos - step] = (byte)(C(ps0 + u) + 128);
                u = C((63 + s * 18) >> 7);
                p[pos + step] = (byte)(C(qs1 - u) + 128);
                p[pos - 2 * step] = (byte)(C(ps1 + u) + 128);
                u = C((63 + s * 9) >> 7);
                p[pos + 2 * step] = (byte)(C(qs2 - u) + 128);
                p[pos - 3 * step] = (byte)(C(ps2 + u) + 128);
            }
        }
    }
}
