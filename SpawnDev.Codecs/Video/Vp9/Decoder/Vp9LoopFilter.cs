// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 loop filter (8-bit), kernel-callable. A port of libvpx v1.17.0 (BSD-3-Clause):
//   vpx_dsp/loopfilter.c          filter_mask / flat_mask4 / flat_mask5 / hev_mask / signed_char_clamp / filter4 /
//                                 filter8 / filter16, vpx_lpf_{horizontal,vertical}_{4,8}[_dual]_c,
//                                 vpx_lpf_horizontal_16[_dual]_c (mb_lpf_horizontal_edge_w),
//                                 vpx_lpf_vertical_16[_dual]_c (mb_lpf_vertical_edge_w)
//   vp9/common/vp9_loopfilter.c   update_sharpness, vp9_loop_filter_init (hev thresholds), vp9_loop_filter_frame_init,
//                                 get_filter_level, vp9_build_mask (the decoder's per-block mask build),
//                                 vp9_adjust_mask, filter_selectively_vert_row2, filter_selectively_horiz,
//                                 filter_selectively_vert, vp9_filter_block_plane_ss00 / _ss11 / _non420, and the
//                                 inner loop of loop_filter_rows (one superblock)
//   vp9/common/vp9_common_data.c  uv_txsize_lookup, num_{4x4,8x8}_blocks_{wide,high}_lookup
//
// The decoder's masks: libvpx's decoder calls vp9_build_mask for every block it decodes (decode_block, when the frame
// filter level is not 0) and loop_filter_rows then runs vp9_adjust_mask on each superblock before filtering it.
// BuildSuperblockMask does exactly that from the mode info grid: every mi position inside the frame that is the
// top-left of its block (the grid points at the record stored at that very position) is a block the decoder decoded
// (decode_partition skips blocks whose top-left lies outside the frame). vp9_build_mask only ORs bits into the masks
// and writes each block's own lfl_y bytes, so the decode order does not matter.
//
// Order between superblocks: a superblock's filters read and write only its own pixels plus the 8 columns left of it
// (its rows) and the 8 rows above it (its columns). loop_filter_rows runs superblocks in raster order, so superblock
// (r, c) must run after (r, c - 1), after (r - 1, c) and after (r - 1, c + 1) (whose left edge filter writes the last
// columns of (r - 1, c), which (r, c)'s top edge filter reads), and before (r, c + 1) and (r + 1, c).
//
// Inside a superblock (FilterSuperblockGroup): one thread group, one lane per pixel row for the vertical edges, a
// group barrier, one lane per pixel column for the horizontal edges. Each lane replays libvpx's walk over the edges and
// filters only its own pixel of each edge, so every row and column sees libvpx's operations in libvpx's order (see
// "filtering" below).
//
// Mode info (the decoder's Vp9Mi layout): records of Vp9Mi.Stride ints at record index mi_row * mi_cols + mi_col of the
// block's top-left, fields SbType, Mode (libvpx mi->mode: for sub8x8 inter blocks the last sub-block's mode), TxSize,
// Skip (the value after decode_block's "no coefficients -> skip = 1" for inter blocks >= 8x8), SegmentId, Ref0
// (0 intra, 1 last, 2 golden, 3 altref); the grid (one int per mi position, at gridOffset + mi_row * mi_cols +
// mi_col) holds the record index of the block covering the position.
// Frame table (Vp9LfTable, ints): lf_info.lvl[seg][ref][mode], lf_info.lfthr[lvl] mblim / lim / hev_thr and the frame
// filter level; FrameInit builds it from the frame header (Vp9LfHdr) on the accelerator.

using System.Runtime.CompilerServices;
using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>
/// Loop filter inputs of one frame header (ints), FrameInit's input: filter_level, sharpness_level,
/// mode_ref_delta_enabled, ref_deltas[4] (intra, last, golden, altref), mode_deltas[2], segmentation enabled,
/// abs_delta (1 = SEGMENT_ABSDATA), per segment whether its SEG_LVL_ALT_LF feature bit is set, and its data.
/// </summary>
public static class Vp9LfHdr
{
    public const int Level = 0, Sharpness = 1, DeltaEnabled = 2, RefDeltas = 3, ModeDeltas = 7, SegEnabled = 9,
        SegAbsDelta = 10, SegAltLfEnabled = 11, SegAltLfData = 19;
    public const int Ints = 27;

    /// <summary>Packs a parsed header's loop filter and segmentation parameters into <paramref name="dst"/>.</summary>
    public static void Pack(Span<int> dst, Vp9LoopFilterInfo lf, Vp9SegmentationInfo seg)
    {
        dst[Level] = lf.Level;
        dst[Sharpness] = lf.Sharpness;
        dst[DeltaEnabled] = lf.DeltaEnabled ? 1 : 0;
        for (int i = 0; i < 4; i++) dst[RefDeltas + i] = lf.RefDeltas[i];
        for (int i = 0; i < 2; i++) dst[ModeDeltas + i] = lf.ModeDeltas[i];
        dst[SegEnabled] = seg.Enabled ? 1 : 0;
        dst[SegAbsDelta] = seg.AbsDelta ? 1 : 0;
        for (int s = 0; s < 8; s++)
        {
            dst[SegAltLfEnabled + s] = (seg.FeatureMask[s] >> 1) & 1;   // SEG_LVL_ALT_LF = 1
            dst[SegAltLfData + s] = seg.FeatureData[s, 1];
        }
    }
}

/// <summary>
/// The per-frame loop filter table (ints): <see cref="Lvl"/> + seg * 8 + ref * 2 + mode = lf_info.lvl[seg][ref][mode];
/// <see cref="Thr"/> + lvl * 3 + 0 / 1 / 2 = lfthr[lvl].mblim / lim / hev_thr; <see cref="FrameLevel"/> = the frame's
/// filter_level (0: libvpx does not filter the frame at all).
/// </summary>
public static class Vp9LfTable
{
    public const int Lvl = 0, Thr = 64, FrameLevel = 256;
    public const int Ints = 257;
}

/// <summary>libvpx LOOP_FILTER_MASK: the bit masks of one 64x64 superblock; lfl_y row r is byte c of <c>LflR</c>.</summary>
public struct Vp9LfMask
{
    public ulong LeftY0, LeftY1, LeftY2, LeftY3;
    public ulong AboveY0, AboveY1, AboveY2, AboveY3;
    public ulong Int4x4Y;
    // uint16_t in libvpx: every store keeps the low 16 bits
    public uint LeftUv0, LeftUv1, LeftUv2, LeftUv3;
    public uint AboveUv0, AboveUv1, AboveUv2, AboveUv3;
    public uint Int4x4Uv;
    public ulong Lfl0, Lfl1, Lfl2, Lfl3, Lfl4, Lfl5, Lfl6, Lfl7;

    public static ulong LeftY(ref Vp9LfMask m, int tx) => tx switch { 0 => m.LeftY0, 1 => m.LeftY1, 2 => m.LeftY2, _ => m.LeftY3 };
    public static ulong AboveY(ref Vp9LfMask m, int tx) => tx switch { 0 => m.AboveY0, 1 => m.AboveY1, 2 => m.AboveY2, _ => m.AboveY3 };
    public static uint LeftUv(ref Vp9LfMask m, int tx) => tx switch { 0 => m.LeftUv0, 1 => m.LeftUv1, 2 => m.LeftUv2, _ => m.LeftUv3 };
    public static uint AboveUv(ref Vp9LfMask m, int tx) => tx switch { 0 => m.AboveUv0, 1 => m.AboveUv1, 2 => m.AboveUv2, _ => m.AboveUv3 };
    public static ulong Lfl(ref Vp9LfMask m, int r) => r switch
    {
        0 => m.Lfl0, 1 => m.Lfl1, 2 => m.Lfl2, 3 => m.Lfl3, 4 => m.Lfl4, 5 => m.Lfl5, 6 => m.Lfl6, _ => m.Lfl7,
    };

    public static void OrLeftY(ref Vp9LfMask m, int tx, ulong v)
    {
        if (tx == 0) m.LeftY0 |= v; else if (tx == 1) m.LeftY1 |= v; else if (tx == 2) m.LeftY2 |= v; else m.LeftY3 |= v;
    }

    public static void OrAboveY(ref Vp9LfMask m, int tx, ulong v)
    {
        if (tx == 0) m.AboveY0 |= v; else if (tx == 1) m.AboveY1 |= v; else if (tx == 2) m.AboveY2 |= v; else m.AboveY3 |= v;
    }

    public static void OrLeftUv(ref Vp9LfMask m, int tx, uint v)
    {
        v &= 0xffff;
        if (tx == 0) m.LeftUv0 |= v; else if (tx == 1) m.LeftUv1 |= v; else if (tx == 2) m.LeftUv2 |= v; else m.LeftUv3 |= v;
    }

    public static void OrAboveUv(ref Vp9LfMask m, int tx, uint v)
    {
        v &= 0xffff;
        if (tx == 0) m.AboveUv0 |= v; else if (tx == 1) m.AboveUv1 |= v; else if (tx == 2) m.AboveUv2 |= v; else m.AboveUv3 |= v;
    }

    /// <summary>memset(&amp;lfl_y[r * 8 + c], level, w).</summary>
    public static void SetLfl(ref Vp9LfMask m, int r, int c, int w, int level)
    {
        ulong bytes = (w >= 8 ? ulong.MaxValue : (1UL << (8 * w)) - 1) << (8 * c);
        ulong fill = (ulong)level * 0x0101010101010101UL;
        switch (r)
        {
            case 0: m.Lfl0 = (m.Lfl0 & ~bytes) | (fill & bytes); break;
            case 1: m.Lfl1 = (m.Lfl1 & ~bytes) | (fill & bytes); break;
            case 2: m.Lfl2 = (m.Lfl2 & ~bytes) | (fill & bytes); break;
            case 3: m.Lfl3 = (m.Lfl3 & ~bytes) | (fill & bytes); break;
            case 4: m.Lfl4 = (m.Lfl4 & ~bytes) | (fill & bytes); break;
            case 5: m.Lfl5 = (m.Lfl5 & ~bytes) | (fill & bytes); break;
            case 6: m.Lfl6 = (m.Lfl6 & ~bytes) | (fill & bytes); break;
            default: m.Lfl7 = (m.Lfl7 & ~bytes) | (fill & bytes); break;
        }
    }
}

/// <summary>
/// The masks of libvpx vp9_filter_block_plane_non420 (chroma of 4:2:2 / 4:4:0) for one superblock, byte r of each = mi
/// row r of the superblock: <c>Vert*</c> = the row's mask_16x16_c / mask_8x8_c / mask_4x4_c (vertical edges, before the
/// image's first column is masked off), <c>Horz*</c> / <c>Int4</c> = mask_16x16[r] / mask_8x8[r] / mask_4x4[r] /
/// mask_4x4_int[r]; lfl row r is <c>LflR</c> (byte c &gt;&gt; ss_x).
/// </summary>
public struct Vp9LfNon420Mask
{
    public ulong Vert16, Vert8, Vert4, Horz16, Horz8, Horz4, Int4;
    public ulong Lfl0, Lfl1, Lfl2, Lfl3, Lfl4, Lfl5, Lfl6, Lfl7;

    public static ulong Lfl(ref Vp9LfNon420Mask m, int r) => r switch
    {
        0 => m.Lfl0, 1 => m.Lfl1, 2 => m.Lfl2, 3 => m.Lfl3, 4 => m.Lfl4, 5 => m.Lfl5, 6 => m.Lfl6, _ => m.Lfl7,
    };

    public static void SetLfl(ref Vp9LfNon420Mask m, int r, ulong v)
    {
        switch (r)
        {
            case 0: m.Lfl0 = v; break;
            case 1: m.Lfl1 = v; break;
            case 2: m.Lfl2 = v; break;
            case 3: m.Lfl3 = v; break;
            case 4: m.Lfl4 = v; break;
            case 5: m.Lfl5 = v; break;
            case 6: m.Lfl6 = v; break;
            default: m.Lfl7 = v; break;
        }
    }
}

/// <summary>VP9 loop filter (libvpx v1.17.0, 8-bit): edge filters and whole-superblock filtering, kernel-callable.</summary>
public static class Vp9Lpf
{
    private const int MaxLoopFilter = 63;

    // ---------------------------------------------------------------- vpx_dsp/loopfilter.c

    private static int Abs(int v) => v < 0 ? -v : v;

    private static int SignedCharClamp(int t) => t < -128 ? -128 : t > 127 ? 127 : t;

    /// <summary>filter_mask: -1 (filter) or 0.</summary>
    private static int FilterMask(int limit, int blimit, int p3, int p2, int p1, int p0, int q0, int q1, int q2, int q3)
    {
        bool off = Abs(p3 - p2) > limit || Abs(p2 - p1) > limit || Abs(p1 - p0) > limit || Abs(q1 - q0) > limit ||
                   Abs(q2 - q1) > limit || Abs(q3 - q2) > limit || Abs(p0 - q0) * 2 + Abs(p1 - q1) / 2 > blimit;
        return off ? 0 : -1;
    }

    /// <summary>flat_mask4 with thresh 1: true when flat.</summary>
    private static bool FlatMask4(int p3, int p2, int p1, int p0, int q0, int q1, int q2, int q3) =>
        !(Abs(p1 - p0) > 1 || Abs(q1 - q0) > 1 || Abs(p2 - p0) > 1 || Abs(q2 - q0) > 1 || Abs(p3 - p0) > 1 || Abs(q3 - q0) > 1);

    /// <summary>hev_mask: -1 (high edge variance) or 0.</summary>
    private static int HevMask(int thresh, int p1, int p0, int q0, int q1) =>
        Abs(p1 - p0) > thresh || Abs(q1 - q0) > thresh ? -1 : 0;

    /// <summary>filter4 on the pixels at s[at - 2 * step] .. s[at + step].</summary>
    private static void Filter4(ArrayView<byte> s, long at, long step, int mask, int thresh)
    {
        int p1 = s[at - 2 * step], p0 = s[at - step], q0 = s[at], q1 = s[at + step];
        // (int8_t)(x ^ 0x80) == x - 128 for a byte x
        int ps1 = p1 - 128, ps0 = p0 - 128, qs0 = q0 - 128, qs1 = q1 - 128;
        int hev = HevMask(thresh, p1, p0, q0, q1);
        int filter = SignedCharClamp(ps1 - qs1) & hev;
        filter = SignedCharClamp(filter + 3 * (qs0 - ps0)) & mask;
        int filter1 = SignedCharClamp(filter + 4) >> 3;
        int filter2 = SignedCharClamp(filter + 3) >> 3;
        s[at] = (byte)(SignedCharClamp(qs0 - filter1) + 128);
        s[at - step] = (byte)(SignedCharClamp(ps0 + filter2) + 128);
        filter = ((filter1 + 1) >> 1) & ~hev;
        s[at + step] = (byte)(SignedCharClamp(qs1 - filter) + 128);
        s[at - 2 * step] = (byte)(SignedCharClamp(ps1 + filter) + 128);
    }

    /// <summary>filter8: the 7-tap filter where flat and masked, else filter4.</summary>
    private static void Filter8(ArrayView<byte> s, long at, long step, int mask, int thresh, bool flat)
    {
        if (flat && mask != 0)
        {
            int p3 = s[at - 4 * step], p2 = s[at - 3 * step], p1 = s[at - 2 * step], p0 = s[at - step];
            int q0 = s[at], q1 = s[at + step], q2 = s[at + 2 * step], q3 = s[at + 3 * step];
            s[at - 3 * step] = (byte)((p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            s[at - 2 * step] = (byte)((p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            s[at - step] = (byte)((p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            s[at] = (byte)((p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            s[at + step] = (byte)((p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            s[at + 2 * step] = (byte)((p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        }
        else
        {
            Filter4(s, at, step, mask, thresh);
        }
    }

    /// <summary>filter16: the 15-tap filter where flat2, flat and masked, else filter8.</summary>
    private static void Filter16(ArrayView<byte> s, long at, long step, int mask, int thresh, bool flat, bool flat2)
    {
        if (flat2 && flat && mask != 0)
        {
            int p7 = s[at - 8 * step], p6 = s[at - 7 * step], p5 = s[at - 6 * step], p4 = s[at - 5 * step];
            int p3 = s[at - 4 * step], p2 = s[at - 3 * step], p1 = s[at - 2 * step], p0 = s[at - step];
            int q0 = s[at], q1 = s[at + step], q2 = s[at + 2 * step], q3 = s[at + 3 * step];
            int q4 = s[at + 4 * step], q5 = s[at + 5 * step], q6 = s[at + 6 * step], q7 = s[at + 7 * step];
            s[at - 7 * step] = (byte)((p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4);
            s[at - 6 * step] = (byte)((p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4);
            s[at - 5 * step] = (byte)((p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4);
            s[at - 4 * step] = (byte)((p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4);
            s[at - 3 * step] = (byte)((p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4);
            s[at - 2 * step] = (byte)((p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4);
            s[at - step] = (byte)((p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4);
            s[at] = (byte)((p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7 + 8) >> 4);
            s[at + step] = (byte)((p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2 + 8) >> 4);
            s[at + 2 * step] = (byte)((p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3 + 8) >> 4);
            s[at + 3 * step] = (byte)((p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4 + 8) >> 4);
            s[at + 4 * step] = (byte)((p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5 + 8) >> 4);
            s[at + 5 * step] = (byte)((p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6 + 8) >> 4);
            s[at + 6 * step] = (byte)((p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7 + 8) >> 4);
        }
        else
        {
            Filter8(s, at, step, mask, thresh, flat);
        }
    }

    /// <summary>One pixel position of a 4-tap edge: the edge lies between s[at - step] and s[at].</summary>
    private static void Lpf4At(ArrayView<byte> s, long at, long step, int blimit, int limit, int thresh)
    {
        int p3 = s[at - 4 * step], p2 = s[at - 3 * step], p1 = s[at - 2 * step], p0 = s[at - step];
        int q0 = s[at], q1 = s[at + step], q2 = s[at + 2 * step], q3 = s[at + 3 * step];
        Filter4(s, at, step, FilterMask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3), thresh);
    }

    private static void Lpf8At(ArrayView<byte> s, long at, long step, int blimit, int limit, int thresh)
    {
        int p3 = s[at - 4 * step], p2 = s[at - 3 * step], p1 = s[at - 2 * step], p0 = s[at - step];
        int q0 = s[at], q1 = s[at + step], q2 = s[at + 2 * step], q3 = s[at + 3 * step];
        int mask = FilterMask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        Filter8(s, at, step, mask, thresh, FlatMask4(p3, p2, p1, p0, q0, q1, q2, q3));
    }

    private static void Lpf16At(ArrayView<byte> s, long at, long step, int blimit, int limit, int thresh)
    {
        int p3 = s[at - 4 * step], p2 = s[at - 3 * step], p1 = s[at - 2 * step], p0 = s[at - step];
        int q0 = s[at], q1 = s[at + step], q2 = s[at + 2 * step], q3 = s[at + 3 * step];
        int mask = FilterMask(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3);
        bool flat = FlatMask4(p3, p2, p1, p0, q0, q1, q2, q3);
        // flat_mask5(1, s[-8], s[-7], s[-6], s[-5], p0, q0, s[4], s[5], s[6], s[7]): every one of p4..p7 and q4..q7
        // within 1 of p0 / q0 (flat_mask4 on p7, p6, p5 / q5, q6, q7, then p4 and q4)
        int p7 = s[at - 8 * step], p6 = s[at - 7 * step], p5 = s[at - 6 * step], p4 = s[at - 5 * step];
        int q4 = s[at + 4 * step], q5 = s[at + 5 * step], q6 = s[at + 6 * step], q7 = s[at + 7 * step];
        bool flat2 = FlatMask4(p7, p6, p5, p0, q0, q5, q6, q7) && !(Abs(p4 - p0) > 1 || Abs(q4 - q0) > 1);
        Filter16(s, at, step, mask, thresh, flat, flat2);
    }

    /// <summary>vpx_lpf_horizontal_4: the horizontal edge above row <paramref name="at"/>, 8 pixels wide.</summary>
    public static void Horizontal4(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 8; i++) Lpf4At(s, at + i, pitch, blimit, limit, thresh);
    }

    public static void Horizontal4Dual(ArrayView<byte> s, long at, int pitch, int blimit0, int limit0, int thresh0,
        int blimit1, int limit1, int thresh1)
    {
        Horizontal4(s, at, pitch, blimit0, limit0, thresh0);
        Horizontal4(s, at + 8, pitch, blimit1, limit1, thresh1);
    }

    /// <summary>vpx_lpf_vertical_4: the vertical edge left of column <paramref name="at"/>, 8 rows high.</summary>
    public static void Vertical4(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 8; i++) Lpf4At(s, at + (long)i * pitch, 1, blimit, limit, thresh);
    }

    public static void Vertical4Dual(ArrayView<byte> s, long at, int pitch, int blimit0, int limit0, int thresh0,
        int blimit1, int limit1, int thresh1)
    {
        Vertical4(s, at, pitch, blimit0, limit0, thresh0);
        Vertical4(s, at + 8L * pitch, pitch, blimit1, limit1, thresh1);
    }

    public static void Horizontal8(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 8; i++) Lpf8At(s, at + i, pitch, blimit, limit, thresh);
    }

    public static void Horizontal8Dual(ArrayView<byte> s, long at, int pitch, int blimit0, int limit0, int thresh0,
        int blimit1, int limit1, int thresh1)
    {
        Horizontal8(s, at, pitch, blimit0, limit0, thresh0);
        Horizontal8(s, at + 8, pitch, blimit1, limit1, thresh1);
    }

    public static void Vertical8(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 8; i++) Lpf8At(s, at + (long)i * pitch, 1, blimit, limit, thresh);
    }

    public static void Vertical8Dual(ArrayView<byte> s, long at, int pitch, int blimit0, int limit0, int thresh0,
        int blimit1, int limit1, int thresh1)
    {
        Vertical8(s, at, pitch, blimit0, limit0, thresh0);
        Vertical8(s, at + 8L * pitch, pitch, blimit1, limit1, thresh1);
    }

    /// <summary>vpx_lpf_horizontal_16 (mb_lpf_horizontal_edge_w, count 1): 8 pixels wide.</summary>
    public static void Horizontal16(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 8; i++) Lpf16At(s, at + i, pitch, blimit, limit, thresh);
    }

    /// <summary>vpx_lpf_horizontal_16_dual (count 2): 16 pixels wide, one set of limits.</summary>
    public static void Horizontal16Dual(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 16; i++) Lpf16At(s, at + i, pitch, blimit, limit, thresh);
    }

    /// <summary>vpx_lpf_vertical_16 (mb_lpf_vertical_edge_w, count 8): 8 rows high.</summary>
    public static void Vertical16(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 8; i++) Lpf16At(s, at + (long)i * pitch, 1, blimit, limit, thresh);
    }

    /// <summary>vpx_lpf_vertical_16_dual (count 16): 16 rows high, one set of limits.</summary>
    public static void Vertical16Dual(ArrayView<byte> s, long at, int pitch, int blimit, int limit, int thresh)
    {
        for (int i = 0; i < 16; i++) Lpf16At(s, at + (long)i * pitch, 1, blimit, limit, thresh);
    }

    // ---------------------------------------------------------------- lookup tables (vp9_common_data.c, vp9_loopfilter.c)

    public static int Num8x8Wide(int b) => b switch { 5 or 6 or 7 => 2, 8 or 9 or 10 => 4, 11 or 12 => 8, _ => 1 };
    public static int Num8x8High(int b) => b switch { 4 or 6 => 2, 7 or 9 or 11 => 4, 8 => 2, 10 or 12 => 8, _ => 1 };
    public static int Num4x4Wide(int b) => b switch { 0 or 1 => 1, 2 or 3 or 4 => 2, 5 or 6 or 7 => 4, 8 or 9 or 10 => 8, _ => 16 };
    public static int Num4x4High(int b) => b switch { 0 or 2 => 1, 1 or 3 or 5 => 2, 4 or 6 or 8 => 4, 7 or 9 or 11 => 8, _ => 16 };

    /// <summary>uv_txsize_lookup[bsize][tx][ss_x][ss_y], each block size's 16 entries packed 2 bits each.</summary>
    public static int UvTxSize(int bsize, int tx, int ssX, int ssY)
    {
        uint packed = bsize switch
        {
            3 => 0x01010100u, 4 => 0x05050500u, 5 => 0x51511100u, 6 => 0x56565500u, 7 => 0x5a5a5500u,
            8 => 0x66665500u, 9 => 0xabaa5500u, 10 => 0xafaa5500u, 11 => 0xbbaa5500u, 12 => 0xffaa5500u, _ => 0u,
        };
        return (int)((packed >> (2 * ((tx << 2) | (ssX << 1) | ssY))) & 3);
    }

    private static ulong LeftPredictionMask(int b) => b switch
    {
        4 or 6 or 8 => 0x0000000000000101UL, 7 or 9 or 11 => 0x0000000001010101UL, 10 or 12 => 0x0101010101010101UL,
        _ => 0x0000000000000001UL,
    };

    private static ulong AbovePredictionMask(int b) => b switch
    {
        5 or 6 or 7 => 0x03UL, 8 or 9 or 10 => 0x0fUL, 11 or 12 => 0xffUL, _ => 0x01UL,
    };

    private static ulong SizeMask(int b) => b switch
    {
        4 => 0x0000000000000101UL, 5 => 0x0000000000000003UL, 6 => 0x0000000000000303UL, 7 => 0x0000000003030303UL,
        8 => 0x0000000000000f0fUL, 9 => 0x000000000f0f0f0fUL, 10 => 0x0f0f0f0f0f0f0f0fUL, 11 => 0x00000000ffffffffUL,
        12 => 0xffffffffffffffffUL, _ => 0x0000000000000001UL,
    };

    private static ulong Left64x64TxMask(int tx) => tx switch
    {
        2 => 0x5555555555555555UL, 3 => 0x1111111111111111UL, _ => 0xffffffffffffffffUL,
    };

    private static ulong Above64x64TxMask(int tx) => tx switch
    {
        2 => 0x00ff00ff00ff00ffUL, 3 => 0x000000ff000000ffUL, _ => 0xffffffffffffffffUL,
    };

    private static uint LeftPredictionMaskUv(int b) => b switch { 7 or 9 or 11 => 0x0011u, 10 or 12 => 0x1111u, _ => 0x0001u };
    private static uint AbovePredictionMaskUv(int b) => b switch { 8 or 9 or 10 => 0x0003u, 11 or 12 => 0x000fu, _ => 0x0001u };

    private static uint SizeMaskUv(int b) => b switch
    {
        7 => 0x0011u, 8 => 0x0003u, 9 => 0x0033u, 10 => 0x3333u, 11 => 0x00ffu, 12 => 0xffffu, _ => 0x0001u,
    };

    private static uint Left64x64TxMaskUv(int tx) => tx switch { 2 => 0x5555u, 3 => 0x1111u, _ => 0xffffu };
    private static uint Above64x64TxMaskUv(int tx) => tx switch { 2 => 0x0f0fu, 3 => 0x000fu, _ => 0xffffu };

    private const ulong LeftBorder = 0x1111111111111111UL, AboveBorder = 0x000000ff000000ffUL;
    private const uint LeftBorderUv = 0x1111u, AboveBorderUv = 0x000fu;

    // ---------------------------------------------------------------- frame init (vp9_loopfilter.c)

    /// <summary>
    /// update_sharpness + vp9_loop_filter_init's hev thresholds + vp9_loop_filter_frame_init: the frame table
    /// (<see cref="Vp9LfTable"/>) at lf[lfOffset..] from the header (<see cref="Vp9LfHdr"/>) at hdr[hdrOffset..].
    /// lvl[seg][INTRA_FRAME][1] is never read (intra modes map to mode_lf_lut 0); libvpx leaves it unchanged when
    /// deltas are enabled, this writes the intra level there.
    /// </summary>
    public static void FrameInit(ArrayView<int> hdr, int hdrOffset, ArrayView<int> lf, int lfOffset)
    {
        int level = hdr[hdrOffset + Vp9LfHdr.Level];
        int sharpness = hdr[hdrOffset + Vp9LfHdr.Sharpness];
        lf[lfOffset + Vp9LfTable.FrameLevel] = level;
        for (int lvl = 0; lvl <= MaxLoopFilter; lvl++)
        {
            int blockInsideLimit = lvl >> ((sharpness > 0 ? 1 : 0) + (sharpness > 4 ? 1 : 0));
            if (sharpness > 0 && blockInsideLimit > 9 - sharpness) blockInsideLimit = 9 - sharpness;
            if (blockInsideLimit < 1) blockInsideLimit = 1;
            int t = lfOffset + Vp9LfTable.Thr + lvl * 3;
            lf[t] = 2 * (lvl + 2) + blockInsideLimit;
            lf[t + 1] = blockInsideLimit;
            lf[t + 2] = lvl >> 4;
        }
        int scale = 1 << (level >> 5);
        bool deltas = hdr[hdrOffset + Vp9LfHdr.DeltaEnabled] != 0;
        bool segEnabled = hdr[hdrOffset + Vp9LfHdr.SegEnabled] != 0;
        bool abs = hdr[hdrOffset + Vp9LfHdr.SegAbsDelta] != 0;
        for (int seg = 0; seg < 8; seg++)
        {
            int lvlSeg = level;
            if (segEnabled && hdr[hdrOffset + Vp9LfHdr.SegAltLfEnabled + seg] != 0)
            {
                int data = hdr[hdrOffset + Vp9LfHdr.SegAltLfData + seg];
                lvlSeg = Clamp(abs ? data : level + data, 0, MaxLoopFilter);
            }
            int at = lfOffset + Vp9LfTable.Lvl + seg * 8;
            if (!deltas)
            {
                for (int k = 0; k < 8; k++) lf[at + k] = lvlSeg;
            }
            else
            {
                int intraLvl = Clamp(lvlSeg + hdr[hdrOffset + Vp9LfHdr.RefDeltas] * scale, 0, MaxLoopFilter);
                lf[at] = intraLvl;
                lf[at + 1] = intraLvl;
                for (int rf = 1; rf < 4; rf++)
                    for (int mode = 0; mode < 2; mode++)
                        lf[at + rf * 2 + mode] = Clamp(lvlSeg + hdr[hdrOffset + Vp9LfHdr.RefDeltas + rf] * scale +
                                                       hdr[hdrOffset + Vp9LfHdr.ModeDeltas + mode] * scale, 0, MaxLoopFilter);
            }
        }
    }

    private static int Clamp(int v, int lo, int hi) => v < lo ? lo : v > hi ? hi : v;

    /// <summary>get_filter_level: lvl[segment_id][ref_frame[0]][mode_lf_lut[mode]] (NEARESTMV, NEARMV, NEWMV -> 1).</summary>
    private static int FilterLevel(ArrayView<int> mi, int b, ArrayView<int> lf, int lfOffset)
    {
        int mode = mi[b + Vp9Mi.Mode];
        int modeLf = mode == 10 || mode == 11 || mode == 13 ? 1 : 0;
        return lf[lfOffset + Vp9LfTable.Lvl + mi[b + Vp9Mi.SegmentId] * 8 + mi[b + Vp9Mi.Ref0] * 2 + modeLf];
    }

    // ---------------------------------------------------------------- masks (vp9_loopfilter.c)

    /// <summary>vp9_build_mask for the block whose record is <paramref name="rec"/>, at mi (row, col) of the superblock.</summary>
    public static void BuildMask(ref Vp9LfMask m, ArrayView<int> mi, int rec, int rowInSb, int colInSb, ArrayView<int> lf, int lfOffset)
    {
        int b = rec * Vp9Mi.Stride;
        int blockSize = mi[b + Vp9Mi.SbType];
        int txY = mi[b + Vp9Mi.TxSize];
        int txUv = UvTxSize(blockSize, txY, 1, 1);
        int level = FilterLevel(mi, b, lf, lfOffset);
        int shiftY = colInSb + (rowInSb << 3);
        int shiftUv = (colInSb >> 1) + ((rowInSb >> 1) << 2);
        bool buildUv = ((rowInSb | colInSb) & 1) == 0;   // first_block_in_16x16
        if (level == 0) return;
        int bw = Num8x8Wide(blockSize), bh = Num8x8High(blockSize);
        for (int i = 0; i < bh; i++) Vp9LfMask.SetLfl(ref m, rowInSb + i, colInSb, bw, level);

        Vp9LfMask.OrAboveY(ref m, txY, AbovePredictionMask(blockSize) << shiftY);
        Vp9LfMask.OrLeftY(ref m, txY, LeftPredictionMask(blockSize) << shiftY);
        if (buildUv)
        {
            Vp9LfMask.OrAboveUv(ref m, txUv, AbovePredictionMaskUv(blockSize) << shiftUv);
            Vp9LfMask.OrLeftUv(ref m, txUv, LeftPredictionMaskUv(blockSize) << shiftUv);
        }
        // no coefficients and not intra: no filtering on the transform edges inside the block
        if (mi[b + Vp9Mi.Skip] != 0 && mi[b + Vp9Mi.Ref0] > 0) return;

        Vp9LfMask.OrAboveY(ref m, txY, (SizeMask(blockSize) & Above64x64TxMask(txY)) << shiftY);
        Vp9LfMask.OrLeftY(ref m, txY, (SizeMask(blockSize) & Left64x64TxMask(txY)) << shiftY);
        if (buildUv)
        {
            Vp9LfMask.OrAboveUv(ref m, txUv, (SizeMaskUv(blockSize) & Above64x64TxMaskUv(txUv)) << shiftUv);
            Vp9LfMask.OrLeftUv(ref m, txUv, (SizeMaskUv(blockSize) & Left64x64TxMaskUv(txUv)) << shiftUv);
        }
        if (txY == 0) m.Int4x4Y |= SizeMask(blockSize) << shiftY;
        if (buildUv && txUv == 0) m.Int4x4Uv = (m.Int4x4Uv | ((SizeMaskUv(blockSize) & 0xffffu) << shiftUv)) & 0xffff;
    }

    /// <summary>vp9_adjust_mask.</summary>
    public static void AdjustMask(ref Vp9LfMask m, int miRows, int miCols, int miRow, int miCol)
    {
        // the largest filter is 16 wide: 32x32 transforms use it too
        m.LeftY2 |= m.LeftY3;
        m.AboveY2 |= m.AboveY3;
        m.LeftUv2 |= m.LeftUv3;
        m.AboveUv2 |= m.AboveUv3;

        // at least an 8 tap filter on every 32x32 edge, even for 4x4 transforms
        m.LeftY1 |= m.LeftY0 & LeftBorder;
        m.LeftY0 &= ~LeftBorder;
        m.AboveY1 |= m.AboveY0 & AboveBorder;
        m.AboveY0 &= ~AboveBorder;
        m.LeftUv1 |= m.LeftUv0 & LeftBorderUv;
        m.LeftUv0 &= ~LeftBorderUv & 0xffff;
        m.AboveUv1 |= m.AboveUv0 & AboveBorderUv;
        m.AboveUv0 &= ~AboveBorderUv & 0xffff;

        if (miRow + 8 > miRows)
        {
            int rows = miRows - miRow;
            ulong maskY = (1UL << (rows << 3)) - 1;
            uint maskUv = ((1u << (((rows + 1) >> 1) << 2)) - 1) & 0xffff;
            // tx sizes 0..2 (TX_32X32's masks are no longer used)
            m.LeftY0 &= maskY; m.LeftY1 &= maskY; m.LeftY2 &= maskY;
            m.AboveY0 &= maskY; m.AboveY1 &= maskY; m.AboveY2 &= maskY;
            m.LeftUv0 &= maskUv; m.LeftUv1 &= maskUv; m.LeftUv2 &= maskUv;
            m.AboveUv0 &= maskUv; m.AboveUv1 &= maskUv; m.AboveUv2 &= maskUv;
            m.Int4x4Y &= maskY;
            m.Int4x4Uv &= maskUv;
            // no wide filter on the last uv block row: the shorter one instead
            if (rows == 1)
            {
                m.AboveUv1 |= m.AboveUv2;
                m.AboveUv2 = 0;
            }
            if (rows == 5)
            {
                m.AboveUv1 |= m.AboveUv2 & 0xff00;
                m.AboveUv2 &= ~(m.AboveUv2 & 0xff00) & 0xffff;
            }
        }

        if (miCol + 8 > miCols)
        {
            int columns = miCols - miCol;
            ulong maskY = (ulong)((1 << columns) - 1) * 0x0101010101010101UL;
            uint maskUv = (uint)(((1 << ((columns + 1) >> 1)) - 1) * 0x1111) & 0xffff;
            // internal edges are not filtered on the last column of the image
            uint maskUvInt = (uint)(((1 << (columns >> 1)) - 1) * 0x1111) & 0xffff;
            m.LeftY0 &= maskY; m.LeftY1 &= maskY; m.LeftY2 &= maskY;
            m.AboveY0 &= maskY; m.AboveY1 &= maskY; m.AboveY2 &= maskY;
            m.LeftUv0 &= maskUv; m.LeftUv1 &= maskUv; m.LeftUv2 &= maskUv;
            m.AboveUv0 &= maskUv; m.AboveUv1 &= maskUv; m.AboveUv2 &= maskUv;
            m.Int4x4Y &= maskY;
            m.Int4x4Uv &= maskUvInt;
            // no wide filter on the last uv column: the shorter one instead
            if (columns == 1)
            {
                m.LeftUv1 |= m.LeftUv2;
                m.LeftUv2 = 0;
            }
            if (columns == 5)
            {
                m.LeftUv1 |= m.LeftUv2 & 0xcccc;
                m.LeftUv2 &= ~(m.LeftUv2 & 0xcccc) & 0xffff;
            }
        }
        // no filter on the first column of the image
        if (miCol == 0)
        {
            m.LeftY0 &= 0xfefefefefefefefeUL; m.LeftY1 &= 0xfefefefefefefefeUL; m.LeftY2 &= 0xfefefefefefefefeUL;
            m.LeftUv0 &= 0xeeee; m.LeftUv1 &= 0xeeee; m.LeftUv2 &= 0xeeee;
        }
    }

    /// <summary>
    /// The masks libvpx's decoder filters the superblock at mi (<paramref name="miRow"/>, <paramref name="miCol"/>)
    /// with: vp9_build_mask for each block decoded in it, then vp9_adjust_mask.
    /// </summary>
    public static void BuildSuperblockMask(ref Vp9LfMask m, ArrayView<int> mi, int gridOffset, int miRows, int miCols,
        int miRow, int miCol, ArrayView<int> lf, int lfOffset)
    {
        m = default;
        for (int r = 0; r < 8 && miRow + r < miRows; r++)
            for (int c = 0; c < 8 && miCol + c < miCols; c++)
            {
                int at = (miRow + r) * miCols + miCol + c;
                // a block's record sits at its top-left position: decode_block runs once per block
                if (mi[gridOffset + at] == at) BuildMask(ref m, mi, at, r, c, lf, lfOffset);
            }
        AdjustMask(ref m, miRows, miCols, miRow, miCol);
    }

    // ---------------------------------------------------------------- filtering (vp9_loopfilter.c), one lane per pixel row / column
    //
    // libvpx filters each plane of a superblock with all of its vertical edges first (filter_selectively_vert_row2 /
    // filter_selectively_vert, block columns left to right), then all of its horizontal edges (filter_selectively_horiz,
    // block rows top to bottom). A vertical edge filter reads and writes pixels of one pixel row only; a horizontal edge
    // filter pixels of one pixel column only. So within the vertical pass the pixel rows are independent of each other,
    // and within the horizontal pass the pixel columns are. A lane that owns one row (one column) and replays libvpx's
    // walk over the edges, filtering only its own pixel of each vpx_lpf_* call, applies exactly libvpx's operations to
    // that row (column), in libvpx's order. The *Lane functions are libvpx's functions with that one change: each
    // vpx_lpf_* call becomes the lane's pixel of it, when the call covers it, with the same arguments. A _dual call is
    // its two 8 pixel halves with their own limits, except vpx_lpf_{vertical,horizontal}_16_dual: one 16 pixel span
    // with the first block's limits (vpx_dsp/loopfilter.c).

    /// <summary>The lane's pixel of a vertical edge filter <paramref name="rows"/> rows high from s: its row <paramref name="i"/>.</summary>
    private static void V4(ArrayView<byte> p, long s, int pitch, int rows, int i, int bl, int li, int th)
    {
        if ((uint)i < (uint)rows) Lpf4At(p, s + (long)i * pitch, 1, bl, li, th);
    }

    private static void V8(ArrayView<byte> p, long s, int pitch, int rows, int i, int bl, int li, int th)
    {
        if ((uint)i < (uint)rows) Lpf8At(p, s + (long)i * pitch, 1, bl, li, th);
    }

    private static void V16(ArrayView<byte> p, long s, int pitch, int rows, int i, int bl, int li, int th)
    {
        if ((uint)i < (uint)rows) Lpf16At(p, s + (long)i * pitch, 1, bl, li, th);
    }

    /// <summary>The lane's pixel of a horizontal edge filter <paramref name="cols"/> pixels wide from s: its column <paramref name="i"/>.</summary>
    private static void H4(ArrayView<byte> p, long s, int pitch, int cols, int i, int bl, int li, int th)
    {
        if ((uint)i < (uint)cols) Lpf4At(p, s + i, pitch, bl, li, th);
    }

    private static void H8(ArrayView<byte> p, long s, int pitch, int cols, int i, int bl, int li, int th)
    {
        if ((uint)i < (uint)cols) Lpf8At(p, s + i, pitch, bl, li, th);
    }

    private static void H16(ArrayView<byte> p, long s, int pitch, int cols, int i, int bl, int li, int th)
    {
        if ((uint)i < (uint)cols) Lpf16At(p, s + i, pitch, bl, li, th);
    }

    private static int LflByte(ulong row, int k) => (int)((row >> (8 * k)) & 0xff);

    private static int Mblim(ArrayView<int> lf, int thr, int lvl) => lf[thr + lvl * 3];
    private static int Lim(ArrayView<int> lf, int thr, int lvl) => lf[thr + lvl * 3 + 1];
    private static int Hev(ArrayView<int> lf, int thr, int lvl) => lf[thr + lvl * 3 + 2];

    /// <summary>
    /// filter_selectively_vert_row2 (two rows of 8x8 blocks from pixel row s; lfl0 / lfl1 are the two rows' levels) for
    /// the lane's pixel row <paramref name="row"/> (0..15) of them.
    /// </summary>
    private static void FilterSelectivelyVertRow2Lane(int subsamplingFactor, ArrayView<byte> p, long s, int pitch, int row,
        uint mask16x16, uint mask8x8, uint mask4x4, uint mask4x4Int, ArrayView<int> lf, int thr, ulong lfl0, ulong lfl1)
    {
        uint dualMaskCutoff = subsamplingFactor != 0 ? 0xffu : 0xffffu;
        int lflForward = subsamplingFactor != 0 ? 4 : 8;
        uint dualOne = 1u | (1u << lflForward);
        int i0 = row, i1 = row - 8;   // the lane's row in the first / the second block row
        int k = 0;
        for (uint mask = (mask16x16 | mask8x8 | mask4x4 | mask4x4Int) & dualMaskCutoff; mask != 0; mask = (mask & ~dualOne) >> 1)
        {
            if ((mask & dualOne) != 0)
            {
                int l0 = LflByte(lfl0, k), l1 = LflByte(lfl1, k);
                long s0 = s, s1 = s + 8L * pitch;

                if ((mask16x16 & dualOne) != 0)
                {
                    if ((mask16x16 & dualOne) == dualOne)
                        V16(p, s0, pitch, 16, i0, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0));   // vertical_16_dual
                    else
                    {
                        bool first = (mask16x16 & 1) != 0;
                        int l = first ? l0 : l1;
                        V16(p, first ? s0 : s1, pitch, 8, first ? i0 : i1, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }

                if ((mask8x8 & dualOne) != 0)
                {
                    if ((mask8x8 & dualOne) == dualOne)
                    {
                        V8(p, s0, pitch, 8, i0, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0));
                        V8(p, s1, pitch, 8, i1, Mblim(lf, thr, l1), Lim(lf, thr, l1), Hev(lf, thr, l1));
                    }
                    else
                    {
                        bool first = (mask8x8 & 1) != 0;
                        int l = first ? l0 : l1;
                        V8(p, first ? s0 : s1, pitch, 8, first ? i0 : i1, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }

                if ((mask4x4 & dualOne) != 0)
                {
                    if ((mask4x4 & dualOne) == dualOne)
                    {
                        V4(p, s0, pitch, 8, i0, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0));
                        V4(p, s1, pitch, 8, i1, Mblim(lf, thr, l1), Lim(lf, thr, l1), Hev(lf, thr, l1));
                    }
                    else
                    {
                        bool first = (mask4x4 & 1) != 0;
                        int l = first ? l0 : l1;
                        V4(p, first ? s0 : s1, pitch, 8, first ? i0 : i1, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }

                if ((mask4x4Int & dualOne) != 0)
                {
                    if ((mask4x4Int & dualOne) == dualOne)
                    {
                        V4(p, s0 + 4, pitch, 8, i0, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0));
                        V4(p, s1 + 4, pitch, 8, i1, Mblim(lf, thr, l1), Lim(lf, thr, l1), Hev(lf, thr, l1));
                    }
                    else
                    {
                        bool first = (mask4x4Int & 1) != 0;
                        int l = first ? l0 : l1;
                        V4(p, (first ? s0 : s1) + 4, pitch, 8, first ? i0 : i1, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }
            }
            s += 8;
            k++;
            mask16x16 >>= 1;
            mask8x8 >>= 1;
            mask4x4 >>= 1;
            mask4x4Int >>= 1;
        }
    }

    /// <summary>
    /// filter_selectively_horiz (one row of 8x8 blocks from pixel s; lfl is the row's levels) for the lane's pixel column
    /// <paramref name="col"/> of it.
    /// </summary>
    private static void FilterSelectivelyHorizLane(ArrayView<byte> p, long s, int pitch, int col, uint mask16x16, uint mask8x8,
        uint mask4x4, uint mask4x4Int, ArrayView<int> lf, int thr, ulong lfl)
    {
        int k = 0;
        int count;
        for (uint mask = mask16x16 | mask8x8 | mask4x4 | mask4x4Int; mask != 0; mask >>= count)
        {
            count = 1;
            if ((mask & 1) != 0)
            {
                int i0 = col - 8 * k, i1 = i0 - 8;   // the lane's column in this block / in the next one
                int l = LflByte(lfl, k);
                int bl = Mblim(lf, thr, l), li = Lim(lf, thr, l), th = Hev(lf, thr, l);
                if ((mask16x16 & 1) != 0)
                {
                    if ((mask16x16 & 3) == 3)
                    {
                        H16(p, s, pitch, 16, i0, bl, li, th);   // horizontal_16_dual
                        count = 2;
                    }
                    else
                    {
                        H16(p, s, pitch, 8, i0, bl, li, th);
                    }
                }
                else if ((mask8x8 & 1) != 0)
                {
                    if ((mask8x8 & 3) == 3)
                    {
                        // the next block's thresholds
                        int n = LflByte(lfl, k + 1);
                        int bn = Mblim(lf, thr, n), ln = Lim(lf, thr, n), tn = Hev(lf, thr, n);
                        H8(p, s, pitch, 8, i0, bl, li, th);
                        H8(p, s + 8, pitch, 8, i1, bn, ln, tn);
                        if ((mask4x4Int & 3) == 3)
                        {
                            H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                            H4(p, s + 8 + 4L * pitch, pitch, 8, i1, bn, ln, tn);
                        }
                        else if ((mask4x4Int & 1) != 0)
                            H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                        else if ((mask4x4Int & 2) != 0)
                            H4(p, s + 8 + 4L * pitch, pitch, 8, i1, bn, ln, tn);
                        count = 2;
                    }
                    else
                    {
                        H8(p, s, pitch, 8, i0, bl, li, th);
                        if ((mask4x4Int & 1) != 0) H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                    }
                }
                else if ((mask4x4 & 1) != 0)
                {
                    if ((mask4x4 & 3) == 3)
                    {
                        int n = LflByte(lfl, k + 1);
                        int bn = Mblim(lf, thr, n), ln = Lim(lf, thr, n), tn = Hev(lf, thr, n);
                        H4(p, s, pitch, 8, i0, bl, li, th);
                        H4(p, s + 8, pitch, 8, i1, bn, ln, tn);
                        if ((mask4x4Int & 3) == 3)
                        {
                            H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                            H4(p, s + 8 + 4L * pitch, pitch, 8, i1, bn, ln, tn);
                        }
                        else if ((mask4x4Int & 1) != 0)
                            H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                        else if ((mask4x4Int & 2) != 0)
                            H4(p, s + 8 + 4L * pitch, pitch, 8, i1, bn, ln, tn);
                        count = 2;
                    }
                    else
                    {
                        H4(p, s, pitch, 8, i0, bl, li, th);
                        if ((mask4x4Int & 1) != 0) H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                    }
                }
                else
                {
                    H4(p, s + 4L * pitch, pitch, 8, i0, bl, li, th);
                }
            }
            s += 8 * count;
            k += count;
            mask16x16 >>= count;
            mask8x8 >>= count;
            mask4x4 >>= count;
            mask4x4Int >>= count;
        }
    }

    /// <summary>
    /// filter_selectively_vert (the non420 path: one row of 8x8 blocks from pixel row s) for the lane's pixel row
    /// <paramref name="row"/> (0..7) of it.
    /// </summary>
    private static void FilterSelectivelyVertLane(ArrayView<byte> p, long s, int pitch, int row, uint mask16x16, uint mask8x8,
        uint mask4x4, uint mask4x4Int, ArrayView<int> lf, int thr, ulong lfl)
    {
        int k = 0;
        for (uint mask = mask16x16 | mask8x8 | mask4x4 | mask4x4Int; mask != 0; mask >>= 1)
        {
            int l = LflByte(lfl, k);
            int bl = Mblim(lf, thr, l), li = Lim(lf, thr, l), th = Hev(lf, thr, l);
            if ((mask & 1) != 0)
            {
                if ((mask16x16 & 1) != 0) V16(p, s, pitch, 8, row, bl, li, th);
                else if ((mask8x8 & 1) != 0) V8(p, s, pitch, 8, row, bl, li, th);
                else if ((mask4x4 & 1) != 0) V4(p, s, pitch, 8, row, bl, li, th);
            }
            if ((mask4x4Int & 1) != 0) V4(p, s + 4, pitch, 8, row, bl, li, th);
            s += 8;
            k++;
            mask16x16 >>= 1;
            mask8x8 >>= 1;
            mask4x4 >>= 1;
            mask4x4Int >>= 1;
        }
    }

    // ---- vp9_filter_block_plane_ss00 / _ss11 / _non420, split into the vertical pass (per pixel row) and the
    //      horizontal pass (per pixel column)

    /// <summary>
    /// vp9_filter_block_plane_ss00's vertical pass (a plane without subsampling: Y, or U / V of 4:4:4; top-left pixel of
    /// the superblock at p[at]) for the lane's pixel row <paramref name="y"/> (0..63).
    /// </summary>
    private static void RowSs00(ArrayView<byte> p, long at, int stride, int miRow, int miRows, ref Vp9LfMask m,
        ArrayView<int> lf, int thr, int y)
    {
        // the serial loop: r = 0, 2, 4, 6 (two block rows at a time) while miRow + r < miRows, masks shifted 16 bits
        // per step, i.e. by 8 * r
        int r = (y >> 3) & ~1;
        if (miRow + r >= miRows) return;
        int sh = 8 * r;
        FilterSelectivelyVertRow2Lane(0, p, at + 8L * r * stride, stride, y & 15, (uint)(m.LeftY2 >> sh),
            (uint)(m.LeftY1 >> sh), (uint)(m.LeftY0 >> sh), (uint)(m.Int4x4Y >> sh), lf, thr,
            Vp9LfMask.Lfl(ref m, r), Vp9LfMask.Lfl(ref m, r + 1));
    }

    /// <summary>vp9_filter_block_plane_ss00's horizontal pass for the lane's pixel column <paramref name="x"/> (0..63).</summary>
    private static void ColSs00(ArrayView<byte> p, long at, int stride, int miRow, int miRows, ref Vp9LfMask m,
        ArrayView<int> lf, int thr, int x)
    {
        for (int r = 0; r < 8 && miRow + r < miRows; r++)
        {
            int sh = 8 * r;
            uint m16 = 0, m8 = 0, m4 = 0;
            if (miRow + r != 0)
            {
                m16 = (uint)((m.AboveY2 >> sh) & 0xff);
                m8 = (uint)((m.AboveY1 >> sh) & 0xff);
                m4 = (uint)((m.AboveY0 >> sh) & 0xff);
            }
            FilterSelectivelyHorizLane(p, at + 8L * r * stride, stride, x, m16, m8, m4, (uint)((m.Int4x4Y >> sh) & 0xff),
                lf, thr, Vp9LfMask.Lfl(ref m, r));
        }
    }

    /// <summary>lfl_uv row from lfl_y row: lfl_uv[c] = lfl_y[c &lt;&lt; 1], c = 0..3.</summary>
    private static ulong UvLflRow(ulong yRow) =>
        (yRow & 0xff) | ((yRow >> 8) & 0xff00) | ((yRow >> 16) & 0xff0000) | ((yRow >> 24) & 0xff000000);

    /// <summary>
    /// vp9_filter_block_plane_ss11's vertical pass (U or V of 4:2:0, the superblock's top-left chroma pixel at p[at]) for
    /// the lane's pixel row <paramref name="y"/> (0..31).
    /// </summary>
    private static void RowSs11(ArrayView<byte> p, long at, int stride, int miRow, int miRows, ref Vp9LfMask m,
        ArrayView<int> lf, int thr, int y)
    {
        // the serial loop: r = 0, 4 (two uv block rows at a time) while miRow + r < miRows; the masks as they are for
        // r = 0, then (mask >> 8) & 0xffff; the levels lfl_uv rows r / 2 and r / 2 + 1 = UvLflRow of lfl_y rows r, r + 2
        int pair = y >> 4, r = pair * 4;
        if (miRow + r >= miRows) return;
        uint mask16x16 = m.LeftUv2, mask8x8 = m.LeftUv1, mask4x4 = m.LeftUv0, mask4x4Int = m.Int4x4Uv;
        if (pair == 1)
        {
            mask16x16 = (mask16x16 >> 8) & 0xffff;
            mask8x8 = (mask8x8 >> 8) & 0xffff;
            mask4x4 = (mask4x4 >> 8) & 0xffff;
            mask4x4Int = (mask4x4Int >> 8) & 0xffff;
        }
        FilterSelectivelyVertRow2Lane(1, p, at + 16L * pair * stride, stride, y & 15, mask16x16, mask8x8, mask4x4, mask4x4Int,
            lf, thr, UvLflRow(Vp9LfMask.Lfl(ref m, r)), UvLflRow(Vp9LfMask.Lfl(ref m, r + 2)));
    }

    /// <summary>vp9_filter_block_plane_ss11's horizontal pass for the lane's pixel column <paramref name="x"/> (0..31).</summary>
    private static void ColSs11(ArrayView<byte> p, long at, int stride, int miRow, int miRows, ref Vp9LfMask m,
        ArrayView<int> lf, int thr, int x)
    {
        // uv block row j = r / 2: masks shifted 4 bits per row (2 * r), pixel rows 8 * j, levels lfl_uv row j (the
        // vertical pass filled it: UvLflRow of lfl_y row r)
        for (int r = 0; r < 8 && miRow + r < miRows; r += 2)
        {
            int sh = 2 * r;
            bool skipBorder4x4R = miRow + r == miRows - 1;
            uint m4i = skipBorder4x4R ? 0u : ((m.Int4x4Uv >> sh) & 0xf);
            uint m16 = 0, m8 = 0, m4 = 0;
            if (miRow + r != 0)
            {
                m16 = (m.AboveUv2 >> sh) & 0xf;
                m8 = (m.AboveUv1 >> sh) & 0xf;
                m4 = (m.AboveUv0 >> sh) & 0xf;
            }
            FilterSelectivelyHorizLane(p, at + 4L * r * stride, stride, x, m16, m8, m4, m4i, lf, thr,
                UvLflRow(Vp9LfMask.Lfl(ref m, r)));
        }
    }

    private static uint GetByte(ulong v, int i) => (uint)((v >> (8 * i)) & 0xff);

    /// <summary>
    /// The masks and levels of vp9_filter_block_plane_non420 (a chroma plane subsampled other than 4:2:0 / 4:4:4: 4:2:2,
    /// 4:4:0), built from the mode info directly. libvpx builds row r's masks right before filtering row r's vertical
    /// edges; building reads no pixels, so building them all first changes nothing. U and V use the same masks.
    /// </summary>
    public static void BuildNon420Mask(ref Vp9LfNon420Mask n, int ssX, int ssY, ArrayView<int> mi, int gridOffset, int miRows,
        int miCols, int miRow, int miCol, ArrayView<int> lf, int lfOffset)
    {
        n = default;
        int rowStep = 1 << ssY, colStep = 1 << ssX;
        for (int r = 0; r < 8 && miRow + r < miRows; r += rowStep)
        {
            uint mask16x16C = 0, mask8x8C = 0, mask4x4C = 0;
            ulong lflRow = 0;
            // the vertical edges that need filtering
            for (int c = 0; c < 8 && miCol + c < miCols; c += colStep)
            {
                int b = mi[gridOffset + (miRow + r) * miCols + miCol + c] * Vp9Mi.Stride;
                int sbType = mi[b + Vp9Mi.SbType];
                bool skipThis = mi[b + Vp9Mi.Skip] != 0 && mi[b + Vp9Mi.Ref0] > 0;
                // the left / top edge of the unit is a block edge: never skipped
                bool blockEdgeLeft = Num4x4Wide(sbType) > 1 ? (c & (Num8x8Wide(sbType) - 1)) == 0 : true;
                bool skipThisC = skipThis && !blockEdgeLeft;
                bool blockEdgeAbove = Num4x4High(sbType) > 1 ? (r & (Num8x8High(sbType) - 1)) == 0 : true;
                bool skipThisR = skipThis && !blockEdgeAbove;
                int txSize = UvTxSize(sbType, mi[b + Vp9Mi.TxSize], ssX, ssY);
                bool skipBorder4x4C = ssX != 0 && miCol + c == miCols - 1;
                bool skipBorder4x4R = ssY != 0 && miRow + r == miRows - 1;
                int cs = c >> ssX, rs = r >> ssY;
                uint bit = 1u << cs;

                int level = FilterLevel(mi, b, lf, lfOffset);
                lflRow = SetByte(lflRow, cs, (uint)level);
                if (level == 0) continue;

                if (txSize == 3)
                {
                    if (!skipThisC && (cs & 3) == 0)
                    {
                        if (!skipBorder4x4C) mask16x16C |= bit;
                        else mask8x8C |= bit;
                    }
                    if (!skipThisR && (rs & 3) == 0)
                    {
                        if (!skipBorder4x4R) n.Horz16 |= (ulong)bit << (8 * r);
                        else n.Horz8 |= (ulong)bit << (8 * r);
                    }
                }
                else if (txSize == 2)
                {
                    if (!skipThisC && (cs & 1) == 0)
                    {
                        if (!skipBorder4x4C) mask16x16C |= bit;
                        else mask8x8C |= bit;
                    }
                    if (!skipThisR && (rs & 1) == 0)
                    {
                        if (!skipBorder4x4R) n.Horz16 |= (ulong)bit << (8 * r);
                        else n.Horz8 |= (ulong)bit << (8 * r);
                    }
                }
                else
                {
                    // force 8x8 filtering on 32x32 boundaries
                    if (!skipThisC)
                    {
                        if (txSize == 1 || (cs & 3) == 0) mask8x8C |= bit;
                        else mask4x4C |= bit;
                    }
                    if (!skipThisR)
                    {
                        if (txSize == 1 || (rs & 3) == 0) n.Horz8 |= (ulong)bit << (8 * r);
                        else n.Horz4 |= (ulong)bit << (8 * r);
                    }
                    if (!skipThis && txSize < 1 && !skipBorder4x4C) n.Int4 |= (ulong)bit << (8 * r);
                }
            }
            n.Vert16 |= (ulong)mask16x16C << (8 * r);
            n.Vert8 |= (ulong)mask8x8C << (8 * r);
            n.Vert4 |= (ulong)mask4x4C << (8 * r);
            Vp9LfNon420Mask.SetLfl(ref n, r, lflRow);
        }
    }

    private static ulong SetByte(ulong v, int i, uint b) => (v & ~(0xffUL << (8 * i))) | ((ulong)(b & 0xff) << (8 * i));

    /// <summary>
    /// vp9_filter_block_plane_non420's vertical pass (the superblock's top-left pixel of the plane at p[at]) for the
    /// lane's pixel row <paramref name="y"/> (0 .. (64 &gt;&gt; ssY) - 1).
    /// </summary>
    private static void RowNon420(ArrayView<byte> p, long at, int stride, int ssY, int miRow, int miRows, int miCol,
        ref Vp9LfNon420Mask n, ArrayView<int> lf, int thr, int y)
    {
        // the serial loop: block row j at mi row r = j * row_step, while miRow + r < miRows
        int j = y >> 3, r = j << ssY;
        if (r >= 8 || miRow + r >= miRows) return;
        // no filtering on the leftmost column of the image
        uint borderMask = miCol == 0 ? ~1u : ~0u;
        FilterSelectivelyVertLane(p, at + 8L * j * stride, stride, y & 7, GetByte(n.Vert16, r) & borderMask,
            GetByte(n.Vert8, r) & borderMask, GetByte(n.Vert4, r) & borderMask, GetByte(n.Int4, r), lf, thr,
            Vp9LfNon420Mask.Lfl(ref n, r));
    }

    /// <summary>vp9_filter_block_plane_non420's horizontal pass for the lane's pixel column <paramref name="x"/>.</summary>
    private static void ColNon420(ArrayView<byte> p, long at, int stride, int ssY, int miRow, int miRows,
        ref Vp9LfNon420Mask n, ArrayView<int> lf, int thr, int x)
    {
        int rowStep = 1 << ssY;
        for (int r = 0, j = 0; r < 8 && miRow + r < miRows; r += rowStep, j++)
        {
            bool skipBorder4x4R = ssY != 0 && miRow + r == miRows - 1;
            uint mask4x4IntR = skipBorder4x4R ? 0u : GetByte(n.Int4, r);
            uint m16 = 0, m8 = 0, m4 = 0;
            if (miRow + r != 0)
            {
                m16 = GetByte(n.Horz16, r);
                m8 = GetByte(n.Horz8, r);
                m4 = GetByte(n.Horz4, r);
            }
            FilterSelectivelyHorizLane(p, at + 8L * j * stride, stride, x, m16, m8, m4, mask4x4IntR, lf, thr,
                Vp9LfNon420Mask.Lfl(ref n, r));
        }
    }

    // ---------------------------------------------------------------- one superblock, one thread group

    /// <summary>Lanes of the thread group that filters one superblock: one per pixel row, then one per pixel column.</summary>
    public const int GroupLanes = 64;

    // The group's shared memory (uint words; a ulong is two words, low first): the superblock's Vp9LfMask, then its
    // Vp9LfNon420Mask.
    private const int ShLeftY = 0, ShAboveY = 8, ShInt4x4Y = 16, ShLfl = 18, ShLeftUv = 34, ShAboveUv = 38, ShInt4x4Uv = 42,
        ShNon420 = 43, ShNon420Lfl = 57;

    /// <summary>The shared memory (uint words) <see cref="FilterSuperblockGroup"/> needs.</summary>
    public const int GroupSharedWords = 73;

    private static void Put(ArrayView<uint> s, int i, ulong v)
    {
        s[i] = (uint)v;
        s[i + 1] = (uint)(v >> 32);
    }

    private static ulong Get(ArrayView<uint> s, int i) => s[i] | ((ulong)s[i + 1] << 32);

    private static void StoreMasks(ArrayView<uint> s, ref Vp9LfMask m, ref Vp9LfNon420Mask n)
    {
        for (int t = 0; t < 4; t++)
        {
            Put(s, ShLeftY + 2 * t, Vp9LfMask.LeftY(ref m, t));
            Put(s, ShAboveY + 2 * t, Vp9LfMask.AboveY(ref m, t));
            s[ShLeftUv + t] = Vp9LfMask.LeftUv(ref m, t);
            s[ShAboveUv + t] = Vp9LfMask.AboveUv(ref m, t);
        }
        Put(s, ShInt4x4Y, m.Int4x4Y);
        s[ShInt4x4Uv] = m.Int4x4Uv;
        for (int r = 0; r < 8; r++)
        {
            Put(s, ShLfl + 2 * r, Vp9LfMask.Lfl(ref m, r));
            Put(s, ShNon420Lfl + 2 * r, Vp9LfNon420Mask.Lfl(ref n, r));
        }
        Put(s, ShNon420, n.Vert16);
        Put(s, ShNon420 + 2, n.Vert8);
        Put(s, ShNon420 + 4, n.Vert4);
        Put(s, ShNon420 + 6, n.Horz16);
        Put(s, ShNon420 + 8, n.Horz8);
        Put(s, ShNon420 + 10, n.Horz4);
        Put(s, ShNon420 + 12, n.Int4);
    }

    private static void LoadMasks(ArrayView<uint> s, ref Vp9LfMask m, ref Vp9LfNon420Mask n)
    {
        m.LeftY0 = Get(s, ShLeftY); m.LeftY1 = Get(s, ShLeftY + 2); m.LeftY2 = Get(s, ShLeftY + 4); m.LeftY3 = Get(s, ShLeftY + 6);
        m.AboveY0 = Get(s, ShAboveY); m.AboveY1 = Get(s, ShAboveY + 2); m.AboveY2 = Get(s, ShAboveY + 4); m.AboveY3 = Get(s, ShAboveY + 6);
        m.Int4x4Y = Get(s, ShInt4x4Y);
        m.LeftUv0 = s[ShLeftUv]; m.LeftUv1 = s[ShLeftUv + 1]; m.LeftUv2 = s[ShLeftUv + 2]; m.LeftUv3 = s[ShLeftUv + 3];
        m.AboveUv0 = s[ShAboveUv]; m.AboveUv1 = s[ShAboveUv + 1]; m.AboveUv2 = s[ShAboveUv + 2]; m.AboveUv3 = s[ShAboveUv + 3];
        m.Int4x4Uv = s[ShInt4x4Uv];
        m.Lfl0 = Get(s, ShLfl); m.Lfl1 = Get(s, ShLfl + 2); m.Lfl2 = Get(s, ShLfl + 4); m.Lfl3 = Get(s, ShLfl + 6);
        m.Lfl4 = Get(s, ShLfl + 8); m.Lfl5 = Get(s, ShLfl + 10); m.Lfl6 = Get(s, ShLfl + 12); m.Lfl7 = Get(s, ShLfl + 14);
        n.Vert16 = Get(s, ShNon420); n.Vert8 = Get(s, ShNon420 + 2); n.Vert4 = Get(s, ShNon420 + 4);
        n.Horz16 = Get(s, ShNon420 + 6); n.Horz8 = Get(s, ShNon420 + 8); n.Horz4 = Get(s, ShNon420 + 10);
        n.Int4 = Get(s, ShNon420 + 12);
        n.Lfl0 = Get(s, ShNon420Lfl); n.Lfl1 = Get(s, ShNon420Lfl + 2); n.Lfl2 = Get(s, ShNon420Lfl + 4); n.Lfl3 = Get(s, ShNon420Lfl + 6);
        n.Lfl4 = Get(s, ShNon420Lfl + 8); n.Lfl5 = Get(s, ShNon420Lfl + 10); n.Lfl6 = Get(s, ShNon420Lfl + 12); n.Lfl7 = Get(s, ShNon420Lfl + 14);
    }

    /// <summary>
    /// The vertical pass of every plane of the superblock for lane <paramref name="lane"/>: luma pixel row lane, then the
    /// chroma pixel rows lane, lane + 64, ... of U's rows followed by V's.
    /// </summary>
    // A real function on Wasm (and on the WebGPU / WebGL backends once ILGPU emits view-taking helpers as functions),
    // not inlined at every call site: keeps the kernels small enough for the browsers' shader and Wasm compilers.
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void SuperblockRowsLane(int lane, ArrayView<byte> p, long yAt, int yStride, long uAt, long vAt, int uvStride,
        int ssX, int ssY, int miRows, int miRow, int miCol, Vp9LfMask m, Vp9LfNon420Mask n, ArrayView<int> lf, int lfOffset)
    {
        int thr = lfOffset + Vp9LfTable.Thr;
        RowSs00(p, yAt, yStride, miRow, miRows, ref m, lf, thr, lane);
        int rows = 64 >> ssY;
        for (int t = lane; t < 2 * rows; t += GroupLanes)
        {
            bool isU = t < rows;
            long at = isU ? uAt : vAt;
            int y = isU ? t : t - rows;
            if (ssX == 1 && ssY == 1) RowSs11(p, at, uvStride, miRow, miRows, ref m, lf, thr, y);
            else if (ssX == 0 && ssY == 0) RowSs00(p, at, uvStride, miRow, miRows, ref m, lf, thr, y);
            else RowNon420(p, at, uvStride, ssY, miRow, miRows, miCol, ref n, lf, thr, y);
        }
    }

    /// <summary>
    /// The horizontal pass of every plane of the superblock for lane <paramref name="lane"/>: luma pixel column lane, then
    /// the chroma pixel columns lane, lane + 64, ... of U's columns followed by V's.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void SuperblockColumnsLane(int lane, ArrayView<byte> p, long yAt, int yStride, long uAt, long vAt, int uvStride,
        int ssX, int ssY, int miRows, int miRow, Vp9LfMask m, Vp9LfNon420Mask n, ArrayView<int> lf, int lfOffset)
    {
        int thr = lfOffset + Vp9LfTable.Thr;
        ColSs00(p, yAt, yStride, miRow, miRows, ref m, lf, thr, lane);
        int cols = 64 >> ssX;
        for (int t = lane; t < 2 * cols; t += GroupLanes)
        {
            bool isU = t < cols;
            long at = isU ? uAt : vAt;
            int x = isU ? t : t - cols;
            if (ssX == 1 && ssY == 1) ColSs11(p, at, uvStride, miRow, miRows, ref m, lf, thr, x);
            else if (ssX == 0 && ssY == 0) ColSs00(p, at, uvStride, miRow, miRows, ref m, lf, thr, x);
            else ColNon420(p, at, uvStride, ssY, miRow, miRows, ref n, lf, thr, x);
        }
    }

    /// <summary>
    /// One iteration of libvpx loop_filter_rows' inner loop, the superblock at mi (<paramref name="miRow"/>,
    /// <paramref name="miCol"/>), all planes, by a thread group of <see cref="GroupLanes"/> lanes: every lane of the group
    /// calls this (barriers inside), with <paramref name="shared"/> = <see cref="GroupSharedWords"/> words of group shared
    /// memory. <paramref name="yAt"/> is the index in <paramref name="p"/> of the superblock's top-left luma pixel,
    /// <paramref name="uAt"/> / <paramref name="vAt"/> of its top-left chroma pixels (row (miRow * 8) &gt;&gt; ssY, column
    /// (miCol * 8) &gt;&gt; ssX of the plane). <paramref name="active"/> false (the same for the whole group): the group
    /// filters nothing but still meets the barriers. Does nothing when the frame's filter level is 0 (libvpx skips the
    /// whole frame). Superblocks run in raster order or on the wavefront (see the file header).
    /// <list type="number">
    /// <item>Lane 0 builds the superblock's masks (vp9_build_mask per block, vp9_adjust_mask; the non420 masks for
    /// 4:2:2 / 4:4:0) into shared memory. Barrier.</item>
    /// <item>Vertical edges: each lane its pixel rows of every plane. Barrier.</item>
    /// <item>Horizontal edges: each lane its pixel columns of every plane.</item>
    /// </list>
    /// The planes never touch each other's pixels, so running U's and V's passes next to Y's changes nothing.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void FilterSuperblockGroup(ArrayView<uint> shared, bool active, ArrayView<byte> p, long yAt, int yStride,
        long uAt, long vAt, int uvStride, int ssX, int ssY, ArrayView<int> mi, int gridOffset, int miRows, int miCols,
        int miRow, int miCol, ArrayView<int> lf, int lfOffset)
    {
        int lane = Group.IdxX;
        bool on = active && lf[lfOffset + Vp9LfTable.FrameLevel] != 0;
        // barriers stay outside every branch: the whole group meets each one
        if (on && lane == 0)
        {
            var built = default(Vp9LfMask);
            BuildSuperblockMask(ref built, mi, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
            var builtNon420 = default(Vp9LfNon420Mask);
            if (ssX != ssY) BuildNon420Mask(ref builtNon420, ssX, ssY, mi, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
            StoreMasks(shared, ref built, ref builtNon420);
        }
        Group.Barrier();
        var m = default(Vp9LfMask);
        var n = default(Vp9LfNon420Mask);
        if (on)
        {
            LoadMasks(shared, ref m, ref n);
            SuperblockRowsLane(lane, p, yAt, yStride, uAt, vAt, uvStride, ssX, ssY, miRows, miRow, miCol, m, n, lf, lfOffset);
        }
        Group.Barrier();
        if (on) SuperblockColumnsLane(lane, p, yAt, yStride, uAt, vAt, uvStride, ssX, ssY, miRows, miRow, m, n, lf, lfOffset);
    }
}
