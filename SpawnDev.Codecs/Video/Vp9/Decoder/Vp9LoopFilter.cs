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
// Mode info (the decoder's Vp9Mi layout): records of Vp9Mi.Stride ints at record index mi_row * mi_cols + mi_col of the
// block's top-left, fields SbType, Mode (libvpx mi->mode: for sub8x8 inter blocks the last sub-block's mode), TxSize,
// Skip (the value after decode_block's "no coefficients -> skip = 1" for inter blocks >= 8x8), SegmentId, Ref0
// (0 intra, 1 last, 2 golden, 3 altref); the grid (one int per mi position, at gridOffset + mi_row * mi_cols +
// mi_col) holds the record index of the block covering the position.
// Frame table (Vp9LfTable, ints): lf_info.lvl[seg][ref][mode], lf_info.lfthr[lvl] mblim / lim / hev_thr and the frame
// filter level; FrameInit builds it from the frame header (Vp9LfHdr) on the accelerator.

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

    // ---------------------------------------------------------------- filtering (vp9_loopfilter.c)

    private static int LflByte(ulong row, int k) => (int)((row >> (8 * k)) & 0xff);

    private static int Mblim(ArrayView<int> lf, int thr, int lvl) => lf[thr + lvl * 3];
    private static int Lim(ArrayView<int> lf, int thr, int lvl) => lf[thr + lvl * 3 + 1];
    private static int Hev(ArrayView<int> lf, int thr, int lvl) => lf[thr + lvl * 3 + 2];

    /// <summary>filter_selectively_vert_row2: two rows of 8x8 blocks; lfl0 / lfl1 are the two rows' levels.</summary>
    private static void FilterSelectivelyVertRow2(int subsamplingFactor, ArrayView<byte> p, long s, int pitch,
        uint mask16x16, uint mask8x8, uint mask4x4, uint mask4x4Int, ArrayView<int> lf, int thr, ulong lfl0, ulong lfl1)
    {
        uint dualMaskCutoff = subsamplingFactor != 0 ? 0xffu : 0xffffu;
        int lflForward = subsamplingFactor != 0 ? 4 : 8;
        uint dualOne = 1u | (1u << lflForward);
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
                        Vertical16Dual(p, s0, pitch, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0));
                    else
                    {
                        bool first = (mask16x16 & 1) != 0;
                        int l = first ? l0 : l1;
                        Vertical16(p, first ? s0 : s1, pitch, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }

                if ((mask8x8 & dualOne) != 0)
                {
                    if ((mask8x8 & dualOne) == dualOne)
                        Vertical8Dual(p, s0, pitch, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0),
                            Mblim(lf, thr, l1), Lim(lf, thr, l1), Hev(lf, thr, l1));
                    else
                    {
                        bool first = (mask8x8 & 1) != 0;
                        int l = first ? l0 : l1;
                        Vertical8(p, first ? s0 : s1, pitch, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }

                if ((mask4x4 & dualOne) != 0)
                {
                    if ((mask4x4 & dualOne) == dualOne)
                        Vertical4Dual(p, s0, pitch, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0),
                            Mblim(lf, thr, l1), Lim(lf, thr, l1), Hev(lf, thr, l1));
                    else
                    {
                        bool first = (mask4x4 & 1) != 0;
                        int l = first ? l0 : l1;
                        Vertical4(p, first ? s0 : s1, pitch, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
                    }
                }

                if ((mask4x4Int & dualOne) != 0)
                {
                    if ((mask4x4Int & dualOne) == dualOne)
                        Vertical4Dual(p, s0 + 4, pitch, Mblim(lf, thr, l0), Lim(lf, thr, l0), Hev(lf, thr, l0),
                            Mblim(lf, thr, l1), Lim(lf, thr, l1), Hev(lf, thr, l1));
                    else
                    {
                        bool first = (mask4x4Int & 1) != 0;
                        int l = first ? l0 : l1;
                        Vertical4(p, (first ? s0 : s1) + 4, pitch, Mblim(lf, thr, l), Lim(lf, thr, l), Hev(lf, thr, l));
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

    /// <summary>filter_selectively_horiz: one row of 8x8 blocks; lfl is the row's levels.</summary>
    private static void FilterSelectivelyHoriz(ArrayView<byte> p, long s, int pitch, uint mask16x16, uint mask8x8,
        uint mask4x4, uint mask4x4Int, ArrayView<int> lf, int thr, ulong lfl)
    {
        int k = 0;
        int count;
        for (uint mask = mask16x16 | mask8x8 | mask4x4 | mask4x4Int; mask != 0; mask >>= count)
        {
            count = 1;
            if ((mask & 1) != 0)
            {
                int l = LflByte(lfl, k);
                int bl = Mblim(lf, thr, l), li = Lim(lf, thr, l), th = Hev(lf, thr, l);
                if ((mask16x16 & 1) != 0)
                {
                    if ((mask16x16 & 3) == 3)
                    {
                        Horizontal16Dual(p, s, pitch, bl, li, th);
                        count = 2;
                    }
                    else
                    {
                        Horizontal16(p, s, pitch, bl, li, th);
                    }
                }
                else if ((mask8x8 & 1) != 0)
                {
                    if ((mask8x8 & 3) == 3)
                    {
                        // the next block's thresholds
                        int n = LflByte(lfl, k + 1);
                        int bn = Mblim(lf, thr, n), ln = Lim(lf, thr, n), tn = Hev(lf, thr, n);
                        Horizontal8Dual(p, s, pitch, bl, li, th, bn, ln, tn);
                        if ((mask4x4Int & 3) == 3)
                            Horizontal4Dual(p, s + 4L * pitch, pitch, bl, li, th, bn, ln, tn);
                        else if ((mask4x4Int & 1) != 0)
                            Horizontal4(p, s + 4L * pitch, pitch, bl, li, th);
                        else if ((mask4x4Int & 2) != 0)
                            Horizontal4(p, s + 8 + 4L * pitch, pitch, bn, ln, tn);
                        count = 2;
                    }
                    else
                    {
                        Horizontal8(p, s, pitch, bl, li, th);
                        if ((mask4x4Int & 1) != 0) Horizontal4(p, s + 4L * pitch, pitch, bl, li, th);
                    }
                }
                else if ((mask4x4 & 1) != 0)
                {
                    if ((mask4x4 & 3) == 3)
                    {
                        int n = LflByte(lfl, k + 1);
                        int bn = Mblim(lf, thr, n), ln = Lim(lf, thr, n), tn = Hev(lf, thr, n);
                        Horizontal4Dual(p, s, pitch, bl, li, th, bn, ln, tn);
                        if ((mask4x4Int & 3) == 3)
                            Horizontal4Dual(p, s + 4L * pitch, pitch, bl, li, th, bn, ln, tn);
                        else if ((mask4x4Int & 1) != 0)
                            Horizontal4(p, s + 4L * pitch, pitch, bl, li, th);
                        else if ((mask4x4Int & 2) != 0)
                            Horizontal4(p, s + 8 + 4L * pitch, pitch, bn, ln, tn);
                        count = 2;
                    }
                    else
                    {
                        Horizontal4(p, s, pitch, bl, li, th);
                        if ((mask4x4Int & 1) != 0) Horizontal4(p, s + 4L * pitch, pitch, bl, li, th);
                    }
                }
                else
                {
                    Horizontal4(p, s + 4L * pitch, pitch, bl, li, th);
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

    /// <summary>filter_selectively_vert (the non420 path): one row of 8x8 blocks.</summary>
    private static void FilterSelectivelyVert(ArrayView<byte> p, long s, int pitch, uint mask16x16, uint mask8x8,
        uint mask4x4, uint mask4x4Int, ArrayView<int> lf, int thr, ulong lfl)
    {
        int k = 0;
        for (uint mask = mask16x16 | mask8x8 | mask4x4 | mask4x4Int; mask != 0; mask >>= 1)
        {
            int l = LflByte(lfl, k);
            int bl = Mblim(lf, thr, l), li = Lim(lf, thr, l), th = Hev(lf, thr, l);
            if ((mask & 1) != 0)
            {
                if ((mask16x16 & 1) != 0) Vertical16(p, s, pitch, bl, li, th);
                else if ((mask8x8 & 1) != 0) Vertical8(p, s, pitch, bl, li, th);
                else if ((mask4x4 & 1) != 0) Vertical4(p, s, pitch, bl, li, th);
            }
            if ((mask4x4Int & 1) != 0) Vertical4(p, s + 4, pitch, bl, li, th);
            s += 8;
            k++;
            mask16x16 >>= 1;
            mask8x8 >>= 1;
            mask4x4 >>= 1;
            mask4x4Int >>= 1;
        }
    }

    /// <summary>
    /// vp9_filter_block_plane_ss00: a plane without subsampling (Y, or U / V of 4:4:4) of the superblock whose
    /// top-left pixel is p[at], with the superblock's masks.
    /// </summary>
    public static void FilterBlockPlaneSs00(ArrayView<byte> p, long at, int stride, int miRow, int miRows,
        ref Vp9LfMask m, ArrayView<int> lf, int lfOffset)
    {
        int thr = lfOffset + Vp9LfTable.Thr;
        long dst = at;
        ulong mask16x16 = m.LeftY2, mask8x8 = m.LeftY1, mask4x4 = m.LeftY0, mask4x4Int = m.Int4x4Y;
        // vertical pass: 2 rows at a time
        for (int r = 0; r < 8 && miRow + r < miRows; r += 2)
        {
            FilterSelectivelyVertRow2(0, p, dst, stride, (uint)mask16x16, (uint)mask8x8, (uint)mask4x4, (uint)mask4x4Int,
                lf, thr, Vp9LfMask.Lfl(ref m, r), Vp9LfMask.Lfl(ref m, r + 1));
            dst += 16L * stride;
            mask16x16 >>= 16;
            mask8x8 >>= 16;
            mask4x4 >>= 16;
            mask4x4Int >>= 16;
        }

        // horizontal pass
        dst = at;
        mask16x16 = m.AboveY2;
        mask8x8 = m.AboveY1;
        mask4x4 = m.AboveY0;
        mask4x4Int = m.Int4x4Y;
        for (int r = 0; r < 8 && miRow + r < miRows; r++)
        {
            uint m16 = 0, m8 = 0, m4 = 0;
            if (miRow + r != 0)
            {
                m16 = (uint)(mask16x16 & 0xff);
                m8 = (uint)(mask8x8 & 0xff);
                m4 = (uint)(mask4x4 & 0xff);
            }
            FilterSelectivelyHoriz(p, dst, stride, m16, m8, m4, (uint)(mask4x4Int & 0xff), lf, thr, Vp9LfMask.Lfl(ref m, r));
            dst += 8L * stride;
            mask16x16 >>= 8;
            mask8x8 >>= 8;
            mask4x4 >>= 8;
            mask4x4Int >>= 8;
        }
    }

    /// <summary>lfl_uv row from lfl_y row: lfl_uv[c] = lfl_y[c &lt;&lt; 1], c = 0..3.</summary>
    private static ulong UvLflRow(ulong yRow) =>
        (yRow & 0xff) | ((yRow >> 8) & 0xff00) | ((yRow >> 16) & 0xff0000) | ((yRow >> 24) & 0xff000000);

    /// <summary>vp9_filter_block_plane_ss11: U or V of 4:2:0, the superblock's top-left chroma pixel at p[at].</summary>
    public static void FilterBlockPlaneSs11(ArrayView<byte> p, long at, int stride, int miRow, int miRows,
        ref Vp9LfMask m, ArrayView<int> lf, int lfOffset)
    {
        int thr = lfOffset + Vp9LfTable.Thr;
        long dst = at;
        ulong lflUv0 = 0, lflUv1 = 0, lflUv2 = 0, lflUv3 = 0;   // lfl_uv[16], 4 per uv row (vp9_zero)
        uint mask16x16 = m.LeftUv2, mask8x8 = m.LeftUv1, mask4x4 = m.LeftUv0, mask4x4Int = m.Int4x4Uv;

        // vertical pass: 2 rows at a time
        for (int r = 0; r < 8 && miRow + r < miRows; r += 4)
        {
            ulong row0 = UvLflRow(Vp9LfMask.Lfl(ref m, r)), row1 = UvLflRow(Vp9LfMask.Lfl(ref m, r + 2));
            if (r == 0) { lflUv0 = row0; lflUv1 = row1; }
            else { lflUv2 = row0; lflUv3 = row1; }
            FilterSelectivelyVertRow2(1, p, dst, stride, mask16x16, mask8x8, mask4x4, mask4x4Int, lf, thr, row0, row1);
            dst += 16L * stride;
            mask16x16 = (mask16x16 >> 8) & 0xffff;
            mask8x8 = (mask8x8 >> 8) & 0xffff;
            mask4x4 = (mask4x4 >> 8) & 0xffff;
            mask4x4Int = (mask4x4Int >> 8) & 0xffff;
        }

        // horizontal pass
        dst = at;
        mask16x16 = m.AboveUv2;
        mask8x8 = m.AboveUv1;
        mask4x4 = m.AboveUv0;
        mask4x4Int = m.Int4x4Uv;
        for (int r = 0; r < 8 && miRow + r < miRows; r += 2)
        {
            bool skipBorder4x4R = miRow + r == miRows - 1;
            uint m4i = skipBorder4x4R ? 0u : (mask4x4Int & 0xf);
            uint m16 = 0, m8 = 0, m4 = 0;
            if (miRow + r != 0)
            {
                m16 = mask16x16 & 0xf;
                m8 = mask8x8 & 0xf;
                m4 = mask4x4 & 0xf;
            }
            int uvRow = r >> 1;
            ulong lfl = uvRow == 0 ? lflUv0 : uvRow == 1 ? lflUv1 : uvRow == 2 ? lflUv2 : lflUv3;
            FilterSelectivelyHoriz(p, dst, stride, m16, m8, m4, m4i, lf, thr, lfl);
            dst += 8L * stride;
            mask16x16 >>= 4;
            mask8x8 >>= 4;
            mask4x4 >>= 4;
            mask4x4Int >>= 4;
        }
    }

    private static ulong SetByte(ulong v, int i, uint b) => (v & ~(0xffUL << (8 * i))) | ((ulong)(b & 0xff) << (8 * i));
    private static uint GetByte(ulong v, int i) => (uint)((v >> (8 * i)) & 0xff);

    /// <summary>
    /// vp9_filter_block_plane_non420: a chroma plane subsampled other than 4:2:0 / 4:4:4 (4:2:2, 4:4:0), masks built
    /// from the mode info directly; the superblock's top-left pixel of the plane at p[at].
    /// </summary>
    public static void FilterBlockPlaneNon420(ArrayView<byte> p, long at, int stride, int ssX, int ssY, ArrayView<int> mi,
        int gridOffset, int miRows, int miCols, int miRow, int miCol, ArrayView<int> lf, int lfOffset)
    {
        int thr = lfOffset + Vp9LfTable.Thr;
        int rowStep = 1 << ssY, colStep = 1 << ssX;
        // per mi row r (byte r): mask_16x16[r], mask_8x8[r], mask_4x4[r], mask_4x4_int[r]; lfl row r = lflRows byte
        ulong mask16Rows = 0, mask8Rows = 0, mask4Rows = 0, mask4IntRows = 0;
        Vp9LfMask lfl = default;
        long dst = at;
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
                        if (!skipBorder4x4R) mask16Rows |= (ulong)bit << (8 * r);
                        else mask8Rows |= (ulong)bit << (8 * r);
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
                        if (!skipBorder4x4R) mask16Rows |= (ulong)bit << (8 * r);
                        else mask8Rows |= (ulong)bit << (8 * r);
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
                        if (txSize == 1 || (rs & 3) == 0) mask8Rows |= (ulong)bit << (8 * r);
                        else mask4Rows |= (ulong)bit << (8 * r);
                    }
                    if (!skipThis && txSize < 1 && !skipBorder4x4C) mask4IntRows |= (ulong)bit << (8 * r);
                }
            }
            SetLflRow(ref lfl, r, lflRow);

            // no filtering on the leftmost column of the image
            uint borderMask = miCol == 0 ? ~1u : ~0u;
            FilterSelectivelyVert(p, dst, stride, mask16x16C & borderMask, mask8x8C & borderMask, mask4x4C & borderMask,
                GetByte(mask4IntRows, r), lf, thr, lflRow);
            dst += 8L * stride;
        }

        // horizontal pass
        dst = at;
        for (int r = 0; r < 8 && miRow + r < miRows; r += rowStep)
        {
            bool skipBorder4x4R = ssY != 0 && miRow + r == miRows - 1;
            uint mask4x4IntR = skipBorder4x4R ? 0u : GetByte(mask4IntRows, r);
            uint m16 = 0, m8 = 0, m4 = 0;
            if (miRow + r != 0)
            {
                m16 = GetByte(mask16Rows, r);
                m8 = GetByte(mask8Rows, r);
                m4 = GetByte(mask4Rows, r);
            }
            FilterSelectivelyHoriz(p, dst, stride, m16, m8, m4, mask4x4IntR, lf, thr, Vp9LfMask.Lfl(ref lfl, r));
            dst += 8L * stride;
        }
    }

    private static void SetLflRow(ref Vp9LfMask m, int r, ulong v)
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

    /// <summary>
    /// One iteration of libvpx loop_filter_rows' inner loop: the superblock at mi (<paramref name="miRow"/>,
    /// <paramref name="miCol"/>), all planes. <paramref name="yAt"/> is the index of its top-left luma pixel,
    /// <paramref name="uAt"/> / <paramref name="vAt"/> of its top-left chroma pixels (row (miRow * 8) &gt;&gt; ssY,
    /// column (miCol * 8) &gt;&gt; ssX of the plane). Does nothing when the frame's filter level is 0 (libvpx skips
    /// the whole frame). Superblocks run in raster order (see the file header).
    /// </summary>
    public static void FilterSuperblock(ArrayView<byte> y, long yAt, int yStride, ArrayView<byte> u, long uAt,
        ArrayView<byte> v, long vAt, int uvStride, int ssX, int ssY, ArrayView<int> mi, int gridOffset, int miRows,
        int miCols, int miRow, int miCol, ArrayView<int> lf, int lfOffset)
    {
        if (lf[lfOffset + Vp9LfTable.FrameLevel] == 0) return;
        Vp9LfMask m = default;
        BuildSuperblockMask(ref m, mi, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
        FilterBlockPlaneSs00(y, yAt, yStride, miRow, miRows, ref m, lf, lfOffset);
        if (ssX == 1 && ssY == 1)
        {
            FilterBlockPlaneSs11(u, uAt, uvStride, miRow, miRows, ref m, lf, lfOffset);
            FilterBlockPlaneSs11(v, vAt, uvStride, miRow, miRows, ref m, lf, lfOffset);
        }
        else if (ssX == 0 && ssY == 0)
        {
            FilterBlockPlaneSs00(u, uAt, uvStride, miRow, miRows, ref m, lf, lfOffset);
            FilterBlockPlaneSs00(v, vAt, uvStride, miRow, miRows, ref m, lf, lfOffset);
        }
        else
        {
            FilterBlockPlaneNon420(u, uAt, uvStride, ssX, ssY, mi, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
            FilterBlockPlaneNon420(v, vAt, uvStride, ssX, ssY, mi, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
        }
    }
}
