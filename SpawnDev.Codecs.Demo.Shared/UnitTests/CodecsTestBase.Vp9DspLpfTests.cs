// Tests for the VP9 loop filter (Vp9Lpf, port of libvpx v1.17.0 vpx_dsp/loopfilter.c + vp9/common/vp9_loopfilter.c).
//
// Fixtures (scripts/video/libvpx_ref/lpf_ref.c via make_lpf_fixture.py, TestData/Vp9Dsp): libvpx's own C on generated
// cases. lpf.gz: every vpx_lpf_* edge filter (4 / 8 / 16 taps, single / dual, horizontal / vertical) on flat, near-flat,
// stepped and noisy 24x24 neighbourhoods with limits derived from every level and sharpness, plus arbitrary bytes.
// lpf_sb.gz: whole superblocks exactly as libvpx's decoder filters them: random legal partitions (cut by the frame's
// right / bottom edge as decode_partition cuts them), tx sizes, skip, reference frames, modes and segments, random
// frame headers (filter level incl. 0, sharpness, ref / mode deltas, segment ALT_LF absolute / delta), 4:2:0, 4:4:4,
// 4:2:2 and 4:4:0; libvpx's vp9_loop_filter_frame_init tables, the masks after vp9_build_mask per block + vp9_adjust_mask,
// and the filtered pixels of the superblock plus the 8 columns / rows left of / above it.
// The pixels are regenerated here from the stored seed with the harness' generator (xorshift32 + gen_plane), checked
// against the harness' FNV-1a hash; the fixtures hold the outputs as (out - in) mod 256.
//
// What must hold: every pixel, every lf table entry libvpx's get_filter_level can read, and every mask equal to libvpx.

using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Vp9DspPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Vp9Dsp.";
    private const int LpfEdgeWindow = 24;                          // 24 x 24, pitch 24, the edge at row 8 / column 8
    private const int LpfEdgeBytes = LpfEdgeWindow * LpfEdgeWindow;
    private const int LpfEdgeArgs = 7;                             // fn, blimit0, limit0, thresh0, blimit1, limit1, thresh1
    private const int LpfSbMeta = 14;
    private const int LpfSbMaskWords = 26;

    // ---- the harness' pixel generator (lpf_ref.c: rnd32 / rnd / gen_plane / fnv), kept identical

    private sealed class LpfRng
    {
        private uint _s;
        public LpfRng(uint seed) => _s = seed != 0 ? seed : 1u;
        public uint Next()
        {
            _s ^= _s << 13;
            _s ^= _s >> 17;
            _s ^= _s << 5;
            return _s;
        }
        public int Rnd(int n) => (int)(Next() % (uint)n);
    }

    private static readonly int[] LpfGenD = { 0, 1, 3, 8, 24, 80 };
    private static readonly int[] LpfGenN = { 0, 0, 1, 2, 5 };

    private static void LpfGenPlane(LpfRng g, byte[] p, int off, int w, int h, int stride)
    {
        var b = new int[32, 32];
        int d = LpfGenD[g.Rnd(6)];
        int n = LpfGenN[g.Rnd(5)];
        int cell = g.Rnd(2) != 0 ? 8 : 4;
        int cw = (w + cell - 1) / cell, ch = (h + cell - 1) / cell;
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++)
            {
                int left = x != 0 ? b[y, x - 1] : y != 0 ? b[y - 1, x] : g.Rnd(256);
                int up = y != 0 ? b[y - 1, x] : left;
                int v = (left + up + 1) >> 1;
                if (d != 0) v += g.Rnd(2 * d + 1) - d;
                b[y, x] = Math.Clamp(v, 0, 255);
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int v = b[y / cell, x / cell];
                if (n != 0) v += g.Rnd(2 * n + 1) - n;
                p[off + y * stride + x] = (byte)Math.Clamp(v, 0, 255);
            }
    }

    private static uint LpfFnv(uint h, byte[] p, int off, int w, int hgt, int stride)
    {
        for (int y = 0; y < hgt; y++)
            for (int x = 0; x < w; x++)
            {
                h ^= p[off + y * stride + x];
                h *= 16777619u;
            }
        return h;
    }

    private static BinaryReader LpfFixture(string name, uint magic)
    {
        var r = new BinaryReader(new MemoryStream(Gunzip(ReadEmbedded(Vp9DspPrefix + name))));
        uint m = r.ReadUInt32();
        if (m != magic) throw new InvalidOperationException($"{name}: bad magic {m:x8}");
        return r;
    }

    // ---- edge filters

    private static void LpfEdgeKernel(Index1D i, ArrayView<byte> pix, ArrayView<int> c)
    {
        long at = (long)i.X * LpfEdgeBytes + 8 * LpfEdgeWindow + 8;
        int k = i * LpfEdgeArgs;
        int b0 = c[k + 1], l0 = c[k + 2], t0 = c[k + 3], b1 = c[k + 4], l1 = c[k + 5], t1 = c[k + 6];
        const int p = LpfEdgeWindow;
        switch (c[k])
        {
            case 0: Vp9Lpf.Horizontal4(pix, at, p, b0, l0, t0); break;
            case 1: Vp9Lpf.Horizontal4Dual(pix, at, p, b0, l0, t0, b1, l1, t1); break;
            case 2: Vp9Lpf.Vertical4(pix, at, p, b0, l0, t0); break;
            case 3: Vp9Lpf.Vertical4Dual(pix, at, p, b0, l0, t0, b1, l1, t1); break;
            case 4: Vp9Lpf.Horizontal8(pix, at, p, b0, l0, t0); break;
            case 5: Vp9Lpf.Horizontal8Dual(pix, at, p, b0, l0, t0, b1, l1, t1); break;
            case 6: Vp9Lpf.Vertical8(pix, at, p, b0, l0, t0); break;
            case 7: Vp9Lpf.Vertical8Dual(pix, at, p, b0, l0, t0, b1, l1, t1); break;
            case 8: Vp9Lpf.Horizontal16(pix, at, p, b0, l0, t0); break;
            case 9: Vp9Lpf.Horizontal16Dual(pix, at, p, b0, l0, t0); break;
            case 10: Vp9Lpf.Vertical16(pix, at, p, b0, l0, t0); break;
            default: Vp9Lpf.Vertical16Dual(pix, at, p, b0, l0, t0); break;
        }
    }

    private static readonly string[] LpfEdgeNames =
    {
        "horizontal_4", "horizontal_4_dual", "vertical_4", "vertical_4_dual", "horizontal_8", "horizontal_8_dual",
        "vertical_8", "vertical_8_dual", "horizontal_16", "horizontal_16_dual", "vertical_16", "vertical_16_dual",
    };

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Vp9Lpf_EdgeFilters_MatchLibvpx()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var r = LpfFixture("lpf.gz", 0x3046504c);
            int count = r.ReadInt32();
            var args = new int[count * LpfEdgeArgs];
            var pix = new byte[(long)count * LpfEdgeBytes];
            var want = new byte[pix.Length];
            var failures = new List<string>();
            int changed = 0;
            var perFn = new int[12];
            for (int i = 0; i < count; i++)
            {
                for (int k = 0; k < LpfEdgeArgs; k++) args[i * LpfEdgeArgs + k] = r.ReadInt32();
                uint seed = r.ReadUInt32(), hash = r.ReadUInt32();
                int off = i * LpfEdgeBytes;
                LpfGenPlane(new LpfRng(seed), pix, off, LpfEdgeWindow, LpfEdgeWindow, LpfEdgeWindow);
                if (LpfFnv(2166136261u, pix, off, LpfEdgeWindow, LpfEdgeWindow, LpfEdgeWindow) != hash)
                    throw new InvalidOperationException($"case {i}: the regenerated pixels differ from the harness' (hash)");
                byte[] delta = r.ReadBytes(LpfEdgeBytes);
                bool any = false;
                for (int k = 0; k < LpfEdgeBytes; k++)
                {
                    want[off + k] = (byte)(pix[off + k] + delta[k]);
                    any |= delta[k] != 0;
                }
                if (any) changed++;
                perFn[args[i * LpfEdgeArgs]]++;
            }

            using var dPix = acc.Allocate1D(pix);
            using var dArgs = acc.Allocate1D(args);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(LpfEdgeKernel);
            kernel(new Index1D(count), dPix.View, dArgs.View);
            await acc.SynchronizeAsync();
            var got = await dPix.CopyToHostAsync();

            int bad = 0;
            for (int i = 0; i < count; i++)
            {
                int off = i * LpfEdgeBytes;
                for (int k = 0; k < LpfEdgeBytes; k++)
                    if (got[off + k] != want[off + k])
                    {
                        bad++;
                        if (failures.Count < 12)
                        {
                            int a = i * LpfEdgeArgs;
                            failures.Add($"case {i} vpx_lpf_{LpfEdgeNames[args[a]]} (blimit/limit/thresh {args[a + 1]}/{args[a + 2]}/{args[a + 3]}, " +
                                         $"{args[a + 4]}/{args[a + 5]}/{args[a + 6]}): first difference at row {k / LpfEdgeWindow} col {k % LpfEdgeWindow}: " +
                                         $"got {got[off + k]}, libvpx {want[off + k]} (input {pix[off + k]})");
                        }
                        break;
                    }
            }
            Console.WriteLine($"{count} edge cases ({string.Join(", ", perFn.Select((n, f) => $"{LpfEdgeNames[f]} {n}"))}), " +
                              $"{changed} change pixels; {bad} differ from libvpx");
            True(bad == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // ---- superblocks

    private static void LpfFrameInitKernel(Index1D i, ArrayView<int> hdr, ArrayView<int> lf) =>
        Vp9Lpf.FrameInit(hdr, i * Vp9LfHdr.Ints, lf, i * Vp9LfTable.Ints);

    private static void LpfSbKernel(Index1D i, ArrayView<byte> pix, ArrayView<int> mi, ArrayView<int> lf, ArrayView<int> meta,
        ArrayView<ulong> masks)
    {
        int k = i * LpfSbMeta;
        int ssX = meta[k], ssY = meta[k + 1], miRows = meta[k + 2], miCols = meta[k + 3], miRow = meta[k + 4], miCol = meta[k + 5];
        var miv = mi.SubView(meta[k + 6], meta[k + 7]);
        int gridOffset = meta[k + 13];
        int lfOffset = i * Vp9LfTable.Ints;
        var m = default(Vp9LfMask);
        Vp9Lpf.BuildSuperblockMask(ref m, miv, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
        long o = (long)i.X * LpfSbMaskWords;
        masks[o] = m.LeftY0; masks[o + 1] = m.LeftY1; masks[o + 2] = m.LeftY2; masks[o + 3] = m.LeftY3;
        masks[o + 4] = m.AboveY0; masks[o + 5] = m.AboveY1; masks[o + 6] = m.AboveY2; masks[o + 7] = m.AboveY3;
        masks[o + 8] = m.Int4x4Y;
        masks[o + 9] = m.LeftUv0; masks[o + 10] = m.LeftUv1; masks[o + 11] = m.LeftUv2; masks[o + 12] = m.LeftUv3;
        masks[o + 13] = m.AboveUv0; masks[o + 14] = m.AboveUv1; masks[o + 15] = m.AboveUv2; masks[o + 16] = m.AboveUv3;
        masks[o + 17] = m.Int4x4Uv;
        masks[o + 18] = m.Lfl0; masks[o + 19] = m.Lfl1; masks[o + 20] = m.Lfl2; masks[o + 21] = m.Lfl3;
        masks[o + 22] = m.Lfl4; masks[o + 23] = m.Lfl5; masks[o + 24] = m.Lfl6; masks[o + 25] = m.Lfl7;
        // the superblock's top-left pixel sits 8 rows / columns into each window
        int pitchY = meta[k + 11], pitchUv = meta[k + 12];
        Vp9Lpf.FilterSuperblock(pix, meta[k + 8] + 8L * pitchY + 8, pitchY, pix, meta[k + 9] + 8L * pitchUv + 8,
            pix, meta[k + 10] + 8L * pitchUv + 8, pitchUv, ssX, ssY, miv, gridOffset, miRows, miCols, miRow, miCol, lf, lfOffset);
    }

    private static readonly string[] LpfMaskNames =
    {
        "left_y[0]", "left_y[1]", "left_y[2]", "left_y[3]", "above_y[0]", "above_y[1]", "above_y[2]", "above_y[3]", "int_4x4_y",
        "left_uv[0]", "left_uv[1]", "left_uv[2]", "left_uv[3]", "above_uv[0]", "above_uv[1]", "above_uv[2]", "above_uv[3]", "int_4x4_uv",
        "lfl_y row 0", "lfl_y row 1", "lfl_y row 2", "lfl_y row 3", "lfl_y row 4", "lfl_y row 5", "lfl_y row 6", "lfl_y row 7",
    };

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Vp9Lpf_Superblocks_MatchLibvpx()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var r = LpfFixture("lpf_sb.gz", 0x3042534c);
            int count = r.ReadInt32();
            var failures = new List<string>();

            // uv_txsize_lookup: the packed copy against libvpx's table
            byte[] uvTx = r.ReadBytes(13 * 4 * 4);
            for (int b = 0; b < 13; b++)
                for (int tx = 0; tx < 4; tx++)
                    for (int sx = 0; sx < 2; sx++)
                        for (int sy = 0; sy < 2; sy++)
                            if (Vp9Lpf.UvTxSize(b, tx, sx, sy) != uvTx[((b * 4 + tx) * 2 + sx) * 2 + sy])
                                failures.Add($"uv_txsize_lookup[{b}][{tx}][{sx}][{sy}]: {Vp9Lpf.UvTxSize(b, tx, sx, sy)}, libvpx {uvTx[((b * 4 + tx) * 2 + sx) * 2 + sy]}");

            var meta = new int[count * LpfSbMeta];
            var hdr = new int[count * Vp9LfHdr.Ints];
            var wantLvl = new byte[count][];
            var wantThr = new byte[count][];
            var wantMask = new ulong[count * LpfSbMaskWords];
            var pixList = new List<byte>();
            var wantPix = new List<byte>();
            var miList = new List<int>();
            int filtered = 0, changed = 0, cut = 0;
            var paths = new int[3];
            for (int i = 0; i < count; i++)
            {
                int k = i * LpfSbMeta;
                int ssX = r.ReadInt32(), ssY = r.ReadInt32(), miRows = r.ReadInt32(), miCols = r.ReadInt32(),
                    miRow = r.ReadInt32(), miCol = r.ReadInt32();
                uint seed = r.ReadUInt32(), hash = r.ReadUInt32();
                for (int h = 0; h < Vp9LfHdr.Ints; h++) hdr[i * Vp9LfHdr.Ints + h] = r.ReadInt32();
                wantLvl[i] = r.ReadBytes(64);
                wantThr[i] = r.ReadBytes(64 * 3);
                for (int w = 0; w < 9; w++) wantMask[i * LpfSbMaskWords + w] = r.ReadUInt64();
                for (int w = 9; w < 18; w++) wantMask[i * LpfSbMaskWords + w] = r.ReadUInt32();
                byte[] lfl = r.ReadBytes(64);
                for (int w = 0; w < 8; w++) wantMask[i * LpfSbMaskWords + 18 + w] = BitConverter.ToUInt64(lfl, w * 8);

                // mode info in the decoder's layout: records at mi_row * mi_cols + mi_col, the grid after them
                int miCount = miRows * miCols;
                int miBase = miList.Count, gridOffset = miCount * Vp9Mi.Stride;
                var mi = new int[miCount * (Vp9Mi.Stride + 1)];
                Array.Fill(mi, -1, gridOffset, miCount);
                int nblocks = r.ReadInt32();
                for (int bk = 0; bk < nblocks; bk++)
                {
                    byte[] f = r.ReadBytes(9);   // row, col, sb_type, mode, uv_mode, tx_size, skip, segment, ref
                    int row = miRow + f[0], col = miCol + f[1];
                    int rec = row * miCols + col, b = rec * Vp9Mi.Stride;
                    mi[b + Vp9Mi.SbType] = f[2];
                    mi[b + Vp9Mi.Mode] = f[3];
                    mi[b + Vp9Mi.UvMode] = f[4];
                    mi[b + Vp9Mi.TxSize] = f[5];
                    mi[b + Vp9Mi.Skip] = f[6];
                    mi[b + Vp9Mi.SegmentId] = f[7];
                    mi[b + Vp9Mi.Ref0] = f[8];
                    mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
                    int bw = Vp9Lpf.Num8x8Wide(f[2]), bh = Vp9Lpf.Num8x8High(f[2]);
                    for (int y = 0; y < bh && row + y < miRows; y++)
                        for (int x = 0; x < bw && col + x < miCols; x++) mi[gridOffset + (row + y) * miCols + col + x] = rec;
                }
                miList.AddRange(mi);

                // the pixel windows: 8 columns / rows left of / above the superblock + the superblock, pitch width + 8
                var g = new LpfRng(seed);
                uint hh = 2166136261u;
                bool any = false;
                int[] at = new int[3];
                int pitchUv = 0, pitchY = 0;
                for (int p = 0; p < 3; p++)
                {
                    int sx = p != 0 ? ssX : 0, sy = p != 0 ? ssY : 0;
                    int pw = 8 + (64 >> sx), ph = 8 + (64 >> sy), pitch = pw + 8;
                    if (p == 0) pitchY = pitch; else pitchUv = pitch;
                    var plane = new byte[pitch * ph];
                    LpfGenPlane(g, plane, 0, pw, ph, pitch);
                    hh = LpfFnv(hh, plane, 0, pw, ph, pitch);
                    at[p] = pixList.Count;
                    pixList.AddRange(plane);
                    var want = (byte[])plane.Clone();
                    byte[] delta = r.ReadBytes(pw * ph);
                    for (int y = 0; y < ph; y++)
                        for (int x = 0; x < pw; x++)
                        {
                            want[y * pitch + x] = (byte)(plane[y * pitch + x] + delta[y * pw + x]);
                            any |= delta[y * pw + x] != 0;
                        }
                    wantPix.AddRange(want);
                }
                if (hh != hash) throw new InvalidOperationException($"case {i}: the regenerated pixels differ from the harness' (hash)");
                if (any) changed++;
                if (hdr[i * Vp9LfHdr.Ints + Vp9LfHdr.Level] != 0)
                {
                    filtered++;
                    paths[ssX == 1 && ssY == 1 ? 0 : ssX == 0 && ssY == 0 ? 1 : 2]++;
                }
                if (miRow + 8 > miRows || miCol + 8 > miCols) cut++;
                meta[k] = ssX; meta[k + 1] = ssY; meta[k + 2] = miRows; meta[k + 3] = miCols; meta[k + 4] = miRow; meta[k + 5] = miCol;
                meta[k + 6] = miBase; meta[k + 7] = mi.Length; meta[k + 8] = at[0]; meta[k + 9] = at[1]; meta[k + 10] = at[2];
                meta[k + 11] = pitchY; meta[k + 12] = pitchUv; meta[k + 13] = gridOffset;
            }

            var pix = pixList.ToArray();
            using var dPix = acc.Allocate1D(pix);
            using var dMi = acc.Allocate1D(miList.ToArray());
            using var dHdr = acc.Allocate1D(hdr);
            using var dMeta = acc.Allocate1D(meta);
            using var dLf = acc.Allocate1D<int>((long)count * Vp9LfTable.Ints);
            using var dMasks = acc.Allocate1D<ulong>((long)count * LpfSbMaskWords);
            var init = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(LpfFrameInitKernel);
            var sb = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>>(LpfSbKernel);
            init(new Index1D(count), dHdr.View, dLf.View);
            sb(new Index1D(count), dPix.View, dMi.View, dLf.View, dMeta.View, dMasks.View);
            await acc.SynchronizeAsync();
            var got = await dPix.CopyToHostAsync();
            var lf = await dLf.CopyToHostAsync();
            var masks = await dMasks.CopyToHostAsync();

            int badTables = 0, badMasks = 0, badPix = 0;
            for (int i = 0; i < count; i++)
            {
                int k = i * LpfSbMeta, h = i * Vp9LfHdr.Ints;
                // libvpx only runs vp9_loop_filter_frame_init and builds masks when the frame level is not 0
                if (hdr[h + Vp9LfHdr.Level] != 0)
                {
                    bool deltas = hdr[h + Vp9LfHdr.DeltaEnabled] != 0;
                    for (int e = 0; e < 64; e++)
                    {
                        // lvl[seg][INTRA_FRAME][1]: never read; libvpx leaves it unchanged when deltas are enabled
                        if (deltas && (e & 7) == 1) continue;
                        int ours = lf[i * Vp9LfTable.Ints + Vp9LfTable.Lvl + e];
                        if (ours != wantLvl[i][e])
                        {
                            badTables++;
                            if (failures.Count < 30)
                                failures.Add($"case {i}: lvl[{e >> 3}][{(e >> 1) & 3}][{e & 1}] = {ours}, libvpx {wantLvl[i][e]}");
                        }
                    }
                    for (int e = 0; e < 64 * 3; e++)
                    {
                        int ours = lf[i * Vp9LfTable.Ints + Vp9LfTable.Thr + e];
                        if (ours != wantThr[i][e])
                        {
                            badTables++;
                            if (failures.Count < 30)
                                failures.Add($"case {i}: lfthr[{e / 3}].{(e % 3 == 0 ? "mblim" : e % 3 == 1 ? "lim" : "hev_thr")} = {ours}, libvpx {wantThr[i][e]}");
                        }
                    }
                    for (int w = 0; w < LpfSbMaskWords; w++)
                    {
                        ulong ours = masks[i * LpfSbMaskWords + w], theirs = wantMask[i * LpfSbMaskWords + w];
                        if (ours != theirs)
                        {
                            badMasks++;
                            if (failures.Count < 30)
                                failures.Add($"case {i} (ss {meta[k]}{meta[k + 1]}, mi {meta[k + 2]}x{meta[k + 3]} at {meta[k + 4]},{meta[k + 5]}): " +
                                             $"{LpfMaskNames[w]} = {ours:x16}, libvpx {theirs:x16}");
                        }
                    }
                }
                int end = i + 1 < count ? meta[k + LpfSbMeta + 8] : pix.Length;
                for (int o = meta[k + 8]; o < end; o++)
                    if (got[o] != wantPix[o])
                    {
                        badPix++;
                        if (failures.Count < 30)
                        {
                            int p = o >= meta[k + 10] ? 2 : o >= meta[k + 9] ? 1 : 0;
                            int pitch = p == 0 ? meta[k + 11] : meta[k + 12];
                            int rel = o - meta[k + 8 + p];
                            failures.Add($"case {i} (ss {meta[k]}{meta[k + 1]}, mi {meta[k + 2]}x{meta[k + 3]} at {meta[k + 4]},{meta[k + 5]}, " +
                                         $"level {hdr[h + Vp9LfHdr.Level]}): plane {p} first difference at window x {rel % pitch} y {rel / pitch} " +
                                         $"(superblock x {rel % pitch - 8} y {rel / pitch - 8}): got {got[o]}, libvpx {wantPix[o]} (input {pix[o]})");
                        }
                        break;
                    }
            }
            Console.WriteLine($"{count} superblocks: {filtered} filtered (4:2:0 {paths[0]}, 4:4:4 {paths[1]}, 4:2:2 / 4:4:0 {paths[2]}), " +
                              $"{changed} with pixels changed, {cut} cut by the frame edge; differ from libvpx: {badPix} superblocks' pixels, " +
                              $"{badMasks} mask words, {badTables} table entries");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
