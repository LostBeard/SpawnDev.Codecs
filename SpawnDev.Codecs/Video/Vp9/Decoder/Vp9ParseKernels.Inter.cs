// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 parse kernel, inter mode info: reference frames, inter modes, interpolation filter, MV candidates
// (dec_find_mv_refs with libvpx's early break, append_sub8x8_mvs_for_idx), MV decoding and counts. Port of libvpx
// v1.17.0 vp9/decoder/vp9_decodemv.c + vp9/common/vp9_mvref_common.h + vp9/common/vp9_entropymv.c (BSD-3-Clause).

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

internal static partial class Vp9ParseKernels
{
    /// <summary>Block geometry for MV clamping (libvpx MACROBLOCKD mb_to_*_edge, 1/8 pel) and tile bounds.</summary>
    private struct BlockPos
    {
        public int MiRow, MiCol, MiColStart, MiColEnd, ToLeft, ToRight, ToTop, ToBottom;
    }

    /// <summary>Record index of the block at (miRow + dr, miCol + dc), -1 when libvpx's is_inside fails.</summary>
    private static int Candidate(ArrayView<int> mi, ArrayView<int> a, in BlockPos p, int dr, int dc)
    {
        int row = p.MiRow + dr, col = p.MiCol + dc;
        if (row < 0 || col < p.MiColStart || row >= a[Vp9PArg.MiRows] || col >= p.MiColEnd) return -1;
        return mi[a[Vp9PArg.GridOffset] + row * a[Vp9PArg.MiCols] + col];
    }

    private static int ClampMvRef(int mv, in BlockPos p)
    {
        int row = MvRow(mv), col = MvCol(mv);
        int minCol = p.ToLeft - MvBorder, maxCol = p.ToRight + MvBorder, minRow = p.ToTop - MvBorder, maxRow = p.ToBottom + MvBorder;
        col = col < minCol ? minCol : col > maxCol ? maxCol : col;
        row = row < minRow ? minRow : row > maxRow ? maxRow : row;
        return PackMv(row, col);
    }

    private static int NegateMv(int mv) => PackMv(-MvRow(mv), -MvCol(mv));

    /// <summary>scale_mv: the candidate's mv[which], negated when its reference has the other sign bias.</summary>
    private static int ScaleMv(ArrayView<int> mi, ArrayView<int> a, int cand, int which, int refFrame)
    {
        int mv = mi[cand * Vp9Mi.Stride + Vp9Mi.Mv0 + which];
        int candRef = mi[cand * Vp9Mi.Stride + Vp9Mi.Ref0 + which];
        return a[Vp9PArg.SignBias + candRef] != a[Vp9PArg.SignBias + refFrame] ? NegateMv(mv) : mv;
    }

    /// <summary>get_sub_block_mv.</summary>
    private static int SubBlockMv(ArrayView<int> mi, int cand, int which, int searchCol, int block)
    {
        int b = cand * Vp9Mi.Stride;
        if (block >= 0 && mi[b + Vp9Mi.SbType] < Block8x8)
        {
            // idx_n_column_to_subblock[block][search_col == 0]: {1,2},{1,3},{3,2},{3,3}
            int sub = searchCol == 0 ? (block == 0 ? 2 : block == 1 ? 3 : block == 2 ? 2 : 3) : (block <= 1 ? 1 : 3);
            return mi[b + (which == 0 ? Vp9Mi.BmiMv0 : Vp9Mi.BmiMv1) + sub];
        }
        return mi[b + Vp9Mi.Mv0 + which];
    }

    /// <summary>
    /// dec_find_mv_refs: up to 2 candidate MVs for refFrame into list0 / list1; returns the count libvpx returns
    /// (2 for NEARMV, else 1, or the early-break count). The list is clamped (clamp_mv_ref).
    /// </summary>
    private static int FindMvRefs(ArrayView<int> mi, ArrayView<int> maps, ArrayView<int> a, ArrayView<int> t, in BlockPos p,
        int bsize, int mode, int refFrame, int block, out int list0, out int list1)
    {
        list0 = 0;
        list1 = 0;
        int count = 0;
        bool differentRefFound = false;
        bool earlyBreak = mode != NearMv;
        int search = Vp9T.MvRefBlocks + bsize * 16;   // [8] {row, col}
        int prev = -1;
        if (a[Vp9PArg.UsePrevMvs] != 0) prev = a[Vp9PArg.PrevMvOffset] + (p.MiRow * a[Vp9PArg.MiCols] + p.MiCol) * Vp9PLayout.MvStride;
        int i = 0;
        bool done = false;
        if (block >= 0)
        {
            for (i = 0; i < 2 && !done; i++)
            {
                int dr = t[search + i * 2], dc = t[search + i * 2 + 1];
                int cand = Candidate(mi, a, p, dr, dc);
                if (cand < 0) continue;
                differentRefFound = true;
                int mv;
                if (Vp9Mi.Ref(mi, cand, 0) == refFrame) mv = SubBlockMv(mi, cand, 0, dc, block);
                else if (Vp9Mi.Ref(mi, cand, 1) == refFrame) mv = SubBlockMv(mi, cand, 1, dc, block);
                else continue;
                done = AddMv(mv, ref count, ref list0, ref list1, earlyBreak);
            }
            if (!done) i = 2;
        }
        for (; i < 8 && !done; i++)
        {
            int cand = Candidate(mi, a, p, t[search + i * 2], t[search + i * 2 + 1]);
            if (cand < 0) continue;
            differentRefFound = true;
            int mv;
            if (Vp9Mi.Ref(mi, cand, 0) == refFrame) mv = mi[cand * Vp9Mi.Stride + Vp9Mi.Mv0];
            else if (Vp9Mi.Ref(mi, cand, 1) == refFrame) mv = mi[cand * Vp9Mi.Stride + Vp9Mi.Mv1];
            else continue;
            done = AddMv(mv, ref count, ref list0, ref list1, earlyBreak);
        }
        if (!done && prev >= 0)
        {
            if (maps[prev] == refFrame) done = AddMv(maps[prev + 2], ref count, ref list0, ref list1, earlyBreak);
            else if (maps[prev + 1] == refFrame) done = AddMv(maps[prev + 3], ref count, ref list0, ref list1, earlyBreak);
        }
        if (!done && differentRefFound)
        {
            for (i = 0; i < 8 && !done; i++)
            {
                int cand = Candidate(mi, a, p, t[search + i * 2], t[search + i * 2 + 1]);
                if (cand < 0 || !Vp9Mi.IsInter(mi, cand)) continue;
                if (Vp9Mi.Ref(mi, cand, 0) != refFrame)
                    done = AddMv(ScaleMv(mi, a, cand, 0, refFrame), ref count, ref list0, ref list1, earlyBreak);
                if (!done && Vp9Mi.HasSecondRef(mi, cand) && Vp9Mi.Ref(mi, cand, 1) != refFrame &&
                    mi[cand * Vp9Mi.Stride + Vp9Mi.Mv1] != mi[cand * Vp9Mi.Stride + Vp9Mi.Mv0])
                    done = AddMv(ScaleMv(mi, a, cand, 1, refFrame), ref count, ref list0, ref list1, earlyBreak);
            }
        }
        if (!done && prev >= 0)
        {
            int r0 = maps[prev], r1 = maps[prev + 1], m0 = maps[prev + 2], m1 = maps[prev + 3];
            if (r0 != refFrame && r0 > Vp9Mi.IntraFrame)
            {
                int mv = a[Vp9PArg.SignBias + r0] != a[Vp9PArg.SignBias + refFrame] ? NegateMv(m0) : m0;
                done = AddMv(mv, ref count, ref list0, ref list1, earlyBreak);
            }
            if (!done && r1 > Vp9Mi.IntraFrame && r1 != refFrame && m1 != m0)
            {
                int mv = a[Vp9PArg.SignBias + r1] != a[Vp9PArg.SignBias + refFrame] ? NegateMv(m1) : m1;
                done = AddMv(mv, ref count, ref list0, ref list1, earlyBreak);
            }
        }
        if (!done) count = mode == NearMv ? 2 : 1;
        // Done: clamp the returned entries
        if (count > 0) list0 = ClampMvRef(list0, p);
        if (count > 1) list1 = ClampMvRef(list1, p);
        return count;
    }

    /// <summary>ADD_MV_REF_LIST_EB; true = "goto Done".</summary>
    private static bool AddMv(int mv, ref int count, ref int list0, ref int list1, bool earlyBreak)
    {
        if (count != 0)
        {
            if (mv != list0)
            {
                list1 = mv;
                count++;
                return true;
            }
            return false;
        }
        list0 = mv;
        count = 1;
        return earlyBreak;
    }

    private static bool UseMvHp(int mv)
    {
        int row = MvRow(mv), col = MvCol(mv);
        return (row < 0 ? -row : row) < 64 && (col < 0 ? -col : col) < 64;
    }

    private static int LowerMvPrecision(int mv, bool allowHp)
    {
        if (allowHp && UseMvHp(mv)) return mv;
        int row = MvRow(mv), col = MvCol(mv);
        if ((row & 1) != 0) row += row > 0 ? -1 : 1;
        if ((col & 1) != 0) col += col > 0 ? -1 : 1;
        return PackMv(row, col);
    }

    private static int ReadMvComponent(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, ArrayView<int> t, int comp, bool useHp)
    {
        int sign = Read(ref r, d, ctx[comp + Vp9Ctx.MvSign]);
        int mvClass = ReadTree(ref r, d, t, Vp9T.MvClassTree, ctx, comp + Vp9Ctx.MvClasses);
        bool class0 = mvClass == 0;
        int mag, dInt;
        if (class0)
        {
            dInt = Read(ref r, d, ctx[comp + Vp9Ctx.MvClass0]);
            mag = 0;
        }
        else
        {
            int n = mvClass + 1 - 1;   // mv_class + CLASS0_BITS - 1
            dInt = 0;
            for (int i = 0; i < n; i++) dInt |= Read(ref r, d, ctx[comp + Vp9Ctx.MvBits + i]) << i;
            mag = 2 << (mvClass + 2);   // CLASS0_SIZE << (mv_class + 2)
        }
        int fr = ReadTree(ref r, d, t, Vp9T.MvFpTree, ctx, class0 ? comp + Vp9Ctx.MvClass0Fp + dInt * 3 : comp + Vp9Ctx.MvFp);
        int hp = useHp ? Read(ref r, d, ctx[class0 ? comp + Vp9Ctx.MvClass0Hp : comp + Vp9Ctx.MvHp]) : 1;
        mag += ((dInt << 3) | (fr << 1) | hp) + 1;
        return sign != 0 ? -mag : mag;
    }

    private static void IncMvComponent(int v, ArrayView<int> counts, int comp)
    {
        int s = v < 0 ? 1 : 0;
        counts[comp + Vp9Cnt.MvSign + s]++;
        int z = (s != 0 ? -v : v) - 1;
        // vp9_get_mv_class
        int c;
        if (z >= 2 * 4096) c = 10;
        else
        {
            int x = z >> 3;
            c = x == 0 ? 0 : 31 - System.Numerics.BitOperations.LeadingZeroCount((uint)x);
        }
        int o = z - (c != 0 ? 2 << (c + 2) : 0);
        counts[comp + Vp9Cnt.MvClasses + c]++;
        int dd = o >> 3, f = (o >> 1) & 3, e = o & 1;
        if (c == 0)
        {
            counts[comp + Vp9Cnt.MvClass0 + dd]++;
            counts[comp + Vp9Cnt.MvClass0Fp + dd * 4 + f]++;
            counts[comp + Vp9Cnt.MvClass0Hp + e]++;   // usehp = 1 in vp9_inc_mv
        }
        else
        {
            int nb = c + 1 - 1;
            for (int i = 0; i < nb; i++) counts[comp + Vp9Cnt.MvBits + i * 2 + ((dd >> i) & 1)]++;
            counts[comp + Vp9Cnt.MvFp + f]++;
            counts[comp + Vp9Cnt.MvHp + e]++;
        }
    }

    /// <summary>read_mv: ref + the coded difference; counts with vp9_inc_mv.</summary>
    private static int ReadMv(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t, int refMv,
        bool allowHp, ArrayView<int> counts, int cnt)
    {
        int joint = ReadTree(ref r, d, t, Vp9T.MvJointTree, ctx, fc + Vp9Ctx.MvJoints);
        bool useHp = allowHp && UseMvHp(refMv);
        int dRow = 0, dCol = 0;
        if (joint == 2 || joint == 3) dRow = ReadMvComponent(ref r, d, ctx, t, fc + Vp9Ctx.MvComp, useHp);
        if (joint == 1 || joint == 3) dCol = ReadMvComponent(ref r, d, ctx, t, fc + Vp9Ctx.MvComp + Vp9Ctx.MvCompStride, useHp);
        if (cnt >= 0)
        {
            int j = dRow == 0 ? (dCol == 0 ? 0 : 1) : (dCol == 0 ? 2 : 3);
            counts[cnt + Vp9Cnt.MvJoints + j]++;
            if (j == 2 || j == 3) IncMvComponent(dRow, counts, cnt + Vp9Cnt.MvComp);
            if (j == 1 || j == 3) IncMvComponent(dCol, counts, cnt + Vp9Cnt.MvComp + Vp9Cnt.MvCompStride);
        }
        return PackMv(MvRow(refMv) + dRow, MvCol(refMv) + dCol);
    }

    private static bool IsMvValid(int mv)
    {
        int row = MvRow(mv), col = MvCol(mv);
        return row > MvLow && row < MvUpp && col > MvLow && col < MvUpp;
    }

    private static int ReadInterMode(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        int modeCtx, ArrayView<int> counts, int cnt)
    {
        int m = ReadTree(ref r, d, t, Vp9T.InterModeTree, ctx, fc + Vp9Ctx.InterMode + modeCtx * 3);
        if (cnt >= 0) counts[cnt + Vp9Cnt.InterMode + modeCtx * 4 + m]++;
        return NearestMv + m;
    }

    /// <summary>read_ref_frames into the record's Ref0 / Ref1.</summary>
    private static void ReadRefFrames(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> a,
        ArrayView<int> mi, int rec, int above, int left, int segId, ArrayView<int> results, ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride;
        if (SegFeatureActive(a, segId, SegLvlRefFrame))
        {
            mi[b + Vp9Mi.Ref0] = SegData(a, segId, SegLvlRefFrame);
            mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
            return;
        }
        int refMode = results[Vp9CompArgs.RRefMode];
        int compFixed = results[Vp9CompArgs.RCompFixed], compVar0 = results[Vp9CompArgs.RCompVar0], compVar1 = results[Vp9CompArgs.RCompVar1];
        int mode = refMode;   // SINGLE 0 / COMPOUND 1 / SELECT 2
        if (refMode == 2)
        {
            int c = Vp9ParseContexts.ReferenceModeContext(mi, above, left, compFixed);
            mode = Read(ref r, d, ctx[fc + Vp9Ctx.CompInter + c]);
            if (cnt >= 0) counts[cnt + Vp9Cnt.CompInter + c * 2 + mode]++;
        }
        if (mode == 1)
        {
            int idx = a[Vp9PArg.SignBias + compFixed];
            int c = Vp9ParseContexts.CompRefContext(mi, above, left, compFixed, compVar0, compVar1, idx);
            int bit = Read(ref r, d, ctx[fc + Vp9Ctx.CompRef + c]);
            if (cnt >= 0) counts[cnt + Vp9Cnt.CompRef + c * 2 + bit]++;
            mi[b + Vp9Mi.Ref0 + idx] = compFixed;
            mi[b + Vp9Mi.Ref0 + (idx == 0 ? 1 : 0)] = bit != 0 ? compVar1 : compVar0;
        }
        else
        {
            int c0 = Vp9ParseContexts.SingleRefP1Context(mi, above, left);
            int bit0 = Read(ref r, d, ctx[fc + Vp9Ctx.SingleRef + c0 * 2]);
            if (cnt >= 0) counts[cnt + Vp9Cnt.SingleRef + c0 * 4 + bit0]++;
            if (bit0 != 0)
            {
                int c1 = Vp9ParseContexts.SingleRefP2Context(mi, above, left);
                int bit1 = Read(ref r, d, ctx[fc + Vp9Ctx.SingleRef + c1 * 2 + 1]);
                if (cnt >= 0) counts[cnt + Vp9Cnt.SingleRef + c1 * 4 + 2 + bit1]++;
                mi[b + Vp9Mi.Ref0] = bit1 != 0 ? Vp9Mi.AltrefFrame : Vp9Mi.GoldenFrame;
            }
            else mi[b + Vp9Mi.Ref0] = Vp9Mi.LastFrame;
            mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
        }
    }

    /// <summary>append_sub8x8_mvs_for_idx: best_sub8x8 for sub-block j and reference ref.</summary>
    private static int AppendSub8x8Mvs(ArrayView<int> mi, ArrayView<int> maps, ArrayView<int> a, ArrayView<int> t, in BlockPos p,
        int rec, int bsize, int bMode, int block, int refIdx)
    {
        int b = rec * Vp9Mi.Stride;
        int bmi = b + (refIdx == 0 ? Vp9Mi.BmiMv0 : Vp9Mi.BmiMv1);
        int refFrame = mi[b + Vp9Mi.Ref0 + refIdx];
        int l0, l1;
        if (block == 0)
        {
            int n = FindMvRefs(mi, maps, a, t, p, bsize, bMode, refFrame, block, out l0, out l1);
            return n - 1 == 0 ? l0 : l1;
        }
        if (block == 1 || block == 2)
        {
            if (bMode == NearestMv) return mi[bmi];
            FindMvRefs(mi, maps, a, t, p, bsize, bMode, refFrame, block, out l0, out l1);
            if (mi[bmi] != l0) return l0;
            if (mi[bmi] != l1) return l1;
            return 0;
        }
        // block 3
        if (bMode == NearestMv) return mi[bmi + 2];
        if (mi[bmi + 2] != mi[bmi + 1]) return mi[bmi + 1];
        if (mi[bmi + 2] != mi[bmi]) return mi[bmi];
        FindMvRefs(mi, maps, a, t, p, bsize, bMode, refFrame, block, out l0, out l1);
        if (mi[bmi + 2] != l0) return l0;
        if (mi[bmi + 2] != l1) return l1;
        return 0;
    }

    /// <summary>read_inter_block_mode_info; returns false when an MV is out of range (libvpx xd->corrupted).</summary>
    private static bool ReadInterBlockModeInfo(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> a, ArrayView<int> mi, ArrayView<int> maps, ArrayView<int> results, in BlockPos p, int rec, int above, int left,
        int bsize, int bmodeWl, int bmodeHl, ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride;
        bool allowHp = a[Vp9PArg.AllowHp] != 0;
        int segId = mi[b + Vp9Mi.SegmentId];
        ReadRefFrames(ref r, d, ctx, fc, a, mi, rec, above, left, segId, results, counts, cnt);
        bool compound = Vp9Mi.HasSecondRef(mi, rec);
        // get_mode_context: from the first two candidates
        int search = Vp9T.MvRefBlocks + bsize * 16, counter = 0;
        for (int i = 0; i < 2; i++)
        {
            int cand = Candidate(mi, a, p, t[search + i * 2], t[search + i * 2 + 1]);
            if (cand >= 0) counter += t[Vp9T.Mode2Counter + mi[cand * Vp9Mi.Stride + Vp9Mi.Mode]];
        }
        int modeCtx = t[Vp9T.CounterToContext + counter];

        if (SegFeatureActive(a, segId, SegLvlSkip))
        {
            mi[b + Vp9Mi.Mode] = ZeroMv;
            if (bsize < Block8x8) return false;   // "Invalid usage of segment feature on small blocks"
        }
        else if (bsize >= Block8x8) mi[b + Vp9Mi.Mode] = ReadInterMode(ref r, d, ctx, fc, t, modeCtx, counts, cnt);

        int interp = a[Vp9PArg.Interp];
        if (interp == (int)Vp9Interp.Switchable)
        {
            int c = Vp9ParseContexts.SwitchableInterpContext(mi, above, left);
            interp = ReadTree(ref r, d, t, Vp9T.SwitchableInterpTree, ctx, fc + Vp9Ctx.SwitchableInterp + c * 2);
            if (cnt >= 0) counts[cnt + Vp9Cnt.SwitchableInterp + c * 3 + interp]++;
        }
        mi[b + Vp9Mi.Interp] = interp;

        bool ok = true;
        if (bsize < Block8x8)
        {
            int n4w = 1 << bmodeWl, n4h = 1 << bmodeHl;
            bool gotNew = false;
            int best0 = 0, best1 = 0, sub0 = 0, sub1 = unchecked((int)0x80008000);
            int bMode = 0;
            for (int idy = 0; idy < 2; idy += n4h)
                for (int idx = 0; idx < 2; idx += n4w)
                {
                    int j = idy * 2 + idx;
                    bMode = ReadInterMode(ref r, d, ctx, fc, t, modeCtx, counts, cnt);
                    if (bMode == NearestMv || bMode == NearMv)
                    {
                        sub0 = AppendSub8x8Mvs(mi, maps, a, t, p, rec, bsize, bMode, j, 0);
                        if (compound) sub1 = AppendSub8x8Mvs(mi, maps, a, t, p, rec, bsize, bMode, j, 1);
                    }
                    else if (bMode == NewMv && !gotNew)
                    {
                        for (int refIdx = 0; refIdx < 1 + (compound ? 1 : 0); refIdx++)
                        {
                            FindMvRefs(mi, maps, a, t, p, bsize, NewMv, mi[b + Vp9Mi.Ref0 + refIdx], -1, out int l0, out _);
                            int best = LowerMvPrecision(l0, allowHp);
                            if (refIdx == 0) best0 = best; else best1 = best;
                            gotNew = true;
                        }
                    }
                    // assign_mv into bmi[j]
                    int mv0, mv1;
                    if (bMode == NewMv)
                    {
                        mv0 = ReadMv(ref r, d, ctx, fc, t, best0, allowHp, counts, cnt);
                        ok &= IsMvValid(mv0);
                        mv1 = mi[b + Vp9Mi.BmiMv1 + j];
                        if (compound)
                        {
                            mv1 = ReadMv(ref r, d, ctx, fc, t, best1, allowHp, counts, cnt);
                            ok &= IsMvValid(mv1);
                        }
                    }
                    else if (bMode == NearMv || bMode == NearestMv) { mv0 = sub0; mv1 = sub1; }
                    else { mv0 = 0; mv1 = 0; }
                    mi[b + Vp9Mi.BmiMv0 + j] = mv0;
                    mi[b + Vp9Mi.BmiMv1 + j] = mv1;
                    mi[b + Vp9Mi.BmiMode + j] = bMode;
                    if (n4h == 2)
                    {
                        mi[b + Vp9Mi.BmiMv0 + j + 2] = mv0;
                        mi[b + Vp9Mi.BmiMv1 + j + 2] = mv1;
                        mi[b + Vp9Mi.BmiMode + j + 2] = bMode;
                    }
                    if (n4w == 2)
                    {
                        mi[b + Vp9Mi.BmiMv0 + j + 1] = mv0;
                        mi[b + Vp9Mi.BmiMv1 + j + 1] = mv1;
                        mi[b + Vp9Mi.BmiMode + j + 1] = bMode;
                    }
                    if (!ok) return false;
                }
            mi[b + Vp9Mi.Mode] = bMode;
            mi[b + Vp9Mi.Mv0] = mi[b + Vp9Mi.BmiMv0 + 3];
            mi[b + Vp9Mi.Mv1] = mi[b + Vp9Mi.BmiMv1 + 3];
            return true;
        }

        int mode = mi[b + Vp9Mi.Mode];
        int ref0 = 0, ref1 = 0;
        if (mode != ZeroMv)
        {
            for (int refIdx = 0; refIdx < 1 + (compound ? 1 : 0); refIdx++)
            {
                int n = FindMvRefs(mi, maps, a, t, p, bsize, mode, mi[b + Vp9Mi.Ref0 + refIdx], -1, out int l0, out int l1);
                int best = LowerMvPrecision(n - 1 == 0 ? l0 : l1, allowHp);
                if (refIdx == 0) ref0 = best; else ref1 = best;
            }
        }
        // assign_mv (near_nearest_mv = best_ref_mvs for blocks of 8x8 and up)
        if (mode == NewMv)
        {
            int mv0 = ReadMv(ref r, d, ctx, fc, t, ref0, allowHp, counts, cnt);
            ok &= IsMvValid(mv0);
            mi[b + Vp9Mi.Mv0] = mv0;
            if (compound)
            {
                int mv1 = ReadMv(ref r, d, ctx, fc, t, ref1, allowHp, counts, cnt);
                ok &= IsMvValid(mv1);
                mi[b + Vp9Mi.Mv1] = mv1;
            }
        }
        else if (mode == NearMv || mode == NearestMv)
        {
            mi[b + Vp9Mi.Mv0] = ref0;
            mi[b + Vp9Mi.Mv1] = ref1;
        }
        else
        {
            mi[b + Vp9Mi.Mv0] = 0;
            mi[b + Vp9Mi.Mv1] = 0;
        }
        return ok;
    }
}
