// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 parse kernel, inter mode info: reference frames, inter modes, interpolation filter, MV candidates
// (dec_find_mv_refs with libvpx's early break, append_sub8x8_mvs_for_idx), MV decoding and counts. Port of libvpx
// v1.17.0 vp9/decoder/vp9_decodemv.c + vp9/common/vp9_mvref_common.h + vp9/common/vp9_entropymv.c (BSD-3-Clause).
// Every helper here has ONE call site in the kernel (see ReadSym): the sub-8x8 sub-blocks, the references and the MV
// components are loop iterations, not repeated calls.

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

internal static partial class Vp9ParseKernels
{
    /// <summary>Block geometry for MV clamping (libvpx MACROBLOCKD mb_to_*_edge, 1/8 pel) and tile bounds.</summary>
    internal struct BlockPos
    {
        public int MiRow, MiCol, MiColStart, MiColEnd, ToLeft, ToRight, ToTop, ToBottom;
    }

    /// <summary>Record index of the block at (miRow + dr, miCol + dc), -1 when libvpx's is_inside fails.</summary>
    internal static int Candidate(ArrayView<int> mi, ArrayView<int> a, in BlockPos p, int dr, int dc)
    {
        int row = p.MiRow + dr, col = p.MiCol + dc;
        int cand = -1;
        if (row >= 0 & col >= p.MiColStart & row < a[Vp9PArg.MiRows] & col < p.MiColEnd) cand = mi[a[Vp9PArg.GridOffset] + row * a[Vp9PArg.MiCols] + col];
        return cand;
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

    /// <summary>get_sub_block_mv.</summary>
    private static int SubBlockMv(ArrayView<int> mi, int cand, int which, int searchCol, int block)
    {
        int b = cand * Vp9Mi.Stride;
        int at = b + Vp9Mi.Mv0 + which;
        if (block >= 0 & mi[b + Vp9Mi.SbType] < Block8x8)
        {
            // idx_n_column_to_subblock[block][search_col == 0]: {1,2},{1,3},{3,2},{3,3}
            int sub = searchCol == 0 ? (block == 0 ? 2 : block == 1 ? 3 : block == 2 ? 2 : 3) : (block <= 1 ? 1 : 3);
            at = b + (which == 0 ? Vp9Mi.BmiMv0 : Vp9Mi.BmiMv1) + sub;
        }
        return mi[at];
    }

    /// <summary>
    /// dec_find_mv_refs: up to 2 candidate MVs for refFrame into list0 / list1; returns the count libvpx returns
    /// (2 for NEARMV, else 1, or the early-break count). The list is clamped (clamp_mv_ref).
    /// libvpx's four searches are one stream of candidates, in its order, with one ADD_MV_REF_LIST_EB site:
    ///   0-7   the spatial neighbours (mv_ref_blocks) with refFrame, the first two through their sub-block MVs (block >= 0);
    ///   8     the previous frame's MV with refFrame;
    ///   9-24  (neighbour (s - 9) / 2, its mv[(s - 9) % 2]) with a different reference, scaled (only if a neighbour was found);
    ///   25-26 the previous frame's mv[s - 25] with a different reference, scaled.
    /// </summary>
    internal static int FindMvRefs(ArrayView<int> mi, ArrayView<int> maps, ArrayView<int> a, ArrayView<int> t, in BlockPos p,
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
        bool done = false;
        for (int s = 0; s < 27 & !done; s++)
        {
            bool spatial = s < 8 | (s >= 9 & s < 25);
            if (spatial ? s >= 9 & !differentRefFound : prev < 0) continue;
            int candRef, candMv, which;
            bool add;
            if (spatial)
            {
                int i = s < 8 ? s : (s - 9) >> 1;
                int dc = t[search + i * 2 + 1];
                int cand = Candidate(mi, a, p, t[search + i * 2], dc);
                if (cand < 0) continue;
                int cb = cand * Vp9Mi.Stride;
                if (s < 8)
                {
                    differentRefFound = true;
                    which = mi[cb + Vp9Mi.Ref0] == refFrame ? 0 : 1;
                    candRef = mi[cb + Vp9Mi.Ref0 + which];
                    add = candRef == refFrame;
                    candMv = SubBlockMv(mi, cand, which, dc, i < 2 ? block : -1);
                }
                else
                {
                    which = (s - 9) & 1;
                    candRef = mi[cb + Vp9Mi.Ref0 + which];
                    candMv = mi[cb + Vp9Mi.Mv0 + which];
                    // IF_DIFF_REF_FRAME_ADD_MV: mv[0] when its reference differs; mv[1] when the block is compound, its
                    // second reference differs and its MVs differ
                    add = mi[cb + Vp9Mi.Ref0] > Vp9Mi.IntraFrame & candRef != refFrame
                        & (which == 0 | (candRef > Vp9Mi.IntraFrame & candMv != mi[cb + Vp9Mi.Mv0]));
                }
            }
            else
            {
                if (s == 8) which = maps[prev] == refFrame ? 0 : 1;
                else which = s - 25;
                candRef = maps[prev + which];
                candMv = maps[prev + 2 + which];
                add = s == 8 ? candRef == refFrame
                    : candRef > Vp9Mi.IntraFrame & candRef != refFrame & (which == 0 | candMv != maps[prev + 2]);
            }
            if (!add) continue;
            // scale_mv: negated when the candidate's reference has the other sign bias
            if (s >= 9 & a[Vp9PArg.SignBias + candRef] != a[Vp9PArg.SignBias + refFrame]) candMv = NegateMv(candMv);
            // ADD_MV_REF_LIST_EB ("goto Done" = done)
            if (count != 0)
            {
                if (candMv != list0)
                {
                    list1 = candMv;
                    count++;
                    done = true;
                }
            }
            else
            {
                list0 = candMv;
                count = 1;
                done = earlyBreak;
            }
        }
        if (!done) count = mode == NearMv ? 2 : 1;
        // Done: clamp the returned entries
        if (count > 0) list0 = ClampMvRef(list0, p);
        if (count > 1) list1 = ClampMvRef(list1, p);
        return count;
    }

    private static bool UseMvHp(int mv)
    {
        int row = MvRow(mv), col = MvCol(mv);
        return (row < 0 ? -row : row) < 64 & (col < 0 ? -col : col) < 64;
    }

    internal static int LowerMvPrecision(int mv, bool allowHp)
    {
        int row = MvRow(mv), col = MvCol(mv);
        if (!(allowHp & UseMvHp(mv)))
        {
            if ((row & 1) != 0) row += row > 0 ? -1 : 1;
            if ((col & 1) != 0) col += col > 0 ? -1 : 1;
        }
        return PackMv(row, col);
    }

    private static int ReadMvComponent(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, ArrayView<int> t, ArrayView<int> a,
        int comp, bool useHp)
    {
        int sign = Read(ref r, d, ctx[comp + Vp9Ctx.MvSign]);
        int mvClass = ReadSym(ref r, d, t, ctx, a, Vp9T.MvClassTree, SrcCtx, comp + Vp9Ctx.MvClasses);
        bool class0 = mvClass == 0;
        // the integer part: class 0 one bit (class0 probability), class c c bits (bits probabilities), LSB first
        int n = class0 ? 1 : mvClass;   // mv_class + CLASS0_BITS - 1
        int dInt = 0;
        for (int i = 0; i < n; i++) dInt |= Read(ref r, d, ctx[class0 ? comp + Vp9Ctx.MvClass0 : comp + Vp9Ctx.MvBits + i]) << i;
        int mag = class0 ? 0 : 2 << (mvClass + 2);   // CLASS0_SIZE << (mv_class + 2)
        int fr = ReadSym(ref r, d, t, ctx, a, Vp9T.MvFpTree, SrcCtx, class0 ? comp + Vp9Ctx.MvClass0Fp + dInt * 3 : comp + Vp9Ctx.MvFp);
        int hp = useHp ? Read(ref r, d, ctx[class0 ? comp + Vp9Ctx.MvClass0Hp : comp + Vp9Ctx.MvHp]) : 1;
        mag += ((dInt << 3) | (fr << 1) | hp) + 1;
        return sign != 0 ? -mag : mag;
    }

    internal static void IncMvComponent(int v, ArrayView<int> counts, int comp)
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

    /// <summary>read_mv: ref + the coded difference; counts with vp9_inc_mv. The components are a loop (row, then col).</summary>
    private static int ReadMv(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t, ArrayView<int> a,
        int refMv, bool allowHp, ArrayView<int> counts, int cnt)
    {
        int joint = ReadSym(ref r, d, t, ctx, a, Vp9T.MvJointTree, SrcCtx, fc + Vp9Ctx.MvJoints);
        bool useHp = allowHp & UseMvHp(refMv);
        int dRow = 0, dCol = 0;
        // MV_JOINT_HNZVZ 1 (col), MV_JOINT_HZVNZ 2 (row), MV_JOINT_HNZVNZ 3 (both): bit 1 = row, bit 0 = col
        for (int k = 0; k < 2; k++)
            if ((joint & (2 >> k)) != 0)
            {
                int v = ReadMvComponent(ref r, d, ctx, t, a, fc + Vp9Ctx.MvComp + k * Vp9Ctx.MvCompStride, useHp);
                if (k == 0) dRow = v; else dCol = v;
            }
        if (cnt >= 0)
        {
            int j = (dRow != 0 ? 2 : 0) | (dCol != 0 ? 1 : 0);   // vp9_get_mv_joint
            counts[cnt + Vp9Cnt.MvJoints + j]++;
            for (int k = 0; k < 2; k++)
                if ((j & (2 >> k)) != 0) IncMvComponent(k == 0 ? dRow : dCol, counts, cnt + Vp9Cnt.MvComp + k * Vp9Cnt.MvCompStride);
        }
        return PackMv(MvRow(refMv) + dRow, MvCol(refMv) + dCol);
    }

    private static bool IsMvValid(int mv)
    {
        int row = MvRow(mv), col = MvCol(mv);
        return row > MvLow & row < MvUpp & col > MvLow & col < MvUpp;
    }

    /// <summary>
    /// read_ref_frames into the record's Ref0 / Ref1: its bools (comp_inter, comp_ref, single_ref p1 / p2) are steps of
    /// one loop over one bool read.
    /// </summary>
    private static void ReadRefFrames(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> a,
        ArrayView<int> mi, int rec, int above, int left, int segId, ArrayView<int> results, ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride;
        bool segRef = SegFeatureActive(a, segId, SegLvlRefFrame);
        if (segRef)
        {
            mi[b + Vp9Mi.Ref0] = SegData(a, segId, SegLvlRefFrame);
            mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
        }
        int refMode = results[Vp9CompArgs.RRefMode];   // SINGLE 0 / COMPOUND 1 / SELECT 2
        int compFixed = results[Vp9CompArgs.RCompFixed], compVar0 = results[Vp9CompArgs.RCompVar0], compVar1 = results[Vp9CompArgs.RCompVar1];
        // step 0: comp_inter (REFERENCE_MODE_SELECT); 1: comp_ref; 2: single_ref p1; 3: single_ref p2; 4 done (the
        // segment's reference: nothing read; no early return, see ReadInterBlockModeInfo)
        int step = segRef ? 4 : refMode == 2 ? 0 : refMode == 1 ? 1 : 2;
        int idx = refMode != 0 ? a[Vp9PArg.SignBias + compFixed] : 0;   // compound: the fixed reference's slot (its sign bias)
        while (step < 4)
        {
            int c, probAt, cntAt;
            if (step == 0)
            {
                c = Vp9ParseContexts.ReferenceModeContext(mi, above, left, compFixed);
                probAt = Vp9Ctx.CompInter + c;
                cntAt = Vp9Cnt.CompInter + c * 2;
            }
            else if (step == 1)
            {
                c = Vp9ParseContexts.CompRefContext(mi, above, left, compFixed, compVar0, compVar1, idx);
                probAt = Vp9Ctx.CompRef + c;
                cntAt = Vp9Cnt.CompRef + c * 2;
            }
            else if (step == 2)
            {
                c = Vp9ParseContexts.SingleRefP1Context(mi, above, left);
                probAt = Vp9Ctx.SingleRef + c * 2;
                cntAt = Vp9Cnt.SingleRef + c * 4;
            }
            else
            {
                c = Vp9ParseContexts.SingleRefP2Context(mi, above, left);
                probAt = Vp9Ctx.SingleRef + c * 2 + 1;
                cntAt = Vp9Cnt.SingleRef + c * 4 + 2;
            }
            int bit = Read(ref r, d, ctx[fc + probAt]);
            if (cnt >= 0) counts[cnt + cntAt + bit]++;
            if (step == 0) step = bit != 0 ? 1 : 2;
            else if (step == 1)
            {
                mi[b + Vp9Mi.Ref0 + idx] = compFixed;
                mi[b + Vp9Mi.Ref0 + (idx == 0 ? 1 : 0)] = bit != 0 ? compVar1 : compVar0;
                step = 4;
            }
            else if (step == 2 && bit != 0) step = 3;
            else
            {
                mi[b + Vp9Mi.Ref0] = step == 2 ? Vp9Mi.LastFrame : bit != 0 ? Vp9Mi.AltrefFrame : Vp9Mi.GoldenFrame;
                mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
                step = 4;
            }
        }
    }

    /// <summary>
    /// read_inter_block_mode_info; returns false when an MV is out of range (libvpx xd->corrupted). Below 8x8 one pass
    /// per sub-block (the 4x8 / 8x4 halves copied), from 8x8 up a single pass (block -1); each pass runs, per reference,
    /// dec_find_mv_refs (for the NEWMV best reference or append_sub8x8_mvs_for_idx's search) and then assign_mv, each
    /// at one call site.
    /// </summary>
    private static bool ReadInterBlockModeInfo(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> a, ArrayView<int> mi, ArrayView<int> maps, ArrayView<int> results, in BlockPos p, int rec, int above, int left,
        int bsize, int bmodeWl, int bmodeHl, ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride;
        bool allowHp = a[Vp9PArg.AllowHp] != 0;
        int segId = mi[b + Vp9Mi.SegmentId];
        ReadRefFrames(ref r, d, ctx, fc, a, mi, rec, above, left, segId, results, counts, cnt);
        bool compound = Vp9Mi.HasSecondRef(mi, rec);
        int refs = compound ? 2 : 1;
        // get_mode_context: from the first two candidates
        int search = Vp9T.MvRefBlocks + bsize * 16, counter = 0;
        for (int i = 0; i < 2; i++)
        {
            int cand = Candidate(mi, a, p, t[search + i * 2], t[search + i * 2 + 1]);
            if (cand >= 0) counter += t[Vp9T.Mode2Counter + mi[cand * Vp9Mi.Stride + Vp9Mi.Mode]];
        }
        int modeCtx = t[Vp9T.CounterToContext + counter];
        bool sub8 = bsize < Block8x8;

        // The block's inter mode (8x8 and up, before the filter) and the switchable interp filter, as steps of one loop
        // over one tree read; below 8x8 the modes are read per sub-block below.
        // No early returns here: inlined, an early exit makes WGSL's structured control flow repeat the code after it.
        bool segSkip = SegFeatureActive(a, segId, SegLvlSkip);
        if (segSkip) mi[b + Vp9Mi.Mode] = ZeroMv;
        bool ok = !(segSkip & sub8);   // "Invalid usage of segment feature on small blocks": corrupt, nothing more is read
        int interp = a[Vp9PArg.Interp];
        for (int step = !ok ? 2 : segSkip | sub8 ? 1 : 0; step < 2; step++)
        {
            if (step == 1 && interp != (int)Vp9Interp.Switchable) break;
            int c = step == 0 ? modeCtx : Vp9ParseContexts.SwitchableInterpContext(mi, above, left);
            int v = ReadSym(ref r, d, t, ctx, a, step == 0 ? Vp9T.InterModeTree : Vp9T.SwitchableInterpTree, SrcCtx,
                fc + (step == 0 ? Vp9Ctx.InterMode + c * 3 : Vp9Ctx.SwitchableInterp + c * 2));
            if (cnt >= 0) counts[cnt + (step == 0 ? Vp9Cnt.InterMode + c * 4 : Vp9Cnt.SwitchableInterp + c * 3) + v]++;
            if (step == 0) mi[b + Vp9Mi.Mode] = NearestMv + v;
            else interp = v;
        }
        if (ok) mi[b + Vp9Mi.Interp] = interp;

        int n4w = sub8 ? 1 << bmodeWl : 2, n4h = sub8 ? 1 << bmodeHl : 2;
        bool gotNew = false;
        int best0 = 0, best1 = 0, sub0 = 0, sub1 = unchecked((int)0x80008000);
        int bMode = mi[b + Vp9Mi.Mode];
        // the passes stop after the first out-of-range MV (ok false): libvpx returns there
        for (int idy = 0; idy < 2 & ok; idy += n4h)
            for (int idx = 0; idx < 2 & ok; idx += n4w)
            {
                int j = idy * 2 + idx;
                if (sub8)
                {
                    int v = ReadSym(ref r, d, t, ctx, a, Vp9T.InterModeTree, SrcCtx, fc + Vp9Ctx.InterMode + modeCtx * 3);
                    if (cnt >= 0) counts[cnt + Vp9Cnt.InterMode + modeCtx * 4 + v]++;
                    bMode = NearestMv + v;
                }
                bool nearOrNearest = bMode == NearestMv | bMode == NearMv;
                for (int refIdx = 0; refIdx < refs; refIdx++)
                {
                    int bmi = b + (refIdx == 0 ? Vp9Mi.BmiMv0 : Vp9Mi.BmiMv1);
                    // 8x8 up: the best reference of any mode but ZEROMV; below 8x8: NEWMV's best reference once per block,
                    // NEAREST / NEAR append_sub8x8_mvs_for_idx, which searches only when the earlier sub-blocks do not decide
                    bool need;
                    if (!sub8) need = bMode != ZeroMv;
                    else if (bMode == NewMv) need = !gotNew;
                    else need = nearOrNearest
                        & (j == 0 | (bMode != NearestMv & (j < 3 | (mi[bmi + 2] == mi[bmi + 1] & mi[bmi + 2] == mi[bmi]))));
                    int n = 0, l0 = 0, l1 = 0;
                    if (need)
                        n = FindMvRefs(mi, maps, a, t, p, bsize, bMode, mi[b + Vp9Mi.Ref0 + refIdx], sub8 & nearOrNearest ? j : -1, out l0, out l1);
                    if (sub8 & nearOrNearest)
                    {
                        // append_sub8x8_mvs_for_idx's result
                        int cmp = mi[bmi + (j == 3 ? 2 : 0)];
                        int v = j == 0 ? (n - 1 == 0 ? l0 : l1)
                            : !need ? (bMode == NearestMv ? cmp : mi[bmi + 2] != mi[bmi + 1] ? mi[bmi + 1] : mi[bmi])
                            : cmp != l0 ? l0 : cmp != l1 ? l1 : 0;
                        if (refIdx == 0) sub0 = v; else sub1 = v;
                    }
                    else if (need)
                    {
                        int best = LowerMvPrecision(n - 1 == 0 ? l0 : l1, allowHp);
                        if (refIdx == 0) best0 = best; else best1 = best;
                    }
                }
                if (sub8 & bMode == NewMv) gotNew = true;
                // assign_mv (near_nearest_mv = best_ref_mvs from 8x8 up, the sub-block's candidates below)
                int mv0 = 0, mv1 = 0;
                if (bMode == NewMv)
                {
                    mv1 = mi[b + (sub8 ? Vp9Mi.BmiMv1 + j : Vp9Mi.Mv1)];   // unchanged unless compound
                    for (int refIdx = 0; refIdx < refs; refIdx++)
                    {
                        int mv = ReadMv(ref r, d, ctx, fc, t, a, refIdx == 0 ? best0 : best1, allowHp, counts, cnt);
                        ok &= IsMvValid(mv);
                        if (refIdx == 0) mv0 = mv; else mv1 = mv;
                    }
                }
                else if (nearOrNearest)
                {
                    mv0 = sub8 ? sub0 : best0;
                    mv1 = sub8 ? sub1 : best1;
                }
                if (sub8)
                {
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
                }
                else
                {
                    mi[b + Vp9Mi.Mv0] = mv0;
                    mi[b + Vp9Mi.Mv1] = mv1;
                }
            }
        if (sub8 & ok)
        {
            mi[b + Vp9Mi.Mode] = bMode;
            mi[b + Vp9Mi.Mv0] = mi[b + Vp9Mi.BmiMv0 + 3];
            mi[b + Vp9Mi.Mv1] = mi[b + Vp9Mi.BmiMv1 + 3];
        }
        return ok;
    }
}
