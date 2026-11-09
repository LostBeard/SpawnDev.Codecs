// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 mode info records on the accelerator + the entropy contexts derived from the neighbouring blocks. A port of
// libvpx v1.17.0 vp9/common/vp9_pred_common.h / .c (BSD-3-Clause): segment id prediction, skip, switchable
// interpolation, intra / inter, reference mode, compound and single reference, and tx size contexts.
// "above" / "left" are record indices of the neighbouring blocks, -1 where libvpx's above_mi / left_mi is NULL
// (above: first mi row of the frame; left: first mi column of the tile).

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>
/// Layout of one block's mode info (libvpx MODE_INFO) in the mode info array: <see cref="Stride"/> ints per record,
/// stored at the block's top-left mi offset (mi_row * mi_cols + mi_col). MVs are libvpx int_mv.as_int
/// (row in the low 16 bits, col in the high 16 bits).
/// </summary>
internal static class Vp9Mi
{
    public const int SbType = 0, Mode = 1, UvMode = 2, TxSize = 3, Skip = 4, SegmentId = 5, SegIdPredicted = 6,
        Ref0 = 7, Ref1 = 8, Interp = 9, Mv0 = 10, Mv1 = 11, BmiMode = 12, BmiMv0 = 16, BmiMv1 = 20;
    public const int Stride = 24;

    // libvpx enums
    public const int IntraFrame = 0, LastFrame = 1, GoldenFrame = 2, AltrefFrame = 3, NoRefFrame = -1;
    public const int SwitchableFilters = 3;   // interp_filter value of intra blocks (no filter)
    public const int Block8x8 = 3;            // BLOCK_8X8

    public static bool IsInter(ArrayView<int> mi, int b) => mi[b * Stride + Ref0] > IntraFrame;
    public static bool HasSecondRef(ArrayView<int> mi, int b) => mi[b * Stride + Ref1] > IntraFrame;
    public static int Ref(ArrayView<int> mi, int b, int i) => mi[b * Stride + Ref0 + i];
}

/// <summary>Entropy contexts from the neighbouring blocks (libvpx vp9_pred_common).</summary>
internal static class Vp9ParseContexts
{
    public static int SegIdPredContext(ArrayView<int> mi, int above, int left) =>
        (above >= 0 ? mi[above * Vp9Mi.Stride + Vp9Mi.SegIdPredicted] : 0) + (left >= 0 ? mi[left * Vp9Mi.Stride + Vp9Mi.SegIdPredicted] : 0);

    public static int SkipContext(ArrayView<int> mi, int above, int left) =>
        (above >= 0 ? mi[above * Vp9Mi.Stride + Vp9Mi.Skip] : 0) + (left >= 0 ? mi[left * Vp9Mi.Stride + Vp9Mi.Skip] : 0);

    public static int SwitchableInterpContext(ArrayView<int> mi, int above, int left)
    {
        int leftType = left >= 0 ? mi[left * Vp9Mi.Stride + Vp9Mi.Interp] : Vp9Mi.SwitchableFilters;
        int aboveType = above >= 0 ? mi[above * Vp9Mi.Stride + Vp9Mi.Interp] : Vp9Mi.SwitchableFilters;
        if (leftType == aboveType) return leftType;
        if (leftType == Vp9Mi.SwitchableFilters) return aboveType;
        if (aboveType == Vp9Mi.SwitchableFilters) return leftType;
        return Vp9Mi.SwitchableFilters;
    }

    public static int IntraInterContext(ArrayView<int> mi, int above, int left)
    {
        bool hasAbove = above >= 0, hasLeft = left >= 0;
        if (hasAbove && hasLeft)
        {
            bool aboveIntra = !Vp9Mi.IsInter(mi, above), leftIntra = !Vp9Mi.IsInter(mi, left);
            return leftIntra && aboveIntra ? 3 : (leftIntra || aboveIntra ? 1 : 0);
        }
        if (hasAbove || hasLeft) return Vp9Mi.IsInter(mi, hasAbove ? above : left) ? 0 : 2;
        return 0;
    }

    public static int ReferenceModeContext(ArrayView<int> mi, int above, int left, int compFixedRef)
    {
        bool hasAbove = above >= 0, hasLeft = left >= 0;
        if (hasAbove && hasLeft)
        {
            bool aSecond = Vp9Mi.HasSecondRef(mi, above), lSecond = Vp9Mi.HasSecondRef(mi, left);
            if (!aSecond && !lSecond)
                return ((Vp9Mi.Ref(mi, above, 0) == compFixedRef) ? 1 : 0) ^ ((Vp9Mi.Ref(mi, left, 0) == compFixedRef) ? 1 : 0);
            if (!aSecond) return 2 + (Vp9Mi.Ref(mi, above, 0) == compFixedRef || !Vp9Mi.IsInter(mi, above) ? 1 : 0);
            if (!lSecond) return 2 + (Vp9Mi.Ref(mi, left, 0) == compFixedRef || !Vp9Mi.IsInter(mi, left) ? 1 : 0);
            return 4;
        }
        if (hasAbove || hasLeft)
        {
            int edge = hasAbove ? above : left;
            if (!Vp9Mi.HasSecondRef(mi, edge)) return Vp9Mi.Ref(mi, edge, 0) == compFixedRef ? 1 : 0;
            return 3;
        }
        return 1;
    }

    public static int CompRefContext(ArrayView<int> mi, int above, int left, int compFixedRef, int compVar0, int compVar1, int fixRefIdx)
    {
        int varRefIdx = fixRefIdx == 0 ? 1 : 0;
        bool aboveIn = above >= 0, leftIn = left >= 0;
        if (aboveIn && leftIn)
        {
            bool aboveIntra = !Vp9Mi.IsInter(mi, above), leftIntra = !Vp9Mi.IsInter(mi, left);
            if (aboveIntra && leftIntra) return 2;
            if (aboveIntra || leftIntra)
            {
                int edge = aboveIntra ? left : above;
                if (!Vp9Mi.HasSecondRef(mi, edge)) return 1 + 2 * (Vp9Mi.Ref(mi, edge, 0) != compVar1 ? 1 : 0);
                return 1 + 2 * (Vp9Mi.Ref(mi, edge, varRefIdx) != compVar1 ? 1 : 0);
            }
            bool lSg = !Vp9Mi.HasSecondRef(mi, left), aSg = !Vp9Mi.HasSecondRef(mi, above);
            int vrfa = aSg ? Vp9Mi.Ref(mi, above, 0) : Vp9Mi.Ref(mi, above, varRefIdx);
            int vrfl = lSg ? Vp9Mi.Ref(mi, left, 0) : Vp9Mi.Ref(mi, left, varRefIdx);
            if (vrfa == vrfl && compVar1 == vrfa) return 0;
            if (lSg && aSg)
            {
                if ((vrfa == compFixedRef && vrfl == compVar0) || (vrfl == compFixedRef && vrfa == compVar0)) return 4;
                if (vrfa == vrfl) return 3;
                return 1;
            }
            if (lSg || aSg)
            {
                int vrfc = lSg ? vrfa : vrfl;
                int rfs = aSg ? vrfa : vrfl;
                if (vrfc == compVar1 && rfs != compVar1) return 1;
                if (rfs == compVar1 && vrfc != compVar1) return 2;
                return 4;
            }
            return vrfa == vrfl ? 4 : 2;
        }
        if (aboveIn || leftIn)
        {
            int edge = aboveIn ? above : left;
            if (!Vp9Mi.IsInter(mi, edge)) return 2;
            if (Vp9Mi.HasSecondRef(mi, edge)) return 4 * (Vp9Mi.Ref(mi, edge, varRefIdx) != compVar1 ? 1 : 0);
            return 3 * (Vp9Mi.Ref(mi, edge, 0) != compVar1 ? 1 : 0);
        }
        return 2;
    }

    public static int SingleRefP1Context(ArrayView<int> mi, int above, int left)
    {
        const int Last = Vp9Mi.LastFrame;
        bool hasAbove = above >= 0, hasLeft = left >= 0;
        if (hasAbove && hasLeft)
        {
            bool aboveIntra = !Vp9Mi.IsInter(mi, above), leftIntra = !Vp9Mi.IsInter(mi, left);
            if (aboveIntra && leftIntra) return 2;
            if (aboveIntra || leftIntra)
            {
                int edge = aboveIntra ? left : above;
                if (!Vp9Mi.HasSecondRef(mi, edge)) return 4 * (Vp9Mi.Ref(mi, edge, 0) == Last ? 1 : 0);
                return 1 + (Vp9Mi.Ref(mi, edge, 0) == Last || Vp9Mi.Ref(mi, edge, 1) == Last ? 1 : 0);
            }
            bool aSecond = Vp9Mi.HasSecondRef(mi, above), lSecond = Vp9Mi.HasSecondRef(mi, left);
            int above0 = Vp9Mi.Ref(mi, above, 0), above1 = Vp9Mi.Ref(mi, above, 1);
            int left0 = Vp9Mi.Ref(mi, left, 0), left1 = Vp9Mi.Ref(mi, left, 1);
            if (aSecond && lSecond)
                return 1 + (above0 == Last || above1 == Last || left0 == Last || left1 == Last ? 1 : 0);
            if (aSecond || lSecond)
            {
                int rfs = !aSecond ? above0 : left0;
                int crf1 = aSecond ? above0 : left0;
                int crf2 = aSecond ? above1 : left1;
                if (rfs == Last) return 3 + (crf1 == Last || crf2 == Last ? 1 : 0);
                return crf1 == Last || crf2 == Last ? 1 : 0;
            }
            return 2 * (above0 == Last ? 1 : 0) + 2 * (left0 == Last ? 1 : 0);
        }
        if (hasAbove || hasLeft)
        {
            int edge = hasAbove ? above : left;
            if (!Vp9Mi.IsInter(mi, edge)) return 2;
            if (!Vp9Mi.HasSecondRef(mi, edge)) return 4 * (Vp9Mi.Ref(mi, edge, 0) == Last ? 1 : 0);
            return 1 + (Vp9Mi.Ref(mi, edge, 0) == Last || Vp9Mi.Ref(mi, edge, 1) == Last ? 1 : 0);
        }
        return 2;
    }

    public static int SingleRefP2Context(ArrayView<int> mi, int above, int left)
    {
        const int Last = Vp9Mi.LastFrame, Golden = Vp9Mi.GoldenFrame, Alt = Vp9Mi.AltrefFrame;
        bool hasAbove = above >= 0, hasLeft = left >= 0;
        if (hasAbove && hasLeft)
        {
            bool aboveIntra = !Vp9Mi.IsInter(mi, above), leftIntra = !Vp9Mi.IsInter(mi, left);
            if (aboveIntra && leftIntra) return 2;
            if (aboveIntra || leftIntra)
            {
                int edge = aboveIntra ? left : above;
                if (!Vp9Mi.HasSecondRef(mi, edge))
                {
                    if (Vp9Mi.Ref(mi, edge, 0) == Last) return 3;
                    return 4 * (Vp9Mi.Ref(mi, edge, 0) == Golden ? 1 : 0);
                }
                return 1 + 2 * (Vp9Mi.Ref(mi, edge, 0) == Golden || Vp9Mi.Ref(mi, edge, 1) == Golden ? 1 : 0);
            }
            bool aSecond = Vp9Mi.HasSecondRef(mi, above), lSecond = Vp9Mi.HasSecondRef(mi, left);
            int above0 = Vp9Mi.Ref(mi, above, 0), above1 = Vp9Mi.Ref(mi, above, 1);
            int left0 = Vp9Mi.Ref(mi, left, 0), left1 = Vp9Mi.Ref(mi, left, 1);
            if (aSecond && lSecond)
            {
                if (above0 == left0 && above1 == left1)
                    return 3 * (above0 == Golden || above1 == Golden || left0 == Golden || left1 == Golden ? 1 : 0);
                return 2;
            }
            if (aSecond || lSecond)
            {
                int rfs = !aSecond ? above0 : left0;
                int crf1 = aSecond ? above0 : left0;
                int crf2 = aSecond ? above1 : left1;
                if (rfs == Golden) return 3 + (crf1 == Golden || crf2 == Golden ? 1 : 0);
                if (rfs == Alt) return crf1 == Golden || crf2 == Golden ? 1 : 0;
                return 1 + 2 * (crf1 == Golden || crf2 == Golden ? 1 : 0);
            }
            if (above0 == Last && left0 == Last) return 3;
            if (above0 == Last || left0 == Last)
            {
                int edge0 = above0 == Last ? left0 : above0;
                return 4 * (edge0 == Golden ? 1 : 0);
            }
            return 2 * (above0 == Golden ? 1 : 0) + 2 * (left0 == Golden ? 1 : 0);
        }
        if (hasAbove || hasLeft)
        {
            int edge = hasAbove ? above : left;
            if (!Vp9Mi.IsInter(mi, edge) || (Vp9Mi.Ref(mi, edge, 0) == Last && !Vp9Mi.HasSecondRef(mi, edge))) return 2;
            if (!Vp9Mi.HasSecondRef(mi, edge)) return 4 * (Vp9Mi.Ref(mi, edge, 0) == Golden ? 1 : 0);
            return 3 * (Vp9Mi.Ref(mi, edge, 0) == Golden || Vp9Mi.Ref(mi, edge, 1) == Golden ? 1 : 0);
        }
        return 2;
    }

    /// <summary>get_tx_size_context (maxTx = max_txsize_lookup[sb_type] of the current block).</summary>
    public static int TxSizeContext(ArrayView<int> mi, int above, int left, int maxTx)
    {
        bool hasAbove = above >= 0, hasLeft = left >= 0;
        int aboveCtx = hasAbove && mi[above * Vp9Mi.Stride + Vp9Mi.Skip] == 0 ? mi[above * Vp9Mi.Stride + Vp9Mi.TxSize] : maxTx;
        int leftCtx = hasLeft && mi[left * Vp9Mi.Stride + Vp9Mi.Skip] == 0 ? mi[left * Vp9Mi.Stride + Vp9Mi.TxSize] : maxTx;
        if (!hasLeft) leftCtx = aboveCtx;
        if (!hasAbove) aboveCtx = leftCtx;
        return aboveCtx + leftCtx > maxTx ? 1 : 0;
    }
}
