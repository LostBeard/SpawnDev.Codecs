// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 backward adaptation on the accelerator: the frame's probabilities move toward what the frame's symbol counts
// say, starting from the context the frame loaded. A port of libvpx v1.17.0 (BSD-3-Clause):
//   vp9/common/vp9_entropy.c      vp9_adapt_coef_probs / adapt_coef_probs
//   vp9/common/vp9_entropymode.c  vp9_adapt_mode_probs, tx_counts_to_branch_counts_*
//   vp9/common/vp9_entropymv.c    vp9_adapt_mv_probs
//   vpx_dsp/prob.h / prob.c       get_prob, get_binary_prob, weighted_prob, merge_probs, mode_mv_merge_probs,
//                                 vpx_tree_merge_probs (recursion -> nodes from last to first: a node's children always
//                                 come later in libvpx's trees, so their totals are ready)

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Arguments of <see cref="Vp9AdaptKernels.AdaptKernel"/>.</summary>
internal static class Vp9AArg
{
    public const int PreSlot = 0, IntraOnly = 1, LastFrameKey = 2, Switchable = 3, AllowHp = 4, Count = 5;
}

internal static class Vp9AdaptKernels
{
    private const int CountSat = 24, MaxUpdateFactor = 112, MaxUpdateFactorKey = 112, MaxUpdateFactorAfterKey = 128;
    private const int ModeMvCountSat = 20;

    /// <summary>One thread per FRAME_COUNTS entry: the per tile column counts summed into <paramref name="sum"/>.</summary>
    public static void SumCountsKernel(Index1D i, ArrayView<int> counts, ArrayView<int> sum, int tileCols)
    {
        int s = 0;
        for (int c = 0; c < tileCols; c++) s += counts[c * Vp9Cnt.Size + i];
        sum[i] = s;
    }

    private static int GetProb(uint num, uint den)
    {
        int p = (int)(((ulong)num * 256 + (den >> 1)) / den);
        return p > 255 ? 255 : p < 1 ? 1 : p;
    }

    private static int GetBinaryProb(uint n0, uint n1)
    {
        uint den = n0 + n1;
        return den == 0 ? 128 : GetProb(n0, den);
    }

    private static int WeightedProb(int prob1, int prob2, int factor) => (prob1 * (256 - factor) + prob2 * factor + 128) >> 8;

    private static int MergeProbs(int pre, uint ct0, uint ct1, uint countSat, uint maxUpdateFactor)
    {
        int prob = GetBinaryProb(ct0, ct1);
        uint count = ct0 + ct1 < countSat ? ct0 + ct1 : countSat;
        uint factor = maxUpdateFactor * count / countSat;
        return WeightedProb(pre, prob, (int)factor);
    }

    /// <summary>count_to_update_factor[count] = MODE_MV_MAX_UPDATE_FACTOR (128) * count / MODE_MV_COUNT_SAT, as libvpx tabulates it.</summary>
    private static int CountToUpdateFactor(uint count) => count switch
    {
        0 => 0, 1 => 6, 2 => 12, 3 => 19, 4 => 25, 5 => 32, 6 => 38, 7 => 44, 8 => 51, 9 => 57, 10 => 64,
        11 => 70, 12 => 76, 13 => 83, 14 => 89, 15 => 96, 16 => 102, 17 => 108, 18 => 115, 19 => 121, _ => 128,
    };

    private static int ModeMvMergeProbs(int pre, uint ct0, uint ct1)
    {
        uint den = ct0 + ct1;
        if (den == 0) return pre;
        uint count = den < ModeMvCountSat ? den : ModeMvCountSat;
        return WeightedProb(pre, GetProb(ct0, den), CountToUpdateFactor(count));
    }

    /// <summary>
    /// vpx_tree_merge_probs over a tree of <paramref name="treeSize"/> entries at t[tree]: probabilities at ctx[pre] / ctx[dst],
    /// counts at counts[cnt]. <paramref name="totals"/> holds one total per node.
    /// </summary>
    private static void TreeMergeProbs(ArrayView<int> t, int tree, int treeSize, ArrayView<int> ctx, int pre, int dst,
        ArrayView<int> counts, int cnt, ArrayView<int> totals)
    {
        for (int i = treeSize - 2; i >= 0; i -= 2)
        {
            int l = t[tree + i], r = t[tree + i + 1];
            uint left = l <= 0 ? (uint)counts[cnt - l] : (uint)totals[l >> 1];
            uint right = r <= 0 ? (uint)counts[cnt - r] : (uint)totals[r >> 1];
            totals[i >> 1] = (int)(left + right);
            ctx[dst + (i >> 1)] = ModeMvMergeProbs(ctx[pre + (i >> 1)], left, right);
        }
    }

    /// <summary>One thread: libvpx's backward adaptation of the current context (slot 4) from the loaded one and the summed counts.</summary>
    public static void AdaptKernel(Index1D index, ArrayView<int> ctx, ArrayView<int> c, ArrayView<int> t, ArrayView<int> args,
        ArrayView<int> results, ArrayView<int> totals)
    {
        int pre = args[Vp9AArg.PreSlot] * Vp9Ctx.Bytes;
        int fc = Vp9Ctx.Current * Vp9Ctx.Bytes;
        bool intraOnly = args[Vp9AArg.IntraOnly] != 0;

        // vp9_adapt_coef_probs
        uint updateFactor = (uint)(intraOnly ? MaxUpdateFactorKey : args[Vp9AArg.LastFrameKey] != 0 ? MaxUpdateFactorAfterKey : MaxUpdateFactor);
        for (int tx = 0; tx < 4; tx++)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 6; k++)
                    {
                        int contexts = k == 0 ? 3 : 6;
                        for (int l = 0; l < contexts; l++)
                        {
                            int block = (((tx * 2 + i) * 2 + j) * 6 + k) * 6 + l;
                            int cnt = Vp9Cnt.Coef + block * 4;
                            uint n0 = (uint)c[cnt], n1 = (uint)c[cnt + 1], n2 = (uint)c[cnt + 2], neob = (uint)c[cnt + 3];
                            uint eob = (uint)c[Vp9Cnt.EobBranch + block];
                            int p = Vp9Ctx.Coef + block * 3;
                            ctx[fc + p] = MergeProbs(ctx[pre + p], neob, eob - neob, CountSat, updateFactor);
                            ctx[fc + p + 1] = MergeProbs(ctx[pre + p + 1], n0, n1 + n2, CountSat, updateFactor);
                            ctx[fc + p + 2] = MergeProbs(ctx[pre + p + 2], n1, n2, CountSat, updateFactor);
                        }
                    }
        if (intraOnly) return;

        // vp9_adapt_mode_probs
        for (int i = 0; i < 4; i++)
            ctx[fc + Vp9Ctx.IntraInter + i] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.IntraInter + i], (uint)c[Vp9Cnt.IntraInter + i * 2], (uint)c[Vp9Cnt.IntraInter + i * 2 + 1]);
        for (int i = 0; i < 5; i++)
            ctx[fc + Vp9Ctx.CompInter + i] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.CompInter + i], (uint)c[Vp9Cnt.CompInter + i * 2], (uint)c[Vp9Cnt.CompInter + i * 2 + 1]);
        for (int i = 0; i < 5; i++)
            ctx[fc + Vp9Ctx.CompRef + i] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.CompRef + i], (uint)c[Vp9Cnt.CompRef + i * 2], (uint)c[Vp9Cnt.CompRef + i * 2 + 1]);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 2; j++)
                ctx[fc + Vp9Ctx.SingleRef + i * 2 + j] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.SingleRef + i * 2 + j],
                    (uint)c[Vp9Cnt.SingleRef + (i * 2 + j) * 2], (uint)c[Vp9Cnt.SingleRef + (i * 2 + j) * 2 + 1]);
        for (int i = 0; i < 7; i++)
            TreeMergeProbs(t, Vp9T.InterModeTree, 6, ctx, pre + Vp9Ctx.InterMode + i * 3, fc + Vp9Ctx.InterMode + i * 3, c, Vp9Cnt.InterMode + i * 4, totals);
        for (int i = 0; i < 4; i++)
            TreeMergeProbs(t, Vp9T.IntraModeTree, 18, ctx, pre + Vp9Ctx.YMode + i * 9, fc + Vp9Ctx.YMode + i * 9, c, Vp9Cnt.YMode + i * 10, totals);
        for (int i = 0; i < 10; i++)
            TreeMergeProbs(t, Vp9T.IntraModeTree, 18, ctx, pre + Vp9Ctx.UvMode + i * 9, fc + Vp9Ctx.UvMode + i * 9, c, Vp9Cnt.UvMode + i * 10, totals);
        for (int i = 0; i < 16; i++)
            TreeMergeProbs(t, Vp9T.PartitionTree, 6, ctx, pre + Vp9Ctx.Partition + i * 3, fc + Vp9Ctx.Partition + i * 3, c, Vp9Cnt.Partition + i * 4, totals);
        if (args[Vp9AArg.Switchable] != 0)
            for (int i = 0; i < 4; i++)
                TreeMergeProbs(t, Vp9T.SwitchableInterpTree, 4, ctx, pre + Vp9Ctx.SwitchableInterp + i * 2, fc + Vp9Ctx.SwitchableInterp + i * 2,
                    c, Vp9Cnt.SwitchableInterp + i * 3, totals);
        if (results[Vp9CompArgs.RTxMode] == 4)
            for (int i = 0; i < 2; i++)
            {
                int c8 = Vp9Cnt.Tx8 + i * 2;
                ctx[fc + Vp9Ctx.Tx8 + i] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Tx8 + i], (uint)c[c8], (uint)c[c8 + 1]);
                int c16 = Vp9Cnt.Tx16 + i * 3;
                ctx[fc + Vp9Ctx.Tx16 + i * 2] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Tx16 + i * 2], (uint)c[c16], (uint)(c[c16 + 1] + c[c16 + 2]));
                ctx[fc + Vp9Ctx.Tx16 + i * 2 + 1] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Tx16 + i * 2 + 1], (uint)c[c16 + 1], (uint)c[c16 + 2]);
                int c32 = Vp9Cnt.Tx32 + i * 4;
                ctx[fc + Vp9Ctx.Tx32 + i * 3] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Tx32 + i * 3], (uint)c[c32], (uint)(c[c32 + 1] + c[c32 + 2] + c[c32 + 3]));
                ctx[fc + Vp9Ctx.Tx32 + i * 3 + 1] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Tx32 + i * 3 + 1], (uint)c[c32 + 1], (uint)(c[c32 + 2] + c[c32 + 3]));
                ctx[fc + Vp9Ctx.Tx32 + i * 3 + 2] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Tx32 + i * 3 + 2], (uint)c[c32 + 2], (uint)c[c32 + 3]);
            }
        for (int i = 0; i < 3; i++)
            ctx[fc + Vp9Ctx.Skip + i] = ModeMvMergeProbs(ctx[pre + Vp9Ctx.Skip + i], (uint)c[Vp9Cnt.Skip + i * 2], (uint)c[Vp9Cnt.Skip + i * 2 + 1]);

        // vp9_adapt_mv_probs
        TreeMergeProbs(t, Vp9T.MvJointTree, 6, ctx, pre + Vp9Ctx.MvJoints, fc + Vp9Ctx.MvJoints, c, Vp9Cnt.MvJoints, totals);
        for (int i = 0; i < 2; i++)
        {
            int cp = Vp9Ctx.MvComp + i * Vp9Ctx.MvCompStride;
            int cc = Vp9Cnt.MvComp + i * Vp9Cnt.MvCompStride;
            ctx[fc + cp + Vp9Ctx.MvSign] = ModeMvMergeProbs(ctx[pre + cp + Vp9Ctx.MvSign], (uint)c[cc + Vp9Cnt.MvSign], (uint)c[cc + Vp9Cnt.MvSign + 1]);
            TreeMergeProbs(t, Vp9T.MvClassTree, 20, ctx, pre + cp + Vp9Ctx.MvClasses, fc + cp + Vp9Ctx.MvClasses, c, cc + Vp9Cnt.MvClasses, totals);
            TreeMergeProbs(t, Vp9T.MvClass0Tree, 2, ctx, pre + cp + Vp9Ctx.MvClass0, fc + cp + Vp9Ctx.MvClass0, c, cc + Vp9Cnt.MvClass0, totals);
            for (int j = 0; j < 10; j++)
                ctx[fc + cp + Vp9Ctx.MvBits + j] = ModeMvMergeProbs(ctx[pre + cp + Vp9Ctx.MvBits + j],
                    (uint)c[cc + Vp9Cnt.MvBits + j * 2], (uint)c[cc + Vp9Cnt.MvBits + j * 2 + 1]);
            for (int j = 0; j < 2; j++)
                TreeMergeProbs(t, Vp9T.MvFpTree, 6, ctx, pre + cp + Vp9Ctx.MvClass0Fp + j * 3, fc + cp + Vp9Ctx.MvClass0Fp + j * 3,
                    c, cc + Vp9Cnt.MvClass0Fp + j * 4, totals);
            TreeMergeProbs(t, Vp9T.MvFpTree, 6, ctx, pre + cp + Vp9Ctx.MvFp, fc + cp + Vp9Ctx.MvFp, c, cc + Vp9Cnt.MvFp, totals);
            if (args[Vp9AArg.AllowHp] != 0)
            {
                ctx[fc + cp + Vp9Ctx.MvClass0Hp] = ModeMvMergeProbs(ctx[pre + cp + Vp9Ctx.MvClass0Hp], (uint)c[cc + Vp9Cnt.MvClass0Hp], (uint)c[cc + Vp9Cnt.MvClass0Hp + 1]);
                ctx[fc + cp + Vp9Ctx.MvHp] = ModeMvMergeProbs(ctx[pre + cp + Vp9Ctx.MvHp], (uint)c[cc + Vp9Cnt.MvHp], (uint)c[cc + Vp9Cnt.MvHp + 1]);
            }
        }
    }
}
