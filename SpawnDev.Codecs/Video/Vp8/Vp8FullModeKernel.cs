// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full-syntax VP8 keyframe decoder, kernel 2: per-macroblock mode info from the first partition (RFC 6386 section 11,
// libvpx read_kf_mb_mode_info), ONE macroblock row per launch (one GPU thread; the bool decoder state is picked up from
// and put back into the state buffer, so no launch runs for long). Per macroblock, in stream order:
//   segment id (only when the segment map is being updated), mb_skip_coeff (only with mb_no_coeff_skip), the 16x16 luma
//   mode (key-frame tree: B_PRED, DC, V, H, TM), for B_PRED the 16 sub-block modes (each read with the
//   kf_bmode_prob[above][left] contexts), and the chroma mode.
// The record written per macroblock always carries 16 sub-block modes: the signalled ones for B_PRED, the implied ones
// (DC->B_DC, V->B_VE, H->B_HE, TM->B_TM) otherwise, so the next macroblock's contexts read them uniformly. Neighbours
// outside the frame are B_DC_PRED.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>Layout of the constant byte table the full decoder kernels share.</summary>
internal static class Vp8FullTables
{
    public const int Zigzag = 0;     // 16
    public const int Bands = 16;     // 17 (the 17th is the sentinel read after position 15)
    public const int Cat3 = 33;      // 3
    public const int Cat4 = 36;      // 4
    public const int Cat5 = 40;      // 5
    public const int Cat6 = 45;      // 11
    public const int KfY = 56;       // 4: key-frame luma mode tree probabilities
    public const int KfUv = 60;      // 3: chroma mode tree probabilities
    public const int KfB = 64;       // 10 x 10 x 9: kf_bmode_prob[above][left][node]
    public const int Total = KfB + 900;

    public static byte[] Build()
    {
        var t = new byte[Total];
        Array.Copy(Vp8CoefTables.ZigzagScan, 0, t, Zigzag, 16);
        Array.Copy(Vp8CoefTables.CoefBands, 0, t, Bands, 17);
        Array.Copy(Vp8CoefTables.Cat3Probs, 0, t, Cat3, 3);
        Array.Copy(Vp8CoefTables.Cat4Probs, 0, t, Cat4, 4);
        Array.Copy(Vp8CoefTables.Cat5Probs, 0, t, Cat5, 5);
        Array.Copy(Vp8CoefTables.Cat6Probs, 0, t, Cat6, 11);
        byte[] ky = { 145, 156, 163, 128 };
        byte[] kuv = { 142, 114, 183 };
        Array.Copy(ky, 0, t, KfY, 4);
        Array.Copy(kuv, 0, t, KfUv, 3);
        for (int a = 0; a < 10; a++)
            for (int l = 0; l < 10; l++)
                for (int n = 0; n < 9; n++)
                    t[KfB + (a * 10 + l) * 9 + n] = Vp8KfBmodeProbs.Probs[a, l, n];
        return t;
    }
}

internal static class Vp8FullModeKernel
{
    public static void Kernel(
        Index1D _,
        ArrayView<byte> enc, ArrayView<int> meta, ArrayView<int> bdStates,
        ArrayView<byte> recs, ArrayView<byte> tables, int mbRow)
    {
        if (meta[Vp8FullMeta.Error] != 0) return;
        int mbCols = meta[Vp8FullMeta.MbCols];
        int segUpdateMap = meta[Vp8FullMeta.SegUpdateMap];
        int segP0 = meta[Vp8FullMeta.SegTreeProb + 0];
        int segP1 = meta[Vp8FullMeta.SegTreeProb + 1];
        int segP2 = meta[Vp8FullMeta.SegTreeProb + 2];
        int noSkip = meta[Vp8FullMeta.MbNoSkipCoeff];
        int probSkip = meta[Vp8FullMeta.ProbSkipFalse];
        var bd = Vp8BdOps.Load(bdStates, 0);

        for (int mbCol = 0; mbCol < mbCols; mbCol++)
        {
            long rec = ((long)mbRow * mbCols + mbCol) * Vp8MbRecord.Size;

            int seg = 0;
            if (segUpdateMap != 0)
            {
                int s0 = Vp8BdOps.Bool(ref bd, enc, segP0);
                if (s0 == 0) seg = Vp8BdOps.Bool(ref bd, enc, segP1);
                else seg = 2 + Vp8BdOps.Bool(ref bd, enc, segP2);
            }
            int skip = 0;
            if (noSkip != 0) skip = Vp8BdOps.Bool(ref bd, enc, probSkip);

            // 16x16 luma mode: kf_ymode_tree {-B_PRED, 2, 4, 6, -DC, -V, -H, -TM}, probs {145, 156, 163, 128}.
            int yMode;
            int b0 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfY + 0]);
            if (b0 == 0)
            {
                yMode = 4;
            }
            else
            {
                int b1 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfY + 1]);
                if (b1 == 0)
                {
                    int b2 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfY + 2]);
                    yMode = b2;                       // 0 DC, 1 V
                }
                else
                {
                    int b2 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfY + 3]);
                    yMode = 2 + b2;                   // 2 H, 3 TM
                }
            }

            if (yMode == 4)
            {
                for (int i = 0; i < 16; i++)
                {
                    int sr = i >> 2;
                    int sc = i & 3;
                    int above = 0;
                    if (sr > 0) above = recs[rec + Vp8MbRecord.BModes + (sr - 1) * 4 + sc];
                    else if (mbRow > 0) above = recs[rec - (long)mbCols * Vp8MbRecord.Size + Vp8MbRecord.BModes + 12 + sc];
                    int left = 0;
                    if (sc > 0) left = recs[rec + Vp8MbRecord.BModes + sr * 4 + sc - 1];
                    else if (mbCol > 0) left = recs[rec - Vp8MbRecord.Size + Vp8MbRecord.BModes + sr * 4 + 3];
                    int pb = Vp8FullTables.KfB + (above * 10 + left) * 9;

                    // vp8_bmode_tree: DC | TM | VE | ( HE | RD VR ) , ( LD | VL | HD HU )
                    int mode;
                    int t0 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 0]);
                    if (t0 == 0) mode = 0;
                    else
                    {
                        int t1 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 1]);
                        if (t1 == 0) mode = 1;
                        else
                        {
                            int t2 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 2]);
                            if (t2 == 0) mode = 2;
                            else
                            {
                                int t3 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 3]);
                                if (t3 == 0)
                                {
                                    int t4 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 4]);
                                    if (t4 == 0) mode = 3;
                                    else
                                    {
                                        int t5 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 5]);
                                        mode = 5 + t5;               // 5 RD, 6 VR
                                    }
                                }
                                else
                                {
                                    int t6 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 6]);
                                    if (t6 == 0) mode = 4;
                                    else
                                    {
                                        int t7 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 7]);
                                        if (t7 == 0) mode = 7;
                                        else
                                        {
                                            int t8 = Vp8BdOps.Bool(ref bd, enc, tables[pb + 8]);
                                            mode = 8 + t8;           // 8 HD, 9 HU
                                        }
                                    }
                                }
                            }
                        }
                    }
                    recs[rec + Vp8MbRecord.BModes + i] = (byte)mode;
                }
            }
            else
            {
                // implied sub-block mode: DC->B_DC(0), V->B_VE(2), H->B_HE(3), TM->B_TM(1)
                int implied = 0;
                if (yMode == 1) implied = 2;
                else if (yMode == 2) implied = 3;
                else if (yMode == 3) implied = 1;
                for (int i = 0; i < 16; i++) recs[rec + Vp8MbRecord.BModes + i] = (byte)implied;
            }

            // chroma mode: uv_mode_tree {-DC, 2, -V, 4, -H, -TM}, probs {142, 114, 183}
            int uvMode;
            int u0 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfUv + 0]);
            if (u0 == 0) uvMode = 0;
            else
            {
                int u1 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfUv + 1]);
                if (u1 == 0) uvMode = 1;
                else
                {
                    int u2 = Vp8BdOps.Bool(ref bd, enc, tables[Vp8FullTables.KfUv + 2]);
                    uvMode = 2 + u2;
                }
            }

            recs[rec + Vp8MbRecord.YMode] = (byte)yMode;
            recs[rec + Vp8MbRecord.UvMode] = (byte)uvMode;
            recs[rec + Vp8MbRecord.Segment] = (byte)seg;
            recs[rec + Vp8MbRecord.Skip] = (byte)skip;
            recs[rec + Vp8MbRecord.HasCoef] = 0;
        }

        Vp8BdOps.Save(bdStates, 0, bd);
        if (bd.Err != 0) meta[Vp8FullMeta.Error] = (int)Vp8DecodeError.FirstPartitionTruncated;
    }
}
