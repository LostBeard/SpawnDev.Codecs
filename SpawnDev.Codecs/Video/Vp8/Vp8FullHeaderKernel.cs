// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full-syntax VP8 keyframe decoder, kernel 1: the frame header (RFC 6386 sections 9.2 - 9.11, 19.2). ONE GPU thread
// parses everything after the 10 uncompressed bytes: colour space / clamping, segmentation (map + data updates),
// loop-filter header with mode/ref deltas, the token partition count, quantizer indices and deltas, the per-frame
// coefficient probability updates and the skip probability. It writes the parsed values to the meta block, builds
// the per-segment dequantizer table, locates every token partition from the partition size table, and leaves the
// first-partition bool decoder (positioned at the first macroblock's mode data) in state slot 0.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp8;

internal static class Vp8FullHeaderKernel
{
    public static void Kernel(
        Index1D _,
        ArrayView<byte> enc, ArrayView<int> meta, ArrayView<int> bdStates,
        ArrayView<byte> coefProbs, ArrayView<byte> defaultProbs, ArrayView<byte> updateProbs,
        ArrayView<int> dcQ, ArrayView<int> acQ)
    {
        int frameLen = meta[Vp8FullMeta.FrameLength];
        int p0Len = meta[Vp8FullMeta.FirstPartLen];
        var bd = Vp8BdOps.Init(enc, 10, p0Len);

        Vp8BdOps.Literal(ref bd, enc, 1);                          // colour space (ignored by every decoder)
        Vp8BdOps.Literal(ref bd, enc, 1);                          // clamping type (decoders always clamp)

        // ---- segmentation ----
        int segEnabled = Vp8BdOps.Literal(ref bd, enc, 1);
        int segUpdateMap = 0, segUpdateData = 0, segAbs = 0;
        for (int i = 0; i < 4; i++) { meta[Vp8FullMeta.SegQ + i] = 0; meta[Vp8FullMeta.SegLf + i] = 0; }
        for (int i = 0; i < 3; i++) meta[Vp8FullMeta.SegTreeProb + i] = 255;
        if (segEnabled != 0)
        {
            segUpdateMap = Vp8BdOps.Literal(ref bd, enc, 1);
            segUpdateData = Vp8BdOps.Literal(ref bd, enc, 1);
            if (segUpdateData != 0)
            {
                segAbs = Vp8BdOps.Literal(ref bd, enc, 1);
                for (int i = 0; i < 4; i++) meta[Vp8FullMeta.SegQ + i] = Vp8BdOps.OptionalSigned(ref bd, enc, 7);
                for (int i = 0; i < 4; i++) meta[Vp8FullMeta.SegLf + i] = Vp8BdOps.OptionalSigned(ref bd, enc, 6);
            }
            if (segUpdateMap != 0)
            {
                for (int i = 0; i < 3; i++)
                {
                    int present = Vp8BdOps.Literal(ref bd, enc, 1);
                    if (present != 0) meta[Vp8FullMeta.SegTreeProb + i] = Vp8BdOps.Literal(ref bd, enc, 8);
                }
            }
        }
        meta[Vp8FullMeta.SegEnabled] = segEnabled;
        meta[Vp8FullMeta.SegUpdateMap] = segUpdateMap;
        meta[Vp8FullMeta.SegAbs] = segAbs;

        // ---- loop filter ----
        meta[Vp8FullMeta.FilterType] = Vp8BdOps.Literal(ref bd, enc, 1);
        meta[Vp8FullMeta.FilterLevel] = Vp8BdOps.Literal(ref bd, enc, 6);
        meta[Vp8FullMeta.Sharpness] = Vp8BdOps.Literal(ref bd, enc, 3);
        int lfDelta = Vp8BdOps.Literal(ref bd, enc, 1);
        for (int i = 0; i < 4; i++) { meta[Vp8FullMeta.RefLfDelta + i] = 0; meta[Vp8FullMeta.ModeLfDelta + i] = 0; }
        if (lfDelta != 0)
        {
            int update = Vp8BdOps.Literal(ref bd, enc, 1);
            if (update != 0)
            {
                for (int i = 0; i < 4; i++) meta[Vp8FullMeta.RefLfDelta + i] = Vp8BdOps.OptionalSigned(ref bd, enc, 6);
                for (int i = 0; i < 4; i++) meta[Vp8FullMeta.ModeLfDelta + i] = Vp8BdOps.OptionalSigned(ref bd, enc, 6);
            }
        }
        meta[Vp8FullMeta.LfDeltaEnabled] = lfDelta;

        // ---- partitions + quantizer ----
        int log2Parts = Vp8BdOps.Literal(ref bd, enc, 2);
        meta[Vp8FullMeta.Log2Parts] = log2Parts;
        meta[Vp8FullMeta.NumParts] = 1 << log2Parts;

        int baseQ = Vp8BdOps.Literal(ref bd, enc, 7);
        int y1DcDelta = Vp8BdOps.OptionalSigned(ref bd, enc, 4);
        int y2DcDelta = Vp8BdOps.OptionalSigned(ref bd, enc, 4);
        int y2AcDelta = Vp8BdOps.OptionalSigned(ref bd, enc, 4);
        int uvDcDelta = Vp8BdOps.OptionalSigned(ref bd, enc, 4);
        int uvAcDelta = Vp8BdOps.OptionalSigned(ref bd, enc, 4);

        Vp8BdOps.Literal(ref bd, enc, 1);                          // refresh_entropy_probs (only matters for the next frame)

        // ---- coefficient probability updates (4 x 8 x 3 x 11 = 1056 entries) ----
        for (int i = 0; i < 1056; i++)
        {
            byte cur = defaultProbs[i];
            int upd = Vp8BdOps.Bool(ref bd, enc, updateProbs[i]);
            if (upd != 0) cur = (byte)Vp8BdOps.Literal(ref bd, enc, 8);
            coefProbs[i] = cur;
        }

        int noSkip = Vp8BdOps.Literal(ref bd, enc, 1);
        int probSkip = 0;
        if (noSkip != 0) probSkip = Vp8BdOps.Literal(ref bd, enc, 8);
        meta[Vp8FullMeta.MbNoSkipCoeff] = noSkip;
        meta[Vp8FullMeta.ProbSkipFalse] = probSkip;

        // ---- per-segment dequantizers (libvpx vp8cx_init_de_quantizer + vp8_mb_init_dequantizer) ----
        for (int seg = 0; seg < 4; seg++)
        {
            int q = baseQ;
            if (segEnabled != 0)
            {
                int sq = meta[Vp8FullMeta.SegQ + seg];
                q = segAbs != 0 ? sq : baseQ + sq;
                q = q < 0 ? 0 : (q > 127 ? 127 : q);
            }
            int d = seg * 6 + Vp8FullMeta.Dequant;
            meta[d + 0] = dcQ[Clamp127(q + y1DcDelta)];
            meta[d + 1] = acQ[q];
            meta[d + 2] = dcQ[Clamp127(q + y2DcDelta)] * 2;
            int y2ac = (acQ[Clamp127(q + y2AcDelta)] * 101581) >> 16;
            meta[d + 3] = y2ac < 8 ? 8 : y2ac;
            int uvdc = dcQ[Clamp127(q + uvDcDelta)];
            meta[d + 4] = uvdc > 132 ? 132 : uvdc;
            meta[d + 5] = acQ[Clamp127(q + uvAcDelta)];
        }

        // The first-partition decoder continues with the macroblock modes.
        int error = 0;
        if (bd.Err != 0) error = (int)Vp8DecodeError.FirstPartitionTruncated;
        Vp8BdOps.Save(bdStates, 0, bd);

        // ---- token partitions (RFC 6386 section 9.5) ----
        int nparts = 1 << log2Parts;
        int tokenStart = 10 + p0Len;
        int tableBytes = 3 * (nparts - 1);
        int offset = tokenStart + tableBytes;
        bool tableBad = (error == 0) & (offset > frameLen);
        if (tableBad) error = (int)Vp8DecodeError.PartitionTableTruncated;
        for (int p = 0; p < nparts; p++)
        {
            int start = offset;
            int len = 0;
            if (error == 0)
            {
                if (p < nparts - 1)
                {
                    int t = tokenStart + 3 * p;
                    len = enc[t] | (enc[t + 1] << 8) | (enc[t + 2] << 16);
                    if (start + len > frameLen) { error = (int)Vp8DecodeError.TokenPartitionExceedsFrame; len = 0; }
                }
                else
                {
                    len = frameLen - start;
                }
                var tp = Vp8BdOps.Init(enc, start, len);
                Vp8BdOps.Save(bdStates, 1 + p, tp);
                offset = start + len;
            }
        }
        meta[Vp8FullMeta.Error] = error;
    }

    private static int Clamp127(int v) => v < 0 ? 0 : (v > 127 ? 127 : v);
}
