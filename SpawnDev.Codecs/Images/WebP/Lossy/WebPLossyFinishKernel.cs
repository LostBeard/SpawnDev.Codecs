// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Lossy WebP encoder, kernel 4 (one thread, a chain of bounded launches - see below): the end of libwebp's VP8EncTokenLoop and VP8EncWrite -
// FinalizeTokenProbas, VP8EmitTokens into the token partition, VP8BitWriterFinish, VP8AdjustFilterStrength (no autofilter),
// GeneratePartition0 (frame header bits, segment / filter headers, quantizers, token probabilities, VP8CodeIntraModes) and
// the RIFF / VP8 chunk / frame tag assembly into the caller's output. The bool encoder is libwebp's VP8BitWriter
// (bit_writer_utils.c) with the delayed-0xff carry run. Simple file format (no alpha). Port of Vp8LossyEncoderReference.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

/// <summary>libwebp VP8BitWriter state, writing into an ArrayView&lt;byte&gt; region [Base, Base + Cap).</summary>
internal struct WebPBoolWriter
{
    public int Range, Value, Run, NbBits, Pos, Base, Cap, Overflow;

    public void Init(int baseOff, int cap)
    {
        Range = 255 - 1; Value = 0; Run = 0; NbBits = -8; Pos = 0; Base = baseOff; Cap = cap; Overflow = 0;
    }
}

internal static class WebPLossyFinishKernel
{
    private static void Put(ArrayView<byte> buf, ref WebPBoolWriter w, int at, int v)
    {
        if (at < w.Cap) buf[w.Base + at] = (byte)v;
        else w.Overflow = 1;
    }

    private static void Flush(ArrayView<byte> buf, ref WebPBoolWriter w)
    {
        int s = 8 + w.NbBits;
        int bits = w.Value >> s;
        w.Value -= bits << s;
        w.NbBits -= 8;
        if ((bits & 0xff) != 0xff)
        {
            int pos = w.Pos;
            bool carry = (bits & 0x100) != 0;
            if (carry & (pos > 0))
            {
                int prev = pos - 1;
                if (prev < w.Cap) buf[w.Base + prev] = (byte)(buf[w.Base + prev] + 1);
            }
            int fill = carry ? 0x00 : 0xff;
            for (int k = 0; k < w.Run; ++k) { Put(buf, ref w, pos, fill); pos++; }
            w.Run = 0;
            Put(buf, ref w, pos, bits & 0xff);
            pos++;
            w.Pos = pos;
        }
        else
        {
            w.Run++;
        }
    }

    public static int PutBit(ArrayView<byte> buf, ref WebPBoolWriter w, ArrayView<int> t, int bit, int prob)
    {
        int split = (w.Range * prob) >> 8;
        if (bit != 0) { w.Value += split + 1; w.Range -= split + 1; }
        else w.Range = split;
        if (w.Range < 127)
        {
            int shift = t[T.Norm + w.Range];
            w.Range = t[T.NewRange + w.Range];
            w.Value <<= shift;
            w.NbBits += shift;
            if (w.NbBits > 0) Flush(buf, ref w);
        }
        return bit;
    }

    public static int PutBitUniform(ArrayView<byte> buf, ref WebPBoolWriter w, ArrayView<int> t, int bit)
    {
        int split = w.Range >> 1;
        if (bit != 0) { w.Value += split + 1; w.Range -= split + 1; }
        else w.Range = split;
        if (w.Range < 127)
        {
            w.Range = t[T.NewRange + w.Range];
            w.Value <<= 1;
            w.NbBits += 1;
            if (w.NbBits > 0) Flush(buf, ref w);
        }
        return bit;
    }

    private static void PutBits(ArrayView<byte> buf, ref WebPBoolWriter w, ArrayView<int> t, int value, int nbBits)
    {
        for (int k = nbBits - 1; k >= 0; --k) PutBitUniform(buf, ref w, t, (value >> k) & 1);
    }

    private static void PutSignedBits(ArrayView<byte> buf, ref WebPBoolWriter w, ArrayView<int> t, int value, int nbBits)
    {
        int nz = value != 0 ? 1 : 0;
        if (PutBitUniform(buf, ref w, t, nz) != 0)
        {
            if (value < 0) PutBits(buf, ref w, t, ((-value) << 1) | 1, nbBits + 1);
            else PutBits(buf, ref w, t, value << 1, nbBits + 1);
        }
    }

    private static void Finish(ArrayView<byte> buf, ref WebPBoolWriter w, ArrayView<int> t)
    {
        PutBits(buf, ref w, t, 0, 9 - w.NbBits);
        w.NbBits = 0;
        Flush(buf, ref w);
    }

    private static int PutI4Mode(ArrayView<byte> buf, ref WebPBoolWriter w, ArrayView<int> t, int mode, int p)
    {
        // B_DC 0, B_TM 1, B_VE 2, B_HE 3, B_RD 4, B_VR 5, B_LD 6, B_VL 7, B_HD 8, B_HU 9
        if (PutBit(buf, ref w, t, mode != 0 ? 1 : 0, t[p + 0]) != 0)
            if (PutBit(buf, ref w, t, mode != 1 ? 1 : 0, t[p + 1]) != 0)
                if (PutBit(buf, ref w, t, mode != 2 ? 1 : 0, t[p + 2]) != 0)
                {
                    if (PutBit(buf, ref w, t, mode >= 6 ? 1 : 0, t[p + 3]) == 0)
                    {
                        if (PutBit(buf, ref w, t, mode != 3 ? 1 : 0, t[p + 4]) != 0)
                            PutBit(buf, ref w, t, mode != 4 ? 1 : 0, t[p + 5]);
                    }
                    else
                    {
                        if (PutBit(buf, ref w, t, mode != 6 ? 1 : 0, t[p + 6]) != 0)
                            if (PutBit(buf, ref w, t, mode != 7 ? 1 : 0, t[p + 7]) != 0)
                                PutBit(buf, ref w, t, mode != 8 ? 1 : 0, t[p + 8]);
                    }
                }
        return mode;
    }

    /// <summary>Tokens per launch of the token partition (one thread; a token is one bool-coder step).</summary>
    public const int TokensPerLaunch = 65536;
    /// <summary>Macroblocks per launch of VP8CodeIntraModes in partition 0.</summary>
    public const int MbsPerLaunch = 2048;

    private static void SaveWriter(ArrayView<int> meta, int at, ref WebPBoolWriter w)
    {
        meta[at + 0] = w.Range; meta[at + 1] = w.Value; meta[at + 2] = w.Run; meta[at + 3] = w.NbBits;
        meta[at + 4] = w.Pos; meta[at + 5] = w.Base; meta[at + 6] = w.Cap; meta[at + 7] = w.Overflow;
    }

    private static WebPBoolWriter LoadWriter(ArrayView<int> meta, int at) => new WebPBoolWriter
    {
        Range = meta[at + 0], Value = meta[at + 1], Run = meta[at + 2], NbBits = meta[at + 3],
        Pos = meta[at + 4], Base = meta[at + 5], Cap = meta[at + 6], Overflow = meta[at + 7],
    };

    // The end of VP8EncTokenLoop and VP8EncWrite runs as a chain of launches (each one thread, each bounded):
    //   FinishProbasKernel (FinalizeTokenProbas, the token writer) -> FinishTokensKernel x ceil(tokens / TokensPerLaunch)
    //   -> FinishHeaderKernel (end of the token partition, VP8AdjustFilterStrength, partition 0 up to the intra modes)
    //   -> FinishModesKernel x ceil(macroblocks / MbsPerLaunch) -> FinishEndKernel (end of partition 0, sizes).
    // The two bool writers and the cursors live in meta[Step ..] between launches. All buffers are the same for every
    // launch: mbRecs, work (preds at 0), state (coeffs / stats via meta offsets), meta, tables, tokens, scratch (partition 0
    // at 0, token partition after it).

    /// <summary>FinalizeTokenProbas and the token partition writer.</summary>
    public static void FinishProbasKernel(
        Index1D _, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> t, ArrayView<int> tokens, ArrayView<byte> scratch)
    {
        if (meta[M.Error] != 0) return;
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var stats = state.SubView(meta[M.StStatsOff], WebPLossySetupKernel.ProbaCount);
        WebPLossyEncodeRowKernel.FinalizeTokenProbas(coeffs, stats, t, meta);
        var tw = new WebPBoolWriter();
        tw.Init(meta[M.Partition0Capacity], meta[M.Token1Capacity]);
        SaveWriter(meta, M.TokWriter, ref tw);
        meta[M.TokNext] = 0;
    }

    /// <summary>VP8EmitTokens(parts[0]): the next <see cref="TokensPerLaunch"/> tokens.</summary>
    public static void FinishTokensKernel(
        Index1D _, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> t, ArrayView<int> tokens, ArrayView<byte> scratch)
    {
        if (meta[M.Error] != 0) return;
        int start = meta[M.TokNext];
        int tokenCount = meta[M.TokenCount];
        if (start >= tokenCount) return;
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var tw = LoadWriter(meta, M.TokWriter);
        int end = start + TokensPerLaunch;
        if (end > tokenCount) end = tokenCount;
        for (int i = start; i < end; ++i)
        {
            int token = tokens[i];
            int bit = (token >> 15) & 1;
            int prob = (token & (1 << 14)) != 0 ? token & 0xff : coeffs[token & 0x3fff];
            PutBit(scratch, ref tw, t, bit, prob);
        }
        SaveWriter(meta, M.TokWriter, ref tw);
        meta[M.TokNext] = end;
    }

    /// <summary>VP8BitWriterFinish of the token partition, VP8AdjustFilterStrength, partition 0 up to the intra modes.</summary>
    public static void FinishHeaderKernel(
        Index1D _, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> t, ArrayView<int> tokens, ArrayView<byte> scratch)
    {
        if (meta[M.Error] != 0) return;
        var coeffs = state.SubView(meta[M.StCoeffsOff], WebPLossySetupKernel.ProbaCount);
        var tw = LoadWriter(meta, M.TokWriter);
        Finish(scratch, ref tw, t);
        SaveWriter(meta, M.TokWriter, ref tw);

        // ---- VP8AdjustFilterStrength (filter_strength > 0, no autofilter)
        if (meta[M.FilterStrength] > 0)
        {
            int maxLevel = 0;
            for (int s = 0; s < 4; ++s)
            {
                int sb = M.Seg + s * M.SegStride;
                int delta = (meta[sb + M.SMaxEdge] * meta[sb + M.SQY2 + 1]) >> 3;
                int pos = delta < 64 ? delta : 63;
                int level = t[T.LevelsFromDelta + meta[M.FilterSharpness] * 64 + pos];
                if (level > meta[sb + M.SFStrength]) meta[sb + M.SFStrength] = level;
                if (maxLevel < meta[sb + M.SFStrength]) maxLevel = meta[sb + M.SFStrength];
            }
            meta[M.FilterLevel] = maxLevel;
        }

        // ---- GeneratePartition0 up to VP8CodeIntraModes
        var bw = new WebPBoolWriter();
        bw.Init(0, meta[M.Partition0Capacity]);
        PutBitUniform(scratch, ref bw, t, 0);   // colorspace
        PutBitUniform(scratch, ref bw, t, 0);   // clamp type
        int numSegments = meta[M.NumSegments];
        bool updateMap = meta[M.UpdateMap] != 0;
        if (PutBitUniform(scratch, ref bw, t, numSegments > 1 ? 1 : 0) != 0)
        {
            PutBitUniform(scratch, ref bw, t, updateMap ? 1 : 0);
            if (PutBitUniform(scratch, ref bw, t, 1) != 0)
            {
                PutBitUniform(scratch, ref bw, t, 1);
                for (int s = 0; s < 4; ++s) PutSignedBits(scratch, ref bw, t, meta[M.Seg + s * M.SegStride + M.SQuant], 7);
                for (int s = 0; s < 4; ++s) PutSignedBits(scratch, ref bw, t, meta[M.Seg + s * M.SegStride + M.SFStrength], 6);
            }
            if (updateMap)
                for (int s = 0; s < 3; ++s)
                {
                    int sp = meta[M.SegProba0 + s];
                    if (PutBitUniform(scratch, ref bw, t, sp != 255 ? 1 : 0) != 0) PutBits(scratch, ref bw, t, sp, 8);
                }
        }
        PutBitUniform(scratch, ref bw, t, meta[M.FilterSimple]);
        PutBits(scratch, ref bw, t, meta[M.FilterLevel], 6);
        PutBits(scratch, ref bw, t, meta[M.FilterSharpness], 3);
        PutBitUniform(scratch, ref bw, t, 0);    // use_lf_delta (i4x4_lf_delta == 0)
        PutBits(scratch, ref bw, t, 0, 2);       // one token partition
        PutBits(scratch, ref bw, t, meta[M.BaseQuant], 7);
        PutSignedBits(scratch, ref bw, t, meta[M.DqY1Dc], 4);
        PutSignedBits(scratch, ref bw, t, meta[M.DqY2Dc], 4);
        PutSignedBits(scratch, ref bw, t, meta[M.DqY2Ac], 4);
        PutSignedBits(scratch, ref bw, t, meta[M.DqUVDc], 4);
        PutSignedBits(scratch, ref bw, t, meta[M.DqUVAc], 4);
        PutBitUniform(scratch, ref bw, t, 0);    // no proba update (refresh_entropy_probs)
        for (int i = 0; i < WebPLossySetupKernel.ProbaCount; ++i)
        {
            int p0 = coeffs[i];
            int update = p0 != t[T.CoeffsProba0 + i] ? 1 : 0;
            if (PutBit(scratch, ref bw, t, update, t[T.CoeffsUpdateProba + i]) != 0) PutBits(scratch, ref bw, t, p0, 8);
        }
        PutBitUniform(scratch, ref bw, t, 0);    // use_skip_proba is 0 in the token loop
        SaveWriter(meta, M.P0Writer, ref bw);
        meta[M.MbNext] = 0;
    }

    /// <summary>VP8CodeIntraModes: the next <see cref="MbsPerLaunch"/> macroblocks (raster order) of partition 0.</summary>
    public static void FinishModesKernel(
        Index1D _, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> t, ArrayView<int> tokens, ArrayView<byte> scratch)
    {
        if (meta[M.Error] != 0) return;
        int mbW = meta[M.MbW], mbH = meta[M.MbH];
        int start = meta[M.MbNext];
        int total = mbW * mbH;
        if (start >= total) return;
        int predsW = 4 * mbW + 1;
        bool updateMap = meta[M.UpdateMap] != 0;
        var bw = LoadWriter(meta, M.P0Writer);
        int end = start + MbsPerLaunch;
        if (end > total) end = total;
        for (int m = start; m < end; ++m)
        {
            int my = m / mbW;
            int mx = m - my * mbW;
            int rec = (my * mbW + mx) * M.MbStride;
            int pIdx = 1 + predsW + my * 4 * predsW + mx * 4;
            if (updateMap)
            {
                int sgm = mbRecs[rec + M.MbSegment];
                int bitHi = sgm >= 2 ? 1 : 0;
                int o = PutBit(scratch, ref bw, t, bitHi, meta[M.SegProba0]) != 0 ? 1 : 0;
                PutBit(scratch, ref bw, t, sgm & 1, meta[M.SegProba0 + o + 1]);
            }
            if (PutBit(scratch, ref bw, t, mbRecs[rec + M.MbType] != 0 ? 1 : 0, 145) != 0)
            {
                int mode = work[pIdx];
                // DC_PRED 0, TM_PRED 1, V_PRED 2, H_PRED 3
                bool tmOrH = (mode == 1) | (mode == 3);
                if (PutBit(scratch, ref bw, t, tmOrH ? 1 : 0, 156) != 0) PutBit(scratch, ref bw, t, mode == 1 ? 1 : 0, 128);
                else PutBit(scratch, ref bw, t, mode == 2 ? 1 : 0, 163);
            }
            else
            {
                int topPred = pIdx - predsW;
                int cur = pIdx;
                for (int y = 0; y < 4; ++y)
                {
                    int left = work[cur - 1];
                    for (int x = 0; x < 4; ++x)
                    {
                        int probIdx = T.BModesProba + (work[topPred + x] * 10 + left) * 9;
                        left = PutI4Mode(scratch, ref bw, t, work[cur + x], probIdx);
                    }
                    topPred = cur;
                    cur += predsW;
                }
            }
            int uvMode = mbRecs[rec + M.MbUVMode];
            if (PutBit(scratch, ref bw, t, uvMode != 0 ? 1 : 0, 142) != 0)
                if (PutBit(scratch, ref bw, t, uvMode != 2 ? 1 : 0, 114) != 0)
                    PutBit(scratch, ref bw, t, uvMode != 3 ? 1 : 0, 183);
        }
        SaveWriter(meta, M.P0Writer, ref bw);
        meta[M.MbNext] = end;
    }

    /// <summary>VP8BitWriterFinish of partition 0, the size checks and the file length.</summary>
    public static void FinishEndKernel(
        Index1D _, ArrayView<int> mbRecs, ArrayView<byte> work, ArrayView<int> state, ArrayView<int> meta,
        ArrayView<int> t, ArrayView<int> tokens, ArrayView<byte> scratch)
    {
        if (meta[M.Error] != 0) return;
        int p0Cap = meta[M.Partition0Capacity];
        int t1Cap = meta[M.Token1Capacity];
        var bw = LoadWriter(meta, M.P0Writer);
        Finish(scratch, ref bw, t);
        var tw = LoadWriter(meta, M.TokWriter);

        if ((bw.Overflow != 0) | (bw.Pos > p0Cap)) { meta[M.Error] = (int)WebPLossyEncError.Partition0ScratchTooSmall; return; }
        if ((tw.Overflow != 0) | (tw.Pos > t1Cap)) { meta[M.Error] = (int)WebPLossyEncError.TokenScratchTooSmall; return; }
        int size0 = bw.Pos;
        if (size0 >= (1 << 19)) { meta[M.Error] = (int)WebPLossyEncError.Partition0TooBig; return; }
        meta[M.Partition0Size] = size0;
        meta[M.Token1Size] = tw.Pos;
        int vp8Size = 10 + size0 + tw.Pos;
        vp8Size += vp8Size & 1;
        meta[M.OutLength] = 8 + 4 + 8 + vp8Size;
    }

    /// <summary>
    /// VP8EncWrite's byte layout, ONE THREAD PER OUTPUT BYTE: "RIFF" size "WEBP" "VP8 " size, the 10-byte frame tag, partition 0,
    /// the token partition, and the pad byte that makes the chunk even (libwebp counts it in the chunk size).
    /// </summary>
    public static void AssembleKernel(Index1D index, ArrayView<int> meta, ArrayView<byte> scratch, ArrayView<byte> output)
    {
        output[index.X] = SimpleFileByte(meta, scratch, index.X);
    }

    /// <summary>
    /// The extended layout of a picture with transparency (VP8EncWrite + VP8X / ALPH of libwebp's syntax_enc.c), ONE THREAD
    /// PER OUTPUT BYTE: "RIFF" size "WEBP", "VP8X" 10 (flags = ALPHA_FLAG 0x10, canvas width - 1 and height - 1 as 24-bit
    /// little-endian), "ALPH" <paramref name="alphaLength"/> + the payload + a pad byte when it is odd, then the "VP8 " chunk
    /// exactly as the simple layout has it (its bytes 12 ..). Launch with <see cref="ExtendedFileLength"/> threads.
    /// </summary>
    public static void AssembleExtendedKernel(Index1D index, ArrayView<int> meta, ArrayView<byte> scratch, ArrayView<byte> alpha,
        ArrayView<byte> output, int alphaLength)
    {
        int i = index.X;
        int vp8Start = 38 + alphaLength + (alphaLength & 1);
        int v;
        if (i < 12)
        {
            int riffSize = vp8Start + meta[M.OutLength] - 12 - 8;
            if (i < 4) v = i == 0 ? 'R' : (i == 1 ? 'I' : 'F');
            else if (i < 8) v = (riffSize >> (8 * (i - 4))) & 0xff;
            else v = i == 8 ? 'W' : (i == 9 ? 'E' : (i == 10 ? 'B' : 'P'));
        }
        else if (i < 30)
        {
            int j = i - 12;
            int w1 = meta[M.Width] - 1, h1 = meta[M.Height] - 1;
            if (j < 4) v = j == 0 ? 'V' : (j == 1 ? 'P' : (j == 2 ? '8' : 'X'));
            else if (j < 8) v = j == 4 ? 10 : 0;
            else if (j == 8) v = 0x10;
            else if (j < 12) v = 0;
            else if (j < 15) v = (w1 >> (8 * (j - 12))) & 0xff;
            else v = (h1 >> (8 * (j - 15))) & 0xff;
        }
        else if (i < 38)
        {
            int j = i - 30;
            if (j < 4) v = j == 0 ? 'A' : (j == 1 ? 'L' : (j == 2 ? 'P' : 'H'));
            else v = (alphaLength >> (8 * (j - 4))) & 0xff;
        }
        else if (i < 38 + alphaLength) v = alpha[i - 38];
        else if (i < vp8Start) v = 0;   // pad of an odd ALPH payload (not counted in its size)
        else v = SimpleFileByte(meta, scratch, i - vp8Start + 12);
        output[i] = (byte)v;
    }

    /// <summary>Length of the <see cref="AssembleExtendedKernel"/> file for a simple file of <paramref name="simpleLength"/> bytes.</summary>
    public static int ExtendedFileLength(int simpleLength, int alphaLength) => 38 + alphaLength + (alphaLength & 1) + simpleLength - 12;

    /// <summary>Byte <paramref name="i"/> of the simple layout (see <see cref="AssembleKernel"/>).</summary>
    private static byte SimpleFileByte(ArrayView<int> meta, ArrayView<byte> scratch, int i)
    {
        int size0 = meta[M.Partition0Size];
        int size1 = meta[M.Token1Size];
        int total = meta[M.OutLength];
        int vp8Size = total - 20;
        int riffSize = total - 8;
        int p0Cap = meta[M.Partition0Capacity];
        int v;
        if (i < 20)
        {
            int sizeField = i < 8 ? riffSize : vp8Size;
            int k = i < 8 ? i - 4 : i - 16;
            if (i < 4) v = i == 0 ? 'R' : (i == 1 ? 'I' : 'F');
            else if (i < 8) v = (sizeField >> (8 * k)) & 0xff;
            else if (i < 12) v = i == 8 ? 'W' : (i == 9 ? 'E' : (i == 10 ? 'B' : 'P'));
            else if (i < 16) v = i == 12 ? 'V' : (i == 13 ? 'P' : (i == 14 ? '8' : ' '));
            else v = (sizeField >> (8 * k)) & 0xff;
        }
        else if (i < 30)
        {
            int bits = (meta[M.Profile] << 1) | (1 << 4) | (size0 << 5);
            int width = meta[M.Width], height = meta[M.Height];
            int j = i - 20;
            if (j < 3) v = (bits >> (8 * j)) & 0xff;
            else if (j == 3) v = 0x9d;
            else if (j == 4) v = 0x01;
            else if (j == 5) v = 0x2a;
            else if (j == 6) v = width & 0xff;
            else if (j == 7) v = (width >> 8) & 0xff;
            else if (j == 8) v = height & 0xff;
            else v = (height >> 8) & 0xff;
        }
        else if (i < 30 + size0) v = scratch[i - 30];
        else if (i < 30 + size0 + size1) v = scratch[p0Cap + i - 30 - size0];
        else v = 0;   // pad
        return (byte)v;
    }
}
