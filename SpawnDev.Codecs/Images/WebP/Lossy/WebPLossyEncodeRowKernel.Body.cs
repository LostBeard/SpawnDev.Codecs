// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernel 3, part 2: the per-macroblock walk (VP8EncTokenLoop body) - import, VP8Decimate (RD_OPT_BASIC), RecordTokens,
// SaveBoundary. Names and order follow libwebp 1.6.0 / Vp8LossyEncoderReference; see the header of part 1.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

/// <summary>The scalar part of a libwebp VP8ModeScore (the level arrays live in the int scratch).</summary>
internal struct WebPModeScore
{
    public long D, SD, H, R, Score;
    public uint Nz;

    public void Init() { D = 0; SD = 0; R = 0; H = 0; Nz = 0; Score = 0x7fffffffffffffL; }
    public void SetRD(int lambda) => Score = (R + H) * lambda + 256L * (D + SD);
}

internal static partial class WebPLossyEncodeRowKernel
{
    // ================================================================== costs (cost_enc.c / dsp/cost.c)

    private static int PIdx(int t, int b, int c) => 11 * (c + 3 * (b + 8 * t));
    private static int CostIdx(int t, int b, int c) => 68 * (c + 3 * (b + 8 * t));

    private static int BitCost(ArrayView<int> t, int bit, int proba)
        => bit == 0 ? t[T.EntropyCost + proba] : t[T.EntropyCost + 255 - proba];

    private static int LevelCost(ArrayView<int> t, ArrayView<int> lc, int table, int level)
        => t[T.LevelFixedCosts + level] + lc[table + (level > 67 ? 67 : level)];

    /// <summary>SetResidualCoeffs: index of the last non-zero level (-1 when none).</summary>
    private static int LastNonZero(ArrayView<int> lv, int o)
    {
        int last = -1;
        for (int n = 0; n < 16; ++n) if (lv[o + n] != 0) last = n;
        return last;
    }

    private static int GetResidualCost(ArrayView<int> t, ArrayView<int> coeffs, ArrayView<int> lc,
        int ctx0, int type, int first, ArrayView<int> lv, int o)
    {
        int last = LastNonZero(lv, o);
        int n = first;
        int p0 = coeffs[PIdx(type, n, ctx0) + 0];
        if (last < 0) return BitCost(t, 0, p0);
        int tbl = CostIdx(type, t[T.EncBands + n], ctx0);
        int cost = ctx0 == 0 ? BitCost(t, 1, p0) : 0;
        for (; n < last; ++n)
        {
            int v = Abs(lv[o + n]);
            int ctx = v >= 2 ? 2 : v;
            cost += LevelCost(t, lc, tbl, v);
            tbl = CostIdx(type, t[T.EncBands + n + 1], ctx);
        }
        int vl = Abs(lv[o + n]);
        cost += LevelCost(t, lc, tbl, vl);
        if (n < 15)
        {
            int b = t[T.EncBands + n + 1];
            int ctx = vl == 1 ? 1 : 2;
            cost += BitCost(t, 0, coeffs[PIdx(type, b, ctx) + 0]);
        }
        return cost;
    }

    // ================================================================== non-zero contexts (iterator_enc.c)

    private static int Bit(int nz, int n) => (nz >> n) & 1;

    private static void NzToBytes(ArrayView<int> nzRow, int x, ArrayView<int> si) => NzToBytes(nzRow, x, si, ITopNz, ILeftNz);

    private static void NzToBytes(ArrayView<int> nzRow, int x, ArrayView<int> si, int top, int left)
    {
        int tnz = nzRow[1 + x], lnz = nzRow[x];
        si[top + 0] = Bit(tnz, 12); si[top + 1] = Bit(tnz, 13); si[top + 2] = Bit(tnz, 14); si[top + 3] = Bit(tnz, 15);
        si[top + 4] = Bit(tnz, 18); si[top + 5] = Bit(tnz, 19);
        si[top + 6] = Bit(tnz, 22); si[top + 7] = Bit(tnz, 23);
        si[top + 8] = Bit(tnz, 24);
        si[left + 0] = Bit(lnz, 3); si[left + 1] = Bit(lnz, 7); si[left + 2] = Bit(lnz, 11); si[left + 3] = Bit(lnz, 15);
        si[left + 4] = Bit(lnz, 17); si[left + 5] = Bit(lnz, 19);
        si[left + 6] = Bit(lnz, 21); si[left + 7] = Bit(lnz, 23);
    }

    private static void BytesToNz(ArrayView<int> nzRow, int x, ArrayView<int> si)
    {
        int nz = 0;
        nz |= (si[ITopNz + 0] << 12) | (si[ITopNz + 1] << 13);
        nz |= (si[ITopNz + 2] << 14) | (si[ITopNz + 3] << 15);
        nz |= (si[ITopNz + 4] << 18) | (si[ITopNz + 5] << 19);
        nz |= (si[ITopNz + 6] << 22) | (si[ITopNz + 7] << 23);
        nz |= si[ITopNz + 8] << 24;
        nz |= (si[ILeftNz + 0] << 3) | (si[ILeftNz + 1] << 7);
        nz |= si[ILeftNz + 2] << 11;
        nz |= (si[ILeftNz + 4] << 17) | (si[ILeftNz + 6] << 21);
        nzRow[1 + x] = nz;
    }

    // ================================================================== reconstruction (quant_enc.c)

    // tmpA / tmpB: 16 ints each of si (one pair per mode lane: the lanes reconstruct in parallel)
    private static int ReconstructIntra4(ArrayView<byte> sb, ArrayView<int> si, ArrayView<int> meta, ArrayView<int> t,
        int seg, int levels, int src, int dst, int mode, int tmpA, int tmpB)
    {
        int refO = BPred + I4Off(mode);
        var tmpV = si.SubView(tmpB, 16);
        FTransform(sb, src, sb, refO, si, tmpA);
        int nz = QuantizeBlock(si, tmpA, si, levels, meta, seg, 0, t);
        ITransformOne(sb, refO, si, tmpA, sb, dst);
        return nz;
    }

    private static int QuantizeSingle(ArrayView<int> v, int o, ArrayView<int> meta, int seg)
    {
        int sbase = M.Seg + seg * M.SegStride;
        int V = v[o];
        bool sign = V < 0;
        if (sign) V = -V;
        int zt = meta[sbase + M.SZtUV];
        if (V > zt)
        {
            int qV = (int)(((uint)V * (uint)meta[sbase + M.SIQUV] + (uint)meta[sbase + M.SBiasUV]) >> 17) * meta[sbase + M.SQUV];
            int err = V - qV;
            v[o] = (short)(sign ? -qV : qV);
            return (sign ? -err : err) >> 1;
        }
        v[o] = 0;
        return (sign ? -V : V) >> 1;
    }

    // uvTmp: the transformed coefficients (128 ints of si, one set per UV mode lane); ILeftDerr / topDerr: read only
    private static void CorrectDCValues(ArrayView<int> si, ArrayView<int> topDerr, int x, ArrayView<int> meta, int seg, int derrOut, int uvTmp)
    {
        for (int ch = 0; ch <= 1; ++ch)
        {
            int topO = x * 4 + ch * 2;
            int leftO = ILeftDerr + ch * 2;
            int c = uvTmp + ch * 4 * 16;
            si[c + 0] = (short)(si[c + 0] + ((7 * topDerr[topO + 0] + 8 * si[leftO + 0]) >> 3));
            int err0 = QuantizeSingle(si, c + 0, meta, seg);
            si[c + 16] = (short)(si[c + 16] + ((7 * topDerr[topO + 1] + 8 * err0) >> 3));
            int err1 = QuantizeSingle(si, c + 16, meta, seg);
            si[c + 32] = (short)(si[c + 32] + ((7 * err0 + 8 * si[leftO + 1]) >> 3));
            int err2 = QuantizeSingle(si, c + 32, meta, seg);
            si[c + 48] = (short)(si[c + 48] + ((7 * err1 + 8 * err2) >> 3));
            int err3 = QuantizeSingle(si, c + 48, meta, seg);
            si[derrOut + ch * 3 + 0] = (sbyte)err1;
            si[derrOut + ch * 3 + 1] = (sbyte)err2;
            si[derrOut + ch * 3 + 2] = (sbyte)err3;
        }
    }

    private static bool IsFlat(ArrayView<int> lv, int o, int numBlocks, int thresh)
    {
        int score = 0;
        bool flat = true;
        for (int b = 0; b < numBlocks; ++b)
            for (int i = 1; i < 16; ++i)
            {
                score += lv[o + b * 16 + i] != 0 ? 1 : 0;
                if (score > thresh) flat = false;
            }
        return flat;
    }

    private static bool IsFlatSource16(ArrayView<byte> sb)
    {
        int v = sb[BIn];
        bool flat = true;
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                if (sb[BIn + y * BPS + x] != v) flat = false;
        return flat;
    }

    private static void CopyBlock(ArrayView<byte> sb, int src, int dst, int w, int h)
    {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) sb[dst + y * BPS + x] = sb[src + y * BPS + x];
    }
}
