// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). CPU reference port of libwebp 1.6.0's lossy (VP8) encoder, BSD
// licence, see NOTICE.md: src/enc/webp_enc.c (InitVP8Encoder, MapConfigToTools), analysis_enc.c, quant_enc.c (RD_OPT_NONE /
// RD_OPT_BASIC paths; trellis is not ported yet, so methods 5 and 6 are rejected), iterator_enc.c, cost_enc.c, dsp/cost.c,
// frame_enc.c (VP8EncTokenLoop for methods >= 3, VP8EncLoop for methods <= 2), token_enc.c, tree_enc.c, filter_enc.c
// (no autofilter), syntax_enc.c. The goal is a BYTE-IDENTICAL bitstream for the same YUV input and config, so the structure,
// the order of every state update and every integer quirk follow the C source (comments name the C function). This is the
// oracle for the accelerator encoder, not the shipping path.

namespace SpawnDev.Codecs.References.Images.WebP.Lossy;

/// <summary>The libwebp WebPConfig fields the lossy encoder uses (defaults = WebPConfigInit, i.e. what Pillow sets).</summary>
public sealed class Vp8LossyEncoderConfig
{
    public float Quality = 75f;
    public int Method = 4;
    public int SnsStrength = 50;
    public int FilterStrength = 60;
    public int FilterSharpness = 0;
    public int FilterType = 1;
    public int Segments = 4;
    public int Partitions = 0;
    public int Pass = 1;
    public int PartitionLimit = 0;
    public int Preprocessing = 0;
    public bool EmulateJpegSize = false;
}

/// <summary>CPU reference of libwebp's VP8 lossy encoder (bit-exact port).</summary>
public static class Vp8LossyEncoderReference
{
    private const int BPS = Vp8EncDsp.BPS;
    private const int NUM_MB_SEGMENTS = 4;
    private const int NUM_TYPES = 4, NUM_BANDS = 8, NUM_CTX = 3, NUM_PROBAS = 11;
    private const int MAX_VARIABLE_LEVEL = 67;
    private const int MAX_ALPHA = 255;
    private const int ALPHA_SCALE = 2 * MAX_ALPHA;
    private const int MAX_ITERS_K_MEANS = 6;
    private const long MAX_COST = 0x7fffffffffffffL;
    private const int RD_DISTO_MULT = 256;
    private const int FLATNESS_LIMIT_I16 = 0, FLATNESS_LIMIT_I4 = 3, FLATNESS_LIMIT_UV = 2, FLATNESS_PENALTY = 140;
    private const int ERROR_DIFFUSION_QUALITY = 98;
    private const int VP8_MAX_PARTITION0_SIZE = 1 << 19;
    private const ulong PARTITION0_SIZE_LIMIT = ((ulong)VP8_MAX_PARTITION0_SIZE - 2048UL) << 11;
    private const int MIN_COUNT = 96;
    private const int SKIP_PROBA_THRESHOLD = 250;
    private const int FIXED_PROBA_BIT = 1 << 14;
    private static readonly ushort[] kWeightY = Vp8EncTables.kWeightY;

    // ======================================================================
    // State (VP8Encoder & friends)
    // ======================================================================

    private sealed class SegmentInfo
    {
        public readonly Vp8Matrix Y1 = new(), Y2 = new(), UV = new();
        public int Alpha, Beta, Quant, FStrength, MaxEdge, MinDisto;
        public int LambdaI16, LambdaI4, LambdaUV, LambdaMode, LambdaTrellis, TLambda;
        public int LambdaTrellisI16, LambdaTrellisI4, LambdaTrellisUV;
        public long I4Penalty;

        public void CopyFrom(SegmentInfo o)
        {
            Y1.CopyFrom(o.Y1); Y2.CopyFrom(o.Y2); UV.CopyFrom(o.UV);
            Alpha = o.Alpha; Beta = o.Beta; Quant = o.Quant; FStrength = o.FStrength; MaxEdge = o.MaxEdge; MinDisto = o.MinDisto;
            LambdaI16 = o.LambdaI16; LambdaI4 = o.LambdaI4; LambdaUV = o.LambdaUV; LambdaMode = o.LambdaMode;
            LambdaTrellis = o.LambdaTrellis; TLambda = o.TLambda; LambdaTrellisI16 = o.LambdaTrellisI16;
            LambdaTrellisI4 = o.LambdaTrellisI4; LambdaTrellisUV = o.LambdaTrellisUV; I4Penalty = o.I4Penalty;
        }
    }

    private struct MBInfo
    {
        public int Type;      // 0 = i4x4, 1 = i16x16
        public int UVMode;
        public int Skip;
        public int Segment;
        public int Alpha;
    }

    private sealed class ModeScore
    {
        public long D, SD, H, R, Score;
        public readonly short[] YDcLevels = new short[16];
        public readonly short[] YAcLevels = new short[16 * 16];
        public readonly short[] UVLevels = new short[8 * 16];
        public int ModeI16;
        public readonly byte[] ModesI4 = new byte[16];
        public int ModeUV;
        public uint Nz;
        public readonly sbyte[] Derr = new sbyte[2 * 3];

        public void InitScore() { D = 0; SD = 0; R = 0; H = 0; Nz = 0; Score = MAX_COST; }
        public void CopyScore(ModeScore s) { D = s.D; SD = s.SD; R = s.R; H = s.H; Nz = s.Nz; Score = s.Score; }
        public void AddScore(ModeScore s) { D += s.D; SD += s.SD; R += s.R; H += s.H; Nz |= s.Nz; Score += s.Score; }

        public void CopyAll(ModeScore s)
        {
            CopyScore(s);
            s.YDcLevels.CopyTo(YDcLevels, 0); s.YAcLevels.CopyTo(YAcLevels, 0); s.UVLevels.CopyTo(UVLevels, 0);
            ModeI16 = s.ModeI16; s.ModesI4.CopyTo(ModesI4, 0); ModeUV = s.ModeUV; s.Derr.CopyTo(Derr, 0);
        }
    }

    private sealed class Proba
    {
        public readonly byte[] Segments = new byte[3];
        public int SkipProba;
        public readonly byte[] Coeffs = new byte[NUM_TYPES * NUM_BANDS * NUM_CTX * NUM_PROBAS];
        public readonly uint[] Stats = new uint[NUM_TYPES * NUM_BANDS * NUM_CTX * NUM_PROBAS];
        public readonly ushort[] LevelCost = new ushort[NUM_TYPES * NUM_BANDS * NUM_CTX * (MAX_VARIABLE_LEVEL + 1)];
        public bool Dirty;
        public bool UseSkipProba;
        public int NbSkip;
    }

    private static int PIdx(int t, int b, int c) => NUM_PROBAS * (c + NUM_CTX * (b + NUM_BANDS * t));
    private static int CostIdx(int t, int b, int c) => (MAX_VARIABLE_LEVEL + 1) * (c + NUM_CTX * (b + NUM_BANDS * t));

    private sealed class Encoder
    {
        public Vp8LossyEncoderConfig Config = null!;
        public byte[] Y = null!, U = null!, V = null!;
        public int YStride, UVStride, Width, Height;

        // filter header
        public bool FilterSimple; public int FilterLevel; public int FilterSharpness; public int I4x4LfDelta;
        // segment header
        public int NumSegments; public bool UpdateMap; public int SegmentHdrSize;
        public int Profile;
        public int MbW, MbH, PredsW;
        public int NumParts;
        public Vp8EncBitWriter Bw = null!;
        public Vp8EncBitWriter[] Parts = null!;
        public readonly List<ushort> Tokens = new();
        public readonly SegmentInfo[] Dqm = { new(), new(), new(), new() };
        public int BaseQuant, Alpha, UVAlpha;
        public int DqY1Dc, DqY2Dc, DqY2Ac, DqUVDc, DqUVAc;
        public readonly Proba Proba = new();
        public int Method;
        public int RdOptLevel;   // 0 none, 1 basic, 2 trellis, 3 trellis all
        public int MaxI4HeaderBits;
        public long MbHeaderLimit;
        public bool DoSearch;
        public bool UseTokens;
        public MBInfo[] MbInfo = null!;
        public byte[] Preds = null!;     // (4*mb_w+1) * (4*mb_h+1); PredsBase = 1 + PredsW (C: enc->preds)
        public int PredsBase;
        public uint[] Nz = null!;        // mb_w + 1; NzBase = 1 (C: enc->nz, nz[-1] at 0)
        public byte[] YTop = null!;      // mb_w*16 luma then mb_w*16 uv (C: y_top, uv_top = y_top + top_stride)
        public int UVTopBase;
        public sbyte[]? TopDerr;         // mb_w * [2][2]
    }

    private sealed class Iterator
    {
        public int X, Y;
        public byte[] YuvIn = new byte[Vp8EncDsp.YUV_SIZE_ENC];
        public byte[] YuvOut = new byte[Vp8EncDsp.YUV_SIZE_ENC];
        public byte[] YuvOut2 = new byte[Vp8EncDsp.YUV_SIZE_ENC];
        public byte[] YuvP = new byte[Vp8EncDsp.PRED_SIZE_ENC];
        public Encoder Enc = null!;
        public int Mb;                   // index into MbInfo
        public Vp8EncBitWriter Bw = null!;
        public int Preds;                // index into Enc.Preds
        public int NzIdx;                // index into Enc.Nz
        public byte[] I4Boundary = new byte[40];
        public int I4Top;
        public int I4;
        public int[] TopNz = new int[9];
        public int[] LeftNz = new int[9];
        public bool DoTrellis;
        public int CountDown;
        public sbyte[] LeftDerr = new sbyte[4];
        public bool HasTopDerr;
        // left samples memory: y_left = 1 (index -1 = 0), u_left = y_left + 32, v_left = u_left + 16
        public byte[] LeftMem = new byte[1 + 16 + 16 + 16 + 8 + 16];
        public const int YLeft = 1, ULeft = 1 + 32, VLeft = 1 + 32 + 16;
        // top samples: either the encoder's top rows or the analysis scratch
        public byte[] TopArr = null!;
        public int YTopOff, UVTopOff;
        public byte[] Tmp32 = new byte[32];
    }

    // ======================================================================
    // Public entry point
    // ======================================================================

    /// <summary>
    /// Encode YUV 4:2:0 planes (tightly packed or strided) into a complete simple-format WebP file ("RIFF....WEBPVP8 ").
    /// The planes are what libwebp's WebPPicture would hold after its RGBA import (and transparent-area cleanup).
    /// </summary>
    public static byte[] EncodeWebP(byte[] y, byte[] u, byte[] v, int width, int height, Vp8LossyEncoderConfig config,
        int yStride = 0, int uvStride = 0)
    {
        var vp8 = EncodeVp8(y, u, v, width, height, config, yStride, uvStride, out _);
        return vp8;
    }

    /// <summary>Same as <see cref="EncodeWebP"/>; also returns the raw VP8 payload offset (20 for the simple format).</summary>
    public static byte[] EncodeVp8(byte[] y, byte[] u, byte[] v, int width, int height, Vp8LossyEncoderConfig config,
        int yStride, int uvStride, out int vp8PayloadOffset)
    {
        if (config.Method > 4) throw new NotSupportedException("methods 5 and 6 need the trellis quantizer, not ported yet");
        if (config.Partitions != 0) throw new NotSupportedException("multiple partitions are not ported yet");
        var enc = InitVP8Encoder(config, y, u, v, width, height, yStride == 0 ? width : yStride, uvStride == 0 ? (width + 1) / 2 : uvStride);
        VP8EncAnalyze(enc);
        if (!enc.UseTokens) VP8EncLoop(enc);
        else VP8EncTokenLoop(enc);
        vp8PayloadOffset = 20;
        return VP8EncWrite(enc);
    }

    // ======================================================================
    // webp_enc.c
    // ======================================================================

    private static Encoder InitVP8Encoder(Vp8LossyEncoderConfig config, byte[] y, byte[] u, byte[] v, int width, int height, int yStride, int uvStride)
    {
        var enc = new Encoder
        {
            Config = config, Y = y, U = u, V = v, Width = width, Height = height, YStride = yStride, UVStride = uvStride,
        };
        bool useFilter = config.FilterStrength > 0;   // || autofilter (not supported)
        enc.MbW = (width + 15) >> 4;
        enc.MbH = (height + 15) >> 4;
        enc.PredsW = 4 * enc.MbW + 1;
        int predsH = 4 * enc.MbH + 1;
        enc.Preds = new byte[enc.PredsW * predsH];
        enc.PredsBase = 1 + enc.PredsW;
        enc.Nz = new uint[enc.MbW + 1];
        enc.MbInfo = new MBInfo[enc.MbW * enc.MbH];
        enc.YTop = new byte[2 * enc.MbW * 16];
        enc.UVTopBase = enc.MbW * 16;
        if (config.Quality <= ERROR_DIFFUSION_QUALITY || config.Pass > 1) enc.TopDerr = new sbyte[enc.MbW * 4];
        enc.NumParts = 1 << config.Partitions;
        enc.Profile = useFilter ? (config.FilterType == 1 ? 0 : 1) : 2;

        // MapConfigToTools
        int method = config.Method;
        int limit = 100 - config.PartitionLimit;
        enc.Method = method;
        enc.RdOptLevel = method >= 6 ? 3 : method >= 5 ? 2 : method >= 3 ? 1 : 0;
        enc.MaxI4HeaderBits = 256 * 16 * 16 * (limit * limit) / (100 * 100);
        enc.MbHeaderLimit = (long)256 * 510 * 8 * 1024 / (enc.MbW * enc.MbH);
        enc.DoSearch = false;   // no target_size / target_PSNR
        enc.UseTokens = enc.RdOptLevel >= 1;
        if (enc.UseTokens) enc.NumParts = 1;

        // VP8DefaultProbas
        enc.Proba.UseSkipProba = false;
        enc.Proba.Segments[0] = enc.Proba.Segments[1] = enc.Proba.Segments[2] = 255;
        System.Array.Copy(Vp8EncTables.VP8CoeffsProba0, enc.Proba.Coeffs, enc.Proba.Coeffs.Length);
        enc.Proba.Dirty = true;
        // ResetSegmentHeader
        enc.NumSegments = config.Segments;
        enc.UpdateMap = enc.NumSegments > 1;
        enc.SegmentHdrSize = 0;
        // ResetFilterHeader
        enc.FilterSimple = true; enc.FilterLevel = 0; enc.FilterSharpness = 0; enc.I4x4LfDelta = 0;
        // ResetBoundaryPredictions: top row (from -1) and left column = B_DC_PRED (0): the array is zeroed already.

        enc.Parts = new Vp8EncBitWriter[enc.NumParts];
        return enc;
    }

    // ======================================================================
    // iterator_enc.c
    // ======================================================================

    private static void InitLeft(Iterator it)
    {
        byte corner = (byte)(it.Y > 0 ? 129 : 127);
        it.LeftMem[Iterator.YLeft - 1] = it.LeftMem[Iterator.ULeft - 1] = it.LeftMem[Iterator.VLeft - 1] = corner;
        for (int i = 0; i < 16; i++) it.LeftMem[Iterator.YLeft + i] = 129;
        for (int i = 0; i < 8; i++) { it.LeftMem[Iterator.ULeft + i] = 129; it.LeftMem[Iterator.VLeft + i] = 129; }
        it.LeftNz[8] = 0;
        if (it.HasTopDerr) System.Array.Clear(it.LeftDerr);
    }

    private static void InitTop(Iterator it)
    {
        var enc = it.Enc;
        System.Array.Fill(enc.YTop, (byte)127);
        System.Array.Clear(enc.Nz);   // nz[-1] stays 0 (constant)
        if (enc.TopDerr != null) System.Array.Clear(enc.TopDerr);
    }

    private static void SetRow(Iterator it, int y)
    {
        var enc = it.Enc;
        it.X = 0;
        it.Y = y;
        it.Bw = enc.Parts[y & (enc.NumParts - 1)];
        it.Preds = enc.PredsBase + y * 4 * enc.PredsW;
        it.NzIdx = 1;
        it.Mb = y * enc.MbW;
        it.TopArr = enc.YTop;
        it.YTopOff = 0;
        it.UVTopOff = enc.UVTopBase;
        InitLeft(it);
    }

    private static Iterator IteratorInit(Encoder enc)
    {
        var it = new Iterator { Enc = enc, HasTopDerr = enc.TopDerr != null };
        // VP8IteratorReset
        SetRow(it, 0);
        it.CountDown = enc.MbW * enc.MbH;
        InitTop(it);
        it.DoTrellis = false;
        return it;
    }

    private static void ImportBlock(byte[] src, int srcO, int srcStride, byte[] dst, int dstO, int w, int h, int size)
    {
        int i;
        for (i = 0; i < h; ++i)
        {
            System.Array.Copy(src, srcO, dst, dstO, w);
            if (w < size)
            {
                byte last = dst[dstO + w - 1];
                for (int k = w; k < size; k++) dst[dstO + k] = last;
            }
            dstO += BPS;
            srcO += srcStride;
        }
        for (; i < size; ++i)
        {
            System.Array.Copy(dst, dstO - BPS, dst, dstO, size);
            dstO += BPS;
        }
    }

    private static void ImportLine(byte[] src, int srcO, int srcStride, byte[] dst, int dstO, int len, int totalLen)
    {
        int i;
        for (i = 0; i < len; ++i, srcO += srcStride) dst[dstO + i] = src[srcO];
        for (; i < totalLen; ++i) dst[dstO + i] = dst[dstO + len - 1];
    }

    private static void IteratorImport(Iterator it, bool withBoundary)
    {
        var enc = it.Enc;
        int x = it.X, y = it.Y;
        int ysrc = (y * enc.YStride + x) * 16;
        int usrc = (y * enc.UVStride + x) * 8;
        int w = Math.Min(enc.Width - x * 16, 16);
        int h = Math.Min(enc.Height - y * 16, 16);
        int uvW = (w + 1) >> 1, uvH = (h + 1) >> 1;

        ImportBlock(enc.Y, ysrc, enc.YStride, it.YuvIn, Vp8EncDsp.Y_OFF_ENC, w, h, 16);
        ImportBlock(enc.U, usrc, enc.UVStride, it.YuvIn, Vp8EncDsp.U_OFF_ENC, uvW, uvH, 8);
        ImportBlock(enc.V, usrc, enc.UVStride, it.YuvIn, Vp8EncDsp.V_OFF_ENC, uvW, uvH, 8);
        if (!withBoundary) return;

        if (x == 0) InitLeft(it);
        else
        {
            if (y == 0)
            {
                it.LeftMem[Iterator.YLeft - 1] = it.LeftMem[Iterator.ULeft - 1] = it.LeftMem[Iterator.VLeft - 1] = 127;
            }
            else
            {
                it.LeftMem[Iterator.YLeft - 1] = enc.Y[ysrc - 1 - enc.YStride];
                it.LeftMem[Iterator.ULeft - 1] = enc.U[usrc - 1 - enc.UVStride];
                it.LeftMem[Iterator.VLeft - 1] = enc.V[usrc - 1 - enc.UVStride];
            }
            ImportLine(enc.Y, ysrc - 1, enc.YStride, it.LeftMem, Iterator.YLeft, h, 16);
            ImportLine(enc.U, usrc - 1, enc.UVStride, it.LeftMem, Iterator.ULeft, uvH, 8);
            ImportLine(enc.V, usrc - 1, enc.UVStride, it.LeftMem, Iterator.VLeft, uvH, 8);
        }

        it.TopArr = it.Tmp32;
        it.YTopOff = 0;
        it.UVTopOff = 16;
        if (y == 0) System.Array.Fill(it.Tmp32, (byte)127);
        else
        {
            ImportLine(enc.Y, ysrc - enc.YStride, 1, it.Tmp32, 0, w, 16);
            ImportLine(enc.U, usrc - enc.UVStride, 1, it.Tmp32, 16, uvW, 8);
            ImportLine(enc.V, usrc - enc.UVStride, 1, it.Tmp32, 16 + 8, uvW, 8);
        }
    }

    private static int Bit(uint nz, int n) => (nz & (1u << n)) != 0 ? 1 : 0;

    private static void NzToBytes(Iterator it)
    {
        uint tnz = it.Enc.Nz[it.NzIdx], lnz = it.Enc.Nz[it.NzIdx - 1];
        var t = it.TopNz; var l = it.LeftNz;
        t[0] = Bit(tnz, 12); t[1] = Bit(tnz, 13); t[2] = Bit(tnz, 14); t[3] = Bit(tnz, 15);
        t[4] = Bit(tnz, 18); t[5] = Bit(tnz, 19);
        t[6] = Bit(tnz, 22); t[7] = Bit(tnz, 23);
        t[8] = Bit(tnz, 24);
        l[0] = Bit(lnz, 3); l[1] = Bit(lnz, 7); l[2] = Bit(lnz, 11); l[3] = Bit(lnz, 15);
        l[4] = Bit(lnz, 17); l[5] = Bit(lnz, 19);
        l[6] = Bit(lnz, 21); l[7] = Bit(lnz, 23);
    }

    private static void BytesToNz(Iterator it)
    {
        uint nz = 0;
        var t = it.TopNz; var l = it.LeftNz;
        nz |= (uint)((t[0] << 12) | (t[1] << 13));
        nz |= (uint)((t[2] << 14) | (t[3] << 15));
        nz |= (uint)((t[4] << 18) | (t[5] << 19));
        nz |= (uint)((t[6] << 22) | (t[7] << 23));
        nz |= (uint)(t[8] << 24);
        nz |= (uint)((l[0] << 3) | (l[1] << 7));
        nz |= (uint)(l[2] << 11);
        nz |= (uint)((l[4] << 17) | (l[6] << 21));
        it.Enc.Nz[it.NzIdx] = nz;
    }

    private static void SaveBoundary(Iterator it)
    {
        var enc = it.Enc;
        int x = it.X, y = it.Y;
        var o = it.YuvOut;
        if (x < enc.MbW - 1)
        {
            for (int i = 0; i < 16; ++i) it.LeftMem[Iterator.YLeft + i] = o[Vp8EncDsp.Y_OFF_ENC + 15 + i * BPS];
            for (int i = 0; i < 8; ++i)
            {
                it.LeftMem[Iterator.ULeft + i] = o[Vp8EncDsp.U_OFF_ENC + 7 + i * BPS];
                it.LeftMem[Iterator.VLeft + i] = o[Vp8EncDsp.U_OFF_ENC + 15 + i * BPS];
            }
            it.LeftMem[Iterator.YLeft - 1] = it.TopArr[it.YTopOff + 15];
            it.LeftMem[Iterator.ULeft - 1] = it.TopArr[it.UVTopOff + 0 + 7];
            it.LeftMem[Iterator.VLeft - 1] = it.TopArr[it.UVTopOff + 8 + 7];
        }
        if (y < enc.MbH - 1)
        {
            System.Array.Copy(o, Vp8EncDsp.Y_OFF_ENC + 15 * BPS, it.TopArr, it.YTopOff, 16);
            System.Array.Copy(o, Vp8EncDsp.U_OFF_ENC + 7 * BPS, it.TopArr, it.UVTopOff, 16);
        }
    }

    private static bool IteratorNext(Iterator it)
    {
        if (++it.X == it.Enc.MbW) SetRow(it, ++it.Y);
        else
        {
            it.Preds += 4;
            it.Mb += 1;
            it.NzIdx += 1;
            it.YTopOff += 16;
            it.UVTopOff += 16;
        }
        return 0 < --it.CountDown;
    }

    private static void SetIntra16Mode(Iterator it, int mode)
    {
        var p = it.Enc.Preds;
        int o = it.Preds;
        for (int y = 0; y < 4; ++y)
        {
            for (int k = 0; k < 4; k++) p[o + k] = (byte)mode;
            o += it.Enc.PredsW;
        }
        it.Enc.MbInfo[it.Mb].Type = 1;
    }

    private static void SetIntra4Mode(Iterator it, byte[] modes)
    {
        var p = it.Enc.Preds;
        int o = it.Preds, m = 0;
        for (int y = 4; y > 0; --y)
        {
            for (int k = 0; k < 4; k++) p[o + k] = modes[m + k];
            o += it.Enc.PredsW;
            m += 4;
        }
        it.Enc.MbInfo[it.Mb].Type = 0;
    }

    private static readonly int[] TopLeftI4 = { 17, 21, 25, 29, 13, 17, 21, 25, 9, 13, 17, 21, 5, 9, 13, 17 };

    private static void StartI4(Iterator it)
    {
        var enc = it.Enc;
        it.I4 = 0;
        it.I4Top = TopLeftI4[0];
        for (int i = 0; i < 17; ++i) it.I4Boundary[i] = it.LeftMem[Iterator.YLeft + 15 - i];
        for (int i = 0; i < 16; ++i) it.I4Boundary[17 + i] = it.TopArr[it.YTopOff + i];
        if (it.X < enc.MbW - 1)
            for (int i = 16; i < 16 + 4; ++i) it.I4Boundary[17 + i] = it.TopArr[it.YTopOff + i];
        else
            for (int i = 16; i < 16 + 4; ++i) it.I4Boundary[17 + i] = it.I4Boundary[17 + 15];
        NzToBytes(it);
    }

    private static bool RotateI4(Iterator it, byte[] yuvOut, int yuvOutOff)
    {
        int blk = yuvOutOff + Vp8EncDsp.Scan[it.I4];
        int top = it.I4Top;
        var b = it.I4Boundary;
        for (int i = 0; i <= 3; ++i) b[top - 4 + i] = yuvOut[blk + i + 3 * BPS];
        if ((it.I4 & 3) != 3)
            for (int i = 0; i <= 2; ++i) b[top + i] = yuvOut[blk + 3 + (2 - i) * BPS];
        else
            for (int i = 0; i <= 3; ++i) b[top + i] = b[top + i + 4];
        ++it.I4;
        if (it.I4 == 16) return false;
        it.I4Top = TopLeftI4[it.I4];
        return true;
    }

    private static void SwapOut(Iterator it) => (it.YuvOut, it.YuvOut2) = (it.YuvOut2, it.YuvOut);

    // ======================================================================
    // analysis_enc.c
    // ======================================================================

    private static int Clip(int v, int m, int M) => v < m ? m : v > M ? M : v;

    private static int GetAlpha(int maxValue, int lastNonZero) => maxValue > 1 ? ALPHA_SCALE * lastNonZero / maxValue : 0;

    private static void MakeLuma16Preds(Iterator it)
    {
        byte[]? left = it.X != 0 ? it.LeftMem : null;
        byte[]? top = it.Y != 0 ? it.TopArr : null;
        Vp8EncDsp.Intra16Preds(it.YuvP, left, Iterator.YLeft, top, it.YTopOff);
    }

    private static void MakeChroma8Preds(Iterator it)
    {
        byte[]? left = it.X != 0 ? it.LeftMem : null;
        byte[]? top = it.Y != 0 ? it.TopArr : null;
        Vp8EncDsp.IntraChromaPreds(it.YuvP, left, Iterator.ULeft, top, it.UVTopOff);
    }

    private static int MBAnalyzeBestIntra16Mode(Iterator it)
    {
        const int maxMode = 2;
        int bestAlpha = -1, bestMode = 0;
        MakeLuma16Preds(it);
        for (int mode = 0; mode < maxMode; ++mode)
        {
            var (mv, lnz) = Vp8EncDsp.CollectHistogram(it.YuvIn, Vp8EncDsp.Y_OFF_ENC, it.YuvP, Vp8EncDsp.I16ModeOffsets[mode], 0, 16);
            int alpha = GetAlpha(mv, lnz);
            if (alpha > bestAlpha) { bestAlpha = alpha; bestMode = mode; }
        }
        SetIntra16Mode(it, bestMode);
        return bestAlpha;
    }

    private static int FastMBAnalyze(Iterator it)
    {
        int q = (int)it.Enc.Config.Quality;
        uint kThreshold = (uint)(8 + (17 - 8) * q / 100);
        Span<uint> dc = stackalloc uint[16];
        for (int k = 0; k < 16; k += 4) Vp8EncDsp.Mean16x4(it.YuvIn, Vp8EncDsp.Y_OFF_ENC + k * BPS, dc.Slice(k, 4));
        uint m = 0, m2 = 0;
        for (int k = 0; k < 16; ++k) { m += dc[k]; m2 += dc[k] * dc[k]; }
        if (kThreshold * m2 < m * m) SetIntra16Mode(it, 0);
        else SetIntra4Mode(it, new byte[16]);
        return 0;
    }

    private static int MBAnalyzeBestUVMode(Iterator it)
    {
        int bestAlpha = -1, smallestAlpha = 0, bestMode = 0;
        const int maxMode = 2;
        MakeChroma8Preds(it);
        for (int mode = 0; mode < maxMode; ++mode)
        {
            var (mv, lnz) = Vp8EncDsp.CollectHistogram(it.YuvIn, Vp8EncDsp.U_OFF_ENC, it.YuvP, Vp8EncDsp.UVModeOffsets[mode], 16, 16 + 4 + 4);
            int alpha = GetAlpha(mv, lnz);
            if (alpha > bestAlpha) bestAlpha = alpha;
            if (mode == 0 || alpha < smallestAlpha) { smallestAlpha = alpha; bestMode = mode; }
        }
        it.Enc.MbInfo[it.Mb].UVMode = bestMode;
        return bestAlpha;
    }

    private static void MBAnalyze(Iterator it, int[] alphas, ref int alpha, ref int uvAlpha)
    {
        var enc = it.Enc;
        SetIntra16Mode(it, 0);
        enc.MbInfo[it.Mb].Skip = 0;
        enc.MbInfo[it.Mb].Segment = 0;
        int bestAlpha = enc.Method <= 1 ? FastMBAnalyze(it) : MBAnalyzeBestIntra16Mode(it);
        int bestUvAlpha = MBAnalyzeBestUVMode(it);
        bestAlpha = (3 * bestAlpha + bestUvAlpha + 2) >> 2;
        bestAlpha = Clip(MAX_ALPHA - bestAlpha, 0, MAX_ALPHA);   // FinalAlphaValue
        alphas[bestAlpha]++;
        enc.MbInfo[it.Mb].Alpha = bestAlpha;
        alpha += bestAlpha;
        uvAlpha += bestUvAlpha;
    }

    private static void SetSegmentAlphas(Encoder enc, int[] centers, int mid)
    {
        int nb = enc.NumSegments;
        int min = centers[0], max = centers[0];
        if (nb > 1)
            for (int n = 0; n < nb; ++n) { if (min > centers[n]) min = centers[n]; if (max < centers[n]) max = centers[n]; }
        if (max == min) max = min + 1;
        for (int n = 0; n < nb; ++n)
        {
            int alpha = 255 * (centers[n] - mid) / (max - min);
            int beta = 255 * (centers[n] - min) / (max - min);
            enc.Dqm[n].Alpha = Clip(alpha, -127, 127);
            enc.Dqm[n].Beta = Clip(beta, 0, 255);
        }
    }

    private static void AssignSegments(Encoder enc, int[] alphas)
    {
        int nb = enc.NumSegments < NUM_MB_SEGMENTS ? enc.NumSegments : NUM_MB_SEGMENTS;
        var centers = new int[NUM_MB_SEGMENTS];
        int weightedAverage = 0;
        var map = new int[MAX_ALPHA + 1];
        var accum = new int[NUM_MB_SEGMENTS];
        var distAccum = new int[NUM_MB_SEGMENTS];
        int n, a, k;
        for (n = 0; n <= MAX_ALPHA && alphas[n] == 0; ++n) { }
        int minA = n;
        for (n = MAX_ALPHA; n > minA && alphas[n] == 0; --n) { }
        int maxA = n;
        int rangeA = maxA - minA;
        for (k = 0, n = 1; k < nb; ++k, n += 2) centers[k] = minA + (n * rangeA) / (2 * nb);

        for (k = 0; k < MAX_ITERS_K_MEANS; ++k)
        {
            for (n = 0; n < nb; ++n) { accum[n] = 0; distAccum[n] = 0; }
            n = 0;
            for (a = minA; a <= maxA; ++a)
            {
                if (alphas[a] != 0)
                {
                    while (n + 1 < nb && Math.Abs(a - centers[n + 1]) < Math.Abs(a - centers[n])) n++;
                    map[a] = n;
                    distAccum[n] += a * alphas[a];
                    accum[n] += alphas[a];
                }
            }
            int displaced = 0;
            weightedAverage = 0;
            int totalWeight = 0;
            for (n = 0; n < nb; ++n)
            {
                if (accum[n] != 0)
                {
                    int newCenter = (distAccum[n] + accum[n] / 2) / accum[n];
                    displaced += Math.Abs(centers[n] - newCenter);
                    centers[n] = newCenter;
                    weightedAverage += newCenter * accum[n];
                    totalWeight += accum[n];
                }
            }
            weightedAverage = (weightedAverage + totalWeight / 2) / totalWeight;
            if (displaced < 5) break;
        }

        for (n = 0; n < enc.MbW * enc.MbH; ++n)
        {
            int alpha = enc.MbInfo[n].Alpha;
            enc.MbInfo[n].Segment = map[alpha];
            enc.MbInfo[n].Alpha = centers[map[alpha]];
        }
        if (nb > 1 && (enc.Config.Preprocessing & 1) != 0) SmoothSegmentMap(enc);
        SetSegmentAlphas(enc, centers, weightedAverage);
    }

    private static void SmoothSegmentMap(Encoder enc)
    {
        int w = enc.MbW, h = enc.MbH;
        var tmp = new int[w * h];
        for (int y = 1; y < h - 1; ++y)
            for (int x = 1; x < w - 1; ++x)
            {
                var cnt = new int[NUM_MB_SEGMENTS];
                int i = x + w * y;
                int majority = enc.MbInfo[i].Segment;
                cnt[enc.MbInfo[i - w - 1].Segment]++; cnt[enc.MbInfo[i - w].Segment]++; cnt[enc.MbInfo[i - w + 1].Segment]++;
                cnt[enc.MbInfo[i - 1].Segment]++; cnt[enc.MbInfo[i + 1].Segment]++;
                cnt[enc.MbInfo[i + w - 1].Segment]++; cnt[enc.MbInfo[i + w].Segment]++; cnt[enc.MbInfo[i + w + 1].Segment]++;
                for (int s = 0; s < NUM_MB_SEGMENTS; ++s) if (cnt[s] >= 5) { majority = s; break; }
                tmp[i] = majority;
            }
        for (int y = 1; y < h - 1; ++y)
            for (int x = 1; x < w - 1; ++x) enc.MbInfo[x + w * y].Segment = tmp[x + y * w];
    }

    private static void VP8EncAnalyze(Encoder enc)
    {
        bool doSegments = enc.Config.EmulateJpegSize || enc.NumSegments > 1 || enc.Method <= 1;
        if (doSegments)
        {
            var alphas = new int[MAX_ALPHA + 1];
            int alpha = 0, uvAlpha = 0;
            var it = IteratorInit(enc);   // single-thread job over all rows
            do
            {
                IteratorImport(it, withBoundary: true);
                MBAnalyze(it, alphas, ref alpha, ref uvAlpha);
            } while (IteratorNext(it));
            int totalMb = enc.MbH * enc.MbW;
            enc.Alpha = alpha / totalMb;
            enc.UVAlpha = uvAlpha / totalMb;
            AssignSegments(enc, alphas);
        }
        else
        {
            for (int n = 0; n < enc.MbW * enc.MbH; ++n) enc.MbInfo[n] = new MBInfo { Type = 1 };
            enc.Dqm[0].Alpha = 0; enc.Dqm[0].Beta = 0;
            enc.Alpha = 0; enc.UVAlpha = 0;
        }
    }

    // ======================================================================
    // quant_enc.c: segment parameters
    // ======================================================================

    private static int ExpandMatrix(Vp8Matrix m, int type)
    {
        for (int i = 0; i < 2; ++i)
        {
            int isAc = i > 0 ? 1 : 0;
            int bias = Vp8EncTables.kBiasMatrices[type * 2 + isAc];
            m.IQ[i] = (ushort)((1 << Vp8EncDsp.QFIX) / m.Q[i]);
            m.Bias[i] = (uint)(bias << (Vp8EncDsp.QFIX - 8));
            m.ZThresh[i] = ((1u << Vp8EncDsp.QFIX) - 1 - m.Bias[i]) / m.IQ[i];
        }
        for (int i = 2; i < 16; ++i)
        {
            m.Q[i] = m.Q[1]; m.IQ[i] = m.IQ[1]; m.Bias[i] = m.Bias[1]; m.ZThresh[i] = m.ZThresh[1];
        }
        int sum = 0;
        for (int i = 0; i < 16; ++i)
        {
            m.Sharpen[i] = type == 0 ? (ushort)((Vp8EncTables.kFreqSharpening[i] * m.Q[i]) >> 11) : (ushort)0;
            sum += m.Q[i];
        }
        return (sum + 8) >> 4;
    }

    private static int CheckLambda(int v) => v < 1 ? 1 : v;

    private static void SetupMatrices(Encoder enc)
    {
        int tlambdaScale = enc.Method >= 4 ? enc.Config.SnsStrength : 0;
        for (int i = 0; i < enc.NumSegments; ++i)
        {
            var m = enc.Dqm[i];
            int q = m.Quant;
            m.Y1.Q[0] = Vp8EncTables.kDcTable[Clip(q + enc.DqY1Dc, 0, 127)];
            m.Y1.Q[1] = Vp8EncTables.kAcTable[Clip(q, 0, 127)];
            m.Y2.Q[0] = (ushort)(Vp8EncTables.kDcTable[Clip(q + enc.DqY2Dc, 0, 127)] * 2);
            m.Y2.Q[1] = Vp8EncTables.kAcTable2[Clip(q + enc.DqY2Ac, 0, 127)];
            m.UV.Q[0] = Vp8EncTables.kDcTable[Clip(q + enc.DqUVDc, 0, 117)];
            m.UV.Q[1] = Vp8EncTables.kAcTable[Clip(q + enc.DqUVAc, 0, 127)];
            int qI4 = ExpandMatrix(m.Y1, 0);
            int qI16 = ExpandMatrix(m.Y2, 1);
            int qUV = ExpandMatrix(m.UV, 2);
            m.LambdaI4 = CheckLambda((3 * qI4 * qI4) >> 7);
            m.LambdaI16 = CheckLambda(3 * qI16 * qI16);
            m.LambdaUV = CheckLambda((3 * qUV * qUV) >> 6);
            m.LambdaMode = CheckLambda((1 * qI4 * qI4) >> 7);
            m.LambdaTrellisI4 = CheckLambda((7 * qI4 * qI4) >> 3);
            m.LambdaTrellisI16 = CheckLambda((qI16 * qI16) >> 2);
            m.LambdaTrellisUV = CheckLambda((qUV * qUV) << 1);
            m.TLambda = CheckLambda((tlambdaScale * qI4) >> 5);
            m.MinDisto = 20 * m.Y1.Q[0];
            m.MaxEdge = 0;
            m.I4Penalty = 1000L * qI4 * qI4;
        }
    }

    private static int FilterStrengthFromDelta(int sharpness, int delta)
    {
        int pos = delta < 64 ? delta : 63;
        return Vp8EncTables.kLevelsFromDelta[sharpness * 64 + pos];
    }

    private static void SetupFilterStrength(Encoder enc)
    {
        int level0 = 5 * enc.Config.FilterStrength;
        for (int i = 0; i < NUM_MB_SEGMENTS; ++i)
        {
            var m = enc.Dqm[i];
            int qstep = Vp8EncTables.kAcTable[Clip(m.Quant, 0, 127)] >> 2;
            int baseStrength = FilterStrengthFromDelta(enc.FilterSharpness, qstep);
            int f = baseStrength * level0 / (256 + m.Beta);
            m.FStrength = f < 2 ? 0 : f > 63 ? 63 : f;
        }
        enc.FilterLevel = enc.Dqm[0].FStrength;
        enc.FilterSimple = enc.Config.FilterType == 0;
        enc.FilterSharpness = enc.Config.FilterSharpness;
    }

    private static double QualityToCompression(double c)
    {
        double linearC = c < 0.75 ? c * (2.0 / 3.0) : 2.0 * c - 1.0;
        return Math.Pow(linearC, 1 / 3.0);
    }

    private static double QualityToJPEGCompression(double c, double alpha)
    {
        const double amin = 0.30, amax = 0.85, expMin = 0.4, expMax = 0.9;
        const double slope = (expMin - expMax) / (amax - amin);
        double expn = alpha > amax ? expMin : alpha < amin ? expMax : expMax + slope * (alpha - amin);
        return Math.Pow(c, expn);
    }

    private static void SimplifySegments(Encoder enc)
    {
        var map = new[] { 0, 1, 2, 3 };
        int numSegments = enc.NumSegments < NUM_MB_SEGMENTS ? enc.NumSegments : NUM_MB_SEGMENTS;
        int numFinal = 1;
        for (int s1 = 1; s1 < numSegments; ++s1)
        {
            var S1 = enc.Dqm[s1];
            bool found = false;
            int s2;
            for (s2 = 0; s2 < numFinal; ++s2)
            {
                var S2 = enc.Dqm[s2];
                if (S1.Quant == S2.Quant && S1.FStrength == S2.FStrength) { found = true; break; }
            }
            map[s1] = s2;
            if (!found)
            {
                if (numFinal != s1) enc.Dqm[numFinal].CopyFrom(enc.Dqm[s1]);
                ++numFinal;
            }
        }
        if (numFinal < numSegments)
        {
            for (int i = enc.MbW * enc.MbH - 1; i >= 0; i--) enc.MbInfo[i].Segment = map[enc.MbInfo[i].Segment];
            enc.NumSegments = numFinal;
            for (int i = numFinal; i < numSegments; ++i) enc.Dqm[i].CopyFrom(enc.Dqm[numFinal - 1]);
        }
    }

    private static void VP8SetSegmentParams(Encoder enc, float quality)
    {
        int numSegments = enc.NumSegments;
        double amp = 0.9 * enc.Config.SnsStrength / 100.0 / 128.0;
        double Q = quality / 100.0;
        double cBase = enc.Config.EmulateJpegSize ? QualityToJPEGCompression(Q, enc.Alpha / 255.0) : QualityToCompression(Q);
        for (int i = 0; i < numSegments; ++i)
        {
            double expn = 1.0 - amp * enc.Dqm[i].Alpha;
            double c = Math.Pow(cBase, expn);
            int q = (int)(127.0 * (1.0 - c));
            enc.Dqm[i].Quant = Clip(q, 0, 127);
        }
        enc.BaseQuant = enc.Dqm[0].Quant;
        for (int i = numSegments; i < NUM_MB_SEGMENTS; ++i) enc.Dqm[i].Quant = enc.BaseQuant;

        int dqUvAc = (enc.UVAlpha - 64) * (6 - (-4)) / (100 - 30);
        dqUvAc = dqUvAc * enc.Config.SnsStrength / 100;
        dqUvAc = Clip(dqUvAc, -4, 6);
        int dqUvDc = -4 * enc.Config.SnsStrength / 100;
        dqUvDc = Clip(dqUvDc, -15, 15);
        enc.DqY1Dc = 0; enc.DqY2Dc = 0; enc.DqY2Ac = 0;
        enc.DqUVDc = dqUvDc; enc.DqUVAc = dqUvAc;

        SetupFilterStrength(enc);
        if (numSegments > 1) SimplifySegments(enc);
        SetupMatrices(enc);
    }

    // ======================================================================
    // cost_enc.c / dsp/cost.c
    // ======================================================================

    private static int BitCost(int bit, int proba) => bit == 0 ? Vp8EncTables.VP8EntropyCost[proba] : Vp8EncTables.VP8EntropyCost[255 - proba];

    private static int VariableLevelCost(int level, byte[] coeffs, int p)
    {
        int pattern = Vp8EncTables.VP8LevelCodes[(level - 1) * 2 + 0];
        int bits = Vp8EncTables.VP8LevelCodes[(level - 1) * 2 + 1];
        int cost = 0;
        for (int i = 2; pattern != 0; ++i)
        {
            if ((pattern & 1) != 0) cost += BitCost(bits & 1, coeffs[p + i]);
            bits >>= 1;
            pattern >>= 1;
        }
        return cost;
    }

    private static void CalculateLevelCosts(Proba proba)
    {
        if (!proba.Dirty) return;
        for (int ctype = 0; ctype < NUM_TYPES; ++ctype)
            for (int band = 0; band < NUM_BANDS; ++band)
                for (int ctx = 0; ctx < NUM_CTX; ++ctx)
                {
                    int p = PIdx(ctype, band, ctx);
                    int table = CostIdx(ctype, band, ctx);
                    int cost0 = ctx > 0 ? BitCost(1, proba.Coeffs[p + 0]) : 0;
                    int costBase = BitCost(1, proba.Coeffs[p + 1]) + cost0;
                    proba.LevelCost[table + 0] = (ushort)(BitCost(0, proba.Coeffs[p + 1]) + cost0);
                    for (int v = 1; v <= MAX_VARIABLE_LEVEL; ++v)
                        proba.LevelCost[table + v] = (ushort)(costBase + VariableLevelCost(v, proba.Coeffs, p));
                }
        proba.Dirty = false;
    }

    private static int LevelCost(ushort[] costs, int table, int level)
        => Vp8EncTables.VP8LevelFixedCosts[level] + costs[table + (level > MAX_VARIABLE_LEVEL ? MAX_VARIABLE_LEVEL : level)];

    /// <summary>VP8Residual.</summary>
    private struct Residual
    {
        public int First, Last, CoeffType;
        public short[] Coeffs;
        public int CoeffsOff;
    }

    private static void InitResidual(int first, int coeffType, ref Residual res)
    {
        res.First = first;
        res.CoeffType = coeffType;
    }

    private static void SetResidualCoeffs(short[] coeffs, int off, ref Residual res)
    {
        res.Last = -1;
        for (int n = 15; n >= 0; --n)
            if (coeffs[off + n] != 0) { res.Last = n; break; }
        res.Coeffs = coeffs;
        res.CoeffsOff = off;
    }

    // remapped_costs[type][n][ctx] = level_cost[type][VP8EncBands[n]][ctx]
    private static int RemappedCost(int type, int n, int ctx) => CostIdx(type, Vp8EncTables.VP8EncBands[n], ctx);

    private static int GetResidualCost(Proba proba, int ctx0, ref Residual res)
    {
        int n = res.First;
        int p0 = proba.Coeffs[PIdx(res.CoeffType, n, ctx0) + 0];
        int t = RemappedCost(res.CoeffType, n, ctx0);
        int cost = ctx0 == 0 ? BitCost(1, p0) : 0;
        if (res.Last < 0) return BitCost(0, p0);
        for (; n < res.Last; ++n)
        {
            int v = Math.Abs(res.Coeffs[res.CoeffsOff + n]);
            int ctx = v >= 2 ? 2 : v;
            cost += LevelCost(proba.LevelCost, t, v);
            t = RemappedCost(res.CoeffType, n + 1, ctx);
        }
        {
            int v = Math.Abs(res.Coeffs[res.CoeffsOff + n]);
            cost += LevelCost(proba.LevelCost, t, v);
            if (n < 15)
            {
                int b = Vp8EncTables.VP8EncBands[n + 1];
                int ctx = v == 1 ? 1 : 2;
                int lastP0 = proba.Coeffs[PIdx(res.CoeffType, b, ctx) + 0];
                cost += BitCost(0, lastP0);
            }
        }
        return cost;
    }

    private static int GetCostLuma4(Iterator it, short[] levels, int off)
    {
        int x = it.I4 & 3, y = it.I4 >> 2;
        var res = new Residual();
        InitResidual(0, 3, ref res);
        int ctx = it.TopNz[x] + it.LeftNz[y];
        SetResidualCoeffs(levels, off, ref res);
        return GetResidualCost(it.Enc.Proba, ctx, ref res);
    }

    private static int GetCostLuma16(Iterator it, ModeScore rd)
    {
        var res = new Residual();
        int R = 0;
        NzToBytes(it);
        InitResidual(0, 1, ref res);
        SetResidualCoeffs(rd.YDcLevels, 0, ref res);
        R += GetResidualCost(it.Enc.Proba, it.TopNz[8] + it.LeftNz[8], ref res);
        InitResidual(1, 0, ref res);
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                int ctx = it.TopNz[x] + it.LeftNz[y];
                SetResidualCoeffs(rd.YAcLevels, (x + y * 4) * 16, ref res);
                R += GetResidualCost(it.Enc.Proba, ctx, ref res);
                it.TopNz[x] = it.LeftNz[y] = res.Last >= 0 ? 1 : 0;
            }
        return R;
    }

    private static int GetCostUV(Iterator it, ModeScore rd)
    {
        var res = new Residual();
        int R = 0;
        NzToBytes(it);
        InitResidual(0, 2, ref res);
        for (int ch = 0; ch <= 2; ch += 2)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                {
                    int ctx = it.TopNz[4 + ch + x] + it.LeftNz[4 + ch + y];
                    SetResidualCoeffs(rd.UVLevels, (ch * 2 + x + y * 2) * 16, ref res);
                    R += GetResidualCost(it.Enc.Proba, ctx, ref res);
                    it.TopNz[4 + ch + x] = it.LeftNz[4 + ch + y] = res.Last >= 0 ? 1 : 0;
                }
        return R;
    }

    // ======================================================================
    // quant_enc.c: reconstruction and mode decision
    // ======================================================================

    private static long Mult8b(int a, int b) => (a * b + 128) >> 8;

    private static void SetRDScore(int lambda, ModeScore rd) => rd.Score = (rd.R + rd.H) * lambda + RD_DISTO_MULT * (rd.D + rd.SD);

    private static bool IsFlat(short[] levels, int off, int numBlocks, int thresh)
    {
        int score = 0;
        while (numBlocks-- > 0)
        {
            for (int i = 1; i < 16; ++i)
            {
                score += levels[off + i] != 0 ? 1 : 0;
                if (score > thresh) return false;
            }
            off += 16;
        }
        return true;
    }

    private static bool IsFlatSource16(byte[] src, int o)
    {
        byte v = src[o];
        for (int i = 0; i < 16; ++i, o += BPS)
            for (int x = 0; x < 16; x++)
                if (src[o + x] != v) return false;
        return true;
    }

    private static uint ReconstructIntra16(Iterator it, ModeScore rd, byte[] yuvOut, int yuvOutOff, int mode)
    {
        var enc = it.Enc;
        int refO = Vp8EncDsp.I16ModeOffsets[mode];
        var dqm = enc.Dqm[enc.MbInfo[it.Mb].Segment];
        uint nz = 0;
        var tmp = new short[16 * 16];
        var dcTmp = new short[16];
        for (int n = 0; n < 16; n += 2)
            Vp8EncDsp.FTransform2(it.YuvIn, Vp8EncDsp.Y_OFF_ENC + Vp8EncDsp.Scan[n], it.YuvP, refO + Vp8EncDsp.Scan[n], tmp, n * 16);
        Vp8EncDsp.FTransformWHT(tmp, 0, dcTmp, 0);
        nz |= (uint)Vp8EncDsp.QuantizeBlock(dcTmp, 0, rd.YDcLevels, 0, dqm.Y2) << 24;
        for (int n = 0; n < 16; n += 2)
        {
            tmp[n * 16] = tmp[(n + 1) * 16] = 0;
            nz |= (uint)Vp8EncDsp.Quantize2Blocks(tmp, n * 16, rd.YAcLevels, n * 16, dqm.Y1) << n;
        }
        Vp8EncDsp.TransformWHT(dcTmp, 0, tmp, 0);
        for (int n = 0; n < 16; n += 2)
            Vp8EncDsp.ITransform(it.YuvP, refO + Vp8EncDsp.Scan[n], tmp, n * 16, yuvOut, yuvOutOff + Vp8EncDsp.Scan[n], true);
        return nz;
    }

    private static int ReconstructIntra4(Iterator it, short[] levels, int levelsOff, byte[] src, int srcO, byte[] yuvOut, int yuvOutOff, int mode)
    {
        var enc = it.Enc;
        int refO = Vp8EncDsp.I4ModeOffsets[mode];
        var dqm = enc.Dqm[enc.MbInfo[it.Mb].Segment];
        var tmp = new short[16];
        Vp8EncDsp.FTransform(src, srcO, it.YuvP, refO, tmp, 0);
        int nz = Vp8EncDsp.QuantizeBlock(tmp, 0, levels, levelsOff, dqm.Y1);
        Vp8EncDsp.ITransform(it.YuvP, refO, tmp, 0, yuvOut, yuvOutOff, false);
        return nz;
    }

    // DC error diffusion: C1 = 7, C2 = 8, DSHIFT = 4, DSCALE = 1
    private static int QuantizeSingle(short[] v, int o, Vp8Matrix mtx)
    {
        int V = v[o];
        bool sign = V < 0;
        if (sign) V = -V;
        if (V > (int)mtx.ZThresh[0])
        {
            int qV = Vp8EncDsp.QuantDiv((uint)V, mtx.IQ[0], mtx.Bias[0]) * mtx.Q[0];
            int err = V - qV;
            v[o] = (short)(sign ? -qV : qV);
            return (sign ? -err : err) >> 1;
        }
        v[o] = 0;
        return (sign ? -V : V) >> 1;
    }

    private static void CorrectDCValues(Iterator it, Vp8Matrix mtx, short[] tmp, ModeScore rd)
    {
        var topDerr = it.Enc.TopDerr!;
        for (int ch = 0; ch <= 1; ++ch)
        {
            int topO = it.X * 4 + ch * 2;
            int leftO = ch * 2;
            int c = ch * 4 * 16;   // &tmp[ch * 4]
            tmp[c + 0 * 16] = (short)(tmp[c + 0 * 16] + ((7 * topDerr[topO + 0] + 8 * it.LeftDerr[leftO + 0]) >> 3));
            int err0 = QuantizeSingle(tmp, c + 0 * 16, mtx);
            tmp[c + 1 * 16] = (short)(tmp[c + 1 * 16] + ((7 * topDerr[topO + 1] + 8 * err0) >> 3));
            int err1 = QuantizeSingle(tmp, c + 1 * 16, mtx);
            tmp[c + 2 * 16] = (short)(tmp[c + 2 * 16] + ((7 * err0 + 8 * it.LeftDerr[leftO + 1]) >> 3));
            int err2 = QuantizeSingle(tmp, c + 2 * 16, mtx);
            tmp[c + 3 * 16] = (short)(tmp[c + 3 * 16] + ((7 * err1 + 8 * err2) >> 3));
            int err3 = QuantizeSingle(tmp, c + 3 * 16, mtx);
            rd.Derr[ch * 3 + 0] = (sbyte)err1;
            rd.Derr[ch * 3 + 1] = (sbyte)err2;
            rd.Derr[ch * 3 + 2] = (sbyte)err3;
        }
    }

    private static void StoreDiffusionErrors(Iterator it, ModeScore rd)
    {
        var topDerr = it.Enc.TopDerr!;
        for (int ch = 0; ch <= 1; ++ch)
        {
            int topO = it.X * 4 + ch * 2;
            int leftO = ch * 2;
            it.LeftDerr[leftO + 0] = rd.Derr[ch * 3 + 0];
            it.LeftDerr[leftO + 1] = (sbyte)(3 * rd.Derr[ch * 3 + 2] >> 2);
            topDerr[topO + 0] = rd.Derr[ch * 3 + 1];
            topDerr[topO + 1] = (sbyte)(rd.Derr[ch * 3 + 2] - it.LeftDerr[leftO + 1]);
        }
    }

    private static uint ReconstructUV(Iterator it, ModeScore rd, byte[] yuvOut, int yuvOutOff, int mode)
    {
        var enc = it.Enc;
        int refO = Vp8EncDsp.UVModeOffsets[mode];
        var dqm = enc.Dqm[enc.MbInfo[it.Mb].Segment];
        uint nz = 0;
        var tmp = new short[8 * 16];
        for (int n = 0; n < 8; n += 2)
            Vp8EncDsp.FTransform2(it.YuvIn, Vp8EncDsp.U_OFF_ENC + Vp8EncDsp.ScanUV[n], it.YuvP, refO + Vp8EncDsp.ScanUV[n], tmp, n * 16);
        if (it.HasTopDerr) CorrectDCValues(it, dqm.UV, tmp, rd);
        for (int n = 0; n < 8; n += 2)
            nz |= (uint)Vp8EncDsp.Quantize2Blocks(tmp, n * 16, rd.UVLevels, n * 16, dqm.UV) << n;
        for (int n = 0; n < 8; n += 2)
            Vp8EncDsp.ITransform(it.YuvP, refO + Vp8EncDsp.ScanUV[n], tmp, n * 16, yuvOut, yuvOutOff + Vp8EncDsp.ScanUV[n], true);
        return nz << 16;
    }

    private static void StoreMaxDelta(SegmentInfo dqm, short[] DCs)
    {
        int v0 = Math.Abs(DCs[1]), v1 = Math.Abs(DCs[2]), v2 = Math.Abs(DCs[4]);
        int maxV = v1 > v0 ? v1 : v0;
        maxV = v2 > maxV ? v2 : maxV;
        if (maxV > dqm.MaxEdge) dqm.MaxEdge = maxV;
    }

    private static void PickBestIntra16(Iterator it, ModeScore rd)
    {
        const int kNumBlocks = 16;
        var dqm = it.Enc.Dqm[it.Enc.MbInfo[it.Mb].Segment];
        int lambda = dqm.LambdaI16;
        int tlambda = dqm.TLambda;
        var rdTmp = new ModeScore();
        ModeScore rdCur = rdTmp, rdBest = rd;
        bool isFlat = IsFlatSource16(it.YuvIn, Vp8EncDsp.Y_OFF_ENC);

        rd.ModeI16 = -1;
        for (int mode = 0; mode < 4; ++mode)
        {
            var tmpDst = it.YuvOut2;   // scratch buffer (Y_OFF_ENC = 0)
            rdCur.ModeI16 = mode;
            rdCur.Nz = ReconstructIntra16(it, rdCur, tmpDst, Vp8EncDsp.Y_OFF_ENC, mode);
            rdCur.D = Vp8EncDsp.SSE(it.YuvIn, Vp8EncDsp.Y_OFF_ENC, tmpDst, Vp8EncDsp.Y_OFF_ENC, 16, 16);
            rdCur.SD = tlambda != 0 ? Mult8b(tlambda, Vp8EncDsp.Disto16x16(it.YuvIn, Vp8EncDsp.Y_OFF_ENC, tmpDst, Vp8EncDsp.Y_OFF_ENC, kWeightY)) : 0;
            rdCur.H = Vp8EncTables.VP8FixedCostsI16[mode];
            rdCur.R = GetCostLuma16(it, rdCur);
            if (isFlat)
            {
                isFlat = IsFlat(rdCur.YAcLevels, 0, kNumBlocks, FLATNESS_LIMIT_I16);
                if (isFlat) { rdCur.D *= 2; rdCur.SD *= 2; }
            }
            SetRDScore(lambda, rdCur);
            if (mode == 0 || rdCur.Score < rdBest.Score)
            {
                (rdCur, rdBest) = (rdBest, rdCur);
                SwapOut(it);
            }
        }
        if (rdBest != rd) rd.CopyAll(rdBest);
        SetRDScore(dqm.LambdaMode, rd);
        SetIntra16Mode(it, rd.ModeI16);
        if ((rd.Nz & 0x100ffff) == 0x1000000 && rd.D > dqm.MinDisto) StoreMaxDelta(dqm, rd.YDcLevels);
    }

    private static int GetCostModeI4Base(Iterator it, byte[] modes)
    {
        int predsW = it.Enc.PredsW;
        int x = it.I4 & 3, y = it.I4 >> 2;
        int left = x == 0 ? it.Enc.Preds[it.Preds + y * predsW - 1] : modes[it.I4 - 1];
        int top = y == 0 ? it.Enc.Preds[it.Preds - predsW + x] : modes[it.I4 - 4];
        return (top * 10 + left) * 10;   // VP8FixedCostsI4[top][left]
    }

    private static bool PickBestIntra4(Iterator it, ModeScore rd)
    {
        var enc = it.Enc;
        var dqm = enc.Dqm[enc.MbInfo[it.Mb].Segment];
        int lambda = dqm.LambdaI4;
        int tlambda = dqm.TLambda;
        // best_blocks = yuv_out2 + Y_OFF_ENC
        var bestBlocks = it.YuvOut2;
        int totalHeaderBits = 0;
        var rdBest = new ModeScore();
        if (enc.MaxI4HeaderBits == 0) return false;

        rdBest.InitScore();
        rdBest.H = 211;
        SetRDScore(dqm.LambdaMode, rdBest);
        StartI4(it);
        var rdI4 = new ModeScore();
        var rdTmp = new ModeScore();
        var tmpLevels = new short[16];
        do
        {
            const int kNumBlocks = 1;
            int bestMode = -1;
            int srcO = Vp8EncDsp.Y_OFF_ENC + Vp8EncDsp.Scan[it.I4];
            int modeCosts = GetCostModeI4Base(it, rd.ModesI4);
            // best_block / tmp_dst are pointers that get swapped: (array, offset)
            byte[] bestBlockA = bestBlocks; int bestBlockO = Vp8EncDsp.Y_OFF_ENC + Vp8EncDsp.Scan[it.I4];
            byte[] tmpDstA = it.YuvP; int tmpDstO = Vp8EncDsp.I4TMP;

            rdI4.InitScore();
            Vp8EncDsp.Intra4Preds(it.YuvP, it.I4Boundary, it.I4Top);
            for (int mode = 0; mode < 10; ++mode)
            {
                rdTmp.Nz = (uint)ReconstructIntra4(it, tmpLevels, 0, it.YuvIn, srcO, tmpDstA, tmpDstO, mode) << it.I4;
                rdTmp.D = Vp8EncDsp.SSE(it.YuvIn, srcO, tmpDstA, tmpDstO, 4, 4);
                rdTmp.SD = tlambda != 0 ? Mult8b(tlambda, Vp8EncDsp.Disto4x4(it.YuvIn, srcO, tmpDstA, tmpDstO, kWeightY)) : 0;
                rdTmp.H = Vp8EncTables.VP8FixedCostsI4[modeCosts + mode];
                if (mode > 0 && IsFlat(tmpLevels, 0, kNumBlocks, FLATNESS_LIMIT_I4)) rdTmp.R = FLATNESS_PENALTY * kNumBlocks;
                else rdTmp.R = 0;

                SetRDScore(lambda, rdTmp);
                if (bestMode >= 0 && rdTmp.Score >= rdI4.Score) continue;

                rdTmp.R += GetCostLuma4(it, tmpLevels, 0);
                SetRDScore(lambda, rdTmp);
                if (bestMode < 0 || rdTmp.Score < rdI4.Score)
                {
                    rdI4.CopyScore(rdTmp);
                    bestMode = mode;
                    (tmpDstA, bestBlockA) = (bestBlockA, tmpDstA);
                    (tmpDstO, bestBlockO) = (bestBlockO, tmpDstO);
                    System.Array.Copy(tmpLevels, 0, rdBest.YAcLevels, it.I4 * 16, 16);
                }
            }
            SetRDScore(dqm.LambdaMode, rdI4);
            rdBest.AddScore(rdI4);
            if (rdBest.Score >= rd.Score) return false;
            totalHeaderBits += (int)rdI4.H;
            if (totalHeaderBits > enc.MaxI4HeaderBits) return false;
            int home = Vp8EncDsp.Y_OFF_ENC + Vp8EncDsp.Scan[it.I4];
            if (bestBlockA != bestBlocks || bestBlockO != home)
                Vp8EncDsp.Copy(bestBlockA, bestBlockO, bestBlocks, home, 4, 4);
            rd.ModesI4[it.I4] = (byte)bestMode;
            it.TopNz[it.I4 & 3] = it.LeftNz[it.I4 >> 2] = rdI4.Nz != 0 ? 1 : 0;
        } while (RotateI4(it, bestBlocks, Vp8EncDsp.Y_OFF_ENC));

        rd.CopyScore(rdBest);
        SetIntra4Mode(it, rd.ModesI4);
        SwapOut(it);
        System.Array.Copy(rdBest.YAcLevels, rd.YAcLevels, rd.YAcLevels.Length);
        return true;
    }

    private static void PickBestUV(Iterator it, ModeScore rd)
    {
        const int kNumBlocks = 8;
        var dqm = it.Enc.Dqm[it.Enc.MbInfo[it.Mb].Segment];
        int lambda = dqm.LambdaUV;
        byte[] tmpDst = it.YuvOut2;   // + U_OFF_ENC
        byte[] dst0 = it.YuvOut;
        byte[] dst = dst0;
        var rdBest = new ModeScore();
        var rdUv = new ModeScore();
        rd.ModeUV = -1;
        rdBest.InitScore();
        for (int mode = 0; mode < 4; ++mode)
        {
            rdUv.Nz = ReconstructUV(it, rdUv, tmpDst, Vp8EncDsp.U_OFF_ENC, mode);
            rdUv.D = Vp8EncDsp.SSE(it.YuvIn, Vp8EncDsp.U_OFF_ENC, tmpDst, Vp8EncDsp.U_OFF_ENC, 16, 8);
            rdUv.SD = 0;
            rdUv.H = Vp8EncTables.VP8FixedCostsUV[mode];
            rdUv.R = GetCostUV(it, rdUv);
            if (mode > 0 && IsFlat(rdUv.UVLevels, 0, kNumBlocks, FLATNESS_LIMIT_UV)) rdUv.R += FLATNESS_PENALTY * kNumBlocks;
            SetRDScore(lambda, rdUv);
            if (mode == 0 || rdUv.Score < rdBest.Score)
            {
                rdBest.CopyScore(rdUv);
                rd.ModeUV = mode;
                System.Array.Copy(rdUv.UVLevels, rd.UVLevels, rd.UVLevels.Length);
                if (it.HasTopDerr) System.Array.Copy(rdUv.Derr, rd.Derr, rd.Derr.Length);
                (dst, tmpDst) = (tmpDst, dst);
            }
        }
        it.Enc.MbInfo[it.Mb].UVMode = rd.ModeUV;
        rd.AddScore(rdBest);
        if (dst != dst0) Vp8EncDsp.Copy(dst, Vp8EncDsp.U_OFF_ENC, dst0, Vp8EncDsp.U_OFF_ENC, 16, 8);
        if (it.HasTopDerr) StoreDiffusionErrors(it, rd);
    }

    private static void RefineUsingDistortion(Iterator it, bool tryBothModes, bool refineUvMode, ModeScore rd)
    {
        long bestScore = MAX_COST;
        uint nz = 0;
        var enc = it.Enc;
        bool isI16 = tryBothModes || enc.MbInfo[it.Mb].Type == 1;
        var dqm = enc.Dqm[enc.MbInfo[it.Mb].Segment];
        const int lambdaDI16 = 106, lambdaDI4 = 11, lambdaDUV = 120;
        long scoreI4 = dqm.I4Penalty;
        long i4BitSum = 0;
        long bitLimit = tryBothModes ? enc.MbHeaderLimit : MAX_COST;

        if (isI16)
        {
            int bestMode = -1;
            for (int mode = 0; mode < 4; ++mode)
            {
                int refO = Vp8EncDsp.I16ModeOffsets[mode];
                long score = (long)Vp8EncDsp.SSE(it.YuvIn, Vp8EncDsp.Y_OFF_ENC, it.YuvP, refO, 16, 16) * RD_DISTO_MULT
                           + Vp8EncTables.VP8FixedCostsI16[mode] * lambdaDI16;
                if (mode > 0 && Vp8EncTables.VP8FixedCostsI16[mode] > bitLimit) continue;
                if (score < bestScore) { bestMode = mode; bestScore = score; }
            }
            if (it.X == 0 || it.Y == 0)
            {
                if (IsFlatSource16(it.YuvIn, Vp8EncDsp.Y_OFF_ENC))
                {
                    bestMode = it.X == 0 ? 0 : 2;
                    tryBothModes = false;
                }
            }
            SetIntra16Mode(it, bestMode);
        }

        if (tryBothModes || !isI16)
        {
            isI16 = false;
            StartI4(it);
            do
            {
                int bestI4Mode = -1;
                long bestI4Score = MAX_COST;
                int srcO = Vp8EncDsp.Y_OFF_ENC + Vp8EncDsp.Scan[it.I4];
                int modeCosts = GetCostModeI4Base(it, rd.ModesI4);
                Vp8EncDsp.Intra4Preds(it.YuvP, it.I4Boundary, it.I4Top);
                for (int mode = 0; mode < 10; ++mode)
                {
                    int refO = Vp8EncDsp.I4ModeOffsets[mode];
                    long score = (long)Vp8EncDsp.SSE(it.YuvIn, srcO, it.YuvP, refO, 4, 4) * RD_DISTO_MULT
                               + Vp8EncTables.VP8FixedCostsI4[modeCosts + mode] * lambdaDI4;
                    if (score < bestI4Score) { bestI4Mode = mode; bestI4Score = score; }
                }
                i4BitSum += Vp8EncTables.VP8FixedCostsI4[modeCosts + bestI4Mode];
                rd.ModesI4[it.I4] = (byte)bestI4Mode;
                scoreI4 += bestI4Score;
                if (scoreI4 >= bestScore || i4BitSum > bitLimit) { isI16 = true; break; }
                nz |= (uint)ReconstructIntra4(it, rd.YAcLevels, it.I4 * 16, it.YuvIn, srcO, it.YuvOut2,
                    Vp8EncDsp.Y_OFF_ENC + Vp8EncDsp.Scan[it.I4], bestI4Mode) << it.I4;
            } while (RotateI4(it, it.YuvOut2, Vp8EncDsp.Y_OFF_ENC));
        }

        if (!isI16)
        {
            SetIntra4Mode(it, rd.ModesI4);
            SwapOut(it);
            bestScore = scoreI4;
        }
        else
        {
            nz = ReconstructIntra16(it, rd, it.YuvOut, Vp8EncDsp.Y_OFF_ENC, enc.Preds[it.Preds]);
        }

        if (refineUvMode)
        {
            int bestMode = -1;
            long bestUvScore = MAX_COST;
            for (int mode = 0; mode < 4; ++mode)
            {
                int refO = Vp8EncDsp.UVModeOffsets[mode];
                long score = (long)Vp8EncDsp.SSE(it.YuvIn, Vp8EncDsp.U_OFF_ENC, it.YuvP, refO, 16, 8) * RD_DISTO_MULT
                           + Vp8EncTables.VP8FixedCostsUV[mode] * lambdaDUV;
                if (score < bestUvScore) { bestMode = mode; bestUvScore = score; }
            }
            enc.MbInfo[it.Mb].UVMode = bestMode;
        }
        nz |= ReconstructUV(it, rd, it.YuvOut, Vp8EncDsp.U_OFF_ENC, enc.MbInfo[it.Mb].UVMode);
        rd.Nz = nz;
        rd.Score = bestScore;
    }

    private static bool VP8Decimate(Iterator it, ModeScore rd, int rdOpt)
    {
        int method = it.Enc.Method;
        rd.InitScore();
        MakeLuma16Preds(it);
        MakeChroma8Preds(it);
        if (rdOpt > 0)
        {
            it.DoTrellis = rdOpt >= 3;
            PickBestIntra16(it, rd);
            if (method >= 2) PickBestIntra4(it, rd);
            PickBestUV(it, rd);
            if (rdOpt == 2) throw new NotSupportedException("trellis");
        }
        else
        {
            RefineUsingDistortion(it, method >= 2, method >= 1, rd);
        }
        bool isSkipped = rd.Nz == 0;
        it.Enc.MbInfo[it.Mb].Skip = isSkipped ? 1 : 0;
        return isSkipped;
    }

    // ======================================================================
    // frame_enc.c
    // ======================================================================

    private static int CalcSkipProba(long nb, long total) => (int)(total != 0 ? (total - nb) * 255 / total : 255);

    private static int FinalizeSkipProba(Encoder enc)
    {
        var proba = enc.Proba;
        int nbMbs = enc.MbW * enc.MbH;
        int nbEvents = proba.NbSkip;
        proba.SkipProba = CalcSkipProba(nbEvents, nbMbs);
        proba.UseSkipProba = proba.SkipProba < SKIP_PROBA_THRESHOLD;
        int size = 256;
        if (proba.UseSkipProba)
        {
            size += nbEvents * BitCost(1, proba.SkipProba) + (nbMbs - nbEvents) * BitCost(0, proba.SkipProba);
            size += 8 * 256;
        }
        return size;
    }

    private static int CalcTokenProba(int nb, int total) => nb != 0 ? (255 - nb * 255 / total) : 255;

    private static int BranchCost(int nb, int total, int proba) => nb * BitCost(1, proba) + (total - nb) * BitCost(0, proba);

    private static int FinalizeTokenProbas(Proba proba)
    {
        bool hasChanged = false;
        int size = 0;
        for (int i = 0; i < proba.Coeffs.Length; i++)
        {
            uint stats = proba.Stats[i];
            int nb = (int)(stats & 0xffff);
            int total = (int)((stats >> 16) & 0xffff);
            int updateProba = Vp8EncTables.VP8CoeffsUpdateProba[i];
            int oldP = Vp8EncTables.VP8CoeffsProba0[i];
            int newP = CalcTokenProba(nb, total);
            int oldCost = BranchCost(nb, total, oldP) + BitCost(0, updateProba);
            int newCost = BranchCost(nb, total, newP) + BitCost(1, updateProba) + 8 * 256;
            bool useNewP = oldCost > newCost;
            size += BitCost(useNewP ? 1 : 0, updateProba);
            if (useNewP)
            {
                proba.Coeffs[i] = (byte)newP;
                hasChanged |= newP != oldP;
                size += 8 * 256;
            }
            else proba.Coeffs[i] = (byte)oldP;
        }
        proba.Dirty = hasChanged;
        return size;
    }

    private static int GetProba(int a, int b)
    {
        int total = a + b;
        return total == 0 ? 255 : (255 * a + total / 2) / total;
    }

    private static void SetSegmentProbas(Encoder enc)
    {
        var p = new int[NUM_MB_SEGMENTS];
        for (int n = 0; n < enc.MbW * enc.MbH; ++n) ++p[enc.MbInfo[n].Segment];
        if (enc.NumSegments > 1)
        {
            var probas = enc.Proba.Segments;
            probas[0] = (byte)GetProba(p[0] + p[1], p[2] + p[3]);
            probas[1] = (byte)GetProba(p[0], p[1]);
            probas[2] = (byte)GetProba(p[2], p[3]);
            enc.UpdateMap = probas[0] != 255 || probas[1] != 255 || probas[2] != 255;
            if (!enc.UpdateMap)
                for (int n = 0; n < enc.MbW * enc.MbH; ++n) enc.MbInfo[n].Segment = 0;
            enc.SegmentHdrSize =
                p[0] * (BitCost(0, probas[0]) + BitCost(0, probas[1])) +
                p[1] * (BitCost(0, probas[0]) + BitCost(1, probas[1])) +
                p[2] * (BitCost(1, probas[0]) + BitCost(0, probas[2])) +
                p[3] * (BitCost(1, probas[0]) + BitCost(1, probas[2]));
        }
        else
        {
            enc.UpdateMap = false;
            enc.SegmentHdrSize = 0;
        }
    }

    // ---- coefficient coding (PutCoeffs) and statistics (VP8RecordCoeffs) ----

    private static int PutCoeffs(Encoder enc, Vp8EncBitWriter bw, int ctx, ref Residual res)
    {
        var coeffs = enc.Proba.Coeffs;
        int t = res.CoeffType;
        int n = res.First;
        int p = PIdx(t, n, ctx);
        if (bw.PutBit(res.Last >= 0, coeffs[p + 0]) == 0) return 0;
        while (n < 16)
        {
            int c = res.Coeffs[res.CoeffsOff + n++];
            bool sign = c < 0;
            int v = sign ? -c : c;
            if (bw.PutBit(v != 0, coeffs[p + 1]) == 0)
            {
                p = PIdx(t, Vp8EncTables.VP8EncBands[n], 0);
                continue;
            }
            if (bw.PutBit(v > 1, coeffs[p + 2]) == 0)
            {
                p = PIdx(t, Vp8EncTables.VP8EncBands[n], 1);
            }
            else
            {
                if (bw.PutBit(v > 4, coeffs[p + 3]) == 0)
                {
                    if (bw.PutBit(v != 2, coeffs[p + 4]) != 0) bw.PutBit(v == 4, coeffs[p + 5]);
                }
                else if (bw.PutBit(v > 10, coeffs[p + 6]) == 0)
                {
                    if (bw.PutBit(v > 6, coeffs[p + 7]) == 0) bw.PutBit(v == 6, 159);
                    else { bw.PutBit(v >= 9, 165); bw.PutBit((v & 1) == 0, 145); }
                }
                else
                {
                    int mask;
                    byte[] tab;
                    if (v < 3 + (8 << 1)) { bw.PutBit(0, coeffs[p + 8]); bw.PutBit(0, coeffs[p + 9]); v -= 3 + (8 << 0); mask = 1 << 2; tab = Vp8EncTables.VP8Cat3; }
                    else if (v < 3 + (8 << 2)) { bw.PutBit(0, coeffs[p + 8]); bw.PutBit(1, coeffs[p + 9]); v -= 3 + (8 << 1); mask = 1 << 3; tab = Vp8EncTables.VP8Cat4; }
                    else if (v < 3 + (8 << 3)) { bw.PutBit(1, coeffs[p + 8]); bw.PutBit(0, coeffs[p + 10]); v -= 3 + (8 << 2); mask = 1 << 4; tab = Vp8EncTables.VP8Cat5; }
                    else { bw.PutBit(1, coeffs[p + 8]); bw.PutBit(1, coeffs[p + 10]); v -= 3 + (8 << 3); mask = 1 << 10; tab = Vp8EncTables.VP8Cat6; }
                    int ti = 0;
                    while (mask != 0) { bw.PutBit((v & mask) != 0, tab[ti++]); mask >>= 1; }
                }
                p = PIdx(t, Vp8EncTables.VP8EncBands[n], 2);
            }
            bw.PutBitUniform(sign);
            if (n == 16 || bw.PutBit(n <= res.Last, coeffs[p + 0]) == 0) return 1;
        }
        return 1;
    }

    private static int RecordStats(int bit, uint[] stats, int i)
    {
        uint p = stats[i];
        if (p >= 0xfffe0000u) p = ((p + 1u) >> 1) & 0x7fff7fffu;
        p += 0x00010000u + (uint)bit;
        stats[i] = p;
        return bit;
    }

    private static int RecordCoeffs(Encoder enc, int ctx, ref Residual res)
    {
        var stats = enc.Proba.Stats;
        int t = res.CoeffType;
        int n = res.First;
        int s = PIdx(t, n, ctx);
        if (res.Last < 0) { RecordStats(0, stats, s + 0); return 0; }
        while (n <= res.Last)
        {
            int v;
            RecordStats(1, stats, s + 0);
            while ((v = res.Coeffs[res.CoeffsOff + n++]) == 0)
            {
                RecordStats(0, stats, s + 1);
                s = PIdx(t, Vp8EncTables.VP8EncBands[n], 0);
            }
            RecordStats(1, stats, s + 1);
            if (RecordStats(2u < (uint)(v + 1) ? 1 : 0, stats, s + 2) == 0)
            {
                s = PIdx(t, Vp8EncTables.VP8EncBands[n], 1);
            }
            else
            {
                v = Math.Abs(v);
                if (v > MAX_VARIABLE_LEVEL) v = MAX_VARIABLE_LEVEL;
                int bits = Vp8EncTables.VP8LevelCodes[(v - 1) * 2 + 1];
                int pattern = Vp8EncTables.VP8LevelCodes[(v - 1) * 2 + 0];
                for (int i = 0; (pattern >>= 1) != 0; ++i)
                {
                    int mask = 2 << i;
                    if ((pattern & 1) != 0) RecordStats((bits & mask) != 0 ? 1 : 0, stats, s + 3 + i);
                }
                s = PIdx(t, Vp8EncTables.VP8EncBands[n], 2);
            }
        }
        if (n < 16) RecordStats(0, stats, s + 0);
        return 1;
    }

    // ---- token buffer (token_enc.c) ----

    private static int AddToken(Encoder enc, int bit, int probaIdx, int statIdx)
    {
        enc.Tokens.Add((ushort)((bit << 15) | probaIdx));
        RecordStats(bit, enc.Proba.Stats, statIdx);
        return bit;
    }

    private static void AddConstantToken(Encoder enc, int bit, int proba)
        => enc.Tokens.Add((ushort)((bit << 15) | FIXED_PROBA_BIT | proba));

    private static int RecordCoeffTokens(Encoder enc, int ctx, ref Residual res)
    {
        int t = res.CoeffType;
        int last = res.Last;
        int n = res.First;
        int baseId = PIdx(t, n, ctx);
        int s = baseId;   // stats share the coeffs layout
        if (AddToken(enc, last >= 0 ? 1 : 0, baseId + 0, s + 0) == 0) return 0;
        while (n < 16)
        {
            int c = res.Coeffs[res.CoeffsOff + n++];
            bool sign = c < 0;
            uint v = (uint)(sign ? -c : c);
            if (AddToken(enc, v != 0 ? 1 : 0, baseId + 1, s + 1) == 0)
            {
                baseId = PIdx(t, Vp8EncTables.VP8EncBands[n], 0);
                s = baseId;
                continue;
            }
            if (AddToken(enc, v > 1 ? 1 : 0, baseId + 2, s + 2) == 0)
            {
                baseId = PIdx(t, Vp8EncTables.VP8EncBands[n], 1);
                s = baseId;
            }
            else
            {
                if (AddToken(enc, v > 4 ? 1 : 0, baseId + 3, s + 3) == 0)
                {
                    if (AddToken(enc, v != 2 ? 1 : 0, baseId + 4, s + 4) != 0) AddToken(enc, v == 4 ? 1 : 0, baseId + 5, s + 5);
                }
                else if (AddToken(enc, v > 10 ? 1 : 0, baseId + 6, s + 6) == 0)
                {
                    if (AddToken(enc, v > 6 ? 1 : 0, baseId + 7, s + 7) == 0) AddConstantToken(enc, v == 6 ? 1 : 0, 159);
                    else { AddConstantToken(enc, v >= 9 ? 1 : 0, 165); AddConstantToken(enc, (v & 1) == 0 ? 1 : 0, 145); }
                }
                else
                {
                    int mask;
                    byte[] tab;
                    uint residue = v - 3;
                    if (residue < (8 << 1)) { AddToken(enc, 0, baseId + 8, s + 8); AddToken(enc, 0, baseId + 9, s + 9); residue -= 8 << 0; mask = 1 << 2; tab = Vp8EncTables.VP8Cat3; }
                    else if (residue < (8 << 2)) { AddToken(enc, 0, baseId + 8, s + 8); AddToken(enc, 1, baseId + 9, s + 9); residue -= 8 << 1; mask = 1 << 3; tab = Vp8EncTables.VP8Cat4; }
                    // libwebp records the statistic of p[10] in slot 9 (s + 9) for Cat5 / Cat6: kept as is.
                    else if (residue < (8 << 3)) { AddToken(enc, 1, baseId + 8, s + 8); AddToken(enc, 0, baseId + 10, s + 9); residue -= 8 << 2; mask = 1 << 4; tab = Vp8EncTables.VP8Cat5; }
                    else { AddToken(enc, 1, baseId + 8, s + 8); AddToken(enc, 1, baseId + 10, s + 9); residue -= 8 << 3; mask = 1 << 10; tab = Vp8EncTables.VP8Cat6; }
                    int ti = 0;
                    while (mask != 0) { AddConstantToken(enc, (residue & mask) != 0 ? 1 : 0, tab[ti++]); mask >>= 1; }
                }
                baseId = PIdx(t, Vp8EncTables.VP8EncBands[n], 2);
                s = baseId;
            }
            AddConstantToken(enc, sign ? 1 : 0, 128);
            if (n == 16 || AddToken(enc, n <= last ? 1 : 0, baseId + 0, s + 0) == 0) return 1;
        }
        return 1;
    }

    private static void EmitTokens(Encoder enc, Vp8EncBitWriter bw)
    {
        var probas = enc.Proba.Coeffs;
        foreach (var token in enc.Tokens)
        {
            int bit = (token >> 15) & 1;
            if ((token & FIXED_PROBA_BIT) != 0) bw.PutBit(bit, token & 0xff);
            else bw.PutBit(bit, probas[token & 0x3fff]);
        }
    }

    private delegate int CoeffSink(Encoder enc, int ctx, ref Residual res);

    /// <summary>The traversal shared by CodeResiduals / RecordResiduals / RecordTokens (only the sink differs).</summary>
    private static void VisitResiduals(Iterator it, ModeScore rd, CoeffSink sink)
    {
        var enc = it.Enc;
        var res = new Residual();
        NzToBytes(it);
        if (enc.MbInfo[it.Mb].Type == 1)
        {
            int ctx = it.TopNz[8] + it.LeftNz[8];
            InitResidual(0, 1, ref res);
            SetResidualCoeffs(rd.YDcLevels, 0, ref res);
            it.TopNz[8] = it.LeftNz[8] = sink(enc, ctx, ref res);
            InitResidual(1, 0, ref res);
        }
        else InitResidual(0, 3, ref res);

        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
            {
                int ctx = it.TopNz[x] + it.LeftNz[y];
                SetResidualCoeffs(rd.YAcLevels, (x + y * 4) * 16, ref res);
                it.TopNz[x] = it.LeftNz[y] = sink(enc, ctx, ref res);
            }

        InitResidual(0, 2, ref res);
        for (int ch = 0; ch <= 2; ch += 2)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                {
                    int ctx = it.TopNz[4 + ch + x] + it.LeftNz[4 + ch + y];
                    SetResidualCoeffs(rd.UVLevels, (ch * 2 + x + y * 2) * 16, ref res);
                    it.TopNz[4 + ch + x] = it.LeftNz[4 + ch + y] = sink(enc, ctx, ref res);
                }
        BytesToNz(it);
    }

    private static void ResetAfterSkip(Iterator it)
    {
        if (it.Enc.MbInfo[it.Mb].Type == 1)
        {
            it.Enc.Nz[it.NzIdx] = 0;
            it.LeftNz[8] = 0;
        }
        else
        {
            it.Enc.Nz[it.NzIdx] &= 1u << 24;
        }
    }

    private static void SetLoopParams(Encoder enc, float q)
    {
        q = q < 0f ? 0f : q > 100f ? 100f : q;
        VP8SetSegmentParams(enc, q);
        SetSegmentProbas(enc);
        // ResetStats
        CalculateLevelCosts(enc.Proba);
        enc.Proba.NbSkip = 0;
    }

    private static void VP8EncTokenLoop(Encoder enc)
    {
        int maxCount = (enc.MbW * enc.MbH) >> 3;
        int numPassLeft = enc.Config.Pass;
        var proba = enc.Proba;
        int rdOpt = enc.RdOptLevel;
        float q = Math.Clamp(enc.Config.Quality, 0f, 100f);   // InitPassStats (qmin 0, qmax 100)
        enc.Parts[0] = new Vp8EncBitWriter();
        if (maxCount < MIN_COUNT) maxCount = MIN_COUNT;

        Iterator it = null!;
        while (numPassLeft-- > 0)
        {
            bool isLastPass = numPassLeft == 0 || enc.MaxI4HeaderBits == 0;   // fabs(dq) <= DQ_LIMIT is false: dq starts at 10
            ulong sizeP0 = 0;
            int cnt = maxCount;
            it = IteratorInit(enc);
            SetLoopParams(enc, q);
            if (isLastPass) System.Array.Clear(proba.Stats);   // ResetTokenStats (+ VP8InitFilter: no autofilter)
            enc.Tokens.Clear();
            var info = new ModeScore();
            do
            {
                IteratorImport(it, withBoundary: false);
                if (--cnt < 0)
                {
                    FinalizeTokenProbas(proba);
                    CalculateLevelCosts(proba);
                    cnt = maxCount;
                }
                VP8Decimate(it, info, rdOpt);
                VisitResiduals(it, info, RecordCoeffTokens);
                sizeP0 += (ulong)info.H;
                SaveBoundary(it);
            } while (IteratorNext(it));

            sizeP0 += (ulong)enc.SegmentHdrSize;
            if (enc.MaxI4HeaderBits > 0 && sizeP0 > PARTITION0_SIZE_LIMIT)
            {
                ++numPassLeft;
                enc.MaxI4HeaderBits >>= 1;
                continue;
            }
            if (isLastPass) break;
        }
        FinalizeTokenProbas(enc.Proba);
        EmitTokens(enc, enc.Parts[0]);
        PostLoopFinalize(enc);
    }

    private static void VP8EncLoop(Encoder enc)
    {
        for (int p = 0; p < enc.NumParts; ++p) enc.Parts[p] = new Vp8EncBitWriter();
        StatLoop(enc);
        var it = IteratorInit(enc);
        var info = new ModeScore();
        do
        {
            bool dontUseSkip = !enc.Proba.UseSkipProba;
            IteratorImport(it, withBoundary: false);
            if (!VP8Decimate(it, info, enc.RdOptLevel) || dontUseSkip)
                VisitResiduals(it, info, (Encoder e, int ctx, ref Residual r) => PutCoeffs(e, it.Bw, ctx, ref r));
            else
                ResetAfterSkip(it);
            SaveBoundary(it);
        } while (IteratorNext(it));
        PostLoopFinalize(enc);
    }

    private static void StatLoop(Encoder enc)
    {
        int method = enc.Method;
        bool fastProbe = method == 0 || method == 3;
        int numPassLeft = enc.Config.Pass;
        int rdOpt = method >= 3 ? 1 : 0;
        int nbMbs = enc.MbW * enc.MbH;
        float q = Math.Clamp(enc.Config.Quality, 0f, 100f);
        System.Array.Clear(enc.Proba.Stats);   // ResetTokenStats
        if (fastProbe)
        {
            if (method == 3) nbMbs = nbMbs > 200 ? nbMbs >> 1 : 100;
            else nbMbs = nbMbs > 200 ? nbMbs >> 2 : 50;
        }
        while (numPassLeft-- > 0)
        {
            bool isLastPass = numPassLeft == 0 || enc.MaxI4HeaderBits == 0;
            // OneStatPass
            ulong sizeP0 = 0;
            var it = IteratorInit(enc);
            SetLoopParams(enc, q);
            var info = new ModeScore();
            int left = nbMbs;
            do
            {
                IteratorImport(it, withBoundary: false);
                if (VP8Decimate(it, info, rdOpt)) ++enc.Proba.NbSkip;
                VisitResiduals(it, info, RecordCoeffs);
                sizeP0 += (ulong)info.H;
                SaveBoundary(it);
            } while (IteratorNext(it) && --left > 0);
            sizeP0 += (ulong)enc.SegmentHdrSize;
            // libwebp: `if (size_p0 == 0) return 0;` - and VP8EncLoop ignores the result. With RD_OPT_NONE (methods 0..2)
            // VP8Decimate never sets info.H, so a single-segment picture (segment map cost 0) returns HERE: no skip
            // probability, no token-probability update, default probabilities stay. Ported as is (byte identity).
            if (sizeP0 == 0) return;
            if (enc.MaxI4HeaderBits > 0 && sizeP0 > PARTITION0_SIZE_LIMIT)
            {
                ++numPassLeft;
                enc.MaxI4HeaderBits >>= 1;
                continue;
            }
            if (isLastPass) break;
        }
        FinalizeSkipProba(enc);
        FinalizeTokenProbas(enc.Proba);
        CalculateLevelCosts(enc.Proba);
    }

    private static void PostLoopFinalize(Encoder enc)
    {
        for (int p = 0; p < enc.NumParts; ++p) enc.Parts[p].Finish();
        // VP8AdjustFilterStrength (no autofilter)
        if (enc.Config.FilterStrength > 0)
        {
            int maxLevel = 0;
            for (int s = 0; s < NUM_MB_SEGMENTS; s++)
            {
                var dqm = enc.Dqm[s];
                int delta = (dqm.MaxEdge * dqm.Y2.Q[1]) >> 3;
                int level = FilterStrengthFromDelta(enc.FilterSharpness, delta);
                if (level > dqm.FStrength) dqm.FStrength = level;
                if (maxLevel < dqm.FStrength) maxLevel = dqm.FStrength;
            }
            enc.FilterLevel = maxLevel;
        }
    }

    // ======================================================================
    // tree_enc.c / syntax_enc.c
    // ======================================================================

    private static int PutI4Mode(Vp8EncBitWriter bw, int mode, int probIdx)
    {
        var prob = Vp8EncTables.kBModesProba;
        // B_DC 0, B_TM 1, B_VE 2, B_HE 3, B_RD 4, B_VR 5, B_LD 6, B_VL 7, B_HD 8, B_HU 9
        if (bw.PutBit(mode != 0, prob[probIdx + 0]) != 0)
            if (bw.PutBit(mode != 1, prob[probIdx + 1]) != 0)
                if (bw.PutBit(mode != 2, prob[probIdx + 2]) != 0)
                {
                    if (bw.PutBit(mode >= 6, prob[probIdx + 3]) == 0)
                    {
                        if (bw.PutBit(mode != 3, prob[probIdx + 4]) != 0) bw.PutBit(mode != 4, prob[probIdx + 5]);
                    }
                    else
                    {
                        if (bw.PutBit(mode != 6, prob[probIdx + 6]) != 0)
                            if (bw.PutBit(mode != 7, prob[probIdx + 7]) != 0) bw.PutBit(mode != 8, prob[probIdx + 8]);
                    }
                }
        return mode;
    }

    private static void PutI16Mode(Vp8EncBitWriter bw, int mode)
    {
        // DC_PRED 0, TM_PRED 1, V_PRED 2, H_PRED 3
        if (bw.PutBit(mode == 1 || mode == 3, 156) != 0) bw.PutBit(mode == 1, 128);
        else bw.PutBit(mode == 2, 163);
    }

    private static void PutUVMode(Vp8EncBitWriter bw, int uvMode)
    {
        if (bw.PutBit(uvMode != 0, 142) != 0)
            if (bw.PutBit(uvMode != 2, 114) != 0) bw.PutBit(uvMode != 3, 183);
    }

    private static void PutSegment(Vp8EncBitWriter bw, int s, byte[] p)
    {
        int o = 0;
        if (bw.PutBit(s >= 2, p[0]) != 0) o = 1;
        bw.PutBit(s & 1, p[o + 1]);
    }

    private static void CodeIntraModes(Encoder enc)
    {
        var bw = enc.Bw;
        var it = IteratorInit(enc);
        do
        {
            var mb = enc.MbInfo[it.Mb];
            int preds = it.Preds;
            if (enc.UpdateMap) PutSegment(bw, mb.Segment, enc.Proba.Segments);
            if (enc.Proba.UseSkipProba) bw.PutBit(mb.Skip, enc.Proba.SkipProba);
            if (bw.PutBit(mb.Type != 0, 145) != 0)
            {
                PutI16Mode(bw, enc.Preds[preds]);
            }
            else
            {
                int predsW = enc.PredsW;
                int topPred = preds - predsW;
                for (int y = 0; y < 4; ++y)
                {
                    int left = enc.Preds[preds - 1];
                    for (int x = 0; x < 4; ++x)
                    {
                        int probIdx = (enc.Preds[topPred + x] * 10 + left) * 9;
                        left = PutI4Mode(bw, enc.Preds[preds + x], probIdx);
                    }
                    topPred = preds;
                    preds += predsW;
                }
            }
            PutUVMode(bw, mb.UVMode);
        } while (IteratorNext(it));
    }

    private static void WriteProbas(Vp8EncBitWriter bw, Proba probas)
    {
        for (int i = 0; i < probas.Coeffs.Length; i++)
        {
            byte p0 = probas.Coeffs[i];
            bool update = p0 != Vp8EncTables.VP8CoeffsProba0[i];
            if (bw.PutBit(update, Vp8EncTables.VP8CoeffsUpdateProba[i]) != 0) bw.PutBits(p0, 8);
        }
        if (bw.PutBitUniform(probas.UseSkipProba) != 0) bw.PutBits((uint)probas.SkipProba, 8);
    }

    private static void PutSegmentHeader(Vp8EncBitWriter bw, Encoder enc)
    {
        if (bw.PutBitUniform(enc.NumSegments > 1) != 0)
        {
            bw.PutBitUniform(enc.UpdateMap);
            if (bw.PutBitUniform(1) != 0)
            {
                bw.PutBitUniform(1);
                for (int s = 0; s < NUM_MB_SEGMENTS; ++s) bw.PutSignedBits(enc.Dqm[s].Quant, 7);
                for (int s = 0; s < NUM_MB_SEGMENTS; ++s) bw.PutSignedBits(enc.Dqm[s].FStrength, 6);
            }
            if (enc.UpdateMap)
                for (int s = 0; s < 3; ++s)
                    if (bw.PutBitUniform(enc.Proba.Segments[s] != 255) != 0) bw.PutBits(enc.Proba.Segments[s], 8);
        }
    }

    private static void PutFilterHeader(Vp8EncBitWriter bw, Encoder enc)
    {
        bool useLfDelta = enc.I4x4LfDelta != 0;
        bw.PutBitUniform(enc.FilterSimple);
        bw.PutBits((uint)enc.FilterLevel, 6);
        bw.PutBits((uint)enc.FilterSharpness, 3);
        if (bw.PutBitUniform(useLfDelta) != 0)
        {
            bool needUpdate = enc.I4x4LfDelta != 0;
            if (bw.PutBitUniform(needUpdate) != 0)
            {
                bw.PutBits(0, 4);
                bw.PutSignedBits(enc.I4x4LfDelta, 6);
                bw.PutBits(0, 3);
            }
        }
    }

    private static void PutQuant(Vp8EncBitWriter bw, Encoder enc)
    {
        bw.PutBits((uint)enc.BaseQuant, 7);
        bw.PutSignedBits(enc.DqY1Dc, 4);
        bw.PutSignedBits(enc.DqY2Dc, 4);
        bw.PutSignedBits(enc.DqY2Ac, 4);
        bw.PutSignedBits(enc.DqUVDc, 4);
        bw.PutSignedBits(enc.DqUVAc, 4);
    }

    private static void GeneratePartition0(Encoder enc)
    {
        enc.Bw = new Vp8EncBitWriter(enc.MbW * enc.MbH * 7 / 8);
        var bw = enc.Bw;
        bw.PutBitUniform(0);   // colorspace
        bw.PutBitUniform(0);   // clamp type
        PutSegmentHeader(bw, enc);
        PutFilterHeader(bw, enc);
        bw.PutBits((uint)(enc.NumParts == 8 ? 3 : enc.NumParts == 4 ? 2 : enc.NumParts == 2 ? 1 : 0), 2);
        PutQuant(bw, enc);
        bw.PutBitUniform(0);   // no proba update
        WriteProbas(bw, enc.Proba);
        CodeIntraModes(enc);
        bw.Finish();
    }

    private static byte[] VP8EncWrite(Encoder enc)
    {
        GeneratePartition0(enc);
        var bw = enc.Bw;
        int size0 = bw.Size;
        if (size0 >= VP8_MAX_PARTITION0_SIZE) throw new InvalidOperationException("partition #0 is too big");
        long vp8Size = 10 + size0 + 3 * (enc.NumParts - 1);
        for (int p = 0; p < enc.NumParts; ++p) vp8Size += enc.Parts[p].Size;
        int pad = (int)(vp8Size & 1);
        vp8Size += pad;
        long riffSize = 4 + 8 + vp8Size;   // TAG_SIZE + CHUNK_HEADER_SIZE + vp8_size (no VP8X / ALPH here)

        var o = new MemoryStream();
        void LE32(uint v) { o.WriteByte((byte)v); o.WriteByte((byte)(v >> 8)); o.WriteByte((byte)(v >> 16)); o.WriteByte((byte)(v >> 24)); }
        o.Write("RIFF"u8); LE32((uint)riffSize); o.Write("WEBP"u8);
        o.Write("VP8 "u8); LE32((uint)vp8Size);
        uint bits = 0 | ((uint)enc.Profile << 1) | (1u << 4) | ((uint)size0 << 5);
        o.WriteByte((byte)bits); o.WriteByte((byte)(bits >> 8)); o.WriteByte((byte)(bits >> 16));
        o.WriteByte(0x9d); o.WriteByte(0x01); o.WriteByte(0x2a);
        o.WriteByte((byte)(enc.Width & 0xff)); o.WriteByte((byte)(enc.Width >> 8));
        o.WriteByte((byte)(enc.Height & 0xff)); o.WriteByte((byte)(enc.Height >> 8));
        o.Write(bw.Buffer);
        for (int p = 0; p < enc.NumParts - 1; ++p)
        {
            int ps = enc.Parts[p].Size;
            o.WriteByte((byte)ps); o.WriteByte((byte)(ps >> 8)); o.WriteByte((byte)(ps >> 16));
        }
        for (int p = 0; p < enc.NumParts; ++p) o.Write(enc.Parts[p].Buffer);
        if (pad != 0) o.WriteByte(0);
        return o.ToArray();
    }
}
