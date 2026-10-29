// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT band recursion on the accelerator, encode side WITH resynthesis, for theta RDO, ported from libopus 1.5.2 (fixed
// point; celt/bands.c quant_all_bands with theta_rdo = stereo, no dual stereo, complexity >= 8): the encoder rebuilds each
// band as the decoder will (folding from the coded lower bands, the stereo merge), and for every band below the intensity
// band codes the split angle rounded down and rounded up, keeping the one with the smaller weighted distortion (the range
// coder state, the bytes it wrote, the band and its folding history saved and restored around the second try). The
// reconstruction is the decoder's (CeltQuantBandsGpu, bit-exact with libopus 1.5.2), with encode calls for the decode ones.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>The CELT band recursion, encode side with resynthesis and theta RDO (libopus 1.5.2, GPU-callable).</summary>
public static class CeltQuantBandsRdoGpu
{
    private const int Nb = CeltModeTables.NbEBands;
    private const int PartitionDepth = CeltQuantBandsGpu.PartitionDepth, MaxBandN = CeltQuantBandsGpu.MaxBandN;
    private const int FrameInts = 20;
    /// <summary>Scratch ints of one band's coding: the partition stack, alg_quant's three vectors, the interleave buffer.</summary>
    private const int BandScratch = PartitionDepth * FrameInts + 4 * MaxBandN;
    // quant_all_bands scratch after the band scratch: norm (2 * 8 * 78), lowband_scratch, X_save, Y_save, X_save2, Y_save2,
    // norm_save2 (176 each), the collapse masks (2 * 21), the saved bytes (1275)
    private const int QNorm = BandScratch, QNormStride = 8 * 78, QLowScratch = QNorm + 2 * QNormStride, QXSave = QLowScratch + MaxBandN,
        QYSave = QXSave + MaxBandN, QXSave2 = QYSave + MaxBandN, QYSave2 = QXSave2 + MaxBandN, QNormSave2 = QYSave2 + MaxBandN,
        QMasks = QNormSave2 + MaxBandN, QBytes = QMasks + 2 * Nb;
    /// <summary>Scratch ints one <see cref="QuantAllBandsEncodeRdo"/> call needs.</summary>
    public const int ScratchInts = QBytes + 1275;

    private const int FX = 0, FN = 1, FBits = 2, FB = 3, FLow = 4, FLM = 5, FGain = 6, FFill = 7, FStage = 8, FMidFirst = 9,
        FB0 = 10, FXSide = 11, FLow2 = 12, FRem = 13, FItheta = 14, FImid = 15, FIside = 16, FMbits = 17, FSbits = 18, FResult = 19;

    private static uint QuantBandN1(ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt, uint storage,
        ArrayView<int> x, int xAt, int yAt, int lowbandOutAt)
    {
        int at = xAt;
        int channels = yAt >= 0 ? 2 : 1;
        for (int c = 0; c < channels; c++)
        {
            int sign = 0;
            if (ctx.RemainingBits >= 1 << BITRES)
            {
                sign = x[at] < 0 ? 1 : 0;
                OpusRangeEncoderGpu.EncodeBits(ref ec, buf, bufAt, storage, (uint)sign, 1);
                ctx.RemainingBits -= 1 << BITRES;
            }
            x[at] = sign != 0 ? -16384 : 16384;   // NORM_SCALING
            at = yAt;
        }
        if (lowbandOutAt >= 0) x[lowbandOutAt] = x[xAt] >> 4;
        return 1;
    }

    // quant_partition (encode, resynthesis), iterative as in the decoder.
    private static uint QuantPartition(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt,
        uint storage, ArrayView<int> x, int xAt, int N, int b, int B, int lowbandAt, int LM, int gain, int fill, ArrayView<int> s, int sAt,
        ArrayView<uint> pvqU)
    {
        int vecAt = sAt + PartitionDepth * FrameInts;
        uint ret = 0;
        int f0 = sAt;
        s[f0 + FX] = xAt; s[f0 + FN] = N; s[f0 + FBits] = b; s[f0 + FB] = B; s[f0 + FLow] = lowbandAt; s[f0 + FLM] = LM;
        s[f0 + FGain] = gain; s[f0 + FFill] = fill; s[f0 + FStage] = 0;
        int sp = 1;
        while (true)
        {
            int f = sAt + (sp - 1) * FrameInts;
            int stage = s[f + FStage];
            if (stage == 0)
            {
                int fx = s[f + FX], fN = s[f + FN], fb = s[f + FBits], fB = s[f + FB], fLow = s[f + FLow], fLM = s[f + FLM];
                int fGain = s[f + FGain], fFill = s[f + FFill];
                int B0 = fB, i = ctx.I;
                int cache = m.CacheIndexOf((fLM + 1) * Nb + i);
                if (fLM != -1 && fb > m.CacheBitsOf(cache + m.CacheBitsOf(cache)) + 12 && fN > 2)
                {
                    fN >>= 1;
                    int fy = fx + fN;
                    fLM -= 1;
                    if (fB == 1) fFill = (fFill & 1) | (fFill << 1);
                    fB = (fB + 1) >> 1;
                    CeltQuantBandsEncGpu.ComputeThetaEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, fx, fy, x, 0, fN, ref fb, fB, B0, fLM, 0,
                        ref fFill, out _, out int imid, out int iside, out int delta, out int itheta, out int qalloc);
                    // Give more bits to low-energy MDCTs than they would otherwise deserve
                    if (B0 > 1 && (itheta & 0x3fff) != 0)
                    {
                        if (itheta > 8192) delta -= delta >> (4 - fLM);
                        else delta = IMIN(0, delta + (fN << BITRES >> (5 - fLM)));
                    }
                    int mbits = IMAX(0, IMIN(fb, (fb - delta) / 2));
                    int sbits = fb - mbits;
                    ctx.RemainingBits -= qalloc;
                    int low2 = fLow >= 0 ? fLow + fN : -1;
                    s[f + FN] = fN; s[f + FB] = fB; s[f + FLM] = fLM; s[f + FFill] = fFill; s[f + FB0] = B0; s[f + FXSide] = fy;
                    s[f + FLow2] = low2; s[f + FRem] = ctx.RemainingBits; s[f + FItheta] = itheta; s[f + FImid] = imid; s[f + FIside] = iside;
                    s[f + FMbits] = mbits; s[f + FSbits] = sbits; s[f + FMidFirst] = mbits >= sbits ? 1 : 0; s[f + FStage] = 1;
                    int c = f + FrameInts;
                    if (mbits >= sbits)
                    {
                        s[c + FX] = fx; s[c + FBits] = mbits; s[c + FLow] = fLow; s[c + FGain] = MULT16_16_P15(fGain, imid); s[c + FFill] = fFill;
                    }
                    else
                    {
                        s[c + FX] = fy; s[c + FBits] = sbits; s[c + FLow] = low2; s[c + FGain] = MULT16_16_P15(fGain, iside); s[c + FFill] = fFill >> fB;
                    }
                    s[c + FN] = fN; s[c + FB] = fB; s[c + FLM] = fLM; s[c + FStage] = 0;
                    sp++;
                    continue;
                }
                // The basic no-split case
                uint cm = 0;
                int q = CeltRateGpu.Bits2Pulses(m, i, fLM, fb);
                int curr_bits = CeltRateGpu.Pulses2Bits(m, i, fLM, q);
                ctx.RemainingBits -= curr_bits;
                while (ctx.RemainingBits < 0 && q > 0)
                {
                    ctx.RemainingBits += curr_bits;
                    q--;
                    curr_bits = CeltRateGpu.Pulses2Bits(m, i, fLM, q);
                    ctx.RemainingBits -= curr_bits;
                }
                if (q != 0)
                {
                    int K = CeltRateGpu.GetPulses(q);
                    cm = CeltPvqEncGpu.AlgQuant(x, fx, fN, K, ctx.Spread, fB, ref ec, buf, bufAt, storage, fGain, 1, s, vecAt, vecAt + MaxBandN,
                        vecAt + 2 * MaxBandN, pvqU, 0);
                }
                else
                {
                    // If there's no pulse, fill the band anyway
                    uint cm_mask = (uint)((1L << fB) - 1);
                    fFill &= (int)cm_mask;
                    if (fFill == 0)
                    {
                        for (int j = 0; j < fN; j++) x[fx + j] = 0;
                    }
                    else
                    {
                        if (fLow < 0)
                        {
                            // Noise
                            for (int j = 0; j < fN; j++)
                            {
                                ctx.Seed = CeltBandsGpu.CeltLcgRand(ctx.Seed);
                                x[fx + j] = (int)ctx.Seed >> 20;
                            }
                            cm = cm_mask;
                        }
                        else
                        {
                            // Folded spectrum, about 48 dB below the "normal" folding level
                            for (int j = 0; j < fN; j++)
                            {
                                ctx.Seed = CeltBandsGpu.CeltLcgRand(ctx.Seed);
                                int tmp = (ctx.Seed & 0x8000) != 0 ? 4 : -4;   // QCONST16(1.0f / 256, 10)
                                x[fx + j] = S16(x[fLow + j] + tmp);
                            }
                            cm = (uint)fFill;
                        }
                        CeltPvqGpu.RenormaliseVector(x, fx, fN, fGain);
                    }
                }
                ret = cm;
                sp--;
                if (sp == 0) break;
                continue;
            }
            if (stage == 1)
            {
                // the first child finished with `ret`
                int fN = s[f + FN], fB = s[f + FB], fLM = s[f + FLM], fFill = s[f + FFill], fGain = s[f + FGain];
                int B0 = s[f + FB0], itheta = s[f + FItheta], mbits = s[f + FMbits], sbits = s[f + FSbits];
                int rebalance = s[f + FRem];
                int c = f + FrameInts;
                if (s[f + FMidFirst] != 0)
                {
                    s[f + FResult] = (int)ret;
                    rebalance = mbits - (rebalance - ctx.RemainingBits);
                    if (rebalance > 3 << BITRES && itheta != 0) sbits += rebalance - (3 << BITRES);
                    s[c + FX] = s[f + FXSide]; s[c + FBits] = sbits; s[c + FLow] = s[f + FLow2];
                    s[c + FGain] = MULT16_16_P15(fGain, s[f + FIside]); s[c + FFill] = fFill >> fB;
                }
                else
                {
                    s[f + FResult] = (int)(ret << (B0 >> 1));
                    rebalance = sbits - (rebalance - ctx.RemainingBits);
                    if (rebalance > 3 << BITRES && itheta != 16384) mbits += rebalance - (3 << BITRES);
                    s[c + FX] = s[f + FX]; s[c + FBits] = mbits; s[c + FLow] = s[f + FLow];
                    s[c + FGain] = MULT16_16_P15(fGain, s[f + FImid]); s[c + FFill] = fFill;
                }
                s[c + FN] = fN; s[c + FB] = fB; s[c + FLM] = fLM; s[c + FStage] = 0;
                s[f + FStage] = 2;
                sp++;
                continue;
            }
            // stage 2: the second child finished with `ret`
            {
                uint result = (uint)s[f + FResult];
                if (s[f + FMidFirst] != 0) result |= ret << (s[f + FB0] >> 1);
                else result |= ret;
                ret = result;
                sp--;
                if (sp == 0) break;
            }
        }
        return ret;
    }

    // quant_band (encode, resynthesis)
    private static uint QuantBand(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt,
        uint storage, ArrayView<int> x, int xAt, int N, int b, int B, int lowbandAt, int LM, int lowbandOutAt, int gain, int lowbandScratchAt,
        int fill, ArrayView<int> s, int sAt, ArrayView<uint> pvqU)
    {
        int tmpAt = sAt + PartitionDepth * FrameInts + 3 * MaxBandN;
        int N0 = N, N_B = N, B0 = B, time_divide = 0, recombine = 0;
        int longBlocks = B0 == 1 ? 1 : 0;
        int tf_change = ctx.TfChange;
        N_B = N_B / B;
        // Special case for one sample
        if (N == 1) return QuantBandN1(ref ctx, ref ec, buf, bufAt, storage, x, xAt, -1, lowbandOutAt);
        if (tf_change > 0) recombine = tf_change;
        // Band recombining to increase frequency resolution
        if (lowbandScratchAt >= 0 && lowbandAt >= 0 && (recombine != 0 || ((N_B & 1) == 0 && tf_change < 0) || B0 > 1))
        {
            for (int j = 0; j < N; j++) x[lowbandScratchAt + j] = x[lowbandAt + j];
            lowbandAt = lowbandScratchAt;
        }
        for (int k = 0; k < recombine; k++)
        {
            CeltBandsGpu.Haar1(x, xAt, N >> k, 1 << k);
            if (lowbandAt >= 0) CeltBandsGpu.Haar1(x, lowbandAt, N >> k, 1 << k);
            fill = CeltQuantBandsGpu.BitInterleave(fill & 0xF) | CeltQuantBandsGpu.BitInterleave(fill >> 4) << 2;
        }
        B >>= recombine;
        N_B <<= recombine;
        // Increasing the time resolution
        while ((N_B & 1) == 0 && tf_change < 0)
        {
            CeltBandsGpu.Haar1(x, xAt, N_B, B);
            if (lowbandAt >= 0) CeltBandsGpu.Haar1(x, lowbandAt, N_B, B);
            fill |= fill << B;
            B <<= 1;
            N_B >>= 1;
            time_divide++;
            tf_change++;
        }
        B0 = B;
        int N_B0 = N_B;
        // Reorganize the samples in time order instead of frequency order
        if (B0 > 1)
        {
            CeltBandsGpu.DeinterleaveHadamard(m, x, xAt, N_B >> recombine, B0 << recombine, longBlocks, s, tmpAt);
            if (lowbandAt >= 0) CeltBandsGpu.DeinterleaveHadamard(m, x, lowbandAt, N_B >> recombine, B0 << recombine, longBlocks, s, tmpAt);
        }
        uint cm = QuantPartition(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, N, b, B, lowbandAt, LM, gain, fill, s, sAt, pvqU);
        // Undo the sample reorganization going from time order to frequency order
        if (B0 > 1) CeltBandsGpu.InterleaveHadamard(m, x, xAt, N_B >> recombine, B0 << recombine, longBlocks, s, tmpAt);
        // Undo time-freq changes that we did earlier
        N_B = N_B0;
        B = B0;
        for (int k = 0; k < time_divide; k++)
        {
            B >>= 1;
            N_B <<= 1;
            cm |= cm >> B;
            CeltBandsGpu.Haar1(x, xAt, N_B, B);
        }
        for (int k = 0; k < recombine; k++)
        {
            cm = CeltQuantBandsGpu.BitDeinterleave(cm);
            CeltBandsGpu.Haar1(x, xAt, N0 >> k, 1 << k);
        }
        B <<= recombine;
        // Scale output for later folding
        if (lowbandOutAt >= 0)
        {
            int n = CeltFixed.celt_sqrt(SHL32(N0, 22));
            for (int j = 0; j < N0; j++) x[lowbandOutAt + j] = MULT16_16_Q15(n, x[xAt + j]);
        }
        cm &= (uint)((1 << B) - 1);
        return cm;
    }

    // quant_band_stereo (encode, resynthesis)
    private static uint QuantBandStereo(CeltModeGpu m, ref CeltBandEncCtx ctx, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt,
        uint storage, ArrayView<int> x, int xAt, int yAt, ArrayView<int> bandE, int bandEAt, int N, int b, int B, int lowbandAt, int LM,
        int lowbandOutAt, int lowbandScratchAt, int fill, ArrayView<int> s, int sAt, ArrayView<uint> pvqU)
    {
        // Special case for one sample
        if (N == 1) return QuantBandN1(ref ctx, ref ec, buf, bufAt, storage, x, xAt, yAt, lowbandOutAt);
        int orig_fill = fill;
        CeltQuantBandsEncGpu.ComputeThetaEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, yAt, bandE, bandEAt, N, ref b, B, B, LM, 1,
            ref fill, out int inv, out int mid, out int side, out int delta, out int itheta, out int qalloc);
        uint cm;
        if (N == 2)
        {
            // N = 2 codes the side with one sign bit (mid and side are orthogonal)
            int mbits = b, sbits = 0;
            if (itheta != 0 && itheta != 16384) sbits = 1 << BITRES;
            mbits -= sbits;
            int c = itheta > 8192 ? 1 : 0;
            ctx.RemainingBits -= qalloc + sbits;
            int x2 = c != 0 ? yAt : xAt, y2 = c != 0 ? xAt : yAt;
            int sign = 0;
            if (sbits != 0)
            {
                sign = x[x2] * x[y2 + 1] - x[x2 + 1] * x[y2] < 0 ? 1 : 0;
                OpusRangeEncoderGpu.EncodeBits(ref ec, buf, bufAt, storage, (uint)sign, 1);
            }
            sign = 1 - 2 * sign;
            // orig_fill: we want to fold the side, but itheta == 16384 cleared the low bits of fill
            cm = QuantBand(m, ref ctx, ref ec, buf, bufAt, storage, x, x2, N, mbits, B, lowbandAt, LM, lowbandOutAt, Q15ONE, lowbandScratchAt,
                orig_fill, s, sAt, pvqU);
            // cm is 1 or 0 (a fold-collapse): no need to mix it with the other channel
            x[y2] = S16(-sign * x[x2 + 1]);
            x[y2 + 1] = S16(sign * x[x2]);
            x[xAt] = MULT16_16_Q15(mid, x[xAt]);
            x[xAt + 1] = MULT16_16_Q15(mid, x[xAt + 1]);
            x[yAt] = MULT16_16_Q15(side, x[yAt]);
            x[yAt + 1] = MULT16_16_Q15(side, x[yAt + 1]);
            int tmp = x[xAt];
            x[xAt] = S16(SUB16(tmp, x[yAt]));
            x[yAt] = ADD16(tmp, x[yAt]);
            tmp = x[xAt + 1];
            x[xAt + 1] = S16(SUB16(tmp, x[yAt + 1]));
            x[yAt + 1] = ADD16(tmp, x[yAt + 1]);
        }
        else
        {
            // "Normal" split code
            int mbits = IMAX(0, IMIN(b, (b - delta) / 2));
            int sbits = b - mbits;
            ctx.RemainingBits -= qalloc;
            int rebalance = ctx.RemainingBits;
            if (mbits >= sbits)
            {
                // The mid is not scaled: its normalized version folds later
                cm = QuantBand(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, N, mbits, B, lowbandAt, LM, lowbandOutAt, Q15ONE, lowbandScratchAt,
                    fill, s, sAt, pvqU);
                rebalance = mbits - (rebalance - ctx.RemainingBits);
                if (rebalance > 3 << BITRES && itheta != 0) sbits += rebalance - (3 << BITRES);
                // For a stereo split, the high bits of fill are always zero, so no folding will be done to the side.
                cm |= QuantBand(m, ref ctx, ref ec, buf, bufAt, storage, x, yAt, N, sbits, B, -1, LM, -1, side, -1, fill >> B, s, sAt, pvqU);
            }
            else
            {
                cm = QuantBand(m, ref ctx, ref ec, buf, bufAt, storage, x, yAt, N, sbits, B, -1, LM, -1, side, -1, fill >> B, s, sAt, pvqU);
                rebalance = sbits - (rebalance - ctx.RemainingBits);
                if (rebalance > 3 << BITRES && itheta != 16384) mbits += rebalance - (3 << BITRES);
                cm |= QuantBand(m, ref ctx, ref ec, buf, bufAt, storage, x, xAt, N, mbits, B, lowbandAt, LM, lowbandOutAt, Q15ONE, lowbandScratchAt,
                    fill, s, sAt, pvqU);
            }
            CeltBandsGpu.StereoMerge(x, xAt, x, yAt, mid, N);
        }
        if (inv != 0)
            for (int j = 0; j < N; j++) x[yAt + j] = S16(-x[yAt + j]);
        return cm;
    }

    /// <summary>
    /// libopus quant_all_bands (encode, stereo, theta RDO: resynthesis, no dual stereo): codes the shapes of bands
    /// start..end-1 of the normalised stereo spectrum (channel 0 at <paramref name="xAt"/>, channel 1 at <paramref name="yAt"/>,
    /// in <paramref name="x"/>), trying both roundings of the split angle of every band below the intensity band. The
    /// arguments are <see cref="CeltQuantBandsEncGpu.QuantAllBandsEncode"/>'s, plus the folding seed (in / out) and
    /// <see cref="ScratchInts"/> ints of scratch at <paramref name="sAt"/>. <paramref name="s"/> must be the view of
    /// <paramref name="x"/>: the folding history (norm), the lowband scratch and the bands are offsets into one view, as in
    /// the decoder.
    /// </summary>
    public static void QuantAllBandsEncodeRdo(CeltModeGpu m, int start, int end, ArrayView<int> x, int xAt, int yAt, ArrayView<int> bandE,
        int bandEAt, ArrayView<int> pulses, int pulsesAt, int shortBlocks, int spread, int intensity, ArrayView<int> tfRes, int tfAt,
        int total_bits, int balance, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt, uint storage, int LM, int codedBands,
        ref uint seed, int disableInv, ArrayView<int> s, int sAt, ArrayView<uint> pvqU)
    {
        const int C = 2;
        int M = 1 << LM;
        int B = shortBlocks != 0 ? M : 1;
        int norm_offset = M * m.EBandsOf(start);
        int normAt = sAt + QNorm, norm2At = normAt + M * m.EBandsOf(Nb - 1) - norm_offset;
        int masks = sAt + QMasks;
        int lowband_offset = 0, update_lowband = 1;
        var ctx = new CeltBandEncCtx { Intensity = intensity, Spread = spread, DisableInv = disableInv, Seed = seed, ThetaRound = 0 };
        // Avoid injecting noise in the first band on transients.
        ctx.AvoidSplitNoise = B > 1 ? 1 : 0;
        for (int i = start; i < end; i++)
        {
            ctx.I = i;
            int last = i == end - 1 ? 1 : 0;
            int X = xAt + M * m.EBandsOf(i), Y = yAt + M * m.EBandsOf(i);
            int N = M * m.EBandsOf(i + 1) - M * m.EBandsOf(i);
            int tell = (int)OpusRangeEncoderGpu.TellFrac(ref ec);
            // Compute how many bits we want to allocate to this band
            if (i != start) balance -= tell;
            int remaining_bits = total_bits - tell - 1;
            ctx.RemainingBits = remaining_bits;
            int b;
            if (i <= codedBands - 1)
            {
                int curr_balance = balance / IMIN(3, codedBands - i);
                b = IMAX(0, IMIN(16383, IMIN(remaining_bits + 1, pulses[pulsesAt + i] + curr_balance)));
            }
            else b = 0;
            if ((M * m.EBandsOf(i) - N >= M * m.EBandsOf(start) || i == start + 1) && (update_lowband != 0 || lowband_offset == 0))
                lowband_offset = i;
            if (i == start + 1) CeltQuantBandsGpu.SpecialHybridFolding(m, s, normAt, norm2At, start, M, 0);
            ctx.TfChange = tfRes[tfAt + i];
            int lowband_scratch = sAt + QLowScratch;   // the encoder's own (resynth_alloc), kept for the last band (theta RDO)
            // A conservative estimate of the collapse masks of the bands we fold from.
            int effective_lowband = -1;
            uint x_cm, y_cm;
            if (lowband_offset != 0 && (spread != 3 || B > 1 || ctx.TfChange < 0))   // SPREAD_AGGRESSIVE
            {
                // This ensures we never repeat spectral content within one band
                effective_lowband = IMAX(0, M * m.EBandsOf(lowband_offset) - norm_offset - N);
                int fold_start = lowband_offset;
                while (M * m.EBandsOf(--fold_start) > effective_lowband + norm_offset) { }
                int fold_end = lowband_offset - 1;
                while (++fold_end < i && M * m.EBandsOf(fold_end) < effective_lowband + norm_offset + N) { }
                x_cm = y_cm = 0;
                int fold_i = fold_start;
                do
                {
                    x_cm |= (uint)s[masks + fold_i * C];
                    y_cm |= (uint)s[masks + fold_i * C + C - 1];
                }
                while (++fold_i < fold_end);
            }
            else x_cm = y_cm = (uint)((1 << B) - 1);   // the LCG folds: all blocks (almost always) non-zero
            int low = effective_lowband != -1 ? normAt + effective_lowband : -1;
            int lowOut = last != 0 ? -1 : normAt + M * m.EBandsOf(i) - norm_offset;
            if (i < intensity)
            {
                // Theta RDO: code the band with theta rounded down, then up, and keep the one with the smaller distortion
                int ex = bandE[bandEAt + i], ey = bandE[bandEAt + i + Nb];
                int minE = IMIN(ex, ey);
                // Adjustment to make the weights a bit more conservative.
                ex += minE / 3;
                ey += minE / 3;
                int shift = celt_ilog2(EPSILON + IMAX(ex, ey)) - 14;
                int w0 = S16(VSHR32(ex, shift)), w1 = S16(VSHR32(ey, shift));
                uint cm = x_cm | y_cm;
                var ec_save = ec;
                var ctx_save = ctx;
                for (int j = 0; j < N; j++)
                {
                    s[sAt + QXSave + j] = x[X + j];
                    s[sAt + QYSave + j] = x[Y + j];
                }
                // Encode and round down.
                ctx.ThetaRound = -1;
                x_cm = QuantBandStereo(m, ref ctx, ref ec, buf, bufAt, storage, x, X, Y, bandE, bandEAt, N, b, B, low, LM, lowOut, lowband_scratch,
                    (int)cm, s, sAt, pvqU);
                int dist0 = MULT16_32_Q15(w0, InnerProdX(s, sAt + QXSave, x, X, N)) + MULT16_32_Q15(w1, InnerProdX(s, sAt + QYSave, x, Y, N));
                // Save first result.
                uint cm2 = x_cm;
                var ec_save2 = ec;
                var ctx_save2 = ctx;
                for (int j = 0; j < N; j++)
                {
                    s[sAt + QXSave2 + j] = x[X + j];
                    s[sAt + QYSave2 + j] = x[Y + j];
                }
                if (last == 0)
                    for (int j = 0; j < N; j++) s[sAt + QNormSave2 + j] = s[normAt + M * m.EBandsOf(i) - norm_offset + j];
                int nstart_bytes = (int)ec_save.Offs, nend_bytes = (int)storage, save_bytes = nend_bytes - nstart_bytes;
                for (int j = 0; j < save_bytes; j++) s[sAt + QBytes + j] = buf[bufAt + nstart_bytes + j];
                // Restore
                ec = ec_save;
                ctx = ctx_save;
                for (int j = 0; j < N; j++)
                {
                    x[X + j] = s[sAt + QXSave + j];
                    x[Y + j] = s[sAt + QYSave + j];
                }
                if (i == start + 1) CeltQuantBandsGpu.SpecialHybridFolding(m, s, normAt, norm2At, start, M, 0);
                // Encode and round up.
                ctx.ThetaRound = 1;
                x_cm = QuantBandStereo(m, ref ctx, ref ec, buf, bufAt, storage, x, X, Y, bandE, bandEAt, N, b, B, low, LM, lowOut, lowband_scratch,
                    (int)cm, s, sAt, pvqU);
                int dist1 = MULT16_32_Q15(w0, InnerProdX(s, sAt + QXSave, x, X, N)) + MULT16_32_Q15(w1, InnerProdX(s, sAt + QYSave, x, Y, N));
                if (dist0 >= dist1)
                {
                    x_cm = cm2;
                    ec = ec_save2;
                    ctx = ctx_save2;
                    for (int j = 0; j < N; j++)
                    {
                        x[X + j] = s[sAt + QXSave2 + j];
                        x[Y + j] = s[sAt + QYSave2 + j];
                    }
                    if (last == 0)
                        for (int j = 0; j < N; j++) s[normAt + M * m.EBandsOf(i) - norm_offset + j] = s[sAt + QNormSave2 + j];
                    for (int j = 0; j < save_bytes; j++) buf[bufAt + nstart_bytes + j] = (byte)s[sAt + QBytes + j];
                }
            }
            else
            {
                ctx.ThetaRound = 0;
                x_cm = QuantBandStereo(m, ref ctx, ref ec, buf, bufAt, storage, x, X, Y, bandE, bandEAt, N, b, B, low, LM, lowOut, lowband_scratch,
                    (int)(x_cm | y_cm), s, sAt, pvqU);
            }
            y_cm = x_cm;
            s[masks + i * C] = (int)(x_cm & 0xFF);
            s[masks + i * C + C - 1] = (int)(y_cm & 0xFF);
            balance += pulses[pulsesAt + i] + tell;
            // Update the folding position only as long as we have 1 bit/sample depth.
            update_lowband = b > N << BITRES ? 1 : 0;
            // We only need to avoid noise on a split for the first band. After that, we have folding.
            ctx.AvoidSplitNoise = 0;
        }
        seed = ctx.Seed;
    }

    // celt_inner_prod of a saved band (in the scratch view) and the band
    private static int InnerProdX(ArrayView<int> s, int sAt, ArrayView<int> x, int xAt, int N)
    {
        int xy = 0;
        for (int i = 0; i < N; i++) xy = MAC16_16(xy, s[sAt + i], x[xAt + i]);
        return xy;
    }
}
