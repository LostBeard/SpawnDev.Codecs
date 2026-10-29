// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT band recursion on the accelerator, encode side, ported from libopus 1.5.2 (fixed point; celt/bands.c
// quant_all_bands, quant_band, quant_band_stereo, quant_band_n1, quant_partition with encode = 1), with and without the
// band resynthesis (libopus resynthesises only for theta RDO: stereo, no dual stereo, complexity >= 8, where every band below
// the intensity band is coded with the split angle rounded down and up and the one with the smaller weighted distortion is
// kept; the range coder state, its bytes, the band and its folding history saved and restored around the second try).
//
// Shaped for WGSL like the decoder (CeltQuantBandsGpu): WebGPU inlines every helper that takes a view, so each heavy helper
// has ONE call site. The partition recursion runs on an explicit frame stack; a band's quant_band calls (mono, dual stereo,
// the stereo N = 2 mid, the stereo mid and side) are jobs through one call site; the two RDO tries loop around one band body.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>The CELT band recursion, encode side (libopus 1.5.2 quant_all_bands, GPU-callable).</summary>
public static class CeltBandsEncGpu
{
    private const int Nb = CeltModeTables.NbEBands;
    private const int PartitionDepth = CeltQuantBandsGpu.PartitionDepth, MaxBandN = CeltQuantBandsGpu.MaxBandN;
    private const int FrameInts = 20;
    private const int BandScratch = PartitionDepth * FrameInts + 4 * MaxBandN;
    // quant_all_bands scratch after the band scratch: norm (2 * 8 * 78), lowband_scratch, X_save, Y_save, X_save2, Y_save2,
    // norm_save2 (176 each), the collapse masks (2 * 21), the saved bytes (1275)
    private const int QNorm = BandScratch, QNormStride = 8 * 78, QLowScratch = QNorm + 2 * QNormStride, QXSave = QLowScratch + MaxBandN,
        QYSave = QXSave + MaxBandN, QXSave2 = QYSave + MaxBandN, QYSave2 = QXSave2 + MaxBandN, QNormSave2 = QYSave2 + MaxBandN,
        QMasks = QNormSave2 + MaxBandN, QBytes = QMasks + 2 * Nb;
    /// <summary>Scratch ints one <see cref="QuantAllBands"/> call needs.</summary>
    public const int ScratchInts = QBytes + 1275;

    private const int FX = 0, FN = 1, FBits = 2, FB = 3, FLow = 4, FLM = 5, FGain = 6, FFill = 7, FStage = 8, FMidFirst = 9,
        FB0 = 10, FXSide = 11, FLow2 = 12, FRem = 13, FItheta = 14, FImid = 15, FIside = 16, FMbits = 17, FSbits = 18, FResult = 19;

    // quant_band_n1: a band of one bin per channel is a sign bit
    private static uint QuantBandN1(ref CeltBandEncCtx ctx, int resynth, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt,
        uint storage, ArrayView<int> x, int xAt, int yAt, int lowbandOutAt)
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
            if (resynth != 0) x[at] = sign != 0 ? -16384 : 16384;   // NORM_SCALING
            at = yAt;
        }
        if (lowbandOutAt >= 0) x[lowbandOutAt] = x[xAt] >> 4;
        return 1;
    }

    // quant_partition, iterative as in the decoder
    private static uint QuantPartition(CeltModeGpu m, ref CeltBandEncCtx ctx, int resynth, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf,
        int bufAt, uint storage, ArrayView<int> x, int xAt, int N, int b, int B, int lowbandAt, int LM, int gain, int fill, int sAt,
        ArrayView<uint> pvqU)
    {
        int vecAt = sAt + PartitionDepth * FrameInts;
        uint ret = 0;
        int f0 = sAt;
        x[f0 + FX] = xAt; x[f0 + FN] = N; x[f0 + FBits] = b; x[f0 + FB] = B; x[f0 + FLow] = lowbandAt; x[f0 + FLM] = LM;
        x[f0 + FGain] = gain; x[f0 + FFill] = fill; x[f0 + FStage] = 0;
        int sp = 1;
        while (true)
        {
            int f = sAt + (sp - 1) * FrameInts;
            int stage = x[f + FStage];
            if (stage == 0)
            {
                int fx = x[f + FX], fN = x[f + FN], fb = x[f + FBits], fB = x[f + FB], fLow = x[f + FLow], fLM = x[f + FLM];
                int fGain = x[f + FGain], fFill = x[f + FFill];
                int B0 = fB, i = ctx.I;
                // If we need 1.5 more bit than we can produce, split the band in two.
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
                        if (itheta > 8192) delta -= delta >> (4 - fLM);   // Rough approximation for pre-echo masking
                        else delta = IMIN(0, delta + (fN << BITRES >> (5 - fLM)));   // a forward-masking slope of 1.5 dB per 10 ms
                    }
                    int mbits = IMAX(0, IMIN(fb, (fb - delta) / 2));
                    int sbits = fb - mbits;
                    ctx.RemainingBits -= qalloc;
                    int low2 = fLow >= 0 ? fLow + fN : -1;
                    x[f + FN] = fN; x[f + FB] = fB; x[f + FLM] = fLM; x[f + FFill] = fFill; x[f + FB0] = B0; x[f + FXSide] = fy;
                    x[f + FLow2] = low2; x[f + FRem] = ctx.RemainingBits; x[f + FItheta] = itheta; x[f + FImid] = imid; x[f + FIside] = iside;
                    x[f + FMbits] = mbits; x[f + FSbits] = sbits; x[f + FMidFirst] = mbits >= sbits ? 1 : 0; x[f + FStage] = 1;
                    int c = f + FrameInts;
                    if (mbits >= sbits)
                    {
                        x[c + FX] = fx; x[c + FBits] = mbits; x[c + FLow] = fLow; x[c + FGain] = MULT16_16_P15(fGain, imid); x[c + FFill] = fFill;
                    }
                    else
                    {
                        x[c + FX] = fy; x[c + FBits] = sbits; x[c + FLow] = low2; x[c + FGain] = MULT16_16_P15(fGain, iside); x[c + FFill] = fFill >> fB;
                    }
                    x[c + FN] = fN; x[c + FB] = fB; x[c + FLM] = fLM; x[c + FStage] = 0;
                    sp++;
                    continue;
                }
                // The basic no-split case
                uint cm = 0;
                int q = CeltRateGpu.Bits2Pulses(m, i, fLM, fb);
                int curr_bits = CeltRateGpu.Pulses2Bits(m, i, fLM, q);
                ctx.RemainingBits -= curr_bits;
                // Ensures we can never bust the budget
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
                    cm = CeltPvqEncGpu.AlgQuant(x, fx, fN, K, ctx.Spread, fB, ref ec, buf, bufAt, storage, fGain, resynth, x, vecAt,
                        vecAt + MaxBandN, vecAt + 2 * MaxBandN, pvqU, 0);
                }
                else if (resynth != 0)
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
                int fN = x[f + FN], fB = x[f + FB], fLM = x[f + FLM], fFill = x[f + FFill], fGain = x[f + FGain];
                int B0 = x[f + FB0], itheta = x[f + FItheta], mbits = x[f + FMbits], sbits = x[f + FSbits];
                int rebalance = x[f + FRem];
                int c = f + FrameInts;
                if (x[f + FMidFirst] != 0)
                {
                    x[f + FResult] = (int)ret;
                    rebalance = mbits - (rebalance - ctx.RemainingBits);
                    if (rebalance > 3 << BITRES && itheta != 0) sbits += rebalance - (3 << BITRES);
                    x[c + FX] = x[f + FXSide]; x[c + FBits] = sbits; x[c + FLow] = x[f + FLow2];
                    x[c + FGain] = MULT16_16_P15(fGain, x[f + FIside]); x[c + FFill] = fFill >> fB;
                }
                else
                {
                    x[f + FResult] = (int)(ret << (B0 >> 1));
                    rebalance = sbits - (rebalance - ctx.RemainingBits);
                    if (rebalance > 3 << BITRES && itheta != 16384) mbits += rebalance - (3 << BITRES);
                    x[c + FX] = x[f + FX]; x[c + FBits] = mbits; x[c + FLow] = x[f + FLow];
                    x[c + FGain] = MULT16_16_P15(fGain, x[f + FImid]); x[c + FFill] = fFill;
                }
                x[c + FN] = fN; x[c + FB] = fB; x[c + FLM] = fLM; x[c + FStage] = 0;
                x[f + FStage] = 2;
                sp++;
                continue;
            }
            // stage 2: the second child finished with `ret`
            {
                uint result = (uint)x[f + FResult];
                if (x[f + FMidFirst] != 0) result |= ret << (x[f + FB0] >> 1);
                else result |= ret;
                ret = result;
                sp--;
                if (sp == 0) break;
            }
        }
        return ret;
    }

    // quant_band: one band of one channel, with the TF resolution changes around quant_partition
    private static uint QuantBand(CeltModeGpu m, ref CeltBandEncCtx ctx, int resynth, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf,
        int bufAt, uint storage, ArrayView<int> x, int xAt, int N, int b, int B, int lowbandAt, int LM, int lowbandOutAt, int gain,
        int lowbandScratchAt, int fill, int sAt, ArrayView<uint> pvqU)
    {
        int tmpAt = sAt + PartitionDepth * FrameInts + 3 * MaxBandN;
        int N0 = N, N_B = N, B0 = B, time_divide = 0, recombine = 0;
        int longBlocks = B0 == 1 ? 1 : 0;
        int tf_change = ctx.TfChange;
        N_B = N_B / B;
        // Special case for one sample
        if (N == 1) return QuantBandN1(ref ctx, resynth, ref ec, buf, bufAt, storage, x, xAt, -1, lowbandOutAt);
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
            CeltBandsGpu.DeinterleaveHadamard(m, x, xAt, N_B >> recombine, B0 << recombine, longBlocks, x, tmpAt);
            if (lowbandAt >= 0) CeltBandsGpu.DeinterleaveHadamard(m, x, lowbandAt, N_B >> recombine, B0 << recombine, longBlocks, x, tmpAt);
        }
        uint cm = QuantPartition(m, ref ctx, resynth, ref ec, buf, bufAt, storage, x, xAt, N, b, B, lowbandAt, LM, gain, fill, sAt, pvqU);
        // This code is used by the decoder and by the resynthesis-enabled encoder
        if (resynth != 0)
        {
            // Undo the sample reorganization going from time order to frequency order
            if (B0 > 1) CeltBandsGpu.InterleaveHadamard(m, x, xAt, N_B >> recombine, B0 << recombine, longBlocks, x, tmpAt);
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
        }
        return cm;
    }

    /// <summary>
    /// libopus quant_all_bands (encode): codes the shapes of bands start..end-1 of the normalised spectrum (channel 0 at
    /// <paramref name="xAt"/>, channel 1 at <paramref name="yAt"/>, -1 for mono; consumed in place) with the allocation's
    /// per-band <paramref name="pulses"/> and the balance. <paramref name="complexity"/> decides theta RDO (stereo, no dual
    /// stereo, complexity >= 8) and with it the resynthesis. <paramref name="bandE"/> holds the linear band energies (channel c
    /// at bandEAt + c * NbEBands). The scratch (<see cref="ScratchInts"/> ints at <paramref name="sAt"/>) is in the view of
    /// the bands: the folding history and the lowband scratch are offsets into it, as in the decoder.
    /// </summary>
    public static void QuantAllBands(CeltModeGpu m, int start, int end, ArrayView<int> x, int xAt, int yAt, ArrayView<int> bandE,
        int bandEAt, ArrayView<int> pulses, int pulsesAt, int shortBlocks, int spread, int dual_stereo, int intensity, ArrayView<int> tfRes,
        int tfAt, int total_bits, int balance, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt, uint storage, int LM,
        int codedBands, ref uint seed, int complexity, int disableInv, int sAt, ArrayView<uint> pvqU)
    {
        int C = yAt >= 0 ? 2 : 1;
        int theta_rdo = yAt >= 0 && dual_stereo == 0 && complexity >= 8 ? 1 : 0;
        int resynth = theta_rdo;
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
            int X = xAt + M * m.EBandsOf(i), Y = yAt >= 0 ? yAt + M * m.EBandsOf(i) : -1;
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
            if (resynth != 0 && (M * m.EBandsOf(i) - N >= M * m.EBandsOf(start) || i == start + 1) && (update_lowband != 0 || lowband_offset == 0))
                lowband_offset = i;
            if (i == start + 1) CeltQuantBandsGpu.SpecialHybridFolding(m, x, normAt, norm2At, start, M, dual_stereo);
            ctx.TfChange = tfRes[tfAt + i];
            // The encoder's own lowband scratch with resynthesis (kept for the last band with theta RDO); without, nothing folds
            int lowband_scratch = resynth != 0 ? sAt + QLowScratch : -1;
            if (last != 0 && theta_rdo == 0) lowband_scratch = -1;
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
                    x_cm |= (uint)x[masks + fold_i * C];
                    y_cm |= (uint)x[masks + fold_i * C + C - 1];
                }
                while (++fold_i < fold_end);
            }
            else x_cm = y_cm = (uint)((1 << B) - 1);   // the LCG folds: all blocks (almost always) non-zero
            if (dual_stereo != 0 && i == intensity)
            {
                // Switch off dual stereo to do intensity.
                dual_stereo = 0;
                if (resynth != 0)
                    for (int j = 0; j < M * m.EBandsOf(i) - norm_offset; j++) x[normAt + j] = HALF32(x[normAt + j] + x[norm2At + j]);
            }
            int low = effective_lowband != -1 ? normAt + effective_lowband : -1;
            int lowOut = last != 0 ? -1 : normAt + M * m.EBandsOf(i) - norm_offset;
            // Theta RDO: the band rounded down, then up; the second try's result stays when it is not worse
            int tries = theta_rdo != 0 && i < intensity ? 2 : 1;
            int w0 = 0, w1 = 0, dist0 = 0;
            uint cm2 = 0;
            var ec_save = ec;
            var ctx_save = ctx;
            var ec_save2 = ec;
            var ctx_save2 = ctx;
            int nstart_bytes = 0, save_bytes = 0;
            uint cmIn = x_cm | y_cm, cmX = 0, cmY = 0;
            for (int t = 0; t < tries; t++)
            {
                if (tries == 2)
                {
                    if (t == 0)
                    {
                        int ex = bandE[bandEAt + i], ey = bandE[bandEAt + i + Nb];
                        int minE = IMIN(ex, ey);
                        // Adjustment to make the weights a bit more conservative.
                        ex += minE / 3;
                        ey += minE / 3;
                        int shift = celt_ilog2(EPSILON + IMAX(ex, ey)) - 14;
                        w0 = S16(VSHR32(ex, shift));
                        w1 = S16(VSHR32(ey, shift));
                        ec_save = ec;
                        ctx_save = ctx;
                        for (int j = 0; j < N; j++)
                        {
                            x[sAt + QXSave + j] = x[X + j];
                            x[sAt + QYSave + j] = x[Y + j];
                        }
                        ctx.ThetaRound = -1;   // Encode and round down.
                    }
                    else
                    {
                        // Save first result.
                        cm2 = cmX;
                        ec_save2 = ec;
                        ctx_save2 = ctx;
                        for (int j = 0; j < N; j++)
                        {
                            x[sAt + QXSave2 + j] = x[X + j];
                            x[sAt + QYSave2 + j] = x[Y + j];
                        }
                        if (last == 0)
                            for (int j = 0; j < N; j++) x[sAt + QNormSave2 + j] = x[lowOut + j];
                        nstart_bytes = (int)ec_save.Offs;
                        save_bytes = (int)storage - nstart_bytes;
                        for (int j = 0; j < save_bytes; j++) x[sAt + QBytes + j] = buf[bufAt + nstart_bytes + j];
                        // Restore
                        ec = ec_save;
                        ctx = ctx_save;
                        for (int j = 0; j < N; j++)
                        {
                            x[X + j] = x[sAt + QXSave + j];
                            x[Y + j] = x[sAt + QYSave + j];
                        }
                        if (i == start + 1) CeltQuantBandsGpu.SpecialHybridFolding(m, x, normAt, norm2At, start, M, 0);
                        ctx.ThetaRound = 1;   // Encode and round up.
                    }
                }
                else ctx.ThetaRound = 0;

                // The band: mono, dual stereo or stereo (quant_band_stereo), its quant_band calls as jobs through one call site
                int bandMode;   // 0 mono, 1 dual stereo, 2 stereo N == 2, 3 stereo split, 4 stereo N == 1 (no job)
                int njobs;
                int sFill = (int)cmIn, sInv = 0, sImid = 0, sIside = 0, sItheta = 0, mbits = 0, sbits = 0, rebalance = 0;
                int n2Sign = 1, n2X = 0, n2Y = 0;
                cmX = 0;
                cmY = 0;
                if (dual_stereo != 0) { bandMode = 1; njobs = 2; }
                else if (Y < 0) { bandMode = 0; njobs = 1; }
                else if (N == 1)
                {
                    bandMode = 4; njobs = 0;
                    cmX = QuantBandN1(ref ctx, resynth, ref ec, buf, bufAt, storage, x, X, Y, lowOut);
                }
                else
                {
                    int sb = b;
                    int orig_fill = sFill;
                    CeltQuantBandsEncGpu.ComputeThetaEncode(m, ref ctx, ref ec, buf, bufAt, storage, x, X, Y, bandE, bandEAt, N, ref sb, B, B, LM,
                        1, ref sFill, out sInv, out sImid, out sIside, out int sDelta, out sItheta, out int sQalloc);
                    if (N == 2)
                    {
                        // N = 2 codes the side with one sign bit (mid and side are orthogonal)
                        bandMode = 2; njobs = 1;
                        sbits = sItheta != 0 && sItheta != 16384 ? 1 << BITRES : 0;
                        mbits = sb - sbits;
                        ctx.RemainingBits -= sQalloc + sbits;
                        bool c = sItheta > 8192;
                        n2X = c ? Y : X;   // libopus x2 / y2
                        n2Y = c ? X : Y;
                        int sign = 0;
                        if (sbits != 0)
                        {
                            sign = x[n2X] * x[n2Y + 1] - x[n2X + 1] * x[n2Y] < 0 ? 1 : 0;
                            OpusRangeEncoderGpu.EncodeBits(ref ec, buf, bufAt, storage, (uint)sign, 1);
                        }
                        n2Sign = 1 - 2 * sign;
                        // orig_fill: we want to fold the side, but itheta == 16384 cleared the low bits of fill
                        sFill = orig_fill;
                    }
                    else
                    {
                        bandMode = 3; njobs = 2;
                        mbits = IMAX(0, IMIN(sb, (sb - sDelta) / 2));
                        sbits = sb - mbits;
                        ctx.RemainingBits -= sQalloc;
                        rebalance = ctx.RemainingBits;
                    }
                }
                for (int job = 0; job < njobs; job++)
                {
                    int jX, jB, jLow, jLowOut, jGain, jScr, jFill;
                    if (bandMode == 0)
                    {
                        jX = X; jB = b; jLow = low; jLowOut = lowOut; jGain = Q15ONE; jScr = lowband_scratch; jFill = sFill;
                    }
                    else if (bandMode == 1)
                    {
                        jB = b / 2; jGain = Q15ONE; jScr = lowband_scratch;
                        if (job == 0) { jX = X; jLow = low; jLowOut = lowOut; jFill = (int)x_cm; }
                        else
                        {
                            jX = Y;
                            jLow = effective_lowband != -1 ? norm2At + effective_lowband : -1;
                            jLowOut = last != 0 ? -1 : norm2At + M * m.EBandsOf(i) - norm_offset;
                            jFill = (int)y_cm;
                        }
                    }
                    else if (bandMode == 2)
                    {
                        jX = n2X; jB = mbits; jLow = low; jLowOut = lowOut; jGain = Q15ONE; jScr = lowband_scratch; jFill = sFill;
                    }
                    else
                    {
                        bool midFirst = mbits >= sbits;
                        if (job == 1)
                        {
                            // Rebalance what the first half did not use into the second.
                            int used = (midFirst ? mbits : sbits) - (rebalance - ctx.RemainingBits);
                            if (used > 3 << BITRES && sItheta != (midFirst ? 0 : 16384))
                            {
                                if (midFirst) sbits += used - (3 << BITRES);
                                else mbits += used - (3 << BITRES);
                            }
                        }
                        if ((job == 0) == midFirst)
                        {
                            // The mid is not scaled: its normalized version folds later
                            jX = X; jB = mbits; jLow = low; jLowOut = lowOut; jGain = Q15ONE; jScr = lowband_scratch; jFill = sFill;
                        }
                        else
                        {
                            // For a stereo split, the high bits of fill are always zero, so no folding will be done to the side.
                            jX = Y; jB = sbits; jLow = -1; jLowOut = -1; jGain = sIside; jScr = -1; jFill = sFill >> B;
                        }
                    }
                    uint r = QuantBand(m, ref ctx, resynth, ref ec, buf, bufAt, storage, x, jX, N, jB, B, jLow, LM, jLowOut, jGain, jScr, jFill,
                        sAt, pvqU);
                    if (bandMode == 1 && job == 1) cmY = r;
                    else cmX |= r;
                }
                if (bandMode == 2)
                {
                    // The side of a 2-bin band from the mid's bins and the sign, then the mid / side mix
                    x[n2Y] = S16(-n2Sign * x[n2X + 1]);
                    x[n2Y + 1] = S16(n2Sign * x[n2X]);
                    if (resynth != 0)
                    {
                        x[X] = MULT16_16_Q15(sImid, x[X]);
                        x[X + 1] = MULT16_16_Q15(sImid, x[X + 1]);
                        x[Y] = MULT16_16_Q15(sIside, x[Y]);
                        x[Y + 1] = MULT16_16_Q15(sIside, x[Y + 1]);
                        int tmp = x[X];
                        x[X] = S16(SUB16(tmp, x[Y]));
                        x[Y] = ADD16(tmp, x[Y]);
                        tmp = x[X + 1];
                        x[X + 1] = S16(SUB16(tmp, x[Y + 1]));
                        x[Y + 1] = ADD16(tmp, x[Y + 1]);
                    }
                }
                if (resynth != 0)
                {
                    if (bandMode == 3) CeltBandsGpu.StereoMerge(x, X, x, Y, sImid, N);
                    if ((bandMode == 2 || bandMode == 3) && sInv != 0)
                        for (int j = 0; j < N; j++) x[Y + j] = S16(-x[Y + j]);
                }
                if (tries == 2 && t == 0)
                {
                    int xs = 0, ys = 0;
                    for (int j = 0; j < N; j++)
                    {
                        xs = MAC16_16(xs, x[sAt + QXSave + j], x[X + j]);
                        ys = MAC16_16(ys, x[sAt + QYSave + j], x[Y + j]);
                    }
                    dist0 = MULT16_32_Q15(w0, xs) + MULT16_32_Q15(w1, ys);
                }
            }
            if (tries == 2)
            {
                int xs = 0, ys = 0;
                for (int j = 0; j < N; j++)
                {
                    xs = MAC16_16(xs, x[sAt + QXSave + j], x[X + j]);
                    ys = MAC16_16(ys, x[sAt + QYSave + j], x[Y + j]);
                }
                int dist1 = MULT16_32_Q15(w0, xs) + MULT16_32_Q15(w1, ys);
                if (dist0 >= dist1)
                {
                    cmX = cm2;
                    ec = ec_save2;
                    ctx = ctx_save2;
                    for (int j = 0; j < N; j++)
                    {
                        x[X + j] = x[sAt + QXSave2 + j];
                        x[Y + j] = x[sAt + QYSave2 + j];
                    }
                    if (last == 0)
                        for (int j = 0; j < N; j++) x[lowOut + j] = x[sAt + QNormSave2 + j];
                    for (int j = 0; j < save_bytes; j++) buf[bufAt + nstart_bytes + j] = (byte)x[sAt + QBytes + j];
                }
            }
            x_cm = cmX;
            y_cm = dual_stereo != 0 ? cmY : cmX;
            x[masks + i * C] = (int)(x_cm & 0xFF);
            x[masks + i * C + C - 1] = (int)(y_cm & 0xFF);
            balance += pulses[pulsesAt + i] + tell;
            // Update the folding position only as long as we have 1 bit/sample depth.
            update_lowband = b > N << BITRES ? 1 : 0;
            // We only need to avoid noise on a split for the first band. After that, we have folding.
            ctx.AvoidSplitNoise = 0;
        }
        seed = ctx.Seed;
    }
}
