// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT band energy encoding on the accelerator, ported from libopus 1.5.2 (fixed point; celt/quant_bands.c
// quant_coarse_energy (+ _impl, loss_distortion), quant_fine_energy, quant_energy_finalise; celt/laplace.c ec_laplace_encode)
// with 1.5.2's C arithmetic (CeltOps) on OpusRangeEncoderGpu. quant_coarse_energy's two-pass mode codes the frame intra,
// snapshots the encoder and the bytes it wrote, codes it again inter from the starting state, and keeps the cheaper: the
// snapshot is the encoder state struct plus those bytes copied to caller scratch.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT band energy encoding (libopus 1.5.2, GPU-callable).</summary>
public static class CeltEnergyEncGpu
{
    private const int Nb = CeltModeTables.NbEBands;
    private const int LaplaceMinP = 1, LaplaceNMin = 16;

    private static int PredCoef(int lm) => lm == 0 ? 29440 : lm == 1 ? 26112 : lm == 2 ? 21248 : 16384;
    private static int BetaCoef(int lm) => lm == 0 ? 30147 : lm == 1 ? 22282 : lm == 2 ? 12124 : 6554;
    private const int BetaIntra = 4915;

    /// <summary>libopus ec_laplace_encode: <paramref name="value"/> with a Laplace-like distribution (P(0) = fs / 32768,
    /// decay Q14); a value past the representable range comes back clamped.</summary>
    public static void LaplaceEncode(ref OpusRangeEncoderGpuState enc, ArrayView<byte> buf, int bufStart, uint storage, ref int value, uint fs, int decay)
    {
        uint fl = 0;
        int val = value;
        if (val != 0)
        {
            int s = -(val < 0 ? 1 : 0);
            val = (val + s) ^ s;
            fl = fs;
            // ec_laplace_get_freq1
            fs = (uint)(32768 - LaplaceMinP * (2 * LaplaceNMin) - (int)fs) * (uint)(16384 - decay) >> 15;
            // Search the decaying part of the PDF.
            int i;
            for (i = 1; fs > 0 && i < val; i++)
            {
                fs *= 2;
                fl += fs + 2 * LaplaceMinP;
                fs = (fs * (uint)decay) >> 15;
            }
            // Everything beyond that has probability LAPLACE_MINP.
            if (fs == 0)
            {
                int ndi_max = (int)(32768 - fl + LaplaceMinP - 1);
                ndi_max = (ndi_max - s) >> 1;
                int di = IMIN(val - i, ndi_max - 1);
                fl += (uint)((2 * di + 1 + s) * LaplaceMinP);
                fs = (uint)IMIN(LaplaceMinP, (int)(32768 - fl));
                value = (i + di + s) ^ s;
            }
            else
            {
                fs += LaplaceMinP;
                fl += fs & (uint)~s;
            }
        }
        OpusRangeEncoderGpu.EncodeBin(ref enc, buf, bufStart, storage, fl, fl + fs, 15);
    }

    /// <summary>ec_enc_icdf over a <see cref="CeltModeGpu"/> table: the same arithmetic as ec_encode_bin with fl / fh from it.</summary>
    public static void EncodeIcdfMode(ref OpusRangeEncoderGpuState enc, ArrayView<byte> buf, int bufStart, uint storage, int s,
        CeltModeGpu m, int icdfAt, int ftb)
    {
        uint ft = 1u << ftb;
        uint fl = s > 0 ? ft - (uint)m.Data[icdfAt + s - 1] : 0u;
        uint fh = ft - (uint)m.Data[icdfAt + s];
        OpusRangeEncoderGpu.EncodeBin(ref enc, buf, bufStart, storage, fl, fh, ftb);
    }

    // libopus loss_distortion
    private static int LossDistortion(ArrayView<int> eBands, int eAt, ArrayView<int> oldEBands, int oldAt, int start, int end, int C)
    {
        int dist = 0;
        for (int c = 0; c < C; c++)
            for (int i = start; i < end; i++)
            {
                int d = SUB16(SHR16(eBands[eAt + i + c * Nb], 3), SHR16(oldEBands[oldAt + i + c * Nb], 3));
                dist = MAC16_16(dist, d, d);
            }
        return IMIN(200, dist >> (2 * DB_SHIFT - 6));
    }

    // libopus quant_coarse_energy_impl: one coding pass (intra or inter); returns the badness.
    private static int QuantCoarseEnergyImpl(CeltModeGpu m, int start, int end, ArrayView<int> eBands, int eAt, ArrayView<int> oldEBands,
        int oldAt, int budget, int tell, int probAt, ArrayView<int> error, int errAt, ref OpusRangeEncoderGpuState enc, ArrayView<byte> buf,
        int bufStart, uint storage, int C, int LM, int intra, int max_decay, int lfe)
    {
        int badness = 0;
        int prev0 = 0, prev1 = 0;
        int coef, beta;
        if (tell + 3 <= budget)
            OpusRangeEncoderGpu.EncodeBitLogP(ref enc, buf, bufStart, storage, intra, 3);
        if (intra != 0)
        {
            coef = 0;
            beta = BetaIntra;
        }
        else
        {
            beta = BetaCoef(LM);
            coef = PredCoef(LM);
        }
        // Encode at a fixed coarse resolution
        for (int i = start; i < end; i++)
        {
            for (int c = 0; c < C; c++)
            {
                int e = i + c * Nb;
                int x = eBands[eAt + e];
                int oldE = IMAX(-9216, oldEBands[oldAt + e]);   // -QCONST16(9.f, DB_SHIFT)
                int prev = c == 0 ? prev0 : prev1;
                int f = SHL32(x, 7) - PSHR32(MULT16_16(coef, oldE), 8) - prev;
                // Rounding to nearest integer here is really important!
                int qi = (f + 65536) >> (DB_SHIFT + 7);         // QCONST32(.5f, DB_SHIFT + 7)
                int decay_bound = S16(IMAX(-28672, oldEBands[oldAt + e] - max_decay));
                // Prevent the energy from going down too quickly (e.g. for bands that have just one bin)
                if (qi < 0 && x < decay_bound)
                {
                    qi += SHR16(SUB16(decay_bound, x), DB_SHIFT);
                    if (qi > 0) qi = 0;
                }
                int qi0 = qi;
                // If we don't have enough bits to encode all the energy, just assume something safe.
                tell = OpusRangeEncoderGpu.Tell(ref enc);
                int bits_left = budget - tell - 3 * C * (end - i);
                if (i != start && bits_left < 30)
                {
                    if (bits_left < 24) qi = IMIN(1, qi);
                    if (bits_left < 16) qi = IMAX(-1, qi);
                }
                if (lfe != 0 && i >= 2) qi = IMIN(qi, 0);
                if (budget - tell >= 15)
                {
                    int pi = 2 * IMIN(i, 20);
                    LaplaceEncode(ref enc, buf, bufStart, storage, ref qi, (uint)(m.EProbModelOf(probAt + pi) << 7), m.EProbModelOf(probAt + pi + 1) << 6);
                }
                else if (budget - tell >= 2)
                {
                    qi = IMAX(-1, IMIN(qi, 1));
                    // ec_enc_icdf(enc, 2*qi^-(qi<0), small_energy_icdf = {2, 1, 0}, 2)
                    int sym = (2 * qi) ^ -(qi < 0 ? 1 : 0);
                    OpusRangeEncoderGpu.EncodeBin(ref enc, buf, bufStart, storage, sym > 0 ? (uint)(4 - (sym == 1 ? 2 : 1)) : 0u,
                        (uint)(4 - (sym == 0 ? 2 : sym == 1 ? 1 : 0)), 2);
                }
                else if (budget - tell >= 1)
                {
                    qi = IMIN(0, qi);
                    OpusRangeEncoderGpu.EncodeBitLogP(ref enc, buf, bufStart, storage, -qi, 1);
                }
                else
                    qi = -1;
                error[errAt + e] = S16(PSHR32(f, 7) - SHL16(qi, DB_SHIFT));
                badness += ABS32(qi0 - qi);
                int q = SHL32(qi, DB_SHIFT);
                int tmp = PSHR32(MULT16_16(coef, oldE), 8) + prev + SHL32(q, 7);
                tmp = IMAX(-3670016, tmp);                       // -QCONST32(28.f, DB_SHIFT + 7)
                oldEBands[oldAt + e] = S16(PSHR32(tmp, 7));
                int next = prev + SHL32(q, 7) - MULT16_16(beta, PSHR32(q, 8));
                if (c == 0) prev0 = next; else prev1 = next;
            }
        }
        return lfe != 0 ? 0 : badness;
    }

    /// <summary>
    /// libopus quant_coarse_energy: the coarse band energies (log2, DB_SHIFT, at <paramref name="eAt"/>; channel c at + c *
    /// NbEBands), intra or predicted from <paramref name="oldEBands"/> (updated), the quantisation error to
    /// <paramref name="error"/>. <paramref name="delayedIntra"/> is the encoder's running intra distortion (in / out).
    /// <paramref name="scratch"/> holds 4 * NbEBands + 1275 ints at <paramref name="scratchAt"/> (the intra pass's energies,
    /// errors and coded bytes).
    /// </summary>
    public static void QuantCoarseEnergy(CeltModeGpu m, int start, int end, int effEnd, ArrayView<int> eBands, int eAt, ArrayView<int> oldEBands,
        int oldAt, int budget, ArrayView<int> error, int errAt, ref OpusRangeEncoderGpuState enc, ArrayView<byte> buf, int bufStart, uint storage,
        int C, int LM, int nbAvailableBytes, int force_intra, ref int delayedIntra, int two_pass, int loss_rate, int lfe,
        ArrayView<int> scratch, int scratchAt)
    {
        int oldIntraAt = scratchAt, errIntraAt = scratchAt + 2 * Nb, bitsAt = scratchAt + 4 * Nb;
        int intra = force_intra != 0 || (two_pass == 0 && delayedIntra > 2 * C * (end - start) && nbAvailableBytes > (end - start) * C) ? 1 : 0;
        // budget is opus_uint32 in libopus, so the product and the division are unsigned
        int intra_bias = (int)((uint)budget * (uint)delayedIntra * (uint)loss_rate / (uint)(C * 512));
        int new_distortion = LossDistortion(eBands, eAt, oldEBands, oldAt, start, effEnd, C);
        int tell = OpusRangeEncoderGpu.Tell(ref enc);
        if (tell + 3 > budget) two_pass = intra = 0;
        int max_decay = 16384;   // QCONST16(16.f, DB_SHIFT)
        if (end - start > 10) max_decay = IMIN(max_decay, SHL32(nbAvailableBytes, DB_SHIFT - 3));
        if (lfe != 0) max_decay = 3072;
        var enc_start_state = enc;
        for (int k = 0; k < C * Nb; k++) scratch[oldIntraAt + k] = oldEBands[oldAt + k];
        int badness1 = 0;
        if (two_pass != 0 || intra != 0)
            badness1 = QuantCoarseEnergyImpl(m, start, end, eBands, eAt, scratch, oldIntraAt, budget, tell, LM * 84 + 42, scratch, errIntraAt,
                ref enc, buf, bufStart, storage, C, LM, 1, max_decay, lfe);
        if (intra == 0)
        {
            int tell_intra = (int)OpusRangeEncoderGpu.TellFrac(ref enc);
            var enc_intra_state = enc;
            int nstart_bytes = (int)enc_start_state.Offs;
            int nintra_bytes = (int)enc_intra_state.Offs;
            // Copy bits from intra bit-stream
            for (int k = 0; k < nintra_bytes - nstart_bytes; k++) scratch[bitsAt + k] = buf[bufStart + nstart_bytes + k];
            enc = enc_start_state;
            int badness2 = QuantCoarseEnergyImpl(m, start, end, eBands, eAt, oldEBands, oldAt, budget, tell, LM * 84 + 42 * intra, error, errAt,
                ref enc, buf, bufStart, storage, C, LM, 0, max_decay, lfe);
            if (two_pass != 0 && (badness1 < badness2 || (badness1 == badness2 && (int)OpusRangeEncoderGpu.TellFrac(ref enc) + intra_bias > tell_intra)))
            {
                enc = enc_intra_state;
                // Copy intra bits to bit-stream
                for (int k = 0; k < nintra_bytes - nstart_bytes; k++) buf[bufStart + nstart_bytes + k] = (byte)scratch[bitsAt + k];
                for (int k = 0; k < C * Nb; k++)
                {
                    oldEBands[oldAt + k] = scratch[oldIntraAt + k];
                    error[errAt + k] = scratch[errIntraAt + k];
                }
                intra = 1;
            }
        }
        else
        {
            for (int k = 0; k < C * Nb; k++)
            {
                oldEBands[oldAt + k] = scratch[oldIntraAt + k];
                error[errAt + k] = scratch[errIntraAt + k];
            }
        }
        if (intra != 0)
            delayedIntra = new_distortion;
        else
            delayedIntra = MULT16_32_Q15(MULT16_16_Q15(PredCoef(LM), PredCoef(LM)), delayedIntra) + new_distortion;
    }

    /// <summary>libopus quant_fine_energy: fine_quant[i] more bits of each band's energy.</summary>
    public static void QuantFineEnergy(int start, int end, ArrayView<int> oldEBands, int oldAt, ArrayView<int> error, int errAt,
        ArrayView<int> fine_quant, int fqAt, ref OpusRangeEncoderGpuState enc, ArrayView<byte> buf, int bufStart, uint storage, int C)
    {
        for (int i = start; i < end; i++)
        {
            int fq = fine_quant[fqAt + i];
            int frac = (short)(1 << fq);
            if (fq <= 0) continue;
            for (int c = 0; c < C; c++)
            {
                int e = i + c * Nb;
                // Has to be without rounding
                int q2 = (error[errAt + e] + 512) >> (DB_SHIFT - fq);
                if (q2 > frac - 1) q2 = frac - 1;
                if (q2 < 0) q2 = 0;
                OpusRangeEncoderGpu.EncodeBits(ref enc, buf, bufStart, storage, (uint)q2, fq);
                int offset = S16(SUB16((SHL32(q2, DB_SHIFT) + 512) >> fq, 512));
                oldEBands[oldAt + e] = S16(oldEBands[oldAt + e] + offset);
                error[errAt + e] = S16(error[errAt + e] - offset);
            }
        }
    }

    /// <summary>libopus quant_energy_finalise: the remaining bits, one per band and channel, by priority.</summary>
    public static void QuantEnergyFinalise(int start, int end, ArrayView<int> oldEBands, int oldAt, ArrayView<int> error, int errAt,
        ArrayView<int> fine_quant, int fqAt, ArrayView<int> fine_priority, int fpAt, int bits_left, ref OpusRangeEncoderGpuState enc,
        ArrayView<byte> buf, int bufStart, uint storage, int C)
    {
        // Use up the remaining bits
        for (int prio = 0; prio < 2; prio++)
        {
            for (int i = start; i < end && bits_left >= C; i++)
            {
                if (fine_quant[fqAt + i] >= 8 || fine_priority[fpAt + i] != prio)   // MAX_FINE_BITS
                    continue;
                for (int c = 0; c < C; c++)
                {
                    int e = i + c * Nb;
                    int q2 = error[errAt + e] < 0 ? 0 : 1;
                    OpusRangeEncoderGpu.EncodeBits(ref enc, buf, bufStart, storage, (uint)q2, 1);
                    int offset = S16(SHR16(SHL16(q2, DB_SHIFT) - 512, fine_quant[fqAt + i] + 1));
                    oldEBands[oldAt + e] = S16(oldEBands[oldAt + e] + offset);
                    error[errAt + e] = S16(error[errAt + e] - offset);
                    bits_left--;
                }
            }
        }
    }
}
