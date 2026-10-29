// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT band energy decoding on the accelerator (libopus celt/quant_bands.c unquant_coarse_energy / unquant_fine_energy /
// unquant_energy_finalise and celt/laplace.c ec_laplace_decode), fixed-point build. Ported statement for statement from
// Concentus 2.2.2 (BSD-3, libopus ~1.1); as in libopus 1.5.2 each band energy store wraps to 16 bits (oldEBands is opus_val16),
// which only a corrupt stream can tell from Concentus' 32-bit ints.
//
// GPU shape: oldEBands is a view at an offset, channel c at + c * NbEBands; fine_quant / fine_priority are per-band views at an
// offset; the coarse-energy Laplace model comes from CeltModeGpu (EProbModel).

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT band energy decode (quant_bands.c + laplace.c), GPU-callable.</summary>
public static class CeltEnergyGpu
{
    private static int PredCoef(int lm) => lm == 0 ? 29440 : lm == 1 ? 26112 : lm == 2 ? 21248 : 16384;
    private static int BetaCoef(int lm) => lm == 0 ? 30147 : lm == 1 ? 22282 : lm == 2 ? 12124 : 6554;
    private const int BetaIntra = 4915;

    private static uint LaplaceGetFreq1(uint fs0, int decay)
    {
        return (uint)((32736 - fs0) * (16384 - decay) >> 15);
    }

    /// <summary>A Laplace-distributed integer with probability of 0 <paramref name="fs"/> / 32768 and the given decay (libopus ec_laplace_decode).</summary>
    public static int LaplaceDecode(ref OpusRangeDecoderGpuState dec, ArrayView<byte> pkt, int pktAt, uint storage, uint fs, int decay)
    {
        int num = 0;
        uint num2 = CeltEntropyGpu.DecodeBin(ref dec, 15);
        uint num3 = 0u;
        if (num2 >= fs)
        {
            num++;
            num3 = fs;
            fs = LaplaceGetFreq1(fs, decay) + 1;
            while (fs > 1 && num2 >= num3 + 2 * fs)
            {
                fs *= 2;
                num3 += fs;
                fs = (uint)((fs - 2) * decay >> 15);
                fs++;
                num++;
            }
            if (fs <= 1)
            {
                int num4 = (int)(num2 - num3) >> 1;
                num += num4;
                num3 += (uint)(2 * num4);
            }
            if (num2 < num3 + fs)
            {
                num = -num;
            }
            else
            {
                num3 += fs;
            }
        }
        CeltEntropyGpu.DecUpdate(ref dec, pkt, pktAt, storage, num3, IMIN(num3 + fs, 32768u), 32768u);
        return num;
    }

    /// <summary>ec_dec_icdf over libopus' small_energy_icdf {2, 1, 0} with ftb 2.</summary>
    private static int DecodeSmallEnergyIcdf(ref OpusRangeDecoderGpuState dec, ArrayView<byte> pkt, int pktAt, uint storage)
    {
        uint num = dec.Rng;
        uint num2 = dec.Val;
        uint num3 = num >> 2;
        int num4 = -1;
        uint num5;
        do
        {
            num5 = num;
            ++num4;
            num = num3 * (uint)(num4 == 0 ? 2 : num4 == 1 ? 1 : 0);
        }
        while (num2 < num);
        dec.Val = num2 - num;
        dec.Rng = num5 - num;
        OpusRangeDecoderGpu.Normalize(ref dec, pkt, pktAt, storage);
        return num4;
    }

    /// <summary>
    /// Coarse band energies (libopus unquant_coarse_energy): predicted from the previous frame (inter) or not (intra),
    /// residuals Laplace-coded. <paramref name="oldEBands"/> holds channel c at <paramref name="eAt"/> + c * NbEBands (Q8 log2 in
    /// libopus' DB_SHIFT units), updated in place.
    /// </summary>
    public static void UnquantCoarseEnergy(CeltModeGpu m, int start, int end, ArrayView<int> oldEBands, int eAt, int intra,
        ref OpusRangeDecoderGpuState dec, ArrayView<byte> pkt, int pktAt, uint storage, int C, int LM)
    {
        int nb = CeltModeTables.NbEBands;
        int probAt = LM * 84 + intra * 42;
        int prev0 = 0, prev1 = 0;   // libopus' prev[2]
        int a;
        int a2;
        if (intra != 0)
        {
            a = 0;
            a2 = BetaIntra;
        }
        else
        {
            a2 = BetaCoef(LM);
            a = PredCoef(LM);
        }
        int num = (int)(storage * 8);
        for (int i = start; i < end; i++)
        {
            int num2 = 0;
            do
            {
                int num3 = CeltEntropyGpu.Tell(ref dec);
                int a3;
                if (num - num3 >= 15)
                {
                    int num4 = 2 * IMIN(i, 20);
                    a3 = LaplaceDecode(ref dec, pkt, pktAt, storage, (uint)(m.EProbModelOf(probAt + num4) << 7), m.EProbModelOf(probAt + num4 + 1) << 6);
                }
                else if (num - num3 < 2)
                {
                    a3 = ((num - num3 < 1) ? (-1) : (-OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, pktAt, storage, 1)));
                }
                else
                {
                    a3 = DecodeSmallEnergyIcdf(ref dec, pkt, pktAt, storage);
                    a3 = (a3 >> 1) ^ -(a3 & 1);
                }
                int a4 = SHL32(a3, 10);
                int e = eAt + i + num2 * nb;
                oldEBands[e] = MAX16(-9216, oldEBands[e]);
                int prev = num2 == 0 ? prev0 : prev1;
                int b = PSHR32(MULT16_16(a, oldEBands[e]), 8) + prev + SHL32(a4, 7);
                b = MAX32(-3670016, b);
                oldEBands[e] = (short)PSHR32(b, 7);   // libopus stores oldEBands as opus_val16
                int next = prev + SHL32(a4, 7) - MULT16_16(a2, PSHR32(a4, 8));
                if (num2 == 0) prev0 = next; else prev1 = next;
            }
            while (++num2 < C);
        }
    }

    /// <summary>Fine band energies (libopus unquant_fine_energy): <paramref name="fineQuant"/> bits per band and channel.</summary>
    public static void UnquantFineEnergy(int start, int end, ArrayView<int> oldEBands, int eAt, ArrayView<int> fineQuant, int fqAt,
        ref OpusRangeDecoderGpuState dec, ArrayView<byte> pkt, int pktAt, uint storage, int C)
    {
        int nb = CeltModeTables.NbEBands;
        for (int i = start; i < end; i++)
        {
            if (fineQuant[fqAt + i] > 0)
            {
                int num = 0;
                do
                {
                    int num2 = SUB16(SHR32(SHL32((int)OpusRangeDecoderGpu.DecodeBits(ref dec, pkt, pktAt, storage, fineQuant[fqAt + i]), 10) + 512, fineQuant[fqAt + i]), 512);
                    oldEBands[eAt + i + num * nb] = (short)(oldEBands[eAt + i + num * nb] + num2);
                }
                while (++num < C);
            }
        }
    }

    /// <summary>The last fine-energy bits, by priority, with what is left of the frame (libopus unquant_energy_finalise).</summary>
    public static void UnquantEnergyFinalise(int start, int end, ArrayView<int> oldEBands, int eAt, ArrayView<int> fineQuant, int fqAt,
        ArrayView<int> finePriority, int fpAt, int bits_left,
        ref OpusRangeDecoderGpuState dec, ArrayView<byte> pkt, int pktAt, uint storage, int C)
    {
        int nb = CeltModeTables.NbEBands;
        for (int i = 0; i < 2; i++)
        {
            for (int j = start; j < end; j++)
            {
                if (bits_left < C)
                {
                    break;
                }
                if (fineQuant[fqAt + j] < 8 && finePriority[fpAt + j] == i)
                {
                    int num = 0;
                    do
                    {
                        int num2 = SHR16(SHL16((int)OpusRangeDecoderGpu.DecodeBits(ref dec, pkt, pktAt, storage, 1), 10) - 512, fineQuant[fqAt + j] + 1);
                        oldEBands[eAt + j + num * nb] = (short)(oldEBands[eAt + j + num * nb] + num2);
                        bits_left--;
                    }
                    while (++num < C);
                }
            }
        }
    }
}
