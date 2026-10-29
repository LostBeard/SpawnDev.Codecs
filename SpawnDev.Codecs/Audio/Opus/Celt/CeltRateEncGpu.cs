// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT bit allocation on the accelerator, encode side, ported from libopus 1.5.2 (fixed point; celt/rate.c
// clt_compute_allocation and interp_bits2pulses with encode = 1): the same per-band shape / fine-energy split as the decoder
// (CeltRateGpu), with the band skip decision (the one non-normative choice, with the depth-threshold hysteresis against the
// previous frame's coded bands), the intensity band and the dual-stereo flag written to the range encoder.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT bit allocation, encode side (libopus 1.5.2 clt_compute_allocation, GPU-callable).</summary>
public static class CeltRateEncGpu
{
    private const int Nb = CeltModeTables.NbEBands;
    private const int NbAllocVectors = 11, AllocSteps = 6, FineOffset = 21, MaxFineBits = 8;

    /// <summary>
    /// libopus clt_compute_allocation (encode): the per-band shape bits (<paramref name="pulses"/>, 1/8 bit), fine-energy bits
    /// (<paramref name="ebits"/>) and fine priorities for <paramref name="total"/> 1/8 bits, coding the skip, intensity and
    /// dual-stereo decisions. The per-band views share <paramref name="at"/>; <paramref name="scratch"/> holds 4 * NbEBands
    /// ints at <paramref name="scratchAt"/>. <paramref name="intensity"/> comes in as the encoder's choice and goes out as
    /// coded; <paramref name="prev"/> is the previous frame's coded band count. Returns the coded band count.
    /// </summary>
    public static int ComputeAllocationEncode(CeltModeGpu m, int start, int end, ArrayView<int> offsets, ArrayView<int> cap,
        int alloc_trim, ref int intensity, ref int dual_stereo, int total, out int balance, ArrayView<int> pulses,
        ArrayView<int> ebits, ArrayView<int> fine_priority, int at, ArrayView<int> scratch, int scratchAt, int C, int LM,
        ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt, uint storage, int prev, int signalBandwidth)
    {
        int bits1 = scratchAt, bits2 = scratchAt + Nb, thresh = scratchAt + 2 * Nb, trim_offset = scratchAt + 3 * Nb;
        total = IMAX(total, 0);
        int skip_start = start;
        // Reserve a bit to signal the end of manually skipped bands.
        int skip_rsv = total >= 1 << BITRES ? 1 << BITRES : 0;
        total -= skip_rsv;
        // Reserve bits for the intensity and dual stereo parameters.
        int intensity_rsv = 0, dual_stereo_rsv = 0;
        if (C == 2)
        {
            intensity_rsv = m.Log2FracTableOf(end - start);
            if (intensity_rsv > total) intensity_rsv = 0;
            else
            {
                total -= intensity_rsv;
                dual_stereo_rsv = total >= 1 << BITRES ? 1 << BITRES : 0;
                total -= dual_stereo_rsv;
            }
        }
        for (int j = start; j < end; j++)
        {
            int w = m.EBandsOf(j + 1) - m.EBandsOf(j);
            // Below this threshold, we're sure not to allocate any PVQ bits
            scratch[thresh + j] = IMAX(C << BITRES, (3 * w << LM << BITRES) >> 4);
            // Tilt of the allocation curve
            scratch[trim_offset + j] = C * w * (alloc_trim - 5 - LM) * (end - j - 1) * (1 << (LM + BITRES)) >> 6;
            // Less resolution to single-coefficient bands: they benefit more from one coarse value per coefficient
            if (w << LM == 1) scratch[trim_offset + j] -= C << BITRES;
        }
        int lo = 1, hi = NbAllocVectors - 1;
        do
        {
            int done = 0, psum = 0;
            int mid = (lo + hi) >> 1;
            for (int j = end; j-- > start;)
            {
                int N = m.EBandsOf(j + 1) - m.EBandsOf(j);
                int bitsj = C * N * m.AllocVectorsOf(mid * Nb + j) << LM >> 2;
                if (bitsj > 0) bitsj = IMAX(0, bitsj + scratch[trim_offset + j]);
                bitsj += offsets[at + j];
                if (bitsj >= scratch[thresh + j] || done != 0)
                {
                    done = 1;
                    // Don't allocate more than we can actually use
                    psum += IMIN(bitsj, cap[at + j]);
                }
                else if (bitsj >= C << BITRES) psum += C << BITRES;
            }
            if (psum > total) hi = mid - 1;
            else lo = mid + 1;
        }
        while (lo <= hi);
        hi = lo--;
        for (int j = start; j < end; j++)
        {
            int N = m.EBandsOf(j + 1) - m.EBandsOf(j);
            int bits1j = C * N * m.AllocVectorsOf(lo * Nb + j) << LM >> 2;
            int bits2j = hi >= NbAllocVectors ? cap[at + j] : C * N * m.AllocVectorsOf(hi * Nb + j) << LM >> 2;
            if (bits1j > 0) bits1j = IMAX(0, bits1j + scratch[trim_offset + j]);
            if (bits2j > 0) bits2j = IMAX(0, bits2j + scratch[trim_offset + j]);
            if (lo > 0) bits1j += offsets[at + j];
            bits2j += offsets[at + j];
            if (offsets[at + j] > 0) skip_start = j;
            bits2j = IMAX(0, bits2j - bits1j);
            scratch[bits1 + j] = bits1j;
            scratch[bits2 + j] = bits2j;
        }
        return InterpBits2PulsesEncode(m, start, end, skip_start, scratch, bits1, bits2, thresh, cap, total, out balance, skip_rsv,
            ref intensity, intensity_rsv, ref dual_stereo, dual_stereo_rsv, pulses, ebits, fine_priority, at, C, LM, ref ec, buf, bufAt,
            storage, prev, signalBandwidth);
    }

    private static int InterpBits2PulsesEncode(CeltModeGpu m, int start, int end, int skip_start, ArrayView<int> s, int bits1,
        int bits2, int thresh, ArrayView<int> cap, int total, out int _balance, int skip_rsv, ref int intensity, int intensity_rsv,
        ref int dual_stereo, int dual_stereo_rsv, ArrayView<int> bits, ArrayView<int> ebits, ArrayView<int> fine_priority, int at,
        int C, int LM, ref OpusRangeEncoderGpuState ec, ArrayView<byte> buf, int bufAt, uint storage, int prev, int signalBandwidth)
    {
        int alloc_floor = C << BITRES;
        int stereo = C > 1 ? 1 : 0;
        int logM = LM << BITRES;
        int lo = 0, hi = 1 << AllocSteps;
        int psum, done;
        for (int i = 0; i < AllocSteps; i++)
        {
            int mid = (lo + hi) >> 1;
            psum = 0;
            done = 0;
            for (int j = end; j-- > start;)
            {
                int tmp = s[bits1 + j] + (mid * s[bits2 + j] >> AllocSteps);
                if (tmp >= s[thresh + j] || done != 0)
                {
                    done = 1;
                    // Don't allocate more than we can actually use
                    psum += IMIN(tmp, cap[at + j]);
                }
                else if (tmp >= alloc_floor) psum += alloc_floor;
            }
            if (psum > total) hi = mid;
            else lo = mid;
        }
        psum = 0;
        done = 0;
        for (int j = end; j-- > start;)
        {
            int tmp = s[bits1 + j] + (lo * s[bits2 + j] >> AllocSteps);
            if (tmp < s[thresh + j] && done == 0)
            {
                if (tmp >= alloc_floor) tmp = alloc_floor;
                else tmp = 0;
            }
            else done = 1;
            // Don't allocate more than we can actually use
            tmp = IMIN(tmp, cap[at + j]);
            bits[at + j] = tmp;
            psum += tmp;
        }
        // Decide which bands to skip, working backwards from the end.
        int codedBands;
        int left, percoeff;
        for (codedBands = end; ; codedBands--)
        {
            int j = codedBands - 1;
            // Never skip the first band, nor a band that has been boosted by dynalloc.
            if (j <= skip_start)
            {
                // Give the bit we reserved to end skipping back.
                total += skip_rsv;
                break;
            }
            // How many left-over bits we would be adding to this band, including bits stolen back from higher, skipped bands.
            left = total - psum;
            percoeff = left / (m.EBandsOf(codedBands) - m.EBandsOf(start));
            left -= (m.EBandsOf(codedBands) - m.EBandsOf(start)) * percoeff;
            int rem = IMAX(left - (m.EBandsOf(j) - m.EBandsOf(start)), 0);
            int band_width = m.EBandsOf(codedBands) - m.EBandsOf(j);
            int band_bits = bits[at + j] + percoeff * band_width + rem;
            // Only code a skip decision if we're above the threshold for this band; otherwise it is force-skipped.
            if (band_bits >= IMAX(s[thresh + j], alloc_floor + (1 << BITRES)))
            {
                // The only part of the allocation that is not a mandatory part of the bitstream: hysteresis keeps bands from
                // fluctuating in and out, without folding below a certain point.
                int depth_threshold = codedBands > 17 ? (j < prev ? 7 : 9) : 0;
                if (codedBands <= start + 2 || (band_bits > (depth_threshold * band_width << LM << BITRES) >> 4 && j <= signalBandwidth))
                {
                    OpusRangeEncoderGpu.EncodeBitLogP(ref ec, buf, bufAt, storage, 1, 1);
                    break;
                }
                OpusRangeEncoderGpu.EncodeBitLogP(ref ec, buf, bufAt, storage, 0, 1);
                // We used a bit to skip this band.
                psum += 1 << BITRES;
                band_bits -= 1 << BITRES;
            }
            // Reclaim the bits originally allocated to this band.
            psum -= bits[at + j] + intensity_rsv;
            if (intensity_rsv > 0) intensity_rsv = m.Log2FracTableOf(j - start);
            psum += intensity_rsv;
            if (band_bits >= alloc_floor)
            {
                // If we have enough for a fine energy bit per channel, use it.
                psum += alloc_floor;
                bits[at + j] = alloc_floor;
            }
            else bits[at + j] = 0;   // Otherwise this band gets nothing at all.
        }
        // Code the intensity and dual stereo parameters.
        if (intensity_rsv > 0)
        {
            intensity = IMIN(intensity, codedBands);
            OpusRangeEncoderGpu.EncodeUint(ref ec, buf, bufAt, storage, (uint)(intensity - start), (uint)(codedBands + 1 - start));
        }
        else intensity = 0;
        if (intensity <= start)
        {
            total += dual_stereo_rsv;
            dual_stereo_rsv = 0;
        }
        if (dual_stereo_rsv > 0) OpusRangeEncoderGpu.EncodeBitLogP(ref ec, buf, bufAt, storage, dual_stereo, 1);
        else dual_stereo = 0;
        // Allocate the remaining bits
        left = total - psum;
        percoeff = left / (m.EBandsOf(codedBands) - m.EBandsOf(start));
        left -= (m.EBandsOf(codedBands) - m.EBandsOf(start)) * percoeff;
        for (int j = start; j < codedBands; j++) bits[at + j] += percoeff * (m.EBandsOf(j + 1) - m.EBandsOf(j));
        for (int j = start; j < codedBands; j++)
        {
            int tmp = IMIN(left, m.EBandsOf(j + 1) - m.EBandsOf(j));
            bits[at + j] += tmp;
            left -= tmp;
        }
        int balance = 0;
        int jj;
        for (jj = start; jj < codedBands; jj++)
        {
            int j = jj;
            int N0 = m.EBandsOf(j + 1) - m.EBandsOf(j);
            int N = N0 << LM;
            int bit = bits[at + j] + balance;
            int excess;
            if (N > 1)
            {
                excess = IMAX(bit - cap[at + j], 0);
                bits[at + j] = bit - excess;
                // Compensate for the extra DoF in stereo
                int den = C * N + (C == 2 && N > 2 && dual_stereo == 0 && j < intensity ? 1 : 0);
                int NClogN = den * (m.LogNOf(j) + logM);
                // Offset for the number of fine bits by log2(N)/2 + FINE_OFFSET compared to their "fair share" of total/N
                int offset = (NClogN >> 1) - den * FineOffset;
                // N=2 is the only point that doesn't match the curve
                if (N == 2) offset += den << BITRES >> 2;
                // Changing the offset for allocating the second and third fine energy bit
                if (bits[at + j] + offset < den * 2 << BITRES) offset += NClogN >> 2;
                else if (bits[at + j] + offset < den * 3 << BITRES) offset += NClogN >> 3;
                // Divide with rounding
                int e = IMAX(0, bits[at + j] + offset + (den << (BITRES - 1)));
                e = (int)((uint)e / (uint)den) >> BITRES;
                // Make sure not to bust
                if (C * e > (bits[at + j] >> BITRES)) e = bits[at + j] >> stereo >> BITRES;
                // More than that is useless because that's about as far as PVQ can go
                e = IMIN(e, MaxFineBits);
                ebits[at + j] = e;
                // If we rounded down or capped this band, make it a candidate for the final fine energy pass
                fine_priority[at + j] = e * (den << BITRES) >= bits[at + j] + offset ? 1 : 0;
                // Remove the allocated fine bits; the rest are assigned to PVQ
                bits[at + j] -= C * e << BITRES;
            }
            else
            {
                // For N=1, all bits go to fine energy except for a single sign bit
                excess = IMAX(0, bit - (C << BITRES));
                bits[at + j] = bit - excess;
                ebits[at + j] = 0;
                fine_priority[at + j] = 1;
            }
            // Fine energy can't take advantage of the re-balancing in quant_all_bands(); do the re-balancing here.
            if (excess > 0)
            {
                int extra_fine = IMIN(excess >> (stereo + BITRES), MaxFineBits - ebits[at + j]);
                ebits[at + j] += extra_fine;
                int extra_bits = extra_fine * C << BITRES;
                fine_priority[at + j] = extra_bits >= excess - balance ? 1 : 0;
                excess -= extra_bits;
            }
            balance = excess;
        }
        // Save any remaining bits over the cap for the rebalancing in quant_all_bands().
        _balance = balance;
        // The skipped bands use all their bits for fine energy.
        for (; jj < end; jj++)
        {
            ebits[at + jj] = bits[at + jj] >> stereo >> BITRES;
            bits[at + jj] = 0;
            fine_priority[at + jj] = ebits[at + jj] < 1 ? 1 : 0;
        }
        return codedBands;
    }
}
