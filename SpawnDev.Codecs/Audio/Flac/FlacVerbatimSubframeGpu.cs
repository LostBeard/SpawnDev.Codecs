// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC VERBATIM subframe encoder + decoder, GPU-callable form.
// Verbatim is the simplest FLAC subframe type: subframe header bits
// declare type=VERBATIM, then samples are written as bps-bit signed
// integers in MSB-first order.
//
// Subframe header (8 bits):
//   bit 0    : zero (reserved, 0)
//   bits 1-6 : type (000001 for VERBATIM)
//   bit 7    : wasted-bits flag (0 = no wasted bits)
// Then for each sample: signed (bps) bits.
//
// V1 FLAC GPU pipeline uses VERBATIM for every subframe. No prediction,
// no Rice coding - just raw samples wrapped in subframe headers. The
// resulting stream is valid FLAC but uncompressed (~bps bits per
// sample plus 8 bits per subframe header).
//
// Every bit of a run of VERBATIM subframes sits at a position known up front (8 + sampleCount * bps bits per
// subframe), so the subframes are packed and unpacked IN PARALLEL: one thread per output byte (PackByteAt) or per
// sample (UnpackSampleAt). No thread walks a whole subframe, so no launch grows with the block size.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC VERBATIM subframe helpers. Mirror of the
/// VERBATIM branch of FlacSubframeWriter.Emit + FlacSubframeDecoder.
/// </summary>
public static class FlacVerbatimSubframeGpu
{
    /// <summary>The 8-bit VERBATIM subframe header: 0, type 000001, no wasted bits.</summary>
    public const int HeaderByte = 0x02;

    /// <summary>Bits of one VERBATIM subframe of <paramref name="sampleCount"/> samples at <paramref name="bps"/> bits.</summary>
    public static long SubframeBits(int sampleCount, int bps) => 8 + (long)sampleCount * bps;

    /// <summary>Bytes of <paramref name="channels"/> consecutive VERBATIM subframes, zero-padded to a byte.</summary>
    public static long SubframesBytes(int sampleCount, int channels, int bps)
        => (SubframeBits(sampleCount, bps) * channels + 7) >> 3;

    /// <summary>
    /// Byte <paramref name="j"/> of <paramref name="channels"/> consecutive VERBATIM subframes packed MSB-first from
    /// bit 0 and zero-padded to a byte (the bytes FlacBitWriter writes for them). Channel c's sample i is
    /// <c>samples[samplesBase + c * channelStride + i]</c>; each is written as a signed <paramref name="bps"/>-bit value.
    /// </summary>
    public static byte PackByteAt(
        ArrayView<int> samples, long samplesBase, long channelStride,
        int sampleCount, int channels, int bps, long j)
    {
        long subBits = SubframeBits(sampleCount, bps);
        long total = subBits * channels;
        int acc = 0;
        for (int k = 0; k < 8; k++)
        {
            long p = j * 8 + k;
            int bit = 0;
            if (p < total)
            {
                long c = p / subBits;
                long r = p - c * subBits;
                if (r < 8)
                {
                    bit = (HeaderByte >> (7 - (int)r)) & 1;
                }
                else
                {
                    long q = r - 8;
                    long i = q / bps;
                    int b = (int)(q - i * bps);
                    uint raw = (uint)samples[samplesBase + c * channelStride + i];
                    bit = (int)((raw >> (bps - 1 - b)) & 1u);
                }
            }
            acc = (acc << 1) | bit;
        }
        return (byte)acc;
    }

    /// <summary>
    /// Reads <paramref name="nBits"/> (1..32) bits MSB-first starting <paramref name="bitPos"/> bits after byte
    /// <paramref name="byteBase"/> of <paramref name="data"/>.
    /// </summary>
    public static uint ReadBitsAt(ArrayView<byte> data, long byteBase, long bitPos, int nBits)
    {
        uint v = 0;
        for (int k = 0; k < nBits; k++)
        {
            long q = bitPos + k;
            int bv = data[byteBase + (q >> 3)];
            uint bit = (uint)((bv >> (7 - (int)(q & 7))) & 1);
            v = (v << 1) | bit;
        }
        return v;
    }

    /// <summary>
    /// 1 when the 8-bit header of VERBATIM subframe <paramref name="channel"/> (of a run starting at byte
    /// <paramref name="byteBase"/>) declares VERBATIM with no wasted bits, else 0 (a non-VERBATIM type the v1 GPU
    /// decoder does not handle).
    /// </summary>
    public static int HeaderOkAt(ArrayView<byte> data, long byteBase, int sampleCount, int bps, int channel)
    {
        uint h = ReadBitsAt(data, byteBase, channel * SubframeBits(sampleCount, bps), 8);
        return h == HeaderByte ? 1 : 0;
    }

    /// <summary>
    /// Sample <paramref name="i"/> of VERBATIM subframe <paramref name="channel"/> of a run starting at byte
    /// <paramref name="byteBase"/>: <paramref name="bps"/> bits, sign-extended.
    /// </summary>
    public static int UnpackSampleAt(
        ArrayView<byte> data, long byteBase, int sampleCount, int bps, int channel, long i)
    {
        long p = channel * SubframeBits(sampleCount, bps) + 8 + i * bps;
        uint u = ReadBitsAt(data, byteBase, p, bps);
        if (bps < 32)
        {
            uint signMask = 1u << (bps - 1);
            if ((u & signMask) != 0) u |= ~((1u << bps) - 1);
        }
        return unchecked((int)u);
    }
}
