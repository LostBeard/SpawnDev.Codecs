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

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC VERBATIM subframe helpers. Mirror of the
/// VERBATIM branch of FlacSubframeWriter.Emit + FlacSubframeDecoder.
/// </summary>
public static class FlacVerbatimSubframeGpu
{
    /// <summary>
    /// Encode a VERBATIM subframe of <paramref name="sampleCount"/>
    /// samples at <paramref name="bps"/> bits per sample. Reads
    /// samples from <paramref name="samples"/> starting at
    /// <paramref name="samplesBase"/>; writes via the FLAC bit writer
    /// state.
    /// </summary>
    public static void Encode(
        ref FlacBitWriterGpuState w, ArrayView<byte> outBuf,
        ArrayView<int> samples, long samplesBase, int sampleCount,
        int bps)
    {
        // Subframe header: 0 (1 bit) + 0b000001 (6 bits) + 0 (1 bit, no wasted bits) = 8 bits.
        FlacBitWriterGpu.Write(ref w, outBuf, 0u, 1);
        FlacBitWriterGpu.Write(ref w, outBuf, 0b000001u, 6);
        FlacBitWriterGpu.Write(ref w, outBuf, 0u, 1);

        // Samples as signed (bps)-bit values.
        for (int i = 0; i < sampleCount; i++)
        {
            FlacBitWriterGpu.WriteSigned(ref w, outBuf, samples[samplesBase + i], bps);
        }
    }

    /// <summary>
    /// Decode a VERBATIM subframe: parse the 8-bit subframe header
    /// (must indicate type=VERBATIM with no wasted bits) then read
    /// <paramref name="sampleCount"/> samples at <paramref name="bps"/>
    /// bits each. Writes decoded samples to <paramref name="samples"/>
    /// starting at <paramref name="samplesBase"/>.
    /// <para>
    /// Returns 1 on success, 0 if the subframe header indicates a
    /// non-VERBATIM type or wasted bits (which the v1 GPU decoder
    /// doesn't handle yet).
    /// </para>
    /// </summary>
    public static int Decode(
        ref FlacBitReaderGpuState r, ArrayView<byte> data,
        ArrayView<int> samples, long samplesBase, int sampleCount,
        int bps)
    {
        // Subframe header.
        uint zero = FlacBitReaderGpu.ReadBits(ref r, data, 1);
        uint type = FlacBitReaderGpu.ReadBits(ref r, data, 6);
        uint wasted = FlacBitReaderGpu.ReadBits(ref r, data, 1);
        if (zero != 0 || type != 0b000001 || wasted != 0) return 0;

        for (int i = 0; i < sampleCount; i++)
        {
            samples[samplesBase + i] = FlacBitReaderGpu.ReadBitsSigned(ref r, data, bps);
        }
        return 1;
    }
}
