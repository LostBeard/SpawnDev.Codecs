// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable composite FLAC VERBATIM subframe decoder. Mirror of the
// VERBATIM branch inside FlacSubframeDecoder.Decode (RFC 9639 Section
// 8.1.2). Reads blockSize signed values at the effective bit depth and
// applies the wasted-bits left-shift if any.
//
// This is the "post-header" composite that pairs with
// FlacSubframeHeaderGpu (header parse) + FlacChannelDecorrelationGpu
// (post-decode stereo conversion). Now all 4 FLAC subframe kinds
// (CONSTANT, VERBATIM, FIXED, LPC) have GPU-callable composite
// post-header decoders.
//
// Sequential per-stream because the bit reader state evolves over
// blockSize sequential reads. Single-thread per stream; multiple FLAC
// channels parallelize across threads.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable composite FLAC VERBATIM subframe decoder. Mirror of the
/// VERBATIM branch in <see cref="FlacSubframeDecoder"/>.Decode.
/// </summary>
public static class FlacVerbatimSubframeDecodeGpu
{
    /// <summary>
    /// Decode one VERBATIM subframe in place. Bit-exact vs the CPU
    /// FlacSubframeDecoder.Decode VERBATIM branch.
    /// </summary>
    /// <param name="state">Bit reader state, positioned at the first sample.</param>
    /// <param name="data">Underlying byte buffer.</param>
    /// <param name="samples">Output PCM (length &gt;= blockSize).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="blockSize">FLAC frame block size.</param>
    /// <param name="effectiveBps">Bit depth = subframeBps - wastedBits.</param>
    /// <param name="wastedBits">Wasted bits per sample (left-shifted at the end).</param>
    public static void DecodeAt(
        ref FlacBitReaderGpuState state,
        ArrayView<byte> data,
        ArrayView<int> samples, long samplesBase,
        int blockSize, int effectiveBps, int wastedBits)
    {
        // Read blockSize signed values at effectiveBps bits each.
        for (int i = 0; i < blockSize; i++)
        {
            int value = FlacBitReaderGpu.ReadBitsSigned(ref state, data, effectiveBps);
            samples[samplesBase + i] = wastedBits > 0 ? (value << wastedBits) : value;
        }
    }
}
