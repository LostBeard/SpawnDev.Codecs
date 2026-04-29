// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable composite FLAC CONSTANT subframe decoder. Mirror of the
// CONSTANT branch inside FlacSubframeDecoder.Decode (RFC 9639 Section
// 8.1.1 constant predictor). Reads a single signed value at the
// effective bit depth and broadcasts it across blockSize samples; then
// applies the wasted-bits left-shift if any.
//
// Sequential per-stream because the bit reader state evolves over one
// signed read. Single-thread per stream; multiple FLAC channels
// parallelize across threads. The fill loop runs as a serial fan-out
// in the same kernel thread (small for blockSize up to 4096; if needed,
// callers can dispatch a separate per-sample kernel for large blocks).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable composite FLAC CONSTANT subframe decoder. Mirror of the
/// CONSTANT branch in <see cref="FlacSubframeDecoder"/>.Decode.
/// </summary>
public static class FlacConstantSubframeDecodeGpu
{
    /// <summary>
    /// Decode one CONSTANT subframe in place. Bit-exact vs the CPU
    /// FlacSubframeDecoder.Decode CONSTANT branch.
    /// </summary>
    /// <param name="state">Bit reader state, positioned at the value field.</param>
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
        int value = FlacBitReaderGpu.ReadBitsSigned(ref state, data, effectiveBps);
        int shifted = wastedBits > 0 ? (value << wastedBits) : value;

        for (int i = 0; i < blockSize; i++)
        {
            samples[samplesBase + i] = shifted;
        }
    }
}
