// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable composite FLAC FIXED subframe decoder. Mirror of the
// FIXED branch inside FlacSubframeDecoder.Decode (RFC 9639 Section
// 8.1.3 fixed predictor). Composes 3 already-shipped GPU primitives in
// a single kernel thread:
//   1. Bit-read order warm-up samples at the effective bit depth.
//   2. FlacResidualDecoderGpu.DecodeAt for Rice-coded residuals.
//   3. FlacFixedReconstructGpu.ReconstructAt to rebuild the signal.
//   4. Optional left-shift by the wasted-bits count.
//
// Sequential per-stream because every stage shares the same bit reader
// state. Single-thread per stream; multiple FLAC channels parallelize
// across threads.
//
// Caller pre-parses the subframe header (kind/order/wastedBits) via
// FlacSubframeHeaderGpu and calls this primitive only when kind == FIXED.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable composite FLAC FIXED subframe decoder. Mirror of the
/// FIXED branch in <see cref="FlacSubframeDecoder"/>.Decode.
/// </summary>
public static class FlacFixedSubframeDecodeGpu
{
    /// <summary>
    /// Decode one FIXED subframe in place. Bit-exact vs the CPU
    /// FlacSubframeDecoder.Decode FIXED branch.
    /// </summary>
    /// <param name="state">Bit reader state, positioned at the warm-up samples.</param>
    /// <param name="data">Underlying byte buffer.</param>
    /// <param name="samples">Output PCM (length &gt;= blockSize).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="blockSize">FLAC frame block size.</param>
    /// <param name="order">FIXED predictor order (0..4).</param>
    /// <param name="effectiveBps">Bit depth for warm-up + residual = subframeBps - wastedBits.</param>
    /// <param name="wastedBits">Wasted bits per sample (left-shifted at the end).</param>
    public static void DecodeAt(
        ref FlacBitReaderGpuState state,
        ArrayView<byte> data,
        ArrayView<int> samples, long samplesBase,
        int blockSize, int order, int effectiveBps, int wastedBits)
    {
        // Step 1: warm-up samples.
        for (int i = 0; i < order; i++)
        {
            samples[samplesBase + i] = FlacBitReaderGpu.ReadBitsSigned(ref state, data, effectiveBps);
        }

        // Step 2: residual into [order..blockSize).
        FlacResidualDecoderGpu.DecodeAt(
            ref state, data,
            samples, samplesBase + order,
            blockSize, order);

        // Step 3: FIXED reconstruction.
        FlacFixedReconstructGpu.ReconstructAt(samples, samplesBase, blockSize, order);

        // Step 4: wasted-bits left-shift.
        if (wastedBits > 0)
        {
            for (int i = 0; i < blockSize; i++)
            {
                samples[samplesBase + i] = samples[samplesBase + i] << wastedBits;
            }
        }
    }
}
