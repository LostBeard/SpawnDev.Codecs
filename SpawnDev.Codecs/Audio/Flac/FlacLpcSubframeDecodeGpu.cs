// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable composite FLAC LPC subframe decoder. Mirror of the LPC
// branch inside FlacSubframeDecoder.Decode (RFC 9639 Section 8.1.4 LPC
// predictor). Composes 4 already-shipped GPU primitives in a single
// kernel thread:
//   1. Bit-read order warm-up samples at the effective bit depth.
//   2. Bit-read 4-bit precision-1 + 5-bit signed quantLevel.
//   3. Bit-read order QLP coefficients at `precision` bits each.
//   4. FlacResidualDecoderGpu.DecodeAt for Rice-coded residuals.
//   5. FlacLpcReconstructGpu.ReconstructAt to rebuild the signal.
//   6. Optional left-shift by the wasted-bits count.
//
// Sequential per-stream because every stage shares the same bit reader
// state. Single-thread per stream; multiple FLAC channels parallelize
// across threads.
//
// Caller pre-parses the subframe header (kind/order/wastedBits) via
// FlacSubframeHeaderGpu and calls this primitive only when kind == LPC.
// Caller also provides a coefs scratch buffer of length >= order.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable composite FLAC LPC subframe decoder. Mirror of the
/// LPC branch in <see cref="FlacSubframeDecoder"/>.Decode.
/// </summary>
public static class FlacLpcSubframeDecodeGpu
{
    /// <summary>
    /// Decode one LPC subframe in place. Bit-exact vs the CPU
    /// FlacSubframeDecoder.Decode LPC branch. Caller must check that
    /// precMinusOne != 0b1111 and quantLevel &gt;= 0 (the spec forbids
    /// these but the GPU primitive does not throw).
    /// </summary>
    /// <param name="state">Bit reader state, positioned at the warm-up samples.</param>
    /// <param name="data">Underlying byte buffer.</param>
    /// <param name="samples">Output PCM (length &gt;= blockSize).</param>
    /// <param name="samplesBase">Base offset.</param>
    /// <param name="coefsScratch">Per-call QLP coefficients scratch (length &gt;= order).</param>
    /// <param name="coefsBase">Base offset.</param>
    /// <param name="blockSize">FLAC frame block size.</param>
    /// <param name="order">LPC predictor order (1..32).</param>
    /// <param name="effectiveBps">Bit depth for warm-up + residual = subframeBps - wastedBits.</param>
    /// <param name="wastedBits">Wasted bits per sample (left-shifted at the end).</param>
    public static void DecodeAt(
        ref FlacBitReaderGpuState state,
        ArrayView<byte> data,
        ArrayView<int> samples, long samplesBase,
        ArrayView<int> coefsScratch, long coefsBase,
        int blockSize, int order, int effectiveBps, int wastedBits)
    {
        // Step 1: warm-up samples.
        for (int i = 0; i < order; i++)
        {
            samples[samplesBase + i] = FlacBitReaderGpu.ReadBitsSigned(ref state, data, effectiveBps);
        }

        // Step 2: precision + quantLevel.
        int precMinusOne = (int)FlacBitReaderGpu.ReadBits(ref state, data, 4);
        int precision = precMinusOne + 1;
        int quantLevel = FlacBitReaderGpu.ReadBitsSigned(ref state, data, 5);

        // Step 3: QLP coefficients.
        for (int i = 0; i < order; i++)
        {
            coefsScratch[coefsBase + i] = FlacBitReaderGpu.ReadBitsSigned(ref state, data, precision);
        }

        // Step 4: residual into [order..blockSize).
        FlacResidualDecoderGpu.DecodeAt(
            ref state, data,
            samples, samplesBase + order,
            blockSize, order);

        // Step 5: LPC reconstruction.
        FlacLpcReconstructGpu.ReconstructAt(
            samples, samplesBase,
            coefsScratch, coefsBase,
            blockSize, order, quantLevel);

        // Step 6: wasted-bits left-shift.
        if (wastedBits > 0)
        {
            for (int i = 0; i < blockSize; i++)
            {
                samples[samplesBase + i] = samples[samplesBase + i] << wastedBits;
            }
        }
    }
}
