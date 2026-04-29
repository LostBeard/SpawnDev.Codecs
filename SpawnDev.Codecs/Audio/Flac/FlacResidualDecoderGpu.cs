// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable FLAC Rice-coded residual decoder. Mirror of
// FlacResidualDecoder.Decode (libFLAC stream_decoder.c
// ::read_residual_partitioned_rice_, RFC 9639 Section 10.4.1).
// Reads (blockSize - predictorOrder) residual samples from a bit
// reader, supporting both PartitionedRice (4-bit parameter) and
// PartitionedRice2 (5-bit parameter) coding methods plus per-partition
// raw-residual escape paths.
//
// Sequential per-stream because the bit reader state evolves
// sample-by-sample (unary + remainder + raw read sequence). One-thread-
// per-stream on the GPU. Multiple FLAC channels parallelize across
// threads.
//
// Composes FlacBitReaderGpu for the actual bit reading.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC Rice-coded residual decoder. Mirror of
/// <see cref="FlacResidualDecoder"/>.Decode.
/// </summary>
public static class FlacResidualDecoderGpu
{
    private const int RESIDUAL_CODING_RICE = 0;
    private const int RESIDUAL_CODING_RICE2 = 1;
    private const int RICE_ESCAPE = 15;
    private const int RICE2_ESCAPE = 31;

    /// <summary>
    /// Decode the residual section of a FIXED or LPC subframe into
    /// <paramref name="residualOut"/>. The output buffer must be sized
    /// to <c>blockSize - predictorOrder</c>. The bit reader state is
    /// updated in place.
    /// </summary>
    /// <param name="state">Bit reader state.</param>
    /// <param name="data">Underlying byte buffer.</param>
    /// <param name="residualOut">Output residuals (length blockSize - predictorOrder).</param>
    /// <param name="residualBase">Base offset.</param>
    /// <param name="blockSize">FLAC frame block size.</param>
    /// <param name="predictorOrder">FIXED or LPC predictor order.</param>
    public static void DecodeAt(
        ref FlacBitReaderGpuState state,
        ArrayView<byte> data,
        ArrayView<int> residualOut, long residualBase,
        int blockSize, int predictorOrder)
    {
        // 2-bit coding method.
        int codingMethod = (int)FlacBitReaderGpu.ReadBits(ref state, data, 2);
        int paramBits;
        int escape;
        if (codingMethod == RESIDUAL_CODING_RICE)
        {
            paramBits = 4;
            escape = RICE_ESCAPE;
        }
        else
        {
            paramBits = 5;
            escape = RICE2_ESCAPE;
        }

        // 4-bit partition order.
        int partitionOrder = (int)FlacBitReaderGpu.ReadBits(ref state, data, 4);
        int partitionCount = 1 << partitionOrder;
        int partitionSizeBase = blockSize >> partitionOrder;

        long residualIndex = residualBase;
        for (int p = 0; p < partitionCount; p++)
        {
            int partitionSize = (p == 0) ? partitionSizeBase - predictorOrder : partitionSizeBase;
            int riceParam = (int)FlacBitReaderGpu.ReadBits(ref state, data, paramBits);

            if (riceParam == escape)
            {
                // Escape: 5-bit raw bit depth, then partitionSize raw signed values.
                int rawBits = (int)FlacBitReaderGpu.ReadBits(ref state, data, 5);
                for (int i = 0; i < partitionSize; i++)
                {
                    residualOut[residualIndex++] = rawBits == 0
                        ? 0
                        : FlacBitReaderGpu.ReadBitsSigned(ref state, data, rawBits);
                }
            }
            else
            {
                for (int i = 0; i < partitionSize; i++)
                {
                    // Unary quotient + riceParam-bit remainder.
                    int q = FlacBitReaderGpu.ReadUnary(ref state, data);
                    uint r = riceParam == 0
                        ? 0u
                        : FlacBitReaderGpu.ReadBits(ref state, data, riceParam);
                    uint zigzag = ((uint)q << riceParam) | r;
                    // LSB-carries-sign zigzag: 0 -> +(zz>>1), 1 -> -(zz>>1) - 1.
                    int value = ((zigzag & 1u) == 0u)
                        ? (int)(zigzag >> 1)
                        : -(int)(zigzag >> 1) - 1;
                    residualOut[residualIndex++] = value;
                }
            }
        }
    }
}
