// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC Rice-coded residual decoder. Matches libFLAC stream_decoder.c
// ::read_residual_partitioned_rice_.
//
// RFC 9639 Section 10.4.1: a residual block of (blockSize - predictorOrder)
// samples is split into 2^partitionOrder partitions. Each partition carries a
// Rice parameter and either Rice-coded residuals or, if the parameter is the
// escape value, raw two's-complement residuals at a shared bit depth.

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacResidualDecoder
{
    /// <summary>
    /// Decode the residual section of a FIXED or LPC subframe into
    /// <paramref name="residualOut"/>. The output must be sized to
    /// <c>blockSize - predictorOrder</c>.
    /// </summary>
    internal static void Decode(
        ref FlacBitReader reader,
        Span<int> residualOut,
        int blockSize,
        int predictorOrder)
    {
        int expected = blockSize - predictorOrder;
        if (residualOut.Length != expected)
            throw new ArgumentException(
                $"residualOut length {residualOut.Length} != blockSize - predictorOrder = {expected}.",
                nameof(residualOut));

        // 2-bit coding method: 0 = 4-bit Rice parameter, 1 = 5-bit Rice parameter.
        int codingMethod = (int)reader.ReadBits(2);
        int paramBits;
        int escape;
        if (codingMethod == FlacConstants.ResidualCodingPartitionedRice)
        {
            paramBits = 4;
            escape = FlacConstants.RiceParameterEscape;
        }
        else if (codingMethod == FlacConstants.ResidualCodingPartitionedRice2)
        {
            paramBits = 5;
            escape = FlacConstants.Rice2ParameterEscape;
        }
        else
        {
            throw new InvalidDataException($"Reserved residual coding method 0b{codingMethod:B2}.");
        }

        // 4-bit partition order: number of partitions = 2^order.
        int partitionOrder = (int)reader.ReadBits(4);
        int partitionCount = 1 << partitionOrder;
        int partitionSizeBase = blockSize >> partitionOrder;
        if (partitionSizeBase == 0 || partitionSizeBase * partitionCount != blockSize)
            throw new InvalidDataException(
                $"Partition order {partitionOrder} invalid for block size {blockSize}.");
        if (partitionSizeBase < predictorOrder)
            throw new InvalidDataException(
                $"First partition size {partitionSizeBase} < predictor order {predictorOrder}.");

        int residualIndex = 0;
        for (int p = 0; p < partitionCount; p++)
        {
            int partitionSize = (p == 0) ? partitionSizeBase - predictorOrder : partitionSizeBase;
            int riceParam = (int)reader.ReadBits(paramBits);
            if (riceParam == escape)
            {
                // Escape: next 5 bits are the bit depth for raw residuals.
                int rawBits = (int)reader.ReadBits(5);
                for (int i = 0; i < partitionSize; i++)
                {
                    residualOut[residualIndex++] = rawBits == 0 ? 0 : reader.ReadBitsSigned(rawBits);
                }
            }
            else
            {
                for (int i = 0; i < partitionSize; i++)
                {
                    // Unary quotient + riceParam-bit remainder, then zigzag un-interleave.
                    int q = reader.ReadUnary();
                    uint r = riceParam == 0 ? 0u : reader.ReadBits(riceParam);
                    uint zigzag = ((uint)q << riceParam) | r;
                    // LSB carries sign: 0 -> positive = zigzag >> 1, 1 -> negative = -(zigzag >> 1) - 1.
                    int value = ((zigzag & 1) == 0)
                        ? (int)(zigzag >> 1)
                        : -(int)(zigzag >> 1) - 1;
                    residualOut[residualIndex++] = value;
                }
            }
        }
    }
}
