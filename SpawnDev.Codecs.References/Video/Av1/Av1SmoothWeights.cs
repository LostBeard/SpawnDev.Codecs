// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 SMOOTH/SMOOTH_V/SMOOTH_H intra prediction weights table.
// Bit-exact port of libaom aom_dsp/intrapred_common.h smooth_weights[].
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Weights are quadratic from '1' to '1 / block_size', scaled by
// 2^SMOOTH_WEIGHT_LOG2_SCALE = 256.
// Lookup: <c>SmoothWeights(blockSize)</c> returns the contiguous weight
// span for a given dimension (4 / 8 / 16 / 32 / 64).

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 SMOOTH intra prediction weights (libaom <c>smooth_weights[]</c>).
/// </summary>
public static class Av1SmoothWeights
{
    /// <summary>libaom <c>SMOOTH_WEIGHT_LOG2_SCALE</c>.</summary>
    public const int Log2Scale = 8;

    /// <summary>libaom <c>scale = 1 &lt;&lt; SMOOTH_WEIGHT_LOG2_SCALE</c> (256).</summary>
    public const int Scale = 1 << Log2Scale;

    /// <summary>
    /// libaom <c>smooth_weights[]</c> - contiguous storage for all
    /// supported block dimensions: bs=4 (4 entries), bs=8 (8 entries),
    /// bs=16 (16), bs=32 (32), bs=64 (64). Total length = 124.
    /// </summary>
    public static readonly byte[] Weights = new byte[]
    {
        // bs = 4
        255, 149, 85, 64,
        // bs = 8
        255, 197, 146, 105, 73, 50, 37, 32,
        // bs = 16
        255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
        // bs = 32
        255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
        66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
        // bs = 64
        255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
        150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
        65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
        13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
    };

    /// <summary>
    /// Get the weights span for dimension <paramref name="dim"/>. Mirrors
    /// libaom <c>smooth_weights + dim - 4</c>.
    /// </summary>
    public static ReadOnlySpan<byte> GetWeights(int dim) => dim switch
    {
        4 => Weights.AsSpan(0, 4),
        8 => Weights.AsSpan(4, 8),
        16 => Weights.AsSpan(12, 16),
        32 => Weights.AsSpan(28, 32),
        64 => Weights.AsSpan(60, 64),
        _ => throw new ArgumentOutOfRangeException(nameof(dim),
            "block dimension must be 4, 8, 16, 32, or 64"),
    };
}
