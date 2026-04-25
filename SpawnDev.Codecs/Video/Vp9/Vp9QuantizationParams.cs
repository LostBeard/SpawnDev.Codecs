// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 quantization parameters parser - the quantization_params
// section of the uncompressed frame header. Mirror of libvpx
// vp9/decoder/vp9_decodeframe.c setup_quantization() and read_delta_q().
//
// Bitstream layout (VP9 spec sec 6.2.8):
//   base_q_idx f(8)
//   y_dc_delta:  delta_present f(1); if present then s(4)
//   uv_dc_delta: delta_present f(1); if present then s(4)
//   uv_ac_delta: delta_present f(1); if present then s(4)
//
// A frame is "lossless" when base_q_idx and all three deltas are 0
// (libvpx VP9_COMMON.lossless). The decoder behaves differently in
// that case (e.g. transform sizes are forced to 4x4, intra-only
// transform path is used).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Per-frame quantization parameters parsed from the uncompressed
/// header. Bit-exact against libvpx <c>setup_quantization</c>.
/// </summary>
public sealed record Vp9QuantizationParams
{
    /// <summary>libvpx <c>QINDEX_BITS</c>.</summary>
    public const int QIndexBits = 8;

    /// <summary>Base quantization index, 0..255 (f(8)).</summary>
    public required int BaseQIndex { get; init; }

    /// <summary>Y plane DC delta-q, signed (4 magnitude bits + sign).</summary>
    public required int YDcDeltaQ { get; init; }

    /// <summary>UV plane DC delta-q, signed.</summary>
    public required int UvDcDeltaQ { get; init; }

    /// <summary>UV plane AC delta-q, signed.</summary>
    public required int UvAcDeltaQ { get; init; }

    /// <summary>
    /// True when <see cref="BaseQIndex"/> and all three deltas are 0.
    /// libvpx forces lossless frames to use 4x4 transforms and the
    /// intra-only inverse transform path.
    /// </summary>
    public bool Lossless => BaseQIndex == 0 && YDcDeltaQ == 0
                            && UvDcDeltaQ == 0 && UvAcDeltaQ == 0;
}

/// <summary>Parser for VP9 quantization parameters in the uncompressed header.</summary>
public static class Vp9QuantizationParamsParser
{
    /// <summary>
    /// Parse quantization parameters from <paramref name="reader"/>.
    /// Reads a minimum of 11 bits (8-bit base_q_idx + three 1-bit delta
    /// flags) and a maximum of 26 bits (all three deltas present).
    /// </summary>
    internal static Vp9QuantizationParams Parse(ref Vp9BitReader reader)
    {
        int baseQIndex = (int)reader.ReadBits(Vp9QuantizationParams.QIndexBits);
        int yDcDeltaQ = ReadDeltaQ(ref reader);
        int uvDcDeltaQ = ReadDeltaQ(ref reader);
        int uvAcDeltaQ = ReadDeltaQ(ref reader);

        return new Vp9QuantizationParams
        {
            BaseQIndex = baseQIndex,
            YDcDeltaQ = yDcDeltaQ,
            UvDcDeltaQ = uvDcDeltaQ,
            UvAcDeltaQ = uvAcDeltaQ,
        };
    }

    /// <summary>
    /// Convenience overload for unit tests. Production callers parse
    /// via the frame header which advances a single shared bit reader.
    /// </summary>
    public static Vp9QuantizationParams Parse(ReadOnlySpan<byte> data)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r);
    }

    private static int ReadDeltaQ(ref Vp9BitReader reader)
        => reader.ReadFlag() ? reader.ReadSignedLiteral(4) : 0;
}
