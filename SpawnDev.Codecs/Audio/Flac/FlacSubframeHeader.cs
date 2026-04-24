// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Decoded subframe header: kind, order (for FIXED/LPC), and wasted-bits count.
/// Parsing follows RFC 9639 Section 10.
/// </summary>
public readonly record struct FlacSubframeHeader(
    FlacSubframeKind Kind,
    int Order,
    int WastedBitsPerSample);

/// <summary>
/// Parser for the 1-2 byte subframe header. The 6-bit type field encodes kind +
/// order; the wasted-bits field is Rice-unary-coded (the sample values output
/// are then left-shifted by this count at the end of subframe decoding).
/// </summary>
internal static class FlacSubframeHeaderParser
{
    /// <summary>Parse the next subframe header from <paramref name="reader"/>.</summary>
    internal static FlacSubframeHeader Parse(ref FlacBitReader reader)
    {
        // 1 reserved bit (must be 0)
        if (reader.ReadBit() != 0)
            throw new InvalidDataException("Subframe header reserved bit must be 0.");

        // 6-bit type field (RFC 9639 Section 10.1):
        //   0b000000        -> CONSTANT
        //   0b000001        -> VERBATIM
        //   0b00001x        -> reserved
        //   0b0001xx        -> reserved
        //   0b001xxx        -> FIXED (xxx = order 0..7, only 0..4 are valid)
        //   0b01xxxx        -> reserved
        //   0b1xxxxx        -> LPC (xxxxx+1 = order 1..32)
        int code = (int)reader.ReadBits(6);
        FlacSubframeKind kind;
        int order;
        if (code == 0) { kind = FlacSubframeKind.Constant; order = 0; }
        else if (code == 1) { kind = FlacSubframeKind.Verbatim; order = 0; }
        else if ((code & 0b111000) == 0b001000)
        {
            kind = FlacSubframeKind.Fixed;
            order = code & 0b000111;
            if (order > FlacConstants.MaxFixedOrder)
                throw new InvalidDataException($"FIXED subframe order {order} exceeds max {FlacConstants.MaxFixedOrder}.");
        }
        else if ((code & 0b100000) == 0b100000)
        {
            kind = FlacSubframeKind.Lpc;
            order = (code & 0b011111) + 1;
        }
        else
        {
            throw new InvalidDataException($"Reserved subframe type code 0b{code:B6}.");
        }

        // 1-bit wasted-bits flag; if set, unary count of extra trailing zero-bits stripped from samples.
        int wastedFlag = (int)reader.ReadBit();
        int wastedBits = 0;
        if (wastedFlag != 0)
        {
            // Per spec the wasted count is encoded as (k-1) zero-bits terminated by a 1.
            // So "1" alone (ReadUnary returns 0) means 1 wasted bit, unary 1 means 2, etc.
            wastedBits = reader.ReadUnary() + 1;
        }

        return new FlacSubframeHeader(kind, order, wastedBits);
    }
}
