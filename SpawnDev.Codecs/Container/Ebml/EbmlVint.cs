// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// EBML variable-length integer encoding per
// https://matroska-org.github.io/libebml/specs.html. Used as the element ID
// and element size primitives in Matroska / WebM containers.
//
// The first byte of a VINT has a "length marker": the position of the first
// set bit indicates how many bytes the VINT occupies. For example:
//   1xxxxxxx            -> 1 byte,  7-bit value (element ID or size).
//   01xxxxxx xxxxxxxx    -> 2 bytes, 14-bit value.
//   001xxxxx ...         -> 3 bytes, 21-bit value.
//   ...
//   00000001 xxxxxxxx...x-> 8 bytes, 56-bit value.
//
// For element IDs the length marker is KEPT in the returned value (so IDs
// are unique across different widths). For sizes the marker is stripped.
// A size with all-ones following the marker signals "unknown size" (common
// in streaming / live Matroska).

namespace SpawnDev.Codecs.Container.Ebml;

/// <summary>
/// EBML variable-length integer (VINT) decoder.
/// </summary>
public static class EbmlVint
{
    /// <summary>Unknown-size sentinel returned for VINT sizes with all-ones tail.</summary>
    public const ulong UnknownSize = ulong.MaxValue;

    /// <summary>
    /// Read a VINT size starting at byte offset <paramref name="offset"/>. The
    /// length marker is stripped; the returned value is the raw size. All-ones
    /// after the marker returns <see cref="UnknownSize"/>.
    /// </summary>
    public static ulong ReadSize(ReadOnlySpan<byte> data, int offset, out int bytesRead)
    {
        ReadVint(data, offset, stripMarker: true, out ulong raw, out bytesRead, out bool isAllOnes);
        return isAllOnes ? UnknownSize : raw;
    }

    /// <summary>
    /// Read a VINT element ID starting at <paramref name="offset"/>. The length
    /// marker is PRESERVED in the returned value (element IDs are unique
    /// across widths because the marker is part of the identifier).
    /// </summary>
    public static ulong ReadId(ReadOnlySpan<byte> data, int offset, out int bytesRead)
    {
        ReadVint(data, offset, stripMarker: false, out ulong raw, out bytesRead, out _);
        return raw;
    }

    private static void ReadVint(
        ReadOnlySpan<byte> data, int offset, bool stripMarker,
        out ulong value, out int bytesRead, out bool isAllOnes)
    {
        if (offset >= data.Length)
            throw new InvalidDataException($"EBML VINT read past end of buffer at {offset}.");
        byte first = data[offset];
        if (first == 0)
            throw new InvalidDataException("EBML VINT first byte 0x00 is reserved (length >= 9 not supported).");

        int width = 0;
        for (int w = 1; w <= 8; w++)
        {
            if ((first & (0x80 >> (w - 1))) != 0)
            {
                width = w;
                break;
            }
        }
        if (width == 0 || offset + width > data.Length)
            throw new InvalidDataException($"EBML VINT truncated at offset {offset}.");
        bytesRead = width;

        ulong v;
        if (stripMarker)
        {
            byte markerMask = (byte)(0x80 >> (width - 1));
            v = (ulong)(first & ~markerMask);
        }
        else
        {
            v = first;
        }
        for (int i = 1; i < width; i++)
            v = (v << 8) | data[offset + i];
        value = v;

        // Check for all-ones tail (strip-marker mode): value's bits after the
        // marker are all 1 across the remaining (width-1) bytes plus the
        // remaining (7-1) bits of the first byte = 7*width bits total.
        isAllOnes = false;
        if (stripMarker)
        {
            int payloadBits = 7 * width;
            ulong maxValue = payloadBits >= 64 ? ulong.MaxValue : ((1UL << payloadBits) - 1);
            isAllOnes = v == maxValue;
        }
    }
}
