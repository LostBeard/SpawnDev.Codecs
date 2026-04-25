// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 Open Bitstream Unit (OBU) parser. The OBU is the framing primitive
// of AV1 - every byte of an AV1 bitstream lives inside one of these.
//
// AV1 spec sec 5.3 OBU syntax:
//   obu_header:
//     obu_forbidden_bit f(1)        // must be 0
//     obu_type          f(4)        // OBU type code
//     obu_extension_flag f(1)       // optional extension byte present?
//     obu_has_size_field f(1)       // optional length-prefix present?
//     obu_reserved_1bit f(1)        // must be 0
//   if (obu_extension_flag) obu_extension_header (1 byte: temporal_id + spatial_id)
//   if (obu_has_size_field) obu_size leb128 (1..N bytes)
//   payload : obu_size bytes (or to end of unit when no size field).
//
// References: AV1 spec sec 5.3 "OBU semantics" + dav1d/src/parse.c
// parse_obus.

using System.Buffers.Binary;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 OBU type (spec sec 6.2.1).</summary>
public enum Av1ObuType : byte
{
    /// <summary>Reserved.</summary>
    Reserved = 0,
    /// <summary>Sequence header (top-of-stream parameters).</summary>
    SequenceHeader = 1,
    /// <summary>Temporal delimiter (stream sync point between TUs).</summary>
    TemporalDelimiter = 2,
    /// <summary>Frame header.</summary>
    FrameHeader = 3,
    /// <summary>Tile group payload.</summary>
    TileGroup = 4,
    /// <summary>Metadata OBU (HDR, ITU T.35, scalability, ...).</summary>
    Metadata = 5,
    /// <summary>Combined frame header + tile group (most common).</summary>
    Frame = 6,
    /// <summary>Redundant frame header (error resilience).</summary>
    RedundantFrameHeader = 7,
    /// <summary>Tile list OBU (large-scale tile coding).</summary>
    TileList = 8,
    /// <summary>Padding OBU.</summary>
    Padding = 15,
}

/// <summary>One parsed AV1 OBU, byte-slicing into the original buffer.</summary>
public readonly record struct Av1Obu(
    Av1ObuType Type,
    int TemporalId,
    int SpatialId,
    bool HasSizeField,
    int PayloadOffset,
    int PayloadLength)
{
    /// <summary>True for OBUs that carry image data (frame, tile group).</summary>
    public bool IsCodedFrameData =>
        Type == Av1ObuType.Frame
        || Type == Av1ObuType.TileGroup
        || Type == Av1ObuType.FrameHeader
        || Type == Av1ObuType.RedundantFrameHeader;
}

/// <summary>Stateless AV1 OBU parser.</summary>
public static class Av1ObuParser
{
    /// <summary>
    /// Enumerate every OBU in <paramref name="data"/>. The returned
    /// records reference <paramref name="data"/> via offset / length.
    /// </summary>
    public static IEnumerable<Av1Obu> EnumerateObus(ReadOnlyMemory<byte> data)
    {
        int pos = 0;
        while (pos < data.Length)
        {
            byte header = data.Span[pos];
            int forbidden = (header >> 7) & 1;
            if (forbidden != 0)
                throw new InvalidDataException($"AV1 obu_forbidden_bit set at offset {pos}.");
            var type = (Av1ObuType)((header >> 3) & 0xF);
            bool extFlag = ((header >> 2) & 1) != 0;
            bool hasSizeField = ((header >> 1) & 1) != 0;
            int reserved = header & 1;
            if (reserved != 0)
                throw new InvalidDataException($"AV1 obu_reserved_1bit non-zero at offset {pos}.");

            int hdrSize = 1;
            int temporalId = 0;
            int spatialId = 0;
            if (extFlag)
            {
                if (pos + hdrSize >= data.Length)
                    throw new InvalidDataException("AV1 OBU header extension byte truncated.");
                byte ext = data.Span[pos + hdrSize];
                temporalId = (ext >> 5) & 0x7;
                spatialId = (ext >> 3) & 0x3;
                hdrSize++;
            }

            int payloadOffset;
            int payloadLength;
            if (hasSizeField)
            {
                int leb128Bytes = ReadLeb128(data.Span.Slice(pos + hdrSize), out long size);
                if (size < 0 || size > int.MaxValue)
                    throw new InvalidDataException($"AV1 OBU size {size} out of range.");
                payloadOffset = pos + hdrSize + leb128Bytes;
                payloadLength = (int)size;
                if (payloadOffset + payloadLength > data.Length)
                    throw new InvalidDataException(
                        $"AV1 OBU at {pos}: declared payload {payloadLength}B overruns buffer end {data.Length}.");
            }
            else
            {
                // No length prefix: payload runs to the end of the buffer
                // (caller must ensure single-OBU buffers when this happens,
                // e.g. AnnexB stream).
                payloadOffset = pos + hdrSize;
                payloadLength = data.Length - payloadOffset;
            }

            yield return new Av1Obu(type, temporalId, spatialId, hasSizeField,
                payloadOffset, payloadLength);

            pos = payloadOffset + payloadLength;
        }
    }

    /// <summary>
    /// Read an unsigned LEB128 integer. AV1 caps the value at 8 bytes
    /// total. Returns the byte count consumed via <paramref name="bytesRead"/>.
    /// </summary>
    private static int ReadLeb128(ReadOnlySpan<byte> data, out long value)
    {
        value = 0;
        int shift = 0;
        for (int i = 0; i < 8; i++)
        {
            if (i >= data.Length)
                throw new InvalidDataException($"AV1 LEB128 truncated at byte {i}.");
            byte b = data[i];
            value |= (long)(b & 0x7F) << shift;
            shift += 7;
            if ((b & 0x80) == 0)
                return i + 1;
        }
        throw new InvalidDataException("AV1 LEB128 exceeds 8 bytes.");
    }
}
