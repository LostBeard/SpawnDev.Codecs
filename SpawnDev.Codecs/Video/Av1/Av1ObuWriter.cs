// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 OBU writer - inverse of Av1ObuParser. Emits the obu_header byte(s)
// + extension + LEB128 size prefix + payload bytes.
//
// Shape of an emitted OBU (matches Av1ObuParser exactly):
//   byte 0:                    [0][type:4][ext_flag][has_size_field][0]
//   byte 1 (if ext_flag):      [temporal_id:3][spatial_id:2][000]
//   bytes (if has_size_field): LEB128(payload_length)
//   bytes:                     payload
//
// First-class building block for any future AV1 encoder. Pairs with
// Av1SequenceHeaderWriter / Av1FrameHeaderWriter to produce an emit
// path that round-trips a real BBB AV1 stream bit-exact.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 OBU writer.</summary>
public static class Av1ObuWriter
{
    /// <summary>
    /// Emit a single OBU around <paramref name="payload"/> and return the
    /// resulting byte sequence. <paramref name="hasSizeField"/> controls
    /// whether a LEB128 length prefix is written (use true when
    /// concatenating multiple OBUs into one buffer; false when each OBU
    /// occupies its own buffer).
    /// </summary>
    public static byte[] EmitObu(
        Av1ObuType type,
        ReadOnlySpan<byte> payload,
        bool hasSizeField = true,
        int temporalId = 0,
        int spatialId = 0,
        bool? hasExtension = null)
    {
        if ((int)type < 0 || (int)type > 15)
            throw new ArgumentOutOfRangeException(nameof(type));
        if ((uint)temporalId > 7)
            throw new ArgumentOutOfRangeException(nameof(temporalId));
        if ((uint)spatialId > 3)
            throw new ArgumentOutOfRangeException(nameof(spatialId));

        bool hasExt = hasExtension ?? (temporalId != 0 || spatialId != 0);
        int hdrSize = 1 + (hasExt ? 1 : 0);
        int sizeFieldLen = hasSizeField ? Leb128Length(payload.Length) : 0;
        int total = hdrSize + sizeFieldLen + payload.Length;
        var buf = new byte[total];

        byte hdr = 0;
        hdr |= (byte)(((int)type & 0xF) << 3);
        if (hasExt) hdr |= 0x04;
        if (hasSizeField) hdr |= 0x02;
        buf[0] = hdr;

        int pos = 1;
        if (hasExt)
        {
            byte ext = 0;
            ext |= (byte)((temporalId & 0x7) << 5);
            ext |= (byte)((spatialId & 0x3) << 3);
            buf[pos++] = ext;
        }

        if (hasSizeField)
        {
            pos += WriteLeb128(buf.AsSpan(pos), payload.Length);
        }

        payload.CopyTo(buf.AsSpan(pos));
        return buf;
    }

    /// <summary>
    /// Re-emit a parsed OBU bit-exactly using the same header flags it
    /// was decoded with. <paramref name="sourceBuffer"/> is the buffer the
    /// OBU was parsed from (PayloadOffset / PayloadLength are slicing
    /// into it).
    /// </summary>
    public static byte[] EmitObu(Av1Obu obu, ReadOnlyMemory<byte> sourceBuffer)
    {
        var payload = sourceBuffer.Span.Slice(obu.PayloadOffset, obu.PayloadLength);
        return EmitObu(
            obu.Type,
            payload,
            hasSizeField: obu.HasSizeField,
            temporalId: obu.TemporalId,
            spatialId: obu.SpatialId,
            hasExtension: obu.HasExtension);
    }

    /// <summary>
    /// Bytes required to LEB128-encode a non-negative integer up to 2^56-1.
    /// </summary>
    public static int Leb128Length(long value)
    {
        if (value < 0)
            throw new ArgumentOutOfRangeException(nameof(value));
        if (value == 0) return 1;
        int n = 0;
        while (value > 0)
        {
            n++;
            value >>= 7;
        }
        return n;
    }

    /// <summary>
    /// Write a LEB128-encoded integer into <paramref name="dest"/>, returning
    /// the number of bytes written. AV1 caps the encoding at 8 bytes.
    /// </summary>
    public static int WriteLeb128(Span<byte> dest, long value)
    {
        if (value < 0)
            throw new ArgumentOutOfRangeException(nameof(value));
        int written = 0;
        do
        {
            byte b = (byte)(value & 0x7F);
            value >>= 7;
            if (value != 0) b |= 0x80;
            if (written >= dest.Length)
                throw new ArgumentException("LEB128 destination buffer too small.", nameof(dest));
            if (written >= 8)
                throw new InvalidOperationException("LEB128 encoding exceeded 8 bytes.");
            dest[written++] = b;
        } while (value != 0);
        return written;
    }
}
