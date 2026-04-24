// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// On-wire Matroska SimpleBlock / Block binary parser. The ID + data-size
// VINTs are the EBML layer's job (handled by SpawnDev.EBML); this class
// parses the BLOCK BODY - the bytes between the size VINT and the end of
// the element - into one or more codec frames.
//
// Block body layout (same for SimpleBlock and Block except for the
// keyframe bit's location):
//
//   vint         track_number
//   int16 BE     relative_timestamp  (added to Cluster.Timestamp)
//   uint8        flags
//                  bit 7   keyframe      (SimpleBlock only; Block uses
//                                         BlockGroup.ReferenceBlock siblings)
//                  bits 5-6 reserved
//                  bit 4   invisible
//                  bits 1-2 lacing_type  (0=none, 1=Xiph, 2=fixed, 3=EBML)
//                  bit 0   discardable   (SimpleBlock only)
//
//   if lacing_type != 0:
//     uint8      frame_count_minus_one
//     [per-lacing-type size data]
//
//   bytes         concatenated frame payloads
//
// Matroska spec reference: https://www.matroska.org/technical/basics.html

namespace SpawnDev.Codecs.Container.Matroska;

internal static class MatroskaBlockParser
{
    /// <summary>
    /// Parse the block body. <paramref name="isSimpleBlock"/> controls
    /// whether the keyframe bit is consulted (SimpleBlock only).
    /// Returns a fully materialised list (not an iterator) so the method
    /// body can use <see cref="ReadOnlySpan{T}"/> without running into
    /// yield-boundary restrictions.
    /// </summary>
    public static IReadOnlyList<MatroskaFrame> Parse(
        ReadOnlySpan<byte> body, long clusterTimestamp, bool isSimpleBlock)
    {
        var frames = new List<MatroskaFrame>();
        int pos = 0;

        // 1) Track number (VINT with length marker stripped).
        ulong trackNumber = ReadVintStripMarker(body, ref pos);

        // 2) int16 BE relative timestamp.
        if (pos + 2 > body.Length) throw new InvalidDataException("block truncated before timestamp");
        short relTimestamp = (short)((body[pos] << 8) | body[pos + 1]);
        pos += 2;
        long absTimestamp = clusterTimestamp + relTimestamp;

        // 3) Flags.
        if (pos + 1 > body.Length) throw new InvalidDataException("block truncated before flags");
        byte flags = body[pos++];
        bool keyframe = isSimpleBlock && ((flags & 0x80) != 0);
        int lacing = (flags >> 1) & 0x03;

        // 4) Per-lacing-type: decode frame sizes into an int[].
        int[] frameSizes;
        if (lacing == 0)
        {
            // No lacing - single frame, size = remaining bytes.
            frameSizes = new[] { body.Length - pos };
        }
        else
        {
            if (pos + 1 > body.Length) throw new InvalidDataException("block truncated before frame count");
            int frameCount = body[pos++] + 1;
            frameSizes = new int[frameCount];
            int laceHeaderStart = pos;
            switch (lacing)
            {
                case 1: // Xiph lacing: first (n-1) sizes encoded as 255-byte increments.
                    ParseXiphSizes(body, ref pos, frameSizes);
                    break;
                case 2: // Fixed-size lacing: (body.length - header) / frameCount each.
                    int fixedTotal = body.Length - pos;
                    if (fixedTotal % frameCount != 0)
                        throw new InvalidDataException(
                            $"fixed-lacing remainder not divisible: total={fixedTotal} frames={frameCount}");
                    int fixedSize = fixedTotal / frameCount;
                    for (int i = 0; i < frameCount; i++) frameSizes[i] = fixedSize;
                    break;
                case 3: // EBML lacing: first size = unsigned VINT, subsequent = signed VINT offsets.
                    ParseEbmlSizes(body, ref pos, frameSizes);
                    break;
                default:
                    throw new InvalidDataException($"unknown lacing type {lacing}");
            }
            // For lacing types 1/3 the last frame's size is "whatever's left" and
            // wasn't encoded. Compute it from the remaining bytes after the sized
            // frames.
            if (lacing != 2)
            {
                int consumed = 0;
                for (int i = 0; i < frameSizes.Length - 1; i++) consumed += frameSizes[i];
                int payloadStart = pos;
                int payloadLen = body.Length - payloadStart;
                int last = payloadLen - consumed;
                if (last < 0)
                    throw new InvalidDataException(
                        $"lacing sizes exceed payload: consumed={consumed} payload={payloadLen}");
                frameSizes[^1] = last;
            }
        }

        // 5) Emit frames.
        int framePos = pos;
        for (int i = 0; i < frameSizes.Length; i++)
        {
            int size = frameSizes[i];
            if (framePos + size > body.Length)
                throw new InvalidDataException(
                    $"frame {i} ({size}B) overruns block body ({body.Length - framePos}B left)");
            var data = body.Slice(framePos, size).ToArray();
            framePos += size;
            frames.Add(new MatroskaFrame
            {
                TrackNumber = trackNumber,
                Timestamp = absTimestamp,
                Data = data,
                IsKeyframe = keyframe,
                LaceIndex = i,
            });
        }
        return frames;
    }

    /// <summary>Read a VINT and return the raw value with the length marker stripped.</summary>
    private static ulong ReadVintStripMarker(ReadOnlySpan<byte> data, ref int pos)
    {
        if (pos >= data.Length) throw new InvalidDataException("VINT read past end");
        byte first = data[pos];
        if (first == 0) throw new InvalidDataException("VINT first byte 0x00 is reserved");
        int width = 0;
        for (int w = 1; w <= 8; w++)
        {
            if ((first & (0x80 >> (w - 1))) != 0) { width = w; break; }
        }
        if (width == 0 || pos + width > data.Length) throw new InvalidDataException("VINT truncated");
        byte marker = (byte)(0x80 >> (width - 1));
        ulong v = (ulong)(first & ~marker);
        for (int i = 1; i < width; i++) v = (v << 8) | data[pos + i];
        pos += width;
        return v;
    }

    /// <summary>
    /// Xiph lacing: for each of the first (n-1) frames, read bytes of value
    /// 0xFF (each adding 255) until a byte &lt; 0xFF (which adds its value
    /// and ends that frame's size encoding).
    /// </summary>
    private static void ParseXiphSizes(ReadOnlySpan<byte> data, ref int pos, int[] sizes)
    {
        for (int i = 0; i < sizes.Length - 1; i++)
        {
            int size = 0;
            while (true)
            {
                if (pos >= data.Length) throw new InvalidDataException("Xiph size truncated");
                byte b = data[pos++];
                size += b;
                if (b != 0xFF) break;
            }
            sizes[i] = size;
        }
    }

    /// <summary>
    /// EBML lacing: first frame's size is an UNSIGNED VINT (marker stripped,
    /// same encoding as element sizes). Each subsequent frame's size is a
    /// SIGNED VINT offset from the previous size; the signed VINT is
    /// biased by 2^(7w-1) - 1 where w is the byte width.
    /// </summary>
    private static void ParseEbmlSizes(ReadOnlySpan<byte> data, ref int pos, int[] sizes)
    {
        // Frame 0 size: unsigned VINT.
        sizes[0] = (int)ReadVintStripMarker(data, ref pos);
        // Frames 1..n-2: signed VINT delta from previous.
        for (int i = 1; i < sizes.Length - 1; i++)
        {
            if (pos >= data.Length) throw new InvalidDataException("EBML size truncated");
            byte first = data[pos];
            int width = 0;
            for (int w = 1; w <= 8; w++)
            {
                if ((first & (0x80 >> (w - 1))) != 0) { width = w; break; }
            }
            if (width == 0 || pos + width > data.Length) throw new InvalidDataException("EBML VINT truncated");
            byte marker = (byte)(0x80 >> (width - 1));
            ulong raw = (ulong)(first & ~marker);
            for (int j = 1; j < width; j++) raw = (raw << 8) | data[pos + j];
            pos += width;
            // Bias: signed delta = raw - (2^(7w-1) - 1).
            int payloadBits = 7 * width;
            long bias = (1L << (payloadBits - 1)) - 1;
            long delta = (long)raw - bias;
            sizes[i] = sizes[i - 1] + (int)delta;
            if (sizes[i] < 0)
                throw new InvalidDataException($"EBML lacing: negative size at frame {i}");
        }
    }
}
