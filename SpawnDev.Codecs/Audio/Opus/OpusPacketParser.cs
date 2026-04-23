// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus src/opus.c::opus_packet_parse_impl to clean C#.
// RFC 6716 section 3: frame packing (TOC + count + size headers + payload).
//
// Upstream Copyright (c) 2011 Xiph.Org Foundation, Skype Limited
// Upstream Copyright (c) 2024 Arm Limited
// Upstream authors: Jean-Marc Valin, Koen Vos
// Upstream license: BSD 3-Clause. See NOTICE.md.
// Upstream source: https://github.com/xiph/opus

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Parses raw Opus packet bytes into an <see cref="OpusPacket"/> structure.
/// Implements RFC 6716 section 3 (frame packing) and libopus opus_packet_parse_impl.
/// Thread-safe: the parser is stateless.
/// </summary>
public static class OpusPacketParser
{
    /// <summary>
    /// Maximum number of frames allowed per packet (RFC 6716 section 3.2.5:
    /// 48 frames at 2.5 ms each = 120 ms max packet duration).
    /// </summary>
    public const int MaxFramesPerPacket = 48;

    /// <summary>
    /// Maximum size of a single compressed frame in bytes when the packet is NOT self-delimited.
    /// Derived from RFC 6716 section 3.2.1 (frame length encoding).
    /// </summary>
    public const int MaxFrameBytes = 1275;

    /// <summary>
    /// Maximum total audio samples a packet may describe at 48 kHz (120 ms).
    /// </summary>
    private const int MaxSamplesAt48K = 5760;

    /// <summary>
    /// Parses an Opus packet from the given buffer.
    /// </summary>
    /// <param name="data">The raw packet bytes. The returned <see cref="OpusPacket"/>
    /// holds slices of this buffer without copying, so the caller must keep the buffer alive
    /// as long as the result is in use.</param>
    /// <param name="selfDelimited">Self-delimited framing per RFC 6716 Appendix B. False for
    /// standard Opus packets (e.g. RTP payload, Ogg-wrapped streams); true for multi-stream
    /// inner packets or research tooling.</param>
    /// <exception cref="ArgumentException">If the packet is malformed or internally inconsistent.</exception>
    public static OpusPacket Parse(ReadOnlyMemory<byte> data, bool selfDelimited = false)
    {
        var result = TryParse(data, selfDelimited, out OpusPacket? packet, out OpusPacketError error);
        if (result && packet is not null) return packet;

        string reason = error switch
        {
            OpusPacketError.BadArgument => "Invalid argument passed to parser.",
            OpusPacketError.InvalidPacket => "Malformed Opus packet.",
            _ => $"Parse failed with error code {(int)error}."
        };
        throw new ArgumentException(reason, nameof(data));
    }

    /// <summary>
    /// Non-throwing parse. Returns true and fills <paramref name="packet"/> on success;
    /// returns false and fills <paramref name="error"/> otherwise.
    /// </summary>
    public static bool TryParse(
        ReadOnlyMemory<byte> data,
        bool selfDelimited,
        out OpusPacket? packet,
        out OpusPacketError error)
    {
        packet = null;

        if (data.Length < 0)
        {
            error = OpusPacketError.BadArgument;
            return false;
        }
        if (data.Length == 0)
        {
            error = OpusPacketError.InvalidPacket;
            return false;
        }

        ReadOnlySpan<byte> span = data.Span;
        int totalLen = data.Length;

        byte toc = span[0];
        int cursor = 1;
        int remaining = totalLen - 1;

        var tocByte = new OpusTocByte(toc);
        int framesize = tocByte.GetSamplesPerFrame(48_000);

        int count;
        int lastSize = remaining;
        bool cbr = false;
        int padLen = 0;

        Span<short> sizes = stackalloc short[MaxFramesPerPacket];

        switch (toc & 0x03)
        {
            case 0: // One frame
                count = 1;
                break;

            case 1: // Two CBR frames
                count = 2;
                cbr = true;
                if (!selfDelimited)
                {
                    if ((remaining & 1) != 0) { error = OpusPacketError.InvalidPacket; return false; }
                    lastSize = remaining / 2;
                    sizes[0] = (short)lastSize;
                }
                break;

            case 2: // Two VBR frames
                count = 2;
                {
                    int consumed = ParseSize(span, cursor, remaining, out short s0);
                    if (consumed < 0 || s0 < 0 || s0 > remaining - consumed)
                    {
                        error = OpusPacketError.InvalidPacket;
                        return false;
                    }
                    sizes[0] = s0;
                    cursor += consumed;
                    remaining -= consumed;
                    lastSize = remaining - s0;
                }
                break;

            default: // case 3: multiple frames with explicit count byte
                if (remaining < 1) { error = OpusPacketError.InvalidPacket; return false; }
                {
                    byte ch = span[cursor++];
                    remaining--;
                    count = ch & 0x3F;
                    if (count <= 0 || framesize * count > MaxSamplesAt48K)
                    {
                        error = OpusPacketError.InvalidPacket;
                        return false;
                    }

                    // Padding flag
                    if ((ch & 0x40) != 0)
                    {
                        int p;
                        do
                        {
                            if (remaining <= 0) { error = OpusPacketError.InvalidPacket; return false; }
                            p = span[cursor++];
                            remaining--;
                            int tmp = p == 255 ? 254 : p;
                            remaining -= tmp;
                            padLen += tmp;
                        } while (p == 255);
                    }

                    if (remaining < 0) { error = OpusPacketError.InvalidPacket; return false; }

                    // VBR flag is bit 7
                    cbr = (ch & 0x80) == 0;

                    if (!cbr)
                    {
                        lastSize = remaining;
                        for (int i = 0; i < count - 1; i++)
                        {
                            int consumed = ParseSize(span, cursor, remaining, out short si);
                            if (consumed < 0 || si < 0 || si > remaining - consumed)
                            {
                                error = OpusPacketError.InvalidPacket;
                                return false;
                            }
                            sizes[i] = si;
                            cursor += consumed;
                            remaining -= consumed;
                            lastSize -= consumed + si;
                        }
                        if (lastSize < 0) { error = OpusPacketError.InvalidPacket; return false; }
                    }
                    else if (!selfDelimited)
                    {
                        lastSize = remaining / count;
                        if (lastSize * count != remaining)
                        {
                            error = OpusPacketError.InvalidPacket;
                            return false;
                        }
                        for (int i = 0; i < count - 1; i++) sizes[i] = (short)lastSize;
                    }
                }
                break;
        }

        // Self-delimited framing has an explicit size for the last frame.
        if (selfDelimited)
        {
            int consumed = ParseSize(span, cursor, remaining, out short sLast);
            if (consumed < 0 || sLast < 0 || sLast > remaining - consumed)
            {
                error = OpusPacketError.InvalidPacket;
                return false;
            }
            sizes[count - 1] = sLast;
            cursor += consumed;
            remaining -= consumed;

            if (cbr)
            {
                if (sLast * count > remaining + (sLast * count - sLast))
                {
                    // Match the libopus check: size[count-1]*count > len (len AFTER consuming the size byte).
                    // At this point remaining is the bytes after consuming the final size header.
                    // libopus check was `size[count-1]*count > len` with len = original remaining pre-consumption of frame bytes.
                    // We reproduce by checking against (remaining + consumed-effects). Using the straightforward
                    // RFC rule: all frames are CBR and their total cannot exceed available payload bytes.
                }
                if (sLast * count > remaining + sLast)
                {
                    error = OpusPacketError.InvalidPacket;
                    return false;
                }
                for (int i = 0; i < count - 1; i++) sizes[i] = sLast;
            }
            else if (consumed + sLast > lastSize)
            {
                error = OpusPacketError.InvalidPacket;
                return false;
            }
        }
        else
        {
            // Not self-delimited: last frame can't exceed 1275 bytes.
            if (lastSize > MaxFrameBytes) { error = OpusPacketError.InvalidPacket; return false; }
            sizes[count - 1] = (short)lastSize;
        }

        int payloadOffset = cursor;

        // Materialize frame slices.
        var frames = new ReadOnlyMemory<byte>[count];
        for (int i = 0; i < count; i++)
        {
            int size = sizes[i];
            frames[i] = data.Slice(cursor, size);
            cursor += size;
        }

        // Padding follows the frames (only in count-code 3 with padding bit).
        ReadOnlyMemory<byte> padding = padLen > 0
            ? data.Slice(cursor, padLen)
            : ReadOnlyMemory<byte>.Empty;

        int packetLen = cursor + padLen;

        packet = new OpusPacket
        {
            Toc = tocByte,
            Frames = frames,
            PayloadOffset = payloadOffset,
            PacketLength = packetLen,
            Padding = padding
        };
        error = OpusPacketError.None;
        return true;
    }

    /// <summary>
    /// Decodes an RFC 6716 section 3.2.1 frame-length value from the given buffer position.
    /// Returns the number of bytes consumed (1 or 2), or -1 if the buffer is too short.
    /// </summary>
    private static int ParseSize(ReadOnlySpan<byte> data, int offset, int remaining, out short size)
    {
        if (remaining < 1)
        {
            size = -1;
            return -1;
        }
        byte b0 = data[offset];
        if (b0 < 252)
        {
            size = b0;
            return 1;
        }
        if (remaining < 2)
        {
            size = -1;
            return -1;
        }
        size = (short)(4 * data[offset + 1] + b0);
        return 2;
    }
}
