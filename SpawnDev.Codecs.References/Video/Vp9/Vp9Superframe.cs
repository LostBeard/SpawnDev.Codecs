// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 Superframe unpacker per the VP9 Bitstream Specification, Annex B.1.
// https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.6-20160331-draft.pdf
//
// A VP9 "superframe" is 1..8 individual VP9 frames concatenated into a
// single container packet (one WebM SimpleBlock, one RTP packet, etc.),
// followed by a compact index telling the decoder where each frame
// starts. Most common case: 2 frames where the first is a non-visible
// alternate-reference frame and the second is the actually-displayed
// frame - this is how VP9 encodes altref-GOP structures without exposing
// the non-shown frame as a separate container packet.
//
// The superframe index is identified by the LAST byte of the packet.
// The VP9 spec encodes the header 3+2+3 bits MSB-first, so layout is:
//
//     bit 7 6 5 4 3 2 1 0
//         +-----+-----+-----+
//         | 1 1 0 |bpfm1| nfm1 |       (marker byte)
//         +-----+-----+-----+
//
//   marker (bits 7-5) : must be 0b110 for this to be a superframe.
//   bpfm1  (bits 4-3) : bytes_per_framesize - 1 (0..3 -> 1..4 bytes)
//   nfm1   (bits 2-0) : number_of_frames - 1  (0..7 -> 1..8 frames)
//
// This matches libvpx's decoder: frames = (marker & 0x07) + 1,
// bytes_per_size = ((marker >> 3) & 0x03) + 1.
//
// If the marker isn't 0b110, the packet is a single VP9 frame and the
// whole payload IS frame 0.
//
// When the marker IS present, the packet layout is:
//
//     [frame_0][frame_1]...[frame_{n-1}][n*bpf size bytes (little-endian)][1 marker byte]
//
// Sizes inside the index are little-endian unsigned integers of
// bpf = bpfm1 + 1 bytes. They describe each frame's length in order.

using System.Buffers.Binary;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Decoded view of a VP9 packet: a list of frame byte-slice ranges within
/// the original packet. Holding ranges (not copies) keeps the hot path
/// allocation-free - decoders slice the original buffer when they're
/// ready to parse each frame.
/// </summary>
public sealed record Vp9Superframe
{
    /// <summary>Individual frame slice descriptors, in packet order.</summary>
    public required IReadOnlyList<Vp9FrameSlice> Frames { get; init; }

    /// <summary>True when the packet carried a superframe index.</summary>
    public required bool HadIndex { get; init; }
}

/// <summary>One entry in <see cref="Vp9Superframe.Frames"/>.</summary>
public readonly record struct Vp9FrameSlice
{
    /// <summary>Offset into the original packet where this frame starts.</summary>
    public int Offset { get; init; }
    /// <summary>Length in bytes of this frame.</summary>
    public int Length { get; init; }
}

/// <summary>Stateless VP9 superframe parser.</summary>
public static class Vp9SuperframeParser
{
    private const byte MarkerMask = 0b1110_0000;
    private const byte MarkerValue = 0b1100_0000;

    /// <summary>
    /// Parse <paramref name="packet"/> into 1..8 frame slices.
    /// The returned slices reference <paramref name="packet"/> by offset
    /// and length; the caller must not mutate <paramref name="packet"/>
    /// while it holds a <see cref="Vp9Superframe"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// The packet has a superframe-index marker but the declared sizes
    /// overflow the packet, or the declared frame count is 0.
    /// </exception>
    public static Vp9Superframe Parse(ReadOnlySpan<byte> packet)
    {
        if (packet.Length == 0)
            throw new InvalidDataException("VP9 packet is empty.");

        byte last = packet[^1];
        if ((last & MarkerMask) != MarkerValue)
        {
            // Not a superframe - one frame, whole payload.
            return new Vp9Superframe
            {
                HadIndex = false,
                Frames = new[] { new Vp9FrameSlice { Offset = 0, Length = packet.Length } },
            };
        }

        int frameCount = (last & 0x07) + 1;
        int bytesPerSize = ((last >> 3) & 0x03) + 1;
        int indexSize = 1 + frameCount * bytesPerSize; // marker + n*size fields

        if (indexSize > packet.Length)
            throw new InvalidDataException(
                $"VP9 superframe index size {indexSize}B exceeds packet size {packet.Length}B.");

        int sizesStart = packet.Length - indexSize; // where the size array begins
        var frames = new Vp9FrameSlice[frameCount];
        int pos = 0;
        for (int i = 0; i < frameCount; i++)
        {
            int sizeOffset = sizesStart + i * bytesPerSize;
            var sizeSpan = packet.Slice(sizeOffset, bytesPerSize);
            int size = ReadLeSize(sizeSpan);
            if (size <= 0)
                throw new InvalidDataException(
                    $"VP9 superframe frame {i} has non-positive size {size}.");
            if (pos + size > sizesStart)
                throw new InvalidDataException(
                    $"VP9 superframe frame {i} ({size}B starting at {pos}) overruns " +
                    $"the start of the size index ({sizesStart}).");
            frames[i] = new Vp9FrameSlice { Offset = pos, Length = size };
            pos += size;
        }
        // Frames need not fill all the space before the index - the spec
        // doesn't forbid padding - but in practice they always do. Treat
        // trailing bytes as benign.

        return new Vp9Superframe
        {
            HadIndex = true,
            Frames = frames,
        };
    }

    /// <summary>Little-endian unsigned integer, 1..4 bytes, into a non-negative int.</summary>
    private static int ReadLeSize(ReadOnlySpan<byte> bytes)
    {
        return bytes.Length switch
        {
            1 => bytes[0],
            2 => BinaryPrimitives.ReadUInt16LittleEndian(bytes),
            3 => bytes[0] | (bytes[1] << 8) | (bytes[2] << 16),
            4 => (int)BinaryPrimitives.ReadUInt32LittleEndian(bytes),
            _ => throw new InvalidDataException(
                    $"VP9 superframe size field must be 1..4 bytes, got {bytes.Length}."),
        };
    }
}
