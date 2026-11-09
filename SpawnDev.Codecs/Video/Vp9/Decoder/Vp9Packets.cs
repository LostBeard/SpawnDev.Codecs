// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 packets -> frames, host side (container bookkeeping: sizes and offsets, no codec data is interpreted beyond the
// one-byte show_existing_frame check below). Follows libvpx v1.17.0 vp9/vp9_dx_iface.c decoder_decode and
// vp9/decoder/vp9_decoder.c vp9_parse_superframe_index (BSD-3-Clause):
//   - a packet whose last byte is a superframe marker (0b110mmfff) with the same byte at the front of the index holds
//     (fff + 1) frames whose sizes (mm + 1 bytes each, little-endian) are listed in the index;
//   - otherwise the packet is decoded frame after frame: a frame normally runs to the end of the packet; a
//     show_existing_frame frame ends after its header (1 byte, 2 for profile 3), zero bytes after a frame are padding.
// A decoder outputs at most one frame per packet: the last shown one (libvpx vp9_get_raw_frame after the whole packet;
// SVC packets show every spatial layer and only the top layer comes out).

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Container.Matroska;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Splitting VP9 packets (WebM blocks, IVF frames) into the frames a decoder decodes, as libvpx does.</summary>
public static class Vp9Packets
{
    /// <summary>The frames in <paramref name="packet"/> as (offset, length) ranges, in decode order.</summary>
    /// <exception cref="InvalidDataException">A superframe index whose two marker bytes differ, or whose sizes run past the packet.</exception>
    public static List<(int Offset, int Length)> Split(ReadOnlySpan<byte> packet)
    {
        var frames = new List<(int, int)>();
        if (packet.IsEmpty) return frames;
        byte marker = packet[^1];
        if ((marker & 0xE0) == 0xC0)
        {
            int count = (marker & 7) + 1, mag = ((marker >> 3) & 3) + 1, indexSize = 2 + mag * count;
            if (packet.Length < indexSize) throw new InvalidDataException("VP9 superframe index larger than its packet.");
            if (packet[packet.Length - indexSize] != marker) throw new InvalidDataException("VP9 superframe index markers differ.");
            int at = 0, x = packet.Length - indexSize + 1;
            for (int i = 0; i < count; i++)
            {
                uint size = 0;
                for (int j = 0; j < mag; j++) size |= (uint)packet[x++] << (8 * j);
                if (size > (uint)(packet.Length - at)) throw new InvalidDataException("VP9 superframe frame size past the packet.");
                frames.Add((at, (int)size));
                at += (int)size;
            }
            return frames;
        }
        int start = 0;
        while (start < packet.Length)
        {
            int length = packet.Length - start;
            int headerOnly = ShowExistingHeaderBytes(packet[start..]);
            if (headerOnly > 0) length = Math.Min(headerOnly, length);
            frames.Add((start, length));
            start += length;
            while (start < packet.Length && packet[start] == 0) start++;   // encoder padding
        }
        return frames;
    }

    /// <summary>The header length in bytes when <paramref name="frame"/> is a show_existing_frame frame, else 0.</summary>
    private static int ShowExistingHeaderBytes(ReadOnlySpan<byte> frame)
    {
        // frame_marker (2) profile_low (1) profile_high (1) [reserved_zero (1) when profile 3] show_existing_frame (1)
        int b = frame[0];
        if ((b >> 6) != 2) return 0;
        int profile = ((b >> 5) & 1) | (((b >> 4) & 1) << 1);
        if (profile == 3)
        {
            bool showExisting = ((b >> 2) & 1) != 0;   // bit 3 is the reserved zero
            return showExisting ? 2 : 0;
        }
        return ((b >> 3) & 1) != 0 ? 1 : 0;
    }

    /// <summary>
    /// The VP9 packets of a WebM / Matroska file (its first V_VP9 track) or an IVF file (FourCC VP90), in file order.
    /// </summary>
    public static IEnumerable<byte[]> ReadFile(byte[] file) => ReadFileWithAlpha(file).Select(p => p.Packet);

    /// <summary>
    /// As <see cref="ReadFile"/>, with each packet's alpha packet (WebM BlockAdditional, BlockAddID 1: a second VP9 stream
    /// holding the alpha plane in its luma) or null.
    /// </summary>
    public static IEnumerable<(byte[] Packet, byte[]? Alpha)> ReadFileWithAlpha(byte[] file)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (file.Length >= 4 && file[0] == (byte)'D' && file[1] == (byte)'K' && file[2] == (byte)'I' && file[3] == (byte)'F')
        {
            var header = IvfReader.ParseHeader(file);
            if (header.FourCc != "VP90") throw new InvalidDataException($"IVF FourCC {header.FourCc} is not VP90.");
            foreach (var f in IvfReader.EnumerateFrames(file)) yield return (f.Data.ToArray(), null);
            yield break;
        }
        var mkv = new MatroskaContainer(new MemoryStream(file, writable: false));
        var track = mkv.Tracks.FirstOrDefault(t => t.CodecId == "V_VP9")
            ?? throw new InvalidDataException("No V_VP9 track in the file.");
        foreach (var f in mkv.Frames)
            if (f.TrackNumber == track.TrackNumber) yield return (f.Data, f.BlockAdditional);
    }
}
