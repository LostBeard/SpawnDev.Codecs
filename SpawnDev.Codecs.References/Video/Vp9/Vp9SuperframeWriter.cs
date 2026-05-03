// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 Superframe writer - inverse of Vp9SuperframeParser. Packs 1..8
// VP9 frames into a single container packet with an optional trailing
// superframe index per VP9 Bitstream Specification, Annex B.1.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 superframe writer.</summary>
public static class Vp9SuperframeWriter
{
    /// <summary>
    /// Pack 1..8 frames into a single VP9 packet. With a single frame
    /// the output is just the frame bytes verbatim (no index marker).
    /// With 2+ frames the output is `frames concat + N size fields + 1
    /// marker byte` per spec Annex B.1.
    /// </summary>
    public static byte[] Emit(IReadOnlyList<byte[]> frames)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
            throw new ArgumentException("Need at least one frame.", nameof(frames));
        if (frames.Count > 8)
            throw new ArgumentException("VP9 superframe holds at most 8 frames.", nameof(frames));

        if (frames.Count == 1)
        {
            // Single frame - no index. Return the frame bytes (or a copy).
            var f = frames[0] ?? throw new ArgumentException("frame[0] is null.", nameof(frames));
            return (byte[])f.Clone();
        }

        // Determine bytes_per_size based on the largest frame.
        int maxLen = 0;
        foreach (var f in frames)
        {
            if (f is null)
                throw new ArgumentException("Null frame in input list.", nameof(frames));
            if (f.Length > maxLen) maxLen = f.Length;
        }
        int bytesPerSize;
        if (maxLen <= 0xFF) bytesPerSize = 1;
        else if (maxLen <= 0xFFFF) bytesPerSize = 2;
        else if (maxLen <= 0xFFFFFF) bytesPerSize = 3;
        else bytesPerSize = 4;

        // Total payload = frames + N * bytesPerSize + 1 marker byte
        long total = 0;
        foreach (var f in frames) total += f.Length;
        total += frames.Count * bytesPerSize + 1;
        if (total > int.MaxValue)
            throw new InvalidOperationException("VP9 superframe exceeds 2 GiB.");

        var output = new byte[total];
        int pos = 0;
        foreach (var f in frames)
        {
            Buffer.BlockCopy(f, 0, output, pos, f.Length);
            pos += f.Length;
        }
        // Size index, little-endian.
        for (int i = 0; i < frames.Count; i++)
        {
            int len = frames[i].Length;
            for (int b = 0; b < bytesPerSize; b++)
            {
                output[pos++] = (byte)((len >> (b * 8)) & 0xFF);
            }
        }
        // Marker byte: 0b110 (bits 7-5) + (bytesPerSize - 1) << 3 + (frameCount - 1).
        byte marker = (byte)(0b1100_0000
            | ((bytesPerSize - 1) << 3)
            | (frames.Count - 1));
        output[pos++] = marker;
        return output;
    }
}
