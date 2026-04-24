// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC frame decoder. Wires together header parser, per-channel subframe decoder,
// channel decorrelation, and CRC-16 footer verification. Matches libFLAC
// stream_decoder.c::read_frame_.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Decodes a full FLAC audio frame: header, N subframes (one per channel),
/// byte alignment, and CRC-16 footer. After decode, decorrelated stereo modes
/// (LeftSide / RightSide / MidSide) are expanded into independent L/R samples.
/// </summary>
public static class FlacFrameDecoder
{
    /// <summary>
    /// Decode a single FLAC frame starting at the beginning of <paramref name="data"/>,
    /// resolving frame header codes against <paramref name="streamInfo"/>.
    /// </summary>
    public static FlacFrame Decode(ReadOnlySpan<byte> data, FlacStreamInfo streamInfo)
    {
        var header = FlacFrameHeaderParser.Parse(data, streamInfo);
        // Resume reading at position after the header's CRC-8 byte.
        var r = new FlacBitReader(data);
        // Advance past header bytes already consumed by the separate parser.
        for (int i = 0; i < header.HeaderBytesConsumed; i++) _ = r.ReadBits(8);

        int channels = header.Channels;
        int blockSize = header.BlockSize;
        int samplesLength = channels * blockSize;
        int[] samples = new int[samplesLength];

        for (int ch = 0; ch < channels; ch++)
        {
            int subframeBps = GetSubframeBitsPerSample(header, ch);
            FlacSubframeDecoder.Decode(ref r, samples.AsSpan(ch * blockSize, blockSize), subframeBps);
        }

        // Frame is byte-aligned after subframes; pad any partial byte (should be no-op in practice).
        r.AlignToByte();
        int crcCoveredBytes = r.Position / 8;
        if (data.Length < crcCoveredBytes + 2)
            throw new InvalidDataException("Frame truncated before CRC-16 footer.");
        ushort expectedCrc = (ushort)r.ReadBits(16);
        ushort actualCrc = FlacCrc.Compute16(data.Slice(0, crcCoveredBytes));
        if (expectedCrc != actualCrc)
            throw new InvalidDataException(
                $"Frame CRC-16 mismatch: expected 0x{expectedCrc:X4}, computed 0x{actualCrc:X4}.");

        ApplyChannelDecorrelation(samples, blockSize, header.ChannelAssignment);

        return new FlacFrame
        {
            Header = header,
            Samples = samples,
            FrameBytesConsumed = crcCoveredBytes + 2,
        };
    }

    /// <summary>
    /// Return the subframe bit depth for channel <paramref name="channelIndex"/>. Side
    /// channels in L/R/M-side stereo modes are one bit wider than the nominal frame
    /// bit depth because they carry signed L-R differences that can exceed the
    /// original dynamic range by one bit.
    /// </summary>
    private static int GetSubframeBitsPerSample(FlacFrameHeader header, int channelIndex)
    {
        return header.ChannelAssignment switch
        {
            FlacChannelAssignment.LeftSide when channelIndex == 1 => header.BitsPerSample + 1,
            FlacChannelAssignment.RightSide when channelIndex == 0 => header.BitsPerSample + 1,
            FlacChannelAssignment.MidSide when channelIndex == 1 => header.BitsPerSample + 1,
            _ => header.BitsPerSample,
        };
    }

    private static void ApplyChannelDecorrelation(int[] samples, int blockSize, FlacChannelAssignment mode)
    {
        if (mode == FlacChannelAssignment.Independent) return;

        for (int n = 0; n < blockSize; n++)
        {
            int a = samples[n];
            int b = samples[blockSize + n];
            switch (mode)
            {
                case FlacChannelAssignment.LeftSide:
                    // ch0 = L (unchanged), ch1 = L - R -> R = L - side = a - b.
                    samples[blockSize + n] = a - b;
                    break;
                case FlacChannelAssignment.RightSide:
                    // ch0 = L - R, ch1 = R (unchanged) -> L = side + R = a + b.
                    samples[n] = a + b;
                    break;
                case FlacChannelAssignment.MidSide:
                    // libFLAC: mid_scaled = (mid << 1) | (side & 1); L = (mid_scaled + side) >> 1; R = (mid_scaled - side) >> 1.
                    int mid = a;
                    int side = b;
                    int midScaled = (mid << 1) | (side & 1);
                    samples[n] = (midScaled + side) >> 1;
                    samples[blockSize + n] = (midScaled - side) >> 1;
                    break;
            }
        }
    }
}
