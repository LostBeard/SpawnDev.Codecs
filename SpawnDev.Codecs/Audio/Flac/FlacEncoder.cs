// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Minimal FLAC encoder. Emits a valid FLAC stream (fLaC marker + STREAMINFO +
// audio frames) using VERBATIM subframes - i.e., no prediction, no Rice
// compression. This is lossless by construction (it's literally raw PCM with
// framing) and proves the encoder-side pipeline end-to-end. Later slices can
// add FIXED/LPC subframes and channel decorrelation analysis for actual
// compression while reusing this skeleton.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Minimal FLAC encoder. Produces a fully valid FLAC byte stream that the
/// built-in <see cref="FlacDecoder"/> (or any conforming decoder) can consume.
/// Does not yet attempt compression; every frame uses VERBATIM subframes.
/// </summary>
public static class FlacEncoder
{
    /// <summary>
    /// Encode PCM samples and write directly to a <c>.flac</c> file on disk.
    /// </summary>
    public static void EncodeToFile(
        string path,
        ReadOnlySpan<int> interleavedSamples,
        int sampleRateHz,
        int channels,
        int bitsPerSample,
        int blockSize = 4096)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        byte[] bytes = EncodeStream(interleavedSamples, sampleRateHz, channels, bitsPerSample, blockSize);
        File.WriteAllBytes(path, bytes);
    }

    /// <summary>
    /// Encode interleaved PCM samples to a full FLAC byte stream.
    /// </summary>
    /// <param name="interleavedSamples">PCM samples as <c>[ch0[0], ch1[0], ch0[1], ...]</c>. Length must be a multiple of <paramref name="channels"/>.</param>
    /// <param name="sampleRateHz">Output sample rate. Must be 1..655350 Hz.</param>
    /// <param name="channels">Channel count. 1..8.</param>
    /// <param name="bitsPerSample">Bits per sample. 4..32, but only the standard FLAC set (8/12/16/20/24/32) is supported by this encoder.</param>
    /// <param name="blockSize">Samples per channel per frame. Defaults to 4096.</param>
    public static byte[] EncodeStream(
        ReadOnlySpan<int> interleavedSamples,
        int sampleRateHz,
        int channels,
        int bitsPerSample,
        int blockSize = 4096)
    {
        ValidateInputs(interleavedSamples.Length, sampleRateHz, channels, bitsPerSample, blockSize);

        int totalPerChannel = interleavedSamples.Length / channels;
        var outputBytes = new List<byte>();

        // "fLaC"
        outputBytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });

        // Metadata block header: isLast=1, type=0 (STREAMINFO), length=34.
        outputBytes.AddRange(new byte[] { 0x80, 0x00, 0x00, 0x22 });

        // Compute MD5 of the decoded PCM for STREAMINFO integrity field.
        byte[] md5 = FlacMd5.Compute(interleavedSamples, bitsPerSample);

        // STREAMINFO payload (34 bytes).
        outputBytes.AddRange(BuildStreamInfoPayload(
            minBlock: blockSize, maxBlock: blockSize,
            sampleRateHz: sampleRateHz, channels: channels, bitsPerSample: bitsPerSample,
            totalSamples: (ulong)totalPerChannel,
            md5Signature: md5));

        // Audio frames. Sample rate / bps codes are stream-wide constants; block-size code
        // and (for stereo) channel-assignment code are resolved PER-FRAME.
        int srateCode = ResolveSampleRateCode(sampleRateHz, out int srateSideBytes, out int srateSideValue);
        int bpsCode = ResolveBpsCode(bitsPerSample);

        int frameIndex = 0;
        for (int frameStart = 0; frameStart < totalPerChannel; frameStart += blockSize)
        {
            int thisBlock = Math.Min(blockSize, totalPerChannel - frameStart);
            int bsizeCode = ResolveBlockSizeCode(thisBlock, out int bsizeSideBytes, out int bsizeSideValue);
            byte[] frameBytes = EncodeFrame(
                interleavedSamples, frameStart, thisBlock, channels, bitsPerSample,
                (uint)frameIndex,
                bsizeCode, bsizeSideBytes, bsizeSideValue,
                srateCode, srateSideBytes, srateSideValue,
                bpsCode);
            outputBytes.AddRange(frameBytes);
            frameIndex++;
        }
        return outputBytes.ToArray();
    }

    private static void ValidateInputs(int totalSamples, int sampleRateHz, int channels, int bps, int blockSize)
    {
        if (channels < 1 || channels > FlacConstants.MaxChannels)
            throw new ArgumentException($"Channels must be 1..{FlacConstants.MaxChannels}.", nameof(channels));
        if (totalSamples % channels != 0)
            throw new ArgumentException(
                $"Interleaved sample length {totalSamples} not a multiple of channels {channels}.",
                nameof(channels));
        if (sampleRateHz < 1 || sampleRateHz > 655350)
            throw new ArgumentException($"Sample rate {sampleRateHz} out of range [1, 655350].", nameof(sampleRateHz));
        if (bps is not (8 or 12 or 16 or 20 or 24 or 32))
            throw new ArgumentException(
                $"Bit depth {bps} not supported by this encoder (use 8/12/16/20/24/32).", nameof(bps));
        if (blockSize < 16 || blockSize > FlacConstants.MaxBlockSize)
            throw new ArgumentException($"Block size {blockSize} out of range [16, {FlacConstants.MaxBlockSize}].", nameof(blockSize));
    }

    private static byte[] BuildStreamInfoPayload(
        int minBlock, int maxBlock, int sampleRateHz, int channels, int bitsPerSample, ulong totalSamples,
        byte[] md5Signature)
    {
        var w = new FlacBitWriter();
        w.Write((uint)minBlock, 16);
        w.Write((uint)maxBlock, 16);
        w.Write(0, 24);                      // MinFrameSize unknown
        w.Write(0, 24);                      // MaxFrameSize unknown
        w.Write((uint)sampleRateHz, 20);
        w.Write((uint)(channels - 1), 3);
        w.Write((uint)(bitsPerSample - 1), 5);
        w.Write((uint)(totalSamples >> 32), 4);
        w.Write((uint)(totalSamples & 0xFFFFFFFF), 32);
        for (int i = 0; i < 16; i++) w.Write(md5Signature[i], 8);
        return w.ToArray();
    }

    private static int ResolveBlockSizeCode(int blockSize, out int sideBytes, out int sideValue)
    {
        sideBytes = 0;
        sideValue = 0;
        return blockSize switch
        {
            192 => 0b0001,
            576 => 0b0010,
            1152 => 0b0011,
            2304 => 0b0100,
            4608 => 0b0101,
            256 => 0b1000,
            512 => 0b1001,
            1024 => 0b1010,
            2048 => 0b1011,
            4096 => 0b1100,
            8192 => 0b1101,
            16384 => 0b1110,
            32768 => 0b1111,
            _ when blockSize <= 256 => Emit8BitSide(blockSize, out sideBytes, out sideValue),
            _ => Emit16BitSide(blockSize, out sideBytes, out sideValue),
        };

        static int Emit8BitSide(int bs, out int sb, out int sv) { sb = 1; sv = bs - 1; return 0b0110; }
        static int Emit16BitSide(int bs, out int sb, out int sv) { sb = 2; sv = bs - 1; return 0b0111; }
    }

    private static int ResolveSampleRateCode(int rate, out int sideBytes, out int sideValue)
    {
        sideBytes = 0;
        sideValue = 0;
        return rate switch
        {
            88200 => 0b0001,
            176400 => 0b0010,
            192000 => 0b0011,
            8000 => 0b0100,
            16000 => 0b0101,
            22050 => 0b0110,
            24000 => 0b0111,
            32000 => 0b1000,
            44100 => 0b1001,
            48000 => 0b1010,
            96000 => 0b1011,
            _ when rate % 1000 == 0 && rate / 1000 <= 255 => Emit8(rate, out sideBytes, out sideValue),
            _ when rate % 10 == 0 => Emit16Deca(rate, out sideBytes, out sideValue),
            _ => Emit16Hz(rate, out sideBytes, out sideValue),
        };

        static int Emit8(int r, out int sb, out int sv) { sb = 1; sv = r / 1000; return 0b1100; }
        static int Emit16Hz(int r, out int sb, out int sv) { sb = 2; sv = r; return 0b1101; }
        static int Emit16Deca(int r, out int sb, out int sv) { sb = 2; sv = r / 10; return 0b1110; }
    }

    private static int ResolveBpsCode(int bps) => bps switch
    {
        8 => 0b001,
        12 => 0b010,
        16 => 0b100,
        20 => 0b101,
        24 => 0b110,
        32 => 0b111,
        _ => throw new ArgumentException($"Unsupported bps {bps} for encoder."),
    };

    private static byte[] EncodeFrame(
        ReadOnlySpan<int> interleaved, int frameStart, int blockSize, int channels, int bps,
        uint frameNumber,
        int bsizeCode, int bsizeSideBytes, int bsizeSideValue,
        int srateCode, int srateSideBytes, int srateSideValue,
        int bpsCode)
    {
        // For stereo, select the cheapest channel decorrelation mode. Resolve
        // per-channel samples under that mode, then emit subframes.
        int[][] channelBuffers;
        int[] perChannelBps;
        int chanCode;
        if (channels == 2)
        {
            (channelBuffers, perChannelBps, chanCode) =
                SelectStereoMode(interleaved, frameStart, blockSize, bps);
        }
        else
        {
            channelBuffers = new int[channels][];
            perChannelBps = new int[channels];
            for (int ch = 0; ch < channels; ch++)
            {
                channelBuffers[ch] = new int[blockSize];
                perChannelBps[ch] = bps;
                for (int n = 0; n < blockSize; n++)
                    channelBuffers[ch][n] = interleaved[(frameStart + n) * channels + ch];
            }
            chanCode = channels - 1; // Independent 1-8 channels.
        }

        // Build frame header into one writer (through CRC-8 byte).
        var header = new FlacBitWriter();
        header.Write((uint)FlacConstants.FrameSyncCode, 14);
        header.Write(0, 1);                  // reserved
        header.Write(0, 1);                  // blocking strategy: fixed-block-size
        header.Write((uint)bsizeCode, 4);
        header.Write((uint)srateCode, 4);
        header.Write((uint)chanCode, 4);
        header.Write((uint)bpsCode, 3);
        header.Write(0, 1);                  // reserved
        WriteUtf8Number(header, frameNumber);
        if (bsizeSideBytes == 1) header.Write((uint)bsizeSideValue, 8);
        else if (bsizeSideBytes == 2) header.Write((uint)bsizeSideValue, 16);
        if (srateSideBytes == 1) header.Write((uint)srateSideValue, 8);
        else if (srateSideBytes == 2) header.Write((uint)srateSideValue, 16);
        header.AlignToByte();
        byte[] headerBytes = header.ToArray();
        byte crc8 = FlacCrc.Compute8(headerBytes);

        var frame = new List<byte>(headerBytes.Length + 1);
        frame.AddRange(headerBytes);
        frame.Add(crc8);

        // Emit each subframe using the unified writer.
        var subframeWriter = new FlacBitWriter();
        for (int ch = 0; ch < channelBuffers.Length; ch++)
            FlacSubframeWriter.Emit(subframeWriter, channelBuffers[ch], perChannelBps[ch]);
        subframeWriter.AlignToByte();
        frame.AddRange(subframeWriter.ToArray());

        ushort crc16 = FlacCrc.Compute16(frame.ToArray());
        frame.Add((byte)(crc16 >> 8));
        frame.Add((byte)crc16);
        return frame.ToArray();
    }

    /// <summary>
    /// Select the best stereo channel-decorrelation mode for this frame. Tries
    /// Independent, LeftSide, RightSide, MidSide; picks the one whose combined
    /// subframe bit estimate is smallest. Returns the two channel buffers, their
    /// per-channel bit depths (side channels are <paramref name="bps"/>+1), and
    /// the FLAC frame-header channel-assignment code.
    /// </summary>
    private static (int[][] buffers, int[] bpsPerChannel, int chanCode) SelectStereoMode(
        ReadOnlySpan<int> interleaved, int frameStart, int blockSize, int bps)
    {
        int[] L = new int[blockSize];
        int[] R = new int[blockSize];
        int[] side = new int[blockSize];
        int[] mid = new int[blockSize];
        for (int n = 0; n < blockSize; n++)
        {
            int l = interleaved[(frameStart + n) * 2 + 0];
            int r = interleaved[(frameStart + n) * 2 + 1];
            L[n] = l;
            R[n] = r;
            side[n] = l - r;
            mid[n] = (l + r) >> 1; // arithmetic shift (floor toward negative infinity)
        }

        long costL = FlacSubframeWriter.EstimateBits(L, bps);
        long costR = FlacSubframeWriter.EstimateBits(R, bps);
        long costSide = FlacSubframeWriter.EstimateBits(side, bps + 1);
        long costMid = FlacSubframeWriter.EstimateBits(mid, bps);

        long independent = costL + costR;
        long leftSide = costL + costSide;
        long rightSide = costSide + costR;
        long midSide = costMid + costSide;

        long min = Math.Min(Math.Min(independent, leftSide), Math.Min(rightSide, midSide));
        if (min == independent)
            return (new[] { L, R }, new[] { bps, bps }, 0b0001);
        if (min == leftSide)
            return (new[] { L, side }, new[] { bps, bps + 1 }, 0b1000);
        if (min == rightSide)
            return (new[] { side, R }, new[] { bps + 1, bps }, 0b1001);
        return (new[] { mid, side }, new[] { bps, bps + 1 }, 0b1010);
    }

    /// <summary>Encode a value as UTF-8-style variable-length integer (FLAC convention).</summary>
    private static void WriteUtf8Number(FlacBitWriter w, ulong value)
    {
        if (value < 0x80)
        {
            w.Write((uint)value, 8);
        }
        else if (value < 0x800)
        {
            w.Write(0b110u, 3); w.Write((uint)(value >> 6), 5);
            w.Write(0b10u, 2);  w.Write((uint)(value & 0x3F), 6);
        }
        else if (value < 0x10000)
        {
            w.Write(0b1110u, 4); w.Write((uint)(value >> 12), 4);
            w.Write(0b10u, 2);   w.Write((uint)((value >> 6) & 0x3F), 6);
            w.Write(0b10u, 2);   w.Write((uint)(value & 0x3F), 6);
        }
        else if (value < 0x200000)
        {
            w.Write(0b11110u, 5); w.Write((uint)(value >> 18), 3);
            for (int i = 2; i >= 0; i--) { w.Write(0b10u, 2); w.Write((uint)((value >> (6 * i)) & 0x3F), 6); }
        }
        else if (value < 0x4000000)
        {
            w.Write(0b111110u, 6); w.Write((uint)(value >> 24), 2);
            for (int i = 3; i >= 0; i--) { w.Write(0b10u, 2); w.Write((uint)((value >> (6 * i)) & 0x3F), 6); }
        }
        else if (value < 0x80000000UL)
        {
            w.Write(0b1111110u, 7); w.Write((uint)(value >> 30), 1);
            for (int i = 4; i >= 0; i--) { w.Write(0b10u, 2); w.Write((uint)((value >> (6 * i)) & 0x3F), 6); }
        }
        else
        {
            w.Write(0b11111110u, 8);
            for (int i = 5; i >= 0; i--) { w.Write(0b10u, 2); w.Write((uint)((value >> (6 * i)) & 0x3F), 6); }
        }
    }
}
