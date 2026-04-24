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

        // STREAMINFO payload (34 bytes).
        outputBytes.AddRange(BuildStreamInfoPayload(
            minBlock: blockSize, maxBlock: blockSize,
            sampleRateHz: sampleRateHz, channels: channels, bitsPerSample: bitsPerSample,
            totalSamples: (ulong)totalPerChannel));

        // Audio frames. Sample rate / channel / bps codes are stream-wide constants; block-size code
        // is resolved PER-FRAME because the final frame may be partial.
        int srateCode = ResolveSampleRateCode(sampleRateHz, out int srateSideBytes, out int srateSideValue);
        int chanCode = channels - 1; // Independent channels for now (codes 0..7 == channels 1..8).
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
                chanCode, bpsCode);
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
        int minBlock, int maxBlock, int sampleRateHz, int channels, int bitsPerSample, ulong totalSamples)
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
        for (int i = 0; i < 16; i++) w.Write(0, 8); // MD5 zero (not computed)
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
        int chanCode, int bpsCode)
    {
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
        // UTF-8-coded frame number (fixed block strategy: frame number).
        WriteUtf8Number(header, frameNumber);
        if (bsizeSideBytes == 1) header.Write((uint)bsizeSideValue, 8);
        else if (bsizeSideBytes == 2) header.Write((uint)bsizeSideValue, 16);
        if (srateSideBytes == 1) header.Write((uint)srateSideValue, 8);
        else if (srateSideBytes == 2) header.Write((uint)srateSideValue, 16);
        // Byte-align & append CRC-8.
        header.AlignToByte();
        byte[] headerBytes = header.ToArray();
        byte crc8 = FlacCrc.Compute8(headerBytes);

        var frame = new List<byte>(headerBytes.Length + 1);
        frame.AddRange(headerBytes);
        frame.Add(crc8);

        // Subframes. CONSTANT when every sample of this channel is equal
        // (typical for DC, silence, or very low-rate pad frames); VERBATIM otherwise.
        var subframeWriter = new FlacBitWriter();
        for (int ch = 0; ch < channels; ch++)
        {
            int firstSample = interleaved[frameStart * channels + ch];
            bool allEqual = true;
            for (int n = 1; n < blockSize; n++)
            {
                if (interleaved[(frameStart + n) * channels + ch] != firstSample)
                {
                    allEqual = false;
                    break;
                }
            }

            if (allEqual)
            {
                // Subframe header: reserved 0, type 0b000000 (CONSTANT), wasted flag 0.
                subframeWriter.Write(0, 1);
                subframeWriter.Write(0b000000, 6);
                subframeWriter.Write(0, 1);
                subframeWriter.WriteSigned(firstSample, bps);
            }
            else
            {
                // Subframe header: reserved 0, type 0b000001 (VERBATIM), wasted flag 0.
                subframeWriter.Write(0, 1);
                subframeWriter.Write(0b000001, 6);
                subframeWriter.Write(0, 1);
                for (int n = 0; n < blockSize; n++)
                {
                    int sample = interleaved[(frameStart + n) * channels + ch];
                    subframeWriter.WriteSigned(sample, bps);
                }
            }
        }
        subframeWriter.AlignToByte();
        frame.AddRange(subframeWriter.ToArray());

        // CRC-16 over full frame so far.
        ushort crc16 = FlacCrc.Compute16(frame.ToArray());
        frame.Add((byte)(crc16 >> 8));
        frame.Add((byte)crc16);
        return frame.ToArray();
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
