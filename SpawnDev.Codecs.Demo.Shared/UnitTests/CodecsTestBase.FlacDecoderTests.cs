using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end tests for the public <see cref="FlacDecoder"/>. Each test
/// builds a complete FLAC byte stream (4-byte "fLaC" marker + STREAMINFO
/// metadata block + N audio frames with CRC-16 footers) and decodes it via
/// the public API, then compares the interleaved PCM output against the
/// original samples used to build the stream.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a FLAC STREAMINFO block bytes (not including the 4-byte metadata
    /// header). Writes the exact 34-byte layout via <see cref="FlacBitWriter"/>.
    /// </summary>
    private static byte[] BuildStreamInfoPayload(
        int minBlock, int maxBlock,
        int sampleRate, int channels, int bps,
        ulong totalSamples)
    {
        var w = new FlacBitWriter();
        w.Write((uint)minBlock, 16);
        w.Write((uint)maxBlock, 16);
        w.Write(0, 24);                      // MinFrameSize
        w.Write(0, 24);                      // MaxFrameSize
        w.Write((uint)sampleRate, 20);
        w.Write((uint)(channels - 1), 3);
        w.Write((uint)(bps - 1), 5);
        // 36-bit total samples: split into 4 high + 32 low to stay within ReadBits(32) domain.
        w.Write((uint)(totalSamples >> 32), 4);
        w.Write((uint)(totalSamples & 0xFFFFFFFF), 32);
        for (int i = 0; i < 16; i++) w.Write(0, 8); // MD5 zero
        return w.ToArray();
    }

    private static byte[] BuildFlacStreamWithMetadata(
        int sampleRate, int channels, int bps, int minBlock, int maxBlock,
        byte[] frameBytes)
    {
        var streamInfoPayload = BuildStreamInfoPayload(minBlock, maxBlock, sampleRate, channels, bps, totalSamples: 0);
        var bytes = new List<byte>();
        // "fLaC"
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        // STREAMINFO header: isLast=1, type=0, length=34
        bytes.AddRange(new byte[] { 0x80, 0x00, 0x00, 0x22 });
        bytes.AddRange(streamInfoPayload);
        bytes.AddRange(frameBytes);
        return bytes.ToArray();
    }

    [TestMethod]
    public void FlacDecoder_Mono_SingleConstantFrame_DecodesToInterleavedPcm()
    {
        // Build one mono CONSTANT frame (value 1234, 4 samples).
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w, 1234, 16);
        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 }, subframeBytes: w.ToArray());

        var stream = BuildFlacStreamWithMetadata(
            sampleRate: 44100, channels: 1, bps: 16, minBlock: 4, maxBlock: 4,
            frameBytes: frameBytes);

        var result = FlacDecoder.Decode(stream);
        Equal(44100, result.StreamInfo.SampleRateHz);
        Equal(1, result.StreamInfo.Channels);
        Equal(4, result.TotalSamplesPerChannel);
        EqualInts(new[] { 1234, 1234, 1234, 1234 }, result.InterleavedSamples);
    }

    [TestMethod]
    public void FlacDecoder_Mono_MultipleFrames_ConcatenatesPcm()
    {
        // Frame 1: CONSTANT value 100, 4 samples.
        // Frame 2: CONSTANT value -200, 4 samples.
        var w1 = new FlacBitWriter();
        WriteSubframeHeader(w1, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w1, 100, 16);
        var frame1 = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 }, subframeBytes: w1.ToArray());

        var w2 = new FlacBitWriter();
        WriteSubframeHeader(w2, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w2, -200, 16);
        var frame2 = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 }, subframeBytes: w2.ToArray());

        var allFrames = new byte[frame1.Length + frame2.Length];
        Array.Copy(frame1, allFrames, frame1.Length);
        Array.Copy(frame2, 0, allFrames, frame1.Length, frame2.Length);

        var stream = BuildFlacStreamWithMetadata(
            sampleRate: 44100, channels: 1, bps: 16, minBlock: 4, maxBlock: 4,
            frameBytes: allFrames);

        var result = FlacDecoder.Decode(stream);
        Equal(8, result.TotalSamplesPerChannel);
        EqualInts(new[] { 100, 100, 100, 100, -200, -200, -200, -200 }, result.InterleavedSamples);
    }

    [TestMethod]
    public void FlacDecoder_Stereo_MidSide_InterleavesLr()
    {
        int[] L = { 5, 5, -10, 100 };
        int[] R = { 3, 2, -6, 100 };
        int[] mid = new int[4];
        int[] side = new int[4];
        for (int i = 0; i < 4; i++)
        {
            mid[i] = (L[i] + R[i]) >> 1;
            side[i] = L[i] - R[i];
        }

        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in mid) WriteSigned(w, v, 16);
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in side) WriteSigned(w, v, 17);

        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b1010, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 }, subframeBytes: w.ToArray());

        var stream = BuildFlacStreamWithMetadata(
            sampleRate: 44100, channels: 2, bps: 16, minBlock: 4, maxBlock: 4,
            frameBytes: frameBytes);

        var result = FlacDecoder.Decode(stream);
        Equal(2, result.StreamInfo.Channels);
        Equal(4, result.TotalSamplesPerChannel);
        // Interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        EqualInts(new[] { L[0], R[0], L[1], R[1], L[2], R[2], L[3], R[3] }, result.InterleavedSamples);
    }

    [TestMethod]
    public void FlacDecoder_FrameByFrame_ReadsExactlyOnce()
    {
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w, 42, 16);
        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 }, subframeBytes: w.ToArray());

        var stream = BuildFlacStreamWithMetadata(
            sampleRate: 44100, channels: 1, bps: 16, minBlock: 4, maxBlock: 4,
            frameBytes: frameBytes);

        var dec = FlacDecoder.Open(stream);
        var first = dec.ReadNextFrame();
        NotNull(first);
        Equal(4, first!.Header.BlockSize);
        True(dec.IsAtEnd, "Decoder should be at end after single frame.");
        Equal<FlacFrame?>(null, dec.ReadNextFrame());
    }

    [TestMethod]
    public void FlacDecoder_MetadataOnly_NoFrames_ReturnsEmpty()
    {
        var stream = BuildFlacStreamWithMetadata(
            sampleRate: 44100, channels: 1, bps: 16, minBlock: 4, maxBlock: 4,
            frameBytes: Array.Empty<byte>());
        var result = FlacDecoder.Decode(stream);
        Equal(0, result.TotalSamplesPerChannel);
        Equal(0, result.InterleavedSamples.Length);
    }

    [TestMethod]
    public void FlacDecoder_BadMarker_Throws()
    {
        byte[] bad = new byte[] { (byte)'X', (byte)'Y', (byte)'Z', (byte)'!' };
        bool threw = false;
        try { _ = FlacDecoder.Decode(bad); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Bad stream marker should throw.");
    }
}
