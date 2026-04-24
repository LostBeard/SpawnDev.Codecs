using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end tests for <see cref="FlacFrameDecoder"/>. Each test hand-builds
/// a full FLAC frame (header + N subframes + alignment + CRC-16 footer) using
/// the <c>FlacBitWriter</c> helper and decodes it back through the production
/// path. This proves the entire pipeline: header parse → per-channel subframe
/// decode → channel decorrelation → CRC-16 verification.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a complete FLAC frame: header bytes (with valid CRC-8) + subframe bits
    /// + CRC-16 footer over the combined payload.
    /// </summary>
    private static byte[] BuildFrame(
        int bsizeCode, int srateCode, int chanCode, int bpsCode, int blocking,
        byte[]? headerSideBytes,
        byte[] subframeBytes)
    {
        var headerBytes = BuildFrameHeaderBytes(
            bsizeCode, srateCode, chanCode, bpsCode, blocking,
            utf8SampleOrFrameNumber: 0x00,
            sideBytes: headerSideBytes);

        var combined = new byte[headerBytes.Length + subframeBytes.Length];
        Array.Copy(headerBytes, combined, headerBytes.Length);
        Array.Copy(subframeBytes, 0, combined, headerBytes.Length, subframeBytes.Length);

        ushort crc = FlacCrc.Compute16(combined);
        var withFooter = new byte[combined.Length + 2];
        Array.Copy(combined, withFooter, combined.Length);
        withFooter[^2] = (byte)(crc >> 8);
        withFooter[^1] = (byte)crc;
        return withFooter;
    }

    private static FlacStreamInfo MonoStreamInfo(int sampleRate = 44100, int bps = 16) => new FlacStreamInfo
    {
        MinBlockSize = 4096,
        MaxBlockSize = 4096,
        MinFrameSize = 0,
        MaxFrameSize = 0,
        SampleRateHz = sampleRate,
        Channels = 1,
        BitsPerSample = bps,
        TotalSamples = 0,
        Md5Signature = new byte[16],
    };

    [TestMethod]
    public void FlacFrame_Mono_ConstantSubframe_DecodesCorrectly()
    {
        // Mono frame with CONSTANT subframe value = 1234 across 4 samples.
        // bsize code 0b0110 + side byte 3 → 4-sample block. chan 0b0000 = 1 channel.
        // bps 0b100 = 16.
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w, 1234, 16);
        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 },
            subframeBytes: w.ToArray());

        var frame = FlacFrameDecoder.Decode(frameBytes, MonoStreamInfo());
        Equal(4, frame.Header.BlockSize);
        Equal(1, frame.Header.Channels);
        Equal(16, frame.Header.BitsPerSample);
        Equal(44100, frame.Header.SampleRateHz);
        Equal(frameBytes.Length, frame.FrameBytesConsumed);
        EqualInts(new[] { 1234, 1234, 1234, 1234 }, frame.Samples);
    }

    [TestMethod]
    public void FlacFrame_Stereo_Independent_TwoVerbatimSubframes()
    {
        // Stereo independent: chan 0b0001, block size 4, both channels VERBATIM 16-bit.
        var w = new FlacBitWriter();
        // Channel 0 VERBATIM with samples [10, 20, 30, 40]
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in new[] { 10, 20, 30, 40 }) WriteSigned(w, v, 16);
        // Channel 1 VERBATIM with samples [-10, -20, -30, -40]
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in new[] { -10, -20, -30, -40 }) WriteSigned(w, v, 16);

        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0001, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 },
            subframeBytes: w.ToArray());

        var streamInfo = MonoStreamInfo() with { Channels = 2 };
        var frame = FlacFrameDecoder.Decode(frameBytes, streamInfo);
        Equal(2, frame.Header.Channels);
        Equal(FlacChannelAssignment.Independent, frame.Header.ChannelAssignment);
        // Channel-major layout: [L0, L1, L2, L3, R0, R1, R2, R3]
        EqualInts(new[] { 10, 20, 30, 40, -10, -20, -30, -40 }, frame.Samples);
    }

    [TestMethod]
    public void FlacFrame_Stereo_LeftSide_Decorrelates()
    {
        // Original L = [100, 200, 300, 400], R = [50, 100, 150, 200].
        // Side = L - R = [50, 100, 150, 200] (same as R in this particular case).
        // Encoded: ch0 = L (verbatim 16-bit), ch1 = side (verbatim 17-bit).
        int[] L = { 100, 200, 300, 400 };
        int[] R = { 50, 100, 150, 200 };
        int[] side = { L[0] - R[0], L[1] - R[1], L[2] - R[2], L[3] - R[3] };

        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in L) WriteSigned(w, v, 16);
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in side) WriteSigned(w, v, 17); // side channel: bps + 1

        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b1000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 },
            subframeBytes: w.ToArray());

        var streamInfo = MonoStreamInfo() with { Channels = 2 };
        var frame = FlacFrameDecoder.Decode(frameBytes, streamInfo);
        Equal(FlacChannelAssignment.LeftSide, frame.Header.ChannelAssignment);
        // After decorrelation: ch0 = L, ch1 = R.
        EqualInts(L, frame.Samples.Take(4).ToArray());
        EqualInts(R, frame.Samples.Skip(4).Take(4).ToArray());
    }

    [TestMethod]
    public void FlacFrame_Stereo_RightSide_Decorrelates()
    {
        int[] L = { 100, 200, 300, 400 };
        int[] R = { 50, 100, 150, 200 };
        int[] side = { L[0] - R[0], L[1] - R[1], L[2] - R[2], L[3] - R[3] };

        var w = new FlacBitWriter();
        // ch0 = side (17-bit), ch1 = R (16-bit)
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in side) WriteSigned(w, v, 17);
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in R) WriteSigned(w, v, 16);

        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b1001, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 },
            subframeBytes: w.ToArray());

        var streamInfo = MonoStreamInfo() with { Channels = 2 };
        var frame = FlacFrameDecoder.Decode(frameBytes, streamInfo);
        Equal(FlacChannelAssignment.RightSide, frame.Header.ChannelAssignment);
        EqualInts(L, frame.Samples.Take(4).ToArray());
        EqualInts(R, frame.Samples.Skip(4).Take(4).ToArray());
    }

    [TestMethod]
    public void FlacFrame_Stereo_MidSide_Decorrelates()
    {
        // L = 5, R = 3 → mid = 4, side = 2.
        // L = 5, R = 2 → mid = 3, side = 3 (tests odd-side case).
        // L = -10, R = -6 → mid = -8, side = -4.
        int[] L = { 5, 5, -10, 100 };
        int[] R = { 3, 2, -6, 100 };
        int[] mid = new int[4];
        int[] side = new int[4];
        for (int i = 0; i < 4; i++)
        {
            mid[i] = (L[i] + R[i]) >> 1; // arithmetic-shift floor
            side[i] = L[i] - R[i];
        }

        var w = new FlacBitWriter();
        // ch0 = mid (16-bit), ch1 = side (17-bit)
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in mid) WriteSigned(w, v, 16);
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in side) WriteSigned(w, v, 17);

        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b1010, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 },
            subframeBytes: w.ToArray());

        var streamInfo = MonoStreamInfo() with { Channels = 2 };
        var frame = FlacFrameDecoder.Decode(frameBytes, streamInfo);
        Equal(FlacChannelAssignment.MidSide, frame.Header.ChannelAssignment);
        EqualInts(L, frame.Samples.Take(4).ToArray());
        EqualInts(R, frame.Samples.Skip(4).Take(4).ToArray());
    }

    [TestMethod]
    public void FlacFrame_BadCrc16_Throws()
    {
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w, 0, 16);
        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 3 },
            subframeBytes: w.ToArray());
        // Corrupt the CRC-16 footer.
        frameBytes[^1] ^= 0xFF;
        bool threw = false;
        try { FlacFrameDecoder.Decode(frameBytes, MonoStreamInfo()); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Bad CRC-16 should throw.");
    }

    [TestMethod]
    public void FlacFrame_Mono_FixedOrder1_EndToEnd()
    {
        // Mono frame with FIXED order 1 subframe, target samples [5, 7, 10, 8, 15].
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Fixed, 1, 0);
        WriteSigned(w, 5, 16);
        WriteRiceHeader(w, 0, 0);
        w.Write(2, 4);
        foreach (var r in new[] { 2, 3, -2, 7 }) WriteRiceCoded(w, r, 2);

        var frameBytes = BuildFrame(
            bsizeCode: 0b0110, srateCode: 0b1001, chanCode: 0b0000, bpsCode: 0b100, blocking: 0,
            headerSideBytes: new byte[] { 4 }, // block size 5
            subframeBytes: w.ToArray());

        var frame = FlacFrameDecoder.Decode(frameBytes, MonoStreamInfo());
        Equal(5, frame.Header.BlockSize);
        EqualInts(new[] { 5, 7, 10, 8, 15 }, frame.Samples);
    }
}
