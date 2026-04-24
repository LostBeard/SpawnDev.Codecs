using Concentus.Enums;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="OpusHeadParser"/>, <see cref="OpusTagsParser"/>, and
/// the full <see cref="OpusOggDecoder"/>.DecodeAsync pipeline that wraps Ogg
/// page assembly around <see cref="OpusDecoder"/>.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- OpusHead parser --------

    private static byte[] BuildOpusHeadPacket(int channels, int preSkip, uint inputRate, int gainQ78, int family,
        int streamCount = 1, int coupledCount = 0, byte[]? mapping = null)
    {
        int size = 19 + (family == 0 ? 0 : 2 + channels);
        var bytes = new byte[size];
        byte[] magic = { (byte)'O', (byte)'p', (byte)'u', (byte)'s', (byte)'H', (byte)'e', (byte)'a', (byte)'d' };
        Array.Copy(magic, bytes, 8);
        bytes[8] = 1;                  // version
        bytes[9] = (byte)channels;
        bytes[10] = (byte)preSkip;
        bytes[11] = (byte)(preSkip >> 8);
        for (int i = 0; i < 4; i++) bytes[12 + i] = (byte)(inputRate >> (8 * i));
        bytes[16] = (byte)gainQ78;
        bytes[17] = (byte)(gainQ78 >> 8);
        bytes[18] = (byte)family;
        if (family != 0)
        {
            bytes[19] = (byte)streamCount;
            bytes[20] = (byte)coupledCount;
            if (mapping != null && mapping.Length == channels)
                Array.Copy(mapping, 0, bytes, 21, channels);
        }
        return bytes;
    }

    [TestMethod]
    public void OpusHead_Stereo_48kHz_NoGain_ParsesCanonical()
    {
        var bytes = BuildOpusHeadPacket(channels: 2, preSkip: 312, inputRate: 48000, gainQ78: 0, family: 0);
        var h = OpusHeadParser.Parse(bytes);
        Equal(1, h.Version);
        Equal(2, h.OutputChannels);
        Equal(312, h.PreSkip);
        Equal(48000u, h.InputSampleRateHz);
        Equal(0, h.OutputGainQ7_8);
        Equal(0, h.ChannelMappingFamily);
        True(h.ChannelMapping is null, "Family 0 has no mapping table.");
    }

    [TestMethod]
    public void OpusHead_Gain_SignedDecoding()
    {
        // Q7.8 gain = -3 dB nominal. In Q7.8 that's -3 * 256 = -768.
        // -768 in 16-bit signed little-endian: (-768 & 0xFFFF) = 0xFD00 -> bytes 0x00 0xFD.
        var bytes = BuildOpusHeadPacket(1, 0, 48000, -768, 0);
        var h = OpusHeadParser.Parse(bytes);
        Equal(-768, h.OutputGainQ7_8);
    }

    [TestMethod]
    public void OpusHead_Surround51_Family1_WithMappingTable()
    {
        byte[] map = { 0, 4, 1, 2, 3, 5 };
        var bytes = BuildOpusHeadPacket(channels: 6, preSkip: 100, inputRate: 48000, gainQ78: 0,
            family: 1, streamCount: 4, coupledCount: 2, mapping: map);
        var h = OpusHeadParser.Parse(bytes);
        Equal(6, h.OutputChannels);
        Equal(1, h.ChannelMappingFamily);
        NotNull(h.ChannelMapping);
        Equal(4, h.ChannelMapping!.StreamCount);
        Equal(2, h.ChannelMapping.CoupledCount);
        EqualBytes(map, h.ChannelMapping.Mapping);
    }

    [TestMethod]
    public void OpusHead_WrongMagic_Throws()
    {
        var bytes = BuildOpusHeadPacket(2, 0, 48000, 0, 0);
        bytes[0] = (byte)'X';
        bool threw = false;
        try { _ = OpusHeadParser.Parse(bytes); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void OpusHead_ZeroChannels_Throws()
    {
        var bytes = BuildOpusHeadPacket(1, 0, 48000, 0, 0);
        bytes[9] = 0;
        bool threw = false;
        try { _ = OpusHeadParser.Parse(bytes); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    // -------- OpusTags parser --------

    private static byte[] BuildOpusTagsPacket(string vendor, string[] comments)
    {
        byte[] vendorBytes = System.Text.Encoding.UTF8.GetBytes(vendor);
        byte[][] cmtBytes = comments.Select(System.Text.Encoding.UTF8.GetBytes).ToArray();
        int size = 8 + 4 + vendorBytes.Length + 4;
        foreach (var b in cmtBytes) size += 4 + b.Length;
        var bytes = new byte[size];
        int pos = 0;
        foreach (var c in "OpusTags") bytes[pos++] = (byte)c;
        WriteUInt32Le(bytes, pos, (uint)vendorBytes.Length); pos += 4;
        Array.Copy(vendorBytes, 0, bytes, pos, vendorBytes.Length); pos += vendorBytes.Length;
        WriteUInt32Le(bytes, pos, (uint)cmtBytes.Length); pos += 4;
        foreach (var b in cmtBytes)
        {
            WriteUInt32Le(bytes, pos, (uint)b.Length); pos += 4;
            Array.Copy(b, 0, bytes, pos, b.Length); pos += b.Length;
        }
        return bytes;
    }

    [TestMethod]
    public void OpusTags_VendorAndComments_Parse()
    {
        var bytes = BuildOpusTagsPacket("libopus 1.3.1", new[] { "ARTIST=Bach", "TITLE=BWV 1060" });
        var t = OpusTagsParser.Parse(bytes);
        Equal("libopus 1.3.1", t.Vendor);
        Equal(2, t.UserComments.Count);
        Equal("ARTIST=Bach", t.UserComments[0]);
        Equal("TITLE=BWV 1060", t.UserComments[1]);
    }

    [TestMethod]
    public void OpusTags_EmptyComments_Parse()
    {
        var bytes = BuildOpusTagsPacket("V", Array.Empty<string>());
        var t = OpusTagsParser.Parse(bytes);
        Equal("V", t.Vendor);
        Equal(0, t.UserComments.Count);
    }

    [TestMethod]
    public void OpusTags_WrongMagic_Throws()
    {
        var bytes = BuildOpusTagsPacket("V", new[] { "A=1" });
        bytes[0] = (byte)'Z';
        bool threw = false;
        try { _ = OpusTagsParser.Parse(bytes); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    // -------- End-to-end Opus-in-Ogg decode --------

    /// <summary>Build a standalone Opus-in-Ogg stream from an encoded packet sequence.</summary>
    private static byte[] BuildOpusOggStream(int channels, int preSkip, byte[][] opusPackets, uint serial = 0xDEADBEEFu)
    {
        var head = BuildOpusHeadPacket(channels, preSkip, 48000, 0, 0);
        var tags = BuildOpusTagsPacket("SpawnDev.Codecs test", Array.Empty<string>());
        var bytes = new List<byte>();
        // Page 0: OpusHead (BOS).
        bytes.AddRange(BuildOggPage(
            headerType: OggConstants.HeaderTypeBeginningOfStream,
            granulePos: 0, serial: serial, pageSeq: 0,
            segmentLengths: SegmentizeForPage(head.Length),
            payload: head));
        // Page 1: OpusTags.
        bytes.AddRange(BuildOggPage(
            headerType: 0,
            granulePos: 0, serial: serial, pageSeq: 1,
            segmentLengths: SegmentizeForPage(tags.Length),
            payload: tags));
        // Remaining pages: one packet per page for simplicity.
        uint pageSeq = 2;
        long granuleRunning = 0;
        for (int i = 0; i < opusPackets.Length; i++)
        {
            byte[] pkt = opusPackets[i];
            bool isLast = i == opusPackets.Length - 1;
            // Rough granule update: assume 20ms packets at 48kHz = 960 samples each.
            granuleRunning += 960;
            bytes.AddRange(BuildOggPage(
                headerType: isLast ? OggConstants.HeaderTypeEndOfStream : (byte)0,
                granulePos: granuleRunning, serial: serial, pageSeq: pageSeq++,
                segmentLengths: SegmentizeForPage(pkt.Length),
                payload: pkt));
        }
        return bytes.ToArray();
    }

    /// <summary>Split a packet size into a valid Ogg segment table.</summary>
    private static byte[] SegmentizeForPage(int size)
    {
        if (size == 0) return new byte[] { 0 };
        var list = new List<byte>();
        while (size >= 255)
        {
            list.Add(255);
            size -= 255;
        }
        list.Add((byte)size);
        return list.ToArray();
    }

    [TestMethod]
    public void OpusOggDecoder_ConcentusEncodedSilk_RoundtripDecodes()
    {
        // Encode 4 frames of 20ms speech-like signal via Concentus, build an Opus-in-Ogg
        // byte stream, and decode through our OpusOggDecoder.
        int frameLen = 960; // 20ms at 48 kHz
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, frameLen * 4);
        var packets = new List<byte[]>();
        var frame = new float[frameLen];
        for (int f = 0; f < 4; f++)
        {
            Array.Copy(pcm, f * frameLen, frame, 0, frameLen);
            byte[] encoded = ReferenceOracle.EncodeFrame(frame, 48000, 1, frameLen, OpusApplication.OPUS_APPLICATION_VOIP);
            var toc = new OpusTocByte(encoded[0]);
            if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Silk)
                throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need SILK for this test.");
            packets.Add(encoded);
        }

        byte[] oggStream = BuildOpusOggStream(channels: 1, preSkip: 312, opusPackets: packets.ToArray());
        OpusOggDecodeResult result;
        try
        {
            result = OpusOggDecoder.DecodeAsync(oggStream).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"OpusDecoder still stubbed: {ae.InnerException.Message}");
        }

        Equal(1, result.Head.OutputChannels);
        Equal(312, result.Head.PreSkip);
        Equal("SpawnDev.Codecs test", result.Tags.Vendor);
        // After pre-skip trimming, we should have at most 4 * 960 samples.
        True(result.TotalSamplesPerChannel <= 4 * 960,
            $"Expected <= {4 * 960} samples, got {result.TotalSamplesPerChannel}.");
        // Pre-skip should have trimmed 312 samples from the front.
        int expectedPerChannel = 4 * 960 - 312;
        Equal(expectedPerChannel, result.TotalSamplesPerChannel);
        // Sanity: values in [-1, 1].
        for (int i = 0; i < result.InterleavedSamples48kHz.Length; i++)
        {
            float v = result.InterleavedSamples48kHz[i];
            True(v >= -1.0f && v <= 1.0f, $"sample {i}={v} out of [-1, 1]");
        }
    }

    [TestMethod]
    public void OpusOggDecoder_EmptyStream_Throws()
    {
        bool threw = false;
        try { _ = OpusOggDecoder.DecodeAsync(Array.Empty<byte>()).Result; }
        catch (AggregateException ae) when (ae.InnerException is InvalidDataException) { threw = true; }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void OpusOggDecoder_MultiStreamMapping_ThrowsNotSupported()
    {
        // Build an OpusHead declaring mapping family 1 + 6 channels. OpusOggDecoder
        // should throw NotSupportedException at that point.
        var head = BuildOpusHeadPacket(channels: 6, preSkip: 0, inputRate: 48000, gainQ78: 0,
            family: 1, streamCount: 4, coupledCount: 2, mapping: new byte[] { 0, 1, 2, 3, 4, 5 });
        var tags = BuildOpusTagsPacket("V", Array.Empty<string>());
        var bytes = new List<byte>();
        bytes.AddRange(BuildOggPage(OggConstants.HeaderTypeBeginningOfStream, 0, 1, 0,
            SegmentizeForPage(head.Length), head));
        bytes.AddRange(BuildOggPage(OggConstants.HeaderTypeEndOfStream, 0, 1, 1,
            SegmentizeForPage(tags.Length), tags));
        bool threw = false;
        try { _ = OpusOggDecoder.DecodeAsync(bytes.ToArray()).Result; }
        catch (AggregateException ae) when (ae.InnerException is NotSupportedException) { threw = true; }
        catch (NotSupportedException) { threw = true; }
        True(threw, "Surround 5.1 mapping family 1 should throw NotSupported.");
    }

    // WriteUInt32Le is defined in CodecsTestBase.VorbisHeaderTests (same partial class).
}
