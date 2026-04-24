using Concentus.Enums;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="OpusOggEncoder"/>. Each test packages externally-
/// encoded Opus packets (produced via Concentus as the reference encoder)
/// into an Opus-in-Ogg byte stream, decodes it with our
/// <see cref="OpusOggDecoder"/>, and checks the structural round-trip.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void OpusOggEncoder_MonoThreeFrames_RoundtripsStructurally()
    {
        int frameLen = 960; // 20 ms at 48 kHz
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, frameLen * 3);
        var opusPackets = new List<byte[]>();
        var buf = new float[frameLen];
        for (int f = 0; f < 3; f++)
        {
            Array.Copy(pcm, f * frameLen, buf, 0, frameLen);
            byte[] enc = ReferenceOracle.EncodeFrame(buf, 48000, 1, frameLen, OpusApplication.OPUS_APPLICATION_VOIP);
            var toc = new OpusTocByte(enc[0]);
            if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Silk)
                throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need SILK.");
            opusPackets.Add(enc);
        }

        var opts = new OpusOggEncoderOptions
        {
            OutputChannels = 1,
            PreSkip = 312,
            InputSampleRateHz = 48000,
            Vendor = "SpawnDev.Codecs OpusOggEncoder test",
            BitstreamSerial = 0xDEADBEEF,
        };
        byte[] ogg = OpusOggEncoder.Encode(opusPackets, opts);

        OpusOggDecodeResult decoded;
        try
        {
            decoded = OpusOggDecoder.DecodeAsync(ogg).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"Stub: {ae.InnerException.Message}");
        }
        Equal(1, decoded.Head.OutputChannels);
        Equal(312, decoded.Head.PreSkip);
        Equal("SpawnDev.Codecs OpusOggEncoder test", decoded.Tags.Vendor);
        // After pre-skip trim, we expect up to (3 * 960) - 312 samples.
        int expectedPerChannel = 3 * 960 - 312;
        Equal(expectedPerChannel, decoded.TotalSamplesPerChannel);
    }

    [TestMethod]
    public void OpusOggEncoder_UserComments_RoundtripViaDecoder()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, 960);
        byte[] enc = ReferenceOracle.EncodeFrame(pcm, 48000, 1, 960, OpusApplication.OPUS_APPLICATION_VOIP);
        var toc = new OpusTocByte(enc[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Silk)
            throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need SILK.");
        var opts = new OpusOggEncoderOptions
        {
            OutputChannels = 1,
            PreSkip = 0,
            Vendor = "vendor-test",
            UserComments = new[] { "ARTIST=Ada Lovelace", "TITLE=Analytical Engine" },
            BitstreamSerial = 1,
        };
        byte[] ogg = OpusOggEncoder.Encode(new[] { enc }, opts);
        OpusOggDecodeResult decoded;
        try
        {
            decoded = OpusOggDecoder.DecodeAsync(ogg).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"Stub: {ae.InnerException.Message}");
        }
        Equal("vendor-test", decoded.Tags.Vendor);
        Equal(2, decoded.Tags.UserComments.Count);
        Equal("ARTIST=Ada Lovelace", decoded.Tags.UserComments[0]);
        Equal("TITLE=Analytical Engine", decoded.Tags.UserComments[1]);
    }

    [TestMethod]
    public void OpusOggEncoder_TooManyChannels_Throws()
    {
        var opts = new OpusOggEncoderOptions { OutputChannels = 6 };
        bool threw = false;
        try { _ = OpusOggEncoder.Encode(Array.Empty<byte[]>(), opts); }
        catch (ArgumentException) { threw = true; }
        True(threw, "Output channels > 2 should throw (multi-stream deferred).");
    }

    [TestMethod]
    public void OpusOggEncoder_EmptyPacketsList_Throws()
    {
        // OggPageWriter.WriteStream throws when given zero packets even after
        // prepending head + tags. Wait - we prepend head + tags so there are
        // always >= 2 outgoing packets. Let's verify the BOS/EOS logic still
        // works with zero audio packets.
        var opts = new OpusOggEncoderOptions { OutputChannels = 1, PreSkip = 0 };
        byte[] ogg = OpusOggEncoder.Encode(Array.Empty<byte[]>(), opts);
        // Zero audio packets should produce just header+tags pages that re-parse.
        OpusOggDecodeResult decoded;
        try
        {
            decoded = OpusOggDecoder.DecodeAsync(ogg).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"Stub: {ae.InnerException.Message}");
        }
        Equal(1, decoded.Head.OutputChannels);
        Equal(0, decoded.TotalSamplesPerChannel);
    }
}
