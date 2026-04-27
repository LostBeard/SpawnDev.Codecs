using Concentus.Enums;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Cross-validation tests for the CELT and Hybrid decode paths. Encodes
/// signals via Concentus at modes/bitrates that force CELT or Hybrid output,
/// decodes via BOTH Concentus (oracle) and our OpusDecoder, and compares
/// the two PCM outputs sample-for-sample.
///
/// Because the runtime CELT path in our OpusDecoder currently delegates to
/// Concentus internally (see Audio/Opus/Celt/CeltDecoder.cs file header for
/// the full provenance and migration plan), bit-exactness is the contract:
/// every produced sample MUST equal the oracle's sample. When a future
/// hand-port replaces the Concentus delegation, these same tests will gate
/// the migration - any divergence from the oracle is a regression.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static double MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new InvalidOperationException($"length mismatch: {a.Length} vs {b.Length}");
        double max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs((double)a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
    }

    [TestMethod]
    public void OpusDecoder_CeltPacket_FullbandMono_BitExactMatchesConcentus()
    {
        // RESTRICTED_LOWDELAY application + 48 kHz + 20 ms frame -> CELT mode
        // Fullband. This is a clean test of the CELT decode path.
        var src = ReferenceOracle.GenerateSineWave(1000, 48000, 1, 960);
        byte[] packet = ReferenceOracle.EncodeFrame(
            src, sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 960,
            application: OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            bitrateBitsPerSecond: 96000);

        var toc = new OpusTocByte(packet[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Celt)
            throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need CELT");

        // Oracle decode.
        float[] oraclePcm = ReferenceOracle.DecodePacket(packet, 48000, 1);

        // Our decode.
        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[960];
        int samples = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;

        Equal(960, samples);
        Equal(oraclePcm.Length, samples);
        // Bit-exact sample comparison. Both decoders run the same Concentus
        // code path, so output MUST match exactly.
        double maxDiff = MaxAbsDiff(oraclePcm.AsSpan(), ourPcm.AsSpan());
        True(maxDiff == 0.0, $"CELT decode should be bit-exact vs Concentus oracle; max diff {maxDiff}");
    }

    [TestMethod]
    public void OpusDecoder_CeltPacket_FullbandStereo_BitExactMatchesConcentus()
    {
        // Stereo CELT. Two channels of a sine sweep, encoded as a
        // RESTRICTED_LOWDELAY packet (CELT mode).
        var src = ReferenceOracle.GenerateSineWave(880, 48000, 2, 960);
        byte[] packet = ReferenceOracle.EncodeFrame(
            src, sampleRateHz: 48000, channelCount: 2, frameSizeSamples: 960,
            application: OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            bitrateBitsPerSecond: 128000);

        var toc = new OpusTocByte(packet[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Celt)
            throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need CELT");

        float[] oraclePcm = ReferenceOracle.DecodePacket(packet, 48000, 2);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 2 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[960 * 2];
        int samples = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;

        Equal(960, samples);
        Equal(oraclePcm.Length, samples * 2);
        double maxDiff = MaxAbsDiff(oraclePcm.AsSpan(), ourPcm.AsSpan());
        True(maxDiff == 0.0, $"Stereo CELT should be bit-exact vs Concentus oracle; max diff {maxDiff}");
    }

    [TestMethod]
    public void OpusDecoder_CeltPacket_2_5msFrame_BitExactMatchesConcentus()
    {
        // 2.5 ms frame at 48 kHz = 120 samples per channel. Smallest CELT frame.
        var src = ReferenceOracle.GenerateSineWave(2000, 48000, 1, 120);
        byte[] packet = ReferenceOracle.EncodeFrame(
            src, sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 120,
            application: OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            bitrateBitsPerSecond: 128000);

        var toc = new OpusTocByte(packet[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Celt)
            throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need CELT");

        float[] oraclePcm = ReferenceOracle.DecodePacket(packet, 48000, 1);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[120];
        int samples = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;

        Equal(120, samples);
        double maxDiff = MaxAbsDiff(oraclePcm.AsSpan(), ourPcm.AsSpan());
        True(maxDiff == 0.0, $"2.5 ms CELT should be bit-exact; max diff {maxDiff}");
    }

    [TestMethod]
    public void OpusDecoder_HybridPacket_FullbandMono_BitExactMatchesConcentus()
    {
        // Hybrid mode is forced when AUDIO/VOIP application hint is paired
        // with SWB or FB at 10/20 ms frames at moderate bitrate. SILK
        // handles the low band, CELT the high band.
        var src = ReferenceOracle.GenerateSineWave(440, 48000, 1, 960);
        byte[] packet = ReferenceOracle.EncodeFrame(
            src, sampleRateHz: 48000, channelCount: 1, frameSizeSamples: 960,
            application: OpusApplication.OPUS_APPLICATION_AUDIO,
            bitrateBitsPerSecond: 32000);

        var toc = new OpusTocByte(packet[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Hybrid)
            throw new UnsupportedTestException($"Concentus chose {toc.Mode}, need Hybrid");

        float[] oraclePcm = ReferenceOracle.DecodePacket(packet, 48000, 1);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[960];
        int samples = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;

        Equal(960, samples);
        double maxDiff = MaxAbsDiff(oraclePcm.AsSpan(), ourPcm.AsSpan());
        True(maxDiff == 0.0, $"Hybrid decode should be bit-exact vs Concentus oracle; max diff {maxDiff}");
    }

    [TestMethod]
    public void OpusDecoder_CeltPacket_MultiPacketStream_StateCarriesAcrossPackets()
    {
        // Encode 3 sequential 20 ms CELT packets - the decoder's MDCT overlap
        // and post-filter taps must carry over correctly between packets,
        // so each successive decode should match the oracle that was
        // initialized fresh and fed the same sequence.
        const int packetCount = 3;
        const int frameSize = 960;
        var src = ReferenceOracle.GenerateSineWave(1500, 48000, 1, frameSize * packetCount);

        // Encode each 960-sample chunk as its own packet via a single encoder
        // (Concentus encoder is stateful, just like our decoder).
        Concentus.OpusCodecFactory.AttemptToUseNativeLibrary = false;
        using var enc = Concentus.OpusCodecFactory.CreateEncoder(48000, 1, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
        enc.Bitrate = 96000;
        var packets = new byte[packetCount][];
        Span<byte> scratch = stackalloc byte[1275];
        for (int p = 0; p < packetCount; p++)
        {
            int srcOffset = p * frameSize;
            int n = enc.Encode(src.AsSpan(srcOffset, frameSize), frameSize, scratch, scratch.Length);
            if (n <= 0) throw new InvalidOperationException($"Concentus encoder returned {n} for packet {p}");
            packets[p] = scratch.Slice(0, n).ToArray();
            // Verify it really is CELT mode.
            var t = new OpusTocByte(packets[p][0]);
            if (t.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Celt)
                throw new UnsupportedTestException($"Concentus produced {t.Mode} for packet {p}, need CELT");
        }

        // Oracle: fresh decoder, decode all packets in sequence.
        using var oracleDec = Concentus.OpusCodecFactory.CreateDecoder(48000, 1);
        var oraclePcm = new float[frameSize * packetCount];
        for (int p = 0; p < packetCount; p++)
        {
            int n = oracleDec.Decode(packets[p].AsSpan(), oraclePcm.AsSpan(p * frameSize, frameSize), frameSize, false);
            Equal(frameSize, n);
        }

        // Our decoder: fresh, decode all packets in sequence via the same
        // OpusDecoder instance to exercise the inter-packet state.
        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var ourDec = new OpusDecoder(config);
        var ourPcm = new float[frameSize * packetCount];
        for (int p = 0; p < packetCount; p++)
        {
            int n = ourDec.DecodePacketAsync(packets[p].AsMemory(), ourPcm.AsMemory(p * frameSize, frameSize)).Result;
            Equal(frameSize, n);
        }

        double maxDiff = MaxAbsDiff(oraclePcm.AsSpan(), ourPcm.AsSpan());
        True(maxDiff == 0.0,
            $"Multi-packet CELT should be bit-exact vs oracle (state carries across packets); max diff {maxDiff}");
    }
}
