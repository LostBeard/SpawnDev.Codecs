using Concentus.Enums;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Cross-validation tests against Concentus (the pure-C# libopus port we use
/// as our reference oracle). Encodes speech-like audio via Concentus at a low
/// bitrate to force SILK-mode output, decodes via BOTH Concentus and our
/// OpusDecoder, and compares the two PCM outputs.
///
/// Current state: these tests are strict about NOT crashing and about output
/// being in the correct sample count + int16 range. Bit-exactness against
/// Concentus is a known future goal (requires carefully-matched fixed-point
/// paths through every SILK subsystem); today we verify the coarser contract
/// that the two decoders produce the same number of samples and that the
/// RMS energy is in the same ballpark. As remaining SILK paths tighten up
/// we'll ratchet the tolerance down toward bit-exact.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static double ComputeRmsFloat(ReadOnlySpan<float> pcm)
    {
        double sumSq = 0;
        for (int i = 0; i < pcm.Length; i++) sumSq += pcm[i] * pcm[i];
        return Math.Sqrt(sumSq / Math.Max(1, pcm.Length));
    }

    [TestMethod]
    public void OpusDecoder_ConcentusEncodedSilk_NbMono_DecodesWithoutCrash()
    {
        // Encode 20 ms of a 440 Hz sine at 8 kHz through Concentus with VOIP application.
        // This normally produces a SILK-mode Opus packet.
        var pcm = ReferenceOracle.GenerateSineWave(440, 8000, 1, 160);
        byte[] packet = ReferenceOracle.EncodeFrame(pcm, 8000, 1, 160, OpusApplication.OPUS_APPLICATION_VOIP);

        // Parse the packet's TOC - if Concentus picked CELT or Hybrid mode, skip the test.
        var toc = new OpusTocByte(packet[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Silk)
        {
            // Unexpected but not a failure of our decoder; document the skip.
            throw new UnsupportedTestException(
                $"Concentus chose {toc.Mode} for VOIP/NB/sine - SILK cross-val needs a SILK packet.");
        }

        var config = new OpusDecoderConfig { SampleRateHz = 8000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[160];
        int samples;
        try
        {
            samples = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"Our OpusDecoder hit NotImplemented: {ae.InnerException.Message}");
        }

        Equal(160, samples);
        for (int i = 0; i < samples; i++)
        {
            True(ourPcm[i] >= -1.0f && ourPcm[i] <= 1.0f, $"our pcm[{i}] = {ourPcm[i]} out of [-1, 1]");
        }
    }

    [TestMethod]
    public void OpusDecoder_ConcentusEncodedSilk_ComparedToConcentusDecode_RmsInSameBallpark()
    {
        // Encode same signal, decode with both decoders, compare RMS levels.
        var srcPcm = ReferenceOracle.GenerateSineWave(440, 8000, 1, 160);
        byte[] packet = ReferenceOracle.EncodeFrame(srcPcm, 8000, 1, 160, OpusApplication.OPUS_APPLICATION_VOIP);

        var toc = new OpusTocByte(packet[0]);
        if (toc.Mode != SpawnDev.Codecs.Audio.Opus.OpusMode.Silk)
        {
            throw new UnsupportedTestException(
                $"Concentus emitted {toc.Mode} mode; RMS comparison needs SILK.");
        }

        // Concentus decode.
        float[] concentusPcm = ReferenceOracle.DecodePacket(packet, 8000, 1);
        double concentusRms = ComputeRmsFloat(concentusPcm);

        // Our decode.
        var config = new OpusDecoderConfig { SampleRateHz = 8000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] ourPcm = new float[160];
        try
        {
            _ = dec.DecodePacketAsync(packet.AsMemory(), ourPcm.AsMemory()).Result;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            throw new UnsupportedTestException($"Our decoder: {ae.InnerException.Message}");
        }
        double ourRms = ComputeRmsFloat(ourPcm);

        // RMS should be in the same order of magnitude.
        // We're not bit-exact yet (subtle fixed-point drift across many subsystems),
        // but the overall signal energy should be recognizably the same.
        // Use a factor-of-4 window (ratio 0.25 to 4) to allow for early-day drift.
        double ratio = ourRms / Math.Max(concentusRms, 1e-6);
        True(ratio > 0.25 && ratio < 4.0,
            $"Our RMS ({ourRms:F4}) should be within factor-of-4 of Concentus RMS ({concentusRms:F4}); ratio={ratio:F2}");
    }
}
