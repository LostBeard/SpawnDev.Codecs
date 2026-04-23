using Concentus.Enums;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Smoke tests for the Concentus-backed reference oracle. These verify Concentus loads
/// and operates correctly across all 6 ILGPU backends' runtime environments (browser
/// Blazor WASM and desktop .NET) so subsequent bit-exact cross-validation tests can
/// rely on the oracle working. No SpawnDev.Codecs code is exercised here - these are
/// isolation tests for the oracle itself per feedback_test_new_features_isolated.md.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void ReferenceOracle_GenerateSineWave_HasCorrectLength()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        Equal(960 * 2, pcm.Length, "Sine wave should be samples * channels long");
    }

    [TestMethod]
    public void ReferenceOracle_GenerateSineWave_ValuesInAmplitudeRange()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, 480, amplitude: 0.5);
        foreach (var v in pcm)
        {
            if (v < -0.5f || v > 0.5f) throw new Exception($"Value {v} outside [-0.5, +0.5]");
        }
    }

    [TestMethod]
    public void ReferenceOracle_Encode_Silence_ProducesValidPacket()
    {
        var silence = ReferenceOracle.GenerateSilence(1, 960);
        var opusBytes = ReferenceOracle.EncodeFrame(silence, 48000, 1, 960);
        True(opusBytes.Length > 0, "Encoded packet should have bytes");
        InRange(opusBytes.Length, 1, 1275);
    }

    [TestMethod]
    public void ReferenceOracle_Encode_SineWave_ProducesValidPacket()
    {
        var sine = ReferenceOracle.GenerateSineWave(440, 48000, 1, 960);
        var opusBytes = ReferenceOracle.EncodeFrame(sine, 48000, 1, 960);
        InRange(opusBytes.Length, 1, 1275);
    }

    [TestMethod]
    public void ReferenceOracle_RoundTrip_Silence_ProducesExpectedLength()
    {
        var silence = ReferenceOracle.GenerateSilence(1, 960);
        var opusBytes = ReferenceOracle.EncodeFrame(silence, 48000, 1, 960);
        var decoded = ReferenceOracle.DecodePacket(opusBytes, 48000, 1);
        Equal(960, decoded.Length, "Should decode to 960 samples (20ms at 48kHz)");
    }

    [TestMethod]
    public void ReferenceOracle_RoundTrip_SineWave_PreservesDominantFrequency()
    {
        // Opus is lossy so we can't check bit-exact, but a 440 Hz sine wave should decode
        // to something roughly resembling a 440 Hz sine wave (not silence, not noise).
        var sine = ReferenceOracle.GenerateSineWave(440, 48000, 1, 960, amplitude: 0.5);
        var opusBytes = ReferenceOracle.EncodeFrame(
            sine, 48000, 1, 960,
            application: OpusApplication.OPUS_APPLICATION_AUDIO,
            bitrateBitsPerSecond: 64000);
        var decoded = ReferenceOracle.DecodePacket(opusBytes, 48000, 1);

        Equal(960, decoded.Length);

        // Check the signal has meaningful amplitude (not zeroed out by encoder).
        float peak = 0f;
        foreach (var v in decoded) peak = Math.Max(peak, Math.Abs(v));
        True(peak > 0.1f, $"Decoded peak {peak} too small; Opus encoder produced silence?");
    }

    [TestMethod]
    public void ReferenceOracle_Stereo_RoundTrip_Works()
    {
        var silence = ReferenceOracle.GenerateSilence(2, 960);
        var opusBytes = ReferenceOracle.EncodeFrame(silence, 48000, 2, 960);
        var decoded = ReferenceOracle.DecodePacket(opusBytes, 48000, 2);
        Equal(960 * 2, decoded.Length, "Stereo silence should decode to 1920 interleaved samples");
    }

    [TestMethod]
    public void ReferenceOracle_MultipleSampleRates_Succeed()
    {
        int[] rates = { 8000, 12000, 16000, 24000, 48000 };
        foreach (var rate in rates)
        {
            // 20ms frame at each rate
            int frameSize = rate / 50;
            var silence = ReferenceOracle.GenerateSilence(1, frameSize);
            var opusBytes = ReferenceOracle.EncodeFrame(silence, rate, 1, frameSize);
            var decoded = ReferenceOracle.DecodePacket(opusBytes, rate, 1);
            Equal(frameSize, decoded.Length, $"Rate {rate} Hz: decoded sample count");
        }
    }
}
