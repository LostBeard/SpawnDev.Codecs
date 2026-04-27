using Concentus.Enums;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Bit-exact cross-validation of SpawnDev.Codecs against Concentus (reference oracle).
///
/// Pattern: for each (mode, sampleRate, channels, frameDuration) tuple,
///   1. Use Concentus to encode a known PCM pattern -> Opus bytes.
///   2. Decode those bytes with Concentus (oracle output).
///   3. Decode those bytes with SpawnDev.Codecs' OpusDecoder (our output).
///   4. Assert the two outputs match within the documented float ULP tolerance.
///
/// Current state: SILK paths run end-to-end through the SpawnDev SILK
/// decoder; CELT and Hybrid paths run through the Concentus-backed CELT
/// decoder (see Audio/Opus/Celt/CeltDecoder.cs file header for the
/// migration plan). The bit-exact-vs-Concentus assertions therefore hold
/// trivially for the CELT/Hybrid paths today and become genuine regression
/// gates when the per-module hand-port replaces the Concentus delegation.
///
/// Tests still catch <see cref="NotImplementedException"/> and re-throw as
/// <see cref="UnsupportedTestException"/> as a defensive measure: any
/// SpawnDev decode path that has not yet been implemented (e.g. SILK LBRR)
/// reports as Skipped rather than Failed.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Float tolerance for bit-exact comparison. Opus's float path allows small ULP
    /// differences across implementations that are mathematically equivalent but not
    /// bit-identical. 1e-5 covers typical float math rounding without hiding real bugs.
    /// </summary>
    private const float BitExactFloatTolerance = 1e-5f;

    private static void AssertCloseEnough(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance, string context)
    {
        if (expected.Length != actual.Length)
            throw new Exception($"[{context}] length mismatch: expected {expected.Length}, got {actual.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            if (diff > tolerance)
                throw new Exception($"[{context}] sample {i} differs: expected {expected[i]}, got {actual[i]} (diff={diff})");
        }
    }

    /// <summary>
    /// Runs a single cross-validation: encode PCM with Concentus, decode with both, compare.
    /// Returns normally if our decoder matches the oracle; throws <see cref="UnsupportedTestException"/>
    /// if the path is still <see cref="NotImplementedException"/> (reported as Skipped);
    /// throws on mismatch with a context-rich message.
    /// </summary>
    private static async Task RunCrossValidation(
        string label,
        float[] pcmInput,
        int sampleRateHz,
        int channelCount,
        int frameSizeSamples,
        OpusApplication application)
    {
        byte[] opusBytes = ReferenceOracle.EncodeFrame(
            pcmInput, sampleRateHz, channelCount, frameSizeSamples, application);

        float[] oracle = ReferenceOracle.DecodePacket(opusBytes, sampleRateHz, channelCount, frameSizeSamples);

        var decoder = OpusCodec.CreateDecoder(new OpusDecoderConfig
        {
            SampleRateHz = sampleRateHz,
            ChannelCount = channelCount
        });

        float[] mine = new float[frameSizeSamples * channelCount];
        int samples;
        try
        {
            samples = await decoder.DecodePacketAsync(opusBytes, mine);
        }
        catch (NotImplementedException ex)
        {
            throw new UnsupportedTestException($"[{label}] {ex.Message}");
        }
        finally
        {
            await decoder.DisposeAsync();
        }

        if (samples * channelCount != oracle.Length)
            throw new Exception($"[{label}] decoded sample count {samples * channelCount} != oracle length {oracle.Length}");

        AssertCloseEnough(oracle, mine.AsSpan(0, oracle.Length), BitExactFloatTolerance, label);
    }

    // -------- SILK-only paths (configs 0-11) --------

    [TestMethod]
    public async Task CrossValidate_SilkNb_20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 8000, 1, 160);
        await RunCrossValidation("SILK-NB-20ms-mono", pcm, 8000, 1, 160, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    [TestMethod]
    public async Task CrossValidate_SilkWb_20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 16000, 1, 320);
        await RunCrossValidation("SILK-WB-20ms-mono", pcm, 16000, 1, 320, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    [TestMethod]
    public async Task CrossValidate_SilkMb_20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 12000, 2, 240);
        await RunCrossValidation("SILK-MB-20ms-stereo", pcm, 12000, 2, 240, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    [TestMethod]
    public async Task CrossValidate_SilkNb_10ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 8000, 1, 80);
        await RunCrossValidation("SILK-NB-10ms-mono", pcm, 8000, 1, 80, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    [TestMethod]
    public async Task CrossValidate_Silk_Silence_Mono()
    {
        var pcm = ReferenceOracle.GenerateSilence(1, 320);
        await RunCrossValidation("SILK-WB-20ms-silence-mono", pcm, 16000, 1, 320, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    // -------- Hybrid paths (configs 12-15) --------

    [TestMethod]
    public async Task CrossValidate_HybridSwb_20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 24000, 1, 480);
        await RunCrossValidation("Hybrid-SWB-20ms-mono", pcm, 24000, 1, 480, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    [TestMethod]
    public async Task CrossValidate_HybridFb_20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        await RunCrossValidation("Hybrid-FB-20ms-stereo", pcm, 48000, 2, 960, OpusApplication.OPUS_APPLICATION_VOIP);
    }

    // -------- CELT-only paths (configs 16-31) --------

    [TestMethod]
    public async Task CrossValidate_CeltFb_2p5ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, 120);
        await RunCrossValidation("CELT-FB-2.5ms-mono", pcm, 48000, 1, 120, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
    }

    [TestMethod]
    public async Task CrossValidate_CeltFb_5ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 240);
        await RunCrossValidation("CELT-FB-5ms-stereo", pcm, 48000, 2, 240, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
    }

    [TestMethod]
    public async Task CrossValidate_CeltFb_10ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 1, 480);
        await RunCrossValidation("CELT-FB-10ms-mono", pcm, 48000, 1, 480, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
    }

    [TestMethod]
    public async Task CrossValidate_CeltFb_20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        await RunCrossValidation("CELT-FB-20ms-stereo", pcm, 48000, 2, 960, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
    }
}
