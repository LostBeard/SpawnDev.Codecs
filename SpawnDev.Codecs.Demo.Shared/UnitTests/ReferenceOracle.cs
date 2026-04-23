using Concentus;
using Concentus.Enums;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Test-only reference Opus codec using Concentus (BSD-3, pure-C# port of libopus) as
/// an oracle for cross-validating SpawnDev.Codecs bit-exactly during Phase 1a development.
///
/// Strategy: every SpawnDev.Codecs decode/encode subsystem we implement must produce
/// output that matches Concentus's output for the same input. When they match, the
/// subsystem is correct. This gives us a high-signal automated gate at every slice.
///
/// The oracle is forced to the pure-managed code path (no native libopus fallback) so
/// Blazor WASM and desktop both compare against the same reference.
///
/// Concentus is referenced by the test project ONLY. The SpawnDev.Codecs library itself
/// has no dependency on Concentus. BSD-3 attribution lives in NOTICE.md.
/// </summary>
internal static class ReferenceOracle
{
    private static readonly object _initLock = new();
    private static bool _initialized;

    /// <summary>Forces Concentus to use its pure managed code path (no native libopus probe).</summary>
    private static void EnsureInitialized()
    {
        if (_initialized) return;
        lock (_initLock)
        {
            if (_initialized) return;
            OpusCodecFactory.AttemptToUseNativeLibrary = false;
            _initialized = true;
        }
    }

    /// <summary>
    /// Encodes one frame of PCM float samples to an Opus packet using Concentus.
    /// </summary>
    /// <param name="pcm">Interleaved PCM samples, length = <paramref name="frameSizeSamples"/> * <paramref name="channelCount"/>. Values clipped to [-1, +1].</param>
    /// <param name="sampleRateHz">8000, 12000, 16000, 24000, or 48000.</param>
    /// <param name="channelCount">1 (mono) or 2 (stereo).</param>
    /// <param name="frameSizeSamples">Opus-legal frame size in samples per channel (2.5/5/10/20/40/60 ms at the given rate).</param>
    /// <param name="application">Opus application hint.</param>
    /// <param name="bitrateBitsPerSecond">Target bitrate. 0 = auto.</param>
    public static byte[] EncodeFrame(
        ReadOnlySpan<float> pcm,
        int sampleRateHz,
        int channelCount,
        int frameSizeSamples,
        OpusApplication application = OpusApplication.OPUS_APPLICATION_AUDIO,
        int bitrateBitsPerSecond = 0)
    {
        EnsureInitialized();
        using var encoder = OpusCodecFactory.CreateEncoder(sampleRateHz, channelCount, application);
        if (bitrateBitsPerSecond > 0) encoder.Bitrate = bitrateBitsPerSecond;

        Span<byte> scratch = stackalloc byte[1275]; // max Opus packet size
        int bytes = encoder.Encode(pcm, frameSizeSamples, scratch, scratch.Length);
        if (bytes <= 0) throw new InvalidOperationException($"Concentus encoder returned {bytes}.");
        return scratch.Slice(0, bytes).ToArray();
    }

    /// <summary>
    /// Decodes an Opus packet to PCM float samples using Concentus (oracle).
    /// </summary>
    /// <param name="opusBytes">The compressed packet.</param>
    /// <param name="sampleRateHz">Output sample rate (does not have to match the encoder's).</param>
    /// <param name="channelCount">Output channels (1 or 2).</param>
    /// <param name="maxFrameSizeSamples">Maximum samples per channel to decode. 5760 at 48kHz covers the 120ms max packet duration.</param>
    /// <returns>Decoded interleaved PCM, length = actualSamples * channelCount.</returns>
    public static float[] DecodePacket(
        ReadOnlySpan<byte> opusBytes,
        int sampleRateHz,
        int channelCount,
        int maxFrameSizeSamples = 5760)
    {
        EnsureInitialized();
        using var decoder = OpusCodecFactory.CreateDecoder(sampleRateHz, channelCount);
        var buffer = new float[maxFrameSizeSamples * channelCount];
        int samples = decoder.Decode(opusBytes, buffer.AsSpan(), maxFrameSizeSamples);
        if (samples <= 0) throw new InvalidOperationException($"Concentus decoder returned {samples}.");
        var result = new float[samples * channelCount];
        Array.Copy(buffer, 0, result, 0, samples * channelCount);
        return result;
    }

    /// <summary>
    /// Generates a sine wave in interleaved PCM float format for test input.
    /// </summary>
    public static float[] GenerateSineWave(
        double frequencyHz,
        int sampleRateHz,
        int channelCount,
        int samples,
        double amplitude = 0.5)
    {
        if (samples <= 0) throw new ArgumentOutOfRangeException(nameof(samples));
        if (amplitude is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(amplitude));

        var result = new float[samples * channelCount];
        double k = 2.0 * Math.PI * frequencyHz / sampleRateHz;
        for (int i = 0; i < samples; i++)
        {
            float v = (float)(amplitude * Math.Sin(k * i));
            for (int c = 0; c < channelCount; c++)
                result[i * channelCount + c] = v;
        }
        return result;
    }

    /// <summary>
    /// Generates a silent PCM buffer (all zeros) for test input.
    /// </summary>
    public static float[] GenerateSilence(int channelCount, int samples)
        => new float[samples * channelCount];
}
