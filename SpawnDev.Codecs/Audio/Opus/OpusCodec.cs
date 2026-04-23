// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Factory entry point for the Opus codec. Separates construction from the decoder
/// interface so that future GPU-accelerated paths can be wired in without changing
/// the consumer-facing API.
/// </summary>
public static class OpusCodec
{
    /// <summary>
    /// Creates a new Opus decoder. Phase 1a state: packet parsing + mode routing wired;
    /// SILK and CELT decode paths throw <see cref="NotImplementedException"/>.
    /// </summary>
    public static IAudioDecoder CreateDecoder(OpusDecoderConfig config)
    {
        return new OpusDecoder(config);
    }
}
