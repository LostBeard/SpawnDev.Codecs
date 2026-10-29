// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.
//
// RFC 6716 section 3.1: the TOC byte encodes mode, bandwidth, frame duration,
// stereo flag, and frame count code.
//
//  0 1 2 3 4 5 6 7
// +-+-+-+-+-+-+-+-+
// | config  |s| c |
// +-+-+-+-+-+-+-+-+

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// The 1-byte TOC header at the start of an Opus packet, decoded into its component fields.
/// </summary>
public readonly struct OpusTocByte
{
    /// <summary>Raw TOC byte value.</summary>
    public byte Value { get; }

    /// <summary>
    /// Constructs a TOC from the raw byte. No validation is performed; every byte value
    /// is a valid TOC per RFC 6716 (all 32 config values, 2 stereo values, 4 count codes).
    /// </summary>
    public OpusTocByte(byte value)
    {
        Value = value;
    }

    /// <summary>The 5-bit configuration number in the high-order bits of the TOC (0-31).</summary>
    public int Config => (Value >> 3) & 0x1F;

    /// <summary>True if the packet carries stereo audio (bit 2 of the TOC).</summary>
    public bool IsStereo => (Value & 0x04) != 0;

    /// <summary>1 for mono, 2 for stereo.</summary>
    public int ChannelCount => IsStereo ? 2 : 1;

    /// <summary>The 2-bit frame count code in the low-order bits of the TOC (0-3).</summary>
    public int FrameCountCode => Value & 0x03;

    /// <summary>
    /// The coding mode selected for this packet, derived from <see cref="Config"/>.
    /// </summary>
    public OpusMode Mode
    {
        get
        {
            int cfg = Config;
            if (cfg < 12) return OpusMode.Silk;
            if (cfg < 16) return OpusMode.Hybrid;
            return OpusMode.Celt;
        }
    }

    /// <summary>
    /// The audio bandwidth carried by this packet, derived from <see cref="Config"/>.
    /// </summary>
    public OpusBandwidth Bandwidth
    {
        get
        {
            int cfg = Config;
            // SILK-only: NB (0-3), MB (4-7), WB (8-11)
            if (cfg < 4) return OpusBandwidth.Narrowband;
            if (cfg < 8) return OpusBandwidth.Mediumband;
            if (cfg < 12) return OpusBandwidth.Wideband;
            // Hybrid: SWB (12-13), FB (14-15)
            if (cfg < 14) return OpusBandwidth.Superwideband;
            if (cfg < 16) return OpusBandwidth.Fullband;
            // CELT-only: NB (16-19), WB (20-23), SWB (24-27), FB (28-31)
            if (cfg < 20) return OpusBandwidth.Narrowband;
            if (cfg < 24) return OpusBandwidth.Wideband;
            if (cfg < 28) return OpusBandwidth.Superwideband;
            return OpusBandwidth.Fullband;
        }
    }

    /// <summary>
    /// Number of audio samples per frame at the given output sample rate, as defined by RFC 6716
    /// for the config encoded in this TOC. (libopus <c>opus_packet_get_samples_per_frame</c>.)
    /// </summary>
    public int GetSamplesPerFrame(int sampleRateHz)
    {
        if (sampleRateHz <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRateHz));

        // CELT-only
        if ((Value & 0x80) != 0)
        {
            int power = (Value >> 3) & 0x03;
            return (sampleRateHz << power) / 400;
        }
        // SILK 10 ms or 20 ms fixed configs
        if ((Value & 0x60) == 0x60)
        {
            return (Value & 0x08) != 0 ? sampleRateHz / 50 : sampleRateHz / 100;
        }
        // SILK variable: 10/20/40/60 ms
        int code = (Value >> 3) & 0x03;
        if (code == 3) return sampleRateHz * 60 / 1000;
        return (sampleRateHz << code) / 100;
    }

    /// <summary>
    /// Frame duration in microseconds for this TOC at the given output sample rate.
    /// Convenience wrapper around <see cref="GetSamplesPerFrame"/>.
    /// </summary>
    public int GetFrameDurationMicroseconds() => GetSamplesPerFrame(48_000) * 1_000_000 / 48_000;

    /// <inheritdoc/>
    public override string ToString() =>
        $"OpusToc(value=0x{Value:X2}, config={Config}, mode={Mode}, bw={Bandwidth}, stereo={IsStereo}, c={FrameCountCode})";
}
