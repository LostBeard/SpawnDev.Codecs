// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// FLAC frame channel assignment. Values <c>0..7</c> are independent-channel
/// modes where the numeric value encodes <c>channel_count - 1</c>. Values
/// <c>8/9/10</c> are stereo decorrelation modes that always decode 2 channels.
/// </summary>
public enum FlacChannelAssignment
{
    /// <summary>Independent channels: 1-8 channels with no decorrelation.</summary>
    Independent = 0,

    /// <summary>Left + side stereo: channel 0 = L, channel 1 = L - R.</summary>
    LeftSide = 8,

    /// <summary>Right + side stereo: channel 0 = L - R, channel 1 = R.</summary>
    RightSide = 9,

    /// <summary>Mid + side stereo: channel 0 = (L+R)&gt;&gt;1 (floor), channel 1 = L - R.</summary>
    MidSide = 10,
}
