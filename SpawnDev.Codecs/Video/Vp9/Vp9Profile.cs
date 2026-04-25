// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 profile enum + helpers. Mirror of libvpx vp9/common/vp9_enums.h
// BITSTREAM_PROFILE.
//
// Profile semantics per VP9 spec sec 7.2:
//   Profile 0: 8-bit, 4:2:0 only
//   Profile 1: 8-bit, 4:2:2 / 4:4:4 / 4:4:0 (any non-420 sampling)
//   Profile 2: 10/12-bit, 4:2:0 only
//   Profile 3: 10/12-bit, 4:2:2 / 4:4:4 / 4:4:0
//
// Bit depth selection within Profile 2/3 is signalled by an extra
// header bit; profile alone tells you 8 vs >8.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 bitstream profile (libvpx <c>BITSTREAM_PROFILE</c>).</summary>
public enum Vp9Profile : byte
{
    /// <summary>8-bit, 4:2:0 only.</summary>
    Profile0 = 0,
    /// <summary>8-bit, non-4:2:0 chroma sampling allowed.</summary>
    Profile1 = 1,
    /// <summary>10 or 12-bit, 4:2:0 only.</summary>
    Profile2 = 2,
    /// <summary>10 or 12-bit, non-4:2:0 chroma sampling allowed.</summary>
    Profile3 = 3,
}

/// <summary>VP9 profile capability helpers.</summary>
public static class Vp9Profiles
{
    /// <summary>libvpx <c>MAX_PROFILES</c>.</summary>
    public const int Count = 4;

    /// <summary>True for profiles that allow &gt; 8-bit sample depth.</summary>
    public static bool IsHighBitDepth(Vp9Profile profile) =>
        profile == Vp9Profile.Profile2 || profile == Vp9Profile.Profile3;

    /// <summary>
    /// True for profiles that allow non-4:2:0 chroma sampling
    /// (4:2:2 / 4:4:4 / 4:4:0). The exact subsampling pair is
    /// signalled separately in the bitstream.
    /// </summary>
    public static bool AllowsNonYuv420(Vp9Profile profile) =>
        profile == Vp9Profile.Profile1 || profile == Vp9Profile.Profile3;

    /// <summary>
    /// True for the most permissive profile (10/12-bit + non-4:2:0).
    /// </summary>
    public static bool IsMostPermissive(Vp9Profile profile) =>
        profile == Vp9Profile.Profile3;
}
