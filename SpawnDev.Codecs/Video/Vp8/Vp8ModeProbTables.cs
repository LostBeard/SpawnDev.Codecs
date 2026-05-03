// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 mode-probability tables extracted to main library so the GPU
// integration classes don't depend on the CPU reference Vp8ModeTrees
// (which lives in SpawnDev.Codecs.References per the 2026-05-03
// architectural directive).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 default keyframe mode probability tables (libvpx
/// <c>vp8_kf_ymode_prob</c> / <c>vp8_kf_uv_mode_prob</c>). These are
/// spec-defined constants from RFC 6386 section 16.2 (probability tables).
/// </summary>
public static class Vp8ModeProbTables
{
    /// <summary>Default keyframe Y mode probabilities (libvpx vp8_kf_ymode_prob).</summary>
    public static readonly byte[] DefaultKfYModeProb = new byte[] { 145, 156, 163, 128 };

    /// <summary>Default keyframe UV mode probabilities (libvpx vp8_kf_uv_mode_prob).</summary>
    public static readonly byte[] DefaultKfUvModeProb = new byte[] { 142, 114, 183 };
}
