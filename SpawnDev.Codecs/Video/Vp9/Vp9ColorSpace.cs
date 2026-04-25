// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 color range enum + color-space helpers. Mirror of libvpx
// vpx_image.h VPX_CS_* and VPX_CR_* constants, signalled by the
// color_config() section of the uncompressed header (VP9 spec sec
// 6.2.4).
//
// Vp9ColorSpace itself is defined alongside Vp9FrameHeader.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 color range (libvpx <c>VPX_CR_STUDIO_RANGE</c> / <c>VPX_CR_FULL_RANGE</c>).</summary>
public enum Vp9ColorRange : byte
{
    /// <summary>
    /// Studio range: Y in [16, 235], U/V in [16, 240] for 8-bit
    /// (scaled appropriately for 10/12-bit).
    /// </summary>
    Studio = 0,
    /// <summary>Full range: Y/U/V cover the full [0, max] range.</summary>
    Full = 1,
}

/// <summary>VP9 color-space helpers.</summary>
public static class Vp9ColorSpaces
{
    /// <summary>
    /// True for color spaces that imply RGB sample storage (no chroma
    /// subsampling). VP9 only defines this for <see cref="Vp9ColorSpace.Srgb"/>.
    /// </summary>
    public static bool IsRgb(Vp9ColorSpace cs) => cs == Vp9ColorSpace.Srgb;

    /// <summary>
    /// Color range is unconditionally Full for RGB color space (VP9
    /// spec sec 6.2.4: when color_space == CS_RGB, color_range is
    /// implicitly full).
    /// </summary>
    public static bool ImpliesFullRange(Vp9ColorSpace cs) => IsRgb(cs);
}
