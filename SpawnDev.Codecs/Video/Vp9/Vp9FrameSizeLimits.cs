// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 frame dimension limits. The VP9 spec encodes frame dimensions
// as f(16) - 1 (subtract-one encoding), so the on-the-wire range
// is [0, 65535] which decodes to actual dimensions [1, 65536].
//
// libvpx in practice also imposes a soft limit on larger dimensions
// to keep buffer allocations reasonable, but the bitstream itself
// allows up to 65536x65536.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 frame dimension limits.</summary>
public static class Vp9FrameSizeLimits
{
    /// <summary>Minimum frame width in pixels per VP9 spec sec 6.2.5.</summary>
    public const int MinWidth = 1;

    /// <summary>Maximum frame width in pixels per VP9 spec sec 6.2.5.</summary>
    public const int MaxWidth = 65536;

    /// <summary>Minimum frame height in pixels.</summary>
    public const int MinHeight = 1;

    /// <summary>Maximum frame height in pixels.</summary>
    public const int MaxHeight = 65536;

    /// <summary>
    /// Validate that a (width, height) tuple is in the legal VP9
    /// range. Throws <see cref="ArgumentOutOfRangeException"/> on
    /// out-of-range values.
    /// </summary>
    public static void Validate(int width, int height)
    {
        if (width < MinWidth || width > MaxWidth)
            throw new ArgumentOutOfRangeException(nameof(width), width,
                $"VP9 frame width must be in [{MinWidth}, {MaxWidth}].");
        if (height < MinHeight || height > MaxHeight)
            throw new ArgumentOutOfRangeException(nameof(height), height,
                $"VP9 frame height must be in [{MinHeight}, {MaxHeight}].");
    }

    /// <summary>
    /// True when (<paramref name="width"/>, <paramref name="height"/>)
    /// is in the legal VP9 range without throwing.
    /// </summary>
    public static bool IsValid(int width, int height) =>
        width >= MinWidth && width <= MaxWidth &&
        height >= MinHeight && height <= MaxHeight;
}
