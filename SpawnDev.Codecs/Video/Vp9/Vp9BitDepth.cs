// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 bit depth enum + resolver. Mirror of libvpx vpx_bit_depth_t.
//
// Bit depth in the VP9 bitstream:
//   Profile 0 / 1 -> always 8-bit (no bit_depth bit in header)
//   Profile 2 / 3 -> read 1 bit:
//                    0 -> 10-bit
//                    1 -> 12-bit
//
// Sample storage:
//   Bits8  : uint8_t per pixel
//   Bits10 : uint16_t per pixel (low 10 bits used)
//   Bits12 : uint16_t per pixel (low 12 bits used)
//
// Higher bit depth raises MAXQ (the dequantizer ceiling) from 255
// at 8-bit to 1023 at 10-bit and 4095 at 12-bit.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 sample bit depth.</summary>
public enum Vp9BitDepth : byte
{
    /// <summary>8-bit samples (Profile 0 / 1).</summary>
    Bits8 = 8,
    /// <summary>10-bit samples (Profile 2 / 3 with bit_depth flag = 0).</summary>
    Bits10 = 10,
    /// <summary>12-bit samples (Profile 2 / 3 with bit_depth flag = 1).</summary>
    Bits12 = 12,
}

/// <summary>VP9 bit depth helpers.</summary>
public static class Vp9BitDepths
{
    /// <summary>
    /// Resolve the bit depth from profile + the high-bit-depth
    /// signal bit. <paramref name="tenOrTwelveBit"/> is ignored for
    /// profiles 0 / 1; for profiles 2 / 3 it picks 10 (false) or
    /// 12 (true) bits.
    /// </summary>
    public static Vp9BitDepth Resolve(Vp9Profile profile, bool tenOrTwelveBit)
    {
        if (!Vp9Profiles.IsHighBitDepth(profile))
            return Vp9BitDepth.Bits8;
        return tenOrTwelveBit ? Vp9BitDepth.Bits12 : Vp9BitDepth.Bits10;
    }

    /// <summary>
    /// Maximum sample value for the given bit depth: 255, 1023, 4095.
    /// </summary>
    public static int MaxSampleValue(Vp9BitDepth depth) =>
        (1 << (int)depth) - 1;

    /// <summary>
    /// Maximum quantizer index (libvpx <c>MAXQ</c>) for the given bit
    /// depth: 255 at 8-bit, 1023 at 10-bit, 4095 at 12-bit.
    /// </summary>
    public static int MaxQuantizerIndex(Vp9BitDepth depth) => depth switch
    {
        Vp9BitDepth.Bits8 => 255,
        Vp9BitDepth.Bits10 => 1023,
        Vp9BitDepth.Bits12 => 4095,
        _ => throw new ArgumentOutOfRangeException(nameof(depth), depth, "Unsupported bit depth."),
    };
}
