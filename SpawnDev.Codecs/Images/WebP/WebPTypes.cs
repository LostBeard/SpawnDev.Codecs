// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WebP-specific facts and options that do not fit the format-neutral ImageInfo.

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>Everything the RIFF container says about a WebP file, without decoding any pixels.</summary>
/// <param name="Info">The format-neutral facts (size, alpha, frame count, EXIF orientation).</param>
/// <param name="IsLossless">True when every image in the file is VP8L (an animation: every frame).</param>
/// <param name="IsExtended">True when the file has a VP8X chunk.</param>
/// <param name="HasIccProfile">True when an ICCP chunk is present.</param>
/// <param name="HasExif">True when an EXIF chunk is present.</param>
/// <param name="HasXmp">True when an XMP chunk is present.</param>
/// <param name="LoopCount">ANIM loop count (0 = forever; 0 for a still).</param>
/// <param name="BackgroundColor">ANIM background colour, bytes B, G, R, A read as a little-endian uint (0 for a still).
/// libwebp's animation decoder ignores it (the canvas starts transparent) and so does <see cref="WebPDecoderGpu"/>.</param>
public sealed record WebPFileInfo(
    ImageInfo Info, bool IsLossless, bool IsExtended, bool HasIccProfile, bool HasExif, bool HasXmp,
    int LoopCount, uint BackgroundColor);

/// <summary>The ICCP / EXIF / XMP chunk payloads of a WebP file (small host blobs; null when absent).</summary>
public sealed record WebPMetadata(byte[]? IccProfile, byte[]? Exif, byte[]? Xmp);

/// <summary>Options of <see cref="WebPLosslessEncoderGpu"/>.</summary>
public sealed class WebPEncodeOptions
{
    /// <summary>
    /// Effort / compression quality 0..100 (default 75), the same meaning as libwebp's lossless quality: higher spends more
    /// work for a smaller file. 0..24 uses one prefix group and no transforms beyond the palette, 25..49 adds subtract
    /// green and the predictor, 50..74 adds the colour cache and LZ77 hash matching, 75..100 searches harder
    /// (more predictor candidates, longer match chains, cache size search).
    /// </summary>
    public int Quality { get; set; } = 75;

    /// <summary>
    /// Preserve the RGB values of fully transparent pixels (default true). When false the encoder is free to replace them
    /// (it zeroes RGB under alpha 0, which compresses better), like libwebp without <c>exact</c>.
    /// </summary>
    public bool Exact { get; set; } = true;

    /// <summary>ICC colour profile written as an ICCP chunk (null = none).</summary>
    public byte[]? IccProfile { get; set; }

    /// <summary>EXIF data (TIFF structure, with or without the "Exif\0\0" prefix) written as an EXIF chunk (null = none).</summary>
    public byte[]? Exif { get; set; }

    /// <summary>XMP packet written as an XMP chunk (null = none).</summary>
    public byte[]? Xmp { get; set; }
}
