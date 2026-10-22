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

/// <summary>The alpha plane (ALPH chunk) of a lossy WebP in managed memory: one byte per pixel, rows top to bottom.</summary>
/// <param name="Width">Plane width (the image or frame width).</param>
/// <param name="Height">Plane height.</param>
/// <param name="Alpha">Width x Height bytes.</param>
public sealed record WebPAlphaPlane(int Width, int Height, byte[] Alpha);

/// <summary>An alpha plane as a JS <see cref="Uint8Array"/> (browser). The caller disposes it; the bytes never entered the .NET heap.</summary>
[System.Runtime.Versioning.SupportedOSPlatform("browser")]
public sealed class WebPAlphaPlaneJS : IDisposable
{
    /// <summary>Plane width.</summary>
    public int Width { get; }
    /// <summary>Plane height.</summary>
    public int Height { get; }
    /// <summary>Width x Height bytes.</summary>
    public SpawnDev.SpawnJS.JSObjects.Uint8Array Alpha { get; }

    internal WebPAlphaPlaneJS(int width, int height, SpawnDev.SpawnJS.JSObjects.Uint8Array alpha)
    {
        Width = width; Height = height; Alpha = alpha;
    }

    /// <summary>Dispose the plane.</summary>
    public void Dispose() => Alpha.Dispose();
}

/// <summary>The ICCP / EXIF / XMP chunk payloads of a WebP file (small host blobs; null when absent).</summary>
public sealed record WebPMetadata(byte[]? IccProfile, byte[]? Exif, byte[]? Xmp);

/// <summary>Options of <see cref="WebPLosslessEncoderGpu"/>.</summary>
public class WebPEncodeOptions
{
    /// <summary>
    /// Effort 0..100 (default 75), the same meaning as libwebp's lossless quality: higher spends more work for a smaller file.
    /// 0..24: palette when the image has at most 256 colours, otherwise subtract green + predictor (5 candidate modes),
    /// LZ77 against the previous pixel / the row above / one hash candidate. 25..49: a longer hash chain, and an image with at
    /// most 256 colours also tries the palette with a predictor. 50..74: the encoder tries several ways to transform the image
    /// (none, green + predictor, + cross colour, palette, palette + predictor) and keeps the one that entropy-codes smallest, and
    /// also tries a colour cache. 75..100: all 14 predictor modes, longer hash chains and
    /// more colour cache sizes.
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

/// <summary>The ALPH chunk's prediction filter: what each alpha sample is predicted from before it is entropy coded.</summary>
public enum WebPAlphaFilter
{
    /// <summary>No prediction.</summary>
    None = 0,
    /// <summary>The sample to the left (the first column from the sample above).</summary>
    Horizontal = 1,
    /// <summary>The sample above (the first row from the sample to the left).</summary>
    Vertical = 2,
    /// <summary>clip(left + above - aboveLeft).</summary>
    Gradient = 3,
    /// <summary>
    /// The accelerator picks: at effort 50 and above it encodes the plane with each of the four filters and keeps the smallest
    /// stream; below, a fast estimate of the residual cost chooses.
    /// </summary>
    Auto = 4,
}

/// <summary>Options of <see cref="WebPLosslessEncoderGpu.EncodeAlphaAsync(ILGPU.ArrayView1D{byte, ILGPU.Stride1D.Dense}, int, int, ILGPU.ArrayView1D{byte, ILGPU.Stride1D.Dense}, WebPAlphaEncodeOptions?)"/>.</summary>
public sealed class WebPAlphaEncodeOptions
{
    /// <summary>Effort 0..100 of the VP8L coder (default 75); same meaning as <see cref="WebPEncodeOptions.Quality"/>.</summary>
    public int Quality { get; set; } = 75;

    /// <summary>The prediction filter (default <see cref="WebPAlphaFilter.Auto"/>).</summary>
    public WebPAlphaFilter Filter { get; set; } = WebPAlphaFilter.Auto;

    /// <summary>
    /// Compress the filtered plane with the lossless coder (default true; the plane is stored raw instead when that is not
    /// larger). False always stores it raw (compression method 0).
    /// </summary>
    public bool Compress { get; set; } = true;
}

/// <summary>Options of the animated <see cref="WebPLosslessEncoderGpu"/> encode: <see cref="WebPEncodeOptions"/> apply to every frame.</summary>
public sealed class WebPAnimationEncodeOptions : WebPEncodeOptions
{
    /// <summary>How many times the animation plays (0 = forever, the default).</summary>
    public int LoopCount { get; set; }

    /// <summary>
    /// ANIM background colour, bytes B, G, R, A read as a little-endian uint (0xAARRGGBB; default 0). Advisory:
    /// libwebp's animation decoder starts from a transparent canvas and ignores it.
    /// </summary>
    public uint BackgroundColor { get; set; }
}
