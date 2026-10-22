// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Shared vocabulary for the image codecs (PNG, JPEG, WebP, GIF, BMP, AVIF, ...). Every image decoder
// produces the same thing: a tightly packed RGBA pixel buffer on the accelerator plus an ImageInfo.
// Every image encoder consumes the same thing.

using System.Runtime.Versioning;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images;

/// <summary>Container / codec of an image file.</summary>
public enum ImageFormat
{
    /// <summary>Not recognised.</summary>
    Unknown = 0,
    /// <summary>Portable Network Graphics (including APNG).</summary>
    Png,
    /// <summary>JPEG (baseline, extended sequential, progressive).</summary>
    Jpeg,
    /// <summary>WebP (lossy, lossless, alpha, animated).</summary>
    WebP,
    /// <summary>GIF (87a / 89a, animated).</summary>
    Gif,
    /// <summary>Windows bitmap.</summary>
    Bmp,
    /// <summary>AVIF (AV1 still / sequence in an ISO-BMFF container).</summary>
    Avif,
}

/// <summary>Layout of a decoded / to-be-encoded pixel buffer.</summary>
public enum PixelFormat
{
    /// <summary>4 bytes per pixel: R, G, B, A, 8 bits each, straight (non-premultiplied) alpha, rows top to bottom, no padding.</summary>
    Rgba8 = 0,
    /// <summary>8 bytes per pixel: R, G, B, A as little-endian 16-bit values, straight alpha, rows top to bottom, no padding.</summary>
    Rgba16 = 1,
}

/// <summary>Static facts about an image file, available without decoding the pixels.</summary>
/// <param name="Format">The container / codec.</param>
/// <param name="Width">Width in pixels (of the first frame / the canvas).</param>
/// <param name="Height">Height in pixels.</param>
/// <param name="HasAlpha">True when the file can carry transparency (alpha channel, palette alpha, colour key).</param>
/// <param name="BitDepth">Bits per channel in the file (1, 2, 4, 8, 16).</param>
/// <param name="FrameCount">1 for a still image; the number of frames for an animation.</param>
/// <param name="Orientation">EXIF orientation 1..8 (1 = upright). Decoders do NOT apply it unless asked.</param>
/// <param name="IsInterlaced">PNG Adam7 / progressive JPEG / interlaced GIF.</param>
public sealed record ImageInfo(
    ImageFormat Format, int Width, int Height, bool HasAlpha, int BitDepth,
    int FrameCount = 1, int Orientation = 1, bool IsInterlaced = false)
{
    /// <summary>Bytes per pixel of <paramref name="format"/>.</summary>
    public static int BytesPerPixel(PixelFormat format) => format == PixelFormat.Rgba16 ? 8 : 4;

    /// <summary>Size in bytes of a tightly packed <paramref name="format"/> buffer for one frame of this image.</summary>
    public long GetPixelBufferSize(PixelFormat format = PixelFormat.Rgba8) =>
        (long)Width * Height * BytesPerPixel(format);
}

/// <summary>A decoded image in managed memory (a terminal CPU sink; in a browser prefer <see cref="DecodedImageJS"/>).</summary>
/// <param name="Info">Facts about the file.</param>
/// <param name="Pixels">Tightly packed pixels in <paramref name="PixelFormat"/>.</param>
/// <param name="PixelFormat">Layout of <paramref name="Pixels"/>.</param>
public sealed record DecodedImage(ImageInfo Info, byte[] Pixels, PixelFormat PixelFormat);

/// <summary>
/// A decoded image as a JS <see cref="Uint8Array"/> (browser). The caller disposes it; the pixels
/// never entered the .NET heap.
/// </summary>
[SupportedOSPlatform("browser")]
public sealed class DecodedImageJS : IDisposable
{
    /// <summary>Facts about the file.</summary>
    public ImageInfo Info { get; }
    /// <summary>Tightly packed pixels in <see cref="PixelFormat"/>.</summary>
    public Uint8Array Pixels { get; }
    /// <summary>Layout of <see cref="Pixels"/>.</summary>
    public PixelFormat PixelFormat { get; }

    internal DecodedImageJS(ImageInfo info, Uint8Array pixels, PixelFormat pixelFormat)
    {
        Info = info; Pixels = pixels; PixelFormat = pixelFormat;
    }

    /// <summary>Dispose the pixel array.</summary>
    public void Dispose() => Pixels.Dispose();
}

/// <summary>A file the decoder understands structurally but cannot decode (corrupt, truncated, unsupported feature).</summary>
public sealed class ImageDecodeException : Exception
{
    /// <summary>Create the exception.</summary>
    public ImageDecodeException(ImageFormat format, string message) : base($"{format}: {message}")
    {
        Format = format;
    }

    /// <summary>The format being decoded.</summary>
    public ImageFormat Format { get; }
}

/// <summary>Identifies an image file from its first bytes.</summary>
public static class ImageFormatDetector
{
    /// <summary>Minimum number of leading bytes needed to recognise every supported format.</summary>
    public const int SniffLength = 32;

    /// <summary>Returns the format whose magic number <paramref name="head"/> starts with, or <see cref="ImageFormat.Unknown"/>.</summary>
    public static ImageFormat Detect(ReadOnlySpan<byte> head)
    {
        if (head.Length >= 8 && head[0] == 0x89 && head[1] == 0x50 && head[2] == 0x4E && head[3] == 0x47
            && head[4] == 0x0D && head[5] == 0x0A && head[6] == 0x1A && head[7] == 0x0A)
            return ImageFormat.Png;
        if (head.Length >= 3 && head[0] == 0xFF && head[1] == 0xD8 && head[2] == 0xFF)
            return ImageFormat.Jpeg;
        if (head.Length >= 12 && head[0] == 'R' && head[1] == 'I' && head[2] == 'F' && head[3] == 'F'
            && head[8] == 'W' && head[9] == 'E' && head[10] == 'B' && head[11] == 'P')
            return ImageFormat.WebP;
        if (head.Length >= 6 && head[0] == 'G' && head[1] == 'I' && head[2] == 'F' && head[3] == '8'
            && (head[4] == '7' || head[4] == '9') && head[5] == 'a')
            return ImageFormat.Gif;
        if (head.Length >= 2 && head[0] == 'B' && head[1] == 'M')
            return ImageFormat.Bmp;
        if (head.Length >= 12 && head[4] == 'f' && head[5] == 't' && head[6] == 'y' && head[7] == 'p'
            && IsAvifBrand(head.Slice(8)))
            return ImageFormat.Avif;
        return ImageFormat.Unknown;
    }

    private static bool IsAvifBrand(ReadOnlySpan<byte> brandAndCompat)
    {
        // Major brand "avif" / "avis"; compatible brands are scanned by the AVIF decoder itself.
        return brandAndCompat.Length >= 4 && brandAndCompat[0] == 'a' && brandAndCompat[1] == 'v'
            && brandAndCompat[2] == 'i' && (brandAndCompat[3] == 'f' || brandAndCompat[3] == 's');
    }
}
