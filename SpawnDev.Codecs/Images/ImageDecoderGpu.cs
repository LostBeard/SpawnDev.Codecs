// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// One entry point for "decode whatever image file this is": the format is sniffed from the first bytes and the
// call is forwarded to the format's decoder (created lazily, so a consumer that only ever sees PNGs never
// compiles the JPEG kernels). Same three-overload shape as every codec in this library.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images;

/// <summary>Decodes any supported image format on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class ImageDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private PngDecoderGpu? _png;
    private readonly StageBuffer<byte> _stageIn;

    /// <summary>Create a decoder; format decoders are compiled on first use.</summary>
    public ImageDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _stageIn = new StageBuffer<byte>(accelerator);
    }

    /// <summary>The format of a file from its first bytes (at least <see cref="ImageFormatDetector.SniffLength"/> if the file is that long).</summary>
    public static ImageFormat Detect(ReadOnlySpan<byte> head) => ImageFormatDetector.Detect(head);

    /// <summary>The PNG decoder (compiled on first access) for PNG-specific calls such as text chunks.</summary>
    public PngDecoderGpu Png => _png ??= new PngDecoderGpu(_accelerator);

    private static Exception NotSupported(ImageFormat format) => format == ImageFormat.Unknown
        ? new ImageDecodeException(ImageFormat.Unknown, "unrecognised image format")
        : new NotSupportedException($"Decoding {format} is not available in this build yet.");

    // ------------------------------------------------------------------
    // ArrayView1D overloads
    // ------------------------------------------------------------------

    private async Task<ImageFormat> DetectOnDeviceAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        int n = Math.Min(fileLength, ImageFormatDetector.SniffLength);
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(fileLength));
        byte[] head = await DeviceTransfer.DownloadAsync(file, n);   // a few bytes of header: metadata
        return ImageFormatDetector.Detect(head);
    }

    /// <summary>Read the facts about an image file that is on the accelerator (no pixels are decoded).</summary>
    public async Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var format = await DetectOnDeviceAsync(file, fileLength);
        switch (format)
        {
            case ImageFormat.Png: return await Png.ReadInfoAsync(file, fileLength);
            default: throw NotSupported(format);
        }
    }

    /// <summary>Decode an image file that is already on the accelerator into a caller-owned pixel view.</summary>
    public async Task<ImageInfo> DecodeAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
    {
        var detected = await DetectOnDeviceAsync(file, fileLength);
        switch (detected)
        {
            case ImageFormat.Png: return await Png.DecodeAsync(file, fileLength, output, format);
            default: throw NotSupported(detected);
        }
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    private void StageManaged(ReadOnlyMemory<byte> file)
    {
        if (file.IsEmpty) throw new ArgumentException("File is empty.", nameof(file));
        _stageIn.Ensure(file.Length);
        DeviceTransfer.Upload(_stageIn.View, file.Span);
    }

    /// <summary>Read the facts about a managed image file.</summary>
    public async Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file)
    {
        var format = ImageFormatDetector.Detect(file.Span.Slice(0, Math.Min(file.Length, ImageFormatDetector.SniffLength)));
        switch (format)
        {
            case ImageFormat.Png: return await Png.ReadInfoAsync(file);
            default: throw NotSupported(format);
        }
    }

    /// <summary>Decode a managed image file; returns the pixels as a managed array (a terminal CPU sink).</summary>
    public async Task<DecodedImage> DecodeAsync(ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8)
    {
        var detected = ImageFormatDetector.Detect(file.Span.Slice(0, Math.Min(file.Length, ImageFormatDetector.SniffLength)));
        switch (detected)
        {
            case ImageFormat.Png: return await Png.DecodeAsync(file, format);
            default: throw NotSupported(detected);
        }
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>Read the facts about an image file held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<ImageInfo> ReadInfoAsync(TypedArray file)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        return await ReadInfoAsync(_stageIn.View, checked((int)length));
    }

    /// <summary>
    /// Decode an image file held in a JS typed array (browser accelerators only); the pixels come back as a JS
    /// <see cref="Uint8Array"/> and never enter the .NET heap. The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedImageJS> DecodeAsync(TypedArray file, PixelFormat format = PixelFormat.Rgba8)
    {
        var head = await DetectTypedArrayAsync(file);
        switch (head)
        {
            case ImageFormat.Png: return await Png.DecodeAsync(file, format);
            default: throw NotSupported(head);
        }
    }

    [SupportedOSPlatform("browser")]
    private async Task<ImageFormat> DetectTypedArrayAsync(TypedArray file)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        return await DetectOnDeviceAsync(_stageIn.View, checked((int)length));
    }

    /// <summary>Release the format decoders and buffers.</summary>
    public void Dispose()
    {
        _png?.Dispose();
        _stageIn.Dispose();
    }
}
