// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// BMP decoder, 100% accelerator-resident. The host is a pure coordinator: it uploads the file (necessary I/O),
// dispatches the BmpKernels chain, and reads back the metadata block (a few dozen ints). The file bytes and the
// RGBA pixels never touch the host unless the caller asks for managed output.
//
// API shape (same as the PNG decoder): three overloads per entry point, all funnelling into the ArrayView1D one.
//   - ArrayView1D : file already on the accelerator, RGBA pixels written into a caller-owned view.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, JS Uint8Array pixels out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; pixels come back as byte[].
//
// Supported: BITMAPCOREHEADER (OS/2 1.x), BITMAPINFOHEADER and the V2 / V3 / V4 / V5 headers; 1, 4, 8 bit palette
// images (palettes shorter than 2^bpp allowed: out-of-range indices are opaque black); 16 bit (RGB555, RGB565,
// any BI_BITFIELDS layout); 24 bit; 32 bit (BI_RGB and BI_BITFIELDS / BI_ALPHABITFIELDS, any contiguous masks);
// RLE4 and RLE8; bottom-up and top-down (negative height) files; rows padded to 4 bytes.
//
// Conventions (documented choices):
//   * Channels narrower than 8 bits expand by bit replication ((v << 3) | (v >> 2) for 5 bits); 16 bit data
//     with no masks is RGB555.
//   * Alpha: a 32-bit BI_RGB image carries alpha in its 4th byte, and so does any image whose BITFIELDS masks
//     include an alpha mask. If every alpha value of such an image is 0 the image is treated as opaque (the file
//     comes from a writer that stores padding there; browsers do the same). Images with no alpha channel are opaque.
//   * RLE pixels not covered by the stream (skipped by delta / end-of-line codes) are transparent black.
//   * Not supported (specific ImageDecodeException): BI_JPEG, BI_PNG, OS/2 2.x headers and Huffman / RLE24
//     compression, bit depths other than 1/4/8/16/24/32.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Bmp;

/// <summary>BMP decoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance (the metadata block and staging buffers are shared).</remarks>
public sealed class BmpDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly BmpKernels _kernels;
    private readonly ImagePixelKernels _pixel;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<byte> _canvas;

    /// <summary>Compile the kernels.</summary>
    public BmpDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new BmpKernels(accelerator);
        _pixel = new ImagePixelKernels(accelerator);
        _meta = accelerator.Allocate1D<int>(BmpMeta.Size);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _canvas = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Layout derived from the header facts (pure metadata arithmetic, 64-bit)
    // ------------------------------------------------------------------

    private sealed class Layout
    {
        public int Width, Height, Bpp, Compression, HeaderSize;
        public bool TopDown, IsRle, HasAlphaChannel;
        public readonly int[] HostSlots = new int[BmpMeta.Size - BmpMeta.HostWrittenStart];

        public ImageInfo ToInfo() => new(
            ImageFormat.Bmp, Width, Height, HasAlphaChannel || IsRle, Bpp <= 8 ? Bpp : 8);
    }

    private static string DescribeScanError(int[] m) => (BmpError)m[BmpMeta.Error] switch
    {
        BmpError.BadSignature => "not a BMP file (bad signature)",
        BmpError.TruncatedHeader => "file is truncated (ends inside the headers)",
        BmpError.UnsupportedHeader => $"unsupported DIB header size {m[BmpMeta.HeaderSize]}",
        var e => e.ToString(),
    };

    private static void SetSlot(Layout l, int slot, int value) => l.HostSlots[slot - BmpMeta.HostWrittenStart] = value;

    /// <summary>Width and shift of a contiguous channel mask; throws when it is not contiguous.</summary>
    private static (int shift, int bits) ParseMask(uint mask, string name, int bpp)
    {
        if (mask == 0) return (0, 0);
        int shift = System.Numerics.BitOperations.TrailingZeroCount(mask);
        uint run = mask >> shift;
        if ((run & (run + 1)) != 0)
            throw new ImageDecodeException(ImageFormat.Bmp, $"{name} channel mask 0x{mask:X8} is not contiguous");
        if (bpp == 16 && mask > 0xFFFF)
            throw new ImageDecodeException(ImageFormat.Bmp, $"{name} channel mask 0x{mask:X8} does not fit a 16 bit pixel");
        return (shift, System.Numerics.BitOperations.PopCount(mask));
    }

    private static Layout ComputeLayout(int[] m)
    {
        if ((BmpError)m[BmpMeta.Error] != BmpError.None)
            throw new ImageDecodeException(ImageFormat.Bmp, DescribeScanError(m));

        var l = new Layout();
        l.HeaderSize = m[BmpMeta.HeaderSize];
        int rawHeight = m[BmpMeta.Height];
        long fileLength = m[BmpMeta.FileLength];
        long offBits = (uint)m[BmpMeta.OffBits];
        l.Width = m[BmpMeta.Width];
        l.TopDown = rawHeight < 0;
        long absHeight = Math.Abs((long)rawHeight);
        l.Bpp = m[BmpMeta.Bpp];
        l.Compression = m[BmpMeta.Compression];

        if (l.Width <= 0 || absHeight <= 0 || absHeight > int.MaxValue)
            throw new ImageDecodeException(ImageFormat.Bmp, $"invalid image size {l.Width}x{rawHeight}");
        if ((long)l.Width * absHeight * 4 > int.MaxValue)
            throw new ImageDecodeException(ImageFormat.Bmp, $"image {l.Width}x{absHeight} is too large");
        l.Height = (int)absHeight;
        if (m[BmpMeta.Planes] != 1)
            throw new ImageDecodeException(ImageFormat.Bmp, $"invalid plane count {m[BmpMeta.Planes]} (must be 1)");

        int bpp = l.Bpp;
        int comp = l.Compression;
        switch (comp)
        {
            case 0:
                if (bpp is not (1 or 4 or 8 or 16 or 24 or 32))
                    throw new ImageDecodeException(ImageFormat.Bmp, $"unsupported bit depth {bpp}");
                break;
            case 1:
                if (bpp != 8) throw new ImageDecodeException(ImageFormat.Bmp, $"RLE8 compression with {bpp} bits per pixel");
                l.IsRle = true;
                break;
            case 2:
                if (bpp != 4) throw new ImageDecodeException(ImageFormat.Bmp, $"RLE4 compression with {bpp} bits per pixel");
                l.IsRle = true;
                break;
            case 3:
            case 6:
                if (bpp is not (16 or 32))
                    throw new ImageDecodeException(ImageFormat.Bmp, $"BITFIELDS compression with {bpp} bits per pixel (only 16 and 32 are valid)");
                break;
            case 4:
                throw new ImageDecodeException(ImageFormat.Bmp, "unsupported compression BI_JPEG");
            case 5:
                throw new ImageDecodeException(ImageFormat.Bmp, "unsupported compression BI_PNG");
            default:
                throw new ImageDecodeException(ImageFormat.Bmp, $"unsupported compression {comp}");
        }
        if (l.IsRle && l.TopDown)
            throw new ImageDecodeException(ImageFormat.Bmp, "RLE bitmaps cannot be top-down");

        if (offBits < 14 + (long)l.HeaderSize)
            throw new ImageDecodeException(ImageFormat.Bmp, $"pixel data offset {offBits} lies inside the headers");

        // Palette.
        int entrySize = l.HeaderSize == 12 ? 3 : 4;
        int palCount = 0;
        if (bpp <= 8)
        {
            long max = 1L << bpp;
            long clr = (uint)m[BmpMeta.ClrUsed];
            palCount = (int)(clr == 0 || clr > max ? max : clr);
            long palEnd = (long)m[BmpMeta.PaletteOffset] + (long)palCount * entrySize;
            if (palEnd > fileLength)
                throw new ImageDecodeException(ImageFormat.Bmp, "file is truncated (ends inside the palette)");
            if (palEnd > offBits)
                throw new ImageDecodeException(ImageFormat.Bmp, "palette overlaps the pixel data");
        }
        if (offBits >= fileLength)
            throw new ImageDecodeException(ImageFormat.Bmp, $"file is truncated (pixel data offset {offBits}, file is {fileLength} bytes)");

        // Channel layout of 16 / 32 bit pixels.
        uint maskR = 0, maskG = 0, maskB = 0, maskA = 0;
        if (bpp == 16 && comp == 0)
        {
            maskR = 0x7C00; maskG = 0x03E0; maskB = 0x001F;
        }
        else if (bpp == 32 && comp == 0)
        {
            maskR = 0x00FF0000; maskG = 0x0000FF00; maskB = 0x000000FF; maskA = 0xFF000000;
        }
        else if (bpp >= 16 && comp != 0)
        {
            maskR = (uint)m[BmpMeta.MaskR]; maskG = (uint)m[BmpMeta.MaskG];
            maskB = (uint)m[BmpMeta.MaskB]; maskA = (uint)m[BmpMeta.MaskA];
            if (maskR == 0 || maskG == 0 || maskB == 0)
                throw new ImageDecodeException(ImageFormat.Bmp, "BITFIELDS image with a zero red, green or blue mask");
        }
        var (shR, bitsR) = ParseMask(maskR, "red", bpp);
        var (shG, bitsG) = ParseMask(maskG, "green", bpp);
        var (shB, bitsB) = ParseMask(maskB, "blue", bpp);
        var (shA, bitsA) = ParseMask(maskA, "alpha", bpp);
        l.HasAlphaChannel = bitsA > 0;

        // Pixel data must lie inside the file.
        if (!l.IsRle)
        {
            long stride = (((long)l.Width * bpp + 31) / 32) * 4;
            long needed = stride * l.Height;
            if (offBits + needed > fileLength)
                throw new ImageDecodeException(ImageFormat.Bmp,
                    $"file is truncated (pixel data needs {needed} bytes at offset {offBits}, file is {fileLength} bytes)");
            SetSlot(l, BmpMeta.Stride, (int)stride);
        }
        SetSlot(l, BmpMeta.TopDown, l.TopDown ? 1 : 0);
        SetSlot(l, BmpMeta.PaletteCount, palCount);
        SetSlot(l, BmpMeta.EntrySize, entrySize);
        SetSlot(l, BmpMeta.ShiftR, shR); SetSlot(l, BmpMeta.BitsR, bitsR);
        SetSlot(l, BmpMeta.ShiftG, shG); SetSlot(l, BmpMeta.BitsG, bitsG);
        SetSlot(l, BmpMeta.ShiftB, shB); SetSlot(l, BmpMeta.BitsB, bitsB);
        SetSlot(l, BmpMeta.ShiftA, shA); SetSlot(l, BmpMeta.BitsA, bitsA);
        SetSlot(l, BmpMeta.HasAlphaChannel, l.HasAlphaChannel ? 1 : 0);
        SetSlot(l, BmpMeta.AlphaSeen, 0);
        SetSlot(l, BmpMeta.AbsHeight, l.Height);
        // The effective masks (the file's, or the BI_RGB defaults) are what the pixel kernels read.
        SetSlot(l, BmpMeta.EffMaskR, (int)maskR);
        SetSlot(l, BmpMeta.EffMaskG, (int)maskG);
        SetSlot(l, BmpMeta.EffMaskB, (int)maskB);
        SetSlot(l, BmpMeta.EffMaskA, (int)maskA);
        return l;
    }

    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------

    private async Task<int[]> ScanAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        if (fileLength <= 0 || fileLength > file.Length) throw new ArgumentOutOfRangeException(nameof(fileLength));
        _kernels.Scan(file, fileLength, _meta.View);
        await _accelerator.SynchronizeAsync();
        return await _meta.View.CopyToHostAsync();
    }

    /// <summary>
    /// Read the facts about a BMP that is on the accelerator (header scan + a metadata-sized readback; no
    /// pixels are decoded). Use the result to size the output buffer for
    /// <see cref="DecodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{byte, Stride1D.Dense}, PixelFormat)"/>.
    /// </summary>
    public async Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var meta = await ScanAsync(file, fileLength);
        return ComputeLayout(meta).ToInfo();
    }

    /// <summary>
    /// Decode a BMP that is already on the accelerator into a caller-owned pixel view. Only metadata scalars
    /// cross to the host.
    /// </summary>
    /// <param name="file">The BMP file bytes (may be longer than <paramref name="fileLength"/>).</param>
    /// <param name="fileLength">Number of valid bytes in <paramref name="file"/>.</param>
    /// <param name="output">Receives the tightly packed pixels; at least <see cref="ImageInfo.GetPixelBufferSize"/> bytes.</param>
    /// <param name="format">Pixel layout to produce (Rgba16 widens each 8-bit sample v to v * 257).</param>
    /// <returns>The facts about the image.</returns>
    /// <exception cref="ImageDecodeException">The file is corrupt, truncated or uses an unsupported feature.</exception>
    public async Task<ImageInfo> DecodeAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
    {
        var meta = await ScanAsync(file, fileLength);
        var layout = ComputeLayout(meta);
        var info = layout.ToInfo();
        long outBytes = info.GetPixelBufferSize(format);
        if (output.Length < outBytes)
            throw new ArgumentException($"output must hold {outBytes} bytes for {info.Width}x{info.Height} {format}.", nameof(output));
        await RunPipelineAsync(file, layout, info, output, format);
        return info;
    }

    private async Task<ImageInfo> DecodeToStageAsync(int fileLength, PixelFormat format)
    {
        var meta = await ScanAsync(_stageIn.View, fileLength);
        var layout = ComputeLayout(meta);
        var info = layout.ToInfo();
        _stageOut.Ensure(info.GetPixelBufferSize(format));
        await RunPipelineAsync(_stageIn.View, layout, info, _stageOut.View, format);
        return info;
    }

    private async Task RunPipelineAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout layout, ImageInfo info,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format)
    {
        int pixels = info.Width * info.Height;
        var canvas = format == PixelFormat.Rgba8 ? output : _canvas.Ensure((long)pixels * 4);

        _meta.View.SubView(BmpMeta.HostWrittenStart, layout.HostSlots.Length)
             .CopyFromCPU(ref layout.HostSlots[0], layout.HostSlots.Length);

        if (layout.IsRle)
        {
            // Pixels the stream never covers stay transparent black.
            canvas.SubView(0, (long)pixels * 4).MemSetToZero();
            _kernels.Rle(file, canvas, _meta.View);
        }
        else
        {
            if (layout.HasAlphaChannel) _kernels.AlphaScan(file, _meta.View, pixels);
            _kernels.Convert(file, canvas, _meta.View, pixels);
        }
        if (format == PixelFormat.Rgba16) _pixel.Widen(canvas, output, pixels * 4);
        await _accelerator.SynchronizeAsync();

        int status = (await _meta.View.SubView(BmpMeta.Error, 1).CopyToHostAsync())[0];
        if (status == (int)BmpError.RleTruncated)
            throw new ImageDecodeException(ImageFormat.Bmp, "RLE data is truncated (ends inside an instruction)");
        if (status == (int)BmpError.RleOutOfBounds)
            throw new ImageDecodeException(ImageFormat.Bmp, "RLE data draws outside the image");
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a managed BMP file (no pixels are decoded).</summary>
    public async Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadInfoAsync(_stageIn.View, file.Length);
    }

    /// <summary>
    /// Decode a managed BMP file. Returns the pixels as a managed array (a terminal CPU sink; in a browser
    /// prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<DecodedImage> DecodeAsync(ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8)
    {
        StageManaged(file);
        var info = await DecodeToStageAsync(file.Length, format);
        var pixels = await DeviceTransfer.DownloadAsync(_stageOut.View, info.GetPixelBufferSize(format));
        return new DecodedImage(info, pixels, format);
    }

    /// <summary>Not async: it holds a span.</summary>
    private void StageManaged(ReadOnlyMemory<byte> file)
    {
        if (file.IsEmpty) throw new ArgumentException("File is empty.", nameof(file));
        _stageIn.Ensure(file.Length);
        DeviceTransfer.Upload(_stageIn.View, file.Span);
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a BMP held in a JS typed array (browser accelerators only).</summary>
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
    /// Decode a BMP held in a JS typed array (browser accelerators only). The bytes go to the GPU with
    /// <c>CopyFromJS</c> and the pixels come back as a JS <see cref="Uint8Array"/>; none of it enters the
    /// .NET heap. The caller disposes the result.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedImageJS> DecodeAsync(TypedArray file, PixelFormat format = PixelFormat.Rgba8)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        var info = await DecodeToStageAsync(checked((int)length), format);
        var pixels = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, info.GetPixelBufferSize(format));
        return new DecodedImageJS(info, pixels, format);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _pixel.Dispose();
        _meta.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
        _canvas.Dispose();
    }
}
