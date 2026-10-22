// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// PNG decoder, 100% accelerator-resident. The host is a pure coordinator: it uploads the file (necessary
// I/O), dispatches the PngKernels chain, and reads back a few scalars (the metadata block and the inflate
// result). The compressed bytes, the inflated scanlines and the RGBA pixels never touch the host unless the
// caller asks for managed output.
//
// API shape (same as the video codecs): three overloads per entry point, all funnelling into the
// ArrayView1D one.
//   - ArrayView1D : file already on the accelerator, RGBA pixels written into a caller-owned view.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, JS Uint8Array pixels out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; pixels come back as byte[].
//
// Supported: every PNG colour type and bit depth (1/2/4/8/16), palette + tRNS (palette alpha, gray/RGB colour
// key), Adam7 interlacing, critical-chunk CRC verification, zlib Adler-32 verification. APNG files decode their
// default image (the IDAT data); ImageInfo.FrameCount reports the animation length.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Png;

/// <summary>PNG decoder running on an ILGPU accelerator.</summary>
/// <remarks>
/// Not thread-safe: one in-flight decode per instance (the metadata block, chunk tables and staging
/// buffers are shared across calls).
/// </remarks>
public sealed class PngDecoderGpu : IDisposable
{
    private const int ChunkCapacity = 1 << 16;

    private readonly Accelerator _accelerator;
    private readonly PngKernels _kernels;
    private readonly InflateKernel _inflate;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _chunkTable;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _idatTable;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _crcTable;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _inflateScratch;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _inflateResult;

    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _zlib;
    private readonly StageBuffer<byte> _raw;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels and upload the CRC table.</summary>
    public PngDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new PngKernels(accelerator);
        _inflate = new InflateKernel(accelerator);

        _chunkTable = accelerator.Allocate1D<int>((long)ChunkCapacity * 4);
        _idatTable = accelerator.Allocate1D<int>((long)ChunkCapacity * 3);
        _meta = accelerator.Allocate1D<int>(PngMeta.Size);
        _crcTable = accelerator.Allocate1D<uint>(256);
        _crcTable.View.CopyFromCPU(Crc32Gpu.BuildTable());
        _inflateScratch = accelerator.Allocate1D<int>(InflateGpu.ScratchInts);
        _inflateResult = accelerator.Allocate1D<int>(1);

        _stageIn = new StageBuffer<byte>(accelerator);
        _zlib = new StageBuffer<byte>(accelerator);
        _raw = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Layout derived from the IHDR facts (pure metadata arithmetic)
    // ------------------------------------------------------------------

    private sealed class Layout
    {
        public int Width, Height, BitDepth, ColorType, Interlace;
        public int PlteLength, TrnsOffset, TrnsLength, ApngFrames, IdatTotal;
        public int Channels, BitsPerPixel, FilterBpp, PassCount;
        public long RawSize;
        public bool HasColorKey;
        public readonly int[] HostSlots = new int[PngMeta.Size - PngMeta.HostWrittenStart];

        public bool HasAlpha => ColorType == 4 || ColorType == 6 || TrnsOffset >= 0;

        public ImageInfo ToInfo() => new(
            ImageFormat.Png, Width, Height, HasAlpha, BitDepth,
            ApngFrames > 1 ? ApngFrames : 1, Orientation: 1, IsInterlaced: Interlace != 0);
    }

    private static readonly int[] AdamXStart = { 0, 4, 0, 2, 0, 1, 0 };
    private static readonly int[] AdamYStart = { 0, 0, 4, 0, 2, 0, 1 };
    private static readonly int[] AdamXStep = { 8, 8, 4, 4, 2, 2, 1 };
    private static readonly int[] AdamYStep = { 8, 8, 8, 4, 4, 2, 2 };

    private static string DescribeScanError(PngError error) => error switch
    {
        PngError.BadSignature => "not a PNG file (bad signature)",
        PngError.Truncated => "file is truncated (chunk list ends before IEND)",
        PngError.BadHeader => "first chunk is not a valid IHDR",
        PngError.TooManyChunks => $"more than {ChunkCapacity} chunks",
        PngError.NoImageData => "no IDAT chunk",
        _ => error.ToString(),
    };

    private static Layout ComputeLayout(int[] m, PixelFormat format)
    {
        var err = (PngError)m[PngMeta.Error];
        if (err != PngError.None)
            throw new ImageDecodeException(ImageFormat.Png, DescribeScanError(err));

        var l = new Layout
        {
            Width = m[PngMeta.Width],
            Height = m[PngMeta.Height],
            BitDepth = m[PngMeta.BitDepth],
            ColorType = m[PngMeta.ColorType],
            Interlace = m[PngMeta.Interlace],
            PlteLength = m[PngMeta.PlteLength],
            TrnsOffset = m[PngMeta.TrnsOffset],
            TrnsLength = m[PngMeta.TrnsLength],
            ApngFrames = m[PngMeta.ApngFrames],
            IdatTotal = m[PngMeta.IdatTotal],
        };

        if (l.Width <= 0 || l.Height <= 0 || (long)l.Width * l.Height > int.MaxValue)
            throw new ImageDecodeException(ImageFormat.Png, $"unsupported image size {l.Width}x{l.Height}");
        if (m[PngMeta.Compression] != 0 || m[PngMeta.FilterMethod] != 0)
            throw new ImageDecodeException(ImageFormat.Png, "unsupported compression or filter method");
        if (l.Interlace > 1)
            throw new ImageDecodeException(ImageFormat.Png, $"unsupported interlace method {l.Interlace}");

        switch (l.ColorType)
        {
            case 0: l.Channels = 1; break;
            case 2: l.Channels = 3; break;
            case 3: l.Channels = 1; break;
            case 4: l.Channels = 2; break;
            case 6: l.Channels = 4; break;
            default:
                throw new ImageDecodeException(ImageFormat.Png, $"unsupported colour type {l.ColorType}");
        }
        bool depthOk = l.ColorType switch
        {
            0 => l.BitDepth is 1 or 2 or 4 or 8 or 16,
            3 => l.BitDepth is 1 or 2 or 4 or 8,
            _ => l.BitDepth is 8 or 16,
        };
        if (!depthOk)
            throw new ImageDecodeException(ImageFormat.Png, $"invalid bit depth {l.BitDepth} for colour type {l.ColorType}");
        if (l.ColorType == 3 && (m[PngMeta.PlteOffset] < 0 || l.PlteLength < 3))
            throw new ImageDecodeException(ImageFormat.Png, "palette image without a PLTE chunk");
        if (m[PngMeta.BadCrcChunk] != 0)
            throw new ImageDecodeException(ImageFormat.Png, $"critical chunk #{m[PngMeta.BadCrcChunk]} has a bad CRC");

        l.BitsPerPixel = l.Channels * l.BitDepth;
        l.FilterBpp = Math.Max(1, (l.BitsPerPixel + 7) / 8);
        l.PassCount = l.Interlace == 0 ? 1 : 7;
        l.HasColorKey = (l.ColorType == 0 && l.TrnsOffset >= 0 && l.TrnsLength >= 2)
                     || (l.ColorType == 2 && l.TrnsOffset >= 0 && l.TrnsLength >= 6);

        long running = 0;
        for (int p = 0; p < l.PassCount; p++)
        {
            int xs = l.Interlace == 0 ? 0 : AdamXStart[p];
            int ys = l.Interlace == 0 ? 0 : AdamYStart[p];
            int dx = l.Interlace == 0 ? 1 : AdamXStep[p];
            int dy = l.Interlace == 0 ? 1 : AdamYStep[p];
            int pw = l.Width > xs ? (l.Width - xs + dx - 1) / dx : 0;
            int ph = l.Height > ys ? (l.Height - ys + dy - 1) / dy : 0;
            long rowBytes = pw > 0 ? ((long)pw * l.BitsPerPixel + 7) / 8 : 0;
            int e = PngMeta.PassTable - PngMeta.HostWrittenStart + p * PngMeta.PassStride;
            if (running > int.MaxValue)
                throw new ImageDecodeException(ImageFormat.Png, "image data too large");
            l.HostSlots[e + PngMeta.PassOffset] = (int)running;
            l.HostSlots[e + PngMeta.PassWidth] = pw;
            l.HostSlots[e + PngMeta.PassHeight] = ph;
            l.HostSlots[e + PngMeta.PassRowBytes] = (int)rowBytes;
            l.HostSlots[e + PngMeta.PassXStart] = xs;
            l.HostSlots[e + PngMeta.PassYStart] = ys;
            l.HostSlots[e + PngMeta.PassXStep] = dx;
            l.HostSlots[e + PngMeta.PassYStep] = dy;
            if (pw > 0 && ph > 0) running += (long)ph * (rowBytes + 1);
        }
        if (running > int.MaxValue)
            throw new ImageDecodeException(ImageFormat.Png, "image data too large");
        l.RawSize = running;

        l.HostSlots[PngMeta.RawSize - PngMeta.HostWrittenStart] = (int)running;
        l.HostSlots[PngMeta.BitsPerPixel - PngMeta.HostWrittenStart] = l.BitsPerPixel;
        l.HostSlots[PngMeta.FilterBytesPerPixel - PngMeta.HostWrittenStart] = l.FilterBpp;
        l.HostSlots[PngMeta.PassCount - PngMeta.HostWrittenStart] = l.PassCount;
        l.HostSlots[PngMeta.HasColorKey - PngMeta.HostWrittenStart] = l.HasColorKey ? 1 : 0;
        l.HostSlots[PngMeta.OutputFormat - PngMeta.HostWrittenStart] = (int)format;
        return l;
    }

    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------

    /// <summary>Scan (+ CRC check) and read back the metadata block.</summary>
    private async Task<int[]> ScanAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        if (fileLength <= 0 || fileLength > file.Length) throw new ArgumentOutOfRangeException(nameof(fileLength));
        _kernels.Scan(file, fileLength, _chunkTable.View, _idatTable.View, _meta.View, ChunkCapacity);
        _kernels.CheckCrc(file, _chunkTable.View, _crcTable.View, _meta.View, ChunkCapacity);
        await _accelerator.SynchronizeAsync();
        return await _meta.View.CopyToHostAsync();
    }

    /// <summary>
    /// Read the facts about a PNG that is on the accelerator (chunk scan + a metadata-sized readback;
    /// no pixels are decoded). Use the result to size the output buffer for
    /// <see cref="DecodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{byte, Stride1D.Dense}, PixelFormat)"/>.
    /// </summary>
    public async Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var meta = await ScanAsync(file, fileLength);
        return ComputeLayout(meta, PixelFormat.Rgba8).ToInfo();
    }

    /// <summary>
    /// Decode a PNG that is already on the accelerator into a caller-owned pixel view. Only metadata
    /// scalars cross to the host.
    /// </summary>
    /// <param name="file">The PNG file bytes (may be longer than <paramref name="fileLength"/>).</param>
    /// <param name="fileLength">Number of valid bytes in <paramref name="file"/>.</param>
    /// <param name="output">Receives the tightly packed pixels; at least <see cref="ImageInfo.GetPixelBufferSize"/> bytes.</param>
    /// <param name="format">Pixel layout to produce.</param>
    /// <returns>The facts about the image.</returns>
    /// <exception cref="ImageDecodeException">The file is corrupt, truncated or uses an unsupported feature.</exception>
    public async Task<ImageInfo> DecodeAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
    {
        var meta = await ScanAsync(file, fileLength);
        var layout = ComputeLayout(meta, format);
        var info = layout.ToInfo();
        long outBytes = info.GetPixelBufferSize(format);
        if (output.Length < outBytes)
            throw new ArgumentException($"output must hold {outBytes} bytes for {info.Width}x{info.Height} {format}.", nameof(output));
        await RunPipelineAsync(file, layout, info, output, format);
        return info;
    }

    /// <summary>
    /// Scan + decode into a staging output sized from the scan, for the managed and TypedArray overloads
    /// (one scan, not two). Returns the info; the pixels are in <see cref="_stageOut"/>.
    /// </summary>
    private async Task<ImageInfo> DecodeToStageAsync(int fileLength, PixelFormat format)
    {
        var meta = await ScanAsync(_stageIn.View, fileLength);
        var layout = ComputeLayout(meta, format);
        var info = layout.ToInfo();
        _stageOut.Ensure(info.GetPixelBufferSize(format));
        await RunPipelineAsync(_stageIn.View, layout, info, _stageOut.View, format);
        return info;
    }

    private async Task RunPipelineAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout layout, ImageInfo info,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format)
    {
        _zlib.Ensure(layout.IdatTotal);
        _raw.Ensure(layout.RawSize);
        // Host-derived layout (pass table, sizes) into the shared metadata block.
        _meta.View.SubView(PngMeta.HostWrittenStart, layout.HostSlots.Length)
             .CopyFromCPU(ref layout.HostSlots[0], layout.HostSlots.Length);

        _kernels.GatherIdat(file, _idatTable.View, _zlib.View, _meta.View, layout.IdatTotal);
        _inflate.Run(_zlib.View, layout.IdatTotal, _raw.View, (int)layout.RawSize,
            _inflateScratch.View, _inflateResult.View, verifyAdler: true);
        _kernels.Unfilter(_raw.View, _meta.View);
        _kernels.ToRgba(file, _raw.View, output, _meta.View, info.Width * info.Height);
        await _accelerator.SynchronizeAsync();

        int inflated = (await _inflateResult.View.CopyToHostAsync())[0];
        int status = (await _meta.View.SubView(PngMeta.Error, 1).CopyToHostAsync())[0];
        if (inflated < 0)
            throw new ImageDecodeException(ImageFormat.Png, $"image data is not valid zlib: {(InflateError)(-inflated)}");
        if (inflated != layout.RawSize)
            throw new ImageDecodeException(ImageFormat.Png, $"image data is {inflated} bytes, expected {layout.RawSize}");
        if (status == (int)PngError.BadFilter)
            throw new ImageDecodeException(ImageFormat.Png, "invalid scanline filter type");
    }

    // ------------------------------------------------------------------
    // Text chunks (metadata: small, so the chunk bytes may visit the host)
    // ------------------------------------------------------------------

    private const uint TypeTExt = 0x74455874;   // tEXt
    private const uint TypeZTxt = 0x7A545874;   // zTXt
    private const uint TypeITxt = 0x69545874;   // iTXt

    /// <summary>
    /// Read the tEXt / zTXt / iTXt chunks of a PNG that is on the accelerator (ComfyUI keeps its "prompt" and
    /// "workflow" JSON there). The chunk walk happens on the accelerator; only the text chunks' bytes are read
    /// back. When a keyword repeats the last chunk wins (as Pillow does).
    /// </summary>
    public async Task<IReadOnlyDictionary<string, string>> ReadTextAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        var meta = await ScanAsync(file, fileLength);
        var err = (PngError)meta[PngMeta.Error];
        // A file without image data still has readable metadata chunks; structural damage does not.
        if (err != PngError.None && err != PngError.NoImageData)
            throw new ImageDecodeException(ImageFormat.Png, DescribeScanError(err));
        int count = meta[PngMeta.ChunkCount];
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (count == 0) return result;
        int[] table = await _chunkTable.View.SubView(0, (long)count * 4).CopyToHostAsync();
        for (int i = 0; i < count; i++)
        {
            uint type = (uint)table[i * 4];
            if (type != TypeTExt && type != TypeZTxt && type != TypeITxt) continue;
            int offset = table[i * 4 + 1];
            int length = table[i * 4 + 2];
            byte[] data = length == 0 ? System.Array.Empty<byte>() : await DeviceTransfer.DownloadAsync(file.SubView(offset, length), length);
            if (TryParseTextChunk(type, data, out var key, out var value)) result[key] = value;
        }
        return result;
    }

    /// <summary>Read the text chunks of a managed PNG file.</summary>
    public async Task<IReadOnlyDictionary<string, string>> ReadTextAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadTextAsync(_stageIn.View, file.Length);
    }

    /// <summary>Read the text chunks of a PNG held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<IReadOnlyDictionary<string, string>> ReadTextAsync(TypedArray file)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        return await ReadTextAsync(_stageIn.View, checked((int)length));
    }

    private static bool TryParseTextChunk(uint type, byte[] data, out string key, out string value)
    {
        key = value = "";
        int nul = System.Array.IndexOf(data, (byte)0);
        if (nul <= 0 || nul > 79) return false;
        key = System.Text.Encoding.Latin1.GetString(data, 0, nul);
        int p = nul + 1;
        if (type == TypeTExt)
        {
            value = System.Text.Encoding.Latin1.GetString(data, p, data.Length - p);
            return true;
        }
        if (type == TypeZTxt)
        {
            if (p >= data.Length || data[p] != 0) return false;   // compression method must be 0 (zlib)
            value = System.Text.Encoding.Latin1.GetString(InflateMetadata(data, p + 1));
            return true;
        }
        // iTXt: flag, method, language tag \0, translated keyword \0, text (UTF-8, optionally zlib).
        if (p + 2 > data.Length) return false;
        bool compressed = data[p] == 1;
        p += 2;
        int langEnd = System.Array.IndexOf(data, (byte)0, p);
        if (langEnd < 0) return false;
        int transEnd = System.Array.IndexOf(data, (byte)0, langEnd + 1);
        if (transEnd < 0) return false;
        p = transEnd + 1;
        value = compressed
            ? System.Text.Encoding.UTF8.GetString(InflateMetadata(data, p))
            : System.Text.Encoding.UTF8.GetString(data, p, data.Length - p);
        return true;
    }

    /// <summary>Inflates a (small) zlib-compressed metadata payload.</summary>
    private static byte[] InflateMetadata(byte[] data, int offset)
    {
        using var input = new MemoryStream(data, offset, data.Length - offset);
        using var z = new System.IO.Compression.ZLibStream(input, System.IO.Compression.CompressionMode.Decompress);
        using var output = new MemoryStream();
        z.CopyTo(output);
        return output.ToArray();
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a managed PNG file (no pixels are decoded).</summary>
    public async Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadInfoAsync(_stageIn.View, file.Length);
    }

    /// <summary>
    /// Decode a managed PNG file. Returns the pixels as a managed array (a terminal CPU sink; in a
    /// browser prefer the <see cref="TypedArray"/> overload).
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

    /// <summary>Read the facts about a PNG held in a JS typed array (browser accelerators only).</summary>
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
    /// Decode a PNG held in a JS typed array (browser accelerators only). The bytes go to the GPU with
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
        _inflate.Dispose();
        _chunkTable.Dispose();
        _idatTable.Dispose();
        _meta.Dispose();
        _crcTable.Dispose();
        _inflateScratch.Dispose();
        _inflateResult.Dispose();
        _stageIn.Dispose();
        _zlib.Dispose();
        _raw.Dispose();
        _stageOut.Dispose();
    }
}
