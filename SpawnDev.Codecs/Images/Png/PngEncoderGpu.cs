// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// PNG encoder, 100% accelerator-resident for the pixel data: the host builds only the tiny header blob
// (signature + IHDR + text chunks) and dispatches PngEncodeKernels + the block-parallel DEFLATE kernels.
// Output is a valid PNG every decoder reads: one IDAT chunk per compressed block, a zlib header on the first
// and an Adler-32 trailer on the last.
//
// API shape (same as the other codecs): three overloads per entry point, all funnelling into the ArrayView1D one.
//   - ArrayView1D : RGBA pixels already on the accelerator, PNG bytes written into a caller-owned view, returns the length.
//   - TypedArray  : browser. JS pixels in via CopyFromJS, JS Uint8Array PNG out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; the PNG comes back as byte[].
//
// Input is RGBA8 (PixelFormat.Rgba8) or RGBA16 (little-endian, PixelFormat.Rgba16 -> 16-bit PNG). Interlaced (Adam7)
// output and palette reduction are not produced yet.

using System.Text;
using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Png;

/// <summary>How the encoder chooses the PNG colour type.</summary>
public enum PngColorMode
{
    /// <summary>Smallest lossless type: gray, gray+alpha, RGB or RGBA depending on the pixels (one analysis pass on the accelerator).</summary>
    Auto = 0,
    /// <summary>Always RGBA.</summary>
    Rgba = 1,
    /// <summary>RGB; alpha is dropped.</summary>
    Rgb = 2,
    /// <summary>Gray from the red channel; alpha is dropped. Use only for gray input.</summary>
    Gray = 3,
    /// <summary>Gray from the red channel plus alpha. Use only for gray input.</summary>
    GrayAlpha = 4,
}

/// <summary>Scanline filter selection.</summary>
public enum PngFilterMode
{
    /// <summary>Per row, the filter with the smallest sum of absolute differences (libpng's heuristic).</summary>
    Adaptive = 0,
    /// <summary>Filter type 0 on every row.</summary>
    None = 1,
    /// <summary>Filter type 1 (Sub) on every row.</summary>
    Sub = 2,
    /// <summary>Filter type 2 (Up) on every row.</summary>
    Up = 3,
    /// <summary>Filter type 3 (Average) on every row.</summary>
    Average = 4,
    /// <summary>Filter type 4 (Paeth) on every row.</summary>
    Paeth = 5,
}

/// <summary>Options of <see cref="PngEncoderGpu"/>.</summary>
public sealed class PngEncodeOptions
{
    /// <summary>0 = stored, 1 = fast, 2 = default, 3 = best (lazy matching).</summary>
    public int CompressionLevel { get; init; } = 2;
    /// <summary>Colour type selection.</summary>
    public PngColorMode ColorMode { get; init; } = PngColorMode.Auto;
    /// <summary>Scanline filtering.</summary>
    public PngFilterMode Filter { get; init; } = PngFilterMode.Adaptive;
    /// <summary>Input pixel layout.</summary>
    public PixelFormat InputFormat { get; init; } = PixelFormat.Rgba8;
    /// <summary>
    /// Text chunks (keyword to text), e.g. ComfyUI's "prompt" and "workflow". Latin-1 text becomes tEXt, anything else iTXt (UTF-8).
    /// They are small host-built metadata chunks (including their CRC-32); the pixel data never touches the host.
    /// </summary>
    public IReadOnlyDictionary<string, string>? TextChunks { get; init; }
    /// <summary>Input bytes per independently compressed block (1 KiB .. 1 MiB). Each block becomes one IDAT chunk.</summary>
    public int BlockSize { get; init; } = ZlibCompressorGpu.DefaultBlockSize;
}

/// <summary>PNG encoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class PngEncoderGpu : IDisposable
{
    private const int HeaderFixedBytes = 8 + 25;   // signature + IHDR chunk

    private readonly Accelerator _accelerator;
    private readonly PngEncodeKernels _kernels;
    private readonly int _wave;
    private readonly Dictionary<int, DeflateBlockRunner> _runners = new();
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _crcTable;
    private readonly StageBuffer<byte> _raw;
    private readonly StageBuffer<byte> _filtered;
    private readonly StageBuffer<byte> _header;
    private readonly StageBuffer<int> _layout;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels.</summary>
    /// <param name="accelerator">The accelerator.</param>
    /// <param name="threadsPerWave">Blocks compressed concurrently.</param>
    public PngEncoderGpu(Accelerator accelerator, int threadsPerWave = 32)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _wave = threadsPerWave;
        _kernels = new PngEncodeKernels(accelerator);
        _meta = accelerator.Allocate1D<int>(PngEncMeta.Size);
        _crcTable = accelerator.Allocate1D<uint>(256);
        _crcTable.View.CopyFromCPU(Crc32Gpu.BuildTable());
        _raw = new StageBuffer<byte>(accelerator);
        _filtered = new StageBuffer<byte>(accelerator);
        _header = new StageBuffer<byte>(accelerator);
        _layout = new StageBuffer<int>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Sizing
    // ------------------------------------------------------------------

    private static (int channels, int colorType) ModeToType(PngColorMode mode) => mode switch
    {
        PngColorMode.Rgb => (3, 2),
        PngColorMode.Gray => (1, 0),
        PngColorMode.GrayAlpha => (2, 4),
        _ => (4, 6),
    };

    /// <summary>Upper bound of the encoded size; the output view of the ArrayView1D overload must be at least this long.</summary>
    public static long GetMaxEncodedSize(int width, int height, PngEncodeOptions? options = null)
    {
        options ??= new PngEncodeOptions();
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        int depthBytes = options.InputFormat == PixelFormat.Rgba16 ? 2 : 1;
        long rowBytes = (long)width * 4 * depthBytes;          // RGBA is the widest colour type
        long filtered = (long)height * (rowBytes + 1);
        long blocks = DeflateBlockRunner.BlocksFor((int)Math.Min(filtered, int.MaxValue), options.BlockSize);
        long header = HeaderFixedBytes + (options.TextChunks is null ? 0 : PngChunkWriter.TextChunksSize(options.TextChunks));
        return header + blocks * (12 + DeflateGpu.MaxBlockOutput(options.BlockSize)) + 6 + 12 + 16;
    }

    // ------------------------------------------------------------------
    // ArrayView1D overload: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode RGBA pixels that are on the accelerator into a PNG written to <paramref name="output"/> (at least
    /// <see cref="GetMaxEncodedSize"/> bytes). Only a few scalars cross to the host. Returns the PNG length.
    /// </summary>
    public async Task<int> EncodeAsync(
        ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, PngEncodeOptions? options = null)
    {
        options ??= new PngEncodeOptions();
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if (options.CompressionLevel < 0 || options.CompressionLevel > 3) throw new ArgumentOutOfRangeException(nameof(options), "CompressionLevel must be 0..3");
        bool is16 = options.InputFormat == PixelFormat.Rgba16;
        int inBpp = is16 ? 8 : 4;
        long pixels = (long)width * height;
        if (pixels > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(width), "image too large");
        if (rgba.Length < pixels * inBpp)
            throw new ArgumentException($"rgba must hold {pixels * inBpp} bytes.", nameof(rgba));
        long maxOut = GetMaxEncodedSize(width, height, options);
        if (output.Length < maxOut)
            throw new ArgumentException($"output must hold {maxOut} bytes.", nameof(output));

        // ---- colour type ----
        int channels, colorType;
        if (options.ColorMode == PngColorMode.Auto)
        {
            // One analysis pass: is every pixel opaque / gray?
            var init = new int[PngEncMeta.Size];
            init[PngEncMeta.Opaque] = 1;
            init[PngEncMeta.Gray] = 1;
            init[PngEncMeta.InputBytesPerPixel] = inBpp;
            _meta.View.CopyFromCPU(ref init[0], init.Length);
            _kernels.Analyze(rgba, _meta.View, (int)pixels);
            await _accelerator.SynchronizeAsync();
            var flags = await _meta.View.SubView(0, 2).CopyToHostAsync();
            bool opaque = flags[PngEncMeta.Opaque] != 0;
            bool gray = flags[PngEncMeta.Gray] != 0;
            (channels, colorType) = gray ? (opaque ? (1, 0) : (2, 4)) : (opaque ? (3, 2) : (4, 6));
        }
        else
        {
            (channels, colorType) = ModeToType(options.ColorMode);
        }

        int depth = is16 ? 16 : 8;
        long rowBytesL = (long)width * channels * (depth / 8);
        long filteredL = (long)height * (rowBytesL + 1);
        if (filteredL > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(width), "image too large");
        int rowBytes = (int)rowBytesL;
        int filteredLen = (int)filteredL;

        // ---- header blob (signature + IHDR + text chunks): tiny host-built metadata ----
        byte[] header = PngChunkWriter.BuildHeader(width, height, depth, colorType, options.TextChunks);
        _header.Ensure(header.Length);
        DeviceTransfer.Upload(_header.View, header);

        var meta = new int[PngEncMeta.Size];
        meta[PngEncMeta.ColorType] = colorType;
        meta[PngEncMeta.BitDepth] = depth;
        meta[PngEncMeta.Channels] = channels;
        meta[PngEncMeta.RowBytes] = rowBytes;
        meta[PngEncMeta.InputBytesPerPixel] = inBpp;
        meta[PngEncMeta.Width] = width;
        meta[PngEncMeta.Height] = height;
        meta[PngEncMeta.FilterMode] = (int)options.Filter;
        meta[PngEncMeta.HeaderLength] = header.Length;
        _meta.View.CopyFromCPU(ref meta[0], meta.Length);

        _raw.Ensure(filteredL);
        _filtered.Ensure(filteredL);
        _kernels.Pack(rgba, _raw.View, _meta.View, (int)pixels);
        _kernels.Filter(_raw.View, _filtered.View, _meta.View, height);

        if (!_runners.TryGetValue(options.BlockSize, out var runner))
        {
            runner = new DeflateBlockRunner(_accelerator, options.BlockSize, _wave);
            _runners[options.BlockSize] = runner;
        }
        runner.Run(_filtered.View, filteredLen, options.CompressionLevel);
        _layout.Ensure(runner.BlockCount + 1);
        _kernels.Layout(runner.Res, runner.Prm, _layout.View, _meta.View);
        _kernels.Frame(runner.Slots, _header.View, output, runner.Res, _layout.View, runner.Prm, _crcTable.View, _meta.View, runner.BlockCount);
        await _accelerator.SynchronizeAsync();

        int total = (await _meta.View.SubView(PngEncMeta.Total, 1).CopyToHostAsync())[0];
        if (total < 0) throw new InvalidOperationException("PNG encode overflowed a block slot (internal error).");
        return total;
    }

    // ------------------------------------------------------------------
    // Managed / TypedArray overloads
    // ------------------------------------------------------------------

    /// <summary>Encode managed RGBA pixels; returns the PNG as a managed array (a terminal CPU sink).</summary>
    public async Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, PngEncodeOptions? options = null)
    {
        options ??= new PngEncodeOptions();
        int inBpp = options.InputFormat == PixelFormat.Rgba16 ? 8 : 4;
        long need = (long)width * height * inBpp;
        if (rgba.Length < need) throw new ArgumentException($"rgba must hold {need} bytes.", nameof(rgba));
        _stageIn.Ensure(need);
        DeviceTransfer.Upload(_stageIn.View, rgba.Span.Slice(0, (int)need));
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int n = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, n);
    }

    /// <summary>Encode RGBA pixels held in a JS typed array (browser accelerators only); returns the PNG as a JS <see cref="Uint8Array"/>.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, PngEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rgba);
        options ??= new PngEncodeOptions();
        _stageIn.Ensure(rgba.ByteLength);
        DeviceTransfer.Upload(_stageIn.View, rgba);
        _stageOut.Ensure(GetMaxEncodedSize(width, height, options));
        int n = await EncodeAsync(_stageIn.View, width, height, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, n);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        foreach (var r in _runners.Values) r.Dispose();
        _runners.Clear();
        _meta.Dispose();
        _crcTable.Dispose();
        _raw.Dispose();
        _filtered.Dispose();
        _header.Dispose();
        _layout.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}

/// <summary>Builds the small host-side PNG chunks (signature, IHDR, text chunks) and reads text chunks back.</summary>
public static class PngChunkWriter
{
    private static readonly uint[] CrcTable = Crc32Gpu.BuildTable();

    /// <summary>CRC-32 of a short byte span (metadata chunks only).</summary>
    internal static uint Crc32(ReadOnlySpan<byte> data)
    {
        uint c = Crc32Gpu.Seed;
        foreach (byte b in data) c = CrcTable[(int)((c ^ b) & 0xFF)] ^ (c >> 8);
        return Crc32Gpu.Finish(c);
    }

    private static void AppendChunk(List<byte> dst, string type, ReadOnlySpan<byte> data)
    {
        var body = new byte[4 + data.Length];
        for (int i = 0; i < 4; i++) body[i] = (byte)type[i];
        data.CopyTo(body.AsSpan(4));
        uint len = (uint)data.Length;
        dst.Add((byte)(len >> 24)); dst.Add((byte)(len >> 16)); dst.Add((byte)(len >> 8)); dst.Add((byte)len);
        dst.AddRange(body);
        uint crc = Crc32(body);
        dst.Add((byte)(crc >> 24)); dst.Add((byte)(crc >> 16)); dst.Add((byte)(crc >> 8)); dst.Add((byte)crc);
    }

    private static bool IsLatin1(string s)
    {
        foreach (char ch in s) if (ch > 0xFF) return false;
        return true;
    }

    private static void ValidateKeyword(string keyword)
    {
        if (string.IsNullOrEmpty(keyword) || keyword.Length > 79)
            throw new ArgumentException("A PNG text keyword must be 1..79 characters.");
        foreach (char ch in keyword)
        {
            bool ok = (ch >= 32 && ch <= 126) || (ch >= 161 && ch <= 255);
            if (!ok) throw new ArgumentException($"PNG text keyword '{keyword}' has a character outside Latin-1 printable.");
        }
        if (keyword[0] == ' ' || keyword[^1] == ' ' || keyword.Contains("  "))
            throw new ArgumentException($"PNG text keyword '{keyword}' has leading, trailing or consecutive spaces.");
    }

    private static byte[] BuildTextData(string keyword, string text, out string type)
    {
        ValidateKeyword(keyword);
        var data = new List<byte>();
        foreach (char ch in keyword) data.Add((byte)ch);
        data.Add(0);
        if (IsLatin1(text))
        {
            type = "tEXt";
            foreach (char ch in text) data.Add((byte)ch);
        }
        else
        {
            type = "iTXt";
            data.Add(0);   // compression flag: uncompressed
            data.Add(0);   // compression method
            data.Add(0);   // language tag (empty)
            data.Add(0);   // translated keyword (empty)
            data.AddRange(Encoding.UTF8.GetBytes(text));
        }
        return data.ToArray();
    }

    /// <summary>Total bytes the chunks for <paramref name="chunks"/> take.</summary>
    internal static long TextChunksSize(IReadOnlyDictionary<string, string> chunks)
    {
        long total = 0;
        foreach (var kv in chunks)
            total += 12 + BuildTextData(kv.Key, kv.Value, out _).Length;
        return total;
    }

    /// <summary>PNG signature + IHDR + the text chunks.</summary>
    internal static byte[] BuildHeader(int width, int height, int depth, int colorType, IReadOnlyDictionary<string, string>? text)
    {
        var dst = new List<byte> { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        var ihdr = new byte[13];
        ihdr[0] = (byte)(width >> 24); ihdr[1] = (byte)(width >> 16); ihdr[2] = (byte)(width >> 8); ihdr[3] = (byte)width;
        ihdr[4] = (byte)(height >> 24); ihdr[5] = (byte)(height >> 16); ihdr[6] = (byte)(height >> 8); ihdr[7] = (byte)height;
        ihdr[8] = (byte)depth;
        ihdr[9] = (byte)colorType;
        AppendChunk(dst, "IHDR", ihdr);
        if (text is not null)
        {
            foreach (var kv in text)
            {
                var data = BuildTextData(kv.Key, kv.Value, out var type);
                AppendChunk(dst, type, data);
            }
        }
        return dst.ToArray();
    }
}
