// Tests for PngEncoderGpu (Images/Png).
//
// Oracle: an INDEPENDENT host-side PNG reader written in this file (chunk parser with CRC-32 check, .NET
// ZLibStream inflate, scanline unfilter, sample extraction). It shares no code with the GPU decoder or encoder, so
// "encode on the accelerator, read on the host, compare to the source pixels" cannot pass if the encoder (or both
// GPU stages together) is wrong. Separately, scripts/images/verify_png_encoder.py opens dumped encoder output with
// Pillow (libpng-equivalent) for a third opinion.
//
// What can fail: wrong colour-type reduction, wrong filters (checked per row), wrong block/IDAT framing (CRC,
// lengths, zlib header, Adler), text chunks, 16-bit byte order, output-buffer handling.

using System.IO.Compression;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private sealed class HostPng
    {
        public int Width, Height, Depth, ColorType, IdatCount;
        public byte[] FilterTypes = System.Array.Empty<byte>();
        public byte[] Rgba8 = System.Array.Empty<byte>();     // only for depth 8
        public ushort[] Rgba16 = System.Array.Empty<ushort>(); // only for depth 16
        public List<(string Type, byte[] Data)> Chunks = new();
    }

    private static readonly uint[] HostCrcTable = BuildHostCrcTable();

    private static uint[] BuildHostCrcTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }

    private static uint HostCrc(ReadOnlySpan<byte> data)
    {
        uint c = 0xFFFFFFFFu;
        foreach (byte b in data) c = HostCrcTable[(int)((c ^ b) & 0xFF)] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    private static HostPng ReadPngOnHost(byte[] file)
    {
        var png = new HostPng();
        True(file.Length > 8 + 12 + 13, "file too short to be a PNG");
        byte[] sig = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        for (int i = 0; i < 8; i++) Equal(sig[i], file[i], $"signature byte {i}");
        int pos = 8;
        var idat = new MemoryStream();
        bool sawIend = false;
        while (pos < file.Length)
        {
            True(pos + 12 <= file.Length, "truncated chunk header");
            uint len = (uint)(file[pos] << 24 | file[pos + 1] << 16 | file[pos + 2] << 8 | file[pos + 3]);
            string type = System.Text.Encoding.ASCII.GetString(file, pos + 4, 4);
            True(pos + 12 + len <= file.Length, $"chunk {type} runs past the file");
            var data = file.AsSpan(pos + 8, (int)len).ToArray();
            uint stored = (uint)(file[pos + 8 + len] << 24 | file[pos + 9 + len] << 16 | file[pos + 10 + len] << 8 | file[pos + 11 + len]);
            Equal(stored, HostCrc(file.AsSpan(pos + 4, 4 + (int)len)), $"CRC of chunk {type}");
            png.Chunks.Add((type, data));
            if (type == "IHDR")
            {
                png.Width = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3];
                png.Height = data[4] << 24 | data[5] << 16 | data[6] << 8 | data[7];
                png.Depth = data[8];
                png.ColorType = data[9];
                Equal(0, data[12], "IHDR interlace");
            }
            else if (type == "IDAT") { idat.Write(data); png.IdatCount++; }
            else if (type == "IEND") { sawIend = true; }
            pos += 12 + (int)len;
            if (sawIend) break;
        }
        True(sawIend, "no IEND");
        Equal(file.Length, pos, "bytes after IEND");

        byte[] raw;
        using (var z = new ZLibStream(new MemoryStream(idat.ToArray()), CompressionMode.Decompress))
        using (var o = new MemoryStream()) { z.CopyTo(o); raw = o.ToArray(); }

        int channels = png.ColorType switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => throw new Exception("unexpected colour type " + png.ColorType) };
        int bytesPerSample = png.Depth / 8;
        int bpp = channels * bytesPerSample;
        int rowBytes = png.Width * bpp;
        Equal(png.Height * (rowBytes + 1), raw.Length, "inflated scanline data length");
        png.FilterTypes = new byte[png.Height];
        var pix = new byte[png.Height * rowBytes];
        for (int y = 0; y < png.Height; y++)
        {
            int ft = raw[y * (rowBytes + 1)];
            png.FilterTypes[y] = (byte)ft;
            True(ft <= 4, $"row {y}: filter type {ft}");
            for (int i = 0; i < rowBytes; i++)
            {
                int x = raw[y * (rowBytes + 1) + 1 + i];
                int a = i >= bpp ? pix[y * rowBytes + i - bpp] : 0;
                int b = y > 0 ? pix[(y - 1) * rowBytes + i] : 0;
                int c = (i >= bpp && y > 0) ? pix[(y - 1) * rowBytes + i - bpp] : 0;
                int v = ft switch
                {
                    0 => x,
                    1 => x + a,
                    2 => x + b,
                    3 => x + ((a + b) >> 1),
                    _ => x + HostPaeth(a, b, c),
                };
                pix[y * rowBytes + i] = (byte)v;
            }
        }
        // Samples -> RGBA.
        if (png.Depth == 8)
        {
            png.Rgba8 = new byte[png.Width * png.Height * 4];
            for (int p = 0; p < png.Width * png.Height; p++)
            {
                int y = p / png.Width, x = p % png.Width;
                int s = y * rowBytes + x * bpp;
                byte r, g, b, a = 255;
                switch (png.ColorType)
                {
                    case 0: r = g = b = pix[s]; break;
                    case 4: r = g = b = pix[s]; a = pix[s + 1]; break;
                    case 2: r = pix[s]; g = pix[s + 1]; b = pix[s + 2]; break;
                    default: r = pix[s]; g = pix[s + 1]; b = pix[s + 2]; a = pix[s + 3]; break;
                }
                png.Rgba8[p * 4] = r; png.Rgba8[p * 4 + 1] = g; png.Rgba8[p * 4 + 2] = b; png.Rgba8[p * 4 + 3] = a;
            }
        }
        else
        {
            png.Rgba16 = new ushort[png.Width * png.Height * 4];
            for (int p = 0; p < png.Width * png.Height; p++)
            {
                int y = p / png.Width, x = p % png.Width;
                int s = y * rowBytes + x * bpp;
                ushort Sample(int k) => (ushort)(pix[s + 2 * k] << 8 | pix[s + 2 * k + 1]);
                ushort r, g, b, a = 65535;
                switch (png.ColorType)
                {
                    case 0: r = g = b = Sample(0); break;
                    case 4: r = g = b = Sample(0); a = Sample(1); break;
                    case 2: r = Sample(0); g = Sample(1); b = Sample(2); break;
                    default: r = Sample(0); g = Sample(1); b = Sample(2); a = Sample(3); break;
                }
                png.Rgba16[p * 4] = r; png.Rgba16[p * 4 + 1] = g; png.Rgba16[p * 4 + 2] = b; png.Rgba16[p * 4 + 3] = a;
            }
        }
        return png;
    }

    private static int HostPaeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        return pb <= pc ? b : c;
    }

    /// <summary>Structured RGBA test picture: gradients, flat areas, edges and noise, so every filter has a use.</summary>
    private static byte[] PicturePixels(int w, int h, int seed, bool opaque, bool gray)
    {
        var rng = new Random(seed);
        var px = new byte[w * h * 4];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int o = (y * w + x) * 4;
                int r = (x * 255 / Math.Max(1, w - 1) + (y / 8 % 2) * 40) & 0xFF;
                int g = (y * 255 / Math.Max(1, h - 1)) & 0xFF;
                int b = ((x ^ y) * 3) & 0xFF;
                if (x > w / 3 && x < w / 2 && y > h / 3 && y < h / 2) { r = 200; g = 30; b = 90; }       // flat block
                if (rng.Next(23) == 0) { r = rng.Next(256); g = rng.Next(256); b = rng.Next(256); }       // noise
                if (gray) { g = r; b = r; }
                px[o] = (byte)r; px[o + 1] = (byte)g; px[o + 2] = (byte)b;
                px[o + 3] = opaque ? (byte)255 : (byte)(x * 255 / Math.Max(1, w - 1));
            }
        return px;
    }

    private async Task<byte[]> EncodePngManagedAsync(byte[] rgba, int w, int h, PngEncodeOptions? options)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PngEncoderGpu(acc);
            var file = await enc.EncodeAsync(rgba, w, h, options);
            DumpPngForPillow(rgba, w, h, options, file);
            return file;
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static int _pngDumpCounter;

    /// <summary>
    /// When CODECS_DUMP_DIR is set (desktop runs), writes each encoded PNG plus its source pixels so
    /// scripts/images/verify_png_encoder.py can open them with Pillow - a third, independent reader.
    /// </summary>
    private static void DumpPngForPillow(byte[] rgba, int w, int h, PngEncodeOptions? options, byte[] file)
    {
        string? dir = Environment.GetEnvironmentVariable("CODECS_DUMP_DIR");
        if (string.IsNullOrEmpty(dir)) return;
        Directory.CreateDirectory(dir);
        int n = System.Threading.Interlocked.Increment(ref _pngDumpCounter);
        string stem = Path.Combine(dir, $"png_{Environment.ProcessId}_{n:D3}_{w}x{h}_{(options?.InputFormat == PixelFormat.Rgba16 ? "rgba16" : "rgba8")}");
        File.WriteAllBytes(stem + ".png", file);
        File.WriteAllBytes(stem + ".src", rgba);
    }

    private static void AssertPixelsEqual(byte[] expected, byte[] actual, string what)
    {
        Equal(expected.Length, actual.Length, $"{what}: length");
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i])
                throw new Exception($"{what}: first mismatch at byte {i} (pixel {i / 4}, channel {i % 4}): got {actual[i]}, expected {expected[i]}");
    }

    [TestMethod]
    public async Task PngEncoderGpu_Rgba8_Auto_RoundTripsOnHostReader()
    {
        const int w = 70, h = 45;
        var src = PicturePixels(w, h, 0x5E01, opaque: false, gray: false);
        var file = await EncodePngManagedAsync(src, w, h, null);
        var png = ReadPngOnHost(file);
        Equal(6, png.ColorType, "RGBA content must stay RGBA");
        Equal(8, png.Depth, "depth");
        Equal(w, png.Width, "width"); Equal(h, png.Height, "height");
        AssertPixelsEqual(src, png.Rgba8, "RGBA8 round trip");
    }

    [TestMethod]
    public async Task PngEncoderGpu_Auto_ReducesOpaqueToRgb_GrayToGray_GrayAlphaToGrayAlpha()
    {
        const int w = 33, h = 21;
        var opaque = PicturePixels(w, h, 0x5E02, opaque: true, gray: false);
        var p1 = ReadPngOnHost(await EncodePngManagedAsync(opaque, w, h, null));
        Equal(2, p1.ColorType, "opaque colour -> RGB");
        AssertPixelsEqual(opaque, p1.Rgba8, "opaque colour");

        var gray = PicturePixels(w, h, 0x5E03, opaque: true, gray: true);
        var p2 = ReadPngOnHost(await EncodePngManagedAsync(gray, w, h, null));
        Equal(0, p2.ColorType, "opaque gray -> gray");
        AssertPixelsEqual(gray, p2.Rgba8, "opaque gray");

        var grayA = PicturePixels(w, h, 0x5E04, opaque: false, gray: true);
        var p3 = ReadPngOnHost(await EncodePngManagedAsync(grayA, w, h, null));
        Equal(4, p3.ColorType, "gray + alpha -> gray+alpha");
        AssertPixelsEqual(grayA, p3.Rgba8, "gray alpha");
    }

    [TestMethod]
    public async Task PngEncoderGpu_Rgba16_Input_Makes16BitPng()
    {
        const int w = 29, h = 19;
        var rng = new Random(0x5E05);
        var src = new byte[w * h * 8];
        rng.NextBytes(src);
        for (int p = 0; p < w * h; p++) { src[p * 8 + 6] = (byte)(p % 255); src[p * 8 + 7] = (byte)(p % 7); }   // varied alpha
        var file = await EncodePngManagedAsync(src, w, h, new PngEncodeOptions { InputFormat = PixelFormat.Rgba16 });
        var png = ReadPngOnHost(file);
        Equal(16, png.Depth, "depth");
        Equal(6, png.ColorType, "colour type");
        for (int i = 0; i < w * h * 4; i++)
        {
            ushort expected = (ushort)(src[i * 2] | src[i * 2 + 1] << 8);   // little-endian input
            if (png.Rgba16[i] != expected) throw new Exception($"16-bit sample {i}: got {png.Rgba16[i]}, expected {expected}");
        }
    }

    [TestMethod]
    public async Task PngEncoderGpu_FixedFilterModes_UseExactlyThatFilter_AndAdaptiveMixes()
    {
        const int w = 64, h = 40;
        var src = PicturePixels(w, h, 0x5E06, opaque: false, gray: false);
        var modes = new[] { PngFilterMode.None, PngFilterMode.Sub, PngFilterMode.Up, PngFilterMode.Average, PngFilterMode.Paeth };
        for (int m = 0; m < modes.Length; m++)
        {
            var png = ReadPngOnHost(await EncodePngManagedAsync(src, w, h, new PngEncodeOptions { Filter = modes[m] }));
            for (int y = 0; y < h; y++) Equal(m, png.FilterTypes[y], $"{modes[m]} row {y} filter type");
            AssertPixelsEqual(src, png.Rgba8, $"{modes[m]} pixels");
        }
        var adaptive = ReadPngOnHost(await EncodePngManagedAsync(src, w, h, new PngEncodeOptions { Filter = PngFilterMode.Adaptive }));
        True(adaptive.FilterTypes.Distinct().Count() >= 2, "adaptive filtering should use more than one filter type on a structured picture");
        AssertPixelsEqual(src, adaptive.Rgba8, "adaptive pixels");
    }

    [TestMethod]
    public async Task PngEncoderGpu_ManyBlocks_MakeOneIdatChunkEach()
    {
        const int w = 100, h = 100;
        var src = PicturePixels(w, h, 0x5E07, opaque: false, gray: false);
        var file = await EncodePngManagedAsync(src, w, h, new PngEncodeOptions { BlockSize = 4096, ColorMode = PngColorMode.Rgba });
        var png = ReadPngOnHost(file);
        int filteredLen = h * (w * 4 + 1);
        Equal((filteredLen + 4095) / 4096, png.IdatCount, "one IDAT per 4096-byte block");
        AssertPixelsEqual(src, png.Rgba8, "multi-block pixels");
    }

    [TestMethod]
    public async Task PngEncoderGpu_CompressionLevels_AllValid_AndCompressionHappens()
    {
        const int w = 120, h = 80;
        var src = PicturePixels(w, h, 0x5E08, opaque: true, gray: false);
        int stored = 0;
        for (int level = 0; level <= 3; level++)
        {
            var file = await EncodePngManagedAsync(src, w, h, new PngEncodeOptions { CompressionLevel = level });
            AssertPixelsEqual(src, ReadPngOnHost(file).Rgba8, $"level {level}");
            if (level == 0) stored = file.Length;
            else True(file.Length < stored * 0.9, $"level {level}: {file.Length} bytes is not smaller than stored {stored}");
        }
    }

    [TestMethod]
    public async Task PngEncoderGpu_TextChunks_RoundTrip_AndHostReaderSeesThem()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PngEncoderGpu(acc);
            using var dec = new PngDecoderGpu(acc);
            const int w = 17, h = 11;
            var src = PicturePixels(w, h, 0x5E09, opaque: true, gray: false);
            var text = new Dictionary<string, string>
            {
                ["prompt"] = "{\"1\": {\"class_type\": \"KSampler\", \"inputs\": {\"seed\": 42}}}",
                ["workflow"] = new string('x', 5000),
                ["Author"] = "Todd Tänner",                        // Latin-1 -> tEXt
                ["Comment"] = "snow ☃ and テスト",    // not Latin-1 -> iTXt (UTF-8)
            };
            var file = await enc.EncodeAsync(src, w, h, new PngEncodeOptions { TextChunks = text });

            var png = ReadPngOnHost(file);   // checks every chunk CRC
            AssertPixelsEqual(src, png.Rgba8, "pixels with text chunks");
            True(png.Chunks.Any(c => c.Type == "tEXt"), "a tEXt chunk");
            True(png.Chunks.Any(c => c.Type == "iTXt"), "an iTXt chunk");

            var read = await dec.ReadTextAsync(file);
            Equal(text.Count, read.Count, "text chunk count");
            foreach (var kv in text) Equal(kv.Value, read[kv.Key], $"text chunk '{kv.Key}'");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task PngEncoderGpu_ArrayViewOverload_WritesCallerBuffer_AndGpuDecoderReadsItBack()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PngEncoderGpu(acc);
            using var dec = new PngDecoderGpu(acc);
            const int w = 50, h = 37;
            var src = PicturePixels(w, h, 0x5E0A, opaque: false, gray: false);
            using var dIn = acc.Allocate1D<byte>(src.Length + 5);
            DeviceTransfer.Upload(dIn.View, src);
            long cap = PngEncoderGpu.GetMaxEncodedSize(w, h);
            using var dOut = acc.Allocate1D<byte>(cap + 9);
            var fill = new byte[dOut.Length];
            System.Array.Fill(fill, (byte)0xAB);
            DeviceTransfer.Upload(dOut.View, fill);
            int n = await enc.EncodeAsync(dIn.View, w, h, dOut.View);
            var file = await DeviceTransfer.DownloadAsync(dOut.View, n);

            AssertPixelsEqual(src, ReadPngOnHost(file).Rgba8, "host reader");
            var decoded = await dec.DecodeAsync(file);
            AssertPixelsEqual(src, decoded.Pixels, "GPU decoder reads the GPU encoder's output");

            // Too-small output is rejected up front.
            using var tiny = acc.Allocate1D<byte>(100);
            ArgumentException? caught = null;
            try { await enc.EncodeAsync(dIn.View, w, h, tiny.View); } catch (ArgumentException ex) { caught = ex; }
            NotNull(caught, "undersized output must throw ArgumentException");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // ---- bounded launches: images that need many launches of every resumable stage ----

    /// <summary>
    /// 1500x900 RGBA: rows wider than one filter-cost segment, 83 blocks in three waves, layout and CRC pieces per chunk;
    /// the host reader checks every chunk CRC, the Adler-32 and the pixels, and the GPU decoder reads it back too.
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task PngEncoderGpu_LargeImage_ManyBlocks_RoundTripsOnHostReader()
    {
        const int w = 1500, h = 900;
        var src = PicturePixels(w, h, 0x5E0B, opaque: false, gray: false);
        var file = await EncodePngManagedAsync(src, w, h, null);
        var png = ReadPngOnHost(file);
        True(png.IdatCount > 64, $"expected many IDAT blocks, got {png.IdatCount}");
        True(png.FilterTypes.Distinct().Count() >= 2, "adaptive filtering should mix filter types");
        AssertPixelsEqual(src, png.Rgba8, "1500x900 host reader");

        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new PngDecoderGpu(acc);
            AssertPixelsEqual(src, (await dec.DecodeAsync(file)).Pixels, "1500x900 GPU decoder");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// Level 3 with 1 MiB blocks (IDAT chunks of 100+ CRC pieces): the parallel parse writes the bytes of the serial
    /// ParseStep oracle, run with a lazy-parse compare budget far below what the blocks need (it finishes only through the
    /// host's unfinished-parse loop).
    /// </summary>
    [TestMethod(Timeout = 600000)]
    public async Task PngEncoderGpu_Level3_BigBlocks_ParallelParse_SameBytesAsSerialParse()
    {
        const int w = 1200, h = 700;
        var src = PicturePixels(w, h, 0x5E0C, opaque: false, gray: false);
        var options = new PngEncodeOptions { CompressionLevel = 3, BlockSize = 1 << 20 };
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var parallel = new PngEncoderGpu(acc);
            using var serial = new PngEncoderGpu(acc) { SerialParse = true, ParseCompareBudget = 4096 };
            byte[] a = await parallel.EncodeAsync(src, w, h, options);
            byte[] b = await serial.EncodeAsync(src, w, h, options);
            EqualBytes(b, a, "level 3 PNG: parallel parse vs the serial ParseStep (4096-compare budget)");
            AssertPixelsEqual(src, ReadPngOnHost(b).Rgba8, "1200x700 level 3 host reader");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
