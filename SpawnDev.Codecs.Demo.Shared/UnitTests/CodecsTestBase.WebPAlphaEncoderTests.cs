// Tests for WebPLosslessEncoderGpu.EncodeAlphaAsync (the ALPH chunk of a lossy WebP).
//
// The truth is the SOURCE plane (TestData/WebP/alph_*.alpha8.gz, the planes the decoder fixtures were built from). The
// encoder is asked for an ALPH payload for each plane x filter x quality; the payload is put into a VP8X file next to a real
// libwebp-made VP8 chunk of the same size (the colour half; an ALPH chunk does not depend on it) and the file must
//   (a) decode to EXACTLY the plane in libwebp (through Pillow: alpha channel of the RGBA decode),
//   (b) decode to EXACTLY the plane in an independent pure-Python ALPH decoder (scripts/images/webp_ref.py),
//   (c) decode to EXACTLY the plane in WebPDecoderGpu.DecodeAlphaAsync.
// The header byte is checked (requested filter, pre-processing 0, reserved 0, method 0 when Compress = false).
// Part (a)/(b) need Pillow: set CODECS_PILLOW_PYTHON (see CodecsTestBase.WebPLosslessEncoderGpuTests.cs).
//
// Everything stays on the accelerator: the plane is uploaded once, the payload is written into a device buffer, the file
// around it is assembled there (the host only uploads the few header bytes of the fixture), it is decoded into another
// device buffer and compared with the plane by DeviceVerify. Only header bytes, lengths and mismatch counts come back.
// The payload is downloaded only when the Pillow check will read the files (desktop with CODECS_PILLOW_PYTHON set).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static readonly string[] WebPAlphaPlaneNames =
    {
        "smooth_64x48", "mask_61x45", "noise_33x29", "steps_50x40", "flat255_32x24", "flat0_17x11",
        "tall_1x37", "wide_41x1", "one_1x1", "big_300x200",
    };

    /// <summary>A source plane (width, height, bytes) and a lossy VP8X file of the same size to host an ALPH chunk.</summary>
    private static (int W, int H, byte[] Plane, byte[] HostFile) WebPAlphaSource(string name)
    {
        string stem = "alph_" + name + (name.StartsWith("big") ? "_f1_vp8l" : "_f0_raw");
        var row = WebPAlphaManifest().Single(r => r.Name == stem);
        return (row.Width, row.Height, WebPAlphaTruth(stem), ReadEmbedded(WebPFixturePrefix + stem + ".webp"));
    }

    private static string AlphaFilterName(WebPAlphaFilter f) => f.ToString().ToLowerInvariant();

    /// <summary>True when the Pillow / libwebp check of written files can run (desktop + CODECS_PILLOW_PYTHON).</summary>
    private static bool PillowAvailable()
    {
        if (OperatingSystem.IsBrowser()) return false;
        var python = Environment.GetEnvironmentVariable("CODECS_PILLOW_PYTHON");
        return !string.IsNullOrEmpty(python) && File.Exists(python);
    }

    /// <summary>
    /// Assemble on the accelerator the host fixture with its ALPH chunk replaced by <paramref name="payload"/>[0..length):
    /// the host uploads the RIFF header and the other chunks (a few hundred bytes of the fixture), the payload is copied
    /// device to device. Returns the file length in <paramref name="file"/>.
    /// </summary>
    private static int AlphaFileOnDevice(byte[] host, ArrayView1D<byte, Stride1D.Dense> payload, int length,
        ArrayView1D<byte, Stride1D.Dense> file)
    {
        var chunks = WebPChunksOf(host);
        int i = chunks.FindIndex(c => c.Fcc == "ALPH");
        if (i < 0) throw new InvalidOperationException("the host fixture has no ALPH chunk");
        using var head = new MemoryStream();
        head.Write("RIFF\0\0\0\0WEBP"u8);
        foreach (var (fcc, p) in chunks.Take(i)) WriteChunk(head, fcc, p);
        head.Write("ALPH"u8);
        head.Write(BitConverter.GetBytes(length));
        using var tail = new MemoryStream();
        if ((length & 1) == 1) tail.WriteByte(0);
        foreach (var (fcc, p) in chunks.Skip(i + 1)) WriteChunk(tail, fcc, p);
        var headBytes = head.ToArray();
        var tailBytes = tail.ToArray();
        int total = headBytes.Length + length + tailBytes.Length;
        BitConverter.GetBytes(total - 8).CopyTo(headBytes, 4);
        if (file.Length < total) throw new ArgumentException($"file buffer holds {file.Length} bytes, {total} needed", nameof(file));
        DeviceTransfer.Upload(file.SubView(0, headBytes.Length), headBytes);
        payload.SubView(0, length).CopyTo(file.SubView(headBytes.Length, length));
        if (tailBytes.Length > 0) DeviceTransfer.Upload(file.SubView(headBytes.Length + length, tailBytes.Length), tailBytes);
        return total;

        static void WriteChunk(MemoryStream ms, string fcc, byte[] p)
        {
            ms.Write(System.Text.Encoding.ASCII.GetBytes(fcc));
            ms.Write(BitConverter.GetBytes(p.Length));
            ms.Write(p);
            if ((p.Length & 1) == 1) ms.WriteByte(0);
        }
    }

    /// <summary>Encode the device plane, decode the result through a device-assembled file and compare on the accelerator.</summary>
    private sealed class AlphaRoundTrip : IDisposable
    {
        public required Accelerator Acc;
        public required WebPLosslessEncoderGpu Enc;
        public required WebPDecoderGpu Dec;
        public required MemoryBuffer1D<byte, Stride1D.Dense> Plane;
        public required MemoryBuffer1D<byte, Stride1D.Dense> Payload;
        public required MemoryBuffer1D<byte, Stride1D.Dense> File;
        public required MemoryBuffer1D<byte, Stride1D.Dense> Back;
        public required byte[] Host;
        public required int W, H;

        public static AlphaRoundTrip Create(Accelerator acc, WebPLosslessEncoderGpu enc, WebPDecoderGpu dec, int w, int h, byte[] plane, byte[] host)
        {
            int max = WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize(w, h);
            return new AlphaRoundTrip
            {
                Acc = acc, Enc = enc, Dec = dec, W = w, H = h, Host = host,
                Plane = DeviceVerify.Upload(acc, plane),
                Payload = acc.Allocate1D<byte>(max),
                File = acc.Allocate1D<byte>(host.Length + max + 16),
                Back = acc.Allocate1D<byte>(Math.Max(1, (long)w * h)),
            };
        }

        /// <summary>Returns the payload length and its header byte; the payload is left in <see cref="Payload"/>.</summary>
        public async Task<(int Length, byte Header)> EncodeAsync(WebPAlphaEncodeOptions options)
        {
            int length = await Enc.EncodeAlphaAsync(Plane.View, W, H, Payload.View, options);
            return (length, await DeviceVerify.ByteAtAsync(Payload.View, 0));
        }

        /// <summary>Null when the payload in <see cref="Payload"/> decodes to exactly the plane in WebPDecoderGpu.</summary>
        public async Task<string?> DecodeAndCompareAsync(int length)
        {
            int fileLength = AlphaFileOnDevice(Host, Payload.View, length, File.View);
            await Dec.DecodeAlphaAsync(File.View, fileLength, Back.View);
            return await DeviceVerify.CompareAsync(Acc, Back.View, Plane.View, (long)W * H, 1, W);
        }

        /// <summary>The assembled file on the host, only for the Pillow check of written files.</summary>
        public async Task<byte[]> DownloadFileAsync(int length)
        {
            int fileLength = AlphaFileOnDevice(Host, Payload.View, length, File.View);
            return await DeviceTransfer.DownloadAsync(File.View, fileLength);
        }

        public void Dispose()
        {
            Plane.Dispose(); Payload.Dispose(); File.Dispose(); Back.Dispose();
        }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_Alpha_EveryPlaneAndFilter_DecodeBitExact_InLibwebpAndBothDecoders()
    {
        bool pillow = PillowAvailable();
        string? dir = pillow ? WebPEncoderOutputDir("alpha") : null;
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        var failures = new List<string>();
        var report = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var name in WebPAlphaPlaneNames)
            {
                var (w, h, plane, host) = WebPAlphaSource(name);
                using var rt = AlphaRoundTrip.Create(acc, enc, dec, w, h, plane, host);
                int maxSize = WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize(w, h);
                bool big = name.StartsWith("big");
                var sizes = new Dictionary<(WebPAlphaFilter, int), int>();

                async Task KeepForPillowAsync(string tag, int length)
                {
                    if (dir == null) return;
                    string stem = Path.Combine(dir, tag);
                    await File.WriteAllBytesAsync(stem + ".webp", await rt.DownloadFileAsync(length));
                    await File.WriteAllBytesAsync(stem + ".alpha8", plane);
                    await File.WriteAllTextAsync(stem + ".size", $"{w} {h}");
                }

                foreach (var filter in new[] { WebPAlphaFilter.None, WebPAlphaFilter.Horizontal, WebPAlphaFilter.Vertical, WebPAlphaFilter.Gradient, WebPAlphaFilter.Auto })
                {
                    foreach (int quality in big ? new[] { 50 } : new[] { 0, 30, 60, 100 })
                    {
                        string tag = $"{name}_{AlphaFilterName(filter)}_q{quality}";
                        try
                        {
                            var (length, header) = await rt.EncodeAsync(new WebPAlphaEncodeOptions { Quality = quality, Filter = filter });
                            if (length > maxSize) failures.Add($"{tag}: {length} bytes exceeds GetMaxAlphaEncodedSize");
                            int method = header & 3, chosen = (header >> 2) & 3, pre = (header >> 4) & 3, rsv = header >> 6;
                            if (pre != 0 || rsv != 0 || method > 1) failures.Add($"{tag}: header byte 0x{header:X2} has pre-processing {pre}, reserved {rsv}, method {method}");
                            if (filter != WebPAlphaFilter.Auto && chosen != (int)filter)
                                failures.Add($"{tag}: asked for filter {(int)filter}, header says {chosen}");
                            sizes[(filter, quality)] = length;
                            if (filter == WebPAlphaFilter.Auto) report.Add($"{name}\tq{quality}\tauto picked filter {chosen}\tmethod {method}\tbytes {length}");
                            var diff = await rt.DecodeAndCompareAsync(length);
                            if (diff != null) failures.Add($"{tag} [our decoder]: {diff}");
                            await KeepForPillowAsync(tag, length);
                        }
                        catch (Exception ex)
                        {
                            failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}");
                        }
                    }
                }
                // From effort 50 on, Auto encodes with every filter and keeps the smallest: it must equal the best explicit filter
                // EXACTLY (a pick by a rougher estimate, or a wrong tie-break, would show up as a larger file somewhere).
                foreach (int quality in big ? new[] { 50 } : new[] { 60, 100 })
                {
                    if (!sizes.ContainsKey((WebPAlphaFilter.Auto, quality))) continue;
                    int best = new[] { WebPAlphaFilter.None, WebPAlphaFilter.Horizontal, WebPAlphaFilter.Vertical, WebPAlphaFilter.Gradient }
                        .Min(f => sizes.TryGetValue((f, quality), out var s) ? s : int.MaxValue);
                    int auto = sizes[(WebPAlphaFilter.Auto, quality)];
                    report.Add($"{name}\tq{quality} auto {auto}\tbest {best}");
                    if (auto != best) failures.Add($"{name} q{quality}: Auto gave {auto} bytes, the best explicit filter {best}");
                }

                // Compress = false: always method 0, exactly 1 + w*h bytes
                foreach (var filter in new[] { WebPAlphaFilter.None, WebPAlphaFilter.Gradient, WebPAlphaFilter.Auto })
                {
                    string tag = $"{name}_{AlphaFilterName(filter)}_raw";
                    try
                    {
                        var (length, header) = await rt.EncodeAsync(new WebPAlphaEncodeOptions { Filter = filter, Compress = false });
                        if ((header & 3) != 0 || length != 1 + w * h) failures.Add($"{tag}: Compress = false must give method 0 and {1 + w * h} bytes, got header 0x{header:X2}, {length} bytes");
                        var diff = await rt.DecodeAndCompareAsync(length);
                        if (diff != null) failures.Add($"{tag} [our decoder]: {diff}");
                        await KeepForPillowAsync(tag, length);
                    }
                    catch (Exception ex)
                    {
                        failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        True(failures.Count == 0, $"{failures.Count} ALPH encoder failures:\n" + string.Join("\n", failures));
        if (dir == null) return; // our decoder checked every payload; the libwebp check needs a desktop with Pillow
        await File.WriteAllLinesAsync(Path.Combine(dir, "autopick.txt"), report);
        var verify = await RunPillowVerifyAsync(dir);
        True(verify.Contains("SUMMARY"), "verification script did not finish");
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_Alpha_IncompressiblePlane_FallsBackToRaw_WithinTheBound()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            int w = 301, h = 251;
            var noise = new byte[w * h];
            new Random(99).NextBytes(noise);
            using var dNoise = DeviceVerify.Upload(acc, noise);
            using var dOut = acc.Allocate1D<byte>(WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize(w, h));
            foreach (var filter in new[] { WebPAlphaFilter.None, WebPAlphaFilter.Auto })
            {
                foreach (int q in new[] { 0, 50, 100 })
                {
                    int length = await enc.EncodeAlphaAsync(dNoise.View, w, h, dOut.View, new WebPAlphaEncodeOptions { Quality = q, Filter = filter });
                    byte header = await DeviceVerify.ByteAtAsync(dOut.View, 0);
                    string tag = $"noise {filter} q{q}";
                    // random bytes cannot be compressed: the payload must be the raw plane (1 header byte + w*h), never larger
                    Equal(1 + w * h, length, tag + " length");
                    Equal(0, header & 3, tag + " method");
                    // stored raw, None and Auto both mean "no filter": the data after the header is the plane itself
                    Equal(0, (header >> 2) & 3, tag + " filter");
                    Equal(null, await DeviceVerify.CompareAsync(acc, dOut.View.SubView(1, w * h), dNoise.View, w * h, 1, w), tag + " raw body");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_Alpha_ArrayViewOverload_WritesOnlyThePayload()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var name in new[] { "smooth_64x48", "mask_61x45", "one_1x1" })
            {
                var (w, h, plane, host) = WebPAlphaSource(name);
                using var dIn = DeviceVerify.Upload(acc, plane, slack: 5);
                int max = WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize(w, h);
                using var dOut = DeviceVerify.Poisoned(acc, max + 64);
                int length = await enc.EncodeAlphaAsync(dIn.View, w, h, dOut.View);
                True(length >= 2 && length <= max, $"{name}: length {length} within [2, {max}]");
                Equal(null, await DeviceVerify.AllEqualAsync(acc, dOut.View.SubView(length, 64), 64, 0xAB), $"{name}: bytes past the end of the payload");
                using var dFile = acc.Allocate1D<byte>(host.Length + max + 16);
                int fileLength = AlphaFileOnDevice(host, dOut.View, length, dFile.View);
                using var dBack = acc.Allocate1D<byte>(Math.Max(1, w * h));
                await dec.DecodeAlphaAsync(dFile.View, fileLength, dBack.View);
                Equal(null, await DeviceVerify.CompareAsync(acc, dBack.View, dIn.View, w * h, 1, w), name);
                // the managed overload (a few KB here) must give the same payload
                var managed = await enc.EncodeAlphaAsync(plane, w, h);
                Equal(length, managed.Length, $"{name}: managed overload length");
                Equal(null, await DeviceVerify.CompareAsync(acc, dOut.View, managed), $"{name}: managed overload bytes");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_Alpha_BadArguments_AreRejected()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            var px = new byte[16 * 16];
            async Task<bool> Throws<T>(Func<Task> action) where T : Exception
            {
                try { await action(); }
                catch (T) { return true; }
                return false;
            }
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 0, 16)), "width 0");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 16, 16385)), "height too large");
            True(await Throws<ArgumentException>(() => enc.EncodeAlphaAsync(px, 17, 16)), "buffer shorter than width*height");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 16, 16, new WebPAlphaEncodeOptions { Quality = -1 })), "quality -1");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 16, 16, new WebPAlphaEncodeOptions { Filter = (WebPAlphaFilter)9 })), "filter 9");
            using var dIn = acc.Allocate1D<byte>(px.Length);
            using var dOut = acc.Allocate1D<byte>(100);
            True(await Throws<ArgumentException>(() => enc.EncodeAlphaAsync(dIn.View, 16, 16, dOut.View)), "output smaller than GetMaxAlphaEncodedSize");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
