// Tests for BmpEncoderGpu (Images/Bmp).
//
// Oracles: (1) our own decoder, which is itself gated against hand-built truth in the decoder tests; (2) the
// documented header layout, checked byte by byte against the BMP structures; (3) Pillow, an independent decoder
// (desktop lanes with SPAWNDEV_IMG_PYTHON set; otherwise reported Unsupported, never passed).
// What can fail: wrong header fields, wrong row padding / bottom-up order, wrong channel order, a palette that is
// unsorted or incomplete, the >256 colour rejection, the ArrayView1D overload writing outside its range.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.Bmp;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Deterministic RGBA noise; <paramref name="colours"/> limits the image to that many distinct opaque colours.</summary>
    private static byte[] MakeRgba(int w, int h, int seed, int colours = 0, bool withAlpha = true)
    {
        var rng = new Random(seed);
        var px = new byte[w * h * 4];
        byte[]? palette = null;
        if (colours > 0)
        {
            palette = new byte[colours * 4];
            // Distinct by construction: the first channel is the index.
            for (int c = 0; c < colours; c++)
            {
                palette[c * 4] = (byte)c;
                palette[c * 4 + 1] = (byte)rng.Next(256);
                palette[c * 4 + 2] = (byte)rng.Next(256);
                palette[c * 4 + 3] = 255;
            }
        }
        for (int i = 0; i < w * h; i++)
        {
            if (palette != null)
            {
                // Make sure every palette colour occurs: first `colours` pixels walk the palette.
                int c = i < colours ? i : rng.Next(colours);
                System.Array.Copy(palette, c * 4, px, i * 4, 4);
            }
            else
            {
                px[i * 4] = (byte)rng.Next(256);
                px[i * 4 + 1] = (byte)rng.Next(256);
                px[i * 4 + 2] = (byte)rng.Next(256);
                px[i * 4 + 3] = withAlpha ? (byte)(1 + rng.Next(255)) : (byte)255;
            }
        }
        return px;
    }

    private static byte[] WithOpaqueAlpha(byte[] rgba)
    {
        var copy = (byte[])rgba.Clone();
        for (int i = 3; i < copy.Length; i += 4) copy[i] = 255;
        return copy;
    }

    private static uint Le32At(byte[] b, int o) => (uint)(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24));

    [TestMethod]
    public async Task BmpEncoderGpu_Bgra32_RoundTrips_AndWritesTheDocumentedHeader()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BmpEncoderGpu(acc);
            using var dec = new BmpDecoderGpu(acc);
            var failures = new List<string>();
            foreach (var (w, h) in new[] { (1, 1), (3, 5), (4, 4), (13, 7), (33, 2), (64, 48) })
            {
                var rgba = MakeRgba(w, h, 100 + w * 31 + h);
                var file = await enc.EncodeAsync(rgba, w, h);
                if (file.Length != BmpEncoderGpu.GetMaxEncodedSize(w, h))
                    failures.Add($"{w}x{h}: length {file.Length} != GetMaxEncodedSize {BmpEncoderGpu.GetMaxEncodedSize(w, h)}");
                if (file[0] != 'B' || file[1] != 'M') failures.Add($"{w}x{h}: no BM signature");
                if (Le32At(file, 2) != (uint)file.Length) failures.Add($"{w}x{h}: bfSize {Le32At(file, 2)} != {file.Length}");
                if (Le32At(file, 10) != 122) failures.Add($"{w}x{h}: bfOffBits {Le32At(file, 10)} != 122");
                if (Le32At(file, 14) != 108) failures.Add($"{w}x{h}: header size {Le32At(file, 14)} != 108 (V4)");
                if ((int)Le32At(file, 18) != w || (int)Le32At(file, 22) != h) failures.Add($"{w}x{h}: header size fields {Le32At(file, 18)}x{(int)Le32At(file, 22)} (positive height = bottom-up)");
                if (file[28] != 32) failures.Add($"{w}x{h}: bpp {file[28]}");
                if (Le32At(file, 30) != 3) failures.Add($"{w}x{h}: compression {Le32At(file, 30)} != BI_BITFIELDS");
                if (Le32At(file, 54) != 0x00FF0000 || Le32At(file, 58) != 0x0000FF00 || Le32At(file, 62) != 0x000000FF || Le32At(file, 66) != 0xFF000000)
                    failures.Add($"{w}x{h}: channel masks are not BGRA8888");
                var back = await dec.DecodeAsync(file);
                var diff = ComparePixels(rgba, back.Pixels, 4, w);
                if (diff != null) failures.Add($"{w}x{h}: round trip: {diff}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task BmpEncoderGpu_Bgr24_RoundTrips_AlphaDropped_RowsPadded()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BmpEncoderGpu(acc);
            using var dec = new BmpDecoderGpu(acc);
            var failures = new List<string>();
            var options = new BmpEncodeOptions { Format = BmpEncodeFormat.Bgr24 };
            foreach (var (w, h) in new[] { (1, 1), (3, 5), (4, 4), (13, 7), (33, 2) })
            {
                var rgba = MakeRgba(w, h, 200 + w * 17 + h);
                var file = await enc.EncodeAsync(rgba, w, h, options);
                int stride = ((w * 24 + 31) / 32) * 4;
                if (file.Length != 54 + stride * h) failures.Add($"{w}x{h}: length {file.Length} != {54 + stride * h}");
                if (file[28] != 24 || Le32At(file, 30) != 0 || Le32At(file, 14) != 40) failures.Add($"{w}x{h}: header is not a 24 bit BITMAPINFOHEADER BI_RGB");
                // Padding bytes are zero.
                for (int row = 0; row < h; row++)
                    for (int p = w * 3; p < stride; p++)
                        if (file[54 + row * stride + p] != 0) failures.Add($"{w}x{h}: row {row} padding byte {p} is not 0");
                // Bottom-up, BGR: the last file row is the first image row.
                int lastRow = 54 + (h - 1) * stride;
                if (file[lastRow] != rgba[2] || file[lastRow + 1] != rgba[1] || file[lastRow + 2] != rgba[0])
                    failures.Add($"{w}x{h}: first image pixel is not stored as B, G, R in the last file row");
                var back = await dec.DecodeAsync(file);
                var diff = ComparePixels(WithOpaqueAlpha(rgba), back.Pixels, 4, w);
                if (diff != null) failures.Add($"{w}x{h}: round trip: {diff}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task BmpEncoderGpu_Indexed8_RoundTrips_WithASortedExactPalette()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BmpEncoderGpu(acc);
            using var dec = new BmpDecoderGpu(acc);
            var failures = new List<string>();
            var options = new BmpEncodeOptions { Format = BmpEncodeFormat.Indexed8 };
            foreach (var (w, h, colours) in new[] { (5, 3, 2), (13, 7, 17), (33, 9, 256), (16, 16, 1) })
            {
                var rgba = MakeRgba(w, h, 300 + colours, colours);
                var file = await enc.EncodeAsync(rgba, w, h, options);
                int stride = ((w * 8 + 31) / 32) * 4;
                if (file[28] != 8) failures.Add($"{w}x{h}x{colours}: bpp {file[28]}");
                if (Le32At(file, 46) != (uint)colours) failures.Add($"{w}x{h}x{colours}: biClrUsed {Le32At(file, 46)} != {colours}");
                if (file.Length != 54 + colours * 4 + stride * h) failures.Add($"{w}x{h}x{colours}: length {file.Length}");
                // Palette entries (B, G, R, 0) ascend by 0xRRGGBB.
                long prev = -1;
                for (int e = 0; e < colours; e++)
                {
                    int o = 54 + e * 4;
                    long rgb = (file[o + 2] << 16) | (file[o + 1] << 8) | file[o];
                    if (rgb <= prev) failures.Add($"{w}x{h}x{colours}: palette entry {e} (0x{rgb:X6}) is not above the previous (0x{prev:X6})");
                    prev = rgb;
                }
                var back = await dec.DecodeAsync(file);
                var diff = ComparePixels(WithOpaqueAlpha(rgba), back.Pixels, 4, w);
                if (diff != null) failures.Add($"{w}x{h}x{colours}: round trip: {diff}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task BmpEncoderGpu_Indexed8_MoreThan256Colours_IsRejected()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BmpEncoderGpu(acc);
            var options = new BmpEncodeOptions { Format = BmpEncodeFormat.Indexed8 };
            // 257 distinct colours: one more than a palette holds.
            var rgba = new byte[17 * 17 * 4];
            for (int i = 0; i < 17 * 17; i++)
            {
                int c = i < 257 ? i : 0;
                rgba[i * 4] = (byte)(c & 0xFF); rgba[i * 4 + 1] = (byte)(c >> 8); rgba[i * 4 + 2] = 7; rgba[i * 4 + 3] = 255;
            }
            InvalidOperationException? caught = null;
            try { await enc.EncodeAsync(rgba, 17, 17, options); }
            catch (InvalidOperationException ex) { caught = ex; }
            NotNull(caught, "257 colours must be rejected");
            Contains("at most 256 distinct colours", caught!.Message, "reason");

            // Exactly 256 colours is fine.
            var ok = MakeRgba(16, 16, 5, 256);
            var file = await enc.EncodeAsync(ok, 16, 16, options);
            Equal(256u, Le32At(file, 46), "256 colours fit");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task BmpEncoderGpu_ArrayViewOverload_MatchesManaged_AndStaysInsideItsRange()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BmpEncoderGpu(acc);
            var failures = new List<string>();
            foreach (var format in new[] { BmpEncodeFormat.Bgra32, BmpEncodeFormat.Bgr24, BmpEncodeFormat.Indexed8 })
            {
                var options = new BmpEncodeOptions { Format = format };
                int w = 21, h = 6;
                var rgba = MakeRgba(w, h, 77, format == BmpEncodeFormat.Indexed8 ? 40 : 0);
                var managed = await enc.EncodeAsync(rgba, w, h, options);

                using var dIn = acc.Allocate1D<byte>(rgba.Length + 5);
                DeviceTransfer.Upload(dIn.View, rgba);
                int max = BmpEncoderGpu.GetMaxEncodedSize(w, h, options);
                using var dOut = acc.Allocate1D<byte>(max + 9);
                var fill = new byte[dOut.Length];
                System.Array.Fill(fill, (byte)0xAB);
                DeviceTransfer.Upload(dOut.View, fill);
                int length = await enc.EncodeAsync(dIn.View, w, h, dOut.View, options);
                if (length != managed.Length) failures.Add($"{format}: ArrayView1D length {length} != managed {managed.Length}");
                var bytes = await DeviceTransfer.DownloadAsync(dOut.View, dOut.Length);
                for (int i = 0; i < managed.Length; i++)
                    if (bytes[i] != managed[i]) { failures.Add($"{format}: byte {i} differs"); break; }
                for (int i = length; i < bytes.Length; i++)
                    if (bytes[i] != 0xAB) { failures.Add($"{format}: wrote past the returned length at byte {i}"); break; }
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task BmpEncoderGpu_PillowDecodesOurFiles_ToTheSamePixels()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BmpEncoderGpu(acc);
            var failures = new List<string>();
            foreach (var format in new[] { BmpEncodeFormat.Bgra32, BmpEncodeFormat.Bgr24, BmpEncodeFormat.Indexed8 })
            {
                var options = new BmpEncodeOptions { Format = format };
                int w = 19, h = 11;
                var rgba = MakeRgba(w, h, 4242, format == BmpEncodeFormat.Indexed8 ? 200 : 0);
                var file = await enc.EncodeAsync(rgba, w, h, options);
                var pillow = PillowOracle.Decode(file, ".bmp");
                if (pillow.Format != "BMP" || pillow.Width != w || pillow.Height != h || pillow.Frames != 1)
                    failures.Add($"{format}: Pillow sees {pillow.Format} {pillow.Width}x{pillow.Height} x{pillow.Frames}");
                var expected = format == BmpEncodeFormat.Bgra32 ? rgba : WithOpaqueAlpha(rgba);
                var diff = ComparePixels(expected, pillow.FramesRgba[0], 4, w);
                if (diff != null) failures.Add($"{format}: Pillow pixels: {diff}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
