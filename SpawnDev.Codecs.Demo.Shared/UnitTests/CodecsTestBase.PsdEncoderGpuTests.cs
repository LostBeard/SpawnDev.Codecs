// The layered PSD writer (PsdEncoderGpu) checked with PsdTestReader, which decodes the whole file from the specification. Every case is
// compared pixel by pixel (layers and merged image), and the metadata (rectangles, blend keys, opacity, flags, names) must be exactly what was asked for.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Psd;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] PsdPixels(int w, int h, int seed, int noise)
    {
        var p = new byte[w * h * 4];
        uint s = (uint)seed * 2654435761u + 1;
        for (int y = 0, i = 0; y < h; y++)
            for (int x = 0; x < w; x++, i++)
            {
                s = s * 1664525 + 1013904223;
                bool flat = noise == 0 || (s >> 16) % 8 >= noise;
                p[i * 4] = flat ? (byte)(x / 8 * 20 + seed) : (byte)(s >> 16);
                p[i * 4 + 1] = flat ? (byte)(y / 8 * 16) : (byte)(s >> 20);
                p[i * 4 + 2] = flat ? (byte)(seed * 37) : (byte)(s >> 8);
                p[i * 4 + 3] = flat ? (byte)((x / 5 + y / 5) % 2 == 0 ? 255 : 90) : (byte)(s >> 12);
            }
        return p;
    }

    private static void ComparePixels(string what, byte[] expected, byte[] actual)
    {
        True(expected.Length == actual.Length, $"{what}: {actual.Length} bytes, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i]) throw new Exception($"{what}: first difference at byte {i} (pixel {i / 4}, channel {i % 4}): {actual[i]}, expected {expected[i]}");
    }

    private static PsdTestReader ReadPsd(byte[] file)
    {
        var r = PsdTestReader.Read(file);
        Equal(1, r.Version, "version");
        Equal(8, r.Depth, "bits per channel");
        Equal(3, r.Mode, "RGB colour mode");
        return r;
    }

    private static void ComparePsd(PsdDocument doc, byte[] file)
    {
        var r = ReadPsd(file);
        Equal(doc.Width, r.Width, "canvas width");
        Equal(doc.Height, r.Height, "canvas height");
        Equal(doc.Composite.HasAlpha ? 4 : 3, r.Channels, "channels of the merged image");
        Equal(doc.Composite.HasAlpha ? -doc.Layers.Count : doc.Layers.Count, r.LayerCount, "layer count");
        Equal(doc.Layers.Count, r.Layers.Count, "layers read");
        for (int i = 0; i < doc.Layers.Count; i++)
        {
            var want = doc.Layers[i]; var got = r.Layers[i];
            Equal(want.Left, got.Left, $"layer {i} left");
            Equal(want.Top, got.Top, $"layer {i} top");
            Equal(want.Left + want.Width, got.Right, $"layer {i} right");
            Equal(want.Top + want.Height, got.Bottom, $"layer {i} bottom");
            True(got.ChannelIds.SequenceEqual(new[] { -1, 0, 1, 2 }), $"layer {i}: channels {string.Join(",", got.ChannelIds)}, expected -1 0 1 2");
            Equal(PsdFormat.BlendKey(want.Blend), got.Key, $"layer {i} blend key");
            Equal((int)want.Opacity, got.Opacity, $"layer {i} opacity");
            Equal(want.Visible ? 0x08 : 0x0A, got.Flags, $"layer {i} flags");
            Equal(want.Name, got.UnicodeName, $"layer {i} Unicode name");
            ComparePixels($"layer {i} pixels", Pixels(want), got.Rgba());
        }
        for (int c = 0; c < 3 + (doc.Composite.HasAlpha ? 1 : 0); c++)
            ComparePixels($"merged image, channel {c}", MergedPlane(doc, c), r.Composite[c]);
    }

    /// <summary>
    /// What the file stores for channel c of the merged picture. With a merged alpha the colours are matted over white, as Photoshop (and ComfyUI's
    /// export: 73, 51, 40 at alpha 204 reads 109, 92, 83 (theirs 108, 91, 82: their rounding)) writes the merged image; alpha stays straight.
    /// </summary>
    private static byte[] MergedPlane(PsdDocument doc, int c)
    {
        var rgba = Pixels(doc.Composite);
        var plane = new byte[doc.Width * doc.Height];
        for (int i = 0; i < plane.Length; i++)
        {
            int a = rgba[i * 4 + 3], v = rgba[i * 4 + c];
            plane[i] = (byte)(!doc.Composite.HasAlpha || c == 3 ? (c == 3 ? a : v) : (v * a + 255 * (255 - a) + 127) / 255);
        }
        return plane;
    }

    // managed pixels of a layer; views are uploaded from the same bytes in the tests
    private static byte[] Pixels(PsdLayer l) => l.Rgba.ToArray();
    private static byte[] Pixels(PsdComposite c) => c.Rgba.ToArray();

    private static byte[] CompositeOf(int w, int h, int seed) => PsdPixels(w, h, seed, 3);

    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task PsdEncoderGpu_Layers_ReadBackExactly()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PsdEncoderGpu(acc);
            await enc.WarmUpAsync();
            var doc = new PsdDocument
            {
                Width = 120, Height = 90,
                Composite = new PsdComposite { Rgba = CompositeOf(120, 90, 5) },
                Layers = new[]
                {
                    new PsdLayer { Name = "Background", Width = 120, Height = 90, Rgba = PsdPixels(120, 90, 1, 0) },
                    new PsdLayer { Name = "on canvas", Left = 10, Top = 8, Width = 64, Height = 33, Rgba = PsdPixels(64, 33, 2, 2), Opacity = 204, Blend = PsdBlendMode.Multiply },
                    new PsdLayer { Name = "noise", Left = 3, Top = 4, Width = 37, Height = 29, Rgba = PsdPixels(37, 29, 3, 8), Opacity = 1 },
                    new PsdLayer { Name = "off canvas negative", Left = -20, Top = -10, Width = 64, Height = 64, Rgba = PsdPixels(64, 64, 4, 1), Blend = PsdBlendMode.Screen },
                    new PsdLayer { Name = "past the canvas", Left = 100, Top = 80, Width = 70, Height = 50, Rgba = PsdPixels(70, 50, 6, 1), Opacity = 0 },
                    new PsdLayer { Name = "hidden", Width = 20, Height = 20, Rgba = PsdPixels(20, 20, 7, 4), Visible = false },
                    new PsdLayer { Name = "one pixel", Left = 119, Top = 89, Width = 1, Height = 1, Rgba = PsdPixels(1, 1, 8, 0) },
                },
            };
            byte[] file = await enc.EncodeAsync(doc);
            True(file.Length <= PsdEncoderGpu.GetMaxEncodedSize(doc), $"{file.Length} B, the maximum said {PsdEncoderGpu.GetMaxEncodedSize(doc)}");
            ComparePsd(doc, file);
            byte[] again = await enc.EncodeAsync(doc);
            True(file.AsSpan().SequenceEqual(again), "a second encode of the same document gives a different file");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task PsdEncoderGpu_EveryBlendMode_HasItsKey()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PsdEncoderGpu(acc);
            var modes = Enum.GetValues<PsdBlendMode>();
            Equal(26, modes.Length, "blend modes");
            var layers = modes.Select((m, i) => new PsdLayer { Name = m.ToString(), Left = i, Top = i, Width = 3, Height = 2, Blend = m, Rgba = PsdPixels(3, 2, i, 2) }).ToArray();
            var doc = new PsdDocument { Width = 40, Height = 40, Layers = layers, Composite = new PsdComposite { Rgba = CompositeOf(40, 40, 1) } };
            byte[] file = await enc.EncodeAsync(doc);
            ComparePsd(doc, file);
            // the keys ComfyUI writes (Lore, 2026-11-25), in the order of the enum
            var keys = new[] { "norm", "mul ", "scrn", "over", "dark", "lite", "div ", "idiv", "hLit", "sLit", "diff", "smud", "lddg", "lbrn", "vLit", "pLit", "lLit", "hMix", "fsub", "fdiv", "norm", "norm", "hue ", "sat ", "colr", "lum " };
            Equal(keys.Length, modes.Length, "keys");
            for (int i = 0; i < modes.Length; i++) Equal(keys[i], PsdFormat.BlendKey(modes[i]), $"key of {modes[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task PsdEncoderGpu_Names_CompositeAlpha_NoLayers()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PsdEncoderGpu(acc);
            // names: Unicode, empty, ASCII longer than a Pascal string, all lengths around the padding
            var names = new List<string> { "Ünïcødé 層 ✓ 🎨", "", "a", "ab", "abc", "abcd", "abcde", new string('x', 300), "tab\tnewline\ntext" };
            var doc = new PsdDocument
            {
                Width = 16, Height = 12,
                Layers = names.Select((n, i) => new PsdLayer { Name = n, Width = 5, Height = 4, Left = i, Rgba = PsdPixels(5, 4, i, 2) }).ToArray(),
                Composite = new PsdComposite { Rgba = CompositeOf(16, 12, 3), HasAlpha = true },
            };
            byte[] file = await enc.EncodeAsync(doc);
            ComparePsd(doc, file);
            var r = ReadPsd(file);
            Equal("?n?c?d? ? ? ??", r.Layers[0].PascalName, "the Pascal name replaces what ASCII cannot hold");
            True(r.Layers[7].PascalName.Length == 255, "the Pascal name is cut at 255 characters");

            var flat = new PsdDocument { Width = 33, Height = 17, Composite = new PsdComposite { Rgba = CompositeOf(33, 17, 9) } };
            byte[] flatFile = await enc.EncodeAsync(flat);
            ComparePsd(flat, flatFile);
            var flatAlpha = flat with { Composite = flat.Composite with { HasAlpha = true } };
            ComparePsd(flatAlpha, await enc.EncodeAsync(flatAlpha));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task PsdEncoderGpu_RunLengthEdges_AndSizes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PsdEncoderGpu(acc);
            // rows that end and start runs at every length around the limits: 1..3, 127..129, 130, 255..257, 384 (128 * 3), 1025
            var widths = new[] { 1, 2, 3, 4, 127, 128, 129, 130, 131, 255, 256, 257, 383, 384, 385, 513, 1025 };
            foreach (int w in widths)
            {
                int h = 6;
                var rgba = new byte[w * h * 4];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        byte v = y switch
                        {
                            0 => 7,                                                   // one run over the whole row
                            1 => (byte)x,                                             // no runs
                            2 => (byte)(x / 2),                                       // pairs
                            3 => (byte)(x / 3 * 5),                                    // runs of exactly 3
                            4 => (byte)(x < w - 2 ? 1 : x == w - 2 ? 1 : 2),           // the end of the row
                            _ => (byte)((x % 128) < 2 ? 9 : 9 + x % 2),                // long runs cut at 128
                        };
                        for (int c = 0; c < 4; c++) rgba[(y * w + x) * 4 + c] = (byte)(v + c * 60);
                    }
                var doc = new PsdDocument
                {
                    Width = w, Height = h,
                    Layers = new[] { new PsdLayer { Name = "edge", Width = w, Height = h, Rgba = rgba } },
                    Composite = new PsdComposite { Rgba = rgba, HasAlpha = w % 2 == 0 },
                };
                ComparePsd(doc, await enc.EncodeAsync(doc));
            }
            // the limits: the longest row and the tallest column PSD allows
            foreach (var (w, h) in new[] { (30000, 1), (1, 30000), (1, 1), (4096, 7) })
            {
                var rgba = PsdPixels(w, h, w + h, 4);
                var doc = new PsdDocument { Width = w, Height = h, Composite = new PsdComposite { Rgba = rgba }, Layers = new[] { new PsdLayer { Name = "long", Width = w, Height = h, Rgba = rgba } } };
                ComparePsd(doc, await enc.EncodeAsync(doc));
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task PsdEncoderGpu_PsdToolsReadsEveryLayer()
    {
        // psd-tools (MIT) reads the file: names, rectangles (off the canvas too), visibility, opacity, blend keys, every layer pixel and the
        // merged image.
        PsdOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PsdEncoderGpu(acc);
            var doc = new PsdDocument
            {
                Width = 100, Height = 80,
                Composite = new PsdComposite { Rgba = CompositeOf(100, 80, 5) },
                Layers = new[]
                {
                    new PsdLayer { Name = "Background", Width = 100, Height = 80, Rgba = PsdPixels(100, 80, 1, 0) },
                    new PsdLayer { Name = "Ünïcødé 層 ✓", Left = 10, Top = 8, Width = 64, Height = 33, Rgba = PsdPixels(64, 33, 2, 2), Opacity = 204, Blend = PsdBlendMode.Multiply },
                    new PsdLayer { Name = "noise", Left = 3, Top = 40, Width = 37, Height = 29, Rgba = PsdPixels(37, 29, 3, 8), Opacity = 1 },
                    new PsdLayer { Name = "negative", Left = -20, Top = -10, Width = 64, Height = 64, Rgba = PsdPixels(64, 64, 4, 1), Blend = PsdBlendMode.Overlay },
                    new PsdLayer { Name = "past", Left = 70, Top = 60, Width = 70, Height = 50, Rgba = PsdPixels(70, 50, 6, 1), Opacity = 0, Visible = false },
                    new PsdLayer { Name = "one pixel", Left = 99, Top = 79, Width = 1, Height = 1, Rgba = PsdPixels(1, 1, 8, 0) },
                },
            };
            foreach (bool alpha in new[] { false, true })
            {
                var d = doc with { Composite = doc.Composite with { HasAlpha = alpha } };
                byte[] file = await enc.EncodeAsync(d);
                var (info, composite, layers) = PsdOracle.Read(file);
                Equal(d.Width, info.Width, "psd-tools canvas width");
                Equal(d.Height, info.Height, "psd-tools canvas height");
                Equal(d.Layers.Count, info.Layers.Count, "psd-tools layer count");
                Equal(d.Width, info.CompositeWidth, "composite width");
                Equal(d.Height, info.CompositeHeight, "composite height");
                // psd-tools un-mattes a merged image with alpha: exact for opaque pixels (and every pixel without merged alpha); the alpha plane is exact always
                for (int i = 0; i < d.Width * d.Height; i++)
                {
                    int a = d.Composite.Rgba.Span[i * 4 + 3];
                    if (alpha && a != 255)
                    {
                        if (composite[i * 4 + 3] != a) throw new Exception($"merged alpha pixel {i}: psd-tools {composite[i * 4 + 3]}, written {a}");
                        continue;
                    }
                    for (int c = 0; c < 3; c++)
                        if (composite[i * 4 + c] != d.Composite.Rgba.Span[i * 4 + c]) throw new Exception($"merged image (alpha {alpha}): channel {c} pixel {i}: psd-tools {composite[i * 4 + c]}, written {d.Composite.Rgba.Span[i * 4 + c]}");
                }
                for (int k = 0; k < d.Layers.Count; k++)
                {
                    var want = d.Layers[k]; var got = info.Layers[k];
                    Equal(want.Name, got.Name, $"layer {k} name");
                    Equal(want.Left, got.Left, $"layer {k} left");
                    Equal(want.Top, got.Top, $"layer {k} top");
                    Equal(want.Left + want.Width, got.Right, $"layer {k} right");
                    Equal(want.Top + want.Height, got.Bottom, $"layer {k} bottom");
                    Equal(want.Visible, got.Visible, $"layer {k} visible");
                    Equal((int)want.Opacity, got.Opacity, $"layer {k} opacity");
                    Equal(PsdFormat.BlendKey(want.Blend), got.Blend, $"layer {k} blend key");
                    Equal(want.Width, got.Width, $"layer {k} pixel width");
                    Equal(want.Height, got.Height, $"layer {k} pixel height");
                    ComparePixels($"layer {k} ({want.Name}) pixels", want.Rgba.ToArray(), layers[k]);
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task PsdEncoderGpu_ViewsEqualManaged_ManyLayers_AndRejects()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new PsdEncoderGpu(acc);
            byte[] layerRgba = PsdPixels(31, 27, 5, 2), compRgba = CompositeOf(31, 27, 4);
            var managed = new PsdDocument
            {
                Width = 31, Height = 27,
                Layers = Enumerable.Range(0, 60).Select(i => new PsdLayer { Name = $"layer {i}", Width = 31, Height = 27, Left = i % 7 - 3, Top = i % 5 - 2, Rgba = layerRgba, Opacity = (byte)(255 - i), Blend = (PsdBlendMode)(i % 26), Visible = i % 3 != 0 }).ToArray(),
                Composite = new PsdComposite { Rgba = compRgba, HasAlpha = true },
            };
            byte[] fileManaged = await enc.EncodeAsync(managed);
            ComparePsd(managed, fileManaged);

            using var dLayer = acc.Allocate1D(layerRgba);
            using var dComp = acc.Allocate1D(compRgba);
            var viewed = managed with
            {
                Layers = managed.Layers.Select(l => l with { Rgba = default, RgbaView = dLayer.View }).ToArray(),
                Composite = managed.Composite with { Rgba = default, RgbaView = dComp.View },
            };
            byte[] fileViews = await enc.EncodeAsync(viewed);
            True(fileViews.AsSpan().SequenceEqual(fileManaged), "the file from views differs from the file from managed bytes");

            // the output too small, and invalid documents, are refused before anything is written
            int max = PsdEncoderGpu.GetMaxEncodedSize(managed);
            var small = new byte[max / 2];
            using (var dOut = acc.Allocate1D(small))
            {
                bool threw = false;
                try { await enc.EncodeAsync(managed, dOut.View); } catch (ArgumentException) { threw = true; }
                True(threw, "an output that is too small must be refused");
            }
            foreach (var (bad, why) in new (PsdDocument, string)[]
            {
                (managed with { Width = 0 }, "width 0"),
                (managed with { Height = 30001 }, "height 30001"),
                (managed with { Layers = new[] { new PsdLayer { Name = "x", Width = 0, Height = 5, Rgba = layerRgba } } }, "layer width 0"),
                (managed with { Layers = new[] { new PsdLayer { Name = "x", Width = 5, Height = 5, Rgba = new byte[10] } } }, "layer pixels too short"),
                (managed with { Composite = new PsdComposite { Rgba = new byte[10] } }, "composite pixels too short"),
            })
            {
                bool threw = false;
                try { await enc.EncodeAsync(bad); } catch (ArgumentException) { threw = true; }
                True(threw, $"must refuse: {why}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
