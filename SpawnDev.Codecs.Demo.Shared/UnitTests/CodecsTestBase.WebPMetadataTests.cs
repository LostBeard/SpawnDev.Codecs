// Tests for the ICCP / EXIF / XMP metadata of WebP (read: WebPDecoderGpu, write: WebPLosslessEncoderGpu).
//
// Fixtures (TestData/WebP/meta2_*, scripts/images/make_webp_meta_fixtures.py): every chunk subset, VP8X flag bits that do not
// match the chunks present, chunk order, a duplicate chunk, an unknown chunk in between, odd payload lengths (RIFF padding),
// metadata written by libwebp's own muxer on a lossy file with alpha and on an animation, and EXIF orientation in both byte
// orders, with and without the "Exif\0\0" prefix, valid 1..8 and ten malformed forms. The truth for the payloads is what
// libwebp's demuxer reports (Pillow Image.info), written next to each file; the orientation truth is Pillow's EXIF parser for a
// valid value and "upright (1)" for a malformed entry.
//
// Reader parity with libwebp: a metadata chunk counts only when its VP8X flag is set, and only the first chunk of a kind.
// Writer: the files our encoder writes are opened by libwebp's demuxer (Pillow), which must report exactly the metadata that was
// passed in, with VP8X flags that say exactly which chunks exist (scripts/images/verify_webp_files.py: check_metadata).

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
    private sealed record WebPMetaCase(
        string Name, int Width, int Height, int Frames, int Orientation, bool Lossless, bool HasIcc, bool HasExif, bool HasXmp, string Note);

    private static List<WebPMetaCase> WebPMetaCases()
    {
        var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPFixturePrefix + "meta2_manifest.tsv"));
        var rows = new List<WebPMetaCase>();
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0) continue;
            var c = line.Split('\t');
            rows.Add(new WebPMetaCase("meta2_" + c[0], int.Parse(c[1]), int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]),
                c[5] == "1", c[6] == "1", c[7] == "1", c[8] == "1", c[9]));
        }
        if (rows.Count == 0) throw new InvalidOperationException("The metadata manifest is empty.");
        return rows;
    }

    private static byte[]? WebPMetaTruth(string name, string ext)
    {
        string resource = WebPFixturePrefix + name + "." + ext;
        return HasEmbedded(resource) ? ReadEmbedded(resource) : null;
    }

    [TestMethod]
    public async Task WebPDecoderGpu_Metadata_EveryCase_MatchesLibwebpDemuxer_ManagedAndArrayView()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        int count = 0;
        try
        {
            using var dec = new WebPDecoderGpu(acc);
            foreach (var c in WebPMetaCases())
            {
                try
                {
                    byte[] file = ReadEmbedded(WebPFixturePrefix + c.Name + ".webp");
                    var truthIcc = WebPMetaTruth(c.Name, "icc");
                    var truthExif = WebPMetaTruth(c.Name, "exif");
                    var truthXmp = WebPMetaTruth(c.Name, "xmp");
                    if ((truthIcc != null) != c.HasIcc || (truthExif != null) != c.HasExif || (truthXmp != null) != c.HasXmp)
                        throw new Exception("manifest and truth sidecars disagree");

                    void CheckBlobs(WebPMetadata md, string via)
                    {
                        foreach (var (what, truth, got) in new[] { ("ICC", truthIcc, md.IccProfile), ("EXIF", truthExif, md.Exif), ("XMP", truthXmp, md.Xmp) })
                        {
                            if (truth is null && got is not null) failures.Add($"{c.Name} [{via}]: {what} reported ({got.Length} bytes), libwebp reports none");
                            else if (truth is not null && got is null) failures.Add($"{c.Name} [{via}]: {what} missing, libwebp reports {truth.Length} bytes");
                            else if (truth is not null && !truth.AsSpan().SequenceEqual(got))
                                failures.Add($"{c.Name} [{via}]: {what} payload differs from libwebp's ({got!.Length} vs {truth.Length} bytes)");
                        }
                    }

                    CheckBlobs(await dec.ReadMetadataAsync(file), "managed");
                    using (var dFile = acc.Allocate1D<byte>(file.Length + 5))
                    {
                        DeviceTransfer.Upload(dFile.View, file);
                        CheckBlobs(await dec.ReadMetadataAsync(dFile.View, file.Length), "ArrayView1D");
                    }

                    var fi = await dec.ReadFileInfoAsync(file);
                    if (fi.HasIccProfile != c.HasIcc || fi.HasExif != c.HasExif || fi.HasXmp != c.HasXmp)
                        failures.Add($"{c.Name}: ReadFileInfo says ICC {fi.HasIccProfile} EXIF {fi.HasExif} XMP {fi.HasXmp}, libwebp: {c.HasIcc} {c.HasExif} {c.HasXmp}");
                    if (fi.IsLossless != c.Lossless) failures.Add($"{c.Name}: IsLossless {fi.IsLossless}");
                    if (fi.Info.Orientation != c.Orientation)
                        failures.Add($"{c.Name}: orientation {fi.Info.Orientation}, expected {c.Orientation}{(c.Note != "-" ? " (" + c.Note + ")" : "")}");
                    if (fi.Info.Width != c.Width || fi.Info.Height != c.Height || fi.Info.FrameCount != c.Frames)
                        failures.Add($"{c.Name}: {fi.Info.Width}x{fi.Info.Height} x{fi.Info.FrameCount} frames");

                    // the metadata chunks must not disturb the picture
                    if (c.Lossless && c.Frames == 1)
                    {
                        var img = await dec.DecodeAsync(file);
                        for (int i = 0; i < img.Pixels.Length; i += 4)
                            if (img.Pixels[i] != 200 || img.Pixels[i + 1] != 100 || img.Pixels[i + 2] != 50 || img.Pixels[i + 3] != 255)
                            {
                                failures.Add($"{c.Name}: pixel {i / 4} decoded to {img.Pixels[i]},{img.Pixels[i + 1]},{img.Pixels[i + 2]},{img.Pixels[i + 3]}");
                                break;
                            }
                    }
                    else if (c.Lossless)
                    {
                        var frame0 = await dec.DecodeAsync(file);
                        if (frame0.Info.Width != c.Width) failures.Add($"{c.Name}: frame 0 width {frame0.Info.Width}");
                    }
                    count++;
                }
                catch (Exception ex) { failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}"); }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        True(count > 35, $"only {count} metadata cases were exercised");
        True(failures.Count == 0, $"{failures.Count} metadata mismatches:\n" + string.Join("\n", failures));
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Metadata_EverySubset_RoundTripsAndOpensInLibwebp()
    {
        byte[] exifBare = new byte[] { (byte)'I', (byte)'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0 };
        byte[] exifPrefixed = System.Text.Encoding.ASCII.GetBytes("Exif\0\0").Concat(exifBare).ToArray();
        byte[] xmpEven = System.Text.Encoding.UTF8.GetBytes("<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF/></x:xmpmeta>");
        byte[] xmpOdd = xmpEven.Concat(new byte[] { (byte)' ' }).ToArray();
        True((xmpEven.Length & 1) == 0 && (xmpOdd.Length & 1) == 1, "test data: one XMP payload even, one odd");
        byte[] iccOne = { 0x42 };
        byte[] iccBig = new byte[200_001];
        new Random(5).NextBytes(iccBig);

        var variants = new (string Name, byte[]? Icc, byte[]? Exif, byte[]? Xmp, byte[]? ExifStored)[]
        {
            ("icc", iccOne, null, null, null), ("icc_big", iccBig, null, null, null),
            ("exif", null, exifBare, null, exifBare), ("exif_prefixed", null, exifPrefixed, null, exifBare),
            ("xmp_even", null, null, xmpEven, null), ("xmp_odd", null, null, xmpOdd, null),
            ("icc_exif", iccOne, exifBare, null, exifBare), ("icc_xmp", iccBig, null, xmpOdd, null),
            ("exif_xmp", null, exifPrefixed, xmpEven, exifBare), ("all", iccBig, exifPrefixed, xmpOdd, exifBare),
        };
        string dir = WebPEncoderOutputDir("metadata_subsets");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var src = WebPSources().Single(r => r.Name == "enc_photo_80x60");
            byte[] px = WebPSource("enc_photo_80x60");
            foreach (var (name, icc, exif, xmp, exifStored) in variants)
            {
                try
                {
                    var options = new WebPEncodeOptions { Quality = 50, IccProfile = icc, Exif = exif, Xmp = xmp };
                    byte[] file = await enc.EncodeAsync(px, src.Width, src.Height, options);
                    var md = await dec.ReadMetadataAsync(file);
                    if (icc != null) { NotNull(md.IccProfile, name + ": ICC missing from the file we wrote"); EqualBytes(icc, md.IccProfile!, name + " ICC"); } else True(md.IccProfile is null, name + ": ICC must be absent");
                    if (exif != null) { NotNull(md.Exif, name + ": EXIF missing from the file we wrote"); EqualBytes(exifStored!, md.Exif!, name + " EXIF"); } else True(md.Exif is null, name + ": EXIF must be absent");
                    if (xmp != null) { NotNull(md.Xmp, name + ": XMP missing from the file we wrote"); EqualBytes(xmp, md.Xmp!, name + " XMP"); } else True(md.Xmp is null, name + ": XMP must be absent");
                    var fi = await dec.ReadFileInfoAsync(file);
                    if (fi.HasIccProfile != (icc != null) || fi.HasExif != (exif != null) || fi.HasXmp != (xmp != null))
                        failures.Add($"{name}: file info flags ICC {fi.HasIccProfile} EXIF {fi.HasExif} XMP {fi.HasXmp}");
                    var decoded = await dec.DecodeAsync(file);
                    var diff = ComparePixels(px, decoded.Pixels, 4, src.Width);
                    if (diff != null) failures.Add($"{name} [pixels]: {diff}");

                    string stem = Path.Combine(dir, name);
                    await File.WriteAllBytesAsync(stem + ".webp", file);
                    await File.WriteAllBytesAsync(stem + ".rgba8", px);
                    await File.WriteAllTextAsync(stem + ".size", $"{src.Width} {src.Height}");
                    if (icc != null) await File.WriteAllBytesAsync(stem + ".icc", icc);
                    if (exif != null) await File.WriteAllBytesAsync(stem + ".exif", exifStored!);
                    if (xmp != null) await File.WriteAllBytesAsync(stem + ".xmp", xmp);
                }
                catch (Exception ex) { failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}"); }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));
        // libwebp's demuxer must report exactly these payloads, with VP8X flags that match, and no stray chunk
        await RunPillowVerifyAsync(dir);
    }
}
