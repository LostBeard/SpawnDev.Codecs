// Browser-lane tests for the TypedArray overloads of WebPDecoderGpu / WebPLosslessEncoderGpu: the JS bytes go to the
// accelerator with CopyFromJS and the results come back as JS typed arrays; each must equal what the managed overload gives.
// Desktop runs skip them (UnsupportedTestException): a desktop accelerator has no JS heap, and a silent managed fallback would
// be the exact copy these overloads exist to avoid.

using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task WebPDecoderGpu_TypedArrayOverloads_MatchManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new WebPDecoderGpu(acc);
            await RunWebPDecoderTypedArrayAsync(dec);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunWebPDecoderTypedArrayAsync(WebPDecoderGpu dec)
    {
        // still image
        byte[] still = WebPFile("mosaic_96x72");
        using (var jStill = new Uint8Array(still.Length))
        {
            jStill.Set(still);
            var managed = await dec.DecodeAsync(still);
            using var js = await dec.DecodeAsync(jStill);
            Equal(managed.Info.Width, js.Info.Width, "still width");
            EqualBytes(managed.Pixels, js.Pixels.ReadBytes(), "still pixels (TypedArray vs managed)");   // the TEST's terminal compare
            Equal(managed.Info.Width, (await dec.ReadInfoAsync(jStill)).Width, "ReadInfo width");
        }
        // animation frame
        byte[] anim = WebPFile("anim_blend_dispose_64x48");
        using (var jAnim = new Uint8Array(anim.Length))
        {
            jAnim.Set(anim);
            var managed = await dec.DecodeFrameAsync(anim, 3);
            using var js = await dec.DecodeFrameAsync(jAnim, 3);
            EqualBytes(managed.Pixels, js.Pixels.ReadBytes(), "animation frame 3 (TypedArray vs managed)");
            Equal((await dec.ReadFramesAsync(anim)).Length, (await dec.ReadFramesAsync(jAnim)).Length, "frame count");
        }
        // metadata
        byte[] meta = ReadEmbedded(WebPFixturePrefix + "meta2_canonical_all.webp");
        using (var jMeta = new Uint8Array(meta.Length))
        {
            jMeta.Set(meta);
            var managed = await dec.ReadMetadataAsync(meta);
            var js = await dec.ReadMetadataAsync(jMeta);
            EqualBytes(managed.IccProfile!, js.IccProfile!, "ICC");
            EqualBytes(managed.Exif!, js.Exif!, "EXIF");
            EqualBytes(managed.Xmp!, js.Xmp!, "XMP");
            var fi = await dec.ReadFileInfoAsync(jMeta);
            True(fi.HasIccProfile && fi.HasExif && fi.HasXmp && fi.IsLossless, "file info flags");
        }
        // alpha plane
        byte[] alph = ReadEmbedded(WebPFixturePrefix + "alph_smooth_64x48_f3_vp8l.webp");
        using (var jAlph = new Uint8Array(alph.Length))
        {
            jAlph.Set(alph);
            using var js = await dec.DecodeAlphaAsync(jAlph);
            Equal(64, js.Width, "alpha width");
            Equal(48, js.Height, "alpha height");
            EqualBytes(WebPAlphaTruth("alph_smooth_64x48_f3_vp8l"), js.Alpha.ReadBytes(), "alpha plane (TypedArray vs libwebp truth)");
        }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_TypedArrayOverloads_MatchManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            await RunWebPEncoderTypedArrayAsync(enc);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunWebPEncoderTypedArrayAsync(WebPLosslessEncoderGpu enc)
    {
        // still image
        var row = WebPSources().Single(r => r.Name == "enc_photo_80x60");
        byte[] px = WebPSource("enc_photo_80x60");
        using (var jPx = new Uint8Array(px.Length))
        {
            jPx.Set(px);
            var managed = await enc.EncodeAsync(px, row.Width, row.Height);
            using var js = await enc.EncodeAsync(jPx, row.Width, row.Height);
            EqualBytes(managed, js.ReadBytes(), "still file (TypedArray vs managed)");
        }
        // alpha plane
        byte[] plane = WebPAlphaTruth("alph_mask_61x45_f0_raw");
        using (var jPlane = new Uint8Array(plane.Length))
        {
            jPlane.Set(plane);
            var managed = await enc.EncodeAlphaAsync(plane, 61, 45);
            using var js = await enc.EncodeAlphaAsync(jPlane, 61, 45);
            EqualBytes(managed, js.ReadBytes(), "ALPH payload (TypedArray vs managed)");
        }
        // animation
        var src = WebPAnimEncodeSources().Single(s => s.Name == "rects_64x48");
        var managedAnim = await enc.EncodeAnimationAsync(src.FramePixels.Select(p => (ReadOnlyMemory<byte>)p).ToArray(), src.Frames, src.CanvasW, src.CanvasH);
        var jFrames = src.FramePixels.Select(p => { var a = new Uint8Array(p.Length); a.Set(p); return (TypedArray)a; }).ToList();
        try
        {
            using var js = await enc.EncodeAnimationAsync(jFrames, src.Frames, src.CanvasW, src.CanvasH);
            EqualBytes(managedAnim, js.ReadBytes(), "animation file (TypedArray vs managed)");
        }
        finally { foreach (var f in jFrames) f.Dispose(); }
    }
}
