// The animation entry points that hand the file back at its exact length (EncodeAnimationToDeviceAsync, and in a browser
// EncodeAnimationToUint8ArrayAsync), and the published upper bounds (GetMaxAnimationEncodedSize for both encoders,
// GetMaxEncodedSize for the lossy one), asked for by SpawnGen (Lore, 2026-11-19: frames stay on the device, the file lands in
// a JS Uint8Array, and an 8 GB card cannot hold a worst-case buffer for 121 frames). The device result must be the managed
// overload's file byte for byte, for the lossless and the lossy encoder; every real file, worst-case content included
// (noise at effort 100), must fit its bound.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static List<byte[]> AnimDeviceOutputFrames(int w, int h)
    {
        var frames = new List<byte[]>();
        for (int f = 0; f < 3; f++)
        {
            var rgba = WebPTokenizeStressPicture(f == 1 ? 1 : 3, w, h);   // noise between two transparent gradients
            for (int i = 0; i < rgba.Length; i += 4 * 37) rgba[i] ^= (byte)(f * 17);
            frames.Add(rgba);
        }
        return frames;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPAnimation_ToDevice_SameFileAsManaged_AndWithinTheBounds()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            const int w = 160, h = 96;
            var frames = AnimDeviceOutputFrames(w, h);
            var managed = frames.Select(f => (ReadOnlyMemory<byte>)f).ToList();
            var device = frames.Select(f => DeviceVerify.Upload(acc, f)).ToList();
            var views = device.Select(b => b.View).ToList();
            var durations = new[] { 41, 41, 41 };
            var anim = new WebPAnimEncoderOptions { Exif = new byte[] { 0x4D, 0x4D, 0, 0x2A, 0, 0, 0, 8, 0, 0 } };
            var failures = new List<string>();
            try
            {
                using (var lossless = new WebPLosslessEncoderGpu(acc))
                {
                    var o = new WebPEncodeOptions { Quality = 80, Exact = false };
                    byte[] want = await lossless.EncodeAnimationAsync(managed, w, h, durations, o, anim);
                    var file = await lossless.EncodeAnimationToDeviceAsync(views, w, h, durations, o, anim);
                    if (file.Length != want.Length) failures.Add($"lossless: device {file.Length} B, managed {want.Length} B");
                    else if (await DeviceVerify.CompareAsync(acc, file, want) is { } d) failures.Add($"lossless: {d}");
                    if (OperatingSystem.IsBrowser())
                    {
                        using var js = await lossless.EncodeAnimationToUint8ArrayAsync(views, w, h, durations, o, anim);
                        if (js.Length != want.Length) failures.Add($"lossless Uint8Array: {js.Length} B, managed {want.Length} B");
                        else if (!js.ReadBytes().AsSpan().SequenceEqual(want)) failures.Add("lossless Uint8Array: bytes differ");
                        // the array is the caller's: the encoder's next encode (other frames) must not change it
                        await lossless.EncodeAnimationToDeviceAsync(views.AsEnumerable().Reverse().ToList(), w, h, durations, o, anim);
                        await acc.SynchronizeAsync();
                        if (!js.ReadBytes().AsSpan().SequenceEqual(want)) failures.Add("lossless Uint8Array: changed by the next encode (aliases the encoder's buffer)");
                    }
                    long bound = WebPLosslessEncoderGpu.GetMaxAnimationEncodedSize(3, w, h, o, anim);
                    if (want.Length > bound) failures.Add($"lossless: {want.Length} B over the bound {bound}");
                }
                using (var lossy = new WebPLossyEncoderGpu(acc))
                {
                    foreach (int q in new[] { 80, 100 })
                    {
                        var o = new WebPLossyEncodeOptions { Quality = q };
                        byte[] want = await lossy.EncodeAnimationAsync(managed, w, h, durations, o, anim);
                        var file = await lossy.EncodeAnimationToDeviceAsync(views, w, h, durations, o, anim);
                        if (file.Length != want.Length) failures.Add($"lossy q{q}: device {file.Length} B, managed {want.Length} B");
                        else if (await DeviceVerify.CompareAsync(acc, file, want) is { } d) failures.Add($"lossy q{q}: {d}");
                        if (OperatingSystem.IsBrowser())
                        {
                            using var js = await lossy.EncodeAnimationToUint8ArrayAsync(views, w, h, durations, o, anim);
                            if (js.Length != want.Length) failures.Add($"lossy q{q} Uint8Array: {js.Length} B, managed {want.Length} B");
                            else if (!js.ReadBytes().AsSpan().SequenceEqual(want)) failures.Add($"lossy q{q} Uint8Array: bytes differ");
                            await lossy.EncodeAnimationToDeviceAsync(views.AsEnumerable().Reverse().ToList(), w, h, durations, o, anim);
                            await acc.SynchronizeAsync();
                            if (!js.ReadBytes().AsSpan().SequenceEqual(want)) failures.Add($"lossy q{q} Uint8Array: changed by the next encode (aliases the encoder's buffer)");
                        }
                        long bound = WebPLossyEncoderGpu.GetMaxAnimationEncodedSize(3, w, h, o, anim);
                        if (want.Length > bound) failures.Add($"lossy q{q}: {want.Length} B over the bound {bound}");
                        // a still of the noise frame, the worst content
                        byte[] still = await lossy.EncodeAsync(frames[1], w, h, o);
                        long stillBound = WebPLossyEncoderGpu.GetMaxEncodedSize(w, h, o);
                        if (still.Length > stillBound) failures.Add($"lossy still q{q}: {still.Length} B over the bound {stillBound}");
                    }
                }
            }
            finally
            {
                foreach (var b in device) b.Dispose();
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
