// Tests for BcnEncoderGpu (Images/Texture): RGBA8 to BC1, BC3, BC4 and BC5 blocks.
//
// Fixtures (TestData/BcnEnc, scripts/texture/make_bcn_encoder_fixtures.py): pictures and the blocks etcpak 0.9.15
// writes for them (its portable paths, the encoder ported): a Big Buck Bunny frame (320 and 316 wide), a synthetic RGBA
// picture (BC1 too: alpha takes no part), noise, and flat / nearly flat blocks (solid detection, 5:6:5-equal neighbours,
// one-step ramps, a single odd pixel). The truth is etcpak's output; a partial-block test reads our blocks back through
// our decoder (bit-exact with the reference decoders) for a quality floor.
//
// What can fail: the solid test, min / max and insets, the summed-range reciprocal, the position rounding and its index
// mapping, 5:6:5 packing and endpoint order, the one-channel reciprocal and its 3-bit packing across the two words, the
// BC3 / BC5 halves, the block order, and writes past the blocks (outputs are poisoned and longer).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Texture;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string BcnEncPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.BcnEnc.";

    private static List<BcnCase> BcnEncCases()
    {
        var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(BcnEncPrefix + "manifest.tsv"));
        var cases = new List<BcnCase>();
        foreach (var line in text.Split('\n'))
        {
            var row = line.TrimEnd('\r');
            if (row.Length == 0) continue;
            var c = row.Split('\t');
            var format = c[1] switch
            {
                "bc1" => BcnFormat.Bc1,
                "bc3" => BcnFormat.Bc3,
                "bc4" => BcnFormat.Bc4,
                "bc5" => BcnFormat.Bc5,
                _ => throw new Exception($"unknown format {c[1]}"),
            };
            cases.Add(new BcnCase(c[0], format, c[1], int.Parse(c[2]), int.Parse(c[3])));
        }
        return cases;
    }

    [TestMethod(Timeout = 600000)]
    public async Task BcnEncoderGpu_AllFixtures_ByteIdenticalWithEtcpak()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BcnEncoderGpu(acc);
            var failures = new List<string>();
            int checkedCount = 0;
            foreach (var c in BcnEncCases())
            {
                byte[] rgba = Gunzip(ReadEmbedded(BcnEncPrefix + c.Name + ".rgba.gz"));
                byte[] expected = ReadEmbedded(BcnEncPrefix + c.Name + "." + c.Ext);
                using var dIn = acc.Allocate1D<byte>(rgba.Length + 9);
                DeviceTransfer.Upload(dIn.View, rgba);
                using var dOut = acc.Allocate1D<byte>(expected.Length + 13);
                DeviceTransfer.Upload(dOut.View, Poison((int)dOut.Length));
                enc.Encode(dIn.View, c.Width, c.Height, c.Format, dOut.View);
                await acc.SynchronizeAsync();
                byte[] all = await DeviceTransfer.DownloadAsync(dOut.View, dOut.Length);
                int bb = BcnDecoderGpu.BlockBytes(c.Format);
                for (int i = 0; i < expected.Length; i++)
                {
                    if (all[i] != expected[i])
                    {
                        int blk = i / bb, bwBlocks = (c.Width + 3) / 4;
                        failures.Add($"{c.Name}.{c.Ext}: first mismatch in block {blk} (x={blk % bwBlocks * 4}, y={blk / bwBlocks * 4}) byte {i % bb}: got {all[i]:X2}, expected {expected[i]:X2}");
                        break;
                    }
                }
                byte[] poison = Poison((int)dOut.Length);
                for (int i = expected.Length; i < all.Length; i++)
                    if (all[i] != poison[i]) { failures.Add($"{c.Name}.{c.Ext}: wrote past the blocks at byte {i}"); break; }
                checkedCount++;
            }
            True(checkedCount >= 19, $"only {checkedCount} fixtures");
            True(failures.Count == 0, $"{failures.Count} of {checkedCount} fixtures differ:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>A picture whose size is not a multiple of 4 (edge pixels repeated) round-trips through our decoder close to the source.</summary>
    [TestMethod(Timeout = 600000)]
    public async Task BcnEncoderGpu_PartialBlocks_ManagedOverload_AndBadArguments()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BcnEncoderGpu(acc);
            using var dec = new BcnDecoderGpu(acc);
            const int w = 37, h = 21;
            var rgba = new byte[w * h * 4];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int i = (y * w + x) * 4;
                    rgba[i] = (byte)(x * 6);
                    rgba[i + 1] = (byte)(y * 11);
                    rgba[i + 2] = (byte)((x + y) * 3);
                    rgba[i + 3] = 255;
                }
            foreach (var f in new[] { BcnFormat.Bc1, BcnFormat.Bc3, BcnFormat.Bc4, BcnFormat.Bc5 })
            {
                byte[] blocks = await enc.EncodeAsync(rgba, w, h, f);
                Equal(BcnDecoderGpu.BlocksSize(f, w, h), (long)blocks.Length, $"{f} block bytes for {w}x{h}");
                byte[] back = await dec.DecodeAsync(blocks, f, w, h);
                int channels = f == BcnFormat.Bc4 ? 1 : (f == BcnFormat.Bc5 ? 2 : 3);
                int worst = 0;
                for (int i = 0; i < w * h; i++)
                    for (int ch = 0; ch < channels; ch++)
                        worst = Math.Max(worst, Math.Abs(back[i * 4 + ch] - rgba[i * 4 + ch]));
                True(worst <= 24, $"{f}: worst channel error {worst} on a smooth gradient (edge blocks repeat the edge pixels)");
            }
            await ThrowsAsync<ArgumentException>(() => enc.EncodeAsync(rgba.AsMemory(0, 16), w, h, BcnFormat.Bc1), "short rgba");
            await ThrowsAsync<ArgumentOutOfRangeException>(() => enc.EncodeAsync(rgba, w, h, BcnFormat.Bc7), "BC7 encode is not written yet");
            await ThrowsAsync<ArgumentOutOfRangeException>(() => enc.EncodeAsync(rgba, 0, h, BcnFormat.Bc1), "zero width");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task BcnEncoderGpu_TypedArrayOverload()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new BcnEncoderGpu(acc);
            await RunBcnEncoderTypedArray(enc);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunBcnEncoderTypedArray(BcnEncoderGpu enc)
    {
        var c = BcnEncCases().First(c => c.Name == "alpha" && c.Format == BcnFormat.Bc3);
        byte[] rgba = Gunzip(ReadEmbedded(BcnEncPrefix + "alpha.rgba.gz"));
        byte[] expected = ReadEmbedded(BcnEncPrefix + "alpha.bc3");
        using var js = new Uint8Array(rgba.Length);
        js.Set(rgba);
        using var outp = await enc.EncodeAsync(js, c.Width, c.Height, BcnFormat.Bc3);
        EqualBytes(expected, outp.ReadBytes(), "TypedArray overload, alpha.bc3");
    }
}
