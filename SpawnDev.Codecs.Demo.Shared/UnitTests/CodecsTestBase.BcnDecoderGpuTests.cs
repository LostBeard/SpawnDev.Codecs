// Tests for BcnDecoderGpu (Images/Texture): BC1, BC3, BC4, BC5 and BC7 blocks to RGBA8.
//
// Fixtures (TestData/Bcn, scripts/texture/make_bcn_fixtures.py): blocks made by the reference encoders (etcpak; bc7enc
// for BC7, also restricted to each mode in turn) over a real picture (a Big Buck Bunny frame) and a synthetic RGBA
// picture, plus random blocks (every bit pattern is a block: all BC7 modes, partitions, rotations, index selections and
// the reserved mode) and 37x21 pictures (partial edge blocks). The truth is texture2ddecoder's output, checked by the
// script against Pillow's decoder (BC3 uses Pillow's spec 4-colour reading; BC7 reserved-mode alpha is the spec's 0).
// Neither side is our code.
//
// What can fail: 5:6:5 expansion, the BC1 3- / 4-colour modes and their rounding, the BC3 forced 4-colour, BC4's 6- and
// 8-value modes and index bits straddling the two words, BC7 mode decoding, partitions and anchors, endpoint channel
// order, p-bits (per endpoint / shared), unquantization, index bit positions, mode 4 index selection, rotation, the
// reserved mode, partial blocks at the right / bottom, and writes past the picture (outputs are poisoned and longer).

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
    private const string BcnPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Bcn.";

    private sealed record BcnCase(string Name, BcnFormat Format, string Ext, int Width, int Height);

    private static List<BcnCase> BcnCases()
    {
        var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(BcnPrefix + "manifest.tsv"));
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
                "bc7" => BcnFormat.Bc7,
                _ => throw new Exception($"unknown format {c[1]}"),
            };
            cases.Add(new BcnCase(c[0], format, c[1], int.Parse(c[2]), int.Parse(c[3])));
        }
        return cases;
    }

    private static string? FirstRgbaDiff(byte[] expected, byte[] got, int w)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != got[i])
            {
                int px = i / 4;
                return $"first mismatch at x={px % w} y={px / w} channel {"RGBA"[i % 4]}: got {got[i]}, expected {expected[i]}";
            }
        }
        return null;
    }

    [TestMethod(Timeout = 600000)]
    public async Task BcnDecoderGpu_AllFixtures_MatchReferenceDecoders()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new BcnDecoderGpu(acc);
            var failures = new List<string>();
            int checkedCount = 0;
            foreach (var c in BcnCases())
            {
                byte[] blocks = ReadEmbedded(BcnPrefix + c.Name + "." + c.Ext);
                byte[] expected = Gunzip(ReadEmbedded(BcnPrefix + c.Name + "." + c.Ext + ".rgba.gz"));
                long n = 4L * c.Width * c.Height;
                using var dIn = acc.Allocate1D<byte>(blocks.Length + 13);
                DeviceTransfer.Upload(dIn.View, blocks);
                using var dOut = acc.Allocate1D<byte>(n + 11);
                DeviceTransfer.Upload(dOut.View, Poison((int)dOut.Length));
                dec.Decode(dIn.View, c.Format, c.Width, c.Height, dOut.View);
                await acc.SynchronizeAsync();
                byte[] all = await DeviceTransfer.DownloadAsync(dOut.View, dOut.Length);
                byte[] got = all.AsSpan(0, (int)n).ToArray();
                string? diff = FirstRgbaDiff(expected, got, c.Width);
                if (diff != null) failures.Add($"{c.Name}.{c.Ext} {c.Width}x{c.Height}: {diff}");
                byte[] poison = Poison((int)dOut.Length);
                for (long i = n; i < dOut.Length; i++)
                    if (all[i] != poison[i]) { failures.Add($"{c.Name}.{c.Ext}: wrote past the picture at byte {i}"); break; }
                checkedCount++;
            }
            True(checkedCount >= 29, $"only {checkedCount} fixtures");
            True(failures.Count == 0, $"{failures.Count} of {checkedCount} fixtures differ:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task BcnDecoderGpu_ManagedOverload_AndBadArguments()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new BcnDecoderGpu(acc);
            var c = BcnCases().First(c => c.Name == "bbb" && c.Format == BcnFormat.Bc7);
            byte[] blocks = ReadEmbedded(BcnPrefix + "bbb.bc7");
            byte[] expected = Gunzip(ReadEmbedded(BcnPrefix + "bbb.bc7.rgba.gz"));
            byte[] got = await dec.DecodeAsync(blocks, BcnFormat.Bc7, c.Width, c.Height);
            EqualBytes(expected, got, "managed overload, bbb.bc7");

            Equal(8L * 10 * 6, BcnDecoderGpu.BlocksSize(BcnFormat.Bc1, 37, 21), "BC1 blocks of 37x21 (10 x 6 blocks of 8 bytes)");
            Equal(16L * 10 * 6, BcnDecoderGpu.BlocksSize(BcnFormat.Bc7, 37, 21), "BC7 blocks of 37x21");
            await ThrowsAsync<ArgumentException>(() => dec.DecodeAsync(blocks.AsMemory(0, 16), BcnFormat.Bc7, c.Width, c.Height), "short blocks");
            await ThrowsAsync<ArgumentOutOfRangeException>(() => dec.DecodeAsync(blocks, BcnFormat.Bc7, 0, 4), "zero width");
            await ThrowsAsync<ArgumentOutOfRangeException>(() => dec.DecodeAsync(blocks, (BcnFormat)2, 4, 4), "BC2 is not supported");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task BcnDecoderGpu_TypedArrayOverload()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new BcnDecoderGpu(acc);
            await RunBcnTypedArray(dec);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunBcnTypedArray(BcnDecoderGpu dec)
    {
        var c = BcnCases().First(c => c.Name == "alpha" && c.Format == BcnFormat.Bc3);
        byte[] blocks = ReadEmbedded(BcnPrefix + "alpha.bc3");
        byte[] expected = Gunzip(ReadEmbedded(BcnPrefix + "alpha.bc3.rgba.gz"));
        using var js = new Uint8Array(blocks.Length);
        js.Set(blocks);
        using var outp = await dec.DecodeAsync(js, BcnFormat.Bc3, c.Width, c.Height);
        EqualBytes(expected, outp.ReadBytes(), "TypedArray overload, alpha.bc3");
    }
}
