// Tests for Lz4FrameDecompressorGpu (Compression/Lz4): frames written by liblz4 1.9.4 in every configuration a decoder
// meets, decoded on the accelerator and compared with their content on the accelerator; damaged frames refused with
// the specific reason; and every frame of the COMPRESSOR's fixtures (liblz4 bytes) decoded back to its input.
//
// Fixtures (TestData/Lz4Dec, scripts/compression/make_lz4_decode_fixtures.py): 64 KB / 256 KB / 1 MB / 4 MB blocks,
// linked and independent; HC levels 9 and 16 (long offsets, optimal parse); block + content checksums; no content
// size; raw (incompressible) blocks; 4 MB zero runs (one long match per block, resumed across launches); an empty frame;
// a legacy frame (two 8 MB blocks); skippable frames between concatenated frames; and damaged frames (header checksum,
// magic, truncation, block checksum, content checksum, a match offset before the block, a dictionary frame).
//
// What can fail: the walk (every header field and frame kind), the size scan, the layout across frames, the
// independent and linked decode with literals / matches resumed mid-run, overlapping matches, the match-target check,
// the checksums and the content size, and writes past the output (poisoned).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression.Lz4;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Lz4DecFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Lz4Dec.";

    private sealed record Lz4DecCase(string Name, string Kind, int Length, int Seed, string Expected, string Description);

    private static List<Lz4DecCase> Lz4DecCases()
    {
        var list = new List<Lz4DecCase>();
        foreach (var raw in ReadEmbeddedText(Lz4DecFixturePrefix + "manifest.tsv").Split('\n'))
        {
            var line = raw.Trim('\r', ' ');
            if (line.Length == 0 || line.StartsWith('#')) continue;
            var f = line.Split('\t');
            list.Add(new Lz4DecCase(f[0], f[1], int.Parse(f[2]), int.Parse(f[3]), f[4], f.Length > 5 ? f[5] : ""));
        }
        return list;
    }

    private static byte[] Lz4DecContent(Lz4DecCase c) => c.Kind switch
    {
        "text" => Lz4GenText(c.Length, c.Seed),
        "noise" => Lz4GenNoise(c.Length, c.Seed),
        "struct" => Lz4GenStruct(c.Length, c.Seed),
        "text2" => Lz4GenText(c.Length / 2, c.Seed).Concat(Lz4GenText(c.Length / 2, c.Seed + 1)).ToArray(),
        _ => throw new InvalidOperationException($"unknown kind {c.Kind}"),
    };

    /// <summary>Decodes a frame through the ArrayView1D overload into a poisoned buffer and compares it with the content on the accelerator.</summary>
    private static async Task<string?> DecodeLz4AndCompareAsync(Accelerator acc, Lz4FrameDecompressorGpu dec, string name, byte[] frame, byte[] content)
    {
        using var dIn = DeviceVerify.Upload(acc, frame, 11);
        using var dOut = DeviceVerify.Poisoned(acc, content.Length + 64);
        int n;
        try { n = await dec.DecompressAsync(dIn.View, frame.Length, dOut.View); }
        catch (Exception ex) { return $"{name}: threw {ex.GetType().Name}: {ex.Message}"; }
        if (n != content.Length) return $"{name}: decoded {n} bytes, content is {content.Length}";
        var expected = new byte[content.Length + 64];
        content.CopyTo(expected, 0);
        System.Array.Fill(expected, (byte)0xAB, content.Length, 64);
        string? diff = await DeviceVerify.CompareAsync(acc, dOut.View, expected);
        return diff == null ? null : $"{name}: {diff}";
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Lz4FrameDecompressorGpu_Liblz4Frames_AllConfigurations_DecodeToTheContent()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new Lz4FrameDecompressorGpu(acc);
            var cases = Lz4DecCases().Where(c => c.Expected == "ok").ToList();
            True(cases.Count >= 16, $"only {cases.Count} good frames embedded");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                string? r = await DecodeLz4AndCompareAsync(acc, dec, c.Name, ReadEmbedded(Lz4DecFixturePrefix + c.Name + ".lz4"), Lz4DecContent(c));
                if (r != null) failures.Add($"{r} ({c.Description})");
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} liblz4 frames decoded wrong:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Lz4FrameDecompressorGpu_DamagedFrames_RefusedWithTheReason()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new Lz4FrameDecompressorGpu(acc);
            var cases = Lz4DecCases().Where(c => c.Expected != "ok").ToList();
            True(cases.Count >= 7, $"only {cases.Count} damaged frames embedded");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                byte[] frame = ReadEmbedded(Lz4DecFixturePrefix + c.Name + ".lz4");
                using var dIn = DeviceVerify.Upload(acc, frame);
                using var dOut = DeviceVerify.Poisoned(acc, c.Length + 64);
                try
                {
                    int n = await dec.DecompressAsync(dIn.View, frame.Length, dOut.View);
                    failures.Add($"{c.Name}: decoded {n} bytes, expected {c.Expected}");
                }
                catch (Lz4FrameDecompressorGpu.Lz4FormatException ex)
                {
                    if (ex.Reason.ToString() != c.Expected) failures.Add($"{c.Name}: refused with {ex.Reason}, expected {c.Expected}");
                }
                catch (Exception ex) { failures.Add($"{c.Name}: threw {ex.GetType().Name}: {ex.Message}"); }
            }
            True(failures.Count == 0, string.Join("\n", failures));

            // The decoder still works after refusing (no state left behind).
            var good = Lz4DecCases().First(c => c.Name == "text_500000_allchecks");
            string? after = await DecodeLz4AndCompareAsync(acc, dec, good.Name, ReadEmbedded(Lz4DecFixturePrefix + good.Name + ".lz4"), Lz4DecContent(good));
            True(after == null, $"after the damaged frames: {after}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Every liblz4 frame of the compressor's fixtures (independent 64 KB blocks, all options, real files) decodes to its input.</summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Lz4FrameDecompressorGpu_CompressorFixtures_DecodeToTheirInputs()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new Lz4FrameDecompressorGpu(acc);
            var failures = new List<string>();
            var cases = Lz4Cases().Where(c => c.Stored && c.Length > 0).ToList();
            foreach (var c in cases)
            {
                string? r = await DecodeLz4AndCompareAsync(acc, dec, c.Name, ReadEmbedded(Lz4FixturePrefix + c.Name + ".lz4"), Lz4Input(c));
                if (r != null) failures.Add(r);
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} frames decoded wrong:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>40 MB round trip on the accelerator: our compressor, then our decompressor, compared with the input on the device.</summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Lz4Frame_RoundTrip40MB_OnTheAccelerator()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var comp = new Lz4FrameCompressorGpu(acc);
            using var dec = new Lz4FrameDecompressorGpu(acc);
            byte[] input = Lz4GenText(40_000_000, 31);
            using var dIn = DeviceVerify.Upload(acc, input);
            using var dFrame = acc.Allocate1D<byte>(Lz4FrameCompressorGpu.GetMaxCompressedSize(input.Length));
            int n = await comp.CompressAsync(dIn.View, input.Length, dFrame.View, new Lz4FrameOptions { ContentChecksum = true, BlockChecksum = true });
            using var dBack = DeviceVerify.Poisoned(acc, input.Length + 64);
            int m = await dec.DecompressAsync(dFrame.View, n, dBack.View);
            Equal(input.Length, m, "decoded length");
            string? diff = await DeviceVerify.CompareAsync(acc, dBack.View, dIn.View, input.Length);
            True(diff == null, $"round trip: {diff}");
            True(await DeviceVerify.AllEqualAsync(acc, dBack.View.SubView(input.Length, 64), 64, 0xAB) == null, "wrote past the output");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task Lz4FrameDecompressorGpu_ManagedOverload_AndOutputTooSmall()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new Lz4FrameDecompressorGpu(acc);
            var c = Lz4DecCases().First(c => c.Name == "concat_2x100000_skippable");
            byte[] frame = ReadEmbedded(Lz4DecFixturePrefix + c.Name + ".lz4");
            EqualBytes(Lz4DecContent(c), await dec.DecompressAsync(frame), "managed overload");
            Equal(0, (await dec.DecompressAsync(ReadEmbedded(Lz4DecFixturePrefix + "empty.lz4"))).Length, "empty frame");
            using var dIn = DeviceVerify.Upload(acc, frame);
            using var small = acc.Allocate1D<byte>(1000);
            try
            {
                await dec.DecompressAsync(dIn.View, frame.Length, small.View);
                throw new Exception("output of 1000 bytes accepted for 200000");
            }
            catch (Lz4FrameDecompressorGpu.Lz4FormatException ex) { Equal(Lz4FrameDecompressorGpu.Lz4Error.OutputTooSmall, ex.Reason, "reason"); }
            Equal(c.Length, await dec.GetDecompressedLengthAsync(dIn.View, frame.Length), "GetDecompressedLengthAsync");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "BrowserOnly,NotOnWebGL")]
    public async Task Lz4FrameDecompressorGpu_TypedArrayOverload()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            using var dec = new Lz4FrameDecompressorGpu(acc);
            await RunLz4DecTypedArray(dec);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunLz4DecTypedArray(Lz4FrameDecompressorGpu dec)
    {
        var c = Lz4DecCases().First(c => c.Name == "text_500000_allchecks");
        byte[] frame = ReadEmbedded(Lz4DecFixturePrefix + c.Name + ".lz4");
        using var js = new Uint8Array(frame.Length);
        js.Set(frame);
        using var outp = await dec.DecompressAsync(js);
        EqualBytes(Lz4DecContent(c), outp.ReadBytes(), "TypedArray overload");
    }
}
