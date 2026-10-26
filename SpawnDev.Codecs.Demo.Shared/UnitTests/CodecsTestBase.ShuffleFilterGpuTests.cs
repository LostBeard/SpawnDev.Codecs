// Tests for ShuffleFilterGpu (Compression): byte shuffle and bitshuffle, both directions, against truth made by the
// REFERENCE libraries (numcodecs / c-blosc Shuffle; the bitshuffle library through its HDF5 filter), never by our code.
//
// Fixtures (TestData/Filters, scripts/compression/make_filter_fixtures.py): byte shuffle for element sizes 1..16 and
// lengths 0..65537 (trailing partial elements included); bitshuffle for element sizes 1, 2, 3, 4, 8, 12 and 1..9000
// elements (partial last blocks, n % 8 leftovers, a 64-element block size); three 16 MiB cases (four launches each,
// thousands of blocks) whose truth is a SHA-256. Inputs are procedural (FilterInput = the script's gen()).
//
// What can fail: the closed-form block / row / bit mapping, the last rounded-down block, the raw tail, the launch
// offsets, writes past the length (outputs are poisoned and longer than needed), and the inverses (each fixture is also
// run backwards and must give the input back).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Compression;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string FilterFixturePrefix = "SpawnDev.Codecs.Demo.Shared.TestData.Filters.";

    /// <summary>make_filter_fixtures.py gen(n, seed): the same bytes.</summary>
    private static byte[] FilterInput(int length, int seed)
    {
        var data = new byte[length];
        for (int k = 0; k < length; k++)
        {
            ulong i = (ulong)k;
            uint x = (uint)(i * 2654435761UL + (ulong)seed * 0x9E37UL);
            data[k] = (byte)(((ulong)(x >> 13) ^ (i >> 9) ^ (ulong)seed) & 0xFF);
        }
        return data;
    }

    private sealed record FilterCase(string Resource, bool Bit, int ElementSize, int Length, int Seed, int Block);

    private static List<FilterCase> FilterCases()
    {
        var cases = new List<FilterCase>();
        foreach (var name in typeof(CodecsTestBase).Assembly.GetManifestResourceNames())
        {
            if (!name.StartsWith(FilterFixturePrefix, StringComparison.Ordinal) || !name.EndsWith(".bin", StringComparison.Ordinal)) continue;
            var parts = name.Substring(FilterFixturePrefix.Length, name.Length - FilterFixturePrefix.Length - 4).Split('_');
            int e = int.Parse(parts[1].Substring(1));
            if (parts[0] == "shuf")
            {
                int bytes = int.Parse(parts[2].Substring(1));
                cases.Add(new FilterCase(name, false, e, bytes, e * 31 + bytes % 97, 0));
            }
            else
            {
                int n = int.Parse(parts[2].Substring(1));
                int bs = int.Parse(parts[3].Substring(2));
                cases.Add(new FilterCase(name, true, e, n * e, e * 7 + n % 89 + bs, bs));
            }
        }
        cases.Sort((a, b) => string.CompareOrdinal(a.Resource, b.Resource));
        return cases;
    }

    /// <summary>Runs one direction into a poisoned buffer and compares it (and the untouched slack) with the truth on the accelerator.</summary>
    private static async Task<string?> RunFilterCaseAsync(Accelerator acc, ShuffleFilterGpu filter, bool bit, bool inverse,
        byte[] input, byte[] expected, int elementSize, int block)
    {
        const int Slack = 37;
        using var dIn = DeviceVerify.Upload(acc, input, Slack);
        using var dOut = DeviceVerify.Poisoned(acc, input.Length + Slack);
        if (!bit && !inverse) await filter.ShuffleAsync(dIn.View, dOut.View, input.Length, elementSize);
        else if (!bit) await filter.UnshuffleAsync(dIn.View, dOut.View, input.Length, elementSize);
        else if (!inverse) await filter.BitShuffleAsync(dIn.View, dOut.View, input.Length, elementSize, block);
        else await filter.BitUnshuffleAsync(dIn.View, dOut.View, input.Length, elementSize, block);
        await acc.SynchronizeAsync();
        var withSlack = new byte[expected.Length + Slack];
        expected.CopyTo(withSlack, 0);
        System.Array.Fill(withSlack, (byte)0xAB, expected.Length, Slack);
        return await DeviceVerify.CompareAsync(acc, dOut.View, withSlack);
    }

    private async Task AssertFilterFixturesAsync(bool bit)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var filter = new ShuffleFilterGpu(acc);
            var cases = FilterCases().Where(c => c.Bit == bit).ToList();
            True(cases.Count >= (bit ? 50 : 42), $"only {cases.Count} fixtures embedded");
            var failures = new List<string>();
            foreach (var c in cases)
            {
                byte[] input = FilterInput(c.Length, c.Seed);
                byte[] truth = ReadEmbedded(c.Resource);
                Equal(input.Length, truth.Length, $"{c.Resource}: fixture length");
                string? fwd = await RunFilterCaseAsync(acc, filter, bit, false, input, truth, c.ElementSize, c.Block);
                if (fwd != null) failures.Add($"{c.Resource} forward: {fwd}");
                string? inv = await RunFilterCaseAsync(acc, filter, bit, true, truth, input, c.ElementSize, c.Block);
                if (inv != null) failures.Add($"{c.Resource} inverse: {inv}");
            }
            True(failures.Count == 0, $"{failures.Count} of {2 * cases.Count} runs differ from the reference:\n" + string.Join("\n", failures.Take(12)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task ShuffleFilterGpu_ByteShuffle_AllFixtures_MatchBlosc_BothDirections() => await AssertFilterFixturesAsync(bit: false);

    [TestMethod(Timeout = 600000)]
    public async Task ShuffleFilterGpu_BitShuffle_AllFixtures_MatchBitshuffleLibrary_BothDirections() => await AssertFilterFixturesAsync(bit: true);

    /// <summary>16 MiB per case (four launches, thousands of blocks): forward output hashed against the reference, inverse compared on the accelerator.</summary>
    [TestMethod(Timeout = 600000)]
    public async Task ShuffleFilterGpu_Large16MiB_MatchReferenceSha256_AndInvert()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var filter = new ShuffleFilterGpu(acc);
            var lines = ReadEmbeddedText(FilterFixturePrefix + "large.txt").Split('\n', StringSplitOptions.RemoveEmptyEntries);
            Equal(3, lines.Length, "large.txt cases");
            foreach (var line in lines)
            {
                var f = line.Trim().Split(' ');
                bool bit = f[0] == "bshuf";
                int e = int.Parse(f[1]), length = int.Parse(f[2]), seed = int.Parse(f[3]);
                string sha = f[4];
                byte[] input = FilterInput(length, seed);
                using var dIn = DeviceVerify.Upload(acc, input);
                using var dMid = DeviceVerify.Poisoned(acc, length);
                using var dBack = DeviceVerify.Poisoned(acc, length);
                if (bit)
                {
                    await filter.BitShuffleAsync(dIn.View, dMid.View, length, e);
                    await filter.BitUnshuffleAsync(dMid.View, dBack.View, length, e);
                }
                else
                {
                    await filter.ShuffleAsync(dIn.View, dMid.View, length, e);
                    await filter.UnshuffleAsync(dMid.View, dBack.View, length, e);
                }
                await acc.SynchronizeAsync();
                string got = Sha256Hex(await DeviceTransfer.DownloadAsync(dMid.View, length));
                Equal(sha, got, $"{line}: SHA-256 of our output");
                string? back = await DeviceVerify.CompareAsync(acc, dBack.View, dIn.View, length);
                True(back == null, $"{line}: inverse: {back}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task ShuffleFilterGpu_ManagedOverloads_MatchFixtures()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var filter = new ShuffleFilterGpu(acc);
            foreach (var c in FilterCases().Where(c => c.Length == 4099 || c.Resource.Contains("_n4103_")))
            {
                byte[] input = FilterInput(c.Length, c.Seed);
                byte[] truth = ReadEmbedded(c.Resource);
                byte[] fwd = c.Bit ? await filter.BitShuffleAsync(input, c.ElementSize, c.Block) : await filter.ShuffleAsync(input, c.ElementSize);
                EqualBytes(truth, fwd, $"{c.Resource}: managed forward");
                byte[] inv = c.Bit ? await filter.BitUnshuffleAsync(truth, c.ElementSize, c.Block) : await filter.UnshuffleAsync(truth, c.ElementSize);
                EqualBytes(input, inv, $"{c.Resource}: managed inverse");
            }
            Equal(0, (await filter.BitShuffleAsync(System.Array.Empty<byte>(), 4)).Length, "empty input");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task ShuffleFilterGpu_TypedArrayOverloads_MatchFixtures()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var filter = new ShuffleFilterGpu(acc);
            await RunShuffleTypedArrayCases(filter);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunShuffleTypedArrayCases(ShuffleFilterGpu filter)
    {
        foreach (var c in FilterCases().Where(c => c.Length == 4099 || c.Resource.Contains("_n4103_")))
        {
            byte[] input = FilterInput(c.Length, c.Seed);
            byte[] truth = ReadEmbedded(c.Resource);
            using var jsIn = new Uint8Array(input.Length);
            jsIn.Set(input);
            using var fwd = c.Bit ? await filter.BitShuffleAsync(jsIn, c.ElementSize, c.Block) : await filter.ShuffleAsync(jsIn, c.ElementSize);
            EqualBytes(truth, fwd.ReadBytes(), $"{c.Resource}: TypedArray forward");
            using var jsTruth = new Uint8Array(truth.Length);
            jsTruth.Set(truth);
            using var inv = c.Bit ? await filter.BitUnshuffleAsync(jsTruth, c.ElementSize, c.Block) : await filter.UnshuffleAsync(jsTruth, c.ElementSize);
            EqualBytes(input, inv.ReadBytes(), $"{c.Resource}: TypedArray inverse");
        }
    }

    [TestMethod]
    public async Task ShuffleFilterGpu_RejectsBadArguments()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var filter = new ShuffleFilterGpu(acc);
            using var a = acc.Allocate1D<byte>(64);
            using var b = acc.Allocate1D<byte>(32);
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await filter.BitShuffleAsync(a.View, a.View, 64, 4, 12), "block not a multiple of 8");
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await filter.ShuffleAsync(a.View, a.View, 64, 0), "element size 0");
            await ThrowsAsync<ArgumentException>(async () => await filter.ShuffleAsync(a.View, b.View, 64, 4), "destination too short");
            await ThrowsAsync<ArgumentOutOfRangeException>(async () => await filter.ShuffleAsync(b.View, a.View, 64, 4), "source too short");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
