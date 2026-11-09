// FLAC encoder: the same file on every backend. The window, autocorrelation and LPC analysis compute in double; the
// coefficients (and so the file) depend on IEEE-754 binary64 rounding. WebGPU and WebGL emulate f64, and the encoder
// compiles those three kernels with ILGPU's bit-exact Ieee emulation (F64EmulationScope). The expected MD5s were
// produced on the desktop CPU accelerator (native IEEE doubles); every backend must write the same bytes.
// Regenerate with FLAC_PRINT_MD5=1 on CpuCodecsTests (prints the table) when the encoder's output changes on purpose.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // (name, channels, bits, level, block size) -> MD5 of the FLAC file the desktop CPU accelerator writes.
    private static readonly (string Name, int Channels, int Bits, int Level, int? Block, string Md5)[] FlacSameFileCases =
    {
        ("stereo16_l0", 2, 16, 0, null, "e9d512d3e1fbb39da80c8ddc741f0a5a"),
        ("stereo16_l5", 2, 16, 5, null, "567d547c0f7279f9ea1a3bdcfec021ac"),
        ("stereo16_l8", 2, 16, 8, null, "621b1c8ed62547a2708ee01721ff61d0"),
        ("mono24_l8", 1, 24, 8, null, "5d934f0bbe59fa968f22cdba9fd09b39"),
        ("stereo16_l8_b1152", 2, 16, 8, 1152, "61db69772f65d2cc7c2e0a5bd52f36fb"),
        ("stereo20_l7_b2048", 2, 20, 7, 2048, "fbc32c3f24a1b134eb8e9bbe8ffb3225"),
    };

    [TestMethod(Timeout = 1200000)]
    public async Task FlacEncoderGpu_SameFileOnEveryBackend()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            bool print = Environment.GetEnvironmentVariable("FLAC_PRINT_MD5") == "1";
            using var enc = new FlacEncoderGpu(acc);
            var failures = new List<string>();
            int seed = 7;
            foreach (var c in FlacSameFileCases)
            {
                int[] pcm = FlacEncSynth(20000, c.Channels, c.Bits, seed++);
                byte[] file = await enc.EncodeAsync(pcm, c.Channels, 44100, c.Bits,
                    new FlacEncodeOptions { CompressionLevel = c.Level, BlockSize = c.Block });
                string md5 = Md5Hex(file);
                if (print) Console.WriteLine($"FLAC_MD5 (\"{c.Name}\", {c.Channels}, {c.Bits}, {c.Level}, {(c.Block?.ToString() ?? "null")}, \"{md5}\"),");
                else if (md5 != c.Md5) failures.Add($"{c.Name}: {md5}, desktop {c.Md5} ({file.Length} bytes)");
            }
            if (print) throw new UnsupportedTestException("FLAC_PRINT_MD5=1: printed the table instead of checking it.");
            True(failures.Count == 0, $"{acc.AcceleratorType}: FLAC files differ from the desktop's:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
