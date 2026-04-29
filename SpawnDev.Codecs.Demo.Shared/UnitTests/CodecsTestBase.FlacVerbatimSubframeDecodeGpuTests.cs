// Cross-backend test for FlacVerbatimSubframeDecodeGpu.DecodeAt. Verifies
// the GPU VERBATIM-subframe composite (per-sample bit reads + wasted-
// bits left shift) round-trips correctly via a synthetic FlacBitWriter
// stream of known sample values.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacVerbatimSubframeDecodeGpu_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await EncodeDecodeAndVerify(acc, blockSize: 256, bps: 16, wastedBits: 0,
                seed: 0xF1AC_4E50u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacVerbatimSubframeDecodeGpu_WastedBits_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 16-bit subframe with 4 wasted bits -> effective 12.
            await EncodeDecodeAndVerify(acc, blockSize: 64, bps: 12, wastedBits: 4,
                seed: 0xF1AC_4E12u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacVerbatimSubframeDecodeGpu_24Bit_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 24-bit hi-res audio.
            await EncodeDecodeAndVerify(acc, blockSize: 128, bps: 24, wastedBits: 0,
                seed: 0xF1AC_4E24u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacVerbatimSubframeDecodeGpu_FullBlock_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Full FLAC default block size.
            await EncodeDecodeAndVerify(acc, blockSize: 4096, bps: 16, wastedBits: 0,
                seed: 0xF1AC_4096u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeDecodeAndVerify(
        Accelerator acc, int blockSize, int bps, int wastedBits, uint seed)
    {
        var rng = new Random(unchecked((int)seed));

        // Build synthetic samples at effective bps.
        int range = 1 << (bps - 1);
        int[] origSignal = new int[blockSize];
        for (int i = 0; i < blockSize; i++) origSignal[i] = rng.Next(-range, range);

        // Build the bit stream: just blockSize signed bps-bit samples.
        var w = new FlacBitWriter();
        for (int i = 0; i < blockSize; i++) w.WriteSigned(origSignal[i], bps);
        w.AlignToByte();
        byte[] encoded = w.ToArray();

        // GPU decode.
        using var dData = acc.Allocate1D<byte>(encoded.Length);
        using var dSamples = acc.Allocate1D<int>(blockSize);
        dData.View.CopyFromCPU(encoded);
        dSamples.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, int, int, int, int>(VerbatimKernel);
        kernel(new Index1D(1), dData.View, dSamples.View,
            encoded.Length, blockSize, bps, wastedBits);
        await acc.SynchronizeAsync();

        var gpuSamples = await dSamples.CopyToHostAsync();

        // Expected: orig signal left-shifted by wastedBits (if any).
        for (int i = 0; i < blockSize; i++)
        {
            int expected = wastedBits > 0 ? (origSignal[i] << wastedBits) : origSignal[i];
            if (gpuSamples[i] != expected)
                throw new Exception($"samples[{i}]: gpu={gpuSamples[i]} expected={expected} (bps={bps}, wasted={wastedBits})");
        }
    }

    private static void VerbatimKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> samples,
        int dataLen, int blockSize, int effectiveBps, int wastedBits)
    {
        var state = FlacBitReaderGpu.Init(dataLen);
        FlacVerbatimSubframeDecodeGpu.DecodeAt(ref state, data, samples, 0,
            blockSize, effectiveBps, wastedBits);
    }
}
