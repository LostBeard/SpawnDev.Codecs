// Cross-backend test for FlacConstantSubframeDecodeGpu.DecodeAt. Verifies
// the GPU CONSTANT-subframe composite (single value read + broadcast +
// wasted-bits left shift) matches the CPU reference behavior bit-exactly.

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
    public async Task FlacConstantSubframeDecodeGpu_PositiveValue_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await EncodeDecodeAndVerify(acc, value: 12345, blockSize: 256,
                effectiveBps: 16, wastedBits: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacConstantSubframeDecodeGpu_NegativeValue_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await EncodeDecodeAndVerify(acc, value: -7891, blockSize: 1024,
                effectiveBps: 16, wastedBits: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacConstantSubframeDecodeGpu_WastedBits_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Wasted bits 3 with 13-bit effective value: orig 16-bit fits.
            await EncodeDecodeAndVerify(acc, value: 256, blockSize: 64,
                effectiveBps: 13, wastedBits: 3);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacConstantSubframeDecodeGpu_FullBlock_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Full FLAC default block size 4096.
            await EncodeDecodeAndVerify(acc, value: -1, blockSize: 4096,
                effectiveBps: 16, wastedBits: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeDecodeAndVerify(
        Accelerator acc, int value, int blockSize, int effectiveBps, int wastedBits)
    {
        // Build the bit stream: just the signed value at effectiveBps.
        var w = new FlacBitWriter();
        w.WriteSigned(value, effectiveBps);
        w.AlignToByte();
        byte[] encoded = w.ToArray();

        // GPU decode.
        using var dData = acc.Allocate1D<byte>(encoded.Length);
        using var dSamples = acc.Allocate1D<int>(blockSize);
        dData.View.CopyFromCPU(encoded);
        dSamples.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, int, int, int, int>(ConstSubframeKernel);
        kernel(new Index1D(1), dData.View, dSamples.View,
            encoded.Length, blockSize, effectiveBps, wastedBits);
        await acc.SynchronizeAsync();

        var gpuSamples = await dSamples.CopyToHostAsync();

        int expected = wastedBits > 0 ? (value << wastedBits) : value;
        for (int i = 0; i < blockSize; i++)
        {
            if (gpuSamples[i] != expected)
                throw new Exception($"samples[{i}]: gpu={gpuSamples[i]} expected={expected} (value={value}, wasted={wastedBits})");
        }
    }

    private static void ConstSubframeKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> samples,
        int dataLen, int blockSize, int effectiveBps, int wastedBits)
    {
        var state = FlacBitReaderGpu.Init(dataLen);
        FlacConstantSubframeDecodeGpu.DecodeAt(ref state, data, samples, 0,
            blockSize, effectiveBps, wastedBits);
    }
}
