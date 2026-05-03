// Cross-backend test for FlacFixedSubframeDecodeGpu.DecodeAt. Verifies the
// GPU FIXED-subframe composite (warm-up reads -> residual decode ->
// reconstruct -> wasted-bits left shift) matches the CPU reference end-
// to-end by encoding a synthetic FIXED subframe and comparing decoded
// PCM bit-exactly.

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
    public async Task FlacFixedSubframeDecodeGpu_Order2_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 2 FIXED, 16-bit samples, no wasted bits.
            await EncodeDecodeAndVerify(acc, order: 2, blockSize: 32, bps: 16, wastedBits: 0,
                seed: 0xF1AC1620u, riceParam: 5);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFixedSubframeDecodeGpu_Order4_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await EncodeDecodeAndVerify(acc, order: 4, blockSize: 64, bps: 16, wastedBits: 0,
                seed: 0xF1AC1640u, riceParam: 6);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFixedSubframeDecodeGpu_Order2_WastedBits_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Wasted-bits path: declared bps 16, wasted 4 -> effective 12.
            await EncodeDecodeAndVerify(acc, order: 2, blockSize: 32, bps: 12, wastedBits: 4,
                seed: 0xF1ACBABEu, riceParam: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeDecodeAndVerify(
        Accelerator acc, int order, int blockSize, int bps,
        int wastedBits, uint seed, int riceParam)
    {
        var rng = new Random(unchecked((int)seed));

        // Build a synthetic signal at effective bps.
        int range = 1 << (bps - 1);
        int[] origSignal = new int[blockSize];
        for (int i = 0; i < blockSize; i++) origSignal[i] = rng.Next(-range / 4, range / 4);

        // Compute residual against the FIXED predictor (encoder side).
        int[] residual = new int[blockSize - order];
        for (int n = order; n < blockSize; n++)
        {
            long pred = 0;
            switch (order)
            {
                case 1: pred = origSignal[n - 1]; break;
                case 2: pred = 2L * origSignal[n - 1] - origSignal[n - 2]; break;
                case 3: pred = 3L * origSignal[n - 1] - 3L * origSignal[n - 2] + origSignal[n - 3]; break;
                case 4: pred = 4L * origSignal[n - 1] - 6L * origSignal[n - 2] + 4L * origSignal[n - 3] - origSignal[n - 4]; break;
            }
            residual[n - order] = (int)(origSignal[n] - pred);
        }

        // Build the bit stream: warm-up + residual block (codingMethod=0, partitionOrder=0).
        var w = new FlacBitWriter();
        for (int i = 0; i < order; i++) w.WriteSigned(origSignal[i], bps);
        w.Write(0, 2); // codingMethod = 0 (4-bit Rice param)
        w.Write(0, 4); // partitionOrder = 0
        w.Write((uint)riceParam, 4);
        for (int i = 0; i < residual.Length; i++)
        {
            int r = residual[i];
            uint u = r >= 0 ? (uint)(r << 1) : (uint)((-r << 1) - 1);
            int q = (int)(u >> riceParam);
            uint rem = u & ((1u << riceParam) - 1);
            w.WriteUnary(q);
            if (riceParam > 0) w.Write(rem, riceParam);
        }
        w.AlignToByte();
        byte[] encoded = w.ToArray();

        // GPU decode.
        using var dData = acc.Allocate1D<byte>(encoded.Length);
        using var dSamples = acc.Allocate1D<int>(blockSize);
        dData.View.CopyFromCPU(encoded);
        dSamples.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>,
            int, int, int, int, int>(SubframeKernel);
        kernel(new Index1D(1), dData.View, dSamples.View,
            encoded.Length, blockSize, order, bps, wastedBits);
        await acc.SynchronizeAsync();

        var gpuSamples = await dSamples.CopyToHostAsync();

        // Expected output: orig signal left-shifted by wastedBits.
        for (int i = 0; i < blockSize; i++)
        {
            int expected = wastedBits > 0 ? (origSignal[i] << wastedBits) : origSignal[i];
            if (gpuSamples[i] != expected)
                throw new Exception($"samples[{i}]: gpu={gpuSamples[i]} expected={expected} (order={order}, wasted={wastedBits})");
        }
    }

    private static void SubframeKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> samples,
        int dataLen, int blockSize, int order, int effectiveBps, int wastedBits)
    {
        var state = FlacBitReaderGpu.Init(dataLen);
        FlacFixedSubframeDecodeGpu.DecodeAt(ref state, data, samples, 0,
            blockSize, order, effectiveBps, wastedBits);
    }
}
