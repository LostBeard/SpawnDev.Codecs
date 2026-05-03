// Cross-backend test for FlacLpcSubframeDecodeGpu.DecodeAt. Verifies the
// GPU LPC-subframe composite (warm-up reads -> precision+quantLevel ->
// QLP coefs -> residual -> reconstruct -> wasted-shift) matches the CPU
// reference end-to-end by encoding a synthetic LPC subframe and comparing
// decoded PCM bit-exactly.

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
    public async Task FlacLpcSubframeDecodeGpu_Order8_Q12_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 8, 12-bit precision, quant 12, 64-block, 16-bit samples.
            int[] coefs = { 100, -75, 50, -25, 12, -6, 3, -1 };
            await EncodeDecodeAndVerify(acc, coefs, blockSize: 64, bps: 16,
                wastedBits: 0, precision: 12, quantLevel: 12, riceParam: 5,
                seed: 0xCDFE0008u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcSubframeDecodeGpu_Order16_Q14_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 16, 14-bit precision, quant 14, 256-block.
            int[] coefs = { 8000, -6000, 4000, -2000, 1000, -500, 250, -100,
                              50,   -25,   12,    -6,    3,   -1,   1,    0 };
            await EncodeDecodeAndVerify(acc, coefs, blockSize: 256, bps: 16,
                wastedBits: 0, precision: 14, quantLevel: 14, riceParam: 6,
                seed: 0xCDFE0010u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacLpcSubframeDecodeGpu_Order4_WastedBits_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Wasted-bits path: declared bps 16, wasted 4 -> effective 12.
            int[] coefs = { 50, -25, 12, -6 };
            await EncodeDecodeAndVerify(acc, coefs, blockSize: 32, bps: 12,
                wastedBits: 4, precision: 8, quantLevel: 7, riceParam: 4,
                seed: 0xCDFE0004u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeDecodeAndVerify(
        Accelerator acc, int[] coefs, int blockSize, int bps,
        int wastedBits, int precision, int quantLevel, int riceParam, uint seed)
    {
        int order = coefs.Length;
        var rng = new Random(unchecked((int)seed));

        // Build synthetic signal at effective bps.
        int range = 1 << (bps - 1);
        int[] origSignal = new int[blockSize];
        for (int i = 0; i < blockSize; i++) origSignal[i] = rng.Next(-range / 8, range / 8);

        // Compute residual against the LPC predictor (encoder side).
        int[] residual = new int[blockSize - order];
        for (int n = order; n < blockSize; n++)
        {
            long pred = 0;
            for (int i = 0; i < order; i++)
                pred += (long)coefs[i] * origSignal[n - 1 - i];
            residual[n - order] = (int)(origSignal[n] - (pred >> quantLevel));
        }

        // Build the bit stream: warm-up + 4-bit precision-1 + 5-bit quantLevel +
        // QLP coefs + residual block (codingMethod=0, partitionOrder=0).
        var w = new FlacBitWriter();
        for (int i = 0; i < order; i++) w.WriteSigned(origSignal[i], bps);
        w.Write((uint)(precision - 1), 4);
        w.WriteSigned(quantLevel, 5);
        for (int i = 0; i < order; i++) w.WriteSigned(coefs[i], precision);
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
        using var dCoefs = acc.Allocate1D<int>(order);
        dData.View.CopyFromCPU(encoded);
        dSamples.MemSetToZero();
        dCoefs.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>,
            int, int, int, int, int>(LpcSubframeKernel);
        kernel(new Index1D(1), dData.View, dSamples.View, dCoefs.View,
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

    private static void LpcSubframeKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> samples, ArrayView<int> coefsScratch,
        int dataLen, int blockSize, int order, int effectiveBps, int wastedBits)
    {
        var state = FlacBitReaderGpu.Init(dataLen);
        FlacLpcSubframeDecodeGpu.DecodeAt(ref state, data, samples, 0,
            coefsScratch, 0, blockSize, order, effectiveBps, wastedBits);
    }
}
