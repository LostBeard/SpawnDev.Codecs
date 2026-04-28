// Tests for Vp8BoolDecoderGpu. CPU encoder produces bytes, GPU
// decoder reads them back, decoded bits MUST match the encoder's
// input bits. Foundation for the GPU-resident decoder side.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp8BoolDecoderGpu_RandomSequence_RoundTripsCpuEncoder()
    {
        // For each of N streams: pick random bits + random probs,
        // encode on CPU via Vp8BoolEncoder, decode on GPU via the test
        // kernel, assert decoded bits == original bits.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8BoolDecoderTestKernel(acc);
            const int streamCount = 4;
            const int bitsPerStream = 1024;
            const int inStride = 4096;

            var rng = new Random(unchecked((int)0xDEC0DE01));
            var origBits = new byte[streamCount * bitsPerStream];
            var probs = new int[streamCount * bitsPerStream];
            for (int i = 0; i < origBits.Length; i++) origBits[i] = (byte)rng.Next(2);
            for (int i = 0; i < probs.Length; i++) probs[i] = 1 + rng.Next(255);

            // CPU encode each stream.
            var encodedBytes = new byte[streamCount * inStride];
            var encodedLens = new int[streamCount];
            for (int s = 0; s < streamCount; s++)
            {
                var enc = new Vp8BoolEncoder();
                for (int b = 0; b < bitsPerStream; b++)
                    enc.EncodeBool(origBits[s * bitsPerStream + b], probs[s * bitsPerStream + b]);
                byte[] bytes = enc.Stop();
                Array.Copy(bytes, 0, encodedBytes, s * inStride, bytes.Length);
                encodedLens[s] = bytes.Length;
            }

            // GPU decode.
            using var dIn = acc.Allocate1D<byte>(encodedBytes.Length);
            using var dInLens = acc.Allocate1D<int>(streamCount);
            using var dProbs = acc.Allocate1D<int>(probs.Length);
            using var dOut = acc.Allocate1D<byte>(origBits.Length);
            dIn.View.CopyFromCPU(encodedBytes);
            dInLens.View.CopyFromCPU(encodedLens);
            dProbs.View.CopyFromCPU(probs);
            dOut.View.MemSetToZero();
            kernel.Run(dIn.View, dInLens.View, dProbs.View, dOut.View,
                streamCount, bitsPerStream, inStride);
            await acc.SynchronizeAsync();

            // GPU-side verification: upload origBits, count mismatches.
            int mismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dOut.View, origBits, origBits.Length);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
