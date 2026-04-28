// Tests for Vp8CoefBlockDecoderGpu - GPU-resident per-block coef
// decoder. Round-trip via CPU encoder produces bytes, GPU decoder
// reads them back, decoded coefs MUST match the encoder's input.

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
    public async Task Vp8CoefBlockDecoderGpu_RandomBlocks_RoundTripsCpuEncoder()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8CoefBlockDecoderTestKernel(acc);
            const int streamCount = 4;
            const int blocksPerStream = 32;
            const int inStride = 32 * 1024;

            var rng = new Random(unchecked((int)0xC0EFDEC0));
            var origCoefs = new short[streamCount * blocksPerStream * 16];
            var ctxs = new int[streamCount * blocksPerStream];
            var firstCoefs = new int[streamCount * blocksPerStream];
            for (int i = 0; i < origCoefs.Length; i++)
                origCoefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-50, 50) : (short)0;
            for (int i = 0; i < ctxs.Length; i++)
            {
                ctxs[i] = rng.Next(3);
                firstCoefs[i] = rng.Next(2);
            }
            // Coefs at the firstCoef-1 position are unused (skipped by both encoder + decoder).
            // For firstCoef=1 blocks, slot 0 must match what the decoder will leave there - which is 0.
            // So zero out slot 0 for firstCoef=1 blocks to make the round-trip comparison work.
            for (int b = 0; b < streamCount * blocksPerStream; b++)
                if (firstCoefs[b] == 1)
                    origCoefs[b * 16 + 0] = 0;

            // Use Vp8DefaultCoefProbs for block type 0.
            var probsFlat = new byte[8 * 3 * 11];
            var defaults = Vp8DefaultCoefProbs.DefaultProbs;
            for (int band = 0; band < 8; band++)
                for (int c = 0; c < 3; c++)
                    for (int n = 0; n < 11; n++)
                        probsFlat[band * 33 + c * 11 + n] = defaults[0, band, c, n];
            var constsFlat = Vp8CoefBlockEncoderGpu.BuildConstsBuffer();

            // CPU encode all streams.
            var encodedBytes = new byte[streamCount * inStride];
            var encodedLens = new int[streamCount];
            for (int s = 0; s < streamCount; s++)
            {
                var probs3d = new byte[8, 3, 11];
                for (int band = 0; band < 8; band++)
                    for (int c = 0; c < 3; c++)
                        for (int n = 0; n < 11; n++)
                            probs3d[band, c, n] = defaults[0, band, c, n];
                var enc = new Vp8BoolEncoder();
                for (int b = 0; b < blocksPerStream; b++)
                {
                    int cParam = ctxs[s * blocksPerStream + b];
                    int firstCoef = firstCoefs[s * blocksPerStream + b];
                    var blockCoefs = origCoefs.AsSpan(
                        (s * blocksPerStream + b) * 16, 16);
                    Vp8CoefBlockEncoder.Encode(enc, probs3d, cParam, firstCoef, blockCoefs);
                }
                byte[] bytes = enc.Stop();
                Array.Copy(bytes, 0, encodedBytes, s * inStride, bytes.Length);
                encodedLens[s] = bytes.Length;
            }

            // GPU decode.
            using var dIn = acc.Allocate1D<byte>(encodedBytes.Length);
            using var dInLens = acc.Allocate1D<int>(streamCount);
            using var dCtxs = acc.Allocate1D<int>(ctxs.Length);
            using var dFirstCoefs = acc.Allocate1D<int>(firstCoefs.Length);
            using var dProbs = acc.Allocate1D<byte>(probsFlat.Length);
            using var dConsts = acc.Allocate1D<byte>(constsFlat.Length);
            using var dCoefsOut = acc.Allocate1D<short>(origCoefs.Length);
            using var dEobsOut = acc.Allocate1D<int>(streamCount * blocksPerStream);
            dIn.View.CopyFromCPU(encodedBytes);
            dInLens.View.CopyFromCPU(encodedLens);
            dCtxs.View.CopyFromCPU(ctxs);
            dFirstCoefs.View.CopyFromCPU(firstCoefs);
            dProbs.View.CopyFromCPU(probsFlat);
            dConsts.View.CopyFromCPU(constsFlat);
            dCoefsOut.View.MemSetToZero();

            kernel.Run(dIn.View, dInLens.View, dCtxs.View, dFirstCoefs.View,
                dProbs.View, dConsts.View, dCoefsOut.View, dEobsOut.View,
                streamCount, blocksPerStream, inStride);
            await acc.SynchronizeAsync();

            // GPU-side verification: count short mismatches against the original coefs.
            int mismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dCoefsOut.View, origCoefs, origCoefs.Length);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
