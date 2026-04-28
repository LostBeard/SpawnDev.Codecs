// Tests for Vp8CoefBlockEncoderGpu - GPU-resident VP8 per-block coef
// encoder. Bit-exact vs Vp8CoefBlockEncoder for arbitrary post-Q
// coef blocks across block-types and contexts.

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
    public async Task Vp8CoefBlockEncoderGpu_RandomBlocks_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8CoefBlockEncoderTestKernel(acc);
            const int streamCount = 4;
            const int blocksPerStream = 32;
            const int outBufStride = 32 * 1024; // generous worst-case

            var rng = new Random(unchecked((int)0xC0EFC0DE));
            // Realistic post-Q coef distribution: mostly zero with some
            // small non-zero spikes; per-block ctx in 0..2; firstCoef 0 or 1.
            var coefs = new short[streamCount * blocksPerStream * 16];
            var ctxs = new int[streamCount * blocksPerStream];
            var firstCoefs = new int[streamCount * blocksPerStream];
            for (int i = 0; i < coefs.Length; i++)
            {
                coefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-50, 50) : (short)0;
            }
            for (int i = 0; i < ctxs.Length; i++)
            {
                ctxs[i] = rng.Next(3);
                firstCoefs[i] = rng.Next(2);
            }

            // Use Vp8DefaultCoefProbs for block type 0 (Y4-no-DC).
            var probsFlat = new byte[8 * 3 * 11];
            var defaults = Vp8DefaultCoefProbs.DefaultProbs;
            for (int band = 0; band < 8; band++)
                for (int c = 0; c < 3; c++)
                    for (int n = 0; n < 11; n++)
                        probsFlat[band * 33 + c * 11 + n] = defaults[0, band, c, n];

            var constsFlat = Vp8CoefBlockEncoderGpu.BuildConstsBuffer();

            // CPU reference per stream.
            var cpuOuts = new byte[streamCount][];
            for (int s = 0; s < streamCount; s++)
            {
                // Materialize the 3D probs view the CPU encoder expects.
                var probs3d = new byte[8, 3, 11];
                for (int band = 0; band < 8; band++)
                    for (int c = 0; c < 3; c++)
                        for (int n = 0; n < 11; n++)
                            probs3d[band, c, n] = defaults[0, band, c, n];
                var enc = new Vp8BoolEncoder();
                for (int b = 0; b < blocksPerStream; b++)
                {
                    int ctxParam = ctxs[s * blocksPerStream + b];
                    int firstCoef = firstCoefs[s * blocksPerStream + b];
                    var blockCoefs = coefs.AsSpan(
                        (s * blocksPerStream + b) * 16, 16);
                    Vp8CoefBlockEncoder.Encode(enc, probs3d, ctxParam, firstCoef, blockCoefs);
                }
                cpuOuts[s] = enc.Stop();
            }

            // GPU encode.
            using var dCoefs = acc.Allocate1D<short>(coefs.Length);
            using var dCtxs = acc.Allocate1D<int>(ctxs.Length);
            using var dFirstCoefs = acc.Allocate1D<int>(firstCoefs.Length);
            using var dProbs = acc.Allocate1D<byte>(probsFlat.Length);
            using var dConsts = acc.Allocate1D<byte>(constsFlat.Length);
            using var dOut = acc.Allocate1D<byte>(streamCount * outBufStride);
            using var dLens = acc.Allocate1D<long>(streamCount);
            dCoefs.View.CopyFromCPU(coefs);
            dCtxs.View.CopyFromCPU(ctxs);
            dFirstCoefs.View.CopyFromCPU(firstCoefs);
            dProbs.View.CopyFromCPU(probsFlat);
            dConsts.View.CopyFromCPU(constsFlat);
            dOut.View.MemSetToZero();
            kernel.Run(dCoefs.View, dCtxs.View, dFirstCoefs.View,
                dProbs.View, dConsts.View, dOut.View, dLens.View,
                streamCount, blocksPerStream, outBufStride);
            await acc.SynchronizeAsync();

            var lensBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dLens);
            var outBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            int totalMismatches = 0;
            int firstBadStream = -1, firstBadByte = -1;
            for (int s = 0; s < streamCount; s++)
            {
                long gpuLen = lensBack[s];
                if ((long)cpuOuts[s].Length != gpuLen)
                {
                    if (firstBadStream < 0) { firstBadStream = s; firstBadByte = -1; }
                    totalMismatches += Math.Abs((int)gpuLen - cpuOuts[s].Length);
                }
                for (int i = 0; i < cpuOuts[s].Length && i < gpuLen; i++)
                {
                    if (cpuOuts[s][i] != outBack[s * outBufStride + i])
                    {
                        if (firstBadStream < 0) { firstBadStream = s; firstBadByte = i; }
                        totalMismatches++;
                    }
                }
            }
            Equal(0, totalMismatches,
                $"first mismatch stream={firstBadStream} byte={firstBadByte}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
