// Tests for Vp8FrameTransformGpu - frame-level GPU-resident transform
// pipeline. Asserts the output of the full FDCT->Walsh->Quant chain
// matches what the CPU produces if it ran the same chain.
//
// This is the v1 GPU-resident pipeline driver: takes residuals on
// device, returns quantized coefs on device, no CPU<->GPU bouncing
// between kernel stages.

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
    public async Task Vp8FrameTransformGpu_RandomMacroblocks_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var pipeline = new Vp8FrameTransformGpu(acc);
            const int mbCount = 8;
            int y4BlockCount = mbCount * 16;
            int uvBlockCount = mbCount * 4;

            var rng = new Random(0x88DA8F);
            var y4Residual = new short[y4BlockCount * 16];
            var uResidual = new short[uvBlockCount * 16];
            var vResidual = new short[uvBlockCount * 16];
            for (int i = 0; i < y4Residual.Length; i++) y4Residual[i] = (short)rng.Next(-128, 128);
            for (int i = 0; i < uResidual.Length; i++) uResidual[i] = (short)rng.Next(-128, 128);
            for (int i = 0; i < vResidual.Length; i++) vResidual[i] = (short)rng.Next(-128, 128);

            // Per-MB quantizers (all blocks in an MB share the same).
            var y1DcQ = new short[mbCount];
            var y1AcQ = new short[mbCount];
            var y2DcQ = new short[mbCount];
            var y2AcQ = new short[mbCount];
            var uvDcQ = new short[mbCount];
            var uvAcQ = new short[mbCount];
            for (int b = 0; b < mbCount; b++)
            {
                y1DcQ[b] = (short)(8 + rng.Next(60));
                y1AcQ[b] = (short)(8 + rng.Next(60));
                y2DcQ[b] = (short)(16 + rng.Next(80));
                y2AcQ[b] = (short)(16 + rng.Next(80));
                uvDcQ[b] = (short)(8 + rng.Next(60));
                uvAcQ[b] = (short)(8 + rng.Next(60));
            }

            // CPU reference: replicate the GPU pipeline step by step.
            var cpuY4 = new short[y4BlockCount * 16];
            var cpuY2 = new short[mbCount * 16];
            var cpuU = new short[uvBlockCount * 16];
            var cpuV = new short[uvBlockCount * 16];

            // 1. FDCT each Y4, U, V block.
            for (int b = 0; b < y4BlockCount; b++)
                Vp8ForwardTransform.ShortFdct4x4(
                    y4Residual.AsSpan(b * 16, 16), 4,
                    cpuY4.AsSpan(b * 16, 16));
            for (int b = 0; b < uvBlockCount; b++)
                Vp8ForwardTransform.ShortFdct4x4(
                    uResidual.AsSpan(b * 16, 16), 4,
                    cpuU.AsSpan(b * 16, 16));
            for (int b = 0; b < uvBlockCount; b++)
                Vp8ForwardTransform.ShortFdct4x4(
                    vResidual.AsSpan(b * 16, 16), 4,
                    cpuV.AsSpan(b * 16, 16));

            // 2. Gather Y4 DCs into pre-Walsh Y2 block (per MB).
            var preWalsh = new short[mbCount * 16];
            for (int mb = 0; mb < mbCount; mb++)
                for (int slot = 0; slot < 16; slot++)
                    preWalsh[mb * 16 + slot] = cpuY4[mb * 16 * 16 + slot * 16 + 0];

            // 3. Forward Walsh per MB.
            for (int mb = 0; mb < mbCount; mb++)
                Vp8ForwardTransform.ShortWalsh4x4(
                    preWalsh.AsSpan(mb * 16, 16), 4,
                    cpuY2.AsSpan(mb * 16, 16));

            // 4. Clear Y4 coef[0] of every block (encoder convention).
            for (int b = 0; b < y4BlockCount; b++) cpuY4[b * 16] = 0;

            // 5. Quantize: Y4 with y1, Y2 with y2, UV with uv.
            for (int b = 0; b < y4BlockCount; b++)
            {
                int mbIdx = b / 16;
                Vp8ForwardQuantizer.QuantizeBlock(cpuY4.AsSpan(b * 16, 16), y1DcQ[mbIdx], y1AcQ[mbIdx]);
            }
            for (int mb = 0; mb < mbCount; mb++)
                Vp8ForwardQuantizer.QuantizeBlock(cpuY2.AsSpan(mb * 16, 16), y2DcQ[mb], y2AcQ[mb]);
            for (int b = 0; b < uvBlockCount; b++)
            {
                int mbIdx = b / 4;
                Vp8ForwardQuantizer.QuantizeBlock(cpuU.AsSpan(b * 16, 16), uvDcQ[mbIdx], uvAcQ[mbIdx]);
                Vp8ForwardQuantizer.QuantizeBlock(cpuV.AsSpan(b * 16, 16), uvDcQ[mbIdx], uvAcQ[mbIdx]);
            }

            // GPU pipeline run.
            using var dY4Res = acc.Allocate1D<short>(y4Residual.Length);
            using var dURes = acc.Allocate1D<short>(uResidual.Length);
            using var dVRes = acc.Allocate1D<short>(vResidual.Length);
            using var dY4Coefs = acc.Allocate1D<short>(y4Residual.Length);
            using var dY2Coefs = acc.Allocate1D<short>(mbCount * 16);
            using var dUCoefs = acc.Allocate1D<short>(uResidual.Length);
            using var dVCoefs = acc.Allocate1D<short>(vResidual.Length);
            using var dY1Dc = acc.Allocate1D<short>(mbCount);
            using var dY1Ac = acc.Allocate1D<short>(mbCount);
            using var dY2Dc = acc.Allocate1D<short>(mbCount);
            using var dY2Ac = acc.Allocate1D<short>(mbCount);
            using var dUvDc = acc.Allocate1D<short>(mbCount);
            using var dUvAc = acc.Allocate1D<short>(mbCount);

            dY4Res.View.CopyFromCPU(y4Residual);
            dURes.View.CopyFromCPU(uResidual);
            dVRes.View.CopyFromCPU(vResidual);
            dY1Dc.View.CopyFromCPU(y1DcQ);
            dY1Ac.View.CopyFromCPU(y1AcQ);
            dY2Dc.View.CopyFromCPU(y2DcQ);
            dY2Ac.View.CopyFromCPU(y2AcQ);
            dUvDc.View.CopyFromCPU(uvDcQ);
            dUvAc.View.CopyFromCPU(uvAcQ);

            pipeline.Run(
                dY4Res.View, dURes.View, dVRes.View,
                dY4Coefs.View, dY2Coefs.View, dUCoefs.View, dVCoefs.View,
                dY1Dc.View, dY1Ac.View, dY2Dc.View, dY2Ac.View, dUvDc.View, dUvAc.View,
                mbCount);
            await acc.SynchronizeAsync();

            // GPU-side verification (Rule 5a): upload CPU expected,
            // run comparison kernel, read back violation count.
            int y4Mismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dY4Coefs.View, cpuY4, cpuY4.Length);
            int y2Mismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dY2Coefs.View, cpuY2, cpuY2.Length);
            int uMismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dUCoefs.View, cpuU, cpuU.Length);
            int vMismatches = await GpuTestVerifyCodecs.CountShortMismatches(
                acc, dVCoefs.View, cpuV, cpuV.Length);

            Equal(0, y4Mismatches, "Y4 plane");
            Equal(0, y2Mismatches, "Y2 plane");
            Equal(0, uMismatches, "U plane");
            Equal(0, vMismatches, "V plane");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
