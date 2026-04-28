// Tests for Vp8FrameReconstructGpu - inverse pipeline driver.
// Verifies the full dequant + inv-Walsh + Y2-DC-inject + IDCT +
// predict-add chain runs on the GPU bit-exactly versus the same
// chain executed on the CPU step by step.

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
    public async Task Vp8FrameReconstructGpu_RandomMacroblocks_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var pipeline = new Vp8FrameReconstructGpu(acc);
            const int mbCount = 8;
            int y4BlockCount = mbCount * 16;
            int uvBlockCount = mbCount * 4;

            var rng = new Random(unchecked((int)0xDEC0FFEE));
            var y4Coefs = new short[y4BlockCount * 16];
            var y2Coefs = new short[mbCount * 16];
            var uCoefs = new short[uvBlockCount * 16];
            var vCoefs = new short[uvBlockCount * 16];
            var y4Pred = new byte[y4BlockCount * 16];
            var uPred = new byte[uvBlockCount * 16];
            var vPred = new byte[uvBlockCount * 16];

            // Fill quantized coefs with values an encoder would produce
            // post-quantization (small ints, mostly zero with some
            // non-zero spikes - typical post-Q distribution).
            for (int i = 0; i < y4Coefs.Length; i++)
                y4Coefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-30, 30) : (short)0;
            for (int i = 0; i < y2Coefs.Length; i++)
                y2Coefs[i] = (rng.Next(4) == 0) ? (short)rng.Next(-200, 200) : (short)0;
            for (int i = 0; i < uCoefs.Length; i++)
                uCoefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-30, 30) : (short)0;
            for (int i = 0; i < vCoefs.Length; i++)
                vCoefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-30, 30) : (short)0;

            for (int i = 0; i < y4Pred.Length; i++) y4Pred[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < uPred.Length; i++) uPred[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < vPred.Length; i++) vPred[i] = (byte)rng.Next(0, 256);

            // Per-MB dequantizers.
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

            // Compute y2HasAc per MB (1 if any AC slot non-zero).
            var y2HasAc = new byte[mbCount];
            for (int mb = 0; mb < mbCount; mb++)
            {
                byte hasAc = 0;
                for (int i = 1; i < 16; i++) if (y2Coefs[mb * 16 + i] != 0) { hasAc = 1; break; }
                y2HasAc[mb] = hasAc;
            }

            // CPU reference: replicate the GPU pipeline.
            var cpuY2Dq = new short[mbCount * 16];
            var cpuY2Inv = new short[mbCount * 16];
            for (int mb = 0; mb < mbCount; mb++)
            {
                cpuY2Dq[mb * 16] = (short)(y2Coefs[mb * 16] * y2DcQ[mb]);
                for (int i = 1; i < 16; i++)
                    cpuY2Dq[mb * 16 + i] = (short)(y2Coefs[mb * 16 + i] * y2AcQ[mb]);
                if (y2HasAc[mb] != 0)
                    Vp8InverseTransform.ShortInvWalsh4x4(
                        cpuY2Dq.AsSpan(mb * 16, 16),
                        cpuY2Inv.AsSpan(mb * 16, 16));
                else
                {
                    short v = (short)((cpuY2Dq[mb * 16] + 3) >> 3);
                    for (int i = 0; i < 16; i++) cpuY2Inv[mb * 16 + i] = v;
                }
            }

            // Y4 dequant + Y2 DC inject + IDCT + predict-add per Y4 block.
            var cpuY4Recon = new byte[y4BlockCount * 16];
            for (int b = 0; b < y4BlockCount; b++)
            {
                int mb = b / 16, slot = b % 16;
                Span<short> dq = stackalloc short[16];
                dq[0] = (short)(y4Coefs[b * 16] * y1DcQ[mb]);
                for (int i = 1; i < 16; i++) dq[i] = (short)(y4Coefs[b * 16 + i] * y1AcQ[mb]);
                dq[0] = cpuY2Inv[mb * 16 + slot]; // Y2 inverse DC injection
                Vp8InverseTransform.ShortIdct4x4Llm(
                    dq,
                    y4Pred.AsSpan(b * 16, 16), 4,
                    cpuY4Recon.AsSpan(b * 16, 16), 4);
            }

            // UV dequant + IDCT + predict-add.
            var cpuURecon = new byte[uvBlockCount * 16];
            var cpuVRecon = new byte[uvBlockCount * 16];
            for (int b = 0; b < uvBlockCount; b++)
            {
                int mb = b / 4;
                Span<short> dqU = stackalloc short[16];
                Span<short> dqV = stackalloc short[16];
                dqU[0] = (short)(uCoefs[b * 16] * uvDcQ[mb]);
                dqV[0] = (short)(vCoefs[b * 16] * uvDcQ[mb]);
                for (int i = 1; i < 16; i++)
                {
                    dqU[i] = (short)(uCoefs[b * 16 + i] * uvAcQ[mb]);
                    dqV[i] = (short)(vCoefs[b * 16 + i] * uvAcQ[mb]);
                }
                Vp8InverseTransform.ShortIdct4x4Llm(
                    dqU, uPred.AsSpan(b * 16, 16), 4,
                    cpuURecon.AsSpan(b * 16, 16), 4);
                Vp8InverseTransform.ShortIdct4x4Llm(
                    dqV, vPred.AsSpan(b * 16, 16), 4,
                    cpuVRecon.AsSpan(b * 16, 16), 4);
            }

            // GPU pipeline.
            using var dY4 = acc.Allocate1D<short>(y4Coefs.Length);
            using var dY2 = acc.Allocate1D<short>(y2Coefs.Length);
            using var dU = acc.Allocate1D<short>(uCoefs.Length);
            using var dV = acc.Allocate1D<short>(vCoefs.Length);
            using var dY4Pred = acc.Allocate1D<byte>(y4Pred.Length);
            using var dUPred = acc.Allocate1D<byte>(uPred.Length);
            using var dVPred = acc.Allocate1D<byte>(vPred.Length);
            using var dY4Recon = acc.Allocate1D<byte>(y4Pred.Length);
            using var dURecon = acc.Allocate1D<byte>(uPred.Length);
            using var dVRecon = acc.Allocate1D<byte>(vPred.Length);
            using var dY1Dc = acc.Allocate1D<short>(mbCount);
            using var dY1Ac = acc.Allocate1D<short>(mbCount);
            using var dY2Dc = acc.Allocate1D<short>(mbCount);
            using var dY2Ac = acc.Allocate1D<short>(mbCount);
            using var dUvDc = acc.Allocate1D<short>(mbCount);
            using var dUvAc = acc.Allocate1D<short>(mbCount);
            using var dY2HasAc = acc.Allocate1D<byte>(mbCount);

            dY4.View.CopyFromCPU(y4Coefs);
            dY2.View.CopyFromCPU(y2Coefs);
            dU.View.CopyFromCPU(uCoefs);
            dV.View.CopyFromCPU(vCoefs);
            dY4Pred.View.CopyFromCPU(y4Pred);
            dUPred.View.CopyFromCPU(uPred);
            dVPred.View.CopyFromCPU(vPred);
            dY1Dc.View.CopyFromCPU(y1DcQ);
            dY1Ac.View.CopyFromCPU(y1AcQ);
            dY2Dc.View.CopyFromCPU(y2DcQ);
            dY2Ac.View.CopyFromCPU(y2AcQ);
            dUvDc.View.CopyFromCPU(uvDcQ);
            dUvAc.View.CopyFromCPU(uvAcQ);
            dY2HasAc.View.CopyFromCPU(y2HasAc);

            pipeline.Run(
                dY4.View, dY2.View, dU.View, dV.View,
                dY4Pred.View, dUPred.View, dVPred.View,
                dY4Recon.View, dURecon.View, dVRecon.View,
                dY1Dc.View, dY1Ac.View, dY2Dc.View, dY2Ac.View, dUvDc.View, dUvAc.View,
                dY2HasAc.View, mbCount);
            await acc.SynchronizeAsync();

            // GPU-side verification.
            int y4Mismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dY4Recon.View, cpuY4Recon, cpuY4Recon.Length);
            int uMismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dURecon.View, cpuURecon, cpuURecon.Length);
            int vMismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dVRecon.View, cpuVRecon, cpuVRecon.Length);

            Equal(0, y4Mismatches, "Y4 recon");
            Equal(0, uMismatches, "U recon");
            Equal(0, vMismatches, "V recon");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
