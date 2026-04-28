// Cross-backend tests for Vp9FrameSequentialEncodeKernel. The kernel
// composes the per-block GPU helpers (Vp9DcPredictorGpu /
// Vp9ForwardDct{16x16,8x8}Gpu / Vp9ForwardQuantizerGpu /
// Vp9DequantBlockGpu / Vp9Idct{16x16,8x8}Gpu) into the full v1
// keyframe forward-then-inverse pipeline.
//
// CPU oracle: a parallel walker in this test file that calls the
// SAME math primitives' CPU references (Vp9DcPredictor /
// Vp9ForwardDct{16x16,8x8} / Vp9ForwardQuantizer / Vp9Dequantizer /
// Vp9Idct{16x16,8x8}Reference) in the SAME order. Both sides are
// bit-exact mirrors of libvpx, so the GPU output must match the
// CPU output byte-for-byte across all backends.
//
// Test sizes:
//   - 1x1 MB (16x16 frame): top-left corner only, variant=None.
//   - 2x2 MBs (32x32 frame): exercises the per-MB walk and the
//     edge-propagation (MB at (0,1) uses left edge from (0,0)'s
//     recon, MB at (1,0) uses above edge from (0,0)'s recon, MB at
//     (1,1) uses both).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// CPU walker that mirrors Vp9FrameSequentialEncodeKernel exactly.
    /// Composes the same math primitives' CPU references in the same
    /// order; outputs are recon planes + per-block quantized coefs.
    /// </summary>
    private static (byte[] yRecon, byte[] uRecon, byte[] vRecon,
                    short[] yCoefs, short[] uCoefs, short[] vCoefs)
        Vp9SequentialEncodeCpu(
            byte[] yPlane, byte[] uPlane, byte[] vPlane,
            int mbCols, int mbRows,
            int yDcQ, int yAcQ, int uvDcQ, int uvAcQ)
    {
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;
        int mbCount = mbCols * mbRows;

        var yRecon = new byte[mbCount * 256];
        var uRecon = new byte[mbCount * 64];
        var vRecon = new byte[mbCount * 64];
        var yCoefs = new short[mbCount * 256];
        var uCoefs = new short[mbCount * 64];
        var vCoefs = new short[mbCount * 64];

        for (int mbRow = 0; mbRow < mbRows; mbRow++)
        for (int mbCol = 0; mbCol < mbCols; mbCol++)
        {
            int mbIdx = mbRow * mbCols + mbCol;
            int yBase = mbRow * 16 * yStride + mbCol * 16;
            int uvBase = mbRow * 8 * uvStride + mbCol * 8;

            bool topAvail = mbRow > 0;
            bool leftAvail = mbCol > 0;

            EncodeLumaBlockCpu(
                yPlane, yRecon, yCoefs, yBase, yStride, mbIdx,
                yDcQ, yAcQ, topAvail, leftAvail);

            EncodeChromaBlockCpu(
                uPlane, uRecon, uCoefs, uvBase, uvStride, mbIdx,
                uvDcQ, uvAcQ, topAvail, leftAvail);

            EncodeChromaBlockCpu(
                vPlane, vRecon, vCoefs, uvBase, uvStride, mbIdx,
                uvDcQ, uvAcQ, topAvail, leftAvail);
        }

        return (yRecon, uRecon, vRecon, yCoefs, uCoefs, vCoefs);
    }

    private static void EncodeLumaBlockCpu(
        byte[] src, byte[] recon, short[] coefsOut,
        int baseOffset, int stride, int mbIdx,
        int dcQ, int acQ,
        bool topAvail, bool leftAvail)
    {
        // Prediction edges from recon.
        var above = new byte[16];
        var left = new byte[16];
        if (topAvail)
            for (int i = 0; i < 16; i++) above[i] = recon[baseOffset - stride + i];
        if (leftAvail)
            for (int r = 0; r < 16; r++) left[r] = recon[baseOffset + r * stride - 1];

        // DC predict into recon at the MB's top-left.
        var predBlock = new byte[16 * 16];
        if (topAvail && leftAvail)
            Vp9DcPredictor.DcPredict(above, left, predBlock, 16, 16);
        else if (topAvail)
            Vp9DcPredictor.DcPredictTop(above, predBlock, 16, 16);
        else if (leftAvail)
            Vp9DcPredictor.DcPredictLeft(left, predBlock, 16, 16);
        else
            Vp9DcPredictor.DcPredict128(predBlock, 16, 16);

        // Copy pred into recon and compute residual src - pred.
        var residual = new short[256];
        for (int r = 0; r < 16; r++)
        {
            for (int c = 0; c < 16; c++)
            {
                byte p = predBlock[r * 16 + c];
                recon[baseOffset + r * stride + c] = p;
                residual[r * 16 + c] = (short)(src[baseOffset + r * stride + c] - p);
            }
        }

        // FDCT 16x16.
        var coefs = new int[256];
        Vp9ForwardDct16x16.Transform(residual, rowStrideShorts: 16, coefs);

        // Forward quantize in place.
        Vp9ForwardQuantizer.QuantizeBlock(coefs, dcQ, acQ);

        // Save quantized coefs (cast to short) and stage for inverse.
        var coefsShort = new short[256];
        for (int i = 0; i < 256; i++)
        {
            short q = (short)coefs[i];
            coefsOut[mbIdx * 256 + i] = q;
            coefsShort[i] = q;
        }

        // Dequant in place.
        Vp9Dequantizer.DequantizeInPlace(coefsShort, new Vp9PlaneQuantizer((short)dcQ, (short)acQ));

        // IDCT + add to recon in place.
        Vp9Idct16x16Reference.Idct16x16_256_Add(coefsShort,
            recon.AsSpan(baseOffset), stride);
    }

    private static void EncodeChromaBlockCpu(
        byte[] src, byte[] recon, short[] coefsOut,
        int baseOffset, int stride, int mbIdx,
        int dcQ, int acQ,
        bool topAvail, bool leftAvail)
    {
        var above = new byte[8];
        var left = new byte[8];
        if (topAvail)
            for (int i = 0; i < 8; i++) above[i] = recon[baseOffset - stride + i];
        if (leftAvail)
            for (int r = 0; r < 8; r++) left[r] = recon[baseOffset + r * stride - 1];

        var predBlock = new byte[8 * 8];
        if (topAvail && leftAvail)
            Vp9DcPredictor.DcPredict(above, left, predBlock, 8, 8);
        else if (topAvail)
            Vp9DcPredictor.DcPredictTop(above, predBlock, 8, 8);
        else if (leftAvail)
            Vp9DcPredictor.DcPredictLeft(left, predBlock, 8, 8);
        else
            Vp9DcPredictor.DcPredict128(predBlock, 8, 8);

        var residual = new short[64];
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                byte p = predBlock[r * 8 + c];
                recon[baseOffset + r * stride + c] = p;
                residual[r * 8 + c] = (short)(src[baseOffset + r * stride + c] - p);
            }
        }

        var coefs = new int[64];
        Vp9ForwardDct8x8.Transform(residual, rowStrideShorts: 8, coefs);
        Vp9ForwardQuantizer.QuantizeBlock(coefs, dcQ, acQ);

        var coefsShort = new short[64];
        for (int i = 0; i < 64; i++)
        {
            short q = (short)coefs[i];
            coefsOut[mbIdx * 64 + i] = q;
            coefsShort[i] = q;
        }

        Vp9Dequantizer.DequantizeInPlace(coefsShort, new Vp9PlaneQuantizer((short)dcQ, (short)acQ));
        Vp9Idct8x8Reference.Idct8x8_64_Add(coefsShort,
            recon.AsSpan(baseOffset), stride);
    }

    private static async Task<(byte[] yRecon, byte[] uRecon, byte[] vRecon,
                               short[] yCoefs, short[] uCoefs, short[] vCoefs)>
        Vp9SequentialEncodeGpuAsync(
            Accelerator acc,
            byte[] yPlane, byte[] uPlane, byte[] vPlane,
            int mbCols, int mbRows,
            int yDcQ, int yAcQ, int uvDcQ, int uvAcQ)
    {
        int mbCount = mbCols * mbRows;
        int yLen = mbCount * 256;
        int uvLen = mbCount * 64;

        using var dY = acc.Allocate1D<byte>(yLen);
        using var dU = acc.Allocate1D<byte>(uvLen);
        using var dV = acc.Allocate1D<byte>(uvLen);
        using var dYRecon = acc.Allocate1D<byte>(yLen);
        using var dURecon = acc.Allocate1D<byte>(uvLen);
        using var dVRecon = acc.Allocate1D<byte>(uvLen);
        using var dYCoefs = acc.Allocate1D<short>(yLen);
        using var dUCoefs = acc.Allocate1D<short>(uvLen);
        using var dVCoefs = acc.Allocate1D<short>(uvLen);
        using var dDequant = acc.Allocate1D<int>(4);

        dY.View.CopyFromCPU(yPlane);
        dU.View.CopyFromCPU(uPlane);
        dV.View.CopyFromCPU(vPlane);
        dYRecon.View.CopyFromCPU(new byte[yLen]);
        dURecon.View.CopyFromCPU(new byte[uvLen]);
        dVRecon.View.CopyFromCPU(new byte[uvLen]);
        dYCoefs.View.CopyFromCPU(new short[yLen]);
        dUCoefs.View.CopyFromCPU(new short[uvLen]);
        dVCoefs.View.CopyFromCPU(new short[uvLen]);
        dDequant.View.CopyFromCPU(new[] { yDcQ, yAcQ, uvDcQ, uvAcQ });

        using var kernel = new Vp9FrameSequentialEncodeKernel(acc);
        kernel.Run(
            dY.View, dU.View, dV.View,
            dYRecon.View, dURecon.View, dVRecon.View,
            dYCoefs.View, dUCoefs.View, dVCoefs.View,
            dDequant.View, mbCols, mbRows);
        await acc.SynchronizeAsync();

        var yRecon = await dYRecon.CopyToHostAsync();
        var uRecon = await dURecon.CopyToHostAsync();
        var vRecon = await dVRecon.CopyToHostAsync();
        var yCoefs = await dYCoefs.CopyToHostAsync();
        var uCoefs = await dUCoefs.CopyToHostAsync();
        var vCoefs = await dVCoefs.CopyToHostAsync();
        return (yRecon, uRecon, vRecon, yCoefs, uCoefs, vCoefs);
    }

    private static async Task AssertVp9SequentialEncodeGpuMatchesCpuAsync(
        Accelerator acc,
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int mbCols, int mbRows,
        int yDcQ, int yAcQ, int uvDcQ, int uvAcQ)
    {
        var (cpuYRecon, cpuURecon, cpuVRecon, cpuYCoefs, cpuUCoefs, cpuVCoefs) =
            Vp9SequentialEncodeCpu(yPlane, uPlane, vPlane, mbCols, mbRows,
                yDcQ, yAcQ, uvDcQ, uvAcQ);
        var (gpuYRecon, gpuURecon, gpuVRecon, gpuYCoefs, gpuUCoefs, gpuVCoefs) =
            await Vp9SequentialEncodeGpuAsync(acc, yPlane, uPlane, vPlane, mbCols, mbRows,
                yDcQ, yAcQ, uvDcQ, uvAcQ);

        for (int i = 0; i < cpuYRecon.Length; i++)
            if (cpuYRecon[i] != gpuYRecon[i])
                throw new Exception($"yRecon mismatch at {i}: cpu={cpuYRecon[i]} gpu={gpuYRecon[i]}");
        for (int i = 0; i < cpuURecon.Length; i++)
            if (cpuURecon[i] != gpuURecon[i])
                throw new Exception($"uRecon mismatch at {i}: cpu={cpuURecon[i]} gpu={gpuURecon[i]}");
        for (int i = 0; i < cpuVRecon.Length; i++)
            if (cpuVRecon[i] != gpuVRecon[i])
                throw new Exception($"vRecon mismatch at {i}: cpu={cpuVRecon[i]} gpu={gpuVRecon[i]}");
        for (int i = 0; i < cpuYCoefs.Length; i++)
            if (cpuYCoefs[i] != gpuYCoefs[i])
                throw new Exception($"yCoefs mismatch at {i}: cpu={cpuYCoefs[i]} gpu={gpuYCoefs[i]}");
        for (int i = 0; i < cpuUCoefs.Length; i++)
            if (cpuUCoefs[i] != gpuUCoefs[i])
                throw new Exception($"uCoefs mismatch at {i}: cpu={cpuUCoefs[i]} gpu={gpuUCoefs[i]}");
        for (int i = 0; i < cpuVCoefs.Length; i++)
            if (cpuVCoefs[i] != gpuVCoefs[i])
                throw new Exception($"vCoefs mismatch at {i}: cpu={cpuVCoefs[i]} gpu={gpuVCoefs[i]}");
    }

    [TestMethod]
    public async Task Vp9FrameSequentialEncodeKernel_SingleMB_FlatGray_MatchesCpu()
    {
        // 1x1 MB frame: 16x16 luma + 8x8 U + 8x8 V. Top-left corner -
        // variant = None for every plane.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var y = new byte[256];
            var u = new byte[64];
            var v = new byte[64];
            for (int i = 0; i < 256; i++) y[i] = 128;
            for (int i = 0; i < 64; i++) { u[i] = 128; v[i] = 128; }

            // baseQ = 30 mirrors the v1 keyframe encoder default.
            int baseQ = 30;
            int yDcQ = Vp9Dequantizer.DcQuant(baseQ, 0);
            int yAcQ = Vp9Dequantizer.AcQuant(baseQ, 0);
            int uvDcQ = yDcQ;
            int uvAcQ = yAcQ;

            await AssertVp9SequentialEncodeGpuMatchesCpuAsync(
                acc, y, u, v, mbCols: 1, mbRows: 1, yDcQ, yAcQ, uvDcQ, uvAcQ);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameSequentialEncodeKernel_SingleMB_RandomContent_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0xF9E12001u));
            var y = new byte[256];
            var u = new byte[64];
            var v = new byte[64];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            int baseQ = 30;
            int yDcQ = Vp9Dequantizer.DcQuant(baseQ, 0);
            int yAcQ = Vp9Dequantizer.AcQuant(baseQ, 0);
            int uvDcQ = yDcQ;
            int uvAcQ = yAcQ;

            await AssertVp9SequentialEncodeGpuMatchesCpuAsync(
                acc, y, u, v, mbCols: 1, mbRows: 1, yDcQ, yAcQ, uvDcQ, uvAcQ);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameSequentialEncodeKernel_TwoByTwoMBs_MatchesCpu()
    {
        // 2x2 MB frame (32x32 luma): exercises the per-MB walk plus
        // edge propagation - MB(0,1) uses left edge from MB(0,0)'s
        // recon, MB(1,0) uses above edge from MB(0,0)'s recon, MB(1,1)
        // uses both.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int mbCols = 2, mbRows = 2;
            int yLen = mbCols * mbRows * 256;
            int uvLen = mbCols * mbRows * 64;
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            var rng = new Random(unchecked((int)0xF9E12222u));
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            int baseQ = 30;
            int yDcQ = Vp9Dequantizer.DcQuant(baseQ, 0);
            int yAcQ = Vp9Dequantizer.AcQuant(baseQ, 0);
            int uvDcQ = yDcQ;
            int uvAcQ = yAcQ;

            await AssertVp9SequentialEncodeGpuMatchesCpuAsync(
                acc, y, u, v, mbCols, mbRows, yDcQ, yAcQ, uvDcQ, uvAcQ);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameSequentialEncodeKernel_QuantizerSweep_MatchesCpu()
    {
        // Sweep across a few baseQ values to flag any quantizer-
        // dependent drift.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] baseQs = { 1, 30, 64, 128, 200 };
            var rng = new Random(unchecked((int)0xF9E10017u));
            var y = new byte[256];
            var u = new byte[64];
            var v = new byte[64];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            foreach (int baseQ in baseQs)
            {
                int yDcQ = Vp9Dequantizer.DcQuant(baseQ, 0);
                int yAcQ = Vp9Dequantizer.AcQuant(baseQ, 0);
                int uvDcQ = yDcQ;
                int uvAcQ = yAcQ;
                await AssertVp9SequentialEncodeGpuMatchesCpuAsync(
                    acc, y, u, v, mbCols: 1, mbRows: 1, yDcQ, yAcQ, uvDcQ, uvAcQ);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
