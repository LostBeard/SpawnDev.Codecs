// Tests for Vp8FramePredictorGpu - frame-level intra predictor builder.
// Verifies the GPU output for every MB matches what the CPU
// reference (Vp8IntraPredictor16x16 + Vp8IntraPredictor8x8) would
// produce when given the same neighbour samples.

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
    public async Task Vp8FramePredictorGpu_AllDcMode_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var pipeline = new Vp8FramePredictorGpu(acc);
            const int mbCols = 4, mbRows = 3;
            int mbCount = mbCols * mbRows;
            int yStride = mbCols * 16;
            int uvStride = mbCols * 8;
            int yPlaneSize = yStride * mbRows * 16;
            int uvPlaneSize = uvStride * mbRows * 8;

            var rng = new Random(0xFA1);
            var yRecon = new byte[yPlaneSize];
            var uRecon = new byte[uvPlaneSize];
            var vRecon = new byte[uvPlaneSize];
            for (int i = 0; i < yRecon.Length; i++) yRecon[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < uRecon.Length; i++) uRecon[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < vRecon.Length; i++) vRecon[i] = (byte)rng.Next(0, 256);

            // All MBs use DC_PRED. haveAbove/haveLeft from MB position.
            var modesY = new byte[mbCount];
            var modesUv = new byte[mbCount];
            for (int mb = 0; mb < mbCount; mb++)
            {
                int mbRow = mb / mbCols, mbCol = mb % mbCols;
                bool haveAbove = mbRow > 0;
                bool haveLeft = mbCol > 0;
                byte mf = (byte)(0 | (haveAbove ? 0x10 : 0) | (haveLeft ? 0x20 : 0));
                modesY[mb] = mf;
                modesUv[mb] = mf;
            }

            // CPU reference: replicate the GPU neighbour gather + intra predict.
            var cpuYPred = new byte[mbCount * 256];
            var cpuUPred = new byte[mbCount * 64];
            var cpuVPred = new byte[mbCount * 64];
            for (int mb = 0; mb < mbCount; mb++)
            {
                int mbRow = mb / mbCols, mbCol = mb % mbCols;
                bool haveAbove = (modesY[mb] & 0x10) != 0;
                bool haveLeft = (modesY[mb] & 0x20) != 0;

                // Gather Y neighbours.
                var yAbove = new byte[16];
                var yLeft = new byte[16];
                byte yTopLeft;
                if (mbRow == 0)
                {
                    for (int c = 0; c < 16; c++) yAbove[c] = 127;
                    yTopLeft = 128;
                }
                else
                {
                    int fRow = (mbRow * 16 - 1) * yStride + mbCol * 16;
                    for (int c = 0; c < 16; c++) yAbove[c] = yRecon[fRow + c];
                    yTopLeft = (mbCol == 0) ? (byte)129 : yRecon[fRow - 1];
                }
                if (mbCol == 0) for (int r = 0; r < 16; r++) yLeft[r] = 129;
                else
                {
                    int fCol = mbCol * 16 - 1;
                    for (int r = 0; r < 16; r++) yLeft[r] = yRecon[(mbRow * 16 + r) * yStride + fCol];
                }
                Vp8IntraPredictor16x16.Predict(
                    Vp8IntraMode16x16.DcPred, yAbove, yLeft, yTopLeft, haveAbove, haveLeft,
                    cpuYPred.AsSpan(mb * 256, 256), 16);

                // Gather UV neighbours (U).
                var uAbove = new byte[8];
                var uLeft = new byte[8];
                byte uTopLeft;
                if (mbRow == 0)
                {
                    for (int c = 0; c < 8; c++) uAbove[c] = 127;
                    uTopLeft = 128;
                }
                else
                {
                    int fRow = (mbRow * 8 - 1) * uvStride + mbCol * 8;
                    for (int c = 0; c < 8; c++) uAbove[c] = uRecon[fRow + c];
                    uTopLeft = (mbCol == 0) ? (byte)129 : uRecon[fRow - 1];
                }
                if (mbCol == 0) for (int r = 0; r < 8; r++) uLeft[r] = 129;
                else
                {
                    int fCol = mbCol * 8 - 1;
                    for (int r = 0; r < 8; r++) uLeft[r] = uRecon[(mbRow * 8 + r) * uvStride + fCol];
                }
                Vp8IntraPredictor8x8.Predict(
                    Vp8IntraMode16x16.DcPred, uAbove, uLeft, uTopLeft, haveAbove, haveLeft,
                    cpuUPred.AsSpan(mb * 64, 64), 8);

                // Same for V.
                var vAbove = new byte[8];
                var vLeft = new byte[8];
                byte vTopLeft;
                if (mbRow == 0)
                {
                    for (int c = 0; c < 8; c++) vAbove[c] = 127;
                    vTopLeft = 128;
                }
                else
                {
                    int fRow = (mbRow * 8 - 1) * uvStride + mbCol * 8;
                    for (int c = 0; c < 8; c++) vAbove[c] = vRecon[fRow + c];
                    vTopLeft = (mbCol == 0) ? (byte)129 : vRecon[fRow - 1];
                }
                if (mbCol == 0) for (int r = 0; r < 8; r++) vLeft[r] = 129;
                else
                {
                    int fCol = mbCol * 8 - 1;
                    for (int r = 0; r < 8; r++) vLeft[r] = vRecon[(mbRow * 8 + r) * uvStride + fCol];
                }
                Vp8IntraPredictor8x8.Predict(
                    Vp8IntraMode16x16.DcPred, vAbove, vLeft, vTopLeft, haveAbove, haveLeft,
                    cpuVPred.AsSpan(mb * 64, 64), 8);
            }

            // GPU pipeline.
            using var dY = acc.Allocate1D<byte>(yPlaneSize);
            using var dU = acc.Allocate1D<byte>(uvPlaneSize);
            using var dV = acc.Allocate1D<byte>(uvPlaneSize);
            using var dModesY = acc.Allocate1D<byte>(mbCount);
            using var dModesUv = acc.Allocate1D<byte>(mbCount);
            using var dYPred = acc.Allocate1D<byte>(mbCount * 256);
            using var dUPred = acc.Allocate1D<byte>(mbCount * 64);
            using var dVPred = acc.Allocate1D<byte>(mbCount * 64);
            dY.View.CopyFromCPU(yRecon);
            dU.View.CopyFromCPU(uRecon);
            dV.View.CopyFromCPU(vRecon);
            dModesY.View.CopyFromCPU(modesY);
            dModesUv.View.CopyFromCPU(modesUv);

            pipeline.Run(
                dY.View, dU.View, dV.View,
                dModesY.View, dModesUv.View,
                dYPred.View, dUPred.View, dVPred.View,
                mbCols, mbRows, yStride, uvStride);
            await acc.SynchronizeAsync();

            int yMis = await GpuTestVerifyCodecs.CountByteMismatches(acc, dYPred.View, cpuYPred, cpuYPred.Length);
            int uMis = await GpuTestVerifyCodecs.CountByteMismatches(acc, dUPred.View, cpuUPred, cpuUPred.Length);
            int vMis = await GpuTestVerifyCodecs.CountByteMismatches(acc, dVPred.View, cpuVPred, cpuVPred.Length);
            Equal(0, yMis, "Y predictor");
            Equal(0, uMis, "U predictor");
            Equal(0, vMis, "V predictor");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
