// Tests for Vp8FrameEntropyKernel - GPU-resident frame-level entropy
// coding. Compares the kernel's bool-stream output (mode partition +
// token partition) against an equivalent CPU encoding of the same
// frame data, both starting from a fresh bool encoder (no frame
// header in either; that's the caller's responsibility).
//
// Bit-exact agreement is mandatory: any drift between GPU and CPU
// would mean the encoded bitstream wouldn't decode correctly.

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
    public async Task Vp8FrameEntropyKernel_AllDcMode_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8FrameEntropyKernel(acc);
            const int mbCols = 4;
            const int mbRows = 3;
            int mbCount = mbCols * mbRows;

            // Realistic post-Q coef distribution.
            var rng = new Random(unchecked((int)0xE0DE0DE0));
            var y4Coefs = new short[mbCount * 16 * 16];
            var y2Coefs = new short[mbCount * 16];
            var uCoefs = new short[mbCount * 4 * 16];
            var vCoefs = new short[mbCount * 4 * 16];
            for (int i = 0; i < y4Coefs.Length; i++)
                y4Coefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-30, 30) : (short)0;
            for (int i = 0; i < y2Coefs.Length; i++)
                y2Coefs[i] = (rng.Next(4) == 0) ? (short)rng.Next(-200, 200) : (short)0;
            for (int i = 0; i < uCoefs.Length; i++)
                uCoefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-30, 30) : (short)0;
            for (int i = 0; i < vCoefs.Length; i++)
                vCoefs[i] = (rng.Next(8) == 0) ? (short)rng.Next(-30, 30) : (short)0;
            // Y4 coef[0] of every block is zero (encoder convention - Y2 carries DC).
            for (int b = 0; b < mbCount * 16; b++) y4Coefs[b * 16] = 0;

            // Coef probs (4 block types * 264 bytes flat).
            var coefProbsByType = new byte[4 * 264];
            var defaults = Vp8DefaultCoefProbs.DefaultProbs;
            for (int t = 0; t < 4; t++)
                for (int band = 0; band < 8; band++)
                    for (int c = 0; c < 3; c++)
                        for (int n = 0; n < 11; n++)
                            coefProbsByType[t * 264 + band * 33 + c * 11 + n] = defaults[t, band, c, n];

            var constsExtended = Vp8FrameEntropyKernel.BuildExtendedConstsBuffer();

            // CPU reference: replicate the per-MB encode pattern, no header.
            var cpuP0 = new Vp8BoolEncoder();
            var cpuTp = new Vp8BoolEncoder();

            // Materialize 3D probs slices for the CPU encoder.
            var probsByType = new byte[4][,,];
            for (int t = 0; t < 4; t++)
            {
                var slice = new byte[8, 3, 11];
                for (int band = 0; band < 8; band++)
                    for (int c = 0; c < 3; c++)
                        for (int n = 0; n < 11; n++)
                            slice[band, c, n] = defaults[t, band, c, n];
                probsByType[t] = slice;
            }

            var cpuContexts = new Vp8EntropyContexts(mbCols);
            var kfYProbs = Vp8ModeTrees.DefaultKfYModeProb;
            var kfUvProbs = Vp8ModeTrees.DefaultKfUvModeProb;

            for (int mbRow = 0; mbRow < mbRows; mbRow++)
            {
                cpuContexts.ClearLeft();
                for (int mbCol = 0; mbCol < mbCols; mbCol++)
                {
                    int mbIdx = mbRow * mbCols + mbCol;
                    var aboveCtx = cpuContexts.GetAbove(mbCol);

                    // Y mode = DC (path: bit 1, 0, 0).
                    cpuP0.EncodeBool(1, kfYProbs[0]);
                    cpuP0.EncodeBool(0, kfYProbs[1]);
                    cpuP0.EncodeBool(0, kfYProbs[2]);
                    // UV mode = DC (path: bit 0).
                    cpuP0.EncodeBool(0, kfUvProbs[0]);

                    // Y2.
                    int y2Ctx = aboveCtx[Vp8EntropyContexts.Plane.Y2Slot]
                        + cpuContexts.Left[Vp8EntropyContexts.Plane.Y2Slot];
                    int y2Eob = Vp8CoefBlockEncoder.Encode(
                        cpuTp, probsByType[1], y2Ctx, 0,
                        y2Coefs.AsSpan(mbIdx * 16, 16));
                    byte y2HasCoef = (byte)(y2Eob > 0 ? 1 : 0);
                    aboveCtx[Vp8EntropyContexts.Plane.Y2Slot] = y2HasCoef;
                    cpuContexts.Left[Vp8EntropyContexts.Plane.Y2Slot] = y2HasCoef;

                    // Y4 16 blocks.
                    for (int by = 0; by < 4; by++)
                    {
                        for (int bx = 0; bx < 4; bx++)
                        {
                            int blockIdxInMb = by * 4 + bx;
                            int aboveSlot = Vp8EntropyContexts.Plane.YBase + bx;
                            int leftSlot = Vp8EntropyContexts.Plane.YBase + by;
                            int c = aboveCtx[aboveSlot] + cpuContexts.Left[leftSlot];
                            int eob = Vp8CoefBlockEncoder.Encode(
                                cpuTp, probsByType[0], c, 1,
                                y4Coefs.AsSpan(mbIdx * 256 + blockIdxInMb * 16, 16));
                            byte h = (byte)(eob > 0 ? 1 : 0);
                            aboveCtx[aboveSlot] = h;
                            cpuContexts.Left[leftSlot] = h;
                        }
                    }
                    // U 4 blocks.
                    for (int by = 0; by < 2; by++)
                    {
                        for (int bx = 0; bx < 2; bx++)
                        {
                            int blockIdx = by * 2 + bx;
                            int aboveSlot = Vp8EntropyContexts.Plane.UBase + bx;
                            int leftSlot = Vp8EntropyContexts.Plane.UBase + by;
                            int c = aboveCtx[aboveSlot] + cpuContexts.Left[leftSlot];
                            int eob = Vp8CoefBlockEncoder.Encode(
                                cpuTp, probsByType[2], c, 0,
                                uCoefs.AsSpan(mbIdx * 64 + blockIdx * 16, 16));
                            byte h = (byte)(eob > 0 ? 1 : 0);
                            aboveCtx[aboveSlot] = h;
                            cpuContexts.Left[leftSlot] = h;
                        }
                    }
                    // V 4 blocks.
                    for (int by = 0; by < 2; by++)
                    {
                        for (int bx = 0; bx < 2; bx++)
                        {
                            int blockIdx = by * 2 + bx;
                            int aboveSlot = Vp8EntropyContexts.Plane.VBase + bx;
                            int leftSlot = Vp8EntropyContexts.Plane.VBase + by;
                            int c = aboveCtx[aboveSlot] + cpuContexts.Left[leftSlot];
                            int eob = Vp8CoefBlockEncoder.Encode(
                                cpuTp, probsByType[2], c, 0,
                                vCoefs.AsSpan(mbIdx * 64 + blockIdx * 16, 16));
                            byte h = (byte)(eob > 0 ? 1 : 0);
                            aboveCtx[aboveSlot] = h;
                            cpuContexts.Left[leftSlot] = h;
                        }
                    }
                }
            }

            byte[] cpuP0Bytes = cpuP0.Stop();
            byte[] cpuTpBytes = cpuTp.Stop();

            // GPU encode.
            const int p0Stride = 32 * 1024;
            const int tp0Stride = 256 * 1024;
            using var dY4 = acc.Allocate1D<short>(y4Coefs.Length);
            using var dY2 = acc.Allocate1D<short>(y2Coefs.Length);
            using var dU = acc.Allocate1D<short>(uCoefs.Length);
            using var dV = acc.Allocate1D<short>(vCoefs.Length);
            using var dProbs = acc.Allocate1D<byte>(coefProbsByType.Length);
            using var dConsts = acc.Allocate1D<byte>(constsExtended.Length);
            using var dP0 = acc.Allocate1D<byte>(p0Stride);
            using var dTp = acc.Allocate1D<byte>(tp0Stride);
            using var dLens = acc.Allocate1D<long>(2);
            using var dAbove = acc.Allocate1D<byte>(mbCols * 9);
            dY4.View.CopyFromCPU(y4Coefs);
            dY2.View.CopyFromCPU(y2Coefs);
            dU.View.CopyFromCPU(uCoefs);
            dV.View.CopyFromCPU(vCoefs);
            dProbs.View.CopyFromCPU(coefProbsByType);
            dConsts.View.CopyFromCPU(constsExtended);
            dP0.View.MemSetToZero();
            dTp.View.MemSetToZero();
            dAbove.View.MemSetToZero();

            kernel.Run(dY4.View, dY2.View, dU.View, dV.View,
                dProbs.View, dConsts.View, dP0.View, dTp.View, dLens.View,
                dAbove.View, mbCols, mbRows);
            await acc.SynchronizeAsync();

            var lens = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dLens);
            var p0Back = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dP0);
            var tpBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dTp);

            long gpuP0Len = lens[0];
            long gpuTpLen = lens[1];
            Equal((long)cpuP0Bytes.Length, gpuP0Len, "partition0 len");
            Equal((long)cpuTpBytes.Length, gpuTpLen, "tokenP0 len");
            int p0Mismatches = 0, tpMismatches = 0;
            int firstP0 = -1, firstTp = -1;
            for (int i = 0; i < cpuP0Bytes.Length; i++)
                if (cpuP0Bytes[i] != p0Back[i]) { if (firstP0 < 0) firstP0 = i; p0Mismatches++; }
            for (int i = 0; i < cpuTpBytes.Length; i++)
                if (cpuTpBytes[i] != tpBack[i]) { if (firstTp < 0) firstTp = i; tpMismatches++; }
            Equal(0, p0Mismatches, $"partition0 first bad byte i={firstP0}");
            Equal(0, tpMismatches, $"tokenP0 first bad byte i={firstTp}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
