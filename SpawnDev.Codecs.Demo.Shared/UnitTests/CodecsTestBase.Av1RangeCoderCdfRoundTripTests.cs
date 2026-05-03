// Minimal DecodeCdfQ15 round-trip test. The existing
// Av1RangeCoderGpuTests only exercise DecodeBoolQ15; this verifies
// DecodeCdfQ15 (used by Av1CoefDecoderGpu) works correctly on every
// backend - especially OpenCL, where the per-block decoder showed
// failures that the range-coder round-trip didn't catch.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1RangeCoderGpu_CdfQ15_RoundTrip_AllBackends()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Build a tiny 4-symbol ICDF + a sequence of 16 syms to encode/decode.
            int nsyms = 4;
            int rowSize = nsyms + 1;
            // Simple ICDF: equal-prob distribution. CDF = [8192, 16384, 24576, 32768].
            // ICDF = [32768-8192, 32768-16384, 32768-24576, 0, 0_pad] = [24576, 16384, 8192, 0, 0]
            ushort[] icdf = new ushort[rowSize];
            icdf[0] = 24576;
            icdf[1] = 16384;
            icdf[2] = 8192;
            icdf[3] = 0;
            icdf[4] = 0;

            int[] inputSyms = new[] { 0, 1, 2, 3, 1, 0, 3, 2, 1, 1, 0, 3, 2, 0, 2, 1 };

            using var dInputSyms = acc.Allocate1D<int>(inputSyms.Length);
            using var dDecodedSyms = acc.Allocate1D<int>(inputSyms.Length);
            using var dIcdf = acc.Allocate1D<ushort>(rowSize);
            using var dScratch = acc.Allocate1D<byte>(256);
            using var dOutLen = acc.Allocate1D<long>(1);

            dInputSyms.View.CopyFromCPU(inputSyms);
            dIcdf.View.CopyFromCPU(icdf);
            dScratch.View.CopyFromCPU(new byte[256]);

            using var kernel = new Av1CdfRoundTripKernel(acc);
            kernel.Run(dInputSyms.View, dDecodedSyms.View, dIcdf.View, dScratch.View, dOutLen.View, inputSyms.Length, nsyms);
            await acc.SynchronizeAsync();

            var decoded = await dDecodedSyms.CopyToHostAsync();
            for (int i = 0; i < inputSyms.Length; i++)
                if (inputSyms[i] != decoded[i])
                    throw new Exception($"sym[{i}]: input={inputSyms[i]} decoded={decoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
