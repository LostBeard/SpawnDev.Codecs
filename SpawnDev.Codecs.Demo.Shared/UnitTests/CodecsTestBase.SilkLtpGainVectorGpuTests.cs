// Cross-backend test for SilkLtpGainVectorGpu.LookupTapAt. Verifies the
// GPU per-subframe LTP gain vector codebook lookup matches the CPU
// reference SilkLtpDecoder.GetGainVector bit-exactly.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task SilkLtpGainVectorGpu_PerIndex0_4Subfr_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Codebook 0 has 8 entries; pick a span of valid indices.
            sbyte[] ltpIndices = { 0, 3, 5, 7 };
            await LookupAndVerify(acc, perIndex: 0, ltpIndices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLtpGainVectorGpu_PerIndex1_4Subfr_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Codebook 1 has 16 entries.
            sbyte[] ltpIndices = { 0, 5, 10, 15 };
            await LookupAndVerify(acc, perIndex: 1, ltpIndices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLtpGainVectorGpu_PerIndex2_4Subfr_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Codebook 2 has 32 entries; cover a wide range.
            sbyte[] ltpIndices = { 0, 10, 20, 31 };
            await LookupAndVerify(acc, perIndex: 2, ltpIndices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLtpGainVectorGpu_2Subfr_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 10 ms frame: 2 subframes only.
            sbyte[] ltpIndices = { 4, 7 };
            await LookupAndVerify(acc, perIndex: 1, ltpIndices);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task LookupAndVerify(Accelerator acc, int perIndex, sbyte[] ltpIndices)
    {
        const int ltpVecSize = 5;
        int nbSubfr = ltpIndices.Length;

        // CPU reference.
        sbyte[] cpuTaps = new sbyte[nbSubfr * ltpVecSize];
        for (int s = 0; s < nbSubfr; s++)
        {
            SilkLtpDecoder.GetGainVector(
                cpuTaps.AsSpan(s * ltpVecSize, ltpVecSize),
                perIndex, ltpIndices[s]);
        }

        // Build flattened codebook (Vq0 + Vq1 + Vq2 = 40 + 80 + 160 = 280 bytes).
        sbyte[] codebook = new sbyte[280];
        Array.Copy(SilkLtpGainTables.Vq0, 0, codebook, 0, 40);
        Array.Copy(SilkLtpGainTables.Vq1, 0, codebook, 40, 80);
        Array.Copy(SilkLtpGainTables.Vq2, 0, codebook, 120, 160);

        // GPU dispatch: nbSubfr * 5 threads.
        using var dTaps = acc.Allocate1D<sbyte>(nbSubfr * ltpVecSize);
        using var dCodebook = acc.Allocate1D<sbyte>(codebook.Length);
        using var dLtpIndices = acc.Allocate1D<sbyte>(nbSubfr);
        dCodebook.View.CopyFromCPU(codebook);
        dLtpIndices.View.CopyFromCPU(ltpIndices);
        dTaps.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<sbyte>, ArrayView<sbyte>, ArrayView<sbyte>, int>(LookupKernel);
        kernel(new Index1D(nbSubfr * ltpVecSize), dTaps.View, dCodebook.View, dLtpIndices.View, perIndex);
        await acc.SynchronizeAsync();

        var gpuTaps = await dTaps.CopyToHostAsync();

        for (int i = 0; i < cpuTaps.Length; i++)
        {
            if (cpuTaps[i] != gpuTaps[i])
                throw new Exception($"taps[{i}]: cpu={cpuTaps[i]} gpu={gpuTaps[i]} (perIndex={perIndex})");
        }
    }

    private static void LookupKernel(
        Index1D index,
        ArrayView<sbyte> taps, ArrayView<sbyte> codebook, ArrayView<sbyte> ltpIndices,
        int perIndex)
    {
        SilkLtpGainVectorGpu.LookupTapAt(taps, 0, codebook, 0, ltpIndices, 0, perIndex, index.X);
    }
}
