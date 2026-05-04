// Cross-backend tests for SilkLtpIndicesDecoderGpu.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static readonly byte[] SilkLtpTest_PerIndex = { 179, 99, 0 };
    private static readonly byte[] SilkLtpTest_LtpScale = { 128, 64, 0 };
    private static readonly byte[] SilkLtpTest_LtpGain0 =
        { 71, 56, 43, 30, 21, 12, 6, 0 };
    private static readonly byte[] SilkLtpTest_LtpGain1 =
    {
        199, 165, 144, 124, 109, 96, 84, 71,
         61,  51,  42,  32,  23, 15,  8,  0,
    };
    private static readonly byte[] SilkLtpTest_LtpGain2 =
    {
        241, 225, 211, 199, 187, 175, 164, 153,
        142, 132, 123, 114, 105,  96,  88,  80,
         72,  64,  57,  50,  44,  38,  33,  29,
         24,  20,  16,  12,   9,   5,   2,   0,
    };

    private static byte[] SilkLtpTest_FlatGains()
    {
        // 8 + 16 + 32 = 56
        var flat = new byte[8 + 16 + 32];
        Array.Copy(SilkLtpTest_LtpGain0, 0, flat, 0, 8);
        Array.Copy(SilkLtpTest_LtpGain1, 0, flat, 8, 16);
        Array.Copy(SilkLtpTest_LtpGain2, 0, flat, 24, 32);
        return flat;
    }

    private static readonly int[] SilkLtpTest_GainOffsets = { 0, 8, 24 };

    private static byte[] SilkLtpTest_SelectGain(int perIndex) => perIndex switch
    {
        0 => SilkLtpTest_LtpGain0,
        1 => SilkLtpTest_LtpGain1,
        2 => SilkLtpTest_LtpGain2,
        _ => throw new ArgumentOutOfRangeException(nameof(perIndex)),
    };

    /// <summary>Encode known LTP indices via the libopus reference encoder.</summary>
    private static byte[] SilkLtpEncodeIndicesCpu(
        int perIndex, int[] gainIndices, int ltpScaleIndex,
        int conditional, int nbSubfr)
    {
        var enc = new OpusRangeEncoder(64);
        enc.EncodeIcdf(perIndex, SilkLtpTest_PerIndex, 8);
        var gainIcdf = SilkLtpTest_SelectGain(perIndex);
        for (int k = 0; k < nbSubfr; k++)
        {
            enc.EncodeIcdf(gainIndices[k], gainIcdf, 8);
        }
        if (conditional == 0)
        {
            enc.EncodeIcdf(ltpScaleIndex, SilkLtpTest_LtpScale, 8);
        }
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> SilkLtpDecodeIndicesGpuAsync(
        Accelerator acc,
        byte[] packet, int conditional, int nbSubfr)
    {
        var flatGains = SilkLtpTest_FlatGains();

        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dPerIndex = acc.Allocate1D<byte>(SilkLtpTest_PerIndex.Length);
        using var dGainFlat = acc.Allocate1D<byte>(flatGains.Length);
        using var dGainOffsets = acc.Allocate1D<int>(SilkLtpTest_GainOffsets.Length);
        using var dLtpScale = acc.Allocate1D<byte>(SilkLtpTest_LtpScale.Length);
        using var dOutput = acc.Allocate1D<int>(2 + nbSubfr);

        dPacket.View.CopyFromCPU(packet);
        dPerIndex.View.CopyFromCPU(SilkLtpTest_PerIndex);
        dGainFlat.View.CopyFromCPU(flatGains);
        dGainOffsets.View.CopyFromCPU(SilkLtpTest_GainOffsets);
        dLtpScale.View.CopyFromCPU(SilkLtpTest_LtpScale);

        var inputs = new SilkLtpIndicesInputs
        {
            LtpPerIndexIcdf = dPerIndex.View,
            LtpGainIcdfFlat = dGainFlat.View,
            LtpGainOffsets = dGainOffsets.View,
            LtpScaleIcdf = dLtpScale.View,
        };

        using var kernel = new SilkLtpIndicesDecoderGpuTestKernel(acc);
        kernel.Run(
            dPacket.View, 0, packet.Length,
            inputs,
            conditional, nbSubfr,
            dOutput.View);
        await acc.SynchronizeAsync();

        var output = await dOutput.CopyToHostAsync();
        var slice = new int[2 + nbSubfr];
        Array.Copy(output, slice, 2 + nbSubfr);
        return slice;
    }

    [TestMethod]
    public async Task SilkLtpIndicesDecoderGpu_Per0Independent4Subfr_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // perIndex=0 (8-symbol gain codebook), 4 subframes, conditional=0 -> ltpScale read.
            int[] gainIdx = { 3, 5, 1, 6 };
            byte[] encoded = SilkLtpEncodeIndicesCpu(
                perIndex: 0, gainIndices: gainIdx, ltpScaleIndex: 1,
                conditional: 0, nbSubfr: 4);

            int[] gpu = await SilkLtpDecodeIndicesGpuAsync(
                acc, encoded, conditional: 0, nbSubfr: 4);

            if (gpu[0] != 0) throw new Exception($"perIndex mismatch: expected 0 got {gpu[0]}");
            if (gpu[1] != 1) throw new Exception($"ltpScaleIndex mismatch: expected 1 got {gpu[1]}");
            for (int k = 0; k < 4; k++)
                if (gpu[2 + k] != gainIdx[k])
                    throw new Exception(
                        $"gainIdx[{k}] mismatch: expected {gainIdx[k]} got {gpu[2 + k]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLtpIndicesDecoderGpu_Per1Conditional2Subfr_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // perIndex=1 (16-symbol gain codebook), 2 subframes, conditional=1 -> scale=0 fixed.
            int[] gainIdx = { 8, 12 };
            byte[] encoded = SilkLtpEncodeIndicesCpu(
                perIndex: 1, gainIndices: gainIdx, ltpScaleIndex: 0,
                conditional: 1, nbSubfr: 2);

            int[] gpu = await SilkLtpDecodeIndicesGpuAsync(
                acc, encoded, conditional: 1, nbSubfr: 2);

            if (gpu[0] != 1) throw new Exception($"perIndex mismatch: expected 1 got {gpu[0]}");
            if (gpu[1] != 0) throw new Exception($"ltpScaleIndex must be 0 when conditional!=0; got {gpu[1]}");
            for (int k = 0; k < 2; k++)
                if (gpu[2 + k] != gainIdx[k])
                    throw new Exception($"gainIdx[{k}] mismatch: expected {gainIdx[k]} got {gpu[2 + k]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLtpIndicesDecoderGpu_Per2Independent4Subfr_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // perIndex=2 (32-symbol gain codebook), 4 subframes, conditional=0.
            int[] gainIdx = { 17, 23, 4, 28 };
            byte[] encoded = SilkLtpEncodeIndicesCpu(
                perIndex: 2, gainIndices: gainIdx, ltpScaleIndex: 2,
                conditional: 0, nbSubfr: 4);

            int[] gpu = await SilkLtpDecodeIndicesGpuAsync(
                acc, encoded, conditional: 0, nbSubfr: 4);

            if (gpu[0] != 2) throw new Exception($"perIndex mismatch: expected 2 got {gpu[0]}");
            if (gpu[1] != 2) throw new Exception($"ltpScaleIndex mismatch: expected 2 got {gpu[1]}");
            for (int k = 0; k < 4; k++)
                if (gpu[2 + k] != gainIdx[k])
                    throw new Exception($"gainIdx[{k}] mismatch: expected {gainIdx[k]} got {gpu[2 + k]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
