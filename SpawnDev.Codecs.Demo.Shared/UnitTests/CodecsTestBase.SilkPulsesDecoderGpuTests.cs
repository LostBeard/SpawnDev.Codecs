// Cross-backend tests for SilkPulsesDecoderGpu - GPU port of
// SilkPulsesDecoder.Decode.

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
    /// <summary>Encode signed pulses via the CPU SilkPulsesDecoder.Encode.</summary>
    private static byte[] SilkPulsesEncodeCpu(
        short[] pulses, int signalType, int quantOffsetType, int frameLength, int rateLevelIndex)
    {
        var enc = new OpusRangeEncoder(256);
        SilkPulsesDecoder.Encode(
            enc,
            pulses.AsSpan(0, ((frameLength + 15) & ~15)),
            signalType, quantOffsetType, frameLength, rateLevelIndex);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<short[]> SilkPulsesDecodeGpuAsync(
        Accelerator acc, byte[] packet,
        int signalType, int quantOffsetType, int frameLength)
    {
        int alignedLen = (frameLength + 15) & ~15;

        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dRateLevels = acc.Allocate1D<byte>(SilkIcdfTables.RateLevels.Length);
        using var dPulsesPerBlock = acc.Allocate1D<byte>(SilkIcdfTables.PulsesPerBlock.Length);
        using var dLsb = acc.Allocate1D<byte>(SilkIcdfTables.Lsb.Length);
        using var dSign = acc.Allocate1D<byte>(SilkIcdfTables.Sign.Length);
        using var dShellOffsets = acc.Allocate1D<byte>(SilkShellCodeTables.Offsets.Length);
        using var dShellTable0 = acc.Allocate1D<byte>(SilkShellCodeTables.Table0.Length);
        using var dShellTable1 = acc.Allocate1D<byte>(SilkShellCodeTables.Table1.Length);
        using var dShellTable2 = acc.Allocate1D<byte>(SilkShellCodeTables.Table2.Length);
        using var dShellTable3 = acc.Allocate1D<byte>(SilkShellCodeTables.Table3.Length);
        using var dSumPulses = acc.Allocate1D<int>(20);
        using var dNLshifts = acc.Allocate1D<int>(20);
        using var dPulsesOut = acc.Allocate1D<short>(alignedLen);

        dPacket.View.CopyFromCPU(packet);
        dRateLevels.View.CopyFromCPU(SilkIcdfTables.RateLevels);
        dPulsesPerBlock.View.CopyFromCPU(SilkIcdfTables.PulsesPerBlock);
        dLsb.View.CopyFromCPU(SilkIcdfTables.Lsb);
        dSign.View.CopyFromCPU(SilkIcdfTables.Sign);
        dShellOffsets.View.CopyFromCPU(SilkShellCodeTables.Offsets);
        dShellTable0.View.CopyFromCPU(SilkShellCodeTables.Table0);
        dShellTable1.View.CopyFromCPU(SilkShellCodeTables.Table1);
        dShellTable2.View.CopyFromCPU(SilkShellCodeTables.Table2);
        dShellTable3.View.CopyFromCPU(SilkShellCodeTables.Table3);

        var inputs = new SilkPulsesInputs
        {
            RateLevelsIcdf = dRateLevels.View,
            PulsesPerBlockIcdf = dPulsesPerBlock.View,
            LsbIcdf = dLsb.View,
            SignIcdf = dSign.View,
            ShellTables = new SilkShellCoderTables
            {
                Offsets = dShellOffsets.View,
                Table0 = dShellTable0.View,
                Table1 = dShellTable1.View,
                Table2 = dShellTable2.View,
                Table3 = dShellTable3.View,
            },
            SumPulsesScratch = dSumPulses.View,
            NLshiftsScratch = dNLshifts.View,
        };

        using var kernel = new SilkPulsesDecoderGpuTestKernel(acc);
        kernel.Run(dPacket.View, 0, packet.Length, inputs,
            signalType, quantOffsetType, frameLength, dPulsesOut.View);
        await acc.SynchronizeAsync();

        var output = await dPulsesOut.CopyToHostAsync();
        var slice = new short[alignedLen];
        Array.Copy(output, slice, alignedLen);
        return slice;
    }

    private static async Task SilkPulsesTest_AssertRoundTrip(
        Accelerator acc, short[] pulses, int signalType, int quantOffsetType,
        int frameLength, int rateLevelIndex)
    {
        byte[] encoded = SilkPulsesEncodeCpu(
            pulses, signalType, quantOffsetType, frameLength, rateLevelIndex);
        short[] gpu = await SilkPulsesDecodeGpuAsync(
            acc, encoded, signalType, quantOffsetType, frameLength);

        for (int i = 0; i < frameLength; i++)
            if (gpu[i] != pulses[i])
                throw new Exception(
                    $"pulse[{i}] mismatch (signalType={signalType}, " +
                    $"frameLength={frameLength}): input={pulses[i]} gpu={gpu[i]}");
    }

    [TestMethod]
    public async Task SilkPulsesDecoderGpu_AllZeros20msNb_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 20ms NB: 8kHz × 20ms = 160 samples = 10 shell blocks of 16.
            // All zeros - tests the empty-block fast path.
            const int frameLength = 160;
            int alignedLen = (frameLength + 15) & ~15;
            var pulses = new short[alignedLen];
            await SilkPulsesTest_AssertRoundTrip(
                acc, pulses, signalType: 1, quantOffsetType: 0,
                frameLength: frameLength, rateLevelIndex: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPulsesDecoderGpu_SparseSigned20msWb_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 20ms WB: 16kHz × 20ms = 320 samples = 20 shell blocks of 16.
            // Sparse signed pulses, mixed signs.
            const int frameLength = 320;
            int alignedLen = (frameLength + 15) & ~15;
            var pulses = new short[alignedLen];
            // Sprinkle signed pulses across various blocks.
            pulses[0] = 1; pulses[5] = -1; pulses[12] = 2;
            pulses[17] = -1; pulses[33] = 1; pulses[80] = -2;
            pulses[150] = 1; pulses[200] = -1; pulses[250] = 1;
            // Voiced (signalType=2), quantOffset=1.
            await SilkPulsesTest_AssertRoundTrip(
                acc, pulses, signalType: 2, quantOffsetType: 1,
                frameLength: frameLength, rateLevelIndex: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPulsesDecoderGpu_DenseLowMagnitude10msMb_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 10ms MB: 12kHz × 10ms = 120 samples (NOT a multiple of 16!).
            // 120 / 16 = 7.5 → 7 full blocks + 1 partial-but-allocated 8th block.
            const int frameLength = 120;
            int alignedLen = (frameLength + 15) & ~15; // 128
            var pulses = new short[alignedLen];
            // Dense ±1 pulses in the first 7 full blocks.
            for (int i = 0; i < 112; i += 4)
                pulses[i] = (short)((i % 8 == 0) ? 1 : -1);
            await SilkPulsesTest_AssertRoundTrip(
                acc, pulses, signalType: 0, quantOffsetType: 0,
                frameLength: frameLength, rateLevelIndex: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
