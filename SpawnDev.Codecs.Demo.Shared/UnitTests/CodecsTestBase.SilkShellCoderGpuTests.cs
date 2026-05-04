// Cross-backend tests for SilkShellCoderGpu - GPU port of SilkShellCoder.Decode.
// Encodes known pulse magnitudes via the CPU reference encoder, decodes on
// GPU, verifies bit-exact match.

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
    /// <summary>Encode a 16-pulse block via the CPU SilkShellCoder.Encode.</summary>
    private static byte[] SilkShellEncodeCpu(short[] pulses, int pulsesTotal)
    {
        var enc = new OpusRangeEncoder(64);
        SilkShellCoder.Encode(enc, pulses.AsSpan(0, 16), pulsesTotal);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<short[]> SilkShellDecodeGpuAsync(
        Accelerator acc, byte[] packet, int pulsesTotal)
    {
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dOffsets = acc.Allocate1D<byte>(SilkShellCodeTables.Offsets.Length);
        using var dTable0 = acc.Allocate1D<byte>(SilkShellCodeTables.Table0.Length);
        using var dTable1 = acc.Allocate1D<byte>(SilkShellCodeTables.Table1.Length);
        using var dTable2 = acc.Allocate1D<byte>(SilkShellCodeTables.Table2.Length);
        using var dTable3 = acc.Allocate1D<byte>(SilkShellCodeTables.Table3.Length);
        using var dPulses = acc.Allocate1D<short>(16);

        dPacket.View.CopyFromCPU(packet);
        dOffsets.View.CopyFromCPU(SilkShellCodeTables.Offsets);
        dTable0.View.CopyFromCPU(SilkShellCodeTables.Table0);
        dTable1.View.CopyFromCPU(SilkShellCodeTables.Table1);
        dTable2.View.CopyFromCPU(SilkShellCodeTables.Table2);
        dTable3.View.CopyFromCPU(SilkShellCodeTables.Table3);

        var tables = new SilkShellCoderTables
        {
            Offsets = dOffsets.View,
            Table0 = dTable0.View,
            Table1 = dTable1.View,
            Table2 = dTable2.View,
            Table3 = dTable3.View,
        };

        using var kernel = new SilkShellCoderGpuTestKernel(acc);
        kernel.Run(dPacket.View, 0, packet.Length, tables, pulsesTotal, dPulses.View);
        await acc.SynchronizeAsync();

        var output = await dPulses.CopyToHostAsync();
        var slice = new short[16];
        Array.Copy(output, slice, 16);
        return slice;
    }

    private static async Task SilkShellTest_AssertRoundTrip(
        Accelerator acc, short[] pulses, int pulsesTotal)
    {
        byte[] encoded = SilkShellEncodeCpu(pulses, pulsesTotal);
        short[] gpu = await SilkShellDecodeGpuAsync(acc, encoded, pulsesTotal);

        for (int i = 0; i < 16; i++)
            if (gpu[i] != pulses[i])
                throw new Exception(
                    $"shell pulse[{i}] mismatch (pulsesTotal={pulsesTotal}): " +
                    $"input={pulses[i]} gpu={gpu[i]}");
    }

    [TestMethod]
    public async Task SilkShellCoderGpu_AllZeros_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // pulsesTotal=0 - all 16 pulses are zero, no DecodeIcdf calls fire.
            var pulses = new short[16];
            await SilkShellTest_AssertRoundTrip(acc, pulses, 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkShellCoderGpu_LowPulses_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 4 pulses spread - exercises the tree but with sparse splits.
            var pulses = new short[16];
            pulses[0] = 1; pulses[3] = 1; pulses[8] = 1; pulses[12] = 1;
            await SilkShellTest_AssertRoundTrip(acc, pulses, 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkShellCoderGpu_DenseUniform_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 16 pulses, one per sample - max coverage of the tree.
            var pulses = new short[16];
            for (int i = 0; i < 16; i++) pulses[i] = 1;
            await SilkShellTest_AssertRoundTrip(acc, pulses, 16);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkShellCoderGpu_AsymmetricMagnitudes_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Asymmetric distribution: most pulses concentrated in left half.
            var pulses = new short[16];
            pulses[0] = 3; pulses[1] = 2; pulses[2] = 1; pulses[3] = 1;
            pulses[4] = 1; pulses[7] = 1;
            pulses[10] = 1;
            int total = 0; for (int i = 0; i < 16; i++) total += pulses[i];
            await SilkShellTest_AssertRoundTrip(acc, pulses, total);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
