using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// A kernel that only copies every field of a <see cref="SilkPulsesInputs"/> argument (two views, ints, a nested struct
/// of ints) into an output buffer. The SILK pulse decode returned all-zero pulses on WebGPU while CUDA / OpenCL / CPU
/// were bit-exact, which is what reading the offsets as 0 would do; this checks the struct argument itself.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static void SilkPulsesInputsEchoKernel(Index1D _, SilkPulsesInputs inputs, ArrayView<int> output)
    {
        output[0] = inputs.RateLevelsIcdfAt;
        output[1] = inputs.PulsesPerBlockIcdfAt;
        output[2] = inputs.LsbIcdfAt;
        output[3] = inputs.SignIcdfAt;
        output[4] = inputs.ShellTables.OffsetsAt;
        output[5] = inputs.ShellTables.Table0At;
        output[6] = inputs.ShellTables.Table1At;
        output[7] = inputs.ShellTables.Table2At;
        output[8] = inputs.ShellTables.Table3At;
        output[9] = inputs.SumPulsesScratchAt;
        output[10] = inputs.NLshiftsScratchAt;
        output[11] = inputs.Bytes[inputs.RateLevelsIcdfAt];        // a read through the first view
        output[12] = inputs.Scratch[inputs.SumPulsesScratchAt];     // a read through the second view
        inputs.Scratch[inputs.NLshiftsScratchAt] = 77;              // a write through the second view
    }

    [TestMethod]
    public async Task SilkPulsesInputs_StructArgument_ReachesTheKernelIntact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var bytes = new PackedDeviceBuffer<byte>();
            var ints = new PackedDeviceBuffer<int>();
            ints.Reserve(3);   // a non-zero start for the scratch regions
            var inputs = SilkPulsesInputs.AddStandard(bytes, ints);
            using var dBytes = bytes.Upload(acc);
            using var dInts = ints.Upload(acc);
            dInts.View.SubView(inputs.SumPulsesScratchAt, 1).CopyFromCPU(new[] { 4242 });
            inputs.Bind(dBytes.View, dInts.View);
            using var dOut = acc.Allocate1D<int>(13);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, SilkPulsesInputs, ArrayView<int>>(SilkPulsesInputsEchoKernel);
            kernel(1, inputs, dOut.View);
            await acc.SynchronizeAsync();
            var got = await dOut.View.CopyToHostAsync();
            var hostBytes = await dBytes.View.CopyToHostAsync();
            int[] expected =
            {
                inputs.RateLevelsIcdfAt, inputs.PulsesPerBlockIcdfAt, inputs.LsbIcdfAt, inputs.SignIcdfAt,
                inputs.ShellTables.OffsetsAt, inputs.ShellTables.Table0At, inputs.ShellTables.Table1At,
                inputs.ShellTables.Table2At, inputs.ShellTables.Table3At,
                inputs.SumPulsesScratchAt, inputs.NLshiftsScratchAt,
                hostBytes[inputs.RateLevelsIcdfAt], 4242,
            };
            string[] names = { "RateLevelsIcdfAt", "PulsesPerBlockIcdfAt", "LsbIcdfAt", "SignIcdfAt", "ShellTables.OffsetsAt",
                "ShellTables.Table0At", "ShellTables.Table1At", "ShellTables.Table2At", "ShellTables.Table3At",
                "SumPulsesScratchAt", "NLshiftsScratchAt", "Bytes[RateLevelsIcdfAt]", "Scratch[SumPulsesScratchAt]" };
            var bad = new List<string>();
            for (int i = 0; i < expected.Length; i++)
                if (got[i] != expected[i]) bad.Add($"{names[i]}: kernel saw {got[i]}, host {expected[i]}");
            var scratch = await dInts.View.CopyToHostAsync();
            if (scratch[inputs.NLshiftsScratchAt] != 77) bad.Add($"write through Scratch landed wrong (slot holds {scratch[inputs.NLshiftsScratchAt]})");
            True(bad.Count == 0, string.Join("\n", bad));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
