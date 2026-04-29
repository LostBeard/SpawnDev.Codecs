// Cross-backend test for FlacSubframeHeaderGpu.ParseAt. Verifies the GPU
// FLAC subframe header parser matches the CPU reference
// FlacSubframeHeaderParser.Parse bit-exactly across all 4 subframe kinds
// + wasted-bits handling.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacSubframeHeaderGpu_Constant_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await EncodeAndVerify(acc, code: 0b000000, wastedBits: 0,
                expectedKind: FlacSubframeHeaderGpu.KIND_CONSTANT, expectedOrder: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacSubframeHeaderGpu_Verbatim_2WastedBits_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await EncodeAndVerify(acc, code: 0b000001, wastedBits: 2,
                expectedKind: FlacSubframeHeaderGpu.KIND_VERBATIM, expectedOrder: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacSubframeHeaderGpu_Fixed_Order3_NoWasted_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await EncodeAndVerify(acc, code: 0b001011, wastedBits: 0,
                expectedKind: FlacSubframeHeaderGpu.KIND_FIXED, expectedOrder: 3);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacSubframeHeaderGpu_Lpc_Order8_5WastedBits_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // LPC order 8 -> code 0b100111 (8-1=7).
            await EncodeAndVerify(acc, code: 0b100111, wastedBits: 5,
                expectedKind: FlacSubframeHeaderGpu.KIND_LPC, expectedOrder: 8);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacSubframeHeaderGpu_Lpc_Order32_MaxWastedBits_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // LPC order 32 -> code 0b111111 (32-1=31).
            await EncodeAndVerify(acc, code: 0b111111, wastedBits: 16,
                expectedKind: FlacSubframeHeaderGpu.KIND_LPC, expectedOrder: 32);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeAndVerify(
        Accelerator acc, int code, int wastedBits, int expectedKind, int expectedOrder)
    {
        // Build a known subframe header bit stream:
        //   1 reserved (0) + 6 type code + 1 wasted flag (+ unary wasted-1 if flag set)
        var w = new FlacBitWriter();
        w.Write(0, 1);                    // reserved 0
        w.Write((uint)code, 6);            // 6-bit type code
        if (wastedBits == 0)
        {
            w.Write(0, 1);                // wasted flag 0
        }
        else
        {
            w.Write(1, 1);                // wasted flag 1
            w.WriteUnary(wastedBits - 1); // unary count = wastedBits - 1
        }
        w.AlignToByte();
        byte[] encoded = w.ToArray();

        // CPU reference.
        var cpuReader = new FlacBitReader(encoded);
        var cpuHeader = FlacSubframeHeaderParser.Parse(ref cpuReader);
        if ((int)cpuHeader.Kind != expectedKind)
            throw new Exception($"CPU kind={cpuHeader.Kind} expected={expectedKind}");
        if (cpuHeader.Order != expectedOrder)
            throw new Exception($"CPU order={cpuHeader.Order} expected={expectedOrder}");
        if (cpuHeader.WastedBitsPerSample != wastedBits)
            throw new Exception($"CPU wastedBits={cpuHeader.WastedBitsPerSample} expected={wastedBits}");

        // GPU dispatch: single-thread.
        using var dData = acc.Allocate1D<byte>(encoded.Length);
        using var dOut = acc.Allocate1D<int>(3);
        dData.View.CopyFromCPU(encoded);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, int>(HeaderKernel);
        kernel(new Index1D(1), dData.View, dOut.View, encoded.Length);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        if (gpuOut[0] != expectedKind)
            throw new Exception($"GPU kind={gpuOut[0]} expected={expectedKind}");
        if (gpuOut[1] != expectedOrder)
            throw new Exception($"GPU order={gpuOut[1]} expected={expectedOrder}");
        if (gpuOut[2] != wastedBits)
            throw new Exception($"GPU wastedBits={gpuOut[2]} expected={wastedBits}");
    }

    private static void HeaderKernel(
        Index1D _,
        ArrayView<byte> data, ArrayView<int> output, int dataLen)
    {
        var state = FlacBitReaderGpu.Init(dataLen);
        FlacSubframeHeaderGpu.ParseAt(ref state, data, output, 0);
    }
}
