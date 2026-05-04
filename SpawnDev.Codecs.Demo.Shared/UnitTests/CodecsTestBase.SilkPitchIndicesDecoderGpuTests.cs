// Cross-backend tests for SilkPitchIndicesDecoderGpu.

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
    // SilkIcdfTables values mirrored locally for tests.
    private static readonly byte[] SilkPitchTest_PitchDelta =
    {
        210, 208, 206, 203, 199, 193, 183, 168,
        142, 104,  74,  52,  37,  27,  20,  14,
         10,   6,   4,   2,   0,
    };
    private static readonly byte[] SilkPitchTest_PitchLag =
    {
        253, 250, 244, 233, 212, 182, 150, 131,
        120, 110,  98,  85,  72,  60,  49,  40,
         32,  25,  19,  15,  13,  11,   9,   8,
          7,   6,   5,   4,   3,   2,   1,   0,
    };
    private static readonly byte[] SilkPitchTest_Uniform4 = { 192, 128, 64, 0 };
    private static readonly byte[] SilkPitchTest_Uniform6 = { 213, 170, 128, 85, 43, 0 };
    private static readonly byte[] SilkPitchTest_Uniform8 = { 224, 192, 160, 128, 96, 64, 32, 0 };
    private static readonly byte[] SilkPitchTest_PitchContour =
    {
        223, 201, 183, 167, 152, 138, 124, 111,
         98,  88,  79,  70,  62,  56,  50,  44,
         39,  35,  31,  27,  24,  21,  18,  16,
         14,  12,  10,   8,   6,   4,   3,   2,
          1,   0,
    };
    private static readonly byte[] SilkPitchTest_PitchContourNb =
    {
        188, 176, 155, 138, 119,  97,  67,  43,
         26,  10,   0,
    };
    private static readonly byte[] SilkPitchTest_PitchContour10Ms =
    {
        165, 119,  80,  61,  47,  35,  27,  20,
         14,   9,   4,   0,
    };
    private static readonly byte[] SilkPitchTest_PitchContour10MsNb =
    {
        113,  63,   0,
    };

    private static byte[] SilkPitchTest_SelectLagLowBits(int fsKHz) =>
        fsKHz switch
        {
            16 => SilkPitchTest_Uniform8,
            12 => SilkPitchTest_Uniform6,
            8 => SilkPitchTest_Uniform4,
            _ => throw new ArgumentException($"fsKHz: {fsKHz}"),
        };

    private static byte[] SilkPitchTest_SelectContour(int fsKHz, int nbSubfr) =>
        (fsKHz == 8)
            ? (nbSubfr == 4 ? SilkPitchTest_PitchContourNb : SilkPitchTest_PitchContour10MsNb)
            : (nbSubfr == 4 ? SilkPitchTest_PitchContour : SilkPitchTest_PitchContour10Ms);

    /// <summary>Encode a known (lagIndex, contourIndex, conditional/delta state) sequence.</summary>
    private static byte[] SilkPitchEncodeIndicesCpu(
        int rawDelta, // 0 = absolute, 1..20 = delta with delta = raw-9
        int coarseLag, int lsb, int contour,
        int fsKHz, int nbSubfr,
        bool conditional, bool prevVoiced)
    {
        var enc = new OpusRangeEncoder(64);
        bool useAbsolute = !(conditional && prevVoiced && rawDelta > 0);
        if (conditional && prevVoiced)
        {
            enc.EncodeIcdf(rawDelta, SilkPitchTest_PitchDelta, 8);
        }
        if (useAbsolute)
        {
            enc.EncodeIcdf(coarseLag, SilkPitchTest_PitchLag, 8);
            enc.EncodeIcdf(lsb, SilkPitchTest_SelectLagLowBits(fsKHz), 8);
        }
        enc.EncodeIcdf(contour, SilkPitchTest_SelectContour(fsKHz, nbSubfr), 8);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> SilkPitchDecodeIndicesGpuAsync(
        Accelerator acc,
        byte[] packet, int fsKHz, int nbSubfr,
        int prevLagIndex, int prevSignalTypeWasVoiced, int conditional)
    {
        var lagLowBits = SilkPitchTest_SelectLagLowBits(fsKHz);
        var contour = SilkPitchTest_SelectContour(fsKHz, nbSubfr);

        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dPitchDelta = acc.Allocate1D<byte>(SilkPitchTest_PitchDelta.Length);
        using var dPitchLag = acc.Allocate1D<byte>(SilkPitchTest_PitchLag.Length);
        using var dLagLowBits = acc.Allocate1D<byte>(lagLowBits.Length);
        using var dContour = acc.Allocate1D<byte>(contour.Length);
        using var dOutput = acc.Allocate1D<int>(2);

        dPacket.View.CopyFromCPU(packet);
        dPitchDelta.View.CopyFromCPU(SilkPitchTest_PitchDelta);
        dPitchLag.View.CopyFromCPU(SilkPitchTest_PitchLag);
        dLagLowBits.View.CopyFromCPU(lagLowBits);
        dContour.View.CopyFromCPU(contour);

        var inputs = new SilkPitchIndicesInputs
        {
            PitchDeltaIcdf = dPitchDelta.View,
            PitchLagIcdf = dPitchLag.View,
            LagLowBitsIcdf = dLagLowBits.View,
            ContourIcdf = dContour.View,
        };

        using var kernel = new SilkPitchIndicesDecoderGpuTestKernel(acc);
        kernel.Run(
            dPacket.View, 0, packet.Length,
            inputs,
            fsKHz, prevLagIndex, prevSignalTypeWasVoiced, conditional,
            dOutput.View);
        await acc.SynchronizeAsync();

        var output = await dOutput.CopyToHostAsync();
        return new int[] { output[0], output[1] };
    }

    [TestMethod]
    public async Task SilkPitchIndicesDecoderGpu_AbsoluteWb20ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // WB (16kHz), 20ms (nbSubfr=4), absolute coding (conditional=0).
            const int fsKHz = 16, nbSubfr = 4;
            const int coarse = 12, lsb = 5, contour = 18;
            int expectedLag = coarse * (fsKHz >> 1) + lsb;

            byte[] encoded = SilkPitchEncodeIndicesCpu(
                0, coarse, lsb, contour, fsKHz, nbSubfr,
                conditional: false, prevVoiced: false);

            int[] gpu = await SilkPitchDecodeIndicesGpuAsync(
                acc, encoded, fsKHz, nbSubfr,
                prevLagIndex: 0, prevSignalTypeWasVoiced: 0, conditional: 0);

            if (gpu[0] != expectedLag)
                throw new Exception($"lagIndex mismatch: expected {expectedLag} got {gpu[0]}");
            if (gpu[1] != contour)
                throw new Exception($"contour mismatch: expected {contour} got {gpu[1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchIndicesDecoderGpu_DeltaCodedNb20ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // NB (8kHz), 20ms, conditional + prevVoiced -> try delta path.
            // rawDelta = 12 -> delta = 12 - 9 = 3, applied to prevLagIndex.
            const int fsKHz = 8, nbSubfr = 4;
            const int rawDelta = 12;
            const int prevLag = 100, contour = 5;
            int expectedLag = prevLag + (rawDelta - 9); // 100 + 3 = 103

            byte[] encoded = SilkPitchEncodeIndicesCpu(
                rawDelta, 0, 0, contour, fsKHz, nbSubfr,
                conditional: true, prevVoiced: true);

            int[] gpu = await SilkPitchDecodeIndicesGpuAsync(
                acc, encoded, fsKHz, nbSubfr,
                prevLagIndex: prevLag, prevSignalTypeWasVoiced: 1, conditional: 1);

            if (gpu[0] != expectedLag)
                throw new Exception($"delta-coded lag mismatch: expected {expectedLag} got {gpu[0]}");
            if (gpu[1] != contour)
                throw new Exception($"contour mismatch: expected {contour} got {gpu[1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchIndicesDecoderGpu_FallthroughAbsoluteMb10ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // MB (12kHz), 10ms, conditional + prevVoiced + rawDelta=0 -> fallthrough to absolute.
            const int fsKHz = 12, nbSubfr = 2;
            const int coarse = 8, lsb = 3, contour = 7;
            int expectedLag = coarse * (fsKHz >> 1) + lsb;

            byte[] encoded = SilkPitchEncodeIndicesCpu(
                0, coarse, lsb, contour, fsKHz, nbSubfr,
                conditional: true, prevVoiced: true);

            int[] gpu = await SilkPitchDecodeIndicesGpuAsync(
                acc, encoded, fsKHz, nbSubfr,
                prevLagIndex: 50, prevSignalTypeWasVoiced: 1, conditional: 1);

            if (gpu[0] != expectedLag)
                throw new Exception($"fallthrough lag mismatch: expected {expectedLag} got {gpu[0]}");
            if (gpu[1] != contour)
                throw new Exception($"contour mismatch: expected {contour} got {gpu[1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
