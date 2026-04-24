using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacResidualDecoder"/>, the Rice-coded residual decoder
/// shared by FIXED and LPC subframes. Test vectors are hand-constructed:
/// each residual is forward-Rice-encoded to produce bit patterns, packed
/// MSB-first into bytes, and then decoded by the library. The zigzag mapping
/// between signed residuals and unsigned Rice codewords is:
/// u = 2x for x ≥ 0 and u = 2|x| - 1 for x &lt; 0.
/// </summary>
public abstract partial class CodecsTestBase
{
    // FlacBitWriter lives in the library (internal); tests access it via InternalsVisibleTo.

    private static void WriteRiceCoded(FlacBitWriter w, int value, int param)
    {
        // Zigzag: u = 2*x for x >= 0, u = 2*|x| - 1 for x < 0.
        uint u = value >= 0 ? (uint)(value << 1) : (uint)((-value << 1) - 1);
        int q = (int)(u >> param);
        uint r = u & ((1u << param) - 1);
        w.WriteUnary(q);
        if (param > 0) w.Write(r, param);
    }

    private static void WriteRiceHeader(FlacBitWriter w, int codingMethod, int partitionOrder)
    {
        w.Write((uint)codingMethod, 2);
        w.Write((uint)partitionOrder, 4);
    }

    private static int[] DecodeRiceResidual(byte[] data, int blockSize, int predictorOrder)
    {
        var residual = new int[blockSize - predictorOrder];
        var r = new FlacBitReader(data);
        FlacResidualDecoder.Decode(ref r, residual, blockSize, predictorOrder);
        return residual;
    }

    [TestMethod]
    public void FlacResidual_SinglePartition_Param0_AllZeros()
    {
        // blockSize=4, predictor=0, partitionOrder=0, k=0. 4 residuals all zero.
        // Header: method=0(2), order=0(4) | Rice param=0(4) = 10 bits.
        // Residual zero with k=0: "1" (1 bit). x4 = 4 bits.
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 0, 0);
        w.Write(0, 4); // param
        for (int i = 0; i < 4; i++) WriteRiceCoded(w, 0, 0);
        var residual = DecodeRiceResidual(w.ToArray(), blockSize: 4, predictorOrder: 0);
        EqualInts(new[] { 0, 0, 0, 0 }, residual);
    }

    [TestMethod]
    public void FlacResidual_SinglePartition_Param2_Mixed_HandComputed()
    {
        // Expected residuals: [0, 1, -1, 3] with k=2.
        var residuals = new[] { 0, 1, -1, 3 };
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 0, 0);
        w.Write(2, 4); // param = 2
        foreach (var v in residuals) WriteRiceCoded(w, v, 2);
        var decoded = DecodeRiceResidual(w.ToArray(), blockSize: 4, predictorOrder: 0);
        EqualInts(residuals, decoded);
    }

    [TestMethod]
    public void FlacResidual_SinglePartition_NegativeResiduals()
    {
        var residuals = new[] { -5, -2, -1, 0, 1, 2, 5 };
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 0, 0);
        w.Write(3, 4); // param = 3
        foreach (var v in residuals) WriteRiceCoded(w, v, 3);
        var decoded = DecodeRiceResidual(w.ToArray(), blockSize: residuals.Length, predictorOrder: 0);
        EqualInts(residuals, decoded);
    }

    [TestMethod]
    public void FlacResidual_PartitionOrder1_TwoPartitions_DifferentParams()
    {
        // blockSize=8, predictorOrder=2, partitionOrder=1 => 2 partitions.
        // Partition 0 size = 8/2 - 2 = 2 samples. Partition 1 size = 4 samples.
        // Partition 0: k=1, residuals [0, 1]. Partition 1: k=3, residuals [-3, 7, -8, 2].
        var part0 = new[] { 0, 1 };
        var part1 = new[] { -3, 7, -8, 2 };
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 0, 1);
        // Partition 0: param=1
        w.Write(1, 4);
        foreach (var v in part0) WriteRiceCoded(w, v, 1);
        // Partition 1: param=3
        w.Write(3, 4);
        foreach (var v in part1) WriteRiceCoded(w, v, 3);

        var decoded = DecodeRiceResidual(w.ToArray(), blockSize: 8, predictorOrder: 2);
        var expected = part0.Concat(part1).ToArray();
        EqualInts(expected, decoded);
    }

    [TestMethod]
    public void FlacResidual_EscapeCode_VerbatimResiduals()
    {
        // blockSize=4, predictor=0, partitionOrder=0, k=escape(15 for method 0).
        // Then 5-bit raw-bits field, then that many bits per residual, two's-complement.
        // Use rawBits=10, residuals [-100, 100, 0, 255].
        var residuals = new[] { -100, 100, 0, 255 };
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 0, 0);
        w.Write(15, 4); // escape
        w.Write(10, 5); // raw bits
        foreach (var v in residuals)
        {
            // Two's complement 10-bit encoding of v.
            uint raw = (uint)(v & 0x3FF);
            w.Write(raw, 10);
        }
        var decoded = DecodeRiceResidual(w.ToArray(), blockSize: 4, predictorOrder: 0);
        EqualInts(residuals, decoded);
    }

    [TestMethod]
    public void FlacResidual_PartitionedRice2_5BitParam()
    {
        // Coding method = 1, Rice parameter field is 5 bits wide (not 4).
        var residuals = new[] { 0, 1, -2, 3 };
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 1, 0); // method=1 (Rice2)
        w.Write(2, 5); // 5-bit param
        foreach (var v in residuals) WriteRiceCoded(w, v, 2);
        var decoded = DecodeRiceResidual(w.ToArray(), blockSize: 4, predictorOrder: 0);
        EqualInts(residuals, decoded);
    }

    [TestMethod]
    public void FlacResidual_ReservedCodingMethod_Throws()
    {
        var w = new FlacBitWriter();
        w.Write(2, 2); // coding method 0b10 reserved
        w.Write(0, 4);
        bool threw = false;
        try { DecodeRiceResidual(w.ToArray(), blockSize: 4, predictorOrder: 0); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Reserved coding method 0b10 should throw.");
    }

    [TestMethod]
    public void FlacResidual_PartitionOrderExceedsBlock_Throws()
    {
        // blockSize=4, partitionOrder=3 (= 8 partitions) → partitionSize=0, invalid.
        var w = new FlacBitWriter();
        WriteRiceHeader(w, 0, 3);
        bool threw = false;
        try { DecodeRiceResidual(w.ToArray(), blockSize: 4, predictorOrder: 0); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Partition order > log2(blockSize) should throw.");
    }
}
