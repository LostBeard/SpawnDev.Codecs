// Tests for Vp9BlockCoefDecoder.DecodeBlockCoefficients (slice 149).
// Drives the per-block scan loop with deterministic bit sequences
// and verifies (a) the coefficient values land at the expected
// raster positions, (b) the EOB count matches the number of decoded
// non-EOB tokens, and (c) the rest of the block stays zero.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static Func<byte, int> BlockBitReader(int[] bits)
    {
        int idx = 0;
        return _ => bits[idx++];
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_EobAtPositionZero_ReturnsZeroAndAllZeros()
    {
        // Bit 0 = 0 means "EOB" at scan position 0.
        var read = BlockBitReader(new int[] { 0 });
        var block = new short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra, block);
        Equal(0, eob);
        for (int i = 0; i < 16; i++) Equal((short)0, block[i]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_OneZeroThenEob_WritesZeroAtScan0RasterPos()
    {
        // c=0: EOB?=1 (not EOB), ZERO?=0 (Zero), advance
        // c=1: EOB?=0 (EOB)
        // Default 4x4 scan position 0 = raster 0, so block[0] stays 0
        // (it was already 0 from the buffer init).
        var read = BlockBitReader(new int[] { 1, 0, 0 });
        var block = new short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra, block);
        Equal(1, eob);
        for (int i = 0; i < 16; i++) Equal((short)0, block[i]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_PositiveOneAtPositionZero_WritesOneAtRasterZero()
    {
        // c=0: EOB?=1, ZERO?=1, ONE?=0 (One token), sign=0 (positive) -> +1
        // c=1: EOB?=0 (EOB)
        var read = BlockBitReader(new int[] { 1, 1, 0, 0, 0 });
        var block = new short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra, block);
        Equal(1, eob);
        // Default 4x4 scan: scan[0] = 0 (DC).
        Equal((short)1, block[0]);
        for (int i = 1; i < 16; i++) Equal((short)0, block[i]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_NegativeOneAtPositionZero_WritesMinusOneAtRasterZero()
    {
        // ...sign = 1 -> -1.
        var read = BlockBitReader(new int[] { 1, 1, 0, 1, 0 });
        var block = new short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra, block);
        Equal(1, eob);
        Equal((short)(-1), block[0]);
        for (int i = 1; i < 16; i++) Equal((short)0, block[i]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_ThreePositiveOnes_LandAtScanRasterPositions()
    {
        // c=0,1,2 each: ONE token positive (4 bits)
        // c=3: EOB
        var perCoef = new int[] { 1, 1, 0, 0 }; // ONE positive
        var bits = new int[4 * 3 + 1];
        Array.Copy(perCoef, 0, bits, 0,  4);
        Array.Copy(perCoef, 0, bits, 4,  4);
        Array.Copy(perCoef, 0, bits, 8,  4);
        bits[12] = 0; // EOB at c=3

        var read = BlockBitReader(bits);
        var block = new short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra, block);
        Equal(3, eob);
        // Default 4x4 scan: scan[0]=0, scan[1]=4, scan[2]=1.
        Equal((short)1, block[0]);
        Equal((short)1, block[4]);
        Equal((short)1, block[1]);
        // All other raster positions stay zero.
        Equal((short)0, block[2]);
        Equal((short)0, block[3]);
        Equal((short)0, block[5]);
        Equal((short)0, block[15]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_TwoTokenAtPositionZero_WritesTwoSignedAtRasterZero()
    {
        // c=0: EOB?=1, ZERO?=1, ONE?=1 (constrained tree), tree:
        //   bit 0 (LOW_VAL) -> i=2 (TWO node)
        //   bit 0 -> -Two leaf
        //   sign=0 -> +2
        // c=1: EOB?=0 (EOB)
        var read = BlockBitReader(new int[] { 1, 1, 1, 0, 0, 0, 0 });
        var block = new short[16];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra, block);
        Equal(1, eob);
        Equal((short)2, block[0]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_8x8Block_RoutesThroughBigPlaneArrays()
    {
        // Smoke test that 8x8 routing pulls the right scan / neighbor /
        // prob arrays. Single ONE positive then EOB.
        var read = BlockBitReader(new int[] { 1, 1, 0, 0, 0 });
        var block = new short[64];
        int eob = Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            read, Vp9TxSize.Tx8x8, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Uv, Vp9BlockCoefDecoder.RefType.Inter, block);
        Equal(1, eob);
        // Default 8x8 scan: scan[0] = 0.
        Equal((short)1, block[0]);
        for (int i = 1; i < 64; i++) Equal((short)0, block[i]);
    }

    [TestMethod]
    public void Vp9BlockCoefDecoder_RejectsUndersizedBlock()
    {
        var read = BlockBitReader(new int[] { 0 });
        Throws<ArgumentException>(() =>
            Vp9BlockCoefDecoder.DecodeBlockCoefficients(
                read, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                new short[15]));
    }
}
