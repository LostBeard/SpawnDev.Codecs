// Round-trip tests for Vp9BlockCoefEncoder. The encoder is the
// bit-exact mirror of Vp9BlockCoefDecoder - a successful encode ->
// bool-stream -> decode -> compare cycle proves the two halves agree
// on the wire format end-to-end.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] EncodeVp9CoefBlock(
        Vp9TxSize txSize,
        Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        ReadOnlySpan<short> block)
    {
        var enc = new Vp9BoolEncoder();
        Vp9BlockCoefEncoder.EncodeBlockCoefficients(
            (prob, bit) => enc.Write(bit, prob),
            txSize, scanType, planeType, refType,
            block);
        return enc.Stop();
    }

    private static int DecodeVp9CoefBlock(
        byte[] encoded,
        Vp9TxSize txSize,
        Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        Span<short> block)
    {
        var dec = new Vp9BoolDecoder(encoded, 0, encoded.Length);
        return Vp9BlockCoefDecoder.DecodeBlockCoefficients(
            prob => dec.Read(prob),
            txSize, scanType, planeType, refType,
            block);
    }

    private static void RoundTripCoefBlock(
        Vp9TxSize txSize, int n,
        Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        ReadOnlySpan<short> input, int expectedEob)
    {
        byte[] encoded = EncodeVp9CoefBlock(txSize, scanType, planeType, refType, input);
        var decoded = new short[n];
        int eob = DecodeVp9CoefBlock(encoded, txSize, scanType, planeType, refType, decoded);
        Equal(expectedEob, eob);
        for (int i = 0; i < n; i++)
        {
            if (input[i] != decoded[i])
                throw new Exception($"mismatch at raster {i}: input {input[i]} decoded {decoded[i]}, eob={eob}");
        }
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_AllZero4x4_RoundTrips()
    {
        var input = new short[16];
        RoundTripCoefBlock(
            Vp9TxSize.Tx4x4, 16, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            input, expectedEob: 0);
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_DcOnly4x4_RoundTrips()
    {
        var input = new short[16];
        input[0] = 17;
        RoundTripCoefBlock(
            Vp9TxSize.Tx4x4, 16, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            input, expectedEob: 1);
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_NegativeOne4x4_RoundTrips()
    {
        var input = new short[16];
        input[0] = -1;
        RoundTripCoefBlock(
            Vp9TxSize.Tx4x4, 16, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            input, expectedEob: 1);
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_AllMagnitudes4x4_RoundTrips()
    {
        // Cover every code path (One, Two, Three, Four, Cat1..Cat6) by
        // assigning magnitudes to scan positions 0..9 so eob = 10 exactly,
        // independent of the raster-vs-scan mapping.
        var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4, Vp9ScanType.Default);
        short[] mags = { 1, -2, 3, -4, 6, -10, 18, -34, 66, -67 };

        var input = new short[16];
        for (int s = 0; s < mags.Length; s++) input[scan[s]] = mags[s];

        RoundTripCoefBlock(
            Vp9TxSize.Tx4x4, 16, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            input, expectedEob: 10);
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_RandomSparse8x8_RoundTrips()
    {
        var rng = new Random(0x8888);
        for (int trial = 0; trial < 4; trial++)
        {
            var input = new short[64];
            // Sparse: ~20% non-zero, magnitudes in [-50, 50].
            int eob = 0;
            for (int i = 0; i < 64; i++)
            {
                if (rng.NextDouble() < 0.2)
                {
                    input[i] = (short)rng.Next(-50, 51);
                }
            }
            for (int i = 63; i >= 0; i--)
            {
                if (input[Vp9ScanTables.GetScan(Vp9TxSize.Tx8x8, Vp9ScanType.Default)[i]] != 0)
                {
                    eob = i + 1;
                    break;
                }
            }
            RoundTripCoefBlock(
                Vp9TxSize.Tx8x8, 64, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                input, expectedEob: eob);
        }
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_Plane_RefType_Variants_RoundTrip()
    {
        // Cover all 4 (planeType, refType) combinations - each maps to a
        // different probability subtable so the bit stream differs but
        // the round-trip must still hold.
        // Non-zero at raster 0 (= scan[0]) and raster 5 (= scan[3] in default scan).
        // That puts the last non-zero at scan position 3 -> eob = 4.
        var input = new short[16];
        input[0] = 7;
        input[5] = -3;
        foreach (Vp9BlockCoefDecoder.PlaneType plane in (Vp9BlockCoefDecoder.PlaneType[])Enum.GetValues(typeof(Vp9BlockCoefDecoder.PlaneType)))
        {
            foreach (Vp9BlockCoefDecoder.RefType refT in (Vp9BlockCoefDecoder.RefType[])Enum.GetValues(typeof(Vp9BlockCoefDecoder.RefType)))
            {
                RoundTripCoefBlock(
                    Vp9TxSize.Tx4x4, 16, Vp9ScanType.Default,
                    plane, refT,
                    input, expectedEob: 4);
            }
        }
    }

    [TestMethod]
    public void Vp9BlockCoefEncoder_Random16x16_RoundTrips()
    {
        var rng = new Random(0x1616);
        var input = new short[256];
        for (int i = 0; i < 256; i++)
        {
            if (rng.NextDouble() < 0.15)
                input[i] = (short)rng.Next(-200, 201);
        }
        int eob = 0;
        var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx16x16, Vp9ScanType.Default);
        for (int i = 255; i >= 0; i--)
        {
            if (input[scan[i]] != 0) { eob = i + 1; break; }
        }
        RoundTripCoefBlock(
            Vp9TxSize.Tx16x16, 256, Vp9ScanType.Default,
            Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
            input, expectedEob: eob);
    }
}
