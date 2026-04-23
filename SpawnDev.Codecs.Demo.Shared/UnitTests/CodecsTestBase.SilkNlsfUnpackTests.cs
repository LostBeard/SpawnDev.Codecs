using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkNlsfUnpack"/> using a synthetic codebook with known
/// <c>ec_sel</c> bytes. Verifies the packed entropy-table index + predictor-variant
/// bits are decoded exactly per the libopus C reference.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Builds a minimal synthetic NLSF codebook for testing NLSF_unpack. Uses
    /// <paramref name="order"/>=10 (NB/MB size). Only the fields Unpack reads
    /// (ec_sel, pred_Q8) have meaningful content; others are zero/empty.
    /// </summary>
    private static SilkNlsfCodebook BuildSyntheticCodebook(short order, byte[] ecSel, byte[] predQ8)
    {
        return new SilkNlsfCodebook
        {
            NVectors = (short)(ecSel.Length / (order / 2)),
            Order = order,
            QuantStepSizeQ16 = 0,
            InvQuantStepSizeQ6 = 0,
            Cb1NlsfQ8 = Array.Empty<byte>(),
            Cb1WghtQ9 = Array.Empty<short>(),
            Cb1Icdf = Array.Empty<byte>(),
            PredQ8 = predQ8,
            EcSel = ecSel,
            EcIcdf = Array.Empty<byte>(),
            EcRatesQ5 = Array.Empty<byte>(),
            DeltaMinQ15 = new short[order + 1],
        };
    }

    [TestMethod]
    public void NlsfUnpack_AllZeros_ProducesZeroIndicesAndPredEntry0()
    {
        // Order=10, 5 ec_sel bytes per vector. All zero ec_sel -> ec_ix all zero.
        // Predictor: (entry & 1) == 0 and (entry >> 4 & 1) == 0, so indices are
        // predQ8[i] and predQ8[i+1] for even i.
        short order = 10;
        byte[] ecSel = new byte[5]; // 1 vector, all zeros
        byte[] predQ8 = new byte[2 * order]; // size generously to cover any synthetic ec_sel pattern
        for (int i = 0; i < predQ8.Length; i++) predQ8[i] = (byte)i;

        var cb = BuildSyntheticCodebook(order, ecSel, predQ8);
        Span<short> ecIx = stackalloc short[10];
        Span<byte> pred = stackalloc byte[10];

        SilkNlsfUnpack.Unpack(ecIx, pred, cb, cb1Index: 0);

        // All ec_ix should be 0.
        for (int i = 0; i < order; i++) Equal((short)0, ecIx[i], $"ecIx[{i}]");

        // predQ8[i] = predQ8[i + 0*9] = predQ8[i] for i=0,2,4,6,8
        // predQ8[i+1] = predQ8[i + 0*9 + 1] = predQ8[i+1] for i=0,2,...
        // So pred[0..9] should equal predQ8[0..9].
        for (int i = 0; i < order; i++) Equal((byte)i, pred[i], $"pred[{i}]");
    }

    [TestMethod]
    public void NlsfUnpack_EntryOne_SelectsPredictorUpperHalf()
    {
        // entry=1 => bit 0 set. Predictor for even i: predQ8[i + 1*(order-1)] = predQ8[i+9].
        // bit 4 not set -> predictor for odd i: predQ8[i + 0*9 + 1] = predQ8[i+1].
        short order = 10;
        byte[] ecSel = new byte[5];
        for (int i = 0; i < ecSel.Length; i++) ecSel[i] = 1;
        byte[] predQ8 = new byte[2 * order];
        for (int i = 0; i < predQ8.Length; i++) predQ8[i] = (byte)(i * 10);

        var cb = BuildSyntheticCodebook(order, ecSel, predQ8);
        Span<short> ecIx = stackalloc short[10];
        Span<byte> pred = stackalloc byte[10];

        SilkNlsfUnpack.Unpack(ecIx, pred, cb, cb1Index: 0);

        // ec_ix: entry=1 -> bits 1..3 == 0 (ecIx[even] = 0), bits 5..7 == 0 (ecIx[odd] = 0).
        for (int i = 0; i < order; i++) Equal((short)0, ecIx[i], $"ecIx[{i}]");

        // Predictor: even indices use upper half: pred[i] = predQ8[i + 9] = (i+9)*10
        //            odd indices use lower half: pred[i] = predQ8[i] * 10 (unchanged)
        // Wait: for even i (i=0,2,4,6,8), bit 0 (of the ec_sel byte shared) is 1.
        //       predQ8[i + 1*(order-1)] = predQ8[i + 9].
        // For odd i, bit 4 of the SAME byte is 0, so predQ8[i + 0*9 + 1] = predQ8[i+1]. But wait,
        //   the "i" in the odd position is the loop variable + 1. Looking at the C code:
        //     predQ8[ i + 1 ] = codebook.PredQ8[ i + (silk_RSHIFT(entry, 4) & 1) * (order - 1) + 1 ];
        //   So the index into predQ8 uses i + 1 via the +1 at the end. For bit4=0: predQ8[i + 1].
        for (int pairI = 0; pairI < order; pairI += 2)
        {
            Equal(predQ8[pairI + 9], pred[pairI], $"pred[{pairI}] upper-half");
            Equal(predQ8[pairI + 1], pred[pairI + 1], $"pred[{pairI + 1}] lower-half");
        }
    }

    [TestMethod]
    public void NlsfUnpack_MaxEntropyIndex_ProducesBoundMultiple()
    {
        // entry=0xFE (bits 1..3 = 7, bits 5..7 = 7). Both ec_ix values should equal 7 * bound.
        short order = 10;
        byte[] ecSel = new byte[5];
        for (int i = 0; i < ecSel.Length; i++) ecSel[i] = 0xFE;
        byte[] predQ8 = new byte[2 * order];

        var cb = BuildSyntheticCodebook(order, ecSel, predQ8);
        Span<short> ecIx = stackalloc short[10];
        Span<byte> pred = stackalloc byte[10];

        SilkNlsfUnpack.Unpack(ecIx, pred, cb, cb1Index: 0);

        int bound = 2 * SilkConstants.NLSF_QUANT_MAX_AMPLITUDE + 1; // 9
        for (int i = 0; i < order; i++)
        {
            Equal((short)(7 * bound), ecIx[i], $"ecIx[{i}]");
        }
    }

    [TestMethod]
    public void NlsfUnpack_CodebookIndexOutOfRange_Throws()
    {
        short order = 10;
        var cb = BuildSyntheticCodebook(order, new byte[5], new byte[20]);
        short[] ecIx = new short[10];
        byte[] pred = new byte[10];

        Throws<ArgumentOutOfRangeException>(() => SilkNlsfUnpack.Unpack(ecIx, pred, cb, -1));
        Throws<ArgumentOutOfRangeException>(() => SilkNlsfUnpack.Unpack(ecIx, pred, cb, 1)); // NVectors=1
    }

    [TestMethod]
    public void NlsfUnpack_OutputBuffersTooSmall_Throws()
    {
        short order = 10;
        var cb = BuildSyntheticCodebook(order, new byte[5], new byte[20]);

        short[] shortEcIx = new short[5];
        byte[] pred = new byte[10];
        Throws<ArgumentException>(() => SilkNlsfUnpack.Unpack(shortEcIx, pred, cb, 0));

        short[] ecIx = new short[10];
        byte[] shortPred = new byte[5];
        Throws<ArgumentException>(() => SilkNlsfUnpack.Unpack(ecIx, shortPred, cb, 0));
    }

    [TestMethod]
    public void NlsfUnpack_SecondCodebookVector_UsesCorrectEcSelOffset()
    {
        // 2 vectors of order 10 -> 2 * 5 = 10 ec_sel bytes.
        // Vector 0: all zeros. Vector 1: all 0x02 (entropy bits 1..3 = 1).
        short order = 10;
        byte[] ecSel = new byte[10];
        for (int i = 5; i < 10; i++) ecSel[i] = 0x02;
        byte[] predQ8 = new byte[2 * order];

        var cb = BuildSyntheticCodebook(order, ecSel, predQ8);
        Span<short> ecIx = stackalloc short[10];
        Span<byte> pred = stackalloc byte[10];

        // Second codebook vector -> ec_ix for even positions should be 1 * bound = 9.
        SilkNlsfUnpack.Unpack(ecIx, pred, cb, cb1Index: 1);
        int bound = 2 * SilkConstants.NLSF_QUANT_MAX_AMPLITUDE + 1;
        for (int i = 0; i < order; i += 2)
        {
            Equal((short)bound, ecIx[i], $"vec=1 ecIx[{i}]");
            Equal((short)0, ecIx[i + 1], $"vec=1 ecIx[{i + 1}]");
        }
    }
}
