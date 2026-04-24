using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacSubframeDecoder"/>. Covers all 4 subframe kinds
/// (CONSTANT, VERBATIM, FIXED orders 0-4, LPC orders 1-32), wasted-bits
/// re-inflation, and the reconstruction math that produces the final PCM
/// samples from warm-up + residual.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Invokes the internal subframe decoder from a freshly-built bit stream.</summary>
    private static int[] RunSubframeDecode(byte[] data, int blockSize, int subframeBps)
    {
        var samples = new int[blockSize];
        var r = new FlacBitReader(data);
        FlacSubframeDecoder.Decode(ref r, samples, subframeBps);
        return samples;
    }

    /// <summary>Write a subframe header (reserved + type + wasted flag + optional unary).</summary>
    private static void WriteSubframeHeader(FlacBitWriter w, FlacSubframeKind kind, int order, int wastedBits)
    {
        int typeCode = kind switch
        {
            FlacSubframeKind.Constant => 0,
            FlacSubframeKind.Verbatim => 1,
            FlacSubframeKind.Fixed => 0b001000 | order,
            FlacSubframeKind.Lpc => 0b100000 | (order - 1),
            _ => throw new ArgumentException(nameof(kind)),
        };
        w.Write(0, 1);                   // reserved
        w.Write((uint)typeCode, 6);      // type code
        if (wastedBits == 0)
        {
            w.Write(0, 1);
        }
        else
        {
            w.Write(1, 1);
            // Unary: wastedBits-1 zeros + terminator.
            for (int i = 0; i < wastedBits - 1; i++) w.Write(0, 1);
            w.Write(1, 1);
        }
    }

    /// <summary>Write a signed value in two's complement at a given bit width.</summary>
    private static void WriteSigned(FlacBitWriter w, int value, int bits)
    {
        // Mask the low `bits` bits from the two's-complement representation.
        uint mask = bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1);
        uint raw = (uint)value & mask;
        w.Write(raw, bits);
    }

    // -------- CONSTANT --------

    [TestMethod]
    public void FlacSubframe_Constant_Positive_Replicates()
    {
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w, 42, 16);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 5, subframeBps: 16);
        EqualInts(new[] { 42, 42, 42, 42, 42 }, samples);
    }

    [TestMethod]
    public void FlacSubframe_Constant_Negative_Replicates()
    {
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, 0);
        WriteSigned(w, -12345, 16);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 3, subframeBps: 16);
        EqualInts(new[] { -12345, -12345, -12345 }, samples);
    }

    // -------- VERBATIM --------

    [TestMethod]
    public void FlacSubframe_Verbatim_MixedSamples()
    {
        var expected = new[] { 10, -10, 20, -20, int.MaxValue >> 16 };
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, 0);
        foreach (var v in expected) WriteSigned(w, v, 16);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: expected.Length, subframeBps: 16);
        EqualInts(expected, samples);
    }

    // -------- FIXED --------

    [TestMethod]
    public void FlacSubframe_FixedOrder0_ResidualIsSample()
    {
        // Order 0: predictor = 0, so samples = residual.
        var expected = new[] { 1, 2, -1, 3, -3, 0 };
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Fixed, 0, 0);
        // No warm-up; go straight to Rice.
        WriteRiceHeader(w, 0, 0);
        w.Write(3, 4); // k=3
        foreach (var v in expected) WriteRiceCoded(w, v, 3);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: expected.Length, subframeBps: 16);
        EqualInts(expected, samples);
    }

    [TestMethod]
    public void FlacSubframe_FixedOrder1_Reconstructs()
    {
        // Target samples: [5, 7, 10, 8, 15].
        // Warm-up = 5. Residuals = [s[n] - s[n-1]] = [2, 3, -2, 7].
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Fixed, 1, 0);
        WriteSigned(w, 5, 8); // warm-up
        WriteRiceHeader(w, 0, 0);
        w.Write(2, 4); // k=2
        foreach (var r in new[] { 2, 3, -2, 7 }) WriteRiceCoded(w, r, 2);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 5, subframeBps: 8);
        EqualInts(new[] { 5, 7, 10, 8, 15 }, samples);
    }

    [TestMethod]
    public void FlacSubframe_FixedOrder2_Reconstructs()
    {
        // Target samples: [3, 6, 10, 15, 21, 28]
        // Differences (order 1): [3, 4, 5, 6, 7]
        // Second differences (order 2): [1, 1, 1, 1]
        // So warm-ups = 3, 6; residuals = [1, 1, 1, 1].
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Fixed, 2, 0);
        WriteSigned(w, 3, 8);
        WriteSigned(w, 6, 8);
        WriteRiceHeader(w, 0, 0);
        w.Write(1, 4); // k=1
        foreach (var r in new[] { 1, 1, 1, 1 }) WriteRiceCoded(w, r, 1);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 6, subframeBps: 8);
        EqualInts(new[] { 3, 6, 10, 15, 21, 28 }, samples);
    }

    [TestMethod]
    public void FlacSubframe_FixedOrder4_Reconstructs()
    {
        // Target: sample[n] = n*n*n (cubic). 4th difference of a cubic is 0, so residuals = 0.
        // samples = [0, 1, 8, 27, 64, 125, 216, 343].
        var expected = new int[8];
        for (int i = 0; i < 8; i++) expected[i] = i * i * i;
        // Warm-ups = first 4 samples. Residuals = 4 zeros.
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Fixed, 4, 0);
        for (int i = 0; i < 4; i++) WriteSigned(w, expected[i], 16);
        WriteRiceHeader(w, 0, 0);
        w.Write(0, 4); // k=0
        for (int i = 0; i < 4; i++) WriteRiceCoded(w, 0, 0);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 8, subframeBps: 16);
        EqualInts(expected, samples);
    }

    // -------- LPC --------

    [TestMethod]
    public void FlacSubframe_Lpc_Order1_Decays()
    {
        // Classic decay: coef = 1, quant level = 1.
        // predictor = (1 * prev) >> 1 = prev / 2 (integer floor).
        // With warm-up = 100, residual = [0, 0, 0, 0], samples decay:
        // s[0]=100, s[1]=50, s[2]=25, s[3]=12, s[4]=6.
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Lpc, 1, 0);
        WriteSigned(w, 100, 16);          // warm-up
        w.Write(3, 4);                    // precision - 1 = 3, so precision = 4 bits
        WriteSigned(w, 1, 5);             // quant level = 1
        WriteSigned(w, 1, 4);             // 1 coefficient = 1 at 4-bit signed
        WriteRiceHeader(w, 0, 0);
        w.Write(0, 4);                    // k = 0
        for (int i = 0; i < 4; i++) WriteRiceCoded(w, 0, 0);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 5, subframeBps: 16);
        EqualInts(new[] { 100, 50, 25, 12, 6 }, samples);
    }

    [TestMethod]
    public void FlacSubframe_Lpc_Order2_MatchesHandComputed()
    {
        // Coefs = [1, 1], quant = 1, precision = 4.
        // predictor = (1*prev + 1*prev2) >> 1.
        // Warm-ups: s[0]=10, s[1]=6. Residuals: [1, -1, 0].
        // Manual trace:
        //   n=2: pred = (1*6 + 1*10) >> 1 = 16 >> 1 = 8; s[2] = 1 + 8 = 9
        //   n=3: pred = (1*9 + 1*6) >> 1 = 15 >> 1 = 7; s[3] = -1 + 7 = 6
        //   n=4: pred = (1*6 + 1*9) >> 1 = 15 >> 1 = 7; s[4] = 0 + 7 = 7
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Lpc, 2, 0);
        WriteSigned(w, 10, 16);
        WriteSigned(w, 6, 16);
        w.Write(3, 4);                    // precision - 1 = 3 => 4-bit coefs
        WriteSigned(w, 1, 5);             // quant = 1
        WriteSigned(w, 1, 4);             // coef[0]
        WriteSigned(w, 1, 4);             // coef[1]
        WriteRiceHeader(w, 0, 0);
        w.Write(1, 4);                    // k = 1
        foreach (var r in new[] { 1, -1, 0 }) WriteRiceCoded(w, r, 1);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 5, subframeBps: 16);
        EqualInts(new[] { 10, 6, 9, 6, 7 }, samples);
    }

    [TestMethod]
    public void FlacSubframe_Lpc_NegativeQuantLevel_Throws()
    {
        // libFLAC treats negative quant level as invalid (encoder never produces them).
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Lpc, 1, 0);
        WriteSigned(w, 0, 16);
        w.Write(3, 4);                    // precision - 1 = 3
        WriteSigned(w, -1, 5);            // negative quant level - should throw
        WriteSigned(w, 0, 4);             // coef
        WriteRiceHeader(w, 0, 0);
        w.Write(0, 4);
        var data = w.ToArray();
        bool threw = false;
        try { RunSubframeDecode(data, blockSize: 2, subframeBps: 16); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "Negative LPC quant level should throw.");
    }

    [TestMethod]
    public void FlacSubframe_Lpc_ReservedPrecision1111_Throws()
    {
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Lpc, 1, 0);
        WriteSigned(w, 0, 16);
        w.Write(0b1111, 4);               // reserved precision
        WriteSigned(w, 0, 5);
        var data = w.ToArray();
        bool threw = false;
        try { RunSubframeDecode(data, blockSize: 2, subframeBps: 16); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "LPC precision 0b1111 should throw.");
    }

    // -------- Wasted bits --------

    [TestMethod]
    public void FlacSubframe_Constant_WithWastedBits_LeftShifts()
    {
        // 3 wasted bits → decoder reads value at (bps - 3), then left-shifts by 3.
        // bps = 16, effective bps = 13. Value 5 at 13 bits, final = 5 << 3 = 40.
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Constant, 0, wastedBits: 3);
        WriteSigned(w, 5, 13);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 4, subframeBps: 16);
        EqualInts(new[] { 40, 40, 40, 40 }, samples);
    }

    [TestMethod]
    public void FlacSubframe_Verbatim_WithWastedBits_LeftShifts()
    {
        // bps=16, wasted=4, effective bps = 12. Samples -100, 50 → -1600, 800 after shift.
        var w = new FlacBitWriter();
        WriteSubframeHeader(w, FlacSubframeKind.Verbatim, 0, wastedBits: 4);
        WriteSigned(w, -100, 12);
        WriteSigned(w, 50, 12);
        var samples = RunSubframeDecode(w.ToArray(), blockSize: 2, subframeBps: 16);
        EqualInts(new[] { -1600, 800 }, samples);
    }
}
