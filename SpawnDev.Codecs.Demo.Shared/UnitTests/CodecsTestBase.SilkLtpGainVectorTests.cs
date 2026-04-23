using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkLtpDecoder.GetGainVector"/> and the underlying
/// <see cref="SilkLtpGainTables"/> codebooks. Verifies the 5-tap Q7 vectors match
/// libopus silk/tables_LTP.c bit-exactly for representative entries in each codebook.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void LtpGainTables_Vq0_HasExpectedShape()
    {
        // 8 entries x 5 taps.
        Equal(40, SilkLtpGainTables.Vq0.Length);
        Equal(5, SilkLtpGainTables.LtpVecSize);
    }

    [TestMethod]
    public void LtpGainTables_Vq1_HasExpectedShape()
    {
        Equal(80, SilkLtpGainTables.Vq1.Length);
    }

    [TestMethod]
    public void LtpGainTables_Vq2_HasExpectedShape()
    {
        Equal(160, SilkLtpGainTables.Vq2.Length);
    }

    // -------- Per-codebook specific-entry checks --------

    [TestMethod]
    public void GetGainVector_Cb0_Entry0_MatchesLibopus()
    {
        // silk_LTP_gain_vq_0[0] = { 4, 6, 24, 7, 5 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 0, ltpIndex: 0);
        EqualSbyteArray(new sbyte[] { 4, 6, 24, 7, 5 }, taps, "Cb0 entry 0");
    }

    [TestMethod]
    public void GetGainVector_Cb0_LastEntry_MatchesLibopus()
    {
        // silk_LTP_gain_vq_0[7] = { 16, 14, 38, -3, 33 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 0, ltpIndex: 7);
        EqualSbyteArray(new sbyte[] { 16, 14, 38, -3, 33 }, taps, "Cb0 entry 7");
    }

    [TestMethod]
    public void GetGainVector_Cb1_Entry0_MatchesLibopus()
    {
        // silk_LTP_gain_vq_1[0] = { 13, 22, 39, 23, 12 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 1, ltpIndex: 0);
        EqualSbyteArray(new sbyte[] { 13, 22, 39, 23, 12 }, taps, "Cb1 entry 0");
    }

    [TestMethod]
    public void GetGainVector_Cb1_LastEntry_MatchesLibopus()
    {
        // silk_LTP_gain_vq_1[15] = { 3, -1, 21, 16, 41 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 1, ltpIndex: 15);
        EqualSbyteArray(new sbyte[] { 3, -1, 21, 16, 41 }, taps, "Cb1 entry 15");
    }

    [TestMethod]
    public void GetGainVector_Cb2_Entry0_MatchesLibopus()
    {
        // silk_LTP_gain_vq_2[0] = { -6, 27, 61, 39, 5 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 2, ltpIndex: 0);
        EqualSbyteArray(new sbyte[] { -6, 27, 61, 39, 5 }, taps, "Cb2 entry 0");
    }

    [TestMethod]
    public void GetGainVector_Cb2_LastEntry_MatchesLibopus()
    {
        // silk_LTP_gain_vq_2[31] = { 2, 0, 9, 10, 88 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 2, ltpIndex: 31);
        EqualSbyteArray(new sbyte[] { 2, 0, 9, 10, 88 }, taps, "Cb2 entry 31");
    }

    // -------- Spot values mid-codebook --------

    [TestMethod]
    public void GetGainVector_Cb2_MidEntry_MatchesLibopus()
    {
        // silk_LTP_gain_vq_2[16] = { -1, 4, 124, 2, -4 }
        sbyte[] taps = new sbyte[5];
        SilkLtpDecoder.GetGainVector(taps, perIndex: 2, ltpIndex: 16);
        EqualSbyteArray(new sbyte[] { -1, 4, 124, 2, -4 }, taps, "Cb2 entry 16");
    }

    // -------- Arg validation --------

    [TestMethod]
    public void GetGainVector_OutputTooSmall_Throws()
    {
        sbyte[] small = new sbyte[4];
        Throws<ArgumentException>(() =>
            SilkLtpDecoder.GetGainVector(small, 0, 0));
    }

    [TestMethod]
    public void GetGainVector_InvalidPerIndex_Throws()
    {
        sbyte[] taps = new sbyte[5];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkLtpDecoder.GetGainVector(taps, perIndex: 3, ltpIndex: 0));
    }

    [TestMethod]
    public void GetGainVector_LtpIndexOutOfRange_Throws()
    {
        sbyte[] taps = new sbyte[5];
        // Cb0 has 8 entries; 8 is out of range.
        Throws<ArgumentOutOfRangeException>(() =>
            SilkLtpDecoder.GetGainVector(taps, perIndex: 0, ltpIndex: 8));
        // Cb1 has 16.
        Throws<ArgumentOutOfRangeException>(() =>
            SilkLtpDecoder.GetGainVector(taps, perIndex: 1, ltpIndex: 16));
        // Cb2 has 32.
        Throws<ArgumentOutOfRangeException>(() =>
            SilkLtpDecoder.GetGainVector(taps, perIndex: 2, ltpIndex: 32));
    }

    // -------- Reference equality --------

    [TestMethod]
    public void LtpGainTables_Select_ReferenceEquality()
    {
        if (!ReferenceEquals(SilkLtpGainTables.Vq0, SilkLtpGainTables.Select(0)))
            throw new Exception("Select(0) should reference Vq0");
        if (!ReferenceEquals(SilkLtpGainTables.Vq1, SilkLtpGainTables.Select(1)))
            throw new Exception("Select(1) should reference Vq1");
        if (!ReferenceEquals(SilkLtpGainTables.Vq2, SilkLtpGainTables.Select(2)))
            throw new Exception("Select(2) should reference Vq2");
    }

    private static void EqualSbyteArray(sbyte[] expected, sbyte[] actual, string label)
    {
        if (expected.Length != actual.Length)
            throw new Exception($"{label}: length mismatch (expected {expected.Length}, got {actual.Length})");
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
                throw new Exception($"{label}: mismatch at [{i}] (expected {expected[i]}, got {actual[i]})");
        }
    }

    private static void EqualSbyteArray(sbyte[] expected, Span<sbyte> actual, string label) =>
        EqualSbyteArray(expected, actual.ToArray(), label);
}
