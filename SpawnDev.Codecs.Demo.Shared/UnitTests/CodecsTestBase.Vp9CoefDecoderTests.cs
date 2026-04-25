// Tests for Vp9CoefDecoder.DecodeOneCoefficient (slice 147). Drives
// the full decode flow via deterministic bit sequences and verifies
// the (token, signed value) result for every branch of the decision
// tree (EOB / ZERO / ONE / TWO / THREE / FOUR / CAT1..CAT6).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static Func<byte, int> ScriptedBitReader(int[] bits)
    {
        int idx = 0;
        return _ => bits[idx++];
    }

    /// <summary>
    /// Build an arbitrary 11-entry full prob vector. The actual
    /// values don't influence DecodeOneCoefficient's logic when the
    /// scripted reader ignores the probability argument.
    /// </summary>
    private static byte[] DummyFullProbs() => new byte[]
    {
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    };

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_EobBranchReturnsEobZeroValue()
    {
        // First bit = 0 -> EOB.
        var read = ScriptedBitReader(new int[] { 0 });
        var r = Vp9CoefDecoder.DecodeOneCoefficient(read, DummyFullProbs());
        Equal(Vp9CoefToken.Eob, r.Token);
        Equal(0, r.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_ZeroBranchReturnsZeroToken()
    {
        // EOB? = 1 (not EOB), ZERO? = 0 -> Zero token, value 0.
        var read = ScriptedBitReader(new int[] { 1, 0 });
        var r = Vp9CoefDecoder.DecodeOneCoefficient(read, DummyFullProbs());
        Equal(Vp9CoefToken.Zero, r.Token);
        Equal(0, r.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_OneBranchPositive_ReturnsPlusOne()
    {
        // EOB?=1, ZERO?=1, ONE?=0 (->One), sign=0 (positive).
        var read = ScriptedBitReader(new int[] { 1, 1, 0, 0 });
        var r = Vp9CoefDecoder.DecodeOneCoefficient(read, DummyFullProbs());
        Equal(Vp9CoefToken.One, r.Token);
        Equal(1, r.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_OneBranchNegative_ReturnsMinusOne()
    {
        // ...ONE?=0, sign=1 -> -1.
        var read = ScriptedBitReader(new int[] { 1, 1, 0, 1 });
        var r = Vp9CoefDecoder.DecodeOneCoefficient(read, DummyFullProbs());
        Equal(Vp9CoefToken.One, r.Token);
        Equal(-1, r.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_TwoBranch_PositiveAndNegative()
    {
        // ...ONE?=1 (constrained tree), tree i=0 bit 0 -> i=2 (TWO),
        //    tree i=2 bit 0 -> -Two leaf, sign 0 -> +2.
        var readPos = ScriptedBitReader(new int[] { 1, 1, 1, 0, 0, 0 });
        var rp = Vp9CoefDecoder.DecodeOneCoefficient(readPos, DummyFullProbs());
        Equal(Vp9CoefToken.Two, rp.Token);
        Equal(2, rp.Value);

        var readNeg = ScriptedBitReader(new int[] { 1, 1, 1, 0, 0, 1 });
        var rn = Vp9CoefDecoder.DecodeOneCoefficient(readNeg, DummyFullProbs());
        Equal(Vp9CoefToken.Two, rn.Token);
        Equal(-2, rn.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_ThreeAndFourBranches()
    {
        // ONE=1 -> tree i=0 bit 0 -> i=2 -> bit 1 -> i=4 (THREE) -> bit 0 -> -Three.
        var readThree = ScriptedBitReader(new int[] { 1, 1, 1, 0, 1, 0, 0 });
        var rt = Vp9CoefDecoder.DecodeOneCoefficient(readThree, DummyFullProbs());
        Equal(Vp9CoefToken.Three, rt.Token);
        Equal(3, rt.Value);

        // ...same path then bit 1 -> -Four leaf, sign 1 -> -4.
        var readFour = ScriptedBitReader(new int[] { 1, 1, 1, 0, 1, 1, 1 });
        var rf = Vp9CoefDecoder.DecodeOneCoefficient(readFour, DummyFullProbs());
        Equal(Vp9CoefToken.Four, rf.Token);
        Equal(-4, rf.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_Category1_BothMagnitudes()
    {
        // ONE=1 -> tree bit 1 -> i=6 -> bit 0 -> i=8 (CAT_ONE) -> bit 0 -> -Cat1.
        // Cat1 reads 1 residual bit. Cat1 magnitudes are 5 (residual 0) or 6 (residual 1).
        // Then sign bit.
        // Path with residual 0, sign 0 -> +5.
        var readPlus5 = ScriptedBitReader(new int[] { 1, 1, 1, 1, 0, 0, 0, 0 });
        var r5 = Vp9CoefDecoder.DecodeOneCoefficient(readPlus5, DummyFullProbs());
        Equal(Vp9CoefToken.Category1, r5.Token);
        Equal(5, r5.Value);

        // Path with residual 1, sign 1 -> -6.
        var readMinus6 = ScriptedBitReader(new int[] { 1, 1, 1, 1, 0, 0, 1, 1 });
        var r6 = Vp9CoefDecoder.DecodeOneCoefficient(readMinus6, DummyFullProbs());
        Equal(Vp9CoefToken.Category1, r6.Token);
        Equal(-6, r6.Value);
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_Category6_LargeMagnitudeProducesExpectedValue()
    {
        // Path: ONE=1 -> tree 1 (HIGH_LOW) -> 1 (CAT_THREEFOUR) ->
        //               1 (CAT_FIVE) -> 1 -> -Cat6 leaf.
        // 5 prefix bits to navigate the tree:  [ONE?=1, tree:1, 1, 1, 1] = 5 bits + sign.
        // Then 14 residual bits all 1 -> 16383 -> 67 + 16383 = 16450.
        // Then sign 0 -> +16450.
        var bits = new int[] {
            1, 1, 1,  // EOB?=1, ZERO?=1, ONE?=1 (NOT ONE)
            1, 1, 1, 1, // tree path to CAT_FIVE -> -Cat6
        };
        var residual = new int[14];
        for (int i = 0; i < 14; i++) residual[i] = 1;
        var combined = new int[bits.Length + residual.Length + 1]; // +1 for sign
        Array.Copy(bits, combined, bits.Length);
        Array.Copy(residual, 0, combined, bits.Length, residual.Length);
        combined[combined.Length - 1] = 0; // positive sign

        var read = ScriptedBitReader(combined);
        var r = Vp9CoefDecoder.DecodeOneCoefficient(read, DummyFullProbs());
        Equal(Vp9CoefToken.Category6, r.Token);
        Equal(67 + 16383, r.Value); // 16450
    }

    [TestMethod]
    public void Vp9CoefDecoder_DecodeOne_RejectsUndersizedProbsVector()
    {
        var read = ScriptedBitReader(new int[] { 0 });
        Throws<ArgumentException>(() =>
            Vp9CoefDecoder.DecodeOneCoefficient(read, new byte[10]));
    }
}
