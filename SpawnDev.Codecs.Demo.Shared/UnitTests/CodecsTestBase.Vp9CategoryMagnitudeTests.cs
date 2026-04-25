// Tests for Vp9CoefProbs.DecodeCategoryMagnitude (slice 146).
// Verifies the cat<N> residual-bit decode produces the expected
// integer magnitude across the full per-category range.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a Func&lt;byte,int&gt; that returns each bit from
    /// <paramref name="bits"/> in order, ignoring the probability
    /// argument (test-only - the probability is what guides a real
    /// arithmetic decoder, but for unit testing the magnitude
    /// decoder we just need a deterministic bit source).
    /// </summary>
    private static Func<byte, int> ConstantBitReader(int[] bits)
    {
        int idx = 0;
        return _ => bits[idx++];
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_CatMinValConstantsMatchSpec()
    {
        Equal(5,  Vp9CoefProbs.CatMinVal.Cat1);
        Equal(7,  Vp9CoefProbs.CatMinVal.Cat2);
        Equal(11, Vp9CoefProbs.CatMinVal.Cat3);
        Equal(19, Vp9CoefProbs.CatMinVal.Cat4);
        Equal(35, Vp9CoefProbs.CatMinVal.Cat5);
        Equal(67, Vp9CoefProbs.CatMinVal.Cat6);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category1_BitZeroProducesMin()
    {
        var read = ConstantBitReader(new int[] { 0 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category1);
        Equal(5, v); // CAT1_MIN_VAL + 0
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category1_BitOneProducesMax()
    {
        var read = ConstantBitReader(new int[] { 1 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category1);
        Equal(6, v); // CAT1_MIN_VAL + 1
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category2_TwoBitsMsbFirst()
    {
        // bits [1, 0] MSB-first -> value 2 -> magnitude = 7 + 2 = 9.
        var read = ConstantBitReader(new int[] { 1, 0 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category2);
        Equal(9, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category2_AllBitsZeroProducesMin()
    {
        var read = ConstantBitReader(new int[] { 0, 0 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category2);
        Equal(7, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category2_AllBitsOneProducesMax()
    {
        var read = ConstantBitReader(new int[] { 1, 1 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category2);
        Equal(10, v); // 7 + 3
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category3_BitsAreMsbFirst()
    {
        // bits [1, 0, 1] MSB-first -> 5 -> 11 + 5 = 16.
        var read = ConstantBitReader(new int[] { 1, 0, 1 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category3);
        Equal(16, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category4_AllOnesProducesMax()
    {
        // 4 bits all ones -> 15 -> 19 + 15 = 34.
        var read = ConstantBitReader(new int[] { 1, 1, 1, 1 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category4);
        Equal(34, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category5_AllOnesProducesMax()
    {
        // 5 bits all ones -> 31 -> 35 + 31 = 66.
        var read = ConstantBitReader(new int[] { 1, 1, 1, 1, 1 });
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category5);
        Equal(66, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category6_8BitProfile_Reads14Bits()
    {
        // 14 bits = 0b10000000000001 = 8193 -> 67 + 8193 = 8260.
        var bits = new int[14] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        var read = ConstantBitReader(bits);
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category6, isHighBitDepth: false);
        Equal(67 + 8193, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_Category6_HighBitDepth_Reads18Bits()
    {
        // 18 bits = 0b100000000000000001 = 131073 -> 67 + 131073 = 131140.
        var bits = new int[18] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        var read = ConstantBitReader(bits);
        int v = Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Category6, isHighBitDepth: true);
        Equal(67 + 131073, v);
    }

    [TestMethod]
    public void Vp9CategoryMagnitude_RejectsNonCategoryTokens()
    {
        var read = ConstantBitReader(new int[] { 0 });
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Zero));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.One));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9CoefProbs.DecodeCategoryMagnitude(read, Vp9CoefToken.Eob));
    }
}
