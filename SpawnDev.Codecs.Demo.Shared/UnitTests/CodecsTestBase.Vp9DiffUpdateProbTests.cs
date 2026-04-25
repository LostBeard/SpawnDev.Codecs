// Tests for Vp9DiffUpdateProb (slice 210). The pure-function
// helpers (InvRecenterNonneg, InvRemapProb, InvMapTable) get
// independent coverage here. The full Read() round-trip needs a
// BoolDecoder driven by a hand-encoded buffer; covered by
// targeted-pattern tests.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9DiffUpdateProb_Constants_MatchLibvpx()
    {
        Equal(252, Vp9DiffUpdateProb.UpdateProb);
        Equal(255, Vp9DiffUpdateProb.MaxProb);
        Equal(255, Vp9DiffUpdateProb.InvMapTable.Length);
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvMapTable_FirstFewEntries()
    {
        // First 19 entries spread early codes across the prob range.
        Equal((byte)7, Vp9DiffUpdateProb.InvMapTable[0]);
        Equal((byte)20, Vp9DiffUpdateProb.InvMapTable[1]);
        Equal((byte)33, Vp9DiffUpdateProb.InvMapTable[2]);
        Equal((byte)254, Vp9DiffUpdateProb.InvMapTable[19]);
        // Then linear fill from 1 starting at index 20.
        Equal((byte)1, Vp9DiffUpdateProb.InvMapTable[20]);
        Equal((byte)2, Vp9DiffUpdateProb.InvMapTable[21]);
        // Last entry is a duplicate of 253.
        Equal((byte)253, Vp9DiffUpdateProb.InvMapTable[253]);
        Equal((byte)253, Vp9DiffUpdateProb.InvMapTable[254]);
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRecenterNonneg_OverflowBranch()
    {
        // v > 2m -> return v unchanged.
        Equal(100, Vp9DiffUpdateProb.InvRecenterNonneg(100, 30));  // 100 > 60
        Equal(50, Vp9DiffUpdateProb.InvRecenterNonneg(50, 10));    // 50 > 20
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRecenterNonneg_OddBranch()
    {
        // v <= 2m, v odd -> m - ((v+1) >> 1).
        // v=1, m=10: 10 - 1 = 9.
        Equal(9, Vp9DiffUpdateProb.InvRecenterNonneg(1, 10));
        // v=3, m=10: 10 - 2 = 8.
        Equal(8, Vp9DiffUpdateProb.InvRecenterNonneg(3, 10));
        // v=5, m=10: 10 - 3 = 7.
        Equal(7, Vp9DiffUpdateProb.InvRecenterNonneg(5, 10));
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRecenterNonneg_EvenBranch()
    {
        // v <= 2m, v even -> m + (v >> 1).
        // v=0, m=10: 10 + 0 = 10.
        Equal(10, Vp9DiffUpdateProb.InvRecenterNonneg(0, 10));
        // v=2, m=10: 10 + 1 = 11.
        Equal(11, Vp9DiffUpdateProb.InvRecenterNonneg(2, 10));
        // v=4, m=10: 10 + 2 = 12.
        Equal(12, Vp9DiffUpdateProb.InvRecenterNonneg(4, 10));
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRemapProb_LowerHalf()
    {
        // m <= MAX_PROB/2 + 0.5 -> use 1 + InvRecenterNonneg(v, m-1).
        // v=0 maps to InvMapTable[0]=7.
        // m=100, m-1=99. (99<<1)=198 <= 255. Branch = 1 + InvRecenterNonneg(7, 99).
        // 7 <= 198? yes. 7 odd -> 99 - 4 = 95. Return 1 + 95 = 96.
        Equal(96, Vp9DiffUpdateProb.InvRemapProb(0, 100));
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRemapProb_UpperHalf()
    {
        // m=200, m-1=199. (199<<1)=398 > 255. Branch = MaxProb - InvRecenterNonneg(v, MaxProb-1-m+1).
        // Wait the formula is MAX_PROB - InvRecenterNonneg(v, MAX_PROB - 1 - m).
        // After m-- in the function, m is now 199. So we use MAX_PROB - 1 - 199 = 55.
        // v=0 -> InvMapTable[0]=7. InvRecenterNonneg(7, 55): 7 <= 110? yes. 7 odd -> 55 - 4 = 51.
        // Return 255 - 51 = 204.
        Equal(204, Vp9DiffUpdateProb.InvRemapProb(0, 200));
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRemapProb_Symmetry()
    {
        // For v=0 (smallest delta), the new prob should be near current m.
        // Quick spot check: m=128 (boundary).
        // m-1=127. (127<<1)=254 <= 255 (yes, the lower-half branch).
        // v=0 -> InvMapTable[0]=7. InvRecenterNonneg(7, 127): 7 <= 254. 7 odd -> 127-4 = 123.
        // Return 1 + 123 = 124.
        Equal(124, Vp9DiffUpdateProb.InvRemapProb(0, 128));
    }

    [TestMethod]
    public void Vp9DiffUpdateProb_InvRemapProb_RejectsOutOfRangeV()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DiffUpdateProb.InvRemapProb(255, 100));  // v == MaxProb
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DiffUpdateProb.InvRemapProb(-1, 100));
    }
}
