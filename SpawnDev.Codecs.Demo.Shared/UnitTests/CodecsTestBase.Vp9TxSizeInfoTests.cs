// Tests for Vp9TxSizeInfo (slice 266).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TxSizeInfo_Constants_MatchLibvpx()
    {
        Equal(4, Vp9TxSizeInfo.TxSizes);
        Equal(4, Vp9TxSizeInfo.SideLength.Length);
        Equal(4, Vp9TxSizeInfo.CoefCounts.Length);
    }

    [TestMethod]
    public void Vp9TxSizeInfo_Side_AllSizes()
    {
        Equal(4, Vp9TxSizeInfo.Side(Vp9TxSize.Tx4x4));
        Equal(8, Vp9TxSizeInfo.Side(Vp9TxSize.Tx8x8));
        Equal(16, Vp9TxSizeInfo.Side(Vp9TxSize.Tx16x16));
        Equal(32, Vp9TxSizeInfo.Side(Vp9TxSize.Tx32x32));
    }

    [TestMethod]
    public void Vp9TxSizeInfo_CoefCount_AllSizes()
    {
        Equal(16, Vp9TxSizeInfo.CoefCount(Vp9TxSize.Tx4x4));
        Equal(64, Vp9TxSizeInfo.CoefCount(Vp9TxSize.Tx8x8));
        Equal(256, Vp9TxSizeInfo.CoefCount(Vp9TxSize.Tx16x16));
        Equal(1024, Vp9TxSizeInfo.CoefCount(Vp9TxSize.Tx32x32));
    }

    [TestMethod]
    public void Vp9TxSizeInfo_Log2Side_AllSizes()
    {
        Equal(2, Vp9TxSizeInfo.Log2Side(Vp9TxSize.Tx4x4));
        Equal(3, Vp9TxSizeInfo.Log2Side(Vp9TxSize.Tx8x8));
        Equal(4, Vp9TxSizeInfo.Log2Side(Vp9TxSize.Tx16x16));
        Equal(5, Vp9TxSizeInfo.Log2Side(Vp9TxSize.Tx32x32));
    }

    [TestMethod]
    public void Vp9TxSizeInfo_CoefCount_EqualsSideSquared()
    {
        for (int i = 0; i < Vp9TxSizeInfo.TxSizes; i++)
        {
            int side = Vp9TxSizeInfo.SideLength[i];
            Equal(side * side, Vp9TxSizeInfo.CoefCounts[i]);
        }
    }

    [TestMethod]
    public void Vp9TxSizeInfo_Side_RejectsOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TxSizeInfo.Side((Vp9TxSize)99));
    }

    [TestMethod]
    public void Vp9TxSizeInfo_CoefCount_RejectsOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TxSizeInfo.CoefCount((Vp9TxSize)99));
    }
}
