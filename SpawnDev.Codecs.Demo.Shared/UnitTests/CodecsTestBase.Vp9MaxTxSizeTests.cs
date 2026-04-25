// Tests for Vp9MaxTxSize (slice 225).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MaxTxSize_Lookup_HasThirteenEntries()
    {
        Equal(13, Vp9MaxTxSize.Lookup.Length);
    }

    [TestMethod]
    public void Vp9MaxTxSize_SmallSizes_AreTx4x4()
    {
        Equal(Vp9TxSize.Tx4x4, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block4x4));
        Equal(Vp9TxSize.Tx4x4, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block4x8));
        Equal(Vp9TxSize.Tx4x4, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block8x4));
    }

    [TestMethod]
    public void Vp9MaxTxSize_8x8Group_IsTx8x8()
    {
        Equal(Vp9TxSize.Tx8x8, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block8x8));
        Equal(Vp9TxSize.Tx8x8, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block8x16));
        Equal(Vp9TxSize.Tx8x8, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block16x8));
    }

    [TestMethod]
    public void Vp9MaxTxSize_16x16Group_IsTx16x16()
    {
        Equal(Vp9TxSize.Tx16x16, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block16x16));
        Equal(Vp9TxSize.Tx16x16, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block16x32));
        Equal(Vp9TxSize.Tx16x16, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block32x16));
    }

    [TestMethod]
    public void Vp9MaxTxSize_32x32AndLarger_AreTx32x32()
    {
        Equal(Vp9TxSize.Tx32x32, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block32x32));
        Equal(Vp9TxSize.Tx32x32, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block32x64));
        Equal(Vp9TxSize.Tx32x32, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block64x32));
        Equal(Vp9TxSize.Tx32x32, Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block64x64));
    }

    [TestMethod]
    public void Vp9MaxTxSize_RejectsOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MaxTxSize.ForBlockSize((Vp9BlockSize)99));
    }
}
