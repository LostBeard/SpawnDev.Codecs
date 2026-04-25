// Tests for Vp9ChromaBlockSize (slice 261).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ChromaBlockSize_Lookup_Length13()
    {
        Equal(13, Vp9ChromaBlockSize.For420.Length);
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_SubEightBlocks_AllMapTo4x4()
    {
        Equal(Vp9BlockSize.Block4x4, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block4x4));
        Equal(Vp9BlockSize.Block4x4, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block4x8));
        Equal(Vp9BlockSize.Block8x4, Vp9ChromaBlockSize.For420[(int)Vp9BlockSize.Block8x4] == Vp9BlockSize.Block4x4 ?
            Vp9BlockSize.Block8x4 : Vp9BlockSize.Block8x4);
        Equal(Vp9BlockSize.Block4x4, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block8x4));
        Equal(Vp9BlockSize.Block4x4, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block8x8));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_64x64_MapsTo32x32()
    {
        Equal(Vp9BlockSize.Block32x32, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block64x64));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_16x16_MapsTo8x8()
    {
        Equal(Vp9BlockSize.Block8x8, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block16x16));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_NonSquare_PreservesAspectRatio()
    {
        Equal(Vp9BlockSize.Block16x32, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block32x64));
        Equal(Vp9BlockSize.Block32x16, Vp9ChromaBlockSize.ForLumaBlock(Vp9BlockSize.Block64x32));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_GetChromaTxSize_SubEightBlock_Force4x4()
    {
        // Sub-8x8 luma always uses Tx4x4 chroma regardless of luma tx_size.
        Equal(Vp9TxSize.Tx4x4, Vp9ChromaBlockSize.GetChromaTxSize(
            Vp9TxSize.Tx32x32, Vp9BlockSize.Block4x4));
        Equal(Vp9TxSize.Tx4x4, Vp9ChromaBlockSize.GetChromaTxSize(
            Vp9TxSize.Tx16x16, Vp9BlockSize.Block4x8));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_GetChromaTxSize_8x8_ChromaIs4x4_TxClampedTo4x4()
    {
        // Block8x8 luma -> Block4x4 chroma -> max chroma tx = Tx4x4.
        // Even if luma tx is 8x8, chroma is clamped to 4x4.
        Equal(Vp9TxSize.Tx4x4, Vp9ChromaBlockSize.GetChromaTxSize(
            Vp9TxSize.Tx8x8, Vp9BlockSize.Block8x8));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_GetChromaTxSize_16x16_ChromaIs8x8_TxClampedTo8x8()
    {
        // Block16x16 luma -> Block8x8 chroma -> max chroma tx = Tx8x8.
        Equal(Vp9TxSize.Tx8x8, Vp9ChromaBlockSize.GetChromaTxSize(
            Vp9TxSize.Tx16x16, Vp9BlockSize.Block16x16));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_GetChromaTxSize_64x64_ChromaIs32x32()
    {
        // Luma 64x64 -> chroma 32x32 (max chroma tx = 32x32).
        Equal(Vp9TxSize.Tx32x32, Vp9ChromaBlockSize.GetChromaTxSize(
            Vp9TxSize.Tx32x32, Vp9BlockSize.Block64x64));
        // Smaller luma tx -> chroma matches.
        Equal(Vp9TxSize.Tx16x16, Vp9ChromaBlockSize.GetChromaTxSize(
            Vp9TxSize.Tx16x16, Vp9BlockSize.Block64x64));
    }

    [TestMethod]
    public void Vp9ChromaBlockSize_RejectsOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9ChromaBlockSize.ForLumaBlock((Vp9BlockSize)99));
    }
}
