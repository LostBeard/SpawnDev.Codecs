// Tests for Vp9TxSizeDecoder (slice 229).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TxSizeDecoder_TxModeToBiggest_MatchesLibvpx()
    {
        var t = Vp9TxSizeDecoder.TxModeToBiggestTxSize;
        Equal(5, t.Length);
        Equal(Vp9TxSize.Tx4x4, t[(int)Vp9TxMode.Only4x4]);
        Equal(Vp9TxSize.Tx8x8, t[(int)Vp9TxMode.AllowOnly8x8]);
        Equal(Vp9TxSize.Tx16x16, t[(int)Vp9TxMode.AllowOnly16x16]);
        Equal(Vp9TxSize.Tx32x32, t[(int)Vp9TxMode.Allow32x32]);
        Equal(Vp9TxSize.Tx32x32, t[(int)Vp9TxMode.TxModeSelect]);
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ForcedMode_ClampsToBlockMax()
    {
        // Block16x16 has max_tx_size = 16; AllowOnly8x8 clamps to 8x8.
        var max = Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block16x16);
        Equal(Vp9TxSize.Tx8x8,
            Vp9TxSizeDecoder.ReadTxSize(Vp9TxMode.AllowOnly8x8, max, null, ReadOnlySpan<byte>.Empty));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ForcedMode_BlockMaxBelowFrameMax()
    {
        // Block8x8 has max_tx_size = 8; Allow32x32 clamps down to 8 (block governs).
        var max = Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block8x8);
        Equal(Vp9TxSize.Tx8x8,
            Vp9TxSizeDecoder.ReadTxSize(Vp9TxMode.Allow32x32, max, null, ReadOnlySpan<byte>.Empty));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ForcedMode_Only4x4_AlwaysReturns4x4()
    {
        // Only4x4 ignores block size entirely.
        Equal(Vp9TxSize.Tx4x4,
            Vp9TxSizeDecoder.ReadTxSize(Vp9TxMode.Only4x4,
                Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block64x64),
                null, ReadOnlySpan<byte>.Empty));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_TxModeSelect_4x4Block_FallsThroughToForcedPath()
    {
        // Block4x4 has max_tx_size = 4x4; TxModeSelect doesn't read,
        // returns clamped (min(4x4, 32x32) = 4x4) without touching reader.
        Equal(Vp9TxSize.Tx4x4,
            Vp9TxSizeDecoder.ReadTxSize(Vp9TxMode.TxModeSelect,
                Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block4x4),
                null, ReadOnlySpan<byte>.Empty));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ReadSelected_Picks4x4_FromZeroBit()
    {
        // probs[0]=128, reader full of zero bits -> reads 0 -> returns 4x4.
        var data = new byte[64];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        byte[] probs = { 128, 128, 128 };
        // ZeroDecoder.Read should return 0 for first bit, picking 4x4.
        Equal(Vp9TxSize.Tx4x4, Vp9TxSizeDecoder.ReadSelectedTxSize(
            Vp9TxSize.Tx32x32, reader, probs));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ReadSelected_RejectsTx4x4Max()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TxSizeDecoder.ReadSelectedTxSize(Vp9TxSize.Tx4x4, reader,
                new byte[] { 128 }));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ReadSelected_RejectsShortProbs()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        // max_tx_size=32x32 needs 3 prob entries, only 2 supplied.
        Throws<ArgumentException>(() =>
            Vp9TxSizeDecoder.ReadSelectedTxSize(Vp9TxSize.Tx32x32, reader,
                new byte[] { 128, 128 }));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_ReadSelected_RejectsNullReader()
    {
        Throws<ArgumentNullException>(() =>
            Vp9TxSizeDecoder.ReadSelectedTxSize(Vp9TxSize.Tx16x16, null!,
                new byte[] { 128, 128 }));
    }

    [TestMethod]
    public void Vp9TxSizeDecoder_TxModeSelect_RequiresReaderFor8x8Plus()
    {
        // Block8x8 with TxModeSelect must read - null reader is an error.
        Throws<ArgumentNullException>(() =>
            Vp9TxSizeDecoder.ReadTxSize(Vp9TxMode.TxModeSelect,
                Vp9MaxTxSize.ForBlockSize(Vp9BlockSize.Block8x8),
                null, ReadOnlySpan<byte>.Empty));
    }
}
