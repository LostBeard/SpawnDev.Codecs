// Tests for Vp9TxMode + Vp9CompressedHeader.ReadTxMode (slice 150).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a literal reader that returns a queue of values in order.
    /// Each call dequeues one value (regardless of the bit-count
    /// argument; the test arrays are constructed to align with the
    /// real call sequence ReadTxMode performs).
    /// </summary>
    private static Func<int, uint> ScriptedLiteralReader(uint[] values)
    {
        int idx = 0;
        return _ => values[idx++];
    }

    [TestMethod]
    public void Vp9TxMode_EnumValuesMatchLibvpxOrdering()
    {
        Equal((byte)0, (byte)Vp9TxMode.Only4x4);
        Equal((byte)1, (byte)Vp9TxMode.AllowOnly8x8);
        Equal((byte)2, (byte)Vp9TxMode.AllowOnly16x16);
        Equal((byte)3, (byte)Vp9TxMode.Allow32x32);
        Equal((byte)4, (byte)Vp9TxMode.TxModeSelect);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_LosslessForcesOnly4x4()
    {
        // Lossless: no bits should be read; the literal reader is
        // never invoked. We pass a reader that throws if called to
        // assert this contract.
        Func<int, uint> failingReader = _ => throw new InvalidOperationException(
            "literal reader called for lossless frame");
        var mode = Vp9CompressedHeader.ReadTxMode(failingReader, isLossless: true);
        Equal(Vp9TxMode.Only4x4, mode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_LiteralZeroProducesOnly4x4()
    {
        // Non-lossless, ReadLiteral(2) = 0 -> tx_mode = Only4x4. No
        // extension bit read.
        var read = ScriptedLiteralReader(new uint[] { 0 });
        var mode = Vp9CompressedHeader.ReadTxMode(read, isLossless: false);
        Equal(Vp9TxMode.Only4x4, mode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_LiteralOneProducesAllowOnly8x8()
    {
        var read = ScriptedLiteralReader(new uint[] { 1 });
        var mode = Vp9CompressedHeader.ReadTxMode(read, isLossless: false);
        Equal(Vp9TxMode.AllowOnly8x8, mode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_LiteralTwoProducesAllowOnly16x16()
    {
        var read = ScriptedLiteralReader(new uint[] { 2 });
        var mode = Vp9CompressedHeader.ReadTxMode(read, isLossless: false);
        Equal(Vp9TxMode.AllowOnly16x16, mode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_LiteralThreeWithExtensionZeroProducesAllow32x32()
    {
        // ReadLiteral(2) = 3, then ReadLiteral(1) = 0 -> 3 + 0 = Allow32x32.
        var read = ScriptedLiteralReader(new uint[] { 3, 0 });
        var mode = Vp9CompressedHeader.ReadTxMode(read, isLossless: false);
        Equal(Vp9TxMode.Allow32x32, mode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_LiteralThreeWithExtensionOneProducesTxModeSelect()
    {
        // ReadLiteral(2) = 3, then ReadLiteral(1) = 1 -> 3 + 1 = TxModeSelect.
        var read = ScriptedLiteralReader(new uint[] { 3, 1 });
        var mode = Vp9CompressedHeader.ReadTxMode(read, isLossless: false);
        Equal(Vp9TxMode.TxModeSelect, mode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_ReadTxMode_BoolDecoderOverload_AllZeroBufferProducesOnly4x4()
    {
        // End-to-end through the real Vp9BoolDecoder: an all-zero
        // buffer drives ReadLiteral(2) to return 0, so the result
        // must be Only4x4.
        var buf = new byte[64];
        var d = new Vp9BoolDecoder(buf, 0, buf.Length);
        var mode = Vp9CompressedHeader.ReadTxMode(d, isLossless: false);
        Equal(Vp9TxMode.Only4x4, mode);
    }
}
