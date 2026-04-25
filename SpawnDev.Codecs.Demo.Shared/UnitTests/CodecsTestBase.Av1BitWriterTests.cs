// Av1BitWriter direct tests - lock the MSB-first bit-packing semantics
// the SH / FH writers rely on. Reads back via Av1BitReader to verify
// each emit-then-read pair round-trips.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1BitWriter_EmptyAfterTrailingBits_IsOneByte()
    {
        var bw = new Av1BitWriter();
        bw.WriteTrailingBits();
        var bytes = bw.ToArray();
        Equal(1, bytes.Length);
        Equal(0x80, (int)bytes[0]); // single trailing-1 in bit 7 + zero pad
    }

    [TestMethod]
    public void Av1BitWriter_SingleByteThroughTrailing_PreservesBits()
    {
        var bw = new Av1BitWriter();
        bw.WriteBits(0xA5, 8); // 10100101
        bw.WriteTrailingBits();
        var bytes = bw.ToArray();
        Equal(2, bytes.Length);
        Equal(0xA5, (int)bytes[0]);
        Equal(0x80, (int)bytes[1]); // trailing-1 then zero pad
    }

    [TestMethod]
    public void Av1BitWriter_MixedFlags_ReadBackInOrder()
    {
        var bw = new Av1BitWriter();
        bw.WriteFlag(true);
        bw.WriteFlag(false);
        bw.WriteFlag(true);
        bw.WriteBits(5, 3);   // 101
        bw.WriteFlag(false);
        bw.WriteBits(7, 4);   // 0111
        bw.WriteTrailingBits();

        var bytes = bw.ToArray();
        var br = new Av1BitReader(bytes);
        Equal(true, br.ReadFlag());
        Equal(false, br.ReadFlag());
        Equal(true, br.ReadFlag());
        Equal((uint)5, br.ReadBits(3));
        Equal(false, br.ReadFlag());
        Equal((uint)7, br.ReadBits(4));
    }

    [TestMethod]
    public void Av1BitWriter_LargeMultiByteField_ReadBack()
    {
        var bw = new Av1BitWriter();
        // 24-bit value: 0xABCDEF.
        bw.WriteBits(0xABCDEF, 24);
        bw.WriteTrailingBits();
        var bytes = bw.ToArray();
        Equal(4, bytes.Length); // 24 bits + trailing -> 4 bytes
        Equal(0xAB, (int)bytes[0]);
        Equal(0xCD, (int)bytes[1]);
        Equal(0xEF, (int)bytes[2]);

        var br = new Av1BitReader(bytes);
        Equal((uint)0xABCDEF, br.ReadBits(24));
    }

    [TestMethod]
    public void Av1BitWriter_RejectsValueOutOfRange()
    {
        var bw = new Av1BitWriter();
        // 5 fits in 3 bits but 8 does not.
        Throws<ArgumentException>(() => bw.WriteBits(8, 3));
        // Negative bit count is invalid.
        Throws<ArgumentOutOfRangeException>(() => bw.WriteBits(0, -1));
        // > 32 bit count is invalid.
        Throws<ArgumentOutOfRangeException>(() => bw.WriteBits(0, 33));
    }

    [TestMethod]
    public void Av1BitWriter_ZeroBitsIsNoop()
    {
        var bw = new Av1BitWriter();
        bw.WriteFlag(true);
        bw.WriteBits(0, 0);
        bw.WriteFlag(false);
        bw.WriteTrailingBits();

        var br = new Av1BitReader(bw.ToArray());
        Equal(true, br.ReadFlag());
        Equal(false, br.ReadFlag());
    }

    [TestMethod]
    public void Av1BitWriter_PositionTracksBitsWritten()
    {
        var bw = new Av1BitWriter();
        Equal(0, bw.BitPosition);
        bw.WriteFlag(true);
        Equal(1, bw.BitPosition);
        bw.WriteBits(0, 7);
        Equal(8, bw.BitPosition);
        bw.WriteBits(0, 16);
        Equal(24, bw.BitPosition);
    }

    [TestMethod]
    public void Av1BitWriter_ToArrayWithoutTrailing_Throws()
    {
        var bw = new Av1BitWriter();
        bw.WriteFlag(true);
        // Position is 1, not byte-aligned. ToArray must reject.
        Throws<InvalidOperationException>(() => bw.ToArray());
    }
}
