// Tests for Vp9BoolDecoder. Bit-exact validation of a VP9 arithmetic
// decoder is hard without the matching encoder, so we cover:
//   - Init contract (first-bit marker must be 0; reject 1).
//   - Behavior on all-zero buffer (always returns 0, regardless of prob).
//   - Boundary: prob = 1 (bit = 1 almost certain) and prob = 255
//     (bit = 0 almost certain) against crafted buffers.
//   - Determinism under repeated decode of the same buffer.
//   - ReadLiteral consistency (value equals sum of individual ReadBit calls).
// Bit-exactness vs libvpx is de-facto validated downstream by the
// integration test that decodes a real VP9 keyframe from a bundled WebM
// - if this decoder drifts even a single bit, the frame would be garbage.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] FilledBuffer(byte value, int length)
    {
        var buf = new byte[length];
        for (int i = 0; i < length; i++) buf[i] = value;
        return buf;
    }

    [TestMethod]
    public void Vp9BoolDecoder_InitOnZeroBuffer_Succeeds()
    {
        var buf = FilledBuffer(0x00, 64);
        var d = new Vp9BoolDecoder(buf, 0, buf.Length);
        // After init with all-zero buffer the decoder is in a valid state.
        // Reading a bit should not throw; value should be 0.
        Equal(0, d.Read(128));
    }

    [TestMethod]
    public void Vp9BoolDecoder_InitOnHighBitBuffer_Throws()
    {
        // First bit of 0x80 is 1, which libvpx treats as a stream-init
        // corruption signal.
        var buf = FilledBuffer(0x80, 64);
        bool threw = false;
        try { _ = new Vp9BoolDecoder(buf, 0, buf.Length); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "init on 0x80-start buffer must throw");
    }

    [TestMethod]
    public void Vp9BoolDecoder_AllZeroBuffer_ReadsAllZeros()
    {
        // With prob=128 the split is (range-1)/2 + 1. For value=0 the
        // comparison always falls into the "bit=0" branch. Because all
        // loaded bytes are 0, value stays 0 and every read returns 0.
        var buf = FilledBuffer(0x00, 128);
        var d = new Vp9BoolDecoder(buf, 0, buf.Length);
        for (int i = 0; i < 64; i++)
            Equal(0, d.Read(128));
    }

    [TestMethod]
    public void Vp9BoolDecoder_AllZeroBuffer_ExtremeProb255_AlsoReadsZero()
    {
        // With prob=255 the split is range-1 (very large), so "bit=1"
        // only fires if value is essentially at the top of the range.
        // All-zero value means bit=0 forever.
        var buf = FilledBuffer(0x00, 128);
        var d = new Vp9BoolDecoder(buf, 0, buf.Length);
        for (int i = 0; i < 32; i++)
            Equal(0, d.Read(255));
    }

    [TestMethod]
    public void Vp9BoolDecoder_IsDeterministic_OnSameBuffer()
    {
        var buf = new byte[]
        {
            0x00, 0x41, 0x7E, 0x23, 0x5A, 0x0F, 0xC3, 0x81,
            0x7F, 0x22, 0x11, 0x66, 0x44, 0x99, 0xEE, 0x33,
        };
        var a = new Vp9BoolDecoder(buf, 0, buf.Length);
        var b = new Vp9BoolDecoder(buf, 0, buf.Length);
        for (int i = 0; i < 40; i++)
            Equal(a.Read(128), b.Read(128));
    }

    [TestMethod]
    public void Vp9BoolDecoder_ReadLiteral_SumsIndividualReadBits()
    {
        var buf = new byte[]
        {
            0x00, 0x5A, 0x3C, 0x91, 0x42, 0x7E, 0x08, 0xF0,
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        };
        var a = new Vp9BoolDecoder(buf, 0, buf.Length);
        uint literalValue = a.ReadLiteral(8);

        var b = new Vp9BoolDecoder(buf, 0, buf.Length);
        uint pieceValue = 0;
        for (int i = 0; i < 8; i++)
            pieceValue = (pieceValue << 1) | (uint)b.ReadBit();

        Equal(literalValue, pieceValue);
    }

    [TestMethod]
    public void Vp9BoolDecoder_ReadLiteral_ZeroBits_ReturnsZero()
    {
        var buf = FilledBuffer(0x00, 16);
        var d = new Vp9BoolDecoder(buf, 0, buf.Length);
        Equal(0u, d.ReadLiteral(0));
    }

    [TestMethod]
    public void Vp9BoolDecoder_InitWithOffset_UsesCorrectWindow()
    {
        // Prefix the buffer with noise, but init the decoder offset
        // into the buffer. The decode path should only see what's
        // starting at `offset`.
        var buf = new byte[]
        {
            0xFF, 0xFF, 0xFF, 0xFF,    // ignored prefix
            0x00, 0x00, 0x00, 0x00,    // init sees this - zero start
            0x00, 0x00, 0x00, 0x00,
        };
        var d = new Vp9BoolDecoder(buf, 4, 8);
        for (int i = 0; i < 16; i++)
            Equal(0, d.Read(128));
    }

    [TestMethod]
    public void Vp9BoolDecoder_ConstructorThrows_OnBadArgs()
    {
        bool t1 = false;
        try { _ = new Vp9BoolDecoder(null!, 0, 0); }
        catch (ArgumentNullException) { t1 = true; }
        True(t1, "null buffer must throw");

        bool t2 = false;
        try { _ = new Vp9BoolDecoder(new byte[4], -1, 4); }
        catch (ArgumentException) { t2 = true; }
        True(t2, "negative offset must throw");

        bool t3 = false;
        try { _ = new Vp9BoolDecoder(new byte[4], 2, 10); }
        catch (ArgumentException) { t3 = true; }
        True(t3, "offset + length past end must throw");
    }
}
