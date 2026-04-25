// Tests for Vp9MvComponentReader (slice 238).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvComponentReader_Constants_MatchLibvpx()
    {
        Equal(1, Vp9MvComponentReader.Class0Bits);
    }

    [TestMethod]
    public void Vp9MvComponentReader_AllZeroBits_IsSmallestPositive()
    {
        // sign=0, class tree all zeros -> Class0,
        //   d=0 (one zero bit), fp all zeros -> Fp0, hp implicit 1.
        // mag = 0 + ((0 << 3) | (0 << 1) | 1) + 1 = 2
        var probs = NewProbs();
        int mv = Vp9MvComponentReader.ReadComponent(
            BuildReader(new int[20]), // all zeros
            probs,
            useHp: false);
        Equal(2, mv);
    }

    [TestMethod]
    public void Vp9MvComponentReader_Class0_Hp_AllZero_Magnitude1()
    {
        // sign=0, class=Class0, d=0, fp=Fp0, hp=0 -> mag = ((0<<3)|(0<<1)|0)+1 = 1
        var probs = NewProbs();
        int mv = Vp9MvComponentReader.ReadComponent(
            BuildReader(new int[20]),
            probs,
            useHp: true);
        Equal(1, mv);
    }

    [TestMethod]
    public void Vp9MvComponentReader_SignBit_Negates()
    {
        // sign=1 -> negate. d=0, fr=0, hp=1 (no HP) -> mag = 2 -> return -2.
        var probs = NewProbs();
        int[] bits = new int[20];
        bits[0] = 1; // sign
        int mv = Vp9MvComponentReader.ReadComponent(
            BuildReader(bits), probs, useHp: false);
        Equal(-2, mv);
    }

    [TestMethod]
    public void Vp9MvComponentReader_Class1_NonClass0Path()
    {
        // sign=0, class tree: 1, 0 -> Class1.
        // class != Class0 -> read n = 1 + CLASS0_BITS = 2 bits as offset.
        // bits[0]=0, bits[1]=0 -> d = 0
        // mag = CLASS0_SIZE << (Class1 + 2) = 2 << 3 = 16
        // fp tree all zeros -> Fp0 (3 bits read)
        // hp implicit 1
        // result = 16 + ((0 << 3) | (0 << 1) | 1) + 1 = 16 + 1 + 1 = 18
        var probs = NewProbs();
        int[] bits = new int[20];
        // 0: sign=0
        // 1: class root bit = 1 -> right
        // 2: class i2 bit = 0 -> -Class1 (leaf)
        bits[1] = 1;
        // 3: offset bit 0 = 0
        // 4: offset bit 1 = 0
        // 5,6,7: fp tree zeros -> Fp0
        int mv = Vp9MvComponentReader.ReadComponent(
            BuildReader(bits), probs, useHp: false);
        Equal(18, mv);
    }

    [TestMethod]
    public void Vp9MvComponentReader_RejectsNullReader()
    {
        var probs = NewProbs();
        Throws<ArgumentNullException>(() =>
            Vp9MvComponentReader.ReadComponent((Vp9BoolDecoder)null!, probs, false));
    }

    [TestMethod]
    public void Vp9MvComponentReader_RejectsNullProbs()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvComponentReader.ReadComponent(p => 0, null!, false));
    }

    [TestMethod]
    public void Vp9MvComponentReader_DelegateOverload_MatchesBoolDecoder()
    {
        // Smoke test: when the bool decoder reads a bitstream of all
        // zero bytes, the prob comparison matches "always 0" output;
        // verify the delegate path agrees with the bool-decoder path.
        // (Exhaustive equality is hard since Vp9BoolDecoder semantics
        // depend on full arithmetic coding; just check both APIs are
        // exercisable without exception.)
        var probs = NewProbs();
        var data = new byte[64];
        var decoder = new Vp9BoolDecoder(data, 0, data.Length);
        int mvDecoder = Vp9MvComponentReader.ReadComponent(decoder, probs, false);
        int mvDelegate = Vp9MvComponentReader.ReadComponent(p => 0, probs, false);
        // Both should produce the smallest positive MV (= 2) since neither
        // bit reader will trip a probability comparison that flips a leaf.
        Equal(mvDelegate, mvDecoder);
    }

    private static Vp9MvComponentProbs NewProbs()
    {
        var p = new Vp9MvComponentProbs();
        // Default to 128 for all probs - irrelevant for delegate-driven
        // tests since the delegate ignores the prob byte entirely.
        p.Sign = 128;
        for (int i = 0; i < p.Classes.Length; i++) p.Classes[i] = 128;
        p.Class0 = 128;
        for (int i = 0; i < p.Bits.Length; i++) p.Bits[i] = 128;
        for (int i = 0; i < p.Class0Fp.GetLength(0); i++)
            for (int j = 0; j < p.Class0Fp.GetLength(1); j++)
                p.Class0Fp[i, j] = 128;
        for (int i = 0; i < p.Fp.Length; i++) p.Fp[i] = 128;
        p.Class0Hp = 128;
        p.Hp = 128;
        return p;
    }

    private static Func<byte, int> BuildReader(int[] bits)
    {
        int idx = 0;
        return _ => idx < bits.Length ? bits[idx++] : 0;
    }
}
