// Tests for Vp9SizeGroupLookup (slice 227).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9SizeGroupLookup_Constants_MatchLibvpx()
    {
        CpuOnlyTest();
        Equal(4, Vp9SizeGroupLookup.Groups);
        Equal(13, Vp9SizeGroupLookup.Lookup.Length);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SizeGroupLookup_Group0()
    {
        CpuOnlyTest();
        Equal(0, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block4x4));
        Equal(0, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block4x8));
        Equal(0, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block8x4));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SizeGroupLookup_Group1()
    {
        CpuOnlyTest();
        Equal(1, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block8x8));
        Equal(1, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block8x16));
        Equal(1, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block16x8));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SizeGroupLookup_Group2()
    {
        CpuOnlyTest();
        Equal(2, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block16x16));
        Equal(2, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block16x32));
        Equal(2, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block32x16));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SizeGroupLookup_Group3()
    {
        CpuOnlyTest();
        Equal(3, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block32x32));
        Equal(3, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block32x64));
        Equal(3, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block64x32));
        Equal(3, Vp9SizeGroupLookup.ForBlockSize(Vp9BlockSize.Block64x64));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SizeGroupLookup_RejectsOutOfRange()
    {
        CpuOnlyTest();
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SizeGroupLookup.ForBlockSize((Vp9BlockSize)99));
    }
}
