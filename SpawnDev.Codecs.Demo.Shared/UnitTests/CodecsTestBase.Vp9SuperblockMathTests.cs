// Tests for Vp9SuperblockMath (slice 262).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SuperblockMath_Constants_MatchLibvpx()
    {
        Equal(3, Vp9SuperblockMath.MiSizeLog2);
        Equal(8, Vp9SuperblockMath.MiSize);
        Equal(3, Vp9SuperblockMath.MiPerSbLog2);
        Equal(8, Vp9SuperblockMath.MiPerSb);
        Equal(6, Vp9SuperblockMath.SbSizeLog2);
        Equal(64, Vp9SuperblockMath.SbSize);
    }

    [TestMethod]
    public void Vp9SuperblockMath_PixelMiRoundtrip()
    {
        Equal(0, Vp9SuperblockMath.PixelToMi(0));
        Equal(0, Vp9SuperblockMath.PixelToMi(7));
        Equal(1, Vp9SuperblockMath.PixelToMi(8));
        Equal(1, Vp9SuperblockMath.PixelToMi(15));
        Equal(2, Vp9SuperblockMath.PixelToMi(16));

        Equal(0, Vp9SuperblockMath.MiToPixel(0));
        Equal(8, Vp9SuperblockMath.MiToPixel(1));
        Equal(64, Vp9SuperblockMath.MiToPixel(8));
    }

    [TestMethod]
    public void Vp9SuperblockMath_MiToSb()
    {
        Equal(0, Vp9SuperblockMath.MiToSb(0));
        Equal(0, Vp9SuperblockMath.MiToSb(7));
        Equal(1, Vp9SuperblockMath.MiToSb(8));
        Equal(2, Vp9SuperblockMath.MiToSb(16));
    }

    [TestMethod]
    public void Vp9SuperblockMath_AlignToSbPixels()
    {
        Equal(0, Vp9SuperblockMath.AlignToSbPixels(0));
        Equal(64, Vp9SuperblockMath.AlignToSbPixels(1));
        Equal(64, Vp9SuperblockMath.AlignToSbPixels(64));
        Equal(128, Vp9SuperblockMath.AlignToSbPixels(65));
        Equal(1280, Vp9SuperblockMath.AlignToSbPixels(1280));
        Equal(1344, Vp9SuperblockMath.AlignToSbPixels(1281));
    }

    [TestMethod]
    public void Vp9SuperblockMath_AlignToSbMi()
    {
        Equal(0, Vp9SuperblockMath.AlignToSbMi(0));
        Equal(8, Vp9SuperblockMath.AlignToSbMi(1));
        Equal(8, Vp9SuperblockMath.AlignToSbMi(8));
        Equal(16, Vp9SuperblockMath.AlignToSbMi(9));
        Equal(160, Vp9SuperblockMath.AlignToSbMi(153));
    }

    [TestMethod]
    public void Vp9SuperblockMath_SbsForPixels()
    {
        Equal(0, Vp9SuperblockMath.SbsForPixels(0));
        Equal(1, Vp9SuperblockMath.SbsForPixels(1));
        Equal(1, Vp9SuperblockMath.SbsForPixels(64));
        Equal(2, Vp9SuperblockMath.SbsForPixels(65));
        // 1280x720 (HD ready): 1280/64 = 20, 720/64 = 11.25 -> 12 rows.
        Equal(20, Vp9SuperblockMath.SbsForPixels(1280));
        Equal(12, Vp9SuperblockMath.SbsForPixels(720));
    }

    [TestMethod]
    public void Vp9SuperblockMath_SbsForMi()
    {
        Equal(0, Vp9SuperblockMath.SbsForMi(0));
        Equal(1, Vp9SuperblockMath.SbsForMi(1));
        Equal(1, Vp9SuperblockMath.SbsForMi(8));
        Equal(2, Vp9SuperblockMath.SbsForMi(9));
        Equal(2, Vp9SuperblockMath.SbsForMi(16));
    }
}
