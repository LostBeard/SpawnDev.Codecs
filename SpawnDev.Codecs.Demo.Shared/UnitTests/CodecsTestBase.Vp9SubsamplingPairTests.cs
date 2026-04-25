// Tests for Vp9SubsamplingPair (slice 265).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SubsamplingPair_NamedSingletons()
    {
        Equal(0, Vp9SubsamplingPair.Yuv444.SubsamplingX);
        Equal(0, Vp9SubsamplingPair.Yuv444.SubsamplingY);
        Equal(true, Vp9SubsamplingPair.Yuv444.Is444);
        Equal(false, Vp9SubsamplingPair.Yuv444.Is420);

        Equal(1, Vp9SubsamplingPair.Yuv422.SubsamplingX);
        Equal(0, Vp9SubsamplingPair.Yuv422.SubsamplingY);

        Equal(0, Vp9SubsamplingPair.Yuv440.SubsamplingX);
        Equal(1, Vp9SubsamplingPair.Yuv440.SubsamplingY);

        Equal(1, Vp9SubsamplingPair.Yuv420.SubsamplingX);
        Equal(1, Vp9SubsamplingPair.Yuv420.SubsamplingY);
        Equal(true, Vp9SubsamplingPair.Yuv420.Is420);
        Equal(false, Vp9SubsamplingPair.Yuv420.Is444);
    }

    [TestMethod]
    public void Vp9SubsamplingPair_Yuv420_HalvesBothAxes()
    {
        Equal(640, Vp9SubsamplingPair.Yuv420.ChromaWidth(1280));
        Equal(360, Vp9SubsamplingPair.Yuv420.ChromaHeight(720));
    }

    [TestMethod]
    public void Vp9SubsamplingPair_Yuv422_HalvesHorizontalOnly()
    {
        Equal(640, Vp9SubsamplingPair.Yuv422.ChromaWidth(1280));
        Equal(720, Vp9SubsamplingPair.Yuv422.ChromaHeight(720));
    }

    [TestMethod]
    public void Vp9SubsamplingPair_Yuv444_PreservesBothAxes()
    {
        Equal(1280, Vp9SubsamplingPair.Yuv444.ChromaWidth(1280));
        Equal(720, Vp9SubsamplingPair.Yuv444.ChromaHeight(720));
    }

    [TestMethod]
    public void Vp9SubsamplingPair_Yuv440_HalvesVerticalOnly()
    {
        Equal(1280, Vp9SubsamplingPair.Yuv440.ChromaWidth(1280));
        Equal(360, Vp9SubsamplingPair.Yuv440.ChromaHeight(720));
    }

    [TestMethod]
    public void Vp9SubsamplingPair_ChromaPixelCount_Yuv420()
    {
        // 1280x720 -> 640x360 = 230400 chroma pixels.
        Equal(230400, Vp9SubsamplingPair.Yuv420.ChromaPixelCount(1280, 720));
    }

    [TestMethod]
    public void Vp9SubsamplingPair_RecordEquality()
    {
        Equal(Vp9SubsamplingPair.Yuv420, new Vp9SubsamplingPair(1, 1));
        Equal(false, Vp9SubsamplingPair.Yuv420 == Vp9SubsamplingPair.Yuv422);
    }
}
