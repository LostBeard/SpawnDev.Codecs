// Tests for Vp9NeighborContexts (slice 230).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothMissing()
    {
        Equal(0, Vp9NeighborContexts.GetSkipContext(null, null));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_OnlyOneSide_Skipped()
    {
        Equal(1, Vp9NeighborContexts.GetSkipContext(true, null));
        Equal(1, Vp9NeighborContexts.GetSkipContext(null, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothPresent_NeitherSkipped()
    {
        Equal(0, Vp9NeighborContexts.GetSkipContext(false, false));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothPresent_OneSkipped()
    {
        Equal(1, Vp9NeighborContexts.GetSkipContext(true, false));
        Equal(1, Vp9NeighborContexts.GetSkipContext(false, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_Skip_BothPresent_BothSkipped()
    {
        Equal(2, Vp9NeighborContexts.GetSkipContext(true, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_BothMissing()
    {
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(null, null));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_OneEdge_Inter()
    {
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(false, null));
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(null, false));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_OneEdge_Intra()
    {
        Equal(2, Vp9NeighborContexts.GetIntraInterContext(true, null));
        Equal(2, Vp9NeighborContexts.GetIntraInterContext(null, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_BothInter()
    {
        Equal(0, Vp9NeighborContexts.GetIntraInterContext(false, false));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_OneIntraOneInter()
    {
        Equal(1, Vp9NeighborContexts.GetIntraInterContext(true, false));
        Equal(1, Vp9NeighborContexts.GetIntraInterContext(false, true));
    }

    [TestMethod]
    public void Vp9NeighborContexts_IntraInter_BothIntra()
    {
        Equal(3, Vp9NeighborContexts.GetIntraInterContext(true, true));
    }
}
