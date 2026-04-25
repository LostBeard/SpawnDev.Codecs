// Tests for Vp9MbModeInfo (slice 251).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MbModeInfo_Intra_IsIntraTrue()
    {
        var info = new Vp9MbModeInfo
        {
            BlockSize = Vp9BlockSize.Block16x16,
            PrimaryRefFrame = Vp9MvReferenceFrame.Intra,
            YMode = Vp9IntraMode.DcPred,
            UvMode = Vp9IntraMode.HPred,
            TxSize = Vp9TxSize.Tx8x8,
            Skip = false,
            SegmentId = 0,
        };
        Equal(true, info.IsIntra);
        Equal(false, info.IsCompound);
        Equal(Vp9IntraMode.DcPred, info.YMode!.Value);
        Equal(Vp9IntraMode.HPred, info.UvMode!.Value);
    }

    [TestMethod]
    public void Vp9MbModeInfo_InterSingleRef_IsIntraFalse_NotCompound()
    {
        var info = new Vp9MbModeInfo
        {
            BlockSize = Vp9BlockSize.Block32x32,
            PrimaryRefFrame = Vp9MvReferenceFrame.Last,
            InterMode = Vp9InterMode.NewMv,
            TxSize = Vp9TxSize.Tx16x16,
            Skip = false,
            SegmentId = 2,
            PrimaryMv = new Vp9Mv(8, -8),
        };
        Equal(false, info.IsIntra);
        Equal(false, info.IsCompound);
        Equal(Vp9MvReferenceFrame.Last, info.PrimaryRefFrame);
        Equal(8, info.PrimaryMv.Row);
        Equal(-8, info.PrimaryMv.Col);
    }

    [TestMethod]
    public void Vp9MbModeInfo_InterCompound_IsCompoundTrue()
    {
        var info = new Vp9MbModeInfo
        {
            BlockSize = Vp9BlockSize.Block16x16,
            PrimaryRefFrame = Vp9MvReferenceFrame.Last,
            CompoundRefFrame = Vp9MvReferenceFrame.AltRef,
            InterMode = Vp9InterMode.ZeroMv,
            TxSize = Vp9TxSize.Tx16x16,
            Skip = false,
            SegmentId = 0,
            PrimaryMv = new Vp9Mv(0, 0),
            CompoundMv = new Vp9Mv(4, 4),
        };
        Equal(false, info.IsIntra);
        Equal(true, info.IsCompound);
        Equal(Vp9MvReferenceFrame.AltRef, info.CompoundRefFrame!.Value);
    }

    [TestMethod]
    public void Vp9MbModeInfo_DefaultsForOmittedFields()
    {
        var info = new Vp9MbModeInfo
        {
            BlockSize = Vp9BlockSize.Block8x8,
            PrimaryRefFrame = Vp9MvReferenceFrame.Last,
            TxSize = Vp9TxSize.Tx4x4,
            Skip = true,
            SegmentId = 7,
        };
        Equal(Vp9Mv.Zero, info.PrimaryMv);
        Equal(Vp9Mv.Zero, info.CompoundMv);
        Equal(Vp9InterpFilter.EightTap, info.InterpFilter);
        Equal(false, info.IsCompound);
        Equal(false, info.SegmentIdPredicted);
    }

    [TestMethod]
    public void Vp9MbModeInfo_RecordEquality()
    {
        var a = new Vp9MbModeInfo
        {
            BlockSize = Vp9BlockSize.Block8x8,
            PrimaryRefFrame = Vp9MvReferenceFrame.Last,
            InterMode = Vp9InterMode.NearestMv,
            TxSize = Vp9TxSize.Tx4x4,
            Skip = false,
            SegmentId = 0,
            PrimaryMv = new Vp9Mv(4, 4),
        };
        var b = a with { };  // exact copy
        Equal(true, a == b);
        Equal(a.GetHashCode(), b.GetHashCode());

        var c = a with { Skip = true };
        Equal(false, a == c);
    }
}
