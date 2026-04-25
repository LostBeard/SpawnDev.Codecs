// Tests for Vp9SubsizeLookup (slice 226).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SubsizeLookup_LookupShape()
    {
        Equal(4, Vp9SubsizeLookup.Lookup.GetLength(0));
        Equal(13, Vp9SubsizeLookup.Lookup.GetLength(1));
    }

    [TestMethod]
    public void Vp9SubsizeLookup_None_PreservesParent()
    {
        for (int i = 0; i < Vp9BlockSizes.Count; i++)
        {
            Equal((Vp9BlockSize)i,
                Vp9SubsizeLookup.Subsize((Vp9BlockSize)i, Vp9PartitionType.None));
        }
    }

    [TestMethod]
    public void Vp9SubsizeLookup_Horz_SquareParents()
    {
        Equal(Vp9BlockSize.Block8x4,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block8x8, Vp9PartitionType.Horz));
        Equal(Vp9BlockSize.Block16x8,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block16x16, Vp9PartitionType.Horz));
        Equal(Vp9BlockSize.Block32x16,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block32x32, Vp9PartitionType.Horz));
        Equal(Vp9BlockSize.Block64x32,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block64x64, Vp9PartitionType.Horz));
    }

    [TestMethod]
    public void Vp9SubsizeLookup_Vert_SquareParents()
    {
        Equal(Vp9BlockSize.Block4x8,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block8x8, Vp9PartitionType.Vert));
        Equal(Vp9BlockSize.Block8x16,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block16x16, Vp9PartitionType.Vert));
        Equal(Vp9BlockSize.Block16x32,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block32x32, Vp9PartitionType.Vert));
        Equal(Vp9BlockSize.Block32x64,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block64x64, Vp9PartitionType.Vert));
    }

    [TestMethod]
    public void Vp9SubsizeLookup_Split_SquareParents()
    {
        Equal(Vp9BlockSize.Block4x4,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block8x8, Vp9PartitionType.Split));
        Equal(Vp9BlockSize.Block8x8,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block16x16, Vp9PartitionType.Split));
        Equal(Vp9BlockSize.Block16x16,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block32x32, Vp9PartitionType.Split));
        Equal(Vp9BlockSize.Block32x32,
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block64x64, Vp9PartitionType.Split));
    }

    [TestMethod]
    public void Vp9SubsizeLookup_NonSquareParent_HorzVertSplit_AreInvalid()
    {
        // Non-square parents and 4x4 cannot legally be re-partitioned
        // by HORZ / VERT / SPLIT in the VP9 bitstream.
        Vp9BlockSize[] illegal =
        [
            Vp9BlockSize.Block4x4,   Vp9BlockSize.Block4x8,   Vp9BlockSize.Block8x4,
            Vp9BlockSize.Block8x16,  Vp9BlockSize.Block16x8,  Vp9BlockSize.Block16x32,
            Vp9BlockSize.Block32x16, Vp9BlockSize.Block32x64, Vp9BlockSize.Block64x32,
        ];
        Vp9PartitionType[] partitions =
        [
            Vp9PartitionType.Horz, Vp9PartitionType.Vert, Vp9PartitionType.Split,
        ];
        foreach (var parent in illegal)
        {
            foreach (var p in partitions)
            {
                Equal(Vp9BlockSize.Invalid, Vp9SubsizeLookup.Subsize(parent, p));
            }
        }
    }

    [TestMethod]
    public void Vp9SubsizeLookup_RejectsOutOfRange()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SubsizeLookup.Subsize(Vp9BlockSize.Block8x8, (Vp9PartitionType)99));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SubsizeLookup.Subsize((Vp9BlockSize)99, Vp9PartitionType.None));
    }
}
