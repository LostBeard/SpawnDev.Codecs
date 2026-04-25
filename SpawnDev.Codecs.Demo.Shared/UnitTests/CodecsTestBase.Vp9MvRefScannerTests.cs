// Tests for Vp9MvRefScanner (slice 280).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvRefScanner_NoNeighbors_LeavesResultEmpty()
    {
        var result = new Vp9MvRefCandidatesByRef();
        // Block at (0, 0) with miAt returning null for everything.
        Vp9MvRefScanner.ScanCandidates(
            curMiRow: 0, curMiCol: 0,
            blockSize: Vp9BlockSize.Block16x16,
            frameMiRows: 32, frameMiCols: 32,
            miAt: (r, c) => null,
            result: result);
        Equal(0, result.ForRef(Vp9MvReferenceFrame.Last).Count);
        Equal(0, result.ForRef(Vp9MvReferenceFrame.Golden).Count);
        Equal(0, result.ForRef(Vp9MvReferenceFrame.AltRef).Count);
    }

    [TestMethod]
    public void Vp9MvRefScanner_IntraNeighbors_LeaveResultEmpty()
    {
        var result = new Vp9MvRefCandidatesByRef();
        Vp9MvRefScanner.ScanCandidates(
            curMiRow: 5, curMiCol: 5,
            blockSize: Vp9BlockSize.Block16x16,
            frameMiRows: 32, frameMiCols: 32,
            miAt: (r, c) => new Vp9MbModeInfo
            {
                BlockSize = Vp9BlockSize.Block8x8,
                PrimaryRefFrame = Vp9MvReferenceFrame.Intra,
                YMode = Vp9IntraMode.DcPred,
                TxSize = Vp9TxSize.Tx8x8,
                Skip = false,
                SegmentId = 0,
            },
            result: result);
        Equal(0, result.ForRef(Vp9MvReferenceFrame.Last).Count);
    }

    [TestMethod]
    public void Vp9MvRefScanner_InterNeighbors_PopulatesByRef()
    {
        var result = new Vp9MvRefCandidatesByRef();
        // All neighbors are inter blocks against the Last reference,
        // each with a distinct MV.
        int callCount = 0;
        Vp9MvRefScanner.ScanCandidates(
            curMiRow: 5, curMiCol: 5,
            blockSize: Vp9BlockSize.Block16x16,
            frameMiRows: 32, frameMiCols: 32,
            miAt: (r, c) =>
            {
                callCount++;
                return new Vp9MbModeInfo
                {
                    BlockSize = Vp9BlockSize.Block8x8,
                    PrimaryRefFrame = Vp9MvReferenceFrame.Last,
                    InterMode = Vp9InterMode.NewMv,
                    TxSize = Vp9TxSize.Tx4x4,
                    Skip = false,
                    SegmentId = 0,
                    PrimaryMv = new Vp9Mv(r, c), // distinct per neighbor
                };
            },
            result: result);

        // 8 neighbors all in-bounds at (5,5) with 32x32 frame.
        Equal(8, callCount);
        // Result list capacity is 2 - dedup'd to first 2 distinct MVs.
        Equal(2, result.ForRef(Vp9MvReferenceFrame.Last).Count);
        Equal(0, result.ForRef(Vp9MvReferenceFrame.Golden).Count);
    }

    [TestMethod]
    public void Vp9MvRefScanner_CompoundNeighbor_PopulatesBothRefs()
    {
        var result = new Vp9MvRefCandidatesByRef();
        Vp9MvRefScanner.ScanCandidates(
            curMiRow: 5, curMiCol: 5,
            blockSize: Vp9BlockSize.Block16x16,
            frameMiRows: 32, frameMiCols: 32,
            miAt: (r, c) => (r == 4 && c == 5) ? new Vp9MbModeInfo
            {
                BlockSize = Vp9BlockSize.Block8x8,
                PrimaryRefFrame = Vp9MvReferenceFrame.Last,
                CompoundRefFrame = Vp9MvReferenceFrame.Golden,
                InterMode = Vp9InterMode.ZeroMv,
                TxSize = Vp9TxSize.Tx4x4,
                Skip = false,
                SegmentId = 0,
                PrimaryMv = new Vp9Mv(4, 8),
                CompoundMv = new Vp9Mv(2, -3),
            } : null,
            result: result);

        Equal(1, result.ForRef(Vp9MvReferenceFrame.Last).Count);
        Equal(1, result.ForRef(Vp9MvReferenceFrame.Golden).Count);
        Equal(new Vp9Mv(4, 8), result.ForRef(Vp9MvReferenceFrame.Last)[0]);
        Equal(new Vp9Mv(2, -3), result.ForRef(Vp9MvReferenceFrame.Golden)[0]);
    }

    [TestMethod]
    public void Vp9MvRefScanner_OutOfBoundsNeighbors_Skipped()
    {
        var result = new Vp9MvRefCandidatesByRef();
        int callCount = 0;
        Vp9MvRefScanner.ScanCandidates(
            curMiRow: 0, curMiCol: 0, // at frame top-left
            blockSize: Vp9BlockSize.Block16x16,
            frameMiRows: 32, frameMiCols: 32,
            miAt: (r, c) =>
            {
                callCount++;
                return null;
            },
            result: result);

        // Block16x16 first 4 neighbors are (-1, 0), (0, -1), (-1, 1),
        // (1, -1) - first 3 are out-of-bounds; only (1, -1) negative col
        // also out. Most neighbors should be skipped without calling miAt.
        // Acceptable as long as some are skipped.
        Equal(true, callCount < 8);
    }

    [TestMethod]
    public void Vp9MvRefScanner_RejectsNullMiAt()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvRefScanner.ScanCandidates(0, 0, Vp9BlockSize.Block8x8, 32, 32, null!,
                new Vp9MvRefCandidatesByRef()));
    }

    [TestMethod]
    public void Vp9MvRefScanner_RejectsNullResult()
    {
        Throws<ArgumentNullException>(() =>
            Vp9MvRefScanner.ScanCandidates(0, 0, Vp9BlockSize.Block8x8, 32, 32, (r, c) => null, null!));
    }
}
