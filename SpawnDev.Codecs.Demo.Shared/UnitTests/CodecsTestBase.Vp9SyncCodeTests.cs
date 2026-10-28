// Tests for Vp9SyncCode (slice 271).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Constants_MatchLibvpx()
    {
        CpuOnlyTest();
        Equal(0x49, Vp9SyncCode.Byte0);
        Equal(0x83, Vp9SyncCode.Byte1);
        Equal(0x42, Vp9SyncCode.Byte2);
        Equal(3, Vp9SyncCode.Length);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_AsArray_ReturnsThreeBytes()
    {
        CpuOnlyTest();
        var arr = Vp9SyncCode.AsArray();
        Equal(3, arr.Length);
        Equal((byte)0x49, arr[0]);
        Equal((byte)0x83, arr[1]);
        Equal((byte)0x42, arr[2]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Matches_TrueForExactSequence()
    {
        CpuOnlyTest();
        var data = new byte[] { 0x49, 0x83, 0x42, 0xFF };
        Equal(true, Vp9SyncCode.Matches(data, 0));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Matches_FalseForWrongFirstByte()
    {
        CpuOnlyTest();
        var data = new byte[] { 0x48, 0x83, 0x42 };
        Equal(false, Vp9SyncCode.Matches(data, 0));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Matches_FalseForTruncated()
    {
        CpuOnlyTest();
        var data = new byte[] { 0x49, 0x83 };
        Equal(false, Vp9SyncCode.Matches(data, 0));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Matches_TrueAtNonZeroOffset()
    {
        CpuOnlyTest();
        var data = new byte[] { 0xFF, 0xFF, 0x49, 0x83, 0x42, 0xFF };
        Equal(true, Vp9SyncCode.Matches(data, 2));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Matches_FalseWhenOffsetPastEnd()
    {
        CpuOnlyTest();
        var data = new byte[] { 0x49, 0x83, 0x42 };
        Equal(false, Vp9SyncCode.Matches(data, 1)); // not enough bytes after offset 1
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9SyncCode_Matches_RejectsNegativeOffset()
    {
        CpuOnlyTest();
        var data = new byte[] { 0x49, 0x83, 0x42 };
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SyncCode.Matches(data, -1));
    }
}
