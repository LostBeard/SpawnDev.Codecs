// IvfDetector tests against real fixtures + synthetic edge cases.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void IvfDetector_BbbAv1Fixture_DetectsAv1()
    {
        var bytes = LoadAv1Fixture();
        Equal(true, IvfDetector.IsIvf(bytes));
        Equal(IvfCodec.Av1, IvfDetector.DetectCodec(bytes));
    }

    [TestMethod]
    public void IvfDetector_RandomBytes_RejectsAsNotIvf()
    {
        var bytes = new byte[64];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i + 1);
        Equal(false, IvfDetector.IsIvf(bytes));
        Equal(IvfCodec.Unknown, IvfDetector.DetectCodec(bytes));
    }

    [TestMethod]
    public void IvfDetector_TruncatedHeader_Rejects()
    {
        var bytes = new byte[16]; // less than 32 bytes
        bytes[0] = (byte)'D'; bytes[1] = (byte)'K'; bytes[2] = (byte)'I'; bytes[3] = (byte)'F';
        Equal(false, IvfDetector.IsIvf(bytes));
        Equal(IvfCodec.Unknown, IvfDetector.DetectCodec(bytes));
    }

    [TestMethod]
    public void IvfDetector_Vp90FourCc_DetectsVp9()
    {
        // Build a minimal IVF header with VP90 fourcc.
        var bytes = new byte[32];
        bytes[0] = (byte)'D'; bytes[1] = (byte)'K'; bytes[2] = (byte)'I'; bytes[3] = (byte)'F';
        bytes[4] = 0; bytes[5] = 0;             // version 0
        bytes[6] = 32; bytes[7] = 0;            // header length 32
        bytes[8] = (byte)'V'; bytes[9] = (byte)'P'; bytes[10] = (byte)'9'; bytes[11] = (byte)'0';
        bytes[12] = 0x40; bytes[13] = 0x01;     // width 320
        bytes[14] = 0xB4; bytes[15] = 0x00;     // height 180
        // remaining fields zero.
        Equal(true, IvfDetector.IsIvf(bytes));
        Equal(IvfCodec.Vp9, IvfDetector.DetectCodec(bytes));
    }
}
