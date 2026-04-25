// Tests for Vp9ReferenceFrameInfoParser (slice 208).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ReferenceFrameInfo_Parse_KnownPattern()
    {
        // 3 references: Last=2 sign=0; Golden=5 sign=1; AltRef=7 sign=0.
        // 12 bits total.
        var data = BitsToBytes(
            (2, 3), (0, 1),  // Last  idx=2, bias=0
            (5, 3), (1, 1),  // Golden idx=5, bias=1
            (7, 3), (0, 1)); // AltRef idx=7, bias=0

        var info = Vp9ReferenceFrameInfoParser.Parse(data);

        Equal(3, info.RefFrameIdx.Length);
        Equal(2, info.RefFrameIdx[(int)Vp9ReferenceSlot.Last]);
        Equal(5, info.RefFrameIdx[(int)Vp9ReferenceSlot.Golden]);
        Equal(7, info.RefFrameIdx[(int)Vp9ReferenceSlot.AltRef]);
        Equal(3, info.RefFrameSignBias.Length);
        Equal(false, info.RefFrameSignBias[(int)Vp9ReferenceSlot.Last]);
        Equal(true, info.RefFrameSignBias[(int)Vp9ReferenceSlot.Golden]);
        Equal(false, info.RefFrameSignBias[(int)Vp9ReferenceSlot.AltRef]);
    }

    [TestMethod]
    public void Vp9ReferenceFrameInfo_Parse_AllZeros()
    {
        var data = BitsToBytes(
            (0, 3), (0, 1),
            (0, 3), (0, 1),
            (0, 3), (0, 1));

        var info = Vp9ReferenceFrameInfoParser.Parse(data);

        for (int i = 0; i < 3; i++)
        {
            Equal(0, info.RefFrameIdx[i]);
            Equal(false, info.RefFrameSignBias[i]);
        }
    }

    [TestMethod]
    public void Vp9ReferenceFrameInfo_Parse_AllMaxValues()
    {
        // All three references: idx=7 (max), bias=1.
        var data = BitsToBytes(
            (7, 3), (1, 1),
            (7, 3), (1, 1),
            (7, 3), (1, 1));

        var info = Vp9ReferenceFrameInfoParser.Parse(data);

        for (int i = 0; i < 3; i++)
        {
            Equal(7, info.RefFrameIdx[i]);
            Equal(true, info.RefFrameSignBias[i]);
        }
    }

    [TestMethod]
    public void Vp9ReferenceFrameInfo_Constants_MatchLibvpx()
    {
        Equal(3, Vp9ReferenceFrameInfo.RefsPerFrame);
        Equal(3, Vp9ReferenceFrameInfo.RefFramesLog2);
        Equal(8, Vp9ReferenceFrameInfo.RefFramesPoolSize);
    }
}
