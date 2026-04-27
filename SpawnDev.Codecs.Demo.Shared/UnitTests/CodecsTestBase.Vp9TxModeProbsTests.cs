// Tests for Vp9TxModeProbsParser (slice 212).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TxModeProbs_Constants_MatchLibvpx()
    {
        Equal(2, Vp9TxModeProbs.TxSizeContexts);
        Equal(4, Vp9TxModeProbs.TxSizes);
    }

    [TestMethod]
    public void Vp9TxModeProbs_TableShapes()
    {
        var probs = new Vp9TxModeProbs();
        // p8x8 = [2][1], p16x16 = [2][2], p32x32 = [2][3].
        Equal(2, probs.P8x8.GetLength(0));
        Equal(1, probs.P8x8.GetLength(1));
        Equal(2, probs.P16x16.GetLength(0));
        Equal(2, probs.P16x16.GetLength(1));
        Equal(2, probs.P32x32.GetLength(0));
        Equal(3, probs.P32x32.GetLength(1));
    }

    [TestMethod]
    public void Vp9TxModeProbs_DefaultsMatchLibvpx()
    {
        // libvpx vp9_entropymode.c default_tx_probs:
        //   p32x32 = { { 3, 136, 37 }, { 5, 52, 13 } }
        //   p16x16 = { { 20, 152 }, { 15, 101 } }
        //   p8x8   = { { 100 }, { 66 } }
        // The compressed header applies diff_update_prob deltas FROM these
        // defaults; zero-init would corrupt every downstream tx_size read
        // when tx_mode == TxModeSelect.
        var probs = new Vp9TxModeProbs();

        Equal((byte)100, probs.P8x8[0, 0]);
        Equal((byte)66,  probs.P8x8[1, 0]);

        Equal((byte)20,  probs.P16x16[0, 0]);
        Equal((byte)152, probs.P16x16[0, 1]);
        Equal((byte)15,  probs.P16x16[1, 0]);
        Equal((byte)101, probs.P16x16[1, 1]);

        Equal((byte)3,   probs.P32x32[0, 0]);
        Equal((byte)136, probs.P32x32[0, 1]);
        Equal((byte)37,  probs.P32x32[0, 2]);
        Equal((byte)5,   probs.P32x32[1, 0]);
        Equal((byte)52,  probs.P32x32[1, 1]);
        Equal((byte)13,  probs.P32x32[1, 2]);
    }

    [TestMethod]
    public void Vp9TxModeProbs_Parse_NoUpdates_LeavesTableUntouched()
    {
        // 12 update flags, all zero. Buffer of zeros suffices.
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);

        var probs = new Vp9TxModeProbs();
        // Seed with non-default values to verify in-place lack of update.
        probs.P8x8[0, 0] = 100;
        probs.P8x8[1, 0] = 110;
        probs.P16x16[0, 0] = 120;
        probs.P32x32[1, 2] = 200;

        Vp9TxModeProbsParser.Read(probs, reader);

        Equal((byte)100, probs.P8x8[0, 0]);
        Equal((byte)110, probs.P8x8[1, 0]);
        Equal((byte)120, probs.P16x16[0, 0]);
        Equal((byte)200, probs.P32x32[1, 2]);
    }

    [TestMethod]
    public void Vp9TxModeProbs_Parse_RejectsNullArgs()
    {
        var data = new byte[8];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        Throws<ArgumentNullException>(() =>
            Vp9TxModeProbsParser.Read(null!, reader));
        Throws<ArgumentNullException>(() =>
            Vp9TxModeProbsParser.Read(new Vp9TxModeProbs(), null!));
    }
}
