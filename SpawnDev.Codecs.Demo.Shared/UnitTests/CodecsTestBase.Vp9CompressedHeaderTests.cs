// Tests for Vp9CompressedHeaderParser (slice 222). Verifies the
// composition wires the right sub-parsers in libvpx's order and
// branches correctly on lossless / intra_only / interp_filter /
// allow_hp / compound_reference_allowed.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CompressedHeader_Lossless_ForcesOnly4x4()
    {
        var data = new byte[64];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var state = new Vp9CompressedHeaderState();
        var inputs = new Vp9CompressedHeaderInputs(
            IsLossless: true,
            IsIntraOnly: true,
            InterpFilter: Vp9InterpFilter.EightTap,
            AllowHighPrecisionMv: false,
            SignBiasLast: false,
            SignBiasGolden: false,
            SignBiasAltRef: false);

        var result = Vp9CompressedHeaderParser.Read(state, inputs, reader);

        Equal(Vp9TxMode.Only4x4, result.TxMode);
        Equal(Vp9ReferenceMode.SingleReference, result.ReferenceMode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_IntraOnly_NoInterFrameTablesRead()
    {
        // Buffer is all zeros so:
        //   tx_mode bits f(2) = 0 -> Only4x4
        //   coef_probs flag for 4x4 = 0 -> no update
        //   skip_probs 3 update flags = 0 0 0 -> no update
        // Then because IsIntraOnly=true the parser stops; no inter
        // tables are touched. Total bits read: 2 + 1 + 3 = 6 bits.
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var state = new Vp9CompressedHeaderState();
        // Seed inter probs so we'd notice if they got touched.
        state.InterModeProbs.Probs[0, 0] = 200;
        state.IntraInterProbs.Probs[0] = 150;
        state.MvProbs.Joints[0] = 99;

        var inputs = new Vp9CompressedHeaderInputs(
            IsLossless: false,
            IsIntraOnly: true,
            InterpFilter: Vp9InterpFilter.EightTap,
            AllowHighPrecisionMv: false,
            SignBiasLast: false,
            SignBiasGolden: false,
            SignBiasAltRef: false);

        var result = Vp9CompressedHeaderParser.Read(state, inputs, reader);

        // Inter-frame tables untouched.
        Equal((byte)200, state.InterModeProbs.Probs[0, 0]);
        Equal((byte)150, state.IntraInterProbs.Probs[0]);
        Equal((byte)99, state.MvProbs.Joints[0]);
        Equal(Vp9ReferenceMode.SingleReference, result.ReferenceMode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_InterFrame_ReadsInterTables()
    {
        // Zero buffer: every update flag reads as 0, every prob unchanged.
        // We just verify the parser walks without throwing and lands at
        // SingleReference (because compound_reference_allowed = false
        // when all sign biases are equal).
        var data = new byte[64];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var state = new Vp9CompressedHeaderState();

        var inputs = new Vp9CompressedHeaderInputs(
            IsLossless: false,
            IsIntraOnly: false,
            InterpFilter: Vp9InterpFilter.EightTap,
            AllowHighPrecisionMv: false,
            SignBiasLast: false,
            SignBiasGolden: false,
            SignBiasAltRef: false);

        var result = Vp9CompressedHeaderParser.Read(state, inputs, reader);

        Equal(Vp9TxMode.Only4x4, result.TxMode);
        // All sign biases equal -> compound not allowed -> single reference.
        Equal(Vp9ReferenceMode.SingleReference, result.ReferenceMode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_InterFrame_CompoundAllowed_BitsConsumed()
    {
        // Different sign biases -> compound_reference_allowed = true ->
        // the parser reads 1 reference_mode bit (0 with zero buffer ->
        // still SingleReference).
        var data = new byte[64];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var state = new Vp9CompressedHeaderState();

        var inputs = new Vp9CompressedHeaderInputs(
            IsLossless: false,
            IsIntraOnly: false,
            InterpFilter: Vp9InterpFilter.EightTap,
            AllowHighPrecisionMv: false,
            SignBiasLast: false,
            SignBiasGolden: true,
            SignBiasAltRef: false);

        var result = Vp9CompressedHeaderParser.Read(state, inputs, reader);

        Equal(Vp9ReferenceMode.SingleReference, result.ReferenceMode);
    }

    [TestMethod]
    public void Vp9CompressedHeader_State_TableShapesInitializedCorrectly()
    {
        var s = new Vp9CompressedHeaderState();
        Equal(4, s.CoefProbs.Length);
        for (int i = 0; i < 4; i++)
            Equal(Vp9CoefProbsParser.FlatSize, s.CoefProbs[i].Length);
        Equal(36, s.YModeProbs.Length);
        Equal(48, s.PartitionProbs.Length);
    }

    [TestMethod]
    public void Vp9CompressedHeader_State_PartitionProbsSeededFromKfDefaults()
    {
        var s = new Vp9CompressedHeaderState();
        for (int i = 0; i < s.PartitionProbs.Length; i++)
            Equal(Vp9PartitionProbs.KfPartitionProbs[i], s.PartitionProbs[i]);
    }

    [TestMethod]
    public void Vp9CompressedHeader_State_YModeProbsSeededFromIfDefaults()
    {
        var s = new Vp9CompressedHeaderState();
        for (int i = 0; i < s.YModeProbs.Length; i++)
            Equal(Vp9IntraModeProbs.DefaultIfYProbs[i], s.YModeProbs[i]);
    }

    [TestMethod]
    public void Vp9CompressedHeader_RejectsNullArgs()
    {
        var data = new byte[16];
        var reader = new Vp9BoolDecoder(data, 0, data.Length);
        var inputs = new Vp9CompressedHeaderInputs(
            true, true, Vp9InterpFilter.EightTap, false, false, false, false);
        Throws<ArgumentNullException>(() =>
            Vp9CompressedHeaderParser.Read(null!, inputs, reader));
        Throws<ArgumentNullException>(() =>
            Vp9CompressedHeaderParser.Read(new Vp9CompressedHeaderState(), inputs, null!));
    }
}
