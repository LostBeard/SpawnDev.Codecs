using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisMappingConfigParser"/> and
/// <see cref="VorbisModeConfigParser"/>. Together these two round out the
/// Vorbis setup-header structural parse (codebook + floor + residue + mapping
/// + mode).
/// </summary>
public abstract partial class CodecsTestBase
{
    private static VorbisMappingConfig ParseMapping(byte[] data, int audioChannels)
    {
        var r = new VorbisBitReader(data);
        return VorbisMappingConfigParser.Parse(ref r, audioChannels);
    }

    private static VorbisModeConfig[] ParseModes(byte[] data, int modeCount)
    {
        var r = new VorbisBitReader(data);
        return VorbisModeConfigParser.Parse(ref r, modeCount);
    }

    // -------- Mapping tests --------

    [TestMethod]
    public void VorbisMapping_MonoSingleSubmap_ParsesMinimal()
    {
        // mono, 1 submap (no submap flag), no coupling, reserved 00, no mux (submaps == 1),
        // submap 0: reserved 8-bit, floor = 0, residue = 1.
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // submaps flag = no
        w.Write(0, 1);           // coupling flag = no
        w.Write(0, 2);           // reserved
        w.Write(0, 8);           // submap 0 reserved
        w.Write(0, 8);           // floor
        w.Write(1, 8);           // residue
        var cfg = ParseMapping(w.ToArray(), audioChannels: 1);
        Equal(1, cfg.Submaps);
        Equal(0, cfg.CouplingMagnitudeChannels.Length);
        EqualInts(new[] { 0 }, cfg.Mux);
        Equal(0, cfg.SubmapFloor[0]);
        Equal(1, cfg.SubmapResidue[0]);
    }

    [TestMethod]
    public void VorbisMapping_StereoWithCoupling_ParsesCouplingPair()
    {
        // Stereo, 1 submap, 1 coupling step (mag = channel 0, ang = channel 1).
        // ilog(audio_channels - 1) = ilog(1) = 1 bit per channel index.
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // submaps flag = no -> 1 submap
        w.Write(1, 1);           // coupling flag = yes
        w.Write(0, 8);           // coupling_steps - 1 = 0 -> 1 step
        w.Write(0, 1); w.Write(1, 1); // mag=0, ang=1 at 1 bit each
        w.Write(0, 2);           // reserved
        // submap 0
        w.Write(0, 8); w.Write(2, 8); w.Write(3, 8); // reserved / floor / residue
        var cfg = ParseMapping(w.ToArray(), audioChannels: 2);
        Equal(1, cfg.CouplingMagnitudeChannels.Length);
        Equal(0, cfg.CouplingMagnitudeChannels[0]);
        Equal(1, cfg.CouplingAngleChannels[0]);
        Equal(2, cfg.SubmapFloor[0]);
        Equal(3, cfg.SubmapResidue[0]);
    }

    [TestMethod]
    public void VorbisMapping_MultipleSubmaps_ReadsMuxPerChannel()
    {
        // 3-channel audio, 2 submaps. Channel 0 -> submap 1, channels 1,2 -> submap 0.
        var w = new VorbisTestWriter();
        w.Write(1, 1);           // submaps flag = yes
        w.Write(1, 4);           // submaps - 1 = 1 -> 2 submaps
        w.Write(0, 1);           // no coupling
        w.Write(0, 2);           // reserved
        w.Write(1, 4); w.Write(0, 4); w.Write(0, 4); // mux[0..2]
        // submap 0
        w.Write(0, 8); w.Write(4, 8); w.Write(5, 8);
        // submap 1
        w.Write(0, 8); w.Write(6, 8); w.Write(7, 8);
        var cfg = ParseMapping(w.ToArray(), audioChannels: 3);
        Equal(2, cfg.Submaps);
        EqualInts(new[] { 1, 0, 0 }, cfg.Mux);
        EqualInts(new[] { 4, 6 }, cfg.SubmapFloor);
        EqualInts(new[] { 5, 7 }, cfg.SubmapResidue);
    }

    [TestMethod]
    public void VorbisMapping_ReservedBitsNonZero_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0, 1); w.Write(0, 1);
        w.Write(1, 2);          // reserved = 01 -> violation
        bool threw = false;
        try { ParseMapping(w.ToArray(), 1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisMapping_CouplingPairWithSameChannel_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0, 1);           // submaps flag = no
        w.Write(1, 1);           // coupling flag = yes
        w.Write(0, 8);           // 1 coupling step
        w.Write(0, 1); w.Write(0, 1); // mag=ang=0, invalid
        bool threw = false;
        try { ParseMapping(w.ToArray(), 2); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    // -------- Mode tests --------

    [TestMethod]
    public void VorbisMode_SingleMode_ShortBlock_Parses()
    {
        var w = new VorbisTestWriter();
        w.Write(0, 1);          // blockflag = short
        w.Write(0, 16);         // windowtype
        w.Write(0, 16);         // transformtype
        w.Write(0, 8);          // mapping
        w.Write(1, 1);          // framing flag
        var modes = ParseModes(w.ToArray(), 1);
        Equal(1, modes.Length);
        False(modes[0].BlockFlag);
        Equal(0, modes[0].Mapping);
    }

    [TestMethod]
    public void VorbisMode_TwoModes_MixedBlockSizesAndMappings()
    {
        var w = new VorbisTestWriter();
        // mode 0 (short, mapping 0)
        w.Write(0, 1); w.Write(0, 16); w.Write(0, 16); w.Write(0, 8);
        // mode 1 (long, mapping 1)
        w.Write(1, 1); w.Write(0, 16); w.Write(0, 16); w.Write(1, 8);
        w.Write(1, 1);
        var modes = ParseModes(w.ToArray(), 2);
        False(modes[0].BlockFlag);
        True(modes[1].BlockFlag);
        Equal(1, modes[1].Mapping);
    }

    [TestMethod]
    public void VorbisMode_MissingFramingFlag_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0, 1); w.Write(0, 16); w.Write(0, 16); w.Write(0, 8);
        w.Write(0, 1); // framing flag = 0, violation
        bool threw = false;
        try { ParseModes(w.ToArray(), 1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisMode_NonZeroWindowType_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0, 1); w.Write(1, 16); w.Write(0, 16); w.Write(0, 8); w.Write(1, 1);
        bool threw = false;
        try { ParseModes(w.ToArray(), 1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisMode_NonZeroTransformType_Throws()
    {
        var w = new VorbisTestWriter();
        w.Write(0, 1); w.Write(0, 16); w.Write(1, 16); w.Write(0, 8); w.Write(1, 1);
        bool threw = false;
        try { ParseModes(w.ToArray(), 1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
