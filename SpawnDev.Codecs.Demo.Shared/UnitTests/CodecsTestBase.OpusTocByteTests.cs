using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="OpusTocByte"/>: config -> mode/bandwidth mapping for all 32 configs
/// plus stereo flag, frame count code, samples-per-frame computation, and ToString formatting.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void TocByte_Config_MapsToCorrectModeAndBandwidth_AllConfigs()
    {
        // RFC 6716 Table 2:
        //   0-3:  SILK  NB
        //   4-7:  SILK  MB
        //   8-11: SILK  WB
        //   12-13: Hybrid SWB
        //   14-15: Hybrid FB
        //   16-19: CELT NB
        //   20-23: CELT WB
        //   24-27: CELT SWB
        //   28-31: CELT FB
        var table = new (int config, OpusMode mode, OpusBandwidth bw)[]
        {
            (0,  OpusMode.Silk,   OpusBandwidth.Narrowband),
            (1,  OpusMode.Silk,   OpusBandwidth.Narrowband),
            (3,  OpusMode.Silk,   OpusBandwidth.Narrowband),
            (4,  OpusMode.Silk,   OpusBandwidth.Mediumband),
            (7,  OpusMode.Silk,   OpusBandwidth.Mediumband),
            (8,  OpusMode.Silk,   OpusBandwidth.Wideband),
            (11, OpusMode.Silk,   OpusBandwidth.Wideband),
            (12, OpusMode.Hybrid, OpusBandwidth.Superwideband),
            (13, OpusMode.Hybrid, OpusBandwidth.Superwideband),
            (14, OpusMode.Hybrid, OpusBandwidth.Fullband),
            (15, OpusMode.Hybrid, OpusBandwidth.Fullband),
            (16, OpusMode.Celt,   OpusBandwidth.Narrowband),
            (19, OpusMode.Celt,   OpusBandwidth.Narrowband),
            (20, OpusMode.Celt,   OpusBandwidth.Wideband),
            (23, OpusMode.Celt,   OpusBandwidth.Wideband),
            (24, OpusMode.Celt,   OpusBandwidth.Superwideband),
            (27, OpusMode.Celt,   OpusBandwidth.Superwideband),
            (28, OpusMode.Celt,   OpusBandwidth.Fullband),
            (31, OpusMode.Celt,   OpusBandwidth.Fullband),
        };
        foreach (var (config, expectedMode, expectedBw) in table)
        {
            byte tocByte = (byte)((config & 0x1F) << 3);
            var toc = new OpusTocByte(tocByte);
            Equal(expectedMode, toc.Mode, $"config {config} Mode");
            Equal(expectedBw, toc.Bandwidth, $"config {config} Bandwidth");
            Equal(config, toc.Config, $"config {config} Config");
        }
    }

    [TestMethod]
    public void TocByte_StereoBit_SetsChannelCount()
    {
        var mono = new OpusTocByte(0x00);
        var stereo = new OpusTocByte(0x04);
        Equal(1, mono.ChannelCount);
        False(mono.IsStereo);
        Equal(2, stereo.ChannelCount);
        True(stereo.IsStereo);
    }

    [TestMethod]
    public void TocByte_FrameCountCode_ExtractsLowTwoBits()
    {
        var cases = new (byte value, int expected)[]
        {
            (0x00, 0),
            (0x01, 1),
            (0x02, 2),
            (0x03, 3),
        };
        foreach (var (value, expected) in cases)
        {
            Equal(expected, new OpusTocByte(value).FrameCountCode, $"value 0x{value:X2}");
        }
    }

    [TestMethod]
    public void TocByte_GetSamplesPerFrame_MatchesRfcTable()
    {
        // (config, expected samples @ 48kHz)
        var cases = new (int config, int expected)[]
        {
            (0,  480),   // SILK NB 10ms
            (1,  960),   // SILK NB 20ms
            (2,  1920),  // SILK NB 40ms
            (3,  2880),  // SILK NB 60ms
            (12, 480),   // Hybrid SWB 10ms
            (13, 960),   // Hybrid SWB 20ms
            (16, 120),   // CELT NB 2.5ms
            (17, 240),   // CELT NB 5ms
            (18, 480),   // CELT NB 10ms
            (19, 960),   // CELT NB 20ms
        };
        foreach (var (config, expected) in cases)
        {
            var toc = new OpusTocByte((byte)((config & 0x1F) << 3));
            Equal(expected, toc.GetSamplesPerFrame(48_000), $"config {config}");
        }
    }

    [TestMethod]
    public void TocByte_GetSamplesPerFrame_ZeroSampleRate_Throws()
    {
        var toc = new OpusTocByte(0x00);
        Throws<ArgumentOutOfRangeException>(() => toc.GetSamplesPerFrame(0));
    }

    [TestMethod]
    public void TocByte_ToString_IncludesAllFields()
    {
        var toc = new OpusTocByte(0x77); // config 14 Hybrid FB, stereo=1, c=3
        string s = toc.ToString();
        Contains("0x77", s);
        Contains("config=14", s);
        Contains("Hybrid", s);
        Contains("stereo=True", s);
    }
}
