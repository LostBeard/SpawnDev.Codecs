using SpawnDev.Codecs.Audio.Opus;

namespace SpawnDev.Codecs.Tests.Audio.Opus;

/// <summary>
/// Tests for <see cref="OpusTocByte"/>. Covers all 32 configs from RFC 6716 Table 2,
/// stereo flag, and frame count code extraction.
/// </summary>
public class OpusTocByteTests
{
    [Theory]
    // Config 0-11: SILK (NB 0-3 / MB 4-7 / WB 8-11)
    [InlineData(0, OpusMode.Silk, OpusBandwidth.Narrowband)]
    [InlineData(1, OpusMode.Silk, OpusBandwidth.Narrowband)]
    [InlineData(3, OpusMode.Silk, OpusBandwidth.Narrowband)]
    [InlineData(4, OpusMode.Silk, OpusBandwidth.Mediumband)]
    [InlineData(7, OpusMode.Silk, OpusBandwidth.Mediumband)]
    [InlineData(8, OpusMode.Silk, OpusBandwidth.Wideband)]
    [InlineData(11, OpusMode.Silk, OpusBandwidth.Wideband)]
    // Config 12-15: Hybrid (SWB 12-13 / FB 14-15)
    [InlineData(12, OpusMode.Hybrid, OpusBandwidth.Superwideband)]
    [InlineData(13, OpusMode.Hybrid, OpusBandwidth.Superwideband)]
    [InlineData(14, OpusMode.Hybrid, OpusBandwidth.Fullband)]
    [InlineData(15, OpusMode.Hybrid, OpusBandwidth.Fullband)]
    // Config 16-31: CELT (NB 16-19 / WB 20-23 / SWB 24-27 / FB 28-31)
    [InlineData(16, OpusMode.Celt, OpusBandwidth.Narrowband)]
    [InlineData(19, OpusMode.Celt, OpusBandwidth.Narrowband)]
    [InlineData(20, OpusMode.Celt, OpusBandwidth.Wideband)]
    [InlineData(23, OpusMode.Celt, OpusBandwidth.Wideband)]
    [InlineData(24, OpusMode.Celt, OpusBandwidth.Superwideband)]
    [InlineData(27, OpusMode.Celt, OpusBandwidth.Superwideband)]
    [InlineData(28, OpusMode.Celt, OpusBandwidth.Fullband)]
    [InlineData(31, OpusMode.Celt, OpusBandwidth.Fullband)]
    public void Config_MapsToCorrectModeAndBandwidth(int config, OpusMode expectedMode, OpusBandwidth expectedBw)
    {
        byte tocByte = (byte)((config & 0x1F) << 3); // stereo=0, count=0
        var toc = new OpusTocByte(tocByte);
        Assert.Equal(expectedMode, toc.Mode);
        Assert.Equal(expectedBw, toc.Bandwidth);
        Assert.Equal(config, toc.Config);
    }

    [Fact]
    public void StereoBit_SetsChannelCountTo2()
    {
        var mono = new OpusTocByte(0x00);
        var stereo = new OpusTocByte(0x04);
        Assert.Equal(1, mono.ChannelCount);
        Assert.False(mono.IsStereo);
        Assert.Equal(2, stereo.ChannelCount);
        Assert.True(stereo.IsStereo);
    }

    [Theory]
    [InlineData(0x00, 0)] // count code 0 = 1 frame
    [InlineData(0x01, 1)] // count code 1 = 2 CBR frames
    [InlineData(0x02, 2)] // count code 2 = 2 VBR frames
    [InlineData(0x03, 3)] // count code 3 = arbitrary
    public void FrameCountCode_ExtractsLowTwoBits(byte value, int expected)
    {
        var toc = new OpusTocByte(value);
        Assert.Equal(expected, toc.FrameCountCode);
    }

    [Theory]
    // SILK-only NB (config 0-3): 10ms / 20ms / 40ms / 60ms
    [InlineData(0, 48_000, 480)]   // config 0, 10ms NB @ 48k
    [InlineData(1, 48_000, 960)]   // config 1, 20ms NB
    [InlineData(2, 48_000, 1920)]  // config 2, 40ms NB
    [InlineData(3, 48_000, 2880)]  // config 3, 60ms NB
    // Hybrid (config 12-15): 10ms or 20ms
    [InlineData(12, 48_000, 480)]  // config 12, 10ms SWB Hybrid
    [InlineData(13, 48_000, 960)]  // config 13, 20ms SWB Hybrid
    // CELT-only (config 16-31): 2.5/5/10/20ms (matches bit 7 = 1 path)
    [InlineData(16, 48_000, 120)]  // config 16, 2.5ms NB CELT
    [InlineData(17, 48_000, 240)]  // config 17, 5ms NB CELT
    [InlineData(18, 48_000, 480)]  // config 18, 10ms NB CELT
    [InlineData(19, 48_000, 960)]  // config 19, 20ms NB CELT
    public void GetSamplesPerFrame_MatchesRfcTable(int config, int sampleRate, int expectedSamples)
    {
        var toc = new OpusTocByte((byte)((config & 0x1F) << 3));
        Assert.Equal(expectedSamples, toc.GetSamplesPerFrame(sampleRate));
    }

    [Fact]
    public void GetSamplesPerFrame_ZeroSampleRate_Throws()
    {
        var toc = new OpusTocByte(0x00);
        Assert.Throws<ArgumentOutOfRangeException>(() => toc.GetSamplesPerFrame(0));
    }

    [Fact]
    public void ToString_IncludesAllFields()
    {
        var toc = new OpusTocByte(0x77); // config 14 (Hybrid FB), stereo=1, count=3
        string s = toc.ToString();
        Assert.Contains("0x77", s);
        Assert.Contains("config=14", s);
        Assert.Contains("Hybrid", s);
        Assert.Contains("stereo=True", s);
    }
}
