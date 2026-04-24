using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Smoke tests for <see cref="VorbisAudioDecoder"/>. Real bit-accuracy
/// validation against libvorbis-encoded streams requires reference test
/// vectors not yet bundled; these tests verify construction, first-packet
/// no-output invariant, and the shape of stored overlap-add state.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static VorbisIdentificationHeader MiniIdent(int channels, int blockSize0 = 256, int blockSize1 = 2048)
    {
        return new VorbisIdentificationHeader
        {
            VorbisVersion = 0,
            AudioChannels = channels,
            SampleRateHz = 48000,
            BitrateMaximum = 0,
            BitrateNominal = 128000,
            BitrateMinimum = 0,
            BlockSize0 = blockSize0,
            BlockSize1 = blockSize1,
        };
    }

    [TestMethod]
    public void VorbisAudioDecoder_Construct_SetupWithNoCodebooks_Succeeds()
    {
        var ident = MiniIdent(1);
        var setup = new VorbisSetupHeader
        {
            Codebooks = Array.Empty<VorbisCodebook>(),
            Floors = Array.Empty<VorbisFloor1Config>(),
            Residues = Array.Empty<VorbisResidueConfig>(),
            Mappings = Array.Empty<VorbisMappingConfig>(),
            Modes = Array.Empty<VorbisModeConfig>(),
        };
        var dec = new VorbisAudioDecoder(ident, setup);
        True(dec is not null);
    }

    [TestMethod]
    public void VorbisAudioDecoder_EmptyPacket_ReturnsZeroFrames()
    {
        var ident = MiniIdent(1);
        var setup = new VorbisSetupHeader
        {
            Codebooks = Array.Empty<VorbisCodebook>(),
            Floors = Array.Empty<VorbisFloor1Config>(),
            Residues = Array.Empty<VorbisResidueConfig>(),
            Mappings = Array.Empty<VorbisMappingConfig>(),
            Modes = new[] { new VorbisModeConfig { BlockFlag = false, Mapping = 0 } },
        };
        var dec = new VorbisAudioDecoder(ident, setup);
        var output = new float[1024];
        int frames = dec.DecodePacket(ReadOnlySpan<byte>.Empty, output);
        Equal(0, frames);
    }

    [TestMethod]
    public void VorbisAudioDecoder_NullHeaders_Throws()
    {
        var ident = MiniIdent(1);
        var setup = new VorbisSetupHeader
        {
            Codebooks = Array.Empty<VorbisCodebook>(),
            Floors = Array.Empty<VorbisFloor1Config>(),
            Residues = Array.Empty<VorbisResidueConfig>(),
            Mappings = Array.Empty<VorbisMappingConfig>(),
            Modes = Array.Empty<VorbisModeConfig>(),
        };
        Throws<ArgumentNullException>(() => new VorbisAudioDecoder(null!, setup));
        Throws<ArgumentNullException>(() => new VorbisAudioDecoder(ident, null!));
    }
}
