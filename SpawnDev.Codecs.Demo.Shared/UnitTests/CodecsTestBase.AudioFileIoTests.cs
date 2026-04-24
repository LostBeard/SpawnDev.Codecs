using SpawnDev.Codecs.Audio.Aiff;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Wav;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the file-path I/O helpers added on top of the byte-array codec
/// surface. Each test writes to a temporary file, reads it back, and
/// verifies lossless sample recovery. Temp files are cleaned up in finally.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void FlacFile_Encode_Decode_DiskRoundtrip()
    {
        var input = GenerateSineInt(samplesPerChannel: 512, channels: 2, sampleRateHz: 44100, bps: 16);
        string path = Path.Combine(Path.GetTempPath(), $"flac_test_{Guid.NewGuid():N}.flac");
        try
        {
            FlacEncoder.EncodeToFile(path, input, 44100, 2, 16, blockSize: 512);
            True(File.Exists(path), "FLAC file must exist after EncodeToFile.");
            True(new FileInfo(path).Length > 0, "FLAC file must be non-empty.");
            var decoded = FlacDecoder.DecodeFile(path);
            Equal(44100, decoded.StreamInfo.SampleRateHz);
            Equal(2, decoded.StreamInfo.Channels);
            EqualInts(input, decoded.InterleavedSamples);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [TestMethod]
    public void WavFile_Write_Read_DiskRoundtrip()
    {
        var input = GenerateSineInt(samplesPerChannel: 256, channels: 1, sampleRateHz: 48000, bps: 16);
        string path = Path.Combine(Path.GetTempPath(), $"wav_test_{Guid.NewGuid():N}.wav");
        try
        {
            WavFileCodec.WriteFile(path, input, 48000, 1, 16);
            var wav = WavFileCodec.ReadFile(path);
            Equal(48000, wav.SampleRateHz);
            EqualInts(input, wav.InterleavedSamples);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [TestMethod]
    public void AiffFile_Write_Read_DiskRoundtrip()
    {
        var input = GenerateSineInt(samplesPerChannel: 128, channels: 2, sampleRateHz: 96000, bps: 24);
        string path = Path.Combine(Path.GetTempPath(), $"aiff_test_{Guid.NewGuid():N}.aiff");
        try
        {
            AiffFileCodec.WriteFile(path, input, 96000, 2, 24);
            var aiff = AiffFileCodec.ReadFile(path);
            Equal(96000, aiff.SampleRateHz);
            Equal(24, aiff.BitsPerSample);
            EqualInts(input, aiff.InterleavedSamples);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [TestMethod]
    public void FileApi_NullPath_Throws()
    {
        Throws<ArgumentNullException>(() => FlacDecoder.DecodeFile(null!));
        Throws<ArgumentNullException>(() => WavFileCodec.ReadFile(null!));
        Throws<ArgumentNullException>(() => AiffFileCodec.ReadFile(null!));
    }

    [TestMethod]
    public void WavToFlacToAiff_FileChain_RoundtripsLosslessly()
    {
        // Full end-to-end: WAV -> FLAC -> AIFF all through disk, samples preserved.
        var input = GenerateSineInt(samplesPerChannel: 256, channels: 1, sampleRateHz: 44100, bps: 16);
        string wavPath = Path.Combine(Path.GetTempPath(), $"chain_{Guid.NewGuid():N}.wav");
        string flacPath = Path.Combine(Path.GetTempPath(), $"chain_{Guid.NewGuid():N}.flac");
        string aiffPath = Path.Combine(Path.GetTempPath(), $"chain_{Guid.NewGuid():N}.aiff");
        try
        {
            WavFileCodec.WriteFile(wavPath, input, 44100, 1, 16);
            var wav = WavFileCodec.ReadFile(wavPath);
            FlacEncoder.EncodeToFile(flacPath, wav.InterleavedSamples, wav.SampleRateHz, wav.Channels, wav.BitsPerSample, blockSize: 256);
            var flac = FlacDecoder.DecodeFile(flacPath);
            AiffFileCodec.WriteFile(aiffPath, flac.InterleavedSamples, flac.StreamInfo.SampleRateHz, flac.StreamInfo.Channels, flac.StreamInfo.BitsPerSample);
            var aiff = AiffFileCodec.ReadFile(aiffPath);
            EqualInts(input, aiff.InterleavedSamples);
        }
        finally
        {
            if (File.Exists(wavPath)) File.Delete(wavPath);
            if (File.Exists(flacPath)) File.Delete(flacPath);
            if (File.Exists(aiffPath)) File.Delete(aiffPath);
        }
    }
}
