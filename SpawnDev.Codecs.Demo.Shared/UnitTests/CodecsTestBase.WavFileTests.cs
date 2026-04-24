using SpawnDev.Codecs.Audio.Wav;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="WavFileCodec"/>. Writes PCM samples to WAV bytes,
/// reads them back, and asserts roundtrip equivalence for 8 / 16 / 24 / 32-bit
/// integer PCM, mono / stereo / multi-channel, and 32-bit IEEE float decoding.
/// Also tests that our FLAC output can be decoded and converted to WAV and
/// that roundtrip through WAV + FLAC is lossless.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Wav_Write_Read_16Bit_Mono_Roundtrips()
    {
        var samples = new[] { 0, 100, -100, 32767, -32768, 1234, -1234 };
        byte[] wav = WavFileCodec.Write(samples, sampleRateHz: 44100, channels: 1, bitsPerSample: 16);
        var parsed = WavFileCodec.Read(wav);
        Equal(44100, parsed.SampleRateHz);
        Equal(1, parsed.Channels);
        Equal(16, parsed.BitsPerSample);
        False(parsed.IsFloat);
        Equal(samples.Length, parsed.TotalSamplesPerChannel);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_Write_Read_16Bit_Stereo_Roundtrips()
    {
        var samples = new[] { 100, -100, 200, -200, 300, -300 };
        byte[] wav = WavFileCodec.Write(samples, 48000, 2, 16);
        var parsed = WavFileCodec.Read(wav);
        Equal(48000, parsed.SampleRateHz);
        Equal(2, parsed.Channels);
        Equal(3, parsed.TotalSamplesPerChannel);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_Write_Read_24Bit_Roundtrips()
    {
        // 24-bit signed range is [-8388608, 8388607].
        var samples = new[] { 0, 1000, -1000, 8388607, -8388608, 123456 };
        byte[] wav = WavFileCodec.Write(samples, 48000, 1, 24);
        var parsed = WavFileCodec.Read(wav);
        Equal(24, parsed.BitsPerSample);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_Write_Read_32Bit_Roundtrips()
    {
        var samples = new[] { 0, int.MaxValue, int.MinValue, 1_000_000, -1_000_000 };
        byte[] wav = WavFileCodec.Write(samples, 96000, 1, 32);
        var parsed = WavFileCodec.Read(wav);
        Equal(32, parsed.BitsPerSample);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_Write_Read_8Bit_UnsignedOffset()
    {
        // 8-bit WAV is UNSIGNED with 128 offset, but our API accepts signed in/out.
        var samples = new[] { 0, 50, -50, 127, -128 };
        byte[] wav = WavFileCodec.Write(samples, 22050, 1, 8);
        var parsed = WavFileCodec.Read(wav);
        Equal(8, parsed.BitsPerSample);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_Write_Read_8Channels()
    {
        // 8 channels x 10 samples = 80 interleaved int values.
        int samplesPerChannel = 10;
        int channels = 8;
        var samples = new int[samplesPerChannel * channels];
        for (int i = 0; i < samples.Length; i++) samples[i] = i * 37 - 500;
        byte[] wav = WavFileCodec.Write(samples, 48000, channels, 16);
        var parsed = WavFileCodec.Read(wav);
        Equal(8, parsed.Channels);
        EqualInts(samples, parsed.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_Write_NonAlignedSize_Throws()
    {
        bool threw = false;
        try { _ = WavFileCodec.Write(new int[5], 44100, 2, 16); }
        catch (ArgumentException) { threw = true; }
        True(threw, "Odd count with 2 channels should throw.");
    }

    [TestMethod]
    public void Wav_Read_BadRiff_Throws()
    {
        var bad = new byte[44];
        bad[0] = (byte)'X'; bad[1] = (byte)'I'; bad[2] = (byte)'F'; bad[3] = (byte)'F';
        bool threw = false;
        try { _ = WavFileCodec.Read(bad); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Wav_Read_MissingFmt_Throws()
    {
        // Build a RIFF/WAVE with only a data chunk, no fmt.
        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'R', (byte)'I', (byte)'F', (byte)'F' });
        bytes.AddRange(new byte[] { 0, 0, 0, 0 });
        bytes.AddRange(new byte[] { (byte)'W', (byte)'A', (byte)'V', (byte)'E' });
        bytes.AddRange(new byte[] { (byte)'d', (byte)'a', (byte)'t', (byte)'a' });
        bytes.AddRange(new byte[] { 4, 0, 0, 0 });
        bytes.AddRange(new byte[4]);
        bool threw = false;
        try { _ = WavFileCodec.Read(bytes.ToArray()); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Wav_WavToFlacToWav_RoundtripsLosslessly()
    {
        // WAV → FLAC → WAV should yield the exact same samples (FLAC is lossless).
        var input = GenerateSineInt(samplesPerChannel: 1024, channels: 2, sampleRateHz: 44100, bps: 16);
        byte[] wavBytes = WavFileCodec.Write(input, 44100, 2, 16);
        // Read the WAV, encode to FLAC.
        var wav = WavFileCodec.Read(wavBytes);
        byte[] flacBytes = SpawnDev.Codecs.Audio.Flac.FlacEncoder.EncodeStream(
            wav.InterleavedSamples, wav.SampleRateHz, wav.Channels, wav.BitsPerSample, blockSize: 1024);
        // Decode FLAC back and re-write as WAV.
        var flac = SpawnDev.Codecs.Audio.Flac.FlacDecoder.Decode(flacBytes);
        byte[] roundTripWav = WavFileCodec.Write(flac.InterleavedSamples, flac.StreamInfo.SampleRateHz,
            flac.StreamInfo.Channels, flac.StreamInfo.BitsPerSample);
        var finalWav = WavFileCodec.Read(roundTripWav);
        EqualInts(input, finalWav.InterleavedSamples);
    }

    [TestMethod]
    public void Wav_SkipsUnknownChunks()
    {
        // Build a WAV with a LIST chunk before the data chunk - parser should skip LIST.
        var fmt = new byte[16];
        fmt[0] = 0x01; // format tag PCM
        fmt[2] = 1;    // 1 channel
        // sample rate 44100 = 0xAC44 -> LE
        fmt[4] = 0x44; fmt[5] = 0xAC;
        // bits per sample 16
        fmt[14] = 16;
        // byte rate = 44100 * 1 * 2 = 88200 = 0x0158 88 hmm
        // Actually 88200 = 0x00015888. LE: 0x88, 0x58, 0x01, 0x00.
        fmt[8] = 0x88; fmt[9] = 0x58; fmt[10] = 0x01;
        fmt[12] = 2; // block align
        // LIST chunk with 4 bytes of junk
        var listChunk = new byte[] {
            (byte)'L', (byte)'I', (byte)'S', (byte)'T',
            4, 0, 0, 0,
            0xAA, 0xBB, 0xCC, 0xDD,
        };
        // data chunk with 2 samples (4 bytes) at 16-bit.
        short s0 = 1234, s1 = -5678;
        var dataChunk = new byte[] {
            (byte)'d', (byte)'a', (byte)'t', (byte)'a',
            4, 0, 0, 0,
            (byte)s0, (byte)(s0 >> 8),
            (byte)s1, (byte)(s1 >> 8),
        };
        var full = new List<byte>();
        full.AddRange(new byte[] { (byte)'R', (byte)'I', (byte)'F', (byte)'F' });
        full.AddRange(new byte[4]); // overall size, ignored
        full.AddRange(new byte[] { (byte)'W', (byte)'A', (byte)'V', (byte)'E' });
        full.AddRange(new byte[] { (byte)'f', (byte)'m', (byte)'t', (byte)' ' });
        full.AddRange(new byte[] { (byte)fmt.Length, 0, 0, 0 });
        full.AddRange(fmt);
        full.AddRange(listChunk);
        full.AddRange(dataChunk);

        var parsed = WavFileCodec.Read(full.ToArray());
        Equal(44100, parsed.SampleRateHz);
        Equal(1, parsed.Channels);
        Equal(16, parsed.BitsPerSample);
        EqualInts(new[] { (int)s0, (int)s1 }, parsed.InterleavedSamples);
    }
}
