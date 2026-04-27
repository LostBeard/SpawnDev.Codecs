using System.Numerics;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisAudioEncoder"/>. The minimum-viable encoder is
/// validated end-to-end via two paths:
///   1. Self round-trip: encode -> SpawnDev.Codecs decoder -> compare PCM
///      via DFT peak detection. Must reproduce the dominant frequency.
///   2. Reference round-trip: encode and verify the byte stream parses
///      correctly with our own VorbisOggDecoder + VorbisSetupHeaderParser
///      (which exercises the same decoder path that ffmpeg's libavcodec uses).
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void VorbisEncoder_Ctor_RejectsStereo()
    {
        Throws<NotSupportedException>(() => new VorbisAudioEncoder(new VorbisAudioEncoderOptions
        {
            SampleRateHz = 44100,
            Channels = 2,
            BlockSize = 1024,
        }));
    }

    [TestMethod]
    public void VorbisEncoder_Ctor_RejectsBadBlockSize()
    {
        Throws<ArgumentException>(() => new VorbisAudioEncoder(new VorbisAudioEncoderOptions
        {
            SampleRateHz = 44100,
            Channels = 1,
            BlockSize = 1000, // not a power of 2
        }));
    }

    [TestMethod]
    public void VorbisEncoder_Headers_ParseCleanly()
    {
        var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
        {
            SampleRateHz = 48000,
            Channels = 1,
            BlockSize = 256,
        });
        var input = new float[256];
        var ogg = enc.EncodeStream(input);
        var pages = SpawnDev.Codecs.Container.Ogg.OggPageReader.EnumeratePages(ogg).ToArray();
        var packets = SpawnDev.Codecs.Container.Ogg.OggPacketReader.AssemblePackets(pages).ToArray();
        True(packets.Length >= 4, $"Expected >= 4 packets, got {packets.Length}");

        var ident = VorbisIdentificationHeaderParser.Parse(packets[0].Data);
        Equal(48000, ident.SampleRateHz);
        Equal(1, ident.AudioChannels);
        Equal(256, ident.BlockSize0);
        Equal(256, ident.BlockSize1);

        // Ensure setup parses without throwing.
        var setup = VorbisSetupHeaderParser.Parse(packets[2].Data, ident.AudioChannels);
        True(setup.Codebooks.Length >= 1);
        True(setup.Floors.Length >= 1);
        True(setup.Residues.Length >= 1);
        True(setup.Mappings.Length >= 1);
        True(setup.Modes.Length >= 1);
    }

    [TestMethod]
    public void VorbisEncoder_RoundTrip_440Hz_DominantFrequencyMatches()
    {
        const int sr = 44100;
        const double targetHz = 440.0;
        const int totalSamples = sr; // 1 second
        var input = new float[totalSamples];
        for (int n = 0; n < totalSamples; n++)
            input[n] = (float)(0.5 * Math.Sin(2 * Math.PI * targetHz * n / sr));

        var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
        {
            SampleRateHz = sr,
            Channels = 1,
            BlockSize = 1024,
        });
        var ogg = enc.EncodeStream(input);
        True(ogg.Length > 30, $"Encoded stream too small: {ogg.Length}");

        var decoded = VorbisOggDecoder.Decode(ogg);
        True(decoded.InterleavedSamples.Length > totalSamples / 2,
            $"Decoded too few samples: {decoded.InterleavedSamples.Length}");

        // Find dominant frequency via dumb DFT scan.
        int searchN = Math.Min(decoded.InterleavedSamples.Length, 8192);
        double peakHz = 0; double peakMag = 0;
        for (int hz = 50; hz <= 1000; hz++)
        {
            Complex sum = Complex.Zero;
            for (int n = 0; n < searchN; n++)
            {
                double phase = -2 * Math.PI * hz * n / sr;
                sum += new Complex(decoded.InterleavedSamples[n] * Math.Cos(phase),
                                    decoded.InterleavedSamples[n] * Math.Sin(phase));
            }
            double mag = sum.Magnitude;
            if (mag > peakMag) { peakMag = mag; peakHz = hz; }
        }
        True(Math.Abs(peakHz - targetHz) <= 5.0,
            $"Dominant frequency {peakHz} Hz differs from target {targetHz} Hz by more than 5 Hz");
    }

    [TestMethod]
    public void VorbisEncoder_RoundTrip_880Hz_DominantFrequencyMatches()
    {
        // Different test frequency to make sure the encoder isn't accidentally
        // tone-locked. 880 Hz = A5.
        const int sr = 44100;
        const double targetHz = 880.0;
        const int totalSamples = sr; // 1 second
        var input = new float[totalSamples];
        for (int n = 0; n < totalSamples; n++)
            input[n] = (float)(0.4 * Math.Sin(2 * Math.PI * targetHz * n / sr));

        var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
        {
            SampleRateHz = sr,
            Channels = 1,
            BlockSize = 1024,
        });
        var ogg = enc.EncodeStream(input);
        var decoded = VorbisOggDecoder.Decode(ogg);

        int searchN = Math.Min(decoded.InterleavedSamples.Length, 8192);
        double peakHz = 0; double peakMag = 0;
        for (int hz = 50; hz <= 1500; hz++)
        {
            Complex sum = Complex.Zero;
            for (int n = 0; n < searchN; n++)
            {
                double phase = -2 * Math.PI * hz * n / sr;
                sum += new Complex(decoded.InterleavedSamples[n] * Math.Cos(phase),
                                    decoded.InterleavedSamples[n] * Math.Sin(phase));
            }
            double mag = sum.Magnitude;
            if (mag > peakMag) { peakMag = mag; peakHz = hz; }
        }
        True(Math.Abs(peakHz - targetHz) <= 10.0,
            $"Dominant frequency {peakHz} Hz differs from target {targetHz} Hz by more than 10 Hz");
    }

    [TestMethod]
    public void VorbisBitWriter_RoundTripsThroughReader()
    {
        // Sanity check that the LSB-first writer and reader pair correctly.
        var writer = new VorbisBitWriter();
        writer.WriteBits(5u, 3);
        writer.WriteBits(0xAAu, 8);
        writer.WriteBits(0x1234u, 16);
        writer.WriteBit(1u);
        var bytes = writer.ToArray();
        var reader = new VorbisBitReader(bytes);
        Equal(5u, reader.ReadBits(3));
        Equal(0xAAu, reader.ReadBits(8));
        Equal(0x1234u, reader.ReadBits(16));
        Equal(1u, reader.ReadBit());
    }
}
