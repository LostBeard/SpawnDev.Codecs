using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the SILK routing inside <see cref="OpusDecoder"/>. Encodes synthetic
/// Opus SILK-mode packets (TOC + VAD/LBRR flags + SILK indices) and verifies
/// the full Opus-level decode produces valid float PCM at the configured output rate.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a single-frame SILK-only Opus packet. Emits TOC byte + VAD (1) + LBRR (0)
    /// prefix bits, then encodes the SILK frame indices via <see cref="EncodeFullSilkFrame"/>
    /// into the same range-coded bitstream.
    /// </summary>
    private static byte[] BuildSilkOnlyOpusPacket(
        int tocConfig,
        bool stereo,
        SilkNlsfCodebook cb,
        SilkDecodedIndices indices,
        int fsKHz,
        int nbSubfr,
        bool vadFlag)
    {
        // TOC byte: config (5 bits) | stereo (1 bit) | frame-count-code (2 bits, single frame = 0).
        byte tocByte = (byte)((tocConfig << 3) | ((stereo ? 1 : 0) << 2));

        // Encode the SILK bitstream: VAD flag + LBRR=0 + SILK indices + pulses.
        var enc = new OpusRangeEncoder(512);
        enc.EncodeBitLogP(vadFlag ? 1 : 0, 1);
        enc.EncodeBitLogP(0, 1); // no LBRR

        // 1. Signal type + offset.
        int combined = indices.QuantOffsetType + 2 * indices.SignalType;
        if (vadFlag)
        {
            enc.EncodeIcdf(combined - 2, SilkIcdfTables.TypeOffsetVad, 8);
        }
        else
        {
            enc.EncodeIcdf(combined, SilkIcdfTables.TypeOffsetNoVad, 8);
        }

        // 2. Gains (independent, since we're on a first frame).
        EncodeGainIndices(enc, indices.GainsIndices.AsSpan(0, nbSubfr),
            signalType: indices.SignalType, conditional: 0, nbSubfr: nbSubfr);

        // 3. NLSFs.
        EncodeNlsfIndices(enc, indices.NlsfIndices.AsSpan(0, cb.Order + 1), cb,
            signalType: indices.SignalType, nbSubfr: nbSubfr,
            interpCoefQ2: indices.NlsfInterpCoefQ2);

        // 4. Pitch + LTP (voiced only) - skipped for simplicity: use inactive/unvoiced only.

        // 5. Seed.
        enc.EncodeIcdf(indices.Seed, SilkIcdfTables.Uniform4, 8);

        // 6. Pulses.
        int frameLength = nbSubfr * 5 * fsKHz;
        short[] pulses = new short[((frameLength + 15) & ~15)]; // aligned to shell boundary
        SilkPulsesDecoder.Encode(enc, pulses, indices.SignalType, indices.QuantOffsetType,
            frameLength: frameLength, rateLevelIndex: 0);

        enc.Done();
        byte[] silkPayload = enc.ToArray();

        // Prepend TOC byte.
        byte[] packet = new byte[1 + silkPayload.Length];
        packet[0] = tocByte;
        silkPayload.CopyTo(packet, 1);
        return packet;
    }

    [TestMethod]
    public void OpusDecoder_NbInactive20Ms_SingleFrame_DecodesViaSilkRouting()
    {
        // Config 1: NB SILK 20 ms, mono. Single Opus frame (frame-count-code 0).
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 15;
        indices.NlsfIndices[0] = 5;

        byte[] packet = BuildSilkOnlyOpusPacket(
            tocConfig: 1, stereo: false,
            cb: SilkNlsfCodebookTables.NbMb,
            indices: indices,
            fsKHz: 8, nbSubfr: 4, vadFlag: false);

        // Decoder configured for 48 kHz output (most common Opus API rate).
        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[960]; // 20 ms at 48 kHz
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(960, samples);
        // Output in [-1, 1].
        for (int i = 0; i < samples; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f, $"pcm[{i}] = {pcm[i]} out of [-1, 1]");
        }
    }

    [TestMethod]
    public void OpusDecoder_WbUnvoiced20Ms_SingleFrame_DecodesViaSilkRouting()
    {
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeUnvoiced,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 20;
        indices.NlsfIndices[0] = 7;

        // Config 9: WB SILK 20 ms, mono.
        byte[] packet = BuildSilkOnlyOpusPacket(
            tocConfig: 9, stereo: false,
            cb: SilkNlsfCodebookTables.Wb,
            indices: indices,
            fsKHz: 16, nbSubfr: 4, vadFlag: true);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[960];
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(960, samples);
        for (int i = 0; i < samples; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
        }
    }

    [TestMethod]
    public void OpusDecoder_SilkAt16kHzOutput_NoResamplingPath()
    {
        // Output rate equal to internal SILK rate -> no resampler instantiated.
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 10;
        indices.NlsfIndices[0] = 3;

        byte[] packet = BuildSilkOnlyOpusPacket(
            tocConfig: 9, stereo: false, // WB SILK 20 ms
            cb: SilkNlsfCodebookTables.Wb,
            indices: indices,
            fsKHz: 16, nbSubfr: 4, vadFlag: false);

        var config = new OpusDecoderConfig { SampleRateHz = 16000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[320]; // 20 ms at 16 kHz
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(320, samples);
        for (int i = 0; i < samples; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
        }
    }

    [TestMethod]
    public void OpusDecoder_StereoSilk_NotYetImplemented_Throws()
    {
        // Stereo SILK is not yet wired; verify we throw the documented exception.
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 10;
        indices.NlsfIndices[0] = 3;

        byte[] packet = BuildSilkOnlyOpusPacket(
            tocConfig: 1, stereo: true,
            cb: SilkNlsfCodebookTables.NbMb,
            indices: indices,
            fsKHz: 8, nbSubfr: 4, vadFlag: false);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 2 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[960 * 2];

        // Stereo SILK is not yet wired; verify the decode throws NotImplementedException.
        bool threw = false;
        try
        {
            _ = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;
        }
        catch (NotImplementedException)
        {
            threw = true;
        }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException)
        {
            threw = true;
        }
        True(threw, "Expected stereo SILK decode to throw NotImplementedException");
    }
}
