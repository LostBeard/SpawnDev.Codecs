using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for 40 ms (config 2/6/10) and 60 ms (config 3/7/11) SILK-only Opus
/// packets. Each contains 2 or 3 internal 20 ms SILK frames that share the
/// same range-coded payload and use conditional gain+NLSF coding after the
/// first frame.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a multi-internal-frame Opus SILK-only packet. Each frame uses the
    /// same <paramref name="indices"/> template for simplicity; the first frame
    /// is independent and subsequent frames are conditional.
    /// </summary>
    private static byte[] BuildMultiFrameSilkOnlyOpusPacket(
        int tocConfig,
        SilkNlsfCodebook cb,
        SilkDecodedIndices indices,
        int fsKHz,
        int nbSubfr,
        int silkFrameCount,
        bool vadFlag)
    {
        byte tocByte = (byte)(tocConfig << 3); // mono, frame-count 0

        var enc = new OpusRangeEncoder(512);

        // VAD flags (one per SILK frame) + LBRR flag (single).
        for (int f = 0; f < silkFrameCount; f++)
        {
            enc.EncodeBitLogP(vadFlag ? 1 : 0, 1);
        }
        enc.EncodeBitLogP(0, 1); // no LBRR

        // Encode each SILK frame's indices. First = independent, subsequent = conditional.
        for (int f = 0; f < silkFrameCount; f++)
        {
            int conditional = f > 0 ? 1 : 0;

            int combined = indices.QuantOffsetType + 2 * indices.SignalType;
            if (vadFlag)
            {
                enc.EncodeIcdf(combined - 2, SilkIcdfTables.TypeOffsetVad, 8);
            }
            else
            {
                enc.EncodeIcdf(combined, SilkIcdfTables.TypeOffsetNoVad, 8);
            }

            EncodeGainIndices(enc, indices.GainsIndices.AsSpan(0, nbSubfr),
                signalType: indices.SignalType, conditional: conditional, nbSubfr: nbSubfr);

            EncodeNlsfIndices(enc, indices.NlsfIndices.AsSpan(0, cb.Order + 1), cb,
                signalType: indices.SignalType, nbSubfr: nbSubfr,
                interpCoefQ2: indices.NlsfInterpCoefQ2);

            enc.EncodeIcdf(indices.Seed, SilkIcdfTables.Uniform4, 8);

            int frameLength = nbSubfr * 5 * fsKHz;
            short[] pulses = new short[((frameLength + 15) & ~15)];
            SilkPulsesDecoder.Encode(enc, pulses, indices.SignalType, indices.QuantOffsetType,
                frameLength: frameLength, rateLevelIndex: 0);
        }

        enc.Done();
        byte[] payload = enc.ToArray();
        byte[] packet = new byte[1 + payload.Length];
        packet[0] = tocByte;
        payload.CopyTo(packet, 1);
        return packet;
    }

    [TestMethod]
    public void OpusDecoder_Nb40Ms_TwoInternalFrames_DecodesCorrectTotalSamples()
    {
        // Config 2: NB SILK 40 ms, mono. 2 x 20 ms internal SILK frames.
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 15;
        indices.NlsfIndices[0] = 5;

        byte[] packet = BuildMultiFrameSilkOnlyOpusPacket(
            tocConfig: 2, cb: SilkNlsfCodebookTables.NbMb, indices: indices,
            fsKHz: 8, nbSubfr: 4, silkFrameCount: 2, vadFlag: false);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[48 * 40]; // 40 ms at 48 kHz = 1920 samples
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(1920, samples);
        for (int i = 0; i < samples; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f, $"pcm[{i}] out of range");
        }
    }

    [TestMethod]
    public void OpusDecoder_Wb60Ms_ThreeInternalFrames_DecodesCorrectTotalSamples()
    {
        // Config 11: WB SILK 60 ms, mono. 3 x 20 ms internal SILK frames.
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeUnvoiced,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 20;
        indices.NlsfIndices[0] = 7;

        byte[] packet = BuildMultiFrameSilkOnlyOpusPacket(
            tocConfig: 11, cb: SilkNlsfCodebookTables.Wb, indices: indices,
            fsKHz: 16, nbSubfr: 4, silkFrameCount: 3, vadFlag: true);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[48 * 60]; // 60 ms at 48 kHz = 2880 samples
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(2880, samples);
        for (int i = 0; i < samples; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f, $"pcm[{i}] out of range");
        }
    }

    [TestMethod]
    public void OpusDecoder_Mb40Ms_TwoInternalFrames_AtNative12kHz()
    {
        // Config 6: MB SILK 40 ms, mono. Output at 12 kHz (no resampling).
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4,
            Seed = 2,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 10;
        indices.NlsfIndices[0] = 3;

        byte[] packet = BuildMultiFrameSilkOnlyOpusPacket(
            tocConfig: 6, cb: SilkNlsfCodebookTables.NbMb, indices: indices,
            fsKHz: 12, nbSubfr: 4, silkFrameCount: 2, vadFlag: false);

        var config = new OpusDecoderConfig { SampleRateHz = 12000, ChannelCount = 1 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[12 * 40]; // 40 ms at 12 kHz = 480 samples
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(480, samples);
        for (int i = 0; i < samples; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
        }
    }
}
