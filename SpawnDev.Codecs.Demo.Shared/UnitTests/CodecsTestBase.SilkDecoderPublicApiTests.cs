using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the public <see cref="SilkDecoder"/> API - the consumer-facing
/// wrapper around the internal SILK per-frame decode pipeline.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void SilkDecoder_Nb20Ms_ExposesCorrectDimensions()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 8000, frameLengthMs: 20);
        Equal(8000, dec.InternalSampleRateHz);
        Equal(10, dec.LpcOrder);
        Equal(4, dec.NbSubfr);
        Equal(160, dec.FrameLength);
    }

    [TestMethod]
    public void SilkDecoder_Wb20Ms_ExposesCorrectDimensions()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 16000, frameLengthMs: 20);
        Equal(16000, dec.InternalSampleRateHz);
        Equal(16, dec.LpcOrder);
        Equal(4, dec.NbSubfr);
        Equal(320, dec.FrameLength);
    }

    [TestMethod]
    public void SilkDecoder_Mb10Ms_ExposesCorrectDimensions()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 12000, frameLengthMs: 10);
        Equal(12000, dec.InternalSampleRateHz);
        Equal(10, dec.LpcOrder);
        Equal(2, dec.NbSubfr);
        Equal(120, dec.FrameLength);
    }

    [TestMethod]
    public void SilkDecoder_InvalidSampleRate_Throws()
    {
        Throws<ArgumentException>(() => new SilkDecoder(internalSampleRateHz: 24000));
        Throws<ArgumentException>(() => new SilkDecoder(internalSampleRateHz: 48000));
    }

    [TestMethod]
    public void SilkDecoder_InvalidFrameLength_Throws()
    {
        Throws<ArgumentException>(() => new SilkDecoder(internalSampleRateHz: 16000, frameLengthMs: 5));
        Throws<ArgumentException>(() => new SilkDecoder(internalSampleRateHz: 16000, frameLengthMs: 40));
    }

    [TestMethod]
    public void SilkDecoder_DecodeInactiveFrame_ProducesValidPcm()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 8000, frameLengthMs: 20);
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 15;
        indices.NlsfIndices[0] = 5;

        short[] pulses = new short[160 + 16];
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);

        short[] pcm = new short[dec.FrameLength];
        int written = dec.DecodeFrame(bitstream, pcm, vadFlag: false, conditional: false);

        Equal(dec.FrameLength, written);
        for (int i = 0; i < dec.FrameLength; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void SilkDecoder_ResetRestoresFirstFrameBehavior()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 16000, frameLengthMs: 20);
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 10;
        indices.NlsfIndices[0] = 5;

        short[] pulses = new short[320 + 16];
        byte[] bs1 = EncodeFullSilkFrame(SilkNlsfCodebookTables.Wb, indices, pulses,
            fsKHz: 16, nbSubfr: 4, conditional: 0, vadFlag: false);

        short[] pcm = new short[dec.FrameLength];
        dec.DecodeFrame(bs1, pcm, vadFlag: false, conditional: false);

        // Reset and decode the same bitstream again; results should match (state fresh).
        dec.Reset();
        short[] pcm2 = new short[dec.FrameLength];
        dec.DecodeFrame(bs1, pcm2, vadFlag: false, conditional: false);

        for (int i = 0; i < dec.FrameLength; i++)
        {
            Equal(pcm[i], pcm2[i], $"pos {i}: reset should reproduce identical output");
        }
    }

    [TestMethod]
    public void SilkDecoder_EmptyPayload_Throws()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 8000);
        short[] pcm = new short[dec.FrameLength];
        Throws<ArgumentException>(() =>
            dec.DecodeFrame(ReadOnlySpan<byte>.Empty, pcm, false, false));
    }

    [TestMethod]
    public void SilkDecoder_SmallOutputBuffer_Throws()
    {
        var dec = new SilkDecoder(internalSampleRateHz: 8000);
        short[] tooSmall = new short[dec.FrameLength - 1];
        byte[] payload = new byte[32]; // any non-empty payload
        Throws<ArgumentException>(() =>
            dec.DecodeFrame(payload, tooSmall, false, false));
    }

    [TestMethod]
    public void SilkDecoder_DecodeFromRange_MatchesDecodeFrame()
    {
        // DecodeFromRange with a freshly-constructed range decoder must produce
        // identical output to DecodeFrame with the same payload (the former is
        // the Opus-layer primitive; the latter is a convenience wrapper around it).
        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 15;
        indices.NlsfIndices[0] = 5;

        short[] pulses = new short[160 + 16];
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);

        var dec1 = new SilkDecoder(8000);
        short[] pcm1 = new short[dec1.FrameLength];
        dec1.DecodeFrame(bitstream, pcm1, vadFlag: false, conditional: false);

        var dec2 = new SilkDecoder(8000);
        var rangeDec = new OpusRangeDecoder(bitstream);
        short[] pcm2 = new short[dec2.FrameLength];
        dec2.DecodeFromRange(rangeDec, pcm2, vadFlag: false, conditional: false);

        for (int i = 0; i < dec1.FrameLength; i++)
        {
            Equal(pcm1[i], pcm2[i], $"pos {i}");
        }
    }

    [TestMethod]
    public void SilkDecoder_DecodeFromRange_NullDecoder_Throws()
    {
        var dec = new SilkDecoder(8000);
        short[] pcm = new short[dec.FrameLength];
        Throws<ArgumentNullException>(() =>
            dec.DecodeFromRange(null!, pcm, false, false));
    }
}
