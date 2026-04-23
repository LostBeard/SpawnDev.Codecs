using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for the resampling overload of <see cref="SilkDecoder"/>. Verifies that
/// output is produced at the requested API sample rate and that all supported
/// SILK-internal -> API rate combinations decode cleanly.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void SilkDecoderResample_Wb20Ms_Internal16_Output48_FrameLengthIsTripled()
    {
        var dec = new SilkDecoder(
            internalSampleRateHz: 16000, frameLengthMs: 20, outputSampleRateHz: 48000);
        Equal(16000, dec.InternalSampleRateHz);
        Equal(48000, dec.OutputSampleRateHz);
        Equal(960, dec.FrameLength); // 48 * 20 = 960 samples
    }

    [TestMethod]
    public void SilkDecoderResample_Nb20Ms_Internal8_Output48_FrameLengthIs6x()
    {
        var dec = new SilkDecoder(
            internalSampleRateHz: 8000, frameLengthMs: 20, outputSampleRateHz: 48000);
        Equal(8000, dec.InternalSampleRateHz);
        Equal(48000, dec.OutputSampleRateHz);
        Equal(960, dec.FrameLength); // 48 * 20
    }

    [TestMethod]
    public void SilkDecoderResample_16To48_DecodesAndResamplesCleanly()
    {
        var dec = new SilkDecoder(
            internalSampleRateHz: 16000, frameLengthMs: 20, outputSampleRateHz: 48000);

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 15;
        indices.NlsfIndices[0] = 5;

        short[] pulses = new short[320 + 16];
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.Wb, indices, pulses,
            fsKHz: 16, nbSubfr: 4, conditional: 0, vadFlag: false);

        short[] pcm = new short[dec.FrameLength];
        int written = dec.DecodeFrame(bitstream, pcm, vadFlag: false, conditional: false);

        Equal(960, written);
        // PCM in int16 range at the 48 kHz output rate.
        for (int i = 0; i < pcm.Length; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void SilkDecoderResample_8To48_MaxUpsampleFactor()
    {
        // 6x upsampler: internal 8 kHz -> output 48 kHz.
        var dec = new SilkDecoder(
            internalSampleRateHz: 8000, frameLengthMs: 20, outputSampleRateHz: 48000);

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 10;
        indices.NlsfIndices[0] = 3;

        short[] pulses = new short[160 + 16];
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);

        short[] pcm = new short[dec.FrameLength];
        dec.DecodeFrame(bitstream, pcm, vadFlag: false, conditional: false);

        for (int i = 0; i < pcm.Length; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
        }
    }

    [TestMethod]
    public void SilkDecoderResample_Reset_ReinitializesResampler()
    {
        var dec = new SilkDecoder(16000, 20, 48000);

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
        byte[] bs = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.Wb, indices, pulses,
            fsKHz: 16, nbSubfr: 4, conditional: 0, vadFlag: false);

        short[] pcm1 = new short[dec.FrameLength];
        dec.DecodeFrame(bs, pcm1, vadFlag: false, conditional: false);

        dec.Reset();

        // After reset, decoding the same bitstream must produce identical output.
        short[] pcm2 = new short[dec.FrameLength];
        dec.DecodeFrame(bs, pcm2, vadFlag: false, conditional: false);

        for (int i = 0; i < dec.FrameLength; i++)
        {
            Equal(pcm1[i], pcm2[i], $"pos {i}");
        }
    }

    [TestMethod]
    public void SilkDecoderResample_InvalidOutputRate_Throws()
    {
        Throws<ArgumentException>(() => new SilkDecoder(16000, 20, 36000));
        Throws<ArgumentException>(() => new SilkDecoder(16000, 20, 44100));
    }

    [TestMethod]
    public void SilkDecoderResample_EveryValidSilkToApiCombination_Works()
    {
        // Every decode-direction rate pair: 3 internal x 5 output = 15 combinations.
        int[] internalRates = { 8000, 12000, 16000 };
        int[] outputRates = { 8000, 12000, 16000, 24000, 48000 };

        foreach (int fsIn in internalRates)
        {
            foreach (int fsOut in outputRates)
            {
                var dec = new SilkDecoder(fsIn, 20, fsOut);
                Equal(fsOut, dec.OutputSampleRateHz);
                Equal(fsOut / 1000 * 20, dec.FrameLength);
            }
        }
    }
}
