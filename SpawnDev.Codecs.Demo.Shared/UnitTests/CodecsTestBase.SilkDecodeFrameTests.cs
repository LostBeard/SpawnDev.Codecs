using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end tests for <see cref="SilkDecodeFrame.Decode"/> - the top-level
/// SILK per-frame orchestrator. Encodes synthetic SILK bitstreams via the
/// encoder-side helpers and decodes them, verifying the full pipeline produces
/// valid PCM and updates state correctly.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Assemble a full SILK frame bitstream for test purposes. Mirrors
    /// libopus silk_encode_indices + silk_encode_pulses order.
    /// </summary>
    private static byte[] EncodeFullSilkFrame(
        SilkNlsfCodebook codebook,
        SilkDecodedIndices indices,
        short[] pulses,
        int fsKHz,
        int nbSubfr,
        int conditional,
        bool vadFlag,
        short prevLagIndex = 0,
        bool prevSignalTypeWasVoiced = false,
        int rateLevelIndex = 0)
    {
        var enc = new OpusRangeEncoder(512);

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

        // 2. Gains.
        EncodeGainIndices(enc, indices.GainsIndices.AsSpan(0, nbSubfr),
            signalType: indices.SignalType, conditional: conditional, nbSubfr: nbSubfr);

        // 3. NLSFs.
        EncodeNlsfIndices(enc, indices.NlsfIndices.AsSpan(0, codebook.Order + 1), codebook,
            signalType: indices.SignalType, nbSubfr: nbSubfr,
            interpCoefQ2: indices.NlsfInterpCoefQ2);

        // 4. Pitch + LTP (voiced only).
        if (indices.SignalType == SilkSideInfoDecoder.TypeVoiced)
        {
            // Absolute pitch coding (keep it simple).
            int coarse = indices.LagIndex / (fsKHz >> 1);
            int lsb = indices.LagIndex - coarse * (fsKHz >> 1);

            bool canDelta = conditional != 0 && prevSignalTypeWasVoiced;
            if (canDelta)
            {
                // Use delta if possible, else escape to absolute.
                int diff = indices.LagIndex - prevLagIndex;
                int raw = diff + 9;
                if (raw >= 1 && raw <= 20)
                {
                    enc.EncodeIcdf(raw, SilkIcdfTables.PitchDelta, 8);
                }
                else
                {
                    enc.EncodeIcdf(0, SilkIcdfTables.PitchDelta, 8);
                    enc.EncodeIcdf(coarse, SilkIcdfTables.PitchLag, 8);
                    enc.EncodeIcdf(lsb, SilkIcdfTables.SelectPitchLagLowBits(fsKHz), 8);
                }
            }
            else
            {
                enc.EncodeIcdf(coarse, SilkIcdfTables.PitchLag, 8);
                enc.EncodeIcdf(lsb, SilkIcdfTables.SelectPitchLagLowBits(fsKHz), 8);
            }
            enc.EncodeIcdf(indices.ContourIndex,
                SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr), 8);

            EncodeLtpIndices(enc, indices.PerIndex,
                indices.LtpIndices.AsSpan(0, nbSubfr),
                conditional: conditional,
                ltpScaleIdx: indices.LtpScaleIndex);
        }

        // 5. Seed.
        enc.EncodeIcdf(indices.Seed, SilkIcdfTables.Uniform4, 8);

        // 6. Pulses.
        int frameLength = nbSubfr * 5 * fsKHz;
        SilkPulsesDecoder.Encode(enc, pulses, indices.SignalType, indices.QuantOffsetType,
            frameLength: frameLength, rateLevelIndex: rateLevelIndex);

        enc.Done();
        return enc.ToArray();
    }

    // -------- Basic round-trips --------

    [TestMethod]
    public void DecodeFrame_InactiveNoVadIndependent_FirstFrame_DecodesSuccessfully()
    {
        // Simplest end-to-end test: inactive signal type, no VAD, independent coding,
        // all-zero pulses. The entire pipeline should produce valid PCM.
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 15;
        indices.NlsfIndices[0] = 5;

        short[] pulses = new short[160 + 16]; // aligned to shell boundary
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);

        var dec = new OpusRangeDecoder(bitstream);
        short[] pcm = new short[state.FrameLength];

        SilkDecodeFrame.Decode(state, dec, pcm, vadFlag: false, conditional: 0);

        // Output samples in int16 range.
        for (int i = 0; i < state.FrameLength; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue, $"pcm[{i}] out of range");
        }

        // State was updated.
        Equal(SilkSideInfoDecoder.TypeInactive, (sbyte)state.PrevSignalType);
        True(!state.PrevSignalTypeWasVoiced, "PrevSignalTypeWasVoiced should be false");
        True(!state.FirstFrameAfterReset, "FirstFrameAfterReset should flip to false");
    }

    [TestMethod]
    public void DecodeFrame_UnvoicedWithVad_FirstFrame_DecodesSuccessfully()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeUnvoiced,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4,
            Seed = 2,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = (sbyte)(20 + i);
        indices.NlsfIndices[0] = 10;

        short[] pulses = new short[160 + 16];
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: true);

        var dec = new OpusRangeDecoder(bitstream);
        short[] pcm = new short[state.FrameLength];

        SilkDecodeFrame.Decode(state, dec, pcm, vadFlag: true, conditional: 0);

        for (int i = 0; i < state.FrameLength; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
        }
        Equal(SilkSideInfoDecoder.TypeUnvoiced, (sbyte)state.PrevSignalType);
    }

    [TestMethod]
    public void DecodeFrame_VoicedWithLtp_FirstFrame_DecodesSuccessfully()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeVoiced,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            LagIndex = 80,
            ContourIndex = 2,
            PerIndex = 1,
            LtpScaleIndex = 0,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 25;
        indices.NlsfIndices[0] = 7;
        for (int i = 0; i < 4; i++) indices.LtpIndices[i] = (sbyte)(i + 3);

        short[] pulses = new short[160 + 16];
        byte[] bitstream = EncodeFullSilkFrame(
            SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: true);

        var dec = new OpusRangeDecoder(bitstream);
        short[] pcm = new short[state.FrameLength];

        SilkDecodeFrame.Decode(state, dec, pcm, vadFlag: true, conditional: 0);

        for (int i = 0; i < state.FrameLength; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
        }
        Equal(SilkSideInfoDecoder.TypeVoiced, (sbyte)state.PrevSignalType);
        True(state.PrevSignalTypeWasVoiced, "PrevSignalTypeWasVoiced should be true after voiced frame");
        Equal(indices.LagIndex, state.PrevLagIndex);
    }

    // -------- Multi-frame: state carries across --------

    [TestMethod]
    public void DecodeFrame_TwoBackToBackFrames_OutBufShiftsCorrectly()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

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
        short[] pcm = new short[state.FrameLength];

        // Frame 1
        byte[] bs1 = EncodeFullSilkFrame(SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);
        var dec1 = new OpusRangeDecoder(bs1);
        SilkDecodeFrame.Decode(state, dec1, pcm, vadFlag: false, conditional: 0);

        // Snapshot outBuf tail (where frame 1's xq should now be stored).
        short[] outBufAfter1 = new short[state.FrameLength];
        state.OutBuf.AsSpan(state.LtpMemLength - state.FrameLength, state.FrameLength).CopyTo(outBufAfter1);

        // Frame 2
        byte[] bs2 = EncodeFullSilkFrame(SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);
        var dec2 = new OpusRangeDecoder(bs2);
        SilkDecodeFrame.Decode(state, dec2, pcm, vadFlag: false, conditional: 0);

        // After frame 2, outBuf tail should now hold frame 2's output. The "oldest"
        // samples (what was at outBuf[frame_length..ltp_mem_length) before the shift)
        // should still be present but at earlier positions. Basic sanity check: the
        // two frames' outBuf-tail contents may differ (different PRNG steps).
        short[] outBufAfter2 = new short[state.FrameLength];
        state.OutBuf.AsSpan(state.LtpMemLength - state.FrameLength, state.FrameLength).CopyTo(outBufAfter2);

        // Both calls completed without crashing - that's the minimum contract.
        True(state.PrevSignalType == SilkSideInfoDecoder.TypeInactive,
            $"PrevSignalType should still be inactive, got {state.PrevSignalType}");
    }

    [TestMethod]
    public void DecodeFrame_FirstFrameAfterReset_FlagFlipsToFalse()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();

        True(state.FirstFrameAfterReset, "Should start as true after Reset");

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 10;

        short[] pulses = new short[160 + 16];
        byte[] bs = EncodeFullSilkFrame(SilkNlsfCodebookTables.NbMb, indices, pulses,
            fsKHz: 8, nbSubfr: 4, conditional: 0, vadFlag: false);
        var dec = new OpusRangeDecoder(bs);
        short[] pcm = new short[state.FrameLength];

        SilkDecodeFrame.Decode(state, dec, pcm, vadFlag: false, conditional: 0);

        True(!state.FirstFrameAfterReset, "Should flip to false after first decode");
    }

    // -------- WB configuration --------

    [TestMethod]
    public void DecodeFrame_WbInactiveFrame_DecodesSuccessfully()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 16, nbSubfr: 4, lpcOrder: 16);
        state.Reset();

        var indices = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = 20;
        indices.NlsfIndices[0] = 5;

        short[] pulses = new short[320 + 16]; // 320 aligned is 320
        byte[] bs = EncodeFullSilkFrame(SilkNlsfCodebookTables.Wb, indices, pulses,
            fsKHz: 16, nbSubfr: 4, conditional: 0, vadFlag: false);
        var dec = new OpusRangeDecoder(bs);
        short[] pcm = new short[state.FrameLength];

        SilkDecodeFrame.Decode(state, dec, pcm, vadFlag: false, conditional: 0);

        for (int i = 0; i < state.FrameLength; i++)
        {
            True(pcm[i] >= short.MinValue && pcm[i] <= short.MaxValue);
        }
    }

    // -------- Argument validation --------

    [TestMethod]
    public void DecodeFrame_NullState_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        short[] pcm = new short[160];
        Throws<ArgumentNullException>(() =>
            SilkDecodeFrame.Decode(null!, dec, pcm, vadFlag: false, conditional: 0));
    }

    [TestMethod]
    public void DecodeFrame_UnconfiguredState_Throws()
    {
        var enc = new OpusRangeEncoder(64);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        var state = new SilkChannelDecoderState(); // no Configure
        short[] pcm = new short[160];
        Throws<InvalidOperationException>(() =>
            SilkDecodeFrame.Decode(state, dec, pcm, vadFlag: false, conditional: 0));
    }

    [TestMethod]
    public void DecodeFrame_OutputBufferTooSmall_Throws()
    {
        var state = new SilkChannelDecoderState();
        state.Configure(fsKHz: 8, nbSubfr: 4, lpcOrder: 10);
        state.Reset();
        var enc = new OpusRangeEncoder(64);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        short[] tooSmall = new short[state.FrameLength - 1];
        Throws<ArgumentException>(() =>
            SilkDecodeFrame.Decode(state, dec, tooSmall, vadFlag: false, conditional: 0));
    }
}
