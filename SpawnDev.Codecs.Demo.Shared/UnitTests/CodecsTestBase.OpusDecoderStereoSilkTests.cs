using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for stereo SILK routing in <see cref="OpusDecoder"/>. Builds minimal
/// stereo Opus SILK packets (TOC + VAD_mid + VAD_side + LBRR prefix + stereo
/// predictors + mid/side SILK indices) and verifies the decoder produces
/// interleaved L/R float PCM without crashing.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a single-internal-frame stereo Opus SILK packet. Each channel gets a
    /// separate SILK-indices block (mid then side) sharing the same range-coded payload.
    /// </summary>
    private static byte[] BuildStereoSilkOpusPacket(
        int tocConfig,
        SilkNlsfCodebook cb,
        SilkDecodedIndices midIdx,
        SilkDecodedIndices sideIdx,
        int fsKHz,
        int nbSubfr,
        bool vadMid,
        bool vadSide,
        // 6 predictor index triples (3 for each channel): ix0_0, ix0_1, ix0_2, ix1_0, ix1_1, ix1_2
        int p0a, int p0b, int p0c,
        int p1a, int p1b, int p1c)
    {
        byte tocByte = (byte)((tocConfig << 3) | (1 << 2)); // stereo bit set, frame-count 0

        var enc = new OpusRangeEncoder(1024);

        // VAD flags: mid, side, then LBRR (single frame = 1 triple).
        enc.EncodeBitLogP(vadMid ? 1 : 0, 1);
        enc.EncodeBitLogP(vadSide ? 1 : 0, 1);
        enc.EncodeBitLogP(0, 1); // no LBRR

        // Stereo predictors via the test-side helper from the SilkStereoDecodePredTests file.
        int joint = 5 * p0c + p1c;
        enc.EncodeIcdf(joint, SilkStereoDecodePred.StereoPredJointIcdf, 8);
        enc.EncodeIcdf(p0a, SilkIcdfTables.Uniform3, 8);
        enc.EncodeIcdf(p0b, SilkIcdfTables.Uniform5, 8);
        enc.EncodeIcdf(p1a, SilkIcdfTables.Uniform3, 8);
        enc.EncodeIcdf(p1b, SilkIcdfTables.Uniform5, 8);

        // Mid-only flag is read iff !vadSide. We set it to 0 for this test (both channels present).
        if (!vadSide)
        {
            enc.EncodeIcdf(0, SilkStereoDecodePred.StereoOnlyCodeMidIcdf, 8);
        }

        // Mid channel indices (independent = first frame).
        int combinedMid = midIdx.QuantOffsetType + 2 * midIdx.SignalType;
        if (vadMid) enc.EncodeIcdf(combinedMid - 2, SilkIcdfTables.TypeOffsetVad, 8);
        else enc.EncodeIcdf(combinedMid, SilkIcdfTables.TypeOffsetNoVad, 8);
        EncodeGainIndices(enc, midIdx.GainsIndices.AsSpan(0, nbSubfr),
            signalType: midIdx.SignalType, conditional: 0, nbSubfr: nbSubfr);
        EncodeNlsfIndices(enc, midIdx.NlsfIndices.AsSpan(0, cb.Order + 1), cb,
            signalType: midIdx.SignalType, nbSubfr: nbSubfr, interpCoefQ2: midIdx.NlsfInterpCoefQ2);
        enc.EncodeIcdf(midIdx.Seed, SilkIcdfTables.Uniform4, 8);
        int frameLength = nbSubfr * 5 * fsKHz;
        short[] midPulses = new short[((frameLength + 15) & ~15)];
        SilkPulsesDecoder.Encode(enc, midPulses, midIdx.SignalType, midIdx.QuantOffsetType,
            frameLength: frameLength, rateLevelIndex: 0);

        // Side channel indices (always decoded when !mid-only).
        int combinedSide = sideIdx.QuantOffsetType + 2 * sideIdx.SignalType;
        if (vadSide) enc.EncodeIcdf(combinedSide - 2, SilkIcdfTables.TypeOffsetVad, 8);
        else enc.EncodeIcdf(combinedSide, SilkIcdfTables.TypeOffsetNoVad, 8);
        EncodeGainIndices(enc, sideIdx.GainsIndices.AsSpan(0, nbSubfr),
            signalType: sideIdx.SignalType, conditional: 0, nbSubfr: nbSubfr);
        EncodeNlsfIndices(enc, sideIdx.NlsfIndices.AsSpan(0, cb.Order + 1), cb,
            signalType: sideIdx.SignalType, nbSubfr: nbSubfr, interpCoefQ2: sideIdx.NlsfInterpCoefQ2);
        enc.EncodeIcdf(sideIdx.Seed, SilkIcdfTables.Uniform4, 8);
        short[] sidePulses = new short[((frameLength + 15) & ~15)];
        SilkPulsesDecoder.Encode(enc, sidePulses, sideIdx.SignalType, sideIdx.QuantOffsetType,
            frameLength: frameLength, rateLevelIndex: 0);

        enc.Done();
        byte[] payload = enc.ToArray();
        byte[] packet = new byte[1 + payload.Length];
        packet[0] = tocByte;
        payload.CopyTo(packet, 1);
        return packet;
    }

    [TestMethod]
    public void OpusDecoder_StereoNbInactive20Ms_InterleavesLR()
    {
        // Config 1 (NB 20 ms) + stereo bit. Both channels inactive (VAD=false, signalType=0).
        var mid = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) mid.GainsIndices[i] = 12;
        mid.NlsfIndices[0] = 4;

        var side = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) side.GainsIndices[i] = 8;
        side.NlsfIndices[0] = 6;

        // Both channels inactive -> vadMid = vadSide = false. vadSide = false triggers
        // the mid-only flag read (which we emit as 0 = both channels coded).
        byte[] packet = BuildStereoSilkOpusPacket(
            tocConfig: 1, cb: SilkNlsfCodebookTables.NbMb,
            midIdx: mid, sideIdx: side,
            fsKHz: 8, nbSubfr: 4, vadMid: false, vadSide: false,
            p0a: 1, p0b: 2, p0c: 1,
            p1a: 1, p1b: 2, p1c: 1);

        // Stereo output at the internal SILK rate (8 kHz) - current stereo slice requires this.
        var config = new OpusDecoderConfig { SampleRateHz = 8000, ChannelCount = 2 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[160 * 2]; // 20 ms at 8 kHz = 160 per channel; interleaved = 320 floats
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(160, samples); // per-channel sample count
        // Every interleaved float in range.
        for (int i = 0; i < pcm.Length; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f, $"pcm[{i}] = {pcm[i]} out of range");
        }
    }

    [TestMethod]
    public void OpusDecoder_StereoWb20Ms_InterleavesLR()
    {
        // Config 9 (WB 20 ms) + stereo.
        var mid = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeUnvoiced,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 2,
        };
        for (int i = 0; i < 4; i++) mid.GainsIndices[i] = 20;
        mid.NlsfIndices[0] = 7;

        var side = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeUnvoiced, // matches vadSide=true
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) side.GainsIndices[i] = 10;
        side.NlsfIndices[0] = 3;

        byte[] packet = BuildStereoSilkOpusPacket(
            tocConfig: 9, cb: SilkNlsfCodebookTables.Wb,
            midIdx: mid, sideIdx: side,
            fsKHz: 16, nbSubfr: 4, vadMid: true, vadSide: true,
            p0a: 0, p0b: 2, p0c: 1,
            p1a: 1, p1b: 2, p1c: 1);

        var config = new OpusDecoderConfig { SampleRateHz = 16000, ChannelCount = 2 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[320 * 2]; // 20 ms at 16 kHz = 320 per channel
        int samples = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result;

        Equal(320, samples);
        for (int i = 0; i < pcm.Length; i++)
        {
            True(pcm[i] >= -1.0f && pcm[i] <= 1.0f);
        }
    }

    [TestMethod]
    public void OpusDecoder_StereoAtNon_InternalRate_ThrowsUntilWired()
    {
        // Until the stereo-path resampler wiring ships, stereo output must be at
        // the internal rate. Requesting 48 kHz output for NB stereo should throw.
        var mid = new SilkDecodedIndices
        {
            SignalType = SilkSideInfoDecoder.TypeInactive,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
        };
        for (int i = 0; i < 4; i++) mid.GainsIndices[i] = 10;
        mid.NlsfIndices[0] = 3;
        var side = mid;

        byte[] packet = BuildStereoSilkOpusPacket(
            tocConfig: 1, cb: SilkNlsfCodebookTables.NbMb,
            midIdx: mid, sideIdx: side,
            fsKHz: 8, nbSubfr: 4, vadMid: false, vadSide: false,
            p0a: 1, p0b: 2, p0c: 1, p1a: 1, p1b: 2, p1c: 1);

        var config = new OpusDecoderConfig { SampleRateHz = 48000, ChannelCount = 2 };
        var dec = new OpusDecoder(config);
        float[] pcm = new float[960 * 2];

        bool threw = false;
        try { _ = dec.DecodePacketAsync(packet.AsMemory(), pcm.AsMemory()).Result; }
        catch (NotImplementedException) { threw = true; }
        catch (AggregateException ae) when (ae.InnerException is NotImplementedException) { threw = true; }

        True(threw, "Expected stereo + non-internal output rate to throw NotImplementedException");
    }
}
