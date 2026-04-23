using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// End-to-end integration tests for <see cref="SilkIndicesDecoder.Decode"/> - the
/// top-level silk_decode_indices orchestrator. Encodes a known side-information
/// block end-to-end (mirroring libopus silk_encode_indices), then verifies that
/// the Decode driver reads back every field in the right order.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Test-side encoder that mirrors libopus silk_encode_indices. Only the paths
    /// exercised by the current tests are handled (full matrix of vad / voiced /
    /// conditional combinations).
    /// </summary>
    private static void EncodeFullIndices(
        OpusRangeEncoder enc,
        SilkDecodedIndices indices,
        SilkNlsfCodebook cb,
        bool vadFlag,
        bool decodeLbrr,
        int fsKHz,
        int nbSubfr,
        int conditional,
        short prevLagIndex,
        bool prevSignalTypeWasVoiced)
    {
        // 1. Signal type + offset.
        bool useVad = vadFlag || decodeLbrr;
        int combined = indices.QuantOffsetType + 2 * indices.SignalType;
        if (useVad)
        {
            // Effective Ix (from decoder side) = raw + 2 -> raw = combined - 2.
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
        EncodeNlsfIndices(enc, indices.NlsfIndices.AsSpan(0, cb.Order + 1), cb,
            signalType: indices.SignalType, nbSubfr: nbSubfr,
            interpCoefQ2: indices.NlsfInterpCoefQ2);

        // 4. Pitch + LTP (voiced only).
        if (indices.SignalType == SilkSideInfoDecoder.TypeVoiced)
        {
            // Pitch lag absolute encode (we always send absolute in these tests unless
            // conditional && prev voiced AND we use delta path - keep it simple by using absolute).
            bool canDelta = conditional != 0 && prevSignalTypeWasVoiced;
            if (!canDelta)
            {
                // Absolute path.
                int coarse = indices.LagIndex / (fsKHz >> 1);
                int lsb = indices.LagIndex - coarse * (fsKHz >> 1);
                enc.EncodeIcdf(coarse, SilkIcdfTables.PitchLag, 8);
                enc.EncodeIcdf(lsb, SilkIcdfTables.SelectPitchLagLowBits(fsKHz), 8);
            }
            else
            {
                // Delta-encode: we need LagIndex = prevLagIndex + (raw - 9) for some raw in [1, 20].
                // If the difference is in range, use delta; otherwise escape to absolute.
                int diff = indices.LagIndex - prevLagIndex;
                int raw = diff + 9;
                if (raw >= 1 && raw <= 20)
                {
                    enc.EncodeIcdf(raw, SilkIcdfTables.PitchDelta, 8);
                }
                else
                {
                    // Escape: emit raw=0 then absolute.
                    enc.EncodeIcdf(0, SilkIcdfTables.PitchDelta, 8);
                    int coarse = indices.LagIndex / (fsKHz >> 1);
                    int lsb = indices.LagIndex - coarse * (fsKHz >> 1);
                    enc.EncodeIcdf(coarse, SilkIcdfTables.PitchLag, 8);
                    enc.EncodeIcdf(lsb, SilkIcdfTables.SelectPitchLagLowBits(fsKHz), 8);
                }
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
    }

    [TestMethod]
    public void IndicesDecoder_Inactive_NoVad_Independent_NbMb_RoundTrip()
    {
        // Simplest path: inactive (signalType=0), no VAD -> uses 2-symbol TypeOffsetNoVad iCDF.
        // Non-voiced means no pitch/LTP block. Conditional = 0 so gains are independent.
        var cb = SilkNlsfCodebookTables.NbMb;
        var indices = new SilkDecodedIndices
        {
            SignalType = 0,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4,
            Seed = 2,
        };
        indices.GainsIndices[0] = 45;
        indices.GainsIndices[1] = 10;
        indices.GainsIndices[2] = 5;
        indices.GainsIndices[3] = 20;
        indices.NlsfIndices[0] = 7;
        // residuals 1..10 left at 0

        var enc = new OpusRangeEncoder(256);
        EncodeFullIndices(enc, indices, cb,
            vadFlag: false, decodeLbrr: false, fsKHz: 8, nbSubfr: 4,
            conditional: 0, prevLagIndex: 0, prevSignalTypeWasVoiced: false);
        enc.Done();

        var decoded = new SilkDecodedIndices();
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkIndicesDecoder.Decode(decoded, dec, cb,
            vadFlag: false, decodeLbrr: false, fsKHz: 8, nbSubfr: 4,
            conditional: 0, prevLagIndex: 0, prevSignalTypeWasVoiced: false);

        Equal(indices.SignalType, decoded.SignalType);
        Equal(indices.QuantOffsetType, decoded.QuantOffsetType);
        for (int i = 0; i < 4; i++) Equal(indices.GainsIndices[i], decoded.GainsIndices[i], $"gain[{i}]");
        for (int i = 0; i <= cb.Order; i++) Equal(indices.NlsfIndices[i], decoded.NlsfIndices[i], $"nlsf[{i}]");
        Equal(indices.NlsfInterpCoefQ2, decoded.NlsfInterpCoefQ2);
        Equal(indices.Seed, decoded.Seed);
        // Voiced-only fields should be zero for inactive frames.
        Equal((short)0, decoded.LagIndex);
        Equal((sbyte)0, decoded.ContourIndex);
        Equal((sbyte)0, decoded.PerIndex);
        Equal((sbyte)0, decoded.LtpScaleIndex);
    }

    [TestMethod]
    public void IndicesDecoder_Unvoiced_Vad_Conditional_NbMb_RoundTrip()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        var indices = new SilkDecodedIndices
        {
            SignalType = 1,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 2,
            Seed = 0,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = (sbyte)(5 + i);
        indices.NlsfIndices[0] = 15;
        for (int i = 1; i <= cb.Order; i++) indices.NlsfIndices[i] = (sbyte)((i % 3) - 1);

        var enc = new OpusRangeEncoder(256);
        EncodeFullIndices(enc, indices, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 8, nbSubfr: 4,
            conditional: 1, prevLagIndex: 0, prevSignalTypeWasVoiced: false);
        enc.Done();

        var decoded = new SilkDecodedIndices();
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkIndicesDecoder.Decode(decoded, dec, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 8, nbSubfr: 4,
            conditional: 1, prevLagIndex: 0, prevSignalTypeWasVoiced: false);

        Equal(indices.SignalType, decoded.SignalType);
        Equal(indices.QuantOffsetType, decoded.QuantOffsetType);
        for (int i = 0; i < 4; i++) Equal(indices.GainsIndices[i], decoded.GainsIndices[i], $"gain[{i}]");
        for (int i = 0; i <= cb.Order; i++) Equal(indices.NlsfIndices[i], decoded.NlsfIndices[i], $"nlsf[{i}]");
        Equal(indices.NlsfInterpCoefQ2, decoded.NlsfInterpCoefQ2);
        Equal(indices.Seed, decoded.Seed);
    }

    [TestMethod]
    public void IndicesDecoder_Voiced_Vad_Independent_Wb_RoundTrip()
    {
        // Full voiced WB path: signal type 2 (voiced), pitch + LTP + LTP scale, all exercised.
        var cb = SilkNlsfCodebookTables.Wb;
        var indices = new SilkDecodedIndices
        {
            SignalType = 2,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4,
            LagIndex = 72,   // coarse 9 * (16/2=8) + lsb 0 = 72
            ContourIndex = 15,
            PerIndex = 1,
            LtpScaleIndex = 1,
            Seed = 3,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = (sbyte)(10 + i * 2);
        indices.NlsfIndices[0] = 20;
        for (int i = 1; i <= cb.Order; i++) indices.NlsfIndices[i] = 0;
        for (int i = 0; i < 4; i++) indices.LtpIndices[i] = (sbyte)(i * 3);

        var enc = new OpusRangeEncoder(256);
        EncodeFullIndices(enc, indices, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 16, nbSubfr: 4,
            conditional: 0, prevLagIndex: 0, prevSignalTypeWasVoiced: false);
        enc.Done();

        var decoded = new SilkDecodedIndices();
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkIndicesDecoder.Decode(decoded, dec, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 16, nbSubfr: 4,
            conditional: 0, prevLagIndex: 0, prevSignalTypeWasVoiced: false);

        Equal(indices.SignalType, decoded.SignalType);
        Equal(indices.QuantOffsetType, decoded.QuantOffsetType);
        for (int i = 0; i < 4; i++) Equal(indices.GainsIndices[i], decoded.GainsIndices[i]);
        for (int i = 0; i <= cb.Order; i++) Equal(indices.NlsfIndices[i], decoded.NlsfIndices[i]);
        Equal(indices.NlsfInterpCoefQ2, decoded.NlsfInterpCoefQ2);
        Equal(indices.LagIndex, decoded.LagIndex);
        Equal(indices.ContourIndex, decoded.ContourIndex);
        Equal(indices.PerIndex, decoded.PerIndex);
        for (int i = 0; i < 4; i++) Equal(indices.LtpIndices[i], decoded.LtpIndices[i], $"ltp[{i}]");
        Equal(indices.LtpScaleIndex, decoded.LtpScaleIndex);
        Equal(indices.Seed, decoded.Seed);
    }

    [TestMethod]
    public void IndicesDecoder_Voiced_DeltaPitchFromPrev_RoundTrip()
    {
        // Voiced frame + conditional + prevVoiced enables delta pitch coding.
        // Pick a pitch lag that's within [prev - 8, prev + 11] so the delta encoder uses raw > 0.
        var cb = SilkNlsfCodebookTables.NbMb;
        short prevLag = 60;
        short newLag = 63; // delta = +3, raw = 3 + 9 = 12 (valid)
        var indices = new SilkDecodedIndices
        {
            SignalType = 2,
            QuantOffsetType = 0,
            NlsfInterpCoefQ2 = 4,
            LagIndex = newLag,
            ContourIndex = 2,
            PerIndex = 2,
            LtpScaleIndex = 0, // forced to 0 when conditional != 0
            Seed = 1,
        };
        for (int i = 0; i < 4; i++) indices.GainsIndices[i] = (sbyte)(8 + i);
        indices.NlsfIndices[0] = 3;
        for (int i = 1; i <= cb.Order; i++) indices.NlsfIndices[i] = 0;
        for (int i = 0; i < 4; i++) indices.LtpIndices[i] = (sbyte)(i + 5);

        var enc = new OpusRangeEncoder(256);
        EncodeFullIndices(enc, indices, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 8, nbSubfr: 4,
            conditional: 1, prevLagIndex: prevLag, prevSignalTypeWasVoiced: true);
        enc.Done();

        var decoded = new SilkDecodedIndices();
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkIndicesDecoder.Decode(decoded, dec, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 8, nbSubfr: 4,
            conditional: 1, prevLagIndex: prevLag, prevSignalTypeWasVoiced: true);

        Equal(indices.SignalType, decoded.SignalType);
        Equal(indices.LagIndex, decoded.LagIndex);
        Equal(indices.ContourIndex, decoded.ContourIndex);
        Equal(indices.PerIndex, decoded.PerIndex);
        for (int i = 0; i < 4; i++) Equal(indices.LtpIndices[i], decoded.LtpIndices[i]);
        // LTP scale was NOT emitted by the encoder (conditional != 0), and decoder forces to 0.
        Equal((sbyte)0, decoded.LtpScaleIndex);
        Equal(indices.Seed, decoded.Seed);
    }

    [TestMethod]
    public void IndicesDecoder_Voiced_TenMs_UsesNarrowPitchContour()
    {
        // 10 ms frame -> nbSubfr = 2; NLSF interp coef is NOT read (forced to 4 by decoder).
        var cb = SilkNlsfCodebookTables.NbMb;
        var indices = new SilkDecodedIndices
        {
            SignalType = 2,
            QuantOffsetType = 1,
            NlsfInterpCoefQ2 = 4, // Must be 4 for 10 ms per libopus semantics
            LagIndex = 40,
            ContourIndex = 1,
            PerIndex = 0,
            LtpScaleIndex = 2,
            Seed = 0,
        };
        indices.GainsIndices[0] = 37;
        indices.GainsIndices[1] = 10;
        indices.NlsfIndices[0] = 5;
        indices.LtpIndices[0] = 3;
        indices.LtpIndices[1] = 7;

        var enc = new OpusRangeEncoder(256);
        EncodeFullIndices(enc, indices, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 8, nbSubfr: 2,
            conditional: 0, prevLagIndex: 0, prevSignalTypeWasVoiced: false);
        enc.Done();

        var decoded = new SilkDecodedIndices();
        var dec = new OpusRangeDecoder(enc.ToArray());
        SilkIndicesDecoder.Decode(decoded, dec, cb,
            vadFlag: true, decodeLbrr: false, fsKHz: 8, nbSubfr: 2,
            conditional: 0, prevLagIndex: 0, prevSignalTypeWasVoiced: false);

        Equal(indices.SignalType, decoded.SignalType);
        Equal((sbyte)4, decoded.NlsfInterpCoefQ2, "10 ms frame should force NLSF interp coef = 4");
        Equal(indices.LagIndex, decoded.LagIndex);
        Equal(indices.ContourIndex, decoded.ContourIndex);
        Equal(indices.PerIndex, decoded.PerIndex);
        Equal(indices.LtpIndices[0], decoded.LtpIndices[0]);
        Equal(indices.LtpIndices[1], decoded.LtpIndices[1]);
        Equal(indices.LtpScaleIndex, decoded.LtpScaleIndex);
        Equal(indices.Seed, decoded.Seed);
    }

    // -------- Arg validation --------

    [TestMethod]
    public void IndicesDecoder_NullIndices_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        Throws<ArgumentNullException>(() =>
            SilkIndicesDecoder.Decode(null!, dec, SilkNlsfCodebookTables.NbMb,
                false, false, 8, 4, 0, 0, false));
    }

    [TestMethod]
    public void IndicesDecoder_NullRangeDecoder_Throws()
    {
        var indices = new SilkDecodedIndices();
        Throws<ArgumentNullException>(() =>
            SilkIndicesDecoder.Decode(indices, null!, SilkNlsfCodebookTables.NbMb,
                false, false, 8, 4, 0, 0, false));
    }

    [TestMethod]
    public void IndicesDecoder_NullCodebook_Throws()
    {
        var indices = new SilkDecodedIndices();
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        Throws<ArgumentNullException>(() =>
            SilkIndicesDecoder.Decode(indices, dec, null!,
                false, false, 8, 4, 0, 0, false));
    }
}
