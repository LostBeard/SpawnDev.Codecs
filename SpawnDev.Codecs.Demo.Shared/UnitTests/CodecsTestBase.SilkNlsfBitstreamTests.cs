using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Bitstream round-trip tests for <see cref="SilkNlsfDecoder.DecodeIndices"/>.
/// Encodes known NLSF indices using the same iCDF tables the decoder reads, then
/// verifies the decoder recovers every value exactly. Also confirms the decoded
/// indices feed cleanly into the existing NLSF decode + NLSF2A pipeline.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Encode the NLSF-index block for test purposes, mirroring the libopus encoder
    /// side of <c>silk_decode_indices</c>.
    /// </summary>
    /// <param name="interpCoefQ2">Q2 interpolation coefficient (ignored when nbSubfr != 4).</param>
    private static void EncodeNlsfIndices(
        OpusRangeEncoder enc,
        ReadOnlySpan<sbyte> nlsfIndices,
        SilkNlsfCodebook cb,
        int signalType,
        int nbSubfr,
        int interpCoefQ2)
    {
        int order = cb.Order;
        int cb1IcdfStart = (signalType >> 1) * cb.NVectors;
        int cb1Index = nlsfIndices[0];
        enc.EncodeIcdf(cb1Index, cb.Cb1Icdf.AsSpan(cb1IcdfStart, cb.NVectors), 8);

        Span<short> ecIx = stackalloc short[SilkConstants.MAX_LPC_ORDER];
        Span<byte> predQ8 = stackalloc byte[SilkConstants.MAX_LPC_ORDER];
        SilkNlsfUnpack.Unpack(ecIx, predQ8, cb, cb1Index);

        int railTop = 2 * SilkConstants.NLSF_QUANT_MAX_AMPLITUDE;
        for (int i = 0; i < order; i++)
        {
            int signedIdx = nlsfIndices[i + 1]; // in [-amp - 6, amp + 6]
            int ix = signedIdx + SilkConstants.NLSF_QUANT_MAX_AMPLITUDE; // Shift to [amp-6, 3*amp+6]
            int core;
            int ext = 0;
            bool useExt = false;
            if (ix <= 0)
            {
                // Map to core=0, ext = -ix
                core = 0;
                ext = -ix; // in [0, 6]
                useExt = true;
            }
            else if (ix >= railTop)
            {
                core = railTop;
                ext = ix - railTop; // in [0, 6]
                useExt = true;
            }
            else
            {
                core = ix;
            }

            enc.EncodeIcdf(core, cb.EcIcdf.AsSpan(ecIx[i], 9), 8);
            if (useExt)
            {
                enc.EncodeIcdf(ext, SilkIcdfTables.NlsfExt, 8);
            }
        }

        if (nbSubfr == SilkConstants.MAX_NB_SUBFR)
        {
            enc.EncodeIcdf(interpCoefQ2, SilkIcdfTables.NlsfInterpolationFactor, 8);
        }
    }

    [TestMethod]
    public void NlsfDecodeIndices_NbMb_ZeroResiduals_AllSignalTypes_RoundTrip()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        int order = cb.Order;
        sbyte[] orig = new sbyte[order + 1];
        sbyte[] decoded = new sbyte[order + 1];

        for (int signalType = 0; signalType < 3; signalType++)
        {
            for (int cb1 = 0; cb1 < cb.NVectors; cb1++)
            {
                orig[0] = (sbyte)cb1;
                for (int i = 1; i <= order; i++) orig[i] = 0;

                var enc = new OpusRangeEncoder(128);
                EncodeNlsfIndices(enc, orig, cb, signalType, nbSubfr: 4, interpCoefQ2: 2);
                enc.Done();

                var dec = new OpusRangeDecoder(enc.ToArray());
                int interpCoef = SilkNlsfDecoder.DecodeIndices(decoded, dec, cb, signalType, nbSubfr: 4);

                for (int i = 0; i <= order; i++)
                {
                    Equal(orig[i], decoded[i], $"signalType={signalType}, cb1={cb1}, idx={i}");
                }
                Equal(2, interpCoef, $"signalType={signalType}, cb1={cb1}: interpCoef");
            }
        }
    }

    [TestMethod]
    public void NlsfDecodeIndices_Wb_ZeroResiduals_AllSignalTypes_RoundTrip()
    {
        var cb = SilkNlsfCodebookTables.Wb;
        int order = cb.Order;
        sbyte[] orig = new sbyte[order + 1];
        sbyte[] decoded = new sbyte[order + 1];

        for (int signalType = 0; signalType < 3; signalType++)
        {
            for (int cb1 = 0; cb1 < cb.NVectors; cb1++)
            {
                orig[0] = (sbyte)cb1;
                for (int i = 1; i <= order; i++) orig[i] = 0;

                var enc = new OpusRangeEncoder(128);
                EncodeNlsfIndices(enc, orig, cb, signalType, nbSubfr: 4, interpCoefQ2: 3);
                enc.Done();

                var dec = new OpusRangeDecoder(enc.ToArray());
                int interpCoef = SilkNlsfDecoder.DecodeIndices(decoded, dec, cb, signalType, nbSubfr: 4);

                for (int i = 0; i <= order; i++)
                {
                    Equal(orig[i], decoded[i], $"signalType={signalType}, cb1={cb1}, idx={i}");
                }
                Equal(3, interpCoef);
            }
        }
    }

    [TestMethod]
    public void NlsfDecodeIndices_NonZeroResiduals_WithinAmplitude_RoundTrip()
    {
        var cb = SilkNlsfCodebookTables.NbMb;
        int order = cb.Order;
        sbyte[] orig = new sbyte[order + 1];
        orig[0] = 5;
        // Values in the "core" range [-amp+1, +amp-1] that do not trip the rail extension.
        sbyte[] pattern = { -3, -2, -1, 0, 1, 2, 3, 0, -1, 2 };
        for (int i = 0; i < order; i++) orig[i + 1] = pattern[i];

        var enc = new OpusRangeEncoder(128);
        EncodeNlsfIndices(enc, orig, cb, signalType: 1, nbSubfr: 2, interpCoefQ2: 4);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[order + 1];
        int interpCoef = SilkNlsfDecoder.DecodeIndices(decoded, dec, cb, signalType: 1, nbSubfr: 2);

        for (int i = 0; i <= order; i++)
        {
            Equal(orig[i], decoded[i], $"idx={i}");
        }
        // nbSubfr != 4 -> interp coef is hard-coded to 4 and NOT read from bitstream.
        Equal(4, interpCoef);
    }

    [TestMethod]
    public void NlsfDecodeIndices_RailExtensionNegative_RoundTrip()
    {
        // Indices at -amp (triggers low rail) and beyond -amp (triggers NlsfExt decode).
        // With NLSF_QUANT_MAX_AMPLITUDE = 4, ix==0 case maps signedIdx = -amp - ext.
        // So signedIdx values -4, -5, ..., -10 all exercise the rail path (ext = 0..6).
        var cb = SilkNlsfCodebookTables.NbMb;
        int order = cb.Order;
        sbyte[] orig = new sbyte[order + 1];
        orig[0] = 10;
        sbyte[] pattern = { -4, -5, -6, -7, -8, -9, -10, -4, -4, -4 };
        for (int i = 0; i < order; i++) orig[i + 1] = pattern[i];

        var enc = new OpusRangeEncoder(128);
        EncodeNlsfIndices(enc, orig, cb, signalType: 2, nbSubfr: 4, interpCoefQ2: 1);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[order + 1];
        int interpCoef = SilkNlsfDecoder.DecodeIndices(decoded, dec, cb, signalType: 2, nbSubfr: 4);

        for (int i = 0; i <= order; i++)
        {
            Equal(orig[i], decoded[i], $"idx={i}");
        }
        Equal(1, interpCoef);
    }

    [TestMethod]
    public void NlsfDecodeIndices_RailExtensionPositive_RoundTrip()
    {
        // signedIdx at +amp (core = railTop), beyond +amp (ext kicks in). Values 4..10.
        var cb = SilkNlsfCodebookTables.NbMb;
        int order = cb.Order;
        sbyte[] orig = new sbyte[order + 1];
        orig[0] = 20;
        sbyte[] pattern = { 4, 5, 6, 7, 8, 9, 10, 4, 4, 4 };
        for (int i = 0; i < order; i++) orig[i + 1] = pattern[i];

        var enc = new OpusRangeEncoder(128);
        EncodeNlsfIndices(enc, orig, cb, signalType: 2, nbSubfr: 4, interpCoefQ2: 0);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[order + 1];
        int interpCoef = SilkNlsfDecoder.DecodeIndices(decoded, dec, cb, signalType: 2, nbSubfr: 4);

        for (int i = 0; i <= order; i++)
        {
            Equal(orig[i], decoded[i], $"idx={i}");
        }
        Equal(0, interpCoef);
    }

    [TestMethod]
    public void NlsfDecodeIndices_ThenFullDecode_ProducesStableLpc()
    {
        // End-to-end: bitstream -> DecodeIndices -> Decode -> Nlsf2A -> stable LPC.
        var cb = SilkNlsfCodebookTables.Wb;
        int order = cb.Order;
        sbyte[] orig = new sbyte[order + 1];
        orig[0] = 7;
        for (int i = 0; i < order; i++) orig[i + 1] = (sbyte)(((i % 3) - 1));

        var enc = new OpusRangeEncoder(128);
        EncodeNlsfIndices(enc, orig, cb, signalType: 2, nbSubfr: 4, interpCoefQ2: 4);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Span<sbyte> decodedIdx = stackalloc sbyte[order + 1];
        int interpCoef = SilkNlsfDecoder.DecodeIndices(decodedIdx, dec, cb, signalType: 2, nbSubfr: 4);

        Span<short> nlsfQ15 = stackalloc short[order];
        SilkNlsfDecoder.Decode(nlsfQ15, decodedIdx, cb);

        Span<short> aQ12 = stackalloc short[order];
        SilkNlsf2A.Compute(aQ12, nlsfQ15, order);

        int invGain = SilkLpcInvPredGain.Compute(aQ12, order);
        True(invGain > 0, $"End-to-end WB NLSF from bitstream should produce a stable LPC; interpCoef={interpCoef}");
    }

    [TestMethod]
    public void NlsfDecodeIndices_InvalidSignalType_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] indices = new sbyte[17];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkNlsfDecoder.DecodeIndices(indices, dec, SilkNlsfCodebookTables.Wb, signalType: 3, nbSubfr: 4));
    }

    [TestMethod]
    public void NlsfDecodeIndices_NullCodebook_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] indices = new sbyte[11];
        Throws<ArgumentNullException>(() =>
            SilkNlsfDecoder.DecodeIndices(indices, dec, null!, signalType: 0, nbSubfr: 4));
    }

    [TestMethod]
    public void NlsfDecodeIndices_NullRangeDecoder_Throws()
    {
        sbyte[] indices = new sbyte[11];
        Throws<ArgumentNullException>(() =>
            SilkNlsfDecoder.DecodeIndices(indices, null!, SilkNlsfCodebookTables.NbMb, signalType: 0, nbSubfr: 4));
    }
}
