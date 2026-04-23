using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Bitstream round-trip tests for <see cref="SilkLtpDecoder.DecodeIndices"/>:
/// decodes PERIndex + per-subframe LTP gain indices + optional LTP scale index
/// for a voiced SILK frame.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static void EncodeLtpIndices(
        OpusRangeEncoder enc,
        int perIdx,
        ReadOnlySpan<sbyte> ltpIndices,
        int conditional,
        int ltpScaleIdx)
    {
        enc.EncodeIcdf(perIdx, SilkIcdfTables.LtpPerIndex, 8);
        byte[] gainIcdf = SilkIcdfTables.SelectLtpGain(perIdx);
        for (int k = 0; k < ltpIndices.Length; k++)
        {
            enc.EncodeIcdf(ltpIndices[k], gainIcdf, 8);
        }
        if (conditional == 0)
        {
            enc.EncodeIcdf(ltpScaleIdx, SilkIcdfTables.LtpScale, 8);
        }
    }

    [TestMethod]
    public void LtpDecoder_CodebookZero_AllGainIndices_RoundTrip()
    {
        // Codebook 0 has 8 symbols. Exercise every index across 4 subframes.
        for (int gainIdx = 0; gainIdx < 8; gainIdx++)
        {
            sbyte[] ltp = { (sbyte)gainIdx, (sbyte)gainIdx, (sbyte)gainIdx, (sbyte)gainIdx };
            var enc = new OpusRangeEncoder(64);
            EncodeLtpIndices(enc, perIdx: 0, ltp, conditional: 0, ltpScaleIdx: 1);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            sbyte[] decoded = new sbyte[4];
            SilkLtpDecoder.DecodeIndices(
                decoded, dec, conditional: 0, nbSubfr: 4,
                out sbyte decPer, out sbyte decScale);

            Equal((sbyte)0, decPer, $"gainIdx={gainIdx}: PERIndex");
            for (int k = 0; k < 4; k++) Equal((sbyte)gainIdx, decoded[k], $"gainIdx={gainIdx}, subframe={k}");
            Equal((sbyte)1, decScale);
        }
    }

    [TestMethod]
    public void LtpDecoder_CodebookOne_MixedIndices_RoundTrip()
    {
        // Codebook 1 has 16 symbols. Mix of values per subframe.
        sbyte[] ltp = { 3, 10, 14, 7 };
        var enc = new OpusRangeEncoder(64);
        EncodeLtpIndices(enc, perIdx: 1, ltp, conditional: 0, ltpScaleIdx: 0);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[4];
        SilkLtpDecoder.DecodeIndices(
            decoded, dec, conditional: 0, nbSubfr: 4,
            out sbyte decPer, out sbyte decScale);

        Equal((sbyte)1, decPer);
        for (int k = 0; k < 4; k++) Equal(ltp[k], decoded[k], $"subframe={k}");
        Equal((sbyte)0, decScale);
    }

    [TestMethod]
    public void LtpDecoder_CodebookTwo_FullRange_RoundTrip()
    {
        // Codebook 2 has 32 symbols. Test the extremes and midpoints.
        sbyte[] ltp = { 0, 31, 15, 16 };
        var enc = new OpusRangeEncoder(64);
        EncodeLtpIndices(enc, perIdx: 2, ltp, conditional: 0, ltpScaleIdx: 2);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[4];
        SilkLtpDecoder.DecodeIndices(
            decoded, dec, conditional: 0, nbSubfr: 4,
            out sbyte decPer, out sbyte decScale);

        Equal((sbyte)2, decPer);
        for (int k = 0; k < 4; k++) Equal(ltp[k], decoded[k], $"subframe={k}");
        Equal((sbyte)2, decScale);
    }

    [TestMethod]
    public void LtpDecoder_ConditionalCoding_ScaleIndexForcedToZero()
    {
        // conditional != 0 -> LTP scale index is NOT read from the bitstream and should be 0.
        sbyte[] ltp = { 5, 3, 7, 1 };
        var enc = new OpusRangeEncoder(64);
        EncodeLtpIndices(enc, perIdx: 1, ltp, conditional: 1, ltpScaleIdx: -1 /* ignored */);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[4];
        SilkLtpDecoder.DecodeIndices(
            decoded, dec, conditional: 1, nbSubfr: 4,
            out sbyte decPer, out sbyte decScale);

        Equal((sbyte)1, decPer);
        for (int k = 0; k < 4; k++) Equal(ltp[k], decoded[k]);
        Equal((sbyte)0, decScale, "scale index should be 0 under conditional coding");
    }

    [TestMethod]
    public void LtpDecoder_TwoSubframes_RoundTrip()
    {
        sbyte[] ltp = { 6, 2 };
        var enc = new OpusRangeEncoder(64);
        EncodeLtpIndices(enc, perIdx: 0, ltp, conditional: 0, ltpScaleIdx: 2);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] decoded = new sbyte[2];
        SilkLtpDecoder.DecodeIndices(
            decoded, dec, conditional: 0, nbSubfr: 2,
            out sbyte decPer, out sbyte decScale);

        Equal((sbyte)0, decPer);
        Equal(ltp[0], decoded[0]);
        Equal(ltp[1], decoded[1]);
        Equal((sbyte)2, decScale);
    }

    [TestMethod]
    public void LtpDecoder_AllThreePerIndices_ThenScale_SequentialStream()
    {
        // Ensure the decoder reads the right iCDF for each per index and advances the
        // bitstream position correctly to read the scale at the end.
        for (int perIdx = 0; perIdx < 3; perIdx++)
        {
            sbyte[] ltp = { 1, 2, 3, 4 };
            int scaleIdx = perIdx; // use perIdx as the scale marker too so we can distinguish them.
            var enc = new OpusRangeEncoder(64);
            EncodeLtpIndices(enc, perIdx, ltp, conditional: 0, ltpScaleIdx: scaleIdx);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            sbyte[] decoded = new sbyte[4];
            SilkLtpDecoder.DecodeIndices(
                decoded, dec, conditional: 0, nbSubfr: 4,
                out sbyte decPer, out sbyte decScale);

            Equal((sbyte)perIdx, decPer);
            for (int k = 0; k < 4; k++) Equal(ltp[k], decoded[k]);
            Equal((sbyte)scaleIdx, decScale);
        }
    }

    // -------- Arg validation --------

    [TestMethod]
    public void LtpDecoder_NullRangeDecoder_Throws()
    {
        sbyte[] ltp = new sbyte[4];
        Throws<ArgumentNullException>(() =>
            SilkLtpDecoder.DecodeIndices(ltp, null!, 0, 4, out _, out _));
    }

    [TestMethod]
    public void LtpDecoder_InvalidNbSubfr_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] ltp = new sbyte[4];
        Throws<ArgumentException>(() =>
            SilkLtpDecoder.DecodeIndices(ltp, dec, 0, 3, out _, out _));
    }

    [TestMethod]
    public void LtpGainSelector_InvalidIndex_Throws()
    {
        Throws<ArgumentOutOfRangeException>(() => SilkIcdfTables.SelectLtpGain(3));
        Throws<ArgumentOutOfRangeException>(() => SilkIcdfTables.SelectLtpGain(-1));
    }

    [TestMethod]
    public void LtpGainSelector_ReturnsReferenceEquality()
    {
        if (!ReferenceEquals(SilkIcdfTables.LtpGain0, SilkIcdfTables.SelectLtpGain(0)))
            throw new Exception("SelectLtpGain(0) should return LtpGain0 ref");
        if (!ReferenceEquals(SilkIcdfTables.LtpGain1, SilkIcdfTables.SelectLtpGain(1)))
            throw new Exception("SelectLtpGain(1) should return LtpGain1 ref");
        if (!ReferenceEquals(SilkIcdfTables.LtpGain2, SilkIcdfTables.SelectLtpGain(2)))
            throw new Exception("SelectLtpGain(2) should return LtpGain2 ref");
    }
}
