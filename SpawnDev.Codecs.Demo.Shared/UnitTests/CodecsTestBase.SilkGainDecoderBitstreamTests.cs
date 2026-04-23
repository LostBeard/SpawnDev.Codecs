using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Bitstream round-trip tests for <see cref="SilkGainDecoder.DecodeIndices"/>.
/// Encodes a known set of gain indices using <see cref="OpusRangeEncoder"/> with the
/// same iCDF tables libopus specifies in its gain-decode block, then decodes them
/// back and verifies the decoded indices match the originals exactly.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Encodes gain indices for a SILK frame, mirroring the code path in libopus
    /// <c>silk_encode_indices</c>. For test purposes only.
    /// </summary>
    private static void EncodeGainIndices(
        OpusRangeEncoder enc,
        ReadOnlySpan<sbyte> indices,
        int signalType,
        int conditional,
        int nbSubfr)
    {
        if (conditional != 0)
        {
            enc.EncodeIcdf(indices[0], SilkIcdfTables.DeltaGain, 8);
        }
        else
        {
            int first = indices[0];
            int msb = first >> 3;
            int lsb = first & 7;
            int gainIcdfStart = SilkIcdfTables.GainIcdfOffset(signalType);
            enc.EncodeIcdf(msb, SilkIcdfTables.Gain.AsSpan(gainIcdfStart, SilkIcdfTables.GainIcdfEntriesPerType), 8);
            enc.EncodeIcdf(lsb, SilkIcdfTables.Uniform8, 8);
        }
        for (int i = 1; i < nbSubfr; i++)
        {
            enc.EncodeIcdf(indices[i], SilkIcdfTables.DeltaGain, 8);
        }
    }

    [TestMethod]
    public void GainDecoder_IndependentCoding_RoundTripsAllSignalTypes()
    {
        // Independent (conditional=0) coding reads a 3-bit MSB + 3-bit LSB for the first subframe.
        // So indices[0] is in [0, 63]. Remaining subframes use 41-symbol delta iCDF so each index is in [0, 40].
        sbyte[] original = { 37, 20, 15, 8 }; // 4 subframes
        sbyte[] decoded = new sbyte[4];
        for (int signalType = 0; signalType < SilkIcdfTables.GainIcdfNumTypes; signalType++)
        {
            var enc = new OpusRangeEncoder(128);
            EncodeGainIndices(enc, original, signalType, conditional: 0, nbSubfr: 4);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            SilkGainDecoder.DecodeIndices(decoded, dec, signalType, conditional: 0, nbSubfr: 4);

            for (int i = 0; i < 4; i++)
            {
                Equal(original[i], decoded[i], $"signalType={signalType}, subframe={i}");
            }
        }
    }

    [TestMethod]
    public void GainDecoder_ConditionalCoding_RoundTrips()
    {
        // Conditional coding: all subframes use the 41-symbol delta iCDF (indices in [0, 40]).
        sbyte[] original = { 14, 28, 7, 33 };
        var enc = new OpusRangeEncoder(128);
        EncodeGainIndices(enc, original, signalType: 1, conditional: 1, nbSubfr: 4);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Span<sbyte> decoded = stackalloc sbyte[4];
        SilkGainDecoder.DecodeIndices(decoded, dec, signalType: 1, conditional: 1, nbSubfr: 4);

        for (int i = 0; i < 4; i++)
        {
            Equal(original[i], decoded[i], $"subframe={i}");
        }
    }

    [TestMethod]
    public void GainDecoder_TwoSubframes_RoundTrips()
    {
        // 10 ms frame uses 2 subframes.
        sbyte[] original = { 25, 5 };
        var enc = new OpusRangeEncoder(128);
        EncodeGainIndices(enc, original, signalType: 2, conditional: 0, nbSubfr: 2);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Span<sbyte> decoded = stackalloc sbyte[2];
        SilkGainDecoder.DecodeIndices(decoded, dec, signalType: 2, conditional: 0, nbSubfr: 2);

        Equal(original[0], decoded[0]);
        Equal(original[1], decoded[1]);
    }

    [TestMethod]
    public void GainDecoder_Dequantize_AfterDecodeIndices_ProducesNonZeroGains()
    {
        // End-to-end: decode indices from a bitstream, then feed into Dequantize to get Q16 gains.
        sbyte[] original = { 40, 15, 10, 20 };
        var enc = new OpusRangeEncoder(128);
        EncodeGainIndices(enc, original, signalType: 2, conditional: 0, nbSubfr: 4);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        Span<sbyte> decoded = stackalloc sbyte[4];
        SilkGainDecoder.DecodeIndices(decoded, dec, signalType: 2, conditional: 0, nbSubfr: 4);

        Span<int> gainsQ16 = stackalloc int[4];
        sbyte prev = 0;
        SilkGainDecoder.Dequantize(gainsQ16, decoded, ref prev, conditional: 0, nbSubfr: 4);

        for (int i = 0; i < 4; i++)
        {
            True(gainsQ16[i] > 0, $"gainQ16[{i}] = {gainsQ16[i]} should be > 0");
        }
    }

    [TestMethod]
    public void GainDecoder_InvalidSignalType_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] indices = new sbyte[4];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkGainDecoder.DecodeIndices(indices, dec, signalType: 3, conditional: 0, nbSubfr: 4));
        Throws<ArgumentOutOfRangeException>(() =>
            SilkGainDecoder.DecodeIndices(indices, dec, signalType: -1, conditional: 0, nbSubfr: 4));
    }

    [TestMethod]
    public void GainDecoder_InvalidSubframeCount_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        sbyte[] indices = new sbyte[4];
        Throws<ArgumentOutOfRangeException>(() =>
            SilkGainDecoder.DecodeIndices(indices, dec, signalType: 0, conditional: 0, nbSubfr: 0));
        Throws<ArgumentOutOfRangeException>(() =>
            SilkGainDecoder.DecodeIndices(indices, dec, signalType: 0, conditional: 0, nbSubfr: 5));
    }

    [TestMethod]
    public void GainDecoder_NullRangeDecoder_Throws()
    {
        sbyte[] indices = new sbyte[4];
        Throws<ArgumentNullException>(() =>
            SilkGainDecoder.DecodeIndices(indices, null!, signalType: 0, conditional: 0, nbSubfr: 4));
    }
}
