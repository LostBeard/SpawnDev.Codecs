using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="SilkSideInfoDecoder"/> - the small scalar side-information
/// decoders (signal type + quantizer offset, PRNG seed). Covers every possible
/// decoded value to confirm the iCDF table maps correctly to each output.
/// </summary>
public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void SideInfo_SignalType_NoVadTable_AllRawValues_RoundTrip()
    {
        // No-VAD iCDF has 2 symbols: Ix in {0, 1}. signalType = 0 (inactive), offset in {0, 1}.
        for (int raw = 0; raw < 2; raw++)
        {
            var enc = new OpusRangeEncoder(64);
            enc.EncodeIcdf(raw, SilkIcdfTables.TypeOffsetNoVad, 8);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            var result = SilkSideInfoDecoder.DecodeSignalType(dec, useVadTable: false);
            Equal((sbyte)(raw >> 1), result.SignalType, $"raw={raw}: signalType");
            Equal((sbyte)(raw & 1), result.QuantOffsetType, $"raw={raw}: quantOffset");
            Equal(SilkSideInfoDecoder.TypeInactive, result.SignalType, $"raw={raw}: should be inactive");
        }
    }

    [TestMethod]
    public void SideInfo_SignalType_VadTable_AllRawValues_RoundTrip()
    {
        // VAD iCDF has 4 symbols: Ix in {0..3}, +2 -> effective in {2..5}.
        // signalType = Ix >> 1 ∈ {1, 2} (unvoiced or voiced), offset ∈ {0, 1}.
        for (int raw = 0; raw < 4; raw++)
        {
            var enc = new OpusRangeEncoder(64);
            enc.EncodeIcdf(raw, SilkIcdfTables.TypeOffsetVad, 8);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            var result = SilkSideInfoDecoder.DecodeSignalType(dec, useVadTable: true);

            int effective = raw + 2;
            Equal((sbyte)(effective >> 1), result.SignalType, $"raw={raw}: signalType");
            Equal((sbyte)(effective & 1), result.QuantOffsetType, $"raw={raw}: quantOffset");

            // First two raw symbols (0, 1) -> signalType 1 (unvoiced). Last two (2, 3) -> voiced.
            if (raw < 2)
                Equal(SilkSideInfoDecoder.TypeUnvoiced, result.SignalType, $"raw={raw}: should be unvoiced");
            else
                Equal(SilkSideInfoDecoder.TypeVoiced, result.SignalType, $"raw={raw}: should be voiced");
        }
    }

    [TestMethod]
    public void SideInfo_Seed_AllValues_RoundTrip()
    {
        // Uniform4 -> seed in {0, 1, 2, 3}.
        for (int s = 0; s < 4; s++)
        {
            var enc = new OpusRangeEncoder(64);
            enc.EncodeIcdf(s, SilkIcdfTables.Uniform4, 8);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            sbyte decoded = SilkSideInfoDecoder.DecodeSeed(dec);
            Equal((sbyte)s, decoded, $"seed={s}");
        }
    }

    [TestMethod]
    public void SideInfo_SignalType_NullRangeDecoder_Throws()
    {
        Throws<ArgumentNullException>(() =>
            SilkSideInfoDecoder.DecodeSignalType(null!, useVadTable: true));
    }

    [TestMethod]
    public void SideInfo_Seed_NullRangeDecoder_Throws()
    {
        Throws<ArgumentNullException>(() => SilkSideInfoDecoder.DecodeSeed(null!));
    }

    [TestMethod]
    public void SideInfo_SequentialDecode_MatchesSequentialEncode()
    {
        // End-to-end: encode signal type + seed in sequence, verify both round-trip in order.
        var enc = new OpusRangeEncoder(64);
        // Signal type (VAD): raw=3 -> signalType=2 (voiced), offset=1.
        enc.EncodeIcdf(3, SilkIcdfTables.TypeOffsetVad, 8);
        // Seed=2.
        enc.EncodeIcdf(2, SilkIcdfTables.Uniform4, 8);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var sigType = SilkSideInfoDecoder.DecodeSignalType(dec, useVadTable: true);
        sbyte seed = SilkSideInfoDecoder.DecodeSeed(dec);

        Equal(SilkSideInfoDecoder.TypeVoiced, sigType.SignalType);
        Equal((sbyte)1, sigType.QuantOffsetType);
        Equal((sbyte)2, seed);
    }
}
