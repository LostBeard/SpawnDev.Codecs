using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Bitstream round-trip tests for <see cref="SilkPitchDecoder.DecodeIndices"/>
/// covering all three paths (absolute lag for each of the three internal SILK
/// sample rates, delta-coded lag, and contour selection for NB/non-NB x 10/20 ms
/// frames).
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Encode an absolute pitch-lag + contour for test purposes, mirroring the
    /// absolute-coding branch of libopus silk_encode_indices.
    /// </summary>
    private static void EncodeAbsolutePitchIndices(
        OpusRangeEncoder enc,
        int fsKHz,
        int nbSubfr,
        int coarseIdx,
        int lsbIdx,
        int contourIdx)
    {
        enc.EncodeIcdf(coarseIdx, SilkIcdfTables.PitchLag, 8);
        enc.EncodeIcdf(lsbIdx, SilkIcdfTables.SelectPitchLagLowBits(fsKHz), 8);
        enc.EncodeIcdf(contourIdx, SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr), 8);
    }

    /// <summary>
    /// Encode a delta-coded pitch lag + contour. <paramref name="deltaRaw"/> is the
    /// raw symbol (1..20); the decoded delta is deltaRaw - 9.
    /// </summary>
    private static void EncodeDeltaPitchIndices(
        OpusRangeEncoder enc,
        int fsKHz,
        int nbSubfr,
        int deltaRaw,
        int contourIdx)
    {
        enc.EncodeIcdf(deltaRaw, SilkIcdfTables.PitchDelta, 8);
        enc.EncodeIcdf(contourIdx, SilkIcdfTables.SelectPitchContour(fsKHz, nbSubfr), 8);
    }

    // -------- Absolute pitch coding --------

    [TestMethod]
    public void PitchDecoder_AbsoluteNb20Ms_RoundTrips()
    {
        // fs_kHz = 8 (NB), 20 ms -> 4 subframes, contour iCDF = PitchContourNb (11 symbols).
        int coarse = 17, lsb = 3, contour = 5;
        var enc = new OpusRangeEncoder(64);
        EncodeAbsolutePitchIndices(enc, 8, 4, coarse, lsb, contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 8, nbSubfr: 4,
            prevLagIndex: 0, prevSignalTypeWasVoiced: false, conditional: 0);

        // Absolute decoded lag = coarse * (fsKHz/2) + lsb = 17*4 + 3 = 71.
        Equal((short)(coarse * 4 + lsb), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    [TestMethod]
    public void PitchDecoder_AbsoluteMb20Ms_RoundTrips()
    {
        // fs_kHz = 12 (MB), uses Uniform6 for LSB. Contour iCDF = PitchContour (34 symbols).
        int coarse = 10, lsb = 4, contour = 22;
        var enc = new OpusRangeEncoder(64);
        EncodeAbsolutePitchIndices(enc, 12, 4, coarse, lsb, contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 12, nbSubfr: 4,
            prevLagIndex: 100, prevSignalTypeWasVoiced: false, conditional: 0);

        // coarse * 6 + lsb = 60 + 4 = 64.
        Equal((short)(coarse * 6 + lsb), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    [TestMethod]
    public void PitchDecoder_AbsoluteWb20Ms_RoundTrips()
    {
        // fs_kHz = 16 (WB), Uniform8 for LSB, PitchContour (non-NB 20 ms).
        int coarse = 25, lsb = 6, contour = 15;
        var enc = new OpusRangeEncoder(64);
        EncodeAbsolutePitchIndices(enc, 16, 4, coarse, lsb, contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 16, nbSubfr: 4,
            prevLagIndex: 0, prevSignalTypeWasVoiced: false, conditional: 0);

        Equal((short)(coarse * 8 + lsb), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    // -------- Delta pitch coding --------

    [TestMethod]
    public void PitchDecoder_DeltaCoding_RawSymbolOne_ProducesNegativeEightDelta()
    {
        // Raw delta = 1 -> decoded delta = 1 - 9 = -8.
        int contour = 10;
        short prevLag = 80;
        var enc = new OpusRangeEncoder(64);
        EncodeDeltaPitchIndices(enc, 16, 4, deltaRaw: 1, contourIdx: contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 16, nbSubfr: 4,
            prevLagIndex: prevLag, prevSignalTypeWasVoiced: true, conditional: 1);

        Equal((short)(prevLag - 8), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    [TestMethod]
    public void PitchDecoder_DeltaCoding_RawSymbolAllNonZero_RoundTrips()
    {
        // All 20 non-zero raw delta symbols produce unique results via prevLag + (raw - 9).
        int contour = 3;
        short prevLag = 100;
        for (int raw = 1; raw <= 20; raw++)
        {
            var enc = new OpusRangeEncoder(64);
            EncodeDeltaPitchIndices(enc, 8, 4, deltaRaw: raw, contourIdx: contour);
            enc.Done();

            var dec = new OpusRangeDecoder(enc.ToArray());
            var res = SilkPitchDecoder.DecodeIndices(
                dec, fsKHz: 8, nbSubfr: 4,
                prevLagIndex: prevLag, prevSignalTypeWasVoiced: true, conditional: 1);

            Equal((short)(prevLag + raw - 9), res.LagIndex, $"raw={raw}");
            Equal((sbyte)contour, res.ContourIndex, $"raw={raw}");
        }
    }

    [TestMethod]
    public void PitchDecoder_DeltaCoding_RawZero_FallsBackToAbsolute()
    {
        // Raw delta == 0 signals "switch to absolute coding".
        int coarse = 12, lsb = 2, contour = 8;
        var enc = new OpusRangeEncoder(64);
        enc.EncodeIcdf(0, SilkIcdfTables.PitchDelta, 8);           // raw delta 0 = escape
        EncodeAbsolutePitchIndices(enc, 16, 4, coarse, lsb, contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 16, nbSubfr: 4,
            prevLagIndex: 50, prevSignalTypeWasVoiced: true, conditional: 1);

        Equal((short)(coarse * 8 + lsb), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    // -------- 10 ms frame variants --------

    [TestMethod]
    public void PitchDecoder_AbsoluteNb10Ms_UsesNarrowContourTable()
    {
        // fs_kHz=8, nb_subfr=2 -> PitchContour10MsNb (3 symbols).
        int coarse = 5, lsb = 1, contour = 2;
        var enc = new OpusRangeEncoder(64);
        EncodeAbsolutePitchIndices(enc, 8, 2, coarse, lsb, contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 8, nbSubfr: 2,
            prevLagIndex: 0, prevSignalTypeWasVoiced: false, conditional: 0);

        Equal((short)(coarse * 4 + lsb), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    [TestMethod]
    public void PitchDecoder_AbsoluteWb10Ms_UsesWideContourTable()
    {
        int coarse = 20, lsb = 7, contour = 11;
        var enc = new OpusRangeEncoder(64);
        EncodeAbsolutePitchIndices(enc, 16, 2, coarse, lsb, contour);
        enc.Done();

        var dec = new OpusRangeDecoder(enc.ToArray());
        var res = SilkPitchDecoder.DecodeIndices(
            dec, fsKHz: 16, nbSubfr: 2,
            prevLagIndex: 0, prevSignalTypeWasVoiced: false, conditional: 0);

        Equal((short)(coarse * 8 + lsb), res.LagIndex);
        Equal((sbyte)contour, res.ContourIndex);
    }

    // -------- Edge / argument cases --------

    [TestMethod]
    public void PitchDecoder_NullRangeDecoder_Throws()
    {
        Throws<ArgumentNullException>(() =>
            SilkPitchDecoder.DecodeIndices(null!, 16, 4, 0, false, 0));
    }

    [TestMethod]
    public void PitchDecoder_UnsupportedFsKHz_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        Throws<ArgumentException>(() =>
            SilkPitchDecoder.DecodeIndices(dec, fsKHz: 24, nbSubfr: 4,
                prevLagIndex: 0, prevSignalTypeWasVoiced: false, conditional: 0));
    }

    [TestMethod]
    public void PitchDecoder_InvalidNbSubfr_Throws()
    {
        var enc = new OpusRangeEncoder(32);
        enc.Done();
        var dec = new OpusRangeDecoder(enc.ToArray());
        Throws<ArgumentException>(() =>
            SilkPitchDecoder.DecodeIndices(dec, fsKHz: 16, nbSubfr: 3,
                prevLagIndex: 0, prevSignalTypeWasVoiced: false, conditional: 0));
    }

    // -------- Table selector sanity --------

    [TestMethod]
    public void PitchTables_SelectorsMatchLibopus()
    {
        Same(SilkIcdfTables.PitchContourNb, SilkIcdfTables.SelectPitchContour(8, 4), "NB 20ms");
        Same(SilkIcdfTables.PitchContour10MsNb, SilkIcdfTables.SelectPitchContour(8, 2), "NB 10ms");
        Same(SilkIcdfTables.PitchContour, SilkIcdfTables.SelectPitchContour(16, 4), "WB 20ms");
        Same(SilkIcdfTables.PitchContour10Ms, SilkIcdfTables.SelectPitchContour(12, 2), "MB 10ms");

        Same(SilkIcdfTables.Uniform4, SilkIcdfTables.SelectPitchLagLowBits(8), "NB low bits");
        Same(SilkIcdfTables.Uniform6, SilkIcdfTables.SelectPitchLagLowBits(12), "MB low bits");
        Same(SilkIcdfTables.Uniform8, SilkIcdfTables.SelectPitchLagLowBits(16), "WB low bits");
        Throws<ArgumentException>(() => SilkIcdfTables.SelectPitchLagLowBits(24));
    }

    private static void Same(byte[] expected, byte[] actual, string label)
    {
        if (!ReferenceEquals(expected, actual))
            throw new Exception($"{label}: expected reference-equal array");
    }
}
