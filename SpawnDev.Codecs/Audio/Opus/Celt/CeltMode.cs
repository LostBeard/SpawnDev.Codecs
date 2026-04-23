// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// C# equivalent of libopus celt/modes.c::OpusCustomMode (the subset used by
// the decoder). For Opus CELT the sample rate is always 48 kHz; the mode
// captures frame-size + effective bandwidth information.
//
// Upstream Copyright (c) 2007-2008 CSIRO, 2007-2009 Xiph.Org Foundation,
// 2008 Gregory Maxwell. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>
/// CELT mode description. Produced by <see cref="Create"/> based on the Opus
/// TOC's frame size + bandwidth. Matches the libopus <c>OpusCustomMode</c>
/// for the standard 48 kHz Opus configurations.
/// </summary>
internal sealed class CeltMode
{
    /// <summary>Sample rate in Hz (48000 for all Opus CELT modes).</summary>
    public int SampleRateHz { get; init; }

    /// <summary>Frame size in samples at <see cref="SampleRateHz"/>.</summary>
    public int FrameSize { get; init; }

    /// <summary>
    /// Total number of frequency bands. 21 at 48 kHz / 5 ms-scale; fewer for shorter
    /// effective bandwidths via <see cref="StartBand"/> / <see cref="EndBand"/>.
    /// </summary>
    public int NbEBands { get; init; }

    /// <summary>
    /// Inclusive-lower band boundary. Always 0 for standard Opus configurations.
    /// </summary>
    public int StartBand { get; init; }

    /// <summary>
    /// Exclusive-upper band boundary. Set by <see cref="Create"/> based on TOC
    /// bandwidth (NB=13, WB=17, SWB=19, FB=21 at 48 kHz).
    /// </summary>
    public int EndBand { get; init; }

    /// <summary>
    /// Per-band bin-boundary table (length = <see cref="NbEBands"/> + 1). Band <c>k</c>
    /// spans MDCT bins <c>[EBands[k], EBands[k+1])</c>.
    /// </summary>
    public required short[] EBands { get; init; }

    /// <summary>
    /// Construct a CELT mode for the given <paramref name="frameSizeSamples"/> at 48 kHz
    /// and the given effective band count (<see cref="EndBand"/>).
    /// </summary>
    /// <param name="frameSizeSamples">Frame size in samples (120, 240, 480, or 960).</param>
    /// <param name="endBand">Upper band limit (13 NB / 17 WB / 19 SWB / 21 FB).</param>
    public static CeltMode Create(int frameSizeSamples, int endBand)
    {
        if (frameSizeSamples is not (CeltConstants.FRAME_SIZE_2_5MS
            or CeltConstants.FRAME_SIZE_5MS
            or CeltConstants.FRAME_SIZE_10MS
            or CeltConstants.FRAME_SIZE_20MS))
        {
            throw new ArgumentException(
                $"frameSizeSamples must be 120/240/480/960, got {frameSizeSamples}.",
                nameof(frameSizeSamples));
        }
        if (endBand < 1 || endBand > CeltConstants.NB_BANDS_FULLBAND)
        {
            throw new ArgumentException(
                $"endBand must be in [1, {CeltConstants.NB_BANDS_FULLBAND}], got {endBand}.",
                nameof(endBand));
        }

        return new CeltMode
        {
            SampleRateHz = CeltConstants.MAX_SAMPLE_RATE_HZ,
            FrameSize = frameSizeSamples,
            NbEBands = CeltConstants.NB_BANDS_FULLBAND,
            StartBand = 0,
            EndBand = endBand,
            EBands = CeltConstants.Eband5Ms,
        };
    }

    /// <summary>
    /// Map an Opus TOC bandwidth to the corresponding CELT <see cref="EndBand"/> value.
    /// </summary>
    public static int EndBandForBandwidth(OpusBandwidth bandwidth) => bandwidth switch
    {
        OpusBandwidth.Narrowband => CeltConstants.NB_BANDS_NB,
        OpusBandwidth.Wideband => CeltConstants.NB_BANDS_WB,
        OpusBandwidth.Superwideband => CeltConstants.NB_BANDS_SWB,
        OpusBandwidth.Fullband => CeltConstants.NB_BANDS_FULLBAND,
        _ => throw new ArgumentException(
            $"CELT does not support Mediumband; got {bandwidth}.", nameof(bandwidth)),
    };
}
