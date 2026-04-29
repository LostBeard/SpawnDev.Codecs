// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable composite primitive: half-band peak + floor headroom
// scale + MagnitudeToFloorY for both endpoints in one call. Mirrors
// VorbisAudioEncoder.EncodeAudioPacket's floor-fitting step
// (lines 235-249).
//
// Per-call output: 2 ints in posteriorsOut[0..1] (low + high). Both
// posteriors are clamped to >= 1 to guard against fully-silent blocks.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis encoder floor-fit composite primitive.
/// Single-thread dispatch (the half-band reduction is naturally
/// sequential per band).
/// </summary>
public static class VorbisEncoderFloorFitGpu
{
    /// <summary>
    /// Compute the two Floor 1 posterior endpoints for one block:
    ///   posteriorsOut[0] = MagnitudeToFloorY(peak(low half) * headroom)
    ///   posteriorsOut[1] = MagnitudeToFloorY(peak(high half) * headroom)
    /// Both posteriors are clamped to >= 1 (silent-block guard).
    /// </summary>
    public static void FitFloorEndpoints(
        ArrayView<float> spectrum, long spectrumBase, int halfBlock,
        float headroom,
        ArrayView<float> inverseDbTable, long inverseDbBase,
        ArrayView<int> posteriorsOut, long posteriorsBase)
    {
        // Peak reduction (mirror of VorbisSpectrumPeakGpu logic inlined
        // here so this primitive is a single GPU dispatch).
        int split = halfBlock >> 1;
        float lowPeak = 0f;
        float highPeak = 0f;
        for (int i = 0; i < split; i++)
        {
            float v = spectrum[spectrumBase + i];
            float a = v < 0 ? -v : v;
            if (a > lowPeak) lowPeak = a;
        }
        for (int i = split; i < halfBlock; i++)
        {
            float v = spectrum[spectrumBase + i];
            float a = v < 0 ? -v : v;
            if (a > highPeak) highPeak = a;
        }

        int yLow = VorbisEncoderHelpersGpu.MagnitudeToFloorY(
            lowPeak * headroom, inverseDbTable, inverseDbBase);
        int yHigh = VorbisEncoderHelpersGpu.MagnitudeToFloorY(
            highPeak * headroom, inverseDbTable, inverseDbBase);

        // Silent-block guard: clamp to >= 1 so floor curve isn't fully
        // silent (which would make residue undefined).
        if (yLow < 1) yLow = 1;
        if (yHigh < 1) yHigh = 1;

        posteriorsOut[posteriorsBase + 0] = yLow;
        posteriorsOut[posteriorsBase + 1] = yHigh;
    }
}
