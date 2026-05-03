// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Unified estimator + emitter for FLAC subframes. Tries CONSTANT, FIXED (via
// FlacFixedSubframeEncoder), and VERBATIM in order; picks the type with the
// smallest encoded bit count. Used both for direct emission and for cost
// comparison during stereo mode selection.

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacSubframeWriter
{
    /// <summary>
    /// Estimate the encoded bit count of the cheapest subframe type for
    /// <paramref name="samples"/> at <paramref name="bps"/> bits per sample.
    /// Does not write any output.
    /// </summary>
    internal static long EstimateBits(ReadOnlySpan<int> samples, int bps)
    {
        if (samples.Length == 0) return 8;
        // CONSTANT fast path.
        bool allEqual = true;
        int first = samples[0];
        for (int i = 1; i < samples.Length; i++)
        {
            if (samples[i] != first) { allEqual = false; break; }
        }
        if (allEqual) return 8 + bps;

        long verbatimBits = 8L + (long)samples.Length * bps;
        long best = verbatimBits;

        var fixedChoice = FlacFixedSubframeEncoder.PickBest(samples, bps);
        if (fixedChoice is not null && fixedChoice.TotalSubframeBits < best)
            best = fixedChoice.TotalSubframeBits;

        var lpcChoice = FlacLpcSubframeEncoder.PickBest(samples, bps, best);
        if (lpcChoice is not null && lpcChoice.TotalSubframeBits < best)
            best = lpcChoice.TotalSubframeBits;

        return best;
    }

    /// <summary>
    /// Emit the cheapest subframe type into <paramref name="w"/>.
    /// </summary>
    internal static void Emit(FlacBitWriter w, ReadOnlySpan<int> samples, int bps)
    {
        if (samples.Length == 0) return;

        bool allEqual = true;
        int first = samples[0];
        for (int i = 1; i < samples.Length; i++)
        {
            if (samples[i] != first) { allEqual = false; break; }
        }
        if (allEqual)
        {
            w.Write(0, 1);
            w.Write(0b000000, 6);
            w.Write(0, 1);
            w.WriteSigned(first, bps);
            return;
        }

        long verbatimBits = 8L + (long)samples.Length * bps;
        var fixedChoice = FlacFixedSubframeEncoder.PickBest(samples, bps);
        long fixedBits = fixedChoice?.TotalSubframeBits ?? long.MaxValue;

        // Cheapest baseline between VERBATIM and FIXED becomes the bar LPC has to beat.
        long bestSoFar = Math.Min(verbatimBits, fixedBits);
        var lpcChoice = FlacLpcSubframeEncoder.PickBest(samples, bps, bestSoFar);

        if (lpcChoice is not null)
        {
            FlacLpcSubframeEncoder.Emit(w, samples, bps, lpcChoice);
            return;
        }
        if (fixedChoice is not null && fixedBits < verbatimBits)
        {
            FlacFixedSubframeEncoder.Emit(w, samples, bps, fixedChoice);
            return;
        }

        // VERBATIM fallback.
        w.Write(0, 1);
        w.Write(0b000001, 6);
        w.Write(0, 1);
        for (int i = 0; i < samples.Length; i++)
            w.WriteSigned(samples[i], bps);
    }
}
