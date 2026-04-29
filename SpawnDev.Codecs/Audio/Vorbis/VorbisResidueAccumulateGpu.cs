// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable codebook vector lookup + accumulate-into-buffer for
// Vorbis residue decode. Combines VorbisCodebookVectorLookupGpu's
// lookup with a per-element add to a channel buffer at a target
// offset, matching the inner loop of VorbisResidueDecoder type 0/1.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable codebook vector lookup + accumulate. Bit-exact mirror
/// of the inner residue decode loop:
///   lookup vec via VorbisResidueDecoder.VorbisCodebookVector.LookupVector
///   for d in [0..dims): targetBuffer[targetBase + d] += vec[d]
/// </summary>
public static class VorbisResidueAccumulateGpu
{
    /// <summary>
    /// Look up a codebook entry's vector and accumulate it into
    /// <paramref name="target"/> at <paramref name="targetBase"/>.
    /// </summary>
    public static void LookupAndAccumulate(
        ArrayView<int> multiplicands, long multBase, int multLen,
        int entry, int entries, int dimensions, int lookupType,
        int quantvals,
        double minValue, double deltaValue, int sequenceP,
        ArrayView<float> target, long targetBase)
    {
        // Type 0 + out-of-range entry -> add zero (no-op).
        if (lookupType == 0 || entry < 0 || entry >= entries) return;

        double last = 0;

        if (lookupType == 1)
        {
            int indexDivisor = 1;
            for (int d = 0; d < dimensions; d++)
            {
                int multIndex = (entry / indexDivisor) % quantvals;
                int m = multIndex < multLen ? multiplicands[multBase + multIndex] : 0;
                int absM = m < 0 ? -m : m;
                double val = (double)absM * deltaValue + minValue + last;
                if (sequenceP != 0) last = val;
                target[targetBase + d] += (float)val;
                indexDivisor *= quantvals;
            }
            return;
        }

        // Lookup type 2: flat table indexed by (entry * dims + d).
        int baseIndex = entry * dimensions;
        for (int d = 0; d < dimensions; d++)
        {
            int flatIndex = baseIndex + d;
            int m = (flatIndex < 0 || flatIndex >= multLen)
                ? 0
                : multiplicands[multBase + flatIndex];
            int absM = m < 0 ? -m : m;
            double val = (double)absM * deltaValue + minValue + last;
            if (sequenceP != 0) last = val;
            target[targetBase + d] += (float)val;
        }
    }
}
