// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis codebook vector lookup. Mirror of
// VorbisCodebookVector.LookupVector inside VorbisResidueDecoder.cs.
//
// Vorbis codebooks have three lookup types:
//   - 0: pure entropy codebook, no vector value (output is always zero)
//   - 1: per-dimension index decoded as digits of `entry` in base
//        `quantvals = lookup1_values(entries, dims)`
//   - 2: flat table indexed by `entry * dims + d`
//
// Multiplicand values are integer indices into a per-codebook quantization
// grid; the actual vector value at dim d is:
//   val[d] = abs(multiplicand) * delta + minValue + (sequenceP ? last : 0)
//   last = sequenceP ? val[d] : 0
//
// Caller flattens VorbisCodebook to flat buffers + scalar params for GPU
// upload (the per-codebook config is a fixed metadata struct setup
// allowed under the CARDINAL rule).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis codebook vector lookup. Mirror of
/// <see cref="VorbisResidueDecoder"/>.LookupVector.
/// </summary>
public static class VorbisCodebookVectorLookupGpu
{
    /// <summary>
    /// Look up the vector value for a codebook entry. Writes the result
    /// into <paramref name="outVec"/> starting at <paramref name="outBase"/>
    /// (length = <paramref name="dimensions"/>).
    /// </summary>
    /// <param name="multiplicands">Flat integer multiplicand array.</param>
    /// <param name="multBase">Base offset into <paramref name="multiplicands"/>.</param>
    /// <param name="multLen">Number of valid multiplicand entries for this codebook.</param>
    /// <param name="entry">Codebook entry index (a negative or out-of-range
    /// value yields the zero-vector regardless of lookup type).</param>
    /// <param name="entries">Total entries in the codebook.</param>
    /// <param name="dimensions">Vector dimensionality.</param>
    /// <param name="lookupType">0, 1, or 2 (0 = no vector, 1 = quant-base digits, 2 = flat).</param>
    /// <param name="quantvals">Number of quantization values per dimension
    /// (lookup1_values(entries, dims), only used for type 1).</param>
    /// <param name="minValue">MinValue scalar.</param>
    /// <param name="deltaValue">DeltaValue scalar.</param>
    /// <param name="sequenceP">SequenceP flag (1 = sequential reconstruction).</param>
    /// <param name="outVec">Output buffer.</param>
    /// <param name="outBase">Base offset into <paramref name="outVec"/>.</param>
    public static void LookupVector(
        ArrayView<int> multiplicands, long multBase, int multLen,
        int entry, int entries, int dimensions, int lookupType,
        int quantvals,
        double minValue, double deltaValue, int sequenceP,
        ArrayView<float> outVec, long outBase)
    {
        // Type 0 + out-of-range entry -> zero vector.
        if (lookupType == 0 || entry < 0 || entry >= entries)
        {
            for (int d = 0; d < dimensions; d++) outVec[outBase + d] = 0f;
            return;
        }

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
                outVec[outBase + d] = (float)val;
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
            outVec[outBase + d] = (float)val;
        }
    }
}
