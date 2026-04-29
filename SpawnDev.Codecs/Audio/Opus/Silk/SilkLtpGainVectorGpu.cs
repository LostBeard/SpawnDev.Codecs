// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK LTP gain vector lookup. Mirror of
// SilkLtpDecoder.GetGainVector (libopus silk_decode_parameters LTP filter
// lookup). Reads one of three codebooks (8 / 16 / 32 entries, selected by
// perIndex) and copies a 5-tap Q7 gain vector for a given ltpIndex.
//
// Per-tap parallel: 5 threads per subframe each copy one tap. Per-frame
// dispatch over all subframes can be (nbSubfr * 5) threads with each
// thread picking out its own (subframe, tap) pair.
//
// All 3 codebooks (Vq0, Vq1, Vq2) flattened into a single ArrayView<sbyte>
// with per-codebook offsets (8*5, 16*5, 32*5 = 40, 80, 160 entries each).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LTP gain vector codebook lookup. Mirror of
/// <see cref="SilkLtpDecoder"/>.GetGainVector.
/// </summary>
public static class SilkLtpGainVectorGpu
{
    /// <summary>5 taps per LTP gain vector (libopus LTP_ORDER).</summary>
    public const int LTP_VEC_SIZE = 5;

    /// <summary>
    /// Look up one tap of one subframe's LTP gain vector. Caller dispatches
    /// nbSubfr * LTP_VEC_SIZE threads, each picking its (subfrIdx, tapIdx).
    /// </summary>
    /// <param name="taps">Output: per-subframe Q7 taps. Length nbSubfr * 5.</param>
    /// <param name="tapsBase">Base offset.</param>
    /// <param name="codebook">Concatenated codebook bytes (Vq0[40] + Vq1[80] + Vq2[160]).</param>
    /// <param name="codebookBase">Base offset.</param>
    /// <param name="ltpIndices">Per-subframe LTP entry index. Length nbSubfr.</param>
    /// <param name="ltpIndicesBase">Base offset.</param>
    /// <param name="perIndex">Periodicity index selecting which codebook (0/1/2).</param>
    /// <param name="threadIdx">Linear thread index in [0, nbSubfr * 5).</param>
    public static void LookupTapAt(
        ArrayView<sbyte> taps, long tapsBase,
        ArrayView<sbyte> codebook, long codebookBase,
        ArrayView<sbyte> ltpIndices, long ltpIndicesBase,
        int perIndex, int threadIdx)
    {
        int subfrIdx = threadIdx / LTP_VEC_SIZE;
        int tapIdx = threadIdx - subfrIdx * LTP_VEC_SIZE;

        // Codebook offset within the flattened table.
        // perIndex 0 -> Vq0 starts at 0
        // perIndex 1 -> Vq1 starts at 8 * 5 = 40
        // perIndex 2 -> Vq2 starts at 40 + 16 * 5 = 120
        int cbOffset;
        if (perIndex == 0) cbOffset = 0;
        else if (perIndex == 1) cbOffset = 40;
        else cbOffset = 120;

        int ltpIndex = ltpIndices[ltpIndicesBase + subfrIdx];
        long srcIdx = codebookBase + cbOffset + ltpIndex * LTP_VEC_SIZE + tapIdx;
        taps[tapsBase + subfrIdx * LTP_VEC_SIZE + tapIdx] = codebook[srcIdx];
    }
}
