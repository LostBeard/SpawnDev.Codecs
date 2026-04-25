// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 default coefficient probabilities for the 4x4 transform.
// Bit-exact copy of libvpx vp9/common/vp9_entropy.c
// `default_coef_probs_4x4`. Slice 142 of Phase 1b.
//
// Layout: byte[2 plane][2 ref][6 band][6 ctx][3 node], stored flat
// in row-major order with the same axis ordering as the libvpx
// 5D array. The flat index for (plane, ref, band, ctx, node) is:
//   ((((plane * 2 + ref) * 6 + band) * 6 + ctx) * 3 + node)
//
// Total length: 2 * 2 * 6 * 6 * 3 = 432 bytes.
//
// Important shape detail: the libvpx C source only initializes 3
// contexts for Band 0 (the DC band - it has fewer entropy contexts
// than the AC bands). The remaining 3 ctx slots in band 0 are
// implicitly zero. To preserve the rectangular flat layout, this
// file zero-pads ctx 3..5 of every band 0 across both planes and
// both reference types - 9 zero bytes per (plane, ref) combination,
// 36 zero bytes total. The coefficient decoder never reads these
// padded slots; they exist only so the index arithmetic stays
// rectangular.
//
// Provenance: extracted directly from libvpx vp9_entropy.c (lines
// 370-533) via raw text fetch + awk pipeline. Pipeline tracked the
// "// Band 0" comment to know when to inject the zero pad. Total
// entries verified at 432 before commit.

namespace SpawnDev.Codecs.Video.Vp9;

public static partial class Vp9CoefProbs
{
    /// <summary>
    /// VP9 default coefficient probabilities for 4x4 transforms.
    /// Flat byte[432] = [2 plane][2 ref][6 band][6 ctx][3 node]
    /// in row-major order. See <see cref="Index4x4"/> for the
    /// per-axis indexing convention.
    /// </summary>
    public static readonly byte[] DefaultCoefProbs4x4 = new byte[]
    {
        195, 29, 183, 84, 49, 136, 8, 42, 71, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 31, 107, 169, 35, 99, 159, 17, 82, 140, 8, 66, 114, 2, 44,
        76, 1, 19, 32, 40, 132, 201, 29, 114, 187, 13, 91, 157, 7, 75, 127,
        3, 58, 95, 1, 28, 47, 69, 142, 221, 42, 122, 201, 15, 91, 159, 6,
        67, 121, 1, 42, 77, 1, 17, 31, 102, 148, 228, 67, 117, 204, 17, 82,
        154, 6, 59, 114, 2, 39, 75, 1, 15, 29, 156, 57, 233, 119, 57, 212,
        58, 48, 163, 29, 40, 124, 12, 30, 81, 3, 12, 31, 191, 107, 226, 124,
        117, 204, 25, 99, 155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 148,
        210, 37, 126, 194, 8, 93, 157, 2, 68, 118, 1, 39, 69, 1, 17, 33,
        41, 151, 213, 27, 123, 193, 3, 82, 144, 1, 58, 105, 1, 32, 60, 1,
        13, 26, 59, 159, 220, 23, 126, 198, 4, 88, 151, 1, 66, 114, 1, 38,
        71, 1, 18, 34, 114, 136, 232, 51, 114, 207, 11, 83, 155, 3, 56, 105,
        1, 33, 65, 1, 17, 34, 149, 65, 234, 121, 57, 215, 61, 49, 166, 28,
        36, 114, 12, 25, 76, 3, 16, 42, 214, 49, 220, 132, 63, 188, 42, 65,
        137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 137, 221, 104, 131, 216,
        49, 111, 192, 21, 87, 155, 2, 49, 87, 1, 16, 28, 89, 163, 230, 90,
        137, 220, 29, 100, 183, 10, 70, 135, 2, 42, 81, 1, 17, 33, 108, 167,
        237, 55, 133, 222, 15, 97, 179, 4, 72, 135, 1, 45, 85, 1, 19, 38,
        124, 146, 240, 66, 124, 224, 17, 88, 175, 4, 58, 122, 1, 36, 75, 1,
        18, 37, 141, 79, 241, 126, 70, 227, 66, 58, 182, 30, 44, 136, 12, 34,
        96, 2, 20, 47, 229, 99, 249, 143, 111, 235, 46, 109, 192, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 82, 158, 236, 94, 146, 224, 25, 117, 191, 9,
        87, 149, 3, 56, 99, 1, 33, 57, 83, 167, 237, 68, 145, 222, 10, 103,
        177, 2, 72, 131, 1, 41, 79, 1, 20, 39, 99, 167, 239, 47, 141, 224,
        10, 104, 178, 2, 73, 133, 1, 44, 85, 1, 22, 47, 127, 145, 243, 71,
        129, 228, 17, 93, 177, 3, 61, 124, 1, 41, 84, 1, 21, 52, 157, 78,
        244, 140, 72, 231, 69, 58, 184, 31, 44, 137, 14, 38, 105, 8, 23, 61,
    };

    /// <summary>
    /// Compute the flat index into <see cref="DefaultCoefProbs4x4"/>
    /// for the given (plane, ref, band, ctx, node) tuple.
    /// </summary>
    public static int Index4x4(int plane, int refType, int band, int ctx, int node)
    {
        return ((((plane * 2 + refType) * 6 + band) * 6 + ctx) * 3 + node);
    }
}
