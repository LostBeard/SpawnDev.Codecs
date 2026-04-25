// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 scan-position neighbor tables - bit-exact copy of libvpx
// vp9/common/vp9_scan.c.
//
// Each scan position i has up to MAX_NEIGHBORS = 2 raster-position
// neighbors that drive the entropy decoder's coefficient context
// (libvpx get_coef_context). The table layout for a scan of N
// coefficients is (N + 1) * 2 entries:
//   neighbors[2*i + 0] and neighbors[2*i + 1] are the two neighbors
//   of scan position i. The trailing pair at index N is a boundary
//   marker (libvpx reads one element past the EOB).
//
// Slice 137 ships the 4x4 and 8x8 tables only (small, hand-verifiable
// from a single libvpx fetch). The 16x16 and 32x32 neighbor tables
// follow in slice 138 after a fresh fetch - the WebFetch summary I
// pulled today appeared to duplicate an interior section of the 32x32
// array, so I'm refusing to copy unverified bytes into the codec.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 coefficient-context neighbor tables (libvpx vp9_scan.c).
/// Each entry is a raster position; pairs are read at indices
/// (2*i, 2*i + 1) for scan position i.
/// </summary>
public static class Vp9NeighborTables
{
    /// <summary>4x4 default-scan neighbors. (16 + 1) * 2 = 34 entries.</summary>
    public static readonly ushort[] DefaultScan4x4Neighbors = new ushort[]
    {
        0, 0, 0, 0, 0, 0, 1, 4, 4, 4, 1, 1, 8, 8, 5, 8,
        2, 2, 2, 5, 9, 12, 6, 9, 3, 6, 10, 13, 7, 10, 11, 14,
        0, 0,
    };

    /// <summary>4x4 row-scan neighbors. 34 entries.</summary>
    public static readonly ushort[] RowScan4x4Neighbors = new ushort[]
    {
        0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 2, 2, 5, 5, 4, 4,
        8, 8, 6, 6, 8, 8, 9, 9, 12, 12, 10, 10, 13, 13, 14, 14,
        0, 0,
    };

    /// <summary>4x4 column-scan neighbors. 34 entries.</summary>
    public static readonly ushort[] ColScan4x4Neighbors = new ushort[]
    {
        0, 0, 0, 0, 4, 4, 0, 0, 8, 8, 1, 1, 5, 5, 1, 1,
        9, 9, 2, 2, 6, 6, 2, 2, 3, 3, 10, 10, 7, 7, 11, 11,
        0, 0,
    };

    /// <summary>8x8 default-scan neighbors. (64 + 1) * 2 = 130 entries.</summary>
    public static readonly ushort[] DefaultScan8x8Neighbors = new ushort[]
    {
        0, 0, 0, 0, 0, 0, 8, 8, 1, 8, 1, 1, 9, 16, 16, 16,
        2, 9, 2, 2, 10, 17, 17, 24, 24, 24, 3, 10, 3, 3, 18, 25,
        25, 32, 11, 18, 32, 32, 4, 11, 26, 33, 19, 26, 4, 4, 33, 40,
        12, 19, 40, 40, 5, 12, 27, 34, 34, 41, 20, 27, 13, 20, 5, 5,
        41, 48, 48, 48, 28, 35, 35, 42, 21, 28, 6, 6, 6, 13, 42, 49,
        49, 56, 36, 43, 14, 21, 29, 36, 7, 14, 43, 50, 50, 57, 22, 29,
        37, 44, 15, 22, 44, 51, 51, 58, 30, 37, 23, 30, 52, 59, 45, 52,
        38, 45, 31, 38, 53, 60, 46, 53, 39, 46, 54, 61, 47, 54, 55, 62,
        0, 0,
    };

    /// <summary>8x8 row-scan neighbors. 130 entries.</summary>
    public static readonly ushort[] RowScan8x8Neighbors = new ushort[]
    {
        0, 0, 0, 0, 1, 1, 0, 0, 8, 8, 2, 2, 8, 8, 9, 9,
        3, 3, 16, 16, 10, 10, 16, 16, 4, 4, 17, 17, 24, 24, 11, 11,
        18, 18, 25, 25, 24, 24, 5, 5, 12, 12, 19, 19, 32, 32, 26, 26,
        6, 6, 33, 33, 32, 32, 20, 20, 27, 27, 40, 40, 13, 13, 34, 34,
        40, 40, 41, 41, 28, 28, 35, 35, 48, 48, 21, 21, 42, 42, 14, 14,
        48, 48, 36, 36, 49, 49, 43, 43, 29, 29, 56, 56, 22, 22, 50, 50,
        57, 57, 44, 44, 37, 37, 51, 51, 30, 30, 58, 58, 52, 52, 45, 45,
        59, 59, 38, 38, 60, 60, 46, 46, 53, 53, 54, 54, 61, 61, 62, 62,
        0, 0,
    };

    /// <summary>8x8 column-scan neighbors. 130 entries.</summary>
    public static readonly ushort[] ColScan8x8Neighbors = new ushort[]
    {
        0, 0, 0, 0, 8, 8, 0, 0, 16, 16, 1, 1, 24, 24, 9, 9,
        1, 1, 32, 32, 17, 17, 2, 2, 25, 25, 10, 10, 40, 40, 2, 2,
        18, 18, 33, 33, 3, 3, 48, 48, 11, 11, 26, 26, 3, 3, 41, 41,
        19, 19, 34, 34, 4, 4, 27, 27, 12, 12, 49, 49, 42, 42, 20, 20,
        4, 4, 35, 35, 5, 5, 28, 28, 50, 50, 43, 43, 13, 13, 36, 36,
        5, 5, 21, 21, 51, 51, 29, 29, 6, 6, 44, 44, 14, 14, 6, 6,
        37, 37, 52, 52, 22, 22, 7, 7, 30, 30, 45, 45, 15, 15, 38, 38,
        23, 23, 53, 53, 31, 31, 46, 46, 39, 39, 54, 54, 47, 47, 55, 55,
        0, 0,
    };

    /// <summary>
    /// Look up the neighbor table for a 4x4 transform with the given
    /// scan flavor. Mirrors libvpx <c>get_scan_and_band</c>.
    /// </summary>
    public static ushort[] GetNeighbors4x4(Vp9ScanType scanType) => scanType switch
    {
        Vp9ScanType.Row => RowScan4x4Neighbors,
        Vp9ScanType.Col => ColScan4x4Neighbors,
        _ => DefaultScan4x4Neighbors,
    };

    /// <summary>Neighbor table for an 8x8 transform with the given scan flavor.</summary>
    public static ushort[] GetNeighbors8x8(Vp9ScanType scanType) => scanType switch
    {
        Vp9ScanType.Row => RowScan8x8Neighbors,
        Vp9ScanType.Col => ColScan8x8Neighbors,
        _ => DefaultScan8x8Neighbors,
    };
}
