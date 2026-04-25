// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 coefficient band tables - bit-exact copy of libvpx
// vp9/common/vp9_entropy.c.
//
// Each scan position belongs to one of six coefficient "bands"
// (0..5). The entropy decoder uses the band as one axis of the
// probability table lookup (the other axes are tx_size, plane_type,
// reference type, and per-coefficient context). Bands group scan
// positions that share statistics: position 0 (DC) is its own
// band, the next two AC positions are band 1, etc.
//
// Two tables exist:
//   - coefband_4x4 (16 entries) for 4x4 transforms.
//   - coefband_trans_8x8plus (1024 entries) shared across 8x8,
//     16x16, and 32x32 transforms - each size uses only its
//     prefix (first 64 / 256 / 1024 entries respectively).
//
// Layout pattern (libvpx convention):
//   coefband_4x4:        bands 0..5 = sizes [1, 2, 3, 4, 3, 3]
//   coefband_8x8plus:    bands 0..5 = sizes [1, 2, 3, 4, 11, 1003]
//
// The two tables differ only in how band 4 and band 5 are sized -
// the AC head-of-scan is identical across all sizes.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 coefficient band lookup tables. Maps scan position to band
/// index 0..5 for the entropy decoder.
/// </summary>
public static class Vp9CoefBands
{
    /// <summary>4x4 coefficient bands (16 entries).</summary>
    public static readonly byte[] CoefBand4x4 = new byte[]
    {
        0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
    };

    /// <summary>
    /// Coefficient bands shared across 8x8 / 16x16 / 32x32 (1024
    /// entries). Each size uses only its prefix.
    /// </summary>
    public static readonly byte[] CoefBandTrans8x8Plus = BuildCoefBandTrans8x8Plus();

    /// <summary>
    /// Look up the coefficient band for a given scan position and
    /// transform size. 4x4 uses CoefBand4x4; every larger size shares
    /// the prefix of CoefBandTrans8x8Plus.
    /// </summary>
    public static byte GetBand(Vp9TxSize txSize, int scanPos)
    {
        if (scanPos < 0) throw new ArgumentOutOfRangeException(nameof(scanPos));
        return txSize switch
        {
            Vp9TxSize.Tx4x4 => scanPos < 16
                ? CoefBand4x4[scanPos]
                : throw new ArgumentOutOfRangeException(nameof(scanPos)),
            Vp9TxSize.Tx8x8 => scanPos < 64
                ? CoefBandTrans8x8Plus[scanPos]
                : throw new ArgumentOutOfRangeException(nameof(scanPos)),
            Vp9TxSize.Tx16x16 => scanPos < 256
                ? CoefBandTrans8x8Plus[scanPos]
                : throw new ArgumentOutOfRangeException(nameof(scanPos)),
            Vp9TxSize.Tx32x32 => scanPos < 1024
                ? CoefBandTrans8x8Plus[scanPos]
                : throw new ArgumentOutOfRangeException(nameof(scanPos)),
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
    }

    /// <summary>
    /// Build the 1024-entry shared band table programmatically.
    /// Initialiser form is preferred over a 1024-literal array
    /// because the first 21 entries are the only non-uniform part:
    /// every position past index 20 is band 5.
    /// </summary>
    private static byte[] BuildCoefBandTrans8x8Plus()
    {
        var arr = new byte[1024];
        arr[0] = 0;                              // band 0: 1 entry (DC)
        arr[1] = 1; arr[2] = 1;                  // band 1: 2 entries
        arr[3] = 2; arr[4] = 2; arr[5] = 2;      // band 2: 3 entries
        arr[6] = 3; arr[7] = 3;                  // band 3: 4 entries
        arr[8] = 3; arr[9] = 3;
        for (int i = 10; i <= 20; i++) arr[i] = 4; // band 4: 11 entries
        for (int i = 21; i < 1024; i++) arr[i] = 5; // band 5: 1003 entries
        return arr;
    }
}
