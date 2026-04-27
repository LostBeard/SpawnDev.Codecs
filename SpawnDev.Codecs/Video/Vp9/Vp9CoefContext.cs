// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 coefficient context computation. Bridges the per-block decoded
// token history (stored as energy classes per slice 145's
// pt_energy_class lookup) and the neighbor tables (slices 137 + 139)
// into a 0/1/2 entropy context value used to index the coefficient
// probability table at each scan position.
//
// libvpx reference: vp9/common/vp9_entropy.h
//   `get_coef_context` and `vp9_pt_energy_class[]`.
//
// The probability lookup that drives the coefficient decoder is
//   probs[tx_size][plane_type][ref_type][band][ctx][model_node]
// where ctx is the 0/1/2 value computed by GetCoefContext from the
// energy classes of the two pre-decoded raster neighbors. This slice
// ships the energy class lookup and the context helper; the
// per-block decoder loop is the next slice.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 per-coefficient entropy context utilities.
/// </summary>
public static class Vp9CoefContext
{
    /// <summary>
    /// Map from <see cref="Vp9CoefToken"/> value (0..11) to energy
    /// class (0..5). Bit-exact with libvpx
    /// <c>vp9_pt_energy_class[ENTROPY_TOKENS]</c>. Energy class 0 is
    /// the "no signal" bucket (Zero / Eob); higher classes correspond
    /// to larger coefficient magnitudes.
    /// </summary>
    public static readonly byte[] PtEnergyClass = new byte[]
    {
        0, // Zero
        1, // One
        2, // Two
        3, // Three
        3, // Four
        4, // Category1
        4, // Category2
        5, // Category3
        5, // Category4
        5, // Category5
        5, // Category6
        0, // Eob (treated as "no signal")
    };

    /// <summary>
    /// Compute the entropy context value (0, 1, or 2) for scan
    /// position <paramref name="scanPos"/> given a prefix of decoded
    /// energy classes laid out by raster position.
    /// </summary>
    /// <param name="neighbors">
    /// Neighbor table for the active (tx_size, scan_type) pair, e.g.
    /// <see cref="Vp9NeighborTables.DefaultScan4x4Neighbors"/>. The
    /// pair (n0, n1) for scan position c lives at indices (2c, 2c+1).
    /// </param>
    /// <param name="tokenCache">
    /// Per-raster-position energy classes of coefficients that have
    /// already been decoded. Indexed by raster position. Initialised
    /// to all-zeros before the scan starts; updated to
    /// <c>PtEnergyClass[token]</c> after each coefficient is
    /// decoded (the caller does this; this helper just reads).
    /// </param>
    /// <param name="scanPos">Scan position whose context is being computed.</param>
    /// <returns>Entropy context value: 0, 1, or 2.</returns>
    public static int GetCoefContext(
        ReadOnlySpan<ushort> neighbors,
        ReadOnlySpan<byte> tokenCache,
        int scanPos)
    {
        if (scanPos < 0) throw new ArgumentOutOfRangeException(nameof(scanPos));
        int n0Index = 2 * scanPos;
        int n1Index = n0Index + 1;
        if (n1Index >= neighbors.Length)
            throw new ArgumentOutOfRangeException(nameof(scanPos),
                "scanPos out of range for the supplied neighbors table");

        ushort n0 = neighbors[n0Index];
        ushort n1 = neighbors[n1Index];
        int e0 = tokenCache[n0];
        int e1 = tokenCache[n1];

        // (1 + e0 + e1) >> 1. libvpx vp9_scan.h get_coef_context returns
        // the raw value with no clamping; result range is [0, MAX_ENERGY_CLASS=5].
        // The coefficient prob table is sized [REF][BAND][COEFF_CONTEXTS=6][3]
        // with BAND_COEFF_CONTEXTS(band) = (band==0 ? 3 : 6); for band 0
        // the only legal scan position is c=0 where tokenCache is all-zero
        // so ctx = 0 naturally. Bands 1..5 must use the full ctx range
        // 0..5; clamping here is what was hiding high-energy probabilities
        // and producing the AC variance under-decode.
        return (1 + e0 + e1) >> 1;
    }
}
