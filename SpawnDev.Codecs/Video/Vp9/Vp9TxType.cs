// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 transform type (TX_TYPE) - the 2D inverse transform code that
// drives row/column iDCT vs iADST selection. Same numeric values
// across all sizes (4x4 / 8x8 / 16x16); 32x32 always uses DCT_DCT.
//
// libvpx reference: vp9/common/vp9_blockd.h
//   typedef enum {
//     DCT_DCT = 0,
//     ADST_DCT = 1,    // ADST in row, DCT in column
//     DCT_ADST = 2,    // DCT in row, ADST in column
//     ADST_ADST = 3,   // ADST in both rows and columns
//     TX_TYPES = 4
//   } TX_TYPE;
//
// And the per-intra-mode mapping:
//   static const TX_TYPE intra_mode_to_tx_type_lookup[INTRA_MODES] = {
//     DCT_DCT,    // DC
//     ADST_DCT,   // V
//     DCT_ADST,   // H
//     DCT_DCT,    // D45
//     ADST_ADST,  // D135
//     ADST_DCT,   // D117
//     DCT_ADST,   // D153
//     DCT_ADST,   // D207
//     ADST_DCT,   // D63
//     ADST_ADST,  // TM
//   };
//
// Existing Vp9TxType4x4, Vp9TxType8x8, Vp9TxType16x16 enums live next
// to the Iht reference functions and use the same numeric values; this
// file adds the single canonical enum + the per-mode lookup.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 transform type. Matches libvpx <c>TX_TYPE</c>. The low bit
/// selects the ROW transform (0 = iDCT, 1 = iADST) and the high bit
/// selects the COLUMN transform (0 = iDCT, 1 = iADST).
/// </summary>
public enum Vp9TxType : byte
{
    /// <summary>iDCT in both row and column passes.</summary>
    DctDct = 0,
    /// <summary>iADST in rows, iDCT in columns.</summary>
    AdstDct = 1,
    /// <summary>iDCT in rows, iADST in columns.</summary>
    DctAdst = 2,
    /// <summary>iADST in both row and column passes.</summary>
    AdstAdst = 3,
}

/// <summary>
/// Mapping from VP9 intra prediction mode to the transform type used
/// to inverse the residual coefficients of an intra block. Mirror of
/// libvpx <c>intra_mode_to_tx_type_lookup</c> in vp9_blockd.h.
/// </summary>
public static class Vp9IntraTxType
{
    /// <summary>
    /// Lookup table indexed by <see cref="Vp9IntraMode"/> (0..9). All
    /// 32x32 blocks use <see cref="Vp9TxType.DctDct"/> regardless of
    /// the intra mode (libvpx hard-codes this); this table only
    /// applies to 4x4 / 8x8 / 16x16 transform sizes.
    /// </summary>
    public static readonly Vp9TxType[] Lookup = new Vp9TxType[10]
    {
        Vp9TxType.DctDct,    // 0  DC_PRED
        Vp9TxType.AdstDct,   // 1  V_PRED
        Vp9TxType.DctAdst,   // 2  H_PRED
        Vp9TxType.DctDct,    // 3  D45_PRED
        Vp9TxType.AdstAdst,  // 4  D135_PRED
        Vp9TxType.AdstDct,   // 5  D117_PRED
        Vp9TxType.DctAdst,   // 6  D153_PRED
        Vp9TxType.DctAdst,   // 7  D207_PRED
        Vp9TxType.AdstDct,   // 8  D63_PRED
        Vp9TxType.AdstAdst,  // 9  TM_PRED
    };

    /// <summary>
    /// Return the inverse transform type for an intra block with
    /// <paramref name="mode"/> at sub-32x32 sizes. Throws if
    /// <paramref name="mode"/> is out of range.
    /// </summary>
    public static Vp9TxType ForMode(Vp9IntraMode mode)
    {
        int idx = (int)mode;
        if ((uint)idx >= 10)
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "mode must be 0..9");
        return Lookup[idx];
    }
}
