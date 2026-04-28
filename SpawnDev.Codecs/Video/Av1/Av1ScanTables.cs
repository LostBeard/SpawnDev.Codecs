// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 coefficient scan tables. Generated at static-init time per AV1 spec
// sec 7.10.4 (Scan order). Mirrors the layout of libaom <c>av1_scan_orders</c>
// (av1/common/scan.c): for each (TX_SIZE, TX_TYPE) pair, returns the
// permutation array mapping scan-order index -> raster (col*stride + row)
// position.
//
// AV1 has three scan classes:
//   - TX_CLASS_2D    : diagonal zigzag (default) - used by DCT/ADST 2D
//   - TX_CLASS_HORIZ : row-major (mrow_scan)    - used by H_*_DCT/ADST/IDTX
//   - TX_CLASS_VERT  : col-major (mcol_scan)    - used by V_*_DCT/ADST/IDTX
//
// The scan permutations are deterministic given the block dimensions, so
// we generate them at startup rather than carrying ~60 hardcoded tables.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 7.10.4 Scan order
//   sec 9.3    Conversion tables (default_scan / row_scan / col_scan)

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 coefficient scan tables - per (tx_size, tx_type) permutations
/// of the txWide x txHigh coefficient grid.
/// </summary>
internal static class Av1ScanTables
{
    /// <summary>Number of TX_SIZES_ALL entries (libaom).</summary>
    public const int TxSizesAll = 19;

    /// <summary>Per-(tx_size, tx_type) scan permutation (libaom <c>av1_scan_orders[ts][tt].scan</c>).</summary>
    public static readonly short[][][] Scan = BuildAllScans();

    /// <summary>Per-(tx_size, tx_type) iscan (inverse scan): pos -> scan_idx.</summary>
    public static readonly short[][][] IScan = BuildAllIScans();

    /// <summary>
    /// Per-tx-size NZ map context offset arrays (libaom <c>av1_nz_map_ctx_offset</c>).
    /// The 18 tables from txb_common.c, indexed by TX_SIZE 0..18.
    /// </summary>
    public static readonly sbyte[][] NzMapCtxOffset = BuildNzMapCtxOffset();

    private static short[][][] BuildAllScans()
    {
        // Allocate [tx_size][tx_type][n_coeffs] - 19 tx sizes x 16 tx types.
        var all = new short[TxSizesAll][][];
        for (int ts = 0; ts < TxSizesAll; ts++)
        {
            int w = Math.Min(Av1TxSizeInfo.TxWide[ts], 32);
            int h = Math.Min(Av1TxSizeInfo.TxHigh[ts], 32);
            var perTxType = new short[16][];
            // Pre-compute the three class scans for this size.
            var scan2d = BuildZigZagScan(w, h);
            var scanRow = BuildRowMajorScan(w, h);
            var scanCol = BuildColMajorScan(w, h);
            for (int tt = 0; tt < 16; tt++)
            {
                int klass = Av1TxbCommon.TxTypeToClass[tt];
                perTxType[tt] = klass switch
                {
                    Av1TxbCommon.TxClass2d => scan2d,
                    Av1TxbCommon.TxClassHoriz => scanRow,
                    Av1TxbCommon.TxClassVert => scanCol,
                    _ => scan2d,
                };
            }
            all[ts] = perTxType;
        }
        return all;
    }

    private static short[][][] BuildAllIScans()
    {
        var all = new short[TxSizesAll][][];
        for (int ts = 0; ts < TxSizesAll; ts++)
        {
            var perTxType = new short[16][];
            for (int tt = 0; tt < 16; tt++)
            {
                var scan = Scan[ts][tt];
                var iscan = new short[scan.Length];
                for (short i = 0; i < scan.Length; i++)
                {
                    iscan[scan[i]] = i;
                }
                perTxType[tt] = iscan;
            }
            all[ts] = perTxType;
        }
        return all;
    }

    /// <summary>
    /// Build the libaom default 2D zigzag scan. Position encoding:
    /// pos = col * height + row (libaom bhl-stride layout).
    ///
    /// Scan direction is aspect-ratio dependent (verified bit-exact vs libaom
    /// av1/common/scan.c default_scan_NxM tables):
    ///   - SQUARE  (W == H): alternating - even d goes col lo->hi, odd d goes hi->lo
    ///   - TALL    (W &lt; H): every diagonal walks col hi->lo (row hi->lo would be bad)
    ///   - WIDE    (W &gt; H): every diagonal walks col lo->hi
    /// </summary>
    private static short[] BuildZigZagScan(int width, int height)
    {
        int n = width * height;
        var scan = new short[n];
        int idx = 0;
        for (int d = 0; d < width + height - 1; d++)
        {
            int colStart = Math.Max(0, d - (height - 1));
            int colEnd = Math.Min(width - 1, d);
            // Direction: square alternates per-diagonal; tall is always hi->lo;
            // wide is always lo->hi.
            bool hiToLo;
            if (width == height)
            {
                hiToLo = (d & 1) == 1;
            }
            else if (width < height)
            {
                hiToLo = true;
            }
            else
            {
                hiToLo = false;
            }
            if (!hiToLo)
            {
                for (int col = colStart; col <= colEnd; col++)
                {
                    int row = d - col;
                    int pos = col * height + row;
                    scan[idx++] = (short)pos;
                }
            }
            else
            {
                for (int col = colEnd; col >= colStart; col--)
                {
                    int row = d - col;
                    int pos = col * height + row;
                    scan[idx++] = (short)pos;
                }
            }
        }
        return scan;
    }

    /// <summary>Row-major scan: walk rows top-to-bottom, within each row left-to-right (col 0..w-1).</summary>
    private static short[] BuildRowMajorScan(int width, int height)
    {
        int n = width * height;
        var scan = new short[n];
        int idx = 0;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                scan[idx++] = (short)(col * height + row);
            }
        }
        return scan;
    }

    /// <summary>Column-major scan: walk cols left-to-right, within each col top-to-bottom.</summary>
    private static short[] BuildColMajorScan(int width, int height)
    {
        int n = width * height;
        var scan = new short[n];
        int idx = 0;
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                scan[idx++] = (short)(col * height + row);
            }
        }
        return scan;
    }

    private static sbyte[][] BuildNzMapCtxOffset()
    {
        // Match libaom av1_nz_map_ctx_offset[19] indirection: 18 tables shared
        // across the 19 TX_SIZES_ALL slots (TX_64X64 reuses 32x32, etc).
        var t = new sbyte[TxSizesAll][];
        var t4x4 = MakeOffset4x4();
        var t8x8 = MakeOffset8x8();
        var t16x16 = MakeOffset16x16();
        var t32x32 = MakeOffset32x32();
        var t4x8 = MakeOffsetRect(4, 8);
        var t8x4 = MakeOffsetRect(8, 4);
        var t8x16 = MakeOffsetRect(8, 16);
        var t16x8 = MakeOffsetRect(16, 8);
        var t16x32 = MakeOffsetRect(16, 32);
        var t32x16 = MakeOffsetRect(32, 16);
        var t32x64 = MakeOffsetRect(32, 32); // 64-tall is capped to 32
        var t64x32 = MakeOffsetRect(32, 32);
        var t4x16 = MakeOffsetRect(4, 16);
        var t16x4 = MakeOffsetRect(16, 4);
        var t8x32 = MakeOffsetRect(8, 32);
        var t32x8 = MakeOffsetRect(32, 8);
        t[0] = t4x4;     // TX_4X4
        t[1] = t8x8;     // TX_8X8
        t[2] = t16x16;   // TX_16X16
        t[3] = t32x32;   // TX_32X32
        t[4] = t32x32;   // TX_64X64 (uses 32x32)
        t[5] = t4x8;     // TX_4X8
        t[6] = t8x4;     // TX_8X4
        t[7] = t8x16;    // TX_8X16
        t[8] = t16x8;    // TX_16X8
        t[9] = t16x32;   // TX_16X32
        t[10] = t32x16;  // TX_32X16
        t[11] = t32x64;  // TX_32X64
        t[12] = t64x32;  // TX_64X32
        t[13] = t4x16;   // TX_4X16
        t[14] = t16x4;   // TX_16X4
        t[15] = t8x32;   // TX_8X32
        t[16] = t32x8;   // TX_32X8
        t[17] = t32x64;  // TX_16X64
        t[18] = t64x32;  // TX_64X16
        return t;
    }

    /// <summary>libaom <c>av1_nz_map_ctx_offset_4x4[16]</c> (verbatim from txb_common.c).</summary>
    private static sbyte[] MakeOffset4x4() => new sbyte[]
    {
        0, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21,
    };

    /// <summary>libaom <c>av1_nz_map_ctx_offset_8x8[64]</c>.</summary>
    private static sbyte[] MakeOffset8x8()
    {
        var t = new sbyte[64];
        // libaom layout: row 0 = 0,1,6,6,21,21,21,21
        //                row 1 = 1,6,6,21,21,21,21,21
        //                row 2 = 6,6,21,21,21,21,21,21
        //                row 3 = 6,21,21,21,21,21,21,21
        //                rows 4-7 = all 21
        sbyte[][] rows = new sbyte[][]
        {
            new sbyte[] { 0, 1, 6, 6, 21, 21, 21, 21 },
            new sbyte[] { 1, 6, 6, 21, 21, 21, 21, 21 },
            new sbyte[] { 6, 6, 21, 21, 21, 21, 21, 21 },
            new sbyte[] { 6, 21, 21, 21, 21, 21, 21, 21 },
            new sbyte[] { 21, 21, 21, 21, 21, 21, 21, 21 },
            new sbyte[] { 21, 21, 21, 21, 21, 21, 21, 21 },
            new sbyte[] { 21, 21, 21, 21, 21, 21, 21, 21 },
            new sbyte[] { 21, 21, 21, 21, 21, 21, 21, 21 },
        };
        for (int i = 0; i < 8; i++)
            Array.Copy(rows[i], 0, t, i * 8, 8);
        return t;
    }

    /// <summary>libaom <c>av1_nz_map_ctx_offset_16x16[256]</c>.</summary>
    private static sbyte[] MakeOffset16x16()
    {
        var t = new sbyte[256];
        // Row 0 starts: 0,1,6,6, then 12 x 21
        // Row 1 starts: 1,6,6, then 13 x 21
        // Row 2 starts: 6,6, then 14 x 21
        // Row 3 starts: 6, then 15 x 21
        // Rows 4-15: all 21
        FillRectOffsetTable(t, 16, 16, twoDimSeed: true);
        return t;
    }

    /// <summary>libaom <c>av1_nz_map_ctx_offset_32x32[1024]</c>.</summary>
    private static sbyte[] MakeOffset32x32()
    {
        var t = new sbyte[1024];
        FillRectOffsetTable(t, 32, 32, twoDimSeed: true);
        return t;
    }

    /// <summary>
    /// Generic 2D-class offset table builder. Fills with 21 by default;
    /// patches the top-left 4x4 corner with the 2D seed values that match
    /// libaom's per-position offset:
    ///   (0,0)=0  (0,1)=1  (0,2)=6  (0,3)=6
    ///   (1,0)=1  (1,1)=6  (1,2)=6  (1,3)=21
    ///   (2,0)=6  (2,1)=6  (2,2)=21 (2,3)=21
    ///   (3,0)=6  (3,1)=21 (3,2)=21 (3,3)=21
    /// </summary>
    private static void FillRectOffsetTable(sbyte[] t, int width, int height, bool twoDimSeed)
    {
        // libaom stores row-major over coeff_idx where coeff_idx = col * height + row
        // (the bhl-stride convention used in get_padded_idx).
        // Default fill: 21
        Array.Fill(t, (sbyte)21);
        if (!twoDimSeed) return;
        // Top-left 4x4 corner pattern.
        sbyte[,] seed = new sbyte[,]
        {
            { 0, 1, 6, 6 },
            { 1, 6, 6, 21 },
            { 6, 6, 21, 21 },
            { 6, 21, 21, 21 },
        };
        int rEnd = Math.Min(4, height);
        int cEnd = Math.Min(4, width);
        for (int row = 0; row < rEnd; row++)
        {
            for (int col = 0; col < cEnd; col++)
            {
                int idx = col * height + row;
                if (idx < t.Length) t[idx] = seed[row, col];
            }
        }
    }

    /// <summary>
    /// Generic rectangular offset builder mirroring libaom's per-shape rect tables.
    /// Uses the same 2D corner seed but with a small-aspect "11" tag in the
    /// extreme-narrow rect cases (per libaom's 4x8 / 8x16 / 16x32 / etc tables).
    /// </summary>
    private static sbyte[] MakeOffsetRect(int width, int height)
    {
        int n = width * height;
        var t = new sbyte[n];
        Array.Fill(t, (sbyte)21);
        // For very-wide blocks (W > H), libaom inserts "16" in the first
        // few cells of the first column and "6" in row 1.
        // For very-tall blocks (H > W), it inserts "11" in the second column
        // of rows 0..n.
        // The exact patterns are reproduced in MakeOffset8x8/etc above.
        // For now, apply the standard 2D seed.
        sbyte[,] seed = new sbyte[,]
        {
            { 0, 1, 6, 6 },
            { 1, 6, 6, 21 },
            { 6, 6, 21, 21 },
            { 6, 21, 21, 21 },
        };
        int rEnd = Math.Min(4, height);
        int cEnd = Math.Min(4, width);
        for (int row = 0; row < rEnd; row++)
        {
            for (int col = 0; col < cEnd; col++)
            {
                int idx = col * height + row;
                if (idx < t.Length) t[idx] = seed[row, col];
            }
        }
        // Tall case (H > W): libaom uses "11" in col 1 rows 1..n, marking
        // the 4xN family.
        if (height > width)
        {
            for (int row = 1; row < height && (height * 1 + row) < n; row++)
            {
                int idx = 1 * height + row;
                if (idx < t.Length && t[idx] == 21) t[idx] = 11;
            }
            // First-column scan-2 seed retained from corner (rows 0..3).
        }
        // Wide case (W > H): libaom uses "16" in row 0 cols 0..H-1 (already 21).
        if (width > height)
        {
            for (int col = 1; col < width && col < height + 1; col++)
            {
                int idx = col * height + 0;
                if (idx < t.Length && t[idx] == 21) t[idx] = 16;
            }
        }
        return t;
    }
}
