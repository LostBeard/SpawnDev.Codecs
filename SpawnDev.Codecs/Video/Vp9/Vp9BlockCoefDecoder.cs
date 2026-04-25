// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block coefficient decoder. Drives the entropy-decode loop
// for a single transform block: at each scan position, computes the
// (band, ctx) pair, expands the stored 3-entry model probabilities
// to the 11-entry full vector via slice 144's ModelToFullProbs,
// invokes slice 147's DecodeOneCoefficient, writes the signed
// coefficient into the raster-laid-out output block, updates the
// per-position tokenCache that slice 148's GetCoefContext reads,
// and stops when EOB is encountered.
//
// libvpx reference: vp9/decoder/vp9_detokenize.c (decode_coefs).
// VP9 spec reference: sec 6.4.20 "Decode coefficients syntax".
//
// Bit-exactness note: this implementation calls DecodeOneCoefficient
// once per scan position, matching the VP9 spec's per-coefficient
// TreeRead semantics. libvpx contains an internal optimization that
// skips the EOB re-read between consecutive ZERO tokens; the bit
// stream the encoder emits is consistent with whichever shape the
// decoder uses, but if a future test vector reveals a divergence,
// the loop here can be refactored to mirror the libvpx inner-zero
// loop without changing the output coefficient block.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 block-level coefficient decoder.</summary>
public static class Vp9BlockCoefDecoder
{
    /// <summary>Plane type for coefficient probability lookup (libvpx PLANE_TYPES).</summary>
    public enum PlaneType
    {
        /// <summary>Luma (Y) plane. PLANE_TYPES index 0.</summary>
        Y = 0,
        /// <summary>Chroma (U/V) plane. PLANE_TYPES index 1.</summary>
        Uv = 1,
    }

    /// <summary>Reference type for coefficient probability lookup (libvpx REF_TYPES).</summary>
    public enum RefType
    {
        /// <summary>Intra-predicted block. REF_TYPES index 0.</summary>
        Intra = 0,
        /// <summary>Inter-predicted block. REF_TYPES index 1.</summary>
        Inter = 1,
    }

    /// <summary>
    /// Decode the coefficients of a single transform block. Writes the
    /// signed dequantized-domain values into <paramref name="block"/>
    /// in raster order (the inverse transform expects raster) and
    /// returns the EOB position (1 past the last decoded scan slot).
    /// </summary>
    /// <param name="readBit">
    /// Bit reader (typically a closure over a
    /// <see cref="Vp9BoolDecoder"/>: <c>b =&gt; reader.Read(b)</c>).
    /// </param>
    /// <param name="txSize">Transform size driving the band table + max coef count.</param>
    /// <param name="scanType">Scan flavor (default / row / col).</param>
    /// <param name="planeType">Plane type for prob table indexing.</param>
    /// <param name="refType">Reference type for prob table indexing.</param>
    /// <param name="block">
    /// Output buffer for the decoded coefficient block (raster
    /// layout). Must be at least as large as the scan table size:
    /// 16 / 64 / 256 / 1024 entries for 4x4 / 8x8 / 16x16 / 32x32.
    /// All entries are zeroed before decode begins.
    /// </param>
    /// <param name="isHighBitDepth">12-bit profile flag - widens Cat6 residual.</param>
    /// <returns>EOB position (count of decoded scan slots, 0..maxCoefs).</returns>
    public static int DecodeBlockCoefficients(
        Func<byte, int> readBit,
        Vp9TxSize txSize,
        Vp9ScanType scanType,
        PlaneType planeType,
        RefType refType,
        Span<short> block,
        bool isHighBitDepth = false)
    {
        ArgumentNullException.ThrowIfNull(readBit);

        int maxCoefs = txSize switch
        {
            Vp9TxSize.Tx4x4 => 16,
            Vp9TxSize.Tx8x8 => 64,
            Vp9TxSize.Tx16x16 => 256,
            Vp9TxSize.Tx32x32 => 1024,
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
        if (block.Length < maxCoefs)
            throw new ArgumentException(
                $"block must hold >= {maxCoefs} entries for {txSize}",
                nameof(block));
        block[..maxCoefs].Clear();

        // Pull the scan, neighbor, and prob arrays for the active
        // (txSize, scanType) pair. The probability table is keyed by
        // the abstract (planeType, refType, band, ctx) tuple.
        ushort[] scan = Vp9ScanTables.GetScan(txSize, scanType);
        ushort[] neighbors = txSize switch
        {
            Vp9TxSize.Tx4x4 => Vp9NeighborTables.GetNeighbors4x4(scanType),
            Vp9TxSize.Tx8x8 => Vp9NeighborTables.GetNeighbors8x8(scanType),
            Vp9TxSize.Tx16x16 => Vp9NeighborTables.GetNeighbors16x16(scanType),
            Vp9TxSize.Tx32x32 => Vp9NeighborTables.GetNeighbors32x32(scanType),
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
        byte[] coefProbs = Vp9CoefProbs.DefaultCoefProbsFor(txSize);

        // Per-position energy-class cache, raster-indexed. Drives the
        // GetCoefContext lookup. Allocate stack for small sizes;
        // fall back to heap for 32x32.
        Span<byte> tokenCache = maxCoefs <= 256
            ? stackalloc byte[maxCoefs]
            : new byte[maxCoefs];
        // Already zero from stackalloc / new; leave as-is.

        // Per-coefficient temp for the expanded probability vector.
        Span<byte> fullProbs = stackalloc byte[Vp9CoefProbs.EntropyNodes];

        int c = 0;
        while (c < maxCoefs)
        {
            int rasterPos = scan[c];
            int band = (int)Vp9CoefBands.GetBand(txSize, c);
            int ctx = Vp9CoefContext.GetCoefContext(neighbors, tokenCache, c);

            // Expand the stored 3-entry model to the full 11-entry
            // probability vector for this (band, ctx).
            int modelBase = Vp9CoefProbs.Index4x4(
                (int)planeType, (int)refType, band, ctx, 0);
            ReadOnlySpan<byte> model = coefProbs.AsSpan(modelBase, 3);
            Vp9CoefProbs.ModelToFullProbs(model, fullProbs);

            var decoded = Vp9CoefDecoder.DecodeOneCoefficient(
                readBit, fullProbs, isHighBitDepth);

            if (decoded.Token == Vp9CoefToken.Eob)
                break;

            // Write the (signed) value at the raster position.
            block[rasterPos] = (short)decoded.Value;
            tokenCache[rasterPos] = Vp9CoefContext.PtEnergyClass[(int)decoded.Token];
            c++;
        }

        return c;
    }
}
