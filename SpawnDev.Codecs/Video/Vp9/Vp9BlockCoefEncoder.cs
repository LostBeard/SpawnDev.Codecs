// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 per-block coefficient encoder. The bit-exact mirror of
// Vp9BlockCoefDecoder: walks the same scan, drives the same
// (band, ctx) lookup, and emits the bool-coded bits the decoder
// will read back. Critically the encoder mirrors the decoder's
// inner ZERO loop semantics: EOB is only re-read at the start of
// each "outer iteration" (after the previous non-zero coefficient
// is emitted), NOT between consecutive ZERO tokens. The bitstream
// would diverge from libvpx if either side re-read EOB inside the
// ZERO loop.
//
// libvpx reference: vp9/encoder/vp9_tokenize.c (tokenize_b) plus
// vp9/encoder/vp9_bitstream.c (write_coef_tokens). This port fuses
// both into a single emit pass for clarity; tokenization and bit
// emission are merged because the entire encoder is ahead-of-time
// (no rate-distortion search loop yet).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 block-level coefficient encoder (mirror of <see cref="Vp9BlockCoefDecoder"/>).</summary>
public static class Vp9BlockCoefEncoder
{
    /// <summary>
    /// Bit-emit callback. Takes a probability and a bit value (0/1)
    /// and writes the encoded bit to the underlying bool encoder.
    /// </summary>
    public delegate void WriteBit(byte prob, int bit);

    /// <summary>
    /// Emit the coefficients of a single transform block. Mirrors
    /// <see cref="Vp9BlockCoefDecoder.DecodeBlockCoefficients"/>
    /// bit-for-bit (i.e. the encoder's output is exactly the bit
    /// sequence the decoder will consume).
    /// </summary>
    /// <param name="writeBit">Bit emit callback over a Vp9 bool encoder.</param>
    /// <param name="txSize">Transform size driving the band table + max coef count.</param>
    /// <param name="scanType">Scan flavor (default / row / col).</param>
    /// <param name="planeType">Plane type for prob table indexing.</param>
    /// <param name="refType">Reference type for prob table indexing.</param>
    /// <param name="block">Signed coefficient block in raster layout (output of the forward transform + quantize).</param>
    /// <param name="isHighBitDepth">12-bit profile flag - widens Cat6 residual.</param>
    /// <param name="coefProbs">Per-frame prob table (defaults to libvpx static defaults if null).</param>
    /// <param name="initialCtx">
    /// Per-plane entropy context for scan position 0. Mirrors
    /// <see cref="Vp9BlockCoefDecoder.DecodeBlockCoefficients"/>'s
    /// <c>initialCtx</c>: libvpx
    /// <c>combine_entropy_contexts(a, b) = (a != 0) + (b != 0)</c>
    /// applied to the per-plane above + left ENTROPY_CONTEXT byte arrays.
    /// Range 0..2. The encoder must derive this from the per-plane
    /// above/left coef context bytes that get set to <c>(eob &gt; 0)</c>
    /// after each tx-block emit. Defaults to 0 for callers that do not
    /// track entropy context (back-compat with isolated round-trip
    /// tests where every block stands alone).
    /// </param>
    /// <returns>EOB position written to the bitstream (count of emitted scan slots, 0..maxCoefs).</returns>
    public static int EncodeBlockCoefficients(
        WriteBit writeBit,
        Vp9TxSize txSize,
        Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        ReadOnlySpan<short> block,
        bool isHighBitDepth = false,
        byte[]? coefProbs = null,
        int initialCtx = 0)
    {
        ArgumentNullException.ThrowIfNull(writeBit);

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

        ushort[] scan = Vp9ScanTables.GetScan(txSize, scanType);
        ushort[] neighbors = txSize switch
        {
            Vp9TxSize.Tx4x4 => Vp9NeighborTables.GetNeighbors4x4(scanType),
            Vp9TxSize.Tx8x8 => Vp9NeighborTables.GetNeighbors8x8(scanType),
            Vp9TxSize.Tx16x16 => Vp9NeighborTables.GetNeighbors16x16(scanType),
            Vp9TxSize.Tx32x32 => Vp9NeighborTables.GetNeighbors32x32(scanType),
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
        coefProbs ??= Vp9CoefProbs.DefaultCoefProbsFor(txSize);

        // EOB position: one past the last scan slot with a non-zero coef.
        // 0 means "block is entirely zero" - the decoder will read the
        // very first EOB bit as 0 and return immediately.
        int eob = 0;
        for (int i = maxCoefs - 1; i >= 0; i--)
        {
            if (block[scan[i]] != 0) { eob = i + 1; break; }
        }

        Span<byte> tokenCache = maxCoefs <= 256
            ? stackalloc byte[maxCoefs]
            : new byte[maxCoefs];
        Span<byte> fullProbs = stackalloc byte[Vp9CoefProbs.EntropyNodes];

        int c = 0;
        // First scan position uses initialCtx (per-plane entropy context);
        // subsequent positions use GetCoefContext from tokenCache. Mirrors
        // Vp9BlockCoefDecoder bit-for-bit.
        bool firstIter = true;
        while (c < maxCoefs)
        {
            ComputeProbs(coefProbs, planeType, refType, txSize, neighbors, tokenCache, c,
                fullProbs, firstIter ? initialCtx : -1);
            firstIter = false;

            if (c == eob)
            {
                writeBit(fullProbs[0], 0);  // EOB (decoder: read==0 -> EOB)
                return eob;
            }
            writeBit(fullProbs[0], 1);      // !EOB

            // Inner ZERO loop - decoder mirrors with `while (readBit(probs[1]) == 0)`.
            // No re-read of EOB between consecutive ZEROs.
            while (block[scan[c]] == 0)
            {
                writeBit(fullProbs[1], 0);  // ZERO
                c++;
                if (c >= maxCoefs) return eob;  // matches decoder line 152
                ComputeProbs(coefProbs, planeType, refType, txSize, neighbors, tokenCache, c,
                    fullProbs, -1);
            }

            // Non-zero token: emit !ZERO, then magnitude tree, then sign.
            writeBit(fullProbs[1], 1);      // !ZERO

            int value = block[scan[c]];
            int magnitude = value < 0 ? -value : value;
            Vp9CoefToken token;

            if (magnitude == 1)
            {
                writeBit(fullProbs[2], 0);  // ONE
                token = Vp9CoefToken.One;
            }
            else
            {
                writeBit(fullProbs[2], 1);  // !ONE
                token = MagnitudeToToken(magnitude, isHighBitDepth);
                EncodeConToken(writeBit, token, fullProbs.Slice(3, 8));
                if (token >= Vp9CoefToken.Category1 && token <= Vp9CoefToken.Category6)
                {
                    EncodeCategoryMagnitude(writeBit, token, magnitude, isHighBitDepth);
                }
            }

            writeBit(128, value < 0 ? 1 : 0);   // sign

            tokenCache[scan[c]] = Vp9CoefContext.PtEnergyClass[(int)token];
            c++;
        }
        return eob;
    }

    private static void ComputeProbs(
        byte[] coefProbs,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        Vp9TxSize txSize,
        ushort[] neighbors,
        ReadOnlySpan<byte> tokenCache,
        int c,
        Span<byte> fullProbs,
        int forcedCtx = -1)
    {
        int band = (int)Vp9CoefBands.GetBand(txSize, c);
        // forcedCtx >= 0 lets the caller override the per-coefficient
        // context (used at scan position 0 to inject the per-plane
        // entropy context per libvpx's vp9_decode_block_tokens). Pass
        // -1 to compute the context from tokenCache as usual.
        int ctx = forcedCtx >= 0
            ? forcedCtx
            : Vp9CoefContext.GetCoefContext(neighbors, tokenCache, c);
        int modelBase = Vp9CoefProbs.Index4x4((int)planeType, (int)refType, band, ctx, 0);
        ReadOnlySpan<byte> model = coefProbs.AsSpan(modelBase, 3);
        Vp9CoefProbs.ModelToFullProbs(model, fullProbs);
    }

    private static Vp9CoefToken MagnitudeToToken(int magnitude, bool isHighBitDepth)
    {
        // Match Vp9CoefProbs.CatMinVal ranges exactly.
        if (magnitude == 2) return Vp9CoefToken.Two;
        if (magnitude == 3) return Vp9CoefToken.Three;
        if (magnitude == 4) return Vp9CoefToken.Four;
        if (magnitude <= 6) return Vp9CoefToken.Category1;
        if (magnitude <= 10) return Vp9CoefToken.Category2;
        if (magnitude <= 18) return Vp9CoefToken.Category3;
        if (magnitude <= 34) return Vp9CoefToken.Category4;
        if (magnitude <= 66) return Vp9CoefToken.Category5;
        // Cat6 caps at 67 + (1 << 14) - 1 for 8-bit (=16450) and
        // 67 + (1 << 18) - 1 for 12-bit. Anything larger means the
        // forward transform produced an out-of-range coefficient that
        // the bitstream cannot represent.
        int max = isHighBitDepth ? (Vp9CoefProbs.CatMinVal.Cat6 + (1 << 18) - 1)
                                 : (Vp9CoefProbs.CatMinVal.Cat6 + (1 << 14) - 1);
        if (magnitude > max)
            throw new ArgumentOutOfRangeException(
                nameof(magnitude),
                $"magnitude {magnitude} exceeds VP9 Cat6 range (max {max})");
        return Vp9CoefToken.Category6;
    }

    /// <summary>
    /// Walk <see cref="Vp9CoefTrees.CoefConTree"/> from the root down
    /// to the target leaf, emitting one bit per internal node visited.
    /// Mirrors the decoder's <see cref="Vp9CoefTrees.DecodeConToken"/>
    /// tree walk so the produced bit sequence is exactly what the
    /// decoder will read.
    /// </summary>
    internal static void EncodeConToken(WriteBit writeBit, Vp9CoefToken token, ReadOnlySpan<byte> probs)
    {
        if (probs.Length < 8)
            throw new ArgumentException("probs must hold >= 8 entries for the constrained tree", nameof(probs));

        sbyte[] tree = Vp9CoefTrees.CoefConTree;
        sbyte target = (sbyte)(-(int)token);
        int i = 0;
        while (true)
        {
            // Pick left vs right by checking which subtree contains the leaf.
            int leftIndex = i;
            int rightIndex = i + 1;
            int bit = SubtreeContains(tree, leftIndex, target) ? 0 : 1;
            writeBit(probs[i >> 1], bit);
            int next = tree[i + bit];
            if (next <= 0)
            {
                if (next != target)
                    throw new InvalidOperationException(
                        $"EncodeConToken landed on token {-next} but expected {(int)token}");
                return;
            }
            i = next;
        }
    }

    /// <summary>True if the subtree rooted at <paramref name="rootIndex"/> contains the target leaf.</summary>
    private static bool SubtreeContains(sbyte[] tree, int rootIndex, sbyte target)
    {
        sbyte v = tree[rootIndex];
        if (v <= 0) return v == target;
        // v is a child node base index; recurse into both halves.
        return SubtreeContains(tree, v, target) || SubtreeContains(tree, (sbyte)(v + 1), target);
    }

    /// <summary>
    /// Emit the residual bits for a Cat1..Cat6 token. Mirrors
    /// <see cref="Vp9CoefProbs.DecodeCategoryMagnitude"/> bit-for-bit:
    /// the decoder reads N bits MSB-first against the same probability
    /// table; the encoder must write them MSB-first too.
    /// </summary>
    internal static void EncodeCategoryMagnitude(WriteBit writeBit, Vp9CoefToken cat, int magnitude, bool isHighBitDepth)
    {
        switch (cat)
        {
            case Vp9CoefToken.Category1:
                WriteResidualMsbFirst(writeBit, Vp9CoefProbs.Cat1Prob, magnitude - Vp9CoefProbs.CatMinVal.Cat1);
                break;
            case Vp9CoefToken.Category2:
                WriteResidualMsbFirst(writeBit, Vp9CoefProbs.Cat2Prob, magnitude - Vp9CoefProbs.CatMinVal.Cat2);
                break;
            case Vp9CoefToken.Category3:
                WriteResidualMsbFirst(writeBit, Vp9CoefProbs.Cat3Prob, magnitude - Vp9CoefProbs.CatMinVal.Cat3);
                break;
            case Vp9CoefToken.Category4:
                WriteResidualMsbFirst(writeBit, Vp9CoefProbs.Cat4Prob, magnitude - Vp9CoefProbs.CatMinVal.Cat4);
                break;
            case Vp9CoefToken.Category5:
                WriteResidualMsbFirst(writeBit, Vp9CoefProbs.Cat5Prob, magnitude - Vp9CoefProbs.CatMinVal.Cat5);
                break;
            case Vp9CoefToken.Category6:
                WriteResidualMsbFirst(
                    writeBit,
                    isHighBitDepth ? Vp9CoefProbs.Cat6ProbHigh12 : Vp9CoefProbs.Cat6Prob,
                    magnitude - Vp9CoefProbs.CatMinVal.Cat6);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(cat),
                    $"Token {cat} is not a category token");
        }
    }

    private static void WriteResidualMsbFirst(WriteBit writeBit, byte[] probs, int value)
    {
        for (int i = 0; i < probs.Length; i++)
        {
            int bit = (value >> (probs.Length - 1 - i)) & 1;
            writeBit(probs[i], bit);
        }
    }
}
