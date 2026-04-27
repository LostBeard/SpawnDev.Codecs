// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 per-block coefficient encoder. Inverse of Vp8CoefBlockDecoder.
// Walks 16 coefficients in zigzag scan order, classifies each into a
// VP8 token, and emits the binary tree path + extra bits + sign through
// the supplied Vp8BoolEncoder.
//
// libvpx reference: vp8/encoder/tokenize.c (tokenize2nd_order_b /
// tokenize1st_order_b) + vp8_short_walsh4x4_c.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 per-block (4x4) coefficient encoder. Inverse of Vp8CoefBlockDecoder.</summary>
public static class Vp8CoefBlockEncoder
{
    /// <summary>
    /// Encode a 16-element coefficient block. The probabilities at each
    /// scan position are looked up via the same (band, ctx) pair the
    /// decoder will use; <paramref name="ctx"/> seeds the initial
    /// previous-coefficient context (0..2).
    /// </summary>
    /// <param name="writer">VP8 bool encoder positioned at the block.</param>
    /// <param name="probs">3D coef probability table [band][prev_ctx][node].</param>
    /// <param name="ctx">Initial previous-coefficient context (0..2).</param>
    /// <param name="firstCoef">
    /// 0 for Y2 / Y_no_Y2 / UV blocks; 1 for Y_after_Y2 (since DC is
    /// covered by Y2). Mirrors the decoder's <c>firstCoef</c> argument.
    /// </param>
    /// <param name="coefs">16-entry input coefficient block (raster order).</param>
    /// <returns>EOB position (last non-zero scan slot + 1; 0 if all zero).</returns>
    public static int Encode(
        Vp8BoolEncoder writer,
        byte[,,] probs,
        int ctx,
        int firstCoef,
        ReadOnlySpan<short> coefs)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(probs);
        if (coefs.Length < 16) throw new ArgumentException("coefs must have 16 entries", nameof(coefs));

        // Find the EOB position by scanning from the end to find the last
        // non-zero coefficient in zigzag order.
        var zigzag = Vp8CoefBlockDecoder.ZigzagScan;
        var bands = Vp8CoefBlockDecoder.CoefBands;
        int eob = 0;
        for (int n = 16 - 1; n >= firstCoef; n--)
        {
            int raster = zigzag[n - firstCoef];
            if (coefs[raster] != 0) { eob = n + 1; break; }
        }
        // Special case: all-zero block from firstCoef onward.
        // The decoder reads ONE bit at probs[bands[firstCoef], ctx, 0]:
        //   0 means "this whole block is empty (EOB)"
        //   1 means "there is at least one coefficient"
        // So for an all-zero block we emit the single 0 bit.
        if (eob == 0)
        {
            writer.EncodeBool(0, probs[bands[firstCoef], ctx, 0]);
            return 0;
        }

        // First non-zero token at firstCoef: emit "block is non-empty" = 1.
        writer.EncodeBool(1, probs[bands[firstCoef], ctx, 0]);

        // Now walk scan positions [firstCoef..15], encoding the per-position
        // token chain (Zero | One | Two..CAT6 | EOB-after-non-zero).
        int n2 = firstCoef;
        while (n2 < 16)
        {
            n2++;
            if (n2 > eob)
            {
                // We've passed the last non-zero coef. Nothing more to emit.
                return eob;
            }

            int band = bands[n2];
            int raster = zigzag[n2 - 1 - firstCoef];
            // Wait - the decoder pulls scan position n THEN advances it.
            // Re-check: the decoder loop is "++n; band = kBands[n]; check ZERO at probs[band, ctx, 1]".
            // The raster index for the just-incremented n is kZigzag[n - 1].
            // So when n2 = firstCoef + 1, the raster index is kZigzag[firstCoef].
            // Let me use n2 - 1 - firstCoef shifted appropriately... actually
            // the decoder uses kZigzag[n - 1] where n is post-increment, so
            // for our n2 (already incremented), raster is kZigzag[n2 - 1].
            raster = zigzag[n2 - 1];
            int v = coefs[raster];

            if (v == 0)
            {
                // Emit ZERO at probs[band, ctx, 1] = 0.
                writer.EncodeBool(0, probs[band, ctx, 1]);
                ctx = 0;
                continue;
            }

            // Non-zero: emit ZERO bit = 1.
            writer.EncodeBool(1, probs[band, ctx, 1]);
            int absV = v < 0 ? -v : v;

            // Emit ONE bit: 0 = exactly 1, 1 = >= 2.
            if (absV == 1)
            {
                writer.EncodeBool(0, probs[band, ctx, 2]);
                ctx = 1;
            }
            else
            {
                writer.EncodeBool(1, probs[band, ctx, 2]);
                EncodeAtLeastTwo(writer, probs, band, ctx, absV);
                ctx = 2;
            }

            // Emit sign bit (flat probability 128).
            writer.EncodeBool(v < 0 ? 1 : 0, 128);

            // Now decide whether this was the last non-zero coef (EOB next).
            // Decoder reads "EOB" at probs[bands[n2], ctx_after, 0]: 0 = EOB.
            // EOB only emitted at scan positions n2 < 16.
            if (n2 < 16)
            {
                if (n2 == eob)
                {
                    // No more non-zero coefs: emit EOB = 0.
                    writer.EncodeBool(0, probs[bands[n2], ctx, 0]);
                    return eob;
                }
                else
                {
                    // More non-zero coefs to come: emit EOB = 1.
                    writer.EncodeBool(1, probs[bands[n2], ctx, 0]);
                }
            }
        }

        return eob;
    }

    private static void EncodeAtLeastTwo(
        Vp8BoolEncoder writer, byte[,,] probs, int band, int ctx, int absV)
    {
        // Walk the constrained tree in reverse to emit the bits the
        // decoder will read.
        if (absV == 2)
        {
            writer.EncodeBool(0, probs[band, ctx, 3]); // low_val
            writer.EncodeBool(0, probs[band, ctx, 4]); // == 2
        }
        else if (absV == 3)
        {
            writer.EncodeBool(0, probs[band, ctx, 3]);
            writer.EncodeBool(1, probs[band, ctx, 4]); // 3 or 4
            writer.EncodeBool(0, probs[band, ctx, 5]); // 3
        }
        else if (absV == 4)
        {
            writer.EncodeBool(0, probs[band, ctx, 3]);
            writer.EncodeBool(1, probs[band, ctx, 4]);
            writer.EncodeBool(1, probs[band, ctx, 5]); // 4
        }
        else
        {
            // High-low: absV >= 5
            writer.EncodeBool(1, probs[band, ctx, 3]); // high_val
            if (absV >= 5 && absV <= 10)
            {
                writer.EncodeBool(0, probs[band, ctx, 6]); // CAT1 or CAT2
                if (absV >= 5 && absV <= 6)
                {
                    writer.EncodeBool(0, probs[band, ctx, 7]); // CAT1
                    // CAT1: 5..6, 1 extra bit
                    writer.EncodeBool(absV - 5, 159);
                }
                else // 7..10
                {
                    writer.EncodeBool(1, probs[band, ctx, 7]); // CAT2
                    int delta = absV - 7;
                    writer.EncodeBool((delta >> 1) & 1, 165);
                    writer.EncodeBool(delta & 1, 145);
                }
            }
            else
            {
                writer.EncodeBool(1, probs[band, ctx, 6]); // CAT3..CAT6
                int cat;
                if (absV <= 18) cat = 0;       // CAT3 (11..18)
                else if (absV <= 34) cat = 1;  // CAT4 (19..34)
                else if (absV <= 66) cat = 2;  // CAT5 (35..66)
                else cat = 3;                  // CAT6 (67+)

                int bit1 = (cat >> 1) & 1;
                int bit0 = cat & 1;
                writer.EncodeBool(bit1, probs[band, ctx, 8]);
                writer.EncodeBool(bit0, probs[band, ctx, 9 + bit1]);

                // Emit category extra bits.
                byte[] tab; int minVal;
                switch (cat)
                {
                    case 0: tab = Vp8CoefBlockDecoder.Cat3Probs; minVal = 11; break;
                    case 1: tab = Vp8CoefBlockDecoder.Cat4Probs; minVal = 19; break;
                    case 2: tab = Vp8CoefBlockDecoder.Cat5Probs; minVal = 35; break;
                    case 3: tab = Vp8CoefBlockDecoder.Cat6Probs; minVal = 67; break;
                    default: throw new InvalidOperationException();
                }
                int extra = absV - minVal;
                int width = tab.Length;
                // Decoder reads bits MSB-first, building v = v + v + bit.
                for (int i = 0; i < width; i++)
                {
                    int bitVal = (extra >> (width - 1 - i)) & 1;
                    writer.EncodeBool(bitVal, tab[i]);
                }
            }
        }
    }
}
