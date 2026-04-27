// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 per-block coefficient decoder. Structural port of libvpx
// vp8/decoder/detokenize.c GetCoeffs() function. RFC 6386 sec 13.
//
// Decodes one 4x4 transform block's coefficients into a 16-entry int16
// array (raster order via the kZigzag table). Returns the position of
// the last non-zero coefficient + 1 (the EOB index, 0 = all zero).
//
// This is the hot inner loop of VP8 decode - called 24 times per
// macroblock (16 Y4 blocks + 8 chroma + optional Y2). The macroblock
// driver (vp8_decode_mb_tokens) walks above/left entropy contexts and
// dispatches to this primitive.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 per-block (4x4) coefficient decoder.</summary>
public static class Vp8CoefBlockDecoder
{
    /// <summary>Scan-position-to-band mapping (libvpx kBands).</summary>
    public static readonly byte[] CoefBands = new byte[]
    {
        0, 1, 2, 3, 6, 4, 5, 6, 6,
        6, 6, 6, 6, 6, 6, 7,
        0, // sentinel - probs lookup at coeff 16 reads kBands[16] which maps to band 0 unused.
    };

    /// <summary>4x4 zigzag scan order (libvpx kZigzag).</summary>
    public static readonly byte[] ZigzagScan = new byte[]
    {
        0, 1,  4,  8,  5, 2,  3,  6,
        9, 12, 13, 10, 7, 11, 14, 15,
    };

    /// <summary>Cat3 extra-bit probabilities (libvpx kCat3 without trailing 0).</summary>
    public static readonly byte[] Cat3Probs = new byte[] { 173, 148, 140 };
    /// <summary>Cat4 extra-bit probabilities (libvpx kCat4 without trailing 0).</summary>
    public static readonly byte[] Cat4Probs = new byte[] { 176, 155, 140, 135 };
    /// <summary>Cat5 extra-bit probabilities (libvpx kCat5 without trailing 0).</summary>
    public static readonly byte[] Cat5Probs = new byte[] { 180, 157, 141, 134, 130 };
    /// <summary>Cat6 extra-bit probabilities (libvpx kCat6 without trailing 0).</summary>
    public static readonly byte[] Cat6Probs = new byte[]
    {
        254, 254, 243, 230, 196, 177,
        153, 140, 133, 130, 129,
    };

    /// <summary>Decode one block. Mirrors libvpx GetCoeffs.</summary>
    /// <param name="reader">VP8 bool decoder positioned at the block's coefficient bits.</param>
    /// <param name="probs">
    /// 4D coefficient probability table (typically a slice into the frame's
    /// fc.coef_probs[block_type] indexed [band][prev_ctx][node]).
    /// </param>
    /// <param name="ctx">Initial previous-coefficient context (0..2).</param>
    /// <param name="firstCoef">
    /// Starting coefficient index. 0 for Y2 / Y_no_Y2 / UV blocks; 1 for
    /// Y_after_Y2 (since DC was decoded by the Y2 block).
    /// </param>
    /// <param name="output">16-entry output buffer in raster order (zigzag-mapped from scan).</param>
    /// <returns>Position of the last non-zero coefficient + 1 (0..16).</returns>
    public static int Decode(
        Vp8BoolDecoder reader,
        byte[,,] probs,
        int ctx,
        int firstCoef,
        Span<short> output)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(probs);
        if (output.Length < 16) throw new ArgumentException("output must hold >= 16 entries", nameof(output));
        if (probs.GetLength(0) < 8 || probs.GetLength(1) < 3 || probs.GetLength(2) < 11)
            throw new ArgumentException("probs must be at least [8 bands][3 ctx][11 nodes]", nameof(probs));

        output.Clear();

        int n = firstCoef;
        // EOB-style first token: if 0, the entire block is zero.
        if (reader.DecodeBool(probs[CoefBands[n], ctx, 0]) == 0)
            return 0;

        while (true)
        {
            n++;
            int band = CoefBands[n];
            // Probability vector for the current (band, ctx) state.
            // probs is rank-3: [band, ctx, node]
            if (reader.DecodeBool(probs[band, ctx, 1]) == 0)
            {
                // Zero coefficient. Move ctx to 0 for next position.
                ctx = 0;
            }
            else
            {
                // Non-zero coefficient.
                int v;
                if (reader.DecodeBool(probs[band, ctx, 2]) == 0)
                {
                    v = 1;
                    ctx = 1; // ctx after a One-token coefficient.
                }
                else
                {
                    if (reader.DecodeBool(probs[band, ctx, 3]) == 0)
                    {
                        if (reader.DecodeBool(probs[band, ctx, 4]) == 0)
                            v = 2;
                        else
                            v = 3 + reader.DecodeBool(probs[band, ctx, 5]);
                    }
                    else
                    {
                        if (reader.DecodeBool(probs[band, ctx, 6]) == 0)
                        {
                            if (reader.DecodeBool(probs[band, ctx, 7]) == 0)
                                v = 5 + reader.DecodeBool(159);
                            else
                            {
                                v = 7 + 2 * reader.DecodeBool(165);
                                v += reader.DecodeBool(145);
                            }
                        }
                        else
                        {
                            int bit1 = reader.DecodeBool(probs[band, ctx, 8]);
                            int bit0 = reader.DecodeBool(probs[band, ctx, 9 + bit1]);
                            int cat = 2 * bit1 + bit0;
                            byte[] tab = cat switch
                            {
                                0 => Cat3Probs,
                                1 => Cat4Probs,
                                2 => Cat5Probs,
                                3 => Cat6Probs,
                                _ => throw new InvalidDataException(),
                            };
                            v = 0;
                            for (int t = 0; t < tab.Length; t++)
                                v = v + v + reader.DecodeBool(tab[t]);
                            v += 3 + (8 << cat);
                        }
                    }
                    ctx = 2; // ctx after a Two-or-greater token.
                }

                int j = ZigzagScan[n - 1];
                output[j] = (short)DecodeSigned(reader, v);

                if (n == 16 || reader.DecodeBool(probs[CoefBands[n], ctx, 0]) == 0)
                    return n; // EOB
            }
            if (n == 16) return 16;
        }
    }

    /// <summary>
    /// Decode the sign bit at flat probability 128 and apply to <paramref name="value"/>.
    /// libvpx GetSigned uses an inline fast path that does the same as
    /// vp8dx_decode_bool(0x80) followed by negation; we use the regular
    /// bool decode for clarity (semantics are identical).
    /// </summary>
    private static int DecodeSigned(Vp8BoolDecoder reader, int value)
    {
        return reader.DecodeBool(128) != 0 ? -value : value;
    }
}
