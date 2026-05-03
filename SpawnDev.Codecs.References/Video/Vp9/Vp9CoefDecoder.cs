// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Single-coefficient decoder. Combines slice 145's constrained tree
// walk, slice 146's category magnitude decode, and the per-token
// sign bit into the canonical VP9 entropy-decode primitive: given
// an 11-element full probability vector and a bit reader, decode
// the next coefficient and return its (token, signed value).
//
// libvpx reference: vp9/decoder/vp9_detokenize.c (decode_coefs).
//
// Probability vector layout (libvpx ENTROPY_NODES enum, after
// ModelToFullProbs has expanded the stored 3-entry model):
//   prob[0]  EOB? branch
//   prob[1]  ZERO? branch
//   prob[2]  ONE? branch  (also the model PIVOT_NODE for pareto8)
//   prob[3]  LOW_VAL constrained tree root
//   prob[4]  TWO        prob[5]  THREE
//   prob[6]  HIGH_LOW   prob[7]  CAT_ONE
//   prob[8]  CAT_THREEFOUR
//   prob[9]  CAT_THREE  prob[10] CAT_FIVE
//
// Decode flow:
//   1. read EOB?  -> if 0 (the unlikely branch), return (Eob, 0)
//   2. read ZERO? -> if 0, return (Zero, 0)
//   3. read ONE?  -> if 0, the magnitude is 1; jump to sign step
//      else walk the constrained tree (slice 145) -> Vp9CoefToken in
//      Two..Category6
//   4. for category tokens, decode N residual bits to recover the
//      magnitude (slice 146)
//   5. read the sign bit at flat probability 128
//   6. return (token, signed magnitude)
//
// Note: libvpx convention is that EOB and ZERO branches read the
// LOW probability (i.e. !vpx_read returns true for EOB / ZERO).
// This port matches that bit-exactly.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Stateless VP9 coefficient decoder. The caller is responsible for
/// computing the per-coefficient context (from neighbor tokens) and
/// expanding the stored model probabilities to the full 11-element
/// vector via <see cref="Vp9CoefProbs.ModelToFullProbs"/> before
/// invoking this helper.
/// </summary>
public static class Vp9CoefDecoder
{
    /// <summary>Result of a single-coefficient decode.</summary>
    /// <param name="Token">
    /// The decoded token (<see cref="Vp9CoefToken.Eob"/>,
    /// <see cref="Vp9CoefToken.Zero"/>, <see cref="Vp9CoefToken.One"/>,
    /// or one of the magnitude tokens).
    /// </param>
    /// <param name="Value">
    /// Signed coefficient value with sign bit applied.
    /// Zero for Eob and Zero tokens.
    /// </param>
    public readonly record struct DecodedCoefficient(Vp9CoefToken Token, int Value);

    /// <summary>
    /// Decode one coefficient using the supplied bit reader and the
    /// 11-element full probability vector. Mirrors libvpx
    /// <c>decode_coefs</c> for a single scan position.
    /// </summary>
    /// <param name="readBit">
    /// Bit reader (typically a closure over a
    /// <see cref="Vp9BoolDecoder"/>: <c>b =&gt; reader.Read(b)</c>).
    /// </param>
    /// <param name="fullProbs">
    /// 11-entry probability vector produced by
    /// <see cref="Vp9CoefProbs.ModelToFullProbs"/>.
    /// </param>
    /// <param name="isHighBitDepth">
    /// True for 12-bit profiles - changes the Cat6 residual width
    /// from 14 to 18 bits.
    /// </param>
    public static DecodedCoefficient DecodeOneCoefficient(
        Func<byte, int> readBit,
        ReadOnlySpan<byte> fullProbs,
        bool isHighBitDepth = false)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (fullProbs.Length < Vp9CoefProbs.EntropyNodes)
            throw new ArgumentException(
                $"fullProbs must hold at least {Vp9CoefProbs.EntropyNodes} entries",
                nameof(fullProbs));

        // libvpx convention: !vpx_read(prob[0]) is the EOB branch.
        // Equivalently: reading 0 means "yes, this is EOB."
        if (readBit(fullProbs[0]) == 0)
            return new DecodedCoefficient(Vp9CoefToken.Eob, 0);

        if (readBit(fullProbs[1]) == 0)
            return new DecodedCoefficient(Vp9CoefToken.Zero, 0);

        Vp9CoefToken token;
        int magnitude;

        // ONE? branch.
        if (readBit(fullProbs[2]) == 0)
        {
            token = Vp9CoefToken.One;
            magnitude = 1;
        }
        else
        {
            // Walk the constrained sub-tree (slice 145) using
            // probs[3..10] as the per-node probabilities.
            token = Vp9CoefTrees.DecodeConToken(readBit, fullProbs.Slice(3, 8));
            magnitude = token switch
            {
                Vp9CoefToken.Two   => 2,
                Vp9CoefToken.Three => 3,
                Vp9CoefToken.Four  => 4,
                Vp9CoefToken.Category1
                    or Vp9CoefToken.Category2
                    or Vp9CoefToken.Category3
                    or Vp9CoefToken.Category4
                    or Vp9CoefToken.Category5
                    or Vp9CoefToken.Category6
                    => Vp9CoefProbs.DecodeCategoryMagnitude(readBit, token, isHighBitDepth),
                _ => throw new InvalidDataException(
                    $"DecodeConToken returned unexpected token {token}"),
            };
        }

        // Sign bit at flat probability 128.
        int sign = readBit(128);
        int value = sign != 0 ? -magnitude : magnitude;
        return new DecodedCoefficient(token, value);
    }
}
