// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector component reader. Composes the per-component
// probability sub-tables (sign, class tree, offset bits, fp tree,
// hp) into a single signed integer magnitude. Mirror of libvpx
// vp9/decoder/vp9_decodemv.c read_mv_component.
//
// Magnitude reconstruction (libvpx layout):
//   sign      = read(probs.Sign)
//   class     = vp9_mv_class_tree(probs.Classes)
//   if class == Class0:
//     d   = read(probs.Class0)         // 1 bit (CLASS0_SIZE = 2)
//     mag = 0
//   else:
//     n   = class + CLASS0_BITS         // CLASS0_BITS = 1
//     d   = sum_{i=0..n-1} read(probs.Bits[i]) << i
//     mag = CLASS0_SIZE << (class + 2)
//   fr  = vp9_mv_fp_tree(class0 ? probs.Class0Fp[d] : probs.Fp)
//   hp  = useHp ? read(class0 ? probs.Class0Hp : probs.Hp) : 1
//   mag += ((d << 3) | (fr << 1) | hp) + 1
//   return sign ? -mag : mag

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector component reader.</summary>
public static class Vp9MvComponentReader
{
    /// <summary>libvpx <c>CLASS0_BITS</c>.</summary>
    public const int Class0Bits = 1;

    /// <summary>
    /// Read a full MV component (sign + class + offset + fp + hp)
    /// from the supplied bool decoder. Returns the signed magnitude.
    /// </summary>
    /// <param name="reader">Compressed-frame arithmetic-coded reader.</param>
    /// <param name="probs">Per-component MV probabilities.</param>
    /// <param name="useHp">
    /// True when 1/8-pel high precision is allowed for this frame.
    /// Maps to libvpx <c>usehp = allow_high_precision_mv &amp;&amp; use_mv_hp(...)</c>.
    /// </param>
    public static int ReadComponent(Vp9BoolDecoder reader, Vp9MvComponentProbs probs, bool useHp)
    {
        ArgumentNullException.ThrowIfNull(reader);
        return ReadComponent(p => reader.Read(p), probs, useHp);
    }

    /// <summary>
    /// Pure-function variant of
    /// <see cref="ReadComponent(Vp9BoolDecoder, Vp9MvComponentProbs, bool)"/>
    /// that takes a bit-reader delegate. Production callers use the
    /// <see cref="Vp9BoolDecoder"/> overload; this one supports
    /// unit-test injection of deterministic bit sequences.
    /// </summary>
    public static int ReadComponent(Func<byte, int> readBit, Vp9MvComponentProbs probs, bool useHp)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        ArgumentNullException.ThrowIfNull(probs);

        int sign = readBit(probs.Sign);
        var mvClass = Vp9MvClassTree.Decode(readBit, probs.Classes);
        bool isClass0 = mvClass == Vp9MvClassType.Class0;

        int d;
        int mag;
        if (isClass0)
        {
            d = readBit(probs.Class0);
            mag = 0;
        }
        else
        {
            int n = (int)mvClass + Class0Bits;
            d = 0;
            for (int i = 0; i < n; i++)
                d |= readBit(probs.Bits[i]) << i;
            mag = Vp9MvComponentProbs.Class0Size << ((int)mvClass + 2);
        }

        ReadOnlySpan<byte> fpProbs;
        if (isClass0)
        {
            // Class0Fp is byte[CLASS0_SIZE, MV_FP_SIZE - 1] - copy the
            // d-th row out into a small stackalloc-style array so we
            // can hand it as a span to the tree decoder.
            byte[] fpRow = new byte[Vp9MvComponentProbs.MvFpSize - 1];
            for (int j = 0; j < fpRow.Length; j++)
                fpRow[j] = probs.Class0Fp[d, j];
            fpProbs = fpRow;
        }
        else
        {
            fpProbs = probs.Fp;
        }
        var fr = Vp9MvFpTree.Decode(readBit, fpProbs);

        int hp = useHp ? readBit(isClass0 ? probs.Class0Hp : probs.Hp) : 1;

        mag += ((d << 3) | ((int)fr << 1) | hp) + 1;
        return sign != 0 ? -mag : mag;
    }
}
