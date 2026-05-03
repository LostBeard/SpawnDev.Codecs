// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 motion vector fractional-part tree. The 4-leaf tree picks
// the quarter-pel fractional offset (Fp0..Fp3) for a motion vector
// component. libvpx reference: vp9/common/vp9_entropymv.c
// vp9_mv_fp_tree.
//
// Tree shape (libvpx layout):
//   ROOT  : -Fp0,  2
//   i=2   : -Fp1,  4
//   i=4   : -Fp2, -Fp3

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 motion vector fractional component value.</summary>
public enum Vp9MvFpType : byte
{
    /// <summary>Fp = 0 (no fractional offset).</summary>
    Fp0 = 0,
    /// <summary>Fp = 1 (1/8 pel).</summary>
    Fp1 = 1,
    /// <summary>Fp = 2 (2/8 pel = 1/4 pel).</summary>
    Fp2 = 2,
    /// <summary>Fp = 3 (3/8 pel).</summary>
    Fp3 = 3,
}

/// <summary>VP9 motion vector fractional-part tree topology and decoder.</summary>
public static class Vp9MvFpTree
{
    /// <summary>libvpx <c>MV_FP_SIZE</c>.</summary>
    public const int FpSize = 4;

    /// <summary>
    /// libvpx <c>vp9_mv_fp_tree</c>, 6 entries (3 internal nodes
    /// x 2 branches). Negative values are leaf fp values; non-negative
    /// values are byte indices of the next node within this same array.
    /// </summary>
    public static readonly sbyte[] Tree = new sbyte[]
    {
        -(sbyte)Vp9MvFpType.Fp0,  2,
        -(sbyte)Vp9MvFpType.Fp1,  4,
        -(sbyte)Vp9MvFpType.Fp2, -(sbyte)Vp9MvFpType.Fp3,
    };

    /// <summary>
    /// Walk the fp tree given a 3-entry probability vector. The
    /// caller passes either <c>nmv_component.fp</c> (non-Class0) or
    /// the appropriate row of <c>nmv_component.class0_fp</c>
    /// (when MvClass == Class0).
    /// </summary>
    public static Vp9MvFpType Decode(Func<byte, int> readBit, ReadOnlySpan<byte> probs)
    {
        ArgumentNullException.ThrowIfNull(readBit);
        if (probs.Length < FpSize - 1)
            throw new ArgumentException(
                $"probs must hold {FpSize - 1} entries for the MV fp tree", nameof(probs));

        int i = 0;
        while (true)
        {
            int bit = readBit(probs[i >> 1]);
            sbyte next = Tree[i + bit];
            if (next <= 0)
                return (Vp9MvFpType)(-next);
            i = next;
        }
    }
}
