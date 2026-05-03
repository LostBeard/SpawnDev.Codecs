// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 compound motion vector pair. Compound inter prediction uses
// two reference frames; each carries its own MV. This record bundles
// the pair for storage in mode info and for inter-prediction
// dispatch.
//
// libvpx reference: vp9/common/vp9_blockd.h MB_MODE_INFO.mv[2].

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 compound MV pair (one per reference for compound blocks).</summary>
public readonly record struct Vp9CompoundMv(Vp9Mv Mv0, Vp9Mv Mv1)
{
    /// <summary>The (zero, zero) compound pair.</summary>
    public static readonly Vp9CompoundMv Zero = new Vp9CompoundMv(Vp9Mv.Zero, Vp9Mv.Zero);

    /// <summary>True when both component MVs are zero.</summary>
    public bool IsZero => Mv0.IsZero && Mv1.IsZero;

    /// <summary>
    /// Component-wise sum of both MVs (useful when averaging compound
    /// candidates).
    /// </summary>
    public Vp9Mv Sum => Mv0 + Mv1;

    /// <summary>Per-component clamp applied to both MVs.</summary>
    public Vp9CompoundMv Clamp() => new Vp9CompoundMv(Mv0.Clamp(), Mv1.Clamp());
}
