// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 compound reference mode resolver. Given the per-frame sign
// biases for the 3 reference slots (Last / Golden / AltRef),
// derives the (fixed, var0, var1) triple used by compound-pred
// blocks. Mirror of libvpx vp9/decoder/vp9_decodeframe.c
// setup_compound_reference_mode.
//
// Intuition: libvpx picks the "odd one out" of the three sign-bias
// flags as the FIXED reference - the one that isn't aligned with
// the other two. The two aligned flags become the VAR refs that the
// bitstream picks between for each compound block.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 compound reference triple: one fixed reference (always
/// available) plus two variable references that the per-block code
/// chooses between when compound prediction is enabled.
/// </summary>
public readonly record struct Vp9CompoundReferenceMode(
    Vp9MvReferenceFrame FixedRef,
    Vp9MvReferenceFrame VarRef0,
    Vp9MvReferenceFrame VarRef1)
{
    /// <summary>
    /// Derive the compound reference triple from per-slot sign-bias
    /// flags. Mirror of libvpx <c>setup_compound_reference_mode</c>.
    /// </summary>
    /// <param name="lastBias">Sign bias for the LAST reference slot.</param>
    /// <param name="goldenBias">Sign bias for the GOLDEN reference slot.</param>
    /// <param name="altRefBias">Sign bias for the ALTREF reference slot.</param>
    public static Vp9CompoundReferenceMode Compute(
        bool lastBias, bool goldenBias, bool altRefBias)
    {
        if (lastBias == goldenBias)
        {
            // Last and Golden agree -> AltRef is the odd one -> fixed.
            return new Vp9CompoundReferenceMode(
                FixedRef: Vp9MvReferenceFrame.AltRef,
                VarRef0: Vp9MvReferenceFrame.Last,
                VarRef1: Vp9MvReferenceFrame.Golden);
        }
        if (lastBias == altRefBias)
        {
            // Last and AltRef agree -> Golden is fixed.
            return new Vp9CompoundReferenceMode(
                FixedRef: Vp9MvReferenceFrame.Golden,
                VarRef0: Vp9MvReferenceFrame.Last,
                VarRef1: Vp9MvReferenceFrame.AltRef);
        }
        // Else: Golden and AltRef agree, Last is fixed.
        return new Vp9CompoundReferenceMode(
            FixedRef: Vp9MvReferenceFrame.Last,
            VarRef0: Vp9MvReferenceFrame.Golden,
            VarRef1: Vp9MvReferenceFrame.AltRef);
    }
}
