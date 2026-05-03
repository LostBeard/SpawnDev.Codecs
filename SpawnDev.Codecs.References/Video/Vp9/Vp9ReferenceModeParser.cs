// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 reference mode parser. Reads the frame-level reference_mode
// from the compressed header. Encoding:
//
//   if (compound_reference_allowed):
//     bit 0:
//       0 -> SINGLE_REFERENCE
//       1 -> bit 1:
//              0 -> COMPOUND_REFERENCE
//              1 -> REFERENCE_MODE_SELECT
//   else:
//     SINGLE_REFERENCE (no bits read)
//
// compound_reference_allowed is true when not all three reference
// frames (LAST, GOLDEN, ALTREF) have the same sign bias.
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c read_frame_reference_mode
// and vp9/common/vp9_pred_common.h vp9_compound_reference_allowed.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 reference-mode parser.</summary>
public static class Vp9ReferenceModeParser
{
    /// <summary>
    /// True when at least one of the three reference frames has a
    /// different sign bias from the others. Mirror of libvpx
    /// <c>vp9_compound_reference_allowed</c>.
    /// </summary>
    /// <param name="signBiasLast">Sign bias bit for the LAST reference.</param>
    /// <param name="signBiasGolden">Sign bias bit for the GOLDEN reference.</param>
    /// <param name="signBiasAltRef">Sign bias bit for the ALTREF reference.</param>
    public static bool CompoundReferenceAllowed(
        bool signBiasLast, bool signBiasGolden, bool signBiasAltRef)
    {
        // libvpx checks pairwise-different; equivalent to "not all the same".
        return !(signBiasLast == signBiasGolden && signBiasGolden == signBiasAltRef);
    }

    /// <summary>
    /// Read the frame-level reference mode from
    /// <paramref name="reader"/>. Reads 0..2 bits depending on
    /// <paramref name="compoundReferenceAllowed"/>.
    /// </summary>
    public static Vp9ReferenceMode Read(Vp9BoolDecoder reader, bool compoundReferenceAllowed)
    {
        ArgumentNullException.ThrowIfNull(reader);
        if (!compoundReferenceAllowed)
            return Vp9ReferenceMode.SingleReference;

        if (reader.ReadBit() == 0)
            return Vp9ReferenceMode.SingleReference;
        return reader.ReadBit() != 0
            ? Vp9ReferenceMode.ReferenceModeSelect
            : Vp9ReferenceMode.CompoundReference;
    }
}
