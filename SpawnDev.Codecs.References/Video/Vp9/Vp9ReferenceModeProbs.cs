// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 reference-mode probability storage + parser. The compressed
// header carries up to 3 sub-tables of update bits gated by the
// frame-level reference_mode (SINGLE / COMPOUND / SELECT):
//
//   reference_mode == REFERENCE_MODE_SELECT
//     comp_inter_prob[COMP_INTER_CONTEXTS=5]
//   reference_mode != COMPOUND_REFERENCE
//     single_ref_prob[REF_CONTEXTS=5][2]
//   reference_mode != SINGLE_REFERENCE
//     comp_ref_prob[REF_CONTEXTS=5]
//
// Mirror of libvpx vp9/decoder/vp9_decodeframe.c
// read_frame_reference_mode_probs.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 frame-level reference mode (libvpx REFERENCE_MODE).</summary>
public enum Vp9ReferenceMode : byte
{
    /// <summary>Single reference for every inter block.</summary>
    SingleReference = 0,
    /// <summary>Compound reference for every inter block.</summary>
    CompoundReference = 1,
    /// <summary>Per-block selection signalled in the bitstream.</summary>
    ReferenceModeSelect = 2,
}

/// <summary>VP9 reference-mode probability tables.</summary>
public sealed class Vp9ReferenceModeProbs
{
    /// <summary>libvpx <c>COMP_INTER_CONTEXTS</c>.</summary>
    public const int CompInterContexts = 5;

    /// <summary>libvpx <c>REF_CONTEXTS</c>.</summary>
    public const int RefContexts = 5;

    /// <summary>5 comp-inter prob bytes (only updated under ReferenceModeSelect).</summary>
    public byte[] CompInterProb { get; } = new byte[CompInterContexts];

    /// <summary>5 contexts x 2 single-ref probs (skipped only under CompoundReference).</summary>
    public byte[,] SingleRefProb { get; } = new byte[RefContexts, 2];

    /// <summary>5 comp-ref prob bytes (skipped only under SingleReference).</summary>
    public byte[] CompRefProb { get; } = new byte[RefContexts];
}

/// <summary>
/// Parser for the read_frame_reference_mode_probs section of the
/// compressed header. Branches gated on <c>reference_mode</c>.
/// </summary>
public static class Vp9ReferenceModeProbsParser
{
    /// <summary>
    /// Apply diff_update_prob to whichever sub-tables the
    /// <paramref name="referenceMode"/> activates. Mirror of libvpx
    /// <c>read_frame_reference_mode_probs</c>.
    /// </summary>
    public static void Read(
        Vp9ReferenceModeProbs probs,
        Vp9ReferenceMode referenceMode,
        Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(probs);
        ArgumentNullException.ThrowIfNull(reader);

        if (referenceMode == Vp9ReferenceMode.ReferenceModeSelect)
            for (int i = 0; i < Vp9ReferenceModeProbs.CompInterContexts; i++)
                probs.CompInterProb[i] = Vp9DiffUpdateProb.Read(reader, probs.CompInterProb[i]);

        if (referenceMode != Vp9ReferenceMode.CompoundReference)
            for (int i = 0; i < Vp9ReferenceModeProbs.RefContexts; i++)
            {
                probs.SingleRefProb[i, 0] = Vp9DiffUpdateProb.Read(reader, probs.SingleRefProb[i, 0]);
                probs.SingleRefProb[i, 1] = Vp9DiffUpdateProb.Read(reader, probs.SingleRefProb[i, 1]);
            }

        if (referenceMode != Vp9ReferenceMode.SingleReference)
            for (int i = 0; i < Vp9ReferenceModeProbs.RefContexts; i++)
                probs.CompRefProb[i] = Vp9DiffUpdateProb.Read(reader, probs.CompRefProb[i]);
    }
}
