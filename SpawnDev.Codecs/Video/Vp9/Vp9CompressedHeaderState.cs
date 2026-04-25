// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 compressed-header state container + composition parser.
//
// The compressed header carries probability updates for all the
// entropy decoders (intra mode tree, inter mode tree, partition
// tree, coef probs, mv probs, etc). Per libvpx VP9_COMMON has
// these as a frame context (FRAME_CONTEXT) plus a few frame-level
// fields (tx_mode, reference_mode, etc).
//
// Vp9CompressedHeaderState bundles all the prob storage records
// (slices 210-221) under one umbrella so callers can pass it
// through the parser as a single argument.
//
// Vp9CompressedHeaderParser.Read composes the per-table parsers
// (read_tx_mode + read_tx_mode_probs + read_coef_probs +
// read_skip_probs + the inter-frame group) in libvpx's exact
// order.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Aggregate of all VP9 frame-context probability tables that the
/// compressed header can update. Mirrors libvpx FRAME_CONTEXT
/// scoped to the entries the decoder actually mutates per frame.
/// </summary>
public sealed class Vp9CompressedHeaderState
{
    /// <summary>tx_mode selection probabilities (read_tx_mode_probs).</summary>
    public Vp9TxModeProbs TxModeProbs { get; } = new();

    /// <summary>
    /// Per-tx-size flat coef-prob tables (read_coef_probs). Index
    /// 0..3 = 4x4 / 8x8 / 16x16 / 32x32; each is byte[432].
    /// </summary>
    public byte[][] CoefProbs { get; } = new byte[4][]
    {
        new byte[Vp9CoefProbsParser.FlatSize],
        new byte[Vp9CoefProbsParser.FlatSize],
        new byte[Vp9CoefProbsParser.FlatSize],
        new byte[Vp9CoefProbsParser.FlatSize],
    };

    /// <summary>Skip-flag probabilities (read_skip_probs).</summary>
    public Vp9SkipProbs SkipProbs { get; } = new();

    /// <summary>Inter mode tree probabilities (read_inter_mode_probs).</summary>
    public Vp9InterModeProbsTable InterModeProbs { get; } = new();

    /// <summary>Switchable interp filter probabilities.</summary>
    public Vp9SwitchableInterpProbs SwitchableInterpProbs { get; } = new();

    /// <summary>Intra-vs-inter probabilities (read_intra_inter_probs).</summary>
    public Vp9IntraInterProbs IntraInterProbs { get; } = new();

    /// <summary>Reference-mode probabilities (comp_inter / single_ref / comp_ref).</summary>
    public Vp9ReferenceModeProbs ReferenceModeProbs { get; } = new();

    /// <summary>
    /// Inter-frame Y intra mode probabilities (4 block size groups,
    /// 9 binary tree leaves). Flat byte[36].
    /// </summary>
    public byte[] YModeProbs { get; } =
        (byte[])Vp9IntraModeProbs.DefaultIfYProbs.Clone();

    /// <summary>
    /// Partition tree probabilities (16 contexts, 3 binary tree
    /// leaves). Flat byte[48], seeded with the libvpx defaults.
    /// </summary>
    public byte[] PartitionProbs { get; } =
        (byte[])Vp9PartitionProbs.KfPartitionProbs.Clone();

    /// <summary>Motion vector probabilities (read_mv_probs).</summary>
    public Vp9MvProbs MvProbs { get; } = new();
}

/// <summary>
/// Inputs the compressed header parser needs from the uncompressed
/// header it can't infer for itself.
/// </summary>
/// <param name="IsLossless">True for lossless frames; forces tx_mode = Only4x4.</param>
/// <param name="IsIntraOnly">True for keyframes and intra-only frames.</param>
/// <param name="InterpFilter">Frame-level interpolation filter selector.</param>
/// <param name="AllowHighPrecisionMv">True when the bitstream allows 1/8-pel MVs.</param>
/// <param name="SignBiasLast">Sign bias for the LAST reference frame.</param>
/// <param name="SignBiasGolden">Sign bias for the GOLDEN reference frame.</param>
/// <param name="SignBiasAltRef">Sign bias for the ALTREF reference frame.</param>
public readonly record struct Vp9CompressedHeaderInputs(
    bool IsLossless,
    bool IsIntraOnly,
    Vp9InterpFilter InterpFilter,
    bool AllowHighPrecisionMv,
    bool SignBiasLast,
    bool SignBiasGolden,
    bool SignBiasAltRef);

/// <summary>
/// Result of parsing the compressed header. Carries the parsed
/// tx_mode + reference_mode (the frame-level decisions; everything
/// else lives in the mutated <see cref="Vp9CompressedHeaderState"/>).
/// </summary>
public sealed record Vp9CompressedHeaderResult
{
    /// <summary>Frame tx_mode (read_tx_mode).</summary>
    public required Vp9TxMode TxMode { get; init; }

    /// <summary>
    /// Frame reference_mode (read_frame_reference_mode). Only meaningful
    /// for non-intra-only frames; defaults to <see cref="Vp9ReferenceMode.SingleReference"/>
    /// for keyframes / intra-only.
    /// </summary>
    public required Vp9ReferenceMode ReferenceMode { get; init; }
}

/// <summary>VP9 compressed header parser.</summary>
public static class Vp9CompressedHeaderParser
{
    /// <summary>
    /// Compose the per-table parsers (slices 210-221) in libvpx's
    /// exact order. Returns the parsed tx_mode and reference_mode;
    /// all probability updates land in <paramref name="state"/>.
    /// </summary>
    public static Vp9CompressedHeaderResult Read(
        Vp9CompressedHeaderState state,
        Vp9CompressedHeaderInputs inputs,
        Vp9BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(reader);

        // tx_mode + tx_mode_probs.
        Vp9TxMode txMode;
        if (inputs.IsLossless)
        {
            txMode = Vp9TxMode.Only4x4;
        }
        else
        {
            txMode = Vp9CompressedHeader.ReadTxMode(
                n => reader.ReadLiteral(n), isLossless: false);
            if (txMode == Vp9TxMode.TxModeSelect)
                Vp9TxModeProbsParser.Read(state.TxModeProbs, reader);
        }

        // coef_probs (always, walks tx_sizes 0..max for the mode).
        Vp9CoefProbsParser.ReadCoefProbs(state.CoefProbs, txMode, reader);

        // skip_probs (always).
        Vp9SkipProbsParser.Read(state.SkipProbs, reader);

        // Default reference_mode for keyframes / intra-only.
        var referenceMode = Vp9ReferenceMode.SingleReference;

        if (!inputs.IsIntraOnly)
        {
            // inter_mode_probs.
            Vp9InterModeProbsParser.Read(state.InterModeProbs, reader);

            // switchable_interp_probs (only when interp_filter == Switchable).
            if (inputs.InterpFilter == Vp9InterpFilter.Switchable)
                Vp9SwitchableInterpProbsParser.Read(state.SwitchableInterpProbs, reader);

            // intra_inter_probs.
            Vp9IntraInterProbsParser.Read(state.IntraInterProbs, reader);

            // frame reference_mode + frame reference_mode_probs.
            bool compoundAllowed = Vp9ReferenceModeParser.CompoundReferenceAllowed(
                inputs.SignBiasLast, inputs.SignBiasGolden, inputs.SignBiasAltRef);
            referenceMode = Vp9ReferenceModeParser.Read(reader, compoundAllowed);
            Vp9ReferenceModeProbsParser.Read(state.ReferenceModeProbs, referenceMode, reader);

            // y_mode_probs.
            Vp9YModeProbsParser.Read(state.YModeProbs, reader);

            // partition_probs.
            Vp9PartitionProbsParser.Read(state.PartitionProbs, reader);

            // mv_probs.
            Vp9MvProbsParser.Read(state.MvProbs, inputs.AllowHighPrecisionMv, reader);
        }

        return new Vp9CompressedHeaderResult
        {
            TxMode = txMode,
            ReferenceMode = referenceMode,
        };
    }
}
