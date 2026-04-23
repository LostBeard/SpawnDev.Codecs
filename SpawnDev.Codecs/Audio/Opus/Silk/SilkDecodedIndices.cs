// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// C# equivalent of libopus silk/structs.h::SideInfoIndices. Holds every scalar
// parameter that silk_decode_indices reads from a single SILK frame.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Per-frame SILK side-information indices as decoded by <see cref="SilkIndicesDecoder"/>.
/// Mirrors the shape of libopus <c>SideInfoIndices</c>: scalar fields plus the
/// <see cref="GainsIndices"/>, <see cref="NlsfIndices"/>, and <see cref="LtpIndices"/>
/// arrays. Array lengths are allocated for the worst case - callers only consume up
/// to the frame's nbSubfr or the codebook's order.
/// </summary>
internal sealed class SilkDecodedIndices
{
    /// <summary>SILK signal type (0 inactive, 1 unvoiced, 2 voiced).</summary>
    public sbyte SignalType { get; set; }

    /// <summary>Quantizer offset type (0 or 1).</summary>
    public sbyte QuantOffsetType { get; set; }

    /// <summary>Gain indices (one per subframe). Length = <see cref="SilkConstants.MAX_NB_SUBFR"/>; only first <c>nbSubfr</c> are populated.</summary>
    public sbyte[] GainsIndices { get; } = new sbyte[SilkConstants.MAX_NB_SUBFR];

    /// <summary>NLSF indices. Length = <see cref="SilkConstants.MAX_LPC_ORDER"/> + 1; only first <c>order + 1</c> are populated.</summary>
    public sbyte[] NlsfIndices { get; } = new sbyte[SilkConstants.MAX_LPC_ORDER + 1];

    /// <summary>NLSF interpolation coefficient in Q2 (0..4). Hard-coded to 4 for 10 ms frames.</summary>
    public sbyte NlsfInterpCoefQ2 { get; set; }

    /// <summary>Pitch lag index (only valid for voiced frames).</summary>
    public short LagIndex { get; set; }

    /// <summary>Pitch contour index (only valid for voiced frames).</summary>
    public sbyte ContourIndex { get; set; }

    /// <summary>LTP periodicity / codebook-selector index (only valid for voiced frames).</summary>
    public sbyte PerIndex { get; set; }

    /// <summary>LTP gain indices (one per subframe, valid only for voiced frames).
    /// Length = <see cref="SilkConstants.MAX_NB_SUBFR"/>; only first <c>nbSubfr</c> are populated.</summary>
    public sbyte[] LtpIndices { get; } = new sbyte[SilkConstants.MAX_NB_SUBFR];

    /// <summary>LTP scale index (only valid for voiced frames under independent coding; else 0).</summary>
    public sbyte LtpScaleIndex { get; set; }

    /// <summary>PRNG seed for the excitation sign-scrambler.</summary>
    public sbyte Seed { get; set; }
}
