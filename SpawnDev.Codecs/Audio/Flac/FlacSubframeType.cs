// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Broad category of a FLAC subframe, before per-category parameters (FIXED order,
/// LPC order, etc.) are resolved.
/// </summary>
public enum FlacSubframeKind
{
    /// <summary>All samples in the block are identical (single encoded value).</summary>
    Constant,

    /// <summary>Raw uncompressed samples at the subframe bit depth.</summary>
    Verbatim,

    /// <summary>Fixed linear predictor (orders 0-4), Rice-coded residual.</summary>
    Fixed,

    /// <summary>LPC (orders 1-32), Rice-coded residual.</summary>
    Lpc,
}
