// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 loop filter parameters parser - the loop_filter_params section
// of the uncompressed frame header. Mirror of libvpx
// vp9/decoder/vp9_decodeframe.c setup_loopfilter().
//
// Bitstream layout (VP9 spec sec 6.2.7):
//   filter_level         f(6)
//   sharpness_level      f(3)
//   mode_ref_delta_enabled f(1)
//   if (mode_ref_delta_enabled) {
//     mode_ref_delta_update f(1)
//     if (mode_ref_delta_update) {
//       for (i = 0; i < MAX_REF_LF_DELTAS = 4; i++) {
//         update_ref_delta f(1)
//         if (update_ref_delta) ref_deltas[i] = s(6)  // signed-magnitude
//       }
//       for (i = 0; i < MAX_MODE_LF_DELTAS = 2; i++) {
//         update_mode_delta f(1)
//         if (update_mode_delta) mode_deltas[i] = s(6)
//       }
//     }
//   }
//
// Note: ref_deltas / mode_deltas are PERSISTENT decoder state - they
// carry forward across frames unless update_ref_delta /
// update_mode_delta is set on a given frame. The parser here returns
// only what the bitstream encoded; the persistent merge-with-previous
// logic is the caller's job (Vp9Decoder state, future slice).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Per-frame loop filter parameters parsed from the uncompressed
/// header. Bit-exact against libvpx <c>setup_loopfilter</c>.
/// </summary>
public sealed record Vp9LoopFilterParams
{
    /// <summary>libvpx <c>MAX_REF_LF_DELTAS</c>.</summary>
    public const int MaxRefDeltas = 4;

    /// <summary>libvpx <c>MAX_MODE_LF_DELTAS</c>.</summary>
    public const int MaxModeDeltas = 2;

    /// <summary>Per-frame filter strength, 0..63 (f(6)).</summary>
    public required int FilterLevel { get; init; }

    /// <summary>Sharpness level, 0..7 (f(3)).</summary>
    public required int SharpnessLevel { get; init; }

    /// <summary>True when the bitstream signals mode/ref deltas.</summary>
    public required bool ModeRefDeltaEnabled { get; init; }

    /// <summary>
    /// True when this frame carries a delta update. Only meaningful
    /// when <see cref="ModeRefDeltaEnabled"/> is true. When false, the
    /// caller carries forward the previous frame's deltas unchanged.
    /// </summary>
    public required bool ModeRefDeltaUpdate { get; init; }

    /// <summary>
    /// Per-reference-frame delta updates, one entry per reference
    /// (Intra, Last, Golden, Alt). A null entry means "no update -
    /// keep previous value"; a non-null entry is the new signed value.
    /// Always 4 entries when <see cref="ModeRefDeltaUpdate"/> is true.
    /// </summary>
    public required int?[] RefDeltas { get; init; } = Array.Empty<int?>();

    /// <summary>
    /// Per-prediction-mode delta updates (zero, new). Same null-means-
    /// keep semantics as <see cref="RefDeltas"/>. Always 2 entries when
    /// <see cref="ModeRefDeltaUpdate"/> is true.
    /// </summary>
    public required int?[] ModeDeltas { get; init; } = Array.Empty<int?>();
}

/// <summary>
/// Parser for VP9 loop filter parameters in the uncompressed header.
/// </summary>
public static class Vp9LoopFilterParamsParser
{
    /// <summary>
    /// Parse loop filter parameters from <paramref name="reader"/>.
    /// Reads at minimum 10 bits (filter_level + sharpness_level +
    /// mode_ref_delta_enabled=0). The parser advances
    /// <paramref name="reader"/> by the number of bits consumed.
    /// </summary>
    internal static Vp9LoopFilterParams Parse(ref Vp9BitReader reader)
    {
        int filterLevel = (int)reader.ReadBits(6);
        int sharpnessLevel = (int)reader.ReadBits(3);
        bool modeRefDeltaEnabled = reader.ReadFlag();
        bool modeRefDeltaUpdate = false;
        int?[] refDeltas = Array.Empty<int?>();
        int?[] modeDeltas = Array.Empty<int?>();

        if (modeRefDeltaEnabled)
        {
            modeRefDeltaUpdate = reader.ReadFlag();
            if (modeRefDeltaUpdate)
            {
                refDeltas = new int?[Vp9LoopFilterParams.MaxRefDeltas];
                for (int i = 0; i < Vp9LoopFilterParams.MaxRefDeltas; i++)
                {
                    if (reader.ReadFlag())
                        refDeltas[i] = reader.ReadSignedLiteral(6);
                }
                modeDeltas = new int?[Vp9LoopFilterParams.MaxModeDeltas];
                for (int i = 0; i < Vp9LoopFilterParams.MaxModeDeltas; i++)
                {
                    if (reader.ReadFlag())
                        modeDeltas[i] = reader.ReadSignedLiteral(6);
                }
            }
        }

        return new Vp9LoopFilterParams
        {
            FilterLevel = filterLevel,
            SharpnessLevel = sharpnessLevel,
            ModeRefDeltaEnabled = modeRefDeltaEnabled,
            ModeRefDeltaUpdate = modeRefDeltaUpdate,
            RefDeltas = refDeltas,
            ModeDeltas = modeDeltas,
        };
    }

    /// <summary>
    /// Parse loop filter parameters from a byte span, starting at the
    /// beginning. For unit tests; production callers parse via the
    /// frame header which advances a single shared bit reader.
    /// </summary>
    public static Vp9LoopFilterParams Parse(ReadOnlySpan<byte> data)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r);
    }
}
