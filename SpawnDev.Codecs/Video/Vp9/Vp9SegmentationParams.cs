// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 segmentation parameters parser - the segmentation_params
// section of the uncompressed frame header. Mirror of libvpx
// vp9/decoder/vp9_decodeframe.c setup_segmentation().
//
// VP9 supports up to 8 segments per frame; each segment can locally
// override base quantizer, loop-filter strength, reference frame
// (inter only), and a skip flag. The bitstream encodes:
//
//   enabled          f(1)
//   if enabled:
//     update_map     f(1)
//     if update_map:
//       tree_probs[0..6]: per-entry update flag + f(8) value (no
//         update -> MAX_PROB = 255)
//       temporal_update f(1)
//       pred_probs[0..2]: when temporal_update -> per-entry update
//         flag + f(8); else MAX_PROB
//     update_data    f(1)
//     if update_data:
//       abs_delta    f(1)
//       For each segment 0..7:
//         For each feature ALT_Q / ALT_LF / REF_FRAME / SKIP:
//           feature_enabled f(1)
//           if feature_enabled and the feature carries data:
//             magnitude = unsigned_max(max_value)
//             if signed: sign bit, negate if 1
//
// Feature data widths (libvpx vp9_seg_feature_data_max + signed-ness):
//   ALT_Q     max=255, signed
//   ALT_LF    max=63,  signed
//   REF_FRAME max=3,   unsigned (intra=0, last=1, golden=2, altref=3)
//   SKIP      max=0,   unsigned (no payload bits, just enable flag)
//
// libvpx reads the magnitude with read_inv_signed_literal-like
// "ceil_log2(max+1) bits". That collapses to:
//   max=0   -> 0 bits
//   max=3   -> 2 bits
//   max=63  -> 6 bits
//   max=255 -> 8 bits

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 per-segment feature index (libvpx SEG_LVL_FEATURES).</summary>
public enum Vp9SegFeature : byte
{
    /// <summary>Alternate quantizer (signed).</summary>
    AltQ = 0,
    /// <summary>Alternate loop filter strength (signed).</summary>
    AltLf = 1,
    /// <summary>Reference frame (unsigned 0..3).</summary>
    RefFrame = 2,
    /// <summary>Skip (no payload).</summary>
    Skip = 3,
}

/// <summary>
/// Parsed VP9 segmentation parameters. Bit-exact against libvpx
/// <c>setup_segmentation</c>.
/// </summary>
public sealed record Vp9SegmentationParams
{
    /// <summary>libvpx <c>SEG_TREE_PROBS</c>.</summary>
    public const int TreeProbs = 7;

    /// <summary>libvpx <c>PREDICTION_PROBS</c>.</summary>
    public const int PredictionProbs = 3;

    /// <summary>libvpx <c>MAX_SEGMENTS</c>.</summary>
    public const int MaxSegments = 8;

    /// <summary>libvpx <c>SEG_LVL_MAX</c>.</summary>
    public const int FeaturesPerSegment = 4;

    /// <summary>Probability sentinel meaning "no update on this frame".</summary>
    public const byte MaxProb = 255;

    /// <summary>True when segmentation is in effect for this frame.</summary>
    public required bool Enabled { get; init; }

    /// <summary>True when this frame carries map probability updates.</summary>
    public required bool UpdateMap { get; init; }

    /// <summary>
    /// 7 binary-tree probabilities for segmentation map decoding.
    /// MaxProb (255) means "no update on this frame".
    /// </summary>
    public required byte[] TreeProbsArray { get; init; } = Array.Empty<byte>();

    /// <summary>
    /// True when the frame uses the temporal predictor for the map
    /// (only meaningful when <see cref="UpdateMap"/> is true).
    /// </summary>
    public required bool TemporalUpdate { get; init; }

    /// <summary>3 prediction probabilities (only meaningful when temporal_update).</summary>
    public required byte[] PredProbs { get; init; } = Array.Empty<byte>();

    /// <summary>True when this frame carries per-segment feature data updates.</summary>
    public required bool UpdateData { get; init; }

    /// <summary>True when feature values are absolute; false means delta from the base.</summary>
    public required bool AbsDelta { get; init; }

    /// <summary>
    /// Feature-enabled flags, indexed [segment, feature]. 8 segments x
    /// 4 features.
    /// </summary>
    public required bool[,] FeatureEnabled { get; init; } = new bool[0, 0];

    /// <summary>
    /// Feature payload values. Only meaningful when the corresponding
    /// <see cref="FeatureEnabled"/> entry is true. SKIP feature entries
    /// are always 0.
    /// </summary>
    public required int[,] FeatureData { get; init; } = new int[0, 0];
}

/// <summary>Parser for VP9 segmentation parameters.</summary>
public static class Vp9SegmentationParamsParser
{
    private static readonly int[] FeatureMaxValue = { 255, 63, 3, 0 };
    private static readonly bool[] FeatureIsSigned = { true, true, false, false };

    /// <summary>
    /// Parse segmentation parameters from <paramref name="reader"/>.
    /// </summary>
    internal static Vp9SegmentationParams Parse(ref Vp9BitReader reader)
    {
        bool enabled = reader.ReadFlag();
        if (!enabled)
        {
            return new Vp9SegmentationParams
            {
                Enabled = false,
                UpdateMap = false,
                TreeProbsArray = Array.Empty<byte>(),
                TemporalUpdate = false,
                PredProbs = Array.Empty<byte>(),
                UpdateData = false,
                AbsDelta = false,
                FeatureEnabled = new bool[0, 0],
                FeatureData = new int[0, 0],
            };
        }

        bool updateMap = reader.ReadFlag();
        var treeProbs = new byte[Vp9SegmentationParams.TreeProbs];
        bool temporalUpdate = false;
        var predProbs = new byte[Vp9SegmentationParams.PredictionProbs];
        if (updateMap)
        {
            for (int i = 0; i < Vp9SegmentationParams.TreeProbs; i++)
                treeProbs[i] = reader.ReadFlag()
                    ? (byte)reader.ReadBits(8)
                    : Vp9SegmentationParams.MaxProb;
            temporalUpdate = reader.ReadFlag();
            for (int i = 0; i < Vp9SegmentationParams.PredictionProbs; i++)
            {
                predProbs[i] = temporalUpdate
                    ? (reader.ReadFlag() ? (byte)reader.ReadBits(8) : Vp9SegmentationParams.MaxProb)
                    : Vp9SegmentationParams.MaxProb;
            }
        }

        bool updateData = reader.ReadFlag();
        bool absDelta = false;
        var featureEnabled = new bool[Vp9SegmentationParams.MaxSegments, Vp9SegmentationParams.FeaturesPerSegment];
        var featureData = new int[Vp9SegmentationParams.MaxSegments, Vp9SegmentationParams.FeaturesPerSegment];
        if (updateData)
        {
            absDelta = reader.ReadFlag();
            for (int seg = 0; seg < Vp9SegmentationParams.MaxSegments; seg++)
            {
                for (int feat = 0; feat < Vp9SegmentationParams.FeaturesPerSegment; feat++)
                {
                    bool feEnabled = reader.ReadFlag();
                    featureEnabled[seg, feat] = feEnabled;
                    int data = 0;
                    if (feEnabled)
                    {
                        int max = FeatureMaxValue[feat];
                        int magBits = MagnitudeBits(max);
                        if (magBits > 0)
                            data = (int)reader.ReadBits(magBits);
                        if (FeatureIsSigned[feat] && data != 0)
                        {
                            if (reader.ReadFlag()) data = -data;
                        }
                    }
                    featureData[seg, feat] = data;
                }
            }
        }

        return new Vp9SegmentationParams
        {
            Enabled = true,
            UpdateMap = updateMap,
            TreeProbsArray = treeProbs,
            TemporalUpdate = temporalUpdate,
            PredProbs = predProbs,
            UpdateData = updateData,
            AbsDelta = absDelta,
            FeatureEnabled = featureEnabled,
            FeatureData = featureData,
        };
    }

    /// <summary>Convenience overload for unit tests.</summary>
    public static Vp9SegmentationParams Parse(ReadOnlySpan<byte> data)
    {
        var r = new Vp9BitReader(data);
        return Parse(ref r);
    }

    /// <summary>
    /// libvpx <c>get_unsigned_bits(max)</c>: minimum number of bits
    /// needed to encode an unsigned integer in [0, max]. Returns 0 when
    /// max == 0 (the SKIP feature carries no payload).
    /// </summary>
    private static int MagnitudeBits(int max)
    {
        if (max == 0) return 0;
        int bits = 0;
        while ((1 << bits) <= max) bits++;
        return bits;
    }
}
