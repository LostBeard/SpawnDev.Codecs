// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame header parser. Decodes the compressed first-partition prefix
// per RFC 6386 sec 9.2-9.10. Begins at the bool-decoder cursor right
// after Vp8FrameTagParser has consumed the 3-byte tag (and 7-byte key
// extension if applicable).
//
// Currently implements the KEY-FRAME path. Inter-frame additions
// (refresh_golden, refresh_alt_ref, ref_frame_sign_bias, intra_y_mode
// prob updates, intra_uv_mode prob updates, mv prob updates) layer on
// top in a future slice.
//
// Frame header order (key frame, RFC 6386 sec 9.2-9.10):
//   colorSpace            1 bit
//   clampingType          1 bit
//   segmentation          variable (enabled + map updates + data updates)
//   filterType            1 bit
//   filterLevel           6 bits
//   sharpnessLevel        3 bits
//   modeRefLfDeltaEnabled 1 bit (+ deltas if update enabled)
//   log2NumPartitions     2 bits
//   quantizer             7 + 5*7 = up to 42 bits (Y_AC + 5 deltas)
//   refreshEntropyProbs   1 bit
//   coefProbUpdates       4D walk against Vp8CoefUpdateProbs (1056 bits max)
//   mbNoSkipCoeff         1 bit (+ 8 bits prob_skip_false if enabled)

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 quantizer indices + deltas (decoded from frame header sec 9.6).</summary>
public sealed record Vp8QuantizerIndices
{
    /// <summary>Base Q index (Y AC, 7 bits).</summary>
    public required int BaseQIndex { get; init; }
    /// <summary>Y1 DC delta (signed, 5 bits).</summary>
    public required int Y1DcDeltaQ { get; init; }
    /// <summary>Y2 DC delta (signed, 5 bits).</summary>
    public required int Y2DcDeltaQ { get; init; }
    /// <summary>Y2 AC delta (signed, 5 bits).</summary>
    public required int Y2AcDeltaQ { get; init; }
    /// <summary>UV DC delta (signed, 5 bits).</summary>
    public required int UvDcDeltaQ { get; init; }
    /// <summary>UV AC delta (signed, 5 bits).</summary>
    public required int UvAcDeltaQ { get; init; }
}

/// <summary>VP8 loop filter parameters (decoded from frame header sec 9.4).</summary>
public sealed record Vp8LoopFilterParams
{
    /// <summary>0 = normal, 1 = simple.</summary>
    public required int FilterType { get; init; }
    /// <summary>0..63.</summary>
    public required int FilterLevel { get; init; }
    /// <summary>0..7.</summary>
    public required int SharpnessLevel { get; init; }
    /// <summary>True if mode/ref-frame loop-filter deltas are in effect.</summary>
    public required bool ModeRefLfDeltaEnabled { get; init; }
    /// <summary>4 ref-frame delta values (signed, valid only if Enabled).</summary>
    public required int[] RefLfDeltas { get; init; }
    /// <summary>4 mode-based delta values (signed, valid only if Enabled).</summary>
    public required int[] ModeLfDeltas { get; init; }
}

/// <summary>VP8 segmentation parameters (decoded from frame header sec 9.3).</summary>
public sealed record Vp8SegmentationParams
{
    /// <summary>True if segmentation is enabled this frame.</summary>
    public required bool Enabled { get; init; }
    /// <summary>True if the segment map is being updated.</summary>
    public required bool UpdateMap { get; init; }
    /// <summary>True if segment feature data is being updated.</summary>
    public required bool UpdateData { get; init; }
    /// <summary>True if feature data values are absolute (vs delta).</summary>
    public required bool AbsDelta { get; init; }
    /// <summary>Per-feature per-segment data (2 features x 4 segments). Q delta + LF delta.</summary>
    public required int[,] FeatureData { get; init; }
    /// <summary>3 segment-tree probabilities for decoding the map (when map updates enabled).</summary>
    public required byte[] SegmentTreeProbs { get; init; }
}

/// <summary>VP8 frame header (decoded). Combines all the structural fields from RFC 6386 sec 9.2-9.10.</summary>
public sealed record Vp8FrameHeader
{
    /// <summary>0 = YUV (single defined value per spec).</summary>
    public required int ColorSpace { get; init; }
    /// <summary>0 = no clamping required, 1 = clamp to [0, 255].</summary>
    public required int ClampingType { get; init; }
    /// <summary>Segmentation parameters.</summary>
    public required Vp8SegmentationParams Segmentation { get; init; }
    /// <summary>Loop filter parameters.</summary>
    public required Vp8LoopFilterParams LoopFilter { get; init; }
    /// <summary>log2 of token partition count: 0=1 partition, 1=2, 2=4, 3=8.</summary>
    public required int Log2NumPartitions { get; init; }
    /// <summary>Quantizer indices.</summary>
    public required Vp8QuantizerIndices Quantizer { get; init; }
    /// <summary>True if the frame's coefficient probabilities should persist past this frame.</summary>
    public required bool RefreshEntropyProbs { get; init; }
    /// <summary>4D coefficient prob table after applying frame-header updates to defaults.</summary>
    public required byte[,,,] CoefProbs { get; init; }
    /// <summary>True if mb_no_skip_coeff feature is enabled.</summary>
    public required bool MbNoSkipCoeffEnabled { get; init; }
    /// <summary>Probability of mb_skip_coeff = 0 (i.e., block has any coefficients to decode), 8-bit.</summary>
    public required int ProbSkipFalse { get; init; }
}

/// <summary>VP8 frame header parser. RFC 6386 sec 9.2-9.10 (key-frame path).</summary>
public static class Vp8FrameHeaderParser
{
    /// <summary>Number of segment-tree probability values for the segment map (libvpx MB_FEATURE_TREE_PROBS).</summary>
    public const int MbFeatureTreeProbs = 3;
    /// <summary>Number of macroblock-level features tracked per segment (Q delta + LF delta).</summary>
    public const int MbLvlMax = 2;
    /// <summary>Maximum number of segments.</summary>
    public const int MaxMbSegments = 4;
    /// <summary>Bit widths for the two feature data values: Q delta = 7 bits, LF delta = 6 bits.</summary>
    public static readonly int[] MbFeatureDataBits = new int[] { 7, 6 };
    /// <summary>Number of mode-based loop filter deltas.</summary>
    public const int MaxModeLfDeltas = 4;
    /// <summary>Number of ref-frame loop filter deltas.</summary>
    public const int MaxRefLfDeltas = 4;

    /// <summary>
    /// Parse the VP8 key-frame header from the bool-decoder stream. The
    /// reader must be positioned at the first bit of the frame header
    /// (i.e., right after the 3-byte frame tag + 7-byte key extension
    /// have been consumed by the caller).
    /// </summary>
    public static Vp8FrameHeader ParseKeyFrameHeader(Vp8BoolDecoder reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        int colorSpace = reader.DecodeValue(1);
        int clampingType = reader.DecodeValue(1);
        var segmentation = ParseSegmentation(reader);
        var loopFilter = ParseLoopFilter(reader);
        int log2NumPartitions = reader.DecodeValue(2);
        var quantizer = ParseQuantizer(reader);
        // Key frames implicitly refresh both gold + alt ref; the
        // refresh_golden_frame / refresh_alt_ref_frame fields aren't
        // present in the key-frame stream.
        bool refreshEntropyProbs = reader.DecodeValue(1) != 0;
        // refresh_last_frame is always 1 for key frames; not in stream.
        var coefProbs = ParseCoefProbUpdates(reader);
        bool mbNoSkipCoeffEnabled = reader.DecodeValue(1) != 0;
        int probSkipFalse = mbNoSkipCoeffEnabled ? reader.DecodeValue(8) : 0;

        return new Vp8FrameHeader
        {
            ColorSpace = colorSpace,
            ClampingType = clampingType,
            Segmentation = segmentation,
            LoopFilter = loopFilter,
            Log2NumPartitions = log2NumPartitions,
            Quantizer = quantizer,
            RefreshEntropyProbs = refreshEntropyProbs,
            CoefProbs = coefProbs,
            MbNoSkipCoeffEnabled = mbNoSkipCoeffEnabled,
            ProbSkipFalse = probSkipFalse,
        };
    }

    private static Vp8SegmentationParams ParseSegmentation(Vp8BoolDecoder reader)
    {
        bool enabled = reader.DecodeValue(1) != 0;
        bool updateMap = false, updateData = false, absDelta = false;
        var featureData = new int[MbLvlMax, MaxMbSegments];
        var segmentTreeProbs = new byte[MbFeatureTreeProbs];
        for (int i = 0; i < MbFeatureTreeProbs; i++) segmentTreeProbs[i] = 255; // libvpx default

        if (enabled)
        {
            updateMap = reader.DecodeValue(1) != 0;
            updateData = reader.DecodeValue(1) != 0;
            if (updateData)
            {
                absDelta = reader.DecodeValue(1) != 0;
                for (int i = 0; i < MbLvlMax; i++)
                {
                    for (int j = 0; j < MaxMbSegments; j++)
                    {
                        if (reader.DecodeValue(1) != 0)
                        {
                            int v = reader.DecodeValue(MbFeatureDataBits[i]);
                            if (reader.DecodeValue(1) != 0) v = -v;
                            featureData[i, j] = v;
                        }
                    }
                }
            }
            if (updateMap)
            {
                for (int i = 0; i < MbFeatureTreeProbs; i++)
                {
                    if (reader.DecodeValue(1) != 0)
                        segmentTreeProbs[i] = (byte)reader.DecodeValue(8);
                }
            }
        }

        return new Vp8SegmentationParams
        {
            Enabled = enabled,
            UpdateMap = updateMap,
            UpdateData = updateData,
            AbsDelta = absDelta,
            FeatureData = featureData,
            SegmentTreeProbs = segmentTreeProbs,
        };
    }

    private static Vp8LoopFilterParams ParseLoopFilter(Vp8BoolDecoder reader)
    {
        int filterType = reader.DecodeValue(1);
        int filterLevel = reader.DecodeValue(6);
        int sharpnessLevel = reader.DecodeValue(3);
        bool modeRefLfDeltaEnabled = reader.DecodeValue(1) != 0;
        var refLfDeltas = new int[MaxRefLfDeltas];
        var modeLfDeltas = new int[MaxModeLfDeltas];

        if (modeRefLfDeltaEnabled)
        {
            bool deltaUpdate = reader.DecodeValue(1) != 0;
            if (deltaUpdate)
            {
                for (int i = 0; i < MaxRefLfDeltas; i++)
                {
                    if (reader.DecodeValue(1) != 0)
                    {
                        int v = reader.DecodeValue(6);
                        if (reader.DecodeValue(1) != 0) v = -v;
                        refLfDeltas[i] = v;
                    }
                }
                for (int i = 0; i < MaxModeLfDeltas; i++)
                {
                    if (reader.DecodeValue(1) != 0)
                    {
                        int v = reader.DecodeValue(6);
                        if (reader.DecodeValue(1) != 0) v = -v;
                        modeLfDeltas[i] = v;
                    }
                }
            }
        }

        return new Vp8LoopFilterParams
        {
            FilterType = filterType,
            FilterLevel = filterLevel,
            SharpnessLevel = sharpnessLevel,
            ModeRefLfDeltaEnabled = modeRefLfDeltaEnabled,
            RefLfDeltas = refLfDeltas,
            ModeLfDeltas = modeLfDeltas,
        };
    }

    private static Vp8QuantizerIndices ParseQuantizer(Vp8BoolDecoder reader)
    {
        int baseQ = reader.DecodeValue(7);
        int y1Dc = ReadSignedDelta(reader, 4);
        int y2Dc = ReadSignedDelta(reader, 4);
        int y2Ac = ReadSignedDelta(reader, 4);
        int uvDc = ReadSignedDelta(reader, 4);
        int uvAc = ReadSignedDelta(reader, 4);
        return new Vp8QuantizerIndices
        {
            BaseQIndex = baseQ,
            Y1DcDeltaQ = y1Dc,
            Y2DcDeltaQ = y2Dc,
            Y2AcDeltaQ = y2Ac,
            UvDcDeltaQ = uvDc,
            UvAcDeltaQ = uvAc,
        };
    }

    /// <summary>libvpx get_delta_q: 1 bit "present" + N bits magnitude + 1 bit sign.</summary>
    private static int ReadSignedDelta(Vp8BoolDecoder reader, int magBits)
    {
        if (reader.DecodeValue(1) == 0) return 0;
        int v = reader.DecodeValue(magBits);
        if (reader.DecodeValue(1) != 0) v = -v;
        return v;
    }

    private static byte[,,,] ParseCoefProbUpdates(Vp8BoolDecoder reader)
    {
        // Start with the default coef probs (4D copy), apply per-entry
        // updates against vp8_coef_update_probs.
        int b = Vp8DefaultCoefProbs.BlockTypes;
        int n = Vp8DefaultCoefProbs.CoefBands;
        int c = Vp8DefaultCoefProbs.PrevCoefContexts;
        int e = Vp8DefaultCoefProbs.EntropyNodes;
        var probs = new byte[b, n, c, e];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < c; k++)
                    for (int l = 0; l < e; l++)
                        probs[i, j, k, l] = Vp8DefaultCoefProbs.DefaultProbs[i, j, k, l];

        for (int i = 0; i < b; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < c; k++)
                    for (int l = 0; l < e; l++)
                        if (reader.DecodeBool(Vp8CoefUpdateProbs.UpdateProbs[i, j, k, l]) != 0)
                            probs[i, j, k, l] = (byte)reader.DecodeValue(8);

        return probs;
    }
}
