// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 frame header writer - encoder-side counterpart of
// Vp8FrameHeaderParser. Emits the compressed first-partition prefix
// matching the bit layout the parser reads, via a Vp8BoolEncoder.
//
// Currently implements the KEY-FRAME path. Inter-frame additions
// (refresh flags, sign biases, intra mode prob updates, MV prob
// updates) layer on later.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 frame header writer (key-frame path).</summary>
public static class Vp8FrameHeaderWriter
{
    /// <summary>
    /// Emit the VP8 key-frame header into the bool encoder. The caller
    /// is responsible for writing the 3+7 byte uncompressed frame tag /
    /// key extension separately (via Vp8FrameTagWriter).
    /// </summary>
    public static void WriteKeyFrameHeader(Vp8BoolEncoder writer, Vp8FrameHeader header)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(header);

        writer.EncodeValue(header.ColorSpace, 1);
        writer.EncodeValue(header.ClampingType, 1);
        WriteSegmentation(writer, header.Segmentation);
        WriteLoopFilter(writer, header.LoopFilter);
        writer.EncodeValue(header.Log2NumPartitions, 2);
        WriteQuantizer(writer, header.Quantizer);
        writer.EncodeValue(header.RefreshEntropyProbs ? 1 : 0, 1);
        WriteCoefProbUpdates(writer, header.CoefProbs);
        writer.EncodeValue(header.MbNoSkipCoeffEnabled ? 1 : 0, 1);
        if (header.MbNoSkipCoeffEnabled)
            writer.EncodeValue(header.ProbSkipFalse, 8);
    }

    private static void WriteSegmentation(Vp8BoolEncoder writer, Vp8SegmentationParams seg)
    {
        writer.EncodeValue(seg.Enabled ? 1 : 0, 1);
        if (!seg.Enabled) return;

        writer.EncodeValue(seg.UpdateMap ? 1 : 0, 1);
        writer.EncodeValue(seg.UpdateData ? 1 : 0, 1);

        if (seg.UpdateData)
        {
            writer.EncodeValue(seg.AbsDelta ? 1 : 0, 1);
            for (int i = 0; i < Vp8FrameHeaderParser.MbLvlMax; i++)
            {
                for (int j = 0; j < Vp8FrameHeaderParser.MaxMbSegments; j++)
                {
                    int v = seg.FeatureData[i, j];
                    if (v != 0)
                    {
                        writer.EncodeValue(1, 1);
                        int absV = v < 0 ? -v : v;
                        writer.EncodeValue(absV, Vp8FrameHeaderParser.MbFeatureDataBits[i]);
                        writer.EncodeValue(v < 0 ? 1 : 0, 1);
                    }
                    else
                    {
                        writer.EncodeValue(0, 1);
                    }
                }
            }
        }

        if (seg.UpdateMap)
        {
            for (int i = 0; i < Vp8FrameHeaderParser.MbFeatureTreeProbs; i++)
            {
                if (seg.SegmentTreeProbs[i] != 255)
                {
                    writer.EncodeValue(1, 1);
                    writer.EncodeValue(seg.SegmentTreeProbs[i], 8);
                }
                else
                {
                    writer.EncodeValue(0, 1);
                }
            }
        }
    }

    private static void WriteLoopFilter(Vp8BoolEncoder writer, Vp8LoopFilterParams lf)
    {
        writer.EncodeValue(lf.FilterType, 1);
        writer.EncodeValue(lf.FilterLevel, 6);
        writer.EncodeValue(lf.SharpnessLevel, 3);
        writer.EncodeValue(lf.ModeRefLfDeltaEnabled ? 1 : 0, 1);

        if (lf.ModeRefLfDeltaEnabled)
        {
            // Decide whether any deltas are non-zero -> need an update emit.
            bool anyDelta = false;
            for (int i = 0; i < Vp8FrameHeaderParser.MaxRefLfDeltas; i++)
                if (lf.RefLfDeltas[i] != 0) { anyDelta = true; break; }
            if (!anyDelta)
            {
                for (int i = 0; i < Vp8FrameHeaderParser.MaxModeLfDeltas; i++)
                    if (lf.ModeLfDeltas[i] != 0) { anyDelta = true; break; }
            }
            writer.EncodeValue(anyDelta ? 1 : 0, 1);

            if (anyDelta)
            {
                for (int i = 0; i < Vp8FrameHeaderParser.MaxRefLfDeltas; i++)
                {
                    int v = lf.RefLfDeltas[i];
                    if (v != 0)
                    {
                        writer.EncodeValue(1, 1);
                        int absV = v < 0 ? -v : v;
                        writer.EncodeValue(absV, 6);
                        writer.EncodeValue(v < 0 ? 1 : 0, 1);
                    }
                    else
                    {
                        writer.EncodeValue(0, 1);
                    }
                }
                for (int i = 0; i < Vp8FrameHeaderParser.MaxModeLfDeltas; i++)
                {
                    int v = lf.ModeLfDeltas[i];
                    if (v != 0)
                    {
                        writer.EncodeValue(1, 1);
                        int absV = v < 0 ? -v : v;
                        writer.EncodeValue(absV, 6);
                        writer.EncodeValue(v < 0 ? 1 : 0, 1);
                    }
                    else
                    {
                        writer.EncodeValue(0, 1);
                    }
                }
            }
        }
    }

    private static void WriteQuantizer(Vp8BoolEncoder writer, Vp8QuantizerIndices q)
    {
        writer.EncodeValue(q.BaseQIndex, 7);
        WriteSignedDelta(writer, q.Y1DcDeltaQ, 4);
        WriteSignedDelta(writer, q.Y2DcDeltaQ, 4);
        WriteSignedDelta(writer, q.Y2AcDeltaQ, 4);
        WriteSignedDelta(writer, q.UvDcDeltaQ, 4);
        WriteSignedDelta(writer, q.UvAcDeltaQ, 4);
    }

    private static void WriteSignedDelta(Vp8BoolEncoder writer, int v, int magBits)
    {
        if (v == 0)
        {
            writer.EncodeValue(0, 1);
            return;
        }
        writer.EncodeValue(1, 1);
        int absV = v < 0 ? -v : v;
        writer.EncodeValue(absV, magBits);
        writer.EncodeValue(v < 0 ? 1 : 0, 1);
    }

    private static void WriteCoefProbUpdates(Vp8BoolEncoder writer, byte[,,,] coefProbs)
    {
        // For each (block, band, ctx, node), if the value differs from the
        // default, emit "update = 1" + the new 8-bit prob; otherwise emit
        // "update = 0".
        int b = Vp8DefaultCoefProbs.BlockTypes;
        int n = Vp8DefaultCoefProbs.CoefBands;
        int c = Vp8DefaultCoefProbs.PrevCoefContexts;
        int e = Vp8DefaultCoefProbs.EntropyNodes;
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < c; k++)
                    for (int l = 0; l < e; l++)
                    {
                        byte updateProb = Vp8CoefUpdateProbs.UpdateProbs[i, j, k, l];
                        if (coefProbs[i, j, k, l] != Vp8DefaultCoefProbs.DefaultProbs[i, j, k, l])
                        {
                            writer.EncodeBool(1, updateProb);
                            writer.EncodeValue(coefProbs[i, j, k, l], 8);
                        }
                        else
                        {
                            writer.EncodeBool(0, updateProb);
                        }
                    }
    }
}
