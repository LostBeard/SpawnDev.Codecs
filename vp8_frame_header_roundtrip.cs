// Round-trip test for Vp8FrameHeaderWriter + Vp8FrameHeaderParser.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using SpawnDev.Codecs.Video.Vp8;

int totalChecks = 0;
int totalFails = 0;

void Test(string name, Vp8FrameHeader input)
{
    var enc = new Vp8BoolEncoder();
    Vp8FrameHeaderWriter.WriteKeyFrameHeader(enc, input);
    var bytes = enc.Stop();

    var dec = new Vp8BoolDecoder(bytes);
    var parsed = Vp8FrameHeaderParser.ParseKeyFrameHeader(dec);

    bool ok =
        parsed.ColorSpace == input.ColorSpace &&
        parsed.ClampingType == input.ClampingType &&
        parsed.Segmentation.Enabled == input.Segmentation.Enabled &&
        parsed.LoopFilter.FilterType == input.LoopFilter.FilterType &&
        parsed.LoopFilter.FilterLevel == input.LoopFilter.FilterLevel &&
        parsed.LoopFilter.SharpnessLevel == input.LoopFilter.SharpnessLevel &&
        parsed.Log2NumPartitions == input.Log2NumPartitions &&
        parsed.Quantizer.BaseQIndex == input.Quantizer.BaseQIndex &&
        parsed.Quantizer.Y1DcDeltaQ == input.Quantizer.Y1DcDeltaQ &&
        parsed.Quantizer.Y2DcDeltaQ == input.Quantizer.Y2DcDeltaQ &&
        parsed.Quantizer.Y2AcDeltaQ == input.Quantizer.Y2AcDeltaQ &&
        parsed.Quantizer.UvDcDeltaQ == input.Quantizer.UvDcDeltaQ &&
        parsed.Quantizer.UvAcDeltaQ == input.Quantizer.UvAcDeltaQ &&
        parsed.RefreshEntropyProbs == input.RefreshEntropyProbs &&
        parsed.MbNoSkipCoeffEnabled == input.MbNoSkipCoeffEnabled &&
        parsed.ProbSkipFalse == input.ProbSkipFalse;

    // Verify coef probs match exactly.
    if (ok)
    {
        for (int i = 0; i < Vp8DefaultCoefProbs.BlockTypes && ok; i++)
            for (int j = 0; j < Vp8DefaultCoefProbs.CoefBands && ok; j++)
                for (int k = 0; k < Vp8DefaultCoefProbs.PrevCoefContexts && ok; k++)
                    for (int l = 0; l < Vp8DefaultCoefProbs.EntropyNodes && ok; l++)
                        if (parsed.CoefProbs[i, j, k, l] != input.CoefProbs[i, j, k, l])
                            ok = false;
    }

    totalChecks++;
    if (!ok)
    {
        totalFails++;
        Console.WriteLine($"  FAIL {name}");
    }
    else
    {
        Console.WriteLine($"  OK   {name,-30}: bytes={bytes.Length}");
    }
}

// Build a minimal default header.
Vp8FrameHeader MakeHeader(int baseQ = 30, int filterLevel = 10, bool segEnabled = false, bool refreshEntropy = true,
                           bool mbNoSkip = true, int probSkipFalse = 100, byte[,,,]? customProbs = null,
                           int y1DcDelta = 0, int y2DcDelta = 0, int y2AcDelta = 0, int uvDcDelta = 0, int uvAcDelta = 0)
{
    var probs = customProbs ?? (byte[,,,])Vp8DefaultCoefProbs.DefaultProbs.Clone();
    return new Vp8FrameHeader
    {
        ColorSpace = 0,
        ClampingType = 0,
        Segmentation = new Vp8SegmentationParams
        {
            Enabled = segEnabled,
            UpdateMap = false,
            UpdateData = false,
            AbsDelta = false,
            FeatureData = new int[2, 4],
            SegmentTreeProbs = new byte[3] { 255, 255, 255 },
        },
        LoopFilter = new Vp8LoopFilterParams
        {
            FilterType = 0,
            FilterLevel = filterLevel,
            SharpnessLevel = 0,
            ModeRefLfDeltaEnabled = false,
            RefLfDeltas = new int[4],
            ModeLfDeltas = new int[4],
        },
        Log2NumPartitions = 0,
        Quantizer = new Vp8QuantizerIndices
        {
            BaseQIndex = baseQ,
            Y1DcDeltaQ = y1DcDelta,
            Y2DcDeltaQ = y2DcDelta,
            Y2AcDeltaQ = y2AcDelta,
            UvDcDeltaQ = uvDcDelta,
            UvAcDeltaQ = uvAcDelta,
        },
        RefreshEntropyProbs = refreshEntropy,
        CoefProbs = probs,
        MbNoSkipCoeffEnabled = mbNoSkip,
        ProbSkipFalse = probSkipFalse,
    };
}

Test("Default", MakeHeader());
Test("HighQ", MakeHeader(baseQ: 100, filterLevel: 50));
Test("LowQ", MakeHeader(baseQ: 4, filterLevel: 5));
Test("WithDeltas", MakeHeader(y1DcDelta: 5, y2DcDelta: -3, y2AcDelta: 2, uvDcDelta: -7, uvAcDelta: 4));
Test("NoSkip", MakeHeader(mbNoSkip: false, probSkipFalse: 0));
Test("NoRefreshEntropy", MakeHeader(refreshEntropy: false));

// Custom coef probs - mutate a few entries
{
    var custom = (byte[,,,])Vp8DefaultCoefProbs.DefaultProbs.Clone();
    custom[0, 0, 0, 0] = 200;
    custom[1, 1, 1, 1] = 100;
    custom[3, 7, 2, 10] = 50;
    Test("WithCoefUpdates", MakeHeader(customProbs: custom));
}

Console.WriteLine();
Console.WriteLine($"VP8 frame header round-trip: {totalChecks - totalFails}/{totalChecks} pass");
if (totalFails == 0)
    Console.WriteLine("VP8 frame header writer + parser round-trip BIT-EXACT.");
else
    Environment.Exit(1);
