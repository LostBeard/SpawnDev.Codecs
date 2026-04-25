// Tests for Vp9SegmentationParamsParser (slice 188).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SegmentationParams_Disabled_ParsesSingleFlag()
    {
        var data = BitsToBytes((0, 1));  // enabled=0
        var s = Vp9SegmentationParamsParser.Parse(data);
        Equal(false, s.Enabled);
        Equal(false, s.UpdateMap);
        Equal(false, s.UpdateData);
    }

    [TestMethod]
    public void Vp9SegmentationParams_EnabledNoUpdates_ReadsTwoFlags()
    {
        // enabled=1, update_map=0, update_data=0.
        var data = BitsToBytes((1, 1), (0, 1), (0, 1));
        var s = Vp9SegmentationParamsParser.Parse(data);
        Equal(true, s.Enabled);
        Equal(false, s.UpdateMap);
        Equal(false, s.UpdateData);
    }

    [TestMethod]
    public void Vp9SegmentationParams_UpdateMap_TreeProbsAndTemporalFlag()
    {
        // enabled=1, update_map=1.
        // 7 tree probs: [50, _, 100, _, 200, _, 75] - alternating present/absent.
        // No temporal update -> all 3 pred probs remain at MaxProb=255.
        // update_data=0.
        var data = BitsToBytes(
            (1, 1),                     // enabled
            (1, 1),                     // update_map
            (1, 1), (50, 8),            // tree[0]=50
            (0, 1),                     // tree[1] no update
            (1, 1), (100, 8),           // tree[2]=100
            (0, 1),                     // tree[3] no update
            (1, 1), (200, 8),           // tree[4]=200
            (0, 1),                     // tree[5] no update
            (1, 1), (75, 8),            // tree[6]=75
            (0, 1),                     // temporal_update=0
            (0, 1));                    // update_data=0

        var s = Vp9SegmentationParamsParser.Parse(data);

        Equal(true, s.Enabled);
        Equal(true, s.UpdateMap);
        Equal(7, s.TreeProbsArray.Length);
        Equal((byte)50, s.TreeProbsArray[0]);
        Equal((byte)Vp9SegmentationParams.MaxProb, s.TreeProbsArray[1]);
        Equal((byte)100, s.TreeProbsArray[2]);
        Equal((byte)Vp9SegmentationParams.MaxProb, s.TreeProbsArray[3]);
        Equal((byte)200, s.TreeProbsArray[4]);
        Equal((byte)Vp9SegmentationParams.MaxProb, s.TreeProbsArray[5]);
        Equal((byte)75, s.TreeProbsArray[6]);
        Equal(false, s.TemporalUpdate);
        for (int i = 0; i < 3; i++)
            Equal((byte)Vp9SegmentationParams.MaxProb, s.PredProbs[i]);
    }

    [TestMethod]
    public void Vp9SegmentationParams_UpdateMapWithTemporalUpdate_ReadsPredProbs()
    {
        var data = BitsToBytes(
            (1, 1),                     // enabled
            (1, 1),                     // update_map
            // 7 tree probs all skipped.
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
            (1, 1),                     // temporal_update=1
            (1, 1), (10, 8),            // pred[0]=10
            (0, 1),                     // pred[1] no update -> MaxProb
            (1, 1), (20, 8),            // pred[2]=20
            (0, 1));                    // update_data=0

        var s = Vp9SegmentationParamsParser.Parse(data);

        Equal(true, s.UpdateMap);
        Equal(true, s.TemporalUpdate);
        Equal((byte)10, s.PredProbs[0]);
        Equal((byte)Vp9SegmentationParams.MaxProb, s.PredProbs[1]);
        Equal((byte)20, s.PredProbs[2]);
    }

    [TestMethod]
    public void Vp9SegmentationParams_UpdateData_AllFeaturesDisabled()
    {
        // enabled=1, update_map=0, update_data=1, abs_delta=0.
        // For 8 segments * 4 features = 32 feature_enabled flags, all 0.
        var fields = new (uint, int)[36];
        fields[0] = (1, 1);  // enabled
        fields[1] = (0, 1);  // update_map
        fields[2] = (1, 1);  // update_data
        fields[3] = (0, 1);  // abs_delta
        for (int i = 0; i < 32; i++) fields[4 + i] = (0, 1);

        var data = BitsToBytes(fields);
        var s = Vp9SegmentationParamsParser.Parse(data);

        Equal(true, s.UpdateData);
        Equal(false, s.AbsDelta);
        for (int seg = 0; seg < 8; seg++)
        for (int feat = 0; feat < 4; feat++)
        {
            Equal(false, s.FeatureEnabled[seg, feat]);
            Equal(0, s.FeatureData[seg, feat]);
        }
    }

    [TestMethod]
    public void Vp9SegmentationParams_UpdateData_AltQAndAltLfWithSign()
    {
        // enabled=1, update_map=0, update_data=1, abs_delta=1.
        // Segment 0: ALT_Q=+30, ALT_LF=-5, REF_FRAME and SKIP disabled.
        // All other segments: all features disabled.
        var fields = new System.Collections.Generic.List<(uint, int)>
        {
            (1, 1),  // enabled
            (0, 1),  // update_map
            (1, 1),  // update_data
            (1, 1),  // abs_delta
            // Segment 0
            (1, 1), (30, 8), (0, 1),  // ALT_Q enabled, mag=30, sign=0 -> +30
            (1, 1), (5, 6), (1, 1),   // ALT_LF enabled, mag=5, sign=1 -> -5
            (0, 1),                    // REF_FRAME disabled
            (0, 1),                    // SKIP disabled
        };
        // Segments 1..7: all 4 features disabled.
        for (int seg = 1; seg < 8; seg++)
            for (int feat = 0; feat < 4; feat++)
                fields.Add((0, 1));

        var data = BitsToBytes(fields.ToArray());
        var s = Vp9SegmentationParamsParser.Parse(data);

        Equal(true, s.UpdateData);
        Equal(true, s.AbsDelta);
        Equal(true, s.FeatureEnabled[0, (int)Vp9SegFeature.AltQ]);
        Equal(30, s.FeatureData[0, (int)Vp9SegFeature.AltQ]);
        Equal(true, s.FeatureEnabled[0, (int)Vp9SegFeature.AltLf]);
        Equal(-5, s.FeatureData[0, (int)Vp9SegFeature.AltLf]);
        Equal(false, s.FeatureEnabled[0, (int)Vp9SegFeature.RefFrame]);
        Equal(false, s.FeatureEnabled[0, (int)Vp9SegFeature.Skip]);
    }

    [TestMethod]
    public void Vp9SegmentationParams_UpdateData_RefFrameAndSkipFeatures()
    {
        // Segment 3: REF_FRAME=2 (golden, 2 bits), SKIP enabled (no payload).
        var fields = new System.Collections.Generic.List<(uint, int)>
        {
            (1, 1),  // enabled
            (0, 1),  // update_map
            (1, 1),  // update_data
            (0, 1),  // abs_delta
        };
        // Segments 0..2: all 4 features disabled.
        for (int seg = 0; seg < 3; seg++)
            for (int feat = 0; feat < 4; feat++)
                fields.Add((0, 1));
        // Segment 3
        fields.Add((0, 1));  // ALT_Q disabled
        fields.Add((0, 1));  // ALT_LF disabled
        fields.Add((1, 1)); fields.Add((2, 2));  // REF_FRAME enabled, value=2 (no sign bit, unsigned)
        fields.Add((1, 1));  // SKIP enabled (no payload bits)
        // Segments 4..7
        for (int seg = 4; seg < 8; seg++)
            for (int feat = 0; feat < 4; feat++)
                fields.Add((0, 1));

        var data = BitsToBytes(fields.ToArray());
        var s = Vp9SegmentationParamsParser.Parse(data);

        Equal(true, s.FeatureEnabled[3, (int)Vp9SegFeature.RefFrame]);
        Equal(2, s.FeatureData[3, (int)Vp9SegFeature.RefFrame]);
        Equal(true, s.FeatureEnabled[3, (int)Vp9SegFeature.Skip]);
        Equal(0, s.FeatureData[3, (int)Vp9SegFeature.Skip]);  // no payload
    }
}
