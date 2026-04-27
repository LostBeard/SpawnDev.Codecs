// Tests for Vp8KeyframeWalker - the macroblock walker that integrates the
// VP8 inverse pipeline (mode info + dequant + IDCT + intra predict + add)
// and writes reconstructed pixels into a Vp8FrameBuffer.
//
// These tests build VP8 keyframe bitstreams using the matching encoder
// (Vp8FrameTagWriter, Vp8FrameHeaderWriter, Vp8BoolEncoder, Vp8CoefBlockEncoder)
// and verify that the walker produces consistent reconstructed YUV output.
// They do NOT depend on ffmpeg - the encoder/decoder pair is the round-trip
// test. The end-to-end ffmpeg-vs-walker comparison is in the demo
// (vp8_keyframe_decode_demo.cs at the repo root).

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8KeyframeWalker_RejectsNonKeyFrame()
    {
        var tag = new Vp8FrameTag
        {
            IsKeyFrame = false,
            Version = Vp8Version.Bicubic,
            ShowFrame = true,
            FirstPartitionSize = 100,
        };
        var hdr = MakeMinimalKeyHeader();
        var bd = new Vp8BoolDecoder(new byte[] { 0x00, 0x00, 0x00, 0x00 });
        var fb = new Vp8FrameBuffer(16, 16);
        var ec = new Vp8EntropyContexts(fb.MbCols);
        Throws<NotImplementedException>(() =>
            Vp8KeyframeWalker.Decode(tag, hdr, bd, new byte[16], fb, ec));
    }

    [TestMethod]
    public void Vp8KeyframeWalker_RejectsMultiPartition()
    {
        var tag = MakeMinimalKeyTag(64, 64);
        var hdr = MakeMinimalKeyHeader() with
        {
            Log2NumPartitions = 1, // 2 partitions - out of scope for this slice
        };
        var bd = new Vp8BoolDecoder(new byte[] { 0x00, 0x00, 0x00, 0x00 });
        var fb = new Vp8FrameBuffer(64, 64);
        var ec = new Vp8EntropyContexts(fb.MbCols);
        Throws<NotImplementedException>(() =>
            Vp8KeyframeWalker.Decode(tag, hdr, bd, new byte[16], fb, ec));
    }

    [TestMethod]
    public void Vp8KeyframeWalker_RejectsMismatchedEntropyContextsCols()
    {
        var tag = MakeMinimalKeyTag(64, 64);
        var hdr = MakeMinimalKeyHeader();
        var bd = new Vp8BoolDecoder(new byte[] { 0x00, 0x00, 0x00, 0x00 });
        var fb = new Vp8FrameBuffer(64, 64); // mbCols = 4
        var ec = new Vp8EntropyContexts(8);  // mismatch!
        Throws<ArgumentException>(() =>
            Vp8KeyframeWalker.Decode(tag, hdr, bd, new byte[16], fb, ec));
    }

    [TestMethod]
    public void Vp8KeyframeWalker_AllSkipMb_ReconstructsBaselinePrediction()
    {
        // Build a minimal frame where every MB is DcPred + skip. The walker
        // should produce a flat-128 luma plane (DC predicted with no neighbors)
        // and chroma equally.
        const int Width = 32, Height = 32; // 2x2 MB grid
        const int MbCols = 2, MbRows = 2;

        var tag = MakeMinimalKeyTag(Width, Height);
        var hdr = MakeMinimalKeyHeader();

        // Encode mode info: 4 MBs, each with DcPred + skip=true.
        // mbNoSkipCoeffEnabled=true (in MakeMinimalKeyHeader). probSkipFalse=128.
        // Mode info per MB:
        //   skip_coeff = 1 (1 bit at probSkipFalse=128)
        //   yMode = DcPred: KfYModeTree walk - bit=1 (not B_PRED, prob 145), then bit=0 (DcPred branch, prob 163)
        //   uvMode = DcPred: UvModeTree walk - bit=0 (DcPred leaf at index 0, prob 142)
        var modeEnc = new Vp8BoolEncoder();
        for (int i = 0; i < MbCols * MbRows; i++)
        {
            modeEnc.EncodeBool(1, hdr.ProbSkipFalse); // skip_coeff = 1
            // yMode = DcPred: KfYModeTree {-BPred, 2, 4, 6, -DcPred, -VPred, -HPred, -TmPred}
            //   probs = {145, 156, 163, 128}
            //   i=0 -> bit=1 (not BPred, prob 145), tree[1]=2 -> i=2
            //   i=2 -> bit=0 (left subtree, prob 156), tree[2]=4 -> i=4
            //   i=4 -> bit=0 (DcPred leaf, prob 163)
            modeEnc.EncodeBool(1, Vp8ModeTrees.DefaultKfYModeProb[0]); // not B_PRED
            modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfYModeProb[1]); // left subtree
            modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfYModeProb[2]); // DcPred
            // uvMode = DcPred: UvModeTree {-DcPred, 2, ...}, probs = {142, ...}
            //   i=0 -> bit=0 (DcPred leaf, prob 142)
            modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfUvModeProb[0]); // DcPred leaf
        }
        var modeBytes = modeEnc.Stop();

        // Token partition is empty since all MBs are skipped.
        var tokenBytes = new byte[] { 0x00, 0x00, 0x00, 0x00 };

        var bd = new Vp8BoolDecoder(modeBytes);
        var fb = new Vp8FrameBuffer(Width, Height);
        var ec = new Vp8EntropyContexts(fb.MbCols);

        Vp8KeyframeWalker.Decode(tag, hdr, bd, tokenBytes, fb, ec);

        // For an all-skip + all-DcPred frame:
        //   MB 0,0 (no above, no left): DC = 128, output = 128
        //   MB 1,0 (no above, has left=128): DC = (16*128+128*16+16)>>5 wait left has 16 samples=128 each, no above so DC = (sum_left + 8) >> 4 = (16*128+8)>>4 = 128
        //   MB 0,1 (has above=128, no left): DC = (sum_above + 8) >> 4 = 128
        //   MB 1,1 (has above=128, has left=128): DC = (sum + 16) >> 5 = 128
        // So entire frame should be 128.
        for (int r = 0; r < Height; r++)
            for (int c = 0; c < Width; c++)
                Equal((byte)128, fb.YPlane[r * fb.YStride + c], $"Y[{r},{c}]");
        // Same for U and V.
        for (int r = 0; r < Height / 2; r++)
            for (int c = 0; c < Width / 2; c++)
            {
                Equal((byte)128, fb.UPlane[r * fb.UvStride + c], $"U[{r},{c}]");
                Equal((byte)128, fb.VPlane[r * fb.UvStride + c], $"V[{r},{c}]");
            }
    }

    [TestMethod]
    public void Vp8KeyframeWalker_SingleMbDcPred_NoResidual_ReconstructsBaseline()
    {
        // 1 MB, DcPred, skip_coeff=true. Y/U/V should all be 128.
        const int Width = 16, Height = 16;
        var tag = MakeMinimalKeyTag(Width, Height);
        var hdr = MakeMinimalKeyHeader();

        var modeEnc = new Vp8BoolEncoder();
        modeEnc.EncodeBool(1, hdr.ProbSkipFalse); // skip
        modeEnc.EncodeBool(1, Vp8ModeTrees.DefaultKfYModeProb[0]); // not B_PRED
        modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfYModeProb[1]); // left subtree
        modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfYModeProb[2]); // DcPred
        modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfUvModeProb[0]); // UV DcPred
        var modeBytes = modeEnc.Stop();
        var tokenBytes = new byte[] { 0x00, 0x00, 0x00, 0x00 };

        var bd = new Vp8BoolDecoder(modeBytes);
        var fb = new Vp8FrameBuffer(Width, Height);
        var ec = new Vp8EntropyContexts(fb.MbCols);

        Vp8KeyframeWalker.Decode(tag, hdr, bd, tokenBytes, fb, ec);

        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                Equal((byte)128, fb.YPlane[i * fb.YStride + j]);
    }

    [TestMethod]
    public void Vp8KeyframeWalker_SingleMb_NoSkip_ZeroResidual_StillBaseline()
    {
        // 1 MB, DcPred, skip_coeff=false but all coef blocks empty.
        // Y2 emits "block empty" bit; each Y4 emits "empty" with firstCoef=1;
        // each UV emits "empty". Total: 1 (Y2) + 16 (Y4) + 8 (UV) = 25 bits.
        const int Width = 16, Height = 16;
        var tag = MakeMinimalKeyTag(Width, Height);
        var hdr = MakeMinimalKeyHeader();

        var modeEnc = new Vp8BoolEncoder();
        modeEnc.EncodeBool(0, hdr.ProbSkipFalse); // skip = false (mb has tokens)
        modeEnc.EncodeBool(1, Vp8ModeTrees.DefaultKfYModeProb[0]); // not B_PRED
        modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfYModeProb[1]); // left subtree
        modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfYModeProb[2]); // DcPred
        modeEnc.EncodeBool(0, Vp8ModeTrees.DefaultKfUvModeProb[0]); // UV DcPred
        var modeBytes = modeEnc.Stop();

        // Encode all-empty coef blocks for the MB. ctx = 0 for all (first MB).
        var coefEnc = new Vp8BoolEncoder();
        var probs = hdr.CoefProbs;
        // Y2 (block_type=1, firstCoef=0): emit "block empty" = bit 0 at probs[1, kBands[0]=0, ctx=0, 0]
        coefEnc.EncodeBool(0, probs[1, 0, 0, 0]);
        // 16 Y4 blocks (block_type=0, firstCoef=1): emit "block empty" each at probs[0, kBands[1]=1, ctx=0, 0]
        for (int i = 0; i < 16; i++)
            coefEnc.EncodeBool(0, probs[0, 1, 0, 0]);
        // 8 UV blocks (block_type=2, firstCoef=0): emit "block empty" each at probs[2, 0, 0, 0]
        for (int i = 0; i < 8; i++)
            coefEnc.EncodeBool(0, probs[2, 0, 0, 0]);
        var tokenBytes = coefEnc.Stop();

        var bd = new Vp8BoolDecoder(modeBytes);
        var fb = new Vp8FrameBuffer(Width, Height);
        var ec = new Vp8EntropyContexts(fb.MbCols);

        Vp8KeyframeWalker.Decode(tag, hdr, bd, tokenBytes, fb, ec);

        // All-zero residual should give just the prediction (DC=128 for unbordered MB).
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                Equal((byte)128, fb.YPlane[i * fb.YStride + j]);
    }

    [TestMethod]
    public void Vp8KeyframeWalker_FrameBufferDimensions_RoundsToMb()
    {
        // Walker should accept non-MB-aligned dimensions and store strides
        // accordingly. Check via the buffer's reported strides.
        var fb = new Vp8FrameBuffer(17, 17);
        Equal(2, fb.MbCols);  // ceil(17/16) = 2
        Equal(2, fb.MbRows);  // ceil(17/16) = 2
        True(fb.YStride >= 32, $"YStride {fb.YStride} should be >= 32 (2 MB cols)");
    }

    // ---------------- Test helpers ----------------

    private static Vp8FrameTag MakeMinimalKeyTag(int width, int height) => new()
    {
        IsKeyFrame = true,
        Version = Vp8Version.Bicubic,
        ShowFrame = true,
        FirstPartitionSize = 100, // not validated by walker
        Width = width,
        Height = height,
        HorizontalScale = 0,
        VerticalScale = 0,
    };

    private static Vp8FrameHeader MakeMinimalKeyHeader()
    {
        // Initialize coef probs from the default table (4D copy).
        var probs = new byte[
            Vp8DefaultCoefProbs.BlockTypes,
            Vp8DefaultCoefProbs.CoefBands,
            Vp8DefaultCoefProbs.PrevCoefContexts,
            Vp8DefaultCoefProbs.EntropyNodes];
        for (int i = 0; i < Vp8DefaultCoefProbs.BlockTypes; i++)
            for (int j = 0; j < Vp8DefaultCoefProbs.CoefBands; j++)
                for (int k = 0; k < Vp8DefaultCoefProbs.PrevCoefContexts; k++)
                    for (int l = 0; l < Vp8DefaultCoefProbs.EntropyNodes; l++)
                        probs[i, j, k, l] = Vp8DefaultCoefProbs.DefaultProbs[i, j, k, l];

        return new Vp8FrameHeader
        {
            ColorSpace = 0,
            ClampingType = 0,
            Segmentation = new Vp8SegmentationParams
            {
                Enabled = false,
                UpdateMap = false,
                UpdateData = false,
                AbsDelta = false,
                FeatureData = new int[2, 4],
                SegmentTreeProbs = new byte[] { 255, 255, 255 },
            },
            LoopFilter = new Vp8LoopFilterParams
            {
                FilterType = 0,
                FilterLevel = 0,
                SharpnessLevel = 0,
                ModeRefLfDeltaEnabled = false,
                RefLfDeltas = new int[4],
                ModeLfDeltas = new int[4],
            },
            Log2NumPartitions = 0,
            Quantizer = new Vp8QuantizerIndices
            {
                BaseQIndex = 4,
                Y1DcDeltaQ = 0,
                Y2DcDeltaQ = 0,
                Y2AcDeltaQ = 0,
                UvDcDeltaQ = 0,
                UvAcDeltaQ = 0,
            },
            RefreshEntropyProbs = false,
            CoefProbs = probs,
            MbNoSkipCoeffEnabled = true,
            ProbSkipFalse = 128,
        };
    }
}
