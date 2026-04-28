// AV1 keyframe encoder tests. Confirms the encoder produces a bitstream that
// our own walker decoder accepts and reconstructs to within reasonable
// quantization tolerance of the source. Bit-exact ffmpeg / libdav1d
// validation lives in Plans/PLAN-Av1-Encoder-Validation.md - tracked
// follow-up while the entropy round-trip is being polished.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1KeyframeEncoder_FlatGray16x16_HeadersParseAndDecode()
    {
        const int W = 16, H = 16;
        var ySrc = new byte[W * H];
        var uSrc = new byte[(W / 2) * (H / 2)];
        var vSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(ySrc, (byte)128);
        Array.Fill(uSrc, (byte)128);
        Array.Fill(vSrc, (byte)128);

        byte[] frame = Av1KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);

        // Parse SH + Frame OBU and confirm self-consistency.
        Av1SequenceHeader? sh = null;
        Av1Obu? frameObu = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(frame))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
                sh = Av1SequenceHeaderParser.Parse(frame.AsSpan(obu.PayloadOffset, obu.PayloadLength));
            else if (obu.Type == Av1ObuType.Frame)
                frameObu = obu;
        }
        NotNull(sh);
        True(frameObu.HasValue, "Frame OBU not produced");

        Equal(0, sh!.SeqProfile);
        Equal(W, sh.MaxFrameWidth);
        Equal(H, sh.MaxFrameHeight);
        Equal(8, sh.BitDepth);
        Equal(1, sh.SubsamplingX);
        Equal(1, sh.SubsamplingY);
        False(sh.Monochrome);

        var fp = frame.AsMemory(frameObu!.Value.PayloadOffset, frameObu.Value.PayloadLength);
        var cfh = Av1CompleteFrameHeaderParser.Parse(fp.Span, sh);
        Equal(Av1TxMode.Largest, cfh.TxMode);
        True(cfh.ReducedTxSetUsed);
        Equal(1, cfh.TileInfo.TileCols);
        Equal(1, cfh.TileInfo.TileRows);
        Equal(32, cfh.Quant.BaseQindex);
    }

    [TestMethod]
    public void Av1KeyframeEncoder_FlatGray16x16_RoundTripsThroughWalker()
    {
        const int W = 16, H = 16;
        var ySrc = new byte[W * H];
        var uSrc = new byte[(W / 2) * (H / 2)];
        var vSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(ySrc, (byte)128);
        Array.Fill(uSrc, (byte)128);
        Array.Fill(vSrc, (byte)128);

        byte[] frame = Av1KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);

        Av1SequenceHeader? sh = null;
        Av1Obu? frameObu = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(frame))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
                sh = Av1SequenceHeaderParser.Parse(frame.AsSpan(obu.PayloadOffset, obu.PayloadLength));
            else if (obu.Type == Av1ObuType.Frame)
                frameObu = obu;
        }

        NotNull(sh);
        var fp = frame.AsMemory(frameObu!.Value.PayloadOffset, frameObu.Value.PayloadLength);
        var cfh = Av1CompleteFrameHeaderParser.Parse(fp.Span, sh!);

        var tile = new Av1TileBuffer(0, 0,
            frameObu.Value.PayloadOffset + cfh.HeaderSizeBytes,
            fp.Length - cfh.HeaderSizeBytes);
        var tg = new Av1TileGroup
        {
            Tiles = new List<Av1TileBuffer> { tile },
            StartTile = 0,
            EndTile = 0,
        };

        var walker = new Av1KeyframeWalker();
        var fb = walker.DecodeFrame(frame, sh!, cfh, tg);

        Equal(W, fb.LumaWidth);
        Equal(H, fb.LumaHeight);
        Equal(W / 2, fb.ChromaWidth);
        Equal(H / 2, fb.ChromaHeight);

        // Flat gray should reconstruct to within ~ a few units of 128.
        double yMean = ComputeMeanBytes(fb.Y);
        double uMean = ComputeMeanBytes(fb.U);
        double vMean = ComputeMeanBytes(fb.V);
        const double tolerance = 4.0;
        True(Math.Abs(yMean - 128.0) < tolerance, $"Y mean {yMean:F2} not within {tolerance} of 128");
        True(Math.Abs(uMean - 128.0) < tolerance, $"U mean {uMean:F2} not within {tolerance} of 128");
        True(Math.Abs(vMean - 128.0) < tolerance, $"V mean {vMean:F2} not within {tolerance} of 128");
    }

    [TestMethod]
    public void Av1KeyframeEncoder_OutputWrapsValidIvf()
    {
        const int W = 32, H = 32;
        var ySrc = new byte[W * H];
        var uSrc = new byte[(W / 2) * (H / 2)];
        var vSrc = new byte[(W / 2) * (H / 2)];
        // Gradient pattern (gentle, so quantizer doesn't trash everything).
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                ySrc[y * W + x] = (byte)(96 + ((x + y) & 0x1F));
        Array.Fill(uSrc, (byte)128);
        Array.Fill(vSrc, (byte)128);

        byte[] frame = Av1KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);

        // IVF wrap should accept it.
        using var ms = new MemoryStream();
        var writer = new IvfWriter(ms, "AV01", W, H);
        writer.WriteFrame(frame, pts: 0);
        writer.Finish();
        byte[] ivf = ms.ToArray();
        True(ivf.Length > 32, $"IVF too short: {ivf.Length}");

        // Walk the IVF and confirm the embedded frame is byte-identical.
        var first = IvfReader.EnumerateFrames(ivf).First();
        EqualBytes(frame, first.Data.ToArray());
    }

    private static double ComputeMeanBytes(byte[] data)
    {
        if (data.Length == 0) return 0;
        long sum = 0;
        for (int i = 0; i < data.Length; i++) sum += data[i];
        return (double)sum / data.Length;
    }
}
