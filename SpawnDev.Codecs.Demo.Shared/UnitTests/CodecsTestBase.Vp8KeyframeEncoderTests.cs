// Tests for Vp8KeyframeEncoder - the keyframe encoder that lays down a
// complete VP8 keyframe (frame tag + first partition + token partition)
// for a single YUV420 input. Round-trip pairing with Vp8KeyframeWalker
// proves bitstream layout matches what our decoder reads (and, by
// extension, what libvpx + ffmpeg's native VP8 decoder read).

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8KeyframeEncoder_FlatLuma_RoundTripsViaWalker()
    {
        // Sweep flat luma values through Encode -> Walker decode. With the
        // Y2 PLANE_TYPE bug (formerly type 3, fixed to type 1) the decode
        // would produce -1353 instead of -76 for luma=64, leading to
        // ~10 instead of 64 in the output Y plane. After the fix every
        // flat value round-trips within a couple of quantization steps.
        const int W = 16, H = 16;
        var uSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(vSrc, (byte)128);

        foreach (int luma in new int[] { 64, 96, 110, 128, 144, 160, 192 })
        {
            var ySrc = new byte[W * H];
            Array.Fill(ySrc, (byte)luma);

            var frame = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);

            var (fb, _) = DecodeFrameViaWalker(frame, W, H);

            int sum = 0, min = 255, max = 0;
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                {
                    byte b = fb.YPlane[r * fb.YStride + c];
                    sum += b;
                    if (b < min) min = b;
                    if (b > max) max = b;
                }
            int mean = sum / (W * H);
            True(Math.Abs(mean - luma) <= 4, $"luma={luma}: decoded mean={mean} (min={min} max={max}) - expected within +/-4 of source");
        }
    }

    [TestMethod]
    public void Vp8KeyframeEncoder_MultiMbGradient_RoundTripsViaWalker()
    {
        // 32x32 = 2x2 MB grid. Without the encoder reconstruction write-back
        // (which mirrors the decoder's inverse pipeline back into recon),
        // MB(0,1) onwards used 127/129 edge fills as predictor, while the
        // decoder used the actual reconstructed neighbors. The drift made
        // multi-MB frames decode to ~200 mean instead of ~110 for a gradient
        // with source mean 110.
        const int W = 32, H = 32;
        var ySrc = new byte[W * H];
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                ySrc[r * W + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(2.0 * Math.PI * c / W) + r * 2, 0, 255);
        var uSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(vSrc, (byte)128);

        int srcSum = 0;
        foreach (var b in ySrc) srcSum += b;
        int srcMean = srcSum / (W * H);

        var frame = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
        var (fb, _) = DecodeFrameViaWalker(frame, W, H);

        int decSum = 0;
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                decSum += fb.YPlane[r * fb.YStride + c];
        int decMean = decSum / (W * H);

        True(Math.Abs(decMean - srcMean) <= 6,
            $"decoded mean={decMean}, source mean={srcMean} - expected within +/-6 (lossy at Q=30 but not amplified)");
    }

    [TestMethod]
    public void Vp8KeyframeEncoder_UsesPlaneType1ForY2_NotPlaneType3()
    {
        // Direct check of the Y2 block type bug. Decode the Y2 coefficient
        // block from the encoded frame using BOTH PLANE_TYPE = 1 (correct,
        // libvpx) and PLANE_TYPE = 3 (the original buggy choice). After the
        // fix, the type-1 decode produces the value the encoder intended
        // (-76 for luma=64 at Q=30 with predictor=128); the type-3 decode
        // produces garbage. Before the fix the polarity was reversed.
        const int W = 16, H = 16;
        var ySrc = new byte[W * H];
        Array.Fill(ySrc, (byte)64);
        var uSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(vSrc, (byte)128);
        var frame = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);

        // Locate the token partition by parsing the tag + first partition.
        var tag = Vp8FrameTagParser.Parse(frame.AsSpan());
        int firstPartOffset = 10;
        int firstPartLen = tag.FirstPartitionSize;
        var firstPart = new byte[firstPartLen];
        Buffer.BlockCopy(frame, firstPartOffset, firstPart, 0, firstPartLen);
        var hdrBd = new Vp8BoolDecoder(firstPart);
        Vp8FrameHeaderParser.ParseKeyFrameHeader(hdrBd);
        int tokenOffset = firstPartOffset + firstPartLen;
        var tokenBytes = new byte[frame.Length - tokenOffset];
        Buffer.BlockCopy(frame, tokenOffset, tokenBytes, 0, tokenBytes.Length);

        // Decode Y2 block using PLANE_TYPE 1 (libvpx PLANE_TYPE_Y2).
        var probsType1 = SliceBlockType(Vp8DefaultCoefProbs.DefaultProbs, 1);
        var bd1 = new Vp8BoolDecoder(tokenBytes);
        var y2_via_type1 = new short[16];
        Vp8CoefBlockDecoder.Decode(bd1, probsType1, 0, 0, y2_via_type1);

        // -76 is the expected qcoef[0] for luma=64 (predictor=128) at Q=30
        // (residual=-64, fdct[0]=-512, walsh[0]~=-4095, /Y2Dc(54) ~ -76).
        Equal((short)-76, y2_via_type1[0], "Y2 qcoef[0] at PLANE_TYPE=1 must equal the expected -76");
    }

    [TestMethod]
    public void Vp8KeyframeEncoder_RejectsBaseQIndex_OutOfRange()
    {
        // BaseQIndex is a 7-bit field per RFC 6386 sec 9.6 (range 0..127).
        // Q >= 128 wraps to 7 bits in the bitstream while the encoder uses
        // the original value internally => decoder + encoder use different
        // quantizers => PSNR collapses to ~9 dB. The encoder must reject at
        // the API boundary so this never silently happens.
        const int W = 16, H = 16;
        var ySrc = new byte[W * H];
        Array.Fill(ySrc, (byte)128);
        var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);

        bool threw128 = false;
        try { Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 128); }
        catch (ArgumentOutOfRangeException) { threw128 = true; }
        True(threw128, "Q=128 (overflow first invalid value) must throw");

        bool threwNeg = false;
        try { Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: -1); }
        catch (ArgumentOutOfRangeException) { threwNeg = true; }
        True(threwNeg, "Q=-1 must throw");

        // 0 and 127 must be accepted (boundaries of the legal range).
        var atZero = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 0);
        True(atZero.Length > 0);
        var atMax = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 127);
        True(atMax.Length > 0);
    }

    /// <summary>Decode a complete VP8 keyframe via Vp8KeyframeWalker and return the buffer + entropy contexts.</summary>
    private static (Vp8FrameBuffer fb, Vp8EntropyContexts ec) DecodeFrameViaWalker(byte[] frame, int width, int height)
    {
        var tag = Vp8FrameTagParser.Parse(frame.AsSpan());
        int firstPartOffset = 10;
        int firstPartLen = tag.FirstPartitionSize;
        var firstPart = new byte[firstPartLen];
        Buffer.BlockCopy(frame, firstPartOffset, firstPart, 0, firstPartLen);
        var bd = new Vp8BoolDecoder(firstPart);
        var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(bd);
        int tokenOffset = firstPartOffset + firstPartLen;
        var tokenBytes = new byte[frame.Length - tokenOffset];
        Buffer.BlockCopy(frame, tokenOffset, tokenBytes, 0, tokenBytes.Length);

        var fb = new Vp8FrameBuffer(width, height);
        var ec = new Vp8EntropyContexts(fb.MbCols);
        Vp8KeyframeWalker.Decode(tag, hdr, bd, tokenBytes, fb, ec);
        return (fb, ec);
    }

    /// <summary>Slice [block_type][band][ctx][node] -> [band][ctx][node] for one block type.</summary>
    private static byte[,,] SliceBlockType(byte[,,,] coefProbs, int blockType)
    {
        int b = coefProbs.GetLength(1);
        int c = coefProbs.GetLength(2);
        int e = coefProbs.GetLength(3);
        var slice = new byte[b, c, e];
        for (int j = 0; j < b; j++)
            for (int k = 0; k < c; k++)
                for (int l = 0; l < e; l++)
                    slice[j, k, l] = coefProbs[blockType, j, k, l];
        return slice;
    }
}
