// Tests for Vp9KeyframeEncoder. Encodes a fresh YUV420 frame, then
// drives it back through the production decoder pipeline + Vp9KeyframeWalker
// to confirm pixels round-trip within Q-induced tolerance.

using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private sealed class IgnoreVp9SinkForEncoder : IVideoFrameSink
    {
        public ValueTask OnFrameAsync(
            ReadOnlyMemory<byte> y, int ys,
            ReadOnlyMemory<byte> u, int us,
            ReadOnlyMemory<byte> v, int vs,
            long pts) => ValueTask.CompletedTask;
    }

    private static (byte[] Y, byte[] U, byte[] V) MakeFlatYuv420(int width, int height,
        byte yValue, byte uValue, byte vValue)
    {
        var y = new byte[width * height];
        var u = new byte[(width / 2) * (height / 2)];
        var v = new byte[(width / 2) * (height / 2)];
        Array.Fill(y, yValue);
        Array.Fill(u, uValue);
        Array.Fill(v, vValue);
        return (y, u, v);
    }

    /// <summary>
    /// Decode an encoded frame via the production pipeline + walker.
    /// Returns the reconstructed YUV planes.
    /// </summary>
    private static async Task<Vp9FrameBuffer> DecodeViaWalker(byte[] frameBytes)
    {
        await using var decoder = new Vp9Decoder();
        await decoder.DecodeFrameAsync(frameBytes, new IgnoreVp9SinkForEncoder());
        var walker = new Vp9KeyframeWalker();
        return walker.DecodeFrame(
            frameBytes,
            decoder.LastCompleteHeader!,
            decoder.LastCompressedState!,
            decoder.LastCompressedResult!,
            decoder.LastTileGroup!);
    }

    /// <summary>
    /// Smallest valid frame: 16x16 flat black. Verify the encoder emits
    /// the VP9 frame_marker (0b10 in top 2 bits => byte starts with
    /// 0b10xxxxxx) and the sync code at the right offset.
    /// </summary>
    [TestMethod]
    public void Vp9KeyframeEncoder_EncodesFlat16x16Black_HasValidSyncCode()
    {
        var (y, u, v) = MakeFlatYuv420(16, 16, yValue: 16, uValue: 128, vValue: 128);
        byte[] frame = Vp9KeyframeEncoder.EncodeKeyFrame(
            y, ySrcStride: 16,
            u, uvSrcStride: 8,
            v,
            width: 16, height: 16,
            baseQIndex: 30);

        True(frame.Length > 0, "encoded frame must be non-empty");
        // First byte: frame_marker(2) profile_low(1) profile_high(1)
        // show_existing_frame(1) frame_type(1) show_frame(1) error_resilient(1)
        // = 1 0 0 0 0 0 1 0 = 0x82 (key frame, show_frame=1, others=0)
        Equal((byte)0x82, frame[0], "first byte must encode frame_marker + key frame + show_frame");

        // Sync code 0x49 0x83 0x42 must appear starting at bit offset 8
        // (right after first byte). Since first byte ended on a byte
        // boundary (8 bits used), sync occupies bytes 1, 2, 3.
        Equal((byte)Vp9SyncCode.Byte0, frame[1], "sync byte 0");
        Equal((byte)Vp9SyncCode.Byte1, frame[2], "sync byte 1");
        Equal((byte)Vp9SyncCode.Byte2, frame[3], "sync byte 2");
    }

    /// <summary>
    /// 16x16 flat-black frame round-trips through encode -> decode.
    /// At low Q the reconstructed luma value should be very close to
    /// the input (within a couple of units).
    /// </summary>
    [TestMethod]
    public async Task Vp9KeyframeEncoder_FlatBlack16x16_RoundTripsViaWalker()
    {
        const byte ySrcVal = 16;
        const byte uSrcVal = 128;
        const byte vSrcVal = 128;
        var (ySrc, uSrc, vSrc) = MakeFlatYuv420(16, 16, ySrcVal, uSrcVal, vSrcVal);

        byte[] frame = Vp9KeyframeEncoder.EncodeKeyFrame(
            ySrc, ySrcStride: 16,
            uSrc, uvSrcStride: 8,
            vSrc,
            width: 16, height: 16,
            baseQIndex: 20);

        var fb = await DecodeViaWalker(frame);
        Equal(16, fb.LumaWidth);
        Equal(16, fb.LumaHeight);
        Equal(8, fb.ChromaWidth);
        Equal(8, fb.ChromaHeight);

        // Verify the reconstructed planes are within tolerance of the source.
        int maxYErr = 0, maxUErr = 0, maxVErr = 0;
        for (int i = 0; i < fb.Y.Length; i++)
            maxYErr = Math.Max(maxYErr, Math.Abs(fb.Y[i] - ySrcVal));
        for (int i = 0; i < fb.U.Length; i++)
            maxUErr = Math.Max(maxUErr, Math.Abs(fb.U[i] - uSrcVal));
        for (int i = 0; i < fb.V.Length; i++)
            maxVErr = Math.Max(maxVErr, Math.Abs(fb.V[i] - vSrcVal));
        True(maxYErr <= 8, $"Y max error = {maxYErr}, expected <= 8");
        True(maxUErr <= 8, $"U max error = {maxUErr}, expected <= 8");
        True(maxVErr <= 8, $"V max error = {maxVErr}, expected <= 8");
    }

    /// <summary>
    /// 32x32 flat gray frame round-trips. Larger frame triggers
    /// recursive 64x64 -> 32x32 -> 16x16 partition decomposition.
    /// </summary>
    [TestMethod]
    public async Task Vp9KeyframeEncoder_FlatGray32x32_RoundTripsViaWalker()
    {
        const byte ySrcVal = 128;
        const byte uSrcVal = 128;
        const byte vSrcVal = 128;
        var (ySrc, uSrc, vSrc) = MakeFlatYuv420(32, 32, ySrcVal, uSrcVal, vSrcVal);

        byte[] frame = Vp9KeyframeEncoder.EncodeKeyFrame(
            ySrc, ySrcStride: 32,
            uSrc, uvSrcStride: 16,
            vSrc,
            width: 32, height: 32,
            baseQIndex: 20);

        var fb = await DecodeViaWalker(frame);
        Equal(32, fb.LumaWidth);
        Equal(32, fb.LumaHeight);
        Equal(16, fb.ChromaWidth);
        Equal(16, fb.ChromaHeight);

        int maxYErr = 0, maxUErr = 0, maxVErr = 0;
        for (int i = 0; i < fb.Y.Length; i++)
            maxYErr = Math.Max(maxYErr, Math.Abs(fb.Y[i] - ySrcVal));
        for (int i = 0; i < fb.U.Length; i++)
            maxUErr = Math.Max(maxUErr, Math.Abs(fb.U[i] - uSrcVal));
        for (int i = 0; i < fb.V.Length; i++)
            maxVErr = Math.Max(maxVErr, Math.Abs(fb.V[i] - vSrcVal));
        True(maxYErr <= 8, $"Y max error = {maxYErr}, expected <= 8");
        True(maxUErr <= 8, $"U max error = {maxUErr}, expected <= 8");
        True(maxVErr <= 8, $"V max error = {maxVErr}, expected <= 8");
    }

    /// <summary>
    /// 32x32 vertical gradient round-trips via the walker. Each row
    /// has a different luma value; reconstruction should preserve the
    /// gradient within Q-induced tolerance. Exercises the forward DCT
    /// and quantizer paths on non-trivial input.
    /// </summary>
    [TestMethod]
    public async Task Vp9KeyframeEncoder_Gradient32x32_RoundTripsWithinTolerance()
    {
        const int W = 32, H = 32;
        var ySrc = new byte[W * H];
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                ySrc[r * W + c] = (byte)(40 + r * 4);  // 40, 44, 48, ... 164
        var uSrc = new byte[(W / 2) * (H / 2)];
        var vSrc = new byte[(W / 2) * (H / 2)];
        Array.Fill(uSrc, (byte)128);
        Array.Fill(vSrc, (byte)128);

        byte[] frame = Vp9KeyframeEncoder.EncodeKeyFrame(
            ySrc, ySrcStride: W,
            uSrc, uvSrcStride: W / 2,
            vSrc,
            width: W, height: H,
            baseQIndex: 30);

        var fb = await DecodeViaWalker(frame);
        Equal(W, fb.LumaWidth);
        Equal(H, fb.LumaHeight);

        int maxYErr = 0;
        for (int i = 0; i < fb.Y.Length; i++)
            maxYErr = Math.Max(maxYErr, Math.Abs(fb.Y[i] - ySrc[i]));
        True(maxYErr <= 16,
            $"gradient Y max error = {maxYErr}, expected <= 16 at Q=30");
    }

    /// <summary>
    /// Sanity: the encoded output is reasonably compact (under a few
    /// KB for a flat 16x16 black frame).
    /// </summary>
    [TestMethod]
    public void Vp9KeyframeEncoder_FlatBlack16x16_OutputIsCompact()
    {
        var (y, u, v) = MakeFlatYuv420(16, 16, yValue: 16, uValue: 128, vValue: 128);
        byte[] frame = Vp9KeyframeEncoder.EncodeKeyFrame(
            y, ySrcStride: 16,
            u, uvSrcStride: 8,
            v,
            width: 16, height: 16,
            baseQIndex: 30);

        // 16x16 raw YUV420 = 384 bytes; flat encoded keyframe should
        // beat that easily. A few hundred bytes max.
        True(frame.Length < 512,
            $"flat 16x16 keyframe size = {frame.Length}, expected < 512 bytes");
    }
}
