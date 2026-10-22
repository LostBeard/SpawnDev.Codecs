// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// One-call CPU reference decode of a complete VP8 keyframe (frame tag + header + modes + tokens + loop filter) to tightly
// packed, cropped Y / U / V planes. Tests compare the accelerator decoder against this AND against ffmpeg.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>CPU reference keyframe decode.</summary>
public static class Vp8KeyframeReference
{
    /// <summary>Decoded planes, tightly packed: Y is width*height, U and V are ceil(w/2)*ceil(h/2).</summary>
    public sealed record Planes(byte[] Y, byte[] U, byte[] V, int Width, int Height);

    /// <summary>Decode one VP8 keyframe (raw frame bytes, tag included).</summary>
    public static Planes Decode(byte[] frame)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var tag = Vp8FrameTagParser.Parse(frame);
        if (!tag.IsKeyFrame) throw new InvalidDataException("Not a key frame.");
        int width = tag.Width!.Value, height = tag.Height!.Value;
        if (width == 0 || height == 0) throw new InvalidDataException("VP8 keyframe signals a zero-sized picture.");
        int p0 = tag.FirstPartitionSize;
        if (10 + p0 > frame.Length) throw new InvalidDataException("First partition size exceeds frame length.");

        var firstPart = new byte[p0];
        Buffer.BlockCopy(frame, 10, firstPart, 0, p0);
        var modeReader = new Vp8BoolDecoder(firstPart);
        var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(modeReader);

        int tokenOffset = 10 + p0;
        var tokenPart = new byte[frame.Length - tokenOffset];
        Buffer.BlockCopy(frame, tokenOffset, tokenPart, 0, tokenPart.Length);

        var fb = new Vp8FrameBuffer(width, height);
        var ec = new Vp8EntropyContexts(fb.MbCols);
        Vp8KeyframeWalker.Decode(tag, hdr, modeReader, tokenPart, fb, ec);

        int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
        var y = new byte[width * height];
        var u = new byte[uvW * uvH];
        var v = new byte[uvW * uvH];
        for (int row = 0; row < height; row++) Buffer.BlockCopy(fb.YPlane, row * fb.YStride, y, row * width, width);
        for (int row = 0; row < uvH; row++)
        {
            Buffer.BlockCopy(fb.UPlane, row * fb.UvStride, u, row * uvW, uvW);
            Buffer.BlockCopy(fb.VPlane, row * fb.UvStride, v, row * uvW, uvW);
        }
        return new Planes(y, u, v, width, height);
    }
}
