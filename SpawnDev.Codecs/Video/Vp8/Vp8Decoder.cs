// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 decoder pipeline. Wires the keyframe walker to the public IVideoDecoder
// surface so callers get real reconstructed YUV420 pixels per frame. Inter
// frames are NotImplementedException (Phase 1c) - keyframe-only streams +
// every-frame-is-keyframe streams (libvpx -keyint_min 1 -g 1) decode today.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 decoder (RFC 6386). Decodes keyframes through the
/// <see cref="Vp8KeyframeWalker"/> pipeline; inter frames remain Phase 1c.
/// </summary>
public sealed class Vp8Decoder : IVideoDecoder
{
    /// <inheritdoc/>
    public VideoCodec Codec => VideoCodec.Vp8;

    /// <inheritdoc/>
    public int Width { get; private set; }

    /// <inheritdoc/>
    public int Height { get; private set; }

    /// <summary>Most recently parsed frame tag; null before the first frame.</summary>
    public Vp8FrameTag? LastFrameTag { get; private set; }

    /// <summary>Most recently parsed compressed frame header (keyframe path).</summary>
    public Vp8FrameHeader? LastFrameHeader { get; private set; }

    /// <summary>Cumulative count of keyframes decoded.</summary>
    public int KeyFrameCount { get; private set; }

    /// <summary>Total frames passed to <see cref="DecodeFrameAsync"/>.</summary>
    public int TotalFrames { get; private set; }

    /// <inheritdoc/>
    public async ValueTask<int> DecodeFrameAsync(
        ReadOnlyMemory<byte> compressedPacket,
        IVideoFrameSink frameSink,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(frameSink);
        ct.ThrowIfCancellationRequested();
        TotalFrames++;

        var frameBytes = compressedPacket.ToArray();
        var tag = Vp8FrameTagParser.Parse(frameBytes.AsSpan());
        LastFrameTag = tag;

        if (!tag.IsKeyFrame)
        {
            throw new NotImplementedException(
                "Vp8Decoder currently handles keyframes only; inter-frame decode is Phase 1c. " +
                "Encode with libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 to get keyframe-only streams.");
        }

        // First-partition starts after the 3-byte tag + 7-byte key extension.
        const int firstPartOffset = 10;
        int firstPartLen = tag.FirstPartitionSize;
        var firstPart = new byte[firstPartLen];
        Buffer.BlockCopy(frameBytes, firstPartOffset, firstPart, 0, firstPartLen);
        var modeReader = new Vp8BoolDecoder(firstPart);
        var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(modeReader);
        LastFrameHeader = hdr;

        int width = tag.Width!.Value;
        int height = tag.Height!.Value;
        Width = width;
        Height = height;

        // Single-token-partition only for now; the walker rejects others itself.
        int tokenOffset = firstPartOffset + firstPartLen;
        int tokenLen = frameBytes.Length - tokenOffset;
        var tokenPart = new byte[tokenLen];
        Buffer.BlockCopy(frameBytes, tokenOffset, tokenPart, 0, tokenLen);

        var fb = new Vp8FrameBuffer(width, height);
        var ec = new Vp8EntropyContexts(fb.MbCols);
        Vp8KeyframeWalker.Decode(tag, hdr, modeReader, tokenPart, fb, ec);

        KeyFrameCount++;

        // Repackage planes into tightly-packed buffers matching the
        // logical width/height (the walker frame buffer pads stride up to
        // the next macroblock boundary so its raw arrays would mislead a
        // sink that doesn't honour stride).
        var y = new byte[width * height];
        int uvW = (width + 1) / 2;
        int uvH = (height + 1) / 2;
        var u = new byte[uvW * uvH];
        var v = new byte[uvW * uvH];
        for (int row = 0; row < height; row++)
            Buffer.BlockCopy(fb.YPlane, row * fb.YStride, y, row * width, width);
        for (int row = 0; row < uvH; row++)
        {
            Buffer.BlockCopy(fb.UPlane, row * fb.UvStride, u, row * uvW, uvW);
            Buffer.BlockCopy(fb.VPlane, row * fb.UvStride, v, row * uvW, uvW);
        }

        await frameSink.OnFrameAsync(y, width, u, uvW, v, uvW, pts: 0L).ConfigureAwait(false);
        return 1;
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
