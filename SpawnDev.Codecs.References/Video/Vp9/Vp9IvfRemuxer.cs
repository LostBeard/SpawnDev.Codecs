// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 IVF remuxer - mirrors Av1IvfRemuxer for VP9 streams. Walks an
// IVF byte stream, re-emits each VP9 packet through Vp9SuperframeWriter,
// and re-packages via IvfWriter.

using SpawnDev.Codecs.Container.Ivf;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>High-level VP9 IVF remux helper.</summary>
public static class Vp9IvfRemuxer
{
    /// <summary>
    /// Read every IVF frame from <paramref name="sourceIvf"/>, re-emit
    /// each VP9 packet through <see cref="Vp9SuperframeWriter"/>, and
    /// write the resulting frames into a new IVF byte array via
    /// <see cref="IvfWriter"/>. Returns the remuxed bytes.
    /// </summary>
    public static byte[] RemuxToBytes(ReadOnlyMemory<byte> sourceIvf)
    {
        var header = IvfReader.ParseHeader(sourceIvf.Span);
        using var ms = new MemoryStream();
        var writer = new IvfWriter(ms, header.FourCc, header.Width, header.Height,
            frameRate: header.FrameRate, timeScale: header.TimeScale);

        foreach (var ivfFrame in IvfReader.EnumerateFrames(sourceIvf))
        {
            // Parse the packet's superframe layout, extract each frame
            // slice as bytes, and re-emit through the writer.
            var data = ivfFrame.Data.ToArray();
            var parsed = Vp9SuperframeParser.Parse(data);
            var frames = new byte[parsed.Frames.Count][];
            for (int i = 0; i < parsed.Frames.Count; i++)
            {
                var slice = parsed.Frames[i];
                var fbytes = new byte[slice.Length];
                Buffer.BlockCopy(data, slice.Offset, fbytes, 0, slice.Length);
                frames[i] = fbytes;
            }
            writer.WriteFrame(Vp9SuperframeWriter.Emit(frames), ivfFrame.Pts);
        }
        writer.Finish();
        return ms.ToArray();
    }
}
