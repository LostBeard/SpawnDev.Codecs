// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 IVF remuxer - high-level helper that walks an AV1 IVF byte
// stream, re-emits every OBU through Av1ObuWriter, and re-packages
// via IvfWriter. The output is byte-equivalent to the source for
// streams whose OBUs have explicit size fields (the BBB fixture
// remuxes 100% bit-exact; ffmpeg+dav1d decode source vs remux to
// pixel-identical YUV).
//
// Use case: drop-in encoder framing for consumers building custom
// AV1 emit paths. Hand them parsed metadata + remux bytes and they
// can substitute their own coded payloads.

using SpawnDev.Codecs.Container.Ivf;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>High-level AV1 IVF remux helper.</summary>
public static class Av1IvfRemuxer
{
    /// <summary>
    /// Read every IVF frame from <paramref name="sourceIvf"/>, re-emit
    /// each OBU through <see cref="Av1ObuWriter"/>, and write the
    /// resulting frames into a new IVF byte array via
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
            using var frameMs = new MemoryStream();
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                byte[] re = Av1ObuWriter.EmitObu(obu, ivfFrame.Data);
                frameMs.Write(re, 0, re.Length);
            }
            writer.WriteFrame(frameMs.ToArray(), ivfFrame.Pts);
        }
        writer.Finish();
        return ms.ToArray();
    }

    /// <summary>
    /// Like <see cref="RemuxToBytes(ReadOnlyMemory{byte})"/> but substitutes
    /// the source's SequenceHeader OBUs with one emitted from
    /// <paramref name="newShConfig"/>. Useful for testing the closed-loop
    /// SH writer path - if the new config produces byte-equivalent SH bytes
    /// to the source, the remux output is byte-equivalent as well.
    /// </summary>
    public static byte[] RemuxToBytesWithShSubstitution(
        ReadOnlyMemory<byte> sourceIvf,
        Av1SequenceHeaderConfig newShConfig)
    {
        ArgumentNullException.ThrowIfNull(newShConfig);
        var header = IvfReader.ParseHeader(sourceIvf.Span);
        var ourShPayload = Av1SequenceHeaderWriter.EmitPayload(newShConfig);
        var ourShObu = Av1ObuWriter.EmitObu(Av1ObuType.SequenceHeader, ourShPayload, hasSizeField: true);

        using var ms = new MemoryStream();
        var writer = new IvfWriter(ms, header.FourCc, header.Width, header.Height,
            frameRate: header.FrameRate, timeScale: header.TimeScale);

        foreach (var ivfFrame in IvfReader.EnumerateFrames(sourceIvf))
        {
            using var frameMs = new MemoryStream();
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                if (obu.Type == Av1ObuType.SequenceHeader)
                {
                    frameMs.Write(ourShObu, 0, ourShObu.Length);
                }
                else
                {
                    byte[] re = Av1ObuWriter.EmitObu(obu, ivfFrame.Data);
                    frameMs.Write(re, 0, re.Length);
                }
            }
            writer.WriteFrame(frameMs.ToArray(), ivfFrame.Pts);
        }
        writer.Finish();
        return ms.ToArray();
    }
}
