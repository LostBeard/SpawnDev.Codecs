// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// IVF (Indeo Video Format) container reader. Minimal sequential reader
// for the de-facto raw VP8 / VP9 / AV1 container format.
//
// Layout:
//   32-byte file header:
//     [0..3]   signature "DKIF"
//     [4..5]   version (u16 LE) = 0
//     [6..7]   header length (u16 LE) = 32
//     [8..11]  fourcc (e.g. "VP90", "AV01")
//     [12..13] width  (u16 LE)
//     [14..15] height (u16 LE)
//     [16..19] frame_rate (u32 LE)
//     [20..23] time_scale (u32 LE)
//     [24..27] num_frames (u32 LE)
//     [28..31] reserved
//   Per frame:
//     [0..3]   frame_size (u32 LE)
//     [4..11]  pts (i64 LE)
//     [12..]   frame_size bytes of payload
//
// Used to iterate AV1 / VP9 frames out of ffmpeg-generated .ivf files
// without depending on a Matroska / WebM container.

using System.Buffers.Binary;

namespace SpawnDev.Codecs.Container.Ivf;

/// <summary>Parsed IVF file header.</summary>
public sealed record IvfHeader
{
    /// <summary>libvpx <c>fourcc</c> identifying the codec.</summary>
    public required string FourCc { get; init; }

    /// <summary>Video width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Video height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Frame rate numerator.</summary>
    public required uint FrameRate { get; init; }

    /// <summary>Frame rate denominator.</summary>
    public required uint TimeScale { get; init; }

    /// <summary>Declared frame count (may be 0 = unknown).</summary>
    public required uint NumFrames { get; init; }
}

/// <summary>One frame from an IVF file.</summary>
public readonly record struct IvfFrame(long Pts, ReadOnlyMemory<byte> Data);

/// <summary>Stateless IVF reader.</summary>
public static class IvfReader
{
    private const int FileHeaderSize = 32;
    private const int FrameHeaderSize = 12;
    private static readonly byte[] DkifSignature = { (byte)'D', (byte)'K', (byte)'I', (byte)'F' };

    /// <summary>Parse the 32-byte IVF file header.</summary>
    public static IvfHeader ParseHeader(ReadOnlySpan<byte> data)
    {
        if (data.Length < FileHeaderSize)
            throw new InvalidDataException(
                $"IVF header truncated: need {FileHeaderSize} bytes, got {data.Length}.");
        if (!data.Slice(0, 4).SequenceEqual(DkifSignature))
            throw new InvalidDataException("IVF signature mismatch (expected 'DKIF').");

        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(4, 2));
        if (version != 0)
            throw new InvalidDataException($"Unsupported IVF version {version}; expected 0.");
        ushort hdrLen = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(6, 2));
        if (hdrLen != FileHeaderSize)
            throw new InvalidDataException($"Unexpected IVF header length {hdrLen}; expected {FileHeaderSize}.");

        string fourCc = System.Text.Encoding.ASCII.GetString(data.Slice(8, 4));
        ushort width = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(12, 2));
        ushort height = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(14, 2));
        uint frameRate = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(16, 4));
        uint timeScale = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(20, 4));
        uint numFrames = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(24, 4));

        return new IvfHeader
        {
            FourCc = fourCc,
            Width = width,
            Height = height,
            FrameRate = frameRate,
            TimeScale = timeScale,
            NumFrames = numFrames,
        };
    }

    /// <summary>
    /// Enumerate every frame in <paramref name="data"/>. Each yielded
    /// <see cref="IvfFrame"/> references a byte slice of the original
    /// <paramref name="data"/> by offset/length.
    /// </summary>
    public static IEnumerable<IvfFrame> EnumerateFrames(ReadOnlyMemory<byte> data)
    {
        ParseHeader(data.Span); // validate header
        int pos = FileHeaderSize;
        while (pos < data.Length)
        {
            int remaining = data.Length - pos;
            if (remaining < FrameHeaderSize)
                throw new InvalidDataException(
                    $"IVF frame header truncated at offset {pos}: need {FrameHeaderSize}B, have {remaining}B.");

            uint frameSize = BinaryPrimitives.ReadUInt32LittleEndian(data.Span.Slice(pos, 4));
            long pts = BinaryPrimitives.ReadInt64LittleEndian(data.Span.Slice(pos + 4, 8));
            int payloadStart = pos + FrameHeaderSize;
            if (payloadStart + frameSize > data.Length)
                throw new InvalidDataException(
                    $"IVF frame at offset {pos}: declared size {frameSize}B overruns file end {data.Length}.");

            yield return new IvfFrame(pts, data.Slice(payloadStart, (int)frameSize));
            pos = payloadStart + (int)frameSize;
        }
    }
}
