// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// IVF container writer - inverse of IvfReader. Writes the 32-byte DKIF
// file header + per-frame 12-byte (size, pts) headers + payload.
//
// Used to wrap AV1 / VP9 / VP8 OBU/superframe bytes into a .ivf file
// that ffmpeg, libvpx, libaom can decode. The bitstream-out side of
// the SpawnDev.Codecs container layer.

using System.Buffers.Binary;

namespace SpawnDev.Codecs.Container.Ivf;

/// <summary>IVF file writer.</summary>
public sealed class IvfWriter
{
    private const int FileHeaderSize = 32;
    private const int FrameHeaderSize = 12;
    private static readonly byte[] DkifSignature = { (byte)'D', (byte)'K', (byte)'I', (byte)'F' };

    private readonly Stream _stream;
    private readonly bool _leaveOpen;
    private long _frameCountPos = -1;
    private uint _framesWritten;

    /// <summary>
    /// Construct around <paramref name="stream"/>. Writes the file header
    /// immediately. Set <paramref name="leaveOpen"/> = true to keep the
    /// stream open when this writer is disposed.
    /// </summary>
    public IvfWriter(
        Stream stream,
        string fourCc,
        int width,
        int height,
        uint frameRate = 30,
        uint timeScale = 1,
        uint numFrames = 0,
        bool leaveOpen = false)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream is not writable.", nameof(stream));
        if (fourCc is null) throw new ArgumentNullException(nameof(fourCc));
        if (fourCc.Length != 4) throw new ArgumentException("fourCc must be exactly 4 ASCII characters.", nameof(fourCc));
        if ((uint)width > 65535)  throw new ArgumentOutOfRangeException(nameof(width));
        if ((uint)height > 65535) throw new ArgumentOutOfRangeException(nameof(height));

        _stream = stream;
        _leaveOpen = leaveOpen;

        Span<byte> hdr = stackalloc byte[FileHeaderSize];
        DkifSignature.CopyTo(hdr);
        BinaryPrimitives.WriteUInt16LittleEndian(hdr.Slice(4, 2), 0);                  // version
        BinaryPrimitives.WriteUInt16LittleEndian(hdr.Slice(6, 2), FileHeaderSize);     // header length
        for (int i = 0; i < 4; i++) hdr[8 + i] = (byte)fourCc[i];
        BinaryPrimitives.WriteUInt16LittleEndian(hdr.Slice(12, 2), (ushort)width);
        BinaryPrimitives.WriteUInt16LittleEndian(hdr.Slice(14, 2), (ushort)height);
        BinaryPrimitives.WriteUInt32LittleEndian(hdr.Slice(16, 4), frameRate);
        BinaryPrimitives.WriteUInt32LittleEndian(hdr.Slice(20, 4), timeScale);
        BinaryPrimitives.WriteUInt32LittleEndian(hdr.Slice(24, 4), numFrames);
        BinaryPrimitives.WriteUInt32LittleEndian(hdr.Slice(28, 4), 0);                 // reserved

        if (_stream.CanSeek)
            _frameCountPos = _stream.Position + 24;
        _stream.Write(hdr);
    }

    /// <summary>
    /// Append one IVF frame (12-byte size+pts header + payload).
    /// </summary>
    public void WriteFrame(ReadOnlySpan<byte> payload, long pts)
    {
        Span<byte> hdr = stackalloc byte[FrameHeaderSize];
        BinaryPrimitives.WriteUInt32LittleEndian(hdr.Slice(0, 4), (uint)payload.Length);
        BinaryPrimitives.WriteInt64LittleEndian(hdr.Slice(4, 8), pts);
        _stream.Write(hdr);
        _stream.Write(payload);
        _framesWritten++;
    }

    /// <summary>
    /// Patch the file header's <c>num_frames</c> with the count actually
    /// written (when the underlying stream supports seeking). No-op
    /// otherwise.
    /// </summary>
    public void Finish()
    {
        if (_frameCountPos < 0) return;
        long save = _stream.Position;
        _stream.Seek(_frameCountPos, SeekOrigin.Begin);
        Span<byte> count = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(count, _framesWritten);
        _stream.Write(count);
        _stream.Seek(save, SeekOrigin.Begin);
    }
}
