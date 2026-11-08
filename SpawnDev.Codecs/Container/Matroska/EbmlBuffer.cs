// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Growable byte buffer with EBML (RFC 8794) element writers, used by WebMWriter to build headers, clusters and cues.
//
// Encoding rules (RFC 8794):
// - Element IDs are written as their full big-endian byte form (the VINT_MARKER is part of the ID value).
// - Element data sizes are VINTs; the all-ones value of each width is reserved for "unknown size", so the largest size a
//   width of n bytes can carry is 2^(7n) - 2. The writers below pick the shortest width unless a fixed width is asked for.
// - Unsigned integers use the fewest big-endian bytes (at least one); signed integers use the fewest two's-complement bytes;
//   floats are written as 8-byte IEEE 754 doubles; strings as UTF-8 without a terminator.

using System.Buffers.Binary;
using System.Text;

namespace SpawnDev.Codecs.Container.Matroska;

internal sealed class EbmlBuffer
{
    private byte[] _data;
    private int _length;

    public EbmlBuffer(int capacity = 256) => _data = new byte[Math.Max(16, capacity)];

    public int Length => _length;
    public ReadOnlySpan<byte> Span => _data.AsSpan(0, _length);
    public Span<byte> WritableSpan => _data.AsSpan(0, _length);
    public void Clear() => _length = 0;

    private Span<byte> Grow(int count)
    {
        int need = _length + count;
        if (need > _data.Length)
        {
            int cap = _data.Length;
            while (cap < need) cap *= 2;
            Array.Resize(ref _data, cap);
        }
        var s = _data.AsSpan(_length, count);
        _length = need;
        return s;
    }

    public void WriteByte(byte b) => Grow(1)[0] = b;
    public void WriteBytes(ReadOnlySpan<byte> bytes) => bytes.CopyTo(Grow(bytes.Length));
    public void WriteZeros(int count) => Grow(count).Clear();

    // ---- primitive encoders

    public static int IdLength(uint id) => id >= 0x1000000 ? 4 : id >= 0x10000 ? 3 : id >= 0x100 ? 2 : 1;

    /// <summary>Shortest VINT width that can carry <paramref name="size"/> as a known size.</summary>
    public static int SizeLength(ulong size)
    {
        int n = 1;
        while (n < 8 && size > (1UL << (7 * n)) - 2) n++;
        return n;
    }

    public static int UIntLength(ulong v)
    {
        int n = 1;
        while (n < 8 && v >> (8 * n) != 0) n++;
        return n;
    }

    public static int IntLength(long v)
    {
        int n = 1;
        while (n < 8)
        {
            long min = -(1L << (8 * n - 1)), max = (1L << (8 * n - 1)) - 1;
            if (v >= min && v <= max) break;
            n++;
        }
        return n;
    }

    /// <summary>Total bytes of an element with the given ID and data length (shortest size field).</summary>
    public static int ElementLength(uint id, int dataLength) => IdLength(id) + SizeLength((ulong)dataLength) + dataLength;

    public void WriteId(uint id)
    {
        int n = IdLength(id);
        var s = Grow(n);
        for (int i = 0; i < n; i++) s[i] = (byte)(id >> (8 * (n - 1 - i)));
    }

    public void WriteSize(ulong size, int width = 0)
    {
        if (width == 0) width = SizeLength(size);
        if (width < 1 || width > 8 || size > (1UL << (7 * width)) - 2)
            throw new ArgumentOutOfRangeException(nameof(size), $"EBML size {size} does not fit a {width}-byte VINT.");
        var s = Grow(width);
        WriteSizeTo(s, size, width);
    }

    public static void WriteSizeTo(Span<byte> dst, ulong size, int width)
    {
        ulong v = size | (1UL << (7 * width));
        for (int i = 0; i < width; i++) dst[i] = (byte)(v >> (8 * (width - 1 - i)));
    }

    /// <summary>The 8-byte "unknown size" VINT (all value bits set).</summary>
    public void WriteUnknownSize()
    {
        var s = Grow(8);
        s[0] = 0x01;
        s.Slice(1).Fill(0xFF);
    }

    public void WriteHeader(uint id, int dataLength, int sizeWidth = 0)
    {
        WriteId(id);
        WriteSize((ulong)dataLength, sizeWidth);
    }

    // ---- typed elements

    public void WriteUInt(uint id, ulong value)
    {
        int n = UIntLength(value);
        WriteHeader(id, n);
        var s = Grow(n);
        for (int i = 0; i < n; i++) s[i] = (byte)(value >> (8 * (n - 1 - i)));
    }

    public void WriteUIntFixed(uint id, ulong value, int bytes)
    {
        WriteHeader(id, bytes);
        var s = Grow(bytes);
        for (int i = 0; i < bytes; i++) s[i] = (byte)(value >> (8 * (bytes - 1 - i)));
    }

    public void WriteInt(uint id, long value)
    {
        int n = IntLength(value);
        WriteHeader(id, n);
        var s = Grow(n);
        for (int i = 0; i < n; i++) s[i] = (byte)(value >> (8 * (n - 1 - i)));
    }

    /// <summary>Writes an 8-byte float element and returns the offset of its 8 data bytes.</summary>
    public int WriteDouble(uint id, double value)
    {
        WriteHeader(id, 8);
        int at = _length;
        BinaryPrimitives.WriteDoubleBigEndian(Grow(8), value);
        return at;
    }

    public void WriteString(uint id, string value)
    {
        int n = Encoding.UTF8.GetByteCount(value);
        WriteHeader(id, n);
        Encoding.UTF8.GetBytes(value, Grow(n));
    }

    public void WriteBinary(uint id, ReadOnlySpan<byte> value)
    {
        WriteHeader(id, value.Length);
        WriteBytes(value);
    }

    /// <summary>Writes a master element whose children are already in <paramref name="body"/>.</summary>
    public void WriteMaster(uint id, EbmlBuffer body, int sizeWidth = 0)
    {
        WriteHeader(id, body.Length, sizeWidth);
        WriteBytes(body.Span);
    }

    /// <summary>
    /// Fills exactly <paramref name="total"/> bytes with a Void element (ID 0xEC) of zero bytes: an 8-byte size field when
    /// the room allows (10 bytes or more), a 1-byte size field otherwise. A Void needs at least 2 bytes.
    /// </summary>
    public void WriteVoid(int total)
    {
        if (total < 2) throw new ArgumentOutOfRangeException(nameof(total), "A Void element needs at least 2 bytes.");
        if (total >= 10)
        {
            WriteId(MatroskaIds.Void);
            WriteSize((ulong)(total - 9), 8);
            WriteZeros(total - 9);
        }
        else
        {
            WriteId(MatroskaIds.Void);
            WriteSize((ulong)(total - 2), 1);
            WriteZeros(total - 2);
        }
    }
}

/// <summary>Element IDs used by the WebM writer (RFC 8794 EBML header, RFC 9559 Matroska).</summary>
internal static class MatroskaIds
{
    public const uint Ebml = 0x1A45DFA3, EbmlVersion = 0x4286, EbmlReadVersion = 0x42F7, EbmlMaxIdLength = 0x42F2,
        EbmlMaxSizeLength = 0x42F3, DocType = 0x4282, DocTypeVersion = 0x4287, DocTypeReadVersion = 0x4285;
    public const uint Void = 0xEC;
    public const uint Segment = 0x18538067;
    public const uint SeekHead = 0x114D9B74, Seek = 0x4DBB, SeekId = 0x53AB, SeekPosition = 0x53AC;
    public const uint Info = 0x1549A966, TimestampScale = 0x2AD7B1, Duration = 0x4489, Title = 0x7BA9, MuxingApp = 0x4D80,
        WritingApp = 0x5741;
    public const uint Tracks = 0x1654AE6B, TrackEntry = 0xAE, TrackNumber = 0xD7, TrackUid = 0x73C5, TrackType = 0x83,
        FlagLacing = 0x9C, FlagDefault = 0x88, FlagForced = 0x55AA, Name = 0x536E, Language = 0x22B59C, CodecId = 0x86,
        CodecPrivate = 0x63A2, DefaultDuration = 0x23E383, CodecDelay = 0x56AA, SeekPreRoll = 0x56BB;
    public const uint Video = 0xE0, PixelWidth = 0xB0, PixelHeight = 0xBA, FlagInterlaced = 0x9A, AlphaMode = 0x53C0,
        DisplayWidth = 0x54B0, DisplayHeight = 0x54BA, Colour = 0x55B0, MatrixCoefficients = 0x55B1, ChromaSitingHorz = 0x55B7,
        ChromaSitingVert = 0x55B8, Range = 0x55B9, TransferCharacteristics = 0x55BA, Primaries = 0x55BB;
    public const uint Audio = 0xE1, Channels = 0x9F, SamplingFrequency = 0xB5, BitDepth = 0x6264;
    public const uint Tags = 0x1254C367, Tag = 0x7373, Targets = 0x63C0, TagTrackUid = 0x63C5, SimpleTag = 0x67C8,
        TagName = 0x45A3, TagString = 0x4487;
    public const uint Cluster = 0x1F43B675, Timestamp = 0xE7, SimpleBlock = 0xA3, BlockGroup = 0xA0, Block = 0xA1,
        BlockDuration = 0x9B, ReferenceBlock = 0xFB, DiscardPadding = 0x75A2, BlockAdditions = 0x75A1, BlockMore = 0xA6,
        BlockAddId = 0xEE, BlockAdditional = 0xA5;
    public const uint Cues = 0x1C53BB6B, CuePoint = 0xBB, CueTime = 0xB3, CueTrackPositions = 0xB7, CueTrack = 0xF7,
        CueClusterPosition = 0xF1, CueRelativePosition = 0xF0;
}
