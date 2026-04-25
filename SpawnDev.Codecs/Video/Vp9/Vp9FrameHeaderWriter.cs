// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Inverse of Vp9FrameHeaderParser: serialize a Vp9FrameHeader back
// into the leading bits of a VP9 uncompressed frame header. Bit-exact
// round-trip with the parser for every keyframe / intra-only / show-
// existing case the parser currently understands.
//
// Bit layout follows VP9 spec sec 6.2 and matches what
// Vp9FrameHeaderParser reads. Writes only the prefix the parser
// understands (through render_size_and_frame_size_different); the
// rest of the uncompressed header (refresh_frame_flags, ref_frame
// info, loop filter / quantization / segmentation / tile info /
// header_size) is the consumer's responsibility for now.
//
// Used by the Vp9 encoder scaffold and round-trip tests that prove
// our parser + writer agree on every bit.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 uncompressed frame-header serializer (writer side).</summary>
public static class Vp9FrameHeaderWriter
{
    private const int FrameMarker = 0b10;

    /// <summary>
    /// Serialize the prefix of a VP9 uncompressed header that
    /// <see cref="Vp9FrameHeaderParser.Parse"/> understands. Returns
    /// a byte array padded to the next byte boundary.
    /// </summary>
    public static byte[] WriteHeaderPrefix(Vp9FrameHeader header)
    {
        ArgumentNullException.ThrowIfNull(header);
        var bw = new Vp9BitWriter();

        // frame_marker: f(2) = 0b10.
        bw.WriteBits((uint)FrameMarker, 2);

        // profile bits: low then high.
        bw.WriteBits((uint)(header.Profile & 0x1), 1);
        bw.WriteBits((uint)((header.Profile >> 1) & 0x1), 1);
        if (header.Profile == 3)
        {
            // reserved_zero_bit
            bw.WriteBits(0u, 1);
        }

        // show_existing_frame.
        if (header.ShowExistingFrame)
        {
            bw.WriteBits(1u, 1);
            bw.WriteBits((uint)header.FrameToShowMapIdx & 0x7u, 3);
            return bw.ToBytes();
        }
        bw.WriteBits(0u, 1);

        // frame_type, show_frame, error_resilient.
        bw.WriteBits((uint)header.FrameType & 0x1u, 1);
        bw.WriteBits(header.ShowFrame ? 1u : 0u, 1);
        bw.WriteBits(header.ErrorResilientMode ? 1u : 0u, 1);

        if (header.FrameType == Vp9FrameType.Key)
        {
            // sync_code f(24).
            bw.WriteBits(Vp9SyncCode.Byte0, 8);
            bw.WriteBits(Vp9SyncCode.Byte1, 8);
            bw.WriteBits(Vp9SyncCode.Byte2, 8);

            WriteColorConfig(bw, header);
            WriteFrameAndRenderSize(bw, header);
            return bw.ToBytes();
        }

        // Non-key: intra_only flag (only when !show_frame).
        if (!header.ShowFrame)
            bw.WriteBits(header.IntraOnly ? 1u : 0u, 1);

        if (!header.ErrorResilientMode)
        {
            // reset_frame_context: parser reads but discards. Emit 0 (no reset).
            bw.WriteBits(0u, 2);
        }

        if (header.IntraOnly)
        {
            bw.WriteBits(Vp9SyncCode.Byte0, 8);
            bw.WriteBits(Vp9SyncCode.Byte1, 8);
            bw.WriteBits(Vp9SyncCode.Byte2, 8);
            if (header.Profile > 0) WriteColorConfig(bw, header);
            // refresh_frame_flags f(8) - parser reads but doesn't expose; emit 0.
            bw.WriteBits(0u, 8);
            WriteFrameAndRenderSize(bw, header);
        }
        // Inter (non-intra-only) frame: parser stops at the intra_only branch
        // today. Writer mirrors that limit.

        return bw.ToBytes();
    }

    private static void WriteColorConfig(Vp9BitWriter bw, Vp9FrameHeader header)
    {
        // High bit depths only when profile >= 2 (parser path). For now we
        // assume profile 0 (8-bit). Profile 2+ writers extend here.
        if (header.Profile >= 2)
        {
            // ten_or_twelve_bit_depth: 0 -> 10, 1 -> 12. We currently never
            // set BitDepth = 12, so pick 0 for >8.
            bw.WriteBits(header.BitDepth == 12 ? 1u : 0u, 1);
        }

        bw.WriteBits((uint)header.ColorSpace & 0x7u, 3);

        if (header.ColorSpace != Vp9ColorSpace.Srgb)
        {
            bw.WriteBits(header.ColorRangeFull ? 1u : 0u, 1);
            if (header.Profile == 1 || header.Profile == 3)
            {
                bw.WriteBits(header.SubsamplingX ? 1u : 0u, 1);
                bw.WriteBits(header.SubsamplingY ? 1u : 0u, 1);
                bw.WriteBits(0u, 1); // reserved
            }
        }
        else
        {
            // Srgb forces full range; subsampling bits implicit.
            if (header.Profile == 1 || header.Profile == 3)
            {
                bw.WriteBits(0u, 1); // reserved
            }
        }
    }

    private static void WriteFrameAndRenderSize(Vp9BitWriter bw, Vp9FrameHeader header)
    {
        bw.WriteBits((uint)(header.FrameWidth - 1) & 0xFFFFu, 16);
        bw.WriteBits((uint)(header.FrameHeight - 1) & 0xFFFFu, 16);
        bool renderDifferent = header.RenderWidth > 0 && header.RenderHeight > 0
            && (header.RenderWidth != header.FrameWidth
                || header.RenderHeight != header.FrameHeight);
        bw.WriteBits(renderDifferent ? 1u : 0u, 1);
        if (renderDifferent)
        {
            bw.WriteBits((uint)(header.RenderWidth - 1) & 0xFFFFu, 16);
            bw.WriteBits((uint)(header.RenderHeight - 1) & 0xFFFFu, 16);
        }
    }
}

/// <summary>
/// Bit-level writer matching <see cref="Vp9BitReader"/>'s MSB-first
/// packing convention. Stateful - accumulates bits into a byte
/// buffer.
/// </summary>
public sealed class Vp9BitWriter
{
    private readonly List<byte> _bytes = new();
    private byte _current;
    private int _bitsInCurrent;

    /// <summary>
    /// Pack <paramref name="numBits"/> bits of <paramref name="value"/>
    /// MSB-first into the buffer. <paramref name="numBits"/> must be in
    /// [0, 32].
    /// </summary>
    public void WriteBits(uint value, int numBits)
    {
        if ((uint)numBits > 32)
            throw new ArgumentOutOfRangeException(nameof(numBits), numBits, "numBits must be in [0, 32].");

        for (int i = numBits - 1; i >= 0; i--)
        {
            int bit = (int)((value >> i) & 1);
            _current = (byte)((_current << 1) | bit);
            _bitsInCurrent++;
            if (_bitsInCurrent == 8)
            {
                _bytes.Add(_current);
                _current = 0;
                _bitsInCurrent = 0;
            }
        }
    }

    /// <summary>
    /// Finalize: pad the current partial byte with zeros and return
    /// the accumulated payload.
    /// </summary>
    public byte[] ToBytes()
    {
        if (_bitsInCurrent > 0)
        {
            _bytes.Add((byte)(_current << (8 - _bitsInCurrent)));
            _current = 0;
            _bitsInCurrent = 0;
        }
        return _bytes.ToArray();
    }
}
