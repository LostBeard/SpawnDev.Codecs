// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 tx_mode field (libvpx TX_MODE) plus the parser that pulls it
// from the compressed frame header. tx_mode constrains which
// transform sizes the rest of the frame may use:
//
//   Only4x4       : every block uses 4x4 (forced for lossless frames)
//   AllowOnly8x8  : 4x4 and 8x8 only
//   AllowOnly16x16: 4x4, 8x8, 16x16
//   Allow32x32    : all four transform sizes available
//   TxModeSelect  : the bitstream signals tx_size per block (i.e. the
//                   per-block tx_size can be any of 4x4..32x32 and is
//                   transmitted explicitly with the block).
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c read_tx_mode.
// VP9 spec reference: sec 6.3.1 "TX mode syntax".

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 tx_mode values (libvpx TX_MODE enum).</summary>
public enum Vp9TxMode : byte
{
    /// <summary>Every block uses 4x4. Forced when the frame is lossless.</summary>
    Only4x4 = 0,
    /// <summary>4x4 and 8x8 transforms allowed.</summary>
    AllowOnly8x8 = 1,
    /// <summary>4x4, 8x8, and 16x16 transforms allowed.</summary>
    AllowOnly16x16 = 2,
    /// <summary>All four transform sizes allowed at the frame level.</summary>
    Allow32x32 = 3,
    /// <summary>Per-block tx_size signalled in the bitstream.</summary>
    TxModeSelect = 4,
}

/// <summary>Compressed-header parsers (libvpx read_tx_mode and friends).</summary>
public static class Vp9CompressedHeader
{
    /// <summary>
    /// Pull the tx_mode field from the compressed frame header. Reads
    /// 0 bits when <paramref name="isLossless"/> is true (forced
    /// Only4x4), 2 bits otherwise; if those 2 bits indicate
    /// <see cref="Vp9TxMode.Allow32x32"/> a third bit is read to
    /// distinguish Allow32x32 from TxModeSelect (the libvpx
    /// "+= read_literal(1)" extension).
    /// </summary>
    public static Vp9TxMode ReadTxMode(Vp9BoolDecoder reader, bool isLossless)
    {
        ArgumentNullException.ThrowIfNull(reader);
        return ReadTxMode(n => reader.ReadLiteral(n), isLossless);
    }

    /// <summary>
    /// Pure-function variant of <see cref="ReadTxMode(Vp9BoolDecoder, bool)"/>
    /// that takes a literal reader delegate (returns the next
    /// <c>n</c> bits as an unsigned integer). Production callers use
    /// the <see cref="Vp9BoolDecoder"/> overload; this overload exists
    /// for unit testing without an arithmetic-coded buffer.
    /// </summary>
    public static Vp9TxMode ReadTxMode(Func<int, uint> readLiteral, bool isLossless)
    {
        ArgumentNullException.ThrowIfNull(readLiteral);
        if (isLossless) return Vp9TxMode.Only4x4;

        int v = (int)readLiteral(2);
        if (v == (int)Vp9TxMode.Allow32x32)
            v += (int)readLiteral(1);
        return (Vp9TxMode)v;
    }
}
