// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC CRC-8 + CRC-16, GPU-callable form. Bit-exact mirror of FlacCrc.
//
// Used by FLAC frame encoder + decoder to compute / verify the
// frame-header CRC-8 (poly 0x07) and frame-footer CRC-16 (poly 0x8005).
// Both start at 0x00, no reflection, no xor-out.
//
// Compute8 / Compute16 are the bit-shift loop forms for SHORT ranges (a frame header is at most 11 bytes).
// Long ranges (a whole frame, up to ~2 MiB) are checksummed in parallel PIECES of PieceBytes so no thread walks
// a whole frame: the register update is linear over GF(2) and both CRCs start at 0, so
// Update(s, D) = Shift(s, |D|) ^ Update(0, D), where Shift advances a register over |D| zero bytes (a 16x16 or
// 8x8 bit matrix). A range cut into a short first piece and full pieces is combined exactly:
// state = Shift(state, PieceBytes) ^ piece, piece by piece (see Crc32Gpu for the CRC-32 twin). The pieces use
// the byte tables of BuildTables (a small constant buffer uploaded once per kernel wrapper).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>GPU-callable FLAC CRC-8 + CRC-16 helpers.</summary>
public static class FlacCrcGpu
{
    /// <summary>Bytes per parallel piece of a long CRC range (one thread checksums one piece).</summary>
    public const int PieceBytes = 4096;

    /// <summary>Pieces one combine launch folds per range (16 or 8 xor-and steps each).</summary>
    public const int CombinePieces = 1024;

    /// <summary>Offsets (ints) inside the constant buffer of <see cref="BuildTables"/>.</summary>
    public const int Table16At = 0, Table8At = 256, Shift16At = 512, Shift8At = 528, TableInts = 536;

    /// <summary>
    /// The constant buffer of the piece-parallel kernels: the 256-entry CRC-16 and CRC-8 byte tables, then the 16
    /// (CRC-16) and 8 (CRC-8) columns of the <see cref="PieceBytes"/> zero-byte shift.
    /// </summary>
    public static int[] BuildTables()
    {
        var all = new int[TableInts];
        for (int n = 0; n < 256; n++)
        {
            int c = n << 8;
            for (int k = 0; k < 8; k++) c = (c & 0x8000) != 0 ? (c << 1) ^ 0x8005 : c << 1;
            all[Table16At + n] = c & 0xFFFF;
            int d = n;
            for (int k = 0; k < 8; k++) d = (d & 0x80) != 0 ? (d << 1) ^ 0x07 : d << 1;
            all[Table8At + n] = d & 0xFF;
        }
        for (int i = 0; i < 16; i++)
        {
            int c = 1 << i;
            for (int k = 0; k < PieceBytes; k++) c = ((c << 8) & 0xFFFF) ^ all[Table16At + ((c >> 8) & 0xFF)];
            all[Shift16At + i] = c;
        }
        for (int i = 0; i < 8; i++)
        {
            int c = 1 << i;
            for (int k = 0; k < PieceBytes; k++) c = all[Table8At + (c & 0xFF)];
            all[Shift8At + i] = c;
        }
        return all;
    }

    /// <summary>
    /// Compute CRC-8 over <paramref name="data"/> bytes [<paramref name="start"/>,
    /// <paramref name="start"/>+<paramref name="length"/>). Polynomial 0x07.
    /// Initial value 0x00. For short ranges only (one thread walks them).
    /// </summary>
    public static byte Compute8(ArrayView<byte> data, long start, int length)
    {
        int crc = 0;
        for (int i = 0; i < length; i++)
        {
            crc ^= data[start + i];
            for (int j = 0; j < 8; j++)
            {
                if ((crc & 0x80) != 0) crc = (crc << 1) ^ 0x07;
                else crc <<= 1;
            }
            crc &= 0xFF;
        }
        return (byte)crc;
    }

    /// <summary>
    /// Compute CRC-16 over <paramref name="data"/> bytes [<paramref name="start"/>,
    /// <paramref name="start"/>+<paramref name="length"/>). Polynomial 0x8005.
    /// Initial value 0x0000. For short ranges only (one thread walks them).
    /// </summary>
    public static ushort Compute16(ArrayView<byte> data, long start, int length)
    {
        int crc = 0;
        for (int i = 0; i < length; i++)
        {
            crc ^= ((int)data[start + i]) << 8;
            for (int j = 0; j < 8; j++)
            {
                if ((crc & 0x8000) != 0) crc = (crc << 1) ^ 0x8005;
                else crc <<= 1;
            }
            crc &= 0xFFFF;
        }
        return (ushort)crc;
    }

    /// <summary>Pieces a range of <paramref name="length"/> bytes is cut into (at least 1).</summary>
    public static int PiecesOf(long length) => length <= PieceBytes ? 1 : (int)((length - 1) / PieceBytes + 1);

    /// <summary>
    /// Start and length of piece <paramref name="piece"/> of a range of <paramref name="length"/> bytes: piece 0 is
    /// the first <c>length - (n - 1) * PieceBytes</c> bytes, every later piece is <see cref="PieceBytes"/> bytes.
    /// </summary>
    public static void PieceSpan(long length, int piece, out long start, out int count)
    {
        int n = PiecesOf(length);
        long first = length - (long)(n - 1) * PieceBytes;
        if (piece == 0)
        {
            start = 0;
            count = (int)first;
        }
        else
        {
            start = first + (long)(piece - 1) * PieceBytes;
            count = PieceBytes;
        }
    }

    /// <summary>CRC-16 register of <paramref name="count"/> bytes at <paramref name="start"/>, folded from 0 (table form).</summary>
    public static int Piece16(ArrayView<byte> data, long start, int count, ArrayView<int> tables)
    {
        int c = 0;
        for (int i = 0; i < count; i++)
            c = ((c << 8) & 0xFFFF) ^ tables[Table16At + (((c >> 8) ^ data[start + i]) & 0xFF)];
        return c;
    }

    /// <summary>CRC-8 register of <paramref name="count"/> bytes at <paramref name="start"/>, folded from 0 (table form).</summary>
    public static int Piece8(ArrayView<byte> data, long start, int count, ArrayView<int> tables)
    {
        int c = 0;
        for (int i = 0; i < count; i++)
            c = tables[Table8At + ((c ^ data[start + i]) & 0xFF)];
        return c;
    }

    /// <summary>Advances CRC-16 register <paramref name="crc"/> over <see cref="PieceBytes"/> zero bytes.</summary>
    public static int Shift16(int crc, ArrayView<int> tables)
    {
        int r = 0;
        for (int i = 0; i < 16; i++)
        {
            int mask = 0 - ((crc >> i) & 1);
            r ^= tables[Shift16At + i] & mask;
        }
        return r;
    }

    /// <summary>Advances CRC-8 register <paramref name="crc"/> over <see cref="PieceBytes"/> zero bytes.</summary>
    public static int Shift8(int crc, ArrayView<int> tables)
    {
        int r = 0;
        for (int i = 0; i < 8; i++)
        {
            int mask = 0 - ((crc >> i) & 1);
            r ^= tables[Shift8At + i] & mask;
        }
        return r;
    }

    /// <summary>
    /// Combine launch <paramref name="step"/> of a range whose <paramref name="pieceCount"/> CRC-16 piece registers
    /// start at <paramref name="at"/> in <paramref name="pieces"/>: folds pieces [1 + step * CombinePieces, ...) into the
    /// accumulator kept in pieces[at]. Returns 1 when this launch folded the last piece (pieces[at] then holds the
    /// CRC-16 of the whole range), 0 otherwise (also for a launch after the end).
    /// </summary>
    public static int CombineStep16(ArrayView<int> pieces, long at, int pieceCount, int step, ArrayView<int> tables)
    {
        int lo = 1 + step * CombinePieces;
        int hi = lo + CombinePieces;
        if (hi > pieceCount) hi = pieceCount;
        int finished = 0;
        bool active = (lo < pieceCount) | (step == 0);
        if (active)
        {
            int state = pieces[at];
            for (int j = lo; j < hi; j++) state = Shift16(state, tables) ^ pieces[at + j];
            pieces[at] = state;
            if (hi == pieceCount) finished = 1;
        }
        return finished;
    }

    /// <summary>CRC-8 twin of <see cref="CombineStep16"/>.</summary>
    public static int CombineStep8(ArrayView<int> pieces, long at, int pieceCount, int step, ArrayView<int> tables)
    {
        int lo = 1 + step * CombinePieces;
        int hi = lo + CombinePieces;
        if (hi > pieceCount) hi = pieceCount;
        int finished = 0;
        bool active = (lo < pieceCount) | (step == 0);
        if (active)
        {
            int state = pieces[at];
            for (int j = lo; j < hi; j++) state = Shift8(state, tables) ^ pieces[at + j];
            pieces[at] = state;
            if (hi == pieceCount) finished = 1;
        }
        return finished;
    }

    /// <summary>Combine launches a range of at most <paramref name="maxPieces"/> pieces needs.</summary>
    public static int CombineSteps(long maxPieces) => (int)Math.Max(1, (maxPieces - 1 + CombinePieces - 1) / CombinePieces);
}
