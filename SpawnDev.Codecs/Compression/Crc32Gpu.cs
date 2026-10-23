// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320), the checksum of PNG chunks and gzip. GPU-callable
// table-driven form: the 256-entry table is a small constant buffer uploaded once per accelerator.
//
// Long ranges are checksummed in parallel PIECES of PieceBytes: the register update is linear over GF(2), so
// Update(s, D) = Shift(s, |D|) ^ Update(0, D), where Shift advances a register over |D| zero bytes (a 32x32 bit
// matrix, BuildShift). A range cut into a short first piece (seeded) and full pieces (from 0) is combined exactly:
// state = Shift(state, PieceBytes) ^ piece, piece by piece. No thread walks a whole long range.

using ILGPU;

namespace SpawnDev.Codecs.Compression;

/// <summary>GPU-callable CRC-32.</summary>
public static class Crc32Gpu
{
    /// <summary>Initial CRC register value (all ones); finish with <see cref="Finish"/>.</summary>
    public const uint Seed = 0xFFFFFFFFu;

    /// <summary>Bytes per parallel piece of a long CRC range (one thread checksums one piece).</summary>
    public const int PieceBytes = 8192;

    /// <summary>
    /// Pieces of one combine launch: a combine folds at most this many pieces per range per launch (32 xor-and steps
    /// each), so the combine of even a gigabyte range stays a chain of short launches.
    /// </summary>
    public const int CombinePieces = 1024;

    /// <summary>The standard 256-entry CRC-32 table (constant data, built on the host once).</summary>
    public static uint[] BuildTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }

    /// <summary>
    /// The 256-entry table followed by the 32 columns of the <see cref="PieceBytes"/> zero-byte shift
    /// (<see cref="BuildShift"/>): the constant buffer the piece-parallel kernels take.
    /// </summary>
    public static uint[] BuildTableWithShift()
    {
        var table = BuildTable();
        var all = new uint[256 + 32];
        table.CopyTo(all, 0);
        BuildShift(table, PieceBytes).CopyTo(all, 256);
        return all;
    }

    /// <summary>
    /// Columns of the linear map that advances a CRC register over <paramref name="zeroBytes"/> zero bytes:
    /// column i is the register 1 &lt;&lt; i after them.
    /// </summary>
    public static uint[] BuildShift(uint[] table, int zeroBytes)
    {
        var cols = new uint[32];
        for (int i = 0; i < 32; i++)
        {
            uint c = 1u << i;
            for (int k = 0; k < zeroBytes; k++) c = table[(int)(c & 0xFF)] ^ (c >> 8);
            cols[i] = c;
        }
        return cols;
    }

    /// <summary>Folds <paramref name="length"/> bytes at <paramref name="offset"/> into <paramref name="crc"/>.</summary>
    public static uint Update(uint crc, ArrayView<byte> data, int offset, int length, ArrayView<uint> table)
    {
        uint c = crc;
        for (int i = 0; i < length; i++)
            c = table[(int)((c ^ data[offset + i]) & 0xFF)] ^ (c >> 8);
        return c;
    }

    /// <summary>
    /// Advances register <paramref name="crc"/> over the zero bytes whose shift columns start at
    /// <paramref name="shiftAt"/> in <paramref name="table"/> (256 for <see cref="BuildTableWithShift"/>).
    /// </summary>
    public static uint Shift(uint crc, ArrayView<uint> table, int shiftAt)
    {
        uint r = 0;
        for (int i = 0; i < 32; i++)
        {
            uint mask = 0u - ((crc >> i) & 1u);
            r ^= table[shiftAt + i] & mask;
        }
        return r;
    }

    /// <summary>
    /// CRC piece <paramref name="piece"/> of the range [<paramref name="offset"/>, <paramref name="offset"/> +
    /// <paramref name="length"/>) (length &gt; 0): piece 0 is the first <c>length - (n - 1) * PieceBytes</c> bytes,
    /// folded from <see cref="Seed"/>; every later piece is <see cref="PieceBytes"/> bytes folded from 0.
    /// </summary>
    public static uint PieceOfRange(ArrayView<byte> data, int offset, int length, int piece, ArrayView<uint> table)
    {
        int n = PiecesOf(length);
        int first = length - (n - 1) * PieceBytes;
        uint c;
        if (piece == 0) c = Update(Seed, data, offset, first, table);
        else c = Update(0u, data, offset + first + (piece - 1) * PieceBytes, PieceBytes, table);
        return c;
    }

    /// <summary>Pieces a range of <paramref name="length"/> bytes is cut into (at least 1).</summary>
    public static int PiecesOf(int length) => length <= PieceBytes ? 1 : (length - 1) / PieceBytes + 1;

    /// <summary>
    /// Combine launch <paramref name="step"/> of a range whose <paramref name="pieceCount"/> piece CRCs start at
    /// <paramref name="at"/> in <paramref name="pieces"/>: folds pieces [1 + step * CombinePieces, ...) into the
    /// accumulator kept in pieces[at]. Returns 1 when this launch folded the last piece (the accumulator then holds the
    /// register of the whole range: <see cref="Finish"/> it), 0 otherwise (also for a launch after the end).
    /// </summary>
    public static int CombineStep(ArrayView<uint> pieces, int at, int pieceCount, int step, ArrayView<uint> table)
    {
        int lo = 1 + step * CombinePieces;
        int hi = lo + CombinePieces;
        if (hi > pieceCount) hi = pieceCount;
        int finished = 0;
        bool active = (lo < pieceCount) | (step == 0);
        if (active)
        {
            uint state = pieces[at];
            for (int j = lo; j < hi; j++) state = Shift(state, table, 256) ^ pieces[at + j];
            pieces[at] = state;
            if (hi == pieceCount) finished = 1;
        }
        return finished;
    }

    /// <summary>Combine launches a range of at most <paramref name="maxPieces"/> pieces needs.</summary>
    public static int CombineSteps(long maxPieces) => (int)Math.Max(1, (maxPieces - 1 + CombinePieces - 1) / CombinePieces);

    /// <summary>Final inversion of the CRC register.</summary>
    public static uint Finish(uint crc) => crc ^ 0xFFFFFFFFu;
}
