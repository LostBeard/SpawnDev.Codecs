// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320), the checksum of PNG chunks and gzip. GPU-callable
// table-driven form: the 256-entry table is a small constant buffer uploaded once per accelerator.

using ILGPU;

namespace SpawnDev.Codecs.Compression;

/// <summary>GPU-callable CRC-32.</summary>
public static class Crc32Gpu
{
    /// <summary>Initial CRC register value (all ones); finish with <see cref="Finish"/>.</summary>
    public const uint Seed = 0xFFFFFFFFu;

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

    /// <summary>Folds <paramref name="length"/> bytes at <paramref name="offset"/> into <paramref name="crc"/>.</summary>
    public static uint Update(uint crc, ArrayView<byte> data, int offset, int length, ArrayView<uint> table)
    {
        uint c = crc;
        for (int i = 0; i < length; i++)
            c = table[(int)((c ^ data[offset + i]) & 0xFF)] ^ (c >> 8);
        return c;
    }

    /// <summary>Final inversion of the CRC register.</summary>
    public static uint Finish(uint crc) => crc ^ 0xFFFFFFFFu;
}
