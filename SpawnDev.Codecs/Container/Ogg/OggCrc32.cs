// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// CRC-32 used by the Ogg container. Polynomial 0x04C11DB7, initial value 0,
// no input reflection, no output reflection, no xor-out. (This is the same
// CRC-32 family as MPEG-2 and ETHERNET but with non-reflected data.)

namespace SpawnDev.Codecs.Container.Ogg;

internal static class OggCrc32
{
    private static readonly uint[] Table = BuildTable();

    private static uint[] BuildTable()
    {
        var t = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i << 24;
            for (int j = 0; j < 8; j++)
                c = ((c & 0x80000000) != 0) ? ((c << 1) ^ 0x04C11DB7) : (c << 1);
            t[i] = c;
        }
        return t;
    }

    /// <summary>Compute Ogg CRC-32 over a span of bytes.</summary>
    internal static uint Compute(ReadOnlySpan<byte> data)
    {
        uint crc = 0;
        for (int i = 0; i < data.Length; i++)
            crc = (crc << 8) ^ Table[(byte)(crc >> 24) ^ data[i]];
        return crc;
    }
}
