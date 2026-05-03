// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 boolean arithmetic encoder. Bit-exact port of libvpx
// vpx_dsp/bitwriter.h (vpx_writer + vpx_write + vpx_stop_encode).
//
// libvpx shares the same range-coder math between VP8 and VP9: vpx_write
// is structurally identical to vp8_encode_bool. Algorithm is unchanged.
// Maintaining a separate Vp9-namespace class keeps the codec boundaries
// clean and avoids cross-codec coupling.
//
// Pairs with <see cref="Vp9BoolDecoder"/>.

using System.Numerics;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 boolean arithmetic encoder. Pairs with <see cref="Vp9BoolDecoder"/>.</summary>
public sealed class Vp9BoolEncoder
{
    private readonly List<byte> _buf = new();
    private uint _lowvalue;
    private uint _range;
    private int _count;

    /// <summary>Construct an empty encoder.</summary>
    public Vp9BoolEncoder()
    {
        Reset();
    }

    /// <summary>Reset encoder state. Mirrors libvpx <c>vpx_start_encode</c>,
    /// including the leading 0 marker bit at flat probability 128 that
    /// <see cref="Vp9BoolDecoder"/> consumes during init.</summary>
    public void Reset()
    {
        _buf.Clear();
        _lowvalue = 0;
        _range = 255;
        _count = -24;
        // libvpx vpx_start_encode emits a leading marker bit (== 0)
        // immediately after seeding the state. The Vp9 bool decoder
        // consumes this bit during init and validates that it is zero.
        Write(0, 128);
    }

    /// <summary>
    /// Encode one bit at <paramref name="probability"/> in [1, 255].
    /// Mirrors libvpx <c>vpx_write</c>.
    /// </summary>
    public void Write(int bit, int probability)
    {
        if (probability < 1 || probability > 255)
            throw new ArgumentOutOfRangeException(nameof(probability), "must be in [1, 255]");

        uint split = 1u + (((_range - 1u) * (uint)probability) >> 8);
        uint range = split;
        uint lowvalue = _lowvalue;
        int count = _count;

        if (bit != 0)
        {
            lowvalue += split;
            range = _range - split;
        }

        int shift = LeadingZeros8((byte)range);
        range <<= shift;
        count += shift;

        if (count >= 0)
        {
            int offset = shift - count;

            if ((((ulong)lowvalue) << (offset - 1) & 0x80000000UL) != 0)
            {
                int x = _buf.Count - 1;
                while (x >= 0 && _buf[x] == 0xFF)
                {
                    _buf[x] = 0;
                    x--;
                }
                if (x >= 0) _buf[x] = (byte)(_buf[x] + 1);
            }

            _buf.Add((byte)((lowvalue >> (24 - offset)) & 0xFF));

            shift = count;
            lowvalue = (lowvalue << offset) & 0xFFFFFFu;
            count -= 8;
        }

        lowvalue <<= shift;

        _lowvalue = lowvalue;
        _range = range;
        _count = count;
    }

    /// <summary>Encode <paramref name="bits"/> raw bits MSB-first at flat probability 128.</summary>
    public void WriteLiteral(int data, int bits)
    {
        if (bits < 0 || bits > 31) throw new ArgumentOutOfRangeException(nameof(bits));
        for (int b = bits - 1; b >= 0; b--) Write((data >> b) & 1, 0x80);
    }

    /// <summary>
    /// Finalize and return encoded bytes. Mirrors libvpx <c>vpx_stop_encode</c>
    /// which emits 32 trailing zero bits at probability 128 to flush.
    /// </summary>
    public byte[] Stop()
    {
        for (int i = 0; i < 32; i++) Write(0, 128);
        return _buf.ToArray();
    }

    private static int LeadingZeros8(byte b)
    {
        if (b == 0) return 0;
        return BitOperations.LeadingZeroCount((uint)b) - 24;
    }
}
