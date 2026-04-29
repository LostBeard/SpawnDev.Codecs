// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis MSB-first... wait, LSB-first bit reader, GPU-callable form.
// Bit-exact mirror of VorbisBitReader. Vorbis I packs integers
// LSB-first (opposite of FLAC), so the bit shuffle inside each byte
// is different. The first bit read appears in bit 0 of the result;
// the nth bit in bit n.
//
// Foundation primitive for the upcoming Vorbis decoder pipeline.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>In-kernel state for the Vorbis bit reader.</summary>
public struct VorbisBitReaderGpuState
{
    /// <summary>Current byte offset.</summary>
    public int BytePos;
    /// <summary>Bits consumed in the current byte (0..7).</summary>
    public int BitPos;
    /// <summary>Length of the input view in bytes.</summary>
    public int Length;
}

/// <summary>
/// GPU-callable LSB-first bit reader matching Vorbis I packing.
/// </summary>
public static class VorbisBitReaderGpu
{
    /// <summary>Initialize for the given input length.</summary>
    public static VorbisBitReaderGpuState Init(int length) => new VorbisBitReaderGpuState
    {
        BytePos = 0,
        BitPos = 0,
        Length = length,
    };

    /// <summary>Bits remaining in the input.</summary>
    public static int BitsRemaining(in VorbisBitReaderGpuState state)
        => (state.Length - state.BytePos) * 8 - state.BitPos;

    /// <summary>True after all bits consumed.</summary>
    public static bool IsEnd(in VorbisBitReaderGpuState state)
        => BitsRemaining(in state) <= 0;

    /// <summary>
    /// Read <paramref name="nBits"/> bits as an unsigned integer
    /// (LSB-first). The first bit read lands in bit 0 of the result.
    /// </summary>
    public static uint ReadBits(
        ref VorbisBitReaderGpuState state, ArrayView<byte> data, int nBits)
    {
        uint result = 0;
        int outBit = 0;
        while (nBits > 0)
        {
            int available = 8 - state.BitPos;
            int take = available < nBits ? available : nBits;
            int b = data[state.BytePos];
            uint chunk = (uint)((b >> state.BitPos) & ((1 << take) - 1));
            result |= chunk << outBit;
            state.BitPos += take;
            nBits -= take;
            outBit += take;
            if (state.BitPos == 8)
            {
                state.BytePos++;
                state.BitPos = 0;
            }
        }
        return result;
    }
}
