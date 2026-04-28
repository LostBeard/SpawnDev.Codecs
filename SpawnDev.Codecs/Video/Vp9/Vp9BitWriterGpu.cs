// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 bit writer, GPU-callable form. Bit-exact mirror of Vp9BitWriter
// (in Vp9FrameHeaderWriter.cs) but with state held in a struct that
// callers pass by ref into static helpers - the same pattern the bool
// encoder/decoder use.
//
// VP9's uncompressed header is RAW BITS (MSB-first byte packing),
// not bool-coded - so we need a separate bit-writer abstraction
// alongside the bool encoder. The compressed header that follows is
// bool-coded; tile data uses the bool coder too.
//
// State holds (currentByte, bitsInCurrent, outLen). The output buffer
// is supplied by the caller as ArrayView<byte>; outLen is the offset
// of the next byte to write (or the partial-byte position when
// bitsInCurrent > 0). On Finalize the partial byte is flushed.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// In-kernel state for the VP9 raw-bit writer. Mirrors
/// <c>Vp9BitWriter</c> from Vp9FrameHeaderWriter.cs.
/// </summary>
public struct Vp9BitWriterGpuState
{
    /// <summary>Partial-byte accumulator (top <see cref="BitsInCurrent"/> bits valid, MSB-first).</summary>
    public uint Current;

    /// <summary>Number of valid bits accumulated in <see cref="Current"/> (0..7).</summary>
    public int BitsInCurrent;

    /// <summary>Number of complete bytes written to the output view so far.</summary>
    public long OutLen;
}

/// <summary>
/// Static GPU-callable helpers for the VP9 raw-bit writer. Mirrors
/// <c>Vp9BitWriter</c> bit-for-bit. Pack ordering is MSB-first.
/// </summary>
public static class Vp9BitWriterGpu
{
    /// <summary>Initial state: no partial byte, OutLen=0.</summary>
    public static Vp9BitWriterGpuState Init() => new Vp9BitWriterGpuState
    {
        Current = 0,
        BitsInCurrent = 0,
        OutLen = 0,
    };

    /// <summary>
    /// Pack <paramref name="numBits"/> bits of <paramref name="value"/>
    /// MSB-first into the buffer. Mirrors <c>Vp9BitWriter.WriteBits</c>.
    /// </summary>
    public static void WriteBits(
        ref Vp9BitWriterGpuState state,
        ArrayView<byte> outBuf,
        uint value,
        int numBits)
    {
        // numBits in [0, 32]; ILGPU happily compiles the per-bit loop.
        for (int i = numBits - 1; i >= 0; i--)
        {
            uint bit = (value >> i) & 1u;
            state.Current = (state.Current << 1) | bit;
            state.BitsInCurrent++;
            if (state.BitsInCurrent == 8)
            {
                outBuf[state.OutLen] = (byte)state.Current;
                state.OutLen++;
                state.Current = 0;
                state.BitsInCurrent = 0;
            }
        }
    }

    /// <summary>
    /// Pad the current byte with zero bits to align on the next byte
    /// boundary. No-op when already aligned. Mirrors
    /// <c>Vp9BitWriter.PadToByte</c>.
    /// </summary>
    public static void PadToByte(
        ref Vp9BitWriterGpuState state,
        ArrayView<byte> outBuf)
    {
        if (state.BitsInCurrent == 0) return;
        WriteBits(ref state, outBuf, 0u, 8 - state.BitsInCurrent);
    }

    /// <summary>
    /// Append <paramref name="data"/> bytes from <paramref name="srcBuf"/>
    /// starting at <paramref name="srcOffset"/> for <paramref name="length"/>
    /// bytes. Caller must ensure byte alignment (BitsInCurrent == 0).
    /// </summary>
    public static void WriteBytes(
        ref Vp9BitWriterGpuState state,
        ArrayView<byte> outBuf,
        ArrayView<byte> srcBuf,
        long srcOffset,
        long length)
    {
        for (long i = 0; i < length; i++)
        {
            outBuf[state.OutLen] = srcBuf[srcOffset + i];
            state.OutLen++;
        }
    }

    /// <summary>
    /// Finalize: pad the current partial byte with zeros and bump
    /// state.OutLen so it reflects the final byte count. Mirrors
    /// <c>Vp9BitWriter.ToBytes</c>'s pad-trailing-zeros behavior.
    /// Renamed from "Finalize" to avoid shadowing <c>Object.Finalize</c>.
    /// </summary>
    public static void Flush(
        ref Vp9BitWriterGpuState state,
        ArrayView<byte> outBuf)
    {
        if (state.BitsInCurrent > 0)
        {
            // CPU writer left-shifts the partial byte to the top of the
            // byte (so unwritten low bits are zero); we mirror that.
            outBuf[state.OutLen] = (byte)(state.Current << (8 - state.BitsInCurrent));
            state.OutLen++;
            state.Current = 0;
            state.BitsInCurrent = 0;
        }
    }
}
