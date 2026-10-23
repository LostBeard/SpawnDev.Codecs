// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC v1 GPU frame writer. Composes FlacBitWriterGpu +
// FlacVerbatimSubframeGpu + FlacCrcGpu into a complete FLAC frame
// containing one VERBATIM subframe per channel.
//
// V1 simplifications (matches the eventual FlacEncoderGpu):
//   - Block size = 4096 (frame-header bsize code 0b1100 = 12)
//   - Sample rate: caller-supplied code (0..0xF per FLAC spec). For
//     unsupported rates the caller can pass 0xC..0xE and provide the
//     side bytes via the explicit srateSideValue parameter (not yet
//     wired - v1 hardcodes 44.1 kHz code 0x9).
//   - Channels: 1..8 independent (no stereo decorrelation)
//   - Bits per sample: 16 (frame-header code 0b100 = 4)
//   - All VERBATIM subframes (no prediction, no Rice coding)
//   - Fixed blocking strategy (no variable-block-size mode)
//
// Frame layout (bytes):
//   [0]      Frame header bits + CRC-8 byte (header byte length varies
//            with frame number's UTF-8 encoding length)
//   [...]    VERBATIM subframes (one per channel, byte-aligned at end)
//   [-2..-1] CRC-16 over all preceding bytes (big-endian)
//
// GPU form: the frame is written by FlacFrameWriterGpuKernel in parallel pieces (no thread walks a whole
// frame): WriteHeader (one thread per frame: header + CRC-8, at most 12 bytes), the VERBATIM subframe bytes
// (one thread per byte, FlacVerbatimSubframeGpu.PackByteAt), the CRC-16 over the frame (FlacCrcGpu pieces,
// combined per frame). Every offset is known from the frame number and the block shape (FrameBytes).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC v1 frame writer pieces: the frame layout (header length from the frame number, VERBATIM
/// subframes, CRC-16 footer) and the header writer. FlacFrameWriterGpuKernel assembles frames from them.
/// </summary>
public static class FlacFrameWriterGpu
{
    /// <summary>Bytes of the UTF-8-style frame number <see cref="WriteUtf8Number"/> writes for <paramref name="value"/>.</summary>
    public static int Utf8Bytes(ulong value)
        => value < 0x80 ? 1
         : value < 0x800 ? 2
         : value < 0x10000 ? 3
         : value < 0x200000 ? 4
         : value < 0x4000000 ? 5
         : 6;

    /// <summary>Frame header bytes before the CRC-8: 32 fixed bits plus the UTF-8 frame number.</summary>
    public static int HeaderBytes(ulong frameNumber) => 4 + Utf8Bytes(frameNumber);

    /// <summary>Frame-relative offset of the first subframe byte (header + CRC-8).</summary>
    public static int SubframesStart(ulong frameNumber) => HeaderBytes(frameNumber) + 1;

    /// <summary>Frame bytes before the CRC-16 footer.</summary>
    public static long PreFooterBytes(int blockSize, int channels, int bps, ulong frameNumber)
        => SubframesStart(frameNumber) + FlacVerbatimSubframeGpu.SubframesBytes(blockSize, channels, bps);

    /// <summary>Total frame bytes (CRC-16 footer included).</summary>
    public static long FrameBytes(int blockSize, int channels, int bps, ulong frameNumber)
        => PreFooterBytes(blockSize, channels, bps, frameNumber) + 2;

    /// <summary>
    /// Writes the frame header bits and the CRC-8 byte at <paramref name="outBase"/> (v1 codes: block size 4096,
    /// 44.1 kHz, independent channels, 16-bit, fixed blocking). Returns <see cref="SubframesStart"/>.
    /// </summary>
    public static int WriteHeader(ArrayView<byte> outBuf, long outBase, int channels, ulong frameNumber)
    {
        // OutLen is buffer-start-relative; pre-seed it to outBase so the
        // bit writer's first byte lands at outBuf[outBase]. This is what
        // makes batch/parallel encoding (multiple threads each writing to
        // its own outBase slot) correct.
        var hdr = FlacBitWriterGpu.Init();
        hdr.OutLen = outBase;

        // Frame header bits.
        FlacBitWriterGpu.Write(ref hdr, outBuf, (uint)FlacConstants.FrameSyncCode, 14);
        FlacBitWriterGpu.Write(ref hdr, outBuf, 0u, 1); // reserved
        FlacBitWriterGpu.Write(ref hdr, outBuf, 0u, 1); // blocking strategy: fixed
        // Block size code: v1 always 4096 -> code 0b1100 = 12 (per FLAC spec).
        FlacBitWriterGpu.Write(ref hdr, outBuf, 12u, 4);
        // Sample rate code: 0x9 = 44.1 kHz (v1 hardcoded).
        FlacBitWriterGpu.Write(ref hdr, outBuf, 0x9u, 4);
        // Channel assignment: independent N channels = N - 1.
        FlacBitWriterGpu.Write(ref hdr, outBuf, (uint)(channels - 1), 4);
        // Sample size code: 0b100 = 16-bit (v1 hardcoded).
        FlacBitWriterGpu.Write(ref hdr, outBuf, 0b100u, 3);
        FlacBitWriterGpu.Write(ref hdr, outBuf, 0u, 1); // reserved
        // Frame number: UTF-8 encoded.
        WriteUtf8Number(ref hdr, outBuf, frameNumber);
        // Block size code 12 means no side bytes (it's the standard
        // 4096 size); sample rate code 9 means no side bytes (44.1 kHz
        // is in the standard table).
        FlacBitWriterGpu.AlignToByte(ref hdr, outBuf);
        int headerBytes = (int)(hdr.OutLen - outBase);

        // CRC-8 over the header bytes.
        outBuf[outBase + headerBytes] = FlacCrcGpu.Compute8(outBuf, outBase, headerBytes);
        return headerBytes + 1;
    }

    /// <summary>
    /// Encode a value as UTF-8-style variable-length integer (FLAC
    /// convention). Mirrors FlacEncoder.WriteUtf8Number bit-for-bit.
    /// </summary>
    public static void WriteUtf8Number(
        ref FlacBitWriterGpuState w, ArrayView<byte> outBuf, ulong value)
    {
        if (value < 0x80)
        {
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)value, 8);
        }
        else if (value < 0x800)
        {
            FlacBitWriterGpu.Write(ref w, outBuf, 0b110u, 3);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value >> 6), 5);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value & 0x3F), 6);
        }
        else if (value < 0x10000)
        {
            FlacBitWriterGpu.Write(ref w, outBuf, 0b1110u, 4);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value >> 12), 4);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 6) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value & 0x3F), 6);
        }
        else if (value < 0x200000)
        {
            FlacBitWriterGpu.Write(ref w, outBuf, 0b11110u, 5);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value >> 18), 3);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 12) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 6) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value & 0x3F), 6);
        }
        else if (value < 0x4000000)
        {
            FlacBitWriterGpu.Write(ref w, outBuf, 0b111110u, 6);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value >> 24), 2);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 18) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 12) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 6) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value & 0x3F), 6);
        }
        else
        {
            FlacBitWriterGpu.Write(ref w, outBuf, 0b1111110u, 7);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value >> 30), 1);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 24) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 18) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 12) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)((value >> 6) & 0x3F), 6);
            FlacBitWriterGpu.Write(ref w, outBuf, 0b10u, 2);
            FlacBitWriterGpu.Write(ref w, outBuf, (uint)(value & 0x3F), 6);
        }
    }
}
