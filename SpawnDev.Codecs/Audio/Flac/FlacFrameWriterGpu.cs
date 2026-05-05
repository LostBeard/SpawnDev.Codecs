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
// GPU note: this is a single-thread kernel-compatible helper. The
// CRC + bit-writer state are sequential by construction; for
// throughput, multiple frames could be emitted in parallel by
// dispatching one thread per frame with disjoint output regions.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// GPU-callable FLAC v1 frame writer. Single-frame helper - assembles
/// the frame header + CRC-8 + VERBATIM subframes + CRC-16 into the
/// caller's output byte buffer.
/// </summary>
public static class FlacFrameWriterGpu
{
    /// <summary>
    /// Encode one FLAC frame. Returns the number of bytes written to
    /// <paramref name="outBuf"/> starting at <paramref name="outBase"/>.
    /// </summary>
    /// <param name="samples">Per-channel sample buffers concatenated:
    /// channel 0 samples, then channel 1, etc.</param>
    /// <param name="samplesBase">Starting offset for channel 0's samples.</param>
    /// <param name="blockSize">Samples per channel (must equal 4096 for v1).</param>
    /// <param name="channels">Channel count (1..8).</param>
    /// <param name="bps">Bits per sample (must be 16 for v1).</param>
    /// <param name="frameNumber">Sequential frame number.</param>
    /// <param name="outBuf">Output byte buffer.</param>
    /// <param name="outBase">Starting offset in <paramref name="outBuf"/>.</param>
    public static long EncodeFrame(
        ArrayView<int> samples, long samplesBase,
        int blockSize, int channels, int bps,
        ulong frameNumber,
        ArrayView<byte> outBuf, long outBase)
    {
        // ---- 1. Build header bits + CRC-8 byte ----
        // OutLen is buffer-start-relative; pre-seed it to outBase so the
        // bit writer's first byte lands at outBuf[outBase]. The frame
        // length returned at the bottom subtracts outBase. This is what
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
        long headerBytes = hdr.OutLen - outBase;

        // CRC-8 over the header bytes.
        byte crc8 = FlacCrcGpu.Compute8(outBuf, outBase, (int)headerBytes);
        outBuf[outBase + headerBytes] = crc8;
        long postHeader = headerBytes + 1; // frame-relative offset after CRC-8

        // ---- 2. Per-channel VERBATIM subframes ----
        var sub = FlacBitWriterGpu.Init();
        // Re-init the writer to point past the header. The OutLen
        // counter accumulates from buffer start; we want it to start
        // at outBase + postHeader.
        sub.OutLen = outBase + postHeader;
        for (int ch = 0; ch < channels; ch++)
        {
            FlacVerbatimSubframeGpu.Encode(ref sub, outBuf,
                samples, samplesBase + (long)ch * blockSize,
                blockSize, bps);
        }
        FlacBitWriterGpu.AlignToByte(ref sub, outBuf);
        long preFooterLen = sub.OutLen - outBase; // bytes written so far (frame-relative)

        // ---- 3. CRC-16 over all preceding frame bytes ----
        ushort crc16 = FlacCrcGpu.Compute16(outBuf, outBase, (int)preFooterLen);
        outBuf[outBase + preFooterLen] = (byte)(crc16 >> 8);
        outBuf[outBase + preFooterLen + 1] = (byte)(crc16 & 0xFF);

        return preFooterLen + 2;
    }

    /// <summary>
    /// Encode a value as UTF-8-style variable-length integer (FLAC
    /// convention). Mirrors FlacEncoder.WriteUtf8Number bit-for-bit.
    /// </summary>
    private static void WriteUtf8Number(
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
