// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC v1 GPU decoder integration class. Symmetric to FlacEncoderGpu.
// Wraps FlacFrameReaderGpuKernel into a callable DecodeStreamAsync
// API that consumes a .flac byte stream produced by either the GPU
// FlacEncoderGpu or any conforming FLAC encoder (within v1's
// constraints: 4096-block, 44.1 kHz, 16-bit, all-VERBATIM
// subframes, fixed blocking).
//
// Stream parse (host - metadata only):
//   1. "fLaC" marker (4 bytes)
//   2. STREAMINFO metadata block (header 4 + payload 34 = 38 bytes)
//      Extract: blockSize, sampleRateHz, channels, bps, totalSamples
//   3. Skip any non-STREAMINFO metadata blocks (until isLast=1)
// Frame decode (GPU - codec-data math):
//   For each frame: dispatch FlacFrameReaderGpuKernel
//
// V1 simplifications match FlacEncoderGpu.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// V1 GPU FLAC decoder integration class. Per-frame decode runs on
/// the accelerator via FlacFrameReaderGpuKernel; stream-level parse
/// (fLaC marker + STREAMINFO + metadata block scan) runs on host.
/// </summary>
public sealed class FlacDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly FlacFrameReaderGpuKernel _frameKernel;

    /// <summary>Construct a decoder bound to <paramref name="accelerator"/>.</summary>
    public FlacDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _frameKernel = new FlacFrameReaderGpuKernel(accelerator);
    }

    /// <summary>
    /// Decode a complete FLAC byte stream. Returns interleaved PCM
    /// samples (channel 0[0], channel 1[0], channel 0[1], ...).
    /// </summary>
    public async Task<FlacGpuDecodeResult> DecodeStreamAsync(byte[] flacBytes)
    {
        if (flacBytes is null) throw new ArgumentNullException(nameof(flacBytes));
        if (flacBytes.Length < 42) // 4 marker + 38 STREAMINFO
            throw new ArgumentException("FLAC stream too short.", nameof(flacBytes));

        // ---- Parse "fLaC" marker (host) ----
        if (flacBytes[0] != 'f' || flacBytes[1] != 'L'
            || flacBytes[2] != 'a' || flacBytes[3] != 'C')
            throw new ArgumentException("Not a FLAC stream (missing fLaC marker).", nameof(flacBytes));

        // ---- Parse metadata blocks (host) ----
        int pos = 4;
        int blockSize = 0;
        int sampleRateHz = 0;
        int channels = 0;
        int bps = 0;
        ulong totalSamples = 0;
        while (true)
        {
            byte hdr = flacBytes[pos];
            bool isLast = (hdr & 0x80) != 0;
            int blockType = hdr & 0x7F;
            int blockLen = (flacBytes[pos + 1] << 16) | (flacBytes[pos + 2] << 8) | flacBytes[pos + 3];
            pos += 4;
            if (blockType == FlacConstants.MetadataStreamInfo)
            {
                // STREAMINFO payload (34 bytes): min/max block (32) +
                // min/max frame (48) + sample_rate (20) + ch-1 (3) +
                // bps-1 (5) + total_samples (36) + md5 (128).
                var r = new SpawnDev.Codecs.Audio.Flac.FlacBitReader(
                    new ReadOnlySpan<byte>(flacBytes, pos, blockLen));
                int minBlock = (int)r.ReadBits(16);
                int maxBlock = (int)r.ReadBits(16);
                blockSize = maxBlock; // V1 fixed block size, min == max.
                int minFrame = (int)r.ReadBits(24);
                int maxFrame = (int)r.ReadBits(24);
                sampleRateHz = (int)r.ReadBits(20);
                channels = (int)r.ReadBits(3) + 1;
                bps = (int)r.ReadBits(5) + 1;
                ulong hi = r.ReadBits(4);
                ulong lo = r.ReadBits(32);
                totalSamples = (hi << 32) | lo;
                // Skip MD5 (16 bytes already counted in blockLen).
            }
            pos += blockLen;
            if (isLast) break;
        }

        if (channels == 0 || blockSize == 0)
            throw new InvalidDataException("STREAMINFO did not provide block size or channels.");
        if (totalSamples == 0)
            throw new InvalidDataException("V1 GPU decoder requires STREAMINFO totalSamples to be set.");

        int totalPerChannel = (int)totalSamples;
        int frameCount = (totalPerChannel + blockSize - 1) / blockSize;
        var output = new int[totalPerChannel * channels];

        // ---- Per-frame decode (GPU) ----
        // Each call uploads the full remaining bytes + decodes one
        // frame. For a v1 demo this is sufficient; throughput
        // optimization (single upload + multiple kernel dispatches)
        // is a follow-up.
        using var dData = _accelerator.Allocate1D<byte>(flacBytes.Length);
        using var dSamples = _accelerator.Allocate1D<int>(blockSize * channels);
        using var dStatus = _accelerator.Allocate1D<int>(1);
        using var dFrameLen = _accelerator.Allocate1D<long>(1);
        dData.View.CopyFromCPU(flacBytes);

        long frameBase = pos;
        for (int frameIdx = 0; frameIdx < frameCount; frameIdx++)
        {
            // The GPU reader needs to know the frame length up front
            // to verify CRC-16. Compute it by scanning forward to the
            // next sync code (or end-of-stream). For v1 with
            // fixed-block-size we can predict the length precisely:
            //   header bytes = 5 (sync + flags + fixed-codes + 1-byte
            //                     UTF-8 frame number assuming idx < 128)
            //   + 1 (CRC8)
            //   + per-channel subframe = 8-bit hdr + samples bps bits
            //                          = 1 + (blockSize * bps + 7) / 8
            //   + 2 (CRC16)
            // For frame numbers >= 128 the UTF-8 length grows; we
            // compute it on host since pos is host-known.
            int fLen = (int)EstimateFrameLength((ulong)frameIdx, blockSize, channels, bps);
            // Alternative: scan from frameBase to find next sync code.
            // For robustness we use the scan when fLen would overshoot.
            long actualFLen = ScanFrameLength(flacBytes, (int)frameBase, fLen);

            _frameKernel.Run(dData.View, dSamples.View, dStatus.View, dFrameLen.View,
                frameBase, (int)actualFLen, blockSize, channels, bps);
            await _accelerator.SynchronizeAsync();

            int status = (await dStatus.CopyToHostAsync())[0];
            long consumed = (await dFrameLen.CopyToHostAsync())[0];
            if (status != 0)
                throw new InvalidDataException(
                    $"FLAC frame {frameIdx} decode failed at byte {frameBase} (status={status}).");

            // Copy decoded samples (channel-major) into output (interleaved).
            var frameSamples = await dSamples.CopyToHostAsync();
            int frameSampleStart = frameIdx * blockSize * channels;
            for (int ch = 0; ch < channels; ch++)
            {
                for (int i = 0; i < blockSize; i++)
                {
                    output[frameSampleStart + i * channels + ch] = frameSamples[ch * blockSize + i];
                }
            }
            frameBase += consumed;
        }

        return new FlacGpuDecodeResult
        {
            SampleRateHz = sampleRateHz,
            Channels = channels,
            BitsPerSample = bps,
            BlockSize = blockSize,
            TotalSamplesPerChannel = totalPerChannel,
            InterleavedSamples = output,
        };
    }

    /// <summary>
    /// Estimate worst-case frame length in bytes for a v1 frame with
    /// the given parameters. Used to size the GPU read range; the
    /// actual length is recovered from the kernel's frameLen output.
    /// </summary>
    private static long EstimateFrameLength(ulong frameIndex, int blockSize, int channels, int bps)
    {
        // Header: 32 bits fixed = 4 bytes + UTF-8 frame number length.
        int utf8Bytes = frameIndex < 0x80 ? 1
            : frameIndex < 0x800 ? 2
            : frameIndex < 0x10000 ? 3
            : frameIndex < 0x200000 ? 4
            : frameIndex < 0x4000000 ? 5
            : frameIndex < 0x80000000 ? 6
            : 7;
        int headerBytes = 4 + utf8Bytes;
        int crc8Bytes = 1;
        // Each VERBATIM subframe: 8-bit header + blockSize * bps bits.
        long subframeBits = (long)channels * (8 + (long)blockSize * bps);
        long subframeBytes = (subframeBits + 7) / 8;
        int crc16Bytes = 2;
        return headerBytes + crc8Bytes + subframeBytes + crc16Bytes;
    }

    /// <summary>
    /// Scan forward from <paramref name="start"/> for the next FLAC
    /// frame sync code (0xFFF8 / 0xFFFA). Used to bound the current
    /// frame's length when the estimate may overshoot. Returns the
    /// distance to the next sync, or remaining bytes if none found.
    /// </summary>
    private static long ScanFrameLength(byte[] data, int start, int hint)
    {
        int searchStart = Math.Min(start + hint - 4, data.Length - 2);
        if (searchStart <= start) return data.Length - start;
        for (int i = searchStart; i + 1 < data.Length; i++)
        {
            // Sync code 0x3FFE in the top 14 bits of the first 2 bytes.
            // Byte 0 = 0xFF, byte 1 = 0xF8 (with bit 1 = blocking strategy).
            if (data[i] == 0xFF && (data[i + 1] & 0xFE) == 0xF8) return i - start;
        }
        return data.Length - start;
    }

    /// <summary>Release accelerator-bound resources.</summary>
    public void Dispose()
    {
        _frameKernel.Dispose();
    }
}

/// <summary>Result of a GPU FLAC decode.</summary>
public sealed record FlacGpuDecodeResult
{
    /// <summary>Sample rate in Hz.</summary>
    public required int SampleRateHz { get; init; }
    /// <summary>Channel count.</summary>
    public required int Channels { get; init; }
    /// <summary>Bits per sample.</summary>
    public required int BitsPerSample { get; init; }
    /// <summary>Block size (samples per frame per channel).</summary>
    public required int BlockSize { get; init; }
    /// <summary>Total samples per channel across the stream.</summary>
    public required int TotalSamplesPerChannel { get; init; }
    /// <summary>Interleaved PCM samples: channel 0[0], channel 1[0], ...</summary>
    public required int[] InterleavedSamples { get; init; }
}
