// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC v1 GPU encoder integration class. Wraps the
// FlacFrameWriterGpuKernel + FlacBitWriterGpu primitives into a
// callable EncodeStream API that produces a complete FLAC byte
// stream the existing CPU FlacDecoder (or any conforming decoder)
// can parse back to PCM.
//
// V1 simplifications:
//   - Block size = 4096 (last frame may be shorter, but v1 requires
//     totalSamples to be a multiple of 4096; we throw otherwise).
//   - Sample rate = 44.1 kHz hardcoded.
//   - Bits per sample = 16 hardcoded.
//   - Channels: 1..8 independent.
//   - No FIXED / LPC predictor (all VERBATIM subframes).
//   - No MD5 signature (zeros - allowed by FLAC spec).
//   - No optional metadata blocks (just fLaC + STREAMINFO).
//
// Stream layout:
//   [0..3]    "fLaC" marker
//   [4..7]    STREAMINFO header: 0x80 (last) | 0x00 (type=0) | 0x000022 (length=34)
//   [8..41]   STREAMINFO payload (34 bytes per FLAC spec)
//   [42..]    Audio frames, one per blockSize samples (per channel)

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// V1 GPU FLAC encoder integration class. Per-frame math runs on
/// the accelerator via FlacFrameWriterGpuKernel; stream-level wrap
/// (fLaC marker + STREAMINFO metadata block) runs on the host since
/// it is one-shot metadata serialization with no codec-data math.
/// </summary>
public sealed class FlacEncoderGpu : IDisposable
{
    /// <summary>V1 constants - block size enforced.</summary>
    public const int BlockSize = 4096;
    /// <summary>V1 constant - sample rate.</summary>
    public const int SampleRateHz = 44100;
    /// <summary>V1 constant - bits per sample.</summary>
    public const int BitsPerSample = 16;

    private readonly Accelerator _accelerator;
    private readonly FlacFrameWriterGpuKernel _frameKernel;

    /// <summary>Construct an encoder bound to <paramref name="accelerator"/>.</summary>
    public FlacEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _frameKernel = new FlacFrameWriterGpuKernel(accelerator);
    }

    /// <summary>
    /// Encode interleaved PCM samples (channel 0[0], channel 1[0],
    /// channel 0[1], channel 1[1], ...) to a complete FLAC byte stream.
    /// Total sample count per channel must be a multiple of BlockSize
    /// (4096) for v1.
    /// </summary>
    public async Task<byte[]> EncodeStreamAsync(
        ReadOnlyMemory<int> interleavedSamples, int channels)
    {
        if (channels < 1 || channels > 8)
            throw new ArgumentException("Channels must be in [1, 8].", nameof(channels));
        int totalSamples = interleavedSamples.Length;
        if (totalSamples % channels != 0)
            throw new ArgumentException(
                "Interleaved sample length must be a multiple of channels.", nameof(interleavedSamples));
        int totalPerChannel = totalSamples / channels;
        if (totalPerChannel == 0 || totalPerChannel % BlockSize != 0)
            throw new ArgumentException(
                $"V1 requires totalPerChannel to be a positive multiple of {BlockSize}.",
                nameof(interleavedSamples));

        // ---- Stream-level metadata (host) ----
        var output = new List<byte>();
        // "fLaC" marker.
        output.AddRange(FlacConstants.StreamMarker);
        // STREAMINFO metadata block: header (4 bytes) + payload (34 bytes).
        // Header: isLast=1, type=STREAMINFO, length=34 (24-bit big-endian).
        output.Add(0x80);
        output.Add(0x00);
        output.Add(0x00);
        output.Add(0x22);
        output.AddRange(BuildStreamInfoPayload(
            blockSize: BlockSize,
            sampleRateHz: SampleRateHz,
            channels: channels,
            bitsPerSample: BitsPerSample,
            totalSamples: (ulong)totalPerChannel));

        // ---- Audio frames (GPU) ----
        int frameCount = totalPerChannel / BlockSize;

        // Per-frame upload + dispatch loop. For v1 simplicity we
        // upload + encode per frame; future optimization is to
        // batch-encode many frames in one dispatch (the kernel is
        // single-thread so the wins come from amortizing the
        // dispatch overhead, not parallelism).
        // Worst-case bytes per frame: header (~16) + samples + CRC (16).
        int worstCasePerFrame = 32 + BlockSize * channels * 4;
        using var dSamples = _accelerator.Allocate1D<int>(BlockSize * channels);
        using var dOut = _accelerator.Allocate1D<byte>(worstCasePerFrame);
        using var dOutLen = _accelerator.Allocate1D<long>(1);

        // Per-channel sample buffer for upload to dSamples (channel-
        // major: ch0[0..N-1], ch1[0..N-1], ...).
        var perChannelBuffer = new int[BlockSize * channels];

        for (int frameIdx = 0; frameIdx < frameCount; frameIdx++)
        {
            // De-interleave: channel-major layout for the GPU encoder.
            int frameSampleStart = frameIdx * BlockSize * channels;
            for (int ch = 0; ch < channels; ch++)
            {
                for (int i = 0; i < BlockSize; i++)
                {
                    perChannelBuffer[ch * BlockSize + i] =
                        interleavedSamples.Span[frameSampleStart + i * channels + ch];
                }
            }

            dSamples.View.CopyFromCPU(perChannelBuffer);
            dOut.View.CopyFromCPU(new byte[worstCasePerFrame]);

            _frameKernel.Run(dSamples.View, dOut.View, dOutLen.View,
                BlockSize, channels, BitsPerSample, (ulong)frameIdx);
            await _accelerator.SynchronizeAsync();

            long frameLen = (await dOutLen.CopyToHostAsync())[0];
            var frameBytes = await dOut.CopyToHostAsync();
            for (int i = 0; i < frameLen; i++) output.Add(frameBytes[i]);
        }

        return output.ToArray();
    }

    /// <summary>
    /// Build the 34-byte STREAMINFO payload per FLAC spec
    /// (sec 11.4 STREAMINFO). MD5 set to all-zeros (v1 simplification;
    /// FLAC spec permits this).
    /// </summary>
    private static byte[] BuildStreamInfoPayload(
        int blockSize, int sampleRateHz, int channels, int bitsPerSample,
        ulong totalSamples)
    {
        var w = new FlacBitWriter();
        // min_block_size (16 bits), max_block_size (16 bits).
        w.Write((uint)blockSize, 16);
        w.Write((uint)blockSize, 16);
        // min_frame_size (24 bits, 0 = unknown), max_frame_size (24 bits, 0 = unknown).
        w.Write(0u, 24);
        w.Write(0u, 24);
        // sample_rate (20 bits), channels-1 (3 bits), bps-1 (5 bits),
        // total_samples (36 bits).
        w.Write((uint)sampleRateHz, 20);
        w.Write((uint)(channels - 1), 3);
        w.Write((uint)(bitsPerSample - 1), 5);
        w.Write((uint)(totalSamples >> 32) & 0xFu, 4);
        w.Write((uint)(totalSamples & 0xFFFFFFFFu), 32);
        // 16-byte MD5 (all zeros for v1).
        for (int i = 0; i < 16; i++) w.Write(0u, 8);
        return w.ToArray();
    }

    /// <summary>Release accelerator-bound resources.</summary>
    public void Dispose()
    {
        _frameKernel.Dispose();
    }
}
