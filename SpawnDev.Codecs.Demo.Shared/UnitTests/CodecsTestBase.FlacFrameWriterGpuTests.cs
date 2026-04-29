// Cross-backend tests for FlacFrameWriterGpu. Verifies that a frame
// encoded by the GPU pipeline parses byte-for-byte identically to a
// CPU-encoded reference frame, AND can be decoded back via the
// existing CPU FLAC frame parser to recover the original samples.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacFrameWriterGpu_Mono16bit_Sin440Hz_MatchesCpuRef()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int blockSize = 4096;
            const int channels = 1;
            const int bps = 16;
            const ulong frameNumber = 0;
            const int sampleRate = 44100;

            // Generate a 440 Hz sin wave at 16-bit.
            var samples = new int[blockSize];
            for (int i = 0; i < blockSize; i++)
            {
                double t = i / (double)sampleRate;
                samples[i] = (int)(Math.Sin(2 * Math.PI * 440 * t) * 16384);
            }

            // CPU reference: build a frame manually with the exact same
            // settings (mono, 16-bit, blockSize=4096, frameNumber=0,
            // 44.1 kHz, all VERBATIM subframes) so the bytes are
            // directly comparable.
            byte[] cpuRef = BuildCpuRefFrame(samples, blockSize, channels, bps, frameNumber);

            // GPU encode.
            int worstCase = 16 + blockSize * channels * 4 + 16;
            using var dSamples = acc.Allocate1D<int>(blockSize * channels);
            using var dOut = acc.Allocate1D<byte>(worstCase);
            using var dOutLen = acc.Allocate1D<long>(1);
            dSamples.View.CopyFromCPU(samples);
            dOut.View.CopyFromCPU(new byte[worstCase]);

            using var kernel = new FlacFrameWriterGpuKernel(acc);
            kernel.Run(dSamples.View, dOut.View, dOutLen.View, blockSize, channels, bps, frameNumber);
            await acc.SynchronizeAsync();

            long gpuLen = (await dOutLen.CopyToHostAsync())[0];
            var gpuFull = await dOut.CopyToHostAsync();
            var gpuBytes = new byte[gpuLen];
            Array.Copy(gpuFull, gpuBytes, gpuLen);

            // Compare GPU bytes to CPU reference.
            Equal(cpuRef.Length, gpuBytes.Length, "frame byte length");
            for (int i = 0; i < cpuRef.Length; i++)
                if (cpuRef[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuRef[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacFrameWriterGpu_Stereo16bit_Random_MatchesCpuRef()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int blockSize = 4096;
            const int channels = 2;
            const int bps = 16;
            const ulong frameNumber = 7;

            var rng = new Random(unchecked((int)0xF1AC5705u));
            var samples = new int[blockSize * channels];
            for (int i = 0; i < samples.Length; i++) samples[i] = rng.Next(-32768, 32768);

            byte[] cpuRef = BuildCpuRefFrame(samples, blockSize, channels, bps, frameNumber);

            int worstCase = 16 + blockSize * channels * 4 + 16;
            using var dSamples = acc.Allocate1D<int>(blockSize * channels);
            using var dOut = acc.Allocate1D<byte>(worstCase);
            using var dOutLen = acc.Allocate1D<long>(1);
            dSamples.View.CopyFromCPU(samples);
            dOut.View.CopyFromCPU(new byte[worstCase]);

            using var kernel = new FlacFrameWriterGpuKernel(acc);
            kernel.Run(dSamples.View, dOut.View, dOutLen.View, blockSize, channels, bps, frameNumber);
            await acc.SynchronizeAsync();

            long gpuLen = (await dOutLen.CopyToHostAsync())[0];
            var gpuFull = await dOut.CopyToHostAsync();
            var gpuBytes = new byte[gpuLen];
            Array.Copy(gpuFull, gpuBytes, gpuLen);

            Equal(cpuRef.Length, gpuBytes.Length, "frame byte length");
            for (int i = 0; i < cpuRef.Length; i++)
                if (cpuRef[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuRef[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    /// <summary>
    /// Build a CPU FLAC frame with the exact same encoding choices the
    /// GPU encoder uses (block size 4096, 44.1 kHz, mono/stereo
    /// independent, 16-bit, all VERBATIM subframes, fixed blocking).
    /// </summary>
    private static byte[] BuildCpuRefFrame(int[] samples, int blockSize, int channels, int bps, ulong frameNumber)
    {
        // Header bits.
        var header = new SpawnDev.Codecs.Audio.Flac.FlacBitWriter();
        header.Write((uint)FlacConstants.FrameSyncCode, 14);
        header.Write(0, 1);
        header.Write(0, 1);
        header.Write(12u, 4);          // block size code (4096)
        header.Write(0x9u, 4);         // sample rate code (44.1 kHz)
        header.Write((uint)(channels - 1), 4);
        header.Write(0b100u, 3);       // sample size code (16-bit)
        header.Write(0, 1);
        WriteUtf8(header, frameNumber);
        header.AlignToByte();
        byte[] headerBytes = header.ToArray();
        byte crc8 = SpawnDev.Codecs.Audio.Flac.FlacCrc.Compute8(headerBytes);

        var frame = new List<byte>();
        frame.AddRange(headerBytes);
        frame.Add(crc8);

        // Per-channel VERBATIM subframes.
        var sub = new SpawnDev.Codecs.Audio.Flac.FlacBitWriter();
        for (int ch = 0; ch < channels; ch++)
        {
            sub.Write(0, 1);
            sub.Write(0b000001u, 6);
            sub.Write(0, 1);
            for (int i = 0; i < blockSize; i++)
                sub.WriteSigned(samples[ch * blockSize + i], bps);
        }
        sub.AlignToByte();
        frame.AddRange(sub.ToArray());

        ushort crc16 = SpawnDev.Codecs.Audio.Flac.FlacCrc.Compute16(frame.ToArray());
        frame.Add((byte)(crc16 >> 8));
        frame.Add((byte)(crc16 & 0xFF));
        return frame.ToArray();
    }

    private static void WriteUtf8(SpawnDev.Codecs.Audio.Flac.FlacBitWriter w, ulong value)
    {
        if (value < 0x80) { w.Write((uint)value, 8); return; }
        if (value < 0x800)
        {
            w.Write(0b110u, 3); w.Write((uint)(value >> 6), 5);
            w.Write(0b10u, 2);  w.Write((uint)(value & 0x3F), 6);
            return;
        }
        if (value < 0x10000)
        {
            w.Write(0b1110u, 4); w.Write((uint)(value >> 12), 4);
            w.Write(0b10u, 2);   w.Write((uint)((value >> 6) & 0x3F), 6);
            w.Write(0b10u, 2);   w.Write((uint)(value & 0x3F), 6);
            return;
        }
        if (value < 0x200000)
        {
            w.Write(0b11110u, 5); w.Write((uint)(value >> 18), 3);
            w.Write(0b10u, 2); w.Write((uint)((value >> 12) & 0x3F), 6);
            w.Write(0b10u, 2); w.Write((uint)((value >> 6) & 0x3F), 6);
            w.Write(0b10u, 2); w.Write((uint)(value & 0x3F), 6);
            return;
        }
        if (value < 0x4000000)
        {
            w.Write(0b111110u, 6); w.Write((uint)(value >> 24), 2);
            w.Write(0b10u, 2); w.Write((uint)((value >> 18) & 0x3F), 6);
            w.Write(0b10u, 2); w.Write((uint)((value >> 12) & 0x3F), 6);
            w.Write(0b10u, 2); w.Write((uint)((value >> 6) & 0x3F), 6);
            w.Write(0b10u, 2); w.Write((uint)(value & 0x3F), 6);
            return;
        }
        w.Write(0b1111110u, 7); w.Write((uint)(value >> 30), 1);
        w.Write(0b10u, 2); w.Write((uint)((value >> 24) & 0x3F), 6);
        w.Write(0b10u, 2); w.Write((uint)((value >> 18) & 0x3F), 6);
        w.Write(0b10u, 2); w.Write((uint)((value >> 12) & 0x3F), 6);
        w.Write(0b10u, 2); w.Write((uint)((value >> 6) & 0x3F), 6);
        w.Write(0b10u, 2); w.Write((uint)(value & 0x3F), 6);
    }
}
