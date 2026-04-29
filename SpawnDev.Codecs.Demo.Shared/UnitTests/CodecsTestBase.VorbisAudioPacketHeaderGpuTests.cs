// Cross-backend test for VorbisAudioPacketHeaderGpu.Parse.
// Verifies the GPU per-packet header parser matches the CPU
// VorbisAudioPacketHeaderParser reference for short and long-block
// configurations.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisAudioPacketHeaderGpu_ShortAndLongModes_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 2 modes: mode 0 short (blockFlag=false), mode 1 long (true).
            // ilog(modes-1) = ilog(1) = 1 bit mode index.
            int[] modeBlockFlags = { 0, 1 };
            const int blockSize0 = 256;
            const int blockSize1 = 2048;
            const int modeBits = 1; // ilog(2-1) = 1

            // --- Test packet 1: short mode (mode 0) ---
            // Bits: type=0, mode=0, no window flags.
            var bw1 = new VorbisBitWriter();
            bw1.WriteBit(0u);  // packet type = 0 (audio)
            bw1.WriteBit(0u);  // mode = 0
            byte[] packet1 = bw1.ToArray();
            var gpu1 = await ParseGpu(acc, packet1, modeBlockFlags, modeBits, blockSize0, blockSize1);
            if (gpu1.ModeNumber != 0) throw new Exception($"short mode: {gpu1.ModeNumber}");
            if (gpu1.BlockSize != blockSize0) throw new Exception($"short blockSize: {gpu1.BlockSize}");
            if (gpu1.IsLongBlock != 0) throw new Exception($"short isLong: {gpu1.IsLongBlock}");

            // --- Test packet 2: long mode (mode 1) with both window flags = 1 ---
            var bw2 = new VorbisBitWriter();
            bw2.WriteBit(0u);  // packet type = 0
            bw2.WriteBit(1u);  // mode = 1 (long)
            bw2.WriteBit(1u);  // prevWindowLong = 1
            bw2.WriteBit(1u);  // nextWindowLong = 1
            byte[] packet2 = bw2.ToArray();
            var gpu2 = await ParseGpu(acc, packet2, modeBlockFlags, modeBits, blockSize0, blockSize1);
            if (gpu2.ModeNumber != 1) throw new Exception($"long mode: {gpu2.ModeNumber}");
            if (gpu2.BlockSize != blockSize1) throw new Exception($"long blockSize: {gpu2.BlockSize}");
            if (gpu2.IsLongBlock != 1) throw new Exception($"long isLong: {gpu2.IsLongBlock}");
            if (gpu2.PreviousWindowLong != 1) throw new Exception($"prev: {gpu2.PreviousWindowLong}");
            if (gpu2.NextWindowLong != 1) throw new Exception($"next: {gpu2.NextWindowLong}");

            // --- Test packet 3: long mode + asymmetric window flags ---
            var bw3 = new VorbisBitWriter();
            bw3.WriteBit(0u);
            bw3.WriteBit(1u);  // long
            bw3.WriteBit(0u);  // prev = 0
            bw3.WriteBit(1u);  // next = 1
            byte[] packet3 = bw3.ToArray();
            var gpu3 = await ParseGpu(acc, packet3, modeBlockFlags, modeBits, blockSize0, blockSize1);
            if (gpu3.PreviousWindowLong != 0) throw new Exception($"asym prev: {gpu3.PreviousWindowLong}");
            if (gpu3.NextWindowLong != 1) throw new Exception($"asym next: {gpu3.NextWindowLong}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task<VorbisAudioPacketHeaderGpuResult> ParseGpu(
        Accelerator acc, byte[] packet, int[] modeBlockFlags,
        int modeBits, int blockSize0, int blockSize1)
    {
        using var dPacket = acc.Allocate1D<byte>(packet.Length);
        using var dFlags = acc.Allocate1D<int>(modeBlockFlags.Length);
        using var dResult = acc.Allocate1D<VorbisAudioPacketHeaderGpuResult>(1);
        dPacket.View.CopyFromCPU(packet);
        dFlags.View.CopyFromCPU(modeBlockFlags);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>,
            ArrayView<VorbisAudioPacketHeaderGpuResult>,
            int, int, int, int>(ParseKernel);
        kernel(new Index1D(1),
            dPacket.View, dFlags.View, dResult.View,
            packet.Length, modeBits, blockSize0, blockSize1);
        await acc.SynchronizeAsync();

        return (await dResult.CopyToHostAsync())[0];
    }

    private static void ParseKernel(
        Index1D _,
        ArrayView<byte> packet, ArrayView<int> modeBlockFlags,
        ArrayView<VorbisAudioPacketHeaderGpuResult> result,
        int packetLen, int modeBits, int blockSize0, int blockSize1)
    {
        var state = VorbisBitReaderGpu.Init(packetLen);
        result[0] = VorbisAudioPacketHeaderGpu.Parse(
            ref state, packet, modeBits,
            modeBlockFlags, 0, blockSize0, blockSize1);
    }
}
