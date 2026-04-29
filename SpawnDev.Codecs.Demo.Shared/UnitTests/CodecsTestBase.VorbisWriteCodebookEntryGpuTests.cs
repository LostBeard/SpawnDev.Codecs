// Cross-backend test for VorbisBitWriterGpu.WriteCodebookEntry.
// Verifies the GPU codebook entry writer matches the CPU
// VorbisAudioEncoder.WriteCodebookEntry reference for a sequence of
// codeword writes.

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
    public async Task VorbisBitWriterGpu_WriteCodebookEntry_SequenceMatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 8-entry canonical Huffman codebook (lengths produce a valid prefix code).
            // Valid prefix code: 2x len2 + 2x len3 + 4x len4 = 0.5+0.25+0.25 = 1.0.
            int[] lengths = { 2, 2, 3, 3, 4, 4, 4, 4 };
            var table = VorbisHuffman.Build(lengths);

            // Sequence to encode + expected stream from CPU writer.
            int[] sequence = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 7, 3, 1 };

            var cpuWriter = new VorbisBitWriter();
            foreach (int e in sequence)
            {
                uint code = table.Codewords[e];
                int len = table.EntryLengths[e];
                for (int b = len - 1; b >= 0; b--)
                    cpuWriter.WriteBit((uint)((code >> b) & 1));
            }
            byte[] cpuBytes = cpuWriter.ToArray();

            // GPU.
            int outBufSize = Math.Max(16, cpuBytes.Length + 4);
            using var dCodes = acc.Allocate1D<uint>(table.Codewords.Length);
            using var dLengths = acc.Allocate1D<int>(table.EntryLengths.Length);
            using var dSequence = acc.Allocate1D<int>(sequence.Length);
            using var dOut = acc.Allocate1D<byte>(outBufSize);
            using var dOutLen = acc.Allocate1D<long>(1);
            dCodes.View.CopyFromCPU(table.Codewords);
            dLengths.View.CopyFromCPU(table.EntryLengths);
            dSequence.View.CopyFromCPU(sequence);
            dOut.View.CopyFromCPU(new byte[outBufSize]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>,
                ArrayView<byte>, ArrayView<long>, int>(WriteCodebookKernel);
            kernel(new Index1D(1),
                dCodes.View, dLengths.View, dSequence.View,
                dOut.View, dOutLen.View, sequence.Length);
            await acc.SynchronizeAsync();

            long gpuLen = (await dOutLen.CopyToHostAsync())[0];
            var gpuFull = await dOut.CopyToHostAsync();
            var gpuBytes = new byte[gpuLen];
            Array.Copy(gpuFull, gpuBytes, gpuLen);

            if (cpuBytes.Length != gpuBytes.Length)
                throw new Exception($"len mismatch: cpu={cpuBytes.Length} gpu={gpuBytes.Length}");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte[{i}]: cpu={cpuBytes[i]:X2} gpu={gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void WriteCodebookKernel(
        Index1D _,
        ArrayView<uint> codes, ArrayView<int> lengths, ArrayView<int> sequence,
        ArrayView<byte> outBuf, ArrayView<long> outLen, int count)
    {
        var state = VorbisBitWriterGpu.Init();
        for (int i = 0; i < count; i++)
        {
            VorbisBitWriterGpu.WriteCodebookEntry(
                ref state, outBuf, codes, 0, lengths, 0, sequence[i]);
        }
        VorbisBitWriterGpu.Finish(ref state, outBuf);
        outLen[0] = state.OutLen;
    }
}
