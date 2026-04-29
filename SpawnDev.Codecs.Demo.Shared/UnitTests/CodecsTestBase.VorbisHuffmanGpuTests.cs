// Cross-backend tests for VorbisHuffmanDecoderGpu.
// Verifies the GPU-callable flat-tree Huffman decoder matches the CPU
// VorbisHuffmanDecoder for arbitrary canonical Vorbis codebooks.

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
    public async Task VorbisHuffmanDecoderGpu_TinyCodebook_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 4 entries with lengths [2, 2, 2, 2] -> codes 00, 01, 10, 11.
            var lengths = new[] { 2, 2, 2, 2 };
            var sequenceToEncode = new[] { 0, 1, 2, 3, 0, 2, 1, 3 };
            await VerifyHuffmanRoundTrip(acc, lengths, sequenceToEncode);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisHuffmanDecoderGpu_MixedLengths_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Mixed lengths exercising deeper tree paths.
            var lengths = new[] { 1, 3, 3, 4, 4, 4, 4 };
            var sequenceToEncode = new[] { 0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 6 };
            await VerifyHuffmanRoundTrip(acc, lengths, sequenceToEncode);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task VerifyHuffmanRoundTrip(
        Accelerator acc, int[] lengths, int[] sequence)
    {
        // Build canonical Huffman codewords from lengths.
        var table = VorbisHuffman.Build(lengths);
        var cpuDecoder = new VorbisHuffmanDecoder(table);

        // Encode the sequence to a bit buffer (LSB-first per Vorbis spec).
        var bw = new VorbisBitWriter();
        foreach (int entry in sequence)
        {
            uint code = table.Codewords[entry];
            int len = table.EntryLengths[entry];
            // Write code MSB-first within the codeword (each bit appended LSB-first to the stream).
            for (int b = len - 1; b >= 0; b--)
            {
                bw.WriteBit((uint)((code >> b) & 1));
            }
        }
        byte[] packetBytes = bw.ToArray();

        // CPU decode reference.
        var cpuReader = new VorbisBitReader(packetBytes);
        var cpuDecoded = new int[sequence.Length];
        for (int i = 0; i < sequence.Length; i++)
        {
            cpuDecoded[i] = cpuDecoder.Decode(ref cpuReader);
        }
        for (int i = 0; i < sequence.Length; i++)
            if (cpuDecoded[i] != sequence[i])
                throw new Exception($"CPU decode mismatch at [{i}]: got {cpuDecoded[i]} expected {sequence[i]}");

        // Flatten tree for GPU.
        var (children, leafToEntry, maxDepth) = cpuDecoder.BuildFlatGpu();

        // GPU decode through a tiny dispatch kernel that calls TryDecode in a loop.
        using var dPacket = acc.Allocate1D<byte>(packetBytes.Length);
        using var dChildren = acc.Allocate1D<int>(children.Length);
        using var dLeafToEntry = acc.Allocate1D<int>(leafToEntry.Length);
        using var dDecoded = acc.Allocate1D<int>(sequence.Length);
        dPacket.View.CopyFromCPU(packetBytes);
        dChildren.View.CopyFromCPU(children);
        dLeafToEntry.View.CopyFromCPU(leafToEntry);

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
            int, int, int>(VorbisHuffmanDecodeKernel);
        kernel(new Index1D(1),
            dPacket.View, dChildren.View, dLeafToEntry.View, dDecoded.View,
            packetBytes.Length, sequence.Length, maxDepth);
        await acc.SynchronizeAsync();

        var gpuDecoded = await dDecoded.CopyToHostAsync();
        for (int i = 0; i < sequence.Length; i++)
            if (gpuDecoded[i] != sequence[i])
                throw new Exception($"GPU decode mismatch at [{i}]: got {gpuDecoded[i]} expected {sequence[i]}");
    }

    private static void VorbisHuffmanDecodeKernel(
        Index1D _,
        ArrayView<byte> packet,
        ArrayView<int> children, ArrayView<int> leafToEntry, ArrayView<int> decoded,
        int packetLen, int count, int maxDepth)
    {
        var state = VorbisBitReaderGpu.Init(packetLen);
        for (int i = 0; i < count; i++)
        {
            int e = VorbisHuffmanDecoderGpu.TryDecode(
                ref state, packet, children, 0, leafToEntry, 0, maxDepth);
            decoded[i] = e;
        }
    }
}
