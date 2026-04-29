// Cross-backend test for VorbisHuffmanCodebookSetGpu.Build. Verifies
// the flattened multi-codebook representation produces the same
// per-codebook Huffman decode results as separate VorbisHuffmanDecoder
// instances when consumed by VorbisHuffmanDecoderGpu.TryDecode.

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
    public async Task VorbisHuffmanCodebookSetGpu_FlattenedDecode_MatchesSingle()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Three small codebooks with different shapes.
            var book0 = new VorbisCodebook
            {
                Dimensions = 2,
                Entries = 4,
                Ordered = false,
                Sparse = false,
                Lengths = new[] { 2, 2, 2, 2 },
                LookupType = 0,
                MinValue = 0,
                DeltaValue = 0,
                ValueBits = 0,
                SequenceP = false,
                Multiplicands = Array.Empty<int>(),
            };
            var book1 = new VorbisCodebook
            {
                Dimensions = 1,
                Entries = 8,
                Ordered = false,
                Sparse = false,
                Lengths = new[] { 1, 3, 3, 4, 4, 4, 4, 0 },
                LookupType = 0,
                MinValue = 0,
                DeltaValue = 0,
                ValueBits = 0,
                SequenceP = false,
                Multiplicands = Array.Empty<int>(),
            };
            var book2 = new VorbisCodebook
            {
                Dimensions = 1,
                Entries = 5,
                Ordered = false,
                Sparse = false,
                Lengths = new[] { 1, 2, 3, 4, 4 },
                LookupType = 0,
                MinValue = 0,
                DeltaValue = 0,
                ValueBits = 0,
                SequenceP = false,
                Multiplicands = Array.Empty<int>(),
            };
            var books = new[] { book0, book1, book2 };
            var flat = VorbisHuffmanCodebookSetGpu.Build(books);

            // Bit pattern: codebook 0 entry 2 + codebook 1 entry 0 + codebook 2 entry 4.
            var bw = new VorbisBitWriter();
            var t0 = VorbisHuffman.Build(book0.Lengths);
            var t1 = VorbisHuffman.Build(book1.Lengths);
            var t2 = VorbisHuffman.Build(book2.Lengths);
            WriteCanonical(bw, t0, 2);
            WriteCanonical(bw, t1, 0);
            WriteCanonical(bw, t2, 4);
            byte[] packet = bw.ToArray();

            // Reference: walk through individual VorbisHuffmanDecoder.Decode.
            var d0 = new VorbisHuffmanDecoder(t0);
            var d1 = new VorbisHuffmanDecoder(t1);
            var d2 = new VorbisHuffmanDecoder(t2);
            var refReader = new VorbisBitReader(packet);
            int e0Cpu = d0.Decode(ref refReader);
            int e1Cpu = d1.Decode(ref refReader);
            int e2Cpu = d2.Decode(ref refReader);
            if (e0Cpu != 2 || e1Cpu != 0 || e2Cpu != 4)
                throw new Exception($"CPU pre-check failed: {e0Cpu}, {e1Cpu}, {e2Cpu}");

            // GPU through flattened set. Pack the 9 per-codebook int params
            // into a flat ArrayView<int> to fit ILGPU's 14-arg generic ceiling.
            using var dPacket = acc.Allocate1D<byte>(packet.Length);
            using var dChildren = acc.Allocate1D<int>(flat.AllChildren.Length);
            using var dLeaf = acc.Allocate1D<int>(flat.AllLeafToEntry.Length);
            using var dDecoded = acc.Allocate1D<int>(3);
            using var dParams = acc.Allocate1D<int>(9);
            dPacket.View.CopyFromCPU(packet);
            dChildren.View.CopyFromCPU(flat.AllChildren);
            dLeaf.View.CopyFromCPU(flat.AllLeafToEntry);
            dParams.View.CopyFromCPU(new[]
            {
                flat.ChildrenOffsets[0], flat.LeafOffsets[0], flat.MaxDepths[0],
                flat.ChildrenOffsets[1], flat.LeafOffsets[1], flat.MaxDepths[1],
                flat.ChildrenOffsets[2], flat.LeafOffsets[2], flat.MaxDepths[2],
            });

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
                ArrayView<int>, int>(SetDecodeKernel);
            kernel(new Index1D(1),
                dPacket.View, dChildren.View, dLeaf.View, dDecoded.View,
                dParams.View, packet.Length);
            await acc.SynchronizeAsync();

            var gpuDecoded = await dDecoded.CopyToHostAsync();
            if (gpuDecoded[0] != 2) throw new Exception($"book0 entry: {gpuDecoded[0]}");
            if (gpuDecoded[1] != 0) throw new Exception($"book1 entry: {gpuDecoded[1]}");
            if (gpuDecoded[2] != 4) throw new Exception($"book2 entry: {gpuDecoded[2]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void WriteCanonical(VorbisBitWriter bw, VorbisHuffmanTable t, int entry)
    {
        uint code = t.Codewords[entry];
        int len = t.EntryLengths[entry];
        for (int b = len - 1; b >= 0; b--)
            bw.WriteBit((uint)((code >> b) & 1));
    }

    private static void SetDecodeKernel(
        Index1D _,
        ArrayView<byte> packet, ArrayView<int> children, ArrayView<int> leaf, ArrayView<int> decoded,
        ArrayView<int> bookParams, int packetLen)
    {
        // bookParams layout: 3 ints per codebook -> [childOff, leafOff, depth] x N.
        var state = VorbisBitReaderGpu.Init(packetLen);
        decoded[0] = VorbisHuffmanDecoderGpu.TryDecode(ref state, packet,
            children, bookParams[0], leaf, bookParams[1], bookParams[2]);
        decoded[1] = VorbisHuffmanDecoderGpu.TryDecode(ref state, packet,
            children, bookParams[3], leaf, bookParams[4], bookParams[5]);
        decoded[2] = VorbisHuffmanDecoderGpu.TryDecode(ref state, packet,
            children, bookParams[6], leaf, bookParams[7], bookParams[8]);
    }
}
