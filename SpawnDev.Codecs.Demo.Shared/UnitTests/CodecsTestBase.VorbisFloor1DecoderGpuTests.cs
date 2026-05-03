// Cross-backend test for VorbisFloor1DecoderGpu. Verifies the GPU
// per-channel posterior decoder matches the CPU VorbisFloor1Decoder
// reference for a small synthetic Floor 1 config.

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
    public async Task VorbisFloor1DecoderGpu_Simple1Class_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Floor 1 config: 1 class, 4 dimensions, 1 codebook (no master).
            // The 4 posterior values come from 4 codebook entries.
            // Total Y values = 2 endpoints + 4 partition values = 6.
            var config = new VorbisFloor1Config
            {
                Partitions = 1,
                PartitionClassList = new[] { 0 },
                ClassDimensions = new[] { 4 },
                ClassSubclasses = new[] { 0 },
                ClassMasterbooks = new[] { -1 },
                ClassSubclassBooks = new[] { new[] { 0 } },
                Multiplier = 1,
                RangeBits = 8,
                XList = new[] { 0, 256, 50, 100, 150, 200 },
            };

            // Simple 4-entry codebook with valid prefix code (lengths 2,2,2,2).
            var codebook = new VorbisCodebook
            {
                Dimensions = 1,
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

            // Build packet bytes:
            //   1 bit nonzero=1
            //   8 bits y[0]=50
            //   8 bits y[1]=200
            //   per partition: cbits=0 -> no master decode; for each cdim=4:
            //     - book 0, decode an entry. cval=0 every iteration so always entry 0.
            //
            // Hold on - with cval=0 and csub=(1<<0)-1=0, we always pick subclass[0]=0
            // (the single codebook). Every partition Y comes from decoding codebook 0.
            // For variety, set y values via different entries.
            int[] partitionEntries = { 1, 2, 3, 0 };

            var bw = new VorbisBitWriter();
            bw.WriteBit(1u);              // nonzero = 1
            bw.WriteBits(50, 8);          // y[0]
            bw.WriteBits(200, 8);         // y[1]

            var t = VorbisHuffman.Build(codebook.Lengths);
            foreach (int e in partitionEntries)
            {
                uint code = t.Codewords[e];
                int len = t.EntryLengths[e];
                for (int b = len - 1; b >= 0; b--)
                    bw.WriteBit((uint)((code >> b) & 1));
            }
            byte[] packet = bw.ToArray();

            // CPU reference.
            var decoder = new VorbisHuffmanDecoder(t);
            var refReader = new VorbisBitReader(packet);
            var cpuY = VorbisFloor1Decoder.Decode(ref refReader, config, new[] { decoder })
                ?? throw new Exception("CPU decoder returned null");

            // GPU.
            var flatSet = VorbisHuffmanCodebookSetGpu.Build(new[] { codebook });

            // Flatten ClassSubclassBooks for the GPU (1 class, 1 slot).
            int classCount = config.ClassDimensions.Length;
            var subBooksOffsets = new int[classCount + 1];
            int totalSlots = 0;
            for (int c = 0; c < classCount; c++)
            {
                subBooksOffsets[c] = totalSlots;
                totalSlots += 1 << config.ClassSubclasses[c];
            }
            subBooksOffsets[classCount] = totalSlots;
            var subBooksFlat = new int[totalSlots];
            for (int c = 0; c < classCount; c++)
                for (int s = 0; s < (1 << config.ClassSubclasses[c]); s++)
                    subBooksFlat[subBooksOffsets[c] + s] = config.ClassSubclassBooks[c][s];

            // Codebook params: 3 ints per book [childOff, leafOff, depth].
            var codebookParams = new int[3];
            codebookParams[0] = flatSet.ChildrenOffsets[0];
            codebookParams[1] = flatSet.LeafOffsets[0];
            codebookParams[2] = flatSet.MaxDepths[0];

            using var dPacket = acc.Allocate1D<byte>(packet.Length);
            using var dPartCls = acc.Allocate1D<int>(config.PartitionClassList.Length);
            using var dDims = acc.Allocate1D<int>(config.ClassDimensions.Length);
            using var dSubcls = acc.Allocate1D<int>(config.ClassSubclasses.Length);
            using var dMaster = acc.Allocate1D<int>(config.ClassMasterbooks.Length);
            using var dSubBooks = acc.Allocate1D<int>(subBooksFlat.Length);
            using var dSubBooksOff = acc.Allocate1D<int>(subBooksOffsets.Length);
            using var dHuffChildren = acc.Allocate1D<int>(flatSet.AllChildren.Length);
            using var dHuffLeaf = acc.Allocate1D<int>(flatSet.AllLeafToEntry.Length);
            using var dCbParams = acc.Allocate1D<int>(codebookParams.Length);
            using var dY = acc.Allocate1D<int>(config.XList.Length);
            using var dYLen = acc.Allocate1D<int>(1);

            dPacket.View.CopyFromCPU(packet);
            dPartCls.View.CopyFromCPU(config.PartitionClassList);
            dDims.View.CopyFromCPU(config.ClassDimensions);
            dSubcls.View.CopyFromCPU(config.ClassSubclasses);
            dMaster.View.CopyFromCPU(config.ClassMasterbooks);
            dSubBooks.View.CopyFromCPU(subBooksFlat);
            dSubBooksOff.View.CopyFromCPU(subBooksOffsets);
            dHuffChildren.View.CopyFromCPU(flatSet.AllChildren);
            dHuffLeaf.View.CopyFromCPU(flatSet.AllLeafToEntry);
            dCbParams.View.CopyFromCPU(codebookParams);
            dY.View.CopyFromCPU(new int[config.XList.Length]);

            // Pack scalars + offsets into a struct to fit ILGPU's 14-arg limit.
            var p = new Floor1DecoderTestParams
            {
                PacketLen = packet.Length,
                Partitions = config.Partitions,
                Multiplier = config.Multiplier,
                XListLen = config.XList.Length,
            };

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>,
                ArrayView<int>, ArrayView<int>, ArrayView<int>,
                ArrayView<int>, ArrayView<int>,
                Floor1DecoderTestParams>(Floor1DecodeKernel);
            kernel(new Index1D(1),
                dPacket.View, dPartCls.View, dDims.View, dSubcls.View,
                dMaster.View, dSubBooks.View, dSubBooksOff.View,
                dHuffChildren.View, dHuffLeaf.View, dCbParams.View,
                dY.View, dYLen.View, p);
            await acc.SynchronizeAsync();

            int gpuYLen = (await dYLen.CopyToHostAsync())[0];
            if (gpuYLen != cpuY.Length)
                throw new Exception($"yLen: cpu={cpuY.Length} gpu={gpuYLen}");
            var gpuY = await dY.CopyToHostAsync();
            for (int i = 0; i < cpuY.Length; i++)
                if (cpuY[i] != gpuY[i])
                    throw new Exception($"Y[{i}]: cpu={cpuY[i]} gpu={gpuY[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    public struct Floor1DecoderTestParams
    {
        public int PacketLen;
        public int Partitions;
        public int Multiplier;
        public int XListLen;
    }

    private static void Floor1DecodeKernel(
        Index1D _,
        ArrayView<byte> packet,
        ArrayView<int> partCls, ArrayView<int> dims, ArrayView<int> subcls,
        ArrayView<int> master, ArrayView<int> subBooks, ArrayView<int> subBooksOff,
        ArrayView<int> huffChildren, ArrayView<int> huffLeaf, ArrayView<int> cbParams,
        ArrayView<int> yOut, ArrayView<int> yLenOut,
        Floor1DecoderTestParams p)
    {
        var state = VorbisBitReaderGpu.Init(p.PacketLen);
        // Pre-zero yOut so silent-floor cases stay zero.
        for (int i = 0; i < p.XListLen; i++) yOut[i] = 0;
        int yLen = VorbisFloor1DecoderGpu.Decode(
            ref state, packet,
            p.Partitions, p.Multiplier, p.XListLen,
            partCls, 0, dims, 0, subcls, 0, master, 0,
            subBooks, 0, subBooksOff, 0,
            huffChildren, huffLeaf, cbParams, 0,
            yOut, 0);
        yLenOut[0] = yLen;
    }
}
