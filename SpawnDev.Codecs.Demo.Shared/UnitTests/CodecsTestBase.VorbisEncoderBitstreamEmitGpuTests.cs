// Cross-backend test for VorbisEncoderBitstreamEmitGpu.EmitPacket.
// Verifies the composite full-packet emit produces byte-identical
// output to a manual CPU sequence of header + posteriors + classbook +
// residue codebook writes.

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
    public async Task VorbisEncoderBitstreamEmitGpu_EmitPacket_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Classbook: 4 entries valid prefix (lengths 2,2,2,2).
            int[] classLengths = { 2, 2, 2, 2 };
            var classTable = VorbisHuffman.Build(classLengths);

            // Residue book: 8 entries valid prefix (lengths 2,2,3,3,4,4,4,4).
            int[] residueLengths = { 2, 2, 3, 3, 4, 4, 4, 4 };
            var residueTable = VorbisHuffman.Build(residueLengths);

            // 16 residueQ entries.
            int[] residueQ = { 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0 };
            const int posteriorY0 = 100;
            const int posteriorY1 = 200;
            const int endpointBits = 8;
            const int modeBits = 0;

            // CPU reference.
            var cpuWriter = new VorbisBitWriter();
            cpuWriter.WriteBit(0u);  // packet type
            cpuWriter.WriteBit(1u);  // floor nonzero
            cpuWriter.WriteBits((uint)posteriorY0, endpointBits);
            cpuWriter.WriteBits((uint)posteriorY1, endpointBits);
            // Classbook entry 0:
            uint cc = classTable.Codewords[0];
            int cl = classTable.EntryLengths[0];
            for (int b = cl - 1; b >= 0; b--) cpuWriter.WriteBit((uint)((cc >> b) & 1));
            // Residue book entries:
            foreach (int e in residueQ)
            {
                uint code = residueTable.Codewords[e];
                int len = residueTable.EntryLengths[e];
                for (int b = len - 1; b >= 0; b--) cpuWriter.WriteBit((uint)((code >> b) & 1));
            }
            byte[] cpuBytes = cpuWriter.ToArray();

            // GPU.
            int outBufSize = Math.Max(32, cpuBytes.Length + 4);
            using var dResidueQ = acc.Allocate1D<int>(residueQ.Length);
            using var dClassCodes = acc.Allocate1D<uint>(classTable.Codewords.Length);
            using var dClassLens = acc.Allocate1D<int>(classTable.EntryLengths.Length);
            using var dResCodes = acc.Allocate1D<uint>(residueTable.Codewords.Length);
            using var dResLens = acc.Allocate1D<int>(residueTable.EntryLengths.Length);
            using var dOut = acc.Allocate1D<byte>(outBufSize);
            using var dOutLen = acc.Allocate1D<long>(1);
            dResidueQ.View.CopyFromCPU(residueQ);
            dClassCodes.View.CopyFromCPU(classTable.Codewords);
            dClassLens.View.CopyFromCPU(classTable.EntryLengths);
            dResCodes.View.CopyFromCPU(residueTable.Codewords);
            dResLens.View.CopyFromCPU(residueTable.EntryLengths);
            dOut.View.CopyFromCPU(new byte[outBufSize]);

            var p = new EmitPacketParams
            {
                Count = residueQ.Length,
                PosteriorY0 = posteriorY0,
                PosteriorY1 = posteriorY1,
                EndpointBits = endpointBits,
                ModeBits = modeBits,
            };

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>,
                ArrayView<uint>, ArrayView<int>, ArrayView<uint>, ArrayView<int>,
                EmitPacketParams>(EmitPacketKernel);
            kernel(new Index1D(1),
                dOut.View, dOutLen.View, dResidueQ.View,
                dClassCodes.View, dClassLens.View, dResCodes.View, dResLens.View, p);
            await acc.SynchronizeAsync();

            long gpuLen = (await dOutLen.CopyToHostAsync())[0];
            var gpuFull = await dOut.CopyToHostAsync();
            if (cpuBytes.Length != gpuLen)
                throw new Exception($"len: cpu={cpuBytes.Length} gpu={gpuLen}");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuFull[i])
                    throw new Exception($"byte[{i}]: cpu={cpuBytes[i]:X2} gpu={gpuFull[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    public struct EmitPacketParams
    {
        public int Count;
        public int PosteriorY0;
        public int PosteriorY1;
        public int EndpointBits;
        public int ModeBits;
    }

    private static void EmitPacketKernel(
        Index1D _,
        ArrayView<byte> outBuf, ArrayView<long> outLen, ArrayView<int> residueQ,
        ArrayView<uint> classCodes, ArrayView<int> classLens,
        ArrayView<uint> resCodes, ArrayView<int> resLens,
        EmitPacketParams p)
    {
        VorbisEncoderBitstreamEmitGpu.EmitPacket(
            outBuf, outLen, residueQ, 0, p.Count,
            p.PosteriorY0, p.PosteriorY1,
            p.EndpointBits, p.ModeBits,
            classCodes, 0, classLens, 0,
            resCodes, 0, resLens, 0);
    }
}
