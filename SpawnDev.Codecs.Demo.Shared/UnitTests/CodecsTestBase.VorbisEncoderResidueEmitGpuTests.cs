// Cross-backend test for VorbisEncoderResidueEmitGpu.EmitAll.
// Verifies the composite per-bin codebook emission produces byte-
// identical output to a manual CPU loop.

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
    public async Task VorbisEncoderResidueEmitGpu_EmitAll_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Codebook: 8 entries, valid prefix code lengths.
            int[] lengths = { 2, 2, 3, 3, 4, 4, 4, 4 };
            var table = VorbisHuffman.Build(lengths);

            // 32-entry residueQ with values from [0, 8).
            var rng = new Random(unchecked((int)0xA1F0C5DEu));
            int[] residueQ = new int[32];
            for (int i = 0; i < residueQ.Length; i++)
                residueQ[i] = rng.Next(0, 8);

            // CPU reference: emit each entry via VorbisBitWriter.
            var cpuWriter = new VorbisBitWriter();
            foreach (int e in residueQ)
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
            using var dResidueQ = acc.Allocate1D<int>(residueQ.Length);
            using var dOut = acc.Allocate1D<byte>(outBufSize);
            using var dOutLen = acc.Allocate1D<long>(1);
            dCodes.View.CopyFromCPU(table.Codewords);
            dLengths.View.CopyFromCPU(table.EntryLengths);
            dResidueQ.View.CopyFromCPU(residueQ);
            dOut.View.CopyFromCPU(new byte[outBufSize]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<uint>, ArrayView<int>, ArrayView<int>,
                ArrayView<byte>, ArrayView<long>, int>(EmitAllKernel);
            kernel(new Index1D(1),
                dCodes.View, dLengths.View, dResidueQ.View,
                dOut.View, dOutLen.View, residueQ.Length);
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

    private static void EmitAllKernel(
        Index1D _,
        ArrayView<uint> codes, ArrayView<int> lengths, ArrayView<int> residueQ,
        ArrayView<byte> outBuf, ArrayView<long> outLen, int count)
    {
        var state = VorbisBitWriterGpu.Init();
        VorbisEncoderResidueEmitGpu.EmitAll(
            ref state, outBuf, residueQ, 0, count,
            codes, 0, lengths, 0);
        VorbisBitWriterGpu.Finish(ref state, outBuf);
        outLen[0] = state.OutLen;
    }
}
