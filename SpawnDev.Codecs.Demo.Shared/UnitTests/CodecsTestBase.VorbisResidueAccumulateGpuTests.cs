// Cross-backend test for VorbisResidueAccumulateGpu.LookupAndAccumulate.
// Verifies the GPU primitive matches the CPU pattern of:
//     LookupVector(book, entry, vec); for (d) target[off+d] += vec[d];

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
    public async Task VorbisResidueAccumulateGpu_Type2_AccumulatesMatchCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var book = new VorbisCodebook
            {
                Dimensions = 4,
                Entries = 3,
                Ordered = false,
                Sparse = false,
                Lengths = new int[3],
                LookupType = 2,
                MinValue = -2.0,
                DeltaValue = 0.5,
                ValueBits = 0,
                SequenceP = false,
                Multiplicands = new[] { 0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11 },
            };

            // Pre-fill buffer with arbitrary non-zero values to verify accumulation.
            var initial = new float[10] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };

            // Multiple accumulation calls at different offsets.
            (int entry, int targetOff)[] calls = { (0, 0), (1, 4), (2, 0), (1, 0) };

            // CPU reference: replicate Lookup + accumulate.
            var cpuOut = (float[])initial.Clone();
            foreach (var (entry, off) in calls)
            {
                var vec = new float[book.Dimensions];
                VorbisCodebookVectorLookupCpu(book, entry, vec);
                for (int d = 0; d < book.Dimensions; d++) cpuOut[off + d] += vec[d];
            }

            // GPU.
            using var dMult = acc.Allocate1D<int>(book.Multiplicands.Length);
            using var dTarget = acc.Allocate1D<float>(initial.Length);
            using var dCalls = acc.Allocate1D<int>(calls.Length * 2);
            dMult.View.CopyFromCPU(book.Multiplicands);
            dTarget.View.CopyFromCPU(initial);
            var callsFlat = new int[calls.Length * 2];
            for (int i = 0; i < calls.Length; i++)
            {
                callsFlat[i * 2 + 0] = calls[i].entry;
                callsFlat[i * 2 + 1] = calls[i].targetOff;
            }
            dCalls.View.CopyFromCPU(callsFlat);

            int callCount = calls.Length;
            int multLen = book.Multiplicands.Length;
            int entries = book.Entries;
            int dims = book.Dimensions;

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<float>, ArrayView<int>,
                int, int, int, int, double, double, int, int>(AccumulateKernel);
            kernel(new Index1D(1),
                dMult.View, dTarget.View, dCalls.View,
                multLen, entries, dims, /*lookupType*/2,
                book.MinValue, book.DeltaValue,
                book.SequenceP ? 1 : 0, callCount);
            await acc.SynchronizeAsync();

            var gpuOut = await dTarget.CopyToHostAsync();
            for (int i = 0; i < initial.Length; i++)
                if (cpuOut[i] != gpuOut[i])
                    throw new Exception($"target[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void VorbisCodebookVectorLookupCpu(VorbisCodebook book, int entry, float[] outVec)
    {
        // Inline reference implementation matching the internal LookupVector.
        int dims = book.Dimensions;
        if (book.LookupType == 0 || entry < 0 || entry >= book.Entries)
        {
            for (int d = 0; d < dims; d++) outVec[d] = 0f;
            return;
        }
        double mindel = book.MinValue;
        double delta = book.DeltaValue;
        double last = 0;
        if (book.LookupType == 1)
        {
            int quantvals = book.Multiplicands.Length;
            int divisor = 1;
            for (int d = 0; d < dims; d++)
            {
                int idx = (entry / divisor) % quantvals;
                double m = book.Multiplicands[idx];
                double val = Math.Abs(m) * delta + mindel + last;
                if (book.SequenceP) last = val;
                outVec[d] = (float)val;
                divisor *= quantvals;
            }
            return;
        }
        int baseIdx = entry * dims;
        for (int d = 0; d < dims; d++)
        {
            int flat = baseIdx + d;
            double m = (flat < 0 || flat >= book.Multiplicands.Length) ? 0 : book.Multiplicands[flat];
            double val = Math.Abs(m) * delta + mindel + last;
            if (book.SequenceP) last = val;
            outVec[d] = (float)val;
        }
    }

    private static void AccumulateKernel(
        Index1D _,
        ArrayView<int> multiplicands, ArrayView<float> target, ArrayView<int> calls,
        int multLen, int entries, int dimensions, int lookupType,
        double minValue, double deltaValue, int sequenceP, int callCount)
    {
        // Single-thread iterates through every accumulation call.
        for (int i = 0; i < callCount; i++)
        {
            int entry = calls[i * 2 + 0];
            int targetOff = calls[i * 2 + 1];
            VorbisResidueAccumulateGpu.LookupAndAccumulate(
                multiplicands, 0, multLen,
                entry, entries, dimensions, lookupType,
                /*quantvals*/0, minValue, deltaValue, sequenceP,
                target, targetOff);
        }
    }
}
