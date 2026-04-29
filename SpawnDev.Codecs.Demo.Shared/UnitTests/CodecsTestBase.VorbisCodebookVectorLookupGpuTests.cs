// Cross-backend tests for VorbisCodebookVectorLookupGpu. Verifies the
// GPU vector lookup matches the CPU VorbisResidueDecoder.LookupVector
// reference for all three Vorbis codebook lookup types (0, 1, 2) and
// for both sequenceP modes (sequential / non-sequential reconstruction).

using System.Reflection;
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
    public async Task VorbisCodebookVectorLookupGpu_Type0_AllZeros_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var book = new VorbisCodebook
            {
                Dimensions = 4,
                Entries = 8,
                Ordered = false,
                Sparse = false,
                Lengths = new int[8],
                LookupType = 0,
                MinValue = 0,
                DeltaValue = 0,
                ValueBits = 0,
                SequenceP = false,
                Multiplicands = Array.Empty<int>(),
            };
            await VerifyAllEntries(acc, book);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisCodebookVectorLookupGpu_Type2_FlatNoSeq_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Type 2: flat table indexed by entry*dims + dim.
            var book = new VorbisCodebook
            {
                Dimensions = 3,
                Entries = 5,
                Ordered = false,
                Sparse = false,
                Lengths = new int[5],
                LookupType = 2,
                MinValue = -1.0,
                DeltaValue = 0.25,
                ValueBits = 4,
                SequenceP = false,
                Multiplicands = new[] { 0, 1, 2,  3, 4, 5,  6, 7, 8,  9, 10, 11,  12, 13, 14 },
            };
            await VerifyAllEntries(acc, book);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisCodebookVectorLookupGpu_Type1_QuantBaseSeqP_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Type 1 with sequenceP=true exercises the sequential reconstruction
            // (each dim accumulates onto the prior dim's value).
            var book = new VorbisCodebook
            {
                Dimensions = 4,
                Entries = 16,           // 16 = 2^4 -> quantvals = 2 per dim
                Ordered = false,
                Sparse = false,
                Lengths = new int[16],
                LookupType = 1,
                MinValue = -0.5,
                DeltaValue = 0.5,
                ValueBits = 1,
                SequenceP = true,
                Multiplicands = new[] { 0, 3 },
            };
            await VerifyAllEntries(acc, book);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task VerifyAllEntries(Accelerator acc, VorbisCodebook book)
    {
        // Use reflection to call the internal LookupVector.
        var lookupMethod = typeof(VorbisResidueDecoder).Assembly
            .GetType("SpawnDev.Codecs.Audio.Vorbis.VorbisResidueDecoder")!
            .GetNestedType("VorbisCodebookVector", BindingFlags.NonPublic)
            ?.GetMethod("LookupVector", BindingFlags.NonPublic | BindingFlags.Static);
        if (lookupMethod is null)
        {
            // The internal helper is in a separate type within the same file;
            // try the file-level assembly directly.
            lookupMethod = typeof(VorbisCodebook).Assembly
                .GetTypes()
                .FirstOrDefault(t => t.Name == "VorbisCodebookVector")
                ?.GetMethod("LookupVector", BindingFlags.NonPublic | BindingFlags.Static);
        }
        if (lookupMethod is null)
            throw new Exception("Could not find VorbisCodebookVector.LookupVector via reflection.");

        // Compute lookup1_values(entries, dims) for type 1 (matches CPU helper).
        int quantvals = book.LookupType == 1 ? Lookup1Values(book.Entries, book.Dimensions) : 0;

        // Upload multiplicands once.
        using var dMult = acc.Allocate1D<int>(Math.Max(1, book.Multiplicands.Length));
        if (book.Multiplicands.Length > 0)
            dMult.View.CopyFromCPU(book.Multiplicands);

        for (int entry = 0; entry < book.Entries; entry++)
        {
            // CPU reference.
            var cpuVec = new float[book.Dimensions];
            var args = new object[] { book, entry, (Memory<float>)cpuVec };
            // LookupVector takes Span<float> - call via Span proxy.
            CallLookupVectorWithSpan(lookupMethod, book, entry, cpuVec);

            // GPU.
            using var dOut = acc.Allocate1D<float>(book.Dimensions);
            dOut.View.CopyFromCPU(new float[book.Dimensions]);
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<float>,
                int, int, int, int, int, int, double, double, int>(LookupVectorKernel);
            kernel(new Index1D(1),
                dMult.View, dOut.View,
                book.Multiplicands.Length, entry, book.Entries, book.Dimensions,
                book.LookupType, quantvals,
                book.MinValue, book.DeltaValue, book.SequenceP ? 1 : 0);
            await acc.SynchronizeAsync();

            var gpuVec = await dOut.CopyToHostAsync();
            for (int d = 0; d < book.Dimensions; d++)
            {
                if (cpuVec[d] != gpuVec[d])
                    throw new Exception(
                        $"Codebook lookup mismatch entry={entry} dim={d}: cpu={cpuVec[d]} gpu={gpuVec[d]}");
            }
        }
    }

    private static void CallLookupVectorWithSpan(
        MethodInfo method, VorbisCodebook book, int entry, float[] outBuf)
    {
        // Wrapper to call internal LookupVector(VorbisCodebook, int, Span<float>).
        // Spans can't be boxed; use a delegate via reflection's MakeGenericMethod-like path.
        // The internal API is `LookupVector(book, entry, outVec)` with `Span<float>` outVec.
        // Workaround: copy outBuf to a fresh array via a `float[]` overload if available.
        // Simpler: call via dynamic invoke with Span<float> wrapped in Memory<float>.Span.
        // .NET doesn't allow Span<T> in object[] - work around via local delegate.
        var del = (LookupDelegate)method.CreateDelegate(typeof(LookupDelegate));
        del(book, entry, outBuf);
    }

    private delegate void LookupDelegate(VorbisCodebook book, int entry, Span<float> outVec);

    private static void LookupVectorKernel(
        Index1D _,
        ArrayView<int> multiplicands, ArrayView<float> outVec,
        int multLen, int entry, int entries, int dimensions, int lookupType,
        int quantvals, double minValue, double deltaValue, int sequenceP)
    {
        VorbisCodebookVectorLookupGpu.LookupVector(
            multiplicands, 0, multLen,
            entry, entries, dimensions, lookupType,
            quantvals, minValue, deltaValue, sequenceP,
            outVec, 0);
    }

    private static int Lookup1Values(int entries, int dimensions)
    {
        // Smallest q such that q^dimensions >= entries.
        int q = 1;
        while (true)
        {
            long pow = 1;
            for (int i = 0; i < dimensions; i++) pow *= q;
            if (pow >= entries) return q;
            q++;
        }
    }
}
