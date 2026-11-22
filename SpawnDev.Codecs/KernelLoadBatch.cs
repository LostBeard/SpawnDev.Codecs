// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Releases the ILGPU compiler cache once per codec kernel load. The ILGPU Context keeps the IR and type information of
// every method it has compiled for its whole life, about 0.9 MiB per large kernel; in a browser the Context usually lives
// as long as the page, so a page that loads several codecs runs the .NET heap out (lossy WebP alone: +20.1 MiB Wasm,
// +28.4 MiB WebGPU, of which the release frees 16.1 / 19.6 MiB; SpawnDev.ILGPU 5.4.44 measurements, 2026-11-20).
// accelerator.ReleaseCompilerCaches() frees it, but the next kernel load re-imports what it shares with earlier ones
// (about 5 to 15 ms), so it belongs after a batch of loads, not after each one.
//
// Codecs nest their loaders (PngDecoderGpu loads PngKernels, then InflateKernel; Vp9KeyframeEncoderGpu's constructor
// builds nine kernel classes), so "one release per loader" would release several times per codec and pay the re-import
// each time. Every loader runs inside RunAsync / Run / Scope instead: a depth counter per Context makes only the
// OUTERMOST batch release, after its nested loads have finished. Loads of different codecs that overlap (two warm-ups
// started together, in the browser's one thread or on desktop threads) count as one batch: nothing is released while any
// Codecs load on that Context is in flight.
//
// A batch releases only if it loaded a kernel (Accelerator.LoadedKernelCount moved: a backend compile, the only thing
// that grows the Context's cache; a hit in the accelerator's own kernel cache does not count). So a warm-up called again,
// or an entry point whose kernels are already loaded, releases nothing and costs nothing.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs;

/// <summary>Runs a codec's kernel loads as one batch and releases the ILGPU compiler cache after the outermost one (see the file header).</summary>
internal static class KernelLoadBatch
{
    internal sealed class Batch
    {
        public int Depth;
        public bool Loaded;
        public int Releases;
        public double ReleaseMs;
    }

    /// <summary>The time the releases of <paramref name="context"/> took in all (the tests' probe).</summary>
    internal static double ReleaseMilliseconds(Context context)
    {
        if (!Batches.TryGetValue(context, out var batch)) return 0;
        lock (batch) return batch.ReleaseMs;
    }

    private static readonly ConditionalWeakTable<Context, Batch> Batches = new();

    /// <summary>
    /// Tests only: false keeps the batches but skips the release, so one test can measure what the release saves (heap)
    /// and costs (load time) on the same codec. Always true outside that test.
    /// </summary>
    internal static bool ReleaseEnabled = true;

    /// <summary>How many times the batches of <paramref name="context"/> have released its compiler cache (the tests' probe).</summary>
    internal static int ReleaseCount(Context context)
    {
        if (!Batches.TryGetValue(context, out var batch)) return 0;
        lock (batch) return batch.Releases;
    }

    /// <summary>Awaits <paramref name="load"/> as part of a batch; the outermost batch releases the compiler cache when it ends (also on failure).</summary>
    public static async Task RunAsync(Accelerator accelerator, Func<Task> load)
    {
        using var scope = Scope(accelerator);
        await load().ConfigureAwait(true);
    }

    /// <summary>Awaits <paramref name="load"/> (one lazily loaded kernel, <c>_k ??= await KernelLoadBatch.RunAsync(a, () => a.Load...Async(...))</c>) as part of a batch.</summary>
    public static async Task<T> RunAsync<T>(Accelerator accelerator, Func<Task<T>> load)
    {
        using var scope = Scope(accelerator);
        return await load().ConfigureAwait(true);
    }

    /// <summary>Runs the synchronous <paramref name="load"/> as part of a batch (the synchronous entry points' loads).</summary>
    public static T Run<T>(Accelerator accelerator, Func<T> load)
    {
        using var scope = Scope(accelerator);
        return load();
    }

    /// <summary>
    /// A batch for loads that cannot sit in a lambda, such as a constructor assigning readonly kernel fields:
    /// <c>using var kernelLoads = KernelLoadBatch.Scope(accelerator);</c> as the constructor's first statement. Disposing it
    /// ends the batch (the outermost one releases the compiler cache if the batch loaded a kernel).
    /// </summary>
    public static LoadScope Scope(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        var batch = Batches.GetValue(accelerator.Context, static _ => new Batch());
        lock (batch) batch.Depth++;
        return new LoadScope(accelerator, batch, accelerator.LoadedKernelCount);
    }

    /// <summary>The batch opened by <see cref="Scope"/>; dispose it once, when the loads are done.</summary>
    public readonly struct LoadScope : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly Batch _batch;
        private readonly long _loadsAtStart;

        internal LoadScope(Accelerator accelerator, Batch batch, long loadsAtStart)
        {
            _accelerator = accelerator;
            _batch = batch;
            _loadsAtStart = loadsAtStart;
        }

        /// <summary>Ends the batch.</summary>
        public void Dispose()
        {
            var batch = _batch;
            if (batch is null) return;   // default(LoadScope)
            lock (batch)
            {
                if (_accelerator.LoadedKernelCount != _loadsAtStart) batch.Loaded = true;
                if (--batch.Depth != 0 || !batch.Loaded) return;
                batch.Loaded = false;
                if (!ReleaseEnabled) return;
                // A codec disposed while its load was in flight: its Context may be gone with it, and a disposed Context
                // has nothing left to release.
                if (_accelerator.IsDisposed || _accelerator.Context.IsDisposed) return;
                // Under the batch's lock: a Codecs load entering now waits for the release to finish.
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                _accelerator.ReleaseCompilerCaches();
                batch.ReleaseMs += System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                batch.Releases++;
            }
        }
    }
}
