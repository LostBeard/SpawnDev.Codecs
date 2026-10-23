// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Keeps the GPU queue short however many bounded launches a stage needs. Every serial codec stage runs as a chain of
// short launches (no launch may hold the GPU long: the desktop stalls and the OS watchdog resets the driver), but short
// launches alone are not enough in a browser: WebGPU records every dispatch into one command encoder until a flush, so a
// thousand short launches would reach the GPU as ONE long command buffer, and a long queue of submitted work has hung
// Chrome's GPU process (device lost) even when every single launch was short. So a launch loop reports each launch here:
// every FlushEvery launches the batch is submitted (Flush: no wait), every WaitEvery launches the loop waits for the GPU
// to finish (SynchronizeAsync), which bounds the work in flight to WaitEvery short launches. On desktop backends Flush is
// a no-op and the waits cost a round trip per WaitEvery launches.

using ILGPU.Runtime;

namespace SpawnDev.Codecs;

/// <summary>Bounds the GPU work in flight of a launch loop (see the file header).</summary>
public sealed class LaunchPacer
{
    /// <summary>Launches per submitted batch.</summary>
    public const int FlushEvery = 8;
    /// <summary>Launches between waits for completion (each launch is bounded to a few ms: at most ~0.3 s in flight).</summary>
    public const int WaitEvery = 32;

    private readonly Accelerator _accelerator;
    private int _count;

    /// <summary>A pacer for launches on <paramref name="accelerator"/>.</summary>
    public LaunchPacer(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Call after each launch: submits every <see cref="FlushEvery"/> launches, waits every <see cref="WaitEvery"/>.</summary>
    public ValueTask LaunchedAsync()
    {
        _count++;
        if (_count % WaitEvery == 0) return new ValueTask(_accelerator.SynchronizeAsync());
        if (_count % FlushEvery == 0) _accelerator.Flush();
        return default;
    }
}
