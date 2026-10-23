// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Paces a host loop that issues many short kernel launches. WebGPU records every dispatch into one
// command encoder until a flush, so hundreds of bounded launches issued before one await still reach
// the GPU as one long command buffer, and a long queue of submitted work has hung Chrome's GPU
// process (device lost). After every launch call LaunchedAsync: every FlushEvery launches it submits
// the recorded work (no wait), every WaitEvery launches it waits for the accelerator to drain.

using ILGPU.Runtime;

namespace SpawnDev.Codecs;

/// <summary>
/// Paces a host loop of many kernel launches: <see cref="Accelerator.Flush"/> every
/// <see cref="FlushEvery"/> launches and <see cref="Accelerator.SynchronizeAsync"/> every
/// <see cref="WaitEvery"/> launches.
/// </summary>
public sealed class LaunchPacer
{
    /// <summary>Launches between submits (Flush, no wait).</summary>
    public const int FlushEvery = 8;
    /// <summary>Launches between waits (SynchronizeAsync).</summary>
    public const int WaitEvery = 32;

    private readonly Accelerator _accelerator;
    private int _launches;

    /// <summary>Pace launches on <paramref name="accelerator"/>.</summary>
    public LaunchPacer(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
    }

    /// <summary>Call after each launch.</summary>
    public ValueTask LaunchedAsync()
    {
        _launches++;
        if (_launches % WaitEvery == 0) return new ValueTask(_accelerator.SynchronizeAsync());
        if (_launches % FlushEvery == 0) _accelerator.Flush();
        return ValueTask.CompletedTask;
    }
}
