// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU.Runtime;

namespace SpawnDev.Codecs;

/// <summary>
/// Paces a host loop that issues many short kernel launches. Bounded launches alone are not enough in the browser:
/// WebGPU records every dispatch into one command encoder until a flush, so hundreds of short launches before one await
/// reach the GPU as one long command buffer (a long queue of submitted work has hung Chrome's GPU). Call
/// <see cref="LaunchedAsync"/> after every launch: every <see cref="FlushEvery"/> launches it submits the recorded work
/// (no wait), every <see cref="WaitEvery"/> launches it waits for the device.
/// </summary>
public sealed class LaunchPacer
{
    /// <summary>Launches between submits.</summary>
    public const int FlushEvery = 8;
    /// <summary>Launches between waits.</summary>
    public const int WaitEvery = 32;

    private readonly Accelerator _accelerator;
    private int _count;

    /// <summary>Paces launches on <paramref name="accelerator"/>.</summary>
    public LaunchPacer(Accelerator accelerator) => _accelerator = accelerator;

    /// <summary>Call after each launch.</summary>
    public ValueTask LaunchedAsync()
    {
        _count++;
        if (_count % WaitEvery == 0) return new ValueTask(_accelerator.SynchronizeAsync());
        if (_count % FlushEvery == 0) _accelerator.Flush();
        return ValueTask.CompletedTask;
    }
}
