// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Where the working threads of a "serial stream" kernel sit inside a launch.
//
// A serial stream kernel has few threads, each doing long, data-dependent work (one range coder per tile, one
// bitstream per frame). An auto-grouped launch packs threads 0..N-1 into the first launch group(s), which is the
// wrong place for such threads on a GPU:
//   - threads of one warp execute in lock-step, so 32 threads taking different branches share one warp's time;
//   - a launch group runs on ONE multiprocessor, so every warp of the group shares that multiprocessor while the
//     others idle.
// Measured, AV1 keyframe tile walker, 832x480 = 104 tile threads, RTX 4070 (46 multiprocessors), 2026-11-13:
//     CUDA   (group 640): packed 20.7 ms per frame, placed 8.8 ms (one thread alone is 8.8 ms)
//     OpenCL (group 256): packed 15.8 ms,           placed 5.1 ms
//     CUDA, 16 frames per launch (1664 threads): packed 4.8 ms per frame, placed 1.1 ms; 64 frames: 1.32 -> 0.83 ms
//     WebGPU (Chrome D3D12, group 64): packed 40.6 - 41.3 ms per frame, 2 threads per group 26.5 ms;
//                                      8 frames per launch: packed 5.1 - 5.2 ms per frame, 14 - 16 per group 3.9 ms
// (a packed CUDA launch of 832 and of 1664 threads took the same 89 ms: one group, then two groups side by side).
//
// The placement spreads the N working threads: one launch group per multiprocessor, inside a group over as many
// warps as it has, and only then several lanes per warp. The launch covers Extent threads; a thread asks
// WorkingThread for its working-thread number and returns at once when it has none. With enough threads to fill
// every group the placement is the packed one.
//
// Applied on CUDA and on OpenCL NVIDIA devices with the device's own multiprocessor count and warp size, and on
// WebGPU, which reports neither, with WebGpuAssumedMultiprocessors groups and two halves per group. Measured on the
// RTX 4070 only. On a device where the spread buys nothing its cost is the idle threads of the launch (each returns
// at its first branch), and with enough working threads to fill the groups the placement is the packed one.
//
// The desktop CPU accelerators have the same problem in another form and get one working thread per group
// (OnePerGroup: the measurements are there). Packed everywhere else: Wasm and the browser's CPU accelerator, WebGL
// (no groups), and the OpenCL devices of other vendors (not measured yet).

using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;

namespace SpawnDev.Codecs;

/// <summary>
/// Placement of the working threads of a serial stream kernel inside a launch (one launch group per multiprocessor,
/// spread over warps, then lanes). A value type passed to the kernel; see the file header for the measurements.
/// </summary>
public struct SerialThreadPlacement
{
    /// <summary>Working threads of the launch.</summary>
    public int ThreadCount;
    /// <summary>Threads per launch group.</summary>
    public int GroupSize;
    /// <summary>Threads per warp.</summary>
    public int WarpSize;
    /// <summary>Warps of a group that hold working threads (its first ones).</summary>
    public int WarpsPerGroup;
    /// <summary>Working threads per used warp (its first lanes).</summary>
    public int LanesPerWarp;

    /// <summary>The launch extent: whole groups, enough of them to hold every working thread.</summary>
    public readonly long Extent
    {
        get
        {
            long perGroup = (long)WarpsPerGroup * LanesPerWarp;
            return (ThreadCount + perGroup - 1) / perGroup * GroupSize;
        }
    }

    /// <summary>
    /// WebGPU reports no multiprocessor count: the placement spreads the working threads over this many groups
    /// before a group gets a second thread (a GPU with fewer multiprocessors runs several small groups on each).
    /// </summary>
    public const int WebGpuAssumedMultiprocessors = 64;

    /// <summary>The packed placement: launch thread i is working thread i.</summary>
    public static SerialThreadPlacement Packed(int threadCount)
    {
        if (threadCount < 1) throw new ArgumentOutOfRangeException(nameof(threadCount));
        return new SerialThreadPlacement { ThreadCount = threadCount, GroupSize = 1, WarpSize = 1, WarpsPerGroup = 1, LanesPerWarp = 1 };
    }

    /// <summary>
    /// The placement of <paramref name="threadCount"/> working threads for a kernel the accelerator launches in
    /// groups of <paramref name="groupSize"/> threads (the kernel's <c>KernelInfo.MinGroupSize</c>).
    /// </summary>
    public static SerialThreadPlacement Create(Accelerator accelerator, int groupSize, int threadCount)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (accelerator.AcceleratorType == AcceleratorType.WebGPU)
            return Create(threadCount, groupSize, groupSize / 2, WebGpuAssumedMultiprocessors);
        // The desktop CPU accelerators (in a browser the CPU accelerator has one thread: nothing to spread).
        if (accelerator.AcceleratorType == AcceleratorType.Velocity)
            return OnePerGroup(threadCount, groupSize, 0);
        if (accelerator.AcceleratorType == AcceleratorType.CPU && !OperatingSystem.IsBrowser())
            return OnePerGroup(threadCount, groupSize, CPUAccelerator.LaneLoopInlineMaxWorkItems);
        bool spread = accelerator.AcceleratorType == AcceleratorType.Cuda
            || (accelerator.AcceleratorType == AcceleratorType.OpenCL && accelerator.Device is CLDevice cl && cl.Vendor == CLDeviceVendor.Nvidia);
        return spread
            ? Create(threadCount, groupSize, accelerator.WarpSize, accelerator.NumMultiprocessors)
            : Packed(threadCount);
    }

    /// <summary>
    /// One working thread per launch group (the group's first thread), for the CPU accelerators, in a launch of
    /// more than <paramref name="minExtentExclusive"/> threads: the groups are whole multiples of
    /// <paramref name="groupSize"/>, as small as that allows. Packed for a single working thread.
    /// </summary>
    /// <remarks>
    /// Velocity runs a group's threads as the lanes of one vector, in lock-step: four tiles in a group took longer
    /// than the four one after another. ILGPU's CPU accelerator runs a group on one core and a launch of at most
    /// <c>CPUAccelerator.LaneLoopInlineMaxWorkItems</c> threads inline on the calling thread, so 104 packed tile
    /// threads ran on one core. Measured, 832x480, Ryzen 5 7500F (6 cores), ms per keyframe, packed -> one per group:
    /// CPU accelerator 66.6 -> 20.3, Velocity 134 -> 23.3 (one tile: 68.7 / 68.9).
    /// </remarks>
    public static SerialThreadPlacement OnePerGroup(int threadCount, int groupSize, int minExtentExclusive)
    {
        if (threadCount < 1) throw new ArgumentOutOfRangeException(nameof(threadCount));
        if (threadCount == 1 || groupSize < 1) return Packed(threadCount);
        // the smallest multiple of groupSize with threadCount * stride > minExtentExclusive
        long groups = Math.Max(1, ((long)minExtentExclusive / threadCount + groupSize) / groupSize);
        long stride = groups * groupSize;
        if (stride == 1) return Packed(threadCount);
        if (stride * threadCount > int.MaxValue) return Packed(threadCount);
        return new SerialThreadPlacement
        {
            ThreadCount = threadCount, GroupSize = (int)stride, WarpSize = (int)stride, WarpsPerGroup = 1, LanesPerWarp = 1,
        };
    }

    /// <summary>
    /// The placement from the device's numbers: one group per multiprocessor, the group's threads spread over its
    /// warps, then over lanes. Packed when the numbers do not describe whole warps per group.
    /// </summary>
    public static SerialThreadPlacement Create(int threadCount, int groupSize, int warpSize, int multiprocessors)
    {
        if (threadCount < 1) throw new ArgumentOutOfRangeException(nameof(threadCount));
        if (warpSize < 2 || multiprocessors < 1 || groupSize < warpSize || groupSize % warpSize != 0) return Packed(threadCount);
        int warps = groupSize / warpSize;
        long perGroup = ((long)threadCount + multiprocessors - 1) / multiprocessors;
        if (perGroup >= groupSize) return Packed(threadCount);   // every group full: nothing to spread
        int lanes = (int)((perGroup + warps - 1) / warps);
        int usedWarps = (int)((perGroup + lanes - 1) / lanes);
        return new SerialThreadPlacement
        {
            ThreadCount = threadCount, GroupSize = groupSize, WarpSize = warpSize, WarpsPerGroup = usedWarps, LanesPerWarp = lanes,
        };
    }

    /// <summary>
    /// In a kernel: the working-thread number (0 .. ThreadCount - 1) of launch thread <paramref name="launchIndex"/>,
    /// or -1 when that thread has no work.
    /// </summary>
    public static int WorkingThread(SerialThreadPlacement p, int launchIndex)
    {
        int group = launchIndex / p.GroupSize;
        int local = launchIndex - group * p.GroupSize;
        int warp = local / p.WarpSize;
        int lane = local - warp * p.WarpSize;
        int thread = (group * p.WarpsPerGroup + warp) * p.LanesPerWarp + lane;
        bool works = (lane < p.LanesPerWarp) & (warp < p.WarpsPerGroup) & (thread < p.ThreadCount);
        return works ? thread : -1;
    }
}
