// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Host-side builder of ONE device buffer from several tables / scratch regions of the same element type. The
// SILK GPU kernels (and the Vorbis floor test) take one buffer per element type plus element offsets (the *At
// fields of their input structs) instead of one view per table: every view is a storage-buffer binding and
// WebGPU allows at most 16 per stage.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs;

/// <summary>
/// Packs tables and reserved (zeroed) regions into one buffer: <see cref="Add(T[])"/> / <see cref="Reserve"/> return
/// each region's element offset, <see cref="Upload"/> allocates the device buffer once.
/// </summary>
public sealed class PackedDeviceBuffer<T> where T : unmanaged
{
    private readonly List<T> _data = new();

    /// <summary>Elements packed so far.</summary>
    public int Length => _data.Count;

    /// <summary>Appends <paramref name="values"/>; returns their offset.</summary>
    public int Add(T[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        int at = _data.Count;
        _data.AddRange(values);
        return at;
    }

    /// <summary>
    /// Appends <paramref name="values"/> in a region of at least <paramref name="minLength"/> elements (zero-padded:
    /// a state buffer the kernel grows into); returns its offset.
    /// </summary>
    public int Add(T[] values, int minLength)
    {
        int at = Add(values);
        for (int i = values.Length; i < minLength; i++) _data.Add(default);
        return at;
    }

    /// <summary>Appends <paramref name="count"/> zeroed elements (a scratch or state region); returns their offset.</summary>
    public int Reserve(int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        int at = _data.Count;
        for (int i = 0; i < count; i++) _data.Add(default);
        return at;
    }

    /// <summary>Allocates the device buffer (at least one element) and uploads the packed contents.</summary>
    public MemoryBuffer1D<T, Stride1D.Dense> Upload(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        var buffer = accelerator.Allocate1D<T>(Math.Max(1, _data.Count));
        if (_data.Count > 0) buffer.View.SubView(0, _data.Count).CopyFromCPU(_data.ToArray());
        return buffer;
    }
}
