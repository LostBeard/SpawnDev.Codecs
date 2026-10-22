// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Grow-only device staging buffer used by the managed and TypedArray overloads of every GPU codec.
// The ArrayView1D overload never needs one: its caller already owns the device buffers.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs;

/// <summary>
/// A device buffer that only ever grows. <see cref="Ensure"/> is called at the START of a public
/// codec call, when every earlier call has already awaited <c>SynchronizeAsync</c>, so replacing the
/// buffer cannot free memory a pending dispatch still reads (which the browser backends turn into
/// reads of freed memory).
/// </summary>
internal sealed class StageBuffer<T> : IDisposable where T : unmanaged
{
    private readonly Accelerator _accelerator;
    private MemoryBuffer1D<T, Stride1D.Dense>? _buffer;

    public StageBuffer(Accelerator accelerator)
    {
        _accelerator = accelerator;
    }

    /// <summary>The whole buffer. Valid after <see cref="Ensure"/>.</summary>
    public ArrayView1D<T, Stride1D.Dense> View =>
        (_buffer ?? throw new InvalidOperationException("Ensure() has not been called.")).View;

    /// <summary>Make the buffer at least <paramref name="length"/> elements; returns the view.</summary>
    public ArrayView1D<T, Stride1D.Dense> Ensure(long length)
    {
        if (length <= 0) length = 1;
        if (_buffer is null || _buffer.Length < length)
        {
            _buffer?.Dispose();
            _buffer = _accelerator.Allocate1D<T>(length);
        }
        return _buffer.View;
    }

    public void Dispose()
    {
        _buffer?.Dispose();
        _buffer = null;
    }
}
