// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The ONE place codec entry points move bytes between the outside world and an accelerator
// buffer. Every GPU codec exposes three overloads (managed, JS TypedArray, ArrayView1D) and all
// three funnel into the ArrayView1D one; these helpers are the managed and TypedArray halves of
// that funnel, so a transfer fix lands once.
//
// Why a TypedArray path exists: in Blazor WASM a byte[] lives on the .NET heap. A decoded frame
// or an encoded stream that arrives as a JS Uint8Array (fetch, OPFS, WebCodecs, canvas) must go
// straight to the GPU through IBrowserMemoryBuffer.CopyFromJS and come back as a Uint8Array, never
// through ReadBytes()/byte[]. The managed overloads stay for desktop and for tiny test inputs.

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs;

/// <summary>
/// Byte transfer between managed memory / JS typed arrays and accelerator buffers. Shared by every
/// codec's managed, <see cref="TypedArray"/> and <see cref="ArrayView1D{T, TStride}"/> overloads.
/// </summary>
public static class DeviceTransfer
{
    /// <summary>
    /// Upload managed bytes into the start of <paramref name="dest"/>. Reads the span in place
    /// (no <c>ToArray()</c> staging copy). The upload is consumed before this returns on every
    /// backend, so the caller may reuse or free the source immediately.
    /// </summary>
    public static void Upload(ArrayView1D<byte, Stride1D.Dense> dest, ReadOnlySpan<byte> src)
    {
        if (src.Length == 0) return;
        if (src.Length > dest.Length)
            throw new ArgumentOutOfRangeException(nameof(src), "Source is larger than the destination view.");
        dest.SubView(0, src.Length).CopyFromCPU(ref MemoryMarshal.GetReference(src), src.Length);
    }

    /// <summary>
    /// Upload a JS typed array into the start of <paramref name="dest"/> without the bytes ever
    /// entering the .NET heap (<see cref="IBrowserMemoryBuffer.CopyFromJS(TypedArray, long)"/>).
    /// The source is not disposed.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// <paramref name="dest"/> is not on a browser accelerator. A desktop accelerator has no JS
    /// heap to copy from; a caller holding a TypedArray there has a wiring mistake, and a silent
    /// managed fallback would be the exact copy this path exists to avoid.
    /// </exception>
    [SupportedOSPlatform("browser")]
    public static void Upload(ArrayView1D<byte, Stride1D.Dense> dest, TypedArray src)
    {
        ArgumentNullException.ThrowIfNull(src);
        // The UNDERLYING MemoryBuffer implements IBrowserMemoryBuffer, not the view or the
        // MemoryBuffer1D wrapper (testing those compiles and is always false).
        var browser = RequireBrowserBuffer(dest, out long byteOffset);
        long srcBytes = (long)src.ByteLength;
        if (srcBytes > dest.Length)
            throw new ArgumentOutOfRangeException(nameof(src), "Source is larger than the destination view.");
        browser.CopyFromJS(src, byteOffset);
    }

    /// <summary>
    /// Read <paramref name="length"/> bytes from the start of <paramref name="src"/> into a new
    /// managed array. This is a terminal CPU sink (a file write, a test compare); in a browser
    /// prefer <see cref="DownloadUint8ArrayAsync"/>.
    /// </summary>
    public static Task<byte[]> DownloadAsync(ArrayView1D<byte, Stride1D.Dense> src, long length)
    {
        if (length < 0 || length > src.Length) throw new ArgumentOutOfRangeException(nameof(length));
        if (length == 0) return Task.FromResult(System.Array.Empty<byte>());
        return src.SubView(0, length).CopyToHostAsync();
    }

    /// <summary>
    /// Read <paramref name="length"/> bytes from the start of <paramref name="src"/> into a new JS
    /// <see cref="Uint8Array"/> (the caller disposes it). The bytes do not enter the .NET heap.
    /// </summary>
    /// <exception cref="NotSupportedException">The view is not on a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public static Task<Uint8Array> DownloadUint8ArrayAsync(ArrayView1D<byte, Stride1D.Dense> src, long length)
    {
        if (length < 0 || length > src.Length) throw new ArgumentOutOfRangeException(nameof(length));
        var browser = RequireBrowserBuffer(src, out long byteOffset);
        return browser.CopyToHostUint8ArrayAsync(byteOffset, length);
    }

    /// <summary>
    /// Upload a managed plane into a staging buffer exactly as supplied (stride included); the
    /// codec's pad/crop kernel strips the stride on the accelerator. Not async: it holds a span.
    /// </summary>
    internal static void UploadPlane(
        StageBuffer<byte> stage, ReadOnlyMemory<byte> src, int stride, int w, int h, string paramName)
    {
        long needed = (long)(h - 1) * stride + w;
        if (src.Length < needed)
            throw new ArgumentException(
                $"Plane is {src.Length} bytes; {needed} are needed for {w}x{h} at stride {stride}.", paramName);
        Upload(stage.Ensure(needed), src.Span.Slice(0, (int)needed));
    }

    /// <summary>Upload one tightly packed plane into slot <c>[offset, offset + length)</c> of an already sized staging buffer.</summary>
    internal static void UploadPlaneSlot(
        StageBuffer<byte> stage, long offset, long length, ReadOnlyMemory<byte> src, string paramName)
    {
        if (src.Length < length)
            throw new ArgumentException($"Plane is {src.Length} bytes; {length} are needed.", paramName);
        Upload(stage.View.SubView(offset, length), src.Span.Slice(0, (int)length));
    }

    /// <summary>
    /// Read <paramref name="count"/> floats from the start of <paramref name="src"/> into a new JS
    /// <see cref="Float32Array"/> (the caller disposes it). The samples do not enter the .NET heap.
    /// </summary>
    /// <exception cref="NotSupportedException">The view is not on a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public static async Task<Float32Array> DownloadFloat32ArrayAsync(ArrayView1D<float, Stride1D.Dense> src, long count)
    {
        if (count < 0 || count > src.Length) throw new ArgumentOutOfRangeException(nameof(count));
        var contiguous = (IContiguousArrayView)src.BaseView;
        if (contiguous.Buffer is not IBrowserMemoryBuffer browser)
            throw new NotSupportedException(
                $"A TypedArray overload needs a browser accelerator buffer; got {contiguous.Buffer.GetType().Name} " +
                $"on {contiguous.Buffer.Accelerator.AcceleratorType}. On a desktop accelerator use the managed or ArrayView1D overload.");
        using var bytes = await browser.CopyToHostUint8ArrayAsync(contiguous.IndexInBytes, count * sizeof(float));
        // A float view needs a 4-aligned byte offset: view the bytes in place when they are, copy (in JS) when not.
        if ((bytes.ByteOffset & 3) == 0) return bytes.ReCast<Float32Array>(0, count);
        var result = new Float32Array(count);
        using var resultBytes = new Uint8Array(result.Buffer);
        resultBytes.Set(bytes);
        return result;
    }

    private static IBrowserMemoryBuffer RequireBrowserBuffer(ArrayView1D<byte, Stride1D.Dense> view, out long byteOffset)
    {
        var contiguous = (IContiguousArrayView)view.BaseView;
        if (contiguous.Buffer is not IBrowserMemoryBuffer browser)
            throw new NotSupportedException(
                $"A TypedArray overload needs a browser accelerator buffer; got {contiguous.Buffer.GetType().Name} " +
                $"on {contiguous.Buffer.Accelerator.AcceleratorType}. On a desktop accelerator use the managed or ArrayView1D overload.");
        // byte view: element index == byte offset.
        byteOffset = contiguous.Index;
        return browser;
    }
}
