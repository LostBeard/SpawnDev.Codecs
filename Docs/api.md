# API shape

Every encoder and decoder exposes each entry point three times. All three funnel into the `ArrayView1D` one, so they
produce identical results (the tests compare them byte for byte).

| Overload | Input | Output | Use it when |
|----------|-------|--------|-------------|
| **`ArrayView1D`** | data already on the accelerator | written into a view **you own**; the call returns only a length or a small info record | the data is on the GPU already (a render target, a model's output, a decoded frame) |
| **`TypedArray`** (browser) | a JS `Uint8Array` / `TypedArray` | a JS `Uint8Array` / `Float32Array` / `DecodedImageJS` you dispose | Blazor WebAssembly: bytes from `fetch`, OPFS, a canvas or WebCodecs |
| **managed** (`ReadOnlyMemory<byte>` / `byte[]`) | managed memory | `byte[]` / `float[]` / `DecodedImage` | desktop, or small inputs |

```csharp
// The same encode, three ways
byte[]     a = await png.EncodeAsync(rgba, w, h);                            // managed
Uint8Array b = await png.EncodeAsync(rgbaJs, w, h);                          // browser
int        n = await png.EncodeAsync(rgbaView, w, h, fileView);              // on the device
```

## Sizing caller-owned buffers

An `ArrayView1D` call writes into your buffer and throws `ArgumentException` naming the needed size when it is too
small. Each class has the sizing helper: `GetMaxEncodedSize(width, height, options)` for encoders, `GetLumaPlaneSize` /
`GetChromaPlaneSize` for video decoders, `MaxSamplesPerPacket` for the Vorbis decoder; image decoders take
`width * height * 4` (or `* 8` for `Rgba16`) from `ReadInfoAsync`. Nothing is written past the returned length (the
tests fill the buffers with a poison byte and check the tail).

## Zero copy in the browser

In Blazor WebAssembly a `byte[]` lives on the .NET heap, and copying a picture through it costs time and memory twice.
The `TypedArray` overloads move bytes with `IBrowserMemoryBuffer.CopyFromJS` (JS to GPU) and read results back as JS
typed arrays, so the pixels never enter the .NET heap. The same applies inside the library: bulk data is never read
back to the host between stages, and only scalars (a length, an error code, a few header fields) cross.

`DeviceTransfer` holds the shared helpers (`Upload` from a span or a `TypedArray`, `DownloadAsync`,
`DownloadUint8ArrayAsync`, `DownloadFloat32ArrayAsync`) for code that builds on the codecs.

## Lifetimes and threading

- The **accelerator belongs to the application.** Codec classes never dispose it.
- A codec instance compiles its kernels once and keeps its working buffers between calls (they only grow). Dispose the
  instance to free them.
- **One call at a time per instance**: the working buffers are shared across calls. Use one instance per concurrent
  stream.
- Every call ends with the result on the device or in your hands: the methods await the accelerator before returning,
  so a buffer you passed may be reused or freed as soon as the task completes.
