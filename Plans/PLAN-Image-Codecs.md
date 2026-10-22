# PLAN - Image codecs in SpawnDev.Codecs

TJ directive 2026-10-22: "A full set of image codecs (encode/decode) needs to be in [SpawnDev.Codecs] also, not just
video." Consumers: SpawnGen (Lore, ComfyUI parity: PNG with text chunks, JPEG, WebP, later animated), everything that
today leans on a browser canvas or a native library.

## Scope (patent-clean only, same rule as video)

| Format | Decode | Encode | Notes |
|---|---|---|---|
| PNG / APNG | done (default image) | PNG encode (deflate compressor on the accelerator) | tEXt/iTXt/zTXt/eXIf metadata read + write; APNG frames |
| JPEG | baseline + extended sequential + progressive, all samplings, restart markers, Adobe CMYK/YCCK | baseline (+ optional progressive) | EXIF orientation reported, optionally applied |
| WebP | VP8L lossless, VP8 lossy (+ALPH), VP8X container, animation | VP8L lossless, VP8 lossy (reuses `Vp8KeyframeEncoderGpu` bitstream) | EXIF/XMP/ICCP chunks |
| GIF | LZW, interlace, transparency, animation | LZW, palette quantization (on the accelerator), animation | |
| BMP | 1/4/8/16/24/32 bpp, RLE4/RLE8, bitfields | 24/32 bpp | |
| AVIF | needs a spec-complete AV1 intra decoder (CDEF, loop restoration, all intra tools) | via `Av1KeyframeEncoderGpu` | LAST; depends on the AV1 decoder work |
| HEIC/HEIF (HEVC) | **out of scope** | **out of scope** | patent-encumbered, same as H.264/H.265 |

## The rules (non-negotiable; see AGENTS.md and this repo's CLAUDE.md)

1. **Host is a pure coordinator.** Every decoder/encoder stage that touches pixel or entropy data is an ILGPU kernel.
   The host may: allocate buffers, upload the source bytes (necessary I/O), dispatch kernels, read back **scalars and
   small metadata blocks** (<= a few hundred ints), build **tiny metadata blobs** (a PNG tEXt chunk, a JPEG header
   blob of < 1 KB), and read back the final output only into a caller-requested managed/JS sink.
2. **Bulk data never enters the .NET WASM heap in the browser.** JS typed arrays go to the accelerator with
   `IBrowserMemoryBuffer.CopyFromJS` (`DeviceTransfer.Upload(view, TypedArray)`) and come back as a `Uint8Array`
   (`DeviceTransfer.DownloadUint8ArrayAsync`).
3. **Three overloads per entry point**, all funnelling into the `ArrayView1D` one (see "API contract").
4. **Sequential entropy coding runs as ONE GPU thread** (or one per independent stream: restart intervals, tiles,
   frames). Still accelerator-resident. Correct first, then parallelize.
5. **Correct first time, then fast.** Every change has a unit test that can fail (red-check it: break the fix, watch the
   test fail by name, restore by inverse edit). Tests use REAL files with an INDEPENDENT oracle.
6. **No CPU fallback branches, no workarounds.** A library bug is fixed in the library (ILGPU codegen bugs go to Data).

## API contract

Namespace `SpawnDev.Codecs.Images.<Format>`, classes `<Format>DecoderGpu` / `<Format>EncoderGpu` (`IDisposable`,
constructed from an `Accelerator`, **not thread-safe**: one in-flight call per instance). Reference implementation:
`Images/Png/PngDecoderGpu.cs`.

```csharp
// ---- decode ----
Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength);     // header facts, no pixels
Task<ImageInfo> DecodeAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
                            ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8);
Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file);                                  // managed
Task<DecodedImage> DecodeAsync(ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8);
[SupportedOSPlatform("browser")] Task<ImageInfo> ReadInfoAsync(TypedArray file);           // browser
[SupportedOSPlatform("browser")] Task<DecodedImageJS> DecodeAsync(TypedArray file, PixelFormat format = PixelFormat.Rgba8);

// ---- encode ----
static int GetMaxEncodedSize(int width, int height, <Format>EncodeOptions? options = null);
Task<int> EncodeAsync(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
                      ArrayView1D<byte, Stride1D.Dense> output, <Format>EncodeOptions? options = null);   // returns length
Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, <Format>EncodeOptions? options = null);
[SupportedOSPlatform("browser")] Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, <Format>EncodeOptions? options = null);
```

* Pixels are tightly packed `PixelFormat.Rgba8` (straight alpha, rows top to bottom); `Rgba16` where the format has 16-bit.
* Errors: `ImageDecodeException(ImageFormat, reason)` with a SPECIFIC reason; never silently produce garbage.
* Decoders do not apply EXIF orientation unless an option says so; `ImageInfo.Orientation` reports it.
* Animation: `ImageInfo.FrameCount`; frame access API is designed once the first animated format lands (shared by
  APNG / GIF / WebP): until then `DecodeAsync` returns the default / first frame.

## Pipeline conventions (learned the hard way; see PngKernels.cs / InflateGpu.cs)

* Scalars and layout between kernels travel in ONE small `ArrayView<int>` "meta" block with named slot constants
  (`PngMeta`), so a kernel and the host cannot disagree about a slot.
* Host reads the meta block back after ONE `await SynchronizeAsync()` to learn sizes, then sizes the next buffers
  (`StageBuffer<T>` is grow-only, grown only at the start of a public call).
* Never dispose a device buffer a pending dispatch still reads: `await SynchronizeAsync()` first (browser backends).
* Kernel dialect rules that have bitten: no `out` parameters; no `a[i]++` on ArrayView elements (write `a[i] = a[i] + 1`);
  no `static readonly` tables in kernels (compute arithmetically, or upload a constant buffer once); loops that need an early
  exit use a flag in the body, not an early `return` from inside; compound short-circuit predicates in a branch condition
  are built into `bool` locals with `&` / `|` first (WGSL has dropped threads on inline `a && b`); struct state passed
  `ref` into static helpers is fine (`InflateState`, `Vp8BoolDecoderGpuState`); keep kernel parameter counts small by
  packing scalars into the meta block.
* Use `DeviceTransfer` / `StageBuffer` / `PlanePadKernel`; do not write new host copy loops.

## Test conventions

* Fixtures: `SpawnDev.Codecs.Demo.Shared/TestData/<Format>/` (embedded resources, so desktop and browser lanes read the
  same bytes), generated by a checked-in script `scripts/images/make_<format>_fixtures.py`. Truth is derived from the
  source samples or from a reference decoder (libjpeg / libwebp / giflib through Pillow), **never from our own code**.
  Pillow + numpy live in a venv (`python -m venv`, `pip install pillow numpy`); run scripts as `python -I script.py`.
* Tests: `UnitTests/CodecsTestBase.<Format>*Tests.cs`; load every fixture, decode on the accelerator, compare to truth,
  collect ALL failures with the fixture name and the first differing byte before failing. Cover: the managed overload,
  the `ArrayView1D` overload into a caller buffer pre-filled with `0xAB`, error paths (bad signature, truncation,
  corruption) asserting the specific reason, and (browser lanes) the TypedArray overload.
* Run on the CPU backend while iterating: `scripts/dev/build-console.ps1` then `scripts/dev/run-desktop-tests.ps1 -Pattern X`.
  CUDA/OpenCL/WebGPU/WebGL/Wasm lanes (PlaywrightMultiTest) are run by the coordinator, one at a time, when the
  shared GPU is free.

## Animation API (APNG / GIF / WebP share it)

`ImageFrameInfo` (Images/ImageTypes.cs) describes a frame's rectangle, duration, disposal and blend. Every animated format
adds, next to `ReadInfoAsync` / `DecodeAsync` (which return the first / default frame as before):

```csharp
Task<ImageFrameInfo[]> ReadFramesAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength);      // small host metadata
Task<ImageInfo> DecodeFrameAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength, int frameIndex,
                                 ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8);
// + managed (ReadOnlyMemory<byte>) and [SupportedOSPlatform("browser")] TypedArray overloads of both
```

`DecodeFrameAsync` returns the FULL CANVAS (`ImageInfo.Width x Height`) as it looks while frame `frameIndex` is displayed:
frames 0..frameIndex are decoded and composited on the accelerator honouring each frame's `Blend` and the previous frames'
`Dispose` (compositing is a kernel; the canvas never visits the host). A decoder may keep the last canvas to make
sequential access O(1) per frame; random access may re-composite from frame 0.

### GIF (first animated format; APNG / WebP follow the same shape)

* `GifDecoderGpu.ReadFramesAsync` / `DecodeFrameAsync` exactly as above; the loop count has no shared type yet, so
  `ReadLoopCountAsync` (three overloads) returns the NETSCAPE value (0 = forever, -1 = no extension before the first image).
  When APNG / WebP land, `num_plays` / loop count should join `ImageInfo` and `ReadLoopCountAsync` can forward to it.
* `GifDecoderGpu.Compositing` (`GifCompositing.Pillow` default, `.Web`): the decoder is bit-exact with Pillow 13
  (`Image.seek(k)` + `convert("RGBA")`) in `Pillow` mode, including Pillow's own quirks (sticky disposal, palette colour kept
  under alpha 0, background-colour disposal fill, identity-ramp palettes read as gray); `Web` is what browsers show.
  Frames that extend past the logical screen are clipped in both (Pillow grows its canvas, so no Pillow truth exists for them).
* Compositing is planned on the host as a few ints per frame and executed by one kernel (`OpFillRect / OpFillCanvas / OpSave /
  OpRestore / OpDraw`); LZW is one thread per frame, frames in parallel (batches of 64, 32 MB of palette indices).
