<div align="center">

# SpawnDev.Codecs

**Image, video and audio codecs in C#, running on any GPU, in the browser or on the desktop.**

Every encoder and decoder is a chain of C# kernels that [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) compiles to WGSL, GLSL, WebAssembly, PTX, OpenCL C or SIMD CPU code.
No native binaries. No server. Patent-clean formats only.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)
[![.NET 10](https://img.shields.io/badge/.NET-10-512bd4.svg)](https://dotnet.microsoft.com/)
[![Sponsor](https://img.shields.io/badge/sponsor-LostBeard-ea4aaa.svg?logo=githubsponsors)](https://github.com/sponsors/LostBeard)

[Docs](Docs/README.md) | [What is verified](Docs/status.md) | [Changelog](CHANGELOG.md) | [Plans](Plans/)

</div>

---

## What it is

SpawnDev.Codecs is a codec library for .NET where the codec work itself, from colour conversion to the entropy coder,
runs on the accelerator you hand it. The host is a pure coordinator: it uploads the input, launches kernels in order
and reads back the finished file or picture. The same library and the same kernels run:

- **in the browser** (Blazor WebAssembly) on WebGPU, WebGL or WebAssembly, with nothing leaving the user's machine, and
- **on the desktop** (Windows, Linux, macOS) on CUDA, OpenCL or the CPU.

## Highlights

- **Image codecs, checked against the reference libraries.** PNG (with APNG frames, Adam7, 16-bit, text chunks), JPEG
  (baseline and progressive), WebP (lossless, lossy, alpha, animation), GIF and BMP, encode and decode, plus one
  `ImageDecoderGpu` that recognizes the format.
- **Byte-identical with the reference encoders where the format allows it:** WebP lossy encode matches libwebp 1.6.0
  (methods 3 and 4, 222 opaque and 145 transparent test pictures) and JPEG encode matches libjpeg-turbo file for file;
  WebP and JPEG decode match libwebp and libjpeg-turbo pixel for pixel.
- **Video:** VP8, VP9 and AV1 keyframe encoders and decoders, including a full-syntax VP8 keyframe decoder that matches
  ffmpeg pixel for pixel on 182 real files.
- **Audio:** an Opus decoder (SILK, CELT, hybrid, loss concealment) and a CELT-only Opus encoder that are byte-identical
  with libopus 1.5.2 on every packet of real streams, Vorbis decoders that match libvorbis, and FLAC.
- **One API shape everywhere.** Every entry point has three overloads: data already on the accelerator (`ArrayView1D`),
  a JS typed array in the browser (`TypedArray`, never touching the .NET heap) and managed memory (`byte[]`).
- **Bounded launches.** Serial stages (bit writers, entropy coders) are resumable chains of short launches, so no kernel
  holds the GPU long enough to stall the desktop or trip the OS watchdog.

## Status

Every test lane is green (2026-10-29): CPU 3281 / 0, CUDA, OpenCL and Velocity 1131 / 0 each on the desktop; WebGPU and
Wasm 1150 / 0 each and WebGL 399 / 0 in Chrome. WebGL has no scatter stores, so the serial frame coders (the video frame
coders, the Opus coders) are not offered there; their building blocks are. [**Docs/status.md**](Docs/status.md) lists
every codec by backend with the evidence; where it disagrees with this page, that file wins.

## Install

```bash
dotnet add package SpawnDev.Codecs
```

Requires **.NET 10**. Depends on [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU),
[SpawnDev.SpawnJS](https://github.com/LostBeard/SpawnDev.SpawnJS) and [SpawnDev.EBML](https://github.com/LostBeard/SpawnDev.EBML).
The CPU reference implementations (for verification, with Concentus for Opus) are a separate project,
`SpawnDev.Codecs.References`, so the main library has no third-party codec dependency.

## Quick start

### 1. Create an accelerator

The application owns the accelerator; codec classes never dispose it.

```csharp
// Desktop: CUDA when present, then OpenCL, then CPU
using ILGPU;
using SpawnDev.ILGPU;

using var context = Context.Create(b => b.CPU().Cuda().OpenCL());
using var accelerator = context.CreatePreferredAccelerator(new AcceleratorRequirements { RequiresAtomics = true });
```

```csharp
// Blazor WebAssembly: WebGPU, falling back to WebGL, then Wasm
using SpawnDev.ILGPU;

var builder = Context.Create();
await builder.AllAcceleratorsAsync();
var context = builder.ToContext();
var accelerator = await context.CreatePreferredAcceleratorAsync();
```

### 2. Decode any image

```csharp
using SpawnDev.Codecs.Images;

using var images = new ImageDecoderGpu(accelerator);
DecodedImage picture = await images.DecodeAsync(File.ReadAllBytes("photo.jpg"));
// picture.Info: Format, Width, Height, HasAlpha, BitDepth, FrameCount, Orientation
// picture.Pixels: Width x Height x 4 bytes of RGBA8
```

### 3. Encode WebP, PNG, JPEG

```csharp
using SpawnDev.Codecs.Images.Png;
using SpawnDev.Codecs.Images.Jpeg;
using SpawnDev.Codecs.Images.WebP.Lossy;

using var webp = new WebPLossyEncoderGpu(accelerator);
byte[] lossy = await webp.EncodeAsync(rgba, width, height, new WebPLossyEncodeOptions { Quality = 80 });   // alpha -> VP8X + ALPH

using var png = new PngEncoderGpu(accelerator);
byte[] lossless = await png.EncodeAsync(rgba, width, height, new PngEncodeOptions
{
    TextChunks = new Dictionary<string, string> { ["parameters"] = "seed 1, 20 steps" },
});

using var jpeg = new JpegEncoderGpu(accelerator);
byte[] photo = await jpeg.EncodeAsync(rgba, width, height, new JpegEncodeOptions { Quality = 90, Progressive = true });
```

### 4. Keep it on the device

When the pixels are already in a GPU buffer (a render target, a model's output), use the `ArrayView1D` overload: the
encoder reads them where they are and writes the file into a buffer you own. Only a few integers cross to the host.

```csharp
long capacity = PngEncoderGpu.GetMaxEncodedSize(width, height);
using var file = accelerator.Allocate1D<byte>(capacity);
int length = await png.EncodeAsync(rgbaOnDevice.View, width, height, file.View);
```

In the browser the `TypedArray` overloads take and return JS typed arrays (`Uint8Array`, `Float32Array`), so a picture
from `fetch`, OPFS or a canvas goes to the GPU and back without entering the .NET heap. See [Docs/api.md](Docs/api.md).

### 5. Video and audio

```csharp
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;

using var vp8 = new Vp8KeyframeEncoderGpu(accelerator);
byte[] frame = await vp8.EncodeKeyFrameAsync(y, width, u, width / 2, v, width, height, baseQIndex: 30);

using var vp8Decoder = new Vp8FullKeyframeDecoderGpu(accelerator);
var decoded = await vp8Decoder.DecodeKeyFrameAsync(frame);   // YPlane, UPlane, VPlane

using var vorbis = new VorbisAudioDecoderGpu(accelerator, identificationHeader, setupHeader);
float[] pcm = await vorbis.DecodePacketAsync(packet);        // interleaved samples of one packet

using var opusEncoder = new OpusEncoderGpu(accelerator, 48000, 2, OpusEncoderApplication.Audio) { Bitrate = 96000 };
byte[] opusPacket = await opusEncoder.EncodePacketAsync(pcm16, frameSize: 960);   // 20 ms of interleaved int16

using var opusDecoder = new OpusDecoderGpu(accelerator, 48000, 2);
float[] opusPcm = await opusDecoder.DecodePacketAsync(opusPacket);
```

## What is in the box

| Area | Encode | Decode |
|------|--------|--------|
| **PNG** | `PngEncoderGpu`: deflate on the accelerator (independent blocks), adaptive filters, RGBA / RGB / grey / grey + alpha, 8 and 16-bit, tEXt / iTXt | `PngDecoderGpu`: all colour types and depths, Adam7, tRNS, CRC and Adler-32 checks, APNG frames, `ReadTextAsync` |
| **JPEG** | `JpegEncoderGpu`: baseline and progressive, 4:4:4 / 4:2:2 / 4:2:0, optimized Huffman tables, restart markers, EXIF orientation; byte-identical with libjpeg-turbo | `JpegDecoderGpu`: baseline, extended and progressive, grey / YCbCr (all common subsamplings) / CMYK / YCCK, restart markers, EXIF orientation; bit-exact with libjpeg-turbo |
| **WebP** | `WebPLosslessEncoderGpu` (VP8L, alpha planes, metadata, animation); `WebPLossyEncoderGpu` (VP8, byte-identical with libwebp, alpha via ALPH) | `WebPDecoderGpu`: lossless, lossy, ALPH, animations (lossy, lossless and mixed), ICC / EXIF / XMP |
| **GIF** | `GifEncoderGpu`: quantization, LZW, animations | `GifDecoderGpu`: LZW, interlace, transparency, disposal, animations |
| **BMP** | `BmpEncoderGpu` | `BmpDecoderGpu` |
| **Any image** | n/a | `ImageDecoderGpu` recognizes PNG, JPEG, WebP, GIF and BMP on the device and dispatches |
| **VP8** | `Vp8KeyframeEncoderGpu` (keyframes, one token partition, batches of frames) | `Vp8FullKeyframeDecoderGpu` (full keyframe syntax, loop filter, segments, B_PRED) |
| **VP9** | `Vp9KeyframeEncoderGpu` (keyframes, batches) | `Vp9KeyframeDecoderGpu` (keyframes) |
| **AV1** | `Av1KeyframeEncoderGpu` (keyframes: DC prediction, 16x16 blocks, OBU framing on the device) | `Av1KeyframeDecoderGpu` (single tile) |
| **Vorbis** | `VorbisAudioEncoderGpu` (first version: mono, fixed block size) | `VorbisAudioDecoderGpu` (any channel count, block switching, libvorbis parity) |
| **FLAC** | `FlacEncoderGpu` (first version: verbatim subframes) | `FlacDecoderGpu` (first version, same subset) |
| **Opus** | `OpusEncoderGpu` (CELT-only: 2.5 .. 120 ms, CBR / VBR / constrained VBR, complexity 0..9, 8 .. 48 kHz; byte-identical with libopus 1.5.2) | `OpusDecoderGpu` (SILK, CELT, hybrid, mode switches, redundancy, loss concealment, DTX; equal to libopus 1.5.2) |
| **Compression** | `ZlibCompressorGpu`, `DeflateGpu` | `InflateGpu` |
| **Containers** | Ogg, IVF, RIFF/WAVE, AIFF; WebM / Matroska through SpawnDev.EBML | the same, plus an MP4 box reader |

Video is keyframe-only today. Limits that remain (for example the VP9 decoder and AV1 at 64-pixel multiples) are listed
per codec in [Docs/video.md](Docs/video.md) and [Docs/status.md](Docs/status.md).

## Backends

| Backend | Runs on | Kernels compile to | Where |
|---------|---------|--------------------|-------|
| **WebGPU** | GPU | WGSL | Browser |
| **WebGL** | GPU | GLSL ES 3.0 (no atomics or scatter stores: some codecs skip it) | Browser |
| **Wasm** | Web Workers | WebAssembly | Browser |
| **CUDA** | NVIDIA GPU | PTX | Desktop |
| **OpenCL** | Any OpenCL GPU | OpenCL C | Desktop |
| **CPU** | CPU cores | .NET (Velocity SIMD by default on 64-bit) | Desktop |

## How it works

```
  input (pixels, PCM, a file)        caller's ArrayView1D | TypedArray | byte[]
        |  one upload (or none: already on the device)
        v
  parallel stages     colour conversion, prediction, transforms, quantization, loop filters, filters
        |             (one thread per pixel / block / candidate)
        v
  serial stages       bit writers, range / bool / Huffman coders, the libwebp RD loop
        |             (one thread per stream, resumable: many short launches, state in device buffers)
        v
  container           RIFF / PNG chunks / JFIF / OBUs written by a kernel, one thread per output byte
        |
        v
  output              into the caller's view, or one readback into a byte[] / Uint8Array
```

The host never touches codec data: no per-pixel loops, no bitstream assembly, no entropy coding on the CPU. Stages that
look inherently sequential run on the GPU as single threads, split into bounded launches; the throughput work happens
inside the kernels. [Docs/architecture.md](Docs/architecture.md) has the rules and the reasons.

## Verification

Correctness is checked against the reference implementations, never against this library's own output:

| Codec | Oracle | Bar |
|-------|--------|-----|
| WebP lossy encode | libwebp 1.6.0 (through Pillow) | byte-identical files (methods 3 / 4) |
| WebP lossy / lossless decode | libwebp | identical pixels |
| WebP lossless encode, PNG, GIF, BMP encode | libwebp / Pillow and our decoders | exact round trip; Pillow decodes our files to the same pixels |
| JPEG encode / decode | libjpeg-turbo (through Pillow) | byte-identical files / identical pixels |
| VP8 keyframe decode | ffmpeg | identical planes on 182 files |
| Vorbis decode | libvorbis (through libsndfile) | within 1e-4 per sample (observed 7.5e-7) |
| Opus decode | libopus 1.5.2 (fixed point, built from source) | equal PCM and final range on every packet of 21 streams |
| Opus encode (CELT-only) | libopus 1.5.2 (fixed point, built from source) | byte-identical packets and equal final range on 49 streams |

Fixtures are made by scripts in [`scripts/`](scripts/) (the truth comes from the reference library), and every guard is
red-checked: the fix is disabled and the test must fail by name.

## Repository layout

| Path | Contents |
|------|----------|
| `SpawnDev.Codecs/` | The library |
| `SpawnDev.Codecs.References/` | CPU reference codecs (verification), Concentus for the CPU Opus reference |
| `SpawnDev.Codecs.Demo/` | Blazor WebAssembly demo and browser test host |
| `SpawnDev.Codecs.Demo.Shared/` | The shared test suite (`CodecsTestBase`) and its fixtures (`TestData/`) |
| `SpawnDev.Codecs.DemoConsole/` | Desktop host for the shared test suite |
| `SpawnDev.Codecs.Tests/` | xUnit tests |
| `PlaywrightMultiTest/` | The test runner (desktop and browser, every backend) |
| `scripts/` | Fixture generators (reference libraries), dev scripts |
| `Plans/` | Design documents and open work |
| `Docs/` | Topic documentation |

## Documentation

| | |
|---|---|
| [Status](Docs/status.md) | Every codec by backend, with the evidence |
| [Images](Docs/images.md) | PNG, JPEG, WebP, GIF, BMP: API, options, limits |
| [Video](Docs/video.md) | VP8, VP9, AV1: API, batches, limits |
| [Audio](Docs/audio.md) | Vorbis, FLAC, Opus, audio containers |
| [API shape](Docs/api.md) | The three overloads, buffers, zero copy in the browser |
| [Architecture](Docs/architecture.md) | Host as coordinator, bounded launches, browser shader budget |
| [Testing](Docs/testing.md) | PlaywrightMultiTest, fixtures and oracles, environment switches |
| [Changelog](CHANGELOG.md) | Every change, in detail |

## Build and test

```bash
dotnet build SpawnDev.Codecs.slnx
```

Tests run through **PlaywrightMultiTest**, which drives the desktop backends and a real Chromium for the browser
backends:

```bash
dotnet test PlaywrightMultiTest --filter "FullyQualifiedName~WebGPUCodecsTests.WebPLossy"
```

Details, environment switches and the per-test desktop script are in [Docs/testing.md](Docs/testing.md).

## Patent policy

Only formats with no patent exposure: PNG, JPEG (baseline and progressive), WebP, GIF, BMP, VP8 / VP9 / AV1 (AOMedia
patent pledge), Opus (RFC 6716), FLAC and Vorbis. H.264, H.265 / HEIC and AAC are out of scope; for those, use platform
encoders through [SpawnDev.MultiMedia](https://github.com/LostBeard/SpawnDev.MultiMedia).

## Contributing

Issues and pull requests are welcome. A bug in this library, in SpawnDev.ILGPU or in SpawnDev.SpawnJS is fixed in that
library, not worked around in a consumer, so a good report is a failing test or a file that decodes wrong, with the
backend named. Run the tests for the lanes you touched before opening a pull request.

## Support the project

SpawnDev is built by one developer and a small team of AI agents. If it is useful to you,
[sponsoring on GitHub](https://github.com/sponsors/LostBeard) directly funds more codecs, more verified formats and more
backends. Stars, issues and sharing what you build with it help just as much.

## Credits

- [**ILGPU**](https://github.com/m4rs-mt/ILGPU) by Marcel Koester and contributors, the compiler that makes C# GPU kernels possible.
- [**SpawnDev.ILGPU**](https://github.com/LostBeard/SpawnDev.ILGPU), which adds the WebGPU, WebGL and Wasm backends and the Velocity CPU backend.
- [**SpawnDev.SpawnJS**](https://github.com/LostBeard/SpawnDev.SpawnJS), typed C# access to the browser's APIs.
- The reference implementations this library is checked against and ported from: libwebp, libjpeg, libpng / zlib,
  libvpx, libaom / dav1d, libvorbis, libopus and Concentus, ffmpeg, libjpeg-turbo and Pillow. Attribution is in [NOTICE.md](NOTICE.md).

Created by Todd Tanner ([@LostBeard](https://github.com/LostBeard)) together with a crew of Claude AI agents: Riker,
Data, Tuvok and Geordi. Live long and prosper. 🖖

## License

[MIT](LICENSE.txt). Upstream attribution for ported reference code is in [NOTICE.md](NOTICE.md).
