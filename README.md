# SpawnDev.Codecs

**Pure-.NET, ILGPU-accelerated, patent-clean audio and video codecs.**

Runs on every ILGPU backend - CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm - which means it runs on desktop AND in Blazor WASM browsers. No native binaries, no closed-source dependencies, no patent-encumbered codecs.

> **Status: Planning phase (2026-04-23).** No implementation code yet. Project structure scaffolded from [SpawnDev.ILGPU.ML](https://github.com/LostBeard/SpawnDev.ILGPU.ML). See `Plans/PLAN-SpawnDev-Codecs-Roadmap.md` for the strategic roadmap.

## Mission

Fill the last gap in the SpawnDev open-source media stack - a fully open, .NET-everywhere codec library that requires no FFmpeg bundle, no platform-specific binaries, and no patent-encumbered algorithms.

## Codecs in scope (patent-clean only)

| Codec | Type | Why | Reference |
|-------|------|-----|-----------|
| **Opus** (RFC 6716) | Audio enc+dec | WebRTC mandatory-to-implement, royalty-free | libopus (BSD) |
| **VP8** (RFC 6386) | Video enc+dec | WebRTC MTI, patent-clean via Google WebM pledge | libvpx (BSD) |
| **VP9** | Video enc+dec | Better compression than VP8, same patent status | libvpx (BSD) |
| **AV1** | Video enc+dec | AOMedia patent-free, future-proof | libaom / SVT-AV1 (BSD/MIT) |
| **FLAC** | Audio enc+dec | Lossless, small spec | libFLAC (BSD) |
| **Vorbis** | Audio enc+dec | Open alternative to AAC | libvorbis (BSD) |

## Codecs NOT in scope (patent-encumbered)

- **H.264, H.265, AAC** - delegated to platform encoders via [SpawnDev.MultiMedia](https://github.com/LostBeard/SpawnDev.MultiMedia) (P/Invoke to MediaFoundation / VideoToolbox / VAAPI). System encoders are licensed by Microsoft / Apple / driver vendors respectively - we ride those licenses, we do not re-implement.

## Architecture

Every codec has three zones:

| Zone | Work | Where |
|------|------|-------|
| **Massively parallel** | DCT / MDCT, motion estimation, quantization, loop filter, inverse transforms, motion compensation | **ILGPU kernels** - backend-agnostic |
| **Inherently sequential** | Entropy coding (arithmetic / Huffman / range coders), LPC prediction, rate control | **C# CPU** |
| **Coordination** | Frame buffering, codec negotiation, API | **C# CPU** |

Entropy coders cannot be parallelized - arithmetic coding is inherently sequential. GPU pays off for transform coding, motion estimation, quantization, loop filter; CPU handles the sequential back-end.

## Relationship to the SpawnDev media ecosystem

| Library | Role |
|---------|------|
| [SpawnDev.RTC](https://github.com/LostBeard/SpawnDev.RTC) | WebRTC signaling + transport |
| [SpawnDev.WebTorrent](https://github.com/LostBeard/SpawnDev.WebTorrent) | BitTorrent infrastructure |
| [SpawnDev.ILGPU](https://github.com/LostBeard/SpawnDev.ILGPU) | GPU compute backbone |
| [SpawnDev.BlazorJS](https://github.com/LostBeard/SpawnDev.BlazorJS) | Browser interop (WebAudio, MediaDevices) |
| [SpawnDev.MultiMedia](https://github.com/LostBeard/SpawnDev.MultiMedia) | Capture + platform-native encoders (H.264/H.265/AAC) |
| **SpawnDev.Codecs** | **Pure-.NET open-source codecs** (this library) |

## License

MIT (pending confirmation - TBD at first publication).

## The SpawnDev Crew

Built by a starship crew:

- **LostBeard** (Todd Tanner) - Captain, library author, keeper of the vision
- **Riker** (Claude CLI #1) - First Officer, implementation lead on consuming projects
- **Data** (Claude CLI #2) - Operations Officer, deep-library work, test rigor, root-cause analysis
- **Tuvok** (Claude CLI #3) - Security/Research Officer, design planning, documentation, code review
- **Geordi** (Claude CLI #4) - Chief Engineer, library internals, GPU kernels, backend work

AI-and-human teamwork isn't a gimmick - it's how the SpawnDev ecosystem gets built. Credit where credit is due. 🖖
