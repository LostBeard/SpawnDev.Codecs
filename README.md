# SpawnDev.Codecs

**Pure-.NET, ILGPU-accelerated, patent-clean audio and video codecs.**

Runs on every ILGPU backend - CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm - which means it runs on desktop AND in Blazor WASM browsers. No native binaries, no closed-source dependencies, no patent-encumbered codecs.

> **Status: Phase 1a shipping.** Audio codec work is live. FLAC (encode + decode), Opus SILK decode + Opus-in-Ogg, native and Ogg-wrapped container formats, and a Vorbis decoder scaffold are all merged to `master`. See the feature matrix below for precise state.

## Current feature matrix

### Audio codecs

| Codec | Decoder | Encoder |
|-------|---------|---------|
| **FLAC (native)** | Complete: CONSTANT/VERBATIM/FIXED/LPC, stereo decorrelation, CRC-8 + CRC-16, MD5 verify, SEEKTABLE + VORBIS_COMMENT metadata | Complete: CONSTANT detection + FIXED order search + LPC via Levinson-Durbin + stereo mode selection, MD5, optional VORBIS_COMMENT tag injection |
| **FLAC-in-Ogg** | Complete | Manual via `OggPageWriter` |
| **Opus (SILK)** | Complete: mono + stereo across NB/MB/WB, 10/20/40/60 ms frames | Not yet |
| **Opus (CELT)** | Stub (`NotImplementedException` with context) | Not yet |
| **Opus-in-Ogg** | Done for SILK - parses `OpusHead` + `OpusTags` + audio packets end-to-end | Done as packager - wraps pre-encoded Opus packets into `.opus` bytes |
| **Vorbis** | Structural decoder: setup parse + floor-1 posterior + curve synthesis + residue framing + inverse coupling + IMDCT + window overlap-add composed; awaits bit-accuracy validation against libvorbis test vectors | Not yet |

### Audio containers

| Container | Read | Write |
|-----------|------|-------|
| RIFF / WAVE | Yes - 8/12/16/20/24/32-bit PCM, 32-bit float, multi-channel, LIST-chunk skipping, WAVE_FORMAT_EXTENSIBLE | Yes |
| AIFF | Yes - 8/16/24/32-bit PCM, IEEE 80-bit extended sample rate | Yes |
| Ogg | Yes - page + packet, CRC-32 per RFC 3533, multi-bitstream demux | Yes |

### Multimedia containers

| Container | Read | Write |
|-----------|------|-------|
| WebM / Matroska (EBML) | Via [`SpawnDev.EBML 3.0.0`](https://www.nuget.org/packages/SpawnDev.EBML) - schema-driven path navigation, non-destructive edits | Via `SpawnDev.EBML` |
| MP4 / ISOBMFF | Structural box reader (ftyp, container recursion, size=0 "rest of file" convention) | Not yet |

### Transforms (shared)

- MDCT + IMDCT reference implementations (CPU, O(N²))
- Round-trip identity `MDCT(IMDCT(X)) = N·X` validated to float precision
- FFT-accelerated CPU and ILGPU-kernel variants planned

### Video codecs

Not started. VP8, VP9, AV1 are planned (patent-clean via AOMedia pledge).

### Out of scope (patent-encumbered)

H.264, H.265, AAC, MP3 - delegated to platform encoders via [SpawnDev.MultiMedia](https://github.com/LostBeard/SpawnDev.MultiMedia).

## Example

```csharp
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Wav;

// WAV -> FLAC -> WAV lossless round-trip.
var wav = WavFileCodec.ReadFile("in.wav");
FlacEncoder.EncodeToFile("out.flac", wav.InterleavedSamples, wav.SampleRateHz, wav.Channels, wav.BitsPerSample);
var flac = FlacDecoder.DecodeFile("out.flac");
WavFileCodec.WriteFile("roundtrip.wav", flac.InterleavedSamples, flac.StreamInfo.SampleRateHz,
    flac.StreamInfo.Channels, flac.StreamInfo.BitsPerSample);
```

## Architecture

Every codec has three zones:

| Zone | Work | Where |
|------|------|-------|
| **Massively parallel** | DCT / MDCT, motion estimation, quantization, loop filter, inverse transforms, motion compensation | **ILGPU kernels** - backend-agnostic |
| **Inherently sequential** | Entropy coding (arithmetic / Huffman / range coders), LPC prediction, rate control | **C# CPU** |
| **Coordination** | Frame buffering, codec negotiation, API | **C# CPU** |

Entropy coders cannot be parallelized - arithmetic coding is inherently sequential. GPU pays off for transform coding, motion estimation, quantization, loop filter; CPU handles the sequential back-end.

## Testing

Every slice is validated through the `PlaywrightMultiTest` harness, which runs the same test suite across every ILGPU backend (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU) and aggregates the results. Thousands of cross-backend test executions gate each merge.

```
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~Flac"
```

The in-browser demo at `SpawnDev.Codecs.Demo/` also ships a `/benchmarks` page that runs throughput + compression measurements against the live library.

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

MIT - see `LICENSE.txt`. Upstream attribution for reference-ported code is in `NOTICE.md`.

## The SpawnDev Crew

Built by a starship crew:

- **LostBeard** (Todd Tanner) - Captain, library author, keeper of the vision
- **Riker** (Claude CLI #1) - First Officer, implementation lead on consuming projects
- **Data** (Claude CLI #2) - Operations Officer, deep-library work, test rigor, root-cause analysis
- **Tuvok** (Claude CLI #3) - Security/Research Officer, design planning, documentation, code review
- **Geordi** (Claude CLI #4) - Chief Engineer, library internals, GPU kernels, backend work

AI-and-human teamwork isn't a gimmick - it's how the SpawnDev ecosystem gets built. Credit where credit is due. 🖖
