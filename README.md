# SpawnDev.Codecs

**Pure-.NET, ILGPU-accelerated, patent-clean audio and video codecs.**

Runs on every ILGPU backend - CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm - which means it runs on desktop AND in Blazor WASM browsers. No native binaries, no closed-source dependencies, no patent-encumbered codecs.

> **Status: Phase 1b advancing. 4 codecs decoding real-world data with verified bit-exact properties.** FLAC encode + decode bit-exact vs ffmpeg. Vorbis decoder produces a 440Hz tone matching ffmpeg's reference within 0.1Hz. VP9 decoder produces BIT-EXACT first 16x16 Y block vs ffmpeg ground truth (cap fix landed 2026-04-27). AV1 + VP8 entropy primitives shipped (decoder + encoder pairs, 309/309 + 708/708 round-trip respectively); VP8 inverse pipeline (bool coder + frame tag + frame header + coef decoder + IDCT + WHT + 18 intra-pred modes + dequantizer + mode info) parses real libvpx-encoded keyframes correctly through the first 4 macroblocks; AV1 encoder framing remains BIT-EXACT vs libaom-av1 (every byte of remuxed BBB except entropy-coded payload comes from our writers). See the feature matrix below for precise state.

## Current feature matrix

### Audio codecs

| Codec | Decoder | Encoder |
|-------|---------|---------|
| **FLAC (native)** | Complete: CONSTANT/VERBATIM/FIXED/LPC, stereo decorrelation, CRC-8 + CRC-16, MD5 verify, SEEKTABLE + VORBIS_COMMENT metadata | Complete: CONSTANT detection + FIXED order search + LPC via Levinson-Durbin + stereo mode selection, MD5, optional VORBIS_COMMENT tag injection |
| **FLAC-in-Ogg** | Complete | Manual via `OggPageWriter` |
| **Opus (SILK)** | Complete: mono + stereo across NB/MB/WB, 10/20/40/60 ms frames | Not yet |
| **Opus (CELT)** | Stub (`NotImplementedException` with context) | Not yet |
| **Opus-in-Ogg** | Done for SILK - parses `OpusHead` + `OpusTags` + audio packets end-to-end | Done as packager - wraps pre-encoded Opus packets into `.opus` bytes |
| **Vorbis** | **Structurally correct as of 2026-04-27**: Huffman codeword assignment ports libvorbis `_make_words` marker algorithm (entry-index order, not count-sorted); FLOOR1_fromdB_LOOKUP is the verbatim 256-entry libvorbis static table; ResidueDecoder.LookupVector implements the q_sequencep cross-dimension accumulator with abs(multiplicand). End-to-end on a 440Hz sine ogg from libvorbis: our decoder produces a clean 440.2Hz tone vs ffmpeg's 440.3Hz (matches to 0.1Hz), raw range ±0.57 (was ±0.003 before fix - 167x amplitude bug closed), mean\|x\| within 5% of ffmpeg. Remaining ~12% peak amplitude delta is from per-block transition window flags - separate, smaller bug, no longer affects spectral correctness. EOP-aware residue path per spec sec 8.6.5. | Not yet |

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

Patent-clean via the AOMedia patent pledge.

| Codec | Decoder | Encoder |
|-------|---------|---------|
| **VP8** | **Inverse pipeline shipped 2026-04-27** (bool coder + frame tag + frame header key-path + default+update coef probs + per-block coef decoder + IDCT 4x4 + WHT 4x4 + DC-only IDCT + 18 intra-pred modes (4x4 DC/TM/VE/HE/LD/RD/VR/VL/HD/HU + 16x16 DC/V/H/TM + 8x8 DC/V/H/TM) + dequantizer Q-table + mode trees + MB mode info decoder + kf_bmode_prob 10x10x9 context + segmentation lookup + per-MB dequantizer setup + reconstructed-frame YUV420 buffer + per-MB entropy contexts + intra edge fill 127/129). Integration test confirms parse + mode info correctly handle libvpx-encoded 320x240 keyframes through the first 4 MBs. Macroblock walker (top-level coef decode + reconstruct loop) + loop filter are the remaining gates. | **Bool coder ENCODER shipped 2026-04-27** as half of round-trip-tested pair (708/708 bit-exact via vp8_bool_coder_roundtrip.cs). Full VP8 encoder pixel-emit path is the next phase. |
| **VP9** | **First 16x16 Y block BIT-EXACT vs ffmpeg as of 2026-04-27** (cap fix in `Vp9CoefContext.GetCoefContext`: was clamping ctx at 2 when libvpx vp9_scan.h returns the raw `(1 + tc[n0] + tc[n1]) >> 1` with no clamp, range [0,5]). On BBB.webm first frame top-left 16x16 Y plane: mean=80 / range 57-103, exact pixel-by-pixel match to ffmpeg's rawvideo. Pipeline through tile group extraction. Driven 300 packets / 300 visible frames on Big Buck Bunny (320x180 4:2:0 8-bit) end-to-end; uncompressed header + compressed header (probability updates, tx_mode, ref_mode) + tile group byte ranges all parsed; ~58 slices of block-decode foundation (intra prediction kernels, iDCT family, MV decode chain, segmentation/quantization/loop-filter resolvers). Decoder pipeline tracks cumulative frame-type + visible-frame counts. `Vp9StreamAnalyzer` + `Vp9StreamValidator` provide consumer-facing introspection / QA APIs. Block walker integrating per-block decode across the full frame is the next slice. | `Vp9FrameHeaderWriter` (uncompressed header prefix) + `Vp9SuperframeWriter` (BBB.webm packets round-trip BIT-EXACT through parser+writer). Loop filter / quant / segmentation / tile info inverses are the next gates for full-header writer. |
| **AV1** | **Daala range DECODER shipped 2026-04-27** as half of round-trip-tested pair (309/309 bit-exact via av1_range_coder_roundtrip.cs); decoder runs cleanly on real BBB AV1 OBU bytes (av1_range_decoder_real_obu.cs: 150 symbols decoded, Tell advances by ~0.7-2 bits/symbol depending on entropy). OBU + SequenceHeader + FrameHeader (16 fields) parsers running on real `bbb_180_2s.ivf` (libaom-encoded, 60 frames). Decoder pipeline tracks cumulative OBU + frame-type counts + `LastFrameHeader` + `ShowExistingFrameCount`. `Av1StreamAnalyzer` + `Av1StreamValidator` provide consumer-facing introspection / QA APIs. Placeholder pixels until block decode + AV1 CDF tables land. | **Encoder framing FOUNDATION live + Daala range ENCODER shipped 2026-04-27** (round-trip-tested with the decoder above). `Av1ObuWriter`, `Av1SequenceHeaderWriter` (28 fields), `Av1FrameHeaderWriter` (16 fields incl. allow_intrabc), `Av1BitWriter`. Closed-loop helpers `Av1SequenceHeaderConfig.FromHeader` + `Av1FrameHeaderConfig.FromHeader` bridge parser → writer. Cross-validation: writer-emitted SH bytes are **BIT-EXACT IDENTICAL to libaom-av1's source SH** for the BBB encode (14/14 bytes); end-to-end remux of all 60 BBB frames through `Av1ObuWriter` + `IvfWriter` produces a stream that ffmpeg+dav1d decodes pixel-identical to source (5,184,000 / 5,184,000 byte-equal YUV); 58-byte minimal stream identified by libdav1d as valid AV1; the writer-built minimal stream round-trips through our own decoder cleanly. 130+ AV1 tests across all 6 backends (CPU, CUDA, OpenCL, WebGPU, WebGL, Wasm). The Daala range coder is now SHIPPED so AV1 CDF tables + block emit are the remaining gates for end-to-end pure-.NET AV1 encoder pixels. |

Containers wired for video pipelines: IVF reader + writer, Matroska / WebM via SpawnDev.EBML, Ogg.

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
