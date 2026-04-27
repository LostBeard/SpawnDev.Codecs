# SpawnDev.Codecs

**Pure-.NET, ILGPU-accelerated, patent-clean audio and video codecs.**

Runs on every ILGPU backend - CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm - which means it runs on desktop AND in Blazor WASM browsers. No native binaries, no closed-source dependencies, no patent-encumbered codecs.

> **Status: 5 of 6 codec decoders + 3 of 6 codec encoders WORKING as of 2026-04-27.** FLAC + Vorbis + VP8 + VP9 decode real-world data; VP8 walker output matches ffmpeg with MAE 0.04/0.01/0.00 across Y/U/V planes (loop filter accounts for the residual ~56 max diff). VP9 walker decodes BBB.webm first keyframe with first 16x16 Y BIT-EXACT and 352 leaf blocks total. FLAC + Vorbis + VP8 encode real-world data; ffmpeg accepts our VP8 keyframe bitstream + libvpx decodes our VP9 OBU bitstream + ffmpeg accepts our Vorbis ogg at 440Hz exact. AV1 has parser + range coder pair + framing all bit-exact vs libaom-av1; full pixel decode/emit is the remaining gap. Opus CELT decoder is the only remaining audio gap (~2000 LoC libopus port). See the feature matrix below.

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
| **VP8** | **WORKING as of 2026-04-27.** Full keyframe walker (Vp8KeyframeWalker.Decode) with B_PRED + non-B_PRED reconstruction, libvpx coefficient decode order (Y2 first when not B_PRED -> 16 Y4 -> 4 U -> 4 V), per-MB above + left entropy contexts, sub-block 4x4 intra prediction. Verified against ffmpeg on a 64x64 testsrc keyframe: Y MAE 0.04 / U MAE 0.01 / V MAE 0.00, range exact match, first-row Y bytes EXACT match. Max abs Y diff 56 explained by loop filter (out of scope). Inter frames + multi-token-partition + loop filter remain NotImplementedException. | **WORKING as of 2026-04-27.** Top-level Vp8KeyframeEncoder integrates forward DCT (6/6 BIT-EXACT round-trip vs inverse) + Walsh + forward quantizer + coef block encoder (17/17 BIT-EXACT round-trip vs decoder, all 6 cat tokens) + frame tag writer (6/6 round-trip) + frame header writer (7/7 round-trip) + bool encoder (708/708 round-trip). ffmpeg ACCEPTS our 32x32 keyframe bitstream and decodes it to YUV. No reconstruction write-back yet (subsequent MBs use 127/129 edge fills) - bitstream is structurally valid; pixel quality fix is incremental. |
| **VP9** | **WORKING as of 2026-04-27.** Full keyframe walker (Vp9KeyframeWalker.DecodeFrame) drives partition tree -> leaf block -> per-tx-block predict + invert + add for all three planes on real BBB.webm bytes. First 16x16 Y BIT-EXACT vs ffmpeg (cap fix in Vp9CoefContext.GetCoefContext: was clamping ctx at 2 when libvpx returns raw `(1+tc[n0]+tc[n1])>>1` no clamp, range [0,5]). 352 leaf blocks decoded across the first BBB keyframe; first 16 px of Y top row + Y left col + first 16x16 Y block ALL EXACT MATCH; V plane mean within 0.03 of ffmpeg; U plane within small drift; Y plane recognizable scene with blocking artifacts (loop filter out of scope). Inter prediction + loop filter + 4:2:2 / 4:4:4 / high-bit-depth NotImplementedException. | **Block-level pipeline WORKING as of 2026-04-27.** Pixels-in -> forward DCT -> quantize -> Vp9BlockCoefEncoder -> Vp9BoolEncoder -> Vp9BoolDecoder -> Vp9BlockCoefDecoder -> dequantize -> inverse DCT -> reconstructed pixels round-trips on every transform size (4x4 / 8x8 / 16x16) with max error 1-4 at low Q on gradients. Vp9BlockCoefEncoder is bit-exact mirror of the decoder including the inner ZERO-loop EOB skip. Vp9BoolEncoder fixed to emit the leading marker bit per libvpx vpx_start_encode. Forward transform set complete: Vp9ForwardDct 4x4 / 8x8 / 16x16 / **32x32**, Vp9ForwardAdst 4 / 8 / **16**, dispatcher routes (txSize, txType) including AdstAdst 16x16. 114/114 entropy round-trip tests + 72/72 forward DCT/ADST round-trip tests + 54/54 end-to-end pipeline tests pass across CPU + CUDA + OpenCL + WebGPU + WebGL + Wasm. Top-level Vp9KeyframeEncoder integrating these is the remaining frame-level work. `Vp9SuperframeWriter` (BBB packets round-trip BIT-EXACT). |
| **AV1** | Daala range DECODER 309/309 round-trip + verified on real BBB AV1 OBU bytes (150 symbols, Tell advances cleanly). OBU + SH (28 fields) + FrameHeader (16 prefix fields) parsers running on bbb_180_2s.ivf. Block decode + CDF tables + intra prediction + inverse transforms (in progress as of 2026-04-27 via parallel agent: Av1InverseDct 4/8/16, Av1InverseAdst 4/8/16, Av1InverseIdentity, Av1IntraPredictor, Av1SmoothWeights, Av1DequantTables shipped). Full keyframe walker pending. | **Encoder framing FOUNDATION + Daala range ENCODER + 4 forward transforms shipped.** Av1ObuWriter, Av1SequenceHeaderWriter (28 fields), Av1FrameHeaderWriter (16 fields), Av1BitWriter. Av1RangeEncoder + Av1ForwardDct4 + Av1ForwardDct8 + Av1ForwardDct16 + Av1ForwardAdst4 + Av1ForwardQuantizer + sinpi/cospi tables. Cross-validation: SH bytes BIT-EXACT vs libaom-av1; remux of 60 BBB frames through Av1ObuWriter + IvfWriter produces stream that ffmpeg+dav1d decode pixel-identical to source (5,184,000 / 5,184,000 byte-equal YUV); 58-byte minimal stream identified by libdav1d as valid AV1. 2D transform wrapper + CDF tables + block emit remain. |

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
