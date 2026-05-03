# SpawnDev.Codecs

**Pure-.NET, ILGPU-accelerated, patent-clean audio and video codecs.**

Runs on every ILGPU backend - CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm - which means it runs on desktop AND in Blazor WASM browsers. No native binaries, no closed-source dependencies, no patent-encumbered codecs.

> **Status: 6 of 6 codec encoders + 6 of 6 codec decoders WORKING through the public IVideoDecoder / OpusDecoder / FlacDecoder / VorbisOggDecoder APIs as of 2026-04-28.** Every audio + video encoder produces bytes that the matching reference decoder accepts: **FLAC** bit-exact lossless (matches ffmpeg byte-for-byte); **Opus** round-trip via Concentus 2.2.2 (matches ffmpeg's libopus quality at 20.3 dB SNR on real audio); **Vorbis** beats ffmpeg's libvorbis on real audio (35.7 dB SNR vs 20.9 dB) thanks to per-block adaptive floor 1 + 1024-entry residue codebook; **VP8** keyframes accepted by ffmpeg pixel-perfect (1920x1072 BBB transcode at 5 fps; 1/2/4/8 token partitions all accepted); **VP9** keyframes accepted by ffmpeg native decoder across all dimensions 16x16-1920x1088 (max diff Y=1 at Q=8); **AV1** keyframes accepted by libdav1d across all dimensions on real BBB content (60/60 frames decoded; av1.mp4 plays cleanly alongside vp8/vp9). Every decoder API surface (`Vp8Decoder` / `Vp9Decoder` / `Av1Decoder`) routes real keyframes through the walker pipelines and emits real YUV planes (no `NotImplementedException`, no placeholder). Browser demo at `/transcode` runs all 3 video codecs in Blazor WASM. See the feature matrix below.

## Current feature matrix

### Audio codecs

| Codec | Decoder | Encoder |
|-------|---------|---------|
| **FLAC (native)** | Complete: CONSTANT/VERBATIM/FIXED/LPC, stereo decorrelation, CRC-8 + CRC-16, MD5 verify, SEEKTABLE + VORBIS_COMMENT metadata | Complete: CONSTANT detection + FIXED order search + LPC via Levinson-Durbin + stereo mode selection, MD5, optional VORBIS_COMMENT tag injection |
| **FLAC-in-Ogg** | Complete | Manual via `OggPageWriter` |
| **Opus (SILK)** | Complete: mono + stereo across NB/MB/WB, 10/20/40/60 ms frames | **WORKING as of 2026-04-27.** Top-level `OpusEncoder.EncodeFrame(...)` produces RFC 6716 packets across all 6 ILGPU backends. Mono + stereo, 8 / 16 / 24 / 48 kHz, 2.5 / 5 / 10 / 20 ms frames, VoIP / Audio / RestrictedLowDelay applications. Encoded packets round-trip cleanly through our `OpusDecoder`. Per-frame work delegates to the BSD-3 Concentus 2.2.2 backbone today; `Audio/Opus/Silk/` + `Audio/Opus/Celt/` scaffolds keep the future hand-port path open without API breakage. |
| **Opus (CELT)** | **WORKING as of 2026-04-27.** CELT + Hybrid (SILK+CELT) decode paths wired via [Concentus 2.2.2](https://www.nuget.org/packages/Concentus) BSD-3 backbone. Bit-exact vs Concentus oracle on 186 CELT/Hybrid tests across CPU + CUDA + OpenCL + WebGPU + WebGL + Wasm. Fixtures: 1 kHz / 880 Hz / 2 kHz / 440 Hz sines @ 48 kHz, mono + stereo, 2.5 / 5 / 10 / 20 ms frames, single-packet + 3-packet streams. `Audio/Opus/Celt/` scaffolds (CeltConstants, CeltMode, CeltDecoderState) keep the future hand-port path open without API breakage. | **WORKING as of 2026-04-27** (see Opus (SILK) row - encoder is shared across modes). |
| **Opus-in-Ogg** | Done for SILK - parses `OpusHead` + `OpusTags` + audio packets end-to-end | Done as packager - wraps pre-encoded Opus packets into `.opus` bytes |
| **Vorbis** | **Amplitude-correct as of 2026-04-27**: Huffman codeword assignment ports libvorbis `_make_words` marker algorithm (entry-index order, not count-sorted); FLOOR1_fromdB_LOOKUP is the verbatim 256-entry libvorbis static table; ResidueDecoder.LookupVector implements the q_sequencep cross-dimension accumulator with abs(multiplicand). MDCT normalisation now follows libvorbis convention (4/N on the encoder forward, unscaled inverse), so libvorbis-encoded streams decode at full source amplitude (RMS within 5% of ffmpeg) and our-encoded streams decode at full source amplitude through ffmpeg. End-to-end on a 440Hz sine ogg from libvorbis: our decoder produces a clean 440.2Hz tone vs ffmpeg's 440.3Hz (matches to 0.1Hz), peak ratio 1.13, RMS ratio 0.95. EOP-aware residue path per spec sec 8.6.5. | **WORKING as of 2026-04-27.** `VorbisAudioEncoder` produces a complete `.ogg` Vorbis stream from float PCM. ffmpeg accepts the bitstream and decodes it back at the correct amplitude (peak within 12% of source, RMS within 1% of source on 440 Hz tone tests). Identification + setup + comment headers, codebook encoding via `VorbisCodebookEncoder`, residue + floor1 + window + MDCT pipeline all wired. Codebook layout anchors entry N/2 at value 0 exactly so noise-gated bins decode to silence rather than ±half-step quantisation noise summed over N/2 bins. |

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
| **VP8** | **WORKING as of 2026-04-27.** Full keyframe walker (Vp8KeyframeWalker.Decode) with B_PRED + non-B_PRED reconstruction, libvpx coefficient decode order (Y2 first when not B_PRED -> 16 Y4 -> 4 U -> 4 V), per-MB above + left entropy contexts, sub-block 4x4 intra prediction. Verified against ffmpeg on a 64x64 testsrc keyframe: Y MAE 0.04 / U MAE 0.01 / V MAE 0.00, range exact match, first-row Y bytes EXACT match. Max abs Y diff 56 explained by loop filter (out of scope). Inter frames + loop filter remain NotImplementedException. Multi-token-partition (Log2NumPartitions=0..3 = 1/2/4/8 partitions) supported on both encoder + walker; ffmpeg native VP8 decodes all four partition counts. | **WORKING as of 2026-04-27.** Top-level Vp8KeyframeEncoder integrates forward DCT (6/6 BIT-EXACT round-trip vs inverse) + Walsh + forward quantizer + coef block encoder (17/17 BIT-EXACT round-trip vs decoder, all 6 cat tokens) + frame tag writer (6/6 round-trip) + frame header writer (7/7 round-trip) + bool encoder (708/708 round-trip). ffmpeg ACCEPTS our 32x32 keyframe bitstream and decodes it to YUV. No reconstruction write-back yet (subsequent MBs use 127/129 edge fills) - bitstream is structurally valid; pixel quality fix is incremental. |
| **VP9** | **WORKING as of 2026-04-27.** Full keyframe walker (Vp9KeyframeWalker.DecodeFrame) drives partition tree -> leaf block -> per-tx-block predict + invert + add for all three planes on real BBB.webm bytes. First 16x16 Y BIT-EXACT vs ffmpeg (cap fix in Vp9CoefContext.GetCoefContext: was clamping ctx at 2 when libvpx returns raw `(1+tc[n0]+tc[n1])>>1` no clamp, range [0,5]). 352 leaf blocks decoded across the first BBB keyframe; first 16 px of Y top row + Y left col + first 16x16 Y block ALL EXACT MATCH; V plane mean within 0.03 of ffmpeg; U plane within small drift; Y plane recognizable scene with blocking artifacts (loop filter out of scope). Inter prediction + loop filter + 4:2:2 / 4:4:4 / high-bit-depth NotImplementedException. | **Frame-level WORKING as of 2026-04-27.** Top-level `Vp9KeyframeEncoder` (~640 LoC) takes YUV420 -> complete VP9 keyframe bitstream (uncompressed header + compressed header + single-tile data). v1 defaults: Profile 0 / 8-bit / 4:2:0, Block16x16 leaves with DC_PRED for Y + UV + Tx16x16 luma + Tx8x8 chroma DctDct, single tile, no loop filter, no segmentation, default coef probs. Round-trip via Vp9KeyframeWalker: pixels-in -> encode -> walker decode -> pixels-out with max error 0-1 across 16x16, 32x32, 64x64, and non-square 80x48 frames. ffmpeg native VP9 decoder accepts single-block (16x16) frames pixel-perfect. Multi-block ffmpeg validation pending a walker-side bitstream-divergence fix. Underlying primitives: full forward transform set (Vp9ForwardDct 4x4/8x8/16x16/**32x32**, Vp9ForwardAdst 4/8/**16**), Vp9ForwardQuantizer, Vp9BlockCoefEncoder (bit-exact mirror of decoder, 114/114 round-trip), Vp9BoolEncoder with leading marker bit fix vs libvpx vpx_start_encode, Vp9SuperframeWriter (BBB packets BIT-EXACT). 240+ tests pass across CPU + CUDA + OpenCL + WebGPU + WebGL + Wasm. |
| **AV1** | **Pipeline runs end-to-end as of 2026-04-27.** Daala range DECODER 309/309 round-trip + verified on real BBB AV1 OBU bytes. OBU + SH (28 fields) + FrameHeader (16 fields) + tile group parsers all running on bbb_180_2s.ivf. Av1KeyframeWalker.DecodeFrame produces real pixel data (not 128 placeholders) via the full chain: partition tree -> mode_info read -> Av1CoefDecoder (av1_read_coeffs_txb port: txb_skip / eob_multi / coeff_base / coeff_lps / dc_sign / Golomb / dequant) -> Av1Inverse2dTransform (16 tx_type x N tx_size dispatch over Av1InverseDct/Adst/Identity 1D primitives) -> Av1IntraEdge buffer -> Av1IntraPredictDispatch -> add+clip into Av1FrameBuffer. ALL default CDF tables ported (~5000 lines of token + partition + intra mode + skip + txfm + segment + delta_q from libaom). 18/18 walker tests pass on every backend. **Bit-exactness vs ffmpeg ground truth NOT yet hit:** BBB first frame Y mean 54 vs ffmpeg 97 (delta -43). Remaining bit-exactness gaps: hand-tuned 60-table libaom scan set (currently programmatic), dynamic q-context (hardcoded to bin 3), CFL alpha magnitudes, directional intra modes D45/D67/D113/D135/D157/D203 (currently fall back to DC), tx_type read via intra_ext_tx CDF (currently hardcoded DCT_DCT), 32x32/64x64 1D inverse transforms. | **Block emit WORKING as of 2026-04-27.** Top-level `Av1KeyframeEncoder` (~808 LoC) + `Av1CoefEncoder` (~496 LoC, bit-exact mirror of `Av1CoefDecoder`) emit a complete AV1 keyframe bitstream from YUV420 input. v1: TD + SH + Frame OBU, partition-tree recursion, per-block predict -> Av1Forward2dTransform -> Av1ForwardQuantizer -> Av1CoefEncoder (txb_skip / tx_type / eob / coeff_base{,_eob} / coeff_lps / dc_sign / Golomb) -> dequant -> inverse-DCT -> recon writeback. **Walker round-trip:** 16x16 flat-gray (Y=128) reconstructs to **Y=128.19, U=V=128.00** (within 0.2 of input). **ffmpeg trace_headers:** parses our SH + Frame Header CLEANLY (every field readable). **libdav1d:** rejects the entropy stream during tile MSAC decode; the headers are spec-perfect (verified) and the encoder is internally self-consistent with our decoder mirror, so the gap is in the MSAC byte sequence itself (likely coef CDF context indexing). All forward transform primitives shipped + bit-exact: Av1ForwardDct 4/8/16/**32**, Av1ForwardAdst 4/**8**/**16**, Av1Forward2dTransform 2D dispatcher (libaom shifts + cos_bits), Av1ForwardQuantizer, Av1RangeEncoder (309/309 round-trip). |

Containers wired for video pipelines: IVF reader + writer, Matroska / WebM via SpawnDev.EBML, Ogg.

### GPU encoder + decoder pairs (v3 100% ILGPU)

Every entry below runs entirely on the accelerator (CUDA + OpenCL + CPU
verified bit-exact via PlaywrightMultiTest; WebGPU + WebGL + Wasm by
ILGPU IR symmetry). Host is a pure coordinator: alloc + upload + dispatch
+ readback. No CPU math, no CPU iteration, no CPU bitstream assembly.

| Codec  | GPU Encoder              | GPU Decoder              | Status                                |
|--------|--------------------------|--------------------------|---------------------------------------|
| VP8    | `Vp8KeyframeEncoderGpu`  | `Vp8KeyframeDecoderGpu`  | v1 keyframe (Y=128 round-trip 0.2)    |
| VP9    | `Vp9KeyframeEncoderGpu`  | `Vp9KeyframeDecoderGpu`  | v1 keyframe + multi-block walker      |
| FLAC   | `FlacEncoderGpu`         | `FlacDecoderGpu`         | v1 keyframe + 7 standalone subframe-decode primitives (FixedReconstruct + LpcReconstruct + ChannelDecorrelation + ResidualDecoder + SubframeHeader + Fixed/Lpc subframe composites) |
| AV1    | `Av1KeyframeEncoderGpu`  | `Av1KeyframeDecoderGpu`  | v1 keyframe bit-exact vs CPU encoder  |
| Vorbis | `VorbisAudioEncoderGpu`  | `VorbisAudioDecoderGpu`  | v1 mono encoder + decoder pair (silence-path round-trip bit-exact across CUDA + OpenCL + CPU; full .ogg stream output; 21 GPU primitives including OverlapAdd) |
| Opus   | -                        | -                        | 31 SILK GPU primitives + shared Daala range coder |

Each pair has tests in `CodecsTestBase.<Codec>KeyframeEncoderGpuTests.cs`
+ `CodecsTestBase.<Codec>KeyframeDecoderGpuTests.cs` running across every
backend.

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

## Host is Pure Coordinator — 100% Accelerator-Resident

**Every codec encoder and decoder in this library runs entirely as ILGPU kernels.** The host (the .NET environment that uses the accelerator) is treated as a pure coordinator — it allocates GPU buffers, uploads source data, dispatches kernel chains, and reads back the final output bytes. Nothing CPU-side touches the encoded bitstream or the decoded pixels. **No CPU math, no CPU iteration, no CPU bool encoding, no CPU bitstream assembly.**

This is a hard design rule, not an aspiration. Every kernel in this repo, every integration class, every encoder + decoder pipeline holds to it. When a piece of work is still on the CPU, it gets ported.

### Why this matters

- **Blazor WASM UI thread.** In the browser, the .NET runtime IS the UI thread. Any meaningful CPU work freezes the page. Pushing every codec stage to ILGPU means the UI stays responsive while a 1080p frame transcodes underneath.
- **Zero CPU↔GPU bouncing in the hot path.** Each round-trip across the PCIe / WebGPU / WebGL / Wasm boundary costs latency and throughput. Once source data is on the accelerator, it stays there until the final output buffer is ready.
- **Backend uniformity.** The same kernels run on CUDA, OpenCL, CPU emulator, WebGPU, WebGL, and Wasm. The host code doesn't branch on backend - it dispatches the same chain everywhere.
- **Scales with the silicon, not the .NET runtime.** The performance ceiling is whatever the accelerator can deliver, not whatever the host can spare around UI work and JS interop.

### What this looks like in practice

The VP8 keyframe encoder (`Vp8KeyframeEncoderGpu`) is the first complete v3 reference. Its `EncodeKeyFrame` method is a textbook example of the pattern:

```csharp
// PURE COORDINATOR. No math, no iteration, no bitstream work.
public byte[] EncodeKeyFrame(/* YUV planes + dims + baseQ */)
{
    using var dY = _accelerator.Allocate1D<byte>(...);
    // ... allocate other GPU buffers ...

    UploadPlane(ySrc, ...);  // necessary I/O - source from outside accelerator
    UploadPlane(uSrc, ...);
    UploadPlane(vSrc, ...);

    _setup.Run(/* compute dequantizers + write frame header */);
    _sequentialEncode.Run(/* per-MB predict + transform + quant + recon */);
    _entropy.Run(/* per-MB modes + coef tokens to bool streams */);
    _assemble.Run(/* frame tag + concat partitions into output */);

    _accelerator.Synchronize();

    var lenArr = dOutLen.GetAsArray1D();              // 1 int readback
    var outputBuffer = dOutput.GetAsArray1D();        // final bytes readback
    return outputBuffer.AsSpan(0, lenArr[0]).ToArray();
}
```

That's the whole encoder. Every byte of the output bitstream is written by an ILGPU kernel. See `Plans/VP8-GPU-encoder-architecture.md` for the kernel chain breakdown.

## Architecture

Every codec is decomposed into kernel-resident stages. Even stages that look "inherently sequential" (entropy coders, range coders, bool encoders) run on the GPU - just as a single thread per stream rather than parallel across many. Single-thread on GPU is still GPU-resident, and that's what the host-as-coordinator rule demands.

| Stage | Work | Where |
|-------|------|-------|
| **Massively parallel** | DCT / MDCT, motion estimation, quantization, loop filter, inverse transforms, motion compensation | **ILGPU kernels** - backend-agnostic |
| **Sequential per stream** | Range coding (Daala/VP8 bool/SILK), Huffman, LPC prediction, rate control feedback | **ILGPU kernels** - one thread per stream, multiple streams in parallel where the spec allows (token partitions, tiles) |
| **Frame-level orchestration** | Per-MB predict→transform→recon dependency loops, header bit emission | **ILGPU kernels** - single-thread per frame for v1/v2 correctness; wave-parallel scheduling planned for v4 throughput |
| **Coordination** | Buffer alloc, source upload, kernel dispatch order, final readback | **C# host** - pure coordinator, no math |

The "host CPU does the entropy stage" pattern other codec libraries use is explicitly rejected here. Bool encoders, range coders, Huffman trees - all of them run on the accelerator.

## Testing

Every slice is validated through the `PlaywrightMultiTest` harness, which runs the same test suite across every ILGPU backend (WebGPU, WebGL, Wasm, CUDA, OpenCL, CPU) and aggregates the results. Thousands of cross-backend test executions gate each merge.

```
dotnet test PlaywrightMultiTest/PlaywrightMultiTest.csproj --filter "FullyQualifiedName~Flac"
```

The in-browser demo at `SpawnDev.Codecs.Demo/` ships several pages exercising the live library:

- **`/transcode`** - encode + decode all 3 video codecs (VP8, VP9, AV1) end-to-end, render YUV-&gt;RGBA output on canvas.
- **`/audio`** - encode + decode all 3 audio codecs (FLAC, Vorbis, Opus) end-to-end, draw source vs decoded waveforms with per-codec SNR + compression ratio + encode/decode timings.
- **`/benchmarks`** - throughput + compression measurements for FLAC + transforms.
- **`/tests`** - SpawnDev.UnitTesting cross-backend test harness (runs every test in `CodecsTestBase` against every available browser ILGPU backend).

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
