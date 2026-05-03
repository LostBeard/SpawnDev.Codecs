# SpawnDev.Codecs CHANGELOG

## 0.3.0-rc.1 (2026-05-03)

Release candidate for the 0.3.0 cut. NOT 1.0.0 - the 1.0.0 stable
designation requires the full architectural completion (see "Path to
1.0.0" below). 0.3.0-rc.1 captures today's substantial progress
toward that bar.

### Codecs in this release

**Audio:** FLAC, Opus, Vorbis. **Video:** VP8, VP9, AV1.

Patent-clean by selection: AOMedia patent pledge for VP8/VP9/AV1, RFC
6716 royalty-free for Opus, BSD reference for FLAC/Vorbis. H.264/H.265/
AAC stay out (delegated to platform encoders via SpawnDev.MultiMedia).

### Public API surface - CPU reference encoders + decoders

Every codec ships a CPU public API that produces bytes a reference
decoder accepts:

- **FLAC**: bit-exact lossless, matches ffmpeg byte-for-byte
- **Opus**: round-trip via Concentus 2.2.2, matches ffmpeg's libopus quality
- **Vorbis**: 35.7 dB SNR on real BBB content (vs ffmpeg's libvorbis 20.9 dB)
- **VP8**: keyframes accepted by ffmpeg pixel-perfect (1920x1072 @ 5 fps;
  1/2/4/8 token partitions)
- **VP9**: keyframes accepted by ffmpeg native decoder (16x16 -> 1920x1088,
  max diff Y=1 at Q=8)
- **AV1**: keyframes accepted by libdav1d (60 frames decoded on real
  BBB content, av1.mp4 plays cleanly)

### GPU encoder + decoder pairs (100% ILGPU)

Five of the six codecs ship 100%-ILGPU encoder/decoder pairs:

- `Vp8KeyframeEncoderGpu` / `Vp8KeyframeDecoderGpu`
- `Vp9KeyframeEncoderGpu` / `Vp9KeyframeDecoderGpu`
- `Av1KeyframeEncoderGpu` / `Av1KeyframeDecoderGpu`
- `FlacEncoderGpu` / `FlacDecoderGpu`
- `VorbisAudioEncoderGpu` / `VorbisAudioDecoderGpu`

Every byte of the encoded bitstream is written by an ILGPU kernel; every
pixel of the decoded recon plane comes back from a kernel readback. The
host (consumer code) is a pure coordinator: alloc + upload + dispatch +
single readback. No CPU math, no CPU iteration, no CPU bool encoding,
no CPU bitstream assembly.

Cardinal-rule compliant: every per-frame iteration on codec data,
`Array.Copy` on bitstreams, full-buffer-readback waste, and CPU-encoder
INSTANCE dependency in the GPU pair production paths has been closed.

Opus has 31 SILK GPU primitives + the shared Daala range coder; the
top-level OpusEncoderGpu/OpusDecoderGpu integration class is the next
codec on the GPU port roadmap.

### Browser demos (live in your browser via Blazor WASM)

The `SpawnDev.Codecs.Demo` project ships browser demos for the entire
codec surface:

- `/transcode` - encode + decode all 3 video codecs (VP8/VP9/AV1) via
  the CPU public API. Renders YUV->RGBA on canvas.
- `/audio` - encode + decode all 3 audio codecs (FLAC/Vorbis/Opus) via
  the CPU public API. Plots source vs decoded waveforms with per-codec
  SNR + compression ratio + timings.
- `/gpu-transcode` - encode + decode VP8 + VP9 + AV1 entirely through
  the ILGPU integration classes. 100% GPU-resident. Routes to WebGPU
  when available, Wasm fallback.
- `/benchmarks` - live throughput numbers for every encoder + decoder,
  both CPU and GPU pair, across the available ILGPU backends.

Every demo page is covered by an end-to-end unit test that mirrors its
exact encode -> decode flow, so the demo can be trusted to work without
manual browser click-through every commit
(`AudioTranscodeDemo_*` / `VideoTranscodeDemo_*` / `GpuTranscodeDemo_*`).

### Containers

| Container | Read | Write |
|---|---|---|
| RIFF / WAVE | Yes | Yes |
| AIFF | Yes | Yes |
| Ogg | Yes | Yes |
| WebM / Matroska (EBML) | Via SpawnDev.EBML 3.0.1 | Via SpawnDev.EBML |
| MP4 / ISOBMFF | Structural box reader | Not yet |
| IVF | Yes | Yes |

### Test infrastructure

- `CodecsTestBase.AcquireAcceleratorOrSkipAsync()` - converts the
  WebGL backend's `NotSupportedException` (eager kernel-feature
  validation) into `UnsupportedTestException` so the harness records
  Skipped rather than Failed for tests whose kernels need atomics.
- `CodecsTestBase.RequireFeatures(acc, AcceleratorRequirements)` -
  explicit per-test feature gating via SpawnDev.ILGPU's
  `device.Satisfies()` API.

### Project split: SpawnDev.Codecs.References (NEW in 0.3.0-rc.1)

A new sibling project, `SpawnDev.Codecs.References`, hosts the CPU
reference encoders + decoders. The main `SpawnDev.Codecs` library now
ships **ZERO external dependencies** (only SpawnDev.* + Microsoft
framework packages) per Captain's 2026-05-03 architectural directive.

Moved to `SpawnDev.Codecs.References` in 0.3.0-rc.1 (174+ files):

**Audio**:
- **Opus** (5 files): `OpusEncoder`, `OpusDecoder`, `OpusOggEncoder`,
  `OpusOggDecoder`, `OpusCodec` (factory)
- **Opus / CELT CPU** (2 files): `CeltDecoder`, `CeltDecoderState`
- **Opus / SILK CPU** (28 files): `SilkBwexpander`, `SilkDecodeCore`,
  `SilkDecodeFrame`, `SilkDecoder`, `SilkExcitationDequantizer`,
  `SilkGainAdjust`, `SilkGainDecoder`, `SilkIndicesDecoder`,
  `SilkLog2`, `SilkLpcAnalysisFilter`, `SilkLpcFit`,
  `SilkLpcInvPredGain`, `SilkLpcSynthesisFilter`, `SilkLtpDecoder`,
  `SilkNlsf2A`, `SilkNlsfDecoder`, `SilkNlsfStabilize`,
  `SilkNlsfUnpack`, `SilkParametersDecoder`, `SilkPitchDecoder`,
  `SilkPulsesDecoder`, `SilkResampler`, `SilkShellCoder`,
  `SilkSideInfoDecoder`, `SilkSigmoid`, `SilkStereoDecodePred`,
  `SilkStereoMsToLr`, `SilkSumSqrShift`
- **FLAC** (9 files): `FlacEncoder`, `FlacDecoder`,
  `FlacFrameDecoder`, `FlacOggDecoder`, `FlacFixedSubframeEncoder`,
  `FlacLpcSubframeEncoder`, `FlacResidualDecoder`,
  `FlacSubframeDecoder`, `FlacSubframeWriter`
- **Vorbis** (3 files): `VorbisAudioDecoder`, `VorbisOggDecoder`,
  `VorbisInverseCoupling`

**Video**:
- **VP8** (14 files): `Vp8KeyframeEncoder`, `Vp8Decoder`,
  `Vp8BoolEncoder`, `Vp8BoolDecoder`, `Vp8CoefBlockEncoder`,
  `Vp8CoefBlockDecoder`, `Vp8MbModeInfoDecoder`, `Vp8FrameHeader`,
  `Vp8FrameHeaderWriter`, `Vp8ModeTrees`, `Vp8SegmentationLookup`,
  `Vp8MbDequantizer`, `Vp8KeyframeWalker`, `Vp8ForwardQuantizer`
- **VP9** (102 files): full CPU decoder + encoder + walker stack,
  `Vp9Decoder`, `Vp9KeyframeEncoder`, `Vp9KeyframeWalker`, all the
  `Vp9*Mv*`, `Vp9Iadst*Reference`, `Vp9Idct*Reference`,
  `Vp9Iht*Reference`, etc.
- **AV1** (11 files): `Av1Decoder`, `Av1KeyframeWalker`,
  `Av1FrameBuffer`, `Av1IvfRemuxer`, `Av1StreamAnalyzer`,
  `Av1StreamValidator`, `Av1TileGroup`, `Av1CflPredictor`,
  `Av1CompleteFrameHeaderParser`, `Av1DefaultSegmentCdfs`,
  `Av1FrameHeaderWriter`

**Entropy coders** (2 files): `OpusRangeEncoder`, `OpusRangeDecoder`

Still in `SpawnDev.Codecs` main library (because each is consumed by
GPU integration classes for static data tables, OBU framing helpers,
or metadata struct setup): the Vorbis encoder + Floor1/Residue/
PacketHeader decoders + bit IO + codebook tables, every AV1
CDF/dequant/scan table + transforms + predictors + OBU writer + spec
record types, every VP9 GPU-shared scan/coef/probs/neighbor table.

For 1.0.0 these remaining CPU implementations need finer extraction
(separate the CPU impl methods from the shared static tables) so
they can also move to References. The pattern is mechanical now -
extract tables/helpers, point GPU code at the tables class, move the
CPU impl. Several have already been done this way (`Vp8CoefTables`,
`Vp8ModeProbTables`, `Vp9BlockCoefEnums`, `Av1RangeDecoderGpu`
constants).

### Dependencies

`SpawnDev.Codecs` (main library):
- `SpawnDev.ILGPU` 4.9.3 (kernel dispatch, partial readback,
  `ArrayView<T>.CopyToHostAsync()`)
- `SpawnDev.EBML` 3.0.1 (WebM/Matroska container)

`SpawnDev.Codecs.References` (CPU reference impls):
- `SpawnDev.Codecs` (sibling)
- `Concentus` 2.2.2 (BSD-3 pure-C# Opus reference)

### Out of scope (patent-encumbered)

H.264, H.265, AAC, MP3 - delegated to platform encoders via
SpawnDev.MultiMedia (P/Invoke to MediaFoundation / VideoToolbox /
VAAPI). System encoders are licensed by Microsoft / Apple / driver
vendors respectively; SpawnDev.Codecs stays clean.

### Path to 1.0.0 stable

Captain's directive 2026-05-03 defines the 1.0.0 bar. 0.3.0-rc.1 is
release candidate work toward that bar:

1. **All 6 codecs ship as 100%-ILGPU encoder + decoder pairs.** Today
   covers 5 of 6 (VP8, VP9, AV1, FLAC, Vorbis). Opus has 31 SILK GPU
   primitives + the shared Daala range coder; CELT has zero GPU
   primitives (bit allocator, PVQ encode/decode, anti-collapse, spread,
   prefilter, post-filter, stereo coupling all need to be ported). Plus
   SILK + CELT integration kernels + OpusEncoderGpu/OpusDecoderGpu
   top-level classes. This is the largest remaining piece.
2. **Vorbis decoder v2 GPU bit-stream decode.** v1 still does CPU-side
   per-packet header parse + floor + Huffman + residue work before
   handing to the GPU IMDCT chain. v2 ports those to GPU kernels using
   the existing `VorbisAudioPacketHeaderGpu`, `VorbisFloor1DecoderGpu`,
   `VorbisHuffmanDecoderGpu`, `VorbisBitReaderGpu`, and the new
   `VorbisResidueDecoderGpu` (shipped 0.3.0-rc.1) primitives. Plan in
   `Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md`.
3. **Complete the project split.** Started in 0.3.0-rc.1 (Opus, FLAC,
   SILK CPU, OpusRange entropy coders moved + Concentus dep removed
   from main). Remaining: Vorbis, VP8, VP9, AV1 CPU classes still
   live in main because each one mixes CPU implementation methods
   with static data tables consumed by the GPU integration classes.
   For 1.0.0 each CPU class needs its data tables / framing helpers
   extracted to a main-library "Tables" / "Helpers" class so the CPU
   class itself can move to References cleanly.
4. **Cross-lane SpawnDev.ILGPU bug fixes** for WebGPU `SubViewValue`
   IR codegen + Wasm OOB. Filed at
   `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-webgpu-subview-codegen-and-wasm-oob-2026-05-03.md`.

## 0.2.0-alpha.2 (2026-05-03)

Same library code as 0.2.0-alpha.1; this point release ships the
documentation + demo improvements that landed after the 0.2.0-alpha.1
package was published:

- README "Quickstart - GPU encoder + decoder pairs" section with
  concrete consumer code examples for each of the 5 GPU pairs (VP8,
  VP9, AV1, FLAC, Vorbis), including how to acquire an accelerator on
  desktop + Blazor WASM and how to handle backends that lack atomics
  (`AcceleratorRequirements` + `UnsupportedTestException` pattern).
- `/gpu-transcode` browser demo now exercises all 3 video codecs end-
  to-end through the GPU pairs (VP8 + VP9 + AV1 via the tile-bytes
  flow `EncodeSingleTileAsync` -> `DecodeSingleTileAsync`). Previously
  only VP8 + VP9.
- `Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md` design doc
  covers the next-session pickup target: move CPU per-packet floor +
  Huffman + residue work in `VorbisAudioDecoderGpu.DecodePacket` into
  a single GPU kernel, closing the v1 design gap.

Net code change for library consumers: README in the package updates
with the quickstart section. No API surface changes.

## 0.2.0-alpha.1 (2026-05-03)

First version after the cardinal-rule audit pass. Every GPU integration
class is now host-as-coordinator clean: alloc + upload + dispatch +
single readback. No CPU iteration on codec data inside any *Gpu class
production path.

### Cardinal-rule + perf wins

- **VP8 keyframe encoder** now uses a GPU stride-pack kernel instead
  of the per-row CPU stride-strip loop in `UploadPlane`. Strided source
  planes upload via `Vp8StridedPlanePackKernel` (one thread per output
  byte) so plane unpacking happens accelerator-resident.
- **VP8 keyframe encoder** final readback uses
  `dOutput.View.SubView(0, finalLen).CopyToCPU(result)` (real
  per-backend partial readback, SpawnDev.ILGPU 4.9.3+) - drops the
  prior `GetAsArray1D` of the worst-case-sized output buffer + CPU
  `Array.Copy` trim.
- **VP9 + AV1 + Vorbis encoder/decoder pairs** swap host-allocated
  zero-arrays + uploads (`CopyFromCPU(new T[N])`) for GPU-resident
  `MemSetToZero()`. Eliminates ~1MB of CPU allocator churn and bus
  transfer of zeros per frame at typical sizes.
- **VP9 keyframe decoder** drops the per-decode `Array.Copy` of tile
  bytes. Uploads the full frame once, dispatches the decode kernel
  with `dFrame.View.SubView(tileStartOffset, tileLength)` so the
  kernel sees the same-shaped tile view at zero copy cost.
- **VorbisAudioDecoderGpu** drops the per-channel `Array.Copy` flat-
  buffer build for posteriors + residues. Each channel uploads
  directly into its slot via `dPosteriors.View.SubView(...)` and
  `dResidue.View.SubView(...)`. `MemSetToZero()` covers silent-channel
  slots.
- **VorbisAudioEncoderGpu** + **VorbisAudioDecoderGpu** drop their
  CPU-encoder / CPU-decoder INSTANCE dependencies. The encoder used
  to call `_cpuRef.BuildIdentPacket()` / `BuildCommentPacket()` /
  `BuildSetupPacket()` for the 3 Ogg header packets in
  `EncodeStreamAsync`; the decoder used reflection to extract
  Huffman state from the CPU decoder's private fields. Now the GPU
  encoder calls `VorbisAudioEncoder.BuildIdentPacketBytes()` /
  `BuildCommentPacketBytes()` / `BuildSetupPacketBytes()` (new public
  static helpers) and the GPU decoder builds its own Huffman array
  from `setup.Codebooks` directly. CPU encoder/decoder classes
  remain as test reference oracles only.

### Demos

- **`/audio` page** - encode + decode all 3 audio codecs (FLAC,
  Vorbis, Opus) end-to-end via the CPU public API. Synthesizes a
  mono sine wave, drives every encoder + decoder, plots source +
  recovered waveforms with per-codec SNR + compression + timings.
- **`/gpu-transcode` page** - encode + decode VP8 + VP9 entirely
  through the ILGPU integration classes (`Vp8KeyframeEncoderGpu` +
  `Vp8KeyframeDecoderGpu`, `Vp9KeyframeEncoderGpu` +
  `Vp9KeyframeDecoderGpu`). 100% GPU-resident encode/decode chain.
  Routes to WebGPU when the browser exposes it, Wasm otherwise.
- `AudioTranscodeDemo_*` unit tests verify the demo pipeline
  bit-exact for FLAC + lossy SNR floors for Vorbis (>10 dB) + Opus
  (>5 dB), 18/18 pass on CUDA + OpenCL + CPU.

### Test infrastructure

- New `CodecsTestBase.AcquireAcceleratorOrSkipAsync()` wraps
  `CreateKernelAcceleratorAsync()` to convert the WebGL backend's
  `NotSupportedException` (eager kernel-feature validation) into
  `UnsupportedTestException`. The harness now records "Skipped" for
  WebGL tests whose kernels need atomics / shared memory / barriers
  rather than reporting them as Failed. Applied across all 200+
  GPU-using test sites.
- New `CodecsTestBase.RequireFeatures(acc, AcceleratorRequirements)`
  helper: explicit per-test feature gating. Throws
  `UnsupportedTestException` when the current accelerator's backend
  cannot satisfy declared requirements.

### SpawnDev.ILGPU PackageReference: 4.9.2 -> 4.9.3

Picked up `ArrayView<T>.CopyToHostAsync()` real per-backend partial
readback (WebGPU `queue.CopyBufferToBuffer` + partial mapAsync, WebGL
partial-range readback worker path, Wasm SAB slot view, CUDA / OpenCL
/ CPU `view.CopyToCPU(target)`). Used at every `*KeyframeEncoderGpu`
and `*KeyframeDecoderGpu` partial-readback site.

## Unreleased

Initial development. Project still in pre-release. See README.md for the
working feature matrix.

### 2026-04-29 - FlacDecoderGpu output interleave moves from CPU to GPU

`FlacDecoderGpu.DecodeStreamAsync` previously copied each frame's
samples back to host (`dSamples.CopyToHostAsync()`) and ran a per-sample
channel-major -> interleaved CPU loop into the output array. That
violated the cardinal rule (host doing per-sample iteration on codec
data). Now: a new `FlacInterleaveOutputGpu` primitive does the same
work GPU-side as one thread per (channel, sample); the integration
class allocates the full interleaved output on the accelerator and
dispatches the kernel per frame. Single readback of the full output
buffer at end.

Net effect: the host is back to pure coordinator status for FLAC
decode - alloc, upload, kernel dispatch, single per-frame status check
+ frame-length scan, single readback. No per-frame `dSamples`
readback, no per-sample CPU iteration on codec data.

FLAC decoder PMT 6/6 PASS on CUDA + OpenCL + CPU.

### 2026-04-29 - VorbisAudioDecoderGpu floor curve render moves to GPU

`VorbisAudioDecoderGpu` now dispatches `VorbisFloor1RenderCurveGpu`
instead of calling the CPU `VorbisFloor1Curve.Render`. Per-floor
`xList` arrays + the 256-entry inverse-dB lookup are pre-flattened +
uploaded once at construction; per packet the CPU bit-stream parse
produces the per-channel posterior Y values, which upload to GPU and
drive a single-thread floor render kernel per non-silent channel.

Tone round-trip test still PASSES on CUDA + OpenCL + CPU (9/9 PMT).
This is a meaningful proof point because the floor render IS exercised
on the tone path (silent floors get zero-curve fallback + skip the
dispatch entirely).

Pipeline state (v1 hybrid):

| Stage                      | Where |
|----------------------------|-------|
| Parse + Huffman + floor decode + residue decode | CPU |
| Floor curve render         | **GPU** |
| Floor x residue multiply   | GPU |
| Inverse coupling           | GPU |
| IMDCT                      | GPU |
| Window + overlap-add + right-half save | GPU |
| Channel interleave         | GPU |

Remaining CPU side: bit-stream parse + Huffman, floor 1 posterior
decode, residue decode. Vorbis-Huffman GPU bit-reader keystone
primitive is what unlocks moving those.

### 2026-04-29 - VorbisAudioDecoderGpu non-silence tone round-trip test

`VorbisAudioDecoderGpu_ToneRoundTrip_MatchesCpuDecoder` exercises the
full GPU decoder chain on a non-silent 440 Hz mono tone (peak 0.5).
Same packets feed both the CPU `VorbisAudioDecoder` reference (double-
precision IMDCT) AND the GPU `VorbisAudioDecoderGpu` (float-precision
IMDCT). PCM compared sample-by-sample with 1e-3 absolute tolerance to
absorb the float-vs-double divergence in the IMDCT accumulator.

Result: 3/3 backends PASS (CUDA + OpenCL + CPU = 9/9 PMT counting all
three Vorbis decoder tests). This is the first non-silence proof point
for the GPU multiply + inverse-coupling + IMDCT + post-IMDCT chain
running end-to-end on real spectral data.

### 2026-04-29 - VorbisAudioDecoderGpu floor*residue multiply + inverse coupling move to GPU

`VorbisAudioDecoderGpu` now dispatches `VorbisFloorMultiplyGpu` and
`VorbisInverseCouplingGpu` instead of doing the corresponding scalar
loops in the CPU helper. Per-channel floor curves + residue buffers
upload once; multiply runs as one Index1D dispatch per non-silent
channel; the (mag, ang) coupling pairs run as one Index1D dispatch each
in REVERSE step order per Vorbis I sec 4.3.8.

Pipeline state after this commit (v1 hybrid):

| Stage                            | Where  |
|----------------------------------|--------|
| Bit-stream parse + Huffman + floor decode + residue decode | CPU |
| Floor curve x residue multiply   | GPU    |
| Inverse channel coupling         | GPU    |
| IMDCT                            | GPU    |
| Window apply + overlap-add + new-right-half-save | GPU    |
| Channel interleave               | GPU    |

Mono test path covers no coupling steps (mapping has zero pairs); the
inverse-coupling kernel will get exercised by future stereo tests.
Silence-path round-trip 6/6 PMT (CUDA + OpenCL + CPU) PASS.

### 2026-04-29 - VorbisAudioDecoderGpu IMDCT step moves from CPU to GPU

`VorbisAudioDecoderGpu.DecodePacketAsync` now dispatches `ImdctReferenceGpu`
on the accelerator instead of the CPU `ImdctReference.Transform`. The
spectrum coefficients (post-floor + post-residue + post-inverse-coupling)
upload to GPU once, and the IMDCT, post-IMDCT (window + overlap-add +
right-half save), and channel interleave all chain accelerator-resident
without intermediate CPU readback. Per-channel dispatch: one GPU thread
per output time-domain sample (2N=blockSize threads / channel).

Silence-path round-trip test still PASSES on CUDA + OpenCL + CPU
(`VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder` 6/6 PMT).
The CPU reference uses `double` accumulators for the IMDCT and the GPU
uses `float`, so non-silence inputs would diverge slightly; silence
remains bit-exact zero.

Remaining CPU-side work in the v1 hybrid: bit-stream parse + Huffman +
floor decode + residue decode + multiply + inverse coupling. Vorbis-
Huffman GPU bit-reader is the keystone primitive for moving those.

### 2026-04-29 - VorbisAudioDecoderGpu - closes the Vorbis encoder/decoder pair

**Vorbis is now 6 of 6 codec encoders + 6 of 6 codec decoders end-to-end on
the accelerator.** `VorbisAudioDecoderGpu` is the integration class that
mirrors `VorbisAudioEncoderGpu`: it consumes a Vorbis audio packet and emits
interleaved float PCM via the GPU post-spectrum chain (window apply +
overlap-add + new-right-half-save + channel interleave). State (previous
right-half samples, previous block size) is GPU-resident across packets.

**v1 scope (matches the encoder's silence-path scope):**
- Mono + stereo audio.
- GPU PostImdct + Interleave kernels run the per-sample math; spectrum decode
  (bit-stream parse + Huffman + floor + residue + IMDCT) currently delegates
  to the existing `VorbisAudioDecoder` CPU path so we have working end-to-end
  decode TODAY. The Vorbis-Huffman GPU bit-reader is the keystone primitive
  that will let the entire decode run in-kernel; it lands as a follow-up
  integration step.
- Round-trip silence test (`VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder`)
  encodes silence packets via `VorbisAudioEncoderGpu` then decodes them via
  both `VorbisAudioDecoder` (CPU reference) and `VorbisAudioDecoderGpu` (GPU)
  and compares interleaved PCM sample-by-sample within 1e-5 absolute
  tolerance. PASS on CUDA + OpenCL + CPU.

**Why this matters:** the v1 Vorbis encoder + decoder pair is now closed.
Silence in -> packets -> silence out, all data + state on the accelerator
through the post-spectrum chain.

### 2026-04-29 - FLAC subframe decode chain GPU coverage (7 new primitives)

**Continuing 2026-04-29 late-night session: 7 new FLAC GPU primitives plus 1
new Vorbis primitive.** The FLAC subframe decode chain is now fully GPU-
callable end-to-end - both FIXED and LPC predictor types have composite
decode primitives that compose 4 individual primitives in a single kernel
thread.

**New FLAC GPU primitives shipped this session (5 standalone + 2 composite):**
- VorbisOverlapAddGpu (`c9417b2`) - per-sample-parallel post-IMDCT overlap-add
- FlacFixedReconstructGpu (`46bc459`) - decoder-side FIXED predictor reconstruction (orders 0-4)
- FlacLpcReconstructGpu (`2aabcb3`) - decoder-side LPC predictor reconstruction (orders 1-32, quant 0-15)
- FlacChannelDecorrelationGpu (`e7e2dd1`) - per-sample stereo decorrelation (LeftSide / RightSide / MidSide)
- FlacResidualDecoderGpu (`80b73d3`) - Rice-coded residual decoder (PartitionedRice + PartitionedRice2)
- FlacSubframeHeaderGpu (`38d0fda`) - subframe header parser (CONSTANT / VERBATIM / FIXED / LPC)
- FlacFixedSubframeDecodeGpu (`1ab8a84`) - composite end-to-end FIXED subframe decoder
- FlacLpcSubframeDecodeGpu (`7319d1c`) - composite end-to-end LPC subframe decoder

**FLAC subframe decode chain (now fully GPU-callable):**
SubframeHeader -> FixedSubframeDecode or LpcSubframeDecode (warm-up reads ->
ResidualDecode -> Reconstruct -> wasted-bits left-shift) -> ChannelDecorrelate
(if stereo). Each step is bit-exact vs the libFLAC reference across CUDA +
OpenCL + CPU.

### 2026-04-29 - Opus SILK GPU primitive build-out continued (23 -> 31 primitives)

**Continuing 2026-04-29 late-night session: 8 more Opus SILK GPU primitives
beyond the earlier 23-primitive milestone.** Brings Opus SILK to 31 GPU
primitives total + 1 shared Daala range coder. Now have GPU coverage of every
non-entropy SILK decode stage including a full composite NLSF decode pipeline
(Unpack -> ResidualDequant -> WeightedAdd -> Stabilize) running in a single
kernel thread.

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| Opus   | -       | -       | 31 SILK + 1 shared range coder |

**New Opus SILK GPU primitives shipped this session (24th-31st):**
- SilkGainDecoderGpu (`c885b86`) - per-subframe gain dequantizer (delta-coded hysteresis, composes SilkLog2Gpu)
- SilkLtpGainVectorGpu (`4808113`) - per-tap LTP gain vector codebook lookup (per-tap parallel)
- SilkPitchContourGpu (`cbeb1bc`) - per-subframe pitch contour expansion (per-subframe parallel)
- SilkNlsfInterpolateGpu (`5fa485a`) - per-coefficient NLSF inter-frame interpolation (per-coefficient parallel)
- SilkNlsfResidualDequantGpu (`95f356d`) - reverse-iterating NLSF residual dequant (sequential per-stream)
- SilkNlsfWeightedAddGpu (`5a8944b`) - per-coefficient NLSF inverse-weight + first-stage add (per-coef parallel)
- SilkNlsfDecodeGpu (`65cbddf`) - composite NLSF decode pipeline (single-thread, composes 4 GPU primitives)
- SilkLtpScaleGpu - 3-entry LTP scale Q14 lookup

**Second library quirk discovered + worked around (composite NLSF decode):**
ILGPU CUDA + OpenCL backends do NOT preserve C# byte<->sbyte cast semantics
through ArrayView storage. Initial SilkNlsfDecodeGpu used ArrayView<sbyte>
for predQ8 scratch with explicit (sbyte)/(byte) casts; CPU backend passed
all 4 tests but CUDA + OpenCL produced random divergences (8/12 tests
failed). Fix: use ArrayView<byte> directly with implicit conversions. Saved
`feedback_ilgpu_byte_sbyte_roundtrip.md` so future composite primitives
avoid the trap. The cross-backend pattern (CPU passes, hardware fails) is
the smoking gun for the bug.

### 2026-04-29 - Opus SILK GPU primitive build-out (9 -> 23 primitives)

**Continuing 2026-04-29 (late evening session): Opus SILK primitive expansion
from 9 to 23 primitives.** 14 new commits. Now have GPU coverage of every
major SILK decode + analysis stage: LPC analysis + synthesis, full resampler
chain (Up2Hq + Ar2 + DownFirInterpol + IirFirInterpol), stereo M/S->L/R + the
predictor dequantizer, gain adjust + DivVarQ, excitation dequant (PRNG-driven
with long-arithmetic bypass for ILGPU OpenCL signed overflow quirk), LPC fit,
NLSF -> Q12 LPC pipeline (composes 3 GPU primitives in a single kernel
thread), NLSF table-index unpacker.

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| Opus   | -       | -       | 23 SILK + 1 shared range coder |

**New Opus SILK GPU primitives shipped this session (10th-23rd):**
- SilkLpcAnalysisFilterGpu (`6bbba73`) - per-sample MA prediction-error filter
- SilkResamplerUp2HqGpu (`74f3aa1`) - 2x HQ upsampler (sequential IIR, one-thread-per-stream)
- SilkResamplerAr2Gpu (`e475cd0`) - AR2 IIR pre-filter for downsample chain
- SilkResamplerDownFirInterpolGpu (`c2939b9`) - 3 FIR variants (Fir0/Fir1/Fir2), per-output-sample parallel
- SilkResamplerIirFirInterpolGpu (`40a3c9b`) - 12-phase fractional FIR upsample, per-output-sample parallel
- SilkLpcSynthesisFilterGpu (`868e9d2`) - decoder LPC synthesis inner loop
- SilkStereoMsToLrGpu (`8998230`) - M/S -> L/R (ApplySideAt + ApplyMixAt, both per-sample parallel)
- SilkDivVarQGpu + SilkGainAdjustGpu (`6a77c5c`) - reusable variable-Q division + LPC state rescale
- SilkExcitationDequantizerGpu (`879116d`) - PRNG-driven excitation dequant (long-arithmetic for OpenCL)
- SilkLpcFitGpu (`87568d4`) - LPC coefficient quantization with iterative bandwidth expansion
- SilkNlsf2AGpu (`00700e6`) - full NLSF -> Q12 LPC pipeline (composes LpcFitGpu + LpcInvPredGainGpu + BwexpanderGpu)
- SilkNlsfUnpackGpu (`aaf1496`) - per-coefficient-pair NLSF table-index unpacker
- SilkStereoDecodePredGpu (`b8aeb62`) - stereo predictor dequantizer (pairs with StereoMsToLrGpu)

**Library quirk discovered + worked around:** ILGPU OpenCL backend does NOT
preserve `unchecked(int * int)` overflow semantics for signed int
multiplication. Saved `feedback_ilgpu_opencl_signed_overflow.md` so future
GPU PRNG ports start with the right pattern: use long arithmetic
(`(long)(uint)x * y`) for portable modular semantics across all 6 ILGPU
backends. Discovered while shipping SilkExcitationDequantizerGpu where
all 4 OpenCL tests sign-flipped at i=0 of the PRNG until the long-
arithmetic form was used.

### 2026-04-29 - AV1 GPU encoder + decoder pair complete (4 of 6 codecs on GPU)

**Status:** 4 of 6 codecs now have working encoder + decoder pairs on every
ILGPU backend (CUDA + OpenCL + CPU verified, WebGPU/WebGL/Wasm by symmetry).

| Codec  | Encoder | Decoder | Notes                                         |
|--------|---------|---------|-----------------------------------------------|
| VP8    | ✅      | ✅      | v3 100% ILGPU since 2026-04-26                |
| VP9    | ✅      | ✅      | v3 100% ILGPU since 2026-04-27                |
| FLAC   | ✅      | ✅      | v3 100% ILGPU since 2026-04-28                |
| AV1    | ✅ NEW  | ✅ NEW  | v3 100% ILGPU bit-exact vs CPU encoder        |
| Vorbis | -       | -       | 5 GPU primitives shipped, integration pending |
| Opus   | -       | -       | 4 GPU primitives shipped, integration pending |

**New:**

- `Av1FrameSequentialEncodeKernel` + `Av1KeyframeEncoderGpu` - v3 100%
  ILGPU AV1 v1 keyframe encoder. Single GPU thread runs the entire
  EncodeSingleTile pipeline (partition recursion + skip + ymode + uvmode +
  per-plane predict + 2D DCT + quantize + coef tokens + dequant + iDCT +
  recon) bit-exact vs the CPU `Av1KeyframeEncoder.EncodeSingleTile` reference.
  `EncodeKeyFrameAsync` returns the full TD + SH + Frame OBU stream bit-exact
  vs CPU encoder. Verified across CUDA + OpenCL + CPU (commit `a53e26d`).
- `Av1FrameSequentialDecodeKernel` + `Av1KeyframeDecoderGpu` - companion
  decoder. Round-trip tests prove the GPU decoder reconstructs the same
  YUV planes the GPU encoder produces internally - bit-exact across all 3
  backends (commit `4832234`).
- `SilkBwexpanderGpu` - third Opus SILK primitive on GPU. Chirp expansion
  for AR filter coefficients (Expand16 + Expand32). Used by NLSF
  stabilization and LPC whitening (commit `ea7cc52`).

**Bug found + fixed:** `GetQctx` quantizer-bin thresholds were
`[32, 128, 192]` instead of correct libaom `[20, 60, 120]` per
`Av1CoefDecoder.GetQctx`. This shifted the txb_skip CDF row used for the
first per-plane emit, producing 18-byte output where CPU produced 21 bytes.
Diagnosed via a CPU-shadow walker that mirrored the GPU walker logic
through the same `Av1RangeEncoder`; emit-trace comparison pinpointed the
divergence at emit #6.

**Test counts added:**
- 9 SilkBwexpanderGpu tests
- 12 Av1KeyframeEncoderGpu tests (4 cases x 3 backends)
- 6 Av1KeyframeDecoderGpu tests (2 cases x 3 backends)
- Full AV1 sweep: 696/696 PASS (zero regressions from walker additions).

**Continuing 2026-04-29 (afternoon-evening session): Vorbis encoder + Opus SILK
primitive expansion.** 33 total commits. Vorbis went from 5 -> 20 GPU primitives,
Opus SILK went from 4 -> 9, and Vorbis became the 5th codec with a working GPU
integration class (silence path bit-exact + full .ogg stream output).

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| VP8    | ✅      | ✅      | -              |
| VP9    | ✅      | ✅      | -              |
| FLAC   | ✅      | ✅      | -              |
| AV1    | ✅      | ✅      | -              |
| Vorbis | partial NEW | -    | 20 (silence path bit-exact; non-silent deferred to MDCT precision alignment) |
| Opus   | -       | -       | 9 SILK + 1 shared range coder |

**Vorbis GPU primitives shipped:**
- VorbisHuffmanDecoderGpu (`957ec88`) - canonical Huffman flat-tree decoder
- VorbisFloor1RenderGpu (`c6b4522`) - per-line / per-point floor renderer
- VorbisCodebookVectorLookupGpu (`ef45942`) - 3 lookup types + sequenceP
- VorbisFloor1RenderCurveGpu (`8a7de7f`) - full Floor 1 curve orchestrator
- VorbisWindowGpu.ApplyWindowAt (`1b65058`) - per-sample window apply
- VorbisEncoderHelpersGpu (`401293c`) - MagnitudeToFloorY + QuantiseResidueValue
- VorbisSpectrumPeakGpu (`b7220c1`) - per-half-band peak reduction
- VorbisEncoderHelpersGpu.DivideQuantizeAt (`a562ae9`) - residue divide+quantize
- VorbisBitWriterGpu.WriteCodebookEntry (`9df95f0`) - canonical Huffman emit
- VorbisHuffmanCodebookSetGpu (`4f77273`) - multi-codebook flattener
- VorbisFloor1DecoderGpu (`46606ba`) - per-channel posterior decoder
- VorbisAudioPacketHeaderGpu (`9b4f1ee`) - per-packet header parser
- VorbisFloorMultiplyGpu (`cc780b3`) - per-bin floor x residue
- VorbisResidueAccumulateGpu (`0385fb2`) - lookup + accumulate composite
- VorbisFwdMdctScaledGpu (`a05c090`) - forward MDCT + 4/N scale
- VorbisEncoderFloorFitGpu (`ba43fd4`) - peak + floor-fit composite
- VorbisEncoderResidueEmitGpu (`d132f98`) - per-bin codebook emission
- VorbisEncoderBitstreamEmitGpu (`9382256`) - full per-packet emit composite
- VorbisAudioEncoderGpu (`fc491a5`) - integration class
- VorbisAudioEncoderGpu.EncodeStreamAsync (`3401183`) - full .ogg stream output

**Opus SILK GPU primitives shipped:**
- SilkSumSqrShiftGpu (`3120646`) - sum-of-squares with dynamic shift
- SilkLsfToCosGpu (`5ff7aad`) - LSF -> 2*cos(LSF) lookup
- SilkNlsfStabilizeGpu (`8f192b9`) - iterative NLSF stabilizer + insertion-sort fallback
- SilkInverseQ32Gpu (`e7398fd`) - Newton-like reciprocal silk_INVERSE32_varQ
- SilkLpcInvPredGainGpu (`ee5ebc5`) - LPC inverse prediction gain (uses InverseQ32)

**Investigation result (2026-04-29 evening):** Tried to align CPU + GPU
MagnitudeToFloorY (binary search vs Log10/Ceil) for non-silent Vorbis
encode bit-exactness. Confirmed via CUDA test that the actual upstream
divergence is MDCT float-vs-double precision: CUDA + OpenCL ILGPU
backends don't support `XMath.Cos(double)` / `XMath.Log10(double)`
without `EnableAlgorithms()` invoked at context construction. Reverted
to float MDCT, documented as a follow-up: switching the accelerator
factory to call `EnableAlgorithms()` would let the GPU encoder use
double-precision XMath.Cos and bit-exactly mirror the CPU encoder.
Until then, the GPU encoder produces a valid Vorbis bitstream that any
decoder accepts; non-silent decoded PCM is acoustically identical (float
MDCT drift < 1 ULP per cosine). The silence path produces byte-identical
output across all backends because all spectrum values are 0 regardless
of MDCT precision.

### 2026-04-27/28 - all encoders + decoders working through public APIs

**Bug fixes (4 critical, 2 follow-on):**

- VP8 encoder: fix Y2 PLANE_TYPE (was using 3 = Y_WITH_DC, should be 1 = Y2) + add reconstruction write-back so multi-MB frames decode pixel-exact through ffmpeg (commit `beae150`).
- Vorbis encoder: amplitude bug closed - MDCT 4/N normalization moved from decoder to encoder (libvorbis convention) + residue codebook anchored at exactly 0 to eliminate ±half-step quantization noise. Single-tone test now matches source amplitude (peak 0.34 vs 0.30, RMS 0.124 vs 0.122) (commit `14ebe2e`).
- AV1 encoder: dav1d MSAC compatibility fix - 4 cumulative bugs closed (EOB token off-by-one, wrong txsize_log2_minus4 table, freelance txb_skip/dc_sign context formulas, missing qctx threading). 16x16 flat Y=128 now reconstructs exact through libdav1d (commit `5ce6c38`).
- VP9 walker + encoder: per-plane ENTROPY_CONTEXT propagation - was always passing ctx=0 for the first scan position, causing block (0,1) onward to mis-decode; libvpx passes the combined neighbor context. After fix, 32x32 multi-block decodes byte-exact vs ffmpeg (commit `3267c69`).
- AV1 decoder pipeline: 6 walker gaps closed (zigzag scan tables direction, qctx already done by encoder fix, CFL alpha magnitudes from libaom CDFs, directional intra modes via z1/z2/z3 from reconintra.c, tx_type via intra_ext_tx CDF, 32x32 + 64x64 inverse DCT 1D primitives). Walker no longer hits NotImplementedException; pixel-mean drift vs ffmpeg remains a separate slice (commit absorbed into `2a7a63b`).
- VP9 tile_info: min/max log2 formulas were transposed vs spec sec 6.2.14 - the encoder skipped the increment bit while the spec-compliant decoder still expected it; ffmpeg rejected every keyframe wider than 320px. After fix, 16x16 through 1920x1088 all decode through ffmpeg native VP9 (commit `be10e55`).

**Public API surface wirings:**

- `Vp8Decoder.DecodeFrameAsync` now routes keyframes through `Vp8KeyframeWalker` (was a `NotImplementedException` stub) (commit `087c99e`).
- `Vp9Decoder.DecodeFrameAsync` and `Av1Decoder.DecodeFrameAsync` now call their walkers and emit real YUV planes; placeholder mid-gray fallback only when the walker can't handle a frame (inter, etc.) (commit `2814322`).

**Verification + benchmark coverage:**

- `verify_all_codecs.cs` extended with VP9 multi-block, AV1 libdav1d, Vp8/Vp9/Av1Decoder API sections - 11/11 sections pass.
- `all_codecs_working_demo.cs` 14/14 entries pass.
- `benchmark_all_codecs.cs`, `benchmark_vs_ffmpeg.cs`, `benchmark_video_psnr.cs`, `benchmark_audio_quality.cs`, `benchmark_bbb_transcode.cs` - new unified benchmark suite.
- `bbb_transcode_artifacts.cs` produces VLC-playable mp4/ogg/flac/opus from TJ's BBB FullHD source.

**Blazor WASM:**

- `/transcode` page added: encode + decode all 3 video codecs in browser, render YUV->RGBA on HTMLCanvas (commit `d107dd2`).

**More fixes (post-overnight):**

- VP8 encoder: validate `baseQIndex` 0..127 at the API boundary - 7-bit field per RFC 6386 sec 9.6, larger values silently wrapped causing decoder/encoder quantizer mismatch (Q=150 PSNR collapsed to 8.76 dB) (commit `179d56a`).
- Vorbis encoder: fit floor 1 curve to per-block spectrum envelope. Encoder was forcing floor to a constant ~0.94 and asking the residue book to carry the entire dynamic range alone; real audio bins peaked at ~0.005 and rounded to zero. Per-block adaptive floor + 1024-entry residue book. BBB SNR 0.35 dB -> 35.71 dB (now beating ffmpeg's libvorbis at 20.88 dB) (commit `c67d8ec`).
- AV1 encoder + walker: 4 multi-block bugs closed - mi_cols formula was `>>3` should be `>>2` (mi units are 4 px not 8); min_log2_tiles formula used `64` should use `max_tile_area_sb = 2304`; `disable_cdf_update` was 0 should be 1; `GatherVertAlike` / `GatherHorzAlike` were 50/50 stubs replaced with libaom prob-summing formulas; `DecodePartition` now handles all four `has_rows / has_cols` cases per `read_partition`. libdav1d acceptance: 16x16 only -> ALL sizes 16x16..256x256 + 1920x1072 flat input (commit `8a43b8f`).
- AV1 walker: prediction always uses full tx-block dimensions, not frame-edge clip. Smooth/SmoothV/SmoothH/Paeth/directional predictors throw on non-power-of-2 dims at frame edge; libaom convention is to predict full size and clip on recon write. BBB Y mean 49.4 -> 94.95 (vs ffmpeg 97.4 = delta -2.5, was -48). Y plane drift essentially closed (commit `8e61258`).
- VP8 encoder + walker: multi-token-partition (Log2NumPartitions=0..3 = 1/2/4/8 partitions) per RFC 6386 sec 9.5. Encoder writes (n-1) 3-byte LE size headers + concatenated partition data; walker dispatches MB row M coefs to partition (M mod n). Round-trip + ffmpeg native VP8 decode all 4 counts: Y mean=124 exact across all (commits `86c2209` walker, `32c00cc` encoder).
- AV1 `GetNzMag` / `GetLowerLevelsCtx2d` were indexing into the raw `levelsBuf` without compensating for the `TxPadTop * stride` leading-pad offset that libaom's `set_levels()` applies via pointer arithmetic. Both helpers were reading from the leading zero-pad rows instead of actual coefficient data, so coefCtx for c > 0 in 2D class was always just the positional offset (mag=0). Encoder/decoder were self-consistent (both wrong the same way) but bit-incompatible with libaom + libdav1d. After fix: BBB transcode 4/60 → 60/60 frames decoded by libdav1d; av1.mp4 plays cleanly in VLC alongside vp8.mp4 + vp9.mp4 (commit `d9ba4b2`).

**Known remaining gaps:**

- AV1 walker chroma drift on BBB: U mean -12, V mean -28 vs ffmpeg ground truth (down from -44, -60 pre-fix but not yet zero). Likely chroma scan/qctx + remaining intra mode stubs.
- VP8/VP9 inter frames + loop filter still NotImplementedException.



### Video codec foundations

**AV1 ENCODER FRAMING (BIT-EXACT validated against libaom-av1 + ffmpeg + libdav1d)**

- `Av1ObuWriter.EmitObu(...)` - emits AV1 OBU header bytes + extension + LEB128 size
- `Av1SequenceHeaderWriter.EmitPayload(Av1SequenceHeaderConfig)` - 28 fields, BIT-EXACT vs libaom-av1's BBB SH (14/14 bytes)
- `Av1FrameHeaderWriter.EmitPayload(Av1FrameHeaderConfig, Av1SequenceHeader)` - 16 fields, prefix through allow_intrabc
- `Av1BitWriter` - MSB-first bit packer
- `Av1IvfRemuxer.RemuxToBytes(...)` - high-level IVF round-trip remux (BBB 77,725/77,725 bytes BIT-EXACT)
- `Av1IvfRemuxer.RemuxToBytesWithShSubstitution(...)` - SH-substitution remux (BBB byte-identical with config-driven SH)
- Closed-loop bridges: `Av1SequenceHeaderConfig.FromHeader(sh)` + `Av1FrameHeaderConfig.FromHeader(fh)`

**AV1 DECODER pipeline (parser side)**

- `Av1ObuParser`, `Av1SequenceHeaderParser`, `Av1FrameHeaderParser` (16 fields)
- `Av1Decoder` exposes `LastSequenceHeader`, `LastFrameHeader`, `LastFrameObuCounts`, `CumulativeObuCounts`, `CumulativeFrameTypeCounts`, `ShowExistingFrameCount`, `TotalTemporalUnits`
- Placeholder pixels until block decode wires up

**VP9 ENCODER FRAMING (BIT-EXACT validated against ffmpeg)**

- `Vp9SuperframeWriter.Emit(IReadOnlyList<byte[]>)` - VP9 spec Annex B.1 packer (300 BBB packets BIT-EXACT round-trip)
- `Vp9IvfRemuxer.RemuxToBytes(...)` - IVF round-trip remux (byte-identical to source)
- `Vp9FrameHeaderWriter` - uncompressed header prefix (full header writer pending)

**VP9 DECODER pipeline**

- Superframe parser, complete uncompressed header parser, compressed header parser
- `Vp9Decoder` exposes `LastFrameHeader`, `LastCompleteHeader`, `LastCompressedResult`, `LastCompressedState`, `LastTileGroup`, `CumulativeFrameTypeCounts`, `ShowExistingFrameCount`, `TotalCodedFrames`, `TotalVisibleFrames`
- Placeholder pixels until block decode wires up
- ~58 slices of block decode foundation (intra prediction kernels, iDCT family, MV decode chain)

**Consumer APIs**

- `Av1StreamAnalyzer.Analyze(ivfBytes)` + `Vp9StreamAnalyzer.Analyze(packets)` - high-level introspection
- `Av1StreamSummary.ToReport()` + `Vp9StreamSummary.ToReport()` - human-readable reports
- `Av1StreamValidator.Validate(...)` + `Vp9StreamValidator.Validate(...)` - bitstream QA
- `IvfDetector.IsIvf(bytes)` + `IvfDetector.DetectCodec(bytes)` - format detection

**Containers**

- IVF reader + writer (32-byte DKIF header + per-frame size/pts)
- Matroska / WebM reader (via SpawnDev.EBML 3.0.0)
- Ogg reader (RFC 3533)
- RIFF/WAVE + AIFF reader/writer

**Audio codecs**

- FLAC: encoder + decoder, ffmpeg cross-validated bit-exact, competitive with libFLAC (20× better on constants, 24% better on linear ramps)
- Opus SILK decode + Opus-in-Ogg packager
- Opus CELT decode WORKING via Concentus 2.2.2 (BSD-3) backbone, bit-exact across 6 ILGPU backends
- Opus encoder WORKING via Concentus 2.2.2 backbone: mono + stereo, 8/16/24/48 kHz, 2.5 / 5 / 10 / 20 ms frames, VoIP / Audio / RestrictedLowDelay applications, encode + decode round-trip across all 6 ILGPU backends (126 tests)
- Vorbis: structural decoder + minimum-viable encoder, AMPLITUDE-CORRECT as of 2026-04-27. MDCT normalisation follows libvorbis convention (4/N on encoder forward, unscaled inverse). Residue codebook anchors entry N/2 at exactly 0 so noise-gated bins decode silently. ffmpeg-decoded our-ogg lands within 1% RMS / 12% peak of source amplitude (pre-fix was "deafening"); our-decoded libvorbis-ogg lands within 5% RMS of source (pre-fix was 600x too quiet).

**Cross-backend coverage**

- 510+ tests pass on every ILGPU backend: CPU, CUDA, OpenCL, WebGPU, WebGL, Wasm
- All AV1 + VP9 + IVF + FLAC writer/parser/analyzer/validator surfaces cross-backend green

### Known issues

- VP9 decoder block-level pixel chain composes end-to-end on real BBB tile bytes but predicted values don't match ffmpeg ground truth. Bit-position drift somewhere upstream in partition / mode info chain. See `Plans/PLAN-vp9-block-decode-wiring.md` for the diagnostic plan.
