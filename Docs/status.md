# Status

What is verified, on which backend, and how. "Verified" means a test that compares against a reference
implementation (never against this library's own output) passed on that backend, with its guard red-checked. Dates are
the last run of the named lane. Updated 2026-10-29 (WebGPU video on ILGPU 5.4.11; browser Images and Audio refreshed on ILGPU 5.4.9-local.4; counts are
passed / failed, skips are gated or CPU-only tests).

Legend: **green** = the codec's suite passes on that lane; **partial** = runs, with the open items named; **open** = not
yet passing or not yet run on the current build; **n/a** = the backend cannot run it (WebGL has no atomics or scatter
stores: its Transform Feedback captures a thread's store only at the thread's own slot, so a serial stream coder, where
one thread writes a whole stream, cannot run there; such tests declare `RequiresScatterStores` and skip on WebGL).

**Fake passes before 2026-10-28 19:00:** a page whose .NET runtime died from an unhandled fatal exception (a
StackOverflowException) kept a frozen DOM, and PlaywrightMultiTest read the later rows as PASS. Fixed in 1cc7c7f (a
Blazor error gives the next test a fresh page; a test whose row did not change fails as "never ran"). Of the browser
runs behind this table, only the 10-28 WebGL video run (VP8 / VP9 rows corrected here) and the 10-27 audio refresh had
such a crash; the later 10-28 all-families runs had none.

**WebGL before 2026-10-26:** the WebGL test class threw `NotSupportedException` for every test, so every "skipped" WebGL
result before that date means "never ran". The lane now creates the real WebGL2 accelerator; the WebGL column is being
filled batch by batch.

**Velocity (the vectorized CPU backend applications get by default):** the CPU column is the ILGPU debug CPU backend,
which interprets the kernels' IL, so Velocity's code generator was never exercised until the `VelocityCodecsTests`
lane (10-26). Its first full run used the Scalar2 device by mistake (`Velocity()` registers Scalar2 first); the lane
runs Vector128, the device applications get, since 10-27. **10-27, ILGPU 5.4.7-local.11: 970 passed, 0 failed** (2181
skipped: CPU-only tests that run once in the CPU lane, browser-only overloads). The runs found four Velocity bugs,
all fixed in ILGPU: a code-generator crash in the VP9 decoder, the WebP lossy encoder's bool NOT on Scalar2, a stack
overflow of large kernels on Vector128 (every VP9 encode, AV1 encode and decode), and loop exit values mixed between
lanes (VP9 batch encode at unaligned sizes).

## Compression

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| Shuffle filters (byte shuffle, bitshuffle) | green | green | green | green (10-26) | green (10-26) | green (10-26, ILGPU 5.4.6) | 94 fixtures from c-blosc / numcodecs and the bitshuffle library, both directions; three 16 MiB SHA-256 cases |
| LZ4 frame compress / decompress | green | green | green | green (10-26, 12 / 0) | n/a (scatter stores) | green (10-26, 12 / 0) | compressor byte-identical with liblz4 1.9.4 (29 frames up to 40 MB); decoder reads every liblz4 configuration (64 KB .. 4 MB blocks, linked, HC, checksums, legacy, skippable) and refuses damaged frames with the reason |
| Snappy compress / decompress (raw + framed) | green (10-26, 10 / 0) | green (10-26) | green (10-26) | green (10-26, 12 / 0) | n/a (scatter stores) | green (10-26, 12 / 0) | compressor byte-identical with the Rust snap crate (python-snappy / cramjam): 28 inputs x raw + framed up to 40 MB; decoder reads every element form (literal tags 60-63, copy-4, offsets 400 KB back), framed chunks of any size incl. stored / padding / skippable, and refuses 15 damaged streams with the reason |
| GDeflate decompress + compress (DirectStorage tile streams) | green (10-26) | green (10-26, 7 / 0) | green (10-26, 7 / 0) | green (10-26, 8 / 0) | n/a (scatter stores) | green (10-26, 8 / 0) | 31 streams written and decoded by Microsoft's reference (built from source) at levels 1 / 6 / 9 / 12, 1 B .. 8 MB, stored / static / dynamic blocks, Deflate64 long matches and far distances; damaged streams refused with the reason. 32 lanes per tile (CUDA, RTX 4070): 4.32 GB/s on BF16 weights, 7.13 GB/s on source text (10-27, shared memory per tile 10.4 -> 3.9 KB; was 3.10 / 6.17), vs the reference CPU decoder on 31 threads 2.01 / 5.65. Compressor: 75 streams at levels 0-4 (CUDA-encoded, incl. 512 MB of BF16 weights) all decoded by the reference; level 3 within -3 .. +5 % of its level 6, BF16 weights 2.3 % smaller. Compress speed (10-27, every stage parallel, same bytes): 512 MB BF16 at 0.47 GB/s (level 1) / 4.81 GB/s (level 4, Huffman only), vs the reference on one CPU thread at level 6 0.066 |
| WeightPack (lossless tensors) | green | green | green | green (10-26, 7 / 0, ILGPU 5.4.7-local.8) | n/a (scatter stores) | green (10-26, 7 / 0) | round trips exact; every chunk inflates with .NET DeflateStream; sizes within 1 % of an optimal Huffman code per chunk. Real weights (CUDA, RTX 4070): BF16 0.676 / FP16 0.667 / FP32 0.608 of the original, unpack 8.79 GB/s (BF16, table decode + one launch per wave 10-26; was 1.2 to 1.9), pack 3.57 GB/s (BF16, parallel Huffman-only encoder 10-26: per-chunk shared histogram, prefix-summed bit offsets; byte-identical with the old encoder; was 0.12 to 0.14) |
| zlib / DEFLATE (+ Huffman-only level) | green | green | green | green (10-27, ILGPU 5.4.9-local.1) | n/a (scatter stores) | green (10-27) | .NET ZLibStream inflates our streams; our InflateGpu agrees; the parallel path (ParallelParse, lanes counting, EmitBlockLanes) is byte-identical with the serial ParseStep / EmitStep oracle at levels 1-4. Speed (10-27, RTX 4070, 64 MB BF16): level 1 0.40 GB/s, level 4 1.30 (default was 0.006 before the parallel parse and the automatic wave) |

## Images

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| PNG decode / encode | green | green | green | green (10-28, 21 / 0) | n/a (scatter stores) | green (10-28, 21 / 0) | 64 decode fixtures; encoder output read by an independent host reader and by Pillow; text chunks |
| JPEG decode / encode | green | green | green | green (10-28, 18 / 0) | n/a (scatter stores) | green (10-28, 18 / 0) | decoder bit-exact with libjpeg-turbo on the fixture set (all subsamplings, progressive, restarts, CMYK / YCCK, EXIF); encoder byte-identical with libjpeg-turbo |
| WebP lossless decode | green | green | green | green (10-28; the WebPDecoderGpu family, 29 / 0, covers lossless and lossy) | n/a (scatter stores) | green (10-28, ILGPU 5.4.9-local.6: Wasm memsets are now ordered with in-flight dispatches; the second decode on a reused decoder read a half-zeroed state before) | libwebp pixels |
| WebP lossless encode (+ ALPH, animation, metadata) | green | green | green | green (10-28, 15 / 0) | n/a (scatter stores) | green (10-28, ILGPU 5.4.9-local.6) | libwebp and both our decoders decode our files to the source pixels |
| WebP lossy decode | green | green | green | green (10-28; TinyPool and metadata now inside the 30 s default) | n/a (scatter stores) | see WebP lossless decode (same decoder family) | 168 fixtures bit-exact with libwebp |
| WebP lossy encode | green | green | green | green (10-28, ILGPU 5.4.9-local.4) | n/a (scatter stores) | green (10-28) | byte-identical with libwebp, 222 fixtures (methods 3 / 4); the RD loop was split into stage kernels for browser shader compilers (10-25) |
| WebP lossy encode with alpha | green | green | green (10-27) | green (10-28) | n/a (scatter stores) | green (10-28) | VP8 chunk byte-identical with libwebp, alpha exact (libwebp and our decoder), 145 fixtures |
| GIF decode / encode | green | green | green | green (10-28, 21 / 0; the slow fixture tests now inside the default) | n/a (scatter stores) | green (10-28, ILGPU 5.4.9-local.6: the next frame's histogram was zeroed under the previous frame's kernels, so a count read 0 and Wasm trapped on the division) | Pillow decodes our files to the same pixels; quantizer PSNR vs Pillow |
| BMP decode / encode | green | green | green | green (10-28, 19 / 0) | green (10-28, 12 / 0; header and alpha scans as gathers); RLE refused (scatter stores) | green (10-28, 19 / 0) | Pillow |

## Textures

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| BCn decode (BC1, BC3, BC4, BC5, BC7) | green (10-27, also Velocity) | green | green | green | green | green | 29 fixtures bit-exact with the reference decoders (texture2ddecoder, cross-checked with Pillow): reference-encoder blocks (etcpak, bc7enc per mode) over a real frame and an RGBA picture, random blocks (every BC7 mode, partition, rotation, the reserved mode), partial edge blocks. Spec choices where the references disagree: BC3 colour always 4-colour, BC1 3-colour index 3 = (0,0,0,0), BC7 reserved mode = (0,0,0,0) |
| BCn encode BC1, BC3, BC4, BC5 | green (10-27, also Velocity) | green | green | green | green | green | 19 fixtures byte-identical with etcpak 0.9.15 (the portable paths its Windows wheel runs; ported, BSD 3-clause in NOTICE.md): a real frame at two widths, an RGBA picture, noise, flat / nearly flat blocks. Sizes that are not a multiple of 4 repeat the edge pixels |
| BCn encode BC7 | green (10-27, also Velocity) | green | green | green | green | green | port of bc7enc (MIT, NOTICE.md) at its default settings, 9 fixtures / 8435 blocks: CPU, Velocity and Wasm byte-identical with bc7enc on every block (required); CUDA 98.97 %, OpenCL 94.5 %, WebGPU 94.7 % identical (ILGPU's GPU float contracts multiply-add / rounds differently; PSNR never more than 0.01 dB lower; GPU gate = PSNR within 0.05 dB + 90 %). ILGPU 5.4.9-local.3 fixed WebGL (unsigned 64-bit compares) and worked around an NVIDIA D3D12 driver miscompile on WebGPU (local.2). Speed 2048x2048 (RTX 4070): CUDA 384 (424 at O2), OpenCL 655 Mpixel/s; bc7enc on one CPU thread 2.2 |

## Video

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| VP8 keyframe encode | green | green (10-23) | green | green (10-29, ILGPU 5.4.11; the entropy step was split 10-26) | n/a for the frame-level coders (serial stream coders: 416 / 159 / 184 transform-feedback outputs, WebGL captures 120); the building blocks (transforms, predictors, bool and coefficient coders) green (10-28, ILGPU 5.4.11-local.2) | green (10-28, ILGPU 5.4.11-local.1: a view's Length was 0 inside non-inlined helpers on Wasm) | ffmpeg decodes our frames; bit-exact with the CPU reference encoder |
| VP8 full keyframe decode | green | green (10-25) | green | green (10-28) | n/a (serial decoder: the full and the older decoder step kernels need scatter stores; the older one is refused at 159 transform-feedback outputs, WebGL captures 120) | green (10-28, ILGPU 5.4.11-local.4: the older decoder path read stream bytes sign-extended inside helpers on Wasm) | 182 files, identical planes to ffmpeg; one CUDA failure in 24 runs on 10-25 traced to an unpinned host copy, fixed in ILGPU 5.4.5 |
| VP9 keyframe encode / decode | green | green (10-23) | green | green (10-29, ILGPU 5.4.11, 267 / 0; video lane 468 / 0) with one **workaround**: a drain after each wavefront group-kernel slice (see Notes, VP9 WebGPU device hang) | n/a for the frame encoder and decoder (serial coders; batch wavefront 561 transform-feedback outputs, WebGL captures 120); the building blocks (transforms, predictors, coefficient coders) green (10-28) | green (10-28, ILGPU 5.4.11-local.1) | ffmpeg / CPU reference |
| AV1 keyframe encode / decode | green | green (10-23) | green | green (10-28, all 144 AV1 tests) | n/a for the frame step kernels (972 transform-feedback outputs, WebGL captures 120; restructure planned, PLAN-Oversized-Kernels.md); transforms and the rest green (10-28, 93 / 0, ILGPU 5.4.11-local.2) | green (10-28, all 144 AV1 tests) | libdav1d decodes our frames; CPU reference |

## Audio

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| Vorbis decode | green | green (10-25) | green | green (10-28; Vorbis family 59 / 0, the rest gated or CPU-only) | green (10-28, Vorbis family 186 / 0) | green (10-28, Vorbis family 188 / 0) | six real libvorbis streams within 1e-4 of libvorbis (observed 7.5e-7) |
| Vorbis encode (first version) | green | green | green | green (10-28, 12 / 0) | green (10-28, 17 / 0) | green (10-28, 17 / 0) | round trip through our decoders and ffmpeg |
| FLAC (first version) | green | green | green | green (10-28, 63 / 0) | green (10-28, 11 / 0; the serial coders gated) | green (10-28, 245 / 0) | round trip; ffmpeg reads our files |
| Opus decode on the accelerator (OpusDecoderGpu) | green (10-29, SILK; also Velocity) | green (10-29, SILK) | green (10-29, SILK) | open (not run in the browser yet) | n/a (serial stream coders) | open (not run yet) | SILK-only streams equal **libopus 1.5.2** (fixed point, built from source; TestData/Opus) on every packet incl. losses (PLC), comfort noise, DTX, FEC, stereo / mono and bandwidth switches, 10..60 ms; desktop CPU, Velocity, CUDA, OpenCL. CELT-only and hybrid frames: the GPU CELT decoder is in progress (Data; decode stages bit-exact vs Concentus), packets refused until then |
| Opus CELT / range coder GPU parts | green | green | green | green (10-28, Opus 7 / 0, CELT 10 / 0) | green (10-28, Opus 75 / 0, CELT 10 / 0) | green (10-28, Opus 77 / 0, CELT 17 / 0) | Concentus |
| SILK GPU primitives | green | green (10-25) | green | green (10-28, 139 / 0) | green (10-28, 202 / 0) | green (10-28, 202 / 0) | Concentus |

## Notes

- **VP9 WebGPU device hang (open, workaround in place, TJ-approved 2026-10-29).** In Chrome WebGPU (D3D12, NVIDIA RTX
  4070), a `Vp9EntropyStepKernel` queued while `Vp9EncodeWavefrontGroupKernel` (the single-dispatch barrier kernel of the
  batch encode) still runs leaves the device in a state where a LATER launch (the next encoder's first
  `Vp9EncodeWavefrontBatchKernel`) dies with DXGI_ERROR_DEVICE_HUNG. Found with Data by 9 pair-run probes (2026-10-28):
  dispatch batching, single pass, the shader caches and the compile path change nothing; one drain after the group
  kernel, or before the entropy step, prevents it. A hand-built WebGPU repro did not hang, but Data's replay of the
  recorded WebGPU calls of the failing page DOES hang in a fresh Chrome with no ILGPU (2 of 2, 2026-10-29), so the fault is
  below ILGPU (Dawn D3D12 or the NVIDIA driver). Minimal so far: one 64x64 single-frame encode's submit, then the batch
  group kernel + one entropy step (both required), then one 1-workgroup `Vp9EncodeWavefrontBatchKernel` dispatch. Workaround: `SynchronizeAsync` after each group-kernel slice in
  `Vp9FrameSequentialEncodeKernel.TryRunBatchSingleDispatchAsync` (`// WORKAROUND:`). Upstream report drafted with a reduced trace that still hangs (SpawnDev.ILGPU
  Plans/webgpu-d3d12-device-hung-trace-replay-upstream-report.md, attachment _upstream_reports/vp9-webgpu-device-hung.zip);
  TJ files it. The drain comes out once the cause is fixed. Regression test:
  WebGPU `Batch_ArrayViewOverload` then `Batch_Unaligned` in one page (the order the suite runs them).

- **ILGPU 5.4.6-local.2** (2026-10-26): integer divisions no longer hoisted out of their guards (Wasm traps), WebGL sends a
  program's GLSL once (not per dispatch) and refuses kernels over the device's Transform Feedback varying limit at load,
  SpawnJS 3.0.5-local.1 (geometric message growth: the WebGL out-of-memory).
- **ILGPU 5.4.5** (released 2026-10-25; this repo is on 5.4.5-local.2) carries the fixes the browser lanes needed: WGSL loop back-edge moves
  (the lossless WebP encode hang), sub-view `.Length` (VP8 / VP9 kernels failed WGSL validation; WebGL silently used the
  whole buffer's length), WebGL inner-loop exits, struct-parameter bindings, pinned host copies, and an O(N^2) memory
  peak in the WGSL generator. The WebGL and Wasm lanes have not been rerun on it yet.
- **Oversized kernels** are tracked in [Plans/PLAN-Oversized-Kernels.md](../Plans/PLAN-Oversized-Kernels.md): nine kernels
  over the browser compile budget, the split strategy, and the finding that the cost comes from unrolled loop nests.
- Before 2026-10-25 the **PlaywrightMultiTest** browser runner reported fake passes for every test after a page's .NET
  runtime died. The browser results above come from runs on the fixed runner (2026-10-25) unless marked otherwise.
