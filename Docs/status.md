# Status

What is verified, on which backend, and how. "Verified" means a test that compares against a reference
implementation (never against this library's own output) passed on that backend, with its guard red-checked. Dates are
the last run of the named lane. Updated 2026-10-26.

Legend: **green** = the codec's suite passes on that lane; **partial** = runs, with the open items named; **open** = not
yet passing or not yet run on the current build; **n/a** = the backend cannot run it (WebGL has no atomics or scatter
stores: its Transform Feedback captures a thread's store only at the thread's own slot, so a serial stream coder, where
one thread writes a whole stream, cannot run there; such tests declare `RequiresScatterStores` and skip on WebGL).

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
| zlib / DEFLATE (+ Huffman-only level) | green | green | green | green (10-25) | n/a (scatter stores) | partial | .NET ZLibStream inflates our streams; our InflateGpu agrees |

## Images

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| PNG decode / encode | green | green | green | partial: the decoder passed (10-22), rerun of both pending | n/a (scatter stores) | partial: decoder passed (10-22) | 64 decode fixtures; encoder output read by an independent host reader and by Pillow; text chunks |
| JPEG decode / encode | green | green | green | open | open | open | decoder bit-exact with libjpeg-turbo on the fixture set (all subsamplings, progressive, restarts, CMYK / YCCK, EXIF); encoder byte-identical with libjpeg-turbo |
| WebP lossless decode | green | green | green | green (10-26) | n/a (scatter stores) | open | libwebp pixels |
| WebP lossless encode (+ ALPH, animation, metadata) | green | green | green | **partial** (10-26): correct, but launch-bound slow; 3 tests over the 30 s default | n/a (scatter stores) | open | libwebp and both our decoders decode our files to the source pixels |
| WebP lossy decode | green | green | green | **partial** (10-26): 2 decoder tests over the 30 s default (TinyPool 85 s, metadata 40 s) | n/a (scatter stores) | open | 168 fixtures bit-exact with libwebp |
| WebP lossy encode | green | green | green | open | open | open | byte-identical with libwebp, 222 fixtures (methods 3 / 4); the RD loop was split into stage kernels for browser shader compilers (10-25) |
| WebP lossy encode with alpha | green | green | open | open | open | open | VP8 chunk byte-identical with libwebp, alpha exact (libwebp and our decoder), 145 fixtures |
| GIF decode / encode | green | green | green | **partial** (10-25, 19 passed): correct, 2 fixture tests take 52 to 73 s | open | open | Pillow decodes our files to the same pixels; quantizer PSNR vs Pillow |
| BMP decode / encode | green | green | green | green (10-25, 19 / 0) | header scan and alpha scan rewritten as gathers 10-26 (rerun pending); RLE refused (scatter stores) | open | Pillow |

## Video

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| VP8 keyframe encode | green | green (10-23) | green | **open**: the wavefront encode kernel is too large for Chrome's shader compiler (the transcode test never finishes) | open | open | ffmpeg decodes our frames; bit-exact with the CPU reference encoder |
| VP8 full keyframe decode | green | green (10-25) | green | green (10-26: all fixtures, ArrayView into poisoned buffers, random corruption) | open | open | 182 files, identical planes to ffmpeg; one CUDA failure in 24 runs on 10-25 traced to an unpinned host copy, fixed in ILGPU 5.4.5 |
| VP9 keyframe encode / decode | green | green (10-23) | green | **partial** (10-26): every decoder / transform / predictor kernel test green (226 + 30); the encoder still kills Chrome's GPU process while compiling (`Vp9KeyframeGpu_TypedArrayOverloads_MatchManaged`) | open | open | ffmpeg / CPU reference |
| AV1 keyframe encode / decode | green | green (10-23) | green | **partial** (10-25, 142 passed): encode and decode kernels oversized (GPU process lost while compiling, or 110 s compiles); batch vs single-frame output differs | open | open | libdav1d decodes our frames; CPU reference |

## Audio

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| Vorbis decode | green | green (10-25) | green | green (10-26, real-GPU lane incl. the libvorbis parity tests) | open | open | six real libvorbis streams within 1e-4 of libvorbis (observed 7.5e-7) |
| Vorbis encode (first version) | green | green | green | open | open | open | round trip through our decoders and ffmpeg |
| FLAC (first version) | green | green | green | green (10-25, 63 / 0) | open | open | round trip; ffmpeg reads our files |
| Opus CELT / range coder GPU parts | green | green | green | green (10-25, 12 / 0) | open | open | Concentus |
| SILK GPU primitives | green | green (10-25) | green | **partial** (10-25): pulse kernels fixed after the run (one buffer was bound twice); the shell-coder kernel needs ILGPU 5.4.5 (struct binding fix) | open | open | Concentus |

## Notes

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
