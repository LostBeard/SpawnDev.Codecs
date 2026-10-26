# Status

What is verified, on which backend, and how. "Verified" means a test that compares against a reference
implementation (never against this library's own output) passed on that backend, with its guard red-checked. Dates are
the last run of the named lane. Updated 2026-10-25.

Legend: **green** = the codec's suite passes on that lane; **partial** = runs, with the open items named; **open** = not
yet passing or not yet run on the current build; **n/a** = the backend cannot run it (WebGL has no atomics or scatter
stores).

## Images

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| PNG decode / encode | green | green | green | partial: the decoder passed (10-22), rerun of both pending | n/a (scatter stores) | partial: decoder passed (10-22) | 64 decode fixtures; encoder output read by an independent host reader and by Pillow; text chunks |
| JPEG decode / encode | green | green | green | open | open | open | decoder bit-exact with libjpeg-turbo on the fixture set (all subsamplings, progressive, restarts, CMYK / YCCK, EXIF); encoder byte-identical with libjpeg-turbo |
| WebP lossless decode | green | green | green | green (10-25, decodes inside the encoder tests) | open | open | libwebp pixels |
| WebP lossless encode (+ ALPH, animation, metadata) | green | green | green | **partial** (10-25): correct, but launch-bound slow; 3 tests over the 30 s default | open | open | libwebp and both our decoders decode our files to the source pixels |
| WebP lossy decode | green | green | green | open | open | open | 168 fixtures bit-exact with libwebp |
| WebP lossy encode | green | green | green | open | open | open | byte-identical with libwebp, 222 fixtures (methods 3 / 4); the RD loop was split into stage kernels for browser shader compilers (10-25) |
| WebP lossy encode with alpha | green | green | open | open | open | open | VP8 chunk byte-identical with libwebp, alpha exact (libwebp and our decoder), 145 fixtures |
| GIF decode / encode | green | green | green | **partial** (10-25, 19 passed): correct, 2 fixture tests take 52 to 73 s | open | open | Pillow decodes our files to the same pixels; quantizer PSNR vs Pillow |
| BMP decode / encode | green | green | green | green (10-25, 19 / 0) | open | open | Pillow |

## Video

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| VP8 keyframe encode | green | green (10-23) | green | **open**: the wavefront encode kernel is too large for Chrome's shader compiler (the transcode test never finishes) | open | open | ffmpeg decodes our frames; bit-exact with the CPU reference encoder |
| VP8 full keyframe decode | green | green (10-25) | green | open | open | open | 182 files, identical planes to ffmpeg; one CUDA failure in 24 runs on 10-25 traced to an unpinned host copy, fixed in ILGPU 5.4.5 |
| VP9 keyframe encode / decode | green | green (10-23) | green | **open**: VP9 wavefront encode kernel oversized (an out-of-memory in the page's WGSL generator on a long lane) | open | open | ffmpeg / CPU reference |
| AV1 keyframe encode / decode | green | green (10-23) | green | **partial** (10-25, 142 passed): encode and decode kernels oversized (GPU process lost while compiling, or 110 s compiles); batch vs single-frame output differs | open | open | libdav1d decodes our frames; CPU reference |

## Audio

| Codec | CPU | CUDA | OpenCL | WebGPU | WebGL | Wasm | Evidence |
|-------|-----|------|--------|--------|-------|------|----------|
| Vorbis decode | green | green (10-25) | green | partial (10-25): the long-block test passed; the libvorbis parity tests not yet run there | open | open | six real libvorbis streams within 1e-4 of libvorbis (observed 7.5e-7) |
| Vorbis encode (first version) | green | green | green | open | open | open | round trip through our decoders and ffmpeg |
| FLAC (first version) | green | green | green | green (10-25, 63 / 0) | open | open | round trip; ffmpeg reads our files |
| Opus CELT / range coder GPU parts | green | green | green | green (10-25, 12 / 0) | open | open | Concentus |
| SILK GPU primitives | green | green (10-25) | green | **partial** (10-25): pulse kernels fixed after the run (one buffer was bound twice); the shell-coder kernel needs ILGPU 5.4.5 (struct binding fix) | open | open | Concentus |

## Notes

- **ILGPU 5.4.5** (being released 2026-10-25; this repo is on 5.4.5-local.2) carries the fixes the browser lanes needed: WGSL loop back-edge moves
  (the lossless WebP encode hang), sub-view `.Length` (VP8 / VP9 kernels failed WGSL validation; WebGL silently used the
  whole buffer's length), WebGL inner-loop exits, struct-parameter bindings, pinned host copies, and an O(N^2) memory
  peak in the WGSL generator. The WebGL and Wasm lanes have not been rerun on it yet.
- **Oversized kernels** are tracked in [Plans/PLAN-Oversized-Kernels.md](../Plans/PLAN-Oversized-Kernels.md): nine kernels
  over the browser compile budget, the split strategy, and the finding that the cost comes from unrolled loop nests.
- Before 2026-10-25 the **PlaywrightMultiTest** browser runner reported fake passes for every test after a page's .NET
  runtime died. The browser results above come from runs on the fixed runner (2026-10-25) unless marked otherwise.
