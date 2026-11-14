# PLAN: VP9 inter frames in the encoder

Item 3 of `PLAN-ComfyUI-Video-IO.md` (file size for 121-frame Wan clips). Started 2026-11-12 (Geordi).
Licensing: libvpx v1.17 (BSD-3) may be ported; FFmpeg and ComfyUI are black boxes only.

## Design (decided 2026-11-12)

**The encoder writes the full decoder's own layouts and runs the decoder's reconstruction kernels on them.**
A frame's mode info (`Vp9Mi` records + grid), block table and eobs (`Vp9PLayout`), dequantized coefficients (`dq`) and
frame buffers (`Vp9FrameBuffersGpu` pool) are exactly what `Vp9TileParserGpu` would produce from the bitstream the
encoder writes. So the reference frame the encoder keeps is what any decoder reconstructs, through the same kernels
(`Vp9ReconKernels.InterPredictKernel`, `InterResidualKernel`, later `Vp9LoopFilterKernels`), and a test proves it frame
by frame (encoder buffer == `Vp9DecoderGpu` output == external decoder).

**A frame whose blocks are all inter needs no wavefront.** Motion search, prediction, forward transform and residual
add run one thread per block / transform block; only the entropy pass is serial (resumable, bounded launches).

Per inter frame (host = coordinator; class `Vp9EncoderGpu`, stateful, composes `Vp9KeyframeEncoderGpu` for key frames,
whose reconstruction is written straight into the pool):

| # | Kernel | Thread | New / reused |
|---|---|---|---|
| 1 | PlanePadKernel | pixel | reused |
| 2 | InterSetup: mi records + grid + block table for the fixed 16x16 leaves (edge leaves 16x8 / 8x16 / 8x8) | 16x16 position | new |
| 3 | Motion search: MV per leaf into `Vp9Mi.Mv0` (stage 2; stage 1 = zero) | leaf | new |
| 4 | `InterPredictKernel`: prediction into the new frame buffer | 4x4 unit | decoder's, unchanged |
| 5 | Forward: source - prediction, FDCT, quantize -> levels (entropy input), dq + eob (decoder layout) | transform block | new |
| 6 | `InterResidualKernel`: residual added in place = the reconstruction | 4x4 unit | decoder's, unchanged |
| 7 | Inter entropy: partition, skip, is_inter, ref, inter mode (`FindMvRefs` on the mi it reads back), MV, tokens; writes `Mode` and `Skip` into mi | frame, resumable | new (mirror of `ReadInterBlockModeInfo`) |
| 8 | Compressed header, uncompressed header (non-key), assemble | 1 | extended |

Bitstream choices, stage 1-3: `error_resilient_mode = 1` (default probabilities every frame, no backward adaptation,
no previous-frame MV candidates), single reference LAST (all three ref slots = slot 0, refresh 0xFF, sign bias 0),
`interp_filter = EIGHTTAP` (not switchable), `allow_high_precision_mv = 0` (quarter-pel MVs), tx_mode ALLOW_32X32,
segmentation off, loop filter level 0, one tile. Fixed quantizer (same q index as the key frames).

## Stages (each: decodable, verified on every backend, then the next)

1. [ ] **Reference plumbing + P-frame syntax.** `Vp9EncoderGpu`; key frame recon in the pool; kernels 2, 4-8; skip
   flag when a leaf has no coefficients; NEARESTMV / NEARMV / ZEROMV / NEWMV chosen in the entropy pass from the
   leaf's MV; MV coding.
   - [x] Code + test `Vp9EncoderGpu_InterFrames_DecoderMatchesEncoderReconstruction` (4affcf4): CPU accelerator,
     red-checked (7 mutations red).
   - [x] External decoder: ffmpeg 8.1 (black box) decodes the clips to the encoder's reconstruction (d9c74f4).
   - [x] CUDA, Velocity, WebGPU lanes (2026-11-12, 418c5b6, ILGPU 5.4.30): VelocityCodecsTests 6/6,
     CudaCodecsTests 6/6, WebGPUCodecsTests 6/6 on `Vp9EncoderGpu_` + `VideoFileEncoderGpu_` (WebGPU: probe of one
     test alone first; no device loss, GPU process peak 1.25 GB). Not run: Wasm and OpenCL lanes (release sweep).
     - [x] Offline WGSL screen + compile-only probe by Data (2026-11-12, ILGPU 5.4.30, RTX 4070 D3D12): all kernels
       valid; the entropy step is 19,464 inlined lines (below three kernels the WebGPU lane already runs), compiles
       in 3.1 s, GPU process peak 342 MB. Nothing was dispatched: run-time behaviour is still unproven there.
   - [ ] Browser's own VP9 decoder in the browser lanes.
   - [x] `VideoFileEncoderGpu` uses it: `VideoFileEncodeOptions.KeyFrameInterval` (default 128); only key frames
     are sync samples (MP4 stss, WebM keyframe flag == the bitstream's frame type); each inter frame within 1 dB of
     the same frame coded as a key frame. CPU accelerator, red-checked (5 mutations).
   - [x] `VideoFileEncoderGpu`: frames already on the accelerator (one ArrayView, N x H x W x 3): the managed
     overload's file byte for byte on the CPU accelerator and CUDA, red-checked.
   - [x] `VideoFileEncoderGpu` TypedArray overload (`IReadOnlyList<Float32Array>` -> `Uint8Array`): the managed
     overload's file byte for byte on WebGPU (`VideoFileEncoderGpu_TypedArrayOverload_MatchesManaged`).
2. [ ] **Motion search.** Pyramid full-pel search (+-38 pixels), half / quarter-pel refinement and a merge with the
   neighbours' MVs, scored with the decoder's own predictor (`Vp9MotionSearchKernels`, d9c74f4).
   - [x] Code + tests on the CPU accelerator: a moving picture's inter frames are 41-42% of zero motion's at q 40.
   - [x] The search finds the motion (`Vp9EncoderGpu_MotionSearch_FindsTheMotion`): a picture moving by a known
     amount, every leaf inside the picture has exactly that MV: (3, 2), (-29, 22), (26, -30), (2.5, -1.5),
     (-0.75, 4.25) pixels per frame. Red-checked on the CPU accelerator, 8 mutations, each red in the case aimed
     at it: full-pel result dropped (also the shrink assertion), 1/4 size level off, full size level off, half-pel
     off, quarter-pel off, merge candidates off, NEWMV charge off. Not red-checked: the 1/2 size level (no case
     isolates it).
   - [x] CUDA, Velocity, WebGPU lanes (with stage 1's, above).
   - [x] Measured on real clips (`DemoConsole vp9-encode-bench <clip.webm>`, CUDA, 2026-11-12, 48 frames each,
     round-to-nearest quantizer): comfyui_vp9_512x512_crf10 with inter frames is 53% / 50% / 47% / 42% of key frames
     only at q 20 / 40 / 60 / 100 (RGB PSNR within 0.11 dB). Speed on the RTX 4070: 9.6 to 15 frames/s at 512x512
     (key frames only: 5.9 to 10).
     Parity gap, stated plainly: libvpx's own file of that clip (crf 10) is 695 KB; ours at q 40 was 1.82 MB with
     round to nearest and is 1.10 MB with the dead zone below. Key frames are DC_PRED 16x16 only (2.3 bits/pixel
     at q 40).
   - [ ] The search's cost settings (`Vp9MArg.LengthCost`, `NewMvCost`) on real clips.
     Known: on a still picture with fresh noise in every frame (+-3) 3 of 120 leaves take a quarter-pel MV and the
     inter frames are 0.4% larger than zero motion's (2.5% without the NEWMV charge): the charge is a little low.
   - [x] Inter quantizer dead zone (`Vp9ForwardQuantizerGpu.QuantizeBlockDeadZone`, `Vp9EncoderGpu.InterZbinQ7` /
     `InterRoundQ7` = 100 / 48, in 1/128 of a step). Rate-distortion sweep on two real clips, q 20 to 130, seven
     settings: at equal luma PSNR 100 / 48 needs 33% (512x512) and 39% (832x480) fewer bytes than round to
     nearest, at equal RGB PSNR 23% and 28% fewer; best of the seven on both clips and both measures. 86 / 42
     (libvpx-like): 30% / 35%. Wider dead zones (to 120) gain nothing in luma and lose in chroma. No decay over
     the clip (last 8 frames within 0.1 dB of the worst frame). Guard test
     `Vp9EncoderGpu_InterDeadZone_SmallerThanRoundToNearest` (red-checked, 3 mutations). Key frames still round
     to nearest (their CPU reference mirrors it byte for byte): the same measurement is owed for them.
3. [ ] **Large skipped partitions** (32x32 / 64x64 when all leaves under them are skipped with one MV).
4. [ ] **Backward adaptation + previous-frame MV candidates** (`error_resilient_mode = 0`; counts written by the
   entropy pass; the decoder's `Vp9AdaptKernels`).
   - [x] Code (2026-11-12): the encoder owns a `Vp9FrameContextsGpu` (the decoder's class): BeginFrame, its own
     compressed header read back by the decoder's reader, the entropy pass codes with that context (coefficient
     probabilities too: `Vp9BlockCoefEncoderGpu.EncodeBlockAdaptive`) and counts every symbol in the decoder's
     FRAME_COUNTS layout (kept in the pass's state buffer: no extra binding), Adapt + EndFrame after the tiles;
     the entropy pass writes the frame's MV map and `FindMvRefs` reads the previous frame's.
     `Vp9EncoderGpu.BackwardAdaptation` (internal, default on; off = the error resilient frames of before).
     Key frames are unchanged (refresh_frame_context = 0: the first inter frame starts from the defaults).
   - [x] CPU accelerator: `Vp9EncoderGpu_BackwardAdaptation_SmallerAndDecodes` (12 frames, 192x128 and 70x50, both
     modes): decoder == encoder reconstruction on every frame, the encoder's symbol counts and four saved contexts
     == the decoder's own after every frame, adapting inter frames 88.7% / 90.2% of the error resilient bytes.
     Red-checked: 20 mutations red (every count, the MV map, the context steps); one (the corner block's implied
     SPLIT) is red only through the counts / contexts comparison: no coded symbol depends on it yet.
   - [ ] Velocity, CUDA, WebGPU lanes (NOT run: the entropy step's WGSL grew: screen it first, watchdog on).
   - [ ] Measure on the real clips (vp9-encode-bench) and against libvpx's files of the same clips at equal PSNR
     (`DemoConsole vp9-psnr`; ffmpeg -c:v libvpx-vp9 -crf N -b:v 0).
   - [ ] Forward probability updates in the compressed header (the reader side is already in the loop).
5. [ ] **Loop filter** on the reconstruction (decoder's kernels; key frames too).
6. [ ] Measure, then decide: intra blocks in inter frames, scene-cut key frames, high-precision MVs, golden / altref.

## Entropy speed and tiles (2026-11-13, Geordi)

- [x] Parallel entropy (c344843, ad5b64b, b1653c9): `Vp8BoolEncoderParallelGpu` + `Vp9KeyframeSymbolsKernel` +
  `Vp9InterSymbols`, byte-identical to the serial walkers (kept as the oracle and as the CPU-accelerator path,
  `Vp8BoolEncoderParallelGpu.PaysOn`). Green on CPU, Velocity, CUDA, OpenCL, WebGPU, Wasm. CUDA q40, same bytes:
  1920x1080 key 1851 -> 43 ms, inter 836 -> 22 ms; 512x512 key 236 -> 13 ms, inter 112 -> 15 ms (with Data's
  `SerialThreadPlacement` on the per-diagonal wavefront).
- [ ] **Tiles** (port of branch `vp9-multi-tile-entropy`, 2026-05-05, NOT merged: built on the May walker; keep the
  branch until this is done). Reuse: the header's tile bits (tile_cols_log2 increments + terminator unless max,
  tile_rows_log2 0 / 10 / 11) and the frame layout (every tile but the last prefixed by its 4-byte big-endian size).
  Do NOT reuse its `GetTileOffsetSb` / `GetTileColRange`: it rounds an mi offset up to a superblock, libvpx
  `get_tile_offset` is `((idx * sb_cols) >> log2) << 3` (min mis). 1920 wide, 4 tile columns: the branch gives
  [0,8) [8,15) [15,23) [23,30), libvpx [0,7) [7,15) [15,22) [22,30); its test asserts the wrong values. Use master's
  formula (decoder `Vp9ParseKernels.Tiles.cs:341`, `Vp9ReconKernels.cs:234`, References `Vp9TileLayout`).
  Steps, each green with tiles off by default:
  1. Header kernels (key `Vp9FrameUncompressedHeaderKernel`, inter `Vp9InterHeaderKernels.UncompressedHeaderKernel`):
     log2TileCols / log2TileRows arguments.
  2. Assembler: multi-tile layout, one thread per output byte like the current one (the branch's is one thread).
  3. Tile-major slot order (tile row, tile col, superblock raster inside the tile); one coder stream per tile.
  4. Contexts at tile boundaries (libvpx decode_tiles): left contexts (entropy, partition) restart at every tile
     column start per superblock row; above contexts are cleared once per frame (not per tile row); inter: the left
     mode-info neighbour is unavailable at a tile column start, MV candidates stay inside the tile's columns
     (`BlockPos.MiColStart / MiColEnd`). In both symbol kernels AND both serial walkers.
  5. Default: the most tile columns the width allows (libvpx's default `--tile-columns 6`), tile rows 0. Tests: both
     entropy paths byte-identical per tile layout, Vp9DecoderGpu (bit-exact libvpx) and ffmpeg decode every layout,
     widths where sb_cols is not a multiple of the tile count (e.g. 1920 / 4 tiles, 1912).
- [ ] VP9 key wavefront, Data's AV1 stage-1 shape (`Plans/AV1-GPU-encoder-architecture.md`, "Parallel inside a tile,
  stage 1"): lanes per diagonal in one group + barrier, working areas in device memory (may retire the WebGPU
  per-launch wait WORKAROUND), a barrier-free entry point for the CPU accelerators. 1080p wavefront is 31 of 43 ms.
- [ ] Re-bench CPU / Velocity on ILGPU 5.4.32 (tier-0 fix) and measure Wasm parallel vs serial; revisit `PaysOn`.

## Found on the way (owned, not yet fixed)
- Key frame compressed header (`Vp9FrameCompressedHeaderKernel` and its CPU reference): after tx_mode it writes 4
  zero bits at probability 128 where the syntax has 5 (the TX_MODE_SELECT bit + 4 coefficient update flags).
  Every decoder reads the same values because every later symbol of that header is a zero (the coded value sits at
  the bottom of the range), so files are valid; the inter header writes the exact sequence. Fix both writers
  together, with the byte comparison tests, when the key frame path is next touched.
- `ForwardKernel` on WebGPU: 59% of its inlined lines are emulated 64-bit operations (144 `i64_mul_i32` in the
  16x16 butterflies; Data's screen, 2026-11-12). libvpx's 8-bit build does this transform in 32 bits
  (`tran_high_t` = int32 without high bit depth), so 32-bit math should be exact here: change it under the byte
  comparison tests of the key frame encoder (same helpers).
- Source planes are zero padded to the 64 grid: at pictures that are not a multiple of 8 the padded strip inside
  the last 8x8 costs bits on every frame. INTER FRAMES FIXED 2026-11-12 (no residual past the picture's edge, the
  motion search scores only pixels inside it): inter / key bytes went from 66% / 72% / 101% to 36% / 33% / 23% at
  80x50 / 70x50 / 200x9; guard in `Vp9EncoderGpu_InterFrames_DecoderMatchesEncoderReconstruction`, red-checked.
  KEY FRAMES still code the zero padding: repeat the edge (or drop the residual) there (changes key frame bytes:
  CPU reference too).

## Tests (per stage)
- Every frame: encoder's frame buffer == `Vp9DecoderGpu` output (no drift over a clip, sizes with every edge leaf
  shape: 80x50, 72x64, 70x50, 200x9).
- Black-box external decode: ffmpeg on the desktop (md5 of decoded planes == ours), the browser's VP9 decoder in the
  browser lanes.
- Fidelity vs the source per frame (no PSNR decay over the clip).
- File size vs key-frame-only at the same q index (recorded in `_TestStats`, the number this item exists for).
- Red-check each guard.
