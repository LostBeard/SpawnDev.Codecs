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
   - [ ] CUDA, Velocity, WebGPU lanes (watchdog on the first WebGPU run, one test alone).
     - [x] Offline WGSL screen + compile-only probe by Data (2026-11-12, ILGPU 5.4.30, RTX 4070 D3D12): all kernels
       valid; the entropy step is 19,464 inlined lines (below three kernels the WebGPU lane already runs), compiles
       in 3.1 s, GPU process peak 342 MB. Nothing was dispatched: run-time behaviour is still unproven there.
   - [ ] Browser's own VP9 decoder in the browser lanes.
   - [x] `VideoFileEncoderGpu` uses it: `VideoFileEncodeOptions.KeyFrameInterval` (default 128); only key frames
     are sync samples (MP4 stss, WebM keyframe flag == the bitstream's frame type); each inter frame within 1 dB of
     the same frame coded as a key frame. CPU accelerator, red-checked (5 mutations).
   - [ ] `VideoFileEncoderGpu`: frames already on the accelerator (ArrayView) and TypedArray overloads.
2. [ ] **Motion search.** Pyramid full-pel search (+-38 pixels), half / quarter-pel refinement and a merge with the
   neighbours' MVs, scored with the decoder's own predictor (`Vp9MotionSearchKernels`, d9c74f4).
   - [x] Code + tests on the CPU accelerator: a moving picture's inter frames are 41-42% of zero motion's at q 40.
   - [x] The search finds the motion (`Vp9EncoderGpu_MotionSearch_FindsTheMotion`): a picture moving by a known
     amount, every leaf inside the picture has exactly that MV: (3, 2), (-29, 22), (26, -30), (2.5, -1.5),
     (-0.75, 4.25) pixels per frame. Red-checked on the CPU accelerator, 8 mutations, each red in the case aimed
     at it: full-pel result dropped (also the shrink assertion), 1/4 size level off, full size level off, half-pel
     off, quarter-pel off, merge candidates off, NEWMV charge off. Not red-checked: the 1/2 size level (no case
     isolates it).
   - [ ] CUDA, Velocity, WebGPU lanes.
   - [ ] Measure on real clips (ComfyUI / Wan): file size and PSNR vs key frames only, and the search's cost
     settings (`Vp9MArg.LengthCost`, `NewMvCost`). Tool: `DemoConsole vp9-encode-bench <clip.webm>`.
     Known: on a still picture with fresh noise in every frame (+-3) 3 of 120 leaves take a quarter-pel MV and the
     inter frames are 0.4% larger than zero motion's (2.5% without the NEWMV charge): the charge is a little low.
   - [ ] Inter quantizer rounding: with round-to-nearest an inter frame re-codes the previous frame's quantization
     noise (a perfectly predicted moving picture was only 2.4x smaller). Measure a dead zone for inter blocks as a
     rate-distortion comparison (PSNR at equal bytes), then decide.
3. [ ] **Large skipped partitions** (32x32 / 64x64 when all leaves under them are skipped with one MV).
4. [ ] **Backward adaptation + previous-frame MV candidates** (`error_resilient_mode = 0`; counts written by the
   entropy pass; the decoder's `Vp9AdaptKernels`).
5. [ ] **Loop filter** on the reconstruction (decoder's kernels; key frames too).
6. [ ] Measure, then decide: intra blocks in inter frames, scene-cut key frames, high-precision MVs, golden / altref.

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
  the last 8x8 costs bits on every frame. Repeat the edge instead (changes key frame bytes: CPU reference too).

## Tests (per stage)
- Every frame: encoder's frame buffer == `Vp9DecoderGpu` output (no drift over a clip, sizes with every edge leaf
  shape: 80x50, 72x64, 70x50, 200x9).
- Black-box external decode: ffmpeg on the desktop (md5 of decoded planes == ours), the browser's VP9 decoder in the
  browser lanes.
- Fidelity vs the source per frame (no PSNR decay over the clip).
- File size vs key-frame-only at the same q index (recorded in `_TestStats`, the number this item exists for).
- Red-check each guard.
