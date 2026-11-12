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

1. [ ] **Reference plumbing + P-frame syntax, zero motion.** `Vp9EncoderGpu`; key frame recon in the pool; kernels
   2, 4-8 with MV = 0; skip flag when a leaf has no coefficients. `VideoFileEncoderGpu` uses it (key frame interval
   option; sync samples only on key frames in MP4 / WebM).
2. [ ] **Motion search.** Full-pel search + sub-pel refinement with the real EIGHTTAP filter; NEARESTMV / NEARMV /
   ZEROMV / NEWMV chosen in the entropy pass from the searched MV; MV coding.
3. [ ] **Large skipped partitions** (32x32 / 64x64 when all leaves under them are skipped with one MV).
4. [ ] **Backward adaptation + previous-frame MV candidates** (`error_resilient_mode = 0`; counts written by the
   entropy pass; the decoder's `Vp9AdaptKernels`).
5. [ ] **Loop filter** on the reconstruction (decoder's kernels; key frames too).
6. [ ] Measure, then decide: intra blocks in inter frames, scene-cut key frames, high-precision MVs, golden / altref.

## Tests (per stage)
- Every frame: encoder's frame buffer == `Vp9DecoderGpu` output (no drift over a clip, sizes with every edge leaf
  shape: 80x50, 72x64, 70x50, 200x9).
- Black-box external decode: ffmpeg on the desktop (md5 of decoded planes == ours), the browser's VP9 decoder in the
  browser lanes.
- Fidelity vs the source per frame (no PSNR decay over the clip).
- File size vs key-frame-only at the same q index (recorded in `_TestStats`, the number this item exists for).
- Red-check each guard.
