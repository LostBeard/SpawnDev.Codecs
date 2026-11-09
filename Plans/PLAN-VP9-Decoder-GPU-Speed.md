# VP9 decoder: GPU speed (2026-11-09)

The decoder is bit-exact with libvpx on CPU, CUDA and OpenCL (`Vp9Decode_EveryFrame_MatchesLibvpx`). It is slow on
GPUs because every stage runs a handful of long sequential threads.

## Measured (RTX 4070 CUDA, ILGPU 5.4.21, 6 ComfyUI files, 252 frames, 512x320, `VP9_STAGE_TIMES=1`)

| stage | threads per launch (512x320) | CUDA ms/frame | CPU accelerator ms/frame |
|---|---|---|---|
| tile parse | 1 per tile column (1) | 19.1 | 3.4 |
| reconstruction | inter: 1 per superblock (40); intra: <= 5 per wavefront step (12 steps) | 16.2 | 9.1 |
| loop filter | <= 5 per wavefront step (k = 2r + c, 17 steps) | 16.5 | 3.5 |

Wasm (browser): the ComfyUI RGBA file takes more than 5 minutes, past PMT's per-test wait.

## Rules
- Bit-exact with libvpx stays the floor: every change keeps `Vp9Decode_EveryFrame_MatchesLibvpx` green on CPU, CUDA and
  OpenCL, red-checked.
- Speed comes from parallel structure, never from skipping work or changing results ([[perf-not-quality]]).
- Measure each step with `VP9_STAGE_TIMES=1` on CUDA (A/B) and log it to _TestStats.

## Steps, by expected gain

### S1 Loop filter: one thread group per superblock (64 lanes)
libvpx filters a superblock's vertical edges (all of them, left to right), then its horizontal edges; its own
multi-threaded loop filter runs superblock rows on a wavefront, which is our k = 2r + c schedule, so that schedule is
already proven equal.

Inside a superblock:
- Vertical edges: each pixel row is independent. Lane i takes row i and walks the edges left to right.
- Horizontal edges: each pixel column is independent. Lane i takes column i and walks the edges top to bottom.
- A group barrier separates the two passes.

That is 64x the lanes per superblock, with the same order per row and column, so the result is bit-exact by
construction. Chroma uses 32 lanes. Needs explicitly grouped kernels (`LoadStreamKernel`) with a group size of 64.

### S2 Reconstruction at transform-block granularity
- **Inter prediction:**
  - One thread per (plane, 4x4 unit) of every inter block, using the mode-info grid. The sub-pel filter is per
    pixel, so this goes from 40 threads to about 15k for 512x320.
  - Then the residual: one thread per transform block (later per row / column of the 1-D transforms).
- **Intra:** an anti-diagonal wavefront over 4x4 transform blocks (each depends on left, above and above-right):
  - step k covers the blocks with 2 * row + col == k;
  - each step can run hundreds of threads, against at most 5 now;
  - the step count grows, so the launches are batched per superblock row.

### S3 Tile parse
The bool decoder is serial within a tile; libvpx parallelises only across tiles. Speed comes from:
- making the serial thread do less: dequantisation and coefficient scatter move into the reconstruction threads,
  and the parse writes compact tokens;
- overlapping parse(N+1) with recon(N) when the next frame does not depend on this frame's backward adaptation
  (`frame_parallel_decoding_mode`, error-resilient streams; ComfyUI's encoder settings to be checked).

### S4 WebGPU / WebGL
- These depend on ILGPU emitting view-taking helpers as WGSL / GLSL functions (Data, SpawnDev.ILGPU
  Plans/view-helpers-as-functions-2026-11-09.md).
- Until then the `GiantShader` fence stays.
- `Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget` tracks the kernel sizes.

## Order
S1 first: contained, bit-exact by construction, and about a third of the frame time. Then S2 inter, S2 intra, S3.
