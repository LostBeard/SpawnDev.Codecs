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

Built 2026-11-09 (`Vp9Lpf.FilterSuperblockGroup`): lane 0 builds the masks into shared memory, barrier, each lane
its luma row plus its chroma rows (U then V), barrier, the same for columns. Measured over the 6 ComfyUI files
(ms per frame of loop filter):
- CUDA: 16.5 -> 1.35 (12x).
- OpenCL: 1.0.
- CPU accelerator: 3.5 -> 351. ILGPU's sequential CPU processor hands off between lanes at every barrier; this is
  with Data.
- Velocity: about 114 -> 133.
Bit-exact on CPU, CUDA and OpenCL.

### S2 Reconstruction at transform-block granularity
- **Inter prediction:**
  - One thread per (plane, 4x4 unit) of every inter block, using the mode-info grid. The sub-pel filter is per
    pixel, so this goes from 40 threads to about 15k for 512x320.
  - Then the residual: one thread per transform block (later per row / column of the 1-D transforms).
  - Built 2026-11-09 (inter half):
    - `InterPredictKernel`: one thread per (plane, 4x4 unit) over the superblock grid. `PredictBlockPlaneRegion`
      does the libvpx call's setup, then runs the same convolve over only its 4x4 part (origin moved by
      `(subpel + offset * step) >> 4`), so the arithmetic per pixel is the same.
    - `InterResidualKernel`: one thread per (plane, 4x4 unit) of the mi grid; the transform block's top-left unit does
      the transform.
    - Offsets: the block-table slot is stored in the mode info (`Vp9Mi.TableSlot`). The j-th transform block of a
      plane is at block start + j * (16 << 2 tx) coefficients and j eobs.
    - Bit-exact on the CPU accelerator: EveryFrame (311 files), PublicApi, and the inter DSP test, now also per unit.
      Red-checked (wrong U offset, residual before prediction).
    - CPU accelerator reconstruction (ComfyUI files): 9.1 -> 4.4 ms/frame.
    - Reconstruction (ComfyUI files), ms per frame: CUDA 16.1 -> 2.10, OpenCL 1.46. Bit-exact on both.
    - The frame on CUDA is now about 23 ms; the one-thread tile parse is 19 ms of it, so S3 is next.
    - Next: one 32x32 transform is still one thread (a row / column split is the follow-up).
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

Measured 2026-11-09 (CPU accelerator, temporary counters in the parse kernel, 6 ComfyUI files / 252 frames):

| per frame (mean) | count | share |
|---|---|---|
| bool decodes | 97.6k | |
| - coefficient tokens | 87.1k | 89.2% |
| - mode info (segment, skip, tx, modes, refs, MVs) | 9.0k | 9.2% |
| - partitions | 1.5k | 1.5% |
| EOB checks / ZERO tokens / non-zero tokens (TWO+) | 12.3k / 29.0k / 9.8k (4.5k) | |
| transform blocks / blocks / partitions | 2.6k / 679 / 864 | |
| blocks: skip / inter / intra (inter with an MV search) | 119 / 611 / 68 (531) | |
| bitstream bytes read by the refill | 7.3k | |

- The parse time is the coefficient loop: about 39k coefficient positions per frame against 679 blocks.
- Header flags (libvpx_ref -h, `libvpx_headers.tsv.gz`):
  - ComfyUI: all 252 frames frame-parallel (no backward adaptation, counts off).
  - Vectors: 1909 of 4858 frames (1470 error-resilient, the rest frame-parallel); only 59 of 305 files on every frame.
- Before, per coefficient the thread made about 8 to 10 device loads, each waited on in order:
  - the band;
  - the three probabilities, one after each bool;
  - the scan position, twice for a non-zero token;
  - two neighbour indices;
  - two token-cache reads of entries it had just written.
  ILGPU has no pass that removes repeated loads, and the views are plain global pointers, so the GPU's assembler cannot
  move a load above the token-cache / coefficient stores between them.
- Dequantisation and the coefficient scatter cost one multiply and one store per non-zero coefficient. A store does not
  stall the thread, so moving them out (the first idea above) would barely shorten the serial chain. Not done.

Built 2026-11-09: the coefficient loop on local memory (`Vp9ParseKernels.DecodeCoefs`):
- One packed scan entry per coefficient (`Vp9PScan`: position | neighbour 0 << 10 | neighbour 1 << 20), fetched one
  coefficient ahead.
- The band computed (`CoefBand`, checked against libvpx's tables when the table is built).
- The three node probabilities of a context loaded together.
- The token cache, the left entropy contexts, the above / left partition contexts and the partition stack in the
  thread's local memory (`Vp9PLocal`, 6.5 KB; a tile column is at most 64 superblocks).
  - The above entropy contexts (another 12 KB as ints) stay in the work buffer, so the locals fit WGSL's 8192-byte limit
    on a function's variables (S4).
  - A version with them local too measured 2.40 ms/frame on the CPU accelerator.
- The bool read's renormalising shift from a leading zero count instead of a compare chain.
- Dequant factors read once per plane of a block instead of per transform block.

Results:
- Bit-exact on the CPU accelerator:
  - EveryFrame: 311 files, 5112 frames, 5000 outputs == libvpx .md5.
  - Parse_ModeInfo: 5112 frames, 2951 with counts checked.
  - PublicApi: 332 pictures.
- Red-checked (prefetch of the wrong entry, wrong energy class, stale probability: all fail).
- ParseKernel WGSL: 87.1 -> 72.9 MB.
- CPU accelerator tile parse (ComfyUI): 3.57 -> 2.59 ms/frame (3.57 = the counter build; the plan's earlier 3.4 had
  no counters). CUDA / OpenCL: to be measured.

Next for S3:
- Mode info (9% of the bools, but many neighbour loads per block): the arguments and neighbour records in locals.
- Frame overlap for frame-parallel streams. This needs the mode info / superblock / coefficient buffers double-buffered
  and a second stream; it can hide at most reconstruction + loop filter.

### S4 WebGPU / WebGL
- These depend on ILGPU emitting view-taking helpers as WGSL / GLSL functions (Data, SpawnDev.ILGPU
  Plans/view-helpers-as-functions-2026-11-09.md).
- Until then the `GiantShader` fence stays.
- `Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget` tracks the kernel sizes.

## Order
S1 first: contained, bit-exact by construction, and about a third of the frame time. Then S2 inter, S2 intra, S3.
