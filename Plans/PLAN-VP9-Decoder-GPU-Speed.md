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
- CPU accelerator tile parse (ComfyUI), quiet A/B (base, S3, base, S3):
  - 3.32 / 3.23 -> 2.67 / 2.68 ms/frame (-18%);
  - without the first frame (kernel compile): 2.34 / 2.31 -> 2.00 / 1.97 (-15%);
  - the whole test is not slower (base 104.8 / 103.2 s, S3 102.2 / 102.3 s).
- CUDA / OpenCL: to be measured.

Next for S3:
- Mode info (9% of the bools, but many neighbour loads per block): the arguments and neighbour records in locals.
- Frame overlap for frame-parallel streams. This needs the mode info / superblock / coefficient buffers double-buffered
  and a second stream; it can hide at most reconstruction + loop filter.

### S4 WebGPU / WebGL
- These depend on ILGPU emitting view-taking helpers as WGSL / GLSL functions (Data, SpawnDev.ILGPU
  Plans/view-helpers-as-functions-2026-11-09.md).
- Until then the `GiantShader` fence stays.
- `Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget` tracks the kernel sizes.
- What counts on Chrome / Windows is the size after full inlining: D3D's compiler (DXC) inlines every WGSL function,
  and making helpers functions ([NoInlining]) made DXC slower, not faster (reverted, f202c88). So the work is fewer
  call sites, not fewer functions.
- Compressed header, done 2026-11-09 (3a20c7c): table-driven, one DiffUpdate call site. WGSL 119,048 -> 7,739 lines,
  DXC 39 s -> 0.9 s.
- Tile parse kernel, done 2026-11-09. Measured on SpawnDev.ILGPU 5.4.23-local.3 (VP9_WGSL_DUMP + an offline inliner
  over the WGSL):
  - Before: 113,564 WGSL lines, but `ParseBlock` was a 111,113-line WGSL function called from 6 sites, so about
    762,800 lines after inlining. Inside it every bool read was inlined at hundreds of sites (5,115 calls of the
    64-bit shift emulation alone).
  - After: 11,464 WGSL lines, about 11,900 fully inlined (64x smaller); 24 bool reads in the whole kernel.
  - What multiplied, and what was done (each body now appears once):
    - `ParseBlock` x 7 sites in the partition walk -> one site in a loop over the partition's blocks.
    - `FindMvRefs` x 8 (`AppendSub8x8Mvs` x 6, NEWMV, 8x8 up), each with 4 `Candidate` and 8 add-to-list sites -> one
      site in one pass per sub-block; inside, libvpx's four searches are one ordered stream of 27 candidates with one
      add-to-list site.
    - `ReadMv` x 4 (8 `ReadMvComponent`, 8 `IncMvComponent`) -> one site; the components are a loop.
    - The keyframe and inter-frame mode info share one sequence (segment id, skip, is_inter, tx size, modes); the y
      and uv modes are steps of one loop over one tree read; reference frames, the segment id's prediction flag and
      tree, the inter mode and the interp filter are loops over one read each.
    - One tree reader (`ReadSym`) with the probability source per value (context / tables / arguments, each buffer
      indexed in its own branch); the partition's edge cases and the coefficient tokens from TWO up are trees in the
      tables (`Vp9PScan`), so `DecodeCoefs` has 6 bool reads instead of 19 (EOB / ZERO / ONE stay unrolled).
    - WGSL's structured control flow copies code: an early return in an inlined helper repeated the helper's tail
      (`ReadInterBlockModeInfo` after its segment-skip return, about 3.3k lines; `DecodeCoefs` after its return inside
      the zero-token loop), and a short-circuit condition in front of a body repeated the body (the refill loop of
      every bool read, `Vp8BoolDecoderGpu.Fill`). Helpers are single-exit and side-effect-free conditions use & / |.
  - CPU accelerator tile parse (ComfyUI, A/B back to back, without the first frame): base 1.91 / 2.00 ms/frame, new
    1.93: unchanged. CUDA and the Chrome / DXC compile: to be measured.
  - Next if needed: the reference-frame contexts (about 1.8k of the 11.9k lines) load the same neighbour fields many
    times; loading them once into locals would shrink them.
- Reconstruction and loop filter: 32-bit offsets, done 2026-11-10. Their pixel, coefficient and scratch offsets were
  `long`, so in WGSL every tap was emulated 64-bit math (helper calls: Wavefront 10,881, IntraWave 6,882,
  InterResidual 5,702, InterPredict 4,621). Now `int`, with host checks that the frame buffers and the scratch fit.
  Fully inlined lines: Wavefront 188,891 -> 116,468, InterPredict 134,149 -> 75,515, InterResidual 101,910 -> 56,457,
  IntraWave 115,872 -> 60,538.
  - Data's `||` dedup fix in ILGPU (5ec0dba: InterPredict 75,562 -> 31,972 WGSL lines on 21d8906) stacks on top.
- Loop filter call sites, done 2026-11-10: one filter body (`LpfAt`, 4 / 8 / 16 sharing the mask and flat path), one
  call site per filter-selectively function (libvpx's calls in order from a loop over call slots), one per pass for
  all plane layouts. Wavefront 9.0 MB -> 0.55 MB, fully inlined 116,468 -> about 11,000 lines (was about 96 filter
  copies).
  - Next if needed: its 64-bit edge masks (`Vp9LfMask`, libvpx's uint64 masks; about 450 emulated calls left) as two
    32-bit halves. Then InterPredict (75k lines; 32k with the dedup fix) is the largest kernel: the same call-site
    treatment for its convolve paths.

## Order
S1 first: contained, bit-exact by construction, and about a third of the frame time. Then S2 inter, S2 intra, S3.
