# AV1 v3 GPU Encoder + Decoder Architecture

**Status:** Shipped 2026-04-29. AV1 v1 keyframe encoder + decoder running
100% on ILGPU; bit-exact vs CPU `Av1KeyframeEncoder.EncodeSingleTile`
across CUDA + OpenCL + CPU. Round-trips with companion GPU decoder.

**Status 2026-11-13: tiles. One GPU thread per tile, every tile of every frame in the same launches.**

Before this the walker was one GPU thread per frame: 832x480 took 766-799 ms on CUDA (RTX 4070), 1.25-1.33 keyframes
per second, about 8 ms per 64x64 superblock. An AV1 tile is independent by definition (its own range coder, its own
contexts, no prediction across its edge), so tiles are the unit of parallelism:

- `Av1KeyframeEncoderGpu.TileSizeSuperblocks` (default 1): the largest tile size in 64x64 superblocks per side.
  1 = one superblock per tile (the most threads); `int.MaxValue` = the fewest tiles the picture size allows (one
  tile up to 4096 wide and 4096 x 2304 in area: the old layout). `Av1TileLayout.Compute` turns it into the uniform
  tile grid of spec 5.9.15 (`TileColsLog2`, `TileRowsLog2` and their codable range). Uniform spacing only halves, so
  a request of 3 on 13 superblock columns gives tiles of 2.
- `Av1KeyframeEncoderGpu.MaxTiles` (default 256 = `Av1TileLayout.HardwareDecoderMaxTiles`): the most tiles of a
  frame. Measured with FFmpeg 8.1 on the RTX 4070 (keyframes of this encoder, one superblock per tile, output
  compared byte for byte with libdav1d's): NVDEC (`av1_cuvid`), D3D11VA and D3D12VA decode 104, 135, 192, 240, 252
  and 256 tiles (256 also as 64 x 4 and 4 x 64) and FAIL at 272, 320 and 510 (NVDEC repeats
  `cuvidDecodePicture failed` without end; D3D: `HW accel decode slice fail`). libdav1d and libaom decode all of
  them, and 2040 tiles at 3840x2160. So when one superblock per tile gives more than `MaxTiles`, the layout is the
  one with the smallest tiles (in superblocks) that stays within it: 1920x1080 = 15 x 17 = 255 tiles of 2 x 1
  superblocks (not 30 x 17 = 510), 3840x2160 = 15 x 17 tiles of 4 x 2. Raise `MaxTiles` only for streams a software
  decoder reads. AV1 LEVEL limits (8 to 128 tiles, 4 to 16 tile columns by level) are not applied: none of the
  six decoders enforces them (832x480 in 13 tile columns decodes on all), and the sequence header signals
  `seq_level_idx` 0 for every picture size, which was already inaccurate above 147,456 pixels before the tiles
  (open: signal the level the picture size needs; the writer has no tier bit for levels above 3.3 yet).
- Kernel (`Av1FrameSequentialEncodeKernel`, ONE kernel for single frames and batches, where there were two):
  working thread = `frame * tileCount + tile`. The thread takes its own copy of the parameters with the tile
  rectangle; the above-context arrays are per tile (`AboveStride` = tile width in mi) and their offsets are rebased
  by `-TileMiColStart`, so the walker's `offset + miCol` indexing is unchanged. A block has an above / left
  neighbour only inside its tile, and the above-right / below-left edge extensions stop at the tile's edge (no
  thread reads pixels another thread is writing). The left contexts reset at the first superblock of every
  superblock row of the tile. Frame source and reconstruction are shared; each tile writes its own rectangle.
- Thread placement (`SerialThreadPlacement`): an auto-grouped launch puts threads 0..N-1 into the first launch
  group, i.e. on ONE multiprocessor (CUDA group size for this kernel: 640), 32 divergent threads per warp. The
  placement spreads the working threads: one launch group per multiprocessor, one thread per warp while there is
  room, then more lanes per warp. The launch covers more threads; the ones without work return at once. Applied on
  CUDA and OpenCL NVIDIA with the device's multiprocessor count and warp size, and on WebGPU (group 64, no
  multiprocessor count reported: 64 groups assumed, two halves per group). Measured on the RTX 4070 only. Packed on
  the CPU accelerators and for OpenCL devices of other vendors (not measured).
- Tile group (`Av1ObuAssembleKernel`, two launches per frame): a one-thread kernel sums the tile lengths into
  offsets; the byte kernel writes the Frame OBU size, the header blob, one 0x00 byte when there is more than one
  tile (`tile_start_and_end_present_flag` = 0 and the byte alignment), and for every tile but the last a 4-byte
  little-endian `tile_size_minus_1` before its bytes. An output byte finds its tile by a binary search of the
  offsets. The frame header codes `context_update_tile_id` = 0 and `tile_size_bytes_minus_1` = 3.
- CPU reference (`Av1KeyframeEncoder.EncodeKeyFrame(..., tileSizeSuperblocks)`): the same tiles, one after another.
- Measured 2026-11-13, RTX 4070, 832x480 q40, planes on the device (`av1-encode-bench`, ms per keyframe, median of
  12; one run each):

  | | CUDA | OpenCL | bytes |
  |---|---|---|---|
  | tile max (1 tile, the old layout) | 791.2 | 450.3 | 264,629 |
  | tile 4 (8 tiles) | 133.2 | 76.6 | 264,819 |
  | tile 2 (28 tiles) | 33.1 | 20.1 | 265,103 |
  | tile 1 (104 tiles), the default | 8.73 | 5.09 | 265,749 (+0.42%) |
  | tile 1, batch of 4 frames | 2.97 | 1.89 | |
  | tile 1, batch of 16 | 1.15 | 0.88 | |
  | tile 1, batch of 64 | 0.82 | 0.79 | |
  | 1280x720 tile 1 (240 tiles) | 9.56 | 5.56 | 613,182 |
  | 1920x1080 default (255 tiles of 2 x 1 superblocks) | 19.3 | 11.1 | 1,379,100 |
  | 1920x1080 default, batch of 8 | 4.43 | 3.46 | |
  | 1920x1080, `MaxTiles` raised (510 tiles; software decoders only) | 10.6 | 6.10 | 1,380,909 |
  | 3840x2160 default (255 tiles of 4 x 2 superblocks) | 73.8 | 44.7 | 5,500,180 |
  | 64x64 (one tile) | 8.63 | 4.98 | |

  Without the placement (tile threads packed into the first group) the same runs took: CUDA tile 1 20.5 ms, batch
  of 16 5.64 ms per frame, 1920x1080 61.1 ms; OpenCL tile 1 15.8 ms, batch of 16 2.04 ms.
- WebGPU (Chrome 157 D3D12, same GPU, `Av1KeyframeEncoderGpu_Tiles_AreFasterThanOneTile`, fastest of 3): one tile
  1828 ms per keyframe (0.55 frames per second), 104 tiles 26.5 ms (38 per second), batch of 8: 4.0 ms per frame.
  Packed, the 104 tiles took 40.6 ms and the batch 5.4 ms per frame. Placement experiment (single / batch of 8, ms
  per frame): packed 41.3 / 5.1, one thread per group 27.4 / 8.4, two per group in separate halves 26.5 / 5.7, four
  per group 25.1 / 3.9, 16 per group 28.7 / 3.9.
- Wasm (same test): one tile 834 ms, 104 tiles 177 ms, batch of 8: 176 ms per frame.
- **Count-leading-zeros in the range coder (later the same day).** `IlogNz` (the bit length of the range, once per
  coded symbol) and the Golomb length were bit-by-bit loops, up to 16 iterations per symbol. NVIDIA's OpenCL
  compiler replaced the loop by one `clz` (its PTX, read back with `av1-encode-bench --dump-kernel`, has 48 `clz`
  and no such loop); ILGPU's PTX, WGSL and Wasm kept the loop. Now `BitOperations.LeadingZeroCount` (one
  instruction on every backend). Same bytes. ms per keyframe, 832x480 q40, before -> after:
  CUDA 8.73 -> 6.31 (batch of 16: 1.15 -> 0.97; one tile 791 -> 570; 1920x1080 19.3 -> 14.1), OpenCL 5.09 -> 5.09,
  WebGPU 26.5 -> 22.7 (batch of 8: 3.99 -> 3.52; one tile 1828 -> 1496), Wasm 177 -> 114. The table above is the
  state before this change. CUDA is now 1.24x slower than OpenCL (was 1.7x). The same kind of loop may sit in the
  VP8 / VP9 coders (`Vp9InterEntropyKernel`, the bool coders' normalization): worth the same check.
- Where a superblock's time goes on one CUDA thread (64x64 picture, after the change): 3.7 ms with no coefficients
  to code (q255: prediction, transforms, quantization, reconstruction: 0.23 ms per 16x16 leaf) and 6.2 ms at q40
  (0.15 ms of entropy coding per leaf on top). A call's own overhead is 0.18 ms (8x8 picture).
- What the numbers say is left:
  - A frame costs what ONE superblock costs on one GPU thread (6.2 ms CUDA, 4.8 ms OpenCL, about 14 ms WebGPU: 16
    leaves in 4 launches). Faster than that needs parallelism inside a superblock: the 16x16 leaves depend on each
    other through DC prediction (7 anti-diagonal waves of leaves instead of 16 steps) and the entropy contexts,
    which with CDF updates disabled depend only on the neighbours' coefficients, so the transform stage could run
    as a wavefront of leaf threads and the range coder as a separate serial pass. Not started.
  - CUDA is 1.24x slower than OpenCL for the same kernel on the same GPU (3.7 vs 2.6 ms per superblock without
    coefficients), WebGPU 2.6x slower than CUDA: code generation questions for SpawnDev.ILGPU, open. Seen in the
    PTX so far: 51 signed divisions by a constant power of two left as `div.s32`, and the quantizer divides every
    coefficient by the quantizer step (384 divisions per leaf).
  - The CPU accelerators (DONE the same day, `SerialThreadPlacement.OnePerGroup`): packed, the 104 tile threads did
    not run in parallel. ILGPU's CPU accelerator runs a launch of up to 256 threads inline on the calling thread
    (`CPUAccelerator.LaneLoopInlineMaxWorkItems`) and a group of 8 on one core; Velocity runs a group of 4 as the
    lanes of one vector, in lock-step, which for four different tiles is slower than one after another. One
    working thread per group (and, on the CPU accelerator, a launch above the inline limit), 832x480 q40, Ryzen 5
    7500F (6 cores), steady state, ms per keyframe: CPU accelerator 66.6 -> 20.5 (one tile 68.4), Velocity 134 ->
    20.5 to 28.5 (one tile 69.0); 256x256: CPU 7.9; 1920x1080: CPU 103, Velocity 133. Wasm is left packed (104
    tiles 110 ms against 721 ms for one tile).
    Small pictures on the CPU accelerators looked far slower in a short benchmark (256x256 as one tile: 209 ms):
    that was the .NET runtime still optimizing the code. Measured on the CPU accelerator: 256x256 one tile 209 ms
    after 1 warm-up frame, 11.2 ms after 20; 64x64 14 ms after 2, 11 ms after 20, 0.88 ms after 200. Use
    `--warmup 20` or more on these two backends, and more for small pictures.
    Left: 3.3x on 6 cores (12 threads), so the cores are not all busy; not investigated.
  - Single tile is 3-5% slower than before the change on CUDA (791 vs 753-766 ms): not investigated.
- Oracles (`CodecsTestBase.Av1KeyframeEncoderGpuTileTests.cs`): FFmpeg (libdav1d) decodes the tiled temporal unit to
  the encoder's reconstruction byte for byte (130x66, 200x9, 320x200, 832x480, 1088x1024 at tile sizes 1 / 2 / 3 /
  max); the kernel's bytes equal the CPU reference's for the same tile size; batches equal single-frame encodes;
  `Av1TileLayout` against values derived by hand; `SerialThreadPlacement` runs every working thread exactly once;
  the default layout at 1088x1024 (144 tiles, not 272) decodes on libaom and on every hardware decoder FFmpeg has on
  the machine; tiles are at least 4x faster than one tile on the GPU backends (`..._AreFasterThanOneTile`, which
  also writes the times to the run log on every lane). 28 red-checks (mutations that must fail a test), all red.
- Run 2026-11-13 on the final code: CPU accelerator 35 / 0, Velocity 20 / 0, CUDA 20 / 0, OpenCL 20 / 0 (encoder,
  layout and placement tests), WebGPU 150 / 0 (every Av1 test), Wasm 16 / 0 (encoder tests); before the tile limit
  was added, every Av1 test: CPU 298 / 0, CUDA 153 / 0, OpenCL 153 / 0.

**Status 2026-11-12: the encoder takes any picture size** (the 2026-05-04 plan below is done, with one difference:
the leaves at the edge are 8x8, not rectangular). What it does at the bottom / right edge:

- Coded size = each dimension rounded up to 8 (`MiCols` / `MiRows` are even). Tight 4:2:0 input planes: luma
  `width x height`, chroma `((width + 1) / 2) x ((height + 1) / 2)`.
- A block the walker splits (64x64, 32x32, and a 16x16 that crosses the edge) codes the AV1 edge syntax (spec
  5.11.4): the full partition symbol when its lower and right halves both start inside the picture,
  `split_or_horz` / `split_or_vert` (SPLIT = 1) when one of them is outside, nothing when both are. The two
  2-symbol CDFs per partition context are derived on the host from the partition CDF rows (libaom
  `partition_gather_vert_alike` / `partition_gather_horz_alike`) and appended to the ushort constants
  (`PartitionSplitOrHorzCdfOffset`, `PartitionSplitOrVertCdfOffset`).
- A 16x16 crossing the edge cannot be `PARTITION_NONE`; it splits into `BLOCK_8X8` leaves (always whole, because
  the coded size is a multiple of 8): luma TX_8X8, chroma TX_4X4 (the Tx4x4 tables prepared in d75c0bc1).
  Rectangular leaves (16x8 / 8x16) would need the rectangular transforms and their scans; 8x8 leaves reuse what
  exists and cost a few bits on one row / column of blocks.
- Source pixels past the picture (up to 7 columns / rows) repeat its last column / row (a clamp in the residual
  loop, no padded copy). Reconstruction planes have the coded size; `ReadReconAsync` crops them to the picture
  on the accelerator.
- Coefficient contexts of chroma are kept at luma-mi positions, both mi of every chroma 4x4 unit written. With
  16x16 leaves only, writing and reading just the first half gave the same contexts; an 8x8 leaf under the right
  half of a 16x16 reads the second half (red-checked: the old layout decodes to garbage at 72x40, 70x50, 17x33).
- Shader size: the leaf pipeline has ONE call site and three `EncodePlane` sites with constant transform sizes
  (16: luma of a 16x16; 8: chroma of a 16x16 and luma of an 8x8; 4: chroma of an 8x8). Fully inlined WGSL
  43,065 -> 46,123 lines (a first version with a separate edge-leaf path was 61,833).
- Oracles: FFmpeg (libdav1d) decodes the temporal unit to the encoder's reconstruction byte for byte
  (`Av1KeyframeEncoderGpu_AnySize_FfmpegDecodesWhatTheEncoderReconstructs`, desktop lanes), the kernel's bytes
  equal the CPU reference's (`..._AnySize_MatchesCpuReference`, every lane that runs serial stream kernels), and
  the CPU reference itself decodes in FFmpeg (`Av1KeyframeEncoder_AnySize_FfmpegDecodesThePicture`).
  Run on c5884c96 (2026-11-12): CPU accelerator, Velocity, CUDA, OpenCL and WebGPU green.
- Cost, measured (Chrome 157 D3D12, RTX 4070, cold, 3 interleaved reps): the encode kernels compile in 9.7 s
  (single) and 11.6 s (batch), up from 7.7 s and 8.0 s, GPU-process peak 1.28 GB (was 0.96). 18,159 of the 46,123
  inlined lines are emulated 64-bit arithmetic, mostly the transform butterflies (`HalfBtf`).
- NOT done: the GPU decoder (`Av1KeyframeDecoderGpu`) still takes multiples of 64 only.

**Status 2026-05-04 (superseded by the above):** AV1 GPU encoder strictly requires width AND height
to be multiples of 64. Non-aligned dim support (e.g. FHD 1920x1080) is the
next major item - VP9 just shipped its boundary forced-split (2026-05-04
walker + sequential encode work at `Vp9FrameEntropyKernel` +
`Vp9FrameSequentialEncodeKernel`); AV1 follows the same template:

- Walker boundary detection at every partition level (SB128/SB64/SB32/SB16).
- AV1 spec sec 5.11.4: at boundary blocks, emit forced-partition CDFs (or
  no emit for both-edge corner). PARTITION_HORZ at the bottom-edge SB16.
- Sequential encode produces 2 Tx8x8 luma + 2 Tx4x4 chroma for boundary MBs.
- AV1 differences from VP9: range encoder (not bool), wider partition CDF
  tree (10 outcomes vs 4), 4x4 mi units (not 8x8). Constants buffer needs
  Tx4x4 entries (similar to what `Vp9KeyframeConstantsGpu` got).

This document describes the v3 100% ILGPU walker pattern used for
`Av1FrameSequentialEncodeKernel` + `Av1FrameSequentialDecodeKernel`.

## Pipeline overview

The host (`Av1KeyframeEncoderGpu`) is a pure coordinator:
1. Allocate GPU buffers (src, recon, tile bytes, scratch).
2. Upload YUV source + constant tables (CDFs, scan, dequant).
3. Dispatch a single kernel invocation per frame.
4. Read back tile bytes + recon planes.

OBU framing (TD + SH + Frame OBU header) is metadata struct setup +
fixed config bit-packing per the CARDINAL rule's "metadata struct setup"
allowance; it stays on host via `Av1KeyframeEncoder.EncodeKeyFrameWithExternalTile`.

### Encoder kernel pipeline

`Av1FrameSequentialEncodeKernel` runs as a single GPU thread:
- For each 64x64 superblock in raster order:
  - Reset left context arrays (entropy, partition, mode, skip).
  - `EncodeSuperblock(64x64 at miRow, miCol)`:
    - Emit partition CDF -> SPLIT.
    - For each 32x32 sub-block:
      - Emit partition CDF -> SPLIT.
      - For each 16x16 sub-block:
        - Emit partition CDF -> NONE.
        - `EncodeLeafBlock(16x16 at miRow, miCol)`:
          - Emit skip CDF (=0).
          - Emit Y mode KF CDF (=DC).
          - Emit UV mode CDF (=DC).
          - `EncodePlane(Y, TX_16X16)` - predict + residual + forward 2D DCT
            + quantize + WriteCoeffsTxb + dequant + iDCT + add to predict
            + clip + recon.
          - `EncodePlane(U, TX_8X8)`.
          - `EncodePlane(V, TX_8X8)`.
          - Update mode + skip arrays.
        - Update partition context.
- `Av1RangeEncoderGpu.Done` -> write tileLen.

### Decoder kernel pipeline

`Av1FrameSequentialDecodeKernel` is the symmetric inverse:
- For each SB, decode partition CDFs then descend.
- For each leaf, decode skip + ymode + uvmode.
- For each plane: `Av1CoefDecoderGpu.ReadCoeffsTxb` (returns dequantized
  coefs directly) -> build edge -> DC predict -> inverse 2D transform
  -> add residual to predict -> clip -> recon.

## Kernel argument budget

ILGPU's `LoadAutoGroupedStreamKernel` ceiling is 14-15 generic args.
The walker packs scalar params + scratch offsets into a struct:
- `Av1FrameSeqEncodeParams` / `Av1FrameSeqDecodeParams`
- 11 ArrayView args + 1 struct arg = 12 generic args (fits comfortably).

Distinct outputs from `WriteCoeffsTxb` (eob + culLevel) are routed via
`scratchInt.SubView(EobSlot, 1)` and `scratchInt.SubView(CulLevelSlot, 1)`
since otherwise they alias the same backing storage.

## Scratch buffer layout

### scratchByte (per-frame state arrays, stable across SBs)

| Region                    | Size                     | Purpose                                |
|---------------------------|--------------------------|----------------------------------------|
| `AboveEntropyOff..`       | 3 planes * frameMiCols   | Above coeff cul_level + dc_sign packed |
| `LeftEntropyOff..`        | 3 planes * 32            | Left ditto, reset per SB row           |
| `AbovePartOff..`          | frameMiCols              | Partition context above bytes          |
| `LeftPartOff..`           | 32                       | Partition context left, reset per SB row|
| `AboveYModeOff..`         | frameMiCols              | Above intra mode (byte = enum value)   |
| `LeftYModeOff..`          | 32                       | Left ditto                             |
| `AboveSkipOff..`          | frameMiCols              | Above skip flag                        |
| `LeftSkipOff..`           | 32                       | Left ditto                             |
| `EdgeAboveOff..`          | 33                       | Per-block edge buffer (above)          |
| `EdgeLeftOff..`           | 33                       | Per-block edge buffer (left)           |
| `PredictOff..`            | 256                      | Per-block predict buffer               |
| `LevelsOff..`             | 1384                     | libaom levels[] padded scratch         |

### scratchInt (per-block working area, reused)

| Range                           | Purpose                                        |
|---------------------------------|------------------------------------------------|
| `[0..n)`                        | coefs (forward output, then quantized)         |
| `[n..2n)`                       | dequant output (overwrites forward scratch)    |
| `[2n..3n)`                      | inverse transform residual output              |
| `[3n..3n+n+16)`                 | inverse transform scratch (272 max for Tx16x16)|
| `[EobSlot=1100]`                | eob output from WriteCoeffsTxb                 |
| `[CulLevelSlot=1101]`           | culLevel output                                |

`MinScratchIntLength = 1102` covers the worst-case Tx16x16 footprint.

### scratchShort (per-block, reused)

| Range       | Purpose                  |
|-------------|--------------------------|
| `[0..n)`    | residual (short, pre-DCT)|

## Constants buffer

`Av1KeyframeConstantsGpu` packs every CDF table the walker needs into
two flat buffers (uploaded once per accelerator, reused across every
frame):

**Byte buffer (~380 bytes):** NzMapCtxOffset (Tx8x8 + Tx16x16),
EobGroupStart, EobOffsetBits, IntraModeContext.

**Ushort buffer (~7300 entries / ~14 KB):** Scan tables (Tx8x8 + Tx16x16
DCT_DCT), TxbSkipCdf, EobMulti64Cdf, EobMulti256Cdf, EobExtraCdf,
CoeffBaseEobMultiCdf, CoeffBaseMultiCdf, CoeffLpsMultiCdf, DcSignCdf,
IntraExtTxCdf (Tx8x8 + Tx16x16 DC), SkipTxfmCdf, PartitionCdf,
KfYModeCdf, UvModeCdfV1Row.

`BuildByteConstsBuffer()` and `BuildUshortConstsBuffer()` produce the
flat byte arrays from the existing `Av1Default*Cdfs` tables.

## v1 simplifications

Mirrors `Av1KeyframeEncoder.cs`:
- Profile 0 (8-bit 4:2:0).
- Any width and height (edge handling: see the 2026-11-12 status at the top).
- DC_PRED only; leaves BLOCK_16X16, BLOCK_8X8 where a 16x16 crosses the picture edge.
- TX_16X16 for Y, TX_8X8 for UV (8x8 leaves: TX_8X8 for Y, TX_4X4 for UV).
- DCT_DCT only.
- `tx_mode = LARGEST`, `reduced_tx_set = 1`.
- Single tile (no per-tile size prefix).
- `disable_cdf_update = 1` (we use static default CDFs throughout).

## Bug found during bring-up

`GetQctx` quantizer-bin thresholds were `[32, 128, 192]` instead of the
correct libaom `[20, 60, 120]` per `Av1CoefDecoder.GetQctx`. This shifted
every per-block CDF row read for the txb_skip + eob + coeff path, making
the GPU walker emit 18 bytes where CPU produced 21 for the same input.

**Diagnostic technique:** Added a temporary trace to
`Av1RangeEncoder.EncodeCdfQ15` capturing `(sym, fl, fh, nsyms)` per
emit. Built a CPU-shadow walker that called the same `Av1RangeEncoder`
with the GPU walker's exact logic. Comparing emit-by-emit traces
pinpointed the divergence at emit #6 (first per-plane txb_skip CDF):
CPU had `fl=867`, shadow had `fl=2811`. Both lookup `DefaultTxbSkipCdf`
at row index 0 of `txsCtx=2`, but `qctx` selected the wrong outer
index. Reading `Av1CoefDecoder.GetQctx` revealed the threshold mismatch.

## Test surface

`CodecsTestBase.Av1KeyframeEncoderGpuTests`:
- `ConstGray64x64` - 1 SB, all eob=0 path.
- `Random64x64` - full coef chain stress.
- `ConstGray64x128` - multi-SB walker.
- `FullKeyFrame_Random64x64_MatchesCpu` - TD + SH + Frame OBU bit-exact
  vs `Av1KeyframeEncoder.EncodeKeyFrame`.

`CodecsTestBase.Av1KeyframeDecoderGpuTests`:
- `ConstGray64x64_RoundTrip` - encode -> decode -> recon all-128.
- `Random64x64_MatchesEncoderRecon` - encode -> decode -> decoder recon
  matches encoder's internal recon.

Total: 18 test method invocations across 3 backends = 18 PMT pass
counts. Plus 696/696 AV1 sweep PASS confirms zero regressions across
the existing AV1 primitive tests.

## Future work

- **Multi-frame** sequences (intra-only stream with multiple keyframes).
- **OBU framing on GPU** - port TD/SH/uncompressed-header bit-writers
  to GPU kernels so the host coordinator no longer touches OBU bytes.
- **Wave-parallel walker** - libaom `tile_groups` + `intra-only` rows
  allow per-row parallelism within a tile; v3 walker is single-thread.
- **Inter frames** - motion compensation + reference frame management
  + delta_q signaling + CDF adaptation.
