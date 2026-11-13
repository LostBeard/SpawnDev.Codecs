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
  decoder reads. AV1 LEVEL limits (8 to 128 tiles, 4 to 16 tile columns by level) are not applied by default:
  none of the six decoders enforces them (832x480 in 13 tile columns decodes on all). `LevelLimit` applies them
  (next item).
- Level (`Av1Level`, AV1 spec Annex A.3; 2026-11-13). The sequence header signals the lowest level and tier the
  stream conforms to, and `seq_level_idx` 31 ("maximum parameters") when it conforms to none. Before, every stream
  said level 2.0, which was untrue above 147,456 pixels and for every grid of more than 8 tiles.
  - What decides. Known before coding: the picture (MaxPicSize, MaxHSize, MaxVSize), the frame rate
    (`SetFrameRate`, default 30: luma samples, frame headers and tiles per second), the tile grid (MaxTiles 8 to
    128, MaxTileCols 4 to 16), and what every level demands (a frame of at least 16 x 16, at least 8 pixels of the
    picture in the last tile column and row, largest tile times frame rate at most 588,251,136). Known only after
    coding: the compression ratio limit on a Frame OBU, and the bit rate (MainMbps, or HighMbps in the High tier,
    which exists from level 4.0). The bit rate is applied as a PEAK: no temporal unit larger than MaxBitrate /
    frame rate, so every unit arrives within its own frame period. That is sufficient for the decoder model of
    Annex E and, for keyframes of similar size, close to the average it needs.
  - Where it is chosen. The host puts one sequence header per possible level and tier into the framing blob, in
    order of preference (levels ascending, Main before High, 31 last), each with the largest Frame OBU it allows
    (`Av1KeyframeFraming.BuildLevelCandidates`), and the offsets thread of `Av1ObuAssembleKernel`, which already
    sums the tile lengths, picks the first one the frame fits. No extra launch and no readback. The CPU reference
    picks from the same list on the host, and the tests check the result with the specification's inequalities on
    the measured bytes (`Av1Level.FirstViolation`), not with the limits the kernel compared against.
  - Files. A file is one coded video sequence with one sequence header (av1C carries it). Each frame signals the
    level a stream of frames of ITS size conforms to, so `VideoFileEncoderGpu` passes its frame rate to the encoder
    and then gives every sample the sequence header of the frame that signalled the highest level
    (`Av1Isobmff.UnifySequenceHeaders`). The level grows with the frame size, so that is the lowest level the whole
    file conforms to. Anything else that stores several frames as one sequence has to do the same.
  - No rate control: a level limit bounds the tiles, not the bytes. A stream above the bit rate of a level signals
    the next tier or level up (lower the quality to lower the level).
  - `LevelLimit` (a `seq_level_idx`; `VideoFileEncodeOptions.Av1LevelLimit`) keeps the tile grid inside a level:
    `Av1Level.TileLayout` applies that level's MaxTiles, MaxTileCols and tiles per second, or those of the lowest
    level the picture size and frame rate need when that is higher, and keeps 8 pixels in the last tile column and
    row. Default: no limit, one superblock per tile, the fastest. Measured on CUDA (RTX 4070, ILGPU 5.4.30,
    av1-encode-bench picture, 30 fps, ms per keyframe, `--level`):
    - 832x480, no limit: 13 x 8 = 104 tiles of 1 x 1 superblocks, 6.3 ms. The grid needs 6.0; signalled 6.0 High at
      q40 (63.8 Mbit/s), 6.0 Main at q120 (33.4 Mbit/s).
    - 832x480, limit 5.1: 7 x 8 = 56 tiles of 2 x 1, 11.7 ms, 5.0 High at q40.
    - 832x480, limit 3.0: 2 x 8 = 16 tiles of 7 x 1, 41.4 ms. Signalled 5.0 High at q40, 4.1 High at q120, 3.1 Main
      at q200 (8.8 Mbit/s), 3.0 Main at q255 (2.7 Mbit/s).
    - 1920x1080, no limit: 15 x 17 = 255 tiles of 2 x 1, 13.3 ms, no level (31).
    - 1920x1080, limit 6.0: 15 x 6 = 90 tiles of 2 x 3, 35.4 ms; limit 5.1: 8 x 6 = 48 tiles of 4 x 3, 70.4 ms;
      limit 4.0: 8 x 4 = 32 tiles of 4 x 5, 120 ms. All three signal 6.1 High at q40 (330 Mbit/s); with the 4.0
      grid: 5.2 High at q120 (172.5 Mbit/s), 4.1 High at q200 (45.2 Mbit/s), 4.0 High at q255 (13.5 Mbit/s).
    What the numbers say: key frames only, at a fine quantizer, are far above the bit rate of the level their
    picture size needs, so there the bit rate sets the level and a smaller grid buys nothing. The limit pays with a
    coarse quantizer, and will pay once inter frames exist. A level-limited frame costs one tile's superblocks in
    sequence (20 at 1920x1080 level 4.0): parallelism inside a tile (wavefront, open) is what would make these
    grids fast.
  - Decoders. Keyframes of this encoder at 832x480 (levels 2.0, 3.0, 6.0, maximum), 1920x1080 (2.0, 4.0, maximum)
    and 3840x2160 (2.0, 5.1, maximum) decode to the same picture on libdav1d, libaom, NVDEC, D3D11VA and D3D12VA
    (FFmpeg 8.1, RTX 4070) whatever level is signalled, and ffprobe reports the level written. So the honest level
    does not break these decoders. Not tested: browsers and players that ask the device whether it supports the
    level before they start (the MP4 codec string carries it), other vendors' hardware decoders.
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
  - A frame cost what ONE superblock costs on one GPU thread (6.2 ms CUDA, 4.8 ms OpenCL, about 14 ms WebGPU: 16
    leaves in 4 launches). Faster than that needs parallelism inside a tile. First half DONE the same day (next
    item): the transform stage is a wavefront of leaf lanes, the entropy coding a separate serial pass.
- **Parallel inside a tile, stage 1: transform wavefront + entropy pass (2026-11-13, afternoon).** The tile walk is
  two passes now (`Av1FrameSequentialEncodeKernel`). Same bytes as before (the CPU reference did not change).
  - Transform pass. A leaf predicts (DC) from the reconstruction above it and to its left, so the 16x16 positions
    on an anti-diagonal of a tile are independent. A launch group is one tile; its lanes take the positions of a
    diagonal (lane L: positions L, L + lanes, ...), with `Group.Barrier()` between diagonals,
    `DiagonalsPerLaunch` (8) diagonals per launch. Each position: predict, forward transform, quantize,
    reconstruct; its quantized coefficients go to a device buffer (`CoefsPerSlot` = 384 ints per 16x16 position,
    indexed by the entropy pass's z-order slot). A lane's working area is device memory too, not LocalMemory.
  - Entropy pass. One thread per tile as before (`SerialThreadPlacement`), now only the partition, mode and
    coefficient syntax, reading the stored coefficients; `LeavesPerLaunch` 4 -> 16.
  - The CPU accelerators (CPU, Velocity, Wasm) have a core per tile, not per lane, and a kernel with barriers runs
    through their group machinery (Velocity: 112 ms for the transform pass of a 832x480 frame, against about 12).
    They load another entry point for the same leaf code (`Av1TransformTileKernel`): one thread per tile walks the
    diagonals alone, no barrier. An accelerator compiles one of the two.
  - The launch shape matters (`GetTransformGroup`: lanes per tile, group threads per lane, tiles per group; CUDA,
    832x480 q40, transform pass of one frame / ms per keyframe in a batch of 16). A leaf's transform code
    branches on its data, so lanes that run in step pay for each other's branches:
    one group per tile, its 4 lanes adjacent: 1.4 ms / 1.02 (the choice for few tiles); the same lanes a warp
    apart: 1.4 / 1.24; the lanes of 8 tiles filling a warp: 2.2 / 1.15; one lane per tile, 2 per warp, 40 tiles
    per group: 4.9 / 0.96 (the choice above 32 tiles per multiprocessor, where the device is full either way and
    total work counts); one lane per tile, a group each: 2.6 / 1.15. Wavefront lanes win up to a batch of 8
    (1.35 against 1.50 ms per frame) and tie from 16. Larger tiles get more lanes (1920x1080 in 32 tiles of
    16 x 20 positions: 16 lanes 8.8 ms, 8 lanes 11.9, 4 lanes 18.9).
  - Measured (RTX 4070, ILGPU 5.4.30, q40, ms per keyframe, before -> after):
    | | CUDA | OpenCL |
    |---|---|---|
    | 832x480, 104 tiles (default) | 6.31 -> 4.90 | 5.1 -> 3.98 |
    | 832x480, batch of 16, per frame | 0.92 -> 0.99 | 0.90 after |
    | 832x480, level 5.1 grid (56 tiles) | 11.7 -> 8.9 | 7.1 after |
    | 832x480, level 3.0 grid (16 tiles) | 41.4 -> 29.0 | 23.4 after |
    | 832x480, one tile | 570 -> 343 | 281 after |
    | 1920x1080, 255 tiles (default) | 13.3 -> 10.2 | 11.1 -> 8.8 |
    | 1920x1080, level 6.0 grid (90 tiles) | 35.4 -> 23.8 | 19.6 after |
    | 1920x1080, level 5.1 grid (48 tiles) | 70.4 -> 45.6 | 37.3 after |
    | 1920x1080, level 4.0 grid (32 tiles) | 120 -> 74.2 | 61.0 after |
    CPU accelerator 20.5 -> 21.4, Velocity 20.5 to 28.5 -> 22.0 (832x480, 104 tiles). WebGPU
    (`Tiles_AreFasterThanOneTile`, two runs): 104 tiles 22.7 -> 17.7 and 20.5, batch of 8: 3.52 -> 3.16 and 3.27
    per frame, one tile 1496 -> 923 and 913; Wasm: 104 tiles 114 -> 96 and 97, one tile 834 -> 320 and 302. No WebGPU device hang with the barrier kernel (the VP9
    wavefront kernel needed a wait per launch; this one keeps its working areas in device memory).
  - The batch of 16 on CUDA is 7% slower than the single-pass walker was (0.92 -> 0.99): two passes store and
    reload the coefficients, and a wavefront lane idles at a barrier. Every shape tried is in the list above.
  - Where the time goes now (`av1-encode-bench --passes 1`, CUDA): 832x480 default grid: transform 1.4 ms,
    entropy 3.4 ms; 1920x1080 level 4.0 grid: transform 7.5 ms, entropy 65.6 ms (0.21 ms per leaf, 320 leaves per
    tile in sequence). The serial entropy pass was what was left: stage 2, next item.
  - Tests: every encoder test runs through the split (bytes equal to the CPU reference, FFmpeg decodes to the
    reconstruction). New: `Av1KeyframeEncoderGpu_Wavefront_EveryLaunchShapeGivesTheReferenceBytes` (forced
    shapes: one lane, several rounds per diagonal, a lane count that is not a power of two, lanes a warp apart,
    several tiles per group) and `..._ManyTilesBatchEqualsSingleFrameEncodes` (1500 one-tile frames: the
    many-tiles shape; the choice itself asserted). 25 red-check mutations, all caught. Two findings from the
    red-checks: lanes in ONE warp advance together, so a missing barrier only shows with lanes a warp apart; and
    the encoder keeps its reconstruction buffer, so a position that reads a neighbour too early finds the right
    pixels of the previous encode of the same picture unless another picture is encoded in between (the test
    does that).
- **Parallel inside a tile, stage 2: symbol pass + range coder pass (2026-11-13, later).** The entropy pass is
  split in two; the tile thread now only runs the range coder. Same bytes.
  - Why it is possible. The keyframe writes `disable_cdf_update = 1`, so a symbol's probabilities are a fixed table
    row picked by a context, and no context depends on the range coder. The serial coder kept "above" and "left"
    arrays and updated them leaf by leaf, but every value it reads was written by the leaf directly above or
    directly to the left (zero at the tile's top row and left column). So: the coefficient contexts (txb_skip,
    dc_sign) come from the neighbour blocks' contexts (cul_level + DC sign), a function of a block's coefficients
    that the transform pass now stores next to them (12 ints per 16x16 slot, `Av1CoefEncoderGpu.ContextOf`); the
    partition contexts come from the neighbour's leaf size, which is geometry (a leaf is 16x16 unless its
    position crosses the picture edge); the skip and Y-mode contexts are constant (every leaf is DC and not
    skipped).
  - Symbol pass (`Av1SymbolizeKernel`): one thread per 16x16 position of every tile at once. The same walk as
    before (`WriteCoeffsTxbSymbols` is `WriteCoeffsTxb` with another sink), writing one 32-bit entry per symbol
    into the position's symbol slot (`Av1SymbolStreamGpu`: the two inverse CDF values shifted right by 6, which
    is all the range coder uses of them, and the EC_MIN_PROB multiple; raw bits as runs of up to 24).
  - Range coder pass (`Av1RangeCodeTileKernel`): one thread per tile feeds the entries of its slots, in order, to
    its range coder (`Av1RangeEncoderGpu.EncodeShifted`).
  - Waves. A slot is reserved at its worst case (3200 entries: at most 8 per coefficient), so the two passes
    alternate over waves of slots: all of a tile's slots, at most 64 (`MaxWaveSlots`), and no more than fit 96 MB
    for all tiles (`SymbolBudgetBytes`); the range coder is kept in device memory between waves. Measured (CUDA):
    1920x1080 level 4.0 grid, slots per wave 4 / 16 / 64 / 320: 58.5 / 45.8 / 43.2 / 46.2 ms; 832x480, 1 / 4 / 16:
    6.0 / 4.1 / 3.6 ms; batch of 16 with a budget of 24 / 96 / 400 MB: 1.14 / 1.02 / 1.07 ms per frame.
  - Measured (RTX 4070, ILGPU 5.4.30, q40, ms per keyframe; this morning -> stage 1 -> stage 2):
    | | CUDA | OpenCL |
    |---|---|---|
    | 832x480, 104 tiles (default) | 6.31 -> 4.90 -> 3.35 | 5.1 -> 3.98 -> 2.83 |
    | 832x480, batch of 16, per frame | 0.92 -> 0.99 -> 1.02 | 0.90 -> 1.00 (stage 1 -> 2) |
    | 832x480, level 5.1 grid (56 tiles) | 11.7 -> 8.9 -> 6.1 | 7.1 -> 4.9 |
    | 832x480, level 3.0 grid (16 tiles) | 41.4 -> 29.0 -> 18.0 | 23.4 -> 14.4 |
    | 832x480, one tile | 570 -> 343 -> 184 | 281 -> 152 |
    | 1920x1080, 255 tiles (default) | 13.3 -> 10.2 -> 7.8 | 11.1 -> 8.8 -> 7.0 |
    | 1920x1080, level 6.0 grid (90 tiles) | 35.4 -> 23.8 -> 15.1 | 19.6 -> 13.0 |
    | 1920x1080, level 5.1 grid (48 tiles) | 70.4 -> 45.6 -> 27.2 | 37.3 -> 22.4 |
    | 1920x1080, level 4.0 grid (32 tiles) | 120 -> 74.2 -> 42.2 | 61.0 -> 35.0 |
    CPU accelerator 20.5 -> 21.4 -> 21.7, Velocity 20.5 to 28.5 -> 22.0 -> 24.0 (832x480, 104 tiles; they have
    a core per tile, and for one core three passes are no less work than one).
    WebGPU (`Tiles_AreFasterThanOneTile`, this morning -> stage 1 -> stage 2): 104 tiles 22.7 -> 17.7 to 20.5 ->
    11.9; batch of 8: 3.52 -> 3.2 -> 3.2 per frame; one tile 1496 -> 913 -> 278. Per pass (104 tiles, a device
    wait after each): transform 3.7, symbols 3.4, range coder 3.4 ms.
  - Wasm is SLOWER on the default grid: 104 tiles 114 -> 96 -> 123 to 127 ms (one tile 834 -> 302 -> 180 to 189;
    batch of 8: 108 -> 139 per frame). Per pass, 104 tiles: transform 31, symbols 87, range coder 52 ms. Against
    the desktop CPU accelerator on the same machine (transform about 10, symbols 7, range coder 1.5) the
    transform is 3x slower in Wasm, the symbol pass 13x and the range coder pass 35x: the two new passes hit
    something the Wasm backend does badly. Both carry their state through nested calls by reference (`ref int n`,
    `ref re`); the transform pass does not. Open, next: pass that state by value and measure.
  - Batches. A batch of 16 on CUDA is 11% slower per frame than this morning (0.92 -> 1.02), batches of 2 / 4 / 8
    are 2.38 / 1.58 / 1.24 ms per frame (stage 1: 3.21 / 1.85 / 1.35), 32: 0.88. Per pass for 16 frames:
    transform 6.7 ms, symbols 3.5, range coder 2.9, where the single pass took about 11.7 for the same work:
    three passes move the coefficients and the symbols through device memory. When the device is full, total
    work is what counts.
  - Where the time goes now (CUDA): 832x480 default grid: transform 1.5 ms, symbols 0.4, range coder 1.8;
    1920x1080 level 4.0 grid: 8.6 / 1.7 / 32.7. What is left is the range coder itself: about 0.1 ms per 16x16
    position at q40 on this picture, one coder per tile by the format. The lever that remains is the cost per
    symbol: the coder keeps a 64-bit window (the reference's layout), which WGSL has to emulate; a 32-bit window
    with a carry buffer is the form to try (open).
  - Tests: the encoder tests (bytes equal to the CPU reference, FFmpeg decodes to the reconstruction, hardware
    decoders) run through the three passes. New: `Av1KeyframeEncoderGpu_SymbolWaves_EveryWaveSizeGivesTheReferenceBytes`
    (waves of 1, 3, 5, 16, 64 slots and a budget that forces one; the wave size asserted),
    `Av1KeyframeEncoderGpu_FinestQuantizerNoise_MatchesCpuReference` (noise at q index 1 and 2: symbol slots near
    their worst case, Golomb codes of a dozen bits; 686 bytes per 16x16 position) and
    `Av1SymbolStreamGpu_Entries_CodeTheBytesOfTheDirectCalls` (900 CDF symbols and bit runs of 1 to 32 bits coded
    directly and through entries: runs above 15 bits do not occur in 8-bit pictures). 39 red-check mutations: 38
    caught, one equivalent (the fh field masked to 9 bits: fh >> 6 never has a tenth).
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

`CodecsTestBase.Av1LevelTests` (2026-11-13; what each oracle can fail on is in the file's header):
- `Av1Level_Select_MatchesTheSpecificationTables`, `Av1Level_FrameSizeLimits_MatchTheSpecificationRules`,
  `Av1Level_TileLayout_StaysInsideTheLevel`, `Av1SequenceHeaderWriter_LevelAndTier_RoundTrip`,
  `Av1Isobmff_UnifySequenceHeaders_TheHighestLevelForEveryUnit` - host arithmetic against values derived by hand
  from Annex A.3.
- `Av1ObuAssembleKernel_Level_PicksTheCandidateAtTheExactLimit` - the kernel's choice at a frame of exactly a
  level's limit and one byte over.
- `Av1KeyframeEncoderGpu_Level_SignalsTheLowestLevelTheStreamConformsTo` - real encodes equal the CPU reference
  and conform to the level they signal and to none below it.
- `Av1KeyframeEncoderGpu_Level_FfmpegReadsTheSignalledLevel` (desktop) - ffprobe reads the same level; libdav1d
  decodes to the encoder's reconstruction.
- `VideoFileEncoderGpu_Av1_OneLevelForTheFile` - one sequence header per file, av1C states its level.
72 red-check mutations, all caught.

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
