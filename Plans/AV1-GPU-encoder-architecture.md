# AV1 v3 GPU Encoder + Decoder Architecture

**Status:** Shipped 2026-04-29. AV1 v1 keyframe encoder + decoder running
100% on ILGPU; bit-exact vs CPU `Av1KeyframeEncoder.EncodeSingleTile`
across CUDA + OpenCL + CPU. Round-trips with companion GPU decoder.

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
- Width + height multiples of 64 (no forced-split edge cases at SB level).
- DC_PRED only, all leaves BLOCK_16X16.
- TX_16X16 for Y, TX_8X8 for UV.
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
