# VP8 GPU Keyframe Encoder Architecture

**Status 2026-05-04:** Native non-aligned dim support shipped. VP8 accepts
any positive (W,H); internal pad to next-16-multiple, frame tag signals
original dims. Spec works at 1920x1080 directly (VP8 has no partition tree
so no boundary forced-split needed). 60-frame BBB FHD batch transcode
clean via libvpx VP8 decoder.

## Mission

Per Captain's directive (2026-04-28): **encoders and decoders must be 100%
ILGPU kernels** to keep data AND processing accelerator-resident. The
host environment that uses the accelerator is treated as a pure
coordinator - it allocates buffers, uploads source data, dispatches
kernels, and reads back the final encoded bytes. **No CPU-side math,
no CPU iteration, no CPU bool-encoding, no CPU bitstream assembly.**

This architecture document records the v3 design of
`Vp8KeyframeEncoderGpu` and the kernel chain it dispatches. Any future
codec (VP9, AV1, Opus, FLAC, Vorbis) follows the same pattern.

## Kernel Chain

```
[host] upload Y / U / V plane bytes to GPU
       (necessary I/O - source comes from outside the accelerator)
[host] dispatch Vp8FrameSetupKernel (1 thread)
       in:  baseQIndex
       in:  dcQLookup [128 ints], acQLookup [128 ints]   (constants)
       in:  defaultCoefProbs [1056 b], updateCoefProbs [1056 b]
       out: dequant [6 ints]                              (Y1/Y2/UV DC+AC)
       out: partition0Out (frame header bool stream)
       out: initialP0State [5 ints]                       (bool encoder snapshot)
[host] dispatch Vp8FrameSequentialEncodeKernel (1 thread)
       in:  yPlane / uPlane / vPlane                      (source)
       in:  yRecon / uRecon / vRecon                      (init zero)
       in:  dequant
       out: y4Coefs / y2Coefs / uCoefs / vCoefs           (quantized post-Q)
       out: yRecon / uRecon / vRecon                      (in-place updated)
[host] dispatch Vp8FrameEntropyKernel (1 thread)
       in:  y4Coefs / y2Coefs / uCoefs / vCoefs
       in:  coefProbsByType [4*264 b], constsExtended [62 b]
       in:  initialP0State                                (resumes from setup)
       in:  partition0Out                                 (header already there)
       out: partition0Out (now header + per-MB modes)
       out: tokenP0Out                                    (per-MB coef tokens)
       out: partLens [2 longs]                            (p0 length, tp0 length)
[host] dispatch Vp8FrameAssembleKernel (1 thread)
       in:  partition0Out, tokenP0Out, partLens, width, height
       out: output                                        (tag + p0 + tp0)
       out: outLen [1 int]                                (final encoded length)
[host] read outLen + output[0..outLen]                    (single readback)
[host] return byte[outLen]
```

## What runs on the host (and why)

1. **Buffer allocation** - necessary, each frame needs sized GPU memory.
2. **Source plane upload (Y, U, V)** - source comes from outside the
   accelerator (a frame grabber, a decoded WebM, a JPEG, etc.). The
   only acceptable host-to-device data move per Captain's directive.
3. **Constant table upload (1x per accelerator)** - DC/AC quantizer
   lookups, default+update coef probs, extended consts. Uploaded
   once at construction; reused across every encoded frame.
4. **Kernel dispatch sequencing** - the host issues 4 kernel dispatches
   per frame. No CPU math between them.
5. **Final byte readback** - the encoded keyframe goes back to whoever
   asked for it. Single read of `output[0..outLen]`.

## What runs on the GPU

1. **Vp8FrameSetupKernel**: dequantizer compute (lookup table indexing
   + the libvpx Y2/UV adjustments), frame header bool emission (~50-60
   bytes for v1 simplifications: colorspace=0, clamping=0, seg=off,
   lf=off, npart=0, baseQ, all deltas=0, refresh probs=true, 1056
   "no coef-prob update" bool emits, mb_no_skip=false).

2. **Vp8FrameSequentialEncodeKernel**: per-MB row-major loop. For each
   MB: gather DC predictor from neighbour recon pixels, compute
   residual, FDCT 16 Y4 + 4 U + 4 V blocks, gather Y4 DCs, Walsh on
   Y2, quantize Y4 + Y2 + UV, write quant coefs to output buffers,
   dequantize, inverse Walsh on Y2, inject Y2 inverse DCs into Y4
   coef[0], IDCT each block + add predictor + clip, write recon back
   to recon plane. ALL inline math - no helper kernels, no CPU
   round-trips.

3. **Vp8FrameEntropyKernel**: continues partition0 from
   `initialP0State` (header bits already emitted by setup). Loops MBs
   row-major, emits Y mode (3 bool bits) + UV mode (1 bit) to
   partition0, emits Y2 + 16 Y4 + 4 U + 4 V coef tokens to tokenP0,
   maintains the 9-slot above-context buffer + 9-slot per-row left
   context. Two bool encoders run in parallel within the kernel.

4. **Vp8FrameAssembleKernel**: writes the 10-byte VP8 keyframe tag
   (3-byte uncompressed tag + 3-byte 0x9D 0x01 0x2A start code +
   2-byte horiz size + 2-byte vert size) at output[0..10], copies
   partition0Out[0..p0Len] to output[10..10+p0Len], copies
   tokenP0Out[0..tp0Len] to output[10+p0Len..], writes final length
   to outLen[0].

## Why single-thread per kernel for v1/v2

The VP8 bitstream has hard sequential dependencies:

- **Per-MB recon dependency**: MB[r][c]'s intra predictor reads
  recon pixels from MB[r-1][c] (above) and MB[r][c-1] (left). The
  recon at those positions has to be written before this MB's
  predictor can read them. Iteration order MUST be row-major.
- **Bool encoder state continuity**: each EncodeBool mutates
  (lowvalue, range, count) and may emit bytes with backward carry
  propagation through the already-written buffer. The state has
  to thread through a single sequential stream per partition.

For v1/v2, every kernel uses **one thread per frame** (Index1D = 1).
The math runs on the GPU but at single-thread throughput.

## Performance v2 vs realistic frame sizes

Single-thread sequential encode per frame, measured on CUDA on TJ's
machine (Apr 2026):

| Frame size  | MBs   | Encode time |
|-------------|-------|-------------|
| 16 x 16     | 1     | ~6 s (incl. JIT first run; ~1s warm)
| 32 x 32     | 4     | ~2 s
| 64 x 64     | 16    | ~3 s
| 128 x 128   | 64    | ~3 s
| 1920 x 1080 | 8160  | (extrapolated) ~6 minutes

Linear scaling means 1080p is far too slow for realtime. **This is
correctness-first, not throughput-first.** The throughput problem is
solved in v3 wave-parallel and v4 partition-parallel scheduling.

## Path to throughput (future v3 / v4)

Captain's note 2026-04-28: "you may not see the path, that does not
mean it isn't there. We will find it."

### Wave-parallel (v3)

MBs at positions (r, c) with r + c = w form **wave w**. Within a wave,
no MB depends on another MB in the same wave - their dependencies are
all in waves 0..w-1. Process the frame as `mbRows + mbCols - 1` waves
sequentially; within each wave, dispatch the per-MB math kernels with
parallelism = wave size.

For 1080p (mbRows=68, mbCols=120):
- 187 waves total
- Largest wave: 68 MBs (mid-diagonal)
- Average wave: ~44 MBs in parallel

That's 44x average parallelism vs single-thread. Expected speedup:
6 minutes -> ~8 seconds. Still not realtime but tractable.

### Partition-parallel entropy (v4)

VP8 supports up to 8 token partitions (`npart` in {1, 2, 4, 8}). Each
partition has its own bool encoder; rows are striped across
partitions (`mbRow & (npart - 1)`). With npart=8, the entropy stage
parallelizes 8x for the bool emit step. Per-MB context bookkeeping
remains shared but bookkeeping is tiny vs entropy emit cost.

Implementation: one thread per partition; each iterates row-major
over the FRAME but only emits bool bits when the current row is
assigned to its partition. The shared above-context buffer needs
careful synchronization (probably barrier per row).

### Multi-frame pipeline

Once v3 + v4 are in, a video transcode pipeline can:
- Dispatch frame N's encode kernels
- While frame N runs on GPU, host uploads frame N+1's source planes
- Read back frame N's bitstream into the muxer

That's the realistic-throughput path.

## v1 simplifications (still active)

These are NOT performance constraints - they're encoder feature gaps
that mirror the existing CPU `Vp8KeyframeEncoder` reference:

- All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED. No B_PRED,
  no V_PRED / H_PRED / TM_PRED.
- No 4x4 sub-block intra modes (B_PRED tree).
- No segmentation.
- Single token partition (Log2NumPartitions = 0).
- No loop filter (FilterLevel = 0).
- No mb_no_skip_coeff flag.
- Default coef probs only (no per-frame prob updates).
- No inter frames (P-frame motion compensation, MV prediction, etc).

Each of these is a future feature; the GPU architecture is general
enough to absorb them.

## Files

| File | Purpose |
|------|---------|
| `Vp8KeyframeEncoderGpu.cs` | Top-level integration class. Pure coordinator. |
| `Vp8FrameSetupKernel.cs` | Dequantizer compute + frame header writer. |
| `Vp8FrameSequentialEncodeKernel.cs` | Per-MB math + recon (single-thread). |
| `Vp8FrameEntropyKernel.cs` | Per-MB modes + coefs to bool streams. |
| `Vp8FrameAssembleKernel.cs` | Frame tag + concatenation. |
| `Vp8BoolEncoderGpu.cs` | GPU-callable VP8 boolean range encoder. |
| `Vp8CoefBlockEncoderGpu.cs` | GPU-callable per-block coef entropy. |

## Test coverage

- `Vp8BoolEncoderGpu_FixedSequence_MatchesCpu` - foundational range coder
- `Vp8BoolEncoderGpu_RandomSequence_MatchesCpu` - 4 streams, 1024 random bits
- `Vp8BoolEncoder_CpuPrefixGpuSuffix_HandoffMatchesAllCpu` - state handoff
- `Vp8CoefBlockEncoderGpu_RandomBlocks_MatchesCpu` - per-block entropy
- `Vp8FrameEntropyKernel_AllDcMode_MatchesCpu` - frame-level entropy
- `Vp8FrameTransformGpu_RandomMacroblocks_MatchesCpuReference` - transform pipeline
- `Vp8FrameReconstructGpu_RandomMacroblocks_MatchesCpuReference` - reconstruct pipeline
- `Vp8FrameSequentialEncodeKernel` (via integration) - full math pipeline
- `Vp8KeyframeEncoderGpu_SingleMbFrame_MatchesCpuEncoder` - 16x16 end-to-end
- `Vp8KeyframeEncoderGpu_MultiMb_2x2_MatchesCpuEncoder` - 32x32
- `Vp8KeyframeEncoderGpu_MultiMb_4x4_MatchesCpuEncoder` - 64x64
- `Vp8KeyframeEncoderGpu_MultiMb_8x8_MatchesCpuEncoder` - 128x128

All currently CUDA-verified bit-exact vs `Vp8KeyframeEncoder` (CPU reference).
OpenCL + CPU backends verified for the per-kernel tests; full
integration cross-backend sweep is the next correctness milestone.
