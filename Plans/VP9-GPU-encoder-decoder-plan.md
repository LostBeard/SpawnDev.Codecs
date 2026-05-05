# VP9 GPU Encoder + Decoder Plan

**Status 2026-05-04:** Native non-aligned dimension support shipped.
1920x1080 source encodes spec-correctly via boundary forced-split path:
walker emits PARTITION_HORZ at SB16 straddling the bottom edge; sequential
encode produces 2 Tx8x8 luma + 2 Tx4x4 chroma transforms over the top
BLOCK_16X8 region; out-of-frame sub-blocks skipped. `MaxMiColsAligned`
lifted 64 -> 512 in `Vp9FrameEntropyKernel` (supports up to 4K width).
Tx4x4 entries added to `Vp9KeyframeConstantsGpu` (Scan4x4, Neighbors4x4,
CoefProbs4x4). All 60 BBB FHD frames decode clean via libvpx-vp9.

AV1 follows the same template (task #23 pending).

Roadmap for applying the v3 host-as-pure-coordinator pattern (proven
on VP8 keyframe encoder + decoder) to VP9.

## Reusable from VP8

The following GPU primitives carry over directly:

- **`Vp8BoolEncoderGpu` / `Vp8BoolDecoderGpu`** — VP9's bool coder is
  structurally identical to VP8's per `Vp9BoolEncoder.cs:6-9`. The
  only difference is VP9 emits a leading marker bit (0 at prob 128)
  on `Reset` which the decoder consumes during init. Caller pattern:
  ```csharp
  var state = Vp8BoolEncoderGpu.Init();
  Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);  // VP9 marker
  // ... rest of stream ...
  ```

- **`Vp8FrameLayoutKernels`** — gather/scatter primitives for
  per-MB-packed buffers work the same way for VP9 SBs (just larger
  block sizes).

- The general kernel-chain shape: `Setup -> SequentialEncode ->
  Entropy -> Assemble` for encoder; `Setup -> Decode -> Readback`
  for decoder.

## What's VP9-specific (NEW GPU code needed)

### Coefficient entropy

VP9 coef encoding is significantly more complex than VP8 (`Vp9BlockCoefEncoder.cs`):

- 4 transform sizes (4x4, 8x8, 16x16, 32x32) with different scan
  tables per size + per-scan-type (default / row / col) = 12 scan
  tables.
- Neighbor tables (used for context computation) - same 12-table
  layout.
- Per-tx-size per-scan-position **tokenCache** byte array (max 1024
  bytes for tx32x32) - significant per-thread storage.
- `Vp9CoefBands.GetBand(txSize, c)` dynamic band lookup.
- `Vp9CoefContext.GetCoefContext(neighbors, tokenCache, c)` -
  neighbor + cache-based context.
- `Vp9CoefProbs.ModelToFullProbs` - converts 3-byte prob model to
  8-byte full prob vector per scan position.
- `Vp9CoefContext.PtEnergyClass` - token-to-energy mapping for cache.

**Port: `Vp9BlockCoefEncoderGpu` + `Vp9BlockCoefDecoderGpu`.** Both
need to handle the largest tx size's tokenCache via
`LocalMemory.Allocate<byte>(1024)` per thread.

Constants to upload once per accelerator:
- 12 scan tables (varying sizes, total ~3KB)
- 12 neighbor tables (same)
- Vp9CoefProbs default tables (~2KB per tx size, ~8KB total)
- Vp9CoefBands tables (small)
- Vp9CoefContext.PtEnergyClass (small)

### Forward + inverse transforms

Already shipped on GPU as kernels:
- `Vp9ForwardDct{4x4,8x8,16x16,32x32}Kernel` ✓
- `Vp9ForwardAdst{4,8,16}Kernel` ✓
- `Vp9ForwardWht4x4Kernel` ✓
- `Vp9Idct{4x4,8x8,16x16}Kernel` ✓ (existing)
- `Vp9Iht{4x4,8x8}Kernel` ✓ (existing)
- `Vp9DequantKernel` ✓ (existing)

For the integration, these get **inlined** into the sequential encode
kernel (one big switch over tx_type * tx_size) rather than dispatched
separately - same pattern as `Vp8FrameSequentialEncodeKernel` did
with FDCT/Walsh/IDCT.

### Quantizer tables

VP9 uses per-frame Y / UV dequantizers (separate from VP8's per-MB
6-tuple). Needs `Vp9DequantizerComputeKernel` that takes baseQIndex
+ y_dc_delta + y_ac_delta + uv_dc_delta + uv_ac_delta and produces
6 dequantizers (Y_DC, Y_AC, UV_DC, UV_AC) via `Vp9Dequantizer.PlaneQuantizer`.

### Frame headers

VP9 has an **uncompressed header** (raw bits) and a **compressed
header** (bool-coded). Both need GPU emission for the encoder + GPU
parsing for the decoder.

- `Vp9FrameHeaderWriter.cs` ports to `Vp9FrameHeaderKernel`. The
  uncompressed header is a fixed sequence of `WriteLiteral`
  bool-coder calls; the compressed header follows the same shape as
  VP8's frame header (tx_mode + coef_prob_updates + skip_prob_updates).

- For v1, all coef probs match defaults, so the 1024+ "no update"
  emits in the compressed header collapse to a flat loop (same trick
  used in `Vp8FrameSetupKernel` to avoid CUDA JIT unroll explosion).

### Partition tree + per-block encode

VP9 uses a recursive partition tree (`PARTITION_NONE`, `_HORZ`,
`_VERT`, `_SPLIT`) with leaf blocks of 4 different sizes. v1 of the
existing `Vp9KeyframeEncoder` simplifies to **Block16x16 leaves
only** with `PARTITION_NONE` at the SB level.

For the GPU sequential encode kernel, the v1 simplification reduces
to: **walk each 64x64 SB, emit `PARTITION_SPLIT` at SB level (to
get four 32x32 children), then `PARTITION_SPLIT` at each 32x32 (to
get four 16x16 children), then `PARTITION_NONE` at 16x16 leaves**.
Each 16x16 leaf encodes as DC_PRED + Tx16x16 luma + Tx8x8 chroma +
DctDct.

This matches the existing `Vp9KeyframeEncoder` v1 shape; the GPU
kernel mirrors it line-for-line.

## Kernel chain (proposed)

```
Vp9KeyframeEncoderGpu.EncodeKeyFrame:
  alloc GPU buffers
  upload Y, U, V planes
  upload (one-time-cached) const tables: scan, neighbor, default coef probs, etc.
  dispatch Vp9FrameSetupKernel
    - compute Y/UV dequantizers from baseQIndex
    - write uncompressed header to outBuf
    - write compressed header bits to compressedBuf
  dispatch Vp9FrameSequentialEncodeKernel
    - per-SB walk (row-major)
    - per-leaf-block: predict + residual + FDCT + Walsh (Y2-style) + Quant
    - save quant coefs
    - inverse: dequant + invWalsh + IDCT + add pred + clip + write recon
    - update entropy contexts
  dispatch Vp9FrameEntropyKernel
    - resume bool encoder from compressed header state
    - per-SB partition + mode info to partition0 stream
    - per-block coefs to tile data stream
  dispatch Vp9FrameAssembleKernel
    - assemble: uncompressed header + compressed header size + compressed header + tile size + tile data
    - write to output buffer
    - write outLen[0]
  read back output[0..outLen]
  return byte[outLen]

Vp9KeyframeDecoderGpu.DecodeKeyFrame:
  alloc GPU buffers
  upload encoded bytes
  parse uncompressed header on host (raw bits in fixed positions - metadata extraction)
  dispatch Vp9FrameSetupKernel (compute dequantizers from baseQIndex)
  dispatch Vp9KeyframeDecodeKernel (single-thread per frame; reads compressed header + tile data, decodes per-SB)
  read back recon planes
  return Y, U, V planes
```

## Sizing estimate

The v3 VP8 encoder shipped at `Vp8FrameSequentialEncodeKernel.cs`
~700 lines + integration class ~250 lines. VP9 will be larger due
to:
- 4 tx sizes × 16/64/256/1024 coef counts (vs VP8's flat 4x4 only)
- Partition tree walk (vs VP8's flat 4x4 grid of MBs)
- Compressed header coef-prob-update loop (1024 emits per tx_size × 4 tx_sizes = 4096 emits total)

Per Captain's directive (Rule 8): **no effort estimates**. Discuss
which approach is correct + performant + high-quality, not how
long. The path exists; building it is the work.

## Open questions

1. Wave-parallel scheduling for multi-SB frames? VP9 SBs have the
   same recon-dependency chain as VP8 MBs (above + left). v1
   single-thread per frame is correct; v4 can wave-parallel later.
2. How to handle the per-tx-size scan/neighbor table dispatch in a
   single kernel without hitting CUDA JIT unroll limits? Likely:
   indirect indexing via an offset table indexed by tx_size, similar
   to how `Vp8FrameSetupKernel`'s 1056-iter loop got fixed.

## Related plans

- `Plans/VP8-GPU-encoder-architecture.md` - the proven v3 reference.
  VP9 follows the same pattern.
- `D:\users\tj\Projects\SpawnDev.Codecs\CLAUDE.md` Cardinal Rule
  section - the binding design constraint.
- `feedback_codecs_host_is_pure_coordinator.md` - the agent-startup
  reminder that this rule applies to every codec.
