# Plan: oversized kernels (browser compile budget)

Status 2026-10-25 (Geordi). Why: a kernel whose WGSL runs to tens of thousands of lines takes Chrome's GPU process
tens of seconds to minutes to compile (Tint -> HLSL -> DXC/FXC inline everything, so `[NoInlining]` helpers do not
shrink the compiled shader). Observed on the real-GPU WebGPU lane: the AV1 decode step kernel compiled in 110 s once
and killed the GPU process ("A valid external Instance reference no longer exists") at 58 s another time; the AV1
batch encode kernel killed it; the VP8 transcode never finished; the lossless WebP emit (42k lines) killed it until it
was split (2dc85b1: largest piece 5.4k lines, 2.4 s on FXC).

**Correction 2026-10-26 (Data):** Chrome's WebGPU here compiles with **DXC** (chrome://gpu `use_dxc`), but the node Dawn
used by the compile screen always takes **FXC** (shader-f16=false with any Dawn flags). Every screen number in this plan
is an FXC number, a proxy, not Chrome's compile time; the real-GPU lane is the measurement that counts. Opaque loop bounds
(ILGPU experiment) gave no gain on FXC over 19 of these kernels; Chrome-DXC timings pending. Splitting into stage kernels
and flattening loop nests stays the lever that has worked in the real browser.

Budget: every kernel at most ~8k lines of WGSL, measured offline with the WgslScreen tool + the Dawn compile screen
(scratchpad dawn_compile/screen_all.sh) before a browser run, and confirmed in the real-GPU WebGPU lane.

## The nine (Dawn/FXC screen, ILGPU 5.4.4/5.4.5 WGSL line counts)

| Kernel | Lines | Who needs it | Order |
|---|---|---|---|
| WebPLossyEncodeRowKernel.EncodeRowKernel | 60,919 | Lore (lossy WebP encode) | 1 |
| WebPKernels.DecodeKernel | 21,925 | Lore (lossless WebP decode) | 2 |
| Av1FrameSequentialEncodeKernel.Av1EncodeFrameStepKernel / BatchStepKernel | 45,046 / 45,158 | AV1 encode | 3 |
| Av1FrameSequentialDecodeKernel.Av1DecodeFrameStepKernel | 29,948 | AV1 decode | 3 |
| Vp8FrameSequentialEncodeKernel.Vp8EncodeWavefrontKernel / BatchKernel | 50,070 / 50,246 | VP8 encode (transcode demo) | 4 |
| Vp8KeyframeDecodeKernel.Vp8KeyframeDecodeStepKernel | 66,565 | VP8 decode | 4 |
| Vp9FrameSequentialEncodeKernel.Vp9EncodeWavefrontBatchKernel | 47,030 | VP9 encode | 5 |

Before item 1: run the real-GPU WebGPU lane on Lore's set (PNG / JPEG / WebP decode + lossless + lossy encode) to see
which of these actually fail there today (the compile time matters only if Chrome hits its limit).

## 2026-10-26: Tint memory, not compile time, was the killer (Data)

ILGPU declared every value as a function-scope `var`; Tint's module step is O(vars in scope x branches), so the giant
kernels ran Chrome's GPU process to ~16 GB (system RAM to ~515 MB) and died ("Instance reference no longer exists").
ILGPU 5.4.6-local.3 emits single-assignment values as `let` (and one `let` per copy of a value duplicated into sibling
scopes): Vp8EntropyStep 58,200 -> 3,113 vars, Vp8KeyframeDecodeStep 23,118 -> 981. Chrome (DXC) after the change:
Av1EncodeFrameStep 0.8 s module + 15.4 s pipeline, Vp8KeyframeDecodeStep 1.2 + 18.6 s (was OOM), Vp9EntropyStep 4.8 +
92.2 s (was OOM; GPU process peaks at 5.85 GB: keep a watchdog), **Vp8EntropyStep still dies (6.2 GB within 5 s)**.

**Vp8EntropyStep (Vp8FrameEntropyKernel) split, mine:** the macroblock loop inlines `Vp8CoefBlockEncoderGpu.Encode` 25
times (Y2, 16 Y blocks, 4 U, 4 V; ~181k lines of WGSL). Design: the encoder allocates the four coefficient arrays as
sub-views of ONE buffer (producers unchanged: they get the sub-views), the entropy kernel takes that one view plus the
four base offsets (in the strides struct), and walks the 25 blocks of a macroblock in bitstream order in one flat loop:
block b -> (probability set, first coefficient, above / left context index, coefficient offset) computed from b, left
contexts kept in the step state, ONE `Encode` call site. Also drops three storage bindings (12 views today). Until it
lands, every browser test that compiles Vp8EntropyStep stays fenced (they can take the whole PC's RAM: never without a memory watchdog).

**Done 2026-10-26:** Vp8EntropyStep (1f58768) and Vp9EntropyStep (db91d83) now inline their block coder ONCE (VP8: the
25 blocks of a macroblock in one loop; VP9: one leaf coder for 16x16 / 16x8 and the 3 or 6 transforms in one loop,
coefficients copied into LocalMemory per job). Tint module step (Data, node Dawn): Vp8EntropyStep 181k -> 12,950 lines,
>6 GB -> 163 MB / 147 ms; Vp9EntropyStep >6 GB -> 17,133 lines, 184 MB / 176 ms. Bit-exact with the CPU encoders on
CPU / CUDA / OpenCL, red-checked. Unfenced for browser runs from ILGPU 5.4.7-local.1 (WGSL write scan through SubViews:
partition0Out of Vp8EntropyStep and the batch kernels' outputs were treated as read-only before).

## Shape of the split (the same for every codec)

The monolith is "for each block, in order: predict, transform, quantize, entropy-code, reconstruct" in one serial
thread. Only the entropy coder is serial by nature; prediction / transform / reconstruction depend only on neighbour
reconstructions. So:

1. **Reconstruction pass, wavefront.** One launch per anti-diagonal of blocks (left + above neighbours done), one thread
   per block on the diagonal (x frames for a batch): predict, forward transform, quantize, store the quantized
   coefficients per block, dequantize, inverse transform, write the reconstruction. One kernel per transform size /
   plane group keeps each kernel small. Launch count = diagonals, not blocks, and it is PARALLEL work for the first time.
   (Intra modes that read above-right need the diagonal skewed by one; v1 AV1 is DC only.)
2. **Entropy pass, serial.** The existing resumable single-thread walker, now reading the stored coefficients: partition
   / mode symbols + the coefficient coder, N blocks per launch (bounded). It no longer contains the transforms.
3. Bit-exactness gates unchanged (CPU references, ffmpeg / libwebp / libaom fixtures), plus a WGSL size gate test.

Decoders mirror it: serial entropy decode -> coefficients per block, then a wavefront reconstruction pass.

## Lossy WebP encoder (item 1) is different

libwebp's RD loop decides modes with token costs that depend on statistics of ALL EARLIER macroblocks, so decisions are
serial; the per-macroblock work is the 4 I16 + 10x16 I4 + 4 UV candidate reconstructions. Split by stage per
macroblock row chunk: (a) candidate predictions + transforms + distortion (parallel over candidates), (b) the decision +
token recording (serial, small), (c) reconstruction of the chosen modes. Keep byte-identity with libwebp (222 + 145
fixtures) as the gate.

## Also on this list (browser lane 2026-10-25)

- Lossless WebP encode on WebGPU is CORRECT but launch-bound slow (Animation_Metadata 78 s, Metadata 41 s, ExtremeShapes
  31 s against the 30 s default; Alpha_EveryPlaneAndFilter 211 s): fewer syncs (CostAsync reads back per trial),
  batch the per-code emit launches.
- Lossy WebP ALPH on CUDA: 79 s for 145 files vs 17 s on the CPU accelerator: the ALPH Auto filter = 5 lossless encodes
  per file; launch-bound.
- GIF decode / encode fixture tests 52-73 s on WebGPU.
- AV1 batch encode differs from the single-frame encode on WebGPU (11445 vs 11382 bytes, frame 0); single-frame
  Random64x128 / Random128x128 tests added to separate size from batching - run them on WebGPU.
