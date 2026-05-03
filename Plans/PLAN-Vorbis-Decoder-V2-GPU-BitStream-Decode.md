# PLAN - Vorbis Decoder v2: Move Bit-Stream Decode to GPU

**Owner:** Tuvok
**Status:** ✓ ALL STEPS SHIPPED (2026-05-03). Vorbis decoder v2 GPU bit-stream decode is live on desktop backends (CPU + CUDA + OpenCL); browser backends (WebGPU + Wasm) keep the v1 hybrid path until cross-lane ILGPU binding-count work lands. PMT verification: 15 PASS / 0 FAIL / 3 SKIP.

## Progress 2026-05-03

- **Step 1 (VorbisResidueDecoderGpu)**: ✓ shipped 0.3.0-rc.1 commit `3a1cb5c`. File: `Audio/Vorbis/VorbisResidueDecoderGpu.cs`. Type 0/1 + Type 2 paths covered. Static GPU-callable.
- **Supporting infrastructure also shipped:** `VorbisSetupHeaderGpu.cs` (flat-pack of full setup header), `VorbisHuffmanCodebookSetGpu.cs` (flat-pack of all codebook trees + multiplicands).
- **Step 2 (VorbisPacketDecodeKernel)**: ✓ shipped commit `da5e064`. File: `Audio/Vorbis/VorbisPacketDecodeKernel.cs`. Single-thread GPU kernel that wires header parse + per-channel floor decode + per-submap residue decode in one dispatch.
- **Step 2 verification (smoke test)**: ✓ commit `23f24e9`. `VorbisPacketDecodeKernel_LoadsOnAccelerator` test PASSED on all 5 available backends (CPU + CUDA + OpenCL + WebGPU + Wasm). Kernel COMPILES on every backend.
- **Step 3a (DecoderGpu uploads + kernel compile in constructor)**: ✓ shipped commit `afeb6b1`. 38 flat-packed buffers uploaded once per stream, kernel compiled. Existing 3 Vorbis decoder GPU tests still PASS on all backends via the v1 path (`DecodeSpectrumOnCpu` unchanged).
- **Step 3b (DecodePacketAsync dispatches the kernel)**: ATTEMPTED + REVERTED. Bit-exact correct on desktop (CPU + CUDA + OpenCL all 3 tests PASS) but **regresses browser backends**:
  - **WebGPU**: `[WebGPU] Kernel 'Kernel_Run' requires 44 storage buffer bindings but this device only supports 10 (maxStorageBuffersPerShaderStage)`. ILGPU's WebGPU backend flattens the `VorbisPacketDecodeStaticInputs` struct's 38 `ArrayView<T>` fields into 38 separate storage-buffer bindings (plus 6 other ArrayView params = 44 total). Chrome's `maxStorageBuffersPerShaderStage = 10` so the kernel cannot be DISPATCHED even though it COMPILES.
  - **Wasm**: memory access OOB on the kernel dispatch. Same root cause class as WebGPU + likely related to Geordi's open Bug 2 (Vp8/9 KeyframeEncoderGpu Wasm OOB).
  
  Reverted Step 3b to keep WebGPU + Wasm decoder tests green via the v1 path. Step 3a infrastructure remains.

- **Step 3b second attempt: dual-path (desktop -> v2, browser -> v1 fallback)**: ATTEMPTED + REVERTED. Logic: at the start of `DecodePacketAsync`, branch on `_accelerator.AcceleratorType`; desktop dispatches via `DecodePacketV2Async`, browser falls through to existing v1 path. Builds clean. PMT result: **5 FAIL / 10 PASS / 3 SKIP** - browser backends (WebGPU + Wasm) PASS via v1 fallback (good), but desktop REGRESSED from the inline Step 3b's 9 PASS to a mixed pattern (CPU 0/3, CUDA 2/3 with Tone fail, OpenCL 2/3 with Tone fail). The extracted-method rewrite introduced subtle bugs the inline version didn't have - likely in return-path / sync timing / buffer lifecycle. Reverted again.

  **Lesson:** the inline Step 3b had `9 PASS desktop`; my from-scratch re-write of the same logic into `DecodePacketV2Async` lost something. Next attempt should bisect the v1-vs-inline-v2 diff rather than reconstruct from scratch.

## Step 3b path forward

Two paths to ship Step 3b without regressing browser backends:

1. **Restructure the kernel to fit ≤10 bindings.** Combine the 36 int fields in `VorbisPacketDecodeStaticInputs` into ONE big `ArrayView<int>` with a header offset table. Each primitive call site uses the same combined ArrayView with different `XBase` offsets (the primitives already support `(ArrayView<int>, long XBase)` patterns). Net: 1 combined int + 2 doubles + packet + outputs + scratch = 7 bindings. Substantial host-side flat-packing infra rewrite + kernel-side offset wiring. Plus all primitive call sites need to know which section offset to pass for each parameter.

2. **Wait for ILGPU WebGPU binding coalescing.** If ILGPU's WebGPU backend can be enhanced to recognize that a `struct` parameter with N `ArrayView` fields = 1 storage-buffer binding (rather than N), the v2 kernel ships as-is. Cross-lane work (Geordi's lane).

Path 1 is doable from Codecs side. Path 2 unblocks all future high-parameter kernels (Opus integration etc.) so it's higher leverage.

Filed with Geordi: `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-vorbis-v2-binding-count-2026-05-03.md`. While we wait, Step 3b stays out of the Vorbis decoder integration.

## Step 3 considerations + sub-steps queued

The kernel's signature has 19 top-level parameters (Index1D + 8 scalars + 1 setup struct + 6 outputs + 3 scratch ArrayViews). `Action<>`'s 16-arity ceiling means `LoadAutoGroupedStreamKernel<...>` won't bind it as-is. Plus the `VorbisPacketDecodeStaticInputs` struct holds 38 ArrayView fields - ILGPU's handling of ArrayView-inside-struct kernel parameters is uncertain (storage-buffer-binding semantics across CUDA / OpenCL / WebGPU differ).

**Step 3 sub-steps:**

1. **Verify struct-of-ArrayView kernel parameter loads on each backend.** Smoke test: `accelerator.LoadAutoGroupedStreamKernel<...>` with the struct param. CUDA/OpenCL/CPU likely fine. WebGPU may need restructuring.
2. **If struct-of-ArrayView is rejected**, restructure: pass the 38 ArrayViews individually OR flatten more aggressively into a few combined ArrayView<int> + ArrayView<double> with offset tables. Either way, parameter count must fit in Action<16>.
3. **Pre-build the flat-packed uploads in `VorbisAudioDecoderGpu` constructor.** New fields: `_dCodebookSet` (one allocation per int/double table - ~15 buffers) + `_dSetupConfig` (similar - ~22 buffers) + `_dCodebookParams` (3-int per codebook for Floor1). Construction-time uploads only (CARDINAL rule "metadata struct setup" carve-out).
4. **Replace `DecodeSpectrumOnCpu` call in `DecodePacketAsync` with the kernel dispatch.** Allocate per-call output buffers (header[5], floorOk[ch], floorIndex[ch], err[1]) + scratch (classifications, doNotDecode, entryVec). Read back the small header arrays.
5. **Verify against existing 3 Vorbis decoder GPU tests** (SilenceRoundTrip, SilenceOutput_IsSilent, ToneRoundTrip) on CPU + CUDA + OpenCL. Expected: bit-exact match vs CPU reference (the existing tests already do this comparison).
6. **WebGPU + Wasm enabling**: depends on (a) the struct/binding question above resolving cleanly, (b) Geordi's Wasm OOB Bug 2, and (c) the WebGPU PMT page-load timeout we're currently observing on existing GPU pair tests.

## Cardinal-rule violation closed in 0.3.0-rc.1 (separate fix)

The `VorbisAudioEncoderGpu` constructor used to call `new VorbisAudioEncoder(options)` to bootstrap its identification + setup headers. Commit `82752c1` replaces this with `VorbisAudioEncoder.BuildResolvedHeaders(options)` static call. The CPU encoder is no longer instantiated from the GPU path.

`VorbisAudioDecoderGpu.DecodePacket` still uses CPU bit-stream + CPU floor decode + CPU residue decode (`DecodeSpectrumOnCpu` private method, line 349 of the file). That's the remaining cardinal-rule violation Steps 2-3 close.

## Problem statement

`VorbisAudioDecoderGpu.DecodePacket` does CPU-side per-packet bit-stream work:

1. Parse audio packet header on the host (CPU bit reader)
2. Per-channel floor decode (CPU `VorbisFloor1Decoder.Decode` + CPU Huffman)
3. Per-submap residue decode (CPU `VorbisResidueDecoder.Decode` + CPU Huffman + CPU codebook vector lookup)
4. Build flat host buffers (posteriors + residues per channel)
5. Upload posteriors + residues to GPU
6. Dispatch the GPU floor render + multiply + IMDCT + post-IMDCT + interleave kernels

Steps 1-4 are host-side per-packet work on codec data. Per CLAUDE.md cardinal rule, the host must be a pure coordinator: alloc + upload + dispatch + readback only. Steps 1-4 violate this.

Why it's still acceptable today: v1 was the earliest cut of the GPU decoder. The integration class delivered a working bit-exact-vs-CPU end-to-end round-trip while the GPU bit-stream-decode primitives were being built up. As of 0.2.0-alpha.1, the GPU primitives EXIST:

- `VorbisAudioPacketHeaderGpu.Parse` (static GPU-callable)
- `VorbisFloor1DecoderGpu.Decode` (static GPU-callable)
- `VorbisHuffmanDecoderGpu` (static GPU-callable)
- `VorbisBitReaderGpu` (state struct + read methods)
- `VorbisCodebookVectorLookupGpu` (static GPU-callable)
- `VorbisHuffmanCodebookSetGpu` (flat-packed codebook set)

What's missing: a `VorbisResidueDecoderGpu` static helper that mirrors `VorbisResidueDecoder.Decode` (Vorbis I sec 8.6.5) in GPU-callable form, AND a top-level `VorbisPacketDecodeKernel` that wires the existing primitives into a single dispatch covering steps 1-4.

## Plan

### Step 1 - Add `VorbisResidueDecoderGpu` static helper

Mirror of `VorbisResidueDecoder.Decode`, GPU-callable shape. Static class with one entry point `Decode(...)` that:

- Takes `ref VorbisBitReaderGpuState` for the bit reader (mutated in-place).
- Takes flat ArrayView<byte> for the encoded packet bytes + packetLen.
- Takes a flat residue config struct (Type 0/1/2, Begin, End, PartitionSize, Classifications, Books, Classbook).
- Takes the flat-packed `VorbisHuffmanCodebookSetGpu` views for codebook lookup.
- Takes `ArrayView<float> residueOutFlat` (channels * n floats, channel-major) for output.
- Takes `ArrayView<int> doNotDecodeFlat` (channels ints, 0 = decode, !=0 = skip).
- Takes a per-channel scratch ArrayView<int> for classifications (sized `channels * partitionsToRead` ints, allocated by caller).
- Returns nothing; signals EOP via the bit-reader state's exhausted flag.

**Type 0/1 path:**
- Same outer 8-pass structure as the CPU port.
- Pass 0 reads classification codewords for each non-skipped channel.
- Passes 0..7 apply per-classification-per-pass books (stored in a flat `books[classification][pass]` int array, -1 for "no book").
- Each partition decode dispatches through `VorbisHuffmanDecoderGpu.TryDecode(...)` to read entries, then `VorbisCodebookVectorLookupGpu.LookupVector(...)` to materialize the multiplicand vector.
- Type 0: stride access `partitionOut[i + d * step]`. Type 1: contiguous `partitionOut[i + d]`.

**Type 2 path:**
- Same as the CPU port: build an interleaved scratch buffer of length `channels * n`, decode it as Type 1, then de-interleave back into the per-channel output rows.
- The interleaved scratch needs to be GPU-allocated by the caller; can re-use a per-decoder scratch buffer at the integration-class level.

**Subtle bits:**
- Stackalloc'd entry vectors in CPU code -> use a fixed-size local array sized to `MaxBookDimensions` (caller-known via setup metadata). The Vorbis spec maxes codebook dimensions at 65536, but in practice all setup-header books are small (<= 64 entries-per-codeword). A `kMaxBookDim = 256` constant for the local array is safe.
- EOP handling: the `TryDecode` returns -1 on bit-reader exhaustion; outer loops break out gracefully. Match the CPU port's `goto eopbreak`-equivalent control flow with a single `eop` flag.
- Float accumulate semantics: `partitionOut[...] += entryVec[d]` - mirror exactly. The output buffer must be pre-zeroed by the caller.

### Step 2 - Build `VorbisPacketDecodeKernel`

Single ILGPU kernel that runs steps 1-4 from the problem statement on GPU. Single-thread-per-packet (sequential by spec), workgroup size 1.

**Inputs (kernel parameters):**
- `ArrayView<byte> packetBytes` - encoded audio packet bytes
- `int packetLen` - packet length
- `int modeBits` - host-precomputed `VorbisMath.Ilog(modeCount - 1)` for the audio packet header parse
- `ArrayView<byte> modeBlockFlags` - flat array of mode block-flag bits indexed by mode number
- `int channels`, `int blockSize0`, `int blockSize1`
- Floor 1 config (one per floor in setup, flat-packed): partitions, multiplier, partitionClassListFlat + offsets, classDimensionsFlat + offsets, classSubclassesFlat + offsets, classMasterbooksFlat + offsets, classSubclassBooksFlat + offsets, xListLengthsFlat + offsets, etc.
- Mapping config (per-mode): submap count, mux per channel, submap floor index, submap residue index
- Residue configs (one per residue in setup, flat-packed): type, begin, end, partitionSize, classifications, classbook, books per (classification, pass)
- `VorbisHuffmanCodebookSetGpu` views (huffmanTreesFlat + offsets + lookup tables)
- Scratch: `ArrayView<int> classificationsScratch` (channels * maxPartitionsToRead ints)
- Scratch: `ArrayView<float> residueScratchFlat` (channels * halfBlock floats, pre-zeroed by caller)
- Outputs:
  - `ArrayView<int> packetHeaderOut` (5 ints: ModeNumber, BlockSize, IsLongBlock, PreviousWindowLong, NextWindowLong)
  - `ArrayView<int> floorOkOut` (channels ints: 1 if floor decoded, 0 if silent)
  - `ArrayView<int> floorIndexPerChannelOut` (channels ints: which floor index each channel uses)
  - `ArrayView<int> posteriorsFlatOut` (channels * maxXListLength ints)
  - `ArrayView<float> residuesFlatOut` (channels * halfBlock floats)
  - `ArrayView<int> errOut` (1 int: 0 = success, !0 = error code)

**Kernel body:**

```
1. var reader = new VorbisBitReaderGpuState { ... };
2. var headerResult = VorbisAudioPacketHeaderGpu.Parse(packetBytes, ..., ref reader, ...);
3. Write headerResult to packetHeaderOut
4. Compute mapping selection from headerResult.ModeNumber
5. For each channel:
   a. Pick its floor (via mapping.Mux + mapping.SubmapFloor)
   b. Call VorbisFloor1DecoderGpu.Decode(...) into posteriorsFlatOut at channel offset
   c. Set floorOkOut[ch] = (yLen > 0)
6. For each submap:
   a. Build a per-submap doNotDecode flag array from floorOk
   b. Call VorbisResidueDecoderGpu.Decode(...) writing to residuesFlatOut at submap offset
7. Write 0 to errOut on success.
```

### Step 3 - Update `VorbisAudioDecoderGpu`

Replace the CPU per-packet bit-stream work in `DecodePacket` with:

1. Allocate per-call scratch: classificationsScratch, residueScratchFlat (or reuse cached buffers).
2. Upload `packetBytes` once.
3. Dispatch the new `VorbisPacketDecodeKernel` with workgroup size 1.
4. Read back `packetHeaderOut` (5 ints), `floorOkOut` (channels ints), `errOut` (1 int) - the only host-side data needed for orchestration.
5. Skip floor-render dispatch for silent-floor channels per existing pattern.
6. Continue with the existing kernel chain (floor render, multiply, inverse coupling, IMDCT, post-IMDCT, interleave).

The host's role is now: alloc + 1 packet upload + dispatch sequence + final PCM readback. No per-packet CPU bit-stream work, no per-channel CPU floor decode, no per-submap CPU residue decode.

### Step 4 - Test bit-exact

The existing `VorbisAudioDecoderGpu_*` tests already verify CPU vs GPU bit-exactness on:
- `SilenceRoundTrip_MatchesCpuDecoder`
- `SilenceOutput_IsSilent`
- `ToneRoundTrip_MatchesCpuDecoder`

After v2 lands, these tests must pass unchanged - same audio output, fewer host-CPU operations. PMT sweep on CUDA + OpenCL + CPU as the floor (browser backends already covered by the existing kernel chain - the new kernel adds atomics-using calls so WebGL stays as Skip via `AcquireAcceleratorOrSkipAsync`).

## Risk + open questions

- **Single-thread-per-packet kernel cost**: the residue decode is a fundamentally sequential bit-stream walk (range coder semantics). Running it as one GPU thread is fine for correctness; throughput recovery for v3 would parallelize across packets (queue multiple packets, dispatch each as a separate single-thread kernel concurrently) or across submaps within a packet.
- **Scratch buffer sizing**: classifications scratch needs `channels * maxPartitionsToRead` ints. `maxPartitionsToRead` is bounded by `(End - Begin) / PartitionSize` per residue config; max across all residues in setup. The integration class can compute this once at construction time.
- **Codebook entry vector size cap**: the `kMaxBookDim = 256` local-array assumption needs verification against the setup-header codebook dimensions. If a real-world Vorbis stream has a larger codebook dimension, the kernel needs a different strategy (shared memory or upload-time codebook flattening).
- **EOP handling**: the bit-reader state needs an `exhausted` flag that propagates through the call chain. Existing `VorbisBitReaderGpuState` should already have this - verify before starting.
- **Classifications array memory**: 2D classifications array `[channels][partitionsToRead]` becomes a flat `channels * partitionsToRead` ArrayView. Indexing must be consistent (channel-major: `classifications[ch * partitionsToRead + pi]`).

## Out of scope

- Multi-channel coupling (the v1 decoder is mono-only; multi-channel is a separate v3 effort).
- Performance optimization: v2 lands a correct GPU-resident decode. v3 parallelizes for throughput.
- Header parser changes: the `VorbisAudioPacketHeaderGpu.Parse` static helper is already GPU-callable.

## Done definition

- `VorbisResidueDecoderGpu` static helper file added with type 0/1/2 paths.
- `VorbisPacketDecodeKernel` integration kernel added.
- `VorbisAudioDecoderGpu.DecodePacket` no longer calls `VorbisFloor1Decoder.Decode` / `VorbisResidueDecoder.Decode` / parses the bit-stream on host.
- Existing `VorbisAudioDecoderGpu_*` PMT tests pass on CUDA + OpenCL + CPU bit-exact (silent + tone).
- WebGL tests skip cleanly (existing AcquireAcceleratorOrSkipAsync wrapper covers this since the new kernel uses atomics).
- CHANGELOG entry covers the v1 -> v2 transition + cardinal-rule closure.
