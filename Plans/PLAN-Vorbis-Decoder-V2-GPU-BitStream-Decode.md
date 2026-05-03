# PLAN - Vorbis Decoder v2: Move Bit-Stream Decode to GPU

**Owner:** Tuvok
**Status:** Step 1 SHIPPED (2026-05-03 commit `3a1cb5c`). Steps 2-3 queued. v1 in production at 0.2.0-alpha.1 + 0.3.0-rc.1; v2 unblocks the last cardinal-rule gap in the Vorbis decode pipeline.

## Progress 2026-05-03

- **Step 1 (VorbisResidueDecoderGpu)**: ✓ shipped 0.3.0-rc.1 commit `3a1cb5c`. File: `Audio/Vorbis/VorbisResidueDecoderGpu.cs`. Type 0/1 + Type 2 paths covered. Static GPU-callable.
- **Supporting infrastructure also shipped:** `VorbisSetupHeaderGpu.cs` (flat-pack of full setup header), `VorbisHuffmanCodebookSetGpu.cs` (flat-pack of all codebook trees + multiplicands).
- **Step 2 (VorbisPacketDecodeKernel)**: NOT YET BUILT. Queued for next focused session.
- **Step 3 (Update VorbisAudioDecoderGpu.DecodePacket)**: NOT YET BUILT. Depends on Step 2.

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
