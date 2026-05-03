# PLAN - OpusEncoderGpu + OpusDecoderGpu (the 6th GPU pair)

**Owner:** Tuvok
**Status:** Design (2026-05-03). Largest remaining 1.0.0 piece. OpusEncoderGpu + OpusDecoderGpu top-level integration classes + SILK integration kernel + CELT GPU primitives all need to land before Opus joins the 5 other 100%-ILGPU encoder/decoder pairs.

## Architectural pattern (validated by Vorbis v2)

The Vorbis v2 GPU integration kernel
(`VorbisPacketDecodeKernel`, shipped 2026-05-03) validates the
architectural pattern Opus will follow:

- **Single-thread integration kernel** (workgroup size 1) wires the
  bit-stream-sequential primitives (header parse + per-channel +
  per-submap) in one dispatch.
- **Plain POD struct** kernel parameter (`VorbisPacketDecodeStaticInputs`,
  38 ArrayView fields) holds the per-stream flat-packed setup +
  codebook tables. Allocated + uploaded ONCE per stream by the host
  (metadata struct setup carve-out per CARDINAL rule).
- **Combined int output buffer** with explicit section layout
  (`PacketHeaderOffset`, `PacketHeaderLength`, `ComputeAllIntOutLength`,
  `ErrOutOffset`) keeps the kernel parameter count under ILGPU's
  Action<16> ceiling.
- **ILGPU's `LoadAutoGroupedStreamKernel`** accepts the struct-of-
  ArrayView pattern on every backend (CPU + CUDA + OpenCL + WebGPU
  + Wasm; WebGL skips via atomics gate). Verified by
  `VorbisPacketDecodeKernel_LoadsOnAccelerator` smoke test.

OpusDecoderGpu / OpusEncoderGpu top-level integration classes will
follow the same pattern: per-stream flat-packed config struct,
combined output buffer with section layout, single-thread mode-
specific kernels (SilkDecodeKernel, CeltDecodeKernel,
HybridDecodeKernel) dispatched based on the parsed TOC byte.

## Why this is "the 6th GPU pair"

Today's 5 ILGPU encoder/decoder pairs:
- `Vp8KeyframeEncoderGpu` / `Vp8KeyframeDecoderGpu`
- `Vp9KeyframeEncoderGpu` / `Vp9KeyframeDecoderGpu`
- `Av1KeyframeEncoderGpu` / `Av1KeyframeDecoderGpu`
- `FlacEncoderGpu` / `FlacDecoderGpu`
- `VorbisAudioEncoderGpu` / `VorbisAudioDecoderGpu`

Opus is the missing pair. Captain's 1.0.0 architectural directive 2026-05-03 requires "ALL encoders and decoders running via ILGPU kernels with the main thread only used to coordinate" - that means Opus must ship a 100%-ILGPU pair too.

## What exists today

**SILK side (substantial scaffolding):**
- 31 GPU primitives in `Audio/Opus/Silk/` (e.g. `SilkBwexpanderGpu`, `SilkExcitationDequantizerGpu`, `SilkGainAdjustGpu`, `SilkLpc*Gpu`, `SilkLtpScaleGpu`, `SilkNlsf*Gpu`, `SilkPitchContourGpu`, `SilkResampler*Gpu`, `SilkSigmoidGpu`, `SilkStereo*Gpu`, etc.)
- `OpusRangeCoderGpu` - the shared Daala range coder (used by both SILK and CELT for entropy)
- File-format types: TOC byte, packet header, OPUS Ogg head, etc.

**CELT side (zero):**
- No GPU primitives yet
- CELT decode requires (per RFC 6716 + libopus): bit allocator, PVQ encode/decode, anti-collapse processing, spreading rotation, prefilter, post-filter, stereo coupling
- Approximate scope: 7 from-scratch GPU primitives + integration

**Top-level dispatch (zero):**
- TOC byte parser (file-format type, exists)
- Mode routing (SILK / CELT / Hybrid based on TOC bits 3-7) - no GPU integration class exists
- Top-level `OpusEncoderGpu` / `OpusDecoderGpu` integration classes - don't exist yet

## Design

### OpusDecoderGpu top-level

Public API mirrors `OpusDecoder` (CPU reference, in References):

```csharp
public sealed class OpusDecoderGpu : IDisposable
{
    public OpusDecoderGpu(Accelerator accelerator, OpusDecoderConfig config);

    public Task<float[]> DecodePacketAsync(ReadOnlyMemory<byte> packet);
}
```

`DecodePacketAsync` per-packet flow (host as coordinator):

1. Parse TOC byte on host (metadata struct setup carve-out per CARDINAL rule).
   - Get mode (SILK / CELT / Hybrid)
   - Get bandwidth (NB / MB / WB / SWB / FB)
   - Get frame size (2.5 / 5 / 10 / 20 / 40 / 60 ms)
   - Get channel count (1 = mono, 2 = stereo)
2. Upload encoded packet bytes to GPU (one upload).
3. Dispatch mode-specific decode kernel:
   - **SILK:** SilkDecodeKernel - chains the 31 SILK primitives. Reads range-coded indices from packet, reconstructs LPC + LTP + excitation, runs synthesis filter, resamples to output rate.
   - **CELT:** CeltDecodeKernel - uses (yet-to-be-built) CELT primitives. Reads range-coded shapes, runs inverse PVQ, applies post-filter, runs IMDCT.
   - **Hybrid:** HybridDecodeKernel - runs SILK then CELT in parallel within the same packet (or sequentially with bandwidth split per RFC 6716 sec 4.4).
4. Single readback: float PCM samples interleaved (channels * samples).

### OpusEncoderGpu top-level

Public API mirrors `OpusEncoder` (CPU reference, in References):

```csharp
public sealed class OpusEncoderGpu : IDisposable
{
    public OpusEncoderGpu(Accelerator accelerator, OpusEncoderOptions options);

    public Task<byte[]> EncodePacketAsync(ReadOnlyMemory<float> samples);
}
```

Per-packet flow:

1. Apply mode selection on host (metadata setup): pick SILK / CELT / Hybrid based on bandwidth + frame size + signal characteristics.
2. Upload PCM samples to GPU (one upload per packet).
3. Dispatch mode-specific encode kernel.
4. Readback encoded bytes (TOC byte prepended on GPU).

## Path to ship

### Step 1 - SILK integration on existing 31 primitives

Build `SilkDecodeKernel` that wires the existing 31 SILK GPU primitives in the libopus order (per RFC 6716 sec 4.2). Inputs: range-coded packet bytes, frame configuration scalars. Outputs: PCM samples + state for next packet.

**Why this first:** The primitives exist; only the integration is missing. Once SILK works end-to-end, OpusDecoderGpu can ship for SILK-mode-only packets while CELT lands. (Per Rule 1, that's still a "compromise" - but it's a meaningful intermediate test point.)

### Step 2 - CELT GPU primitives

Build the 7-ish primitives CELT needs:

1. `CeltBitAllocatorGpu` - dynamic bit allocation per band (RFC 6716 sec 4.3.3)
2. `CeltPvqDecodeGpu` - inverse PVQ (sec 4.3.4)
3. `CeltPvqEncodeGpu` - forward PVQ (encoder side)
4. `CeltAntiCollapseGpu` - anti-collapse processing (sec 4.3.5)
5. `CeltSpreadGpu` - rotation / spread (sec 4.3.4.5)
6. `CeltPrefilterGpu` / `CeltPostfilterGpu` - pitch / stereo prefilter + postfilter (sec 4.3.7)
7. `CeltStereoCouplingGpu` - mid-side stereo (sec 4.3.4.13)

Plus shared MDCT (Vorbis has `VorbisFwdMdctScaledGpu` - reusable architecture; the constants differ).

Bit-exact testing strategy: each primitive ships with NUnit tests against the CPU reference (Concentus or libopus output) on real Opus packets.

### Step 3 - CELT integration kernel

`CeltDecodeKernel` chains the 7 primitives + IMDCT in libopus order.

### Step 4 - Hybrid mode

Hybrid is SILK + CELT side-by-side (SILK below 8kHz, CELT above). The dispatcher splits the packet into the SILK sub-packet and the CELT sub-packet, runs both, mixes in the time domain.

`HybridDecodeKernel` orchestrates. The SILK and CELT sub-decodes run as separate kernel dispatches sequentially (or in parallel if state independence permits).

### Step 5 - OpusDecoderGpu integration class

Top-level dispatch + per-packet routing. Uses the existing OPUS Ogg parsing types (already in References) + the new mode kernels.

### Step 6 - OpusEncoderGpu integration class

Mirror of decoder. Mode selection + encode dispatch.

### Step 7 - Bit-exact verification

PMT sweep: every existing OpusDecoder CPU-vs-GPU bit-exact test passes on the GPU pair. Plus PMT cross-validation against ffmpeg's libopus on real Opus streams.

## Risks + open questions

- **CELT primitives are non-trivial.** Bit allocator + PVQ are the hot path; we need to match libopus bit-by-bit. References available: libopus source + Concentus port + RFC 6716.
- **Range coder state across SILK + CELT in Hybrid.** Both use OpusRangeCoder but with separate ec_dec / ec_enc states for the SILK sub-packet vs CELT sub-packet. Verify the existing `OpusRangeCoderGpu` supports this cleanly.
- **Multi-frame Opus packets.** Per RFC 6716 sec 3.2, an Opus packet can contain multiple frames concatenated. The integration kernel needs to handle the per-packet count + frame loop.
- **Stereo handling.** SILK mid-side prediction + CELT mid-side coupling are separate paths. Make sure the integration class handles stereo correctly across modes.

## Out of scope for v1

- Variable bitrate (VBR) optimization - encoder ships at constant bitrate first.
- Forward error correction (FEC) - not implemented in CPU OpusEncoder either.
- Custom modes (non-Opus-spec sample rates / frame sizes) - libopus has these, we won't.

## Done definition

- `OpusEncoderGpu` + `OpusDecoderGpu` integration classes shipped, with mode dispatch.
- SILK + CELT + Hybrid all decode bit-exact vs CPU reference (Concentus) on real Opus packets.
- PMT sweep green on CPU + CUDA + OpenCL + WebGPU + Wasm (modulo backend feature gaps).
- 6 of 6 GPU pairs shipped per Captain's 2026-05-03 architectural directive.
- CHANGELOG entry documenting Opus pair completion.

## When this lands

Likely a 0.4.0-rc.1 cut alongside the Vorbis decoder v2 GPU integration kernel. 1.0.0 stable depends on both.
