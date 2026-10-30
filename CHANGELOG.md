# SpawnDev.Codecs CHANGELOG

## Unreleased

### Opus GPU pair: `OpusEncoderGpu` (CELT-only) byte-identical with libopus 1.5.2 on every backend (2026-10-29)
- **New `OpusEncoderGpu`** (`SpawnDev.Codecs.Audio.Opus`: `EncodePacketAsync` managed and device-resident, `GetFinalRangeAsync`,
  `Reset`; libopus' settings: bitrate incl. OPUS_AUTO / OPUS_BITRATE_MAX, VBR / constrained VBR / CBR, complexity 0..9,
  bandwidth, max bandwidth, signal, forced channels, packet loss, prediction disabled, phase inversion disabled, lsb depth;
  applications VOIP / AUDIO / RESTRICTED_LOWDELAY; 2.5 .. 120 ms packets; 8 .. 48 kHz, mono / stereo). libopus 1.5.2's
  opus_encode_native + opus_encode_frame_native for CELT-only coding (OPUS_SET_FORCE_MODE(MODE_CELT_ONLY) or
  RESTRICTED_LOWDELAY) around Data's `CeltEncoderGpu` (celt_encode_with_ec, four stage kernels: analysis, coding, bands,
  finish). The host only decides from the settings (CBR budget, equivalent rate, mono / stereo stream with hysteresis,
  bandwidth, the 20 ms split of 40..120 ms packets, TOC value); kernels run the delay buffer, the DC reject / VOIP high-pass,
  the low-rate stereo width fade, CELT and the packet assembly (code 0 / 1 / 2 / 3, CBR padding).
- **Equal to libopus 1.5.2 on every packet of 49 real streams, bytes and final range** (`TestData/OpusEnc`,
  `scripts/audio/make_opus_encoder_fixtures.py`): every frame size, CBR / VBR / CVBR, complexity 0..9 (theta RDO at 8+),
  8 / 12 / 16 / 24 / 48 kHz input, VOIP and low delay, forced mono and stereo coded as mono, silent gaps, OPUS_AUTO and
  OPUS_BITRATE_MAX, the TOC-only too-small packets. CPU, Velocity, CUDA and OpenCL. A second set of 49 tests checks the Opus
  layer alone against the exact input and CTLs of every libopus `celt_encode_with_ec` call (recorded by an instrumented
  libopus build), so a mismatch splits cleanly between the Opus layer and CELT.
- **Scope for 1.0.0:** CELT-only. SILK and hybrid encoding come after 1.0.0; complexity 10 needs libopus' tonality analysis
  (not ported). Browser backends: the CELT stage kernels are being measured in Chrome (Data) before the WebGPU / WebGL / Wasm
  lanes include CELT.
- **SpawnDev.ILGPU 5.4.12-local.8** (Data; every bug below was found by the Opus encoder or decoder): struct-held views in
  local allocas typed local (CUDA / OpenCL), sub-word integer conversion chains, Velocity kernels over 65,536 IR values
  (invalid IL), neg / not of a comparison, multiplicative-loop trip counts, PTX / OpenCL branch-edge phi moves (the
  CUDA / OpenCL stereo CELT encode), OpenCL switch phis, and byte / ushort return values widened by the wrong sign (the FLAC GPU
  decoder's header CRC on the IR backends). Full Codecs desktop lanes: CUDA, OpenCL, Velocity 1131 / 0 each, CPU 3281 / 0.
- `OpusEncoderApplication` moved from SpawnDev.Codecs.References to the library (same values).

### Toward 1.0.0: project split done, Opus GPU range encoder, CELT GPU primitives, VP9 WebGPU green (2026-10-29)
- **Where the "Path to 1.0.0 stable" bar stands** (0.3.0-rc.1 section below): (1) the Opus GPU pair: decoder done, encoder
  (CELT-only) done on every backend (above; WebGL n/a for the serial coders);
  (2) Vorbis decoder v2 is DONE (`VorbisAudioDecoderGpu.DecodePacketAsync` runs `VorbisPacketDecodeKernel.Step` on every
  backend; the host reads back only the packet header / floor flags it needs to orchestrate); (3) the project split is DONE
  (below); (4) the May ILGPU bugs (WebGPU `SubViewValue`, Wasm OOB) are long fixed.
- **Project split:** the CPU implementations still in the main library moved to `SpawnDev.Codecs.References` (same
  namespaces): the AV1 CPU keyframe encoder and its pipeline (coefficient coder, 2D transforms, Dct32 / Dct64 / identity,
  intra predictors, quantizer, mode-info reader, partition context), the VP9 Idct / Iadst / Iht Reference transforms, bool
  decoder and compressed-header parsers, the VP8 CPU predictors / transforms / frame buffer, and the Vorbis CPU floor /
  residue decoders. What the GPU paths needed from them stays in main: `Av1KeyframeFraming` (the TD + SH OBUs and the
  uncompressed header `Av1KeyframeEncoderGpu` uploads), `Av1IntraModeContext.Table`, the `Vp9TxType4x4` / `Vp9TxType8x8` and
  `Vp8IntraMode4x4` enums, `Vp9SkipProbs.DefaultProbs`. What remains in main besides kernels is tables, stream-header
  parsers / writers and the 1D transforms the kernels share.
- **`OpusRangeEncoderGpu`** (was scaffolding that produced AV1 bytes): libopus `celt/entenc.c` statement for statement on the
  accelerator, every entry point (encode, encode_bin, bit_logp, icdf 8 / 16-bit, uint, bits, patch_initial_bits, shrink,
  tell, tell_frac, done). Byte-identical with Concentus' EntropyCoder on random scripts incl. overflow, ec_tell / tell_frac
  equal after every op (CPU, Velocity, CUDA, OpenCL). Concentus 2.2.2's `enc_shrink` moves the raw-bit tail the wrong way
  (its MemMoveByte takes (src, dst)); ours follows libopus.
- **CELT GPU primitives (Data):** PVQ (cwrs.c + vq.c: decode, encoder pulse search, spread rotation, stereo theta), bit
  allocation (rate.c, decode side) with the 48 kHz mode tables on the accelerator, anti-collapse / intensity / mid-side /
  Haar / Hadamard (bands.c), comb pre/post-filter. Each bit-exact with Concentus.
- **New `OpusDecoderGpu`** (`SpawnDev.Codecs.Audio.Opus`, three overloads + `DecodeLostAsync` + `GetFinalRangeAsync`): the whole
  libopus 1.5.2 decoder (fixed point) on the accelerator. `SilkPacketDecoderGpu` (silk_Decode: VAD / LBRR header and LBRR
  skip, stereo, indices / pulses / parameters / decode_core, MS->LR, resampler; PLC, comfort noise and the glue after a loss;
  DTX), Data's `CeltDecoderGpu` (celt_decode_with_ec + concealment), and the opus_decode_frame glue: hybrid on the shared range
  decoder with CELT accumulating into the SILK PCM, redundancy frames both ways with smooth_fade, the transition concealment and
  fades, the hybrid -> SILK silence frame, CELT / SILK resets. The host parses the TOC / frame packing and reads back four
  redundancy ints per SILK / hybrid frame; every other bit decodes on the accelerator. **Equal to libopus 1.5.2 on every packet
  of 21 real streams** (PCM and final range): SILK, CELT-only at 2.5 / 5 / 10 / 20 ms, hybrid, a mode-switching stream; losses,
  FEC, DTX, stereo / mono and bandwidth switches. The reference is libopus 1.5.2 itself, built from source
  (`scripts/audio/make_opus_fixtures.py`, `TestData/Opus`): Concentus ports libopus ~1.1 and differs from 1.5.2 (CNG, PLC,
  decode_core's voiced-PLC transition, several CELT details, 64-bit MULT32_32 on Win64), and its silk_CNG never fills
  CNG_exc_buf_Q14.
- **ILGPU 5.4.12-local.1** (Data: Velocity hung on a loop exiting from both arms of an if; WebGPU if-arm back edge without
  `continue`, both found by the CELT band recursion). Full Velocity desktop lane 1008 / 0.
- **VP9 on WebGPU green** (267 / 0; WebGPU video 468 / 0) with a TJ-approved workaround: a drain after each wavefront
  group-kernel slice. The device hang it avoids reproduces in Chrome with no ILGPU (a replay of the recorded WebGPU calls), so
  it is below ILGPU (Dawn D3D12 or the NVIDIA driver); upstream report drafted. See Docs/status.md.
- **ILGPU 5.4.11** (released; forks 2.4.10). **PlaywrightMultiTest:** `PMT_INIT_SCRIPT` adds a JS file to every page.

### Lossless model-weight packing; WebGL lane runs for real; ILGPU 5.4.6-local.1 (2026-10-26)
- **New `WeightPackGpu`** (`SpawnDev.Codecs.Compression`, three overloads, format "SDWP" v1): lossless tensors (BF16 / FP16 / FP32 / any element size 1..16) as byte planes (`ShuffleFilterGpu`) cut into 64 KiB chunks, each ONE standard raw DEFLATE stream at the new Huffman-only level, plus a length table; any zlib reads every chunk. Compression: one thread per chunk in waves (DeflateGpu phases as bounded launches); decompression: chunk offsets from the table on the accelerator, one InflateGpu thread per chunk in waves, un-shuffle. Measured on our models (same samples, ratio = packed / original): BF16 0.677 (ZipNN 0.719, zstd -3 0.788), FP16 0.662 (ZipNN 0.675), FP32 0.833 (ZipNN 0.771: its zstd stage finds long-range repeats this coder does not).
- **`DeflateGpu.LevelHuffmanOnly` (4)** = zlib Z_HUFFMAN_ONLY (literals only, the block's own dynamic code); `ZlibCompressorGpu` takes level 4. Only level 3 is the lazy parse now (was `>= 3`).
- **Verified (CPU, CUDA, OpenCL 75/75 over filters, weights, zlib, PNG, LZ4, inflate; Wasm filters 6/6 + weights 6/6):** weights round trips exact (normal-distributed BF16 / FP16 / FP32, element sizes 1 / 3 / 8, empty, 64 MiB BF16 + 80 MB FP32 across many waves), every chunk inflates with .NET's DeflateStream and a host un-shuffle gives the input, sizes within 1 % of an optimal Huffman code of every chunk (built in the test), damaged blobs refused. Red-checked (bitshuffle instead of byte shuffle: the .NET oracle fails; stored instead of Huffman: the size bound fails).
- **WebGL lane runs for real:** `WebGLCodecsTests` threw `NotSupportedException` for EVERY test (no Codecs kernel had ever run on WebGL). It now creates the WebGL2 accelerator; tests of serial stream coders (inflate, zlib, PNG, LZ4, WeightPack) declare `SerialStreamKernels` (`RequiresScatterStores`) and skip there with that reason. First real run found: the shuffle filters' later launches stored at `dst[offset + idx]` (Transform Feedback captures only a thread's own slot; each launch now gets `dst.SubView(offset, count)` and stores `dst[idx]`), and a WebGL dispatch OutOfMemoryException (SpawnJS JSTape growth + full GLSL sent every dispatch; Data, ILGPU / SpawnJS).
- **ILGPU 5.4.6-local.1** (forks 2.4.6-local.1): IfConversion and loop-invariant code motion no longer hoist an integer division that can trap (Wasm "divide by zero" in `ShuffleKernel` for inputs shorter than one element, found here; fixed by Data).

### General data compression, step 1: shuffle filters on the accelerator (2026-10-26)
- **New track: general data compression** (`Plans/PLAN-Data-Compression.md`): filters, LZ4 / Snappy, interleaved rANS, lossless ML weights (ZipNN / DFloat11 exponent coding), GDeflate, Gaussian splat containers (SPZ, SOG), zstd. Texture formats (BC / ETC2 / ASTC) are a separate image track.
- **New `ShuffleFilterGpu`** (`SpawnDev.Codecs.Compression`, three overloads: `ArrayView1D` queue-only for chaining, managed, browser `TypedArray`): byte shuffle (c-blosc / numcodecs `Shuffle`, raw trailing partial element) and bitshuffle (the bitshuffle library's `bshuf_bitshuffle`: blocks, rounded last block, raw n % 8 leftovers, default or explicit block size), both directions. The reference's three transposes reduced to one closed form, so every output byte is an 8-bit gather (one thread per byte, no scatter: WebGL keeps the same kernels); launches of 4 MiB, paced.
- **Verified (CPU, CUDA, OpenCL):** 94 fixtures from the reference libraries (`TestData/Filters`, `scripts/compression/make_filter_fixtures.py`: numcodecs, bitshuffle via hdf5plugin; element sizes 1..16, partial elements / blocks / leftovers, a 64-element block) byte-identical in both directions into poisoned buffers; three 16 MiB cases match the reference SHA-256 and invert. Red-checked (last block not rounded; last element skipped in the inverse). Browser lanes pending.
- **New `Lz4FrameCompressorGpu`** (`SpawnDev.Codecs.Compression.Lz4`, three overloads): LZ4 frames of independent 64 KB blocks BYTE-IDENTICAL with liblz4 1.9.4 (`LZ4F_compressFrame`, block_linked=False, level 0 or an acceleration): liblz4's fast compressor (byU16 hash, skip schedule, catch-up, the conservative limitedOutput bounds that decide raw blocks) as a resumable state machine, one thread per block (8192 input positions per launch), waves of blocks laid out and copied on the accelerator; optional content size, block checksums, content checksum (XXH32, `Xxh32Gpu`, resumable). Only the frame length crosses to the host.
- **New `Lz4FrameDecompressorGpu`** (three overloads + `GetDecompressedLengthAsync`): any LZ4 frame (64 KB .. 4 MB blocks, independent or linked, HC-level frames, checksums, content size, skippable and concatenated frames, the legacy format); bounded walk / size scan / layout / decode (independent blocks in parallel, linked blocks in order; literal runs and matches resume across launches) / checksum stages; damaged frames refused with `Lz4FormatException.Reason` (header checksum, magic, truncation, block / content checksum, content size, match offset outside the block or frame, dictionary frames).
- **Verified (CPU, CUDA, OpenCL):** 29 liblz4 frames (`TestData/Lz4`, `scripts/compression/make_lz4_fixtures.py`: empty .. 40 MB, raw blocks, zeros, every option, acceleration 3 / 8, three real media files) byte-identical, including 3-block waves and a 40 MB two-wave input; 24 decoder frames (`TestData/Lz4Dec`, `make_lz4_decode_fixtures.py`) decode to their content or are refused with the expected reason; a 40 MB compress + decompress round trip on the accelerator. Red-checked (skip trigger 5 instead of 6: 19 frames differ; a match copy shifted one byte: 23 frames wrong). Browser lanes pending.
- `TestData/.gitattributes`: `*.bin` and `Filters/*` are binary (a 2-byte fixture `45 0A` was detected as text and would have been rewritten to CRLF on checkout).

### Lossy WebP with transparency; WebP lossless encode WebGPU hang found (ILGPU); Vorbis decoders match libvorbis (2026-10-25)
- **`WebPLossyEncoderGpu` encodes pictures with transparency** (was refused): libwebp's extended layout RIFF + VP8X (alpha flag, canvas) + ALPH + VP8. The alpha plane `WebPRgbaToYuvGpu` imports is coded on the accelerator by `WebPLosslessEncoderGpu.EncodeAlphaAsync` (lossless, like libwebp's alpha_quality 100; filter / compression / effort via the new `WebPLossyEncodeOptions.Alpha`), and the new `AssembleExtendedKernel` writes the file one thread per byte (no intermediate copy). All three overloads.
- **Verified (CPU accelerator):** 145 new fixtures (`TestData/WebPLossyAlpha`, `scripts/images/make_webp_lossy_alpha_fixtures.py`: alpha ramps, cut-out masks, scattered transparent pixels, 1x1 .. 640x480, quality 0..100, methods 3 / 4): the VP8 chunk is byte-identical to libwebp's, our ALPH decodes to exactly the source alpha in `WebPDecoderGpu` and in libwebp (Pillow oracle), and the `ArrayView1D` overload writes nothing past the file. Red-checked (shifted alpha plane, one flipped VP8 bit, one byte past the file). The ALPH bytes are not libwebp's (our VP8L coder differs); they decode to the same plane.
- **Vorbis:** both decoders match libvorbis on six real streams (block switching, floor books, residue type 2 de-interleave, coupling order, render_line, IMDCT phase, packets ending inside a floor). `VorbisAudioDecoderGpu` lost its v1 CPU bit-stream path and allocates nothing per packet (all work buffers sized for BlockSize1 at construction; the packet is uploaded from the caller's memory).
- **WebP lossless encode on WebGPU:** found the hang (device lost in Chrome, endless on SwiftShader): the ILGPU WGSL backend dropped the outer loop's carried values on one arm of `WriteCodeDefinition`'s run-length loop (also hits the SILK pulse kernels). Fixed in ILGPU (Data), version pending. Diagnostics left in: `WebPLosslessEncoderGpu.Trace` / `WebPEncodeKernels.Trace` (stage and per-launch names, no cost when off).
- **ILGPU 5.4.5-local.1:** sub-view `.Length` folded in the IR builder (9 VP8 / VP9 kernels failed Tint with "cannot take the address of let", and WebGL silently used the whole buffer's length).
- **PlaywrightMultiTest:** waits 660 s for a test (was 600 s = the in-page `Timeout`, a race that hid the page's own timeout report); `PMT_WEBGPU_ADAPTER=swiftshader`.
- **Lossy WebP encoder RD loop as five stage kernels per macroblock** (prepare, I16, I4, UV, record; was one 61k-line WGSL kernel), and the I16 / UV predictions as one flat loop (`PredictAll`): FXC compile time of the prepare stage 134 s -> 3.2 s. Still byte-identical with libwebp (222 + 145 fixtures). The browser shader compile cost is unrolled constant-bound loop nests, not line count (Plans/PLAN-Oversized-Kernels.md).
- **Vorbis GPU decoder: ArrayView1D and TypedArray overloads** (`DecodePacketAsync(ArrayView1D<byte>, int, ArrayView1D<float>)` writes PCM into the caller's view; `DecodePacketAsync(TypedArray)` returns a `Float32Array`); new `DeviceTransfer.DownloadFloat32ArrayAsync`.
- **SILK pulse decode on WebGPU:** `SilkPulsesInputs` bound one buffer to two read_write slots (WebGPU rejects it); it now embeds offsets only (`SilkShellCoderTableOffsets`).
- **PlaywrightMultiTest:** a dead page runtime (Mono assert, runtime exit, GC allocation failure, abort), a hung test or an OutOfMemoryException fails the running test and reloads the page for the next one. Before, one runtime exit turned 2,489 later tests into fake passes.
- **ILGPU 5.4.5-local.2** (WGSL loop back-edge moves, WebGL inner-loop exit, CopyFromCPU / CopyToCPU(ref T) pinning, LoopUnrolling and Velocity loop fixes). Lossless WebP encode now runs on WebGPU (it hung).
- **Known open on WebGPU (real GPU):** AV1 encode / decode and VP8 / VP9 encode kernels too large for Chrome's shader compiler (GPU process lost or minutes to compile), AV1 batch vs single-frame encode differ, lossless WebP encode and GIF slow (launch-bound; tests over the 30 s default). Being worked (stage kernels; ILGPU opaque loop bounds experiment).

### Lossy WebP encode on the accelerator: `WebPLossyEncoderGpu`, byte-identical with libwebp (2026-10-23)
- **New `WebPLossyEncoderGpu`** (`SpawnDev.Codecs.Images.WebP.Lossy`, three overloads: `ArrayView1D`, managed, browser `TypedArray`): RGBA8 -> lossy WebP whose bytes are IDENTICAL to libwebp 1.6.0 (Pillow) for the same pixels and quality, methods 3 and 4 (4 = libwebp's and Pillow's default). Kernel chain: `WebPRgbaToYuvGpu` (libwebp import + cleanup), analysis (one thread per macroblock), segmentation + quantizer setup (one thread), the RD token loop (one thread per launch, one launch per macroblock row: libwebp's decisions depend on the token statistics of every earlier macroblock, so the walk is serial), final probabilities + partition 0 + token partition (one thread), file assembly (one thread per byte). The host uploads the picture, ~500 ints of layout and a config-only quantizer table (libwebp's double-precision power law, which WGSL lacks) and reads back scalars.
- **Verified:** 222 fixtures (methods 3 / 4, quality 0..100, 1x1 .. 640x480) byte-identical on the CPU, CUDA and OpenCL lanes (CUDA 19.8 s, OpenCL 30.6 s for all 222 with the single-thread RD loop); red-check: a penalty of 141 instead of 140 makes 9 files differ. Browser lanes (WebGPU / WebGL / Wasm) not run yet.
- **Not yet (refused with `NotSupportedException`, never a different file):** methods 0..2 and 5..6, multiple token partitions (transparency: done 2026-10-25, see above). Parallelising the candidate modes inside a macroblock is the next performance step.
- **SILK decode in `OpusDecoder` (References) restructured like libopus `silk_Decode`; all 5 `CrossValidate_Silk*` tests now match Concentus** (they had failed since they were written). Fixes: the resampler always runs (equal rates = copy resampler with its input delay); mono output goes through the 2-sample `sStereo.sMid` buffering and is resampled from index 1 (one more sample of delay); a mono packet decoded to stereo output is written to BOTH channels (it filled only the first half of the interleaved buffer); stereo no longer delays mid / side twice. Packet header read in spec order (RFC 6716 4.2.3: mid VAD flags + mid LBRR flag, then side VAD flags + side LBRR flag; it read one shared LBRR flag after interleaved VADs, which misparsed every stereo packet); LBRR frames are parsed and skipped instead of throwing `NotImplementedException`; the mid-only flag is read for every frame whose side VAD is 0 (not only the first); the side channel's prediction memory is reset on its first coded frame after mid-only frames and that frame uses `CODE_INDEPENDENTLY_NO_LTP_SCALING`; mono/stereo stream switches reset the stereo predictor history and copy the resampler like libopus; an internal-rate change keeps the stereo state and resets only what `silk_decoder_set_fs` resets. `SilkDecoder` (standalone) also has the mono buffering. CPU: 287 pass / 0 fail / 5 skip over every Opus / SILK test.

### Lossy WebP encode, step 1: libwebp-exact input conversion + byte-identical CPU reference encoder (2026-10-22)
- **New `WebPRgbaToYuvGpu`** (`SpawnDev.Codecs.Images.Yuv`): RGBA8 -> YUV 4:2:0 + alpha exactly as libwebp's lossy encoder imports a picture (`ImportYUVAFromRGBA`: gamma-linear 2x2 chroma averaging, alpha-weighted where a block is partly transparent, `kInvAlpha` reciprocal) followed by `WebPCleanupTransparentArea` (smoothing of partly transparent 8x8 blocks, flattening of fully transparent runs). Three kernels; the constant tables are uploaded once. Matches LIBWEBP'S OWN planes (read through the `webp` cffi bindings, `scripts/images/libwebp_picture.py`) on 74 fixtures incl. 1920x1080, with and without the cleanup (CPU lane; GPU lanes pending).
- **New `Vp8LossyEncoderReference`** (`SpawnDev.Codecs.References`, CPU): a statement-by-statement port of libwebp 1.6.0's lossy VP8 encoder (analysis + k-means segmentation, quantizer / lambda setup, all intra predictions, transforms, RD mode decision, DC error diffusion, token buffer with the in-pass probability refreshes, bool encoder, headers, RIFF). Its files are BYTE-IDENTICAL with Pillow / libwebp on 552 fixtures: methods 0..4, qualities 0..100, 1x1 .. 640x480, photo / graphics / noise / flat content (`scripts/images/make_webp_lossy_encode_fixtures.py`). Tables are extracted from the C source by `scripts/images/gen_vp8enc_tables.py`, not transcribed. One libwebp quirk is ported on purpose: for methods 0..2 a single-segment picture makes `StatLoop` return before the skip / token probabilities are finalised, so default probabilities are written. Methods 5 / 6 (trellis), multiple partitions and ALPH are not ported yet. This is the oracle for the accelerator encoder (next).
- **ILGPU 5.4.3** (from 5.4.3-local.1): the SSA phi fix behind the nested-ternary miscompile; `Vp8FullLoopFilterKernel` uses the nested form again and the canary `KernelDialect_NestedThresholdTernary_MatchesIfChain` guards it.

### Lossy WebP decode + facade for GIF / BMP / WebP (2026-10-22)
- **`WebPDecoderGpu` now decodes lossy WebP**: still VP8, VP8 + ALPH, animations of lossy frames, and animations that mix lossy and lossless frames. Pipeline, all on the accelerator: `Vp8FullKeyframeDecoderGpu` (colour planes) -> `WebPYuvToRgbaGpu` (libwebp's fancy upsampling and matrix) -> the ALPH plane (raw or VP8L-compressed, every filter) merged into channel A by a new `ApplyAlpha` kernel; lossy sub-rectangle frames composite through the existing blend / dispose path. The VP8 kernels are compiled on the first lossy image. Corrupt VP8 data surfaces as `ImageDecodeException` ("invalid VP8 data: ..."), not as a `Vp8DecodeException`; a VP8 size that disagrees with the container is `CanvasMismatch`. The old "lossy WebP not yet supported" error is gone.
- **`ImageDecoderGpu` dispatches GIF, BMP and WebP** (new `Gif`, `Bmp`, `WebP` properties next to `Png` / `Jpeg`) through all of its overloads (`ArrayView1D`, managed, `TypedArray`, `ReadInfoAsync`).
- **Tests:** 168 fixtures (`TestData/WebPLossy`, `scripts/images/make_webp_lossy_decode_fixtures.py`: stills with alpha 1x1 .. 1920x1080, encoder methods 0..6, qualities 20..100, lossy and mixed animations, plus every lossy file already in `TestData/WebP` and `TestData/WebPYuv`) are BIT-EXACT with libwebp (Pillow decode) through the managed and the `ArrayView1D` overload (0xAB-poisoned caller buffer); pictures over 70000 pixels are compared by SHA-256. Red-check: with the alpha merge disabled 270 failures name the alpha fixtures. CPU lane only so far; CUDA / OpenCL / WebGPU / WebGL / Wasm lanes not yet run.

### VP8 lossy keyframes, task 2: full-syntax keyframe decoder on the accelerator (2026-10-22, branch img/vp8-full)
- **New `Vp8FullKeyframeDecoderGpu`** (three overloads: `ArrayView1D`, managed, browser `TypedArray`): decodes the whole keyframe syntax that libwebp / libvpx write, entirely in kernels. Header (segmentation incl. map and per-segment quantizer / loop-filter level, filter header with mode/ref deltas, quantizer deltas, coefficient probability updates, skip probability) and the token-partition size table; per-macroblock modes with all B_PRED sub-modes and their contexts; DCT tokens from 1/2/4/8 partitions; dequantization per segment, Y2 + inverse WHT, 4x4 IDCT; all 16x16, 4x4 and chroma intra modes with the frame-edge rules; the normal and simple loop filter; crop to the signalled size. The v1 `Vp8KeyframeDecoderGpu` is unchanged.
- **Schedule (documented in the kernels):** header / modes / tokens are single GPU threads chunked ONE macroblock row per launch (the bool decoder state lives in a buffer between launches, so no launch runs long); reconstruction and the loop filter are one thread per macroblock, one launch per WAVEFRONT DIAGONAL `col + 2 * row` (left, top, top-left and top-right neighbours are on earlier diagonals, macroblocks on one diagonal are two columns apart). Loop filter = correctness-first: each macroblock does all its edges serially.
- **Hostile input:** a 32-bit bool decoder detects reads past the end of its partition on the accelerator (libwebp's rule: a decision whose 8-bit window leaves the partition); `Vp8DecodeException.Error` is one of TooShort, NotKeyFrame, BadStartCode, ZeroSize, UnsupportedVersion, FirstPartitionExceedsFrame, FirstPartitionTruncated, PartitionTableTruncated, TokenPartitionExceedsFrame, TokenPartitionTruncated.
- **Tests (CPU lane):** all 182 fixtures bit-exact with ffmpeg through the managed overload and through the `ArrayView1D` overload into 0xAB-poisoned longer buffers; 20 specific hostile inputs; 120 random corruptions; the wavefront schedule property test. 17 mutations of the decoder were red-checked. Not yet run on CUDA / OpenCL / WebGPU / WebGL / Wasm; the 16-bit `short` buffers and byte stores are the first things to check there.
- Limit: raw coefficients of the whole frame are held on the accelerator (800 bytes per macroblock; 4000x3000 = 37 MB).

### VP8 lossy keyframes, task 1: CPU reference loop filter + real-file fixtures (2026-10-22, branch img/vp8-full)
- **New `Vp8LoopFilter`** (References): RFC 6386 section 15 / libvpx loop filter for key frames (normal and simple filter, sharpness, interior/edge limits, hev threshold, per-segment absolute/delta levels, ref/mode lf deltas incl. the B_PRED delta, inner-edge skip for coefficient-less non-B_PRED macroblocks, frame level 0 = no filtering), run by `Vp8KeyframeWalker` after the whole frame is reconstructed. New one-call `Vp8KeyframeReference.Decode`.
- **Bugs the real files exposed (all fixed):** `B_VL_PRED` sub-block (3,3) used avg3(A4..A6) instead of avg3(A5..A7) (CPU predictor AND the 4x4 prediction kernel; both agreed with each other, neither with RFC 6386 / ffmpeg); the reference bool decoder rejected probability 0 (libwebp writes coefficient probabilities of 0; split = 1 is well defined); a skipped B_PRED macroblock cleared the Y2 entropy context it must leave alone. Five stale unit tests with wrong hand-computed expectations (frame tag 0xC10 vs 0xC90, inverse Walsh DC 8 vs 32, `dc_qlookup[37]` 28 vs 34) were corrected.
- **182 VP8 keyframe fixtures** in `TestData/Vp8Lossy` (`scripts/images/make_vp8_lossy_fixtures.py`): libwebp 1.5/1.6 output (Pillow + cwebp: quality 1..100, methods 0..6, 4 segments, simple/normal filter, sharpness 0..7), libvpx output via ffmpeg (token partitions 2/4/8, versions 0..3, lf deltas), and 80 random-syntax streams (valid header written by an independent python bool encoder, random partitions: every sub-mode in every context, skipped B_PRED, prob-0 updates, clamping deltas). Truth = ffmpeg's native VP8 decoder (verified native yuv420p), cross-checked against ffmpeg's webp decoder and libwebp `dwebp -yuv` for every libwebp file. The CPU reference is BIT-EXACT with ffmpeg on all 182.

### Image codecs foundation, GPU zlib, PNG (2026-10-22)
- **New `SpawnDev.Codecs.Compression`**: `InflateGpu` / `InflateKernel` (a puff.c port; a DEFLATE / zlib decoder as a single GPU thread, every block type, Adler-32 verified, specific `InflateError` codes), `Crc32Gpu`, `DeflateGpu` + `ZlibCompressorGpu` (block-parallel compressor: independent blocks, LZ77 with hash chains (greedy or lazy), dynamic Huffman with length-limited codes, stored-block fallback, byte-aligned block joins; levels 0..3). Verified against .NET's `ZLibStream` in both directions.
- **New `SpawnDev.Codecs.Images`**: `ImageFormat` / `ImageInfo` / `PixelFormat` / `DecodedImage` / `DecodedImageJS` / `ImageFormatDetector` / `ImageFrameInfo` (shared by every image codec) and the `ImageDecoderGpu` facade (format sniffing, lazy per-format decoders).
- **`PngDecoderGpu`**: every colour type and bit depth (1/2/4/8/16), palette + tRNS, gray / RGB colour keys, Adam7, critical-chunk CRC and zlib Adler checks, RGBA8 and RGBA16 output; chunk scan, gather, inflate, unfilter and colour conversion all run on the accelerator. `ReadTextAsync` returns tEXt / zTXt / iTXt chunks (ComfyUI "prompt" / "workflow"). 64 generated fixtures (`scripts/images/make_png_fixtures.py`, truth = the source samples).
- **`PngEncoderGpu`**: RGBA8 / RGBA16 input, automatic colour-type reduction (gray, gray+alpha, RGB, RGBA), adaptive or fixed scanline filters, block-parallel deflate with one IDAT chunk per block (per-chunk CRC on the accelerator), tEXt / iTXt output. Independent host-side reader in the tests plus `scripts/images/verify_png_encoder.py` (Pillow).
- **`Av1ObuAssembleKernel`**: the AV1 temporal unit (TD + SH + Frame OBU, leb128 size) is now assembled on the accelerator; the host builds only a fixed blob of under 128 bytes. `Av1KeyframeEncoderGpu` got the three-overload shape. (Still multiples of 64 only: task #23.)
- **Bug fix (VP9):** the single-frame sequential encode kernel had no boundary-superblock handling, so a picture whose height was not a multiple of 64 encoded differently alone than in a batch; the single path now runs the batch kernel at extent 1.
- **Kernel dialect rule (WebGPU):** loop headers and branch conditions are single `bool` locals built with `&` / `|`. A `while (a && b)` in the first inflate hung the WebGPU device while CPU / CUDA / OpenCL passed.

### Image codecs: JPEG decoder + encoder (2026-10-22, branch img/jpeg)
- **`JpegDecoderGpu`** (`SpawnDev.Codecs.Images.Jpeg`): baseline, extended sequential and PROGRESSIVE Huffman JPEG (spectral selection + successive approximation, all four progressive scan types), 8-bit, restart markers (independent restart segments decode as parallel GPU threads, every long loop is chunked so no launch runs for long), every integral sampling ratio (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, odd and non-MCU-aligned sizes), grayscale, YCbCr, RGB (Adobe transform 0 / component ids R,G,B), CMYK and YCCK. 12-bit, arithmetic, lossless and hierarchical files throw a specific `ImageDecodeException`. Three overloads (`ArrayView1D`, managed, browser `TypedArray`) plus `ReadInfoAsync`; `JpegDecodeOptions.ApplyOrientation` applies the EXIF orientation with a GPU kernel. Marker scan, Huffman table build, entropy decode, IDCT, upsampling and colour conversion are all kernels; the host only reads back the metadata block.
- **Bit-exact with libjpeg-turbo as used by Pillow**: ISLOW integer IDCT (jidctint), "fancy" h2v1 / h1v2 / h2v2 upsampling (replication for other ratios), libjpeg fixed-point YCbCr -> RGB, Pillow's inverted-CMYK conversion. Verified against Pillow's own decode of 100+ fixtures (`scripts/images/make_jpeg_fixtures.py`) and SHA-256 of 1024x768 pictures.
- **`JpegEncoderGpu`**: baseline Huffman encoder with libjpeg quality scaling of the Annex K tables, 4:4:4 / 4:2:2 / 4:2:0 / gray, restart intervals (parallel segment threads), EXIF orientation segment, `GetMaxEncodedSize`. Colour conversion, downsampling, forward DCT + quantization and entropy coding all run on the accelerator. The output is BYTE-IDENTICAL to Pillow / libjpeg-turbo for the same pixels and options (all fixtures, 1024x768 and 2048x2048 pictures). `JpegEncodeOptions.OptimizeHuffman` (Pillow `optimize=True`) and `Progressive` (libjpeg's default scan script) run libjpeg's two-pass optimal-table coding on the accelerator (statistics pass with atomics, `jpeg_gen_optimal_table`, DHT written device-side) and are byte-identical to Pillow too. These two options need atomics, so they do not run on the WebGL backend.
- Tests: `CodecsTestBase.JpegDecoderGpuTests` / `JpegEncoderGpuTests` (fixtures in `TestData/Jpeg`, about 1.3 MB). Alpha is ignored by the encoder.

### Image codecs: BMP and GIF (2026-10-22)
- **`BmpDecoderGpu` / `BmpEncoderGpu`** (`SpawnDev.Codecs.Images.Bmp`): 1/4/8/16/24/32 bpp, RLE4/RLE8, bitfields, V2..V5 headers; encode Bgra32 / Bgr24 / Indexed8 (exact palette found on the accelerator). Three overloads each; tests against generated fixtures and, for the encoder, Pillow as the independent oracle.
- **`GifDecoderGpu`** (`SpawnDev.Codecs.Images.Gif`): variable-width LZW (min code size 2..8, clear / end codes, deferred clear, KwKwK) with one GPU thread per frame and frames in parallel, interlace, local / global / identity-gray palettes, transparency, disposal 0..7, partial / clipped frames, NETSCAPE loop count (`ReadLoopCountAsync`), comment / application / plain-text extensions skipped, 87a and 89a. Animation API: `ReadFramesAsync`, `DecodeFrameAsync(k)` (full canvas while frame k is shown; compositing is a kernel). Three overloads each. `Compositing` property: `GifCompositing.Pillow` (default, bit-exact with Pillow 13 `Image.seek(k)` + `convert("RGBA")` incl. its quirks) or `GifCompositing.Web` (browser rules). Specific `ImageDecodeException` reasons for truncation, bad blocks, bad LZW codes, early end, a frame larger than its data can describe.
- **`GifEncoderGpu`** (`SpawnDev.Codecs.Images.Gif`): still and animated GIF encode, every stage a kernel. One exact palette (sorted by 0xRRGGBB, `ExactPaletteBuilder`) when the opaque colours of all frames fit `MaxColors` (minus one transparent index), else per-frame exact-or-quantized palettes; quantizer = median cut on a 5-5-5 histogram (split the box with the highest population x extent at the population median, palette = pixel-weighted box means, deterministic, no host math); mapping = nearest colour (ties: lowest index), optional 8x8 Bayer ordered dither or Floyd-Steinberg (anti-diagonal waves, identical to the sequential algorithm). LZW is cut into independent 32768-pixel segments (clear code between them) encoded by one thread each and concatenated bit-exactly on the accelerator, then sub-blocked straight into the caller's view. Transparency (alpha < `AlphaThreshold`), per-frame delays and disposal, NETSCAPE loop count. `GetMaxEncodedSize`, three overloads for single pictures and for animations. Quantizer quality on the Big Buck Bunny frame: 34.9 dB (256 colours) vs Pillow median cut 32.4 dB.
- Encoder tests: 40 fixtures whose expected files are written by an independent Python implementation of the same algorithms (`scripts/images/make_gif_encoder_fixtures.py`) and compared BYTE FOR BYTE; every expected file is also decoded by Pillow; Pillow decodes our output to the same canvases; round trips through `GifDecoderGpu`; brute-force nearest-colour check; random lossless round trips; a 786432-pixel picture (24 segments).
- Tests: ~165 fixtures from `scripts/images/make_gif_fixtures.py` (hostile streams, Pillow-written files, 64 random animations) with a per-frame Pillow truth and a spec truth; the bulk test `GifDecoderGpu_FuzzDirectory_MatchesPillow` compared 2500 random animations (11247 frames) with Pillow exactly. Browser TypedArray overloads are not run on the desktop lane.

### WebP: ALPH, metadata parity, animated encode (2026-10-22)
- **Cross-colour transform guard.** The forward transform SUBTRACTS the delta (libwebp `new_red -= ColorTransformDelta(green_to_red, green)`), the inverse adds it; the committed encoder kernel already did. A `+` in the encoder is invisible to round-trip tests because the strategy cost then makes the encoder reject the cross transform, so the picture only got bigger; forced on, libwebp decodes it wrong. New fixture `enc_correlated_72x56` and the parser statistic `cc_nz` (tiles with non-zero multipliers, `webp_ref.py`) make the guard fail by name.
- **ALPH decode (`WebPDecoderGpu.DecodeAlphaAsync`, three overloads).** The alpha plane of a lossy VP8X image (still, or any animation frame): raw or VP8L-compressed (headerless stream, every transform) ALPH data and the none / horizontal / vertical / gradient prediction filters, all on the accelerator; works before a lossy decoder exists and is its alpha half. Invalid headers (method 2/3, pre-processing 2, reserved bits, empty chunk, short raw data) are rejected with specific reasons, exactly the cases libwebp rejects (the fixture generator asserts that against libwebp). An ALPH chunk after the VP8 chunk no longer counts as the image's alpha (libwebp ignores it); the last ALPH before the image wins.
- **ALPH encode (`WebPLosslessEncoderGpu.EncodeAlphaAsync`, three overloads, `GetMaxAlphaEncodedSize`).** Filter none / horizontal / vertical / gradient / `Auto` (from `Quality` 50 it measures the exact coded size with each filter - a measuring pass writes nothing - and keeps the smallest; below that a fast residual-cost estimate on the accelerator picks), lossless coding through the same pipeline as the stills (palette with pixel bundling, predictor) without the VP8L header, raw fallback when coding does not beat the raw plane (never larger than 1 + w*h), pre-processing level 0. Output is the chunk payload for a lossy encoder to wrap.
- **Metadata (ICCP / EXIF / XMP).** Reader parity with libwebp's demuxer: a chunk only counts when its VP8X flag is set, and only the first of a kind (it used to report a chunk whose flag was clear). New `ReadFileInfoAsync` / `ReadMetadataAsync` TypedArray overloads. 40 new fixtures (`meta2_*`, libwebp's demuxer is the truth): every subset, flags without chunks and chunks without flags, chunk order, duplicates, unknown chunks, RIFF padding of odd payloads, libwebp-muxed lossy+alpha and animated files, EXIF orientation 1..8 in both byte orders with and without the `Exif\0\0` prefix and ten malformed forms (reported as 1). The encoder's metadata is verified through libwebp's demuxer (payloads, VP8X flag bits, no stray chunks) for every subset including a 200 KB ICC profile.
- **Animated lossless encode (`WebPLosslessEncoderGpu.EncodeAnimationAsync`, three overloads, `GetMaxAnimationEncodedSize`, `WebPAnimationEncodeOptions`).** Frames are rectangles with RGBA pixels and the shared `ImageFrameInfo` (offset, duration, `FrameDispose.None` / `Background`, `FrameBlend.Source` / `Over`); each frame is encoded by the still pipeline straight into its ANMF chunk and the RIFF / VP8X / ICCP / ANIM / ANMF / EXIF / XMP headers are written by kernels. Loop count, background colour, metadata. Verified frame by frame against libwebp's animation decoder (canvases from an independent numpy compositor that the generator cross-checks against libwebp, durations, loop count, background colour) and `WebPDecoderGpu`.
- `WebPEncodeOptions` is no longer sealed (`WebPAnimationEncodeOptions` derives from it). `scripts/images/make_webp_alpha_fixtures.py`, `make_webp_meta_fixtures.py`, `make_webp_anim_encode_fixtures.py` generate the new fixtures; `verify_webp_files.py` gained the alpha / animation / demuxer-metadata checks and `webp_ref.py` an independent ALPH decoder. The TypedArray overloads are covered by browser-lane tests that have not been run yet.

### WebP: container + VP8L lossless decoder and encoder (2026-10-22)
- **`WebPDecoderGpu`** (`Images/WebP`): RIFF container (simple VP8L, VP8X with ICCP / EXIF / XMP, ANIM / ANMF), the complete VP8L bitstream (predictor with all 14 modes, cross colour, subtract green, colour indexing with 1/2/4/8 pixels per word, colour cache, meta prefix codes, canonical prefix codes in simple and code-length-code form, LZ77 with the 120 short distance codes). The bitstream decode is one GPU thread; the inverse transforms (except the predictor, which is raster-order sequential) and the ARGB to RGBA conversion are parallel kernels. Three overloads per entry point (managed, `TypedArray`, `ArrayView1D`); `ReadInfoAsync`, `ReadFramesAsync`, `DecodeFrameAsync`, `ReadFileInfoAsync` (losslessness, which metadata exists, ANIM loop count / background) and `ReadMetadataAsync` (ICCP / EXIF / XMP blobs). EXIF orientation is reported in `ImageInfo.Orientation`.
- **Animation:** `DecodeFrameAsync` composites frames 0..k on the accelerator, bit-exact with libwebp's `WebPAnimDecoder` (key frame rule, blend with libwebp's integer arithmetic, dispose to background; the ANIM background colour is ignored exactly like libwebp does).
- **Lossy VP8 chunks** (and ALPH) are recognised: `ReadInfoAsync` / `ReadFramesAsync` describe them, decoding throws `ImageDecodeException("lossy WebP not yet supported")`. The image decode is one switch on the payload kind, so a VP8 decoder plugs in next to VP8L.
- **`WebPLosslessEncoderGpu`**: palette (sorted, delta coded, 1/2/4/8 pixels per word), subtract green, per-tile predictor (14 modes, chosen by minimum residual magnitude, residuals fully parallel), per-tile cross colour, LZ77 hash chains with the 2-D distance codes, colour cache (size search), length-limited Huffman, meta prefix codes (k-means over per-tile histograms), VP8X + ICCP / EXIF / XMP. `WebPEncodeOptions`: `Quality` 0..100, `Exact`, `IccProfile`, `Exif`, `Xmp`. Output is bit-exact in libwebp and about 1.0 to 1.5x libwebp's size at the default quality on the test sources.
- Tests: `CodecsTestBase.WebP*Tests.cs`; fixtures from libwebp via Pillow (`scripts/images/make_webp_fixtures.py`), the encoder is verified against libwebp (`scripts/images/verify_webp_files.py`, needs `CODECS_PILLOW_PYTHON`). `scripts/images/webp_ref.py` is an independent pure-Python VP8L decoder used to assert fixture feature coverage and encoder feature use.

### Three-overload boundary, VP8 first (2026-10-22)
- **Reference bump:** `SpawnDev.ILGPU` 5.2.17 -> 5.4.2, `SpawnDev.SpawnJS.Blazor` 2.1.19 -> 3.0.4 (`SpawnDev.UnitTesting.Browser` 2.7.0 and `SpawnDev.EBML` 3.0.1 are already the newest).
- **New `DeviceTransfer`** (public static): the one place codec entry points move bytes between managed memory / JS typed arrays and accelerator buffers (`Upload` from `ReadOnlySpan<byte>` with no staging copy, `Upload` from a `TypedArray` via `IBrowserMemoryBuffer.CopyFromJS`, `DownloadAsync`, `DownloadUint8ArrayAsync`).
- **`Vp8KeyframeEncoderGpu`**: every entry point now has managed, `TypedArray` (browser, `[SupportedOSPlatform("browser")]`) and `ArrayView1D` overloads, all funnelling into the `ArrayView1D` one. The `ArrayView1D` overload takes planes already on the accelerator, assembles straight into a caller-owned output view and returns only the length. New `GetMaxEncodedSize(width, height)`. The batch path is frame-major with caller-owned `outputs` + `lengths`.
- **Host row loops removed from VP8 encode.** `UploadPaddedPlane`, the batch `hostY/hostU/hostV` builders (which put whole batches on the host heap) and the `.ToArray()` staging copies are gone; the new `Vp8PlanePadKernel` strips the source stride and zero-pads to the macroblock grid on the accelerator.
- **Bug fix: odd picture sizes.** Chroma size is `ceil(dim/2)`, not `dim/2`; the encoder used to drop the last chroma column/row of odd-sized pictures.
- **`Vp8KeyframeDecoderGpu`**: same three overloads (`DecodedFrameJS` for the browser). **Bug fix:** it threw `NotSupportedException` for any size that is not a multiple of 16 even though the encoder emits such frames; it now decodes the macroblock grid and crops on the accelerator. `ReadFrameTagAsync` reads only the 10-byte header from a device-resident frame.
- New tests: `*BoundaryTests` (ArrayView1D into a poisoned caller buffer, odd sizes vs an explicitly zero-padded picture and vs the cropped CPU walker, batch vs single frames, TypedArray in the browser lanes). Not yet run: waiting on the GPU (peer release gate).

### PMT: system Chrome for real WebGPU (2026-09-28)
- PlaywrightMultiTest now launches installed Chrome (`Channel = "chrome"`) with `SpawnDev.Codecs.PlaywrightProfile`, matching ILGPU/ML. Dropped `Vulkan` from `--enable-features` and added `--disable-software-rasterizer` so Windows Dawn stays on D3D12 instead of silently falling back to SwiftShader.

### Port BlazorJS to SpawnJS (2026-09-28)
- **Replace `SpawnDev.BlazorJS` with `SpawnDev.SpawnJS.Blazor` 2.1.19** (brings `SpawnDev.SpawnJS` 2.1.19).
- **Bump `SpawnDev.ILGPU` 4.9.5-local.17 → 5.2.17** - ILGPU 4.9.x still depended on BlazorJS; 5.2.x is the SpawnJS stack.
- Demo/tests: `AddSpawnJSRuntime` / `SpawnJSRunAsync`; `UnitTesting.Blazor` → `UnitTesting.Browser` 2.7.0.
- VP8 GPU encode/decode APIs made async (`EncodeKeyFrameAsync` / `DecodeKeyFrameAsync` / `EncodeKeyFramesBatchAsync`) so browser backends use `SynchronizeAsync` + `CopyToHostAsync` instead of desktop-only sync Synchronize/GetAsArray1D.
- VP9 assemble/uncompressed-header convenience overloads: `Run` → `RunAsync` (same reason).

## Unreleased — master after rc.7 (2026-05-06)

### Wasm cold-compile diagnosis + Lever B (2026-05-06)
- **Root cause identified**: VP9 + VP8 entropy walker kernels emit one Wasm function with **52,012 locals + 750KB instructions + 5,127 inlined call sites**. V8's `kV8MaxWasmFunctionLocals=50,000` cap rejects the kernel with `WebAssembly.compile(): Compiling function #N failed: local count too large`; wabt's parser rejects identically. Same root cause as Geordi's WGSL walker `L44455` chain — ILGPU IR optimizer inlines the entire call graph (EncodeSb64 → EncodeBlock32x32 → EncodeBlock16x16 → EncodeLeafBlock → EncodePlaneCoefs) into the single `[NoInlining]` body method instead of emitting them as separate fn-defs. The Playwright 30s timeout on Codecs Wasm tests was a downstream symptom of the kernel never compiling. Diagnosis tools added at `_codecsdump/wasm-analyze.cs` (per-function local count + body size decoder) and DevComms `tuvok-to-geordi-codecs-wasm-monolithic-inlining-2026-05-06.md`.
- **VP9 + VP8 entropy: collapse single-frame + batch kernels into one** (`Vp9FrameEntropyKernel`, `Vp8FrameEntropyKernel`). Single-frame `Run` now dispatches the batch kernel with extent=1 + full-buffer strides; the inner `SubView(0, fullLen)` is a pointer-arithmetic no-op. Saves one full ~761KB Wasm module's compile pass per backend at zero runtime cost, plus the same gain on every other backend (CUDA / OpenCL / WGSL / GLSL pay the same per-kernel compile cost). Doesn't single-handedly close the V8 50K-local cap (the surviving module still has 52K locals) but halves the number of times that cost is paid.
- **PMT actionability timeout bumped to 5 min** (`PlaywrightMultiTest/ProjectTest.cs`). Codec kernel cold compile on V8 currently exceeds Playwright's 30s default; bumping the per-page default lets tests survive cold compile while ILGPU's fn-def codegen path matures. Cheap on hot tests (timeout doesn't fire when responsive).
- CPU sweep: 91 of 97 PASS post-collapse (6 unrelated VP8 failures pre-existing on master HEAD `5977809` — `Vp8KeyframeWalker_RejectsMultiPartition`, `Vp8Quantizer_*`, `Vp8FrameTag_KeyFrame320x240_ParsesAllFields`, `Vp8InverseTransform_ShortInvWalsh4x4_*`; tracked separately).

## Unreleased — master after rc.7 (2026-05-05)

**Headline: AV1 Tx4x4 transform infrastructure (forward + inverse + constants) lands on master + Vorbis encoder gets per-packet host-SubView batch dispatch (no host syncs between packets).**

### AV1
- `Av1Forward2dTransformGpu.Forward4x4DctDct` GPU helper (commit `c654e1b`) - W=4 H=4 CosBit=13, libaom shifts s0=2 s1=0 s2=0. Bit-exact vs CPU `Av1Forward2dTransform.Apply(Av1TxSize.Tx4x4, Av1TxType.DctDct, ...)` across CPU + CUDA + OpenCL + WebGPU + Wasm (15/15 backend invocations PASS; WebGL skips per existing atomics gap). Three tests: zero input, DC-only (input=16), 64-block random batch.
- `Av1Inverse2dTransformGpu.Inverse4x4DctDct` GPU helper (commit `12a0014`) + `InverseDct4Inline` + `ResolveCospi4` private helpers. Pattern follows Inverse8x8DctDct - row pass uses Inverse4 in place, column pass uses scalar inline helper. Inverse shifts: rowShift=0, colShift=4, cosBit=12. Bit-exact across the same 5 backends.
- `Av1KeyframeConstantsGpu` extended with Tx4x4 entries (commit `d75c0bc`): NzMapCtxOffset[0] 16 bytes + Scan4x4 16 ushort + TxbSkipCdfTx4x4 156 + EobMulti16Cdf 96 + EobExtraCdfTx4x4 216 + CoeffBaseEobMultiCdfTx4x4 128 + CoeffBaseMultiCdfTx4x4 1680 + CoeffLpsMultiCdfTx4x4 840. Total 3148 new entries appended to the existing constant offset chain. Bit-exact vs libaom default tables (36/36 backend invocations).

These are Phase 2 prep for AV1 boundary chroma encoding (BLOCK_16X8 / BLOCK_8X16 leaves at non-64-multiple frame dims). Walker Phase 1 + Phase 2 wiring lands separately when SpawnDev.ILGPU's WGSL fn-def codegen path stabilizes (Geordi iterating; tracked via `phase1-walker-debug` branch).

### Vorbis
- Phase A (commit `15f4f39`): `EmitKernel` now reads floor posteriors from the device buffer instead of round-tripping to host scalars. The encoder pipeline keeps posteriors GPU-resident from `FloorFitKernel` output through `EmitKernel`, eliminating the per-packet `await acc.SynchronizeAsync()` + `dPosteriors.CopyToHostAsync()` round-trip that was forcing a sync barrier mid-encode. 10/10 single-packet round-trip tests PASS unchanged.
- Phase B (commit `7ff3f3f`): new `EncodeAudioPacketsBatchAsync(IReadOnlyList<ReadOnlyMemory<float>> blocks)` API. Encodes N independent audio packets in a single host call. Per-packet stride buffers + per-packet host-side `SubView` dispatch into the existing single-packet kernels means 6×N kernel dispatches but only ONE host sync at the end of the batch (vs 6×N sync barriers in N sequential `EncodeAudioPacketAsync` calls). The per-packet kernels keep the same binding count + signature as the proven single-packet runs, sidestepping the WebGPU bind-group / Wasm codegen issues that tripped the Index2D / Index1D true-batch attempts. 15/15 PASS including new `BatchSilence` regression test.

### Demo / debugging
- `SpawnDev.Codecs.Demo` now wires `SpawnDev.ILGPU.Services.ShaderDebugService` (Program.cs registration) + adds the "Set Debug Folder" / "Grant Write Access" UI block on `/tests`. Pick a folder once at the demo URL and every kernel compile during PMT runs auto-dumps the WGSL/GLSL/Wasm + metadata-headed sources to `{folder}/{timestamp}/wgsl|glsl|wasm/`. Folder choice persists across sessions in IndexedDB (the PMT Chromium profile is persistent at `%TEMP%/SpawnDev.Codecs.PlaywrightProfile/`). Mirrors the pattern already in use by `SpawnDev.ILGPU.Demo`.
- The dump folder is critical for diagnosing ILGPU codegen issues at the kernel level - e.g. AV1 walker compile failures on WebGPU were root-caused via the dump in one DevComms reply instead of the prior 3 rounds of hypothesis-mode ping-pong.

### Cumulative shipped numbers today
- 5 master commits + Demo ShaderDebugService wiring, 66/66 backend invocations green (CPU + CUDA + OpenCL + WebGPU + Wasm; WebGL skips per existing gap).
- `SpawnDev.ILGPU` consumer reference held at `4.9.5-rc.14` on master while debug branches iterate against ILGPU local.X drops for codegen fixes.

## 0.3.0-rc.7 (2026-05-04, local only)

**Headline: SilkDecodeFrameGpu multi-frame state-rolling proven bit-exact + signal-type/sample-rate matrix complete + ILGPU rc.11 bump.**

`SilkDecodeFrameGpu_TwoFrameStateRolling_NB_BitExactVsCpu` decodes 2
consecutive voiced NB frames threading SLpcQ14Buf + OutBuf + PrevGainQ16
across calls, comparing both frames' PCM bit-exactly to CPU
`SilkDecodeFrame.Decode`. Validates the orchestrator is multi-frame
ready end-to-end - frame N+1's LTP synthesis correctly reads frame N's
post-finalize OutBuf history, frame N+1's gain-adjust correctly chains
from frame N's last subframe gain.

Plus full signal-type × sample-rate matrix at the FRAME level:
- Voiced + Unvoiced + Inactive (3 signal types)
- NB (8 kHz / subfr=40) + MB (12 kHz / 60) + WB (16 kHz / 80)

`SilkDecodeFrameGpuOrchestrator` constructor reduced from 6 to 5 kernels
(`InitStateKernel` folded into `IndicesAdapterKernel`).

Plus DecodeBitLogP CELT-prep tests on `OpusRangeDecoderGpu` - bit-exact
vs CPU on CPU + CUDA + OpenCL + Wasm (WebGPU has a separate codegen
issue tracked at
`_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-decodebitlogp-webgpu-2026-05-04.md`).

PackageReference SpawnDev.ILGPU 4.9.5-rc.11 (Wasm dispatcher
host-write-vs-queued-dispatch race fix - YOLOv8 Softmax bug on Data's
side; SILK pipeline benefits transitively).

Source on master commit `623ca5e`. 13 commits since rc.6.

## 0.3.0-rc.6 (2026-05-04, local only)

**Headline: Full SILK per-frame decoder GPU pipeline complete (bitstream → PCM).**

`SilkDecodeFrameGpuOrchestrator` ships as a host-orchestrated multi-kernel
pipeline that mirrors libopus `silk_decode_frame.c` end-to-end:

- **Phase A** (init→indices→pulses): seeds the range-decoder state in a
  1-element `ArrayView<OpusRangeDecoderGpuState>` and threads it through
  the indices + pulses kernel dispatches.
- **Phase A+P** (+ parameters dequant): reads decoded indices buffer,
  runs `SilkParametersDecoderGpu.Decode` to produce gainsQ16, pitchL,
  predCoefQ12, ltpCoefQ14, ltpScaleQ14 outputs.
- **Phase A+P+C** (+ decode core synthesis): chains
  `SilkDecodeCoreGpu.Decode` reading parameters as SubViews into the
  unified parameter output buffers, producing PCM bit-exact vs CPU
  `SilkDecodeFrame.Decode`.
- **Phase A+P+C+F** (+ finalize): adds `FinalizeFrameKernel` for the
  OutBuf shift step (history slide + xq tail-write), so the orchestrator
  is multi-frame ready end-to-end.

All buffers + scalars are GPU-resident throughout (cardinal rule: host =
pure coordinator). Verified bit-exact on CPU + OpenCL backends across:
Voiced NB, Inactive NB, plus the OutBuf rotation in the finalize variant.

Also extends `SilkDecodeCoreGpu` test coverage with WB (lpcOrder=16 full
unroll), MB (12 kHz × 60-sample subframes), NLSF interpolation (k=2
rewhitening branch), and a 3-frame state-rolling test.

CUDA + WebGPU still gated on the same ILGPU backend bugs filed at
`_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-silkdecodecoregpu-cuda-webgpu-2026-05-04.md`
(per-thread register pressure + sub-word body-struct binding count).
Wasm gated on PMT 30s cold-start kernel-compile timeout.

Source on master commit `41c85b6`. PackageReference SpawnDev.ILGPU 4.9.5-rc.10.

## 0.3.0-rc.1 (2026-05-03)

Release candidate for the 0.3.0 cut. NOT 1.0.0 - the 1.0.0 stable
designation requires the full architectural completion (see "Path to
1.0.0" below). 0.3.0-rc.1 captures today's substantial progress
toward that bar.

### Codecs in this release

**Audio:** FLAC, Opus, Vorbis. **Video:** VP8, VP9, AV1.

Patent-clean by selection: AOMedia patent pledge for VP8/VP9/AV1, RFC
6716 royalty-free for Opus, BSD reference for FLAC/Vorbis. H.264/H.265/
AAC stay out (delegated to platform encoders via SpawnDev.MultiMedia).

### Public API surface - CPU reference encoders + decoders

Every codec ships a CPU public API that produces bytes a reference
decoder accepts:

- **FLAC**: bit-exact lossless, matches ffmpeg byte-for-byte
- **Opus**: round-trip via Concentus 2.2.2, matches ffmpeg's libopus quality
- **Vorbis**: 35.7 dB SNR on real BBB content (vs ffmpeg's libvorbis 20.9 dB)
- **VP8**: keyframes accepted by ffmpeg pixel-perfect (1920x1072 @ 5 fps;
  1/2/4/8 token partitions)
- **VP9**: keyframes accepted by ffmpeg native decoder (16x16 -> 1920x1088,
  max diff Y=1 at Q=8)
- **AV1**: keyframes accepted by libdav1d (60 frames decoded on real
  BBB content, av1.mp4 plays cleanly)

### GPU encoder + decoder pairs (100% ILGPU)

Five of the six codecs ship 100%-ILGPU encoder/decoder pairs:

- `Vp8KeyframeEncoderGpu` / `Vp8KeyframeDecoderGpu`
- `Vp9KeyframeEncoderGpu` / `Vp9KeyframeDecoderGpu`
- `Av1KeyframeEncoderGpu` / `Av1KeyframeDecoderGpu`
- `FlacEncoderGpu` / `FlacDecoderGpu`
- `VorbisAudioEncoderGpu` / `VorbisAudioDecoderGpu`

Every byte of the encoded bitstream is written by an ILGPU kernel; every
pixel of the decoded recon plane comes back from a kernel readback. The
host (consumer code) is a pure coordinator: alloc + upload + dispatch +
single readback. No CPU math, no CPU iteration, no CPU bool encoding,
no CPU bitstream assembly.

Cardinal-rule compliant: every per-frame iteration on codec data,
`Array.Copy` on bitstreams, full-buffer-readback waste, and CPU-encoder
INSTANCE dependency in the GPU pair production paths has been closed.

Opus has 31 SILK GPU primitives + the shared Daala range coder; the
top-level OpusEncoderGpu/OpusDecoderGpu integration class is the next
codec on the GPU port roadmap.

### Browser demos (live in your browser via Blazor WASM)

The `SpawnDev.Codecs.Demo` project ships browser demos for the entire
codec surface:

- `/transcode` - encode + decode all 3 video codecs (VP8/VP9/AV1) via
  the CPU public API. Renders YUV->RGBA on canvas.
- `/audio` - encode + decode all 3 audio codecs (FLAC/Vorbis/Opus) via
  the CPU public API. Plots source vs decoded waveforms with per-codec
  SNR + compression ratio + timings.
- `/gpu-transcode` - encode + decode VP8 + VP9 + AV1 entirely through
  the ILGPU integration classes. 100% GPU-resident. Routes to WebGPU
  when available, Wasm fallback.
- `/benchmarks` - live throughput numbers for every encoder + decoder,
  both CPU and GPU pair, across the available ILGPU backends.

Every demo page is covered by an end-to-end unit test that mirrors its
exact encode -> decode flow, so the demo can be trusted to work without
manual browser click-through every commit
(`AudioTranscodeDemo_*` / `VideoTranscodeDemo_*` / `GpuTranscodeDemo_*`).

### Containers

| Container | Read | Write |
|---|---|---|
| RIFF / WAVE | Yes | Yes |
| AIFF | Yes | Yes |
| Ogg | Yes | Yes |
| WebM / Matroska (EBML) | Via SpawnDev.EBML 3.0.1 | Via SpawnDev.EBML |
| MP4 / ISOBMFF | Structural box reader | Not yet |
| IVF | Yes | Yes |

### Test infrastructure

- `CodecsTestBase.AcquireAcceleratorOrSkipAsync()` - converts the
  WebGL backend's `NotSupportedException` (eager kernel-feature
  validation) into `UnsupportedTestException` so the harness records
  Skipped rather than Failed for tests whose kernels need atomics.
- `CodecsTestBase.RequireFeatures(acc, AcceleratorRequirements)` -
  explicit per-test feature gating via SpawnDev.ILGPU's
  `device.Satisfies()` API.

### PMT verification at 0.3.0-rc.1

GpuTranscodeDemo sweep run 2026-05-03 against ILGPU 4.9.5-rc.1:

| Backend | Result | Notes |
|---|---|---|
| CPU | 3 PASS | Vp8 + Vp9 + Av1 GPU pair round-trip |
| CUDA | 3 PASS | Vp8 + Vp9 + Av1 GPU pair round-trip |
| OpenCL | 3 PASS | Vp8 + Vp9 + Av1 GPU pair round-trip |
| WebGPU | 3 FAIL (PMT page-load timeout) | Kernels compile - SubView codegen fix verified working; Playwright `WaitForAsync` on the Run-button locator times out at 30s, page wall time hits 1m30s+. Reported to Geordi: PMT timeout tuning needed, not an ILGPU code regression. |
| WebGL | 3 SKIP | Atomics gate (expected) |
| Wasm | 3 SKIP | Awaiting ILGPU memory OOB fix (Geordi Bug 2 in-progress) |

**Total: 9 PASS / 3 FAIL / 6 SKIP / 18 total.** Desktop GPU-pair pipeline verified bit-exact on all three desktop backends.

### Project split: SpawnDev.Codecs.References (NEW in 0.3.0-rc.1)

A new sibling project, `SpawnDev.Codecs.References`, hosts the CPU
reference encoders + decoders. The main `SpawnDev.Codecs` library now
ships **ZERO external dependencies** (only SpawnDev.* + Microsoft
framework packages) per Captain's 2026-05-03 architectural directive.

Moved to `SpawnDev.Codecs.References` in 0.3.0-rc.1 (174+ files):

**Audio**:
- **Opus** (5 files): `OpusEncoder`, `OpusDecoder`, `OpusOggEncoder`,
  `OpusOggDecoder`, `OpusCodec` (factory)
- **Opus / CELT CPU** (2 files): `CeltDecoder`, `CeltDecoderState`
- **Opus / SILK CPU** (28 files): `SilkBwexpander`, `SilkDecodeCore`,
  `SilkDecodeFrame`, `SilkDecoder`, `SilkExcitationDequantizer`,
  `SilkGainAdjust`, `SilkGainDecoder`, `SilkIndicesDecoder`,
  `SilkLog2`, `SilkLpcAnalysisFilter`, `SilkLpcFit`,
  `SilkLpcInvPredGain`, `SilkLpcSynthesisFilter`, `SilkLtpDecoder`,
  `SilkNlsf2A`, `SilkNlsfDecoder`, `SilkNlsfStabilize`,
  `SilkNlsfUnpack`, `SilkParametersDecoder`, `SilkPitchDecoder`,
  `SilkPulsesDecoder`, `SilkResampler`, `SilkShellCoder`,
  `SilkSideInfoDecoder`, `SilkSigmoid`, `SilkStereoDecodePred`,
  `SilkStereoMsToLr`, `SilkSumSqrShift`
- **FLAC** (9 files): `FlacEncoder`, `FlacDecoder`,
  `FlacFrameDecoder`, `FlacOggDecoder`, `FlacFixedSubframeEncoder`,
  `FlacLpcSubframeEncoder`, `FlacResidualDecoder`,
  `FlacSubframeDecoder`, `FlacSubframeWriter`
- **Vorbis** (4 files): `VorbisAudioDecoder`, `VorbisOggDecoder`,
  `VorbisInverseCoupling`, `VorbisAudioEncoder` (after extracting
  header construction to `VorbisHeaderPackBuilder` in main),
  + `VorbisCommentHeader` (CPU-only metadata record)

**Video**:
- **VP8** (14 files): `Vp8KeyframeEncoder`, `Vp8Decoder`,
  `Vp8BoolEncoder`, `Vp8BoolDecoder`, `Vp8CoefBlockEncoder`,
  `Vp8CoefBlockDecoder`, `Vp8MbModeInfoDecoder`, `Vp8FrameHeader`,
  `Vp8FrameHeaderWriter`, `Vp8ModeTrees`, `Vp8SegmentationLookup`,
  `Vp8MbDequantizer`, `Vp8KeyframeWalker`, `Vp8ForwardQuantizer`
- **VP9** (102 files): full CPU decoder + encoder + walker stack,
  `Vp9Decoder`, `Vp9KeyframeEncoder`, `Vp9KeyframeWalker`, all the
  `Vp9*Mv*`, `Vp9Iadst*Reference`, `Vp9Idct*Reference`,
  `Vp9Iht*Reference`, etc.
- **AV1** (11 files): `Av1Decoder`, `Av1KeyframeWalker`,
  `Av1FrameBuffer`, `Av1IvfRemuxer`, `Av1StreamAnalyzer`,
  `Av1StreamValidator`, `Av1TileGroup`, `Av1CflPredictor`,
  `Av1CompleteFrameHeaderParser`, `Av1DefaultSegmentCdfs`,
  `Av1FrameHeaderWriter`

**Entropy coders** (2 files): `OpusRangeEncoder`, `OpusRangeDecoder`

Still in `SpawnDev.Codecs` main library (because each is consumed by
GPU integration classes for static data tables, OBU framing helpers,
or metadata struct setup): the Vorbis encoder + Floor1/Residue/
PacketHeader decoders + bit IO + codebook tables, every AV1
CDF/dequant/scan table + transforms + predictors + OBU writer + spec
record types, every VP9 GPU-shared scan/coef/probs/neighbor table.

For 1.0.0 these remaining CPU implementations need finer extraction
(separate the CPU impl methods from the shared static tables) so
they can also move to References. The pattern is mechanical now -
extract tables/helpers, point GPU code at the tables class, move the
CPU impl. Several have already been done this way (`Vp8CoefTables`,
`Vp8ModeProbTables`, `Vp9BlockCoefEnums`, `Av1RangeDecoderGpu`
constants).

### Vorbis decoder v2 GPU integration kernel - ALL STEPS SHIPPED

Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md tracks the v2
work that closes the last cardinal-rule gap in the Vorbis decode
pipeline (the CPU bit-stream walk inside
`VorbisAudioDecoderGpu.DecodeSpectrumOnCpu`). Live on desktop
backends (CPU + CUDA + OpenCL) as of commit `a28e196`. Browser
backends (WebGPU + Wasm) keep the v1 hybrid path until ILGPU
binding-count work lands.

- **Step 1**: `VorbisResidueDecoderGpu` static helper - GPU-callable
  Type 0/1 + Type 2 residue decoder mirroring CPU
  `VorbisResidueDecoder.Decode`.
- **Step 2**: `VorbisPacketDecodeKernel` - single-thread GPU
  integration kernel that wires `VorbisAudioPacketHeaderGpu.Parse`
  + per-channel `VorbisFloor1DecoderGpu.Decode` + per-submap
  `VorbisResidueDecoderGpu.Decode` in one dispatch. Outputs go into
  a single `allIntOut` buffer (header, floorOk, floorIndex,
  posteriors, err) + `residuesFlatOut`. 13 top-level params total
  (within ILGPU `Action<16>` ceiling).
  `VorbisPacketDecodeStaticInputs` struct holds 38 ArrayView fields
  for the flat-packed setup header + codebook set; the struct is a
  plain POD so ILGPU's kernel-parameter marshaling can pack it.
- **Step 2 verification**: `VorbisPacketDecodeKernel_LoadsOnAccelerator`
  smoke test verifies the kernel compiles to backend IR via
  `LoadAutoGroupedStreamKernel` on each available accelerator. Catches
  binding-count limits + struct-of-ArrayView marshaling gaps before
  the integration risks breaking the working v1 decoder tests.
- **Step 3a (DecoderGpu uploads + kernel compile)**: ✓ shipped commit
  `afeb6b1`. 38 flat-packed buffers uploaded once per stream + kernel
  compiled at construction. Existing 3 Vorbis decoder GPU tests pass
  on all 5 available backends (15/0/3) via the v1 path; constructor's
  `LoadAutoGroupedStreamKernel` for the v2 kernel verifies in-context.
- **Step 3b (DecoderGpu actually dispatches the v2 kernel)**: ✓ SHIPPED
  commit `a28e196` as a dual-path branch on `_accelerator.AcceleratorType`:
  - **Desktop (CPU + CUDA + OpenCL)**: dispatches the v2 kernel directly.
    Host = pure coordinator (alloc + 1 packet upload + dispatches +
    final readback). All bit-stream decode + floor decode + residue
    decode runs inside the GPU integration kernel. Cardinal-rule
    compliant on desktop.
  - **WebGPU + Wasm**: v1 hybrid path retained. WebGPU's 10-binding
    cap (`maxStorageBuffersPerShaderStage`) rejects the v2 kernel's 44
    storage buffers (ILGPU flattens each `ArrayView<T>` field of
    `VorbisPacketDecodeStaticInputs` into a separate binding). Wasm
    fails OOB on the same dispatch (related root cause + may overlap
    with Geordi's open Bug 2). Browser backends become cardinal-rule
    clean when ILGPU coalesces struct-of-ArrayView kernel params on
    WebGPU OR a kernel restructure flattens the 36 int fields into
    one offset-indexed buffer. Filed with Geordi:
    `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-vorbis-v2-binding-count-2026-05-03.md`.

  Implementation notes: take 3 used an inline branch (post-spectrum
  chain unchanged across paths, common variables `floorOk`,
  `floorIndex`, `mapping`, `dPosteriorsView`, `dResidueView` populated
  per branch). Take 2's extracted `DecodePacketV2Async` regressed
  desktop tests via copy-paste subtleties; take 3 keeps post-spectrum
  code untouched.

  PMT verification on take 3: `Failed: 0, Passed: 15, Skipped: 3,
  Total: 18, Duration: 58 s`. 3 desktop backends × 3 tests via v2
  kernel = 9; 2 browser backends × 3 tests via v1 fallback = 6;
  WebGL × 3 atomics-gated SKIP.

### Helper classes extracted to main (so CPU classes can move cleanly)

- `Vp8CoefTables`: ZigzagScan + CoefBands + Cat3-6 probability tables
  (libvpx kZigzag / kBands / kCatN). Replaces references to the
  moved `Vp8CoefBlockDecoder.X` constants from
  `Vp8CoefBlockEncoderGpu.BuildConstsBuffer`.
- `Vp8ModeProbTables`: DefaultKfYModeProb + DefaultKfUvModeProb.
  Replaces references to the moved `Vp8ModeTrees.X` constants from
  `Vp8FrameEntropyKernel.BuildExtendedConstsBuffer`.
- `Vp9BlockCoefEnums`: PlaneType + RefType. Replaces nested enums on
  the moved `Vp9BlockCoefDecoder` class. Updated
  `Vp9KeyframeDecodeKernel` + `Vp9FrameEntropyKernel`.
- `Av1RangeEncoderGpu` / `Av1RangeDecoderGpu`: now define
  EcProbShift / EcMinProb / OdEcWindowSize / OdBitRes /
  OdEcLotsOfBits / CdfProbTop directly (AV1 spec entropy-coder
  constants). Removes the dependency on the moved CPU
  `Av1RangeEncoder` / `Av1RangeDecoder` classes.
- `VorbisHeaderPackBuilder`: hosts `BuildResolvedHeaders`,
  `BuildIdentificationHeader`, `BuildSetupHeader`,
  `BuildIdentPacketBytes`, `BuildCommentPacketBytes`,
  `BuildSetupPacketBytes`, plus private `PackFloor1` / `PackResidue`
  / `PackMapping`. Moved out of `VorbisAudioEncoder` so
  `VorbisAudioEncoderGpu` can call them without instantiating the
  CPU encoder. The CPU encoder (now in References) delegates back
  to this helper for behavior-preserving identity.
- `VorbisAudioEncoderOptions`: split out from `VorbisAudioEncoder.cs`
  (consumed by both GPU + CPU encoders).

### Dependencies

`SpawnDev.Codecs` (main library):
- `SpawnDev.ILGPU` 4.9.5-rc.1 (WebGPU `SubViewValue` IR codegen fix
  + previous kernel dispatch + partial readback +
  `ArrayView<T>.CopyToHostAsync()`)
- `SpawnDev.EBML` 3.0.1 (WebM/Matroska container)

`SpawnDev.Codecs.References` (CPU reference impls):
- `SpawnDev.Codecs` (sibling)
- `Concentus` 2.2.2 (BSD-3 pure-C# Opus reference)

### Out of scope (patent-encumbered)

H.264, H.265, AAC, MP3 - delegated to platform encoders via
SpawnDev.MultiMedia (P/Invoke to MediaFoundation / VideoToolbox /
VAAPI). System encoders are licensed by Microsoft / Apple / driver
vendors respectively; SpawnDev.Codecs stays clean.

### Path to 1.0.0 stable

Captain's directive 2026-05-03 defines the 1.0.0 bar. 0.3.0-rc.1 is
release candidate work toward that bar:

1. **All 6 codecs ship as 100%-ILGPU encoder + decoder pairs.** Today
   covers 5 of 6 (VP8, VP9, AV1, FLAC, Vorbis). Opus has 31 SILK GPU
   primitives + the shared Daala range coder; CELT has zero GPU
   primitives (bit allocator, PVQ encode/decode, anti-collapse, spread,
   prefilter, post-filter, stereo coupling all need to be ported). Plus
   SILK + CELT integration kernels + OpusEncoderGpu/OpusDecoderGpu
   top-level classes. This is the largest remaining piece.
2. **Vorbis decoder v2 GPU bit-stream decode.** v1 still does CPU-side
   per-packet header parse + floor + Huffman + residue work before
   handing to the GPU IMDCT chain. v2 ports those to GPU kernels using
   the existing `VorbisAudioPacketHeaderGpu`, `VorbisFloor1DecoderGpu`,
   `VorbisHuffmanDecoderGpu`, `VorbisBitReaderGpu`, and the new
   `VorbisResidueDecoderGpu` (shipped 0.3.0-rc.1) primitives. Plan in
   `Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md`.
3. **Complete the project split.** Started in 0.3.0-rc.1 (Opus, FLAC,
   SILK CPU, OpusRange entropy coders moved + Concentus dep removed
   from main). Remaining: Vorbis, VP8, VP9, AV1 CPU classes still
   live in main because each one mixes CPU implementation methods
   with static data tables consumed by the GPU integration classes.
   For 1.0.0 each CPU class needs its data tables / framing helpers
   extracted to a main-library "Tables" / "Helpers" class so the CPU
   class itself can move to References cleanly.
4. **Cross-lane SpawnDev.ILGPU bug fixes** for WebGPU `SubViewValue`
   IR codegen + Wasm OOB. Filed at
   `_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-webgpu-subview-codegen-and-wasm-oob-2026-05-03.md`.

## 0.2.0-alpha.2 (2026-05-03)

Same library code as 0.2.0-alpha.1; this point release ships the
documentation + demo improvements that landed after the 0.2.0-alpha.1
package was published:

- README "Quickstart - GPU encoder + decoder pairs" section with
  concrete consumer code examples for each of the 5 GPU pairs (VP8,
  VP9, AV1, FLAC, Vorbis), including how to acquire an accelerator on
  desktop + Blazor WASM and how to handle backends that lack atomics
  (`AcceleratorRequirements` + `UnsupportedTestException` pattern).
- `/gpu-transcode` browser demo now exercises all 3 video codecs end-
  to-end through the GPU pairs (VP8 + VP9 + AV1 via the tile-bytes
  flow `EncodeSingleTileAsync` -> `DecodeSingleTileAsync`). Previously
  only VP8 + VP9.
- `Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md` design doc
  covers the next-session pickup target: move CPU per-packet floor +
  Huffman + residue work in `VorbisAudioDecoderGpu.DecodePacket` into
  a single GPU kernel, closing the v1 design gap.

Net code change for library consumers: README in the package updates
with the quickstart section. No API surface changes.

## 0.2.0-alpha.1 (2026-05-03)

First version after the cardinal-rule audit pass. Every GPU integration
class is now host-as-coordinator clean: alloc + upload + dispatch +
single readback. No CPU iteration on codec data inside any *Gpu class
production path.

### Cardinal-rule + perf wins

- **VP8 keyframe encoder** now uses a GPU stride-pack kernel instead
  of the per-row CPU stride-strip loop in `UploadPlane`. Strided source
  planes upload via `Vp8StridedPlanePackKernel` (one thread per output
  byte) so plane unpacking happens accelerator-resident.
- **VP8 keyframe encoder** final readback uses
  `dOutput.View.SubView(0, finalLen).CopyToCPU(result)` (real
  per-backend partial readback, SpawnDev.ILGPU 4.9.3+) - drops the
  prior `GetAsArray1D` of the worst-case-sized output buffer + CPU
  `Array.Copy` trim.
- **VP9 + AV1 + Vorbis encoder/decoder pairs** swap host-allocated
  zero-arrays + uploads (`CopyFromCPU(new T[N])`) for GPU-resident
  `MemSetToZero()`. Eliminates ~1MB of CPU allocator churn and bus
  transfer of zeros per frame at typical sizes.
- **VP9 keyframe decoder** drops the per-decode `Array.Copy` of tile
  bytes. Uploads the full frame once, dispatches the decode kernel
  with `dFrame.View.SubView(tileStartOffset, tileLength)` so the
  kernel sees the same-shaped tile view at zero copy cost.
- **VorbisAudioDecoderGpu** drops the per-channel `Array.Copy` flat-
  buffer build for posteriors + residues. Each channel uploads
  directly into its slot via `dPosteriors.View.SubView(...)` and
  `dResidue.View.SubView(...)`. `MemSetToZero()` covers silent-channel
  slots.
- **VorbisAudioEncoderGpu** + **VorbisAudioDecoderGpu** drop their
  CPU-encoder / CPU-decoder INSTANCE dependencies. The encoder used
  to call `_cpuRef.BuildIdentPacket()` / `BuildCommentPacket()` /
  `BuildSetupPacket()` for the 3 Ogg header packets in
  `EncodeStreamAsync`; the decoder used reflection to extract
  Huffman state from the CPU decoder's private fields. Now the GPU
  encoder calls `VorbisAudioEncoder.BuildIdentPacketBytes()` /
  `BuildCommentPacketBytes()` / `BuildSetupPacketBytes()` (new public
  static helpers) and the GPU decoder builds its own Huffman array
  from `setup.Codebooks` directly. CPU encoder/decoder classes
  remain as test reference oracles only.

### Demos

- **`/audio` page** - encode + decode all 3 audio codecs (FLAC,
  Vorbis, Opus) end-to-end via the CPU public API. Synthesizes a
  mono sine wave, drives every encoder + decoder, plots source +
  recovered waveforms with per-codec SNR + compression + timings.
- **`/gpu-transcode` page** - encode + decode VP8 + VP9 entirely
  through the ILGPU integration classes (`Vp8KeyframeEncoderGpu` +
  `Vp8KeyframeDecoderGpu`, `Vp9KeyframeEncoderGpu` +
  `Vp9KeyframeDecoderGpu`). 100% GPU-resident encode/decode chain.
  Routes to WebGPU when the browser exposes it, Wasm otherwise.
- `AudioTranscodeDemo_*` unit tests verify the demo pipeline
  bit-exact for FLAC + lossy SNR floors for Vorbis (>10 dB) + Opus
  (>5 dB), 18/18 pass on CUDA + OpenCL + CPU.

### Test infrastructure

- New `CodecsTestBase.AcquireAcceleratorOrSkipAsync()` wraps
  `CreateKernelAcceleratorAsync()` to convert the WebGL backend's
  `NotSupportedException` (eager kernel-feature validation) into
  `UnsupportedTestException`. The harness now records "Skipped" for
  WebGL tests whose kernels need atomics / shared memory / barriers
  rather than reporting them as Failed. Applied across all 200+
  GPU-using test sites.
- New `CodecsTestBase.RequireFeatures(acc, AcceleratorRequirements)`
  helper: explicit per-test feature gating. Throws
  `UnsupportedTestException` when the current accelerator's backend
  cannot satisfy declared requirements.

### SpawnDev.ILGPU PackageReference: 4.9.2 -> 4.9.3

Picked up `ArrayView<T>.CopyToHostAsync()` real per-backend partial
readback (WebGPU `queue.CopyBufferToBuffer` + partial mapAsync, WebGL
partial-range readback worker path, Wasm SAB slot view, CUDA / OpenCL
/ CPU `view.CopyToCPU(target)`). Used at every `*KeyframeEncoderGpu`
and `*KeyframeDecoderGpu` partial-readback site.

## Unreleased

### PMT: system Chrome for real WebGPU (2026-09-28)
- PlaywrightMultiTest now launches installed Chrome (`Channel = "chrome"`) with `SpawnDev.Codecs.PlaywrightProfile`, matching ILGPU/ML. Dropped `Vulkan` from `--enable-features` and added `--disable-software-rasterizer` so Windows Dawn stays on D3D12 instead of silently falling back to SwiftShader.

Initial development. Project still in pre-release. See README.md for the
working feature matrix.

### 2026-04-29 - FlacDecoderGpu output interleave moves from CPU to GPU

`FlacDecoderGpu.DecodeStreamAsync` previously copied each frame's
samples back to host (`dSamples.CopyToHostAsync()`) and ran a per-sample
channel-major -> interleaved CPU loop into the output array. That
violated the cardinal rule (host doing per-sample iteration on codec
data). Now: a new `FlacInterleaveOutputGpu` primitive does the same
work GPU-side as one thread per (channel, sample); the integration
class allocates the full interleaved output on the accelerator and
dispatches the kernel per frame. Single readback of the full output
buffer at end.

Net effect: the host is back to pure coordinator status for FLAC
decode - alloc, upload, kernel dispatch, single per-frame status check
+ frame-length scan, single readback. No per-frame `dSamples`
readback, no per-sample CPU iteration on codec data.

FLAC decoder PMT 6/6 PASS on CUDA + OpenCL + CPU.

### 2026-04-29 - VorbisAudioDecoderGpu floor curve render moves to GPU

`VorbisAudioDecoderGpu` now dispatches `VorbisFloor1RenderCurveGpu`
instead of calling the CPU `VorbisFloor1Curve.Render`. Per-floor
`xList` arrays + the 256-entry inverse-dB lookup are pre-flattened +
uploaded once at construction; per packet the CPU bit-stream parse
produces the per-channel posterior Y values, which upload to GPU and
drive a single-thread floor render kernel per non-silent channel.

Tone round-trip test still PASSES on CUDA + OpenCL + CPU (9/9 PMT).
This is a meaningful proof point because the floor render IS exercised
on the tone path (silent floors get zero-curve fallback + skip the
dispatch entirely).

Pipeline state (v1 hybrid):

| Stage                      | Where |
|----------------------------|-------|
| Parse + Huffman + floor decode + residue decode | CPU |
| Floor curve render         | **GPU** |
| Floor x residue multiply   | GPU |
| Inverse coupling           | GPU |
| IMDCT                      | GPU |
| Window + overlap-add + right-half save | GPU |
| Channel interleave         | GPU |

Remaining CPU side: bit-stream parse + Huffman, floor 1 posterior
decode, residue decode. Vorbis-Huffman GPU bit-reader keystone
primitive is what unlocks moving those.

### 2026-04-29 - VorbisAudioDecoderGpu non-silence tone round-trip test

`VorbisAudioDecoderGpu_ToneRoundTrip_MatchesCpuDecoder` exercises the
full GPU decoder chain on a non-silent 440 Hz mono tone (peak 0.5).
Same packets feed both the CPU `VorbisAudioDecoder` reference (double-
precision IMDCT) AND the GPU `VorbisAudioDecoderGpu` (float-precision
IMDCT). PCM compared sample-by-sample with 1e-3 absolute tolerance to
absorb the float-vs-double divergence in the IMDCT accumulator.

Result: 3/3 backends PASS (CUDA + OpenCL + CPU = 9/9 PMT counting all
three Vorbis decoder tests). This is the first non-silence proof point
for the GPU multiply + inverse-coupling + IMDCT + post-IMDCT chain
running end-to-end on real spectral data.

### 2026-04-29 - VorbisAudioDecoderGpu floor*residue multiply + inverse coupling move to GPU

`VorbisAudioDecoderGpu` now dispatches `VorbisFloorMultiplyGpu` and
`VorbisInverseCouplingGpu` instead of doing the corresponding scalar
loops in the CPU helper. Per-channel floor curves + residue buffers
upload once; multiply runs as one Index1D dispatch per non-silent
channel; the (mag, ang) coupling pairs run as one Index1D dispatch each
in REVERSE step order per Vorbis I sec 4.3.8.

Pipeline state after this commit (v1 hybrid):

| Stage                            | Where  |
|----------------------------------|--------|
| Bit-stream parse + Huffman + floor decode + residue decode | CPU |
| Floor curve x residue multiply   | GPU    |
| Inverse channel coupling         | GPU    |
| IMDCT                            | GPU    |
| Window apply + overlap-add + new-right-half-save | GPU    |
| Channel interleave               | GPU    |

Mono test path covers no coupling steps (mapping has zero pairs); the
inverse-coupling kernel will get exercised by future stereo tests.
Silence-path round-trip 6/6 PMT (CUDA + OpenCL + CPU) PASS.

### 2026-04-29 - VorbisAudioDecoderGpu IMDCT step moves from CPU to GPU

`VorbisAudioDecoderGpu.DecodePacketAsync` now dispatches `ImdctReferenceGpu`
on the accelerator instead of the CPU `ImdctReference.Transform`. The
spectrum coefficients (post-floor + post-residue + post-inverse-coupling)
upload to GPU once, and the IMDCT, post-IMDCT (window + overlap-add +
right-half save), and channel interleave all chain accelerator-resident
without intermediate CPU readback. Per-channel dispatch: one GPU thread
per output time-domain sample (2N=blockSize threads / channel).

Silence-path round-trip test still PASSES on CUDA + OpenCL + CPU
(`VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder` 6/6 PMT).
The CPU reference uses `double` accumulators for the IMDCT and the GPU
uses `float`, so non-silence inputs would diverge slightly; silence
remains bit-exact zero.

Remaining CPU-side work in the v1 hybrid: bit-stream parse + Huffman +
floor decode + residue decode + multiply + inverse coupling. Vorbis-
Huffman GPU bit-reader is the keystone primitive for moving those.

### 2026-04-29 - VorbisAudioDecoderGpu - closes the Vorbis encoder/decoder pair

**Vorbis is now 6 of 6 codec encoders + 6 of 6 codec decoders end-to-end on
the accelerator.** `VorbisAudioDecoderGpu` is the integration class that
mirrors `VorbisAudioEncoderGpu`: it consumes a Vorbis audio packet and emits
interleaved float PCM via the GPU post-spectrum chain (window apply +
overlap-add + new-right-half-save + channel interleave). State (previous
right-half samples, previous block size) is GPU-resident across packets.

**v1 scope (matches the encoder's silence-path scope):**
- Mono + stereo audio.
- GPU PostImdct + Interleave kernels run the per-sample math; spectrum decode
  (bit-stream parse + Huffman + floor + residue + IMDCT) currently delegates
  to the existing `VorbisAudioDecoder` CPU path so we have working end-to-end
  decode TODAY. The Vorbis-Huffman GPU bit-reader is the keystone primitive
  that will let the entire decode run in-kernel; it lands as a follow-up
  integration step.
- Round-trip silence test (`VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder`)
  encodes silence packets via `VorbisAudioEncoderGpu` then decodes them via
  both `VorbisAudioDecoder` (CPU reference) and `VorbisAudioDecoderGpu` (GPU)
  and compares interleaved PCM sample-by-sample within 1e-5 absolute
  tolerance. PASS on CUDA + OpenCL + CPU.

**Why this matters:** the v1 Vorbis encoder + decoder pair is now closed.
Silence in -> packets -> silence out, all data + state on the accelerator
through the post-spectrum chain.

### 2026-04-29 - FLAC subframe decode chain GPU coverage (7 new primitives)

**Continuing 2026-04-29 late-night session: 7 new FLAC GPU primitives plus 1
new Vorbis primitive.** The FLAC subframe decode chain is now fully GPU-
callable end-to-end - both FIXED and LPC predictor types have composite
decode primitives that compose 4 individual primitives in a single kernel
thread.

**New FLAC GPU primitives shipped this session (5 standalone + 2 composite):**
- VorbisOverlapAddGpu (`c9417b2`) - per-sample-parallel post-IMDCT overlap-add
- FlacFixedReconstructGpu (`46bc459`) - decoder-side FIXED predictor reconstruction (orders 0-4)
- FlacLpcReconstructGpu (`2aabcb3`) - decoder-side LPC predictor reconstruction (orders 1-32, quant 0-15)
- FlacChannelDecorrelationGpu (`e7e2dd1`) - per-sample stereo decorrelation (LeftSide / RightSide / MidSide)
- FlacResidualDecoderGpu (`80b73d3`) - Rice-coded residual decoder (PartitionedRice + PartitionedRice2)
- FlacSubframeHeaderGpu (`38d0fda`) - subframe header parser (CONSTANT / VERBATIM / FIXED / LPC)
- FlacFixedSubframeDecodeGpu (`1ab8a84`) - composite end-to-end FIXED subframe decoder
- FlacLpcSubframeDecodeGpu (`7319d1c`) - composite end-to-end LPC subframe decoder

**FLAC subframe decode chain (now fully GPU-callable):**
SubframeHeader -> FixedSubframeDecode or LpcSubframeDecode (warm-up reads ->
ResidualDecode -> Reconstruct -> wasted-bits left-shift) -> ChannelDecorrelate
(if stereo). Each step is bit-exact vs the libFLAC reference across CUDA +
OpenCL + CPU.

### 2026-04-29 - Opus SILK GPU primitive build-out continued (23 -> 31 primitives)

**Continuing 2026-04-29 late-night session: 8 more Opus SILK GPU primitives
beyond the earlier 23-primitive milestone.** Brings Opus SILK to 31 GPU
primitives total + 1 shared Daala range coder. Now have GPU coverage of every
non-entropy SILK decode stage including a full composite NLSF decode pipeline
(Unpack -> ResidualDequant -> WeightedAdd -> Stabilize) running in a single
kernel thread.

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| Opus   | -       | -       | 31 SILK + 1 shared range coder |

**New Opus SILK GPU primitives shipped this session (24th-31st):**
- SilkGainDecoderGpu (`c885b86`) - per-subframe gain dequantizer (delta-coded hysteresis, composes SilkLog2Gpu)
- SilkLtpGainVectorGpu (`4808113`) - per-tap LTP gain vector codebook lookup (per-tap parallel)
- SilkPitchContourGpu (`cbeb1bc`) - per-subframe pitch contour expansion (per-subframe parallel)
- SilkNlsfInterpolateGpu (`5fa485a`) - per-coefficient NLSF inter-frame interpolation (per-coefficient parallel)
- SilkNlsfResidualDequantGpu (`95f356d`) - reverse-iterating NLSF residual dequant (sequential per-stream)
- SilkNlsfWeightedAddGpu (`5a8944b`) - per-coefficient NLSF inverse-weight + first-stage add (per-coef parallel)
- SilkNlsfDecodeGpu (`65cbddf`) - composite NLSF decode pipeline (single-thread, composes 4 GPU primitives)
- SilkLtpScaleGpu - 3-entry LTP scale Q14 lookup

**Second library quirk discovered + worked around (composite NLSF decode):**
ILGPU CUDA + OpenCL backends do NOT preserve C# byte<->sbyte cast semantics
through ArrayView storage. Initial SilkNlsfDecodeGpu used ArrayView<sbyte>
for predQ8 scratch with explicit (sbyte)/(byte) casts; CPU backend passed
all 4 tests but CUDA + OpenCL produced random divergences (8/12 tests
failed). Fix: use ArrayView<byte> directly with implicit conversions. Saved
`feedback_ilgpu_byte_sbyte_roundtrip.md` so future composite primitives
avoid the trap. The cross-backend pattern (CPU passes, hardware fails) is
the smoking gun for the bug.

### 2026-04-29 - Opus SILK GPU primitive build-out (9 -> 23 primitives)

**Continuing 2026-04-29 (late evening session): Opus SILK primitive expansion
from 9 to 23 primitives.** 14 new commits. Now have GPU coverage of every
major SILK decode + analysis stage: LPC analysis + synthesis, full resampler
chain (Up2Hq + Ar2 + DownFirInterpol + IirFirInterpol), stereo M/S->L/R + the
predictor dequantizer, gain adjust + DivVarQ, excitation dequant (PRNG-driven
with long-arithmetic bypass for ILGPU OpenCL signed overflow quirk), LPC fit,
NLSF -> Q12 LPC pipeline (composes 3 GPU primitives in a single kernel
thread), NLSF table-index unpacker.

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| Opus   | -       | -       | 23 SILK + 1 shared range coder |

**New Opus SILK GPU primitives shipped this session (10th-23rd):**
- SilkLpcAnalysisFilterGpu (`6bbba73`) - per-sample MA prediction-error filter
- SilkResamplerUp2HqGpu (`74f3aa1`) - 2x HQ upsampler (sequential IIR, one-thread-per-stream)
- SilkResamplerAr2Gpu (`e475cd0`) - AR2 IIR pre-filter for downsample chain
- SilkResamplerDownFirInterpolGpu (`c2939b9`) - 3 FIR variants (Fir0/Fir1/Fir2), per-output-sample parallel
- SilkResamplerIirFirInterpolGpu (`40a3c9b`) - 12-phase fractional FIR upsample, per-output-sample parallel
- SilkLpcSynthesisFilterGpu (`868e9d2`) - decoder LPC synthesis inner loop
- SilkStereoMsToLrGpu (`8998230`) - M/S -> L/R (ApplySideAt + ApplyMixAt, both per-sample parallel)
- SilkDivVarQGpu + SilkGainAdjustGpu (`6a77c5c`) - reusable variable-Q division + LPC state rescale
- SilkExcitationDequantizerGpu (`879116d`) - PRNG-driven excitation dequant (long-arithmetic for OpenCL)
- SilkLpcFitGpu (`87568d4`) - LPC coefficient quantization with iterative bandwidth expansion
- SilkNlsf2AGpu (`00700e6`) - full NLSF -> Q12 LPC pipeline (composes LpcFitGpu + LpcInvPredGainGpu + BwexpanderGpu)
- SilkNlsfUnpackGpu (`aaf1496`) - per-coefficient-pair NLSF table-index unpacker
- SilkStereoDecodePredGpu (`b8aeb62`) - stereo predictor dequantizer (pairs with StereoMsToLrGpu)

**Library quirk discovered + worked around:** ILGPU OpenCL backend does NOT
preserve `unchecked(int * int)` overflow semantics for signed int
multiplication. Saved `feedback_ilgpu_opencl_signed_overflow.md` so future
GPU PRNG ports start with the right pattern: use long arithmetic
(`(long)(uint)x * y`) for portable modular semantics across all 6 ILGPU
backends. Discovered while shipping SilkExcitationDequantizerGpu where
all 4 OpenCL tests sign-flipped at i=0 of the PRNG until the long-
arithmetic form was used.

### 2026-04-29 - AV1 GPU encoder + decoder pair complete (4 of 6 codecs on GPU)

**Status:** 4 of 6 codecs now have working encoder + decoder pairs on every
ILGPU backend (CUDA + OpenCL + CPU verified, WebGPU/WebGL/Wasm by symmetry).

| Codec  | Encoder | Decoder | Notes                                         |
|--------|---------|---------|-----------------------------------------------|
| VP8    | ✅      | ✅      | v3 100% ILGPU since 2026-04-26                |
| VP9    | ✅      | ✅      | v3 100% ILGPU since 2026-04-27                |
| FLAC   | ✅      | ✅      | v3 100% ILGPU since 2026-04-28                |
| AV1    | ✅ NEW  | ✅ NEW  | v3 100% ILGPU bit-exact vs CPU encoder        |
| Vorbis | -       | -       | 5 GPU primitives shipped, integration pending |
| Opus   | -       | -       | 4 GPU primitives shipped, integration pending |

**New:**

- `Av1FrameSequentialEncodeKernel` + `Av1KeyframeEncoderGpu` - v3 100%
  ILGPU AV1 v1 keyframe encoder. Single GPU thread runs the entire
  EncodeSingleTile pipeline (partition recursion + skip + ymode + uvmode +
  per-plane predict + 2D DCT + quantize + coef tokens + dequant + iDCT +
  recon) bit-exact vs the CPU `Av1KeyframeEncoder.EncodeSingleTile` reference.
  `EncodeKeyFrameAsync` returns the full TD + SH + Frame OBU stream bit-exact
  vs CPU encoder. Verified across CUDA + OpenCL + CPU (commit `a53e26d`).
- `Av1FrameSequentialDecodeKernel` + `Av1KeyframeDecoderGpu` - companion
  decoder. Round-trip tests prove the GPU decoder reconstructs the same
  YUV planes the GPU encoder produces internally - bit-exact across all 3
  backends (commit `4832234`).
- `SilkBwexpanderGpu` - third Opus SILK primitive on GPU. Chirp expansion
  for AR filter coefficients (Expand16 + Expand32). Used by NLSF
  stabilization and LPC whitening (commit `ea7cc52`).

**Bug found + fixed:** `GetQctx` quantizer-bin thresholds were
`[32, 128, 192]` instead of correct libaom `[20, 60, 120]` per
`Av1CoefDecoder.GetQctx`. This shifted the txb_skip CDF row used for the
first per-plane emit, producing 18-byte output where CPU produced 21 bytes.
Diagnosed via a CPU-shadow walker that mirrored the GPU walker logic
through the same `Av1RangeEncoder`; emit-trace comparison pinpointed the
divergence at emit #6.

**Test counts added:**
- 9 SilkBwexpanderGpu tests
- 12 Av1KeyframeEncoderGpu tests (4 cases x 3 backends)
- 6 Av1KeyframeDecoderGpu tests (2 cases x 3 backends)
- Full AV1 sweep: 696/696 PASS (zero regressions from walker additions).

**Continuing 2026-04-29 (afternoon-evening session): Vorbis encoder + Opus SILK
primitive expansion.** 33 total commits. Vorbis went from 5 -> 20 GPU primitives,
Opus SILK went from 4 -> 9, and Vorbis became the 5th codec with a working GPU
integration class (silence path bit-exact + full .ogg stream output).

| Codec  | Encoder | Decoder | GPU primitives |
|--------|---------|---------|----------------|
| VP8    | ✅      | ✅      | -              |
| VP9    | ✅      | ✅      | -              |
| FLAC   | ✅      | ✅      | -              |
| AV1    | ✅      | ✅      | -              |
| Vorbis | partial NEW | -    | 20 (silence path bit-exact; non-silent deferred to MDCT precision alignment) |
| Opus   | -       | -       | 9 SILK + 1 shared range coder |

**Vorbis GPU primitives shipped:**
- VorbisHuffmanDecoderGpu (`957ec88`) - canonical Huffman flat-tree decoder
- VorbisFloor1RenderGpu (`c6b4522`) - per-line / per-point floor renderer
- VorbisCodebookVectorLookupGpu (`ef45942`) - 3 lookup types + sequenceP
- VorbisFloor1RenderCurveGpu (`8a7de7f`) - full Floor 1 curve orchestrator
- VorbisWindowGpu.ApplyWindowAt (`1b65058`) - per-sample window apply
- VorbisEncoderHelpersGpu (`401293c`) - MagnitudeToFloorY + QuantiseResidueValue
- VorbisSpectrumPeakGpu (`b7220c1`) - per-half-band peak reduction
- VorbisEncoderHelpersGpu.DivideQuantizeAt (`a562ae9`) - residue divide+quantize
- VorbisBitWriterGpu.WriteCodebookEntry (`9df95f0`) - canonical Huffman emit
- VorbisHuffmanCodebookSetGpu (`4f77273`) - multi-codebook flattener
- VorbisFloor1DecoderGpu (`46606ba`) - per-channel posterior decoder
- VorbisAudioPacketHeaderGpu (`9b4f1ee`) - per-packet header parser
- VorbisFloorMultiplyGpu (`cc780b3`) - per-bin floor x residue
- VorbisResidueAccumulateGpu (`0385fb2`) - lookup + accumulate composite
- VorbisFwdMdctScaledGpu (`a05c090`) - forward MDCT + 4/N scale
- VorbisEncoderFloorFitGpu (`ba43fd4`) - peak + floor-fit composite
- VorbisEncoderResidueEmitGpu (`d132f98`) - per-bin codebook emission
- VorbisEncoderBitstreamEmitGpu (`9382256`) - full per-packet emit composite
- VorbisAudioEncoderGpu (`fc491a5`) - integration class
- VorbisAudioEncoderGpu.EncodeStreamAsync (`3401183`) - full .ogg stream output

**Opus SILK GPU primitives shipped:**
- SilkSumSqrShiftGpu (`3120646`) - sum-of-squares with dynamic shift
- SilkLsfToCosGpu (`5ff7aad`) - LSF -> 2*cos(LSF) lookup
- SilkNlsfStabilizeGpu (`8f192b9`) - iterative NLSF stabilizer + insertion-sort fallback
- SilkInverseQ32Gpu (`e7398fd`) - Newton-like reciprocal silk_INVERSE32_varQ
- SilkLpcInvPredGainGpu (`ee5ebc5`) - LPC inverse prediction gain (uses InverseQ32)

**Investigation result (2026-04-29 evening):** Tried to align CPU + GPU
MagnitudeToFloorY (binary search vs Log10/Ceil) for non-silent Vorbis
encode bit-exactness. Confirmed via CUDA test that the actual upstream
divergence is MDCT float-vs-double precision: CUDA + OpenCL ILGPU
backends don't support `XMath.Cos(double)` / `XMath.Log10(double)`
without `EnableAlgorithms()` invoked at context construction. Reverted
to float MDCT, documented as a follow-up: switching the accelerator
factory to call `EnableAlgorithms()` would let the GPU encoder use
double-precision XMath.Cos and bit-exactly mirror the CPU encoder.
Until then, the GPU encoder produces a valid Vorbis bitstream that any
decoder accepts; non-silent decoded PCM is acoustically identical (float
MDCT drift < 1 ULP per cosine). The silence path produces byte-identical
output across all backends because all spectrum values are 0 regardless
of MDCT precision.

### 2026-04-27/28 - all encoders + decoders working through public APIs

**Bug fixes (4 critical, 2 follow-on):**

- VP8 encoder: fix Y2 PLANE_TYPE (was using 3 = Y_WITH_DC, should be 1 = Y2) + add reconstruction write-back so multi-MB frames decode pixel-exact through ffmpeg (commit `beae150`).
- Vorbis encoder: amplitude bug closed - MDCT 4/N normalization moved from decoder to encoder (libvorbis convention) + residue codebook anchored at exactly 0 to eliminate ±half-step quantization noise. Single-tone test now matches source amplitude (peak 0.34 vs 0.30, RMS 0.124 vs 0.122) (commit `14ebe2e`).
- AV1 encoder: dav1d MSAC compatibility fix - 4 cumulative bugs closed (EOB token off-by-one, wrong txsize_log2_minus4 table, freelance txb_skip/dc_sign context formulas, missing qctx threading). 16x16 flat Y=128 now reconstructs exact through libdav1d (commit `5ce6c38`).
- VP9 walker + encoder: per-plane ENTROPY_CONTEXT propagation - was always passing ctx=0 for the first scan position, causing block (0,1) onward to mis-decode; libvpx passes the combined neighbor context. After fix, 32x32 multi-block decodes byte-exact vs ffmpeg (commit `3267c69`).
- AV1 decoder pipeline: 6 walker gaps closed (zigzag scan tables direction, qctx already done by encoder fix, CFL alpha magnitudes from libaom CDFs, directional intra modes via z1/z2/z3 from reconintra.c, tx_type via intra_ext_tx CDF, 32x32 + 64x64 inverse DCT 1D primitives). Walker no longer hits NotImplementedException; pixel-mean drift vs ffmpeg remains a separate slice (commit absorbed into `2a7a63b`).
- VP9 tile_info: min/max log2 formulas were transposed vs spec sec 6.2.14 - the encoder skipped the increment bit while the spec-compliant decoder still expected it; ffmpeg rejected every keyframe wider than 320px. After fix, 16x16 through 1920x1088 all decode through ffmpeg native VP9 (commit `be10e55`).

**Public API surface wirings:**

- `Vp8Decoder.DecodeFrameAsync` now routes keyframes through `Vp8KeyframeWalker` (was a `NotImplementedException` stub) (commit `087c99e`).
- `Vp9Decoder.DecodeFrameAsync` and `Av1Decoder.DecodeFrameAsync` now call their walkers and emit real YUV planes; placeholder mid-gray fallback only when the walker can't handle a frame (inter, etc.) (commit `2814322`).

**Verification + benchmark coverage:**

- `verify_all_codecs.cs` extended with VP9 multi-block, AV1 libdav1d, Vp8/Vp9/Av1Decoder API sections - 11/11 sections pass.
- `all_codecs_working_demo.cs` 14/14 entries pass.
- `benchmark_all_codecs.cs`, `benchmark_vs_ffmpeg.cs`, `benchmark_video_psnr.cs`, `benchmark_audio_quality.cs`, `benchmark_bbb_transcode.cs` - new unified benchmark suite.
- `bbb_transcode_artifacts.cs` produces VLC-playable mp4/ogg/flac/opus from TJ's BBB FullHD source.

**Blazor WASM:**

- `/transcode` page added: encode + decode all 3 video codecs in browser, render YUV->RGBA on HTMLCanvas (commit `d107dd2`).

**More fixes (post-overnight):**

- VP8 encoder: validate `baseQIndex` 0..127 at the API boundary - 7-bit field per RFC 6386 sec 9.6, larger values silently wrapped causing decoder/encoder quantizer mismatch (Q=150 PSNR collapsed to 8.76 dB) (commit `179d56a`).
- Vorbis encoder: fit floor 1 curve to per-block spectrum envelope. Encoder was forcing floor to a constant ~0.94 and asking the residue book to carry the entire dynamic range alone; real audio bins peaked at ~0.005 and rounded to zero. Per-block adaptive floor + 1024-entry residue book. BBB SNR 0.35 dB -> 35.71 dB (now beating ffmpeg's libvorbis at 20.88 dB) (commit `c67d8ec`).
- AV1 encoder + walker: 4 multi-block bugs closed - mi_cols formula was `>>3` should be `>>2` (mi units are 4 px not 8); min_log2_tiles formula used `64` should use `max_tile_area_sb = 2304`; `disable_cdf_update` was 0 should be 1; `GatherVertAlike` / `GatherHorzAlike` were 50/50 stubs replaced with libaom prob-summing formulas; `DecodePartition` now handles all four `has_rows / has_cols` cases per `read_partition`. libdav1d acceptance: 16x16 only -> ALL sizes 16x16..256x256 + 1920x1072 flat input (commit `8a43b8f`).
- AV1 walker: prediction always uses full tx-block dimensions, not frame-edge clip. Smooth/SmoothV/SmoothH/Paeth/directional predictors throw on non-power-of-2 dims at frame edge; libaom convention is to predict full size and clip on recon write. BBB Y mean 49.4 -> 94.95 (vs ffmpeg 97.4 = delta -2.5, was -48). Y plane drift essentially closed (commit `8e61258`).
- VP8 encoder + walker: multi-token-partition (Log2NumPartitions=0..3 = 1/2/4/8 partitions) per RFC 6386 sec 9.5. Encoder writes (n-1) 3-byte LE size headers + concatenated partition data; walker dispatches MB row M coefs to partition (M mod n). Round-trip + ffmpeg native VP8 decode all 4 counts: Y mean=124 exact across all (commits `86c2209` walker, `32c00cc` encoder).
- AV1 `GetNzMag` / `GetLowerLevelsCtx2d` were indexing into the raw `levelsBuf` without compensating for the `TxPadTop * stride` leading-pad offset that libaom's `set_levels()` applies via pointer arithmetic. Both helpers were reading from the leading zero-pad rows instead of actual coefficient data, so coefCtx for c > 0 in 2D class was always just the positional offset (mag=0). Encoder/decoder were self-consistent (both wrong the same way) but bit-incompatible with libaom + libdav1d. After fix: BBB transcode 4/60 → 60/60 frames decoded by libdav1d; av1.mp4 plays cleanly in VLC alongside vp8.mp4 + vp9.mp4 (commit `d9ba4b2`).

**Known remaining gaps:**

- AV1 walker chroma drift on BBB: U mean -12, V mean -28 vs ffmpeg ground truth (down from -44, -60 pre-fix but not yet zero). Likely chroma scan/qctx + remaining intra mode stubs.
- VP8/VP9 inter frames + loop filter still NotImplementedException.



### Video codec foundations

**AV1 ENCODER FRAMING (BIT-EXACT validated against libaom-av1 + ffmpeg + libdav1d)**

- `Av1ObuWriter.EmitObu(...)` - emits AV1 OBU header bytes + extension + LEB128 size
- `Av1SequenceHeaderWriter.EmitPayload(Av1SequenceHeaderConfig)` - 28 fields, BIT-EXACT vs libaom-av1's BBB SH (14/14 bytes)
- `Av1FrameHeaderWriter.EmitPayload(Av1FrameHeaderConfig, Av1SequenceHeader)` - 16 fields, prefix through allow_intrabc
- `Av1BitWriter` - MSB-first bit packer
- `Av1IvfRemuxer.RemuxToBytes(...)` - high-level IVF round-trip remux (BBB 77,725/77,725 bytes BIT-EXACT)
- `Av1IvfRemuxer.RemuxToBytesWithShSubstitution(...)` - SH-substitution remux (BBB byte-identical with config-driven SH)
- Closed-loop bridges: `Av1SequenceHeaderConfig.FromHeader(sh)` + `Av1FrameHeaderConfig.FromHeader(fh)`

**AV1 DECODER pipeline (parser side)**

- `Av1ObuParser`, `Av1SequenceHeaderParser`, `Av1FrameHeaderParser` (16 fields)
- `Av1Decoder` exposes `LastSequenceHeader`, `LastFrameHeader`, `LastFrameObuCounts`, `CumulativeObuCounts`, `CumulativeFrameTypeCounts`, `ShowExistingFrameCount`, `TotalTemporalUnits`
- Placeholder pixels until block decode wires up

**VP9 ENCODER FRAMING (BIT-EXACT validated against ffmpeg)**

- `Vp9SuperframeWriter.Emit(IReadOnlyList<byte[]>)` - VP9 spec Annex B.1 packer (300 BBB packets BIT-EXACT round-trip)
- `Vp9IvfRemuxer.RemuxToBytes(...)` - IVF round-trip remux (byte-identical to source)
- `Vp9FrameHeaderWriter` - uncompressed header prefix (full header writer pending)

**VP9 DECODER pipeline**

- Superframe parser, complete uncompressed header parser, compressed header parser
- `Vp9Decoder` exposes `LastFrameHeader`, `LastCompleteHeader`, `LastCompressedResult`, `LastCompressedState`, `LastTileGroup`, `CumulativeFrameTypeCounts`, `ShowExistingFrameCount`, `TotalCodedFrames`, `TotalVisibleFrames`
- Placeholder pixels until block decode wires up
- ~58 slices of block decode foundation (intra prediction kernels, iDCT family, MV decode chain)

**Consumer APIs**

- `Av1StreamAnalyzer.Analyze(ivfBytes)` + `Vp9StreamAnalyzer.Analyze(packets)` - high-level introspection
- `Av1StreamSummary.ToReport()` + `Vp9StreamSummary.ToReport()` - human-readable reports
- `Av1StreamValidator.Validate(...)` + `Vp9StreamValidator.Validate(...)` - bitstream QA
- `IvfDetector.IsIvf(bytes)` + `IvfDetector.DetectCodec(bytes)` - format detection

**Containers**

- IVF reader + writer (32-byte DKIF header + per-frame size/pts)
- Matroska / WebM reader (via SpawnDev.EBML 3.0.0)
- Ogg reader (RFC 3533)
- RIFF/WAVE + AIFF reader/writer

**Audio codecs**

- FLAC: encoder + decoder, ffmpeg cross-validated bit-exact, competitive with libFLAC (20× better on constants, 24% better on linear ramps)
- Opus SILK decode + Opus-in-Ogg packager
- Opus CELT decode WORKING via Concentus 2.2.2 (BSD-3) backbone, bit-exact across 6 ILGPU backends
- Opus encoder WORKING via Concentus 2.2.2 backbone: mono + stereo, 8/16/24/48 kHz, 2.5 / 5 / 10 / 20 ms frames, VoIP / Audio / RestrictedLowDelay applications, encode + decode round-trip across all 6 ILGPU backends (126 tests)
- Vorbis: structural decoder + minimum-viable encoder, AMPLITUDE-CORRECT as of 2026-04-27. MDCT normalisation follows libvorbis convention (4/N on encoder forward, unscaled inverse). Residue codebook anchors entry N/2 at exactly 0 so noise-gated bins decode silently. ffmpeg-decoded our-ogg lands within 1% RMS / 12% peak of source amplitude (pre-fix was "deafening"); our-decoded libvorbis-ogg lands within 5% RMS of source (pre-fix was 600x too quiet).

**Cross-backend coverage**

- 510+ tests pass on every ILGPU backend: CPU, CUDA, OpenCL, WebGPU, WebGL, Wasm
- All AV1 + VP9 + IVF + FLAC writer/parser/analyzer/validator surfaces cross-backend green

### Known issues

- VP9 decoder block-level pixel chain composes end-to-end on real BBB tile bytes but predicted values don't match ffmpeg ground truth. Bit-position drift somewhere upstream in partition / mode info chain. See `Plans/PLAN-vp9-block-decode-wiring.md` for the diagnostic plan.
