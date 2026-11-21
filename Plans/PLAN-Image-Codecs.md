# PLAN - Image codecs in SpawnDev.Codecs

TJ directive 2026-10-22: "A full set of image codecs (encode/decode) needs to be in [SpawnDev.Codecs] also, not just
video." Consumers: SpawnGen (Lore, ComfyUI parity: PNG with text chunks, JPEG, WebP, later animated), everything that
today leans on a browser canvas or a native library.

## Scope (patent-clean only, same rule as video)

| Format | Decode | Encode | Notes |
|---|---|---|---|
| PNG / APNG | done (default image) | PNG encode (deflate compressor on the accelerator) | tEXt/iTXt/zTXt/eXIf metadata read + write; APNG frames |
| JPEG | baseline + extended sequential + progressive, all samplings, restart markers, Adobe CMYK/YCCK | baseline (+ optional progressive) | EXIF orientation reported, optionally applied |
| WebP | VP8L lossless, VP8 lossy (+ALPH), VP8X container, animation | VP8L lossless, VP8 lossy (reuses `Vp8KeyframeEncoderGpu` bitstream) | EXIF/XMP/ICCP chunks |
| GIF | LZW, interlace, transparency, animation | LZW, palette quantization (on the accelerator), animation | |
| BMP | 1/4/8/16/24/32 bpp, RLE4/RLE8, bitfields | 24/32 bpp | |
| AVIF | needs a spec-complete AV1 intra decoder (CDEF, loop restoration, all intra tools) | via `Av1KeyframeEncoderGpu` | LAST; depends on the AV1 decoder work |
| HEIC/HEIF (HEVC) | **out of scope** | **out of scope** | patent-encumbered, same as H.264/H.265 |

## The rules (non-negotiable; see AGENTS.md and this repo's CLAUDE.md)

1. **Host is a pure coordinator.** Every decoder/encoder stage that touches pixel or entropy data is an ILGPU kernel.
   The host may: allocate buffers, upload the source bytes (necessary I/O), dispatch kernels, read back **scalars and
   small metadata blocks** (<= a few hundred ints), build **tiny metadata blobs** (a PNG tEXt chunk, a JPEG header
   blob of < 1 KB), and read back the final output only into a caller-requested managed/JS sink.
2. **Bulk data never enters the .NET WASM heap in the browser.** JS typed arrays go to the accelerator with
   `IBrowserMemoryBuffer.CopyFromJS` (`DeviceTransfer.Upload(view, TypedArray)`) and come back as a `Uint8Array`
   (`DeviceTransfer.DownloadUint8ArrayAsync`).
3. **Three overloads per entry point**, all funnelling into the `ArrayView1D` one (see "API contract").
4. **Sequential entropy coding runs as ONE GPU thread** (or one per independent stream: restart intervals, tiles,
   frames). Still accelerator-resident. Correct first, then parallelize.
5. **Correct first time, then fast.** Every change has a unit test that can fail (red-check it: break the fix, watch the
   test fail by name, restore by inverse edit). Tests use REAL files with an INDEPENDENT oracle.
6. **No CPU fallback branches, no workarounds.** A library bug is fixed in the library (ILGPU codegen bugs go to Data).

## API contract

Namespace `SpawnDev.Codecs.Images.<Format>`, classes `<Format>DecoderGpu` / `<Format>EncoderGpu` (`IDisposable`,
constructed from an `Accelerator`, **not thread-safe**: one in-flight call per instance). Reference implementation:
`Images/Png/PngDecoderGpu.cs`.

```csharp
// ---- decode ----
Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength);     // header facts, no pixels
Task<ImageInfo> DecodeAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
                            ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8);
Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file);                                  // managed
Task<DecodedImage> DecodeAsync(ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8);
[SupportedOSPlatform("browser")] Task<ImageInfo> ReadInfoAsync(TypedArray file);           // browser
[SupportedOSPlatform("browser")] Task<DecodedImageJS> DecodeAsync(TypedArray file, PixelFormat format = PixelFormat.Rgba8);

// ---- encode ----
static int GetMaxEncodedSize(int width, int height, <Format>EncodeOptions? options = null);
Task<int> EncodeAsync(ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
                      ArrayView1D<byte, Stride1D.Dense> output, <Format>EncodeOptions? options = null);   // returns length
Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, <Format>EncodeOptions? options = null);
[SupportedOSPlatform("browser")] Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, <Format>EncodeOptions? options = null);
```

* Pixels are tightly packed `PixelFormat.Rgba8` (straight alpha, rows top to bottom); `Rgba16` where the format has 16-bit.
* Errors: `ImageDecodeException(ImageFormat, reason)` with a SPECIFIC reason; never silently produce garbage.
* Decoders do not apply EXIF orientation unless an option says so; `ImageInfo.Orientation` reports it.
* Animation: `ImageInfo.FrameCount`; frame access API is designed once the first animated format lands (shared by
  APNG / GIF / WebP): until then `DecodeAsync` returns the default / first frame.

## Pipeline conventions (learned the hard way; see PngKernels.cs / InflateGpu.cs)

* Scalars and layout between kernels travel in ONE small `ArrayView<int>` "meta" block with named slot constants
  (`PngMeta`), so a kernel and the host cannot disagree about a slot.
* Host reads the meta block back after ONE `await SynchronizeAsync()` to learn sizes, then sizes the next buffers
  (`StageBuffer<T>` is grow-only, grown only at the start of a public call).
* Never dispose a device buffer a pending dispatch still reads: `await SynchronizeAsync()` first (browser backends).
* Kernel dialect rules that have bitten: no `out` parameters; no `a[i]++` on ArrayView elements (write `a[i] = a[i] + 1`);
  no `static readonly` tables in kernels (compute arithmetically, or upload a constant buffer once); loops that need an early
  exit use a flag in the body, not an early `return` from inside; compound short-circuit predicates in a branch condition
  are built into `bool` locals with `&` / `|` first (WGSL has dropped threads on inline `a && b`); struct state passed
  `ref` into static helpers is fine (`InflateState`, `Vp8BoolDecoderGpuState`); keep kernel parameter counts small by
  packing scalars into the meta block.
* **Loop and branch conditions are single `bool` locals.** A loop header with `&&` / `||` (`while (a && b)`) HUNG the WebGPU
  device (DXGI_ERROR_DEVICE_HUNG, and the hang poisons every later test in that Chrome) while CPU/CUDA/OpenCL passed. Write
  `bool go = (a) & (b); while (go) { ...; go = (a) & (b); }`. The same goes for compound `if` conditions (build a `bool` with
  `&` / `|` first). A kernel that passes the CPU lane has NOT been proven for WebGPU/Wasm: the coordinator runs those lanes.
* Use `DeviceTransfer` / `StageBuffer` / `PlanePadKernel`; do not write new host copy loops.

## Test conventions

* Fixtures: `SpawnDev.Codecs.Demo.Shared/TestData/<Format>/` (embedded resources, so desktop and browser lanes read the
  same bytes), generated by a checked-in script `scripts/images/make_<format>_fixtures.py`. Truth is derived from the
  source samples or from a reference decoder (libjpeg / libwebp / giflib through Pillow), **never from our own code**.
  Pillow + numpy live in a venv (`python -m venv`, `pip install pillow numpy`); run scripts as `python -I script.py`.
* Tests: `UnitTests/CodecsTestBase.<Format>*Tests.cs`; load every fixture, decode on the accelerator, compare to truth,
  collect ALL failures with the fixture name and the first differing byte before failing. Cover: the managed overload,
  the `ArrayView1D` overload into a caller buffer pre-filled with `0xAB`, error paths (bad signature, truncation,
  corruption) asserting the specific reason, and (browser lanes) the TypedArray overload.
* Run on the CPU backend while iterating: `scripts/dev/build-console.ps1` then `scripts/dev/run-desktop-tests.ps1 -Pattern X`.
  CUDA/OpenCL/WebGPU/WebGL/Wasm lanes (PlaywrightMultiTest) are run by the coordinator, one at a time, when the
  shared GPU is free.

## WebP lossless ENCODE: size parity with libwebp (2026-11-14)

Measured on SpawnGen's 25 LTX frames (512x352, q80; Pillow 13 / libwebp 1.6.0, method 4 = ComfyUI's default):
full-frame stills ours 3,124,892 B vs libwebp 2,957,198 (+5.7%); WebPAnimEncoder animation ours 3,540,954 vs 3,240,800
(+9.3%, Codecs 5901c0f). Speed is no longer the gap (0.20 s / 0.32 s per frame on the RTX 4070 vs libwebp's 5.9 s
clip). First target: the predictor-mode choice, which both paths share.

libwebp 1.6.0 at method 4 (src/enc/predictor_enc.c, BSD): predictor bits = min(7 - method, 5) = 3 (8x8 tiles, then
ClampBits), min_bits == max_bits (the multi-resolution search is method 5..6 only). Tiles in raster order; for each,
all 14 modes are scored on the tile's residuals (exact = 0: transparent pixels take their prediction inside the tile;
rows above / the left column / the top-right are the original picture) by
PredictionCostSpatialHistogram = sum over the 4 channels of PredictionCostBias(tile, 1, 0.94) (favours residuals near
0: -(c[0] + sum_{i=1..15} 0.94 * 0.6^(i-1) * (c[i] + c[256-i])) / 10 bits) + VP8LCombinedShannonEntropy(tile,
accumulated) (the entropy of the tile and of tile + all earlier tiles' chosen histograms), minus 15 bits when the mode
equals the left tile's and again for the above tile's. The chosen mode's histogram joins `accumulated`. Then
VP8LOptimizeSampling coarsens the predictor image when its modes are uniform in 2^k blocks.

GPU form (planned): the choice is serial over tiles (accumulated + neighbour bias), the work inside a tile is not.
Every term that is the same for all modes drops out: sum_Y slog(Y), the tile's pixel count and the accumulated total,
so a mode's relative cost is sum over the tile's NONZERO bins of slog(Y_i) - slog(X_i) - slog(X_i + Y_i) (at most 64
per channel for an 8x8 tile) + the bias terms; float32 with the stable difference X log2(X+Y) + Y log2(1 + X/Y). One
work-group walks the tiles (bounded launches, state in device memory): 14 threads walk the tile once per mode (the
fill-aware residuals, as PredictorModeKernel does today), 56 (mode, channel) threads count their 64 values and sum
the cost terms, one thread picks the first minimum and adds the chosen histogram to `accumulated`. Then the predictor
image at the measured bits (and the uniform-block coarsening). Check: size on Lore's frames and every WebP fixture vs
today; the existing decode / libwebp oracle tests keep the pixels exact. Next after it: cross-colour choice
(PredictionCostCrossColor, the same accumulated scheme) and the entropy-image clustering.

## WebP lossy ENCODE (next; Lore/SpawnGen priority 2)

State 2026-10-22: lossy DECODE is done (VP8 full decoder + libwebp-exact YUV->RGBA + ALPH, bit-exact with libwebp on 168
fixtures). The existing `Vp8KeyframeEncoderGpu` is a v1 (every macroblock DC_PRED, no loop filter, default probabilities,
one partition): valid VP8, far from libwebp in size at equal quality. Wrapping it in RIFF alone would ship a encoder that
loses to every generator, so the encoder is upgraded first. Parity target: libwebp `-m 4` (Pillow `method=4`, the default).

- [~] E1 RGBA -> YUV 4:2:0 kernel = libwebp's default import (picture_csp_enc.c: Y from linear coefficients, U/V from the
      gamma-corrected 2x2 average, `USE_GAMMA_COMPRESSION`) + `WebPCleanupTransparentArea`: `WebPRgbaToYuvGpu`, equal to
      libwebp's own planes on 74 fixtures on the CPU lane; GPU / browser lanes still to run.
- [x] E0 CPU reference: `Vp8LossyEncoderReference` (References) byte-identical with libwebp on 552 files, methods 0..4.
      The accelerator encoder (E2..E6) is checked against it stage by stage (same YUV in -> same bytes out).
  Accelerator design (from the port): (a) analysis is per-macroblock independent (libwebp imports the boundary from the SOURCE),
  so it is one thread per macroblock; (b) k-means + segment parameters are one thread; the quality -> quantizer power law
  (`pow` in doubles, which WGSL lacks and which must round like libwebp's) becomes a host-built table indexed by the segment
  alpha: the host computes it from the CONFIG only (quality, sns), never from pixels; (c) the RD loop is serial across
  macroblocks by construction (the token probabilities are refreshed every mb_count/8 macroblocks from the statistics of all
  earlier ones, and every decision uses them), so it runs as ONE work-group walking the macroblocks in raster order, chunked
  per macroblock row per launch, with the parallelism INSIDE a macroblock: the 4 I16, 10 I4 (per 4x4 block) and 4 UV candidate
  reconstructions are evaluated by separate threads and reduced to libwebp's choice (first index of the minimum score; the
  I4 early-out never changes that result; the I16 flatness flag is a prefix-AND over the modes); (d) token emission, header and
  RIFF assembly are single threads.
- [x] E2..E6 (opaque pictures, methods 3 / 4, one partition): `WebPLossyEncoderGpu` byte-identical with libwebp on CPU,
      CUDA, OpenCL (222 fixtures). Open: browser lanes; methods 0..2, 5..6 (trellis); 2/4/8 partitions;
      parallel candidate evaluation inside a macroblock (performance).
- [ ] **Lossy in the BROWSER (2026-11-19, top priority)**: SpawnGen's site Text-to-video (25 LTX frames 512x352, lossy q90
      method 4) spends 120.6 s of ~155 s in Save Animated WEBP in Chrome (WebGPU / D3D12) on an RX 590: ~4.8 s per frame
      (Lore, Codecs 1.1.1). Same GPU, same tests: WebGPU 6-7x CUDA on the lossy animation tests (39.4 vs 5.4 s), Wasm
      ~14x. MEASURED 2026-11-19 (Probe_DispatchCost_AndLossyFrameTime, 4070, warm, 512x352 q90): the RD loop is 5 x 704
      = 3,520 launches per frame (not 17,600 as first said); one empty dispatch costs CUDA 7.3 us, WebGPU 43.6 us, Wasm
      528 us; the frame takes CUDA 272 ms, WebGPU 759 ms, Wasm 9.6 s: dispatch overhead is 9-20% of a frame, NOT the main
      cost. ~80% is the per-macroblock work run one macroblock at a time (each stage one small work-group: the GPU is
      nearly idle). The wavefront below gains mainly by running a diagonal's macroblocks at once (occupancy).
      DESIGN of the macroblock wavefront, byte-identical (checked against the stage code 2026-11-19):
      - order inside a probability window (the refresh every max(mbCount >> 3, 96) macroblocks in raster order): diagonals
        d = x + 2y; (x, y) needs (x - 1, y) (left samples, left nz / derr) and (x + 1, y - 1) (top-right samples); yTop[x]
        is written by (x, y) after (x, y + 1)'s readers of row y ran (x + 2y < x + 2y + 2), so one yTop row stays correct;
      - per-row state for the rows in flight: left samples (sb BLeft), left nz (si ILeftNz), left derr (si ILeftDerr), the
        output-buffer swap (meta RowOutBuf / RowOut2Buf); per-macroblock scratch (sb BIn / BOut*, si IRdScalars, ...) for
        every macroblock of a diagonal;
      - tokens: RecordTokens appends to one stream in raster order and RecordStats updates saturating counters (halving at
        0xfffe0000: ORDER-DEPENDENT once a counter saturates inside a window) -> per-macroblock token slots (with the stat
        index), then one serial pass per window in raster order applies the stats and compacts the tokens; SizeP0 is a sum
        (order-free);
      - launches: each stage once per diagonal (a work-group per macroblock): ~mbW + 2 mbH diagonals per frame plus the
        window seams, ~400-500 dispatches per 512x352 frame instead of 3,520, each running up to ~20 macroblocks at once.
      Guard: every lossy byte-identical-with-libwebp test (they pin the bytes), the Pillow oracles, all six backends.
      DONE 2026-11-20 (see CHANGELOG): stages per diagonal, per-row scratch and token buffers, window pass (atomic counts;
      ordered replay only for a counter that can saturate inside the window). 222/222 libwebp cases; RTX 4070 512x352 q90
      272 -> 185 ms CUDA. Traced CUDA frame after it: RD stages 109 ms, window 30 (one group, before the atomic version),
      finish 43 (the token partition on ONE thread).
- [ ] NEXT (measured 2026-11-20, webp-anim-bench lossy q90 m4, Lore's 25 frames, OpenCL RTX 4070, warm): 414 ms per frame,
      302,636 B; the ALPH plane's EXACT filter trials (4 full lossless encodes, each with strategy trials + cache search,
      host readback per step) are ~2/3 of it, the lossy RD stages ~1/3. The fast estimate (--alpha-fast 1) gives 148 ms but
      307,014 B (+1.4%): not allowed (size is never traded for speed). Plan: the 4 trials in LOCKSTEP with the same choices
      and bytes - a child encoder per trial (own buffers; the plane is small), their awaits interleaved so the host round
      trips overlap (one sync covers 4 trials' queued kernels). Open point: on desktop the interleaved continuations may
      launch from different threads on one stream - run them under a single-threaded scheduler, or prove the stream is
      thread-safe. Guard: ALPH bytes identical with and without lockstep on every lane (a tests-only switch), Lore's
      302,636 B file unchanged.
- [x] DONE 2026-11-20 (CUDA 43 -> 1 ms per 512x352 frame; frame 272 -> 96 ms with the wavefront): the token partition in parallel, byte-identical (VP8 bool coder; the finish step's 43 ms on CUDA, far more on
      WebGPU where one thread is slow). The coded stream is the big number sum over tokens with bit 1 of
      (split + 1) << (shifts after the token), written in bytes; the 0xff run and the carry into the previous byte are that
      big number's carries. So: (1) the range sequence: chunks of K tokens, each chunk run from all 128 possible start
      ranges (range is 127..254 after normalisation) -> per chunk a map start -> (end range, shift total); compose the maps
      over the chunks in order (one thread, N / K steps) -> each chunk's true start range and bit offset; (2) each chunk
      rebuilds its slice of the big number from its true start (its own bits plus a carry out); (3) add the slices at their
      bit offsets with a carry-lookahead scan over bytes (generate / propagate); (4) the final flush as VP8BitWriterFinish.
      Guard: the token partition bytes of every lossy libwebp case, plus a stress stream of all-0xff runs and long carries.
      MODEL CHECKED 2026-11-20 (Python, 5,000 random streams incl. extreme-probability runs): after any token the writer
      state equals the big number exactly: int(written bytes ++ 0xff x run) << (16 + nb) + value == sum(add_i << (own shift
      + all later shifts)), with 8 + nb + 8 x (bytes + run) == the total shift count. So the parallel pass can produce the
      bytes plus the end state (value, nb, run, range), and VP8BitWriterFinish runs on it unchanged (one thread, a few bits).
      Rebuilding the end state: with t = total shifts - 8, bytes + run = ceil(t / 8) and nb = t - 8 x that (in (-8, 0]);
      the writer may hold a pending carry in value (220 of 5,862 model streams), so its bytes can differ from the big
      number's digits; a writer rebuilt as (digits up to the trailing 0xff run, run = that run, value = big mod
      2^(16 + nb), the range from the scan) finishes to the SAME bytes in all 5,862 (scratchpad boolstate.py). Chunk
      slices can be summed into the global number by 32-bit atomic adds with carry propagation (order-free); slices overlap
      deeply where a chunk shifts little (high-probability runs), so no fixed neighbour-overlap assumption.
- [ ] Speed (2026-11-14, Lore: SpawnGen would use lossy for quick previews if it were fast). Measured: 25 LTX frames 512x352
      q80 on the RTX 4070, 2.15 s per frame (libwebp 40 ms, same bytes). The RD loop is 5 one-thread launches per macroblock
      (3,520 per frame at 512x352). Two independent steps, both byte-identical by construction:
      (1) candidates in parallel inside a macroblock, as (c) above; (2) a wavefront INSIDE each probability window: the level
      costs change only at the refresh every max(mb_count >> 3, 96) macroblocks (raster order), so between two refreshes a
      macroblock depends on earlier ones only through its left / top / top-right reconstruction, the nz and DC-error
      contexts: (x, y) runs after (x - 1, y) and (x + 1, y - 1), the window's first macroblock after the previous window.
      Tokens go to per-macroblock slots (compacted in raster order before the finish), statistics by atomics (sums).
      Steps per window ~ mb_w + 2 * rows: 1080p 8160 macroblocks -> ~1100 steps. To check against Vp8LossyEncoderReference statement by
      statement before building: no other cross-macroblock state (the I4 header-bit limit is per macroblock; the skip
      probability is computed after the loop).
- [ ] Speed, measured 2026-11-16 after the parallel mode decisions (Intra4 b15ccc0, Intra16 b2fd417, UV a112805; same bytes):
      Lore's LTX frames (512x352 q80, CUDA, traced, ms per frame): RD prepare 174 (the probability refresh + import + the
      I16/UV predictions, one thread), I16 89, I4 83, UV 41, record 24; analysis + setup ~220 (includes the first frame's
      compile); the ALPH plane ~516 (the frames have transparency): `EncodeAlphaAsync` with `Filter = Auto` at quality >= 50
      (the lossy default: quality 75) encodes the plane with EACH of the 4 filters to measure them, then once more - five
      full lossless encodes. libwebp's default (`alpha_filtering = 1`, fast) picks the filter from an estimate and encodes
      once. Next: measure the ALPH size of the estimate vs the exact trial on Lore's frames and the fixtures; if the exact
      trial is smaller, keep it but make it cheaper (the 4 measuring passes share the analysis; run them as one pass with
      4 outputs); never ship a bigger default without that measurement (speed is never bought with size).
      2026-11-16/18 state: the exact trial is kept (the estimate is +2% on Lore's animation, up to +41% on small planes;
      `ExactFilterChoice = false` opts out). 3b837d3: no fifth encode (the winner's stream is kept) and the colour analysis
      in parallel: 16 frames 912 -> 759 ms per frame. Traced totals left over 16 frames (CUDA): trial predictor 1553, trial
      raw 1138, cache 712, emit header 573 ms. NEXT (byte-identical by construction): the 4 filter trials in LOCKSTEP -
      EncodeStreamAsync's host decisions (strategy by estimated bits, colour-cache size, meta groups) are per trial and
      independent, so run the four trials' stages side by side (four planes in flight on separate work / token buffers,
      one SynchronizeAsync + one meta readback per stage for all four instead of one per trial and stage). A 512x352 plane
      fills a small part of the GPU per launch; four at once fill more, and the host round trips drop 4x. Measure the
      split between GPU time and sync / launch overhead per stage first (trace with and without the waits), then build.
      MEASURED 2026-11-19 (Nsight Systems, lossy, 8 frames): GPU-bound, not sync-bound - 4.5 s of kernel time, the LZ77
      copy search (SearchKernel 1310 + LinksKernel 443 ms) the largest part, 32 tokenisations per alpha frame; 45,556
      launches cost 233 ms of API time. So lockstep is NOT the lever; d79a21b reuses the copies across the cache trials
      instead (lossless anim -16%, lossy -4%). Left: a strategy chosen before the last trial is re-applied and searched
      again (keep its copies, or its tokens, instead); the search itself (chain depth 32 on every position).
      DONE 588aabe (best strategy's copies saved / restored). PROFILE 2026-11-19 after 1.1.1 (Nsight, lossless WebPAnimEncoder
      animation, 25 LTX frames, RTX 4070): 8.2 s of kernel time, of which 2.2 s is the bench's PNG decode (InflateStepKernel).
      Encode: ParseKernel 1228 ms (one thread: the greedy parse), TransparentFillKernel 1196 (one thread per chunk: transparent
      pixels take their predictions in raster order), LinksKernel 1048 (one thread per 16384-position block: ~11 threads per
      frame), ClusterAssign 669, chooser WalkKernel 581, Emit 263, BuildCodes 189; SearchKernel no longer in the top. NEXT,
      byte-identical by construction: (1) parse in parallel segments: each segment parses speculatively from its own start;
      a segment's true entry is where the previous segment's parse crosses into it, and the speculative parse is right
      from the first start both parses share (greedy parses resynchronise within a few tokens), so only the short prefix
      before that meeting point is re-parsed; (2) the fill as a wavefront (x + 2y, as the chooser's fill) or over runs of
      transparent pixels only; (3) the links as a sort by (key, position) instead of a serial walk per block.
- [x] ALPH (transparency, 2026-10-25): VP8X + ALPH + VP8, VP8 chunk byte-identical with libwebp, alpha exact (145 fixtures, CPU).
- (superseded by the summary above; kept for the design notes) E2 intra mode decision: I16 (DC/V/H/TM), I4 (10 modes) and UV (4 modes) chosen by distortion + lambda * rate like
      libwebp method 4 (`VP8Decimate`: score = SSE * RD_DISTO_MULT + lambda * bits, with the spectral-distortion term
      `tlambda` for I4/I16); per-macroblock recon stays on the accelerator (wavefront, like the decoder).
- (superseded) E3 segmentation (4 segments from the per-macroblock alpha = DCT-histogram "complexity", k-means like
      `VP8SegmentMap`), per-segment quantizer and filter strength; the quality -> quantizer mapping of `VP8SetSegmentParams`.
- (superseded) E4 token statistics pass + coefficient probability updates (`VP8WriteProbas` decisions by savings), skip probability,
      `mb_no_skip_coeff`; partitions 1/2/4/8.
- (superseded) E5 loop filter strength (`VP8AdjustFilterStrength` default: from quantizer and sharpness 0), simple/normal filter.
- (superseded) E6 container on the accelerator: RIFF + `VP8 ` (simple) or VP8X + `ALPH` (alpha via `WebPLosslessEncoderGpu.EncodeAlphaAsync`)
      + ICCP/EXIF/XMP; animation of lossy frames.
- [ ] Gate (the JPEG encoder's bar; methods 3 / 4 opaque + alpha met at up to 640x480 - 1920x1080 / 4000x3000 and methods 0..2, 5, 6 open): libwebp's lossy encoder is deterministic, so for Pillow's defaults (method 4 =
      RD_OPT_BASIC, no trellis, sns 50, filter strength 60, sharpness 0, 4 segments, 1 partition, pass 1) our file is
      BYTE-IDENTICAL to libwebp's for the same pixels and quality, on a corpus of photos / graphics / text / alpha at
      quality 0..100, including 1920x1080 and 4000x3000. Intermediate checks while porting: our CPU reference decoder parses
      both files, so segment map, quantizers, filter level, per-macroblock modes and probabilities can be diffed field by
      field before the bytes match. Other methods (0..3, 5, 6) after method 4 is byte-identical.

## Animation API (APNG / GIF / WebP share it)

`ImageFrameInfo` (Images/ImageTypes.cs) describes a frame's rectangle, duration, disposal and blend. Every animated format
adds, next to `ReadInfoAsync` / `DecodeAsync` (which return the first / default frame as before):

```csharp
Task<ImageFrameInfo[]> ReadFramesAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength);      // small host metadata
Task<ImageInfo> DecodeFrameAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength, int frameIndex,
                                 ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8);
// + managed (ReadOnlyMemory<byte>) and [SupportedOSPlatform("browser")] TypedArray overloads of both
```

`DecodeFrameAsync` returns the FULL CANVAS (`ImageInfo.Width x Height`) as it looks while frame `frameIndex` is displayed:
frames 0..frameIndex are decoded and composited on the accelerator honouring each frame's `Blend` and the previous frames'
`Dispose` (compositing is a kernel; the canvas never visits the host). A decoder may keep the last canvas to make
sequential access O(1) per frame; random access may re-composite from frame 0.

### GIF (first animated format; APNG / WebP follow the same shape)

* `GifDecoderGpu.ReadFramesAsync` / `DecodeFrameAsync` exactly as above; the loop count has no shared type yet, so
  `ReadLoopCountAsync` (three overloads) returns the NETSCAPE value (0 = forever, -1 = no extension before the first image).
  When APNG / WebP land, `num_plays` / loop count should join `ImageInfo` and `ReadLoopCountAsync` can forward to it.
* `GifDecoderGpu.Compositing` (`GifCompositing.Pillow` default, `.Web`): the decoder is bit-exact with Pillow 13
  (`Image.seek(k)` + `convert("RGBA")`) in `Pillow` mode, including Pillow's own quirks (sticky disposal, palette colour kept
  under alpha 0, background-colour disposal fill, identity-ramp palettes read as gray); `Web` is what browsers show.
  Frames that extend past the logical screen are clipped in both (Pillow grows its canvas, so no Pillow truth exists for them).
* Compositing is planned on the host as a few ints per frame and executed by one kernel (`OpFillRect / OpFillCanvas / OpSave /
  OpRestore / OpDraw`); LZW is one thread per frame, frames in parallel (batches of 64, 32 MB of palette indices).
* Encoder: `GifEncoderGpu` (`EncodeAsync` for a picture, `EncodeAnimationAsync` for frames stored back to back, three overloads each). Exact palette via
  `ExactPaletteBuilder`, else median cut + nearest colour on the accelerator, `GifDither.Ordered / FloydSteinberg` optional; LZW in 32768 pixel segments
  (one thread each, a clear code between them, bits concatenated on the accelerator). The algorithms are documented in `GifEncoderKernels.cs` and
  re-implemented independently in `scripts/images/make_gif_encoder_fixtures.py`, whose files the tests compare byte for byte.
