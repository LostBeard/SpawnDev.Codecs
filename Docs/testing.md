# Testing

## One suite, six backends

The tests live once, in `SpawnDev.Codecs.Demo.Shared/UnitTests` (`CodecsTestBase` and its partial files), and run as
one class per backend: `CpuCodecsTests`, `CudaCodecsTests`, `OpenCLCodecsTests` and `VelocityCodecsTests` (the SIMD CPU
backend; desktop, through `SpawnDev.Codecs.DemoConsole`) and `WebGPUCodecsTests`, `WebGLCodecsTests`, `WasmCodecsTests` (in a real Chromium,
through the Blazor demo's `/tests` page). A test that a backend cannot run (WebGL without atomics, a TypedArray
overload on the desktop, a CPU-only reference check) reports **Skipped** with the reason, never passes silently.

## PlaywrightMultiTest

`PlaywrightMultiTest/` is the runner. It publishes the demo apps, serves the Blazor build over HTTPS, drives Chromium
with Playwright and reports every test as an NUnit test:

```bash
dotnet test PlaywrightMultiTest --filter "FullyQualifiedName~WebGPUCodecsTests.WebPLossy"
dotnet test PlaywrightMultiTest --filter "FullyQualifiedName~CudaCodecsTests&FullyQualifiedName~Vorbis"
```

Environment switches:

| Variable | Effect |
|----------|--------|
| `PMT_WEBGPU_ADAPTER=swiftshader` | runs WebGPU on Chrome's CPU adapter: a kernel that never ends just burns CPU instead of losing the GPU (use it to chase hangs) |
| `PMT_CDP_PORT` | Chrome DevTools port of the test browser |
| `PMT_BROWSER_CHANNEL` | Playwright browser channel |
| `PMT_SELFTEST_EXIT_RUNTIME=<test name>` | exits the page's .NET runtime before that test: it must fail and the next test must pass after a reload (the runner's dead-runtime guard) |
| `PMT_INIT_SCRIPT` | a JS file added to every page |
| `CODECS_VELOCITY_DEVICE=Vector256` | runs `VelocityCodecsTests` on that Velocity device (default Vector128) |
| `CODECS_VELOCITY_HOST_COPY=0` | runs Velocity host kernels as the original methods instead of SpawnDev.ILGPU 5.4.32's optimized copies (also read by `vp9-key-bench`), for comparing the two |
| `VP9_STAGE_TIMES=1` | prints the VP9 decoder's time per stage |

The runner fails the running test and reloads the page when the page's .NET runtime dies (Mono assert, runtime exit, GC
allocation failure, abort), when a test never finishes, or after an `OutOfMemoryException`. Browser console lines
containing `CODEC_TRACE` (and WGSL / Wasm dumps) are copied to `PlaywrightMultiTest/WGSLDumps/browser_console.log`; a
test can print stage timings that way (see `WebPLosslessEncoderGpu.Trace`). A browser test may run as long as its own
limit plus 60 s (the result rows carry `data-test-timeout`); the VP9 every-frame decode has a 1 h limit.

## Desktop runs one test at a time

For CPU / CUDA / OpenCL work, `scripts/dev` runs the desktop suite test by test, each in its own process at below-normal
priority, so a long run does not disturb other work on the machine:

```powershell
pwsh scripts/dev/build-console.ps1
pwsh scripts/dev/run-desktop-tests.ps1 -Class CudaCodecsTests -Pattern '(?i)webp|png'
```

`-TimeoutSec` (default 600) limits each test; `-OutDir <dir>` keeps each test's full console output as
`<dir>\<Class.Method>.log` (diagnostics such as `VP9_STAGE_TIMES=1` end up there). `scripts/dev/gpu_watchdog.ps1` kills the GPU process of the
Playwright-launched Chrome when its working set passes `-LimitGB` (default 3), so a giant shader compile cannot starve
the PC, and logs the peak it saw.

## Fixtures and oracles

The truth never comes from this library. Fixture generators in `scripts/` produce the input and the expected output with
the reference implementation and write them into `SpawnDev.Codecs.Demo.Shared/TestData/` (embedded in the test
assembly):

| Script | Oracle | Fixtures |
|--------|--------|----------|
| `scripts/images/make_png_fixtures.py` | hand-written PNG writer, truth = the source samples | `TestData/Png` |
| `scripts/images/make_jpeg_fixtures.py` | Pillow (libjpeg-turbo) | `TestData/Jpeg` |
| `scripts/images/make_webp_fixtures.py`, `make_webp_lossy_decode_fixtures.py` | libwebp (Pillow) | `TestData/WebP`, `TestData/WebPLossy` |
| `scripts/images/make_webp_lossy_encode_fixtures.py`, `make_webp_lossy_alpha_fixtures.py` | libwebp's own imported planes and files | `TestData/WebPLossyEnc`, `TestData/WebPLossyAlpha` |
| `scripts/images/make_vp8_lossy_fixtures.py` | ffmpeg / libvpx | `TestData/Vp8Lossy` |
| `scripts/images/make_gif_fixtures.py`, `make_gif_encoder_fixtures.py`, `make_bmp_fixtures.py` | Pillow | `TestData/Gif`, `GifEnc`, `Bmp` |
| `scripts/audio/make_vorbis_fixtures.py` | libvorbis encode, libsndfile decode | `TestData/Vorbis` |
| `scripts/audio/make_opus_fixtures.py`, `make_opus_encoder_fixtures.py` | libopus 1.5.2 (fixed point, built from source) | `TestData/Opus`, `TestData/OpusEnc` |
| `scripts/audio/make_flac_decoder_fixtures.py`, `make_flac_encoder_fixtures.py`, `flac_decode.py` | FFmpeg and libFLAC | `TestData/FlacDec`, `TestData/FlacEnc` |
| `scripts/audio/make_mp3_decoder_fixtures.py`, `minimp3_ref.c` | minimp3 | `TestData/Mp3Dec` |
| `scripts/audio/make_ogg_opus_fixtures.py`, `make_ogg_opus_layout_fixtures.py` | the ffmpeg binary, as a black box | `TestData/OggOpus`, `TestData/OggOpusLayout` |
| `scripts/images/make_webp_anim_lossy_fixtures.py` | Pillow 13 / libwebp 1.6.0 | `TestData/WebPAnimLossy` |
| `scripts/video/make_webm_mux_fixtures.py`, `make_mp4_mux_fixtures.py` | `ffmpeg -c copy -fflags +bitexact` | `TestData/WebMMux`, `TestData/Mp4Mux` |
| `scripts/video/libvpx_ref`, `make_vp9_header_fixtures.py` | libvpx v1.17.0 built from source with stage hooks | VP9 decoder oracle |
| `scripts/video/make_loadvideo_rgb_fixtures.py` | ComfyUI's LoadVideo, run as a black box | `TestData/LoadVideo`, `TestData/Yuv420Rgb16` |
| `scripts/compression/*`, `scripts/texture/*` | c-blosc / numcodecs, liblz4, snap, Microsoft's GDeflate, texture2ddecoder, etcpak, bc7enc | compression and BCn fixtures |

Some tests call Pillow at run time on files this library wrote (does libwebp / libjpeg / Pillow decode **our** output to
the same pixels?). They need a Python with Pillow and numpy:

| Variable | Used by |
|----------|---------|
| `SPAWNDEV_IMG_PYTHON` | the Pillow oracle tests (`PillowOracle`) and the libFLAC check of the FLAC encoder; without it they report Skipped or Unsupported |
| `CODECS_PILLOW_PYTHON` | the WebP encoder tests' libwebp checks (Pillow opens our files) |

The oracle tests find their Python under either name, so setting one of them is enough.

## Red-checks

A test that cannot fail is not a test. Every guard is red-checked: the fix is disabled (or the output perturbed by one
byte, one sample, one bit) and the test must fail, by name, with a message that points at the cause. The commit message
records the red-check. `scripts/dev/redcheck.py <mutations.json>` automates it: for each mutation it edits a source file
(an exact text that must occur once, multi-line and CRLF files included), rebuilds, runs the test and expects it to fail,
restores the file byte for byte, and finally rebuilds and expects a pass; the exit code is 0 only when every mutation
failed the test.

## Benches (DemoConsole)

`SpawnDev.Codecs.DemoConsole.exe <command>` runs timing tools on a chosen backend (`--backend cpu|velocity|cuda|opencl`;
`vp9-lf-bench` also takes `velocity256`):

| Command | Measures |
|---------|----------|
| `vp9-key-bench` | `Vp9KeyframeEncoderGpu` (or `Vp9EncoderGpu` with `--inter N`) on a synthetic picture, per stage with `--stages 1` |
| `vp9-encode-bench <clip.webm>` | `VideoFileEncoderGpu` (VP9) per quantizer and key frame interval: bytes, PSNR, time |
| `vp9-psnr <source.webm> <encoded.webm...>` | bytes and PSNR of encoded WebM files against a source clip |
| `vp9-lf-bench <file.webm or .ivf>` | the VP9 decoder's loop filter (`--frames`, `--reps`, `--skip-pass`) |
| `av1-encode-bench` | `Av1KeyframeEncoderGpu` per keyframe (`--tile`, `--maxtiles`, `--batch`, `--fps`, `--level`) |
| `webp-anim-bench <frames folder>` | the animated WebP encoders (`--mode rect\|anim\|lossy`, `--stages 1`) |
| `weightpack-bench`, `gdeflate-bench` | `WeightPackGpu` and GDeflate speed |

## Run records

Every run (scoped or full, desktop or browser) is logged with its command, environment, machine, counts, failures and
slowest tests, and compared with the last comparable run before it is reported.
