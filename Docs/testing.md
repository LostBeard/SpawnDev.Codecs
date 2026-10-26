# Testing

## One suite, six backends

The tests live once, in `SpawnDev.Codecs.Demo.Shared/UnitTests` (`CodecsTestBase` and its partial files), and run as
one class per backend: `CpuCodecsTests`, `CudaCodecsTests`, `OpenCLCodecsTests` (desktop, through
`SpawnDev.Codecs.DemoConsole`) and `WebGPUCodecsTests`, `WebGLCodecsTests`, `WasmCodecsTests` (in a real Chromium,
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

The runner fails the running test and reloads the page when the page's .NET runtime dies (Mono assert, runtime exit, GC
allocation failure, abort), when a test never finishes, or after an `OutOfMemoryException`. Browser console lines
containing `CODEC_TRACE` (and WGSL / Wasm dumps) are copied to `PlaywrightMultiTest/WGSLDumps/browser_console.log`; a
test can print stage timings that way (see `WebPLosslessEncoderGpu.Trace`).

## Desktop runs one test at a time

For CPU / CUDA / OpenCL work, `scripts/dev` runs the desktop suite test by test, each in its own process at below-normal
priority, so a long run does not disturb other work on the machine:

```powershell
pwsh scripts/dev/build-console.ps1
pwsh scripts/dev/run-desktop-tests.ps1 -Class CudaCodecsTests -Pattern '(?i)webp|png'
```

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

Some tests call Pillow at run time on files this library wrote (does libwebp / libjpeg / Pillow decode **our** output to
the same pixels?). They need a Python with Pillow and numpy:

| Variable | Used by |
|----------|---------|
| `SPAWNDEV_IMG_PYTHON` | the Pillow oracle tests (`PillowOracle`); without it they report Skipped |
| `CODECS_PILLOW_PYTHON` | the WebP encoder tests' libwebp checks (Pillow opens our files); same Python works for both |

## Red-checks

A test that cannot fail is not a test. Every guard is red-checked: the fix is disabled (or the output perturbed by one
byte, one sample, one bit) and the test must fail, by name, with a message that points at the cause. The commit message
records the red-check.

## Run records

Every run (scoped or full, desktop or browser) is logged with its command, environment, machine, counts, failures and
slowest tests, and compared with the last comparable run before it is reported.
