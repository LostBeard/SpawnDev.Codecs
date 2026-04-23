# Phase 1 - Opus Encoder + Decoder

**Status:** Design LOCKED (2026-04-23). All 6 open questions resolved with Captain. Ready for Phase 1a implementation start.
**Depends on:** `PLAN-SpawnDev-Codecs-Roadmap.md`

This is the **first codec we build** and becomes the template for everything after (VP8 decoder, VP8 encoder, VP9, AV1, FLAC, Vorbis). Every architectural decision here echoes through the rest of the library, so we get it right once.

---

## Why Opus first

- **Smallest spec** of all the codecs in scope (RFC 6716, ~300 pages including appendices)
- **Royalty-free** - no patent exposure
- **WebRTC mandatory-to-implement** - instant utility the moment it ships (replaces SipSorcery's Opus path in SpawnDev.RTC)
- **Covers both transform and predictive coding** - CELT (MDCT-based, GPU-friendly) + SILK (LPC-based, inherently CPU). Any later video codec we build uses one or both of these paradigms. Doing Opus first forces us to solve entropy coding, buffer management, frame-by-frame streaming, and GPU-CPU hand-off cleanly - the scaffolding all subsequent codecs will reuse.
- **Reference test vectors exist** (RFC 6716 conformance vectors) - bit-exact verification is possible from day one

---

## Design principles (derived from Captain's direction 2026-04-23)

### 1. Accelerator buffers are the universal I/O currency for raw audio/video data

- Every codec in SpawnDev.Codecs takes and returns `ArrayView<T>` from ILGPU for raw samples / frames
- On WebGPU, `ArrayView<T>` is backed by a native `GPUBuffer` - directly renderable via WebGPU command encoder, pipe-able into `texImage2D(canvas)` for zero-copy display (per Data's VR.1c hybrid bridge pattern)
- On WebGL, backed by a native `WebGLBuffer` / `WebGLTexture`
- On Wasm, backed by `SharedArrayBuffer`
- On CUDA / OpenCL / CPU, backed by the respective native buffer
- **ILGPU exposes the native JS type when available** - we inherit that for free. Consumers needing raw JS access use the existing ILGPU buffer-casting path.

### 2. Compressed data is CPU-side only (hard boundary)

- Bitstream bytes come from network / disk / memory - that data is CPU by necessity
- Compressed data shape: `ReadOnlyMemory<byte>` (input) / `Memory<byte>` or `byte[]` (output)
- No attempt to stage compressed bitstream on GPU - entropy coding is sequential CPU work anyway

### 3. Codec does NOT own the Accelerator

- Accelerator is passed in, never allocated or disposed by the codec
- Matches Captain's rule ("Never Dispose the Accelerator from Library Code" - from the old ML project CLAUDE.md, still correct for us)
- Same Accelerator can back multiple concurrent codec instances (e.g. 100+ Opus decoders on server-side for conference mixing)

### 4. Zero-copy pipelines are the default, not the exception

- Video decode: `CompressedPacket → CPU entropy decode → GPU: dequant / IDCT / motion comp / loop filter → GPU frame buffer → WebGPU canvas` (no intermediate CPU readback)
- Audio decode (single stream): CPU is fine, but the API still speaks `ArrayView` so **server-side multi-stream mixing** works GPU-native without rewriting the API
- Audio encode (single stream from microphone): microphone produces CPU samples, so input is CPU; `ArrayView` wraps it for uniform API

### 5. No CPU fallback branches

- ILGPU's CPU backend handles non-GPU environments - one implementation runs everywhere
- If GPU is overkill for a case (single Opus stream on desktop), user passes a CPU-backed Accelerator; same code path, no `if (hasGpu) else` branches

### 6. Patent cleanliness is non-negotiable

- VP8 / VP9 / AV1 / Opus / FLAC / Vorbis only
- H.264 / H.265 / AAC stay in SpawnDev.MultiMedia via platform P/Invoke

---

## Proposed public API - Phase 1 (Opus)

### Audio encoder

```csharp
namespace SpawnDev.Codecs.Audio;

/// <summary>
/// Encodes raw PCM audio to a compressed format (Opus in Phase 1).
/// Disposable. Not thread-safe - create one per stream.
/// </summary>
public interface IAudioEncoder : IAsyncDisposable
{
    Accelerator Accelerator { get; }
    AudioCodec Codec { get; }                // Opus, FLAC, Vorbis (future)
    int SampleRateHz { get; }                // 8000, 12000, 16000, 24000, 48000 for Opus
    int ChannelCount { get; }                // 1=mono, 2=stereo

    /// <summary>
    /// Encode one frame of PCM samples (float: -1.0 to +1.0 range).
    /// Frame size inferred from <c>pcmFrame.Length / ChannelCount</c>. Opus-legal sizes: 120, 240, 480, 960, 1920, 2880 samples @ 48kHz.
    /// Returns number of compressed bytes written to <paramref name="outputBuffer"/>.
    /// </summary>
    ValueTask<int> EncodeFrameAsync(
        ArrayView<float> pcmFrame,
        Memory<byte> outputBuffer,
        CancellationToken ct = default);

    /// <summary>
    /// Convenience overload for 16-bit signed PCM (WebRTC RTP native, capture device native).
    /// Equivalent to widening to float internally. Same frame-size rules.
    /// </summary>
    ValueTask<int> EncodeFrameAsync(
        ArrayView<short> pcmFrame,
        Memory<byte> outputBuffer,
        CancellationToken ct = default);
}
```

### Audio decoder

```csharp
namespace SpawnDev.Codecs.Audio;

public interface IAudioDecoder : IAsyncDisposable
{
    Accelerator Accelerator { get; }
    AudioCodec Codec { get; }
    int SampleRateHz { get; }
    int ChannelCount { get; }

    /// <summary>
    /// Decode one compressed packet into PCM samples (float: -1.0 to +1.0 range).
    /// Caller sizes <paramref name="pcmOutput"/> to <c>max_frame_samples * ChannelCount</c>
    /// (2880 * ChannelCount at 48kHz covers the largest legal Opus packet).
    /// Returns actual number of sample frames written (not bytes, not interleaved count).
    /// </summary>
    ValueTask<int> DecodeFrameAsync(
        ReadOnlyMemory<byte> compressedPacket,
        ArrayView<float> pcmOutput,
        CancellationToken ct = default);

    /// <summary>Convenience overload for 16-bit signed PCM output.</summary>
    ValueTask<int> DecodeFrameAsync(
        ReadOnlyMemory<byte> compressedPacket,
        ArrayView<short> pcmOutput,
        CancellationToken ct = default);
}
```

### Factory / configuration

```csharp
namespace SpawnDev.Codecs.Audio;

public static class OpusCodec
{
    /// <summary>Create an Opus encoder. Frame duration is chosen per-call via buffer size.</summary>
    public static ValueTask<IAudioEncoder> CreateEncoderAsync(
        Accelerator accelerator,
        OpusEncoderConfig config,
        CancellationToken ct = default);

    public static ValueTask<IAudioDecoder> CreateDecoderAsync(
        Accelerator accelerator,
        OpusDecoderConfig config,
        CancellationToken ct = default);
}

public sealed record OpusEncoderConfig(
    int SampleRateHz,
    int ChannelCount,
    OpusApplication Application,   // Voip | Audio | LowDelay
    int BitrateBitsPerSecond);
    // NOTE: no FrameDuration - frame size is inferred per-call from ArrayView.Length

public enum OpusApplication { Voip, Audio, LowDelay }
```

### Usage example (SpawnDev.RTC integration, float path - browser-native)

```csharp
// RTC-side: create Opus encoder using the RTC's Accelerator (shared across decoders/encoders)
var encoder = await OpusCodec.CreateEncoderAsync(accelerator, new OpusEncoderConfig(
    SampleRateHz: 48000,
    ChannelCount: 1,
    Application: OpusApplication.Voip,
    BitrateBitsPerSecond: 32000));

// Per-call frame size - consumer picks 20ms (960 samples) for VoIP
// Float32Array from WebAudio getChannelData maps directly; no conversion needed
using var pcmBuffer = accelerator.Allocate1D<float>(960);
pcmBuffer.View.CopyFromCPU(webAudioFloat32Array);

var outBuf = new byte[1275];  // Opus max packet size
int encodedBytes = await encoder.EncodeFrameAsync(pcmBuffer.View, outBuf);

// Ship outBuf[..encodedBytes] across the WebRTC data channel / RTP / whatever
```

### Usage example (short path - WebRTC RTP direct)

```csharp
// Same encoder, different input format - short overload dispatches
using var pcmBuffer = accelerator.Allocate1D<short>(960);
pcmBuffer.View.CopyFromCPU(rtpSamples);
int encodedBytes = await encoder.EncodeFrameAsync(pcmBuffer.View, outBuf);
```

---

## Opus-specific architecture

### Opus mode split

Opus runs in three modes selected per frame by the encoder:

| Mode | Used for | Algorithm |
|------|----------|-----------|
| **SILK-only** | Speech at ≤16 kHz, low bitrate | LPC predictive coding (inherited from Skype's SILK) |
| **CELT-only** | Music, low-latency, high bitrate | MDCT transform coding |
| **Hybrid** | Speech + some music, 16-48 kHz | SILK up to 8 kHz + CELT above |

### GPU / CPU split per module

| Module | GPU viable? | Why |
|--------|-------------|-----|
| **MDCT / IMDCT** (CELT) | ✅ YES | Fourier-family transform, per-block parallel |
| **MDCT window + overlap** (CELT) | ✅ YES | Trivially parallel |
| **Pitch prediction** (CELT) | ⚠️  Marginal | Correlation search, parallel but small |
| **Quantization** (CELT) | ✅ YES | Per-coefficient parallel |
| **Range coder** (CELT + SILK entropy) | ❌ NO | Sequential by nature |
| **LPC analysis** (SILK) | ❌ NO | Sequential recursion |
| **LSF quantization** (SILK) | ❌ NO | Sequential codebook search |
| **Voice activity detection** | ⚠️  Marginal | Small FFT + heuristics, overhead > benefit on single stream |

**Realistic per-Opus-frame pipeline (encoder):**

```
PCM samples (CPU or GPU)
    ↓
Upload to GPU if needed
    ↓
GPU: MDCT + windowing (CELT layer only)           ← ILGPU kernel
    ↓
CPU: SILK LPC analysis                             ← Sequential C#
    ↓
CPU: mode decision (SILK / CELT / Hybrid)
    ↓
GPU: CELT quantization (if CELT/Hybrid)            ← ILGPU kernel
    ↓
CPU: readback quantized coefficients
    ↓
CPU: range coder entropy encode                    ← Sequential C#
    ↓
CPU: bitstream packing → output bytes
```

**Single-stream value proposition:** modest - MDCT + windowing are a small fraction of Opus encoder cost. The big win comes from **batching multiple streams**.

**Multi-stream / server-side (conference bridge) value proposition:** substantial - run 100+ Opus encoders in parallel on one GPU dispatch, each stream's MDCT handled by a workgroup. This is why the API speaks `ArrayView` even when single-stream use could trivially use `Span<short>`.

### Phase 1 implementation strategy

**ILGPU kernels from day 1 for parallelizable parts.** No "CPU-first, port later" intermediate step - that would violate `feedback_ilgpu_all_compute.md`, `feedback_no_cpu_fallback.md`, and `feedback_always_use_gpu_sort.md`. Sequential-by-physics work (range coder, LPC) stays CPU because forcing it into a kernel adds overhead with zero parallelism benefit. Everything else is a kernel.

Three phases:

1. **Phase 1a: Full Opus decoder.**
   - `EntropyCoders/ArithmeticCoderBase.cs` - shared base class for Opus/VP8/AV1 entropy coders (pure C#)
   - `EntropyCoders/OpusRangeCoder.cs` - RFC 6716 range coder (pure C#, sequential by design)
   - `Audio/Opus/SilkDecoder.cs` - SILK LPC synthesis (pure C#, sequential recursion)
   - `Audio/Opus/Kernels/CeltDequantKernel.cs` - ILGPU kernel (parallel per coefficient)
   - `Audio/Opus/Kernels/CeltImdctKernel.cs` - ILGPU kernel (inverse MDCT, per-block parallel)
   - `Audio/Opus/Kernels/CeltWindowingKernel.cs` - ILGPU kernel (window + overlap-add, trivially parallel)
   - `Audio/Opus/Kernels/CeltPostFilterKernel.cs` - ILGPU kernel (pitch post-filter)
   - `Audio/Opus/Kernels/PcmConvertKernel.cs` - ILGPU kernel (float ↔ short for the Q1 overloads)
   - `Audio/Opus/OpusDecoder.cs` - orchestrator (C#, stitches sequential + kernels)
   - **Acceptance:** RFC 6716 Appendix A conformance vectors decode bit-exact on ALL 6 backends (CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm) via ILGPU.
2. **Phase 1b: Full Opus encoder.**
   - `EntropyCoders/OpusRangeEncoder.cs` - encoder side of the range coder (pure C#)
   - `Audio/Opus/SilkEncoder.cs` - SILK LPC analysis + quantization (pure C#)
   - `Audio/Opus/Kernels/CeltMdctKernel.cs` - ILGPU kernel (forward MDCT)
   - `Audio/Opus/Kernels/CeltQuantKernel.cs` - ILGPU kernel (quantization)
   - `Audio/Opus/OpusEncoder.cs` - orchestrator + mode decision logic
   - **Acceptance:** `decode(encode(pcm))` round-trips with PESQ ≥ 4.0 at 32 kbps; output bitstream round-trips through `libopus` reference decoder unchanged.
3. **Phase 1c: Multi-stream batch dispatch.** Optimize the GPU-side kernels to batch N streams in a single dispatch. Unlocks server-side conference mixing at 100+ concurrent Opus streams on one GPU. Benchmark-driven (test suite measures throughput vs single-stream baseline).

**Phase 1a alone ships a complete, GPU-accelerated Opus decoder on all 6 backends.** Phase 1b adds encoder. Phase 1c is the multi-stream perf unlock.

---

## SipSorcery integration strategy

SipSorcery today uses `Concentus` (a C# port of the reference Opus C code) via its internal `AudioEncoder`. The SpawnDev fork can swap to SpawnDev.Codecs.Opus the moment Phase 1a ships.

**Migration path:**

1. **Parallel availability (post-Phase 1a):** Both `Concentus`-backed path and SpawnDev.Codecs path available in the fork. `WaitForIceGatheringToComplete`-style config flag selects.
2. **Default flip (post-Phase 1b):** After passing the full RTC regression suite with SpawnDev.Codecs encoder+decoder, flip the default.
3. **Concentus removal (future):** Once SpawnDev.Codecs Opus is the only path in use for 30+ days on nuget.org stable, remove Concentus dependency from the fork.

---

## Test harness strategy

### Reference vectors

RFC 6716 Appendix A ships conformance test vectors. Every implementation of Opus decode must produce bit-exact PCM output from these vectors.

**Phase 1a acceptance criterion:** all RFC 6716 Appendix A decode vectors produce bit-exact PCM output on ALL 6 ILGPU backends (CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm).

### Encoder quality (not bit-exact)

Opus encoder output is NOT bit-exact - different valid encoders produce different valid bitstreams for the same PCM input. Encoder test criterion is **round-trip**: decode(encode(pcm)) must produce PCM that passes PEAQ / PESQ quality thresholds vs original.

**Phase 1b acceptance criterion:** PESQ ≥ 4.0 (toll-quality) on standard test signals at 32 kbps; decode(encode()) bitstream is RFC-compliant (round-trips through `libopus` reference decoder).

### Test infrastructure

- **`SpawnDev.Codecs.Tests`** (xunit, desktop): unit tests, range coder, LPC analysis, individual modules
- **`SpawnDev.Codecs.Demo`** (Blazor WASM via PlaywrightMultiTest): browser-side end-to-end tests, real WebAudio microphone + Speaker roundtrip, WebRTC loopback
- **Reference vectors ship as embedded resources** in the Tests project

---

## Design decisions (resolved with Captain 2026-04-23)

### Q1. PCM format - RESOLVED: **both `ArrayView<float>` and `ArrayView<short>` overloads**

Float is the browser-native PCM format ([MDN AudioBuffer.getChannelData returns Float32Array](https://developer.mozilla.org/en-US/docs/Web/API/AudioBuffer/getChannelData)), so Blazor WASM consumers pass float buffers natively. Short is native to WebRTC RTP and most microphone capture paths. Providing both overloads avoids consumer-side conversion in either direction.

Implementation: encoder/decoder logic operates internally on `float` (matches Opus reference); `short` overloads delegate with a lightweight widen / narrow pass (GPU kernel if buffer is GPU-resident, trivial SIMD otherwise).

### Q2. Async vs sync - RESOLVED: **async only (`ValueTask<int>`)**

Browser backends (WebGPU via `mapAsync`, etc.) **require** async for any GPU-touching path. Sync APIs would fail the moment we move work to a GPU kernel. Matches the SpawnDev `CopyToHostAsync` convention throughout.

### Q3. Frame size - RESOLVED: **variable per-call, inferred from buffer length**

Captain's principle: *"this is for other devs to use, so flexibility is good."* (Saved as `feedback_library_consumer_flexibility.md`.) Library consumers have diverse use cases - locking frame size at config time would be overconstrained for a library even though it'd be fine for an application.

Implementation: the encoder validates `pcmFrame.Length / ChannelCount` against the set of Opus-legal sample counts (120 / 240 / 480 / 960 / 1920 / 2880 samples at 48kHz, corresponding to 2.5 / 5 / 10 / 20 / 40 / 60 ms). Invalid sizes throw `ArgumentException` with a clear message listing the legal sizes. `OpusEncoderConfig.FrameDuration` removed from the config record.

### Q4. Range coder location - RESOLVED: **shared `EntropyCoders/` folder**

VP8's boolean arithmetic coder, Opus's range coder, and AV1's range coder share mathematical structure (all three are variants of arithmetic coding). Generic base class now saves reimplementation later.

Layout:
```
SpawnDev.Codecs/
├── EntropyCoders/
│   ├── ArithmeticCoderBase.cs       (shared base class)
│   ├── OpusRangeCoder.cs            (RFC 6716 range coder)
│   └── ... (VP8 BoolCoder, AV1 RangeCoder added in later phases)
└── Audio/
    └── Opus/
        ├── OpusEncoder.cs
        ├── OpusDecoder.cs
        └── ...
```

### Q5. License - RESOLVED: **MIT with NOTICE file**

Captain: *"as permissive as possible (allowed)."* MIT is the maximum practical permissiveness in .NET ecosystem - allows commercial use, closed-source derivatives, sublicensing, no copyleft.

Constraint: if we port structural code from libopus (BSD-ish Xiph.Org license), the BSD attribution requirement must be honored. A `NOTICE.md` at repo root credits:
- **libopus** (Xiph.Org Foundation, 2012-present) - for the reference Opus implementation we structurally consult
- **Concentus** (Logan Stromberg) - a BSD-licensed C# port of libopus, likely referenced during porting
- **RFC 6716** (Valin, Vos, Terriberry 2012) - the Opus specification
- **IETF** - for RFC stewardship

Downstream consumers of SpawnDev.Codecs get MIT terms + the NOTICE attribution; they do not need to sublicense Opus reference code themselves. This is the standard .NET codec-library posture (matches how managed ports of libvorbis, libFLAC, etc. are typically licensed).

### Q6. Repo topology - RESOLVED: **own GitHub repo `LostBeard/SpawnDev.Codecs`**

Matches every other SpawnDev library. Independent release cadence. Clean contribution story. Create at the start of Phase 1 implementation so the scaffolding commit is the first commit on master.

---

## Deferred from Phase 1 (Phase 2+)

- VP8 decoder (next after Opus)
- VP9, AV1, FLAC, Vorbis
- Multi-stream batch GPU dispatch API (Phase 1c)
- Video-side `IVideoEncoder` / `IVideoDecoder` interfaces (designed in Phase 2 when VP8 work starts)
- `CanvasRendererFactory` integration for zero-copy video-to-canvas (Phase 2)

---

## Next actions (design locked, ready to execute)

1. **Create `LostBeard/SpawnDev.Codecs` GitHub repo** - Captain's action when ready; scaffolding + Plans committed as first commit on master. (Blocks: nothing; just a bureaucratic step.)
2. **Research pass on the 4 roadmap unknowns** - Tuvok spawns subagent to close:
   - Existing pure-.NET codec implementations on nuget.org (Opus, VP8, VP9, AV1, FLAC, Vorbis) - might bootstrap from existing work
   - libopus / libvpx / libaom actual LoC - realistic effort sizing
   - RFC 6716 + RFC 6386 conformance test vector availability
   - Prior art: any existing ILGPU / GPU-accelerated codec implementations
   Runs in parallel; doesn't block Phase 1a.
3. **Phase 1a editor assignment - RESOLVED (2026-04-23): Tuvok takes SpawnDev.Codecs as full editor project.** Geordi stays on SpawnDev.ILGPU.P2P until 100%. Tuvok owns plans, code, tests, releases for SpawnDev.Codecs. Still covers research/planning/synthesis for the rest of the crew when needed.
4. **Phase 1a kick-off** - after repo exists + research pass returns:
   - Implement `EntropyCoders/ArithmeticCoderBase.cs` + `EntropyCoders/OpusRangeCoder.cs`
   - Implement `Audio/Opus/SilkDecoder.cs` (LPC synthesis, pure C# sequential)
   - Implement `Audio/Opus/Kernels/Celt*.cs` (ILGPU kernels for dequant/IMDCT/windowing/post-filter)
   - Implement `Audio/Opus/OpusDecoder.cs` (orchestrator stitching sequential + kernel parts)
   - Wire up test harness: embed RFC 6716 conformance vectors, validate bit-exact PCM output on all 6 backends
   - Target: Phase 1a green means `SpawnDev.Codecs.OpusDecoder` produces bit-exact RFC 6716 output on CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm via ILGPU

🖖 Tuvok
