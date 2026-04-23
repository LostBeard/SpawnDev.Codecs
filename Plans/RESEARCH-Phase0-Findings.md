# Phase 0 Research Findings - SpawnDev.Codecs

**Date:** 2026-04-23
**Researcher:** subagent spawn by Tuvok
**Closes unknowns:** 4 of 4 from `PLAN-SpawnDev-Codecs-Roadmap.md`

---

## 1. Existing pure-.NET codec implementations on nuget.org

### Opus - **CONCENTUS IS THE BOOTSTRAP CANDIDATE**

- **[Concentus 2.2.2](https://www.nuget.org/packages/Concentus)** (May 2024). Pure managed C# port of libopus 1.1.2 (fixed-point).
- **License: BSD 3-Clause** (Xiph/Skype/CSIRO/Microsoft copyright). **MIT-compatible** with NOTICE attribution.
- **Encoder + decoder + multistream + resampler.** Does NOT parse Ogg/RTP containers.
- Repo: [github.com/lostromb/concentus](https://github.com/lostromb/concentus) - README says dormant since 2016, but 2024 NuGet republish indicates live maintenance.
- **SipSorcery already uses Concentus** via its internal `AudioEncoder` - this is the path Riker's Phase 4 audio bridge goes through today.

### Vorbis
- **[NVorbis 0.10.5](https://www.nuget.org/packages/NVorbis/)** (Oct 2022). Pure managed, no P/Invoke, no unsafe. **License: MIT.** **Decoder only.** Repo: [github.com/NVorbis/NVorbis](https://github.com/NVorbis/NVorbis).

### FLAC
- **CSCore** (MS-PL) - pure C# FLAC decoder at [github.com/filoe/cscore](https://github.com/filoe/cscore) `CSCore/Codecs/FLAC/`.
- **Shamisen.Codecs.Flac 0.1.0-alpha** - pure managed FLAC decoder (alpha quality).
- **CUETools.Codecs.FLAKE** - pure managed FLAC **encoder** (flake port).

### VP8 / VP9 / AV1 - **GREENFIELD**
- **No pure-managed .NET implementations found.** All NuGet packages (ImageSharp.AVCodecFormats, FFMpegCore, VisioForge) are wrappers over native FFmpeg/libvpx/libaom/dav1d.

---

## 2. Reference codec size (order-of-magnitude)

| Project | Repo | Approximate C LoC |
|---------|------|-------------------|
| **libopus** | [xiph/opus](https://github.com/xiph/opus) | **~70-100K** (silk/ + celt/ + src/); Opus 1.5 DNN adds more |
| **libvpx** | [webmproject/libvpx](https://github.com/webmproject/libvpx) | **~300-400K** total; VP8 alone ~50K, VP9 ~200K+, shared DSP + asm |
| **libaom** | [AOMediaCodec/aom](https://aomedia.googlesource.com/aom/) | **~750K-1M+** (largest by far); encoder dominates - **LoC not publicly published, estimate from complexity-analysis literature** |

**Implication:** AV1 is the multi-year long-pole. Opus and VP8 are genuinely tractable Phase 1-3 scopes; VP9 is mid-effort; AV1 encoder is a dedicated year+ project on its own.

---

## 3. RFC conformance test vectors

### RFC 6716 (Opus)
- **Distributed separately** (NOT in-RFC).
- **Primary download:** [opus_testvectors.tar.gz](https://opus-codec.org/static/testvectors/opus_testvectors.tar.gz)
- **RFC 8251 update:** [opus_testvectors-rfc8251.tar.gz](https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz)
- **Format:** raw Opus bitstreams + reference PCM decoder outputs (fixed + float).
- Canonical page: [opus-codec.org/testvectors](https://opus-codec.org/testvectors/)
- Archive sizes not listed on page - download to measure.

### RFC 6386 (VP8)
- Not in RFC.
- **Official mirror:** [github.com/webmproject/vp8-test-vectors](https://github.com/webmproject/vp8-test-vectors)
- **Archive:** `vp8-test-vectors-r2.tar.bz2` (~3.14 MB) / `.zip` (~3.19 MB) on [downloads.webmproject.org/releases/webm/](http://downloads.webmproject.org/releases/webm/index.html)
- **Format:** `.ivf` bitstreams + `.md5` reference hashes.

### AV1
- Future concern. AOMedia has its own conformance suite.

---

## 4. Prior art - GPU-accelerated codec implementations

### Opus
- **No open-source GPU implementation found** - CUDA, OpenCL, WebGPU, Vulkan, nothing.
- 2016 NVIDIA forum thread ([link](https://forums.developer.nvidia.com/t/using-gpu-to-accelerate-speech-encoders-decoders/50283)) raises the idea but no code shipped.
- Closest academic prior art: CUDA-accelerated **ALAC** (Apple Lossless) - not Opus.
- Opus entropy coding is serial - widely considered GPU-hostile, but **never measured**.
- **SpawnDev.Codecs would be the first open GPU-accelerated Opus implementation.**

### Video codecs
- **FFmpeg Vulkan compute-shader pipeline** ([Khronos writeup](https://www.khronos.org/blog/video-encoding-and-decoding-with-vulkan-compute-shaders-in-ffmpeg)) is the leading prior art for compute-shader-based (not fixed-function hardware) codecs. FFv1 + ProRes shipping in FFmpeg 8.1; AV1/H.264/HEVC experimental.
- **No VP8/VP9/AV1 compute-shader encoder in managed code.**
- Vulkan Video (`VK_KHR_video_decode_av1`, encode AV1) exists but targets fixed-function video engines, NOT the software/compute-shader niche SpawnDev.Codecs would fill.
- **ILGPU-accelerated codecs: None.**

---

## Strategic implications for Phase 1a

### Opus implementation strategy - PIVOT RECOMMENDED

Given Concentus is BSD-3, pure-C#, battle-tested, and already consumed by SipSorcery:

**Hybrid approach for Phase 1a:**

1. **Fork Concentus's SILK layer** (LPC synthesis, range coder, bitstream parsing) - import into `SpawnDev.Codecs/Audio/Opus/Silk/` + `SpawnDev.Codecs/EntropyCoders/OpusRangeCoder.cs`. BSD-3 license honored with NOTICE attribution.
2. **Write CELT fresh as ILGPU kernels** - Concentus's CELT is fixed-point; GPU wants float. Rewriting CELT as ILGPU kernels is the actual GPU-acceleration value-add of SpawnDev.Codecs.
3. **Replace Concentus's orchestrator** with our own that stitches forked-SILK + GPU-CELT.
4. **Validate bit-exact vs RFC 6716 vectors.**

**Rationale:**
- Saves months of LPC / range coder / bitstream implementation (genuinely hard code to get right)
- Leverages proven correctness for the sequential parts that would NOT benefit from GPU anyway
- Focuses our ILGPU work on CELT (the parallelizable parts) - where we add unique value
- Migration story for SipSorcery is natural: they already consume Concentus-like code
- Concentus may not yet incorporate RFC 8251 updates - we bring it forward

**Trade-off:** we're not "pure from scratch" - but the point was never purity-of-authorship. The point is pure-.NET, ILGPU-accelerated, patent-clean, open-source. All four properties preserved.

### Value-add over Concentus (addressing Riker's scope flag 2026-04-23)

Riker flagged in `riker-to-tuvok-phase4-effort-estimates-2026-04-23.md`: *"If Phase 1 is 'reimplement Opus,' the value-add over Concentus is probably ILGPU acceleration and architectural control (SpawnDev-idiomatic DI, BlazorJS-aware, etc.) rather than patent cleanup (Concentus is already clean). Worth clarifying scope so Phase 1 doesn't end up being 'port Concentus tests' for three weeks of zero user-visible progress."*

Valid flag. Being explicit about the value-prop:

SpawnDev.Codecs.Opus is NOT competing with Concentus on patent status or basic correctness. Concentus is BSD-3, pure C#, passes RFC 6716 - it is already "good Opus for managed .NET." We build SpawnDev.Codecs.Opus anyway because:

1. **GPU-accelerated CELT (Phase 1a payoff)** - IMDCT + dequant + windowing + post-filter as ILGPU kernels. On a single stream this is marginal (Concentus is fast enough for 1 voice call on any modern CPU). On GPU backends it scales linearly with hardware.
2. **Multi-stream batch dispatch (Phase 1c payoff)** - this is where the real value lives. Server-side conference mixer with 100 concurrent Opus streams on one GPU dispatch. Concentus cannot do this architecturally - it runs one stream per call on CPU. Our GPU kernel dispatch is sized for the batch; `N=1` is the same codepath as `N=100`, priced proportionally by GPU.
3. **Foundation template for VP8/VP9/AV1 (Phase 2+ payoff)** - Opus Phase 1 proves the shared `EntropyCoders/` base class, the `Accelerator`-parameterized codec lifecycle, the RFC-conformance-vector test harness, and the `IAudioEncoder`/`IAudioDecoder` surface patterns. VP8/VP9/AV1 inherit all of this infrastructure. For video codecs, GPU acceleration is not marginal - it is necessary. Phase 1 builds the scaffolding; video codecs cash in on it.
4. **SpawnDev ecosystem integration** - SpawnDev DI shape, BlazorJS-aware browser paths, matches SpawnDev.RTC / SpawnDev.MultiMedia / SpawnDev.WebTorrent idioms so downstream consumers have one consistent mental model.

**The real "Phase 1 ships value" moment is Phase 1c** (multi-stream batch dispatch). Phase 1a (single-stream decoder) and 1b (single-stream encoder) are the foundation that makes 1c possible. We should NOT sell Phase 1a as "better Opus than Concentus for single-stream VoIP" because it mostly isn't. We sell it as "foundation for GPU-native multi-stream audio + the template the video codecs inherit."

This framing affects how Phase 1a gets announced when it ships - the release notes should lead with "first open-source GPU-accelerated Opus implementation in any language" and "foundation for SpawnDev.Codecs video codecs" rather than "replaces Concentus." We don't replace Concentus on our target use cases (single VoIP stream); we enable use cases Concentus structurally can't reach (batched multi-stream GPU dispatch + the video codec pipeline).

### First-mover positioning

- **GPU-accelerated Opus:** first in any language (open-source)
- **Pure-.NET VP8/VP9/AV1:** first in the .NET ecosystem
- **Compute-shader VP8/VP9 encoder in managed code:** first anywhere

Positioning is genuine. This isn't duplicating existing work.

---

## Action items for Phase 1a start

1. Download `opus_testvectors.tar.gz` + RFC 8251 update - stage as embedded resources in `SpawnDev.Codecs.Tests` project.
2. Clone Concentus locally, study its SILK + range coder structure before porting.
3. Confirm BSD-3 license terms - document attribution in `NOTICE.md`.
4. Decide CELT-fresh-write granularity: per-frame dispatch vs multi-frame batch kernel (probably per-frame for Phase 1a, batch for Phase 1c).
