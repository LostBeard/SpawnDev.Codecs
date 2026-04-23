# SpawnDev.Codecs - Roadmap

**Status:** Planning (Phase 0, 2026-04-23)
**Planning owners:** Tuvok (Research/Planning) + Captain
**Editor assignment:** TBD - this library is in planning phase only. No code written yet.

---

## Mission

Pure-.NET, ILGPU-accelerated, patent-clean audio and video codecs. The missing piece of the SpawnDev open-source media stack.

Runs on every ILGPU backend - CUDA, OpenCL, CPU, WebGPU, WebGL, Wasm - which means it runs in desktop AND Blazor WASM browser.

## Strategic positioning

Fills the last gap in the SpawnDev media ecosystem:

| Library | Role | Status |
|---------|------|--------|
| SpawnDev.RTC | Signaling + transport | Shipped, nuget.org (1.1.2) |
| SpawnDev.WebTorrent | Torrent infrastructure | Shipped, nuget.org (3.1.2) |
| SpawnDev.ILGPU | GPU compute | Shipped, nuget.org (4.9.2-rc.8) |
| SpawnDev.BlazorJS | Browser interop | Shipped, nuget.org (3.5.4) |
| SpawnDev.MultiMedia | Capture + platform encoders (P/Invoke to MF/VideoToolbox/VAAPI) | In progress (Riker) |
| **SpawnDev.Codecs** | **Pure-.NET open-source codecs** | **PLANNING (this doc)** |

## Codecs in scope (patent-clean only)

### Audio
- **Opus** (RFC 6716) - WebRTC mandatory-to-implement, royalty-free. Phase 1 target.
- **FLAC** - lossless, small spec, BSD reference (libFLAC).
- **Vorbis** - open alternative to AAC, BSD reference (libvorbis).

### Video
- **VP8** (RFC 6386) - WebRTC mandatory-to-implement, patent-clean via Google's WebM patent pledge, libvpx BSD reference.
- **VP9** - same patent status as VP8, significantly better compression.
- **AV1** - AOMedia patent-free, future-proof, libaom / SVT-AV1 references (BSD/MIT).

## Codecs explicitly NOT in scope (patent-encumbered)

These delegate to platform encoders via **SpawnDev.MultiMedia** (P/Invoke to MediaFoundation / VideoToolbox / VAAPI). System encoders are licensed by Microsoft / Apple / driver vendors respectively - we ride those licenses, we do not re-implement:

- **H.264** - MPEG-LA patent pool
- **H.265 / HEVC** - multiple overlapping patent pools
- **AAC** - patent-encumbered

**This line is critical.** Writing an H.264 encoder from scratch would open SpawnDev to patent exposure. Writing VP8 / VP9 / AV1 does not. SpawnDev.Codecs stays clean; SpawnDev.MultiMedia carries the patent-encumbered load via licensed system encoders.

## Architectural pattern (applies to every codec)

Every codec has three zones:

| Zone | Work | Where |
|------|------|-------|
| **Massively parallel** | DCT / MDCT, motion estimation, quantization, loop filter, inverse transforms, motion compensation, color-space conversion | **ILGPU kernels** - backend-agnostic |
| **Inherently sequential** | Entropy coding (arithmetic / Huffman / range coders), LPC prediction, rate control feedback | **C# CPU** - no way around it |
| **Coordination** | Frame buffering, codec negotiation, API surface | **C# CPU** |

**Entropy coders cannot be parallelized.** Arithmetic coding is inherently sequential - each symbol updates state the next symbol depends on. This is a hard wall in all modern codecs. GPU pays off for transform coding, motion estimation, quantization, loop filter; CPU handles the sequential back-end. This hybrid model is the same for every codec in scope.

**No CPU fallback branches.** Per Captain's rule, ILGPU runs on all 6 backends including CPU. One implementation, runs everywhere.

## Phasing

### Phase 0 - Planning (current, 2026-04-23)

- This roadmap doc
- Phase 1 (Opus) breakdown (collaborative with Captain)
- Research pass on 4 unknowns (below)
- Project skeleton (deferred until Phase 0 planning complete)

### Phase 1 - Opus encoder + decoder

- Smallest codec, fastest path to "SpawnDev has shipped its first working codec"
- Serves as the template for all subsequent codecs:
  - Entropy coder implementation (range coder)
  - Frame management infrastructure
  - Test harness scaffolding (bit-exact vs reference)
  - Public API surface shape
- CELT layer (transform-based) is MDCT - GPU-friendly via ILGPU
- SILK layer (predictive / LPC) is sequential - CPU
- Realistic effort: 4-8 focused sessions over a few weeks

### Phase 2 - VP8 decoder

- Simpler than encoder (no motion estimation, no rate control, no mode decision)
- Enables receiving VP8 video in pure .NET
- Pairs with Phase 1 Opus for a complete Opus + VP8 decode stack
- Realistic effort: several weeks

### Phase 3 - VP8 encoder

- Intra-only first (validates the full DCT / quant / entropy pipeline end-to-end)
- Then motion estimation → inter frames → rate control
- Where ILGPU really pays off (motion estimation is massively parallel block-matching)
- Realistic effort: 2-4 months

### Phase 4+ - VP9, AV1 decoder, AV1 encoder, FLAC, Vorbis

Each subsequent codec reuses the entropy coders / frame coordinators / test harness / GPU kernel patterns from Phases 1-3. Each has decreasing marginal cost as infrastructure accumulates.

**Full production-quality suite: multi-year effort.** Not a deterrent - this is foundational infrastructure that SpawnDev projects will depend on for years. Going in eyes open.

## Unknowns to close (Rule 4b - no guessing)

These get resolved in the Phase 0 research pass before we commit to implementation scope:

| # | Question | Why it matters | Owner | Status |
|---|----------|----------------|-------|--------|
| 1 | Existing pure-.NET codec implementations on nuget.org (Opus, VP8, VP9, AV1, FLAC, Vorbis) | If compatible-licensed .NET impls exist, we may fork instead of writing from scratch | Tuvok | Open |
| 2 | libopus / libvpx / libaom actual LoC | Informs realistic effort estimates | Tuvok | Open |
| 3 | Reference test vector availability (RFC 6716 for Opus, RFC 6386 for VP8, AV1 spec) | Needed for bit-exact verification per Rule #1 "real tests" | Tuvok | Open |
| 4 | Prior art: existing ILGPU / GPU-accelerated codec implementations | Informs our GPU kernel architecture | Tuvok | Open |

## Relationship to other SpawnDev libraries

- **SpawnDev.ILGPU** - required dependency. All GPU kernels via ILGPU. Probably target `4.9.2` or later on nuget.org.
- **SpawnDev.RTC** - will eventually consume SpawnDev.Codecs Opus to replace the SipSorcery Opus path when Phase 1 ships.
- **SpawnDev.MultiMedia** - complementary: MultiMedia handles patent-encumbered codecs via platform P/Invoke (H.264, H.265, AAC); Codecs handles patent-clean codecs via pure-.NET. Both libraries feed into SpawnDev.RTC's codec pipeline.
- **SipSorcery fork (in SpawnDev.RTC tree)** - current Opus source will be swappable to SpawnDev.Codecs when Phase 1 ships. No urgency - SipSorcery's Opus works fine today.
- **No circular dependencies.** SpawnDev.Codecs sits between ILGPU and consumers (RTC, MultiMedia).

## Relationship to Riker's Phase 4 video call work (approved 2026-04-23)

Riker's near-term Phase 4 delivery path (C + B, approved separately in `tuvok-to-riker-phase4-c-plus-b-approved-2026-04-23.md`):

- **C today:** SipSorcery Opus audio bridge + real desktop↔browser e2e call test
- **B next session:** MediaFoundation H.264 encoder P/Invoke in SpawnDev.MultiMedia (Windows)
- **Linux + macOS later:** VAAPI + VideoToolbox via same P/Invoke pattern

**SpawnDev.Codecs is a parallel long-term track:**

- When Codecs Phase 1 ships → SpawnDev.RTC swaps SipSorcery Opus for SpawnDev.Codecs Opus
- When Codecs VP8 enc/dec ship → SpawnDev.MultiMedia gains a pure-.NET open-source VP8 path alongside the platform H.264 paths
- **Both tracks converge on "pure-SpawnDev open-source media stack."** Neither blocks the other.

## Next actions (today)

1. **Collaborative Phase 1 (Opus) planning session with Captain** - API surface, project target framework, test harness design, Opus-specific GPU/CPU split, SipSorcery integration strategy
2. **Research pass on 4 unknowns** - close the Rule 4b unknowns with evidence
3. **Project skeleton** - only AFTER planning is coherent (csproj, solution, initial directory layout, README, reference to ILGPU)
4. **Phase 1 implementation** - editor assignment TBD, begins once skeleton lands

## Open questions for Captain (for the collaborative planning session)

1. **Repo structure:** own GitHub repo (`LostBeard/SpawnDev.Codecs`) or subproject under an existing SpawnDev repo?
2. **Target framework:** .NET 10 (matching SpawnDev.ILGPU 4.9.2)?
3. **Planning depth:** formal API design now, or defer that until after the research pass informs our options?
4. **Long-term ownership:** this is a multi-year editor project. Dedicated crew member, or rotate the team through scope-sized chunks? My take: Tuvok takes planning + research; editor assignment once Phase 1 is ready to start.
5. **Versioning strategy:** start at 0.1.0-alpha during Phase 1 development? Or match a SpawnDev convention I should check?
6. **License:** presumably MIT matching the rest of SpawnDev, but worth confirming given codec-world licensing is more charged than typical code.

---

🖖 Tuvok
