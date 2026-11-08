# PLAN: clean-room rewrite of three components (2026-11-08)

## Why
On 2026-11-08 three components were written as ports of FFmpeg 8.1 source (libavformat matroskaenc.c / oggenc.c /
flacenc.c, libavcodec flacenc.c / lpc.c / libopusenc.c). FFmpeg is licensed LGPL-2.1-or-later; SpawnDev.Codecs is
MIT. TJ decided (2026-11-08): delete the ported code and rewrite clean-room, MIT. The ported files were removed in the
commit that adds this plan. Their acceptance tests are parked in
`SpawnDev.Codecs.Demo.Shared/UnitTests/CleanRoomPending/` (excluded from compilation) and come back as each rewrite lands.

## Clean-room rules (for the implementers)
- Do NOT read FFmpeg source code (any file, any version, any mirror), libopus' FFmpeg wrapper, or LAME.
- Do NOT read the removed files or their history: no `git log -p`, `git show`, `git diff` or blame on
  `SpawnDev.Codecs/Container/Matroska/WebMWriter.cs`, `SpawnDev.Codecs/Container/Ogg/OggOpusWriter.cs`,
  `SpawnDev.Codecs/Audio/Opus/OggOpusEncoder.cs`, `SpawnDev.Codecs/Audio/Flac/FlacEncoderGpu.Stream.cs`,
  `FlacStreamKernels.cs`, `FlacStreamWriteKernels.cs`, and no commits before this plan that touched them.
- Do NOT open anything under any Claude scratchpad / temp directory.
- DO use: the public specifications (below); the fixtures in `SpawnDev.Codecs.Demo.Shared/TestData/` and the scripts
  that generate them (they only run the ffmpeg binary); FFmpeg / ffprobe / mkvinfo-style tools as black boxes (run
  them, inspect their output files); the parked tests; the rest of this repository.
- Matching FFmpeg's OUTPUT by observing it is fine (formats and observable behaviour are not code).

## Shared project rules
Read `D:\users\tj\Projects\AGENTS.md` and `D:\users\tj\Projects\SpawnDev.Codecs\CLAUDE.md`. In short: every encoder /
decoder stage that touches codec data is an ILGPU kernel (the host only allocates, uploads, dispatches, reads back);
container writers that frame already-encoded packets into a caller's Stream may run on the host (precedent:
`Container/Ivf/IvfWriter.cs`); bounded launches (see `LaunchPacer.cs`); no em dashes; tests in
`SpawnDev.Codecs.Demo.Shared/UnitTests`, run with `pwsh scripts/dev/build-console.ps1` then
`pwsh scripts/dev/run-desktop-tests.ps1 -Class CpuCodecsTests -Pattern <regex>` (CPU accelerator; the GPU is in use by
other sessions, do not run GPU lanes); red-check every guard (break it, see the test fail, restore).

## Component A: Ogg Opus writer + Ogg Opus encoder
Specs: RFC 3533 (Ogg), RFC 7845 (Ogg encapsulation for Opus, OpusHead / OpusTags), RFC 6716 (Opus TOC byte).
Public API to provide (namespaces as listed):
- `SpawnDev.Codecs.Container.Ogg.OggOpusWriterOptions` (record: `Channels` 1|2 required, `PreSkip` required,
  `InputSampleRate` = 48000, `OutputGain` (short), `Vendor` = "ffmpeg", `Tags` (IReadOnlyList<KeyValuePair<string,string>>?,
  written in order as "KEY=value"), `Serial` (uint), `PageDurationUs` = 1_000_000).
- `SpawnDev.Codecs.Container.Ogg.OggOpusWriter(Stream, OggOpusWriterOptions, bool leaveOpen = false)` with
  `WritePacket(ReadOnlySpan<byte> packet, long pts, int duration)` (pts in 48 kHz samples, the first packet at -PreSkip),
  `Finish()`, `Dispose()`.
- `SpawnDev.Codecs.Audio.Opus.OggOpusEncodeOptions` (record: `Bitrate` int?, `Complexity` = 9, `Vbr` = true,
  `VbrConstraint` = false, `Application` = Audio, `PacketLossPercent`, `PhaseInversionDisabled`, `Vendor`, `Tags`,
  `Serial`, `InputSampleRate` int?) and static `OggOpusEncoder.EncodeAsync(Accelerator, ReadOnlyMemory<short> |
  ReadOnlyMemory<float>, int sampleRate, int channels, [Stream output], OggOpusEncodeOptions?)` (byte[] overloads and
  Stream overloads), `OggOpusEncoder.WarmUpAsync(...)`, and a float -> int16 kernel class `PcmConvertKernels`
  (`FloatToS16Kernel`: lrintf(x * 32768) semantics, round half to even, clipped to int16; NaN and values beyond the int32
  range -> -32768).
Observable behaviour to match (from the fixtures in TestData/OggOpus, made by `scripts/audio/make_ogg_opus_fixtures.py`):
- Given the packets inside each fixture file, the writer reproduces the fixture file byte for byte (serial 0, vendor and
  tags as in the fixture's OpusTags). Derive the page packing rules from the fixtures and RFC 3533 / 7845 (granule
  positions, when pages close, flags of the first and last pages, CRC).
- The encoder uses `OpusEncoderGpu` (existing, CELT-only) with application Audio, unconstrained VBR, 20 ms frames,
  `Lookahead` as the pre-skip (at 48 kHz), and produces the same page layout (page count, granules, flags), OpusHead and
  OpusTags as each fixture for the same input; its packets need not equal the fixture's (different Opus encoder).
Acceptance tests: `CleanRoomPending/CodecsTestBase.OggOpusTests.cs` (move it back to UnitTests/).

## Component B: WebM writer
Specs: the Matroska specification (RFC 9559, EBML RFC 8794) and the WebM container guidelines (webmproject.org).
Public API: re-create the API the parked test `CleanRoomPending/CodecsTestBase.WebMWriterTests.cs` uses
(`SpawnDev.Codecs.Container.Matroska.WebMWriter` and its options / track types, as named in that test).
Observable behaviour: given the packets and timestamps in each fixture (TestData/WebMMux, made by
`scripts/video/make_webm_mux_fixtures.py` with `ffmpeg -c copy -fflags +bitexact`), write the fixture file byte for
byte. Work out the element layout by inspecting the fixtures (EBML structure, element order, sizes, void elements,
cluster / cue rules) and from the Matroska / WebM specs.
Acceptance tests: the parked WebMWriter test.
Status (2026-11-08): done. `Container/Matroska/WebMWriter.cs` + `EbmlBuffer.cs`; the test is back in `UnitTests/` and
passes on every fixture (21 cases, 10 of them added to pin the cluster rules and track header fields).

## Component C: FLAC stream encoder
Specs: RFC 9639 (FLAC); the literature on linear prediction (autocorrelation + Levinson-Durbin, windowing) and Rice
coding.
Public API: `FlacEncoderGpu.EncodeAsync(ReadOnlyMemory<int> interleaved, int channels, int sampleRate, int bitsPerSample
(16|24), FlacEncodeOptions?)` (+ `Int32Array` and `ArrayView1D<int>` overloads as the parked test uses), `WarmUpAsync`,
and `FlacEncodeOptions` (record: `CompressionLevel` 0..8 default 5, `BlockSize` int?, `ChannelMode` (independent /
left-side / right-side / mid-side / auto), `Padding` = 8192, `Vendor` = "ffmpeg", `Tags`, `ChannelMask` int?). All
codec-data work in kernels (the existing v1 class `FlacEncoderGpu` is the place; keep its v1 API working).
Observable behaviour (TestData/FlacEnc, `scripts/audio/make_flac_encoder_fixtures.py`):
- Lossless: every file decodes to the input exactly (libFLAC oracle and our FlacDecoderGpu).
- Metadata blocks as the fixtures have them: STREAMINFO (with the MD5 of the samples; use `FlacMd5Kernels`),
  VORBIS_COMMENT (vendor + tags in order), PADDING of `Padding` bytes; block sizes per level as the fixtures show.
- Compression: each level within a few percent of the fixture file's size (state the measured ratios). Byte identity of
  the frames with FFmpeg is NOT required.
Acceptance tests: the parked FLAC encoder test, adapted (byte identity of frames -> lossless + size ratio; metadata
bytes still exact). Lore (SpawnGen) checks ComfyUI parity afterwards.
