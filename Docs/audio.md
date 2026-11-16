# Audio codecs

Namespaces `SpawnDev.Codecs.Audio.Vorbis`, `.Flac`, `.Mp3`, `.Opus`, and `SpawnDev.Codecs.Container.Ogg`. `.Wav` and `.Aiff`
are in the `SpawnDev.Codecs.References` assembly.

## Vorbis

`VorbisAudioDecoderGpu` decodes Vorbis audio packets on the accelerator: the bit-stream decode (floors, residues,
codebooks) is a resumable single-thread step kernel, then floor rendering, inverse channel coupling, the floor x residue
product, the IMDCT and the windowed overlap-add with block switching run as parallel kernels. Any channel count. It
matches libvorbis within 1e-4 per sample on six real libvorbis streams (two floors, block switching, coupled stereo with
residue type 2, several quality setups; the observed maximum is 7.5e-7).

```csharp
using SpawnDev.Codecs.Audio.Vorbis;

using var dec = new VorbisAudioDecoderGpu(accelerator, identificationHeader, setupHeader);
foreach (var packet in audioPackets)
{
    float[] samples = await dec.DecodePacketAsync(packet);   // interleaved; empty for the first packet
}
```

The three overloads:

| Overload | Returns |
|----------|---------|
| `DecodePacketAsync(ReadOnlyMemory<byte> packet)` | `float[]` (a terminal CPU sink) |
| `DecodePacketAsync(ArrayView1D<byte> packet, int packetLength, ArrayView1D<float> pcmOut)` | sample frames written into `pcmOut` (size it with `MaxSamplesPerPacket`); the packet is read where it is |
| `DecodePacketAsync(TypedArray packet)` (browser) | a JS `Float32Array`; nothing enters the .NET heap |

The first packet only primes the overlap; every later packet returns `previousBlockSize / 4 + blockSize / 4` frames,
as libvorbis does. Ogg parsing (`OggPageReader`, `OggPacketReader`) and the header parsers
(`VorbisIdentificationHeaderParser`, `VorbisSetupHeaderParser`) are in the library; `VorbisOggDecoder` (CPU, in
`SpawnDev.Codecs.References`) decodes a whole `.ogg` and is the reference the GPU decoder is checked against.

`VorbisAudioEncoderGpu` is a first version: mono, one fixed block size, a two-point floor and a single residue book. It
writes valid streams (our decoders and ffmpeg read them); a libvorbis-grade encoder is planned.

## FLAC

`FlacEncoderGpu.EncodeAsync` encodes interleaved integer PCM (any rate FLAC carries, 1..8 channels, 4..24 bits, any
length) to a complete FLAC file at compression levels 0..8 (`FlacEncodeOptions`: level, block size, stereo mode,
padding, vendor, tags, channel mask). Levels 0..2 use the fixed polynomial predictors, 3..8 add LPC (windowed
autocorrelation, Levinson-Durbin, 15-bit coefficients, orders up to 12); every subframe takes the cheapest of constant /
verbatim / fixed / LPC with exact Rice parameters (escape-coded partitions included), every stereo frame the cheapest of
independent / left-side / right-side / mid-side. STREAMINFO carries the frame sizes and the MD5 of the samples. All of it
runs on the accelerator; on the test fixtures the files are lossless (our decoder and libFLAC), carry the same metadata
blocks as FFmpeg's and their frames are 0.80 to 1.00 times the size of FFmpeg's at the same level (0.989 over all
fixtures together). The encoder is written from RFC 9639 and the linear prediction and Rice coding literature (no FFmpeg
or libFLAC code). Overloads: managed, JS `Int32Array`, and
`ArrayView1D` in and out; `WarmUpAsync` compiles the kernels ahead of time. The v1 `EncodeStreamAsync` (VERBATIM
subframes, 4096-sample blocks) is kept. The file does not depend on the backend: the analysis window is a series in
IEEE double arithmetic and the LPC kernels load with strict IEEE double emulation on WebGPU and WebGL (checked on the CPU
accelerator, CUDA and OpenCL by `FlacEncoderGpu_SameFileOnEveryBackend`).

`FlacDecoderGpu.DecodeAsync` decodes any conforming FLAC stream (RFC 9639) on the accelerator: fixed or variable block
size, 4..32 bits, 1..8 channels, every subframe type, Rice / Rice2 with escapes, wasted bits, every stereo mode, MD5
check; `DecodeFloatAsync` gives float32 planar samples as ComfyUI's Load Audio sees them. `FlacDecodeResult` carries the
rate, channels, bit depth, samples, the VORBIS_COMMENT vendor and tags and the MD5 check; a damaged frame decodes to
silence of its length and the samples after it keep their time, as libFLAC does.

## Opus

`OpusDecoderGpu` decodes SILK, CELT and hybrid packets on the accelerator, with mode switches, redundancy frames, loss
concealment and DTX. On 21 real streams made by libopus 1.5.2 (fixed point, built from source) the PCM and the final range
are equal on every packet. `OpusEncoderGpu` is a CELT-only encoder (2.5 to 120 ms frames, CBR / VBR / constrained VBR,
complexity 0..9, 8 to 48 kHz); its packets and final range are byte-identical with libopus 1.5.2 on 49 streams.

```csharp
using var encoder = new OpusEncoderGpu(accelerator, 48000, 2, OpusEncoderApplication.Audio) { Bitrate = 96000 };
byte[] packet = await encoder.EncodePacketAsync(pcm16, frameSize: 960);        // 20 ms of interleaved int16

using var decoder = new OpusDecoderGpu(accelerator, 48000, 2);
float[] pcm = await decoder.DecodePacketAsync(packet);
```

`OggOpusEncoder.EncodeAsync` encodes interleaved int16 or float PCM (8, 12, 16, 24 or 48 kHz, 1 or 2 channels) to a
whole Ogg Opus file or into a `Stream` (managed, JS typed array and `ArrayView1D` overloads): application Audio,
unconstrained VBR, 20 ms packets, 64000 b/s mono or 96000 b/s stereo unless `OggOpusEncodeOptions.Bitrate` says
otherwise, pre-skip equal to the encoder's lookahead. `EncodePacketsAsync` returns the packets, the OpusHead and the
pre-skip for a container writer (`VideoFileEncoderGpu` uses it); `WarmUpAsync` loads the kernels ahead of time.
`OggOpusWriter` frames already-encoded packets into Ogg pages (RFC 3533, RFC 7845 OpusHead / OpusTags). The Ogg Opus
writer and encoder are written from those specifications; the stream layout was worked out by running ffmpeg and reading
its output, and no FFmpeg code is used.

The CPU path (SILK, CELT and hybrid, encode and decode) is in `SpawnDev.Codecs.References` on top of
[Concentus](https://github.com/lostromb/concentus) (BSD-3); it is the reference the GPU classes are checked against, and
the SILK and hybrid encoders exist only there.

## MP3

`Mp3EncoderGpu.EncodeAsync` encodes float or int32 interleaved PCM (mono or stereo, every Layer III sample rate from 8 to
48 kHz: MPEG-1, 2 and 2.5) to an MP3 file. `Mp3EncodeOptions`: `BitrateKbps` (CBR, default 128), `VbrQuality` (0..9, VBR),
`MidSide` (default on), `ForceLongBlocks`, `BitReservoir` (default on), ID3v2 tags (`WriteId3`, `Tags`, `Id3Encoder`) and the
Info frame (`WriteInfoFrame`, `InfoEncoder`), which carries the encoder delay and padding so decoders play the file
gaplessly. Everything that touches samples or bits runs on the accelerator: polyphase analysis, MDCT, an MDCT-domain
psychoacoustic model, the scalefactor and quantization loops, window switching (short blocks, pre-echo control), Huffman
coding, the bit reservoir, mid / side choice and frame assembly. It is written from ISO 11172-3 and 13818-3 and the public
literature; it contains no LAME code. FFmpeg decodes every test file without errors and gaplessly (exact input length).

`Mp3DecoderGpu` is a port of minimp3 (CC0) as kernels: `DecodeAsync` returns interleaved float32, `DecodeFloatAsync` planar
float32, `WarmUpAsync` loads the kernels. MPEG-1 / 2 / 2.5 Layer III, mono and stereo, mid / side and intensity stereo,
short and mixed blocks, free format, CRC frames and ID3v2 tags; gapless with the Info frame's delay and padding. Damaged
frames come out as silence and keep their time. Bit-exact with minimp3 on 21 fixtures.

## Audio containers and helpers

| | Read | Write |
|---|---|---|
| RIFF / WAVE | 8 to 32-bit PCM, 32-bit float, multi-channel, WAVE_FORMAT_EXTENSIBLE | yes |
| AIFF | 8 to 32-bit PCM, 80-bit extended sample rate | yes |
| Ogg | pages, packets, CRC-32, multiplexed streams | yes (`OggPageWriter`; `OggOpusWriter` for Opus files) |

Transforms: GPU MDCT / IMDCT kernels (the Vorbis decoder's IMDCT computes each output sample directly; a fast transform
is planned) and CPU references.
