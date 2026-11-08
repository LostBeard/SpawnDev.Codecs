# Audio codecs

Namespaces `SpawnDev.Codecs.Audio.Vorbis`, `.Flac`, `.Opus`, `.Wav`, `.Aiff`, and `SpawnDev.Codecs.Container.Ogg`.

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
blocks as FFmpeg's and are no larger than FFmpeg's at the same level. Overloads: managed, JS `Int32Array`, and
`ArrayView1D` in and out; `WarmUpAsync` compiles the kernels ahead of time. The v1 `EncodeStreamAsync` (VERBATIM
subframes, 4096-sample blocks) is kept.

`FlacDecoderGpu.DecodeAsync` decodes any conforming FLAC stream (RFC 9639) on the accelerator: fixed or variable block
size, 4..32 bits, 1..8 channels, every subframe type, Rice / Rice2 with escapes, wasted bits, every stereo mode, MD5
check; `DecodeFloatAsync` gives float32 planar samples as ComfyUI's Load Audio sees them.

## Opus

The CPU path (SILK, CELT and hybrid, encode and decode) is in `SpawnDev.Codecs.References` on top of
[Concentus](https://github.com/lostromb/concentus) (BSD-3). On the accelerator the library has the Opus range coder and
the SILK decode primitives (indices, pulses, shell coder, LPC, synthesis), checked against Concentus; the GPU Opus
encoder / decoder pair is planned ([Plans/PLAN-Opus-GPU-Pair.md](../Plans/PLAN-Opus-GPU-Pair.md)).

## Audio containers and helpers

| | Read | Write |
|---|---|---|
| RIFF / WAVE | 8 to 32-bit PCM, 32-bit float, multi-channel, WAVE_FORMAT_EXTENSIBLE | yes |
| AIFF | 8 to 32-bit PCM, 80-bit extended sample rate | yes |
| Ogg | pages, packets, CRC-32, multiplexed streams | yes |

Transforms: GPU MDCT / IMDCT kernels (the Vorbis decoder's IMDCT computes each output sample directly; a fast transform
is planned) and CPU references.
