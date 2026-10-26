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

`FlacEncoderGpu` and `FlacDecoderGpu` are first versions with a fixed subset: 4096-sample blocks, 44.1 kHz, 16-bit,
VERBATIM subframes, up to 8 channels. Frames are encoded and decoded in parallel on the accelerator. The full codec
(FIXED and LPC prediction, Rice coding, stereo decorrelation, MD5) exists as the CPU reference (`FlacEncoder` /
`FlacDecoder` in References) and is the next GPU port.

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
