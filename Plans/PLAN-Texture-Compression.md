# PLAN: GPU texture compression (BCn first) (Geordi, 2026-10-27)

TJ 10-26 asked for general data compression "with Google's suggestions BC7, bitshuffle, LZ4". Bitshuffle (filters) and
LZ4 are done in PLAN-Data-Compression.md; that plan left texture formats to their own track. This is it. Consumers:
SpawnScene / VoxelEngine / LostSpawns textures (WebGPU `texture-compression-bc` on desktop; ETC2 / ASTC on mobile and
Quest later), and anything that uploads big RGBA textures every frame.

Same rules as every codec here: ILGPU kernels on all six backends, host = coordinator, three overloads (managed /
TypedArray / ArrayView1D), no bulk bytes on the .NET heap, a reference oracle for every byte.

## Formats, in build order

1. **Decoders BC1, BC3, BC4, BC5, BC7** (exact; the verification side and the fallback where the GPU lacks the format).
   One thread per output pixel (gather form: a thread reads its block and decodes only what its pixel needs), so it
   also runs on WebGL. BC7 decode is integer-exact by the spec; BC1's 3- and 4-colour interpolation rounding differs
   between decoders, so the oracle and its rounding are named in the tests.
2. **Encoders BC1 / BC4 / BC5** (range fit + refinement) and **BC3** (BC1 colour + BC4 alpha).
3. **Encoder BC7** (modes 1 / 6 first, then the partitioned modes and alpha modes; bc7enc's search). Encoders are not
   unique: the gate is that the reference decoders read our blocks and the PSNR is within a stated bound of the
   reference encoder at comparable settings, logged with the losses.
4. Later: BC6H, ETC2 / EAC, ASTC (the same oracles cover them).

## Done

- 10-27 **Decoders** BC1 / BC3 / BC4 / BC5 / BC7 (`BcnDecoderGpu`, gather form): 29 fixtures bit-exact, all 7 lanes. BC7
  tables read back from a reference decoder (`scripts/texture/derive_bc7_tables.py`).
- 10-27 **Encoders** BC1 / BC3 / BC4 / BC5 (`BcnEncoderGpu`, one thread per output word): a port of etcpak's portable
  DXTC compressors, byte-identical on 19 fixtures, all 7 lanes. (The etcpak 0.9.15 Windows wheel runs the portable paths:
  its BC4 matches that arithmetic 256 / 256 blocks, the SSE4.1 variant 246; the portable BC1 ignores alpha.) Quality is
  etcpak's (fast range-fit); a higher-quality mode (cluster fit / least squares) can follow, gated on PSNR.
- 10-27 **Encoder BC7** (`Bc7EncoderGpu`, behind `BcnEncoderGpu` with `BcnFormat.Bc7`): a port of bc7enc (the copy etcpak
  0.9.15 builds) at its default settings: modes 6 + 1 for opaque blocks, 6 + 5 + 7 with alpha, perceptual weights, 64
  partitions with the filter bank, least squares. One launch per mode (one thread per block, bc7enc's order, a packed
  40-byte best-so-far per block), then a packing launch (one thread per output word). 9 fixtures (8435 blocks): CPU and
  Velocity byte-identical with bc7enc on every block. GPU lanes: see the test log.
  The etcpak wheel's SIMD modules are tagged cp37 and do not load on Python 3.13+, so the oracle runs `_etcpak_none`
  (portable code); the fixture script asserts it.

## Oracles (Python, oracle venv)

- `etcpak` (K0lb3's bindings of etcpak; BC7 through bc7enc): `compress_bc1/3/4/5/7`. Its `decompress_*` are broken in
  0.9.15 (MemoryError / native faults), and bc7enc with a mode-5-only mask crashes on some pictures; mode-mask encodes
  are not deterministic.
- `texture2ddecoder` (K0lb3, from AssetStudio): `decode_bc1/3/4/5/6/7`, ETC, ASTC: a second, independent decoder.
- Fixtures: `scripts/texture/make_bcn_fixtures.py` writes blocks made by the reference encoders over real pictures
  (TestData photos, alpha pictures) plus synthetic blocks that hit every BC7 mode / partition / rotation, and the RGBA
  both reference decoders produce for them.

## Gates

- Decode: bit-exact with the reference decoder(s) on every fixture (BC1: the named rounding).
- Encode: reference decoders read our blocks; PSNR vs the reference encoder logged in _TestStats (losses included).
- Green on CPU, Velocity, CUDA, OpenCL, WebGPU, Wasm, WebGL.
