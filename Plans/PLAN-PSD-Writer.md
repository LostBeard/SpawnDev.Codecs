# Plan: layered PSD writer (PsdEncoderGpu)

Status: approved by TJ 2026-11-25 (request from Lore: SpawnGen's compositor "Export PSD"). Written from Adobe's published
Photoshop File Formats Specification and ComfyUI's output measured as a black box (Lore); no GPL/LGPL reader or writer is read.
Cardinal rule: the pixel bytes of the file are produced by kernels; the host coordinates and prepares metadata (layer names, rectangles,
blend keys, the lengths it learns from the kernels).

## What it writes
PSD version 1, 8 bits per channel, RGB colour mode (3), RLE (PackBits) everywhere:
1. header (26 bytes), colour mode data (length 0), image resources (one block: 0x03ED resolution, 72 dpi), layer and mask info, merged
   image data.
2. layers bottom first: record (rectangle top/left/bottom/right, channel list -1 R G B with lengths, 8BIM blend signature + key, opacity,
   clipping 0, flags, name as Pascal string + 'luni' Unicode block), then channel data (compression 1 + row byte counts + PackBits).
   - flags 0x08 when visible; hidden sets the bit measured from ComfyUI's output (see "measurements needed").
   - rectangles may be negative or run past the canvas; the layer's pixels are stored as given.
   - layer count: positive when the composite is RGB, negative (with a merged alpha channel) when it is RGBA.
3. composite: 3 or 4 channels, RLE, caller-supplied.
Limits: canvas and layer sides 1..30000 (PSD v1); up to 8192 layers.

## Shape (kernels)
Per image plane (layer channel or composite channel) and row:
1. `PackRowKernel` (one thread per plane row): reads the interleaved RGBA8 of the source, writes the PackBits stream of one channel row into
   a fixed slot (stride = width + (width + 127) / 128 + 2 bytes) and its length to `rowLen[]`. Launched in bounded batches
   (rows x width per launch capped) through LaunchPacer.
2. `RowTableKernel` + host read of the per-plane totals: lengths of every row table (2 bytes each) and the plane's packed size are
   scalars the host needs for the records.
3. offsets: serial scan over the rows (one thread, bounded launches: an exclusive sum of up to 4 x 30000 rows per plane).
4. `AssembleKernel`: one thread per output byte: the host-built metadata blob (header, records, resources) and, by section, row tables and packed rows
   at their offsets; output through the caller's view.
Padded row-length tables are big-endian u16 (the count cannot exceed 30000 + 235, well under 65535).

PackBits choice: greedy with runs of at least 3 as repeats and literals up to 128 (any valid stream decodes the same; the stream is not
pinned to another encoder's bytes). Tests compare decoded pixels and structure, not the encoded bytes.

## API
`PsdEncoderGpu` (EncodeAsync of a `PsdDocument` into a caller view; managed byte[] and Uint8Array overloads as the other encoders),
`GetMaxEncodedSize`, `WarmUpAsync`. Input: `PsdDocument { Width, Height, IReadOnlyList<PsdLayer> Layers (bottom first), Composite }`,
`PsdLayer { Name, Left, Top, Width, Height, Rgba (accelerator view), Blend, Opacity, Visible }`, `PsdBlendMode` (the 26 of SpawnGen's compositor).

## Tests (each with a red check)
- Pillow (PsdImagePlugin, test oracle venv) opens the file: every layer's RGBA, placement, name, blend key and visibility, and the composite, match what
  was encoded, exact pixels; PSD's layer order.
- psd-tools is not installed in the oracle venv: if needed, layer-record bytes are also checked by a small parser written in the test.
- Edge cases: a layer with a negative left/top, a layer past the canvas, an opacity 0 and a hidden layer, a one-pixel layer, a row of all-equal and
  a row of incompressible bytes, Unicode and >255-byte names, no layers (composite only), RGBA composite, 8192 layers (a count test), sizes 1x1 and
  30000 x 1 (row length limit).
- Every backend that runs codec kernels; Lore: open in Photoshop (if available), GIMP, Krita, Photopea, plus SpawnGen's own round trip.

## Measurements needed from Lore (ComfyUI black box, asked 2026-11-25)
Blend key of each of the 26 modes (grain extract / merge have no Photoshop key), hidden-layer flags byte, whether the flags' bit 1 means hidden
(the spec calls it "visible"), how Photoshop-compatible readers treat `luni` padding.

## Open
- Pascal name encoding: ASCII characters as given, others replaced by '?', first 31 characters; the real name is in `luni` (UTF-16BE).
- Global layer mask info: 0 length. Document-level additional layer info: none.
