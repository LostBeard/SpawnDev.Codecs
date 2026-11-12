# ComfyUI LoadVideo: YUV 4:2:0 (8-bit, limited range) -> RGB, derived as a black box

ComfyUI (GPL-3) and FFmpeg (LGPL) were not read. Everything here was measured (2026-11-11) by running programs and
comparing their outputs:

1. Lore ran ComfyUI's `LoadVideo -> GetVideoComponents` on three VP9 WebMs written by ComfyUI and dumped the IMAGE
   tensors (`D:\users\tj\Projects\_DevComms\codecs-load-refs`).
2. Every value is an exact multiple of 1/65535 (16-bit RGB), and all 108 frames are bit-identical to
   `ffmpeg -sws_flags bilinear -pix_fmt gbrp16le` on the same files. A raw yuv420p input gives the same output as the
   WebM, so the ffmpeg command line tool served as the oracle for synthetic probes.
3. Probes: flat chroma (matrix and luma), chroma columns / rows alternating 68 / 188 (exact per-column weights,
   16384ths horizontally, 4096ths vertically, for 20 widths and 18 heights), impulse responses, and random images.

## The conversion

Inputs: Y (W x H), U and V (cw x ch, cw = (W + 1) / 2, ch = (H + 1) / 2), 8-bit. Output: R, G, B 16-bit.

**Chroma taps, per axis** (chroma size n, output size N, output index o; K = 16384 horizontally, 4096 vertically):

```
inc = ((n << 16) + (N >> 1)) / N                 // 16.16, rounded
xp  = o * inc + ((inc - 1) >> 1) - 32768         // centred siting; arithmetic shifts
if n < 4:  tap = clamp((xp + 32768) >> 16, 0, n - 1), weight K   (nearest)
else if xp < 0: tap 0, weight K
else:
    i = xp >> 16, f = xp & 0xFFFF
    if f >= 65405: i += 1, r = 0                 // positions just below an integer snap to it
    else: r = (f * K + 32768) >> 16
    if i >= n - 1: tap n - 1, weight K
    else: taps (i, K - r), (i + 1, r)
```

**Chroma value at 9 fraction bits** (horizontal first, truncated to 11 fraction bits, then vertical):

```
h(row)  = (wl * C[row][cl] + wr * C[row][cr]) >> 3          // 14-bit weights
c9      = (wt * h(rowT) + wb * h(rowB)) >> 14                // 12-bit weights; c9 = chroma * 512
```

**RGB** (BT.601 limited range, 8.8-ish scale; clip to [0, 65535]):

```
yt = (Y - 16) * 9539 * 512 + 8192
R = clip((yt + (c9v - 65536) * 13075) >> 14)
G = clip((yt - (c9u - 65536) * 3209 - (c9v - 65536) * 6660) >> 14)
B = clip((yt + (c9u - 65536) * 16525) >> 14)
```

**Float** (ComfyUI's IMAGE): `R * (1f / 65535f)` in float32 (a multiply by the float32 reciprocal; plain division
differs on about 0.6% of values).

Validated (a Python model of exactly this) with zero mismatches against the oracle on 150+ random image sizes from
1x1 to 1920x8 and 8x1080, odd and even, including the n < 4 cases. The accelerator kernel's tests compare against the
108 ComfyUI reference frames directly.
