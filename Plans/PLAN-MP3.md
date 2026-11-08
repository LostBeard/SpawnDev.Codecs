# PLAN: MP3 (MPEG-1/2/2.5 Layer III) encoder + decoder

Started 2026-11-08 for SpawnGen's ComfyUI parity (Save Audio (MP3) / Save Audio (Advanced) mp3, Load Audio).

## Decisions (TJ, 2026-11-08)
- MP3 is in scope (all MP3 patents expired in 2017; TJ approved the charter change directly).
- ENCODER: clean-room, MIT, from ISO/IEC 11172-3 (and 13818-3 for the lower rates) and public papers. NO LAME code:
  LAME is GNU Library GPL v2 and its LICENSE requires modifications to be released under the LGPL. The files are valid
  MP3 that every decoder plays; they are not byte-identical with ComfyUI's (LAME) files.
- DECODER: a port of minimp3 (CC0 / public domain, github.com/lieff/minimp3). ComfyUI decodes MP3 with FFmpeg's own
  decoder (LGPL, not ported); report how close the floats are.
- Never read or port LAME's or FFmpeg's MP3 code. Tables that the standard defines (Huffman codes, scalefactor bands,
  window / filterbank coefficients) may be taken from minimp3 (CC0) or computed from the standard's formulas.

## What ComfyUI asks of the encoder (Lore, PyAV 18.1 / FFmpeg 8)
- libmp3lame gets s32p (planar int32) from PyAV.
- "V0": codec_context.qscale = 1 (VBR, highest VBR quality); "128k" and "320k": bit_rate (CBR).
- Container: FFmpeg's mp3 muxer: ID3v2 tag with the metadata (prompt / workflow), a Xing/Info frame. Exact layout to be
  measured by Lore; the container is plain spec work and can match byte for byte.

## Architecture (host = pure coordinator; every stage a kernel)
| Stage | Work | Parallelism |
|---|---|---|
| A. Analysis filterbank | 512-tap polyphase window + 32-band matrixing per time slot | thread per (channel, slot, band) |
| P. Psychoacoustics | FFT 1024 / 3 x 256 per granule, energies and spreading per partition, tonality, masking threshold per scalefactor band, perceptual entropy | thread per (channel, granule) (+ inner parallel FFT later) |
| W. Window switching | block type sequence (long / start / short / stop) from attacks | one thread per channel (cheap pass) |
| B. MDCT | alias reduction + 18-point (long) or 3 x 6-point (short) MDCT | thread per (channel, granule, band) |
| S. Stereo | M/S decision per granule | thread per granule |
| R. Bit budget | CBR: per-granule demand from PE, then the reservoir allocation (sequential, cheap); VBR: none | one thread |
| Q. Quantization | outer (noise vs threshold, scalefactor amplification) / inner (global gain for the budget) loops, Huffman table choice, region split, count1 | thread per (granule, channel), all granules in parallel |
| F. Frame assembly | VBR: smallest bitrate index per frame + reservoir; main_data_begin; header + side info; bit copy | sequential offsets (one thread), parallel copy |
| C. Container | Xing / Info frame, ID3v2 tag | host-side file sink (like OggOpusWriter / WebMWriter) |

Granule bit budgets are fixed before quantization, so quantization runs every granule in parallel; only the budget
pass and the frame offsets are sequential, and both are cheap.

## Steps
1. Tables + bitstream: frame header, side info (MPEG-1 and MPEG-2/2.5 LSF), Huffman code tables (built from the
   standard's tables), scalefactor band tables. Test: hand-built frames (silence, a single tone with fixed quantization)
   decode in FFmpeg.
2. Filterbank + MDCT (long blocks): test by encoding with a very fine fixed quantization at 320k and decoding with
   FFmpeg: reconstruction SNR well above 60 dB on tones.
3. Quantization + Huffman + CBR reservoir (long blocks, L/R): 128k / 320k files FFmpeg decodes, SNR on music-like tones.
4. Psychoacoustic model (ISO model 2 as described in Annex D), window switching (short blocks), M/S stereo.
5. VBR (V0 quality target), Xing / Info frame, ID3v2 tag; LSF rates (16 / 22.05 / 24, 8 / 11.025 / 12 kHz).
6. Decoder: minimp3 port as kernels (frame sync, side info, Huffman, requantize, stereo, reorder, alias, IMDCT, synthesis),
   tested against minimp3 itself (bit-exact float target) and against FFmpeg (report the difference).
7. Warm-ups, browser lanes, quality report (SNR / bitrate vs LAME at the same settings, losses included).

## Oracles
- FFmpeg 8.1 decode (desktop tests): files must decode without errors; SNR vs the source.
- minimp3 (C, built locally or via a Python binding) for the decoder's bit-exact target.
- LAME through FFmpeg only as a quality reference (bitrate / SNR comparison), never as code.
