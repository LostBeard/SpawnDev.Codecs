/*
 * SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
 *
 * Reference for Vp9IntraPred (SpawnDev.Codecs/Video/Vp9/Decoder/Vp9IntraPredict.cs): calls libvpx v1.17.0's own
 * vp9_predict_intra_block (vp9/common/vp9_reconintra.c, 8-bit, predictors from vpx_dsp/intrapred.c; BSD-3-Clause)
 * on deterministic pseudo-random cases and records inputs + output. Built and run by make_intra_fixture.py.
 *
 * Usage: intra_ref <out.bin>
 *
 * Every case is a transform block the way libvpx's decoder (vp9_decodeframe.c predict_recon_intra / set_offsets /
 * set_mi_row_col / set_plane_n4) reaches it: a random luma frame size (the 8-aligned decoded size: y_width = mi_cols * 8,
 * uv_width = y_width >> ss_x), a prediction block of 8x8..64x64 luma at a block-aligned mi position (bwl, bhl 1..4 as
 * decode_block gets them), a transform block inside it at a step-aligned (row, col) < (max_blocks_high, max_blocks_wide),
 * above_mi present unless mi_row == 0, left_mi present unless mi_col is the start of its tile column (random 64-pixel
 * tile column starts). Positions are biased to the right / bottom frame edges so the edge replication paths run.
 * The plane is filled with random pixels (also outside the frame, so a read that libvpx does not make shows).
 *
 * File: int32 magic 'VP9I' (0x49395056), int32 version 1, int32 case count; per case 16 int32
 *   tx_size, mode, bwl_in (pd->n4_wl), aoff (col), loff (row), above_mi != NULL, left_mi != NULL,
 *   mb_to_left_edge, mb_to_top_edge, mb_to_right_edge, mb_to_bottom_edge, subsampling_x, subsampling_y,
 *   frame_width, frame_height (the plane's y_width/y_height or uv_width/uv_height), plane,
 * then bytes: the row above the block from x0 - 1 to x0 + 2 * bs - 1 (2 * bs + 1 pixels), the column left of the block
 * for rows y0 .. y0 + bs - 1 (bs pixels), and libvpx's prediction (bs * bs pixels, row major). These are every pixel
 * build_intra_predictors can read outside the block (above_ref[-1 .. 2 * bs - 1], ref[i * stride - 1]).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_reconintra.h"
#include "vpx_scale/yv12config.h"

static unsigned long long rng_state = 0x5650394952414E44ULL; /* fixed seed */
static unsigned rnd(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (unsigned)(rng_state >> 33);
}
static int rnd_below(int n) { return (int)(rnd() % (unsigned)n); }

static void put32(FILE *f, int v) { fwrite(&v, 4, 1, f); }

#define MARGIN 96

int main(int argc, char **argv) {
  const int cases_per_combo = 32;
  const int count = 10 * 4 * 3 * cases_per_combo;
  int i;
  FILE *f;
  MODE_INFO dummy_mi;
  YV12_BUFFER_CONFIG buf;
  MACROBLOCKD xd;
  if (argc < 2) {
    fprintf(stderr, "usage: intra_ref out.bin\n");
    return 2;
  }
  f = fopen(argv[1], "wb");
  if (!f) return 1;
  vp9_init_intra_predictors();
  put32(f, 0x49395056);
  put32(f, 1);
  put32(f, count);
  memset(&dummy_mi, 0, sizeof(dummy_mi));
  for (i = 0; i < count; ++i) {
    const int mode = i % 10;
    const int tx = (i / 10) % 4;
    const int plane = (i / 40) % 3;
    int ssx = 0, ssy = 0, W, H, mi_cols, mi_rows, bwl, bhl, bw, bh, mi_col, mi_row, n4_wl, n4_hl;
    int fw, fh, max_w, max_h, col, row, tile_start, step, bs, x0, y0, stride, rows, k, r, c;
    unsigned char *plane_buf, *dst;
    for (;;) {
      if (plane) {
        const int fmt = rnd_below(4); /* 4:2:0, 4:2:2, 4:4:0, 4:4:4 */
        ssx = (fmt == 0 || fmt == 1);
        ssy = (fmt == 0 || fmt == 2);
      }
      bwl = 1 + rnd_below(4);
      bhl = 1 + rnd_below(4);
      if (rnd_below(3) == 0) bhl = bwl; /* square blocks are the common case */
      n4_wl = bwl - ssx;
      n4_hl = bhl - ssy;
      if (tx > n4_wl || tx > n4_hl) continue;
      W = 1 + rnd_below(rnd_below(4) == 0 ? 40 : 320);
      H = 1 + rnd_below(rnd_below(4) == 0 ? 40 : 320);
      mi_cols = (W + 7) >> 3;
      mi_rows = (H + 7) >> 3;
      bw = 1 << (bwl - 1);
      bh = 1 << (bhl - 1);
      /* block-aligned mi position inside the frame; half the time the last one (touching the right / bottom edge) */
      mi_col = rnd_below(2) ? ((mi_cols - 1) / bw) * bw : rnd_below((mi_cols - 1) / bw + 1) * bw;
      mi_row = rnd_below(2) ? ((mi_rows - 1) / bh) * bh : rnd_below((mi_rows - 1) / bh + 1) * bh;
      break;
    }
    fw = (mi_cols * 8) >> ssx;
    fh = (mi_rows * 8) >> ssy;
    memset(&buf, 0, sizeof(buf));
    buf.y_width = mi_cols * 8;
    buf.y_height = mi_rows * 8;
    buf.uv_width = buf.y_width >> ssx;
    buf.uv_height = buf.y_height >> ssy;
    memset(&xd, 0, sizeof(xd));
    xd.cur_buf = &buf;
    for (k = 0; k < MAX_MB_PLANE; ++k) {
      xd.plane[k].subsampling_x = k ? ssx : 0;
      xd.plane[k].subsampling_y = k ? ssy : 0;
      xd.plane[k].n4_w = (bw << 1) >> xd.plane[k].subsampling_x;
      xd.plane[k].n4_h = (bh << 1) >> xd.plane[k].subsampling_y;
      xd.plane[k].n4_wl = bwl - xd.plane[k].subsampling_x;
      xd.plane[k].n4_hl = bhl - xd.plane[k].subsampling_y;
    }
    /* set_mi_row_col */
    xd.mb_to_top_edge = -((mi_row * MI_SIZE) * 8);
    xd.mb_to_bottom_edge = ((mi_rows - bh - mi_row) * MI_SIZE) * 8;
    xd.mb_to_left_edge = -((mi_col * MI_SIZE) * 8);
    xd.mb_to_right_edge = ((mi_cols - bw - mi_col) * MI_SIZE) * 8;
    tile_start = (mi_col >= 8 && rnd_below(3) == 0) ? (rnd_below(mi_col / 8) + 1) * 8 : 0;
    if (tile_start > mi_col) tile_start = 0;
    xd.above_mi = mi_row != 0 ? &dummy_mi : NULL;
    xd.left_mi = mi_col > tile_start ? &dummy_mi : NULL;
    /* predict_recon_intra: transform blocks inside the visible part of the block */
    {
      const struct macroblockd_plane *pd = &xd.plane[plane];
      max_w = pd->n4_w + (xd.mb_to_right_edge >= 0 ? 0 : xd.mb_to_right_edge >> (5 + pd->subsampling_x));
      max_h = pd->n4_h + (xd.mb_to_bottom_edge >= 0 ? 0 : xd.mb_to_bottom_edge >> (5 + pd->subsampling_y));
    }
    step = 1 << tx;
    col = rnd_below((max_w + step - 1) / step) * step;
    row = rnd_below((max_h + step - 1) / step) * step;
    if (rnd_below(3) == 0) col = ((max_w - 1) / step) * step; /* last transform column: right-edge paths */
    bs = 4 << tx;
    x0 = ((mi_col * 8) >> ssx) + col * 4;
    y0 = ((mi_row * 8) >> ssy) + row * 4;
    if (!(x0 < fw && y0 < fh)) {
      fprintf(stderr, "case %d: block outside the frame\n", i);
      return 1;
    }
    /* the plane with a margin on every side, random pixels everywhere */
    stride = fw + 2 * MARGIN;
    rows = fh + 2 * MARGIN;
    plane_buf = (unsigned char *)malloc((size_t)stride * rows);
    for (k = 0; k < stride * rows; ++k) plane_buf[k] = (unsigned char)(rnd() >> 7);
    dst = plane_buf + (size_t)(MARGIN + y0) * stride + MARGIN + x0;
    /* record the inputs before predicting (libvpx predicts in place) */
    put32(f, tx);
    put32(f, mode);
    put32(f, xd.plane[plane].n4_wl);
    put32(f, col);
    put32(f, row);
    put32(f, xd.above_mi != NULL);
    put32(f, xd.left_mi != NULL);
    put32(f, xd.mb_to_left_edge);
    put32(f, xd.mb_to_top_edge);
    put32(f, xd.mb_to_right_edge);
    put32(f, xd.mb_to_bottom_edge);
    put32(f, ssx * (plane != 0));
    put32(f, ssy * (plane != 0));
    put32(f, plane ? buf.uv_width : buf.y_width);
    put32(f, plane ? buf.uv_height : buf.y_height);
    put32(f, plane);
    fwrite(dst - stride - 1, 1, 2 * bs + 1, f);
    for (r = 0; r < bs; ++r) fputc(dst[r * stride - 1], f);
    vp9_predict_intra_block(&xd, xd.plane[plane].n4_wl, (TX_SIZE)tx, (PREDICTION_MODE)mode, dst, stride, dst, stride,
                            col, row, plane);
    for (r = 0; r < bs; ++r)
      for (c = 0; c < bs; ++c) fputc(dst[r * stride + c], f);
    free(plane_buf);
  }
  fclose(f);
  printf("%d cases\n", count);
  return 0;
}
