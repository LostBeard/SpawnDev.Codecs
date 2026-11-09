/* lpf_ref.c: VP9 loop filter reference outputs from libvpx v1.17.0's own C (BSD-3-Clause), for
 * SpawnDev.Codecs' Vp9Lpf port (SpawnDev.Codecs/Video/Vp9/Decoder/Vp9LoopFilter.cs). Built and run by
 * make_lpf_fixture.py, linked against the libvpx objects build_libvpx_ref.py compiled.
 *
 *   lpf_ref edge <count> <out>   vpx_lpf_{horizontal,vertical}_{4,8,16}[_dual]_c on generated pixel neighbourhoods
 *   lpf_ref sb <count> <out>     whole superblocks exactly as libvpx's decoder filters them: one VP9_COMMON kept
 *                                across cases (as across frames), vp9_loop_filter_frame_init when the frame level is
 *                                not 0, vp9_build_mask per decoded block in decode order (decode_block), then
 *                                loop_filter_rows' inner loop: vp9_adjust_mask, vp9_filter_block_plane_ss00 (Y), and
 *                                for U and V ss11 (4:2:0), ss00 (4:4:4) or non420 (4:2:2, 4:4:0).
 *
 * Pixels come from gen_plane() driven by a xorshift32 stream: the C# test regenerates them from the stored seed with
 * the same code and checks the FNV-1a hash stored here, so the fixture holds only the outputs (as out - in, mod 256).
 * All integers are little-endian.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vp9/common/vp9_common_data.h"
#include "vp9/common/vp9_loopfilter.h"
#include "vp9/common/vp9_onyxc_int.h"

/* patched_vp9_decodeframe.obj (build_libvpx_ref.py's instrumented decoder) references this hook */
struct VP9Decoder;
void (*vp9ref_hook)(struct VP9Decoder *pbi, int stage) = 0;

static uint32_t rs;
static void seed(uint32_t s) { rs = s ? s : 1u; }
static uint32_t rnd32(void) {
  rs ^= rs << 13;
  rs ^= rs >> 17;
  rs ^= rs << 5;
  return rs;
}
static int rnd(int n) { return (int)(rnd32() % (uint32_t)n); }

static const int kD[6] = { 0, 1, 3, 8, 24, 80 };
static const int kN[5] = { 0, 0, 1, 2, 5 };

/* cells of 4 or 8 pixels whose levels walk from their left / above neighbours by up to +-d, plus +-n noise per
 * pixel: flat, near-flat, steps and noise, with cell edges on the 8x8 (and 4x4) block edges of the window */
static void gen_plane(uint8_t *p, int w, int h, int stride) {
  int b[32][32];
  const int d = kD[rnd(6)];
  const int n = kN[rnd(5)];
  const int cell = rnd(2) ? 8 : 4;
  const int cw = (w + cell - 1) / cell, ch = (h + cell - 1) / cell;
  int x, y;
  for (y = 0; y < ch; y++)
    for (x = 0; x < cw; x++) {
      int left, up, v;
      if (x)
        left = b[y][x - 1];
      else if (y)
        left = b[y - 1][x];
      else
        left = rnd(256);
      up = y ? b[y - 1][x] : left;
      v = (left + up + 1) >> 1;
      if (d) v += rnd(2 * d + 1) - d;
      b[y][x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  for (y = 0; y < h; y++)
    for (x = 0; x < w; x++) {
      int v = b[y / cell][x / cell];
      if (n) v += rnd(2 * n + 1) - n;
      p[y * stride + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}

static uint32_t fnv(uint32_t h, const uint8_t *p, int w, int hgt, int stride) {
  int x, y;
  for (y = 0; y < hgt; y++)
    for (x = 0; x < w; x++) {
      h ^= p[y * stride + x];
      h *= 16777619u;
    }
  return h;
}

static FILE *out;
static void w8(int v) { fputc(v & 255, out); }
static void w32(int v) {
  w8(v);
  w8(v >> 8);
  w8(v >> 16);
  w8(v >> 24);
}
static void w64(uint64_t v) {
  w32((int)(uint32_t)v);
  w32((int)(uint32_t)(v >> 32));
}
static void wdelta(const uint8_t *now, const uint8_t *was, int w, int h, int stride) {
  int x, y;
  for (y = 0; y < h; y++)
    for (x = 0; x < w; x++) w8(now[y * stride + x] - was[y * stride + x]);
}

/* limits as update_sharpness + vp9_loop_filter_init derive them (3 of 4), or any bytes */
static void thresholds(int *blimit, int *limit, int *thresh) {
  if (rnd(4) < 3) {
    const int lvl = rnd(64);
    const int sharp = rnd(8);
    int lim = lvl >> ((sharp > 0) + (sharp > 4));
    if (sharp > 0 && lim > 9 - sharp) lim = 9 - sharp;
    if (lim < 1) lim = 1;
    *blimit = 2 * (lvl + 2) + lim;
    *limit = lim;
    *thresh = lvl >> 4;
  } else {
    *blimit = rnd(256);
    *limit = rnd(256);
    *thresh = rnd(256);
  }
}

#define EW 24 /* edge window: 24 x 24, pitch 24, the edge at row 8 / column 8 */
static int edge_cases(int count) {
  int i, changed = 0;
  w32(0x3046504c); /* "LPF0" */
  w32(count);
  for (i = 0; i < count; i++) {
    uint8_t buf[EW * EW], was[EW * EW];
    uint8_t b0, l0, t0, b1, l1, t1;
    int fn, bl[2], li[2], th[2];
    uint32_t pseed;
    uint8_t *s = buf + 8 * EW + 8;
    seed(0x9E3779B9u * (uint32_t)(i + 1));
    fn = rnd(12);
    thresholds(&bl[0], &li[0], &th[0]);
    thresholds(&bl[1], &li[1], &th[1]);
    pseed = rnd32();
    b0 = (uint8_t)bl[0], l0 = (uint8_t)li[0], t0 = (uint8_t)th[0];
    b1 = (uint8_t)bl[1], l1 = (uint8_t)li[1], t1 = (uint8_t)th[1];
    seed(pseed);
    gen_plane(buf, EW, EW, EW);
    memcpy(was, buf, sizeof(buf));
    switch (fn) {
      case 0: vpx_lpf_horizontal_4_c(s, EW, &b0, &l0, &t0); break;
      case 1: vpx_lpf_horizontal_4_dual_c(s, EW, &b0, &l0, &t0, &b1, &l1, &t1); break;
      case 2: vpx_lpf_vertical_4_c(s, EW, &b0, &l0, &t0); break;
      case 3: vpx_lpf_vertical_4_dual_c(s, EW, &b0, &l0, &t0, &b1, &l1, &t1); break;
      case 4: vpx_lpf_horizontal_8_c(s, EW, &b0, &l0, &t0); break;
      case 5: vpx_lpf_horizontal_8_dual_c(s, EW, &b0, &l0, &t0, &b1, &l1, &t1); break;
      case 6: vpx_lpf_vertical_8_c(s, EW, &b0, &l0, &t0); break;
      case 7: vpx_lpf_vertical_8_dual_c(s, EW, &b0, &l0, &t0, &b1, &l1, &t1); break;
      case 8: vpx_lpf_horizontal_16_c(s, EW, &b0, &l0, &t0); break;
      case 9: vpx_lpf_horizontal_16_dual_c(s, EW, &b0, &l0, &t0); break;
      case 10: vpx_lpf_vertical_16_c(s, EW, &b0, &l0, &t0); break;
      default: vpx_lpf_vertical_16_dual_c(s, EW, &b0, &l0, &t0); break;
    }
    changed += memcmp(was, buf, sizeof(buf)) != 0;
    w32(fn);
    w32(b0), w32(l0), w32(t0), w32(b1), w32(l1), w32(t1);
    w32((int)pseed);
    w32((int)fnv(2166136261u, was, EW, EW, EW));
    wdelta(buf, was, EW, EW, EW);
  }
  fprintf(stderr, "edge: %d cases, %d with pixels changed\n", count, changed);
  return 0;
}

/* ---- superblocks ---- */
static VP9_COMMON *cm;
static MODE_INFO mis[16 * 16];
static MODE_INFO *grid[16 * 16];
static LOOP_FILTER_MASK lfms[4];
static int ssx, ssy;
static int nblocks;
static int blocks[64][9];
static int block_hist[BLOCK_SIZES];

static void add_block(int mi_row, int mi_col, BLOCK_SIZE bsize, int bwl, int bhl) {
  const int bw = 1 << (bwl - 1), bh = 1 << (bhl - 1);
  const int x_mis = VPXMIN(bw, cm->mi_cols - mi_col);
  const int y_mis = VPXMIN(bh, cm->mi_rows - mi_row);
  MODE_INFO *mi = &mis[mi_row * cm->mi_cols + mi_col];
  int x, y, *rec;
  memset(mi, 0, sizeof(*mi));
  mi->sb_type = bsize;
  mi->tx_size = (TX_SIZE)rnd(max_txsize_lookup[bsize] + 1);
  mi->skip = (int8_t)rnd(2);
  mi->segment_id = (int8_t)rnd(8);
  mi->ref_frame[0] = (MV_REFERENCE_FRAME)rnd(4);
  mi->ref_frame[1] = NO_REF_FRAME;
  mi->mode = (PREDICTION_MODE)(mi->ref_frame[0] == INTRA_FRAME ? rnd(INTRA_MODES) : NEARESTMV + rnd(INTER_MODES));
  mi->uv_mode = (PREDICTION_MODE)rnd(INTRA_MODES);
  for (y = 0; y < y_mis; y++)
    for (x = 0; x < x_mis; x++) grid[(mi_row + y) * cm->mi_stride + mi_col + x] = mi;
  rec = blocks[nblocks++];
  rec[0] = mi_row & 7, rec[1] = mi_col & 7, rec[2] = mi->sb_type, rec[3] = mi->mode, rec[4] = mi->uv_mode;
  rec[5] = mi->tx_size, rec[6] = mi->skip, rec[7] = mi->segment_id, rec[8] = mi->ref_frame[0];
  block_hist[bsize]++;
  /* decode_block: "if (cm->lf.filter_level) vp9_build_mask(cm, mi, mi_row, mi_col, bw, bh);" */
  if (cm->lf.filter_level) vp9_build_mask(cm, mi, mi_row, mi_col, bw, bh);
}

static int valid_for_planes(BLOCK_SIZE b) {
  /* decode_block rejects a block >= 8x8 whose chroma block size does not exist ("Invalid block size.") */
  return b < BLOCK_8X8 || ss_size_lookup[b][ssx][ssy] != BLOCK_INVALID;
}

/* the partition walk of decode_partition */
static void gen_partition(int mi_row, int mi_col, BLOCK_SIZE bsize, int n4x4_l2) {
  const int n8x8_l2 = n4x4_l2 - 1;
  const int num_8x8_wh = 1 << n8x8_l2;
  const int hbs = num_8x8_wh >> 1;
  const int has_rows = (mi_row + hbs) < cm->mi_rows;
  const int has_cols = (mi_col + hbs) < cm->mi_cols;
  PARTITION_TYPE p;
  BLOCK_SIZE subsize;
  if (mi_row >= cm->mi_rows || mi_col >= cm->mi_cols) return;
  if (has_rows && has_cols)
    p = (PARTITION_TYPE)rnd(PARTITION_TYPES);
  else if (!has_rows && has_cols)
    p = rnd(2) ? PARTITION_SPLIT : PARTITION_HORZ;
  else if (has_rows && !has_cols)
    p = rnd(2) ? PARTITION_SPLIT : PARTITION_VERT;
  else
    p = PARTITION_SPLIT;
  if (!valid_for_planes(subsize_lookup[p][bsize])) p = PARTITION_SPLIT;
  subsize = subsize_lookup[p][bsize];
  if (!hbs) {
    add_block(mi_row, mi_col, subsize, 1, 1);
  } else {
    switch (p) {
      case PARTITION_NONE: add_block(mi_row, mi_col, subsize, n4x4_l2, n4x4_l2); break;
      case PARTITION_HORZ:
        add_block(mi_row, mi_col, subsize, n4x4_l2, n8x8_l2);
        if (has_rows) add_block(mi_row + hbs, mi_col, subsize, n4x4_l2, n8x8_l2);
        break;
      case PARTITION_VERT:
        add_block(mi_row, mi_col, subsize, n8x8_l2, n4x4_l2);
        if (has_cols) add_block(mi_row, mi_col + hbs, subsize, n8x8_l2, n4x4_l2);
        break;
      default:
        gen_partition(mi_row, mi_col, subsize, n8x8_l2);
        gen_partition(mi_row, mi_col + hbs, subsize, n8x8_l2);
        gen_partition(mi_row + hbs, mi_col, subsize, n8x8_l2);
        gen_partition(mi_row + hbs, mi_col + hbs, subsize, n8x8_l2);
        break;
    }
  }
}

static int delta6(void) { return rnd(2) ? rnd(17) - 8 : rnd(127) - 63; }

static int sb_cases(int count) {
  int i, k, changed = 0, filtered = 0, paths[3] = { 0, 0, 0 }, cut = 0;
  cm = (VP9_COMMON *)calloc(1, sizeof(*cm));
  cm->lf.sharpness_level = 0;
  vp9_loop_filter_init(cm);
  cm->lf.lfm = lfms;
  w32(0x3042534c); /* "LSB0" */
  w32(count);
  /* uv_txsize_lookup, for the test's check of the packed copy */
  for (k = 0; k < BLOCK_SIZES * TX_SIZES * 4; k++) w8(((const TX_SIZE *)uv_txsize_lookup)[k]);
  for (i = 0; i < count; i++) {
    /* windows: 8 pixels left / above the superblock (what its edges read and write) + the superblock */
    int pw[3], ph[3], pitch[3], p;
    uint8_t *buf[3], *was[3];
    struct macroblockd_plane planes[MAX_MB_PLANE];
    int mi_row, mi_col, level, sk, s;
    uint32_t pseed, hash;
    LOOP_FILTER_MASK *lfm;
    seed(0x85EBCA6Bu * (uint32_t)(i + 1) + 0x1234567u);
    k = rnd(20);
    ssx = k < 12 ? 1 : k < 15 ? 0 : k < 18 ? 1 : 0;
    ssy = k < 12 ? 1 : k < 15 ? 0 : k < 18 ? 0 : 1;
    /* 1 or 2 superblocks each way; half the time the last one is cut by the frame edge */
    cm->mi_rows = 8 * (1 + rnd(2));
    if (rnd(2)) cm->mi_rows -= 1 + rnd(7);
    cm->mi_cols = 8 * (1 + rnd(2));
    if (rnd(2)) cm->mi_cols -= 1 + rnd(7);
    cm->mi_stride = cm->mi_cols;
    cm->lf.lfm_stride = (cm->mi_cols + 7) >> 3;
    mi_row = rnd((cm->mi_rows + 7) >> 3) * 8;
    mi_col = rnd((cm->mi_cols + 7) >> 3) * 8;
    cut += mi_row + 8 > cm->mi_rows || mi_col + 8 > cm->mi_cols;
    /* frame header */
    level = rnd(10) == 0 ? 0 : rnd(64);
    if (rnd(3) == 0) cm->lf.sharpness_level = rnd(8);
    cm->lf.filter_level = level;
    cm->lf.mode_ref_delta_enabled = rnd(4) != 0;
    for (k = 0; k < MAX_REF_FRAMES; k++) cm->lf.ref_deltas[k] = (signed char)delta6();
    for (k = 0; k < MAX_MODE_LF_DELTAS; k++) cm->lf.mode_deltas[k] = (signed char)delta6();
    cm->seg.enabled = (uint8_t)rnd(2);
    cm->seg.abs_delta = (uint8_t)rnd(2);
    for (s = 0; s < MAX_SEGMENTS; s++) {
      /* the other features' bits must not matter */
      cm->seg.feature_mask[s] = (uint32_t)rnd(16);
      cm->seg.feature_data[s][SEG_LVL_ALT_LF] = (int16_t)(rnd(127) - 63);
    }
    if (level) vp9_loop_filter_frame_init(cm, level);
    memset(lfms, 0, sizeof(lfms));
    memset(grid, 0, sizeof(grid));
    nblocks = 0;
    gen_partition(mi_row, mi_col, BLOCK_64X64, 4);
    pseed = rnd32();
    seed(pseed);
    hash = 2166136261u;
    for (p = 0; p < 3; p++) {
      const int sx = p ? ssx : 0, sy = p ? ssy : 0;
      pw[p] = 8 + (64 >> sx);
      ph[p] = 8 + (64 >> sy);
      pitch[p] = pw[p] + 8;
      buf[p] = (uint8_t *)calloc(pitch[p] * (ph[p] + 8), 1);
      was[p] = (uint8_t *)calloc(pitch[p] * (ph[p] + 8), 1);
      gen_plane(buf[p], pw[p], ph[p], pitch[p]);
      hash = fnv(hash, buf[p], pw[p], ph[p], pitch[p]);
      memcpy(was[p], buf[p], pitch[p] * (ph[p] + 8));
      memset(&planes[p], 0, sizeof(planes[p]));
      planes[p].subsampling_x = sx;
      planes[p].subsampling_y = sy;
    }
    lfm = get_lfm(&cm->lf, mi_row, mi_col);
    /* vp9_loop_filter_frame: "if (!frame_filter_level) return;" then loop_filter_rows' inner loop */
    if (level) {
      vp9_adjust_mask(cm, mi_row, mi_col, lfm);
      for (p = 0; p < 3; p++) {
        planes[p].dst.buf = buf[p] + 8 * pitch[p] + 8;
        planes[p].dst.stride = pitch[p];
      }
      vp9_filter_block_plane_ss00(cm, &planes[0], mi_row, lfm);
      for (p = 1; p < 3; p++) {
        if (ssx == 1 && ssy == 1)
          vp9_filter_block_plane_ss11(cm, &planes[p], mi_row, lfm);
        else if (ssx == 0 && ssy == 0)
          vp9_filter_block_plane_ss00(cm, &planes[p], mi_row, lfm);
        else
          vp9_filter_block_plane_non420(cm, &planes[p], grid + mi_row * cm->mi_stride + mi_col, mi_row, mi_col);
      }
      filtered++;
      paths[ssx == 1 && ssy == 1 ? 0 : ssx == 0 && ssy == 0 ? 1 : 2]++;
    }
    /* the windows hold every pixel the superblock's filters touch: nothing outside them changed */
    for (p = 0; p < 3; p++) {
      int x, y;
      for (y = 0; y < ph[p] + 8; y++)
        for (x = 0; x < pitch[p]; x++)
          if ((y >= ph[p] || x >= pw[p]) && buf[p][y * pitch[p] + x] != was[p][y * pitch[p] + x]) {
            fprintf(stderr, "case %d plane %d: pixel (%d,%d) outside the window changed\n", i, p, x, y);
            return 1;
          }
    }
    sk = 0;
    for (p = 0; p < 3; p++) sk |= memcmp(buf[p], was[p], pitch[p] * (ph[p] + 8)) != 0;
    changed += sk;
    w32(ssx), w32(ssy), w32(cm->mi_rows), w32(cm->mi_cols), w32(mi_row), w32(mi_col);
    w32((int)pseed), w32((int)hash);
    /* header, in Vp9LfHdr's layout */
    w32(level), w32(cm->lf.sharpness_level), w32(cm->lf.mode_ref_delta_enabled);
    for (k = 0; k < MAX_REF_FRAMES; k++) w32(cm->lf.ref_deltas[k]);
    for (k = 0; k < MAX_MODE_LF_DELTAS; k++) w32(cm->lf.mode_deltas[k]);
    w32(cm->seg.enabled), w32(cm->seg.abs_delta);
    for (s = 0; s < MAX_SEGMENTS; s++) w32((cm->seg.feature_mask[s] >> SEG_LVL_ALT_LF) & 1);
    for (s = 0; s < MAX_SEGMENTS; s++) w32(get_segdata(&cm->seg, s, SEG_LVL_ALT_LF));
    /* lf_info: lvl[seg][ref][mode], then lfthr[lvl] mblim / lim / hev_thr */
    for (k = 0; k < MAX_SEGMENTS * MAX_REF_FRAMES * MAX_MODE_LF_DELTAS; k++)
      w8(((const uint8_t *)cm->lf_info.lvl)[k]);
    for (k = 0; k <= MAX_LOOP_FILTER; k++) {
      w8(cm->lf_info.lfthr[k].mblim[0]);
      w8(cm->lf_info.lfthr[k].lim[0]);
      w8(cm->lf_info.lfthr[k].hev_thr[0]);
    }
    /* the masks loop_filter_rows filters with (after vp9_adjust_mask) */
    for (k = 0; k < TX_SIZES; k++) w64(lfm->left_y[k]);
    for (k = 0; k < TX_SIZES; k++) w64(lfm->above_y[k]);
    w64(lfm->int_4x4_y);
    for (k = 0; k < TX_SIZES; k++) w32(lfm->left_uv[k]);
    for (k = 0; k < TX_SIZES; k++) w32(lfm->above_uv[k]);
    w32(lfm->int_4x4_uv);
    for (k = 0; k < 64; k++) w8(lfm->lfl_y[k]);
    /* the blocks in decode order: row, col in the superblock, sb_type, mode, uv_mode, tx_size, skip, segment, ref */
    w32(nblocks);
    for (k = 0; k < nblocks; k++)
      for (s = 0; s < 9; s++) w8(blocks[k][s]);
    for (p = 0; p < 3; p++) wdelta(buf[p], was[p], pw[p], ph[p], pitch[p]);
    for (p = 0; p < 3; p++) free(buf[p]), free(was[p]);
  }
  fprintf(stderr, "sb: %d cases, %d filtered (420 %d, 444 %d, 422/440 %d), %d with pixels changed, %d cut by the frame edge\n",
          count, filtered, paths[0], paths[1], paths[2], changed, cut);
  fprintf(stderr, "block sizes:");
  for (k = 0; k < BLOCK_SIZES; k++) fprintf(stderr, " %d", block_hist[k]);
  fprintf(stderr, "\n");
  return 0;
}

int main(int argc, char **argv) {
  int r;
  if (argc != 4) {
    fprintf(stderr, "usage: lpf_ref edge|sb <count> <out>\n");
    return 2;
  }
  out = fopen(argv[3], "wb");
  if (!out) return 2;
  r = strcmp(argv[1], "edge") == 0 ? edge_cases(atoi(argv[2])) : sb_cases(atoi(argv[2]));
  fclose(out);
  return r;
}
