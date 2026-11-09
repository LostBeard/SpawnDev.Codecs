/* inter_ref.c - libvpx v1.17.0 (BSD-3-Clause) VP9 decoder inter prediction as the reference for SpawnDev.Codecs'
 * Vp9InterPred (SpawnDev.Codecs/Video/Vp9/Decoder/Vp9InterPredict.cs). Built and run by make_inter_fixture.py, which
 * writes the parameter file this reads and gzips the binary it writes into TestData/Vp9Dsp/inter.gz.
 *
 * The prediction itself is libvpx's own code: build_mc_border, extend_and_predict, dec_build_inter_predictors,
 * dec_build_inter_predictors_sb and set_plane_n4 are static in vp9/decoder/vp9_decodeframe.c, so
 * make_inter_fixture.py copies their text verbatim from the libvpx source into inter_dec_copy.h (included below);
 * everything they call (vp9_setup_scale_factors_for_frame, vp9_setup_pre_planes, average_split_mvs,
 * clamp_mv_to_umv_border_sb, the vpx_convolve*_c / vpx_scaled_*_c functions, vpx_realloc_frame_buffer,
 * vpx_extend_frame_borders) is linked from the compiled libvpx objects.
 *
 * Reference frames: vpx_realloc_frame_buffer with the decoder's border (VP9_DEC_BORDER_IN_PIXELS); the crop
 * rectangle of every plane is filled from an LCG the C# test repeats (seed, kind). Every case is predicted twice:
 *   run A: the whole allocation (border, the 8-aligned area beyond the crop size) and the decoder's mc_buf hold
 *          garbage, which is what the decoder has (it never extends a reference frame's border);
 *   run B: after vpx_extend_frame_borders (border = the crop edge replicated) and other mc_buf garbage.
 * Run B is recorded (the borderless port clamps reads to the crop rectangle, so it reproduces B on the whole block).
 * The harness fails unless A == B on every pixel inside the visible frame, which proves libvpx's visible output never
 * depends on what lies outside the crop rectangle; it reports how many blocks differ outside the visible frame.
 *
 * Usage: inter_ref <params.txt> <out.bin>
 * params: nScenarios nCases, then per scenario "ssx ssy curW curH  w0 h0 seed0 kind0  w1 h1 seed1 kind1",
 * then per case "scenario miRow miCol bsize filter compound  mv0 mv1  bmi0[0..3]  bmi1[0..3]" (int_mv.as_int values).
 * out (little-endian int32 unless noted): "VP9I", version 1, nScenarios, nCases; per scenario ssx ssy curW curH and per
 * reference w h seed kind x_scale_fp y_scale_fp x_step_q4 y_step_q4; per case the 16 parameter ints, then the
 * predicted planes 0, 1, 2 as bytes (n4_w * 4 x n4_h * 4 each, row-major).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "./vpx_scale_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_scale/vpx_scale.h"
#include "vpx_scale/yv12config.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_common_data.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_scale.h"
#include "vp9/decoder/vp9_decoder.h"

/* patched_vp9_decodeframe.obj (the oracle decoder's instrumented copy) references the hook libvpx_ref.c defines */
struct VP9Decoder;
void (*vp9ref_hook)(struct VP9Decoder *pbi, int stage) = 0;

#include "inter_dec_copy.h"

typedef struct {
  int ssx, ssy, cur_w, cur_h;
  int w[2], h[2], seed[2], kind[2];
} Scenario;

typedef struct {
  int p[16]; /* scenario miRow miCol bsize filter compound mv0 mv1 bmi0[4] bmi1[4] */
} Case;

static uint32_t rng;
static int next8(void) {
  rng = rng * 1664525u + 1013904223u;
  return (int)(rng >> 24);
}

/* the crop rectangle of one plane; the C# test repeats this */
static void fill_plane(uint8_t *buf, int stride, int cw, int ch, int kind) {
  int x, y;
  for (y = 0; y < ch; ++y)
    for (x = 0; x < cw; ++x) {
      const int r = next8();
      int v;
      if (kind == 0) {
        v = r;
      } else if (kind == 1) {
        v = ((x * 255) / (cw > 1 ? cw - 1 : 1) + (y * 255) / (ch > 1 ? ch - 1 : 1)) / 2 + (r & 31) - 16;
        v = v < 0 ? 0 : v > 255 ? 255 : v;
      } else {
        v = (r & 1) ? 255 : 0;
      }
      buf[y * stride + x] = (uint8_t)v;
    }
}

static void fill_frame(YV12_BUFFER_CONFIG *fb, int seed, int kind, uint32_t garbage) {
  size_t i;
  rng = garbage;
  for (i = 0; i < fb->frame_size; ++i) fb->buffer_alloc[i] = (uint8_t)next8();
  rng = (uint32_t)seed;
  fill_plane(fb->y_buffer, fb->y_stride, fb->y_crop_width, fb->y_crop_height, kind);
  fill_plane(fb->u_buffer, fb->uv_stride, fb->uv_crop_width, fb->uv_crop_height, kind);
  fill_plane(fb->v_buffer, fb->uv_stride, fb->uv_crop_width, fb->uv_crop_height, kind);
}

static void fill_garbage(uint8_t *p, size_t n, uint32_t seed) {
  size_t i;
  rng = seed;
  for (i = 0; i < n; ++i) p[i] = (uint8_t)next8();
}

static int log2i(int v) {
  int l = 0;
  while ((1 << l) < v) ++l;
  return l;
}

/* one block, all three planes, as decode_block sets the block up and dec_build_inter_predictors_sb predicts it */
static void predict_case(VP9Decoder *pbi, TileWorkerData *twd, const Scenario *s, const Case *c,
                         uint8_t out[3][64 * 64], uint32_t garbage) {
  VP9_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &twd->xd;
  MODE_INFO mi;
  MODE_INFO *grid[2];
  TileInfo tile;
  const int mi_row = c->p[1], mi_col = c->p[2], bsize = c->p[3];
  const int bw = num_8x8_blocks_wide_lookup[bsize], bh = num_8x8_blocks_high_lookup[bsize];
  int i, plane;

  memset(&mi, 0, sizeof(mi));
  mi.sb_type = (BLOCK_SIZE)bsize;
  mi.interp_filter = (INTERP_FILTER)c->p[4];
  mi.ref_frame[0] = LAST_FRAME;
  mi.ref_frame[1] = c->p[5] ? GOLDEN_FRAME : NO_REF_FRAME;
  mi.mv[0].as_int = (uint32_t)c->p[6];
  mi.mv[1].as_int = (uint32_t)c->p[7];
  for (i = 0; i < 4; ++i) {
    mi.bmi[i].as_mv[0].as_int = (uint32_t)c->p[8 + i];
    mi.bmi[i].as_mv[1].as_int = (uint32_t)c->p[12 + i];
  }
  grid[0] = NULL;
  grid[1] = &mi;
  xd->mi = &grid[1];
  xd->mi_stride = 1;
  memset(&tile, 0, sizeof(tile));
  tile.mi_col_end = cm->mi_cols;
  tile.mi_row_end = cm->mi_rows;

  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    xd->plane[plane].subsampling_x = plane ? s->ssx : 0;
    xd->plane[plane].subsampling_y = plane ? s->ssy : 0;
  }
  /* decode_block: bw = 1 << (bwl - 1) */
  set_plane_n4(xd, bw, bh, log2i(bw) + 1, log2i(bh) + 1);
  set_mi_row_col(xd, &tile, mi_row, bh, mi_col, bw, cm->mi_rows, cm->mi_cols);
  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    fill_garbage(out[plane], 64 * 64, garbage + (uint32_t)plane);
    xd->plane[plane].dst.buf = out[plane];
    xd->plane[plane].dst.stride = 64;
  }
  fill_garbage((uint8_t *)twd->extend_and_predict_buf, sizeof(twd->extend_and_predict_buf), garbage ^ 0x5bd1e995u);
  dec_build_inter_predictors_sb(twd, pbi, xd, mi_row, mi_col);
}

int main(int argc, char **argv) {
  FILE *fin, *fout;
  int ns, nc, i, k, plane, x, y;
  Scenario *sc;
  Case *cs;
  VP9Decoder *pbi;
  TileWorkerData *twd;
  BufferPool *pool;
  static uint8_t out_a[3][64 * 64], out_b[3][64 * 64];
  int invisible_blocks = 0;

  if (argc != 3) {
    fprintf(stderr, "usage: inter_ref <params.txt> <out.bin>\n");
    return 2;
  }
  fin = fopen(argv[1], "r");
  if (!fin || fscanf(fin, "%d %d", &ns, &nc) != 2) return 3;
  sc = (Scenario *)calloc(ns, sizeof(Scenario));
  cs = (Case *)calloc(nc, sizeof(Case));
  for (i = 0; i < ns; ++i) {
    Scenario *s = &sc[i];
    if (fscanf(fin, "%d %d %d %d %d %d %d %d %d %d %d %d", &s->ssx, &s->ssy, &s->cur_w, &s->cur_h, &s->w[0], &s->h[0],
               &s->seed[0], &s->kind[0], &s->w[1], &s->h[1], &s->seed[1], &s->kind[1]) != 12)
      return 4;
  }
  for (i = 0; i < nc; ++i)
    for (k = 0; k < 16; ++k)
      if (fscanf(fin, "%d", &cs[i].p[k]) != 1) return 5;
  fclose(fin);

  pbi = (VP9Decoder *)vpx_memalign(64, sizeof(*pbi));
  twd = (TileWorkerData *)vpx_memalign(64, sizeof(*twd));
  pool = (BufferPool *)vpx_memalign(64, sizeof(*pool));
  memset(pbi, 0, sizeof(*pbi));
  memset(twd, 0, sizeof(*twd));
  memset(pool, 0, sizeof(*pool));
  pbi->common.buffer_pool = pool;

  fout = fopen(argv[2], "wb");
  fwrite("VP9I", 1, 4, fout);
  {
    int hdr[3] = { 1, ns, nc };
    fwrite(hdr, 4, 3, fout);
  }
  for (i = 0; i < ns; ++i) {
    const Scenario *s = &sc[i];
    int hdr[4] = { s->ssx, s->ssy, s->cur_w, s->cur_h };
    fwrite(hdr, 4, 4, fout);
    for (k = 0; k < 2; ++k) {
      struct scale_factors sf;
      int rec[8];
      vp9_setup_scale_factors_for_frame(&sf, s->w[k], s->h[k], s->cur_w, s->cur_h);
      if (!vp9_is_valid_scale(&sf)) {
        fprintf(stderr, "scenario %d ref %d: invalid scale\n", i, k);
        return 6;
      }
      rec[0] = s->w[k]; rec[1] = s->h[k]; rec[2] = s->seed[k]; rec[3] = s->kind[k];
      rec[4] = sf.x_scale_fp; rec[5] = sf.y_scale_fp; rec[6] = sf.x_step_q4; rec[7] = sf.y_step_q4;
      fwrite(rec, 4, 8, fout);
    }
  }

  for (i = 0; i < ns; ++i) {
    const Scenario *s = &sc[i];
    VP9_COMMON *const cm = &pbi->common;
    int first = -1, last = -1, run;
    static uint8_t *saved_a = NULL;
    static size_t saved_cap = 0;
    for (k = 0; k < nc; ++k)
      if (cs[k].p[0] == i) {
        if (first < 0) first = k;
        last = k;
      }
    if (first < 0) continue;
    for (k = first; k <= last; ++k)
      if (cs[k].p[0] != i) {
        fprintf(stderr, "cases of scenario %d are not contiguous\n", i);
        return 7;
      }
    if (saved_cap < (size_t)(last - first + 1) * 3 * 64 * 64) {
      saved_cap = (size_t)(last - first + 1) * 3 * 64 * 64;
      saved_a = (uint8_t *)realloc(saved_a, saved_cap);
    }
    cm->width = s->cur_w;
    cm->height = s->cur_h;
    cm->mi_cols = (s->cur_w + 7) >> 3;
    cm->mi_rows = (s->cur_h + 7) >> 3;
    cm->subsampling_x = s->ssx;
    cm->subsampling_y = s->ssy;
    for (k = 0; k < 2; ++k) {
      YV12_BUFFER_CONFIG *fb = &pool->frame_bufs[k].buf;
      if (vpx_realloc_frame_buffer(fb, s->w[k], s->h[k], s->ssx, s->ssy, VP9_DEC_BORDER_IN_PIXELS,
                                   cm->byte_alignment, NULL, NULL, NULL)) {
        fprintf(stderr, "frame buffer allocation failed\n");
        return 8;
      }
      fill_frame(fb, s->seed[k], s->kind[k], 0x9e3779b9u ^ (uint32_t)(i * 2 + k));
      cm->frame_refs[k].idx = k;
      cm->frame_refs[k].buf = fb;
      vp9_setup_scale_factors_for_frame(&cm->frame_refs[k].sf, s->w[k], s->h[k], s->cur_w, s->cur_h);
    }
    for (run = 0; run < 2; ++run) {
      if (run == 1)
        for (k = 0; k < 2; ++k) vpx_extend_frame_borders(&pool->frame_bufs[k].buf);
      for (k = first; k <= last; ++k) {
        const Case *c = &cs[k];
        uint8_t(*out)[64 * 64] = run == 0 ? out_a : out_b;
        int bsize = c->p[3];
        int n4w, n4h, vis_diff = 0, invis_diff = 0;
        predict_case(pbi, twd, s, c, out, run == 0 ? 0x1234567u + (uint32_t)k : 0x7654321u + (uint32_t)k);
        if (run == 0) {
          memcpy(saved_a + (size_t)(k - first) * 3 * 64 * 64, out_a, sizeof(out_a));
          continue;
        }
        memcpy(out_a, saved_a + (size_t)(k - first) * 3 * 64 * 64, sizeof(out_a));
        fwrite(c->p, 4, 16, fout);
        for (plane = 0; plane < 3; ++plane) {
          const int ssx = plane ? s->ssx : 0, ssy = plane ? s->ssy : 0;
          const int vis_w = plane ? (s->cur_w + ssx) >> ssx : s->cur_w;
          const int vis_h = plane ? (s->cur_h + ssy) >> ssy : s->cur_h;
          const int px0 = (c->p[2] * 8) >> ssx, py0 = (c->p[1] * 8) >> ssy;
          n4w = (num_8x8_blocks_wide_lookup[bsize] << 1) >> ssx;
          n4h = (num_8x8_blocks_high_lookup[bsize] << 1) >> ssy;
          for (y = 0; y < n4h * 4; ++y) {
            fwrite(out_b[plane] + y * 64, 1, n4w * 4, fout);
            for (x = 0; x < n4w * 4; ++x)
              if (out_a[plane][y * 64 + x] != out_b[plane][y * 64 + x]) {
                if (px0 + x < vis_w && py0 + y < vis_h)
                  ++vis_diff;
                else
                  ++invis_diff;
              }
          }
        }
        if (vis_diff) {
          fprintf(stderr, "case %d: %d visible pixels depend on memory outside the crop rectangle\n", k, vis_diff);
          return 9;
        }
        if (invis_diff) ++invisible_blocks;
      }
    }
  }
  fclose(fout);
  printf("%d scenarios, %d cases; %d blocks differ outside the visible frame between the garbage and the extended border\n",
         ns, nc, invisible_blocks);
  return 0;
}
