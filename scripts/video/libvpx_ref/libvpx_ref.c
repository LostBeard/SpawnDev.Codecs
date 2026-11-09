/*
 * SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
 *
 * Reference VP9 decoder for the VP9 decoder's tests: libvpx (BSD-3-Clause, github.com/webmproject/libvpx) built as
 * plain C (generic target, single thread, 8-bit) by build_libvpx_ref.py. libvpx is NOT in this repository.
 *
 * Usage: libvpx_ref [-h] [-c ctx.bin] [-m] in.ivf [out.i420]
 *   decodes every frame and prints one line per output frame, "<n> <width>x<height> <md5>", where the MD5 is computed
 *   exactly as libvpx's test/md5_helper.h does (each plane row by row over the displayed width; chroma sizes rounded
 *   up), so the lines compare with the conformance vectors' .md5 files. With out.i420 the cropped planes of every
 *   output frame are appended to that file.
 *   -h: packets are split at their superframe index and decoded one frame at a time; after each frame a line
 *   "H <packet> <frame in packet> key=value ..." gives libvpx's parsed header fields (uncompressed + compressed header).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"
#include "md5_utils.h"
#include "vp9/vp9_dx_iface.h"
#include "vp9/decoder/vp9_decoder.h"

/* -c: frame context dumps. Stage 0 = the context the frame loads; stage 1 = after its compressed header, followed by
 * tx_mode, reference_mode, comp_fixed_ref, comp_var_ref[0], comp_var_ref[1] (one byte each). */
#define CTX_BYTES ((int)offsetof(FRAME_CONTEXT, initialized))
static FILE *ctx_out;
static int mi_mode, hooked_frames;
void (*vp9ref_hook)(struct VP9Decoder *pbi, int stage);

static void md5_hex(MD5Context *md5, char out[33]) {
  unsigned char d[16];
  int i;
  MD5Final(d, md5);
  for (i = 0; i < 16; ++i) sprintf(out + 2 * i, "%02x", d[i]);
}

/* -m: per decoded frame "M <n> mi=<md5> counts=<md5|->" after the tiles and "A <n> <md5>" (context after backward
 * adaptation). The mode info MD5 runs over every mi position in raster order, as int32s: sb_type, segment_id, skip,
 * tx_size, is_inter, then for intra blocks mode, uv_mode (+ bmi[0..3].as_mode below 8x8), for inter blocks mode,
 * ref_frame[0], ref_frame[1], interp_filter, mv[0], mv[1] (+ bmi[0..3].as_mv[0], as_mv[1] below 8x8). Only fields the
 * block type defines (libvpx leaves the others stale). counts = FRAME_COUNTS bytes, frames with backward adaptation. */
static void mi_hook(struct VP9Decoder *pbi, int stage) {
  const VP9_COMMON *cm = &pbi->common;
  MD5Context md5;
  char hex[33];
  int r, c, k;
  if (stage == 2) {
    MD5Init(&md5);
    for (r = 0; r < cm->mi_rows; ++r)
      for (c = 0; c < cm->mi_cols; ++c) {
        const MODE_INFO *mi = cm->mi_grid_visible[r * cm->mi_stride + c];
        int v[24], n = 0;
        const int inter = mi->ref_frame[0] > INTRA_FRAME;
        v[n++] = mi->sb_type;
        v[n++] = mi->segment_id;
        v[n++] = mi->skip;
        v[n++] = mi->tx_size;
        v[n++] = inter;
        if (!inter) {
          v[n++] = mi->mode;
          v[n++] = mi->uv_mode;
          if (mi->sb_type < BLOCK_8X8)
            for (k = 0; k < 4; ++k) v[n++] = mi->bmi[k].as_mode;
        } else {
          v[n++] = mi->mode;
          v[n++] = mi->ref_frame[0];
          v[n++] = mi->ref_frame[1];
          v[n++] = mi->interp_filter;
          v[n++] = (int)mi->mv[0].as_int;
          v[n++] = (int)mi->mv[1].as_int;
          if (mi->sb_type < BLOCK_8X8)
            for (k = 0; k < 4; ++k) {
              v[n++] = (int)mi->bmi[k].as_mv[0].as_int;
              v[n++] = (int)mi->bmi[k].as_mv[1].as_int;
            }
        }
        MD5Update(&md5, (const unsigned char *)v, n * 4);
      }
    md5_hex(&md5, hex);
    printf("M %d mi=%s counts=", hooked_frames, hex);
    if (!cm->error_resilient_mode && !cm->frame_parallel_decoding_mode) {
      MD5Init(&md5);
      MD5Update(&md5, (const unsigned char *)&cm->counts, sizeof(cm->counts));
      md5_hex(&md5, hex);
      printf("%s\n", hex);
    } else {
      printf("-\n");
    }
  } else if (stage == 3) {
    MD5Init(&md5);
    MD5Update(&md5, (const unsigned char *)cm->fc, CTX_BYTES - 1);
    md5_hex(&md5, hex);
    printf("A %d %s\n", hooked_frames, hex);
  }
}

static void ctx_hook(struct VP9Decoder *pbi, int stage);
static void hook(struct VP9Decoder *pbi, int stage) {
  if (stage == 0) ++hooked_frames;
  if (ctx_out && stage <= 1) ctx_hook(pbi, stage);
  if (mi_mode && stage >= 2) mi_hook(pbi, stage);
}

static void ctx_hook(struct VP9Decoder *pbi, int stage) {
  const VP9_COMMON *cm = &pbi->common;
  unsigned char extra[5];
  fputc(stage, ctx_out);
  fwrite(cm->fc, 1, CTX_BYTES, ctx_out);
  if (stage == 1) {
    extra[0] = (unsigned char)cm->tx_mode;
    extra[1] = (unsigned char)cm->reference_mode;
    extra[2] = (unsigned char)cm->comp_fixed_ref;
    extra[3] = (unsigned char)cm->comp_var_ref[0];
    extra[4] = (unsigned char)cm->comp_var_ref[1];
    fwrite(extra, 1, 5, ctx_out);
  }
}

static void print_header(vpx_codec_ctx_t *codec, int packet, int sub) {
  const VP9Decoder *pbi = ((vpx_codec_alg_priv_t *)codec->priv)->pbi;
  const VP9_COMMON *cm = &pbi->common;
  const struct loopfilter *lf = &cm->lf;
  const struct segmentation *seg = &cm->seg;
  int i, j;
  printf("H %d %d show_existing=%d", packet, sub, cm->show_existing_frame);
  if (cm->show_existing_frame) {
    printf("\n");
    return;
  }
  printf(" profile=%d bit_depth=%d frame_type=%d show=%d intra_only=%d error_res=%d size=%dx%d render=%dx%d ss=%d,%d"
         " reset_ctx=%d refresh=0x%02x sign_bias=%d,%d,%d allow_hp=%d interp=%d refresh_ctx=%d frame_parallel=%d"
         " ctx_idx=%u lf_level=%d sharpness=%d lf_delta_enabled=%d ref_deltas=%d,%d,%d,%d mode_deltas=%d,%d"
         " base_q=%d dq=%d,%d,%d seg=%d seg_update_map=%d seg_temporal=%d seg_abs=%d tile_log2=%d,%d"
         " tx_mode=%d ref_mode=%d comp_fixed=%d comp_var=%d,%d mi=%dx%d",
         cm->profile, cm->bit_depth, cm->frame_type, cm->show_frame, cm->intra_only, cm->error_resilient_mode,
         cm->width, cm->height, cm->render_width, cm->render_height, cm->subsampling_x, cm->subsampling_y,
         cm->reset_frame_context, pbi->refresh_frame_flags, cm->ref_frame_sign_bias[1], cm->ref_frame_sign_bias[2],
         cm->ref_frame_sign_bias[3], cm->allow_high_precision_mv, cm->interp_filter, cm->refresh_frame_context,
         cm->frame_parallel_decoding_mode, cm->frame_context_idx, lf->filter_level, lf->sharpness_level,
         lf->mode_ref_delta_enabled, lf->ref_deltas[0], lf->ref_deltas[1], lf->ref_deltas[2], lf->ref_deltas[3],
         lf->mode_deltas[0], lf->mode_deltas[1], cm->base_qindex, cm->y_dc_delta_q, cm->uv_dc_delta_q,
         cm->uv_ac_delta_q, seg->enabled, seg->update_map, seg->temporal_update, seg->abs_delta, cm->log2_tile_cols,
         cm->log2_tile_rows, cm->tx_mode, cm->reference_mode, cm->comp_fixed_ref, cm->comp_var_ref[0],
         cm->comp_var_ref[1], cm->mi_cols, cm->mi_rows);
  printf(" use_prev_mvs=%d lossless=%d", cm->use_prev_frame_mvs, pbi->mb.lossless);
  if (!frame_is_intra_only(cm)) {
    printf(" ref_sizes=");
    for (i = 0; i < REFS_PER_FRAME; ++i)
      printf(i ? ",%dx%d" : "%dx%d", cm->frame_refs[i].buf->y_crop_width, cm->frame_refs[i].buf->y_crop_height);
  }
  printf(" dequant=");
  for (i = 0; i < (seg->enabled ? MAX_SEGMENTS : 1); ++i)
    printf(i ? ";%d,%d,%d,%d" : "%d,%d,%d,%d", cm->y_dequant[i][0], cm->y_dequant[i][1], cm->uv_dequant[i][0],
           cm->uv_dequant[i][1]);
  if (seg->enabled) {
    printf(" seg_tree=");
    for (i = 0; i < SEG_TREE_PROBS; ++i) printf(i ? ",%d" : "%d", seg->tree_probs[i]);
    printf(" seg_pred=");
    for (i = 0; i < PREDICTION_PROBS; ++i) printf(i ? ",%d" : "%d", seg->pred_probs[i]);
    printf(" seg_features=");
    for (i = 0; i < MAX_SEGMENTS; ++i) {
      printf(i ? ";%x" : "%x", seg->feature_mask[i]);
      for (j = 0; j < SEG_LVL_MAX; ++j) printf(",%d", seg->feature_data[i][j]);
    }
  }
  printf("\n");
}

static unsigned int rd32(const unsigned char *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned)p[3] << 24); }

int main(int argc, char **argv) {
  FILE *in, *out = NULL;
  unsigned char hdr[32], fh[12];
  unsigned char *buf = NULL;
  size_t cap = 0;
  vpx_codec_ctx_t codec;
  vpx_codec_dec_cfg_t cfg;
  int n = 0, plane, y, headers = 0, packet = 0;
  for (;;) {
    if (argc > 1 && strcmp(argv[1], "-h") == 0) {
      headers = 1;
      ++argv;
      --argc;
    } else if (argc > 2 && strcmp(argv[1], "-c") == 0) {
      if (!(ctx_out = fopen(argv[2], "wb"))) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
      }
      vp9ref_hook = hook;
      fprintf(stderr, "ctx_bytes=%d coef=%d switchable=%d inter_mode=%d tx=%d skip=%d nmvc=%d\n", CTX_BYTES,
              (int)offsetof(FRAME_CONTEXT, coef_probs), (int)offsetof(FRAME_CONTEXT, switchable_interp_prob),
              (int)offsetof(FRAME_CONTEXT, inter_mode_probs), (int)offsetof(FRAME_CONTEXT, tx_probs),
              (int)offsetof(FRAME_CONTEXT, skip_probs), (int)offsetof(FRAME_CONTEXT, nmvc));
      argv += 2;
      argc -= 2;
    } else if (argc > 1 && strcmp(argv[1], "-m") == 0) {
      mi_mode = 1;
      vp9ref_hook = hook;
      ++argv;
      --argc;
    } else {
      break;
    }
  }
  if (argc < 2) {
    fprintf(stderr, "usage: libvpx_ref [-h] in.ivf [out.i420]\n");
    return 2;
  }
  in = fopen(argv[1], "rb");
  if (!in || fread(hdr, 1, 32, in) != 32 || memcmp(hdr, "DKIF", 4) != 0) {
    fprintf(stderr, "not an IVF file: %s\n", argv[1]);
    return 2;
  }
  if (argc > 2 && !(out = fopen(argv[2], "wb"))) {
    fprintf(stderr, "cannot write %s\n", argv[2]);
    return 2;
  }
  memset(&cfg, 0, sizeof(cfg));
  cfg.threads = 1;
  if (vpx_codec_dec_init(&codec, vpx_codec_vp9_dx(), &cfg, 0)) {
    fprintf(stderr, "init failed: %s\n", vpx_codec_error(&codec));
    return 1;
  }
  while (fread(fh, 1, 12, in) == 12) {
    size_t size = rd32(fh);
    vpx_codec_iter_t iter = NULL;
    vpx_image_t *img;
    if (size > cap) {
      cap = size * 2;
      buf = (unsigned char *)realloc(buf, cap);
    }
    if (fread(buf, 1, size, in) != size) {
      fprintf(stderr, "truncated frame\n");
      return 1;
    }
    ++packet;
    if (headers) {
      /* one frame at a time (the decoder does the same split internally), so every frame's header can be read */
      uint32_t sizes[8];
      int count = 0, k;
      size_t at = 0;
      if (vp9_parse_superframe_index(buf, size, sizes, &count, NULL, NULL) != VPX_CODEC_OK) {
        fprintf(stderr, "bad superframe index\n");
        return 1;
      }
      if (count == 0) {
        sizes[0] = (uint32_t)size;
        count = 1;
      }
      for (k = 0; k < count; ++k) {
        if (vpx_codec_decode(&codec, buf + at, sizes[k], NULL, 0)) {
          fprintf(stderr, "decode failed: %s (%s)\n", vpx_codec_error(&codec), vpx_codec_error_detail(&codec));
          return 1;
        }
        print_header(&codec, packet, k);
        at += sizes[k];
      }
    } else if (vpx_codec_decode(&codec, buf, (unsigned int)size, NULL, 0)) {
      fprintf(stderr, "decode failed: %s (%s)\n", vpx_codec_error(&codec), vpx_codec_error_detail(&codec));
      return 1;
    }
    while ((img = vpx_codec_get_frame(&codec, &iter)) != NULL) {
      MD5Context md5;
      unsigned char digest[16];
      int i;
      MD5Init(&md5);
      for (plane = 0; plane < 3; ++plane) {
        const unsigned char *p = img->planes[plane];
        const int bps = (img->fmt & VPX_IMG_FMT_HIGHBITDEPTH) ? 2 : 1;
        const int h = plane ? (int)((img->d_h + img->y_chroma_shift) >> img->y_chroma_shift) : (int)img->d_h;
        const int w = (plane ? (int)((img->d_w + img->x_chroma_shift) >> img->x_chroma_shift) : (int)img->d_w) * bps;
        for (y = 0; y < h; ++y) {
          MD5Update(&md5, p, w);
          if (out) fwrite(p, 1, w, out);
          p += img->stride[plane];
        }
      }
      MD5Final(digest, &md5);
      printf("%d %ux%u ", ++n, img->d_w, img->d_h);
      for (i = 0; i < 16; ++i) printf("%02x", digest[i]);
      printf("\n");
    }
  }
  vpx_codec_destroy(&codec);
  if (out) fclose(out);
  free(buf);
  return 0;
}
