/*
 * SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
 *
 * Reference VP9 decoder for the VP9 decoder's tests: libvpx (BSD-3-Clause, github.com/webmproject/libvpx) built as
 * plain C (generic target, single thread, 8-bit) by build_libvpx_ref.py. libvpx is NOT in this repository.
 *
 * Usage: libvpx_ref in.ivf [out.i420]
 *   decodes every frame and prints one line per output frame, "<n> <width>x<height> <md5>", where the MD5 is computed
 *   exactly as libvpx's test/md5_helper.h does (each plane row by row over the displayed width; chroma sizes rounded
 *   up), so the lines compare with the conformance vectors' .md5 files. With out.i420 the cropped planes of every
 *   output frame are appended to that file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"
#include "md5_utils.h"

static unsigned int rd32(const unsigned char *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned)p[3] << 24); }

int main(int argc, char **argv) {
  FILE *in, *out = NULL;
  unsigned char hdr[32], fh[12];
  unsigned char *buf = NULL;
  size_t cap = 0;
  vpx_codec_ctx_t codec;
  vpx_codec_dec_cfg_t cfg;
  int n = 0, plane, y;
  if (argc < 2) {
    fprintf(stderr, "usage: libvpx_ref in.ivf [out.i420]\n");
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
    if (vpx_codec_decode(&codec, buf, (unsigned int)size, NULL, 0)) {
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
