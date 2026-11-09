/* VP9 inverse transform fixture: libvpx v1.17.0's own C (BSD-3-Clause) on deterministic pseudo-random blocks.
 *
 * Calls the functions libvpx's decoder calls (vp9/common/vp9_idct.c: vp9_iwht4x4_add, vp9_iht4x4_add, vp9_iht8x8_add,
 * vp9_iht16x16_add, vp9_idct32x32_add), which dispatch through the generic C rtcd to vpx_dsp/inv_txfm.c (8-bit build:
 * tran_low_t int16, tran_high_t int32, no CONFIG_EMULATE_HARDWARE). Every tx size x tx type (32x32: DCT only) plus the
 * lossless 4x4 WHT, at eobs on both sides of every eob threshold of the dispatch, nonzero coefficients only at the first
 * eob positions of the scan the decoder uses (vp9_scan_orders[tx][type]; lossless: the default 4x4 scan), magnitudes
 * from +-1 to the int16 extremes, random destination pixels in a block with stride n + 3 (the pad bytes must survive).
 *
 * Output (little endian): int32 count, then per case: int32 txSize, txType, lossless, eob, stride;
 * int16 coef[n * n] (raster order, as libvpx's dqcoeff); uint8 dstIn[n * stride]; uint8 dstOut[n * stride].
 * Build + run: make_itx_fixture.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"
#include "vp9/common/vp9_idct.h"
#include "vp9/common/vp9_scan.h"

/* libvpx_ref.c's instrumentation hook, referenced by the patched vp9_decodeframe.obj in the same build folder; the
 * harness never decodes, the definition only satisfies that object if the linker pulls it */
struct VP9Decoder;
void (*vp9ref_hook)(struct VP9Decoder *pbi, int stage) = 0;

static unsigned int g_seed = 0x9E3779B9u;
static unsigned int rnd(void) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return g_seed >> 8; /* 24 bits */
}
static int rnd_range(int lo, int hi) { /* inclusive */
  return lo + (int)(rnd() % (unsigned int)(hi - lo + 1));
}

/* one coefficient value of magnitude class m (sign random, never 0) */
static int coef_value(int m, int is_dc) {
  int sign = (rnd() & 1) ? -1 : 1;
  switch (m) {
    case 0: return sign * rnd_range(1, 4);
    case 1: return sign * rnd_range(1, 64);
    case 2: return sign * rnd_range(1, 1024);
    case 3: return sign * rnd_range(1, 8192);
    case 4: { /* the whole int16 range */
      int v = rnd_range(-32768, 32767);
      return v == 0 ? 1 : v;
    }
    case 5: { /* the extremes */
      static const int ext[] = { -32768, 32767, -32767, 32766, -16384, 16384, -24000, 24000 };
      return ext[rnd() % 8];
    }
    default: /* a large DC over small ACs, the common shape after quantization */
      return is_dc ? rnd_range(-32768, 32767) : sign * rnd_range(1, 48);
  }
}

#define MAGNITUDES 7

static FILE *g_out;
static int g_count;

static void put32(int v) { fwrite(&v, 4, 1, g_out); }

static void run_case(int tx, int type, int lossless, int eob, int m) {
  const int n = 4 << tx, stride = n + 3;
  const int16_t *scan = lossless ? vp9_default_scan_orders[0].scan : vp9_scan_orders[tx][type].scan;
  DECLARE_ALIGNED(32, tran_low_t, coef[32 * 32]);
  DECLARE_ALIGNED(32, tran_low_t, work[32 * 32]);
  uint8_t dst_in[32 * 35], dst[32 * 35];
  int16_t c16[32 * 32];
  int i;
  memset(coef, 0, sizeof(coef));
  for (i = 0; i < eob; i++) {
    /* the last coefficient before eob is nonzero (an eob token follows a nonzero token); earlier ones may be zero tokens */
    if (i == eob - 1 || rnd() % 10 >= 3) coef[scan[i]] = (tran_low_t)coef_value(m, scan[i] == 0);
  }
  for (i = 0; i < n * stride; i++) dst_in[i] = (uint8_t)rnd();
  /* runs of dark / bright pixels so the clamps at 0 and 255 are reached with small residuals too */
  if (rnd() % 4 == 0)
    for (i = 0; i < n * stride; i++) dst_in[i] = (uint8_t)((rnd() & 1) ? rnd_range(0, 6) : rnd_range(249, 255));
  memcpy(dst, dst_in, sizeof(dst));
  /* the transforms read their input only; work on a copy so the stored input is exactly what was passed */
  memcpy(work, coef, sizeof(coef));
  if (lossless) {
    vp9_iwht4x4_add(work, dst, stride, eob);
  } else {
    switch (tx) {
      case 0: vp9_iht4x4_add((TX_TYPE)type, work, dst, stride, eob); break;
      case 1: vp9_iht8x8_add((TX_TYPE)type, work, dst, stride, eob); break;
      case 2: vp9_iht16x16_add((TX_TYPE)type, work, dst, stride, eob); break;
      default: vp9_idct32x32_add(work, dst, stride, eob); break;
    }
  }
  put32(tx);
  put32(type);
  put32(lossless);
  put32(eob);
  put32(stride);
  for (i = 0; i < n * n; i++) c16[i] = (int16_t)coef[i];
  fwrite(c16, 2, n * n, g_out);
  fwrite(dst_in, 1, n * stride, g_out);
  fwrite(dst, 1, n * stride, g_out);
  g_count++;
}

/* eobs on both sides of every threshold of vp9_idct*_add's dispatch (1 / 12 / 10, 38 / 34, 135) plus mid values;
 * -1 = a random eob in [1, n*n] */
static const int eobs4[] = { 1, 2, 3, 6, 10, 16, -1 };
static const int eobs8[] = { 1, 2, 6, 12, 13, 30, 63, 64, -1 };
static const int eobs16[] = { 1, 2, 10, 11, 38, 39, 100, 255, 256, -1 };
static const int eobs32[] = { 1, 2, 34, 35, 135, 136, 500, 1023, 1024, -1 };

static void run_set(int tx, int type, int lossless, const int *eobs, int neobs, int reps) {
  int e, m, r;
  for (r = 0; r < reps; r++)
    for (e = 0; e < neobs; e++)
      for (m = 0; m < MAGNITUDES; m++) {
        int eob = eobs[e] < 0 ? rnd_range(1, (4 << tx) * (4 << tx)) : eobs[e];
        run_case(tx, type, lossless, eob, m);
      }
}

int main(int argc, char **argv) {
  int type;
  if (argc != 2) {
    fprintf(stderr, "usage: itx_ref <out.bin>\n");
    return 2;
  }
  g_out = fopen(argv[1], "wb");
  if (!g_out) return 1;
  put32(0); /* count, patched at the end */
  run_set(0, 0, 1, eobs4, 7, 6);
  for (type = 0; type < 4; type++) run_set(0, type, 0, eobs4, 7, 6);
  for (type = 0; type < 4; type++) run_set(1, type, 0, eobs8, 9, 4);
  for (type = 0; type < 4; type++) run_set(2, type, 0, eobs16, 10, 2);
  run_set(3, 0, 0, eobs32, 10, 3);
  fseek(g_out, 0, SEEK_SET);
  put32(g_count);
  fclose(g_out);
  printf("%d cases\n", g_count);
  return 0;
}
