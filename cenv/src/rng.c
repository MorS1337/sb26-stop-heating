/* MT19937 shared by every stochastic component.
 *
 * numpy's legacy RandomState and numba's np.random are the same stream: both
 * seed with Knuth's init_genrand, both temper with genrand_int32, both build a
 * double out of two consecutive words. The only divergence in the simulator is
 * `np.random.randint(n)`, which inside njit is numpy's masked rk_interval.
 *
 * Handle table: one process, one simulator, no threading; a handle is an index
 * into a small array of pointers, 0 meaning "none".
 */

#include "stochastic_private.h"

#include <stdlib.h>
#include <string.h>

#define CENV_MT_N 624
#define CENV_MT_M 397
#define CENV_MT_MATRIX_A 0x9908b0dfu
#define CENV_MT_UPPER_MASK 0x80000000u
#define CENV_MT_LOWER_MASK 0x7fffffffu

/* Same flat-table reason as occupancy.c: an environment of a batch owns two
 * generators (its occupancy chain and its convection draws), so M environments
 * need 2M handles -- 1024 covers M = 128 with slack, for 8 kB of pointers. */
#define CENV_MT_MAX_HANDLES 1024

static cenv_mt *g_mt[CENV_MT_MAX_HANDLES];

cenv_mt *cenv__mt_get(int64_t handle) {
  if (handle <= 0 || handle >= CENV_MT_MAX_HANDLES) return NULL;
  return g_mt[handle];
}

/* Knuth's PRNG as used by the reference MT19937 and by both numpy's
 * mt19937_seed and numba's numba_rnd_init. */
void cenv__mt_init(cenv_mt *st, uint32_t seed) {
  uint32_t pos;
  seed &= 0xffffffffu;
  for (pos = 0; pos < CENV_MT_N; pos++) {
    st->mt[pos] = seed;
    seed = (1812433253u * (seed ^ (seed >> 30)) + pos + 1u) & 0xffffffffu;
  }
  st->index = CENV_MT_N;
  st->words = 0;
}

void cenv__mt_shuffle(cenv_mt *st) {
  int i;
  uint32_t y;
  for (i = 0; i < CENV_MT_N - CENV_MT_M; i++) {
    y = (st->mt[i] & CENV_MT_UPPER_MASK) | (st->mt[i + 1] & CENV_MT_LOWER_MASK);
    st->mt[i] = st->mt[i + CENV_MT_M] ^ (y >> 1) ^
                ((uint32_t)(-(int32_t)(y & 1)) & CENV_MT_MATRIX_A);
  }
  for (; i < CENV_MT_N - 1; i++) {
    y = (st->mt[i] & CENV_MT_UPPER_MASK) | (st->mt[i + 1] & CENV_MT_LOWER_MASK);
    st->mt[i] = st->mt[i + (CENV_MT_M - CENV_MT_N)] ^ (y >> 1) ^
                ((uint32_t)(-(int32_t)(y & 1)) & CENV_MT_MATRIX_A);
  }
  y = (st->mt[CENV_MT_N - 1] & CENV_MT_UPPER_MASK) |
      (st->mt[0] & CENV_MT_LOWER_MASK);
  st->mt[CENV_MT_N - 1] = st->mt[CENV_MT_M - 1] ^ (y >> 1) ^
                          ((uint32_t)(-(int32_t)(y & 1)) & CENV_MT_MATRIX_A);
}

uint32_t cenv__mt_next_uint32(cenv_mt *st) {
  uint32_t y;
  if (st->index >= CENV_MT_N) {
    cenv__mt_shuffle(st);
    st->index = 0;
  }
  y = st->mt[st->index++];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= y >> 18;
  st->words++;
  return y;
}

/* numpy rk_double / numba get_next_double. Dividing by 2^53 and multiplying by
 * 1/2^53 give the same double (the reciprocal is exact), so either form is
 * bit-identical; the numerator is < 2^53 and exact. */
double cenv__mt_next_double(cenv_mt *st) {
  double a = (double)(cenv__mt_next_uint32(st) >> 5);
  double b = (double)(cenv__mt_next_uint32(st) >> 6);
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

/* np.random.randint(max_incl + 1) as numba compiles it:
 *   n == 1                 -> 0, no word consumed
 *   otherwise              -> low nbits = 64 - clz(n - 1) bits of a word,
 *                             rejected while the value is >= n.
 * (numba's _randrange_impl with ty = int64, state = "np", step = 1.) */
int64_t cenv__mt_randint(cenv_mt *st, int64_t max_incl) {
  uint64_t n = (uint64_t)max_incl + 1u;
  if (n == 1u) return 0;
  uint64_t nm1 = n - 1u;
  int nbits = 64;
  while (!(nm1 & ((uint64_t)1 << 63))) {
    nm1 <<= 1;
    nbits--;
  }
  /* nbits in [1, 32] for every realistic room; the > 32 branch is numpy's
   * "high bits first" form and is kept for completeness. */
  if (nbits <= 32) {
    uint32_t mask = 0xffffffffu >> (32 - nbits);
    uint32_t r;
    do {
      r = cenv__mt_next_uint32(st) & mask;
    } while ((uint64_t)r >= n);
    return (int64_t)r;
  }
  uint32_t high_mask = 0xffffffffu >> (32 - (nbits - 32));
  uint64_t value;
  do {
    uint32_t high = cenv__mt_next_uint32(st) & high_mask;
    uint32_t low = cenv__mt_next_uint32(st);
    value = (uint64_t)low + ((uint64_t)high << 32);
  } while (value >= n);
  return (int64_t)value;
}

/* ---- handles ------------------------------------------------------------ */

int64_t cenv_mt_create(uint32_t seed) {
  int64_t h;
  for (h = 1; h < CENV_MT_MAX_HANDLES; h++) {
    if (g_mt[h] == NULL) {
      cenv_mt *st = (cenv_mt *)calloc(1, sizeof(cenv_mt));
      if (st == NULL) return 0;
      cenv__mt_init(st, seed);
      g_mt[h] = st;
      return h;
    }
  }
  return 0;
}

void cenv_mt_destroy(int64_t h) {
  if (h <= 0 || h >= CENV_MT_MAX_HANDLES) return;
  free(g_mt[h]);
  g_mt[h] = NULL;
}

int32_t cenv_mt_seed(int64_t h, uint32_t seed) {
  cenv_mt *st = cenv__mt_get(h);
  if (st == NULL) return 1;
  cenv__mt_init(st, seed);
  return 0;
}

int32_t cenv_mt_set_state(int64_t h, const uint32_t *mt, int32_t index) {
  cenv_mt *st = cenv__mt_get(h);
  if (st == NULL || mt == NULL) return 1;
  if (index < 0 || index > CENV_MT_N) return 1;
  for (int i = 0; i < CENV_MT_N; i++) st->mt[i] = mt[i];
  st->index = index;
  st->words = 0;
  return 0;
}

int32_t cenv_mt_get_state(int64_t h, uint32_t *mt, int32_t *index) {
  cenv_mt *st = cenv__mt_get(h);
  if (st == NULL || mt == NULL || index == NULL) return 1;
  for (int i = 0; i < CENV_MT_N; i++) mt[i] = st->mt[i];
  *index = st->index;
  return 0;
}

int32_t cenv_mt_fill_double(int64_t h, double *out, int64_t n) {
  cenv_mt *st = cenv__mt_get(h);
  if (st == NULL || (out == NULL && n > 0) || n < 0) return 1;
  for (int64_t i = 0; i < n; i++) out[i] = cenv__mt_next_double(st);
  return 0;
}

int32_t cenv_mt_fill_uint32(int64_t h, uint32_t *out, int64_t n) {
  cenv_mt *st = cenv__mt_get(h);
  if (st == NULL || (out == NULL && n > 0) || n < 0) return 1;
  for (int64_t i = 0; i < n; i++) out[i] = cenv__mt_next_uint32(st);
  return 0;
}

int32_t cenv_mt_fill_randint(int64_t h, int64_t max_incl, int64_t n,
                             int64_t *out) {
  cenv_mt *st = cenv__mt_get(h);
  if (st == NULL || (out == NULL && n > 0) || n < 0 || max_incl < 0) return 1;
  for (int64_t i = 0; i < n; i++) out[i] = cenv__mt_randint(st, max_incl);
  return 0;
}
