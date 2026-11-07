/* Exact C port of StochasticConvectionSimulator's per-step shuffle.
 *
 * Python (sims/<tree>/smart_buildings/smart_control/simulator/
 * stochastic_convection_simulator.py):
 *
 *   apply_convection(room_dict, temp)        for each room, skipping
 *                                            exterior_space / interior_wall
 *     _shuffle_max_dist(p, v, distance, temp, room_key)
 *       _room_arrays(room_key, v, max_dist, unlimited)    once, cached
 *       _convect_room(cvs, cand_ptr, cand_idx, unlimited, p, temp)   numba
 *
 * `_convect_room` draws one double per control volume, keeps the CV when the
 * draw is <= p, pairs it with a uniformly chosen candidate (drawn only when
 * the candidate list is non-empty), Fisher-Yates over the pairs, then applies
 * the swaps *sequentially* -- a CV may appear in several pairs, so the final
 * field depends on the order the pairs are applied in.
 *
 * The candidate structure is the room's CVs at offsets (d0, d1) with
 * d0^2 + d1^2 <= max_dist^2, iterated d0-major, stored as CSR (ptr over CVs,
 * idx holding positions within the room).
 *
 * The one case Python routes elsewhere is `distance == -1 and p == 1`, which
 * uses CPython's `random.shuffle` on a list of coordinates -- a different
 * generator (CPython's Mersenne Twister through _randbelow). No shipped config
 * sets distance = -1, so the shim leaves that branch to the Python method.
 */

#include "cenv.h"

#include <stdlib.h>
#include <string.h>

#include "stochastic_private.h"

/* Room handles are a flat table (the simulator's shim keeps the int handle in
 * a numpy array, so it has to be an int32/int64, not a pointer). One
 * environment registers one handle per room of its floor plan -- ~30-60 for the
 * training buildings -- so a batch of M environments of one scenario needs
 * M * n_rooms entries: 8192 leaves room for M = 128 on the largest shipping
 * building, and costs 64 kB of pointers. */
#define CENV_CONV_MAX_ROOMS 8192

typedef struct {
  int64_t n;
  int64_t *rows;
  int64_t *cols;
  int64_t *ptr;   /* n + 1 */
  int64_t *idx;   /* nnz */
  int64_t nnz;
  int64_t *left;  /* n, scratch */
  int64_t *right; /* n, scratch */
  int32_t max_dist;
  int32_t unlimited;
} conv_room;

static conv_room *g_rooms[CENV_CONV_MAX_ROOMS];

static uint64_t coord_key(int64_t r, int64_t c) {
  return ((uint64_t)(uint32_t)r << 32) | (uint64_t)(uint32_t)c;
}

/* Open-addressing table of coord_key -> room position; size is a power of two
 * with at least 2n slots. Duplicate coordinates keep the last position, which
 * is what Python's `{cv: i for i, cv in enumerate(v)}` does. */
static uint64_t *hash_keys;
static int64_t *hash_pos;
static int64_t hash_mask;

static int hash_build(const int64_t *rows, const int64_t *cols, int64_t n) {
  int64_t size = 8;
  while (size < 2 * n) size <<= 1;
  hash_keys = (uint64_t *)malloc((size_t)size * sizeof(uint64_t));
  hash_pos = (int64_t *)malloc((size_t)size * sizeof(int64_t));
  if (hash_keys == NULL || hash_pos == NULL) {
    free(hash_keys);
    free(hash_pos);
    hash_keys = NULL;
    hash_pos = NULL;
    return -1;
  }
  for (int64_t i = 0; i < size; i++) hash_pos[i] = -1;
  hash_mask = size - 1;
  for (int64_t i = 0; i < n; i++) {
    uint64_t key = coord_key(rows[i], cols[i]);
    uint64_t slot = (key * 0x9e3779b97f4a7c15ull) >> 17;
    slot &= (uint64_t)hash_mask;
    while (hash_pos[slot] >= 0 && hash_keys[slot] != key) {
      slot = (slot + 1) & (uint64_t)hash_mask;
    }
    hash_keys[slot] = key;
    hash_pos[slot] = i;
  }
  return 0;
}

static void hash_free(void) {
  free(hash_keys);
  free(hash_pos);
  hash_keys = NULL;
  hash_pos = NULL;
}

static int64_t hash_get(int64_t r, int64_t c) {
  uint64_t key = coord_key(r, c);
  uint64_t slot = (key * 0x9e3779b97f4a7c15ull) >> 17;
  slot &= (uint64_t)hash_mask;
  while (hash_pos[slot] >= 0) {
    if (hash_keys[slot] == key) return hash_pos[slot];
    slot = (slot + 1) & (uint64_t)hash_mask;
  }
  return -1;
}

int64_t cenv_conv_room_create(const int64_t *rows, const int64_t *cols,
                              int64_t n, int32_t max_dist, int32_t unlimited) {
  int64_t h;
  conv_room *rm;
  if (rows == NULL || cols == NULL || n < 0 || max_dist < -1) return 0;
  for (h = 1; h < CENV_CONV_MAX_ROOMS; h++) {
    if (g_rooms[h] == NULL) break;
  }
  if (h == CENV_CONV_MAX_ROOMS) return 0;
  rm = (conv_room *)calloc(1, sizeof(conv_room));
  if (rm == NULL) return 0;
  rm->n = n;
  rm->max_dist = max_dist;
  rm->unlimited = unlimited ? 1 : 0;
  if (n > 0) {
    rm->rows = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    rm->cols = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    rm->left = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    rm->right = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (rm->rows == NULL || rm->cols == NULL || rm->left == NULL ||
        rm->right == NULL) {
      free(rm->rows);
      free(rm->cols);
      free(rm->left);
      free(rm->right);
      free(rm);
      return 0;
    }
    memcpy(rm->rows, rows, (size_t)n * sizeof(int64_t));
    memcpy(rm->cols, cols, (size_t)n * sizeof(int64_t));
  }
  if (!rm->unlimited) {
    /* Offsets in the same order as Python's list comprehension. */
    int64_t n_off = 0;
    int64_t side = 2 * (int64_t)max_dist + 1;
    int64_t *off_r = (int64_t *)malloc((size_t)(side * side) * sizeof(int64_t));
    int64_t *off_c = (int64_t *)malloc((size_t)(side * side) * sizeof(int64_t));
    if (off_r == NULL || off_c == NULL) {
      free(off_r);
      free(off_c);
      free(rm->rows);
      free(rm->cols);
      free(rm->left);
      free(rm->right);
      free(rm);
      return 0;
    }
    for (int64_t d0 = -max_dist; d0 <= max_dist; d0++) {
      for (int64_t d1 = -max_dist; d1 <= max_dist; d1++) {
        if (d0 * d0 + d1 * d1 <= (int64_t)max_dist * max_dist) {
          off_r[n_off] = d0;
          off_c[n_off] = d1;
          n_off++;
        }
      }
    }
    rm->ptr = (int64_t *)calloc((size_t)(n + 1), sizeof(int64_t));
    if (rm->ptr == NULL) {
      free(off_r);
      free(off_c);
      free(rm->rows);
      free(rm->cols);
      free(rm->left);
      free(rm->right);
      free(rm);
      return 0;
    }
    int64_t cap = n * n_off + 1;
    rm->idx = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    if (rm->idx == NULL || hash_build(rows, cols, n) != 0) {
      if (rm->idx == NULL) hash_free();
      free(rm->idx);
      free(off_r);
      free(off_c);
      free(rm->ptr);
      free(rm->rows);
      free(rm->cols);
      free(rm->left);
      free(rm->right);
      free(rm);
      return 0;
    }
    int64_t nnz = 0;
    for (int64_t i = 0; i < n; i++) {
      for (int64_t k = 0; k < n_off; k++) {
        int64_t j = hash_get(rows[i] + off_r[k], cols[i] + off_c[k]);
        if (j >= 0) rm->idx[nnz++] = j;
      }
      rm->ptr[i + 1] = nnz;
    }
    rm->nnz = nnz;
    hash_free();
    free(off_r);
    free(off_c);
  }
  g_rooms[h] = rm;
  return h;
}

void cenv_conv_room_destroy(int64_t h) {
  conv_room *rm;
  if (h <= 0 || h >= CENV_CONV_MAX_ROOMS) return;
  rm = g_rooms[h];
  if (rm == NULL) return;
  free(rm->rows);
  free(rm->cols);
  free(rm->ptr);
  free(rm->idx);
  free(rm->left);
  free(rm->right);
  free(rm);
  g_rooms[h] = NULL;
}

int64_t cenv_conv_room_size(int64_t h) {
  if (h <= 0 || h >= CENV_CONV_MAX_ROOMS || g_rooms[h] == NULL) return -1;
  return g_rooms[h]->n;
}

int64_t cenv_conv_room_nnz(int64_t h) {
  if (h <= 0 || h >= CENV_CONV_MAX_ROOMS || g_rooms[h] == NULL) return -1;
  return g_rooms[h]->nnz;
}

/* Copies the candidate CSR out (n + 1 offsets, nnz indices), so the gate can
 * diff it against StochasticConvectionSimulator._room_arrays. */
int32_t cenv_conv_room_csr(int64_t h, int64_t *ptr, int64_t *idx) {
  conv_room *rm;
  if (h <= 0 || h >= CENV_CONV_MAX_ROOMS || (rm = g_rooms[h]) == NULL) return 1;
  if (rm->ptr == NULL || rm->idx == NULL) return 2; /* unlimited: no lists */
  if (ptr == NULL || idx == NULL) return 1;
  memcpy(ptr, rm->ptr, (size_t)(rm->n + 1) * sizeof(int64_t));
  memcpy(idx, rm->idx, (size_t)rm->nnz * sizeof(int64_t));
  return 0;
}

/* _convect_room(cvs, cand_ptr, cand_idx, unlimited, p, temp), numba body. */
int32_t cenv_conv_room_apply(int64_t h, int64_t rng, double *temp,
                             int64_t nrows, int64_t ncols, double p) {
  conv_room *rm;
  cenv_mt *st;
  int64_t m = 0;
  if (h <= 0 || h >= CENV_CONV_MAX_ROOMS || (rm = g_rooms[h]) == NULL) return 1;
  st = cenv__mt_get(rng);
  if (st == NULL || temp == NULL || nrows <= 0 || ncols <= 0) return 1;
  if (rm->n == 0) return 0;

  const int64_t n = rm->n;
  const int64_t *rows = rm->rows;
  const int64_t *cols = rm->cols;
  int64_t *left = rm->left;
  int64_t *right = rm->right;

  for (int64_t i = 0; i < n; i++) {
    if (cenv__mt_next_double(st) > p) continue;
    int64_t j;
    if (rm->unlimited) {
      j = cenv__mt_randint(st, n - 1);
    } else {
      const int64_t start = rm->ptr[i];
      const int64_t end = rm->ptr[i + 1];
      if (end <= start) continue;
      j = rm->idx[start + cenv__mt_randint(st, end - start - 1)];
    }
    left[m] = i;
    right[m] = j;
    m++;
  }

  for (int64_t i = m - 1; i > 0; i--) {
    const int64_t j = cenv__mt_randint(st, i);
    const int64_t tl = left[i];
    left[i] = left[j];
    left[j] = tl;
    const int64_t tr = right[i];
    right[i] = right[j];
    right[j] = tr;
  }

  for (int64_t k = 0; k < m; k++) {
    const int64_t a = left[k];
    const int64_t b = right[k];
    const int64_t ia = rows[a] * ncols + cols[a];
    const int64_t ib = rows[b] * ncols + cols[b];
    const double held = temp[ia];
    temp[ia] = temp[ib];
    temp[ib] = held;
  }
  return 0;
}

int32_t cenv_conv_apply(const int64_t *rooms, int64_t n_rooms, int64_t rng,
                        double *temp, int64_t nrows, int64_t ncols, double p) {
  if (rooms == NULL || n_rooms < 0) return 1;
  for (int64_t r = 0; r < n_rooms; r++) {
    int32_t rc = cenv_conv_room_apply(rooms[r], rng, temp, nrows, ncols, p);
    if (rc != 0) return rc;
  }
  return 0;
}
