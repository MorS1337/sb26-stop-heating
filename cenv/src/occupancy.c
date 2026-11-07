/* Exact C port of RandomizedArrivalDepartureOccupancy.
 *
 * Python (sims/<tree>/smart_buildings/smart_control/simulator/
 * randomized_arrival_departure_occupancy.py):
 *
 *   average_zone_occupancy(zone_id, start_time, end_time)
 *     - lazily creates zone_assignment ZoneOccupants for a new zone (no random
 *       draws, so allocation order does not touch the stream);
 *     - memoises per (start_time, end_time): a *different* key drops the whole
 *       cache, which happens once per simulated interval, and a repeated key
 *       returns the per-zone value without advancing any chain. Per step the
 *       forward window is asked 2-3 times (internal gains, reward, and the
 *       parity mode's second reward read) and the trailing window once, so the
 *       cache is what makes the chain advance exactly once per interval;
 *     - walks t = start_time; while t < end_time; t += step_size, and counts
 *       the occupants whose peek(t) is WORK.
 *
 *   ZoneOccupant.peek(t)
 *     day = local date of t
 *     if not is_work_day(day):            state = AWAY          (no draw)
 *     elif state == AWAY: if arrived(t):  state = WORK
 *     else:               if departed(t): state = AWAY
 *
 *   _occupant_arrived(t):  hour >= 6 and hour <= 12  -> draw < p_arrival
 *   _occupant_departed(t): hour <  13 -> False
 *                          hour >= 19 -> True                  (no draw)
 *                          else       -> draw < p_departure
 *
 * The draws come from one shared np.random.RandomState(seed) (53-bit doubles,
 * cenv__mt_next_double), consumed in zone visit order and then occupant order.
 *
 * Local time is not computed here: the tz database, the DST rules and the
 * holidays package live on the Python side, so the shim hands over, for every
 * step of the episode grid, the local hour and the is_work_day flag (see
 * stochastic.h).
 * That is exact for the same reason it is cheap -- the simulator's clock only
 * ever takes values on that grid.
 */

#include "cenv.h"

#include <stdlib.h>
#include <string.h>

#include "stochastic_private.h"

/* Handles are a flat table because the shim keeps them in numpy int arrays.
 * One environment of a batch owns one occupancy stream, so M environments need
 * M handles: 512 covers the batch's 128-environment ceiling (cenv/batch.py)
 * with room to spare, for 4 kB of pointers. */
#define CENV_OCC_MAX_HANDLES 512
#define CENV_OCC_MAX_ZONES 512

typedef struct {
  int32_t zone_assignment;
  int32_t arr_early, arr_late;
  int32_t dep_early, dep_late;
  double p_arrival, p_departure;
  int64_t step_ns;
  int64_t origin_ns;
  int64_t n_grid;
  int32_t *local_hour;
  uint8_t *work_day;
  int64_t rng;              /* cenv_mt handle */

  int32_t n_zones;
  uint8_t *state;           /* n_zones * zone_assignment: 0 AWAY, 1 WORK */

  int32_t cache_valid;
  int64_t cache_start;
  int64_t cache_end;
  uint8_t *cache_has;
  double *cache_val;
} occ_state;

static occ_state *g_occ[CENV_OCC_MAX_HANDLES];

static int32_t *copy_i32(const int32_t *src, int64_t n) {
  if (n <= 0) return NULL;
  int32_t *dst = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  if (dst != NULL) memcpy(dst, src, (size_t)n * sizeof(int32_t));
  return dst;
}

static uint8_t *copy_u8(const uint8_t *src, int64_t n) {
  if (n <= 0) return NULL;
  uint8_t *dst = (uint8_t *)malloc((size_t)n * sizeof(uint8_t));
  if (dst != NULL) memcpy(dst, src, (size_t)n * sizeof(uint8_t));
  return dst;
}

static int32_t set_grid(occ_state *o, int64_t origin_ns, int64_t n_grid,
                        const int32_t *local_hour, const uint8_t *work_day) {
  if (n_grid <= 0 || local_hour == NULL || work_day == NULL) return 1;
  int32_t *hours = copy_i32(local_hour, n_grid);
  uint8_t *work = copy_u8(work_day, n_grid);
  if (hours == NULL || work == NULL) {
    free(hours);
    free(work);
    return 1;
  }
  free(o->local_hour);
  free(o->work_day);
  o->local_hour = hours;
  o->work_day = work;
  o->origin_ns = origin_ns;
  o->n_grid = n_grid;
  return 0;
}

static void occ_free(occ_state *o) {
  if (o == NULL) return;
  cenv_mt_destroy(o->rng);
  free(o->local_hour);
  free(o->work_day);
  free(o->state);
  free(o->cache_has);
  free(o->cache_val);
  free(o);
}

int64_t cenv_occ_create(const cenv_occ_config *cfg) {
  int64_t h;
  if (cfg == NULL || cfg->zone_assignment < 0 ||
      cfg->zone_assignment > 64 || cfg->step_ns <= 0) {
    return 0;
  }
  for (h = 1; h < CENV_OCC_MAX_HANDLES; h++) {
    if (g_occ[h] == NULL) break;
  }
  if (h == CENV_OCC_MAX_HANDLES) return 0;
  occ_state *o = (occ_state *)calloc(1, sizeof(occ_state));
  if (o == NULL) return 0;
  o->zone_assignment = cfg->zone_assignment;
  o->arr_early = cfg->earliest_arrival_hour;
  o->arr_late = cfg->latest_arrival_hour;
  o->dep_early = cfg->earliest_departure_hour;
  o->dep_late = cfg->latest_departure_hour;
  o->p_arrival = cfg->p_arrival;
  o->p_departure = cfg->p_departure;
  o->step_ns = cfg->step_ns;
  o->n_zones = CENV_OCC_MAX_ZONES;
  o->state = (uint8_t *)calloc((size_t)o->n_zones *
                                   (size_t)(cfg->zone_assignment > 0
                                                ? cfg->zone_assignment
                                                : 1),
                               sizeof(uint8_t));
  o->cache_has = (uint8_t *)calloc((size_t)o->n_zones, sizeof(uint8_t));
  o->cache_val = (double *)calloc((size_t)o->n_zones, sizeof(double));
  o->rng = cenv_mt_create(cfg->seed);
  if (o->state == NULL || o->cache_has == NULL || o->cache_val == NULL ||
      o->rng == 0 ||
      set_grid(o, cfg->origin_ns, cfg->n_grid, cfg->local_hour,
               cfg->work_day) != 0) {
    occ_free(o);
    return 0;
  }
  g_occ[h] = o;
  return h;
}

void cenv_occ_destroy(int64_t h) {
  if (h <= 0 || h >= CENV_OCC_MAX_HANDLES) return;
  occ_free(g_occ[h]);
  g_occ[h] = NULL;
}

int32_t cenv_occ_set_grid(int64_t h, int64_t origin_ns, int64_t n_grid,
                          const int32_t *local_hour, const uint8_t *work_day) {
  occ_state *o;
  if (h <= 0 || h >= CENV_OCC_MAX_HANDLES || (o = g_occ[h]) == NULL) return 1;
  return set_grid(o, origin_ns, n_grid, local_hour, work_day);
}

int64_t cenv_occ_rng(int64_t h) {
  if (h <= 0 || h >= CENV_OCC_MAX_HANDLES || g_occ[h] == NULL) return 0;
  return g_occ[h]->rng;
}

int64_t cenv_occ_draws(int64_t h) {
  cenv_mt *st;
  if (h <= 0 || h >= CENV_OCC_MAX_HANDLES || g_occ[h] == NULL) return -1;
  st = cenv__mt_get(g_occ[h]->rng);
  return st == NULL ? -1 : st->words;
}

/* Grid index of a utc instant, or -1 when it is not on the grid. */
static int64_t grid_index(const occ_state *o, int64_t ns) {
  int64_t delta = ns - o->origin_ns;
  if (delta % o->step_ns != 0) return -1;
  int64_t idx = delta / o->step_ns;
  if (idx < 0 || idx >= o->n_grid) return -1;
  return idx;
}

/* One peek() at grid index `idx` for occupant slot `slot`; returns 1 if WORK. */
static int peek(const occ_state *o, int32_t idx, uint8_t *state) {
  if (!o->work_day[idx]) {
    *state = 0;
    return 0;
  }
  cenv_mt *st = cenv__mt_get(o->rng);
  const int32_t hour = o->local_hour[idx];
  if (*state == 0) {
    if (hour < o->arr_early || hour > o->arr_late) return 0;
    if (cenv__mt_next_double(st) < o->p_arrival) *state = 1;
  } else {
    if (hour < o->dep_early) return 1;
    if (hour >= o->dep_late) {
      *state = 0;
      return 0;
    }
    if (cenv__mt_next_double(st) < o->p_departure) *state = 0;
  }
  return *state;
}

int32_t cenv_occ_average(int64_t h, int32_t zone, int64_t start_ns,
                         int64_t end_ns, double *out) {
  occ_state *o;
  if (h <= 0 || h >= CENV_OCC_MAX_HANDLES || (o = g_occ[h]) == NULL) return 1;
  if (out == NULL) return 1;
  if (zone < 0 || zone >= o->n_zones) return 2;

  const int64_t start_idx = grid_index(o, start_ns);
  if (start_idx < 0) return 3;

  if (!o->cache_valid || o->cache_start != start_ns || o->cache_end != end_ns) {
    o->cache_valid = 1;
    o->cache_start = start_ns;
    o->cache_end = end_ns;
    memset(o->cache_has, 0, (size_t)o->n_zones * sizeof(uint8_t));
  } else if (o->cache_has[zone]) {
    *out = o->cache_val[zone];
    return 0;
  }

  uint8_t *slots = o->state + (size_t)zone * (size_t)(o->zone_assignment > 0
                                                          ? o->zone_assignment
                                                          : 1);
  double total = 0.0;
  int64_t steps = 0;
  int64_t t = start_ns;
  int64_t idx = start_idx;
  while (t < end_ns) {
    double num = 0.0;
    for (int32_t k = 0; k < o->zone_assignment; k++) {
      if (peek(o, idx, &slots[k])) num += 1.0;
    }
    total += num;
    steps++;
    t += o->step_ns;
    idx++;
    if (idx >= o->n_grid) {
      /* Only reachable with a window that runs past the grid; the shim sizes
       * the grid to cover every window it asks about. */
      return 3;
    }
  }

  double value;
  if (steps) {
    value = total / (double)steps;
  } else {
    /* Degenerate interval (end_time <= start_time): one sample, as Python. */
    double num = 0.0;
    for (int32_t k = 0; k < o->zone_assignment; k++) {
      if (peek(o, start_idx, &slots[k])) num += 1.0;
    }
    value = num;
  }
  o->cache_has[zone] = 1;
  o->cache_val[zone] = value;
  *out = value;
  return 0;
}
