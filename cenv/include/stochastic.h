/* Stochastic components: MT19937, convection, occupancy, weather replay.
 *
 * Everything here is a bit-exact C replacement for a Python function the Smart
 * Buildings simulator calls every step. Each entry point names the Python it
 * reproduces; the Python shim (cenv/stochastic.py) monkeypatches the class that
 * holds it and the shadow gate (cenv/verify/stochastic_gate.py) compares the
 * two on recorded episodes.
 *
 *   1. MT19937 -- numpy legacy RandomState and numba's np.random are the same
 *      generator. Both seed with Knuth's init_genrand
 *      (1812433253 * (x ^ (x >> 30)) + pos + 1), numpy in
 *      `mt19937_seed` (numpy/random/src/mt19937/mt19937.c), numba in
 *      `numba_rnd_init` (numba/_random.c). Both temper with the reference
 *      genrand_int32. Both draw a double as
 *      ((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0 with two
 *      consecutive words, so a given seed gives one stream, not two.
 *
 *      `np.random.randint(n)` inside an njit function is the *legacy* path:
 *      typing dispatches to `_randrange_impl` (numba/cpython/randomimpl.py),
 *      which is numpy's rk_interval -- low `nbits = 64 - clz(n - 1)` bits of a
 *      word, rejected while >= n -- and which draws nothing at all when
 *      n == 1. (Not the Lemire path the Generator methods use; those are a
 *      different function.) Implemented exactly so the stream stays aligned.
 *
 *   2. Convection -- stochastic_convection_simulator._convect_room:
 *      one double per CV, `> p` skips, a Lemire integer picks the partner in
 *      the room's CSR candidate list, then a Fisher-Yates pass over the pairs
 *      (numba's shuffle loop) and the sequential swaps.
 *
 *   3. Occupancy -- RandomizedArrivalDepartureOccupancy.average_zone_occupancy
 *      and ZoneOccupant.peek, tick by tick over [start, end) in step_size
 *      increments, with the two cache rules (a new (start, end) key drops the
 *      whole cache; a repeat key returns the memoised per-zone value).
 *
 *      Local time is the one thing C cannot compute alone (tz database, DST,
 *      holidays), so the shim precomputes, on the episode's own step grid, the
 *      local hour and local date of every step (vectorised in pandas, once per
 *      episode) and the set of US holiday dates. Everything else is here.
 *
 *   4. Weather -- ReplayWeatherController.read_weather_csv (CSV, UTC epoch
 *      seconds) plus _get_interpolated_value, which is np.interp on the
 *      precomputed arrays, plus fahrenheit_to_kelvin.
 *
 * All state lives behind int64 handles in a process-local table, matching the
 * Python objects' lifetimes (one per process, no threading).
 */

#ifndef CENV_STOCHASTIC_H
#define CENV_STOCHASTIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 1. MT19937 --------------------------------------------------------- */

/* A fresh generator seeded exactly like numpy RandomState(seed) and numba's
 * np.random.seed(seed). Returns a handle > 0. */
int64_t cenv_mt_create(uint32_t seed);

/* Releases a handle (no-op for unknown handles). */
void cenv_mt_destroy(int64_t h);

/* Re-seeds in place; equivalent to creating a new one with the same handle. */
int32_t cenv_mt_seed(int64_t h, uint32_t seed);

/* Imports/exports the raw generator state (624 words + block index), which is
 * exactly numpy's RandomState.get_state()["state"][1:3] and numba's
 * _helperlib.rnd_get_state(ptr). The shim uses it so a C generator continues
 * an existing Python stream instead of restarting it -- and so the gate can
 * prove both sides consumed the same words. */
int32_t cenv_mt_set_state(int64_t h, const uint32_t *mt, int32_t index);
int32_t cenv_mt_get_state(int64_t h, uint32_t *mt, int32_t *index);

/* n consecutive 53-bit doubles, i.e. RandomState.random_sample() / numba
 * np.random.random(). */
int32_t cenv_mt_fill_double(int64_t h, double *out, int64_t n);

/* n consecutive raw genrand_int32 words (RandomState.tomaxint is not this;
 * this is the underlying uint32 stream, exposed for the stream proof). */
int32_t cenv_mt_fill_uint32(int64_t h, uint32_t *out, int64_t n);

/* n draws of np.random.randint(max_incl + 1) under numba's Lemire path:
 * values in [0, max_incl], and no word consumed at all when max_incl == 0. */
int32_t cenv_mt_fill_randint(int64_t h, int64_t max_incl, int64_t n,
                             int64_t *out);

/* ---- 2. Convection ------------------------------------------------------ */

/* Builds the CSR candidate structure for one room, exactly as
 * StochasticConvectionSimulator._room_arrays does it: candidates are the CMVs
 * of the room at offsets (d0, d1) with d0^2 + d1^2 <= max_dist^2, iterated
 * d0-major then d1, looked up by coordinate. `unlimited` (max_dist == -1)
 * stores no candidate list: the kernel then draws directly in [0, n).
 *
 *   rows, cols  [n] the room's control-volume coordinates, in room order
 *
 * Returns a handle > 0, or 0 on allocation failure / bad input.
 */
int64_t cenv_conv_room_create(const int64_t *rows, const int64_t *cols,
                              int64_t n, int32_t max_dist, int32_t unlimited);

void cenv_conv_room_destroy(int64_t room);

/* Number of CVs in a room, or -1 for an unknown handle. */
int64_t cenv_conv_room_size(int64_t room);

/* The room's candidate CSR (`_room_arrays`' cand_ptr/cand_idx), for the gate:
 * `nnz` candidates in total, `ptr` holding n + 1 offsets. 2 when the room is
 * `unlimited` and has no candidate lists. */
int64_t cenv_conv_room_nnz(int64_t room);
int32_t cenv_conv_room_csr(int64_t room, int64_t *ptr, int64_t *idx);

/* One room's `_convect_room` call: draws, pairs, Fisher-Yates, swap, with
 * `temp` a C-contiguous (nrows, ncols) float64 grid modified in place.
 * `rng` is a cenv_mt handle. Returns 0 on success. */
int32_t cenv_conv_room_apply(int64_t room, int64_t rng, double *temp,
                             int64_t nrows, int64_t ncols, double p);

/* The whole `apply_convection` loop: applies every room in `rooms` (the
 * room_dict iteration order already filtered of exterior_space/interior_wall)
 * with the shared generator. Returns 0 on success. */
int32_t cenv_conv_apply(const int64_t *rooms, int64_t n_rooms, int64_t rng,
                        double *temp, int64_t nrows, int64_t ncols, double p);

/* ---- 3. Occupancy ------------------------------------------------------- */

/* Builds one RandomizedArrivalDepartureOccupancy. Probabilities are computed
 * by the shim exactly as ZoneOccupant._get_event_probability does (a pandas
 * Timedelta division), and the local-time grid is the episode's own step grid:
 * grid entry i is the local wall clock at utc instant origin_ns + i*step_ns.
 *
 *   local_hour  [n_grid] local hour, 0-23
 *   work_day    [n_grid] 1 when conversion_utils.is_work_day(local date) is
 *                        true -- computed by the shim, so the tz database,
 *                        DST rules and the holidays package stay in Python
 */
typedef struct cenv_occ_config {
  int32_t zone_assignment;        /* occupants per zone (work_occupancy) */
  int32_t earliest_arrival_hour;
  int32_t latest_arrival_hour;
  int32_t earliest_departure_hour;
  int32_t latest_departure_hour;
  double p_arrival;               /* 1 / n_halfway, as ZoneOccupant computes it */
  double p_departure;
  int64_t step_ns;                /* time_step_sec in nanoseconds */
  int64_t origin_ns;              /* utc instant of grid entry 0 */
  int64_t n_grid;
  const int32_t *local_hour;
  const uint8_t *work_day;
  uint32_t seed;                  /* RandomizedArrivalDepartureOccupancy.seed */
} cenv_occ_config;

int64_t cenv_occ_create(const cenv_occ_config *cfg);

void cenv_occ_destroy(int64_t h);

/* Replaces the local-time grid (used when the shim extends its horizon).
 * Occupant state and the generator are untouched, exactly like the Python
 * object, whose chain survives a later query. */
int32_t cenv_occ_set_grid(int64_t h, int64_t origin_ns, int64_t n_grid,
                          const int32_t *local_hour, const uint8_t *work_day);

/* `average_zone_occupancy(zone_id, start_time, end_time)` for zone index
 * `zone` (the shim's stable order of first query). Times are utc nanoseconds.
 *
 * Returns 0 and writes `out`, or
 *   1 unknown handle, 2 zone index out of range,
 *   3 start/end not on the local-time grid (shim must extend it).
 */
int32_t cenv_occ_average(int64_t h, int32_t zone, int64_t start_ns,
                         int64_t end_ns, double *out);

/* The handle's generator, so the shim can import the state of the Python
 * RandomState the model was constructed with (and the gate can read it back).
 * Returns 0 for an unknown handle. */
int64_t cenv_occ_rng(int64_t h);

/* Words consumed so far, for the gate. */
int64_t cenv_occ_draws(int64_t h);

/* ---- 4. Weather --------------------------------------------------------- */

/* Parses a weather CSV the way ReplayWeatherController.read_weather_csv does:
 * `Time` in `%Y%m%d-%H%M` read as UTC -> seconds since the Unix epoch, plus
 * the `TempF` and `Humidity` columns. Columns are located by header name and
 * quoted fields are honoured, so the station-name column's embedded comma does
 * not shift anything. Returns a handle > 0, 0 on failure. */
int64_t cenv_weather_load_csv(const char *path);

/* Same table from arrays the caller already has (xp = epoch seconds). */
int64_t cenv_weather_create(const double *xp, const double *temp_f,
                            const double *humidity, int64_t n);

void cenv_weather_destroy(int64_t h);

/* `which`: 0 = TempF, 1 = Humidity. `t_ns` is the utc instant in nanoseconds.
 * Writes the interpolated value. Returns 1 when the instant is outside the
 * table (np.interp clamps; the Python original raises before it gets there),
 * 0 on success, 2 for a bad handle/selector. */
int32_t cenv_weather_query(int64_t h, int64_t t_ns, int32_t which, double *out);

/* np.interp element-wise on one scalar / array, exposed for the gate. */
double cenv_weather_interp(const double *xp, const double *fp, int64_t n,
                           double x);

/* temperature_conversion.fahrenheit_to_kelvin, bit for bit. */
double cenv_weather_fahrenheit_to_kelvin(double temp_f);

/* Copies the table out (used by the shim to build the DataFrame, and by the
 * gate to diff against pandas). Each output array holds n doubles. */
int32_t cenv_weather_copy(int64_t h, double *xp, double *temp_f,
                          double *humidity);

/* Table bounds as seconds since epoch (out may be NULL). */
int32_t cenv_weather_bounds(int64_t h, double *out_min, double *out_max,
                            int64_t *out_n);

#ifdef __cplusplus
}
#endif

#endif /* CENV_STOCHASTIC_H */
