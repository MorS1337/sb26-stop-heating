/* cenv -- C ports of the Smart Buildings simulator's per-step bookkeeping.
 *
 * Contract with the simulator: every entry point here reproduces, bit-for-bit
 * where the arithmetic order allows it, the Python function named in its
 * comment. Nothing in this library is allowed to change behaviour: it is
 * installed by monkeypatch at runtime and gated against the Python original
 * (see cenv/shadow.py) before a component is switched on.
 *
 * Numerics: compiled with -O3 -ffp-contract=off, never -ffast-math. Sums that
 * stand in for numpy reductions implement numpy's pairwise summation (block 128,
 * eight accumulators) so they match `np.add.reduce` exactly.
 *
 * Component prefixes: cenv_bk_* (bookkeeping: zone statistics and the reward
 * arithmetic), cenv_thermal_*, cenv_hvac_*, cenv_stoch_*. Peer components own
 * their own headers; they are picked up below when present, so this file never
 * needs editing to expose a new component.
 */

#ifndef CENV_H
#define CENV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CENV_VERSION_MAJOR 0
#define CENV_VERSION_MINOR 1

/* ---- common ------------------------------------------------------------- */

/* Returns a static version string, e.g. "cenv 0.1". */
const char *cenv_version(void);

/* Self-test: returns 0 when the library round-trips a f64 buffer through a
 * predictable computation (used by cenv/verify/smoke.py). */
int32_t cenv_selftest(double *buf, int64_t n, double scale);

/* numpy-compatible pairwise summation of n doubles (see numpy's
 * pairwise_sum: 8 accumulators, block size 128, n < 8 sequential from -0.0). */
double cenv_pairwise_sum(const double *a, int64_t n);

/* ---- bookkeeping: zone statistics --------------------------------------- *
 * `Building.get_zone_temp_stats(zone)` == (min, max, mean) over the CVs of one
 * zone; `Building.get_zone_average_temps()` == the same mean for every physical
 * zone. Zone membership is fixed at construction, so the Python side builds the
 * plan once (a CSR of flat CV indices per zone, in numpy's gather order).
 *
 * A plan handle is an int64 index into a process-local table; 0 means "no
 * plan". All plan functions are not thread-safe (one plan is used from one
 * process/one env, exactly like the Python object it replaces).
 */

/* Creates a plan over n_zones zones whose CV indices are
 * cells[offsets[z] .. offsets[z+1]) for z in [0, n_zones).
 * Returns the handle, or 0 on allocation failure / malformed input. */
int64_t cenv_bk_plan_create(int64_t n_zones, const int64_t *offsets,
                            const int64_t *cells);

/* Releases a plan (no-op for unknown handles). */
void cenv_bk_plan_destroy(int64_t handle);

/* Number of zones in a plan, or -1 for an unknown handle. */
int64_t cenv_bk_plan_n_zones(int64_t handle);

/* Per-zone min/max/mean of `temp` over the selected zones, in selection order.
 * `sel` holds n_sel zone indices into the plan; `temp_n` is the length of the
 * flat grid. `out_*` may be NULL to skip that output, otherwise each holds
 * n_sel doubles (mean) / int64 (count). An out-of-range CV or zone index
 * returns -2 (a plan/grid mismatch is a bug, not a numerical difference).
 * The sums are np.add.reduce's pairwise order and min/max are np.min/np.max, so
 * the results match the numpy original bit-for-bit. Returns 0 on success. */
int32_t cenv_bk_zone_stats(int64_t handle, const double *temp, int64_t temp_n,
                           const int64_t *sel, int64_t n_sel, double *out_min,
                           double *out_max, double *out_mean,
                           int64_t *out_count);

/* ---- bookkeeping: reward arithmetic ------------------------------------- *
 * Reproduces BaseSetpointEnergyCarbonRewardFunction._sum_zone_productivities /
 * _get_zone_productivity_reward, _sum_electricity_energy_rate,
 * _sum_natural_gas_energy_rate, ElectricityEnergyCost.cost/carbon,
 * NaturalGasEnergyCost.cost/carbon and SetpointEnergyCarbonRegretFunction
 * .compute_reward -- the arithmetic only; the Python side still owns the
 * RewardInfo/RewardResponse protobufs the env API requires.
 *
 * Inputs are the flat per-zone and per-device numbers plus the step's scalar
 * constants. The outputs land in `out`, indexed by the cenv_reward_out enum.
 */

enum cenv_reward_out {
  CENV_RW_AGENT_REWARD_VALUE = 0,
  CENV_RW_PRODUCTIVITY_REWARD,          /* response.productivity_reward */
  CENV_RW_NATURAL_GAS_ENERGY_COST,      /* response.natural_gas_energy_cost */
  CENV_RW_ELECTRICITY_ENERGY_COST,      /* response.electricity_energy_cost */
  CENV_RW_CARBON_EMITTED,               /* response.carbon_emitted */
  CENV_RW_PRODUCTIVITY_REGRET,          /* response.productivity_regret */
  CENV_RW_NORMALIZED_PRODUCTIVITY_REGRET,
  CENV_RW_NORMALIZED_ENERGY_COST,
  CENV_RW_NORMALIZED_CARBON_EMISSION,
  CENV_RW_TOTAL_OCCUPANCY,
  CENV_RW_ELECTRICITY_ENERGY_RATE,
  CENV_RW_NATURAL_GAS_ENERGY_RATE,
  CENV_RW_MAX_ELECTRICITY_COST,
  CENV_RW_MAX_NATURAL_GAS_COST,
  CENV_RW_MAX_ELECTRICITY_CARBON,
  CENV_RW_MAX_NATURAL_GAS_CARBON,
  CENV_RW_MAX_PRODUCTIVITY,
  CENV_RW_MIN_PRODUCTIVITY,
  CENV_RW_LEN
};

/* Scalar constants of one reward step. Every value is taken from the reward
 * function / cost object the Python side already holds, so a gin override
 * flows through untouched. */
typedef struct cenv_reward_params {
  double delta_time_sec;          /* (end - start).total_seconds() */
  double max_productivity_personhour_usd;
  double min_productivity_personhour_usd;
  double productivity_deviation_ref;
  double productivity_weight;
  double energy_cost_weight;
  double carbon_emission_weight;
  double max_electricity_rate;    /* W, the normalisation denominator */
  double max_natural_gas_rate;    /* W */
  double electricity_price;       /* USD / W / s for this step (hour + day type) */
  double electricity_carbon_rate; /* kg / W / s for this step (hour) */
  double gas_price;               /* USD / J for this step (month) */
  double gas_carbon_rate;         /* kg / J */
} cenv_reward_params;

/* Zone arrays, all length n_zones, in the order the Python loop visits them
 * (the RewardInfo map insertion order). */
typedef struct cenv_zone_inputs {
  const double *heating_setpoint; /* K */
  const double *cooling_setpoint; /* K */
  const double *zone_temp;        /* K */
  const double *occupancy;        /* people, fractional */
} cenv_zone_inputs;

/* Plant inputs. Each array is summed in the order the Python loops use:
 * air handlers first (blower + |cooling|), then boiler pumps, then heat pumps. */
typedef struct cenv_plant_inputs {
  const double *ahu_blower_rate;      /* W, n_ahu */
  const double *ahu_cooling_rate;     /* W, n_ahu, sign-carrying */
  int64_t n_ahu;
  const double *boiler_pump_rate;     /* W, n_boiler */
  const double *boiler_gas_rate;      /* W, n_boiler */
  int64_t n_boiler;
  const double *hp_pump_rate;         /* W, n_heat_pump */
  const double *hp_electric_heat_rate;/* W, n_heat_pump */
  int64_t n_heat_pump;
} cenv_plant_inputs;

/* Computes the whole reward step. `out` has CENV_RW_LEN doubles.
 * Returns 0 on success, -1 on a NULL argument / negative length. */
int32_t cenv_bk_reward(int64_t n_zones, const cenv_zone_inputs *zones,
                       const cenv_plant_inputs *plant,
                       const cenv_reward_params *params, double *out);

/* Productivity of one zone, exposed for the shadow harness. */
double cenv_bk_zone_productivity(double heating_setpoint, double cooling_setpoint,
                                 double zone_temp, double delta_time_sec,
                                 double occupancy, double max_personhour_usd,
                                 double deviation_ref);

/* ---- peer components (headers appear as they land) ---------------------- */

#if defined(__has_include)
#if __has_include("thermal.h")
#include "thermal.h"
#endif
#if __has_include("hvac.h")
#include "hvac.h"
#endif
#if __has_include("stochastic.h")
#include "stochastic.h"
#endif
#if __has_include("observation.h")
#include "observation.h"
#endif
#if __has_include("fused.h")
#include "fused.h"
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif /* CENV_H */
