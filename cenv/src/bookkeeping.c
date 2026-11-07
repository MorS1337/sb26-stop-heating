/* Bookkeeping: per-zone temperature statistics and the reward arithmetic.
 *
 * Each function reproduces a specific Python function of the simulator; the
 * comment on the entry point names it. The Python side keeps the protobufs and
 * the object graph, passes flat arrays in, and copies flat results out.
 */

#include "cenv.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- plan table ---------------------------------------------------------
 * A plan is a CSR of flat control-volume indices per zone. Handles are 1-based
 * indices into a process-local table so the Python side can hold an int64.
 */

typedef struct {
  int64_t n_zones;
  int64_t n_cells;
  int64_t *offsets; /* n_zones + 1 */
  int64_t *cells;   /* n_cells */
  double *scratch;  /* max zone size, reused for the pairwise-sum gather */
  int64_t scratch_n;
  int live;
} cenv_bk_plan;

static cenv_bk_plan **g_plans = NULL;
static int64_t g_plans_len = 0;
static int64_t g_plans_cap = 0;

static cenv_bk_plan *plan_get(int64_t handle) {
  if (handle <= 0 || handle > g_plans_len) return NULL;
  cenv_bk_plan *p = g_plans[handle - 1];
  return (p != NULL && p->live) ? p : NULL;
}

/* Returns a free slot index, growing the table as needed (0 on OOM). */
static int64_t plan_slot(cenv_bk_plan *p) {
  for (int64_t i = 0; i < g_plans_len; ++i) {
    if (g_plans[i] == NULL || !g_plans[i]->live) {
      if (g_plans[i] != NULL) {
        free(g_plans[i]->offsets);
        free(g_plans[i]->cells);
        free(g_plans[i]->scratch);
        free(g_plans[i]);
      }
      g_plans[i] = p;
      return i + 1;
    }
  }
  if (g_plans_len == g_plans_cap) {
    int64_t cap = g_plans_cap ? g_plans_cap * 2 : 8;
    cenv_bk_plan **grown = realloc(g_plans, (size_t)cap * sizeof(*grown));
    if (grown == NULL) return 0;
    g_plans = grown;
    g_plans_cap = cap;
  }
  g_plans[g_plans_len] = p;
  return ++g_plans_len;
}

int64_t cenv_bk_plan_create(int64_t n_zones, const int64_t *offsets,
                            const int64_t *cells) {
  if (n_zones < 0 || offsets == NULL || (n_zones > 0 && cells == NULL))
    return 0;
  if (offsets[0] != 0) return 0;
  for (int64_t z = 0; z < n_zones; ++z)
    if (offsets[z + 1] < offsets[z]) return 0;
  cenv_bk_plan *p = calloc(1, sizeof(*p));
  if (p == NULL) return 0;
  p->n_zones = n_zones;
  p->n_cells = offsets[n_zones];
  p->offsets = malloc((size_t)(n_zones + 1) * sizeof(int64_t));
  p->scratch_n = 0;
  for (int64_t z = 0; z < n_zones; ++z) {
    int64_t len = offsets[z + 1] - offsets[z];
    if (len > p->scratch_n) p->scratch_n = len;
  }
  p->cells = malloc((size_t)(p->n_cells > 0 ? p->n_cells : 1) * sizeof(int64_t));
  p->scratch = malloc((size_t)(p->scratch_n > 0 ? p->scratch_n : 1) *
                      sizeof(double));
  if (p->offsets == NULL || p->cells == NULL || p->scratch == NULL) {
    free(p->offsets);
    free(p->cells);
    free(p->scratch);
    free(p);
    return 0;
  }
  memcpy(p->offsets, offsets, (size_t)(n_zones + 1) * sizeof(int64_t));
  if (p->n_cells > 0)
    memcpy(p->cells, cells, (size_t)p->n_cells * sizeof(int64_t));
  p->live = 1;
  int64_t handle = plan_slot(p);
  if (handle == 0) {
    free(p->offsets);
    free(p->cells);
    free(p->scratch);
    free(p);
    return 0;
  }
  return handle;
}

void cenv_bk_plan_destroy(int64_t handle) {
  cenv_bk_plan *p = plan_get(handle);
  if (p == NULL) return;
  p->live = 0;
  free(p->offsets);
  free(p->cells);
  free(p->scratch);
  p->offsets = NULL;
  p->cells = NULL;
  p->scratch = NULL;
  free(p);
  g_plans[handle - 1] = NULL;
}

int64_t cenv_bk_plan_n_zones(int64_t handle) {
  cenv_bk_plan *p = plan_get(handle);
  return p == NULL ? -1 : p->n_zones;
}

int32_t cenv_bk_zone_stats(int64_t handle, const double *temp, int64_t temp_n,
                           const int64_t *sel, int64_t n_sel, double *out_min,
                           double *out_max, double *out_mean,
                           int64_t *out_count) {
  cenv_bk_plan *p = plan_get(handle);
  if (p == NULL || temp == NULL || sel == NULL || n_sel < 0) return -1;
  for (int64_t s = 0; s < n_sel; ++s) {
    const int64_t z = sel[s];
    if (z < 0 || z >= p->n_zones) return -2;
    const int64_t begin = p->offsets[z];
    const int64_t n = p->offsets[z + 1] - begin;
    double *scratch = p->scratch;
    double mn = INFINITY, mx = -INFINITY;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t idx = p->cells[begin + i];
      if (idx < 0 || idx >= temp_n) return -2;
      const double v = temp[idx];
      scratch[i] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    /* np.min / np.max over the gathered array. */
    if (out_min != NULL) out_min[s] = n > 0 ? mn : INFINITY;
    if (out_max != NULL) out_max[s] = n > 0 ? mx : -INFINITY;
    /* np.mean(gathered) == np.add.reduce(gathered) / n, pairwise. */
    if (out_mean != NULL)
      out_mean[s] = n > 0 ? cenv_pairwise_sum(scratch, n) / (double)n : 0.0;
    if (out_count != NULL) out_count[s] = n;
  }
  return 0;
}

/* ---- reward -------------------------------------------------------------
 * Straight transcription of the Python, including its operation order: every
 * `a * b / c` below is `((a * b) / c)` as Python evaluates it.
 */

double cenv_bk_zone_productivity(double heating_setpoint, double cooling_setpoint,
                                 double zone_temp, double delta_time_sec,
                                 double occupancy, double max_personhour_usd,
                                 double deviation_ref) {
  /* deviation = max(0.0, zone_temp - cooling_setpoint, heating_setpoint - zone_temp) */
  double deviation = 0.0;
  const double over = zone_temp - cooling_setpoint;
  const double under = heating_setpoint - zone_temp;
  if (over > deviation) deviation = over;
  if (under > deviation) deviation = under;
  double shortfall;
  if (deviation <= deviation_ref) {
    shortfall = deviation / deviation_ref;
  } else {
    const double excess = (deviation - deviation_ref) / deviation_ref;
    shortfall = 1.0 + excess * excess; /* Python: ** 2.0 */
  }
  const double productivity = max_personhour_usd * (1.0 - shortfall);
  return productivity * occupancy * delta_time_sec / 3600.0;
}

int32_t cenv_bk_reward(int64_t n_zones, const cenv_zone_inputs *zones,
                       const cenv_plant_inputs *plant,
                       const cenv_reward_params *params, double *out) {
  if (zones == NULL || plant == NULL || params == NULL || out == NULL ||
      n_zones < 0)
    return -1;

  /* --- productivity ---------------------------------------------------- */
  double cumulative_productivity = 0.0;
  double total_occupancy = 0.0;
  for (int64_t z = 0; z < n_zones; ++z) {
    const double occupancy = zones->occupancy[z];
    total_occupancy += occupancy;
    cumulative_productivity += cenv_bk_zone_productivity(
        zones->heating_setpoint[z], zones->cooling_setpoint[z],
        zones->zone_temp[z], params->delta_time_sec, occupancy,
        params->max_productivity_personhour_usd,
        params->productivity_deviation_ref);
  }

  const double max_productivity = params->max_productivity_personhour_usd *
                                  total_occupancy * params->delta_time_sec /
                                  3600.0;
  const double min_productivity = params->min_productivity_personhour_usd *
                                  total_occupancy * params->delta_time_sec /
                                  3600.0;
  double normalized_productivity_regret;
  if (total_occupancy > 0.0) {
    normalized_productivity_regret =
        (cumulative_productivity - min_productivity) /
            (max_productivity - min_productivity) -
        1.0;
  } else {
    normalized_productivity_regret = 0.0;
  }

  /* --- plant rates ----------------------------------------------------- */
  double elec_rate = 0.0;
  for (int64_t i = 0; i < plant->n_ahu; ++i) {
    /* Python: += blower + abs(air_conditioning)  (one addition, then one +=) */
    elec_rate += plant->ahu_blower_rate[i] + fabs(plant->ahu_cooling_rate[i]);
  }
  for (int64_t i = 0; i < plant->n_boiler; ++i)
    elec_rate += plant->boiler_pump_rate[i];
  for (int64_t i = 0; i < plant->n_heat_pump; ++i) {
    elec_rate += fabs(plant->hp_pump_rate[i]);
    elec_rate += fabs(plant->hp_electric_heat_rate[i]);
  }
  double gas_rate = 0.0;
  for (int64_t i = 0; i < plant->n_boiler; ++i) gas_rate += plant->boiler_gas_rate[i];

  /* --- costs and carbon (ElectricityEnergyCost.cost/carbon) ------------ */
  const double dt = params->delta_time_sec;
  const double elec_cost = params->electricity_price * fabs(elec_rate) * dt;
  const double max_elec_cost =
      params->electricity_price * fabs(params->max_electricity_rate) * dt;
  const double elec_carbon =
      params->electricity_carbon_rate * fabs(elec_rate) * dt;
  const double max_elec_carbon =
      params->electricity_carbon_rate * fabs(params->max_electricity_rate) * dt;

  /* --- gas (NaturalGasEnergyCost.cost/carbon): negative is clamped to 0 -- */
  const double gas_rate_clamped = gas_rate < 0.0 ? 0.0 : gas_rate;
  const double max_gas_rate_clamped =
      params->max_natural_gas_rate < 0.0 ? 0.0 : params->max_natural_gas_rate;
  const double gas_cost = params->gas_price * (gas_rate_clamped * dt);
  const double max_gas_cost = params->gas_price * (max_gas_rate_clamped * dt);
  const double gas_carbon = params->gas_carbon_rate * (gas_rate_clamped * dt);
  const double max_gas_carbon =
      params->gas_carbon_rate * (max_gas_rate_clamped * dt);

  /* --- normalisation --------------------------------------------------- */
  const double combined_cost = elec_cost + gas_cost;
  const double denom_cost = max_elec_cost + max_gas_cost;
  const double normalized_energy_cost =
      denom_cost > 0.0 ? combined_cost / denom_cost : 0.0;

  const double combined_carbon = elec_carbon + gas_carbon;
  const double denom_carbon = max_elec_carbon + max_gas_carbon;
  const double normalized_carbon =
      denom_carbon > 0.0 ? combined_carbon / denom_carbon : 0.0;

  const double raw = normalized_productivity_regret * params->productivity_weight -
                     normalized_energy_cost * params->energy_cost_weight -
                     normalized_carbon * params->carbon_emission_weight;
  const double agent_reward_value =
      raw / (params->productivity_weight + params->energy_cost_weight +
             params->carbon_emission_weight);

  out[CENV_RW_AGENT_REWARD_VALUE] = agent_reward_value;
  out[CENV_RW_PRODUCTIVITY_REWARD] = cumulative_productivity;
  out[CENV_RW_NATURAL_GAS_ENERGY_COST] = gas_cost;
  out[CENV_RW_ELECTRICITY_ENERGY_COST] = elec_cost;
  out[CENV_RW_CARBON_EMITTED] = combined_carbon;
  out[CENV_RW_PRODUCTIVITY_REGRET] = cumulative_productivity - max_productivity;
  out[CENV_RW_NORMALIZED_PRODUCTIVITY_REGRET] = normalized_productivity_regret;
  out[CENV_RW_NORMALIZED_ENERGY_COST] = normalized_energy_cost;
  out[CENV_RW_NORMALIZED_CARBON_EMISSION] = normalized_carbon;
  out[CENV_RW_TOTAL_OCCUPANCY] = total_occupancy;
  out[CENV_RW_ELECTRICITY_ENERGY_RATE] = elec_rate;
  out[CENV_RW_NATURAL_GAS_ENERGY_RATE] = gas_rate;
  out[CENV_RW_MAX_ELECTRICITY_COST] = max_elec_cost;
  out[CENV_RW_MAX_NATURAL_GAS_COST] = max_gas_cost;
  out[CENV_RW_MAX_ELECTRICITY_CARBON] = max_elec_carbon;
  out[CENV_RW_MAX_NATURAL_GAS_CARBON] = max_gas_carbon;
  out[CENV_RW_MAX_PRODUCTIVITY] = max_productivity;
  out[CENV_RW_MIN_PRODUCTIVITY] = min_productivity;
  return 0;
}
