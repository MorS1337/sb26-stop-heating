/* Fused per-step kernel: one C call for everything `env.step` does.
 *
 * The Python side of the simulator (see cenv/fused.py for the full map) still
 * owns gin parsing, construction, the action request protobufs, reset and the
 * policy. Everything between "the action has been applied to the devices" and
 * "here is the next observation, the reward-info numbers and the new device
 * state" happens in `cenv_fused_run`, in the exact order the Python code runs
 * it -- including the boiler's sensor reads, whose side effects the shipped
 * tree's tank follower depends on.
 *
 * The kernel is a port, not a model: every expression follows the Python it
 * replaces (same order, no reassociation), and the physics/rng/occupancy
 * pieces are the already-gated entry points of thermal.c, hvac.c,
 * bookkeeping.c and the stochastic sources, called in the same order as the
 * Python. Nothing here decides a simulation detail on its own.
 *
 * Allocation: everything is fixed at `cenv_fused_create`; the step allocates
 * nothing and returns 0 (or a positive error code: 1 no plan, 2 bad argument,
 * 3 size mismatch). Numerical divergence from the Python (a NaN, a
 * temperature outside the physical bounds) is reported through
 * `views->flags`, not by raising.
 *
 * Built for a *batched* future: the state is one struct with no globals and no
 * dependence on the Python objects, so several handles can be stepped from
 * one thread group, and the air-side term list is indexed the same way the
 * solver's `solve_many` will want it. Nothing here calls back into Python.
 */

#ifndef CENV_FUSED_H
#define CENV_FUSED_H

#include <stdint.h>

#include "thermal.h"
#include "thermal_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cenv_fused cenv_fused;

/* ---- the observation plan ----------------------------------------------- *
 * One entry per observation vector slot the kernel fills. `index` selects a
 * device (VAV / AHU) or is unused (HWS slots are singletons); `slot` is the
 * position to write; the value is normalised as (raw - mean) / scale, with 0
 * written when scale == 0 -- exactly what the Python assembler does.
 *
 * Kinds are the observable attribute behind one measurement name, not the
 * name itself, because several names alias one attribute (`supply_air_
 * flowrate_setpoint` and `..._sensor` are both `AirHandler.air_flow_rate`).
 */
enum cenv_obs_kind {
  CENV_OBS_VAV_DAMPER = 0,
  CENV_OBS_VAV_VALVE,
  CENV_OBS_VAV_MAX_FLOW,
  CENV_OBS_VAV_SETPOINT_HEAT,
  CENV_OBS_VAV_SETPOINT_COOL,
  CENV_OBS_VAV_ZONE_TEMP,
  CENV_OBS_AHU_COOLING_REQUESTS,
  CENV_OBS_AHU_STATIC_SETPOINT, /* setpoint and sensor are one attribute */
  CENV_OBS_AHU_FLOW,            /* flowrate setpoint and sensor */
  CENV_OBS_AHU_SUPPLY_TEMP_SENSOR,
  CENV_OBS_AHU_SUPPLY_TEMP_SETPOINT,
  CENV_OBS_AHU_RUN_COMMAND,     /* supervisor / supply fan / exhaust fan */
  CENV_OBS_AHU_FAN_SPEED,       /* supply and discharge fan speed */
  CENV_OBS_AHU_OUTSIDE_FLOW,
  CENV_OBS_AHU_RECIRC,
  CENV_OBS_AHU_OUTSIDE_TEMP,
  CENV_OBS_HWS_DIFFP_SENSOR,
  CENV_OBS_HWS_DIFFP_SETPOINT,
  CENV_OBS_HWS_HEATING_REQUESTS,
  CENV_OBS_HWS_RUN_STATUS,
  CENV_OBS_HWS_SUPPLY_WATER_TEMP, /* has a side effect on the tank follower */
  CENV_OBS_HWS_SUPPLY_WATER_SETPOINT,
  CENV_OBS_AUX_COMFORT_NOW,
  CENV_OBS_AUX_COMFORT_SOON,
  CENV_OBS_AUX_NUM_OCCUPANTS,
  CENV_OBS_TIME,                  /* `index` = row of the feature table */
  CENV_OBS_KIND_LEN
};

/* Steps the kernel's own time is split into (and reported through
 * `cenv_fused_views.phase`, in CPU seconds, accumulated over the run). */
enum cenv_phase_kind {
  CENV_PH_ZONE_MEANS = 0,
  CENV_PH_VAV_UPDATE,
  CENV_PH_RECIRC,
  CENV_PH_AHU_DEMAND,
  CENV_PH_COUPLED,
  CENV_PH_GAINS,
  CENV_PH_INPUT_Q,
  CENV_PH_THERMAL,
  CENV_PH_AIR_FOLDBACK,
  CENV_PH_CONVECTION,
  CENV_PH_RETURN_WATER,
  CENV_PH_HEAT_DRAW,
  CENV_PH_CLIP,
  CENV_PH_OBSERVATION,
  CENV_PH_REWARD_INFO,
  CENV_PH_DEVICE_STATE,
  CENV_PH_LEN
};

typedef struct cenv_obs_slot {
  int32_t kind;
  int32_t index;
  int32_t slot;
  double mean;
  double scale;
} cenv_obs_slot;

/* One VAV: geometry-independent parameters plus its place in the plans.
 * `zones` are the indices (into the physical-zone list) of
 * `hvac.get_zones_for_vav(vav_id)` filtered to zones that have a temperature,
 * in that order, and `area` is the sum of their control-volume counts. */
typedef struct cenv_vav_cfg {
  double max_air_flow_rate;
  double min_damper;
  double max_discharge_temp;
  double max_flow_static_pressure;
  double reheat_max_flow_factor;
  int32_t ahu;
  int32_t coupled;      /* air-side column, or -1 */
  int32_t zone_off;     /* into cenv_fused_cfg.vav_zone */
  int32_t zone_n;
  int32_t area;
  int32_t area_all;
} cenv_vav_cfg;

typedef struct cenv_ahu_cfg {
  double recirculation;
  double supply_setpoint;
  double fan_static_pressure;
  double fan_efficiency;
  double max_air_flow_rate;
  double min_static_pressure;
  double max_recirculation;
  double cooling_cop;
  int32_t run_command;  /* RunStatus as an int: 1 ON, 0 OFF */
} cenv_ahu_cfg;

typedef struct cenv_fused_cfg {
  /* -- geometry --------------------------------------------------------- */
  int32_t nrows;
  int32_t ncols;
  int32_t n_zones;              /* physical zones, room_dict order */
  const int32_t *zone_off;      /* [n_zones + 1] into zone_cells */
  const int32_t *zone_cells;    /* flat CV indices, room_dict order */
  const int32_t *zone_of_cv;    /* [n_full], zone index or -1 */
  const int32_t *zone_area;     /* [n_zones] CV count of each zone */

  /* -- air side --------------------------------------------------------- */
  int32_t n_coupled;
  const int32_t *coupled_vav;   /* [n_coupled] -> VAV index */
  int32_t n_vav_zones;          /* total entries in .vav_zone */
  int32_t n_zone_vav;           /* total entries in .zone_vav */

  /* -- thermal ---------------------------------------------------------- */
  void *thermal;                /* cenv_thermal * from cenv_thermal_create */

  /* -- devices ---------------------------------------------------------- */
  int32_t n_vav;
  const cenv_vav_cfg *vavs;
  const int32_t *vav_zone;      /* [vavs[n_vav-1].zone_off + zone_n] */
  const int32_t *zone_vav_off;  /* [n_zones + 1] into zone_vav */
  const int32_t *zone_vav;      /* [..] VAV indices serving each zone */
  int32_t n_ahu;
  const cenv_ahu_cfg *ahus;

  /* -- hot water -------------------------------------------------------- */
  int32_t hws_energy_balance;   /* corrected tree's tank energy balance */
  int32_t hws_has_tank;
  int32_t hws_run;              /* HotWaterSystem run command */
  int32_t pump_run;
  double hws_header_resistance;
  double boiler_heating_rate;
  double boiler_cooling_rate;
  double boiler_convection_coef;
  double boiler_tank_length;
  double boiler_tank_radius;
  double boiler_water_capacity;
  double boiler_insulation_conductivity;
  double boiler_insulation_thickness;
  double boiler_setpoint;
  double boiler_return_temp;
  double pump_head;
  double pump_efficiency;

  /* -- stochastic ------------------------------------------------------- */
  int64_t occ_handle;           /* cenv_occ_* handle */
  const int32_t *occ_zone_id;   /* [n_zones] zone index inside that handle */
  int32_t n_occ_all;
  const int32_t *occ_all;       /* [n_occ_all] zone indices, building.zones order */
  int32_t n_rooms;
  const int64_t *rooms;         /* [n_rooms] cenv_conv_room handles */
  int64_t conv_rng;             /* cenv_mt handle for the convection draws */
  double conv_p;

  /* -- per-step time series --------------------------------------------- *
   * Everything that depends only on the timestamp, precomputed by the Python
   * side for the whole episode: the tz database, DST, holidays, pandas
   * Timestamp arithmetic and numpy's cos/sin stay in Python, where they are
   * already correct, and the kernel only indexes them. */
  int64_t step_ns;
  int64_t origin_ns;            /* utc ns of step 0 */
  int32_t n_horizon;
  const double *ambient;        /* [n_horizon] outside air, K */
  const uint8_t *comfort;       /* [n_horizon] schedule.is_comfort_mode */
  const double *conv_coef;      /* [n_horizon] air convection coefficient */
  const int32_t *feat_slot;     /* [n_feat] observation slots */
  const float *feat;            /* [n_feat][n_horizon], row-major */
  int32_t n_feat;

  /* -- schedule windows ------------------------------------------------- */
  double comfort_heat;
  double comfort_cool;
  double eco_heat;
  double eco_cool;

  /* -- simulator constants ---------------------------------------------- */
  double internal_gain_w_per_m2;
  double internal_gain_standby_fraction;
  double occupant_heat_w;
  double cv_area_m2;
  double time_step_sec;

  /* -- observation plan ------------------------------------------------- */
  int32_t n_obs;
  const cenv_obs_slot *obs;

  /* -- batched stepping ------------------------------------------------- *
   * Exactly one of `thermal` and `thermal_many` is set. `thermal_many` is the
   * scenario's shared batched solve (`cenv_tbatch`, thermal_batch.h): the
   * handle then has no factorization of its own and `cenv_fused_run` refuses
   * to step it, because the M environments' solves have to be issued together
   * (cenv_batch_run in batch.c). Everything else -- the geometry plan, the
   * device state machines, the occupancy and convection streams -- is the same
   * per-environment state either way, so a batch is M of these handles.
   * Appended after the shipped fields so the layout above never moves. */
  void *thermal_many;
} cenv_fused_cfg;

/* Mutable device state the Python side reads back each step, and the outputs
 * of every step. The pointers are owned by the handle and stable for its
 * lifetime, so the caller resolves them once. */
typedef struct cenv_fused_views {
  float *obs;                   /* [n_obs] */
  double *mass;                 /* [n_solved] interior mass nodes */
  double *zone_temp;            /* [n_zones] reward-info inputs, room order */
  double *zone_heat_sp;
  double *zone_cool_sp;
  double *zone_flow_sp;
  double *zone_flow;
  double *zone_occupancy;
  double *ahu_intake;           /* [n_ahu] W, intake fan */
  double *ahu_exhaust;          /* [n_ahu] W, exhaust fan */
  double *ahu_cooling;          /* [n_ahu] W electrical */
  double *vav_damper;           /* [n_vav] */
  double *vav_valve;
  double *vav_zone_temp;
  double *vav_heat_sp;
  double *vav_cool_sp;
  double *vav_flow;             /* flow_rate_demand after the handler cap */
  double *vav_ratio;            /* supply_air_flow_ratio */
  int32_t *vav_mode;            /* Thermostat.Mode value */
  int32_t *vav_prev;            /* step index of the last update, -1 = None */
  double *ahu_flow;             /* [n_ahu] _air_flow_rate, W guard not applied */
  int32_t *ahu_requests;        /* [n_ahu] cooling_request_count */
  double *hws_flow_factor_sum;
  int32_t *hws_requests;
  double *hws_return;           /* return_water_temperature_sensor */
  double *hws_heating;          /* compute_thermal_energy_rate, W */
  double *hws_pump;             /* compute_pump_power, W */
  double *boiler_temp;          /* _current_temperature */
  double *boiler_change;        /* _step_tank_temperature_change */
  double *boiler_last_dur;      /* _last_step_duration, s */
  double *boiler_burner;        /* _burner_heat_rate, W */
  int32_t *flags;               /* bit 0: temperature clip engaged */
  double *diag;                 /* [4] last step: total reheat W, water
                                 * capacity rate W/K, supply water K, coils */
  double *phase;                /* [n_phase] CPU seconds per step phase,
                                 * accumulated (diagnostics, always on) */
  int32_t *n_phase;             /* [1] number of phase slots, 0 if none */
  int32_t *counts;              /* [4] zones, vavs, ahus, coupled */
} cenv_fused_views;

/* Builds a handle from the plan, allocating every array the step needs and
 * copying the device parameters. Returns NULL on a malformed plan. */
cenv_fused *cenv_fused_create(const cenv_fused_cfg *cfg);

void cenv_fused_free(cenv_fused *f);

int32_t cenv_fused_views_get(cenv_fused *f, cenv_fused_views *out);

/* Overwrites the mutable device state from the caller's arrays -- called once
 * after reset (and whenever the Python objects were changed behind the
 * kernel's back). `mode` / `prev` / timestamps are step indices; `prev` -1 for
 * "no previous timestamp", `ts_action` -1 for "no action applied yet". */
int32_t cenv_fused_sync_full(
    cenv_fused *f,
    const double *vav_damper, const double *vav_valve,
    const double *vav_zone_temp, const double *vav_heat_sp,
    const double *vav_cool_sp, const double *vav_ratio,
    const int32_t *vav_mode, const int32_t *vav_prev,
    const double *ahu_air_flow, const int32_t *ahu_requests,
    double hws_flow_factor_sum, int32_t hws_requests,
    double boiler_temp, double boiler_setpoint, double boiler_return_temp,
    double boiler_change, double boiler_last_dur, double boiler_burner,
    int64_t ts_obs, int64_t ts_action);

/* One simulator step, in the order simulator_flexible_floor_plan
 * .execute_step_sim runs it, followed by the observation and the reward-info
 * numbers at the post-step timestamp.
 *
 *   step      index of this step in the episode (the timestamp is
 *             origin + step * step_ns)
 *   temp      [n_full] in/out temperature field (building.temp)
 *   input_q   [n_full] in/out applied power (building.input_q)
 *   sync      mutable command state, layout:
 *               [0, n_vav)                 damper_command    (NAN = None)
 *               [n_vav, 2*n_vav)           reheat_valve_command (NAN = None)
 *               4 * n_ahu:                 run, supply_setpoint, fan_static_
 *                                          pressure, recirculation
 *               then 4:                    hws run, hws supply setpoint,
 *                                          pump differential head (m), pump run
 *               then 2:                    ts_obs, ts_action (step indices)
 *
 * Returns 0, or 1 no plan / factorisation, 2 bad argument, 3 bad size. */
int32_t cenv_fused_run(cenv_fused *f, int32_t step, double *temp,
                       double *input_q, const double *sync);

/* ---- batched stepping (batch.c) ----------------------------------------- *
 * M environments of one scenario, one C call per step, one batched thermal
 * solve. Every environment keeps its own device state, tank, temperature
 * field, mass nodes, occupancy stream and convection generator: the handles
 * are independent, and the only thing shared is the batched solve
 * (`cenv_tbatch`, thermal_batch.h), which is the step's whole cost.
 *
 *   temps[e]   [n_full] in/out field of environment e (building.temp)
 *   qs[e]      [n_full] in/out input_q of environment e (building.input_q)
 *   syncs[e]   that environment's command block, as in cenv_fused_run
 *   tb         the shared batched solve; its k must match every handle's
 *
 * All environments must be at the same `step` (they share the scenario's
 * ambient table). Returns 0, or the first error any slice reported (the step
 * is then incomplete and the caller must not continue the episode). */
int32_t cenv_batch_run(cenv_fused **handles, int32_t m, int32_t step,
                       double *const *temps, double *const *qs,
                       const double *const *syncs, cenv_tbatch *tb);

/* The slices of one step, for a driver that batches the solve itself (the
 * paths above use them in this order). `cenv_fused_batch_finalize` copies the
 * solved field into `temp`, folds the air stream back into `input_q` and runs
 * the epilogue. Each returns 0, 2 bad argument or 3 bad step. */
int32_t cenv_fused_batch_prelude(cenv_fused *f, int32_t step, double *temp,
                                 double *input_q, const double *sync,
                                 double *ambient);
/* `sensed` comes back as the caller's buffer for the batched solve's
 * Q^T T' (n_coupled doubles, owned by the handle). */
int32_t cenv_fused_batch_state(cenv_fused *f, const double **g,
                               const double **ts, double **mass,
                               double **out_full, double **sensed);
int32_t cenv_fused_batch_finalize(cenv_fused *f, int32_t step,
                                  const double *sensed, double *input_q,
                                  double *temp);

#ifdef __cplusplus
}
#endif

#endif /* CENV_FUSED_H */
