/* Fused per-step kernel. See fused.h for the contract and cenv/fused.py for
 * the map of the Python it reproduces.
 *
 * _POSIX_C_SOURCE must come before any header: the phase clock is
 * clock_gettime(CLOCK_PROCESS_CPUTIME_ID), which glibc hides under a strict
 * -std=c11 unless a POSIX level is requested. Harmless on macOS, where the
 * declaration is unconditional.
 *
 * Reading order: `cenv_fused_run` is the whole step and reads top to bottom in
 * the same order as simulator_flexible_floor_plan.execute_step_sim and then
 * environment.step's observation and reward-info queries. Each block names the
 * Python function it transcribes. The device state machines (thermostat, tank
 * follower, tank energy balance) are at the top, because their *call order* is
 * what the step is made of: the shipped boiler's supply-water sensor advances
 * the tank on every read, so the number and position of those reads is part of
 * the physics.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cenv.h"
#include "fused.h"
#include "thermal_batch.h"
#include "hvac.h"
#include "stochastic.h"

/* simulator/constants.py (device side) and utils/constants.py (water side).
 * Passed nowhere: they are physical constants of the simulator, not of a
 * building, and the Python modules define them once. */
#define FUSED_AIR_DENSITY 1.2
#define FUSED_AIR_CP 1006.0
#define FUSED_WATER_CP 4180.0
#define FUSED_WATER_DENSITY 1000.0
#define FUSED_GRAVITY 9.8
#define FUSED_PASCALS_PER_BAR 100000.0

#define FUSED_FLAG_CLIPPED 1

/* Thermostat.Mode */
#define MODE_OFF 0
#define MODE_HEAT 1
#define MODE_COOL 2
#define MODE_PASSIVE_COOL 3

/* Per-phase CPU clock. `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` on a shared
 * machine: wall time here would report other agents' load, not this kernel's
 * work. The cost is ~25 ns per call, i.e. well under 1 us per step for the 16
 * marks below -- which is why it is always on rather than behind a flag. */
#if defined(CLOCK_PROCESS_CPUTIME_ID)
static double fused_clock(void) {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#else
/* ISO C fallback: 1 us resolution, plenty for phase totals over a run. */
static double fused_clock(void) {
  return (double)clock() / (double)CLOCKS_PER_SEC;
}
#endif

typedef struct {
  /* parameters, fixed at create */
  double max_flow;
  double min_damper;
  double max_discharge;
  double design_static;
  double reheat_max_flow;
  int32_t ahu;
  int32_t zoff;
  int32_t zn;
  int32_t area;      /* CVs of the serviced zones that have a temperature */
  int32_t area_all;  /* CVs of every serviced zone (reward-info area share) */
  int32_t coupled;   /* air-side column, or -1 */
  /* state */
  double damper;     /* _damper_setting */
  double valve;      /* _reheat_valve_setting */
  double zone_temp;  /* _zone_air_temperature */
  double heat_sp;    /* _zone_air_heating_temperature_setpoint */
  double cool_sp;    /* _zone_air_cooling_temperature_setpoint */
  double ratio;      /* _supply_air_flow_ratio */
  double flow;       /* flow_rate_demand, after the handler cap */
  double dcmd;       /* damper_command this step (NAN = None) */
  double vcmd;       /* reheat_valve_command this step (NAN = None) */
  int32_t mode;      /* thermostat._current_mode */
  int32_t prev_step; /* thermostat._previous_timestamp as a step index */
} fused_vav;

typedef struct {
  int32_t run;
  double recirc;
  double supply_sp;
  double fan_sp;
  double fan_eff;
  double max_flow;
  double min_sp;
  double max_recirc;
  double cop;
  double air_flow;    /* _air_flow_rate */
  int32_t requests;   /* _cooling_request_count */
  double supply_temp; /* get_supply_air_temp this step */
  double recirc_temp; /* recirculation temperature this step */
} fused_ahu;

typedef struct {
  int32_t energy_balance;
  int32_t has_tank;
  int32_t run;
  int32_t pump_run;
  double header_resistance;
  double heating_rate;
  double cooling_rate;
  double conv_coef;
  double tank_len;
  double tank_radius;
  double water_capacity;
  double insul_cond;
  double insul_thick;
  double setpoint;
  double current;
  double return_temp;
  double step_change;
  double last_dur;
  double burner;
  int64_t ts_action;  /* -1 = boiler._action_timestamp is None */
  int64_t ts_obs;
  double flow_factor_sum;
  double dp;          /* pump differential pressure setpoint, bar */
  double pump_head;
  double pump_eff;
  int32_t requests;
} fused_hws;

struct cenv_fused {
  int32_t nrows;
  int32_t ncols;
  int32_t n_full;
  int32_t n_rows;     /* solved control volumes */
  int32_t n_zones;
  int32_t n_vav;
  int32_t n_ahu;
  int32_t n_coupled;
  int32_t n_obs;
  int32_t n_feat;
  int32_t n_horizon;
  int32_t n_occ_all;
  int32_t n_rooms;

  int32_t *zone_off;
  int32_t *zone_cells;
  int32_t *zone_of_cv;
  int32_t *zone_area;

  cenv_thermal *thermal;      /* single-environment solve, NULL in batch mode */
  cenv_tbatch *thermal_many;  /* the batch's shared solve, NULL otherwise */
  double *mass;               /* n_rows */
  double *temp_out;           /* n_full scratch */
  double total_reheat_w;      /* prelude -> epilogue (diag and return water) */

  int32_t *coupled_vav;
  int32_t *vav_zone;    /* zone indices of every VAV, concatenated */
  int32_t *zone_vav_off;
  int32_t *zone_vav;
  fused_vav *vav;
  fused_ahu *ahu;
  fused_hws hws;

  int64_t occ;
  int32_t *occ_id;
  int32_t *occ_all;
  int64_t *rooms;
  int64_t conv_rng;
  double conv_p;

  int64_t step_ns;
  int64_t origin_ns;
  double *ambient;
  uint8_t *comfort;
  double *conv_coef;
  int32_t *feat_slot;
  float *feat;

  double comfort_heat;
  double comfort_cool;
  double eco_heat;
  double eco_cool;

  double internal_gain;
  double standby_fraction;
  double occupant_heat;
  double cv_area_m2;
  double dt;

  cenv_obs_slot *obs;

  /* per-step scratch */
  double *gather;
  double *avg;
  double *vav_vtemp;
  double *zone_occ;
  double *q_zone;
  uint8_t *q_used;
  double *g;
  double *ts;
  double *sensed;
  double *air_q;
  double *coil;
  int32_t coil_n;
  double *ahu_sum;
  double *ahu_area;
  double *ahu_flow_demand;
  double *recirc;
  double *rc_fsum;
  double *rc_ftotal;
  double *rc_asum;
  double *rc_atotal;

  /* outputs */
  cenv_fused_views v;
  float *obs_out;
  double *o_zone_temp;
  double *o_zone_heat;
  double *o_zone_cool;
  double *o_zone_flow_sp;
  double *o_zone_flow;
  double *o_zone_occ;
  double *o_ahu_intake;
  double *o_ahu_exhaust;
  double *o_ahu_cooling;
  double *o_vav_damper;
  double *o_vav_valve;
  double *o_vav_temp;
  double *o_vav_heat;
  double *o_vav_cool;
  double *o_vav_flow;
  double *o_vav_ratio;
  int32_t *o_vav_mode;
  int32_t *o_vav_prev;
  double *o_ahu_flow;
  int32_t *o_ahu_requests;
  double *o_hws_fsum;
  int32_t *o_hws_requests;
  double *o_hws_return;
  double *o_hws_heating;
  double *o_hws_pump;
  double *o_boiler_temp;
  double *o_boiler_change;
  double *o_boiler_dur;
  double *o_boiler_burner;
  int32_t *o_flags;
  int32_t *o_counts;
  double *o_diag;
  double phase[CENV_PH_LEN];
  int32_t n_phase;
};

/* ------------------------------------------------------------------------- */
/* allocation helpers                                                        */
/* ------------------------------------------------------------------------- */
static void *fzalloc(size_t count, size_t size) {
  if (count == 0) {
    return NULL;
  }
  return calloc(count, size);
}

static int32_t *fzalloc_i(int32_t n) {
  return (int32_t *)fzalloc((size_t)(n > 0 ? n : 0), sizeof(int32_t));
}

static double *fzalloc_d(int32_t n) {
  return (double *)fzalloc((size_t)(n > 0 ? n : 0), sizeof(double));
}

static int copy_i32(int32_t *dst, const int32_t *src, int32_t n) {
  if (n <= 0 || dst == NULL) {
    return 0;
  }
  if (src == NULL) {
    return -1;
  }
  memcpy(dst, src, (size_t)n * sizeof(int32_t));
  return 0;
}

/* ------------------------------------------------------------------------- */
/* device state machines                                                     */
/* ------------------------------------------------------------------------- */

/* thermostat.Thermostat._default_control */
static int32_t thermostat_default(int32_t mode, double zone_temp, double heat_sp,
                                  double cool_sp) {
  const double mid = 0.5 * (cool_sp - heat_sp) + heat_sp;
  if (zone_temp < heat_sp) {
    return MODE_HEAT;
  }
  if (zone_temp > cool_sp) {
    return MODE_COOL;
  }
  if (zone_temp < mid && mode == MODE_HEAT) {
    return MODE_HEAT;
  }
  if (zone_temp > mid && mode == MODE_COOL) {
    return MODE_COOL;
  }
  return MODE_OFF;
}

/* thermostat.Thermostat.update. `prev_comfort` is the schedule's comfort flag
 * at the thermostat's previous timestamp. */
static int32_t thermostat_update(fused_vav *v, double zone_temp, int32_t comfort,
                                 int32_t prev_valid, int32_t prev_comfort,
                                 double heat_sp, double cool_sp) {
  if (comfort) {
    v->mode = thermostat_default(v->mode, zone_temp, heat_sp, cool_sp);
  } else if (prev_valid && prev_comfort) {
    v->mode = MODE_PASSIVE_COOL;
  } else if (v->mode == MODE_PASSIVE_COOL && zone_temp > heat_sp &&
             zone_temp < cool_sp) {
    v->mode = MODE_PASSIVE_COOL;
  } else {
    v->mode = thermostat_default(v->mode, zone_temp, heat_sp, cool_sp);
  }
  return v->mode;
}

/* boiler.Boiler._set_current_temperature, run on every supply-water sensor
 * read. In the shipped tree that read is what advances the tank, so this is
 * called exactly as often, and exactly where, the Python reads it. */
static void boiler_set_current_temperature(fused_hws *h, double dt) {
  if (h->ts_action >= 0) {
    h->last_dur = (double)(h->ts_obs - h->ts_action) * dt;
  } else {
    h->ts_action = h->ts_obs;
  }
  if (h->ts_action >= 0 && h->cooling_rate > 0.0 && h->heating_rate > 0.0 &&
      h->has_tank) {
    const double begin = h->current;
    h->current = cenv_hvac_tank_temperature(h->setpoint, begin, h->last_dur,
                                            h->heating_rate, h->cooling_rate);
    h->step_change = h->current - begin;
  } else {
    h->current = h->setpoint;
  }
}

/* hot_water_system / boiler: `supply_water_temperature_sensor`. */
static double hws_supply_sensor(cenv_fused *f) {
  if (!f->hws.energy_balance) {
    boiler_set_current_temperature(&f->hws, f->dt);
  }
  return f->hws.current;
}

static double pump_dp(const fused_hws *h);

/* hot_water_system.HotWaterSystem.total_flow_rate. */
static double hws_total_flow(const fused_hws *h) {
  return cenv_hvac_water_flow_rate(h->dp, h->flow_factor_sum,
                                   h->header_resistance);
}

/* pump.WaterPump.differential_pressure_setpoint. */
static double pump_dp(const fused_hws *h) {
  if (!h->pump_run) {
    return 0.0;
  }
  return cenv_hvac_head_to_pressure(h->pump_head, FUSED_WATER_DENSITY,
                                    FUSED_GRAVITY, FUSED_PASCALS_PER_BAR);
}

/* air_handler.AirHandler.supply_air_static_pressure_setpoint (= ._sensor). */
static double ahu_static_pressure(const fused_ahu *a) {
  if (!a->run) {
    return 0.0;
  }
  return (a->fan_sp > a->min_sp) ? a->fan_sp : a->min_sp;
}

/* air_handler.AirHandler.air_flow_rate (0 while the unit is commanded off). */
static double ahu_air_flow(const fused_ahu *a) {
  return a->run ? a->air_flow : 0.0;
}

/* vav.Vav._compute_flow_rate_demand, uncapped. */
static double vav_unrestricted_flow(const cenv_fused *f, int32_t v) {
  const fused_vav *s = &f->vav[v];
  return cenv_hvac_vav_flow_demand(s->damper, s->max_flow,
                                   ahu_static_pressure(&f->ahu[s->ahu]),
                                   s->design_static);
}

/* vav.Vav._reheat_flow_rate. */
static double vav_reheat_flow(const cenv_fused *f, const fused_vav *s) {
  if (f->hws.flow_factor_sum == 0.0) {
    return 0.0;
  }
  return hws_total_flow(&f->hws) *
         (s->valve * s->reheat_max_flow / f->hws.flow_factor_sum);
}

/* vav.Vav.compute_reheat_energy_rate(current_supply_temp,
 *                                    hws.supply_water_temperature_sensor). */
static double vav_reheat_energy(const cenv_fused *f, const fused_vav *s,
                                double supply_air_temp, double water_temp) {
  const double air_capacity_rate = FUSED_AIR_CP * FUSED_AIR_DENSITY * s->flow;
  const double water_capacity_rate =
      FUSED_WATER_CP * FUSED_WATER_DENSITY * vav_reheat_flow(f, s);
  return cenv_hvac_reheat_energy_rate(air_capacity_rate, water_capacity_rate,
                                      FUSED_AIR_CP, FUSED_AIR_DENSITY, s->flow,
                                      supply_air_temp, water_temp,
                                      s->max_discharge);
}

/* ------------------------------------------------------------------------- */
/* public: free                                                              */
/* ------------------------------------------------------------------------- */
void cenv_fused_free(cenv_fused *f) {
  if (f == NULL) {
    return;
  }
#define FREE_P(p) free(f->p)
  FREE_P(zone_off);
  FREE_P(zone_cells);
  FREE_P(zone_of_cv);
  FREE_P(zone_area);
  FREE_P(mass);
  FREE_P(temp_out);
  FREE_P(coupled_vav);
  FREE_P(vav_zone);
  FREE_P(zone_vav_off);
  FREE_P(zone_vav);
  FREE_P(vav);
  FREE_P(ahu);
  FREE_P(occ_id);
  FREE_P(occ_all);
  FREE_P(rooms);
  FREE_P(ambient);
  FREE_P(comfort);
  FREE_P(conv_coef);
  FREE_P(feat_slot);
  FREE_P(feat);
  FREE_P(obs);
  FREE_P(gather);
  FREE_P(avg);
  FREE_P(vav_vtemp);
  FREE_P(zone_occ);
  FREE_P(q_zone);
  FREE_P(q_used);
  FREE_P(g);
  FREE_P(ts);
  FREE_P(sensed);
  FREE_P(air_q);
  FREE_P(coil);
  FREE_P(ahu_sum);
  FREE_P(ahu_area);
  FREE_P(ahu_flow_demand);
  FREE_P(recirc);
  FREE_P(rc_fsum);
  FREE_P(rc_ftotal);
  FREE_P(rc_asum);
  FREE_P(rc_atotal);
  FREE_P(obs_out);
  FREE_P(o_zone_temp);
  FREE_P(o_zone_heat);
  FREE_P(o_zone_cool);
  FREE_P(o_zone_flow_sp);
  FREE_P(o_zone_flow);
  FREE_P(o_zone_occ);
  FREE_P(o_ahu_intake);
  FREE_P(o_ahu_exhaust);
  FREE_P(o_ahu_cooling);
  FREE_P(o_vav_damper);
  FREE_P(o_vav_valve);
  FREE_P(o_vav_temp);
  FREE_P(o_vav_heat);
  FREE_P(o_vav_cool);
  FREE_P(o_vav_flow);
  FREE_P(o_vav_ratio);
  FREE_P(o_vav_mode);
  FREE_P(o_vav_prev);
  FREE_P(o_ahu_flow);
  FREE_P(o_ahu_requests);
  FREE_P(o_hws_fsum);
  FREE_P(o_hws_requests);
  FREE_P(o_hws_return);
  FREE_P(o_hws_heating);
  FREE_P(o_hws_pump);
  FREE_P(o_boiler_temp);
  FREE_P(o_boiler_change);
  FREE_P(o_boiler_dur);
  FREE_P(o_boiler_burner);
  FREE_P(o_flags);
  FREE_P(o_counts);
  FREE_P(o_diag);
#undef FREE_P
  free(f);
}

/* ------------------------------------------------------------------------- */
/* public: create                                                           */
/* ------------------------------------------------------------------------- */

/* A zone's mean temperature -- Building.get_zone_temp_stats' third return.
 * numpy gathers the zone's CVs in room order and reduces them with its
 * pairwise summation, which cenv_pairwise_sum reproduces exactly. */
static void zone_means(cenv_fused *f, const double *temp, double *out) {
  for (int32_t z = 0; z < f->n_zones; ++z) {
    const int32_t n = f->zone_off[z + 1] - f->zone_off[z];
    if (n <= 0) {
      out[z] = NAN;
      continue;
    }
    const int32_t *cells = f->zone_cells + f->zone_off[z];
    for (int32_t i = 0; i < n; ++i) {
      f->gather[i] = temp[cells[i]];
    }
    out[z] = cenv_pairwise_sum(f->gather, (int64_t)n) / (double)n;
  }
}

static double flat_mean(const cenv_fused *f, const double *temp) {
  return cenv_pairwise_sum(temp, (int64_t)f->n_full) / (double)f->n_full;
}

/* One VAV's sensing temperature from a zone-mean array: the area-weighted mean
 * of the zones it serves, which is what the Python reward path builds per VAV
 * (`_get_air_handler_reward_infos`: ``sum(avg_temps[z] * room_areas[z]) /
 * v_area`` over `get_zones_for_vav` order). Taking `avg` as an argument, not
 * `f->vav_vtemp`, is the point: the prelude needs the *pre-step* field and the
 * epilogue needs the *post-step* one, and `vav_vtemp` only ever holds the
 * prelude's. */
static double vav_sensing_temp(const cenv_fused *f, const double *avg,
                               int32_t v) {
  const fused_vav *s = &f->vav[v];
  double sum = 0.0;
  for (int32_t i = s->zoff; i < s->zoff + s->zn; ++i) {
    sum += avg[f->vav_zone[i]] * (double)f->zone_area[f->vav_zone[i]];
  }
  return sum / (double)s->area;
}

/* simulator_flexible_floor_plan's area-weighted recirculation temperature (the
 * shipped tree): every VAV counts with the floor area of its zones. */
static void recirc_area(cenv_fused *f, const double *avg, const double *temp,
                        double *out) {
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    f->ahu_sum[a] = 0.0;
    f->ahu_area[a] = 0.0;
  }
  for (int32_t v = 0; v < f->n_vav; ++v) {
    const fused_vav *s = &f->vav[v];
    if (s->zn <= 0 || s->area <= 0) {
      continue;
    }
    const double v_area = (double)s->area;
    f->ahu_sum[s->ahu] += vav_sensing_temp(f, avg, v) * v_area;
    f->ahu_area[s->ahu] += v_area;
  }
  const double mean_all = flat_mean(f, temp);
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    out[a] = (f->ahu_area[a] > 0.0) ? f->ahu_sum[a] / f->ahu_area[a] : mean_all;
  }
}

/* The corrected tree's _recirculation_temps: each VAV counts with the air it
 * actually moves, and falls back to area when no air moves at all. */
static void recirc_flow(cenv_fused *f, const double *avg, const double *temp,
                        double *out) {
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    f->rc_fsum[a] = 0.0;
    f->rc_ftotal[a] = 0.0;
    f->rc_asum[a] = 0.0;
    f->rc_atotal[a] = 0.0;
  }
  for (int32_t v = 0; v < f->n_vav; ++v) {
    const fused_vav *s = &f->vav[v];
    if (s->zn <= 0 || s->area <= 0) {
      continue;
    }
    const double v_area = (double)s->area;
    const double v_temp = vav_sensing_temp(f, avg, v);
    const double v_flow =
        (s->damper > 0.0 && s->max_flow > 0.0) ? vav_unrestricted_flow(f, v)
                                               : 0.0;
    f->rc_fsum[s->ahu] += v_temp * v_flow;
    f->rc_ftotal[s->ahu] += v_flow;
    f->rc_asum[s->ahu] += v_temp * v_area;
    f->rc_atotal[s->ahu] += v_area;
  }
  const double mean_all = flat_mean(f, temp);
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    if (f->rc_ftotal[a] > 0.0) {
      out[a] = f->rc_fsum[a] / f->rc_ftotal[a];
    } else if (f->rc_atotal[a] > 0.0) {
      out[a] = f->rc_asum[a] / f->rc_atotal[a];
    } else {
      out[a] = mean_all;
    }
  }
}

/* air_handler.AirHandler.add_demand. */
static void ahu_add_demand(fused_ahu *a, double flow) {
  a->air_flow += flow;
  if (a->air_flow > a->max_flow) {
    a->air_flow = a->max_flow;
  }
  a->requests += 1;
}

cenv_fused *cenv_fused_create(const cenv_fused_cfg *cfg) {
  if (cfg == NULL || cfg->nrows <= 0 || cfg->ncols <= 0 || cfg->n_zones < 0 ||
      cfg->n_vav < 0 || cfg->n_ahu <= 0 || cfg->n_obs < 0 ||
      cfg->n_horizon <= 0 || cfg->occ_handle <= 0) {
    return NULL;
  }
  if ((cfg->thermal == NULL) == (cfg->thermal_many == NULL)) {
    return NULL; /* exactly one of the two: single-environment or batched */
  }
  int t_full = 0, t_n = 0, t_k = 0;
  if (cfg->thermal != NULL) {
    if (cenv_thermal_sizes((const cenv_thermal *)cfg->thermal, &t_full, &t_n,
                           &t_k) != 0) {
      return NULL;
    }
  } else if (cenv_tbatch_sizes((const cenv_tbatch *)cfg->thermal_many, &t_full,
                               &t_n, &t_k, NULL) != 0) {
    return NULL;
  }
  if (t_full != cfg->nrows * cfg->ncols) {
    return NULL;
  }
  cenv_fused *f = (cenv_fused *)calloc(1, sizeof(cenv_fused));
  if (f == NULL) {
    return NULL;
  }
  f->nrows = cfg->nrows;
  f->ncols = cfg->ncols;
  f->n_full = t_full;
  f->n_rows = t_n;
  f->n_zones = cfg->n_zones;
  f->n_vav = cfg->n_vav;
  f->n_ahu = cfg->n_ahu;
  f->n_coupled = cfg->n_coupled;
  f->n_obs = cfg->n_obs;
  f->n_feat = cfg->n_feat;
  f->n_horizon = cfg->n_horizon;
  f->n_occ_all = cfg->n_occ_all;
  f->n_rooms = cfg->n_rooms;
  f->thermal = (cenv_thermal *)cfg->thermal;
  f->thermal_many = (cenv_tbatch *)cfg->thermal_many;
  f->occ = cfg->occ_handle;
  f->conv_rng = cfg->conv_rng;
  f->conv_p = cfg->conv_p;
  f->step_ns = cfg->step_ns;
  f->origin_ns = cfg->origin_ns;
  f->comfort_heat = cfg->comfort_heat;
  f->comfort_cool = cfg->comfort_cool;
  f->eco_heat = cfg->eco_heat;
  f->eco_cool = cfg->eco_cool;
  f->internal_gain = cfg->internal_gain_w_per_m2;
  f->standby_fraction = cfg->internal_gain_standby_fraction;
  f->occupant_heat = cfg->occupant_heat_w;
  f->cv_area_m2 = cfg->cv_area_m2;
  f->dt = cfg->time_step_sec;

  int32_t max_zone = 1;
  f->zone_off = fzalloc_i(cfg->n_zones + 1);
  f->zone_cells = fzalloc_i(cfg->zone_off[cfg->n_zones]);
  f->zone_of_cv = fzalloc_i(f->n_full);
  f->zone_area = fzalloc_i(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->mass = fzalloc_d(f->n_rows);
  f->temp_out = fzalloc_d(f->n_full);
  f->coupled_vav = fzalloc_i(cfg->n_coupled);
  f->vav_zone = fzalloc_i(cfg->n_vav_zones > 0 ? cfg->n_vav_zones : 1);
  f->zone_vav_off = fzalloc_i(cfg->n_zones + 1);
  f->zone_vav = fzalloc_i(cfg->n_zone_vav);
  f->vav = (fused_vav *)fzalloc((size_t)(cfg->n_vav > 0 ? cfg->n_vav : 1),
                                sizeof(fused_vav));
  f->ahu = (fused_ahu *)fzalloc((size_t)cfg->n_ahu, sizeof(fused_ahu));
  f->occ_id = fzalloc_i(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->occ_all = fzalloc_i(cfg->n_occ_all);
  f->rooms = (int64_t *)fzalloc((size_t)(cfg->n_rooms > 0 ? cfg->n_rooms : 1),
                                sizeof(int64_t));
  f->ambient = fzalloc_d(cfg->n_horizon);
  f->comfort = (uint8_t *)fzalloc((size_t)cfg->n_horizon, sizeof(uint8_t));
  f->conv_coef = fzalloc_d(cfg->n_horizon);
  f->feat_slot = fzalloc_i(cfg->n_feat);
  f->feat = (float *)fzalloc((size_t)cfg->n_feat * (size_t)cfg->n_horizon,
                             sizeof(float));
  f->obs = (cenv_obs_slot *)fzalloc((size_t)(cfg->n_obs > 0 ? cfg->n_obs : 1),
                                    sizeof(cenv_obs_slot));
  f->avg = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->vav_vtemp = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->zone_occ = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->q_zone = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->q_used = (uint8_t *)fzalloc((size_t)(cfg->n_zones > 0 ? cfg->n_zones : 1),
                                 sizeof(uint8_t));
  f->g = fzalloc_d(cfg->n_coupled);
  f->ts = fzalloc_d(cfg->n_coupled);
  f->sensed = fzalloc_d(cfg->n_coupled);
  f->air_q = fzalloc_d(cfg->n_coupled);
  f->coil = fzalloc_d(cfg->n_coupled);
  f->ahu_sum = fzalloc_d(cfg->n_ahu);
  f->ahu_area = fzalloc_d(cfg->n_ahu);
  f->ahu_flow_demand = fzalloc_d(cfg->n_ahu);
  f->recirc = fzalloc_d(cfg->n_ahu);
  f->rc_fsum = fzalloc_d(cfg->n_ahu);
  f->rc_ftotal = fzalloc_d(cfg->n_ahu);
  f->rc_asum = fzalloc_d(cfg->n_ahu);
  f->rc_atotal = fzalloc_d(cfg->n_ahu);
  f->obs_out = (float *)fzalloc((size_t)(cfg->n_obs > 0 ? cfg->n_obs : 1),
                                sizeof(float));
  f->o_zone_temp = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->o_zone_heat = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->o_zone_cool = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->o_zone_flow_sp = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->o_zone_flow = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->o_zone_occ = fzalloc_d(cfg->n_zones > 0 ? cfg->n_zones : 1);
  f->o_ahu_intake = fzalloc_d(cfg->n_ahu);
  f->o_ahu_exhaust = fzalloc_d(cfg->n_ahu);
  f->o_ahu_cooling = fzalloc_d(cfg->n_ahu);
  f->o_vav_damper = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_valve = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_temp = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_heat = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_cool = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_flow = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_ratio = fzalloc_d(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_mode = fzalloc_i(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_vav_prev = fzalloc_i(cfg->n_vav > 0 ? cfg->n_vav : 1);
  f->o_ahu_flow = fzalloc_d(cfg->n_ahu);
  f->o_ahu_requests = fzalloc_i(cfg->n_ahu);
  f->o_hws_fsum = fzalloc_d(1);
  f->o_hws_requests = fzalloc_i(1);
  f->o_hws_return = fzalloc_d(1);
  f->o_hws_heating = fzalloc_d(1);
  f->o_hws_pump = fzalloc_d(1);
  f->o_boiler_temp = fzalloc_d(1);
  f->o_boiler_change = fzalloc_d(1);
  f->o_boiler_dur = fzalloc_d(1);
  f->o_boiler_burner = fzalloc_d(1);
  f->o_flags = fzalloc_i(1);
  f->o_counts = fzalloc_i(4);
  f->o_diag = fzalloc_d(4);

  if (f->zone_off == NULL || f->zone_area == NULL || f->vav == NULL ||
      f->ahu == NULL || f->obs == NULL || f->ambient == NULL ||
      f->comfort == NULL || f->feat == NULL || f->obs_out == NULL ||
      (cfg->n_zones > 0 && (f->zone_cells == NULL || f->avg == NULL ||
                            f->vav_vtemp == NULL || f->zone_occ == NULL)) ||
      (cfg->n_coupled > 0 && (f->g == NULL || f->ts == NULL)) ||
      (cfg->n_vav > 0 && f->vav_zone == NULL) || f->temp_out == NULL) {
    cenv_fused_free(f);
    return NULL;
  }

  copy_i32(f->coupled_vav, cfg->coupled_vav, cfg->n_coupled);
  copy_i32(f->zone_off, cfg->zone_off, cfg->n_zones + 1);
  copy_i32(f->zone_cells, cfg->zone_cells, f->zone_off[cfg->n_zones]);
  copy_i32(f->zone_of_cv, cfg->zone_of_cv, f->n_full);
  copy_i32(f->zone_area, cfg->zone_area, cfg->n_zones);
  for (int32_t v = 0; v < cfg->n_vav; ++v) {
    const cenv_vav_cfg *c = &cfg->vavs[v];
    fused_vav *s = &f->vav[v];
    s->max_flow = c->max_air_flow_rate;
    s->min_damper = c->min_damper;
    s->max_discharge = c->max_discharge_temp;
    s->design_static = c->max_flow_static_pressure;
    s->reheat_max_flow = c->reheat_max_flow_factor;
    s->ahu = c->ahu;
    s->zoff = c->zone_off;
    s->zn = c->zone_n;
    s->area = c->area;
    s->area_all = c->area_all;
    s->coupled = c->coupled;
    s->damper = c->min_damper;
    s->valve = 0.0;
    s->ratio = 1.0;
    s->mode = MODE_OFF;
    s->prev_step = -1;
    if (c->zone_n > 0) {
      copy_i32(f->vav_zone + c->zone_off, cfg->vav_zone + c->zone_off,
               c->zone_n);
    }
  }
  for (int32_t i = 0; i < f->n_zones; ++i) {
    const int32_t a = f->zone_area[i];
    if (a > max_zone) {
      max_zone = a;
    }
  }
  f->gather = fzalloc_d(max_zone);
  if (f->gather == NULL) {
    cenv_fused_free(f);
    return NULL;
  }
  copy_i32(f->zone_vav_off, cfg->zone_vav_off, cfg->n_zones + 1);
  copy_i32(f->zone_vav, cfg->zone_vav, cfg->zone_vav_off[cfg->n_zones]);
  copy_i32(f->occ_id, cfg->occ_zone_id, cfg->n_zones);
  copy_i32(f->occ_all, cfg->occ_all, cfg->n_occ_all);
  if (cfg->n_rooms > 0) {
    memcpy(f->rooms, cfg->rooms, (size_t)cfg->n_rooms * sizeof(int64_t));
  }
  memcpy(f->ambient, cfg->ambient, (size_t)cfg->n_horizon * sizeof(double));
  memcpy(f->comfort, cfg->comfort, (size_t)cfg->n_horizon * sizeof(uint8_t));
  memcpy(f->conv_coef, cfg->conv_coef, (size_t)cfg->n_horizon * sizeof(double));
  copy_i32(f->feat_slot, cfg->feat_slot, cfg->n_feat);
  memcpy(f->feat, cfg->feat,
         (size_t)cfg->n_feat * (size_t)cfg->n_horizon * sizeof(float));
  memcpy(f->obs, cfg->obs, (size_t)cfg->n_obs * sizeof(cenv_obs_slot));

  for (int32_t a = 0; a < cfg->n_ahu; ++a) {
    const cenv_ahu_cfg *c = &cfg->ahus[a];
    fused_ahu *s = &f->ahu[a];
    s->run = c->run_command;
    s->recirc = c->recirculation;
    s->supply_sp = c->supply_setpoint;
    s->fan_sp = c->fan_static_pressure;
    s->fan_eff = c->fan_efficiency;
    s->max_flow = c->max_air_flow_rate;
    s->min_sp = c->min_static_pressure;
    s->max_recirc = c->max_recirculation;
    s->cop = c->cooling_cop;
  }

  f->hws.energy_balance = cfg->hws_energy_balance;
  f->hws.has_tank = cfg->hws_has_tank;
  f->hws.run = cfg->hws_run;
  f->hws.pump_run = cfg->pump_run;
  f->hws.header_resistance = cfg->hws_header_resistance;
  f->hws.heating_rate = cfg->boiler_heating_rate;
  f->hws.cooling_rate = cfg->boiler_cooling_rate;
  f->hws.conv_coef = cfg->boiler_convection_coef;
  f->hws.tank_len = cfg->boiler_tank_length;
  f->hws.tank_radius = cfg->boiler_tank_radius;
  f->hws.water_capacity = cfg->boiler_water_capacity;
  f->hws.insul_cond = cfg->boiler_insulation_conductivity;
  f->hws.insul_thick = cfg->boiler_insulation_thickness;
  f->hws.setpoint = cfg->boiler_setpoint;
  f->hws.current = cfg->boiler_setpoint;
  f->hws.return_temp = cfg->boiler_return_temp;
  f->hws.pump_head = cfg->pump_head;
  f->hws.pump_eff = cfg->pump_efficiency;
  f->hws.ts_action = -1;
  f->hws.ts_obs = 0;
  f->hws.dp = pump_dp(&f->hws);

  f->o_counts[0] = f->n_zones;
  f->o_counts[1] = f->n_vav;
  f->o_counts[2] = f->n_ahu;
  f->o_counts[3] = f->n_coupled;
  return f;
}

int32_t cenv_fused_views_get(cenv_fused *f, cenv_fused_views *out) {
  if (f == NULL || out == NULL) {
    return 2;
  }
  out->obs = f->obs_out;
  out->mass = f->mass;
  out->zone_temp = f->o_zone_temp;
  out->zone_heat_sp = f->o_zone_heat;
  out->zone_cool_sp = f->o_zone_cool;
  out->zone_flow_sp = f->o_zone_flow_sp;
  out->zone_flow = f->o_zone_flow;
  out->zone_occupancy = f->o_zone_occ;
  out->ahu_intake = f->o_ahu_intake;
  out->ahu_exhaust = f->o_ahu_exhaust;
  out->ahu_cooling = f->o_ahu_cooling;
  out->vav_damper = f->o_vav_damper;
  out->vav_valve = f->o_vav_valve;
  out->vav_zone_temp = f->o_vav_temp;
  out->vav_heat_sp = f->o_vav_heat;
  out->vav_cool_sp = f->o_vav_cool;
  out->vav_flow = f->o_vav_flow;
  out->vav_ratio = f->o_vav_ratio;
  out->vav_mode = f->o_vav_mode;
  out->vav_prev = f->o_vav_prev;
  out->ahu_flow = f->o_ahu_flow;
  out->ahu_requests = f->o_ahu_requests;
  out->hws_flow_factor_sum = f->o_hws_fsum;
  out->hws_requests = f->o_hws_requests;
  out->hws_return = f->o_hws_return;
  out->hws_heating = f->o_hws_heating;
  out->hws_pump = f->o_hws_pump;
  out->boiler_temp = f->o_boiler_temp;
  out->boiler_change = f->o_boiler_change;
  out->boiler_last_dur = f->o_boiler_dur;
  out->boiler_burner = f->o_boiler_burner;
  out->flags = f->o_flags;
  out->diag = f->o_diag;
  out->phase = f->phase;
  out->n_phase = &f->n_phase;
  out->counts = f->o_counts;
  return 0;
}

int32_t cenv_fused_sync_full(
    cenv_fused *f, const double *vav_damper, const double *vav_valve,
    const double *vav_zone_temp, const double *vav_heat_sp,
    const double *vav_cool_sp, const double *vav_ratio, const int32_t *vav_mode,
    const int32_t *vav_prev, const double *ahu_air_flow,
    const int32_t *ahu_requests, double hws_flow_factor_sum, int32_t hws_requests,
    double boiler_temp, double boiler_setpoint, double boiler_return_temp,
    double boiler_change, double boiler_last_dur, double boiler_burner,
    int64_t ts_obs, int64_t ts_action) {
  if (f == NULL) {
    return 2;
  }
  for (int32_t v = 0; v < f->n_vav; ++v) {
    fused_vav *s = &f->vav[v];
    if (vav_damper != NULL) s->damper = vav_damper[v];
    if (vav_valve != NULL) s->valve = vav_valve[v];
    if (vav_zone_temp != NULL) s->zone_temp = vav_zone_temp[v];
    if (vav_heat_sp != NULL) s->heat_sp = vav_heat_sp[v];
    if (vav_cool_sp != NULL) s->cool_sp = vav_cool_sp[v];
    if (vav_ratio != NULL) s->ratio = vav_ratio[v];
    if (vav_mode != NULL) s->mode = vav_mode[v];
    if (vav_prev != NULL) s->prev_step = vav_prev[v];
  }
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    if (ahu_air_flow != NULL) f->ahu[a].air_flow = ahu_air_flow[a];
    if (ahu_requests != NULL) f->ahu[a].requests = ahu_requests[a];
  }
  f->hws.flow_factor_sum = hws_flow_factor_sum;
  f->hws.requests = hws_requests;
  f->hws.current = boiler_temp;
  f->hws.setpoint = boiler_setpoint;
  f->hws.return_temp = boiler_return_temp;
  f->hws.step_change = boiler_change;
  f->hws.last_dur = boiler_last_dur;
  f->hws.burner = boiler_burner;
  f->hws.ts_obs = ts_obs;
  f->hws.ts_action = ts_action;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* observation slot values                                                   */
/* ------------------------------------------------------------------------- */
static double obs_raw(cenv_fused *f, const cenv_obs_slot *s, int32_t j,
                      const double ambient1) {
  switch (s->kind) {
    case CENV_OBS_VAV_DAMPER:
      return f->vav[s->index].damper;
    case CENV_OBS_VAV_VALVE:
      return f->vav[s->index].valve;
    case CENV_OBS_VAV_MAX_FLOW:
      return f->vav[s->index].max_flow;
    case CENV_OBS_VAV_SETPOINT_HEAT:
      return f->vav[s->index].heat_sp;
    case CENV_OBS_VAV_SETPOINT_COOL:
      return f->vav[s->index].cool_sp;
    case CENV_OBS_VAV_ZONE_TEMP:
      return f->vav[s->index].zone_temp;
    case CENV_OBS_AHU_COOLING_REQUESTS:
      return (double)f->ahu[s->index].requests;
    case CENV_OBS_AHU_STATIC_SETPOINT:
      return ahu_static_pressure(&f->ahu[s->index]);
    case CENV_OBS_AHU_FLOW:
      return ahu_air_flow(&f->ahu[s->index]);
    case CENV_OBS_AHU_SUPPLY_TEMP_SENSOR:
    case CENV_OBS_AHU_SUPPLY_TEMP_SETPOINT:
      return f->ahu[s->index].supply_sp;
    case CENV_OBS_AHU_RUN_COMMAND:
      return (double)f->ahu[s->index].run;
    case CENV_OBS_AHU_FAN_SPEED:
      return f->ahu[s->index].air_flow / f->ahu[s->index].max_flow;
    case CENV_OBS_AHU_OUTSIDE_FLOW:
      return (1.0 - f->ahu[s->index].recirc) * f->ahu[s->index].air_flow;
    case CENV_OBS_AHU_RECIRC:
      return f->ahu[s->index].recirc;
    case CENV_OBS_AHU_OUTSIDE_TEMP:
      return ambient1;
    case CENV_OBS_HWS_DIFFP_SENSOR:
    case CENV_OBS_HWS_DIFFP_SETPOINT:
      return pump_dp(&f->hws);
    case CENV_OBS_HWS_HEATING_REQUESTS:
      return (double)f->hws.requests;
    case CENV_OBS_HWS_RUN_STATUS:
      return (double)f->hws.run;
    case CENV_OBS_HWS_SUPPLY_WATER_TEMP:
      /* The one sensor read in the observation with a side effect, and the
       * only one whose elapsed time is non-zero: the observation is at the
       * post-step timestamp, the action at the pre-step one. */
      f->hws.ts_obs = j;
      return hws_supply_sensor(f);
    case CENV_OBS_HWS_SUPPLY_WATER_SETPOINT:
      return f->hws.setpoint;
    default:
      (void)j;
      return 0.0;
  }
}

/* ------------------------------------------------------------------------- */
/* the step, in the slices a batch needs                                     */
/* ------------------------------------------------------------------------- */
/* `cenv_fused_run` is one step for one environment. A batch of M environments
 * of the same scenario runs the same slices with the thermal solve replaced by
 * one batched call (batch.c, through the cenv_fused_batch_* entry points at
 * the end of this file):
 *
 *   prelude   items 1-10 of fused.py's map: the device maths, the loads and
 *             the assembled input_q (writes input_q)
 *   thermal   item 11, the finite-difference solve, plus Q^T of its result
 *   foldback  item 12: the air stream folded back into input_q
 *   epilogue  items 13-19: convection, return water, tank, clip, observation,
 *             reward info, device state (mutates temp)
 *
 * Splitting there is the only structure batching needs: the slices already
 * talk to each other through the handle (f->total_reheat_w, f->coil,
 * f->q_zone, f->g, f->ts, f->mass), so a driver can run the prelude for every
 * environment, solve once for all of them, then fold back and finish each.
 *
 * The per-phase CPU timers (fused.h) stay where they were; the batching driver
 * only moves the thermal phase out of the handle (its time is measured by the
 * batched kernel, cenv_tbatch_phase).
 */
#define PH(idx)                             \
  do {                                      \
    const double ph_now = fused_clock();    \
    f->phase[idx] += ph_now - *ph_mark;     \
    *ph_mark = ph_now;                      \
  } while (0)

static int32_t fused_prelude(cenv_fused *f, int32_t step, const double *temp,
                             double *input_q, const double *sync,
                             double *ambient_out, double *ph_mark) {
  const int32_t comfort = f->comfort[step] ? 1 : 0;
  const double ambient = f->ambient[step];
  *ambient_out = ambient;

  /* -- sync: the commands the action request just applied ----------------- */
  const double *cmd_damper = sync;
  const double *cmd_valve = sync + f->n_vav;
  const double *cmd_ahu = cmd_valve + f->n_vav;
  const double *cmd_hws = cmd_ahu + 4 * f->n_ahu;
  for (int32_t v = 0; v < f->n_vav; ++v) {
    f->vav[v].dcmd = cmd_damper[v];
    f->vav[v].vcmd = cmd_valve[v];
  }
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    fused_ahu *d = &f->ahu[a];
    const double *c = cmd_ahu + 4 * a;
    d->run = (c[0] != 0.0) ? 1 : 0;
    d->supply_sp = c[1];
    d->fan_sp = c[2];
    d->recirc = c[3];
  }
  f->hws.run = (cmd_hws[0] != 0.0) ? 1 : 0;
  f->hws.setpoint = cmd_hws[1];
  f->hws.pump_head = cmd_hws[2];
  f->hws.pump_run = (cmd_hws[3] != 0.0) ? 1 : 0;
  f->hws.ts_obs = (int64_t)cmd_hws[4];
  if ((int64_t)cmd_hws[5] != -2) {
    /* -2 keeps the kernel's own latch (the Python object has no timestamp). */
    f->hws.ts_action = (int64_t)cmd_hws[5];
  }
  f->hws.dp = pump_dp(&f->hws);

  /* -- 1. zone average temperatures -------------------------------------- */
  zone_means(f, temp, f->avg);
  PH(CENV_PH_ZONE_MEANS);

  /* -- 2. per-VAV thermostat, damper and reheat valve -------------------- */
  for (int32_t v = 0; v < f->n_vav; ++v) {
    fused_vav *s = &f->vav[v];
    if (s->zn <= 0) {
      continue;
    }
    double sum = 0.0;
    for (int32_t i = s->zoff; i < s->zoff + s->zn; ++i) {
      sum += f->avg[f->vav_zone[i]] * (double)f->zone_area[f->vav_zone[i]];
    }
    const double v_temp = sum / (double)s->area;
    f->vav_vtemp[v] = v_temp;
    s->zone_temp = v_temp;

    const double heat_sp = comfort ? f->comfort_heat : f->eco_heat;
    const double cool_sp = comfort ? f->comfort_cool : f->eco_cool;
    s->heat_sp = heat_sp;
    s->cool_sp = cool_sp;
    const int32_t prev_valid = (s->prev_step >= 0 && s->prev_step < step) ? 1 : 0;
    const int32_t prev_comfort =
        prev_valid ? (int32_t)f->comfort[s->prev_step] : 0;
    const int32_t mode = thermostat_update(
        s, v_temp, comfort, prev_valid, prev_comfort, heat_sp, cool_sp);
    s->prev_step = step;

    double damper;
    double valve;
    if (mode == MODE_HEAT) {
      damper = 1.0;
      valve = 1.0;
    } else if (mode == MODE_COOL) {
      damper = 1.0;
      valve = 0.0;
    } else if (mode == MODE_OFF || mode == MODE_PASSIVE_COOL) {
      damper = s->min_damper;
      valve = 0.0;
    } else {
      damper = s->damper;
      valve = s->valve;
    }
    if (!isnan(s->dcmd)) {
      damper = s->dcmd;
    }
    if (!isnan(s->vcmd)) {
      valve = s->vcmd;
    }
    s->damper = (damper > s->min_damper) ? damper : s->min_damper;
    s->valve = valve;
  }
  PH(CENV_PH_VAV_UPDATE);

  /* -- 3. recirculation temperature per air handler ---------------------- */
  if (f->hws.energy_balance) {
    recirc_flow(f, f->avg, temp, f->recirc);
  } else {
    recirc_area(f, f->avg, temp, f->recirc);
  }
  PH(CENV_PH_RECIRC);

  /* -- 4. supply air temperature per air handler ------------------------ */
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    fused_ahu *d = &f->ahu[a];
    d->recirc_temp = f->recirc[a];
    d->supply_temp = cenv_hvac_supply_air_temp(d->recirc, f->recirc[a], ambient,
                                               d->supply_sp, d->run);
  }

  /* -- 5. reset the demand accumulators --------------------------------- */
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    f->ahu[a].air_flow = 0.0;
    f->ahu[a].requests = 0;
  }
  f->hws.flow_factor_sum = 0.0;
  f->hws.requests = 0;
  for (int32_t v = 0; v < f->n_vav; ++v) {
    const fused_vav *s = &f->vav[v];
    f->hws.flow_factor_sum += s->valve * s->reheat_max_flow;
    f->hws.requests += 1;
  }

  /* -- 6. the handler's flow cap, before anything is charged for air ----- */
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    f->ahu_flow_demand[a] = 0.0;
  }
  for (int32_t v = 0; v < f->n_vav; ++v) {
    f->ahu_flow_demand[f->vav[v].ahu] += vav_unrestricted_flow(f, v);
  }
  for (int32_t v = 0; v < f->n_vav; ++v) {
    fused_vav *s = &f->vav[v];
    const double demanded = f->ahu_flow_demand[s->ahu];
    s->ratio = (demanded > 0.0) ? fmin(1.0, f->ahu[s->ahu].max_flow / demanded)
                                : 1.0;
    s->flow = vav_unrestricted_flow(f, v) * s->ratio;
  }
  PH(CENV_PH_AHU_DEMAND);

  /* -- 7. air-side terms, reheat and per-zone loads --------------------- */
  for (int32_t c = 0; c < f->n_coupled; ++c) {
    f->g[c] = 0.0;
    f->ts[c] = 0.0;
  }
  for (int32_t z = 0; z < f->n_zones; ++z) {
    f->q_zone[z] = 0.0;
    f->q_used[z] = 0;
  }
  double total_reheat_w = 0.0;
  f->coil_n = 0;
  for (int32_t c = 0; c < f->n_coupled; ++c) {
    const int32_t v = f->coupled_vav[c];
    fused_vav *s = &f->vav[v];
    if (s->zn <= 0) {
      continue;
    }
    const double current_supply_temp = f->ahu[s->ahu].supply_temp;
    f->ts[c] = current_supply_temp;
    /* hws.supply_water_temperature_sensor -- advances the shipped tank. */
    const double water_temp = hws_supply_sensor(f);
    const double reheat_w =
        vav_reheat_energy(f, s, current_supply_temp, water_temp);
    total_reheat_w += reheat_w;
    f->coil[f->coil_n++] = current_supply_temp;
    if (s->damper > 0.0 && s->max_flow > 0.0) {
      f->g[c] = s->flow * FUSED_AIR_DENSITY * FUSED_AIR_CP;
      for (int32_t i = s->zoff; i < s->zoff + s->zn; ++i) {
        const int32_t z = f->vav_zone[i];
        f->q_zone[z] += reheat_w * (double)f->zone_area[z] / (double)s->area;
        f->q_used[z] = 1;
      }
    }
    if (s->flow > 0.0) {
      ahu_add_demand(&f->ahu[s->ahu], s->flow);
    }
  }
  PH(CENV_PH_COUPLED);

  /* -- 8. internal gains (lighting, equipment) -------------------------- */
  if (f->internal_gain > 0.0) {
    const double fraction = comfort ? 1.0 : f->standby_fraction;
    const double watts_per_cv = f->internal_gain * f->cv_area_m2 * fraction;
    for (int32_t z = 0; z < f->n_zones; ++z) {
      f->q_zone[z] += watts_per_cv * (double)f->zone_area[z];
      f->q_used[z] = 1;
    }
  }

  /* -- 9. occupant gains
  PH(CENV_PH_GAINS);, on the same interval the reward asks for ------- */
  if (f->occupant_heat > 0.0) {
    const int64_t start_ns = f->origin_ns + (int64_t)step * f->step_ns;
    const int64_t end_ns = start_ns + f->step_ns;
    for (int32_t z = 0; z < f->n_zones; ++z) {
      double occ = 0.0;
      if (cenv_occ_average(f->occ, f->occ_id[z], start_ns, end_ns, &occ) != 0) {
        return 5;
      }
      f->zone_occ[z] = occ;
      f->q_zone[z] += occ * f->occupant_heat;
      f->q_used[z] = 1;
    }
  }

  /* -- 10. apply_thermal_power_zone for the assembled loads -------------- */
  for (int32_t z = 0; z < f->n_zones; ++z) {
    if (!f->q_used[z] || f->zone_area[z] <= 0) {
      continue;
    }
    const double share = f->q_zone[z] / (double)f->zone_area[z];
    const int32_t *cells = f->zone_cells + f->zone_off[z];
    for (int32_t i = f->zone_off[z]; i < f->zone_off[z + 1]; ++i) {
      input_q[cells[i - f->zone_off[z]]] = share;
    }
  }
  f->total_reheat_w = total_reheat_w;
  return 0;
}

static int32_t fused_thermal_single(cenv_fused *f, const double *temp,
                                    const double *input_q, double ambient,
                                    double *ph_mark) {
  /* -- 11. the finite-difference step ----------------------------------- */
  if (f->thermal == NULL) {
    return 1; /* batch mode: the batch driver runs the batched solve */
  }
  {
    const double *g = (f->n_coupled > 0) ? f->g : NULL;
    const double *ts = (f->n_coupled > 0) ? f->ts : NULL;
    const int32_t rc = cenv_thermal_step(
        f->thermal, temp, input_q, ambient, g, ts, f->temp_out, f->mass, NULL);
    if (rc != 0) {
      return 4;
    }
    memcpy((void *)temp, f->temp_out, (size_t)f->n_full * sizeof(double));
    if (f->n_coupled > 0) {
      /* What the foldback needs: Q^T of the field just solved for (and not
       * Q^T y + S q -- see thermal.h's cenv_thermal_sense_last). */
      if (cenv_thermal_sense_last(f->thermal, f->sensed) != 0) {
        return 4;
      }
    }
  }
  PH(CENV_PH_THERMAL);
  return 0;
}

static int32_t fused_foldback(cenv_fused *f, const double *sensed,
                              double *input_q) {
  /* -- 12. fold the air stream back into input_q ------------------------ */
  if (f->n_coupled > 0) {
    for (int32_t c = 0; c < f->n_coupled; ++c) {
      f->air_q[c] = f->g[c] * (f->ts[c] - sensed[c]);
    }
    for (int32_t c = 0; c < f->n_coupled; ++c) {
      const int32_t v = f->coupled_vav[c];
      const fused_vav *s = &f->vav[v];
      if (s->zn <= 0) {
        continue;
      }
      for (int32_t i = s->zoff; i < s->zoff + s->zn; ++i) {
        const int32_t z = f->vav_zone[i];
        f->q_zone[z] += f->air_q[c] * (double)f->zone_area[z] / (double)s->area;
        f->q_used[z] = 1;
      }
    }
    for (int32_t z = 0; z < f->n_zones; ++z) {
      if (!f->q_used[z] || f->zone_area[z] <= 0) {
        continue;
      }
      const double share = f->q_zone[z] / (double)f->zone_area[z];
      const int32_t *cells = f->zone_cells + f->zone_off[z];
      for (int32_t i = f->zone_off[z]; i < f->zone_off[z + 1]; ++i) {
        input_q[cells[i - f->zone_off[z]]] = share;
      }
    }
  }
  return 0;
}

static int32_t fused_epilogue(cenv_fused *f, int32_t step, double *temp,
                              double *ph_mark) {
  const int32_t j = step + 1;
  const int32_t comfort1 = f->comfort[j] ? 1 : 0;
  const double ambient = f->ambient[step];
  const double ambient1 = f->ambient[j];

  /* -- 13. convection
  PH(CENV_PH_AIR_FOLDBACK); -------------------------------------------------- */
  if (f->n_rooms > 0 && f->conv_p > 0.0) {
    if (cenv_conv_apply(f->rooms, f->n_rooms, f->conv_rng, temp, f->nrows,
                        f->ncols, f->conv_p) != 0) {
      return 6;
    }
  }
  PH(CENV_PH_CONVECTION);

  /* -- 14. return water temperature from the coils --------------------- */
  {
    const double water_capacity_rate =
        FUSED_WATER_CP * FUSED_WATER_DENSITY * hws_total_flow(&f->hws);
    if (!(water_capacity_rate <= 0.0) && f->coil_n > 0) {
      const double supply_temp = hws_supply_sensor(f);
      double return_temp = supply_temp - f->total_reheat_w / water_capacity_rate;
      double air_temp = f->coil[0];
      if (f->total_reheat_w >= 0.0) {
        for (int32_t i = 1; i < f->coil_n; ++i) {
          if (f->coil[i] < air_temp) {
            air_temp = f->coil[i];
          }
        }
      } else {
        for (int32_t i = 1; i < f->coil_n; ++i) {
          if (f->coil[i] > air_temp) {
            air_temp = f->coil[i];
          }
        }
      }
      const double low =
          (supply_temp < air_temp) ? supply_temp : air_temp;
      const double high =
          (supply_temp > air_temp) ? supply_temp : air_temp;
      if (return_temp < low) {
        return_temp = low;
      }
      if (return_temp > high) {
        return_temp = high;
      }
      f->hws.return_temp = return_temp;
      f->o_diag[0] = f->total_reheat_w;
      f->o_diag[1] = water_capacity_rate;
      f->o_diag[2] = supply_temp;
      f->o_diag[3] = (double)f->coil_n;
    }
  }
  PH(CENV_PH_RETURN_WATER);

  /* -- 15. corrected tree: take the step's heat out of the tank ---------- */
  if (f->hws.energy_balance) {
    const double coil_heat_w =
        FUSED_WATER_CP * FUSED_WATER_DENSITY * hws_total_flow(&f->hws) *
        (hws_supply_sensor(f) - f->hws.return_temp);
    f->hws.burner = cenv_hvac_tank_energy_balance(
        &f->hws.current, coil_heat_w, ambient, f->dt, f->hws.setpoint,
        f->hws.water_capacity, FUSED_WATER_DENSITY, FUSED_WATER_CP,
        f->hws.heating_rate, f->hws.tank_len, f->hws.tank_radius,
        f->hws.insul_thick, f->hws.insul_cond, f->hws.conv_coef);
  }
  PH(CENV_PH_HEAT_DRAW);

  /* -- 16. the solver-divergence guard --------------------------------- */
  {
    double lo = temp[0];
    double hi = temp[0];
    for (int32_t i = 1; i < f->n_full; ++i) {
      if (temp[i] < lo) lo = temp[i];
      if (temp[i] > hi) hi = temp[i];
    }
    if (lo < 150.0 || hi > 500.0) {
      for (int32_t i = 0; i < f->n_full; ++i) {
        if (temp[i] < 150.0) {
          temp[i] = 150.0;
        } else if (temp[i] > 500.0) {
          temp[i] = 500.0;
        }
      }
      f->o_flags[0] |= FUSED_FLAG_CLIPPED;
    }
  }
  PH(CENV_PH_CLIP);

  /* -- 17. the observation, at the post-step timestamp ------------------ */
  for (int32_t k = 0; k < f->n_obs; ++k) {
    const cenv_obs_slot *s = &f->obs[k];
    if (s->kind == CENV_OBS_AUX_COMFORT_NOW ||
        s->kind == CENV_OBS_AUX_COMFORT_SOON ||
        s->kind == CENV_OBS_AUX_NUM_OCCUPANTS) {
      continue;
    }
    /* The response's `continuous_value` is a protobuf float: the simulator
     * rounds the raw measurement to float32 before the normaliser sees it, so
     * the kernel does too (the remaining arithmetic is float64, as in
     * StandardScoreObservationNormalizer._normalize_one). */
    const float raw32 = (float)obs_raw(f, s, j, ambient1);
    f->obs_out[s->slot] =
        (s->scale > 0.0) ? (float)(((double)raw32 - s->mean) / s->scale) : 0.0f;
  }
  for (int32_t i = 0; i < f->n_feat; ++i) {
    f->obs_out[f->feat_slot[i]] = f->feat[(size_t)i * f->n_horizon + j];
  }
  for (int32_t k = 0; k < f->n_obs; ++k) {
    const cenv_obs_slot *s = &f->obs[k];
    double value;
    if (s->kind == CENV_OBS_AUX_COMFORT_NOW) {
      value = (double)comfort1;
    } else if (s->kind == CENV_OBS_AUX_COMFORT_SOON) {
      value = (double)f->comfort[j + 12 < f->n_horizon ? j + 12 : f->n_horizon - 1];
    } else if (s->kind == CENV_OBS_AUX_NUM_OCCUPANTS) {
      /* building.num_occupants: the sum over every zone of its occupancy over
       * the last five minutes (the interval the step already queried), then
       * int(). */
      const int64_t start_ns =
          f->origin_ns + (int64_t)(step + 1) * f->step_ns - 300000000000LL;
      const int64_t end_ns = f->origin_ns + (int64_t)(step + 1) * f->step_ns;
      double total = 0.0;
      for (int32_t i = 0; i < f->n_occ_all; ++i) {
        double occ = 0.0;
        if (cenv_occ_average(f->occ, f->occ_all[i], start_ns, end_ns, &occ) !=
            0) {
          return 5;
        }
        total += occ;
      }
      value = (double)(int64_t)total;
    } else {
      continue;
    }
    f->obs_out[s->slot] =
        (s->scale > 0.0) ? (float)((value - s->mean) / s->scale) : 0.0f;
  }

  PH(CENV_PH_OBSERVATION);

  /* -- 18. reward info numbers at the post-step timestamp --------------- */
  zone_means(f, temp, f->avg);
  const double heat_j = comfort1 ? f->comfort_heat : f->eco_heat;
  const double cool_j = comfort1 ? f->comfort_cool : f->eco_cool;
  {
    const int64_t t1 = f->origin_ns + (int64_t)j * f->step_ns;
    const int64_t t2 = t1 + f->step_ns;
    for (int32_t z = 0; z < f->n_zones; ++z) {
      f->o_zone_temp[z] = f->avg[z];
      f->o_zone_heat[z] = heat_j;
      f->o_zone_cool[z] = cool_j;
      double flow_sp = 0.0;
      double flow = 0.0;
      for (int32_t i = f->zone_vav_off[z]; i < f->zone_vav_off[z + 1]; ++i) {
        const int32_t v = f->zone_vav[i];
        const double share =
            (double)f->zone_area[z] / (double)f->vav[v].area_all;
        flow_sp += f->vav[v].max_flow * share;
        flow += f->vav[v].flow * share;
      }
      f->o_zone_flow_sp[z] = flow_sp;
      f->o_zone_flow[z] = flow;
      double occ = 0.0;
      if (cenv_occ_average(f->occ, f->occ_id[z], t1, t2, &occ) != 0) {
        return 5;
      }
      f->o_zone_occ[z] = occ;
    }
  }
  if (f->hws.energy_balance) {
    recirc_flow(f, f->avg, temp, f->recirc);
  } else {
    recirc_area(f, f->avg, temp, f->recirc);
  }
  for (int32_t a = 0; a < f->n_ahu; ++a) {
    const fused_ahu *d = &f->ahu[a];
    const double flow = ahu_air_flow(d);
    const double sp = ahu_static_pressure(d);
    f->o_ahu_intake[a] = cenv_hvac_fan_power(flow, sp, d->fan_eff);
    f->o_ahu_exhaust[a] =
        cenv_hvac_fan_power(flow * (1.0 - d->recirc), sp, d->fan_eff);
    const double thermal =
        cenv_hvac_ahu_thermal_rate(flow, d->recirc, f->recirc[a], ambient1,
                                   d->supply_sp, FUSED_AIR_DENSITY, FUSED_AIR_CP,
                                   d->run);
    f->o_ahu_cooling[a] = thermal / d->cop;
    f->o_ahu_flow[a] = d->air_flow;
    f->o_ahu_requests[a] = d->requests;
  }
  {
    const double flow = hws_total_flow(&f->hws);
    if (f->hws.energy_balance) {
      f->o_hws_heating[0] = f->hws.burner;
    } else {
      f->o_hws_heating[0] = cenv_hvac_boiler_thermal_rate(
          f->hws.setpoint, f->hws.return_temp, ambient1, flow,
          FUSED_WATER_DENSITY, FUSED_WATER_CP, f->hws.water_capacity,
          f->hws.step_change, f->hws.last_dur, f->hws.tank_len,
          f->hws.tank_radius, f->hws.insul_thick, f->hws.insul_cond,
          f->hws.conv_coef);
    }
    const double head = f->hws.pump_run ? f->hws.pump_head : 0.0;
    f->o_hws_pump[0] =
        cenv_hvac_pump_power(flow, head, FUSED_WATER_DENSITY, FUSED_GRAVITY,
                             f->hws.pump_eff);
    f->o_hws_return[0] = f->hws.return_temp;
    f->o_hws_fsum[0] = f->hws.flow_factor_sum;
    f->o_hws_requests[0] = f->hws.requests;
  }
  PH(CENV_PH_REWARD_INFO);

  /* -- 19. device state for the Python objects ------------------------- */
  for (int32_t v = 0; v < f->n_vav; ++v) {
    const fused_vav *s = &f->vav[v];
    f->o_vav_damper[v] = s->damper;
    f->o_vav_valve[v] = s->valve;
    f->o_vav_temp[v] = s->zone_temp;
    f->o_vav_heat[v] = s->heat_sp;
    f->o_vav_cool[v] = s->cool_sp;
    f->o_vav_flow[v] = s->flow;
    f->o_vav_ratio[v] = s->ratio;
    f->o_vav_mode[v] = s->mode;
    f->o_vav_prev[v] = s->prev_step;
  }
  f->o_boiler_temp[0] = f->hws.current;
  f->o_boiler_change[0] = f->hws.step_change;
  f->o_boiler_dur[0] = f->hws.last_dur;
  f->o_boiler_burner[0] = f->hws.burner;
  PH(CENV_PH_DEVICE_STATE);
#undef PH
  return 0;
}
#undef PH

/* ------------------------------------------------------------------------- */
/* the entry points                                                          */
/* ------------------------------------------------------------------------- */
int32_t cenv_fused_run(cenv_fused *f, int32_t step, double *temp,
                       double *input_q, const double *sync) {
  if (f == NULL || temp == NULL || input_q == NULL || sync == NULL) {
    return 2;
  }
  if (step < 0 || step + 1 >= f->n_horizon) {
    return 3;
  }
  double ph_mark = fused_clock();
  double ambient = 0.0;
  int32_t rc = fused_prelude(f, step, temp, input_q, sync, &ambient, &ph_mark);
  if (rc != 0) {
    return rc;
  }
  rc = fused_thermal_single(f, temp, input_q, ambient, &ph_mark);
  if (rc != 0) {
    return rc;
  }
  rc = fused_foldback(f, f->sensed, input_q);
  if (rc != 0) {
    return rc;
  }
  return fused_epilogue(f, step, temp, &ph_mark);
}

/* -- the batch halves (batch.c drives these) ------------------------------ *
 * The same error codes cenv_fused_run uses: 2 bad argument, 3 bad step. */

int32_t cenv_fused_batch_prelude(cenv_fused *f, int32_t step, double *temp,
                                 double *input_q, const double *sync,
                                 double *ambient) {
  if (f == NULL || temp == NULL || input_q == NULL || sync == NULL ||
      ambient == NULL) {
    return 2;
  }
  if (step < 0 || step + 1 >= f->n_horizon) {
    return 3;
  }
  if (f->n_phase == 0) {
    f->n_phase = CENV_PH_LEN;
  }
  double ph_mark = fused_clock();
  return fused_prelude(f, step, temp, input_q, sync, ambient, &ph_mark);
}

int32_t cenv_fused_batch_state(cenv_fused *f, const double **g,
                               const double **ts, double **mass,
                               double **out_full, double **sensed) {
  if (f == NULL) {
    return 2;
  }
  if (sensed != NULL) {
    *sensed = f->sensed; /* [n_coupled], the batched solve writes it */
  }
  if (g != NULL) {
    *g = (f->n_coupled > 0) ? f->g : NULL;
  }
  if (ts != NULL) {
    *ts = (f->n_coupled > 0) ? f->ts : NULL;
  }
  if (mass != NULL) {
    *mass = f->mass;
  }
  if (out_full != NULL) {
    *out_full = f->temp_out;
  }
  return 0;
}

int32_t cenv_fused_batch_finalize(cenv_fused *f, int32_t step,
                                  const double *sensed, double *input_q,
                                  double *temp) {
  if (f == NULL || input_q == NULL || temp == NULL ||
      (f->n_coupled > 0 && sensed == NULL)) {
    return 2;
  }
  if (step < 0 || step + 1 >= f->n_horizon) {
    return 3;
  }
  if (f->n_phase == 0) {
    f->n_phase = CENV_PH_LEN;
  }
  memcpy(temp, f->temp_out, (size_t)f->n_full * sizeof(double));
  int32_t rc = fused_foldback(f, sensed, input_q);
  if (rc != 0) {
    return rc;
  }
  double ph_mark = fused_clock();
  return fused_epilogue(f, step, temp, &ph_mark);
}

