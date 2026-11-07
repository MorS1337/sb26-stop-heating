/* Batched stepping: M environments of one scenario, one C call per step.
 *
 * `cenv_fused_run` (step.c) is one environment's step: prelude -> thermal ->
 * foldback -> epilogue. The thermal solve is 68-81 % of it, and in a batch the
 * M environments share the matrix (same building, same weather, same
 * convection coefficient), so the solves are better issued once for all of
 * them (cenv_tbatch_step_many: one multi-RHS sweep over the factors). That is
 * all this file does -- the physics stays in step.c, one environment at a time:
 *
 *   for e in 0..M-1:  cenv_fused_batch_prelude   (device state, loads, input_q)
 *   once:             cenv_tbatch_step_many      (the M solves)
 *   for e in 0..M-1:  cenv_fused_batch_finalize  (field, air foldback, the rest)
 *
 * The interleaving is why this is not M calls to cenv_fused_run: the solve has
 * to see every environment's right-hand side at once. Order within an
 * environment is untouched, so each environment's trajectory is the one a
 * single-environment run produces (modulo the solver's rounding: the batch
 * factorizes with cenv/solver.py's Cholesky instead of SuperLU).
 *
 * No allocation, no globals, no Python callbacks: the pointer arrays are on the
 * stack (CENV_BATCH_MAX), so the per-step cost is the work itself.
 */

#include <stddef.h>

#include "fused.h"
#include "thermal_batch.h"

/* One batch, one call: the environments are a fixed set for the lifetime of a
 * BatchEnv (cenv/batch.py), and 128 is far above the M that fits in memory
 * (M=64 is already ~6 GB of simulator state on the smallest gate building). */
#define CENV_BATCH_MAX 128

int32_t cenv_batch_run(cenv_fused **handles, int32_t m, int32_t step,
                       double *const *temps, double *const *qs,
                       const double *const *syncs, cenv_tbatch *tb) {
  if (handles == NULL || temps == NULL || qs == NULL || syncs == NULL ||
      tb == NULL) {
    return 2;
  }
  if (m <= 0 || m > CENV_BATCH_MAX) {
    return 3;
  }
  int tb_k = 0;
  int tb_m = 0;
  if (cenv_tbatch_sizes(tb, NULL, NULL, &tb_k, &tb_m) != 0) {
    return 2;
  }
  if (m > tb_m) {
    return 3;
  }

  const double *g[CENV_BATCH_MAX];
  const double *ts[CENV_BATCH_MAX];
  const double *tin[CENV_BATCH_MAX];
  const double *qin[CENV_BATCH_MAX];
  double *mass[CENV_BATCH_MAX];
  double *out[CENV_BATCH_MAX];
  double *sensed[CENV_BATCH_MAX];

  double ambient = 0.0;
  for (int32_t e = 0; e < m; ++e) {
    cenv_fused *f = handles[e];
    if (f == NULL) {
      return 2;
    }
    int32_t rc = cenv_fused_batch_prelude(f, step, temps[e], qs[e], syncs[e],
                                          &ambient);
    if (rc != 0) {
      return rc;
    }
    rc = cenv_fused_batch_state(f, &g[e], &ts[e], &mass[e], &out[e],
                                &sensed[e]);
    if (rc != 0) {
      return rc;
    }
    tin[e] = temps[e];
    qin[e] = qs[e];
    if (tb_k > 0 && sensed[e] == NULL) {
      return 2;
    }
  }

  int32_t rc = cenv_tbatch_step_many(tb, m, tin, qin, ambient, g, ts, out,
                                     mass, sensed);
  if (rc != 0) {
    return rc;
  }

  for (int32_t e = 0; e < m; ++e) {
    rc = cenv_fused_batch_finalize(handles[e], step, sensed[e], qs[e],
                                   temps[e]);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}
