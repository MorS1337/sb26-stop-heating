/* Batched thermal step: M environments of one scenario, one factorization.
 *
 * `cenv_thermal_step` (thermal.h) is the single-environment kernel: it solves
 * one right-hand side with scipy's SuperLU factors plus a Woodbury correction
 * for the air side. With M environments that share the geometry -- the same
 * building at the same convection coefficient, which is what a training
 * rollout is -- the matrix is the same for all of them and only the
 * right-hand sides differ, so the step becomes one multi-RHS solve.
 *
 * This handle owns that: one `cenv_chol` (LL^T of the symmetrically permuted
 * A, built by `cenv/solver.py` from the same `solver.matrix(h)` the single
 * kernel factorizes), the dense response R = A^-1 P and S = Q^T A^-1 P built
 * from it, and the width-M work buffers. The per-environment state (the
 * temperature field, the mass nodes, the air-side conductances) is the
 * caller's: `cenv_fused` keeps one set per environment.
 *
 * Not thread-safe: the factor caches its solve workspace. One batch, one
 * thread.
 */

#ifndef CENV_THERMAL_BATCH_H
#define CENV_THERMAL_BATCH_H

#include "solver_chol.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cenv_tbatch cenv_tbatch;

/* Same geometry arguments as `cenv_thermal_create` (thermal.h), plus `m`, the
 * largest number of environments the handle will be asked to step at once.
 * Returns NULL on allocation failure or inconsistent sizes. */
cenv_tbatch *cenv_tbatch_create(int n_full, int n, int k, int m,
                                const int *solved, const double *cap,
                                const double *dir, const double *amb_cond,
                                const double *mc, const double *mr,
                                const double *mg, const int *Pp, const int *Pi,
                                const double *Px, int Pnnz, const int *Sp,
                                const int *Si, const double *Sx, int Snnz);

void cenv_tbatch_free(cenv_tbatch *t);

/* Sizes the handle was created with (any output may be NULL). */
int cenv_tbatch_sizes(const cenv_tbatch *t, int *n_full, int *n, int *k,
                      int *m);

/* Takes ownership of `chol` (n must match) and builds R and S. The Python
 * side hands over the handle of a `cenv.solver.Factor` it does not use any
 * more; freeing the batch frees the factor. Returns 0, 2 bad argument,
 * 3 size mismatch, 4 allocation failure, 5 solve failure. */
int cenv_tbatch_set_factor(cenv_tbatch *t, cenv_chol *chol);

/* One step for `m` (<= capacity) environments, all at the same ambient
 * temperature.
 *
 *   temp_full[e]  [n_full]  the field at the start of the step
 *   q_full[e]     [n_full]  input_q for the step
 *   ambient                 outside air temperature, K (one scenario)
 *   g[e][k]        air-side conductance per term, or NULL to switch it off
 *   ts[e][k]       supply temperature per term, or NULL
 *   out_full[e]   [n_full]  the field after the step (pinned CVs = ambient)
 *   mass[e]       [n]       interior-mass nodes, updated in place
 *   out_sensed[e] [k]       Q^T T' for the caller's air-side foldback, or NULL
 *
 * Returns 0, 1 no factorization, 2 bad argument, 3 bad size, 4 a singular
 * air-side system, 5 solve failure.
 */
int cenv_tbatch_step_many(cenv_tbatch *t, int m,
                          const double *const *temp_full,
                          const double *const *q_full, double ambient,
                          const double *const *g, const double *const *ts,
                          double *const *out_full, double *const *mass,
                          double *const *out_sensed);

/* Diagnostics: CPU seconds in the steps so far, their number, and the split
 * into (forcing, solve, air side, scatter+mass) seconds. */
double cenv_tbatch_seconds(const cenv_tbatch *t);
int cenv_tbatch_steps(const cenv_tbatch *t);
int cenv_tbatch_phase(const cenv_tbatch *t, double *out); /* [4] */

#ifdef __cplusplus
}
#endif

#endif /* CENV_THERMAL_BATCH_H */
