/* Exact C port of FiniteDifferenceSolver's per-step linear algebra.
 *
 * See sims/<tree>/smart_buildings/smart_control/simulator/finite_difference_solver.py
 * (`solve` at ~:968 and `_add_air_side` at ~:1027). The timestep matrix is
 * assembled and factorized in Python with scipy's SuperLU exactly as the
 * simulator does it; this module takes the exported factors and the
 * geometry-dependent vectors and runs the step:
 *
 *   f   = (C/dt)*T + q + T_amb*(d + g_amb) + M_c*T_m       (forcing, :978)
 *   y   = A^-1 f                                          (one CSC solve)
 *   q_a = (I + G S)^-1 G (T_s - Q^T y),  S = Q^T A^-1 P    (Woodbury, :1027)
 *   T'  = y + R q_a,                     R = A^-1 P        (dense, precomputed)
 *   T_m'= r*T_m + g*T'
 *   T'[pinned] = T_amb
 *
 * R is the one thing the Python code deliberately does not keep (it re-solves
 * for `applied @ power` every step); precomputing it once per factorization
 * turns the second triangular solve into a dense matvec. S is precomputed the
 * same way, exactly.
 *
 * Identity of results: the Python path calls scipy's SuperLU solve; this one
 * does a scalar column-oriented forward/back substitution on the same L, U and
 * permutations, so the only difference is summation order inside the
 * triangular solves (agrees to ~1e-15 relative, i.e. a few ULP per entry).
 * Nothing here reorders or reassociates a floating-point expression beyond
 * that: -ffp-contract=off is required (see cenv/build.py).
 *
 * All arrays are 1-D, contiguous, float64 (or int32 for indices), owned by the
 * caller except the ones the constructor copies. The state that changes per
 * step (`T_m`, the output field) stays on the Python side.
 */

#ifndef CENV_THERMAL_H
#define CENV_THERMAL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-(solver, factorization) handle. */
typedef struct cenv_thermal cenv_thermal;

/* Creates a handle with the static, geometry-dependent vectors.
 *
 *   n_full   number of control volumes in the field (rows*cols)
 *   n        number of solved CVs (n_full minus the pinned exterior ones)
 *   k        number of air-side terms (0 for none)
 *   solved   [n]   positions of the solved CVs in the flat field
 *   cap      [n]   _capacitance_rate, W/K
 *   dir      [n]   _dirichlet_conductance, W/K
 *   amb      [n]   ambient_conductance(h), W/K
 *   mc       [n]   _mass_coupling, W/K
 *   mr       [n]   _mass_retained, -
 *   mg       [n]   _mass_gained, -
 *   Pp,Pi,Px CSC of P, the (n x k) "where each term's power lands" matrix,
 *                  column j = Pp[j]..Pp[j+1] over rows Pi with values Px. Only
 *                  P q is ever formed from it, so it stays sparse.
 *   Pnnz     number of stored entries in P
 *   Sp,Si,Sx CSR of Q^T, the (k x n) "which temperature each term reads"
 *                  matrix: row j = Sp[j]..Sp[j+1] over columns Si. Sparse,
 *                  because a dense copy is k*n doubles -- 164 MB per step on
 *                  building_train_2 against 2.9 MB of stored entries.
 *   Snnz     number of stored entries in Q^T
 *
 * Returns NULL on allocation failure or on inconsistent sizes.
 */
cenv_thermal *cenv_thermal_create(
    int n_full, int n, int k,
    const int *solved,
    const double *cap,
    const double *dir,
    const double *amb,
    const double *mc,
    const double *mr,
    const double *mg,
    const int *Pp,
    const int *Pi,
    const double *Px,
    int Pnnz,
    const int *Sp,
    const int *Si,
    const double *Sx,
    int Snnz);

void cenv_thermal_free(cenv_thermal *t);

/* Installs a factorization and precomputes the two responses it implies.
 *
 *   Lp, Li, Lx   [n+1], [Lnnz], [Lnnz]  scipy lu.L.tocsc() with its explicit
 *                                        unit diagonal
 *   Up, Ui, Ux   [n+1], [Unnz], [Unnz]  scipy lu.U.tocsc()
 *   ipr          [n]  argsort(lu.perm_r) — the gather that applies Pr
 *   pc           [n]  lu.perm_c        — the gather that applies Pc^-1
 *
 * Solving A x = b is then: y = b[ipr]; forward-solve L; back-solve U;
 * x[i] = y[pc[i]], which is the convention scipy's SuperLU object uses
 * (verified: Pr A Pc = L U with Pr[i, ipr[i]] = 1 and Pc[pc[i], i] = 1).
 *
 * Cost is k triangular solves; the handle keeps Q^T A^-1 P (k*k) always and
 * A^-1 P (n*k) only when that is cheaper per step than re-solving for P q (see
 * set_factor in thermal.c). Returns 0 on success.
 */
int cenv_thermal_set_factor(
    cenv_thermal *t,
    const int *Lp, const int *Li, const double *Lx, int Lnnz,
    const int *Up, const int *Ui, const double *Ux, int Unnz,
    const int *ipr,
    const int *pc);

/* One backward-Euler step.
 *
 *   temp_full  [n_full] temperature field at the start of the step
 *   q_full     [n_full] input_q (building.input_q)
 *   ambient    ambient temperature, K
 *   g          [k] air-side conductance W/K, or NULL to switch the air side off
 *   ts         [k] supply temperature K per term, or NULL
 *   out_full   [n_full] the field after the step (pinned CVs = ambient)
 *   mass_temp  [n] interior-mass node temperature; updated in place
 *   out_power  [k] delivered air-side power G*(T_s - Q^T T'), or NULL
 *
 * Returns 0 on success, 1 if the handle has no factorization.
 */
int cenv_thermal_step(
    cenv_thermal *t,
    const double *temp_full,
    const double *q_full,
    double ambient,
    const double *g,
    const double *ts,
    double *out_full,
    double *mass_temp,
    double *out_power);

/* Residual diagnostics: copies the forcing vector of the last step. */
int cenv_thermal_last_forcing(const cenv_thermal *t, double *out);


/* Sizes the handle was created with (any output may be NULL). Exposed so a
 * caller that fuses the step around this solve can size its own state. */
int cenv_thermal_sizes(const cenv_thermal *t, int *n_full, int *n, int *k);

/* out[j] = (Q^T T')[j] for the field the last cenv_thermal_step produced, with
 * the same additions scipy's `sensed @ temp` makes (ascending column order).
 * The Python step folds the air side back with the field it just solved for,
 * not with Q^T y + S q, so a fused step that must match it bit for bit asks
 * here instead of using cenv_thermal_step's `out_power`. */
int cenv_thermal_sense_last(const cenv_thermal *t, double *out);

/* Requested at create() time by the Python shim, which needs the two response
 * matrices for its own shadow-mode checks. The pointers are valid until the
 * next cenv_thermal_set_factor / cenv_thermal_free. */
const double *cenv_thermal_response(const cenv_thermal *t);       /* n*k */
const double *cenv_thermal_sensed_response(const cenv_thermal *t); /* k*k */

/* The air-side k-by-k solve, (I + diag(g) S) q = rhs, by dense LU with partial
 * pivoting: `a` is row-major k*k and is destroyed, `b` is k and comes back as
 * the solution. Shared with the batched kernel (thermal_batch.h), which solves
 * one such system per environment. Returns 0, or 1 if the system is singular. */
int cenv_thermal_small_solve(int k, double *a, double *b);

#ifdef __cplusplus
}
#endif

#endif /* CENV_THERMAL_H */
