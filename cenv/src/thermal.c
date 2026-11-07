/* C port of FiniteDifferenceSolver.solve / _add_air_side. See thermal.h. */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "thermal.h"

struct cenv_thermal {
  int n_full;
  int n;
  int k;

  int *solved;
  double *cap;
  double *dir;
  double *amb;
  double *mc;
  double *mr;
  double *mg;

  /* Air-side geometry. P is kept sparse (640 nonzeros against 2.4M dense
   * entries on building_train_8): the step only needs P q, which is one
   * accumulate per stored entry. Q^T is kept dense because it has an entry in
   * every CV of every served zone and a dense read beats reading column
   * indices alongside the values. */
  int *Pp;   /* k+1 */
  int *Pi;   /* Pnnz, rows within the solved vector */
  double *Px; /* Pnnz */
  int Pnnz;
  int *Sp;   /* k+1 */
  int *Si;   /* Snnz, solved-vector columns */
  double *Sx; /* Snnz */
  int Snnz;
  double *pvec; /* n, P q */

  /* Factorization. */
  int *Lp;
  int *Li;
  double *Lx;
  int Lnnz;
  int *Up;
  int *Ui;
  double *Ux;
  double *Udiag; /* U's diagonal, gathered once: back substitution needs the
                  * pivot before it can propagate the column. */
  int Unnz;
  int *ipr;
  int *pc;

  /* A^-1 P (k*n column-major), Q^T A^-1 P (k*k row-major) and whether the step
   * uses the dense response or a second triangular solve. */
  double *response;
  double *sresp;
  int use_dense;

  /* Per-step work. */
  double *forcing; /* n */
  double *y;       /* n */
  double *ywork;   /* n */
  double *acc;     /* n */
  double *sy;      /* k */
  double *rhs;     /* k */
  double *amat;    /* k*k */
  int *piv;        /* k */
};

static void *xmalloc(size_t count, size_t size) {
  if (count == 0) {
    return NULL;
  }
  return calloc(count, size);
}

static int *alloc_i(int n) { return (int *)xmalloc((size_t)n, sizeof(int)); }
static double *alloc_d(int n) { return (double *)xmalloc((size_t)n, sizeof(double)); }

static int copy_i(int *dst, const int *src, int n) {
  if (n <= 0) {
    return 0;
  }
  memcpy(dst, src, (size_t)n * sizeof(int));
  return 0;
}

static int copy_d(double *dst, const double *src, int n) {
  if (n <= 0) {
    return 0;
  }
  memcpy(dst, src, (size_t)n * sizeof(double));
  return 0;
}

void cenv_thermal_free(cenv_thermal *t) {
  if (t == NULL) {
    return;
  }
  free(t->solved);
  free(t->cap);
  free(t->dir);
  free(t->amb);
  free(t->mc);
  free(t->mr);
  free(t->mg);
  free(t->Pp);
  free(t->Pi);
  free(t->Px);
  free(t->Sp);
  free(t->Si);
  free(t->Sx);
  free(t->pvec);
  free(t->Lp);
  free(t->Li);
  free(t->Lx);
  free(t->Up);
  free(t->Ui);
  free(t->Ux);
  free(t->Udiag);
  free(t->ipr);
  free(t->pc);
  free(t->response);
  free(t->sresp);
  free(t->forcing);
  free(t->y);
  free(t->ywork);
  free(t->acc);
  free(t->sy);
  free(t->rhs);
  free(t->amat);
  free(t->piv);
  free(t);
}

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
    int Snnz) {
  if (n_full <= 0 || n <= 0 || n > n_full || k < 0) {
    return NULL;
  }
  cenv_thermal *t = (cenv_thermal *)calloc(1, sizeof(cenv_thermal));
  if (t == NULL) {
    return NULL;
  }
  t->n_full = n_full;
  t->n = n;
  t->k = k;

  t->solved = alloc_i(n);
  t->cap = alloc_d(n);
  t->dir = alloc_d(n);
  t->amb = alloc_d(n);
  t->mc = alloc_d(n);
  t->mr = alloc_d(n);
  t->mg = alloc_d(n);
  t->forcing = alloc_d(n);
  t->y = alloc_d(n);
  t->ywork = alloc_d(n);
  t->acc = alloc_d(n);
  if (k > 0) {
    t->Pp = alloc_i(k + 1);
    t->Pi = alloc_i(Pnnz);
    t->Px = alloc_d(Pnnz);
    t->Sp = alloc_i(k + 1);
    t->Si = alloc_i(Snnz);
    t->Sx = alloc_d(Snnz);
    t->pvec = alloc_d(n);
    t->sy = alloc_d(k);
    t->rhs = alloc_d(k);
    t->amat = alloc_d(k * k);
    t->piv = alloc_i(k);
  }
  if (t->solved == NULL || t->cap == NULL || t->forcing == NULL ||
      t->y == NULL || t->ywork == NULL || t->acc == NULL ||
      (k > 0 && (t->Pp == NULL || t->pvec == NULL || t->Sp == NULL ||
                 t->sy == NULL || t->rhs == NULL || t->amat == NULL ||
                 t->piv == NULL || (Pnnz > 0 && (t->Pi == NULL ||
                                                t->Px == NULL)) ||
                 (Snnz > 0 && (t->Si == NULL || t->Sx == NULL))))) {
    cenv_thermal_free(t);
    return NULL;
  }

  copy_i(t->solved, solved, n);
  copy_d(t->cap, cap, n);
  copy_d(t->dir, dir, n);
  copy_d(t->amb, amb, n);
  copy_d(t->mc, mc, n);
  copy_d(t->mr, mr, n);
  copy_d(t->mg, mg, n);
  if (k > 0) {
    copy_i(t->Pp, Pp, k + 1);
    copy_i(t->Pi, Pi, Pnnz);
    copy_d(t->Px, Px, Pnnz);
    t->Pnnz = Pnnz;
    copy_i(t->Sp, Sp, k + 1);
    copy_i(t->Si, Si, Snnz);
    copy_d(t->Sx, Sx, Snnz);
    t->Snnz = Snnz;
  }
  return t;
}

/* Solves A x = b for one right-hand side.
 *
 * `out` and `b` may not alias. Uses t->y as scratch. The permutation gathers
 * are exactly the ones scipy's SuperLU object applies:
 *   y = b[ipr]; L and U triangular solves; out[i] = y[pc[i]].
 * L carries an explicit unit diagonal (scipy returns it) and the entries of a
 * column are not sorted, which is irrelevant to a column-oriented substitution:
 * column j only writes to rows i > j, all of which are still unmodified.
 */
static void tri_solve(const cenv_thermal *t, const double *b, double *out) {
  const int n = t->n;
  double *y = t->y;
  const int *ipr = t->ipr;
  const int *pc = t->pc;

  for (int i = 0; i < n; ++i) {
    y[i] = b[ipr[i]];
  }

  /* Forward substitution. L carries an explicit unit diagonal and its columns
   * write only to rows below it, so y[j] is already final when column j is
   * reached. */
  for (int j = 0; j < n; ++j) {
    const double yj = y[j];
    const int end = t->Lp[j + 1];
    for (int p = t->Lp[j]; p < end; ++p) {
      const int i = t->Li[p];
      if (i != j) {
        y[i] -= t->Lx[p] * yj;
      }
    }
  }

  /* Back substitution. The pivot has to be applied before the column
   * propagates, so the diagonal is gathered once at factor installation. */
  for (int j = n - 1; j >= 0; --j) {
    const double yj = y[j] / t->Udiag[j];
    y[j] = yj;
    const int end = t->Up[j + 1];
    for (int p = t->Up[j]; p < end; ++p) {
      const int i = t->Ui[p];
      if (i != j) {
        y[i] -= t->Ux[p] * yj;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    out[i] = y[pc[i]];
  }
}

/* out[j] = (Q^T x)[j], summed over Q's stored entries in ascending column
 * order -- the same additions scipy's `sensed @ x` makes. */
static void sensed_apply(const cenv_thermal *t, const double *x, double *out) {
  const int k = t->k;
  for (int j = 0; j < k; ++j) {
    double acc = 0.0;
    const int end = t->Sp[j + 1];
    for (int p = t->Sp[j]; p < end; ++p) {
      acc += t->Sx[p] * x[t->Si[p]];
    }
    out[j] = acc;
  }
}

/* Dense k-by-k solve, LU with partial pivoting (row-major `a`, destroyed).
 * Also used by the batched kernel (thermal_batch.c), which solves the same
 * air-side system once per environment. */
int cenv_thermal_small_solve(int k, double *a, double *b) {
  for (int c = 0; c < k; ++c) {
    int pivot = c;
    double best = fabs(a[(size_t)c * k + c]);
    for (int r = c + 1; r < k; ++r) {
      const double v = fabs(a[(size_t)r * k + c]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best == 0.0) {
      return 1; /* singular */
    }
    if (pivot != c) {
      for (int j = c; j < k; ++j) {
        double tmp = a[(size_t)c * k + j];
        a[(size_t)c * k + j] = a[(size_t)pivot * k + j];
        a[(size_t)pivot * k + j] = tmp;
      }
      double tb = b[c];
      b[c] = b[pivot];
      b[pivot] = tb;
    }
    const double inv = 1.0 / a[(size_t)c * k + c];
    for (int r = c + 1; r < k; ++r) {
      const double factor = a[(size_t)r * k + c] * inv;
      if (factor == 0.0) {
        continue;
      }
      a[(size_t)r * k + c] = 0.0;
      for (int j = c + 1; j < k; ++j) {
        a[(size_t)r * k + j] -= factor * a[(size_t)c * k + j];
      }
      b[r] -= factor * b[c];
    }
  }
  for (int r = k - 1; r >= 0; --r) {
    double acc = b[r];
    for (int j = r + 1; j < k; ++j) {
      acc -= a[(size_t)r * k + j] * b[j];
    }
    b[r] = acc / a[(size_t)r * k + r];
  }
  return 0;
}

static void drop_factor(cenv_thermal *t) {
  free(t->Lp);
  free(t->Li);
  free(t->Lx);
  free(t->Up);
  free(t->Ui);
  free(t->Ux);
  free(t->ipr);
  free(t->pc);
  free(t->response);
  free(t->sresp);
  t->Lp = NULL;
  t->Li = NULL;
  t->Lx = NULL;
  t->Up = NULL;
  t->Ui = NULL;
  t->Ux = NULL;
  t->Udiag = NULL;
  t->ipr = NULL;
  t->pc = NULL;
  t->response = NULL;
  t->sresp = NULL;
  t->Lnnz = 0;
  t->Unnz = 0;
}

int cenv_thermal_set_factor(
    cenv_thermal *t,
    const int *Lp, const int *Li, const double *Lx, int Lnnz,
    const int *Up, const int *Ui, const double *Ux, int Unnz,
    const int *ipr,
    const int *pc) {
  if (t == NULL || Lp == NULL || Li == NULL || Lx == NULL || ipr == NULL ||
      pc == NULL) {
    return 2;
  }
  const int n = t->n;
  drop_factor(t);

  t->Lp = alloc_i(n + 1);
  t->Li = alloc_i(Lnnz);
  t->Lx = alloc_d(Lnnz);
  t->Up = alloc_i(n + 1);
  t->Ui = alloc_i(Unnz);
  t->Ux = alloc_d(Unnz);
  t->Udiag = alloc_d(n);
  t->ipr = alloc_i(n);
  t->pc = alloc_i(n);
  t->response = NULL; /* allocated once use_dense is known, below */
  t->sresp = (t->k > 0) ? alloc_d((size_t)t->k * (size_t)t->k) : NULL;
  if (t->Lp == NULL || t->Up == NULL || t->ipr == NULL || t->pc == NULL ||
      (Lnnz > 0 && (t->Li == NULL || t->Lx == NULL)) ||
      (Unnz > 0 && (t->Ui == NULL || t->Ux == NULL)) ||
      t->Udiag == NULL || (t->k > 0 && t->sresp == NULL)) {
    drop_factor(t);
    return 3;
  }
  copy_i(t->Lp, Lp, n + 1);
  copy_i(t->Li, Li, Lnnz);
  copy_d(t->Lx, Lx, Lnnz);
  copy_i(t->Up, Up, n + 1);
  copy_i(t->Ui, Ui, Unnz);
  copy_d(t->Ux, Ux, Unnz);
  copy_i(t->ipr, ipr, n);
  copy_i(t->pc, pc, n);
  t->Lnnz = Lnnz;
  t->Unnz = Unnz;
  for (int i = 0; i < n; ++i) {
    t->Udiag[i] = 0.0;
  }
  for (int j = 0; j < n; ++j) {
    const int end = t->Up[j + 1];
    for (int p = t->Up[j]; p < end; ++p) {
      if (t->Ui[p] == j) {
        t->Udiag[j] = t->Ux[p];
        break;
      }
    }
  }

  if (t->k > 0) {
    /* Which is cheaper per step: one dense (n x k) matvec, or a second
     * triangular solve whose right-hand side is P q (P q costs Pnnz multiply
     * adds, so P's sparsity is what makes the second solve viable)? Both are
     * bandwidth-bound, so compare the bytes they stream: the dense path reads
     * n*k doubles, the second solve reads the factors. Measured on
     * building_train_8 (n*k = 2.4M against 3.5M factor entries) the two are
     * within 10 % of each other; on building_train_2 (n*k = 14.8M against
     * ~9M) the dense path is the slower one by a third of a step.
     *
     * A factorized block of k right-hand sides still has to be solved once to
     * build S = Q^T A^-1 P either way, so this only decides the per-step
     * branch. */
    t->use_dense = ((double)n * (double)t->k <
                    0.75 * (double)(t->Lnnz + t->Unnz));
    if (t->use_dense) {
      t->response = alloc_d((size_t)t->k * (size_t)n);
      if (t->response == NULL) {
        drop_factor(t);
        return 4;
      }
      for (int j = 0; j < t->k; ++j) {
        double *rhs = t->ywork;
        for (int i = 0; i < n; ++i) {
          rhs[i] = 0.0;
        }
        for (int p = t->Pp[j]; p < t->Pp[j + 1]; ++p) {
          rhs[t->Pi[p]] = t->Px[p];
        }
        tri_solve(t, rhs, t->response + (size_t)j * n);
      }
    }
    /* Q^T A^-1 P, exactly the matrix the Woodbury step inverts: column m is
     * Q^T applied to the m-th solved image of a unit air-side term. */
    double *col = alloc_d(n);
    if (col == NULL) {
      drop_factor(t);
      return 5;
    }
    for (int m = 0; m < t->k; ++m) {
      if (t->use_dense) {
        memcpy(col, t->response + (size_t)m * n, (size_t)n * sizeof(double));
      } else {
        for (int i = 0; i < n; ++i) {
          col[i] = 0.0;
        }
        for (int p = t->Pp[m]; p < t->Pp[m + 1]; ++p) {
          col[t->Pi[p]] = t->Px[p];
        }
        tri_solve(t, col, t->acc);
        memcpy(col, t->acc, (size_t)n * sizeof(double));
      }
      sensed_apply(t, col, t->sy);
      for (int j = 0; j < t->k; ++j) {
        t->sresp[(size_t)j * t->k + m] = t->sy[j];
      }
    }
    free(col);
  }
  return 0;
}

const double *cenv_thermal_response(const cenv_thermal *t) {
  return (t == NULL) ? NULL : t->response;
}

const double *cenv_thermal_sensed_response(const cenv_thermal *t) {
  return (t == NULL) ? NULL : t->sresp;
}

int cenv_thermal_sizes(const cenv_thermal *t, int *n_full, int *n, int *k) {
  if (t == NULL) {
    return 2;
  }
  if (n_full != NULL) {
    *n_full = t->n_full;
  }
  if (n != NULL) {
    *n = t->n;
  }
  if (k != NULL) {
    *k = t->k;
  }
  return 0;
}

int cenv_thermal_sense_last(const cenv_thermal *t, double *out) {
  if (t == NULL || out == NULL) {
    return 2;
  }
  if (t->k <= 0) {
    return 1;
  }
  /* t->ywork holds the solved field of the last step (T' over the unknowns). */
  sensed_apply(t, t->ywork, out);
  return 0;
}

int cenv_thermal_last_forcing(const cenv_thermal *t, double *out) {
  if (t == NULL || out == NULL) {
    return 2;
  }
  copy_d(out, t->forcing, t->n);
  return 0;
}

int cenv_thermal_step(
    cenv_thermal *t,
    const double *temp_full,
    const double *q_full,
    double ambient,
    const double *g,
    const double *ts,
    double *out_full,
    double *mass_temp,
    double *out_power) {
  if (t == NULL || temp_full == NULL || q_full == NULL || out_full == NULL ||
      mass_temp == NULL) {
    return 2;
  }
  if (t->Up == NULL || (t->k > 0 && t->sresp == NULL)) {
    return 1; /* no factorization installed */
  }
  const int n = t->n;
  const int *solved = t->solved;
  double *f = t->forcing;

  /* 1. Forcing: same expression order as finite_difference_solver.py:978-987.
   * cap*T + q, then + T_amb*(d + g_amb), then + M_c*T_m. */
  for (int i = 0; i < n; ++i) {
    const int c = solved[i];
    f[i] = t->cap[i] * temp_full[c] + q_full[c];
  }
  for (int i = 0; i < n; ++i) {
    f[i] += ambient * (t->dir[i] + t->amb[i]);
  }
  for (int i = 0; i < n; ++i) {
    f[i] += t->mc[i] * mass_temp[i];
  }

  /* 2. y = A^-1 f. */
  double *y = t->ywork;
  tri_solve(t, f, y);

  /* 3. Air side, if any term has been registered. */
  const int k = t->k;
  if (k > 0 && g != NULL && ts != NULL) {
    double *sy = t->sy;
    sensed_apply(t, y, sy);
    for (int j = 0; j < k; ++j) {
      t->rhs[j] = g[j] * (ts[j] - sy[j]);
    }
    /* (I + diag(g) S) q = g (T_s - Q^T y) */
    for (int j = 0; j < k; ++j) {
      for (int m = 0; m < k; ++m) {
        double v = g[j] * t->sresp[(size_t)j * k + m];
        if (j == m) {
          v += 1.0;
        }
        t->amat[(size_t)j * k + m] = v;
      }
    }
    if (cenv_thermal_small_solve(k, t->amat, t->rhs) != 0) {
      return 4;
    }
    /* T' = y + A^-1 P q. Either the dense precomputed A^-1 P is applied, or
     * P q is formed from P's sparsity and solved for -- same result, different
     * memory traffic; see the choice in set_factor. */
    if (t->use_dense) {
      double *acc = t->acc;
      for (int i = 0; i < n; ++i) {
        acc[i] = 0.0;
      }
      for (int j = 0; j < k; ++j) {
        const double pj = t->rhs[j];
        if (pj == 0.0) {
          continue;
        }
        const double *col = t->response + (size_t)j * n;
        for (int i = 0; i < n; ++i) {
          acc[i] += col[i] * pj;
        }
      }
      for (int i = 0; i < n; ++i) {
        y[i] += acc[i];
      }
    } else {
      double *pvec = t->pvec;
      for (int i = 0; i < n; ++i) {
        pvec[i] = 0.0;
      }
      for (int j = 0; j < k; ++j) {
        const double pj = t->rhs[j];
        if (pj == 0.0) {
          continue;
        }
        for (int p = t->Pp[j]; p < t->Pp[j + 1]; ++p) {
          pvec[t->Pi[p]] += t->Px[p] * pj;
        }
      }
      tri_solve(t, pvec, t->acc);
      for (int i = 0; i < n; ++i) {
        y[i] += t->acc[i];
      }
    }
    if (out_power != NULL) {
      /* G * (T_s - Q^T T') = G * (T_s - Q^T y - S q). */
      for (int j = 0; j < k; ++j) {
        double acc2 = 0.0;
        for (int m = 0; m < k; ++m) {
          acc2 += t->sresp[(size_t)j * k + m] * t->rhs[m];
        }
        out_power[j] = g[j] * (ts[j] - (sy[j] + acc2));
      }
    }
  }

  /* 4. Interior mass follows the settled air. */
  for (int i = 0; i < n; ++i) {
    double mt = mass_temp[i] * t->mr[i];
    mt += t->mg[i] * y[i];
    mass_temp[i] = mt;
  }

  /* 5. Scatter back into the full field; pinned CVs stay at ambient. */
  for (int i = 0; i < t->n_full; ++i) {
    out_full[i] = ambient;
  }
  for (int i = 0; i < n; ++i) {
    out_full[solved[i]] = y[i];
  }
  return 0;
}
