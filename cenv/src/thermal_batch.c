/* Batched version of `cenv_thermal_step`: M environments, one factorization.
 *
 * See thermal.h for the single-environment contract and the formula map
 * (finite_difference_solver.py:968-1060). Everything here is the same step; the
 * only differences are bookkeeping, and they are deliberate:
 *
 *  - the factorization is an LL^T of the symmetrically permuted A
 *    (`cenv_chol`), because a multi-right-hand-side solve is what makes M
 *    environments cheaper than M solves. One factor per scenario, shared by
 *    every environment in the batch (the factor caches one workspace, so a
 *    batch must be stepped from one thread);
 *  - the Woodbury correction uses the dense `A^-1 P` response for every M
 *    (the single-environment kernel picks between it and a second triangular
 *    solve by streaming bytes; with M environments in the inner loop the
 *    response is read once per *step*, not once per environment, so it wins
 *    from M = 2 upwards -- see cenv/batch.py);
 *  - every per-environment vector is a row of a width-M block, so the loops
 *    over environments are contiguous (`restrict`ed, vectorizable) and the
 *    shared data (the factors, R, Q^T) is streamed once per step.
 *
 * FP semantics: each environment's arithmetic is the same expression tree in
 * the same order as `cenv_thermal_step` -- the forcing terms, the Q^T sums
 * (ascending stored entries), the k-by-k pivoted solve, `y + R q` accumulated
 * over terms in ascending order and the mass update are all summed in the same
 * order, so the only difference from the single-environment kernel is the
 * triangular solve itself (SuperLU's LU against this Cholesky: 1e-14 relative,
 * see docs/solver_cpu.md). Build with -ffp-contract=off, as ever.
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "thermal.h"
#include "thermal_batch.h"

#if defined(CLOCK_PROCESS_CPUTIME_ID)
static double tb_clock(void) {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#else
static double tb_clock(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }
#endif

/* Steps the batched kernel's time is split into, in `cenv_tbatch_phase`. */
enum { TB_FORCE = 0, TB_SOLVE, TB_AIR, TB_SCATTER, TB_PHASE_LEN };

struct cenv_tbatch {
  int n_full;
  int n;
  int k;
  int m; /* capacity, in environments */

  int *solved;
  double *cap;
  double *dir;
  double *amb_cond;
  double *mc;
  double *mr;
  double *mg;

  int *Pp;
  int *Pi;
  double *Px;
  int Pnnz;
  int *Sp;
  int *Si;
  double *Sx;
  int Snnz;

  cenv_chol *chol;
  double *response; /* k*n, R = A^-1 P, term j at response[j*n + i] */
  double *sresp;    /* k*k, S = Q^T A^-1 P, entry (j, m) at sresp[j*k + m] */

  /* Width-M work buffers, allocated once for the capacity. */
  double *F;      /* n*m, forcing in, solution out */
  double *acc;    /* rows_block*m, the dense Woodbury accumulation */
  double *sy;     /* m*k, Q^T y */
  double *sensed; /* m*k, Q^T T' of the last step, term-major [j][e] */
  double *rhs;    /* m*k, the air-side correction per environment */
  double *amat;   /* k*k, one environment's (I + diag(g) S), destroyed */
  int rows_block;

  double phase[TB_PHASE_LEN];
  double seconds;
  long long steps;
};

static void *tb_alloc(size_t count, size_t size) {
  if (count == 0) {
    return NULL;
  }
  return calloc(count, size);
}

static int tb_copy_i(int *dst, const int *src, int n) {
  if (n <= 0) {
    return 0;
  }
  memcpy(dst, src, (size_t)n * sizeof(int));
  return 0;
}

static int tb_copy_d(double *dst, const double *src, int n) {
  if (n <= 0) {
    return 0;
  }
  memcpy(dst, src, (size_t)n * sizeof(double));
  return 0;
}

void cenv_tbatch_free(cenv_tbatch *t) {
  if (t == NULL) {
    return;
  }
  free(t->solved);
  free(t->cap);
  free(t->dir);
  free(t->amb_cond);
  free(t->mc);
  free(t->mr);
  free(t->mg);
  free(t->Pp);
  free(t->Pi);
  free(t->Px);
  free(t->Sp);
  free(t->Si);
  free(t->Sx);
  if (t->chol != NULL) {
    cenv_chol_free(t->chol);
  }
  free(t->response);
  free(t->sresp);
  free(t->F);
  free(t->acc);
  free(t->sy);
  free(t->sensed);
  free(t->rhs);
  free(t->amat);
  free(t);
}

/* Number of rows the dense Woodbury accumulation blocks over: keep one block
 * of `rows * m` doubles inside L2 (16 MB on the P cores of an M3) with room to
 * spare, and never under a cache line per environment. */
static int tb_rows_block(int m) {
  const size_t budget = (size_t)1 << 20; /* 1 MiB of accumulators */
  int rows = (int)(budget / ((size_t)m * sizeof(double)));
  if (rows < 8) {
    rows = 8;
  }
  if (rows > 1024) {
    rows = 1024;
  }
  return rows;
}

cenv_tbatch *cenv_tbatch_create(int n_full, int n, int k, int m,
                                const int *solved, const double *cap,
                                const double *dir, const double *amb_cond,
                                const double *mc, const double *mr,
                                const double *mg, const int *Pp, const int *Pi,
                                const double *Px, int Pnnz, const int *Sp,
                                const int *Si, const double *Sx, int Snnz) {
  if (n_full <= 0 || n <= 0 || n > n_full || k < 0 || m <= 0) {
    return NULL;
  }
  cenv_tbatch *t = (cenv_tbatch *)calloc(1, sizeof(cenv_tbatch));
  if (t == NULL) {
    return NULL;
  }
  t->n_full = n_full;
  t->n = n;
  t->k = k;
  t->m = m;
  t->steps = 0;

  t->solved = (int *)tb_alloc((size_t)n, sizeof(int));
  t->cap = (double *)tb_alloc((size_t)n, sizeof(double));
  t->dir = (double *)tb_alloc((size_t)n, sizeof(double));
  t->amb_cond = (double *)tb_alloc((size_t)n, sizeof(double));
  t->mc = (double *)tb_alloc((size_t)n, sizeof(double));
  t->mr = (double *)tb_alloc((size_t)n, sizeof(double));
  t->mg = (double *)tb_alloc((size_t)n, sizeof(double));
  t->F = (double *)tb_alloc((size_t)n * (size_t)m, sizeof(double));
  t->acc = (double *)tb_alloc((size_t)tb_rows_block(m) * (size_t)m,
                              sizeof(double));
  t->rows_block = tb_rows_block(m);
  if (k > 0) {
    t->Pp = (int *)tb_alloc((size_t)k + 1, sizeof(int));
    t->Pi = (int *)tb_alloc((size_t)Pnnz, sizeof(int));
    t->Px = (double *)tb_alloc((size_t)Pnnz, sizeof(double));
    t->Sp = (int *)tb_alloc((size_t)k + 1, sizeof(int));
    t->Si = (int *)tb_alloc((size_t)Snnz, sizeof(int));
    t->Sx = (double *)tb_alloc((size_t)Snnz, sizeof(double));
    t->sy = (double *)tb_alloc((size_t)m * (size_t)k, sizeof(double));
    t->sensed = (double *)tb_alloc((size_t)m * (size_t)k, sizeof(double));
    t->rhs = (double *)tb_alloc((size_t)m * (size_t)k, sizeof(double));
    t->amat = (double *)tb_alloc((size_t)k * (size_t)k, sizeof(double));
  }
  if (t->solved == NULL || t->cap == NULL || t->dir == NULL ||
      t->amb_cond == NULL || t->mc == NULL || t->mr == NULL || t->mg == NULL ||
      t->F == NULL || t->acc == NULL ||
      (k > 0 && (t->Pp == NULL || t->Sp == NULL || t->sy == NULL ||
                 t->sensed == NULL || t->rhs == NULL || t->amat == NULL ||
                 (Pnnz > 0 && (t->Pi == NULL || t->Px == NULL)) ||
                 (Snnz > 0 && (t->Si == NULL || t->Sx == NULL))))) {
    cenv_tbatch_free(t);
    return NULL;
  }

  tb_copy_i(t->solved, solved, n);
  tb_copy_d(t->cap, cap, n);
  tb_copy_d(t->dir, dir, n);
  tb_copy_d(t->amb_cond, amb_cond, n);
  tb_copy_d(t->mc, mc, n);
  tb_copy_d(t->mr, mr, n);
  tb_copy_d(t->mg, mg, n);
  if (k > 0) {
    tb_copy_i(t->Pp, Pp, k + 1);
    tb_copy_i(t->Pi, Pi, Pnnz);
    tb_copy_d(t->Px, Px, Pnnz);
    t->Pnnz = Pnnz;
    tb_copy_i(t->Sp, Sp, k + 1);
    tb_copy_i(t->Si, Si, Snnz);
    tb_copy_d(t->Sx, Sx, Snnz);
    t->Snnz = Snnz;
  }
  return t;
}

int cenv_tbatch_sizes(const cenv_tbatch *t, int *n_full, int *n, int *k,
                      int *m) {
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
  if (m != NULL) {
    *m = t->m;
  }
  return 0;
}

double cenv_tbatch_seconds(const cenv_tbatch *t) {
  return (t == NULL) ? 0.0 : t->seconds;
}

int cenv_tbatch_phase(const cenv_tbatch *t, double *out) {
  if (t == NULL || out == NULL) {
    return 2;
  }
  for (int i = 0; i < TB_PHASE_LEN; ++i) {
    out[i] = t->phase[i];
  }
  return 0;
}

int cenv_tbatch_steps(const cenv_tbatch *t) {
  return (t == NULL) ? 0 : (int)t->steps;
}

/* out[j] = (Q^T x)[j] for every environment at once. `X` is (n, m) row-major,
 * so one stored entry of Q^T reads `m` contiguous doubles and the k*m
 * accumulator (85*64*8 = 43 kB on the largest building) stays in cache while
 * Q^T's values -- the only thing read from memory -- are streamed once. */
static void sensed_many(const cenv_tbatch *t, const double *X, int m,
                        double *out) {
  const int k = t->k;
  const size_t width = (size_t)m;
  for (size_t i = 0; i < (size_t)k * width; ++i) {
    out[i] = 0.0;
  }
  for (int j = 0; j < k; ++j) {
    double *row = out + (size_t)j * width;
    const int end = t->Sp[j + 1];
    for (int p = t->Sp[j]; p < end; ++p) {
      const double v = t->Sx[p];
      const double *x = X + (size_t)t->Si[p] * width;
      for (size_t e = 0; e < width; ++e) {
        row[e] += v * x[e];
      }
    }
  }
}

int cenv_tbatch_set_factor(cenv_tbatch *t, cenv_chol *chol) {
  if (t == NULL || chol == NULL) {
    return 2;
  }
  if (cenv_chol_n(chol) != t->n) {
    return 3;
  }
  if (t->chol != NULL) {
    cenv_chol_free(t->chol);
  }
  t->chol = chol;
  free(t->response);
  free(t->sresp);
  t->response = NULL;
  t->sresp = NULL;
  const int n = t->n;
  const int k = t->k;
  if (k > 0) {
    t->sresp = (double *)tb_alloc((size_t)k * (size_t)k, sizeof(double));
    t->response = (double *)tb_alloc((size_t)k * (size_t)n, sizeof(double));
    if (t->sresp == NULL || t->response == NULL) {
      return 4;
    }
    /* R = A^-1 P, one column per air-side term, solved in blocks so the wide
     * triangular sweeps do the work. P's columns are the unit vectors'
     * sparse images, so P e_j only has Pnnz worth of nonzeros (640 on
     * building_train_8, 1,973 on building_train_2). */
    const int block = 16;
    double *block_buf = (double *)tb_alloc((size_t)n * (size_t)block,
                                           sizeof(double));
    if (block_buf == NULL) {
      return 4;
    }
    for (int j0 = 0; j0 < k; j0 += block) {
      const int width = (k - j0 < block) ? (k - j0) : block;
      for (size_t i = 0; i < (size_t)n * (size_t)width; ++i) {
        block_buf[i] = 0.0;
      }
      for (int j = 0; j < width; ++j) {
        for (int p = t->Pp[j0 + j]; p < t->Pp[j0 + j + 1]; ++p) {
          block_buf[(size_t)t->Pi[p] * (size_t)width + (size_t)j] = t->Px[p];
        }
      }
      if (cenv_chol_solve_many(t->chol, block_buf, block_buf, width, 0) != 0) {
        free(block_buf);
        return 5;
      }
      for (int j = 0; j < width; ++j) {
        for (int i = 0; i < n; ++i) {
          t->response[(size_t)(j0 + j) * (size_t)n + (size_t)i] =
              block_buf[(size_t)i * (size_t)width + (size_t)j];
        }
      }
    }
    free(block_buf);
    /* S = Q^T A^-1 P: column m is Q^T applied to the m-th response column,
     * accumulating in ascending stored order exactly like sensed_apply. */
    for (int m = 0; m < k; ++m) {
      for (int j = 0; j < k; ++j) {
        double acc = 0.0;
        const int end = t->Sp[j + 1];
        for (int p = t->Sp[j]; p < end; ++p) {
          acc += t->Sx[p] * t->response[(size_t)m * (size_t)n +
                                        (size_t)t->Si[p]];
        }
        t->sresp[(size_t)j * (size_t)k + (size_t)m] = acc;
      }
    }
  }
  return 0;
}

int cenv_tbatch_step_many(cenv_tbatch *t, int m,
                          const double *const *temp_full,
                          const double *const *q_full, double ambient,
                          const double *const *g, const double *const *ts,
                          double *const *out_full, double *const *mass,
                          double *const *out_sensed) {
  if (t == NULL || temp_full == NULL || q_full == NULL || out_full == NULL ||
      mass == NULL) {
    return 2;
  }
  if (m <= 0 || m > t->m) {
    return 3;
  }
  if (t->chol == NULL || (t->k > 0 && t->sresp == NULL)) {
    return 1; /* no factorization installed */
  }
  const int n = t->n;
  const int k = t->k;
  const size_t width = (size_t)m;
  double *restrict F = t->F;
  const double entry = tb_clock();
  double mark = entry;

  /* 1. Forcing, per environment, in the order finite_difference_solver.py
   * spells it: cap*T + q, then + T_amb*(d + g_amb), then + M_c*T_m. */
  for (int e = 0; e < m; ++e) {
    const double *restrict T = temp_full[e];
    const double *restrict Q = q_full[e];
    const double *restrict mtemp = mass[e];
    double *restrict row = F + (size_t)e;
    for (int i = 0; i < n; ++i) {
      const int c = t->solved[i];
      row[(size_t)i * width] = t->cap[i] * T[c] + Q[c];
    }
    for (int i = 0; i < n; ++i) {
      row[(size_t)i * width] += ambient * (t->dir[i] + t->amb_cond[i]);
    }
    for (int i = 0; i < n; ++i) {
      row[(size_t)i * width] += t->mc[i] * mtemp[i];
    }
  }
  double now = tb_clock();
  t->phase[TB_FORCE] += now - mark;
  mark = now;

  /* 2. Y = A^-1 F for all M right-hand sides, in one sweep over the factor
   * (cenv_chol_solve_many copies the input into its own workspace first, so
   * in-place is safe). */
  if (cenv_chol_solve_many(t->chol, F, F, m, (m >= 24) ? 0 : 1) != 0) {
    return 5;
  }
  now = tb_clock();
  t->phase[TB_SOLVE] += now - mark;
  mark = now;

  /* 3. Air side: q = (I + diag(g) S)^-1 g (T_s - Q^T y), then
   * T' = y + R q, with R read once for the whole batch. */
  if (k > 0 && g != NULL && ts != NULL) {
    sensed_many(t, F, m, t->sy);
    for (int e = 0; e < m; ++e) {
      double *rhs = t->rhs + (size_t)e * (size_t)k;
      for (int j = 0; j < k; ++j) {
        rhs[j] = g[e][j] * (ts[e][j] - t->sy[(size_t)j * width + (size_t)e]);
      }
      for (int j = 0; j < k; ++j) {
        for (int q = 0; q < k; ++q) {
          double v = g[e][j] * t->sresp[(size_t)j * (size_t)k + (size_t)q];
          if (j == q) {
            v += 1.0;
          }
          t->amat[(size_t)j * (size_t)k + (size_t)q] = v;
        }
      }
      if (cenv_thermal_small_solve(k, t->amat, rhs) != 0) {
        return 4;
      }
    }
    /* T' = y + R q, blocked over rows: the accumulator for one block is
     * `rows_block*m` doubles, and each element of R is loaded once and used
     * for all M environments (which is what makes the dense response pay here
     * where it does not for M = 1). */
    const int RB = t->rows_block;
    for (int i0 = 0; i0 < n; i0 += RB) {
      int i1 = i0 + RB;
      if (i1 > n) {
        i1 = n;
      }
      const int rows = i1 - i0;
      for (size_t i = 0; i < (size_t)rows * width; ++i) {
        t->acc[i] = 0.0;
      }
      for (int j = 0; j < k; ++j) {
        const double *restrict col = t->response +
                                     (size_t)j * (size_t)n + (size_t)i0;
        const double *restrict pj = t->rhs + (size_t)j;
        for (int i = 0; i < rows; ++i) {
          const double r = col[i];
          double *restrict a = t->acc + (size_t)i * width;
          for (size_t e = 0; e < width; ++e) {
            a[e] += r * pj[(size_t)e * (size_t)k];
          }
        }
      }
      for (int i = 0; i < rows; ++i) {
        const double *restrict a = t->acc + (size_t)i * width;
        double *restrict y = F + (size_t)(i0 + i) * width;
        for (size_t e = 0; e < width; ++e) {
          y[e] += a[e];
        }
      }
    }
    /* Q^T T' for the caller's air-side foldback (cenv_thermal_sense_last's
     * job in the single-environment kernel). sensed_many accumulates
     * term-major, which is what keeps Q^T's values read once per step; the
     * caller wants one row per environment, so transpose on the way out (k*m
     * doubles, still in cache). */
    if (out_sensed != NULL) {
      sensed_many(t, F, m, t->sensed);
      for (int e = 0; e < m; ++e) {
        double *restrict row = out_sensed[e];
        for (int j = 0; j < k; ++j) {
          row[j] = t->sensed[(size_t)j * width + (size_t)e];
        }
      }
    }
  }
  now = tb_clock();
  t->phase[TB_AIR] += now - mark;
  mark = now;

  /* 4. Interior mass follows the settled air; 5. scatter back into the full
   * field, pinned CVs at ambient. */
  for (int e = 0; e < m; ++e) {
    double *restrict mtemp = mass[e];
    const double *restrict row = F + (size_t)e;
    for (int i = 0; i < n; ++i) {
      double mt = mtemp[i] * t->mr[i];
      mt += t->mg[i] * row[(size_t)i * width];
      mtemp[i] = mt;
    }
    double *restrict out = out_full[e];
    for (int i = 0; i < t->n_full; ++i) {
      out[i] = ambient;
    }
    for (int i = 0; i < n; ++i) {
      out[t->solved[i]] = row[(size_t)i * width];
    }
  }
  now = tb_clock();
  t->phase[TB_SCATTER] += now - mark;
  t->steps += 1;
  t->seconds += now - entry;
  return 0;
}
