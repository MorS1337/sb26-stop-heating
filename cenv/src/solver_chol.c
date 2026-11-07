/* See solver_chol.h. Exact sparse Cholesky + supernodal blocked solves. */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "solver_chol.h"

/* Widening the panels by merging consecutive supernodes was measured on the
 * six shipped buildings and does not pay: it doubles the factor (padding to
 * w x w where the block is not nested) and the panel sweep then becomes bound
 * by the padded factor instead of by the right-hand sides, ending level with
 * the CSC sweeps at every M (train_8: 1.40 vs 1.31 ms per right-hand side at
 * M=64, and 1.7x slower at M=1). The machinery is left in place behind this
 * knob for matrices whose supernodes are genuinely narrow *and* numerous. */
#ifndef CENV_CHOL_RELAX_WIDTH
#define CENV_CHOL_RELAX_WIDTH 1 /* max columns in one dense panel */
#endif
#ifndef CENV_CHOL_RELAX_PAD
#define CENV_CHOL_RELAX_PAD 3 /* padded block <= PAD/2 x the exact storage */
#endif
#define RELAX_WIDTH CENV_CHOL_RELAX_WIDTH
#define RELAX_PAD CENV_CHOL_RELAX_PAD

struct cenv_chol {
  int n;
  int *perm;  /* new -> old */
  int *pinv;  /* old -> new */

  /* CSC of L, diagonal first in every column, rows ascending after it. */
  int *Lp;
  int *Li;
  double *Lx;
  int nnz;

  /* CSC of L^T (row-wise sweep of L), same diagonal-first convention. */
  int *Tp;
  int *Ti;
  double *Tx;

  /* Fundamental supernodes: runs of columns with nested patterns. */
  int nsuper;
  int maxwidth;
  int *Sfirst;
  int *Swidth;
  int *Supd;        /* update rows below the diagonal block */
  int *Srowp;       /* nsuper + 1 offsets into Srow */
  int *Srow;        /* concatenated update row indices */
  long long *Svalp; /* nsuper + 1 offsets into Sval */
  double *Sval;     /* panels: (w + m) rows of w doubles; rows 0..w-1 are the
                     dense lower block L11, rows w..w+m-1 the update rows L21 */

  double flops;
  double *y;  /* n, single-rhs scratch */
  double *t1; /* maxwidth, single-rhs panel scratch */

  /* Batched workspace, grown on demand and reused: a 64-wide block on the
   * largest shipped floor plan is 40 MB, and page-faulting that on every call
   * costs more than the solve. The solver is single threaded (one environment
   * per process), so one cache per factor is safe; concurrent use from several
   * threads would have to pass its own buffers. */
  double *yw;
  size_t yw_cap;
  double *tw;
  size_t tw_cap;
};

static void *xcalloc(size_t count, size_t size) {
  if (count == 0) {
    return NULL;
  }
  return calloc(count, size);
}

static int ensure_ws(cenv_chol *f, int M) {
  const size_t need = (size_t)f->n * (size_t)M;
  if (need > f->yw_cap) {
    double *p = realloc(f->yw, need * sizeof(double));
    if (p == NULL) {
      return 1;
    }
    f->yw = p;
    f->yw_cap = need;
  }
  const size_t needt = (size_t)(f->maxwidth > 0 ? f->maxwidth : 1) * (size_t)M;
  if (needt > f->tw_cap) {
    double *p = realloc(f->tw, needt * sizeof(double));
    if (p == NULL) {
      return 1;
    }
    f->tw = p;
    f->tw_cap = needt;
  }
  return 0;
}

void cenv_chol_free(cenv_chol *f) {
  if (f == NULL) {
    return;
  }
  free(f->perm);
  free(f->pinv);
  free(f->Lp);
  free(f->Li);
  free(f->Lx);
  free(f->Tp);
  free(f->Ti);
  free(f->Tx);
  free(f->Sfirst);
  free(f->Swidth);
  free(f->Supd);
  free(f->Srowp);
  free(f->Srow);
  free(f->Svalp);
  free(f->Sval);
  free(f->y);
  free(f->t1);
  free(f->yw);
  free(f->tw);
  free(f);
}

/* ------------------------------------------------------------------ */
/* symbolic                                                            */
/* ------------------------------------------------------------------ */
/* Pattern of row k of L, in ascending column order: walk the elimination tree
 * from the columns i < k that A(i,k) touches. B is the permuted A in CSC with
 * sorted columns, so the entries with row >= k can be skipped wholesale. */
static int ereach(int n, const int *Bp, const int *Bi, int k, const int *parent,
                  int *s, int *wmark) {
  int top = n;
  wmark[k] = k;
  const int end = Bp[k + 1];
  for (int p = Bp[k]; p < end; ++p) {
    int i = Bi[p];
    if (i >= k) {
      break;
    }
    if (wmark[i] == k) {
      continue;
    }
    int len = 0;
    while (i >= 0 && wmark[i] != k) {
      s[len++] = i;
      wmark[i] = k;
      i = parent[i];
    }
    while (len > 0) {
      s[--top] = s[--len];
    }
  }
  return top;
}

/* Sort one column of the permuted matrix by row, summing duplicates. Returns
 * the new end of the compacted column. */
static int sort_column(int start, int end, int *Bi, double *Bx) {
  for (int p = start + 1; p < end; ++p) {
    const int r = Bi[p];
    const double v = Bx[p];
    int q = p - 1;
    while (q >= start && Bi[q] > r) {
      Bi[q + 1] = Bi[q];
      Bx[q + 1] = Bx[q];
      --q;
    }
    Bi[q + 1] = r;
    Bx[q + 1] = v;
  }
  int out = start;
  for (int p = start; p < end; ++p) {
    if (out > start && Bi[out - 1] == Bi[p]) {
      Bx[out - 1] += Bx[p];
    } else {
      Bi[out] = Bi[p];
      Bx[out] = Bx[p];
      ++out;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* supernodes                                                          */
/* ------------------------------------------------------------------ */
static int build_supernodes(cenv_chol *f) {
  const int n = f->n;
  const int *Lp = f->Lp, *Li = f->Li;
  int cap = 256, nsuper = 0;
  int *first = xcalloc((size_t)cap, sizeof(int));
  int *width = xcalloc((size_t)cap, sizeof(int));
  if (first == NULL || width == NULL) {
    free(first);
    free(width);
    return 1;
  }
  int j = 0;
  while (j < n) {
    int w = 1;
    while (j + w < n) {
      const int prev = j + w - 1;
      const int cnt_prev = Lp[prev + 1] - Lp[prev];
      const int cnt_next = Lp[prev + 2] - Lp[prev + 1];
      if (cnt_next != cnt_prev - 1) {
        break;
      }
      if (Li[Lp[prev] + 1] != prev + 1) {
        break; /* the diagonal block would not be dense */
      }
      const int *a = &Li[Lp[prev + 1]];      /* this column, from its diagonal */
      const int *b = &Li[Lp[prev] + 1];      /* previous column, one lower */
      int ok = 1;
      for (int t = 1; t < cnt_next; ++t) {
        if (a[t] != b[t]) {
          ok = 0;
          break;
        }
      }
      if (!ok) {
        break;
      }
      ++w;
      if (w >= 4096) {
        break; /* no realistic supernode is this wide; keeps the scan bounded */
      }
    }
    if (nsuper == cap) {
      cap *= 2;
      int *nf = xcalloc((size_t)cap, sizeof(int));
      int *nw = xcalloc((size_t)cap, sizeof(int));
      if (nf == NULL || nw == NULL) {
        free(nf);
        free(nw);
        free(first);
        free(width);
        return 1;
      }
      memcpy(nf, first, (size_t)nsuper * sizeof(int));
      memcpy(nw, width, (size_t)nsuper * sizeof(int));
      free(first);
      free(width);
      first = nf;
      width = nw;
    }
    first[nsuper] = j;
    width[nsuper] = w;
    ++nsuper;
    j += w;
  }
  /* ---- relaxed amalgamation ----
   * On these floor-plan lattices the fundamental supernodes are almost all one
   * column wide (62k of them for 81k unknowns), which leaves the panel path
   * with nothing to amortize: it is the *dense* panel that turns w scattered
   * rank-1 updates into one (rows x w) block whose right-hand sides stream
   * contiguously. Merging consecutive supernodes pads the block with zeros but
   * divides the traffic per right-hand side by about w, so it is a large win as
   * soon as several environments share the factorization (see
   * scratch/SolverCPU/bench.py) and a small memory cost otherwise. The padding
   * is bounded: a group is accepted only while the padded block stays within
   * RELAX_PAD/2 of the exact storage it replaces. */
  {
    int *mf = xcalloc((size_t)(nsuper ? nsuper : 1), sizeof(int));
    int *mw = xcalloc((size_t)(nsuper ? nsuper : 1), sizeof(int));
    if (mf == NULL || mw == NULL) {
      free(mf);
      free(mw);
      free(first);
      free(width);
      return 1;
    }
    int m = 0, s0 = 0;
    while (s0 < nsuper) {
      int total = width[s0], maxc = 0, e = s0;
      long long sumc = 0;
      for (int c = first[s0]; c < first[s0] + width[s0]; ++c) {
        const int cc = Lp[c + 1] - Lp[c];
        if (cc > maxc) {
          maxc = cc;
        }
        sumc += cc;
      }
      while (e + 1 < nsuper && total + width[e + 1] <= RELAX_WIDTH) {
        int max2 = maxc;
        long long sum2 = sumc;
        for (int c = first[e + 1]; c < first[e + 1] + width[e + 1]; ++c) {
          const int cc = Lp[c + 1] - Lp[c];
          if (cc > max2) {
            max2 = cc;
          }
          sum2 += cc;
        }
        const long long padded = (long long)max2 * (long long)(total + width[e + 1]);
        if (2 * padded > (long long)RELAX_PAD * sum2) {
          break;
        }
        total += width[e + 1];
        maxc = max2;
        sumc = sum2;
        ++e;
      }
      mf[m] = first[s0];
      mw[m] = total;
      ++m;
      s0 = e + 1;
    }
    free(first);
    free(width);
    f->nsuper = m;
    f->Sfirst = mf;
    f->Swidth = mw;
    return 0;
  }
}

/* Panels: rows 0..w-1 of a panel are the w columns' own (lower triangular)
 * block over the panel's column range, rows w..w+m-1 are the other rows those
 * columns touch. Panels are wide enough (RELAX_WIDTH) that the block is not
 * necessarily exactly nested, so entries are placed by row index and every
 * other slot stays zero. */
static int build_super_storage(cenv_chol *f) {
  const int nsuper = f->nsuper, n = f->n;
  const int *Lp = f->Lp, *Li = f->Li;
  const double *Lx = f->Lx;
  int *Supd = xcalloc((size_t)(nsuper ? nsuper : 1), sizeof(int));
  int *Srowp = xcalloc((size_t)nsuper + 1, sizeof(int));
  long long *Svalp = xcalloc((size_t)nsuper + 1, sizeof(long long));
  int *mark = xcalloc((size_t)(n ? n : 1), sizeof(int));
  int *slot = xcalloc((size_t)(n ? n : 1), sizeof(int));
  int rc = 1;
  if (Supd == NULL || Srowp == NULL || Svalp == NULL || mark == NULL ||
      slot == NULL) {
    goto done;
  }
  int maxwidth = 0, total_rows = 0;
  for (int s = 0; s < nsuper; ++s) {
    const int fc = f->Sfirst[s], w = f->Swidth[s], lim = fc + w - 1;
    int m = 0;
    for (int c = fc; c <= lim; ++c) {
      for (int p = Lp[c] + 1; p < Lp[c + 1]; ++p) {
        const int r = Li[p];
        if (r > lim && mark[r] == 0) {
          mark[r] = 1;
          ++m;
        }
      }
    }
    for (int c = fc; c <= lim; ++c) {
      for (int p = Lp[c] + 1; p < Lp[c + 1]; ++p) {
        if (Li[p] > lim) {
          mark[Li[p]] = 0;
        }
      }
    }
    Supd[s] = m;
    Srowp[s] = total_rows;
    total_rows += m;
    Svalp[s + 1] = Svalp[s] + (long long)w * (long long)(w + m);
    if (w > maxwidth) {
      maxwidth = w;
    }
  }
  Srowp[nsuper] = total_rows;
  f->Srow = xcalloc((size_t)(total_rows ? total_rows : 1), sizeof(int));
  f->Sval = xcalloc((size_t)(Svalp[nsuper] ? Svalp[nsuper] : 1), sizeof(double));
  if (f->Srow == NULL || f->Sval == NULL) {
    goto done;
  }
  for (int s = 0; s < nsuper; ++s) {
    const int fc = f->Sfirst[s], w = f->Swidth[s], lim = fc + w - 1;
    const int m = Supd[s];
    double *v = &f->Sval[Svalp[s]];
    int t = 0;
    for (int c = fc; c <= lim; ++c) {
      for (int p = Lp[c] + 1; p < Lp[c + 1]; ++p) {
        const int r = Li[p];
        if (r > lim && slot[r] == 0) {
          slot[r] = ++t; /* 1-based so 0 can mean "not a panel row" */
          f->Srow[Srowp[s] + t - 1] = r;
        }
      }
    }
    if (t != m) {
      goto done; /* the two passes disagreed */
    }
    for (int c = fc; c <= lim; ++c) {
      const int cc = c - fc;
      for (int p = Lp[c]; p < Lp[c + 1]; ++p) {
        const int r = Li[p];
        if (r <= lim) {
          v[(size_t)(r - fc) * (size_t)w + (size_t)cc] = Lx[p];
        } else {
          v[(size_t)(w + slot[r] - 1) * (size_t)w + (size_t)cc] = Lx[p];
        }
      }
    }
    for (int c = fc; c <= lim; ++c) {
      for (int p = Lp[c] + 1; p < Lp[c + 1]; ++p) {
        if (Li[p] > lim) {
          slot[Li[p]] = 0;
        }
      }
    }
  }
  f->Supd = Supd;
  f->Srowp = Srowp;
  f->Svalp = Svalp;
  f->maxwidth = maxwidth;
  Supd = NULL;
  Srowp = NULL;
  Svalp = NULL;
  rc = 0;
done:
  free(Supd);
  free(Srowp);
  free(Svalp);
  free(mark);
  free(slot);
  return rc;
}

/* CSC of L^T with the diagonal first in each column. */
static int build_transpose(cenv_chol *f) {
  const int n = f->n, nnz = f->nnz;
  int rc = 1;
  int *Tp = xcalloc((size_t)n + 1, sizeof(int));
  int *Ti = xcalloc((size_t)(nnz ? nnz : 1), sizeof(int));
  double *Tx = xcalloc((size_t)(nnz ? nnz : 1), sizeof(double));
  int *fill = xcalloc((size_t)n, sizeof(int));
  if (Tp == NULL || Ti == NULL || Tx == NULL || fill == NULL) {
    goto done;
  }
  for (int j = 0; j < n; ++j) {
    for (int p = f->Lp[j] + 1; p < f->Lp[j + 1]; ++p) {
      Tp[f->Li[p] + 1] += 1;
    }
  }
  for (int j = 0; j < n; ++j) {
    Tp[j + 1] += 1; /* the diagonal of row j of L */
    Tp[j + 1] += Tp[j];
  }
  for (int j = 0; j < n; ++j) {
    Ti[Tp[j]] = j;
    Tx[Tp[j]] = f->Lx[f->Lp[j]];
    fill[j] = Tp[j] + 1;
  }
  for (int j = 0; j < n; ++j) {
    for (int p = f->Lp[j] + 1; p < f->Lp[j + 1]; ++p) {
      const int i = f->Li[p];
      const int q = fill[i]++;
      Ti[q] = j;
      Tx[q] = f->Lx[p];
    }
  }
  f->Tp = Tp;
  f->Ti = Ti;
  f->Tx = Tx;
  Tp = NULL;
  Ti = NULL;
  Tx = NULL;
  rc = 0;
done:
  free(Tp);
  free(Ti);
  free(Tx);
  free(fill);
  return rc;
}

/* ------------------------------------------------------------------ */
/* factorization                                                       */
/* ------------------------------------------------------------------ */
cenv_chol *cenv_chol_create(int n, const int *Ap, const int *Ai, const double *Ax,
                            const int *perm) {
  if (n <= 0 || Ap == NULL || Ai == NULL || Ax == NULL) {
    return NULL;
  }
  cenv_chol *f = xcalloc(1, sizeof(cenv_chol));
  int *parent = NULL, *wmark = NULL, *s = NULL, *count = NULL, *cnext = NULL;
  int *Bp = NULL, *Bi = NULL, *start = NULL;
  double *Bx = NULL;
  if (f == NULL) {
    return NULL;
  }
  f->n = n;
  f->perm = xcalloc((size_t)n, sizeof(int));
  f->pinv = xcalloc((size_t)n, sizeof(int));
  if (f->perm == NULL || f->pinv == NULL) {
    goto fail;
  }
  for (int i = 0; i < n; ++i) {
    f->perm[i] = perm ? perm[i] : i;
  }
  for (int i = 0; i < n; ++i) {
    const int old = f->perm[i];
    if (old < 0 || old >= n) {
      goto fail;
    }
    f->pinv[old] = i;
  }

  /* ---- permuted A: CSC, sorted rows, no explicit zeros ---- */
  int bnnz = 0;
  for (int j = 0; j < n; ++j) {
    for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
      if (Ax[p] != 0.0) {
        ++bnnz;
      }
    }
  }
  if (bnnz == 0) {
    goto fail;
  }
  start = xcalloc((size_t)n + 1, sizeof(int)); /* column starts, then reused */
  Bp = xcalloc((size_t)n + 1, sizeof(int));    /* fill pointers */
  int *cnt = xcalloc((size_t)n + 1, sizeof(int));
  Bi = xcalloc((size_t)bnnz, sizeof(int));
  Bx = xcalloc((size_t)bnnz, sizeof(double));
  if (start == NULL || Bp == NULL || cnt == NULL || Bi == NULL || Bx == NULL) {
    free(cnt);
    goto fail;
  }
  for (int j = 0; j < n; ++j) {
    const int nj = f->pinv[j];
    for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
      if (Ax[p] != 0.0) {
        cnt[nj] += 1;
      }
    }
  }
  {
    int running = 0;
    for (int j = 0; j < n; ++j) {
      start[j] = running;
      running += cnt[j];
    }
    start[n] = running;
  }
  free(cnt);
  for (int j = 0; j < n; ++j) {
    Bp[j] = start[j];
  }
  for (int j = 0; j < n; ++j) {
    const int nj = f->pinv[j];
    for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
      if (Ax[p] == 0.0) {
        continue;
      }
      const int q = Bp[nj]++;
      Bi[q] = f->pinv[Ai[p]];
      Bx[q] = Ax[p];
    }
  }
  /* sort each column, drop duplicates by summing, and compact in place */
  {
    int out = 0;
    for (int j = 0; j < n; ++j) {
      const int col_start = start[j], col_end = start[j + 1];
      const int new_end = sort_column(col_start, col_end, Bi, Bx);
      const int len = new_end - col_start;
      if (out != col_start) {
        memmove(&Bi[out], &Bi[col_start], (size_t)len * sizeof(int));
        memmove(&Bx[out], &Bx[col_start], (size_t)len * sizeof(double));
      }
      Bp[j] = out;
      out += len;
    }
    Bp[n] = out;
    bnnz = out;
    memcpy(start, Bp, (size_t)(n + 1) * sizeof(int));
  }
  free(start);
  start = NULL;

  /* ---- elimination tree ---- */
  parent = xcalloc((size_t)n, sizeof(int));
  int *ancestor = xcalloc((size_t)n, sizeof(int));
  if (parent == NULL || ancestor == NULL) {
    free(ancestor);
    goto fail;
  }
  for (int i = 0; i < n; ++i) {
    parent[i] = -1;
    ancestor[i] = -1;
  }
  for (int k = 0; k < n; ++k) {
    for (int p = Bp[k]; p < Bp[k + 1]; ++p) {
      int i = Bi[p];
      if (i >= k) {
        break;
      }
      while (i >= 0 && i < k) {
        const int next = ancestor[i];
        ancestor[i] = k;
        if (next < 0) {
          parent[i] = k;
        }
        i = next;
      }
    }
  }
  free(ancestor);

  /* ---- column counts ---- */
  wmark = xcalloc((size_t)n, sizeof(int));
  s = xcalloc((size_t)n, sizeof(int));
  count = xcalloc((size_t)n, sizeof(int));
  if (wmark == NULL || s == NULL || count == NULL) {
    goto fail;
  }
  for (int i = 0; i < n; ++i) {
    wmark[i] = -1;
  }
  for (int k = 0; k < n; ++k) {
    const int top = ereach(n, Bp, Bi, k, parent, s, wmark);
    for (int p = top; p < n; ++p) {
      count[s[p]] += 1;
    }
    count[k] += 1;
  }
  f->Lp = xcalloc((size_t)n + 1, sizeof(int));
  if (f->Lp == NULL) {
    goto fail;
  }
  for (int j = 0; j < n; ++j) {
    f->Lp[j + 1] = f->Lp[j] + count[j];
  }
  f->nnz = f->Lp[n];
  f->Li = xcalloc((size_t)f->nnz, sizeof(int));
  f->Lx = xcalloc((size_t)f->nnz, sizeof(double));
  cnext = xcalloc((size_t)n, sizeof(int));
  if (f->Li == NULL || f->Lx == NULL || cnext == NULL) {
    goto fail;
  }
  for (int j = 0; j < n; ++j) {
    cnext[j] = f->Lp[j];
  }

  /* ---- numeric: up-looking Cholesky (Davis, cs_chol) ---- */
  {
    double *x = xcalloc((size_t)n, sizeof(double));
    if (x == NULL) {
      goto fail;
    }
    double flops = 0.0;
    for (int k = 0; k < n; ++k) {
      int top = ereach(n, Bp, Bi, k, parent, s, wmark);
      for (int p = Bp[k]; p < Bp[k + 1]; ++p) {
        const int i = Bi[p];
        if (i > k) {
          break;
        }
        x[i] = Bx[p];
      }
      double d = x[k];
      x[k] = 0.0;
      for (; top < n; ++top) {
        const int i = s[top];
        const double lki = x[i] / f->Lx[f->Lp[i]];
        x[i] = 0.0;
        const int end = cnext[i];
        for (int p = f->Lp[i] + 1; p < end; ++p) {
          x[f->Li[p]] -= f->Lx[p] * lki;
          flops += 1.0;
        }
        d -= lki * lki;
        flops += 1.0;
        const int q = cnext[i]++;
        f->Li[q] = k;
        f->Lx[q] = lki;
      }
      if (!(d > 0.0)) { /* not positive definite (or NaN) */
        free(x);
        goto fail;
      }
      const int q = cnext[k]++;
      f->Li[q] = k;
      f->Lx[q] = sqrt(d);
      flops += 1.0;
    }
    free(x);
    f->flops = flops;
  }
  for (int j = 0; j < n; ++j) {
    if (cnext[j] != f->Lp[j + 1]) {
      goto fail; /* symbolic counts disagreed with the numeric pattern */
    }
  }
  free(parent);
  free(wmark);
  free(s);
  free(count);
  free(cnext);
  free(Bp);
  free(Bi);
  free(Bx);

  if (build_supernodes(f) != 0 || build_super_storage(f) != 0 ||
      build_transpose(f) != 0) {
    goto fail;
  }
  f->y = xcalloc((size_t)n, sizeof(double));
  f->t1 = xcalloc((size_t)(f->maxwidth > 0 ? f->maxwidth : 1), sizeof(double));
  if (f->y == NULL || f->t1 == NULL) {
    goto fail;
  }
  return f;

fail:
  free(parent);
  free(wmark);
  free(s);
  free(count);
  free(cnext);
  free(start);
  free(Bp);
  free(Bi);
  free(Bx);
  cenv_chol_free(f);
  return NULL;
}

/* ------------------------------------------------------------------ */
/* solves                                                              */
/* ------------------------------------------------------------------ */
/* Supernodal forward substitution, M right-hand sides, Y is n x M row-major
 * (unknown-major). Each panel row is read once, sequentially; the diagonal
 * block is solved in the gather form so the panel is walked only forwards. */
static void forward_super(const cenv_chol *f, double *Y, int M) {
  const int nsuper = f->nsuper;
  const int *Sfirst = f->Sfirst, *Swidth = f->Swidth, *Supd = f->Supd;
  const int *Srowp = f->Srowp, *Srow = f->Srow;
  const long long *Svalp = f->Svalp;
  const double *Sval = f->Sval;
  for (int s = 0; s < nsuper; ++s) {
    const int fc = Sfirst[s], w = Swidth[s], m = Supd[s];
    const int STRIDE = w;
    const double *v = &Sval[Svalp[s]];
    double *Yf = &Y[(size_t)fc * M];
    for (int i = 0; i < w; ++i) {
      const double *vi = &v[(size_t)i * STRIDE];
      double *restrict yi = Yf + (size_t)i * M;
      for (int c = 0; c < i; ++c) {
        const double lic = vi[c];
        if (lic == 0.0) {
          continue;
        }
        const double *restrict yc = Yf + (size_t)c * M;
        for (int t = 0; t < M; ++t) {
          yi[t] -= lic * yc[t];
        }
      }
      const double d = vi[i];
      for (int t = 0; t < M; ++t) {
        yi[t] /= d;
      }
    }
    const int *rows = &Srow[Srowp[s]];
    for (int q = 0; q < m; ++q) {
      const double *vr = &v[(size_t)(w + q) * STRIDE];
      double *restrict yr = &Y[(size_t)rows[q] * M];
      for (int c = 0; c < w; ++c) {
        const double lrc = vr[c];
        if (lrc == 0.0) {
          continue;
        }
        const double *restrict yc = Yf + (size_t)c * M;
        for (int t = 0; t < M; ++t) {
          yr[t] -= lrc * yc[t];
        }
      }
    }
  }
}

/* Supernodal back substitution. T is scratch of maxwidth * M doubles. */
static void backward_super(const cenv_chol *f, double *Y, int M, double *T) {
  const int nsuper = f->nsuper;
  const int *Sfirst = f->Sfirst, *Swidth = f->Swidth, *Supd = f->Supd;
  const int *Srowp = f->Srowp, *Srow = f->Srow;
  const long long *Svalp = f->Svalp;
  const double *Sval = f->Sval;
  for (int s = nsuper - 1; s >= 0; --s) {
    const int fc = Sfirst[s], w = Swidth[s], m = Supd[s];
    const int STRIDE = w;
    const double *v = &Sval[Svalp[s]];
    double *Yf = &Y[(size_t)fc * M];
    const size_t wm = (size_t)w * M;
    memcpy(T, Yf, wm * sizeof(double));
    const int *rows = &Srow[Srowp[s]];
    for (int q = 0; q < m; ++q) {
      const double *vr = &v[(size_t)(w + q) * STRIDE];
      const double *restrict yr = &Y[(size_t)rows[q] * M];
      for (int c = 0; c < w; ++c) {
        const double lrc = vr[c];
        if (lrc == 0.0) {
          continue;
        }
        double *restrict tc = T + (size_t)c * M;
        for (int t = 0; t < M; ++t) {
          tc[t] -= lrc * yr[t];
        }
      }
    }
    /* Solve L11^T T = T, column oriented from the last column backwards: the
     * unknown of column c is finalized (divide) and then removes itself from
     * the rows above it through L11(c, i) = L^T(i, c). */
    for (int c = w - 1; c >= 0; --c) {
      const double *vc = &v[(size_t)c * STRIDE];
      double *restrict tc = T + (size_t)c * M;
      const double d = vc[c];
      for (int t = 0; t < M; ++t) {
        tc[t] /= d;
      }
      for (int i = 0; i < c; ++i) {
        const double lci = vc[i];
        if (lci == 0.0) {
          continue;
        }
        double *restrict ti = T + (size_t)i * M;
        const double *restrict srcc = tc;
        for (int t = 0; t < M; ++t) {
          ti[t] -= lci * srcc[t];
        }
      }
    }
    memcpy(Yf, T, wm * sizeof(double));
  }
}

/* The CSC sweep pair: the fastest kernel for one to a couple of dozen right-hand
 * sides. Each column adds itself into the rows *below* the diagonal, which are
 * ascending in every column, so those stores are sequential and the source
 * value is read once. The gather form -- one dot product per row over a
 * contiguous run of the factor, no transpose copy needed -- was measured 1.6x
 * slower at M=1 and 1.3x at M=8 on train_8, so it is not kept. */
static void forward_csc_many(const cenv_chol *f, double *Y, int M) {
  const int n = f->n;
  const int *Lp = f->Lp, *Li = f->Li;
  const double *Lx = f->Lx;
  for (int j = 0; j < n; ++j) {
    double *restrict yj = Y + (size_t)j * M;
    const double d = Lx[Lp[j]];
    for (int t = 0; t < M; ++t) {
      yj[t] /= d;
    }
    const int end = Lp[j + 1];
    for (int p = Lp[j] + 1; p < end; ++p) {
      const double a = Lx[p];
      double *restrict yi = Y + (size_t)Li[p] * M;
      const double *restrict src = yj;
      for (int t = 0; t < M; ++t) {
        yi[t] -= a * src[t];
      }
    }
  }
}

static void backward_csc_many(const cenv_chol *f, double *Y, int M) {
  const int n = f->n;
  const int *Tp = f->Tp, *Ti = f->Ti;
  const double *Tx = f->Tx;
  for (int j = n - 1; j >= 0; --j) {
    double *restrict yj = Y + (size_t)j * M;
    const double d = Tx[Tp[j]];
    for (int t = 0; t < M; ++t) {
      yj[t] /= d;
    }
    const int end = Tp[j + 1];
    for (int p = Tp[j] + 1; p < end; ++p) {
      const double a = Tx[p];
      double *restrict yi = Y + (size_t)Ti[p] * M;
      const double *restrict src = yj;
      for (int t = 0; t < M; ++t) {
        yi[t] -= a * src[t];
      }
    }
  }
}

static int solve_impl(const cenv_chol *fc, const double *B, double *X, int M,
                      int mode) {
  /* The workspace cache is the only mutation a solve performs, and one factor
   * is owned by one environment, so the constness of the handle is a promise
   * about the *factorization*, not about this scratch space. */
  cenv_chol *f = (cenv_chol *)fc;
  const int n = f->n;
  if (M <= 0) {
    return 0;
  }
  double *Y;
  const int own = (M == 1 && f->y != NULL);
  if (own) {
    Y = f->y;
  } else {
    if (ensure_ws(f, M) != 0) {
      return 1;
    }
    Y = f->yw;
  }
  for (int i = 0; i < n; ++i) {
    memcpy(&Y[(size_t)i * M], &B[(size_t)f->perm[i] * M],
           (size_t)M * sizeof(double));
  }
  if (mode == 0) {
    const double *T = (M == 1 && f->t1 != NULL) ? f->t1 : f->tw;
    if (T == NULL) {
      return 1;
    }
    forward_super(f, Y, M);
    backward_super(f, Y, M, (double *)T);
  } else {
    forward_csc_many(f, Y, M);
    backward_csc_many(f, Y, M);
  }
  for (int i = 0; i < n; ++i) {
    memcpy(&X[(size_t)f->perm[i] * M], &Y[(size_t)i * M],
           (size_t)M * sizeof(double));
  }
  return 0;
}

int cenv_chol_solve(const cenv_chol *f, const double *b, double *x, int mode) {
  return solve_impl(f, b, x, 1, mode);
}

int cenv_chol_solve_many(const cenv_chol *f, const double *B, double *X, int m,
                         int mode) {
  return solve_impl(f, B, X, m, mode);
}

int cenv_chol_n(const cenv_chol *f) { return f->n; }
int cenv_chol_nnz(const cenv_chol *f) { return f->nnz; }
int cenv_chol_nsuper(const cenv_chol *f) { return f->nsuper; }
int cenv_chol_maxwidth(const cenv_chol *f) { return f->maxwidth; }
double cenv_chol_flops(const cenv_chol *f) { return f->flops; }
const int *cenv_chol_perm(const cenv_chol *f) { return f->perm; }
const int *cenv_chol_colptr(const cenv_chol *f) { return f->Lp; }
const int *cenv_chol_rowind(const cenv_chol *f) { return f->Li; }
const double *cenv_chol_values(const cenv_chol *f) { return f->Lx; }
long long cenv_chol_nnz_super(const cenv_chol *f) {
  return f->Svalp ? f->Svalp[f->nsuper] : 0;
}
