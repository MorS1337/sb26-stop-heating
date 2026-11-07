/* Exact sparse Cholesky (LL^T) of the timestep matrix, with a supernodal
 * blocked triangular solve for one or many right-hand sides.
 *
 * The matrix is symmetric positive definite by construction (see
 * finite_difference_solver.py), so an LL^T factorization of a *symmetrically
 * permuted* A is available: half the storage of an LU and one triangular sweep
 * per solve instead of two. The permutation is supplied by the caller, so the
 * ordering study (MMD / AMD / METIS nested dissection / natural) is a Python
 * decision and this file never has to know a graph partitioner.
 *
 * Numeric factorization is the classic up-looking (row-by-row) algorithm of
 * Davis' CSparse (``cs_chol``): for row k the pattern of row k of L is found by
 * walking the elimination tree from the columns of A(:,k) below the diagonal,
 * then a dense stack solve produces L(k,:). Total work is O(flops), memory is
 * nnz(L) int32 + nnz(L) double.
 *
 * The solve converts L into *fundamental supernodes* (runs of consecutive
 * columns whose patterns are nested) and stores each as one dense rectangle, so
 * a substitution becomes short dense dot products / rank-w updates instead of
 * one dependent scalar update per nonzero. With M right-hand sides the same
 * blocks turn into (w x w) triangular solves and (rows x w) rank-w updates over
 * M-wide vectors, laid out M-contiguous per unknown.
 *
 * FP semantics: the build pins -ffp-contract=off and never -ffast-math, so
 * every floating-point operation here rounds exactly as written.
 */
#ifndef CENV_SOLVER_CHOL_H
#define CENV_SOLVER_CHOL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cenv_chol cenv_chol;

/* Factorizes P A P^T = L L^T where perm[new] = old (perm may be NULL for the
 * natural order; ``permute`` then ignored). A is n x n in CSC with full
 * symmetric storage (both triangles); explicit zeros are dropped. Returns NULL
 * if A is not numerically positive definite or allocation fails. */
cenv_chol *cenv_chol_create(int n, const int *Ap, const int *Ai, const double *Ax,
                            const int *perm);

void cenv_chol_free(cenv_chol *f);

int cenv_chol_n(const cenv_chol *f);
int cenv_chol_nnz(const cenv_chol *f);              /* nnz(L) */
long long cenv_chol_nnz_super(const cenv_chol *f);  /* doubles in the blocked copy */
int cenv_chol_nsuper(const cenv_chol *f);
int cenv_chol_maxwidth(const cenv_chol *f);
double cenv_chol_flops(const cenv_chol *f);         /* multiply-adds in the factorization */
const int *cenv_chol_perm(const cenv_chol *f);      /* perm[new] = old */
const int *cenv_chol_colptr(const cenv_chol *f);    /* L's CSC, for inspection */
const int *cenv_chol_rowind(const cenv_chol *f);
const double *cenv_chol_values(const cenv_chol *f);

/* Solve A x = b. mode 0 = supernodal blocked, 1 = plain CSC sweep. */
int cenv_chol_solve(const cenv_chol *f, const double *b, double *x, int mode);

/* Solve A X = B for m right-hand sides. B and X are (n, m) row-major, i.e.
 * entry (i, j) lives at i*m + j: the m values of one unknown are contiguous.
 * mode 0 = supernodal blocked, 1 = plain CSC sweep per right-hand side. */
int cenv_chol_solve_many(const cenv_chol *f, const double *B, double *X, int m,
                         int mode);

#ifdef __cplusplus
}
#endif

#endif /* CENV_SOLVER_CHOL_H */
