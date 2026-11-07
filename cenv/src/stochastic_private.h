/* Shared internals between the stochastic component files.
 *
 * Not part of the public surface (that is cenv/include/stochastic.h): the
 * generator state is shared by the convection kernel, the occupancy chain and
 * the Python shim's stream proof, but nothing outside cenv/src needs it.
 */

#ifndef CENV_STOCHASTIC_PRIVATE_H
#define CENV_STOCHASTIC_PRIVATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CENV_MT_N 624

typedef struct cenv_mt {
  uint32_t mt[CENV_MT_N];
  int index;      /* next word in mt[], CENV_MT_N == "shuffle first" */
  int64_t words;  /* words consumed so far; diagnostics only */
} cenv_mt;

/* Looks a generator up by handle; NULL when the handle is unknown. */
cenv_mt *cenv__mt_get(int64_t handle);

/* Knuth init_genrand, as numpy's mt19937_seed / numba's numba_rnd_init. */
void cenv__mt_init(cenv_mt *st, uint32_t seed);

void cenv__mt_shuffle(cenv_mt *st);
uint32_t cenv__mt_next_uint32(cenv_mt *st);
double cenv__mt_next_double(cenv_mt *st);

/* np.random.randint(max_incl + 1) under numba's legacy (numpy-compatible)
 * path: masked rejection, and no draw at all when max_incl == 0. */
int64_t cenv__mt_randint(cenv_mt *st, int64_t max_incl);

#ifdef __cplusplus
}
#endif

#endif /* CENV_STOCHASTIC_PRIVATE_H */
