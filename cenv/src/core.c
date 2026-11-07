/* Library glue: version banner, self-test, numpy-compatible pairwise sum. */

#include "cenv.h"

#include <math.h>
#include <string.h>

const char *cenv_version(void) {
  return "cenv 0.1";
}

int32_t cenv_selftest(double *buf, int64_t n, double scale) {
  if (buf == NULL || n < 0) return -1;
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    buf[i] = buf[i] * scale + (double)i;
    sum += buf[i];
  }
  return sum == sum ? 0 : -2; /* reject NaN */
}

/* numpy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src):
 * 8 accumulators below a 128-element block, halves split on an 8-element
 * boundary above it, strictly sequential below 8 elements starting at -0.0.
 * Same operation order in IEEE double == bit-identical to np.add.reduce on a
 * contiguous 1-D float64 array. */
#define CENV_PW_BLOCKSIZE 128

static double pairwise_sum_impl(const double *a, int64_t n) {
  if (n < 8) {
    double res = -0.0;
    for (int64_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= CENV_PW_BLOCKSIZE) {
    double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
    double r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
    int64_t i = 8;
    const int64_t stop = n - (n % 8);
    for (; i < stop; i += 8) {
      r0 += a[i + 0];
      r1 += a[i + 1];
      r2 += a[i + 2];
      r3 += a[i + 3];
      r4 += a[i + 4];
      r5 += a[i + 5];
      r6 += a[i + 6];
      r7 += a[i + 7];
    }
    double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum_impl(a, n2) + pairwise_sum_impl(a + n2, n - n2);
}

double cenv_pairwise_sum(const double *a, int64_t n) {
  if (a == NULL || n <= 0) return -0.0;
  return pairwise_sum_impl(a, n);
}
