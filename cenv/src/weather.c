/* Exact C port of the weather replay.
 *
 * Python (sims/<tree>/smart_buildings/smart_control/simulator/
 * weather_controller.py):
 *
 *   read_weather_csv(path)
 *     pd.read_csv; `Time` parsed as UTC from "%Y%m%d-%H%M"; the index becomes
 *     (Time - 1970-01-01 UTC).total_seconds(), i.e. exact integer seconds.
 *
 *   _get_interpolated_value(timestamp, values)
 *     localize/convert to UTC, range-check, then np.interp on
 *     (epoch seconds, values). np.interp: clamp at the ends, exact hit returns
 *     the sample, otherwise slope * (x - xp[j]) + fp[j] with
 *     slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j]) on the bracketing interval
 *     found by binary search.
 *
 *   get_current_temp = temperature_conversion.fahrenheit_to_kelvin(interp(TempF))
 *                     = ((f - 32.0) * 5.0 / 9.0) + 273.15
 *
 * The CSV is parsed here column-by-column by header name (quoting honoured, so
 * the station-name column's embedded comma cannot shift the fields), because
 * everything downstream of read_weather_csv reads only Time, TempF, Humidity
 * and the index.
 */

#include "cenv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CENV_WEATHER_MAX_HANDLES 8

typedef struct {
  int64_t n;
  double *xp;      /* epoch seconds */
  double *temp_f;
  double *humidity;
} weather_table;

static weather_table *g_wx[CENV_WEATHER_MAX_HANDLES];

/* Howard Hinnant's days_from_civil: days since 1970-01-01, proleptic Gregorian. */
static int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  int64_t era = (y >= 0 ? y : y - 399) / 400;
  unsigned yoe = (unsigned)(y - era * 400);
  unsigned doy = (153u * (m + (m > 2 ? (unsigned)-3 : 9u)) + 2u) / 5u + d - 1u;
  unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
  return era * 146097 + (int64_t)doe - 719468;
}

/* "YYYYMMDD-HHMM" read as UTC -> seconds since the Unix epoch.
 * Returns 0 on success. */
static int parse_epoch_seconds(const char *s, size_t len, double *out) {
  if (len < 13) return -1;
  int64_t y = 0;
  for (int i = 0; i < 4; i++) {
    if (s[i] < '0' || s[i] > '9') return -1;
    y = y * 10 + (s[i] - '0');
  }
  unsigned mo = (unsigned)((s[4] - '0') * 10 + (s[5] - '0'));
  unsigned d = (unsigned)((s[6] - '0') * 10 + (s[7] - '0'));
  if (mo < 1 || mo > 12 || d < 1 || d > 31) return -1;
  int64_t hh = (s[9] - '0') * 10 + (s[10] - '0');
  int64_t mi = (s[11] - '0') * 10 + (s[12] - '0');
  if (hh < 0 || hh > 23 || mi < 0 || mi > 59) return -1;
  *out = (double)(days_from_civil(y, mo, d) * 86400 + hh * 3600 + mi * 60);
  return 0;
}

/* Splits one CSV line in place; returns the number of fields and fills
 * `starts`/`lens` with the trimmed field extents. Honour double quotes. */
#define CENV_CSV_MAX_FIELDS 32

static int split_csv(char *line, size_t len, const char **starts, size_t *lens) {
  int n = 0;
  size_t i = 0;
  int in_quotes = 0;
  char *write = line;
  size_t wstart = 0;
  (void)len;
  while (i <= len && n < CENV_CSV_MAX_FIELDS) {
    char ch = (i < len) ? line[i] : ',';
    if (in_quotes) {
      if (ch == '"') {
        in_quotes = 0;
      } else {
        *write++ = ch;
      }
    } else if (ch == '"') {
      in_quotes = 1;
    } else if (ch == ',' || i == len) {
      size_t fend = (size_t)(write - line);
      starts[n] = line + wstart;
      lens[n] = fend - wstart;
      n++;
      wstart = fend;
      i++;
      continue;
    } else {
      *write++ = ch;
    }
    i++;
  }
  /* Trim surrounding whitespace and \r. */
  for (int f = 0; f < n; f++) {
    const char *s = starts[f];
    size_t l = lens[f];
    while (l > 0 && (s[0] == ' ' || s[0] == '\t')) {
      s++;
      l--;
    }
    while (l > 0 && (s[l - 1] == ' ' || s[l - 1] == '\t' || s[l - 1] == '\r')) {
      l--;
    }
    starts[f] = s;
    lens[f] = l;
  }
  return n;
}

static int field_is(const char *s, size_t len, const char *want) {
  size_t wl = strlen(want);
  return len == wl && memcmp(s, want, wl) == 0;
}

static int ascii_digit(char ch) { return ch >= '0' && ch <= '9'; }
static int ascii_space(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' ||
         ch == '\v';
}

/* pandas' float parser, ported from pandas/_libs/src/parser/tokenizer.c.
 *
 * read_csv's default is `float_precision=None`, which parsers.pyx maps to
 * `precise_xstrtod` (NOT strtod, NOT fast_float, NOT the legacy xstrtod). It
 * accumulates at most 17 significant digits into a double, drops any further
 * fraction digits without counting them, and then applies the decimal exponent
 * with ONE multiply or divide by an exact tabulated power of ten. Because that
 * accumulation rounds above 2^53 and the single division rounds again, about
 * 0.4 percent of the values in the shipped weather CSVs land one ULP away from
 * the correctly rounded double. ReplayWeatherController replays whatever
 * pandas produced, so this has to replicate the rounding pattern rather than
 * the mathematically ideal result.
 */
static const double pandas_pow10[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
    1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23,
    1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30, 1e31,
    1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39,
    1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47,
    1e48, 1e49, 1e50, 1e51, 1e52, 1e53, 1e54, 1e55,
    1e56, 1e57, 1e58, 1e59, 1e60, 1e61, 1e62, 1e63,
    1e64, 1e65, 1e66, 1e67, 1e68, 1e69, 1e70, 1e71,
    1e72, 1e73, 1e74, 1e75, 1e76, 1e77, 1e78, 1e79,
    1e80, 1e81, 1e82, 1e83, 1e84, 1e85, 1e86, 1e87,
    1e88, 1e89, 1e90, 1e91, 1e92, 1e93, 1e94, 1e95,
    1e96, 1e97, 1e98, 1e99, 1e100, 1e101, 1e102, 1e103,
    1e104, 1e105, 1e106, 1e107, 1e108, 1e109, 1e110, 1e111,
    1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
    1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127,
    1e128, 1e129, 1e130, 1e131, 1e132, 1e133, 1e134, 1e135,
    1e136, 1e137, 1e138, 1e139, 1e140, 1e141, 1e142, 1e143,
    1e144, 1e145, 1e146, 1e147, 1e148, 1e149, 1e150, 1e151,
    1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
    1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167,
    1e168, 1e169, 1e170, 1e171, 1e172, 1e173, 1e174, 1e175,
    1e176, 1e177, 1e178, 1e179, 1e180, 1e181, 1e182, 1e183,
    1e184, 1e185, 1e186, 1e187, 1e188, 1e189, 1e190, 1e191,
    1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
    1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207,
    1e208, 1e209, 1e210, 1e211, 1e212, 1e213, 1e214, 1e215,
    1e216, 1e217, 1e218, 1e219, 1e220, 1e221, 1e222, 1e223,
    1e224, 1e225, 1e226, 1e227, 1e228, 1e229, 1e230, 1e231,
    1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
    1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247,
    1e248, 1e249, 1e250, 1e251, 1e252, 1e253, 1e254, 1e255,
    1e256, 1e257, 1e258, 1e259, 1e260, 1e261, 1e262, 1e263,
    1e264, 1e265, 1e266, 1e267, 1e268, 1e269, 1e270, 1e271,
    1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
    1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287,
    1e288, 1e289, 1e290, 1e291, 1e292, 1e293, 1e294, 1e295,
    1e296, 1e297, 1e298, 1e299, 1e300, 1e301, 1e302, 1e303,
    1e304, 1e305, 1e306, 1e307, 1e308,
};

#define PANDAS_MAX_DIGITS 17

static double pandas_atof(const char *s) {
  const char *p = s;
  while (ascii_space(*p)) p++;
  int negative = 0;
  if (*p == '-') {
    negative = 1;
    p++;
  } else if (*p == '+') {
    p++;
  }
  double number = 0.0;
  int exponent = 0;
  int num_digits = 0;
  int num_decimals = 0;

  while (ascii_digit(*p)) {
    if (num_digits < PANDAS_MAX_DIGITS) {
      number = fma(number, 10.0, (double)(*p - '0'));
      num_digits++;
    } else {
      ++exponent;
    }
    p++;
  }

  if (*p == '.') {
    p++;
    while (num_digits < PANDAS_MAX_DIGITS && ascii_digit(*p)) {
      number = fma(number, 10.0, (double)(*p - '0'));
      p++;
      num_digits++;
      num_decimals++;
    }
    if (num_digits >= PANDAS_MAX_DIGITS) {
      while (ascii_digit(*p)) ++p;
    }
    exponent -= num_decimals;
  }

  if (num_digits == 0) return 0.0;
  if (negative) number = -number;

  if (*p == 'e' || *p == 'E') {
    negative = 0;
    switch (*++p) {
      case '-':
        negative = 1;
        /* fall through */
      case '+':
        p++;
        break;
      default:
        break;
    }
    num_digits = 0;
    int n = 0;
    while (num_digits < PANDAS_MAX_DIGITS && ascii_digit(*p)) {
      n = n * 10 + (*p - '0');
      num_digits++;
      p++;
    }
    if (negative) {
      exponent -= n;
    } else {
      exponent += n;
    }
    if (num_digits == 0) p--;
  }

  if (exponent > 308) {
    return 0.0; /* HUGE_VAL in pandas; the weather CSVs never reach it */
  } else if (exponent > 0) {
    number *= pandas_pow10[exponent];
  } else if (exponent < -308) {
    if (exponent < -616) {
      number = 0.0;
    } else {
      number /= pandas_pow10[-308 - exponent];
      number /= pandas_pow10[308];
    }
  } else {
    number /= pandas_pow10[-exponent];
  }
  return number;
}

static int64_t alloc_handle(weather_table *t) {
  for (int64_t h = 1; h < CENV_WEATHER_MAX_HANDLES; h++) {
    if (g_wx[h] == NULL) {
      g_wx[h] = t;
      return h;
    }
  }
  return 0;
}

/* Takes ownership of the three arrays. */
static int64_t table_from_arrays(double *xp, double *temp_f, double *humidity,
                                 int64_t n) {
  weather_table *t = (weather_table *)calloc(1, sizeof(weather_table));
  if (t == NULL) return 0;
  t->n = n;
  t->xp = xp;
  t->temp_f = temp_f;
  t->humidity = humidity;
  int64_t h = alloc_handle(t);
  if (h == 0) {
    free(t);
    return 0;
  }
  return h;
}

int64_t cenv_weather_create(const double *xp, const double *temp_f,
                            const double *humidity, int64_t n) {
  if (xp == NULL || temp_f == NULL || humidity == NULL || n <= 0) return 0;
  double *a = (double *)malloc((size_t)n * sizeof(double));
  double *b = (double *)malloc((size_t)n * sizeof(double));
  double *c = (double *)malloc((size_t)n * sizeof(double));
  if (a == NULL || b == NULL || c == NULL) {
    free(a);
    free(b);
    free(c);
    return 0;
  }
  memcpy(a, xp, (size_t)n * sizeof(double));
  memcpy(b, temp_f, (size_t)n * sizeof(double));
  memcpy(c, humidity, (size_t)n * sizeof(double));
  int64_t h = table_from_arrays(a, b, c, n);
  if (h == 0) {
    free(a);
    free(b);
    free(c);
  }
  return h;
}

void cenv_weather_destroy(int64_t h) {
  weather_table *t;
  if (h <= 0 || h >= CENV_WEATHER_MAX_HANDLES) return;
  t = g_wx[h];
  if (t == NULL) return;
  free(t->xp);
  free(t->temp_f);
  free(t->humidity);
  free(t);
  g_wx[h] = NULL;
}

int64_t cenv_weather_load_csv(const char *path) {
  if (path == NULL) return 0;
  FILE *f = fopen(path, "rb");
  if (f == NULL) return 0;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return 0;
  }
  long size = ftell(f);
  if (size <= 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return 0;
  }
  char *buf = (char *)malloc((size_t)size + 1);
  if (buf == NULL) {
    fclose(f);
    return 0;
  }
  size_t got = fread(buf, 1, (size_t)size, f);
  fclose(f);
  buf[got] = '\0';

  /* Count lines first so the arrays can be allocated once. */
  int64_t n_lines = 0;
  for (size_t i = 0; i < got; i++) {
    if (buf[i] == '\n') n_lines++;
  }
  if (n_lines < 2) {
    free(buf);
    return 0;
  }

  double *xp = (double *)malloc((size_t)(n_lines) * sizeof(double));
  double *tfp = (double *)malloc((size_t)(n_lines) * sizeof(double));
  double *hup = (double *)malloc((size_t)(n_lines) * sizeof(double));
  if (xp == NULL || tfp == NULL || hup == NULL) {
    free(buf);
    free(xp);
    free(tfp);
    free(hup);
    return 0;
  }

  int col_time = -1, col_tempf = -1, col_hum = -1;
  int64_t n = 0;
  int header_done = 0;
  size_t pos = 0;
  const char *sstarts[CENV_CSV_MAX_FIELDS];
  size_t slens[CENV_CSV_MAX_FIELDS];

  while (pos <= got) {
    size_t end = pos;
    while (end < got && buf[end] != '\n') end++;
    size_t line_len = end - pos;
    if (line_len > 0) {
      /* split_csv rewrites the line in place (quote removal), which is safe:
       * the line is never needed again. */
      char *line = buf + pos;
      int nf = split_csv(line, line_len, sstarts, slens);
      if (!header_done) {
        for (int i = 0; i < nf; i++) {
          if (field_is(sstarts[i], slens[i], "Time")) col_time = i;
          else if (field_is(sstarts[i], slens[i], "TempF")) col_tempf = i;
          else if (field_is(sstarts[i], slens[i], "Humidity")) col_hum = i;
        }
        if (col_time < 0 || col_tempf < 0 || col_hum < 0) break;
        header_done = 1;
      } else if (col_time < nf && col_tempf < nf && col_hum < nf) {
        double seconds;
        if (parse_epoch_seconds(sstarts[col_time], slens[col_time], &seconds) ==
            0) {
          char tmp[64];
          size_t tl = slens[col_tempf];
          char tmp2[64];
          size_t tl2 = slens[col_hum];
          if (tl < sizeof(tmp) && tl2 < sizeof(tmp2)) {
            memcpy(tmp, sstarts[col_tempf], tl);
            tmp[tl] = '\0';
            memcpy(tmp2, sstarts[col_hum], tl2);
            tmp2[tl2] = '\0';
            xp[n] = seconds;
            tfp[n] = pandas_atof(tmp);
            hup[n] = pandas_atof(tmp2);
            n++;
          }
        }
      }
    }
    if (end >= got) break;
    pos = end + 1;
  }
  free(buf);
  if (!header_done || n <= 0) {
    free(xp);
    free(tfp);
    free(hup);
    return 0;
  }
  int64_t h = table_from_arrays(xp, tfp, hup, n);
  if (h == 0) {
    free(xp);
    free(tfp);
    free(hup);
  }
  return h;
}

/* np.interp for one scalar: clamp outside, exact sample on a node, else the
 * slope form on the bracketing interval, exactly as numpy's arr_interp does
 * (binary search for the interval, dy[j] on an exact node). */
double cenv_weather_interp(const double *xp, const double *fp, int64_t n,
                           double x) {
  if (n <= 0) return 0.0;
  if (n == 1) return fp[0];
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  int64_t lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    int64_t mid = (lo + hi) / 2;
    if (xp[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  if (x == xp[lo]) return fp[lo];
  double slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  /* numpy compiles "slope * (x - dx[j]) + dy[j]" with the multiply-add fused
   * (one rounding); -ffp-contract=off here, so call fma() explicitly. */
  return fma(slope, x - xp[lo], fp[lo]);
}

/* nanoseconds since the epoch -> seconds, exact for the simulator's clock
 * (every timestamp is a whole number of seconds). */
static double ns_to_seconds(int64_t ns) {
  int64_t q = ns / 1000000000LL;
  int64_t r = ns % 1000000000LL;
  if (r == 0) return (double)q;
  return (double)q + (double)r * 1e-9;
}

int32_t cenv_weather_query(int64_t h, int64_t t_ns, int32_t which, double *out) {
  weather_table *t;
  if (h <= 0 || h >= CENV_WEATHER_MAX_HANDLES || (t = g_wx[h]) == NULL) {
    return 2;
  }
  if (out == NULL || which < 0 || which > 1) return 2;
  const double x = ns_to_seconds(t_ns);
  const double *fp = which == 0 ? t->temp_f : t->humidity;
  *out = cenv_weather_interp(t->xp, fp, t->n, x);
  return (x < t->xp[0] || x > t->xp[t->n - 1]) ? 1 : 0;
}

double cenv_weather_fahrenheit_to_kelvin(double temp_f) {
  double temp_c = (temp_f - 32.0) * 5.0 / 9.0;
  return temp_c + 273.15;
}

int32_t cenv_weather_copy(int64_t h, double *xp, double *temp_f,
                          double *humidity) {
  weather_table *t;
  if (h <= 0 || h >= CENV_WEATHER_MAX_HANDLES || (t = g_wx[h]) == NULL) {
    return 1;
  }
  if (xp == NULL || temp_f == NULL || humidity == NULL) return 1;
  memcpy(xp, t->xp, (size_t)t->n * sizeof(double));
  memcpy(temp_f, t->temp_f, (size_t)t->n * sizeof(double));
  memcpy(humidity, t->humidity, (size_t)t->n * sizeof(double));
  return 0;
}

int32_t cenv_weather_bounds(int64_t h, double *out_min, double *out_max,
                            int64_t *out_n) {
  weather_table *t;
  if (h <= 0 || h >= CENV_WEATHER_MAX_HANDLES || (t = g_wx[h]) == NULL) {
    return 1;
  }
  if (out_min != NULL) *out_min = t->xp[0];
  if (out_max != NULL) *out_max = t->xp[t->n - 1];
  if (out_n != NULL) *out_n = t->n;
  return 0;
}
