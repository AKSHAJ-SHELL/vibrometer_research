// Port of src/vibedge/device_ref.py — keep the two in step.
// Speed: the ESP32-S3 has a single-precision FPU only (double is emulated in software), so
// per-sample work is float32. Sums are kept accurate by accumulating 256 samples in float,
// then flushing to a double (BlockSum). Band edges use exact double comparisons on index ranges.
#include "vibedge_dsp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "vibedge_config.h"
#include "vibedge_model.h"

static const double kEps = 1e-12;
static const double kPi = 3.14159265358979323846;

static float* g_x = nullptr;     // [VB_N]          signal, later the envelope
static float* g_re = nullptr;    // [VB_N + 2*PAD]  FFT real part / filtfilt extension buffer
static float* g_im = nullptr;    // [VB_N]          FFT imaginary part
static float* g_mag = nullptr;   // [VB_N/2 + 1]    one-sided spectrum magnitude
static float* g_sin = nullptr;   // [VB_N/4 + 1]    quarter-wave sine table

static const int kPadMax = (VB_BP_PAD > VB_DEC_PAD) ? VB_BP_PAD : VB_DEC_PAD;

bool vb_init() {
  g_x = (float*)malloc(sizeof(float) * VB_N);
  g_re = (float*)malloc(sizeof(float) * (VB_N + 2 * kPadMax));
  g_im = (float*)malloc(sizeof(float) * VB_N);
  g_mag = (float*)malloc(sizeof(float) * (VB_N / 2 + 1));
  g_sin = (float*)malloc(sizeof(float) * (VB_N / 4 + 1));
  if (!g_x || !g_re || !g_im || !g_mag || !g_sin) { vb_free(); return false; }
  for (int k = 0; k <= VB_N / 4; ++k) g_sin[k] = (float)sin(2.0 * kPi * k / VB_N);
  return true;
}

void vb_free() {
  free(g_x); free(g_re); free(g_im); free(g_mag); free(g_sin);
  g_x = g_re = g_im = g_mag = g_sin = nullptr;
}

// cos/sin(2*pi*k/VB_N) for 0 <= k < VB_N/2, from the quarter-wave table.
static inline void twiddle(int k, float* c, float* s) {
  const int q = VB_N / 4;
  if (k <= q) { *s = g_sin[k]; *c = g_sin[q - k]; }
  else        { *s = g_sin[VB_N / 2 - k]; *c = -g_sin[k - q]; }
}

// cos(2*pi*i/n) for n a power of two <= VB_N and 0 <= i < n, from the table.
static inline float cos_n(int i, int n) {
  int k = i * (VB_N / n);              // same angle on the VB_N grid, in [0, VB_N)
  if (k > VB_N / 2) k = VB_N - k;      // cos is symmetric
  if (k == VB_N / 2) return -1.0f;
  float c, s;
  twiddle(k, &c, &s);
  return c;
}

// Float accumulation in blocks of 256, flushed to double: accurate, and cheap on a float-only FPU.
struct BlockSum {
  double total = 0; float block = 0; int n = 0;
  inline void add(float v) { block += v; if (++n == 256) { total += block; block = 0; n = 0; } }
  inline double value() const { return total + block; }
};

// Bins k in [0, nbin) with lo <= k*df <= hi (the same inclusive test as numpy), as [k0, k1].
static inline bool bin_range(double lo, double hi, double df, int nbin, int* k0, int* k1) {
  int a = (int)floor(lo / df); if (a < 0) a = 0;
  while (a < nbin && a * df < lo) ++a;
  int b = (int)ceil(hi / df); if (b > nbin - 1) b = nbin - 1;
  while (b >= 0 && b * df > hi) --b;
  *k0 = a; *k1 = b;
  return a <= b;
}

// In-place radix-2 complex FFT of size n (power of two, n <= VB_N). inverse: unscaled.
static void fft(float* re, float* im, int n, bool inverse) {
  for (int i = 1, j = 0; i < n; ++i) {             // bit reversal
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  const float sgn = inverse ? 1.0f : -1.0f;
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1, stride = VB_N / len;
    for (int i = 0; i < n; i += len) {
      for (int j = 0; j < half; ++j) {
        float c, s;
        twiddle(j * stride, &c, &s);
        s *= sgn;
        const int a = i + j, b = a + half;
        const float tr = re[b] * c - im[b] * s;
        const float ti = re[b] * s + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
      }
    }
  }
}

// One-sided Hann-windowed magnitude |rfft(x*w)| * 2 / sum(w) into mag[0..n/2]. Uses g_re/g_im.
static void spectrum(const float* x, int n, float* mag) {
  BlockSum wsum_b;
  for (int i = 0; i < n; ++i) {
    const float w = 0.5f - 0.5f * cos_n(i, n);   // periodic Hann
    wsum_b.add(w);
    g_re[i] = x[i] * w;
    g_im[i] = 0.0f;
  }
  const double wsum = wsum_b.value();
  fft(g_re, g_im, n, false);
  const float g = (float)(2.0 / (wsum + kEps));
  for (int k = 0; k <= n / 2; ++k) mag[k] = sqrtf(g_re[k] * g_re[k] + g_im[k] * g_im[k]) * g;
}

// ---------------------------------------------------------------- time domain (10)
static void time_features(const float* x, int n, float* out) {
  BlockSum s, s2, sa, ssa;
  float peak = 0, mx = x[0], mn = x[0];
  for (int i = 0; i < n; ++i) {
    const float v = x[i], a = fabsf(v);
    s.add(v); s2.add(v * v); sa.add(a); ssa.add(sqrtf(a));
    if (a > peak) peak = a;
    if (v > mx) mx = v;
    if (v < mn) mn = v;
  }
  const float mu = (float)(s.value() / n);
  BlockSum b2, b3, b4;
  for (int i = 0; i < n; ++i) {
    const float d = x[i] - mu, d2 = d * d;
    b2.add(d2); b3.add(d2 * d); b4.add(d2 * d2);
  }
  const double m2 = b2.value() / n, m3 = b3.value() / n, m4 = b4.value() / n;
  const double rms = sqrt(s2.value() / n), mean_abs = sa.value() / n;
  const double msa = (ssa.value() / n) * (ssa.value() / n);
  out[0] = (float)rms;
  out[1] = peak;
  out[2] = mx - mn;
  out[3] = (float)sqrt(m2);
  out[4] = (float)(m4 / (m2 * m2));            // Pearson kurtosis, population moments
  out[5] = (float)(m3 / pow(m2, 1.5));
  out[6] = (float)(peak / (rms + kEps));
  out[7] = (float)(rms / (mean_abs + kEps));
  out[8] = (float)(peak / (mean_abs + kEps));
  out[9] = (float)(peak / (msa + kEps));
}

// ---------------------------------------------------------------- broadband (7)
static double power_sum(const float* mag, int k0, int k1) {
  BlockSum b;
  for (int k = k0; k <= k1; ++k) b.add(mag[k] * mag[k]);
  return b.value();
}

static void broadband(const float* mag, int nbin, double df, double fs, float* out) {
  const double ptot = power_sum(mag, 0, nbin - 1), psum = ptot + kEps;
  const float inv = (float)(1.0 / psum);
  BlockSum cen, ent, lsum, msum;
  for (int k = 0; k < nbin; ++k) {
    const float p = mag[k] * mag[k], pn = p * inv;
    cen.add((float)(k * df) * pn);
    if (pn > 1e-30f) ent.add(-pn * log2f(pn));   // terms below 1e-30 contribute < 1e-28
    lsum.add(logf(mag[k] + 1e-12f));
    msum.add(mag[k]);
  }
  const double flat = exp(lsum.value() / nbin) / (msum.value() / nbin + kEps);
  const double target = 0.85 * ptot;
  BlockSum c; int ridx = nbin - 1;
  for (int k = 0; k < nbin; ++k) { c.add(mag[k] * mag[k]); if (c.value() >= target) { ridx = k; break; } }
  const double hi46 = (6000.0 < fs / 2 - 1) ? 6000.0 : fs / 2 - 1;
  // half-open bands [lo, hi): drop a bin that lands exactly on hi
  auto band = [&](double lo, double hi) {
    int k0, k1;
    if (!bin_range(lo, hi, df, nbin, &k0, &k1)) return 0.0;
    if (k1 * df >= hi) --k1;
    return k0 <= k1 ? power_sum(mag, k0, k1) / psum : 0.0;
  };
  out[0] = (float)cen.value(); out[1] = (float)ent.value(); out[2] = (float)flat; out[3] = (float)(ridx * df);
  out[4] = (float)band(1000, 2000); out[5] = (float)band(2000, 4000); out[6] = (float)band(4000, hi46);
}

// ---------------------------------------------------------------- order domain (6)
static void order_features(const float* mag, int nbin, double df, double fr, float* out) {
  static const double orders[6] = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0};
  const double total = power_sum(mag, 0, nbin - 1) + kEps;
  for (int o = 0; o < 6; ++o) {
    const double c = orders[o] * fr;
    int k0, k1;
    out[o] = bin_range(c * (1 - VB_TOL), c * (1 + VB_TOL), df, nbin, &k0, &k1)
                 ? (float)(power_sum(mag, k0, k1) / total) : 0.0f;
  }
}

// ---------------------------------------------------------------- filters
// scipy filtfilt/sosfiltfilt semantics: odd extension by pad, DF-II-transposed sections,
// steady-state initial conditions zi * first sample, forward then backward. In place on x[0..n).
static void sos_pass(const float (*sos)[6], const float (*zi)[2], int nsec, float* y, int len, double x0) {
  for (int s = 0; s < nsec; ++s) {
    const float b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2], a1 = sos[s][4], a2 = sos[s][5];
    float z0 = (float)(zi[s][0] * x0), z1 = (float)(zi[s][1] * x0);
    // scipy cascades section by section with each section's zi scaled by the ORIGINAL x0
    for (int i = 0; i < len; ++i) {
      const float v = y[i];
      const float out = b0 * v + z0;
      z0 = b1 * v - a1 * out + z1;
      z1 = b2 * v - a2 * out;
      y[i] = out;
    }
  }
}

static void filtfilt(const float (*sos)[6], const float (*zi)[2], int nsec, int pad, float* x, int n) {
  float* e = g_re;  // extension buffer, n + 2*pad
  for (int i = 0; i < pad; ++i) e[i] = 2.0f * x[0] - x[pad - i];
  memcpy(e + pad, x, sizeof(float) * n);
  for (int i = 0; i < pad; ++i) e[pad + n + i] = 2.0f * x[n - 1] - x[n - 2 - i];
  const int len = n + 2 * pad;
  sos_pass(sos, zi, nsec, e, len, e[0]);
  for (int i = 0, j = len - 1; i < j; ++i, --j) { float t = e[i]; e[i] = e[j]; e[j] = t; }
  sos_pass(sos, zi, nsec, e, len, e[0]);
  for (int i = 0; i < n; ++i) x[i] = e[len - 1 - pad - i];  // reverse back and drop the padding
}

// ---------------------------------------------------------------- envelope
static double band_energy(const float* spec, int nbin, double df, double center) {
  if (center <= 0) return 0.0;
  int k0, k1;
  if (bin_range(center * (1 - VB_TOL), center * (1 + VB_TOL), df, nbin, &k0, &k1))
    return power_sum(spec, k0, k1);
  int best = (int)floor(center / df + 0.5);        // nearest bin (numpy argmin: first on a tie)
  if (best > nbin - 1) best = nbin - 1;
  if (best > 0 && fabs((best - 1) * df - center) <= fabs(best * df - center)) --best;
  return (double)spec[best] * spec[best];
}

// x holds the signal on entry; on exit g_mag[0..nd/2] holds the envelope spectrum. Returns nd.
static int envelope_spectrum(float* x, int n) {
  filtfilt(VB_BP_SOS, VB_BP_ZI, VB_BP_NSEC, VB_BP_PAD, x, n);          // band-pass 2–5 kHz
  for (int i = 0; i < n; ++i) { g_re[i] = x[i]; g_im[i] = 0.0f; }       // analytic signal (Hilbert)
  fft(g_re, g_im, n, false);
  for (int k = 1; k < n / 2; ++k) { g_re[k] *= 2.0f; g_im[k] *= 2.0f; }
  for (int k = n / 2 + 1; k < n; ++k) { g_re[k] = 0.0f; g_im[k] = 0.0f; }
  fft(g_re, g_im, n, true);
  const float inv = 1.0f / n;
  for (int i = 0; i < n; ++i) x[i] = sqrtf(g_re[i] * g_re[i] + g_im[i] * g_im[i]) * inv;
  filtfilt(VB_DEC_SOS, VB_DEC_ZI, VB_DEC_NSEC, VB_DEC_PAD, x, n);     // anti-alias, then decimate
  const int nd = n / VB_DECIM;
  BlockSum m;
  for (int i = 0; i < nd; ++i) { x[i] = x[i * VB_DECIM]; m.add(x[i]); }
  const float mean = (float)(m.value() / nd);
  for (int i = 0; i < nd; ++i) x[i] -= mean;
  spectrum(x, nd, g_mag);
  return nd;
}

static void envelope_features(const float* spec, int nbin, double df, double fr, float* out) {
  const double rd = (VB_GEO_D / VB_GEO_PITCH) * cos(VB_GEO_PHI_DEG * kPi / 180.0);
  const double bpfo = VB_GEO_N * fr / 2.0 * (1.0 - rd);
  const double bpfi = VB_GEO_N * fr / 2.0 * (1.0 + rd);
  const double ftf = fr / 2.0 * (1.0 - rd);
  const double bsf = VB_GEO_PITCH * fr / (2.0 * VB_GEO_D) * (1.0 - rd * rd);
  const double ball = 2.0 * bsf;
  const double total = power_sum(spec, 0, nbin - 1) + kEps;
  const double funds[4] = {bpfo, bpfi, ball, ftf};
  int o = 0;
  for (int f = 0; f < 4; ++f)
    for (int h = 1; h <= 3; ++h) out[o++] = (float)(band_energy(spec, nbin, df, funds[f] * h) / total);
  auto sb = [&](double c, double sp) {
    return (band_energy(spec, nbin, df, c - sp) + band_energy(spec, nbin, df, c + sp)) / total;
  };
  out[o++] = (float)sb(bpfi, fr);
  out[o++] = (float)sb(ball, ftf);
  out[o++] = (float)sb(bpfo, fr);
  out[o++] = (float)sb(2.0 * bpfi, fr);
}

// ---------------------------------------------------------------- public API
static void load(const int16_t* q, float scale) {
  for (int i = 0; i < VB_N; ++i) g_x[i] = q[i] * scale;
}

void vb_features_time(const int16_t* q, float scale, float* out) {
  load(q, scale);
  time_features(g_x, VB_N, out);                 // [0..10)
  spectrum(g_x, VB_N, g_mag);
  broadband(g_mag, VB_N / 2 + 1, (double)VB_FS / VB_N, VB_FS, out + 10);   // [10..17)
}

void vb_features_env(const int16_t* q, float scale, float fr_hz, float* out) {
  // column order = device_ref.SET_NAMES["envelope_ratio"]: time(10) order(6) env(12) sideband(4) broadband(7)
  load(q, scale);
  time_features(g_x, VB_N, out);
  spectrum(g_x, VB_N, g_mag);
  const double df = (double)VB_FS / VB_N;
  order_features(g_mag, VB_N / 2 + 1, df, fr_hz, out + 10);
  broadband(g_mag, VB_N / 2 + 1, df, VB_FS, out + 32);
  const int nd = envelope_spectrum(g_x, VB_N);
  envelope_features(g_mag, nd / 2 + 1, ((double)VB_FS / VB_DECIM) / nd, fr_hz, out + 16);
}

template <int NF, int NC>
static int predict(const float* f, const float* mean, const float* scale, const float (*coef)[NF],
                   const float* intercept, const int* classes) {
  float z[NF];
  for (int i = 0; i < NF; ++i) z[i] = (f[i] - mean[i]) / scale[i];
  int best = 0; float bestv = -1e30f;
  for (int c = 0; c < NC; ++c) {
    float s = intercept[c];
    for (int i = 0; i < NF; ++i) s += coef[c][i] * z[i];
    if (s > bestv) { bestv = s; best = c; }
  }
  return classes[best];
}

int vb_predict_time(const float* f) {
  return predict<VB_TIME_NF, VB_TIME_NC>(f, VB_TIME_MEAN, VB_TIME_SCALE, VB_TIME_COEF, VB_TIME_INTERCEPT, VB_TIME_CLASSES);
}
int vb_predict_env(const float* f) {
  return predict<VB_ENV_NF, VB_ENV_NC>(f, VB_ENV_MEAN, VB_ENV_SCALE, VB_ENV_COEF, VB_ENV_INTERCEPT, VB_ENV_CLASSES);
}
float vb_standardise_time(int i, float v) { return (v - VB_TIME_MEAN[i]) / VB_TIME_SCALE[i]; }
float vb_standardise_env(int i, float v) { return (v - VB_ENV_MEAN[i]) / VB_ENV_SCALE[i]; }
