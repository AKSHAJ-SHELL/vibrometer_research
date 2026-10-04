// Parity check shared by the host harness and the ESP32-S3 sketch: runs the DSP core on the
// exported windows and compares with vibedge_golden.h (Python reference, float64).
// Pass = every in-band feature within 1e-3 of the reference in standardised (model-input)
// units, and every prediction identical.
#pragma once
#include <math.h>
#include <stdio.h>

#include "vibedge_config.h"
#include "vibedge_dsp.h"
#include "vibedge_golden.h"
#include "vibedge_model.h"
#include "vibedge_windows.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr double kTol = 1e-3;

// Envelope features whose ±2% band holds no envelope-spectrum bin (centre above the envelope
// Nyquist, VB_FS / VB_DECIM / 2). Both implementations then read the last bin, deep in the
// decimation filter's stopband (~1e-8 of total energy), so float32 vs float64 differ in noise.
// Defined from the physics, before looking at errors. Index = position in the env feature vector.
inline void out_of_band(double fr, bool* oob) {
  const double rd = (VB_GEO_D / VB_GEO_PITCH) * cos(VB_GEO_PHI_DEG * M_PI / 180.0);
  const double bpfo = VB_GEO_N * fr / 2 * (1 - rd), bpfi = VB_GEO_N * fr / 2 * (1 + rd);
  const double ftf = fr / 2 * (1 - rd), ball = 2 * VB_GEO_PITCH * fr / (2 * VB_GEO_D) * (1 - rd * rd);
  const int nd = VB_N / VB_DECIM;
  const double df = (VB_FS / VB_DECIM) / nd, fmax = (nd / 2) * df;
  auto empty = [&](double c) { return c > 0 && c * (1 - VB_TOL) > fmax; };
  for (int i = 0; i < VB_ENV_FEATURES; ++i) oob[i] = false;
  const double funds[4] = {bpfo, bpfi, ball, ftf};
  for (int f = 0; f < 4; ++f) for (int h = 1; h <= 3; ++h) oob[16 + f * 3 + (h - 1)] = empty(funds[f] * h);
  oob[28] = empty(bpfi + fr);        // a sideband pair is out of band if either side is
  oob[29] = empty(ball + ftf);
  oob[30] = empty(bpfo + fr);
  oob[31] = empty(2 * bpfi + fr);
}

template <int NF, int NC>
inline bool vb_check(const char* name, const double (*gold)[NF], const int* gold_pred, bool env,
                     int (*predict)(const float*), float (*stdz)(int, float), const float (*coef)[NF],
                     double (*now_us)(), void (*say)(const char*)) {
  char line[200];
  double worst = 0, worst_oob = 0, worst_score = 0; int worst_w = -1, worst_f = -1, pred_ok = 0, n_oob = 0;
  double us = 0;
  for (int w = 0; w < VB_NWIN; ++w) {
    float f[NF];
    bool oob[VB_ENV_FEATURES] = {false};
    if (env) out_of_band(VB_WIN_FR[w], oob);
    const double t0 = now_us();
    if (env) vb_features_env(VB_WIN[w], VB_WIN_SCALE[w], VB_WIN_FR[w], f);
    else vb_features_time(VB_WIN[w], VB_WIN_SCALE[w], f);
    const int p = predict(f);
    us += now_us() - t0;
    pred_ok += (p == gold_pred[w]);
    double dz[NF];
    for (int i = 0; i < NF; ++i) {
      dz[i] = (double)stdz(i, f[i]) - (double)stdz(i, (float)gold[w][i]);
      const double e = fabs(dz[i]);
      if (oob[i]) { ++n_oob; if (e > worst_oob) worst_oob = e; continue; }
      if (e > worst) { worst = e; worst_w = w; worst_f = i; }
    }
    for (int c = 0; c < NC; ++c) {           // effect on each class's decision score
      double ds = 0;
      for (int i = 0; i < NF; ++i) ds += coef[c][i] * dz[i];
      if (fabs(ds) > worst_score) worst_score = fabs(ds);
    }
  }
  const bool ok = worst <= kTol && pred_ok == VB_NWIN;
  snprintf(line, sizeof line, "%-15s in-band worst standardised error %.2e (window %d, feature %d)  predictions %d/%d  %s",
           name, worst, worst_w, worst_f, pred_ok, VB_NWIN, ok ? "PASS" : "FAIL");
  say(line);
  if (env) {
    snprintf(line, sizeof line, "%-15s out-of-band features: %d (window x feature) cases, worst standardised error %.2e", "", n_oob, worst_oob);
    say(line);
  }
  snprintf(line, sizeof line, "%-15s worst change in any class decision score %.2e   %.0f us/window", "", worst_score, us / VB_NWIN);
  say(line);
  return ok;
}


// Both feature sets; returns true if both pass.
inline bool vb_check_all(double (*now_us)(), void (*say)(const char*)) {
  bool ok = vb_check<VB_TIME_FEATURES, VB_TIME_NC>("time_only", VB_GOLD_TIME, VB_GOLD_TIME_PRED, false,
                                                    vb_predict_time, vb_standardise_time, VB_TIME_COEF, now_us, say);
  ok &= vb_check<VB_ENV_FEATURES, VB_ENV_NC>("envelope_ratio", VB_GOLD_ENV, VB_GOLD_ENV_PRED, true,
                                             vb_predict_env, vb_standardise_env, VB_ENV_COEF, now_us, say);
  return ok;
}
