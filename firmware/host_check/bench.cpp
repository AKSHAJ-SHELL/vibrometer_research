// Native timing of the on-device pipeline (the exact ESP32 code) on this computer, one core.
// Prints one line per feature set: "<set> <microseconds per window>".
//   built and run by scripts/bench_gateway.py
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <vector>

#include "vibedge_config.h"
#include "vibedge_dsp.h"
#include "vibedge_windows.h"

static volatile int g_sink = 0;

template <bool ENV>
static double us_per_window(int reps) {
  std::vector<double> per_rep;
  float f[VB_ENV_FEATURES];
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int w = 0; w < VB_NWIN; ++w) {
      if (ENV) { vb_features_env(VB_WIN[w], VB_WIN_SCALE[w], VB_WIN_FR[w], f); g_sink += vb_predict_env(f); }
      else     { vb_features_time(VB_WIN[w], VB_WIN_SCALE[w], f);              g_sink += vb_predict_time(f); }
    }
    per_rep.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / VB_NWIN);
  }
  std::sort(per_rep.begin(), per_rep.end());
  return per_rep[per_rep.size() / 2];   // median over repetitions
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 30;
  if (!vb_init()) { fprintf(stderr, "vb_init failed\n"); return 2; }
  us_per_window<false>(2); us_per_window<true>(2);   // warm-up
  printf("time_only %.2f\n", us_per_window<false>(reps));
  printf("envelope_ratio %.2f\n", us_per_window<true>(reps));
  printf("window_samples %d\nfs %.1f\n", VB_N, (double)VB_FS);
  vb_free();
  return 0;
}
