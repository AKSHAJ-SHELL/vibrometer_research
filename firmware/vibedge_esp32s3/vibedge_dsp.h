// vibedge on-device feature extraction + logistic regression.
// A line-by-line port of src/vibedge/device_ref.py (the Python reference it is checked against).
// Plain C++: compiles for the ESP32-S3 (Arduino) and on a desktop for the host check.
#pragma once
#include <stdint.h>

#define VB_TIME_FEATURES 17   // time_only:      10 time-domain + 7 broadband
#define VB_ENV_FEATURES  39   // envelope_ratio: + 6 order + 12 envelope harmonics + 4 sidebands

// Allocate work buffers (~225 KB) and the twiddle table. Returns false if memory runs out.
bool vb_init();
void vb_free();

// Features for one window of VB_N int16 samples (value = q[i] * scale).
// fr_hz = shaft rate (measured speed). Writes VB_TIME_FEATURES / VB_ENV_FEATURES floats.
void vb_features_time(const int16_t* q, float scale, float* out);
void vb_features_env(const int16_t* q, float scale, float fr_hz, float* out);

// Predicted class index (same numbering as vibedge.taxonomy.CLASSES).
int vb_predict_time(const float* f);
int vb_predict_env(const float* f);

// Standardised feature (f - mean) / scale, for comparing against the reference in model units.
float vb_standardise_time(int i, float v);
float vb_standardise_env(int i, float v);
