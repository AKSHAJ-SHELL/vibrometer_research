// vibedge on an ESP32 (classic ESP32 e.g. NodeMCU-32S, or ESP32-S3): feature extraction + logistic regression on 8 stored real CWRU windows,
// with a marker pin the Arduino Uno power meter reads. See firmware/README.md for wiring.
//
// Boot:  parity check against the Python reference (prints PASS/FAIL over USB serial at 115200),
//        then per-window timing for each feature set.
// Loop:  forever:  idle 3 s  →  code pulses  →  marker HIGH while computing BLOCK_REPS × 8 windows  →  LOW
//        code: 1 pulse = time_only, 2 pulses = envelope_ratio (40 ms high / 40 ms low, then 400 ms low).
// No Wi-Fi or Bluetooth is started; CPU at 240 MHz.

#include <Arduino.h>

#include "vibedge_check.h"
#include "vibedge_dsp.h"

static const int MARKER_PIN = 4;          // → Uno A1 (see README)
static const int BLOCK_REPS = 10;         // 10 × 8 windows = 80 windows per block
static const uint32_t IDLE_MS = 3000;

static volatile int g_sink = 0;           // keeps the compiler from dropping the work

static double now_us() { return (double)esp_timer_get_time(); }
static void say(const char* s) { Serial.println(s); }

static void code_pulses(int n) {
  for (int i = 0; i < n; ++i) {
    digitalWrite(MARKER_PIN, HIGH); delay(40);
    digitalWrite(MARKER_PIN, LOW);  delay(40);
  }
  delay(400);
}

// One block: marker high while computing BLOCK_REPS passes over the stored windows. Returns µs per window.
static double run_block(bool env) {
  float f[VB_ENV_FEATURES];
  const int64_t t0 = esp_timer_get_time();
  digitalWrite(MARKER_PIN, HIGH);
  for (int r = 0; r < BLOCK_REPS; ++r) {
    for (int w = 0; w < VB_NWIN; ++w) {
      if (env) { vb_features_env(VB_WIN[w], VB_WIN_SCALE[w], VB_WIN_FR[w], f); g_sink += vb_predict_env(f); }
      else     { vb_features_time(VB_WIN[w], VB_WIN_SCALE[w], f);              g_sink += vb_predict_time(f); }
    }
  }
  digitalWrite(MARKER_PIN, LOW);
  return (double)(esp_timer_get_time() - t0) / (BLOCK_REPS * VB_NWIN);
}

void setup() {
  pinMode(MARKER_PIN, OUTPUT);
  digitalWrite(MARKER_PIN, LOW);
  setCpuFrequencyMhz(240);
  Serial.begin(115200);
#if ARDUINO_USB_CDC_ON_BOOT
  Serial.setTxTimeoutMs(0);               // never stall when no USB host is attached (power runs)
#endif
  delay(2000);
  Serial.println("\nvibedge ESP32");
  Serial.printf("chip %s rev %d, %d cores\n", ESP.getChipModel(), (int)ESP.getChipRevision(), (int)ESP.getChipCores());
  Serial.printf("CPU %lu MHz, free heap %lu bytes, PSRAM %lu bytes\n", (unsigned long)getCpuFrequencyMhz(),
                (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getPsramSize());
  if (!vb_init()) {
    Serial.println("vb_init FAILED: not enough memory for the work buffers");
    while (true) delay(1000);
  }
  Serial.printf("free heap after buffers %lu bytes\n", (unsigned long)ESP.getFreeHeap());
  const bool ok = vb_check_all(now_us, say);
  Serial.println(ok ? "DEVICE CHECK PASS" : "DEVICE CHECK FAIL");
}

void loop() {
  static uint32_t cycle = 0;
  delay(IDLE_MS);
  code_pulses(1);
  const double t_time = run_block(false);
  delay(IDLE_MS);
  code_pulses(2);
  const double t_env = run_block(true);
  Serial.printf("cycle %lu  time_only %.0f us/window  envelope_ratio %.0f us/window  (%d windows per block)\n",
                (unsigned long)++cycle, t_time, t_env, BLOCK_REPS * VB_NWIN);
}
