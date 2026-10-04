// Host check: runs the firmware's DSP core on this computer and compares it with the Python
// reference (see vibedge_check.h). Build and run with `make device-check` from the repo root.
#include <stdio.h>
#include <chrono>

#include "vibedge_check.h"

static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void say(const char* s) { printf("%s\n", s); }

int main() {
  if (!vb_init()) { printf("vb_init failed\n"); return 2; }
  const bool ok = vb_check_all(now_us, say);
  vb_free();
  printf(ok ? "HOST CHECK PASS\n" : "HOST CHECK FAIL\n");
  return ok ? 0 : 1;
}
