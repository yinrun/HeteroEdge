#pragma once
#include "common.h"

// NPU RMSNorm engine using QNN HTP.
// Accepts external ION buffers for zero-copy sharing with GPU.
// Init must be called from main thread (QNN is not thread-safe for init).
// execute_blocking() can be called from any thread.

bool npu_init(int hidden_dim, float epsilon,
              const IonBuffer& ion_input, const IonBuffer& ion_output);

// Blocking: calls graphExecute. Returns wall-clock time in us.
double npu_execute_blocking();

void npu_print_info();
void npu_cleanup();
