#pragma once
#include "common.h"

// NPU RMSNorm engine using QNN HTP.
// Accepts external ION buffers for zero-copy sharing with GPU.
// Init must be called from main thread (QNN is not thread-safe for init).
// execute_blocking() can be called from any thread.

bool npu_init(int hidden_dim, float epsilon,
              const IonBuffer& ion_input, const IonBuffer& ion_output);

// Initialize NPU with custom FlagPoll RMSNorm op.
// flag ION buffer is shared with GPU: GPU writes flag=1 after computation,
// HVX kernel polls on it before computing RMSNorm.
// poll_count ION buffer receives the number of poll iterations (diagnostic).
bool npu_init_flagpoll(int hidden_dim, float epsilon,
                       const IonBuffer& ion_input, const IonBuffer& ion_output,
                       const IonBuffer& ion_flag, const IonBuffer& ion_poll_count);

// Blocking: calls graphExecute. Returns wall-clock time in us.
double npu_execute_blocking();

// Get the poll_count value from the last NPU_POLL execution (diagnostic).
uint32_t npu_get_last_poll_count();

void npu_print_info();
void npu_cleanup();
