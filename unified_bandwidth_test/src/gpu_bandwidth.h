#pragma once
#include "common.h"

// Initialize OpenCL: import ION buffers, compile kernel.
// offset/partition_size: sub-region within unified buffer (0/0 = full buffer).
bool gpu_init(const IonBuffer& A, const IonBuffer& B, const IonBuffer& C,
              const char* kernel_path,
              size_t offset = 0, size_t partition_size = 0);

// Run bandwidth test.  If barrier != nullptr, waits on it after warmup.
BandwidthResult gpu_run(int num_warmup, int num_iters, SpinBarrier* barrier);

// Print device info.
void gpu_print_info();

// Release all OpenCL resources.
void gpu_cleanup();
