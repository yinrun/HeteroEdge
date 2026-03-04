#pragma once
#include "common.h"

// Initialize QNN HTP: dlopen, backend, device, power, graph, register buffers.
// The IonBuffer A/B/C are allocated externally (main) and registered here.
// partition_size: NPU processes [0, partition_size) of each buffer (0 = full buffer).
// force_cores: override auto-detected core count (0 = auto).
bool htp_init(const IonBuffer& A, const IonBuffer& B, const IonBuffer& C,
              size_t partition_size = 0, int force_cores = 0);

// Run bandwidth test.  If barrier != nullptr, waits on it after warmup.
BandwidthResult htp_run(int num_warmup, int num_iters, SpinBarrier* barrier);

// Print device info.
void htp_print_info();

// Release all QNN resources (deregister mem, free graph/context/backend, dlclose).
void htp_cleanup();
