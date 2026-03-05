#pragma once
#include "common.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

// GPU RMSNorm engine with blocking and non-blocking execution modes.
// Accepts external ION buffers for zero-copy sharing with NPU.

bool gpu_init(int hidden_dim, float epsilon,
              const IonBuffer& ion_input, const IonBuffer& ion_output,
              const char* kernel_path);

// Enable flag-based fast sync: GPU kernel writes flag to shared memory on completion.
// Must be called after gpu_init(). Pass an ION buffer of >= 4 bytes.
bool gpu_enable_flag(const IonBuffer& ion_flag);

// Disable flag mode: kernel arg set to NULL, kernel skips flag write.
void gpu_disable_flag();

// Get CPU-mapped flag pointer (for direct polling). Only valid when flag is enabled.
volatile uint32_t* gpu_get_flag_ptr();

// Blocking: enqueue + clFinish. Returns total wall-clock time in us.
// Fills gpu_compute_us via profiling if profiling is enabled.
double gpu_execute_blocking(double* gpu_compute_us);

// Non-blocking: enqueue + clFlush. Returns cl_event for polling.
cl_event gpu_execute_nonblocking();

// Submit for flag-based sync: reset flag, enqueue, clFlush. No waiting.
void gpu_submit();

// Poll event status. Returns true if CL_COMPLETE.
bool gpu_poll_event(cl_event evt);

// Get GPU compute time from profiling info on a completed event. Returns us.
double gpu_event_compute_us(cl_event evt);

// Get full profiling breakdown from a completed event. All values in us.
struct GpuProfilingInfo {
  double queued_us;    // COMMAND_QUEUED (device clock, relative to first event)
  double submit_us;    // COMMAND_SUBMIT
  double start_us;     // COMMAND_START
  double end_us;       // COMMAND_END
  double queue_delay;  // submit - queued: driver processing before GPU submission
  double submit_delay; // start - submit: GPU scheduling latency
  double compute;      // end - start: actual kernel execution
  double total_device; // end - queued: total device-side time
};
GpuProfilingInfo gpu_event_profiling(cl_event evt);

// Execute blocking without profiling (uses clFinish only, returns wall time)
double gpu_execute_blocking_noprof();

void gpu_print_info();
void gpu_cleanup();
