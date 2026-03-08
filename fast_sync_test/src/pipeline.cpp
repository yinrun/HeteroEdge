#include "pipeline.h"
#include "gpu_engine.h"
#include "npu_engine.h"

#include <atomic>
#include <thread>
#include <random>
#include <unistd.h>
#include <sched.h>

// ARM yield hint: lighter than sched_yield (no context switch), just a CPU hint
#if defined(__aarch64__)
static inline void cpu_pause() { asm volatile("yield"); }
#else
static inline void cpu_pause() { }
#endif

// Pin current thread to a specific CPU core. Returns true on success.
static bool pin_to_core(int core_id) {
  if (core_id < 0) return false;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(core_id, &mask);
  return sched_setaffinity(0, sizeof(mask), &mask) == 0;
}

// ── Mode 1: Sequential Blocking ─────────────────────────────────────────────
static PipelineResult run_sequential(int num_steps) {
  PipelineResult result;
  result.steps.reserve(num_steps);

  double total_t0 = now_us();
  for (int i = 0; i < num_steps; ++i) {
    StepTiming st = {};
    double step_t0 = now_us();

    // GPU: enqueue + clFinish
    double gpu_compute = 0;
    double gpu_wall = gpu_execute_blocking(&gpu_compute);
    st.gpu_compute_us = gpu_compute;
    st.gpu_sync_us = gpu_wall - gpu_compute;

    // NPU: blocking graphExecute
    double npu_t0 = now_us();
    st.npu_compute_us = npu_execute_blocking();
    st.npu_sync_us = 0;  // embedded in graphExecute

    st.step_total_us = now_us() - step_t0;
    result.steps.push_back(st);
  }
  result.total_us = now_us() - total_t0;
  result.num_steps = num_steps;
  result.success = true;
  return result;
}

// ── Mode 2: Threaded + clFinish ──────────────────────────────────────────────
static PipelineResult run_threaded_clfinish(int num_steps, int npu_core) {
  PipelineResult result;
  result.steps.reserve(num_steps);

  std::atomic<uint32_t> npu_start{0};
  std::atomic<uint32_t> npu_done{0};
  std::atomic<bool> running{true};
  std::vector<double> npu_times;
  npu_times.reserve(num_steps);

  // NPU worker thread
  std::thread npu_thread([&, npu_core]() {
    pin_to_core(npu_core);
    while (running.load(std::memory_order_acquire)) {
      // Wait for signal to start (yield to reduce contention)
      while (!npu_start.load(std::memory_order_acquire)) {
        if (!running.load(std::memory_order_relaxed)) return;
        cpu_pause();
      }
      npu_start.store(0, std::memory_order_relaxed);

      double exec_us = npu_execute_blocking();
      npu_times.push_back(exec_us);

      npu_done.store(1, std::memory_order_release);
    }
  });

  double total_t0 = now_us();
  for (int i = 0; i < num_steps; ++i) {
    StepTiming st = {};
    double step_t0 = now_us();

    // GPU: enqueue + clFinish (still blocking)
    double gpu_compute = 0;
    double gpu_wall = gpu_execute_blocking(&gpu_compute);
    st.gpu_compute_us = gpu_compute;
    st.gpu_sync_us = gpu_wall - gpu_compute;

    // Signal NPU to start
    npu_start.store(1, std::memory_order_release);

    // Wait for NPU completion (spin with yield)
    double npu_wait_t0 = now_us();
    while (!npu_done.load(std::memory_order_acquire))
      cpu_pause();
    npu_done.store(0, std::memory_order_relaxed);
    double npu_wait_t1 = now_us();

    st.npu_compute_us = npu_times.back();
    st.npu_sync_us = (npu_wait_t1 - npu_wait_t0) - st.npu_compute_us;
    if (st.npu_sync_us < 0) st.npu_sync_us = 0;

    st.step_total_us = now_us() - step_t0;
    result.steps.push_back(st);
  }
  result.total_us = now_us() - total_t0;

  running.store(false, std::memory_order_release);
  npu_start.store(1, std::memory_order_release);  // wake thread to exit
  npu_thread.join();

  result.num_steps = num_steps;
  result.success = true;
  return result;
}

// ── Mode 3: Event Poll (clFlush + cl_event polling, driver-level) ────────────
static PipelineResult run_event_poll(int num_steps, int usleep_hint, int npu_core) {
  PipelineResult result;
  result.steps.reserve(num_steps);

  std::atomic<uint32_t> npu_start{0};
  std::atomic<uint32_t> npu_done{0};
  std::atomic<bool> running{true};
  std::vector<double> npu_times;
  npu_times.reserve(num_steps);

  // NPU worker thread (yield while waiting to reduce CPU contention)
  std::thread npu_thread([&, npu_core]() {
    pin_to_core(npu_core);
    while (running.load(std::memory_order_acquire)) {
      while (!npu_start.load(std::memory_order_acquire)) {
        if (!running.load(std::memory_order_relaxed)) return;
        cpu_pause();
      }
      npu_start.store(0, std::memory_order_relaxed);

      double exec_us = npu_execute_blocking();
      npu_times.push_back(exec_us);

      npu_done.store(1, std::memory_order_release);
    }
  });

  double total_t0 = now_us();
  for (int i = 0; i < num_steps; ++i) {
    StepTiming st = {};
    double step_t0 = now_us();

    // GPU: non-blocking enqueue + flush (measure submit time)
    double submit_t0 = now_us();
    cl_event gpu_evt = gpu_execute_nonblocking();
    double submit_t1 = now_us();

    // Poll for GPU completion (cl_event polling, still goes through driver)
    int poll_count = 0;
    while (!gpu_poll_event(gpu_evt))
      poll_count++;
    double poll_t1 = now_us();

    st.gpu_compute_us = gpu_event_compute_us(gpu_evt);
    // gpu_sync = total GPU overhead = submit + poll - compute
    st.gpu_sync_us = (poll_t1 - submit_t0) - st.gpu_compute_us;
    clReleaseEvent(gpu_evt);

    // Signal NPU to start
    npu_start.store(1, std::memory_order_release);

    // Wait for NPU with optional sleep hint
    double npu_wait_t0 = now_us();
    if (usleep_hint > 0)
      usleep(usleep_hint);
    while (!npu_done.load(std::memory_order_acquire))
      cpu_pause();
    npu_done.store(0, std::memory_order_relaxed);
    double npu_wait_t1 = now_us();

    st.npu_compute_us = npu_times.back();
    st.npu_sync_us = (npu_wait_t1 - npu_wait_t0) - st.npu_compute_us;
    if (st.npu_sync_us < 0) st.npu_sync_us = 0;

    st.step_total_us = now_us() - step_t0;
    result.steps.push_back(st);
  }
  result.total_us = now_us() - total_t0;

  running.store(false, std::memory_order_release);
  npu_start.store(1, std::memory_order_release);
  npu_thread.join();

  result.num_steps = num_steps;
  result.success = true;
  return result;
}

// ── Mode 4: Fast Sync (paper Section 4.3: shared memory flag polling) ────────
static PipelineResult run_fast_sync(int num_steps, int usleep_hint, int npu_core) {
  PipelineResult result;
  result.steps.reserve(num_steps);

  volatile uint32_t* flag_ptr = gpu_get_flag_ptr();
  if (!flag_ptr) {
    result.error = "Flag not enabled";
    return result;
  }

  std::atomic<uint32_t> npu_start{0};
  std::atomic<uint32_t> npu_done{0};
  std::atomic<bool> running{true};
  std::vector<double> npu_times;
  npu_times.reserve(num_steps);

  // NPU worker thread
  std::thread npu_thread([&, npu_core]() {
    pin_to_core(npu_core);
    while (running.load(std::memory_order_acquire)) {
      while (!npu_start.load(std::memory_order_acquire)) {
        if (!running.load(std::memory_order_relaxed)) return;
        cpu_pause();
      }
      npu_start.store(0, std::memory_order_relaxed);

      double exec_us = npu_execute_blocking();
      npu_times.push_back(exec_us);

      npu_done.store(1, std::memory_order_release);
    }
  });

  double total_t0 = now_us();
  for (int i = 0; i < num_steps; ++i) {
    StepTiming st = {};
    double step_t0 = now_us();

    // GPU: submit (clEnqueue + clFlush, flag auto-reset)
    gpu_submit();

    // Poll shared memory flag directly (no OpenCL driver!)
    double poll_t0 = now_us();
    if (usleep_hint > 0)
      usleep(usleep_hint);
    while (*flag_ptr == 0)
      cpu_pause();
    double poll_t1 = now_us();

    // gpu_sync = total time from step start to flag detection - kernel compute (unknown without profiling)
    // For flag mode, we report total GPU wall time as gpu_sync since we can't get profiling without cl_event
    st.gpu_compute_us = 0;  // no profiling available in flag mode
    st.gpu_sync_us = poll_t1 - step_t0;  // total GPU time (submit + wait)

    // Signal NPU to start
    npu_start.store(1, std::memory_order_release);

    // Wait for NPU with optional sleep hint
    double npu_wait_t0 = now_us();
    while (!npu_done.load(std::memory_order_acquire))
      cpu_pause();
    npu_done.store(0, std::memory_order_relaxed);
    double npu_wait_t1 = now_us();

    st.npu_compute_us = npu_times.back();
    st.npu_sync_us = (npu_wait_t1 - npu_wait_t0) - st.npu_compute_us;
    if (st.npu_sync_us < 0) st.npu_sync_us = 0;

    st.step_total_us = now_us() - step_t0;
    result.steps.push_back(st);
  }
  result.total_us = now_us() - total_t0;

  running.store(false, std::memory_order_release);
  npu_start.store(1, std::memory_order_release);
  npu_thread.join();

  result.num_steps = num_steps;
  result.success = true;
  return result;
}

// ── Mode 5: NPU Poll (NPU-side HVX flag polling, GPU+NPU concurrent launch) ─
static PipelineResult run_npu_poll(int num_steps) {
  PipelineResult result;
  result.steps.reserve(num_steps);

  volatile uint32_t* flag_ptr = gpu_get_flag_ptr();
  if (!flag_ptr) {
    result.error = "Flag not enabled";
    return result;
  }

  double total_t0 = now_us();
  for (int i = 0; i < num_steps; ++i) {
    StepTiming st = {};
    double step_t0 = now_us();

    // Reset flag (CPU side) before GPU starts
    *flag_ptr = 0;

    // GPU: non-blocking submit (clEnqueue + clFlush, will write flag=1 on completion)
    gpu_submit();
    double gpu_submit_t = now_us();

    // NPU: blocking graphExecute with custom FlagPoll op
    // The HVX kernel internally polls the flag, then computes RMSNorm.
    // NPU RPC dispatch overlaps with GPU execution!
    double npu_wall = npu_execute_blocking();

    // When NPU returns, BOTH GPU and NPU are done
    st.gpu_compute_us = 0;  // no profiling available
    st.gpu_sync_us = gpu_submit_t - step_t0;  // just submit overhead
    st.npu_compute_us = npu_wall;  // includes HVX polling + RMSNorm
    st.npu_sync_us = 0;  // no CPU-side NPU wait

    st.step_total_us = now_us() - step_t0;
    result.steps.push_back(st);
  }
  result.total_us = now_us() - total_t0;
  result.num_steps = num_steps;
  result.success = true;
  return result;
}

// ── Public API ───────────────────────────────────────────────────────────────
PipelineResult run_pipeline(const PipelineConfig& config, const char* kernel_path) {
  PipelineResult result;
  int hidden = config.hidden_dim;
  size_t tensor_bytes = (size_t)hidden * 2;  // FP16, batch=1

  // Allocate shared ION buffers (ping-pong)
  IonBuffer ion_buf0, ion_buf1;
  if (!allocIonBuffer(tensor_bytes, 0, ion_buf0) ||
      !allocIonBuffer(tensor_bytes, 0, ion_buf1)) {
    result.error = "ION alloc failed";
    return result;
  }

  // Fill buf0 with random FP16 data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);
  uint16_t* ptr = reinterpret_cast<uint16_t*>(ion_buf0.ptr);
  for (int i = 0; i < hidden; ++i)
    ptr[i] = float_to_half(dist(rng));

  // Init GPU: reads buf0, writes buf1
  if (!gpu_init(hidden, config.epsilon, ion_buf0, ion_buf1, kernel_path)) {
    result.error = "GPU init failed";
    freeIonBuffer(ion_buf0); freeIonBuffer(ion_buf1);
    return result;
  }

  // Allocate flag buffer for modes that need it (FAST_SYNC and NPU_POLL)
  IonBuffer ion_flag = {};
  IonBuffer ion_poll_count = {};
  bool need_flag = (config.mode == SyncMode::FAST_SYNC || config.mode == SyncMode::NPU_POLL);

  if (need_flag) {
    if (!allocIonBuffer(sizeof(uint32_t), 0, ion_flag)) {
      result.error = "ION flag alloc failed";
      gpu_cleanup(); freeIonBuffer(ion_buf0); freeIonBuffer(ion_buf1);
      return result;
    }
  }
  if (config.mode == SyncMode::NPU_POLL) {
    if (!allocIonBuffer(sizeof(uint32_t), 0, ion_poll_count)) {
      result.error = "ION poll_count alloc failed";
      gpu_cleanup(); freeIonBuffer(ion_buf0); freeIonBuffer(ion_buf1); freeIonBuffer(ion_flag);
      return result;
    }
  }

  // Init NPU: reads buf1, writes buf0
  bool npu_ok = false;
  if (config.mode == SyncMode::NPU_POLL) {
    // NPU_POLL uses custom FlagPoll RMSNorm op that polls flag internally
    npu_ok = npu_init_flagpoll(hidden, config.epsilon, ion_buf1, ion_buf0, ion_flag, ion_poll_count);
  } else {
    npu_ok = npu_init(hidden, config.epsilon, ion_buf1, ion_buf0);
  }
  if (!npu_ok) {
    result.error = "NPU init failed";
    gpu_cleanup(); freeIonBuffer(ion_buf0); freeIonBuffer(ion_buf1);
    freeIonBuffer(ion_flag); freeIonBuffer(ion_poll_count);
    return result;
  }

  if (need_flag) {
    if (!gpu_enable_flag(ion_flag)) {
      result.error = "GPU flag enable failed";
      npu_cleanup(); gpu_cleanup();
      freeIonBuffer(ion_buf0); freeIonBuffer(ion_buf1);
      freeIonBuffer(ion_flag); freeIonBuffer(ion_poll_count);
      return result;
    }
  }

  // Pin main thread if requested
  if (config.main_core >= 0) {
    if (pin_to_core(config.main_core))
      printf("  Main thread pinned to core %d\n", config.main_core);
    else
      printf("  WARNING: Failed to pin main thread to core %d\n", config.main_core);
  }

  // Warmup
  if (config.mode == SyncMode::NPU_POLL) {
    // For NPU_POLL: warmup with flag pre-set to 1 so HVX doesn't block
    volatile uint32_t* fp = gpu_get_flag_ptr();
    for (int i = 0; i < config.num_warmup; ++i) {
      if (fp) *fp = 1;  // pre-set so HVX kernel proceeds immediately
      gpu_execute_blocking(nullptr);
      npu_execute_blocking();
    }
  } else {
    // For other modes: sequential blocking warmup without flag
    gpu_disable_flag();
    for (int i = 0; i < config.num_warmup; ++i) {
      gpu_execute_blocking(nullptr);
      npu_execute_blocking();
    }
    // Re-enable flag for FAST_SYNC
    if (config.mode == SyncMode::FAST_SYNC)
      gpu_enable_flag(ion_flag);
  }

  // Run pipeline
  switch (config.mode) {
    case SyncMode::SEQUENTIAL_BLOCKING:
      result = run_sequential(config.num_steps);
      break;
    case SyncMode::THREADED_CLFINISH:
      result = run_threaded_clfinish(config.num_steps, config.npu_core);
      break;
    case SyncMode::EVENT_POLL:
      result = run_event_poll(config.num_steps, config.usleep_hint, config.npu_core);
      break;
    case SyncMode::FAST_SYNC:
      result = run_fast_sync(config.num_steps, config.usleep_hint, config.npu_core);
      break;
    case SyncMode::NPU_POLL:
      result = run_npu_poll(config.num_steps);
      break;
  }

  result.avg_step_us = result.total_us / result.num_steps;

  // Cleanup
  gpu_disable_flag();
  npu_cleanup();
  gpu_cleanup();
  freeIonBuffer(ion_buf0);
  freeIonBuffer(ion_buf1);
  freeIonBuffer(ion_flag);
  freeIonBuffer(ion_poll_count);

  return result;
}

// ── GPU-only diagnostic ────────────────────────────────────────────────────
void run_gpu_diagnostic(int hidden_dim, int num_steps, const char* kernel_path) {
  size_t tensor_bytes = (size_t)hidden_dim * 2;
  IonBuffer ion_in, ion_out;
  if (!allocIonBuffer(tensor_bytes, 0, ion_in) ||
      !allocIonBuffer(tensor_bytes, 0, ion_out)) {
    printf("[Diag] ION alloc failed\n");
    return;
  }

  // Fill with data
  uint16_t* ptr = reinterpret_cast<uint16_t*>(ion_in.ptr);
  for (int i = 0; i < hidden_dim; ++i) ptr[i] = float_to_half(1.0f);

  if (!gpu_init(hidden_dim, 1e-6f, ion_in, ion_out, kernel_path)) {
    printf("[Diag] GPU init failed\n");
    freeIonBuffer(ion_in); freeIonBuffer(ion_out);
    return;
  }

  // Warmup
  for (int i = 0; i < 10; ++i) gpu_execute_blocking(nullptr);

  // Test 0: clFinish without profiling event (no event allocation/tracking)
  std::vector<double> noprof_wall;
  for (int i = 0; i < num_steps; ++i) {
    noprof_wall.push_back(gpu_execute_blocking_noprof());
  }

  // Test 1: clFinish timing (with profiling event)
  std::vector<double> blocking_wall, blocking_compute;
  for (int i = 0; i < num_steps; ++i) {
    double compute = 0;
    double wall = gpu_execute_blocking(&compute);
    blocking_wall.push_back(wall);
    blocking_compute.push_back(compute);
  }

  // Test 1b: Profiling breakdown (QUEUED → SUBMIT → START → END)
  std::vector<double> prof_queue_delay, prof_submit_delay, prof_compute, prof_total_device;
  for (int i = 0; i < num_steps; ++i) {
    cl_event evt = gpu_execute_nonblocking();
    clWaitForEvents(1, &evt);

    GpuProfilingInfo prof = gpu_event_profiling(evt);
    clReleaseEvent(evt);

    prof_queue_delay.push_back(prof.queue_delay);
    prof_submit_delay.push_back(prof.submit_delay);
    prof_compute.push_back(prof.compute);
    prof_total_device.push_back(prof.total_device);
  }

  // Test 2: clFlush + event poll timing (broken down)
  std::vector<double> submit_times, poll_times, total_nonblock, poll_compute;
  std::vector<int> poll_counts;
  for (int i = 0; i < num_steps; ++i) {
    double t0 = now_us();
    cl_event evt = gpu_execute_nonblocking();
    double t1 = now_us();

    int count = 0;
    while (!gpu_poll_event(evt)) count++;
    double t2 = now_us();

    double compute = gpu_event_compute_us(evt);
    clReleaseEvent(evt);

    submit_times.push_back(t1 - t0);
    poll_times.push_back(t2 - t1);
    total_nonblock.push_back(t2 - t0);
    poll_compute.push_back(compute);
    poll_counts.push_back(count);
  }

  // Test 3: clFlush + clWaitForEvents (OS-level wait vs spin poll)
  std::vector<double> wait_evt_times, wait_evt_compute;
  for (int i = 0; i < num_steps; ++i) {
    double t0 = now_us();
    cl_event evt = gpu_execute_nonblocking();
    clWaitForEvents(1, &evt);
    double t1 = now_us();
    double compute = gpu_event_compute_us(evt);
    clReleaseEvent(evt);
    wait_evt_times.push_back(t1 - t0);
    wait_evt_compute.push_back(compute);
  }

  // Test 4: clFlush + shared memory flag poll (paper's approach — ground truth)
  IonBuffer ion_flag;
  std::vector<double> flag_times;
  std::vector<int> flag_poll_counts;
  bool flag_ok = allocIonBuffer(sizeof(uint32_t), 0, ion_flag) && gpu_enable_flag(ion_flag);
  if (flag_ok) {
    // Warmup with flag kernel
    for (int i = 0; i < 10; ++i) {
      gpu_submit();
      volatile uint32_t* fp = gpu_get_flag_ptr();
      while (*fp == 0) ;
    }
    // Measure
    for (int i = 0; i < num_steps; ++i) {
      volatile uint32_t* fp = gpu_get_flag_ptr();
      double t0 = now_us();
      gpu_submit();  // resets flag, enqueue, flush
      int count = 0;
      while (*fp == 0) count++;
      double t1 = now_us();
      flag_times.push_back(t1 - t0);
      flag_poll_counts.push_back(count);
    }
    gpu_disable_flag();
  }

  // Print results
  auto p50 = [](std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
  };
  auto p50i = [](std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
  };

  printf("--- GPU Sync Diagnostic (hidden=%d, %d iters) ---\n", hidden_dim, num_steps);
  printf("  Method                  total_p50    extra\n");
  printf("  clFinish (blocking)    %8.1f us\n", p50(noprof_wall));
  printf("  clFlush+cl_event poll  %8.1f us   submit=%.1f poll=%.1f polls=%d\n",
         p50(total_nonblock),
         p50(submit_times), p50(poll_times), p50i(poll_counts));
  printf("  clFlush+WaitForEvents  %8.1f us\n", p50(wait_evt_times));
  if (flag_ok) {
    printf("  clFlush+flag poll ★    %8.1f us   polls=%d  ← ground truth\n",
           p50(flag_times), p50i(flag_poll_counts));
  }

  printf("\n  GPU kernel compute (profiling): %.1f us\n", p50(blocking_compute));

  // Profiling breakdown: how GPU command pipeline works
  printf("\n  Profiling breakdown (device clock, p50):\n");
  printf("    QUEUED→SUBMIT (driver→GPU):  %8.1f us\n", p50(prof_queue_delay));
  printf("    SUBMIT→START  (GPU sched):   %8.1f us\n", p50(prof_submit_delay));
  printf("    START→END     (kernel exec):  %8.1f us\n", p50(prof_compute));
  printf("    QUEUED→END    (total device): %8.1f us\n", p50(prof_total_device));

  if (flag_ok) {
    double clfinish_overhead = p50(noprof_wall) - p50(flag_times);
    double clevent_overhead = p50(total_nonblock) - p50(flag_times);
    printf("\n  Overhead analysis (vs flag ground truth):\n");
    printf("    clFinish overhead:  %8.1f us (clFinish - flag)\n", clfinish_overhead);
    printf("    cl_event overhead:  %8.1f us (cl_event - flag)\n", clevent_overhead);
    printf("    Flag wall time:     %8.1f us (submit + kernel + flag detect)\n", p50(flag_times));
    printf("    Kernel compute:     %8.1f us (profiling START→END)\n", p50(blocking_compute));
    printf("    Submit+dispatch:    %8.1f us (flag_wall - kernel_compute)\n",
           p50(flag_times) - p50(blocking_compute));
  }

  gpu_cleanup();
  freeIonBuffer(ion_in);
  freeIonBuffer(ion_out);
  if (flag_ok) freeIonBuffer(ion_flag);
}
