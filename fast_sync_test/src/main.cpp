#include "common.h"
#include "gpu_engine.h"
#include "npu_engine.h"
#include "pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --hidden-dim N   hidden dimension (default: 4096)\n");
  printf("  --steps N        measured iterations (default: 100)\n");
  printf("  --warmup N       warmup iterations (default: 10)\n");
  printf("  --usleep-hint N  NPU wait hint in us (default: 0 = pure spin)\n");
  printf("  --mode MODE      seq|threaded|event|fast|all (default: all)\n");
}

static void print_stats_row(const char* label, Stats& s) {
  printf("  %-14s %8.1f %8.1f %8.1f %8.1f %8.1f\n",
         label, s.min, s.p50, s.avg, s.p99, s.max);
}

static void print_mode_result(const char* mode_name, PipelineResult& r) {
  if (!r.success) {
    printf("\n=== %s: FAILED (%s) ===\n", mode_name, r.error.c_str());
    return;
  }

  std::vector<double> gpu_compute, gpu_sync, npu_compute, npu_sync, step_total;
  for (auto& st : r.steps) {
    gpu_compute.push_back(st.gpu_compute_us);
    gpu_sync.push_back(st.gpu_sync_us);
    npu_compute.push_back(st.npu_compute_us);
    npu_sync.push_back(st.npu_sync_us);
    step_total.push_back(st.step_total_us);
  }

  Stats s_gc = compute_stats(gpu_compute);
  Stats s_gs = compute_stats(gpu_sync);
  Stats s_nc = compute_stats(npu_compute);
  Stats s_ns = compute_stats(npu_sync);
  Stats s_st = compute_stats(step_total);

  printf("\n=== %s ===\n", mode_name);
  printf("  %-14s %8s %8s %8s %8s %8s\n", "(us)", "min", "p50", "avg", "p99", "max");
  print_stats_row("gpu_compute", s_gc);
  print_stats_row("gpu_sync", s_gs);
  print_stats_row("npu_compute", s_nc);
  print_stats_row("npu_sync", s_ns);
  print_stats_row("step_total", s_st);
}

struct ModeResult {
  SyncMode mode;
  const char* name;
  PipelineResult result;
  Stats step_stats;
  Stats gpu_sync_stats;
  Stats npu_sync_stats;
};

int main(int argc, char* argv[]) {
  int hidden_dim  = 4096;
  int steps       = 100;
  int warmup      = 10;
  int usleep_hint = 0;
  bool run_seq = true, run_threaded = true, run_event = true, run_fast = true;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--hidden-dim") && i+1 < argc) hidden_dim = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i+1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--usleep-hint") && i+1 < argc) usleep_hint = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mode") && i+1 < argc) {
      ++i;
      run_seq = run_threaded = run_event = run_fast = false;
      if (!strcmp(argv[i], "seq")) run_seq = true;
      else if (!strcmp(argv[i], "threaded")) run_threaded = true;
      else if (!strcmp(argv[i], "event")) run_event = true;
      else if (!strcmp(argv[i], "fast")) run_fast = true;
      else { run_seq = run_threaded = run_event = run_fast = true; }
    }
    else if (!strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
  }

  printf("=== Fast Sync Benchmark: GPU<->NPU Pipeline ===\n");
  printf("Platform: SM8850, Adreno 840 + Hexagon V81\n");
  printf("Config: hidden=%d, batch=1, FP16, steps=%d, warmup=%d\n", hidden_dim, steps, warmup);
  if (usleep_hint > 0)
    printf("NPU usleep hint: %d us\n", usleep_hint);
  printf("\n");

  // Print device info
  {
    size_t tmp_bytes = (size_t)hidden_dim * 2;
    IonBuffer tmp0, tmp1;
    allocIonBuffer(tmp_bytes, 0, tmp0);
    allocIonBuffer(tmp_bytes, 0, tmp1);

    printf("--- Device Info ---\n");
    if (gpu_init(hidden_dim, 1e-6f, tmp0, tmp1, "kernels/rmsnorm.cl")) {
      gpu_print_info();
      gpu_cleanup();
    }
    if (npu_init(hidden_dim, 1e-6f, tmp1, tmp0)) {
      npu_print_info();
      npu_cleanup();
    }
    freeIonBuffer(tmp0);
    freeIonBuffer(tmp1);
    printf("\n");
  }

  const char* kernel_path = "kernels/rmsnorm.cl";

  // GPU-only diagnostic: isolate clFinish overhead
  run_gpu_diagnostic(hidden_dim, steps, kernel_path);
  printf("\n");

  std::vector<ModeResult> results;

  // Run each mode
  SyncMode modes[] = {SyncMode::SEQUENTIAL_BLOCKING, SyncMode::THREADED_CLFINISH, SyncMode::EVENT_POLL, SyncMode::FAST_SYNC};
  bool     run_flags[] = {run_seq, run_threaded, run_event, run_fast};
  const char* names[] = {"Seq Blocking", "Thread+clFinish", "Event Poll", "Fast Sync"};

  for (int m = 0; m < 4; ++m) {
    if (!run_flags[m]) continue;

    PipelineConfig cfg;
    cfg.hidden_dim  = hidden_dim;
    cfg.epsilon     = 1e-6f;
    cfg.num_warmup  = warmup;
    cfg.num_steps   = steps;
    cfg.usleep_hint = usleep_hint;
    cfg.mode        = modes[m];

    printf("Running %s...\n", names[m]);
    PipelineResult r = run_pipeline(cfg, kernel_path);
    print_mode_result(names[m], r);

    if (r.success) {
      ModeResult mr;
      mr.mode = modes[m];
      mr.name = names[m];
      mr.result = r;

      std::vector<double> step_vals, gpu_sync_vals, npu_sync_vals;
      for (auto& st : r.steps) {
        step_vals.push_back(st.step_total_us);
        gpu_sync_vals.push_back(st.gpu_sync_us);
        npu_sync_vals.push_back(st.npu_sync_us);
      }
      mr.step_stats = compute_stats(step_vals);
      mr.gpu_sync_stats = compute_stats(gpu_sync_vals);
      mr.npu_sync_stats = compute_stats(npu_sync_vals);
      results.push_back(mr);
    }
  }

  // Summary table
  if (results.size() > 1) {
    printf("\n=== Summary ===\n");
    printf("%-20s %10s %10s %10s %10s %10s\n",
           "Mode", "step_p50", "gpu_sync", "npu_sync", "sync_tot", "speedup");
    for (int i = 0; i < 72; ++i) printf("-");
    printf("\n");

    double baseline_step = results[0].step_stats.p50;
    for (auto& mr : results) {
      double sync_tot = mr.gpu_sync_stats.p50 + mr.npu_sync_stats.p50;
      double speedup = baseline_step / mr.step_stats.p50;
      printf("%-20s %8.1f us %8.1f us %8.1f us %8.1f us %8.2fx\n",
             mr.name,
             mr.step_stats.p50,
             mr.gpu_sync_stats.p50,
             mr.npu_sync_stats.p50,
             sync_tot,
             speedup);
    }

    printf("\nPaper prediction: 2-4x speedup from fast sync (Section 5.5, Figure 17)\n");
  }

  return 0;
}
