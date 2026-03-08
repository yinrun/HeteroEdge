#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <vector>
#include <functional>
#include <atomic>
#include <semaphore.h>
#include <time.h>

#include "common.h"
#include "qnn_setup.h"
#include "graph_builder.h"

static bool pin_to_core(int core_id) {
  if (core_id < 0) return false;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(core_id, &mask);
  return sched_setaffinity(0, sizeof(mask), &mask) == 0;
}

static void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("  --hidden-dim N   hidden dimension (default: 512)\n");
  printf("  --inter-dim N    intermediate dimension (default: 1376)\n");
  printf("  --warmup N       warmup iterations (default: 10)\n");
  printf("  --iters N        measurement iterations (default: 100)\n");
  printf("  --core N         pin to CPU core N\n");
  printf("  --legacy         run legacy ElementWiseMultiply sweep\n");
  printf("  --async          test graphExecuteAsync on HTP\n");
}

// Reinit QNN backend+context. Returns false on failure.
static uint32_t g_async_queue_depth = 0;
static bool qnn_reinit() {
  qnn_cleanup();
  return qnn_init(g_async_queue_depth);
}

// Benchmark a FusedResult: warmup + measure, return stats. Empty stats on failure.
static Stats benchFused(FusedResult& r, int warmup, int iters) {
  auto exec = [&]() {
    return g_qnn->graphExecute(r.graph, &r.exec_in, 1, &r.exec_out, 1, nullptr, nullptr);
  };
  if (exec() != 0) return {};
  for (int w = 0; w < warmup; ++w) exec();
  std::vector<double> t; t.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    exec();
    t.push_back(now_us() - t0);
  }
  return compute_stats(t);
}

// ── Legacy benchmark (ElementWiseMultiply chain) ──

static void run_legacy(int hidden_dim, int warmup, int iters) {
  printf("=== Legacy: ElementWiseMultiply Chain ===\n");
  printf("Shape: [1,1,1,%d] FP16 | Warmup: %d | Iters: %d\n\n", hidden_dim, warmup, iters);

  printf("%-6s | %-6s | %10s | %12s | %10s | %10s | %s\n",
         "N_ops", "Mode", "Create(us)", "Exec avg(us)", "Exec p50", "Exec p99", "Ratio");
  printf("-------+--------+------------+--------------+------------+------------+-------\n");

  auto printRow = [](int n, const char* mode, double create, Stats s, double base_avg) {
    if (base_avg > 0)
      printf("%5d | %-6s | %10.1f | %12.1f | %10.1f | %10.1f | %.2fx\n",
             n, mode, create, s.avg, s.p50, s.p99, s.avg / base_avg);
    else
      printf("%5d | %-6s | %10.1f | %12.1f | %10.1f | %10.1f |\n",
             n, mode, create, s.avg, s.p50, s.p99);
  };

  int sweep[] = {1, 2, 4, 8, 16, 32};
  for (int n : sweep) {
    // ── Fused ──
    FusedResult fused;
    if (!build_fused_graph(n, hidden_dim, fused)) {
      cleanup_fused(fused); continue;
    }
    Stats fs = benchFused(fused, warmup, iters);
    double fc = fused.create_us;
    uint32_t in_id = fused.input_tensor_id, out_id = fused.output_tensor_id;
    char gname[64];
    snprintf(gname, sizeof(gname), "%s", fused.graph_name);

    // ── Cached ──
    std::vector<uint8_t> binary;
    bool cache_ok = qnn_context_get_binary(binary);
    cleanup_fused(fused);

    Stats cs = {};
    double cc = 0;
    if (cache_ok) {
      qnn_cleanup();
      if (qnn_init()) {
        double load_us = 0;
        if (qnn_context_create_from_binary(binary, load_us)) {
          FusedResult cached;
          if (retrieve_fused_from_cache(n, hidden_dim, gname, in_id, out_id, cached)) {
            cc = load_us;
            cs = benchFused(cached, warmup, iters);
            cleanup_fused(cached);
          }
        }
      }
      if (!qnn_reinit()) { printf("QNN re-init failed\n"); return; }
    }

    // ── Split ──
    SplitResult split;
    if (!build_split_graphs(n, hidden_dim, split)) {
      cleanup_split(split);
      printRow(n, "Fused", fc, fs, 0);
      if (cs.avg > 0) printRow(n, "Cached", cc, cs, fs.avg);
      printf("-------+--------+------------+--------------+------------+------------+-------\n");
      continue;
    }
    for (int w = 0; w < warmup; ++w)
      for (int g = 0; g < n; ++g)
        g_qnn->graphExecute(split.graphs[g], &split.exec_ins[g], 1, &split.exec_outs[g], 1, nullptr, nullptr);
    std::vector<double> st; st.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      double t0 = now_us();
      for (int g = 0; g < n; ++g)
        g_qnn->graphExecute(split.graphs[g], &split.exec_ins[g], 1, &split.exec_outs[g], 1, nullptr, nullptr);
      st.push_back(now_us() - t0);
    }
    Stats ss = compute_stats(st);
    cleanup_split(split);

    printRow(n, "Fused", fc, fs, 0);
    if (cs.avg > 0) printRow(n, "Cached", cc, cs, fs.avg);
    printRow(n, "Split", split.create_us, ss, fs.avg);
    printf("-------+--------+------------+--------------+------------+------------+-------\n");

    if (!qnn_reinit()) { printf("QNN re-init failed\n"); return; }
  }
}

// ── FFN block benchmark ──

static void run_ffn(const FFNConfig& cfg, int warmup, int iters) {
  printf("=== FFN Block: RMSNorm → FC → GELU → FC → Add ===\n");
  printf("hidden=%d, inter=%d | 5 ops | FP16 | Warmup: %d | Iters: %d\n",
         cfg.hidden_dim, cfg.intermediate_dim, warmup, iters);
  printf("NPU cores: %u\n\n", g_coreCount);

  printf("%-8s | %6s | %10s | %12s | %10s | %10s | %s\n",
         "Mode", "Graphs", "Create(ms)", "Exec avg(us)", "Exec p50", "Exec p99", "vs Fused");
  printf("---------+--------+------------+--------------+------------+------------+---------\n");

  double fused_avg = 0;

  // Build → benchmark → cleanup → reinit, all in one helper
  auto runMode = [&](const char* label, std::function<bool(FFNResult&)> builder) {
    FFNResult res;
    if (!builder(res)) {
      printf("[ERROR] %s build failed\n", label);
      cleanup_ffn(res);
      if (!qnn_reinit()) { printf("QNN re-init failed\n"); return; }
      return;
    }
    int ng = (int)res.graphs.size();

    // Verify
    for (int g = 0; g < ng; ++g) {
      auto err = g_qnn->graphExecute(res.graphs[g],
          res.graph_execs[g].ins.data(), (uint32_t)res.graph_execs[g].ins.size(),
          res.graph_execs[g].outs.data(), (uint32_t)res.graph_execs[g].outs.size(),
          nullptr, nullptr);
      if (err != 0) {
        printf("[ERROR] %s graph %d/%d exec failed: %lu\n", label, g, ng, (unsigned long)err);
        cleanup_ffn(res);
        if (!qnn_reinit()) printf("QNN re-init failed\n");
        return;
      }
    }

    // Warmup + measure
    for (int w = 0; w < warmup; ++w)
      for (int g = 0; g < ng; ++g)
        g_qnn->graphExecute(res.graphs[g],
            res.graph_execs[g].ins.data(), (uint32_t)res.graph_execs[g].ins.size(),
            res.graph_execs[g].outs.data(), (uint32_t)res.graph_execs[g].outs.size(),
            nullptr, nullptr);

    std::vector<double> times; times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      double t0 = now_us();
      for (int g = 0; g < ng; ++g)
        g_qnn->graphExecute(res.graphs[g],
            res.graph_execs[g].ins.data(), (uint32_t)res.graph_execs[g].ins.size(),
            res.graph_execs[g].outs.data(), (uint32_t)res.graph_execs[g].outs.size(),
            nullptr, nullptr);
      times.push_back(now_us() - t0);
    }
    Stats s = compute_stats(times);
    if (fused_avg == 0) fused_avg = s.avg;

    double ratio = (fused_avg > 0) ? s.avg / fused_avg : 0;
    printf("%-8s | %6d | %10.1f | %12.1f | %10.1f | %10.1f | %.2fx\n",
           label, ng, res.create_us / 1000.0, s.avg, s.p50, s.p99, ratio);

    cleanup_ffn(res);
    if (!qnn_reinit()) printf("QNN re-init failed\n");
  };

  runMode("Fused",   [&](FFNResult& r) { return build_ffn_block(SplitMode::FUSED, cfg, r); });
  runMode("Cached",  [&](FFNResult& r) { double us; return build_ffn_cached(cfg, r, us); });
  runMode("2-Block", [&](FFNResult& r) { return build_ffn_block(SplitMode::BLOCK_2, cfg, r); });
  runMode("Per-Op",  [&](FFNResult& r) { return build_ffn_block(SplitMode::PER_OP, cfg, r); });

  printf("---------+--------+------------+--------------+------------+------------+---------\n");
  printf("\n--- Analysis ---\n");
  printf("Per-graph launch overhead = (Per-Op_avg - Fused_avg) / 4\n");
  printf("2-Block overhead = 2-Block_avg - Fused_avg (one extra graphExecute)\n");
  printf("Cached vs Fused: creation time difference shows cache benefit\n");
  printf("Cached vs Fused: exec time should be ~equal (both use graphExecute)\n");
}

// ── Async execution test ──

struct AsyncNotifyData {
  sem_t sem;
  Qnn_ErrorHandle_t error;
  double callback_time_us;
};

static void asyncCallback(void* param, Qnn_NotifyStatus_t status) {
  auto* data = static_cast<AsyncNotifyData*>(param);
  data->callback_time_us = now_us();
  data->error = status.error;
  sem_post(&data->sem);
}

static void run_async_test(const FFNConfig& cfg, int warmup, int iters) {
  printf("=== Async Execution Test: graphExecuteAsync on HTP ===\n");
  printf("hidden=%d, inter=%d | 5 ops | FP16 | Warmup: %d | Iters: %d\n\n",
         cfg.hidden_dim, cfg.intermediate_dim, warmup, iters);

  // Build fused FFN graph
  FFNResult res;
  if (!build_ffn_block(SplitMode::FUSED, cfg, res)) {
    printf("[ERROR] Fused build failed\n");
    cleanup_ffn(res);
    return;
  }

  auto& ge = res.graph_execs[0];

  // First verify sync works
  auto sync_err = g_qnn->graphExecute(res.graphs[0],
      ge.ins.data(), (uint32_t)ge.ins.size(),
      ge.outs.data(), (uint32_t)ge.outs.size(),
      nullptr, nullptr);
  if (sync_err != 0) {
    printf("[ERROR] Sync graphExecute failed: %lu\n", (unsigned long)sync_err);
    cleanup_ffn(res);
    return;
  }
  printf("[OK] Sync graphExecute works\n");

  // Check if graphExecuteAsync function pointer exists
  if (!g_qnn->graphExecuteAsync) {
    printf("[ERROR] graphExecuteAsync function pointer is NULL\n");
    cleanup_ffn(res);
    return;
  }
  printf("[OK] graphExecuteAsync function pointer exists\n");

  // Try async execution
  AsyncNotifyData notify;
  sem_init(&notify.sem, 0, 0);
  notify.error = 0;
  notify.callback_time_us = 0;

  double t0 = now_us();
  auto async_err = g_qnn->graphExecuteAsync(res.graphs[0],
      ge.ins.data(), (uint32_t)ge.ins.size(),
      ge.outs.data(), (uint32_t)ge.outs.size(),
      nullptr, nullptr,
      asyncCallback, &notify);
  double t_enqueue = now_us() - t0;

  if (async_err != 0) {
    printf("[ERROR] graphExecuteAsync returned error: %lu\n", (unsigned long)async_err);
    printf("  (Error 1000 = QNN_GRAPH_ERROR_GENERAL, 6000 = QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE)\n");
    sem_destroy(&notify.sem);
    cleanup_ffn(res);
    return;
  }

  printf("[OK] graphExecuteAsync enqueue succeeded (%.1f us)\n", t_enqueue);

  // Wait for callback
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += 5;  // 5s timeout
  if (sem_timedwait(&notify.sem, &ts) != 0) {
    printf("[ERROR] Async callback timed out (5s)\n");
    sem_destroy(&notify.sem);
    cleanup_ffn(res);
    return;
  }

  double t_total = notify.callback_time_us - t0 + t_enqueue;  // enqueue was already subtracted
  printf("[OK] Async callback received, exec error: %lu, total time: %.1f us\n",
         (unsigned long)notify.error, notify.callback_time_us - (t0 - t_enqueue));

  if (notify.error != 0) {
    printf("[ERROR] Async execution failed with error: %lu\n", (unsigned long)notify.error);
    sem_destroy(&notify.sem);
    cleanup_ffn(res);
    return;
  }

  // Benchmark: sync vs async
  printf("\n--- Benchmark: Sync vs Async ---\n");
  printf("%-6s | %12s | %10s | %10s | %10s\n",
         "Mode", "Exec avg(us)", "Exec p50", "Exec p99", "Enqueue(us)");
  printf("-------+--------------+------------+------------+------------\n");

  // Sync benchmark
  for (int w = 0; w < warmup; ++w)
    g_qnn->graphExecute(res.graphs[0],
        ge.ins.data(), (uint32_t)ge.ins.size(),
        ge.outs.data(), (uint32_t)ge.outs.size(),
        nullptr, nullptr);
  std::vector<double> sync_times; sync_times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double s0 = now_us();
    g_qnn->graphExecute(res.graphs[0],
        ge.ins.data(), (uint32_t)ge.ins.size(),
        ge.outs.data(), (uint32_t)ge.outs.size(),
        nullptr, nullptr);
    sync_times.push_back(now_us() - s0);
  }
  Stats ss = compute_stats(sync_times);
  printf("%-6s | %12.1f | %10.1f | %10.1f | %10s\n",
         "Sync", ss.avg, ss.p50, ss.p99, "N/A");

  // Async benchmark (fire + wait for callback each iteration)
  for (int w = 0; w < warmup; ++w) {
    notify.error = 0;
    g_qnn->graphExecuteAsync(res.graphs[0],
        ge.ins.data(), (uint32_t)ge.ins.size(),
        ge.outs.data(), (uint32_t)ge.outs.size(),
        nullptr, nullptr, asyncCallback, &notify);
    sem_wait(&notify.sem);
  }
  std::vector<double> async_times, enqueue_times;
  async_times.reserve(iters); enqueue_times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    notify.error = 0;
    double a0 = now_us();
    g_qnn->graphExecuteAsync(res.graphs[0],
        ge.ins.data(), (uint32_t)ge.ins.size(),
        ge.outs.data(), (uint32_t)ge.outs.size(),
        nullptr, nullptr, asyncCallback, &notify);
    double a_enq = now_us() - a0;
    sem_wait(&notify.sem);
    double a_total = notify.callback_time_us - a0;
    enqueue_times.push_back(a_enq);
    async_times.push_back(a_total);
  }
  Stats as = compute_stats(async_times);
  Stats es = compute_stats(enqueue_times);
  printf("%-6s | %12.1f | %10.1f | %10.1f | %10.1f\n",
         "Async", as.avg, as.p50, as.p99, es.avg);

  printf("-------+--------------+------------+------------+------------\n");
  printf("\nAsync/Sync ratio: %.2fx\n", as.avg / ss.avg);
  printf("Avg enqueue time (async return before completion): %.1f us\n", es.avg);
  if (es.avg < ss.avg * 0.5)
    printf("** Async IS truly non-blocking — enqueue returns before DSP completion **\n");
  else
    printf("** Async appears to block similarly to sync **\n");

  sem_destroy(&notify.sem);
  cleanup_ffn(res);
}

int main(int argc, char** argv) {
  FFNConfig cfg;
  int warmup = 10;
  int iters  = 100;
  int core   = -1;
  bool legacy = false;
  bool async_test = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--hidden-dim") && i + 1 < argc) cfg.hidden_dim = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--inter-dim") && i + 1 < argc) cfg.intermediate_dim = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--core") && i + 1 < argc) core = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--legacy")) legacy = true;
    else if (!strcmp(argv[i], "--async")) async_test = true;
    else if (!strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
  }

  if (async_test) g_async_queue_depth = 4;

  if (core >= 0) {
    if (pin_to_core(core))
      printf("CPU affinity: pinned to core %d\n", core);
    else
      printf("WARNING: failed to pin to core %d\n", core);
  }

  if (!qnn_init(g_async_queue_depth)) { printf("QNN init failed\n"); return 1; }
  printf("NPU cores: %u\n\n", g_coreCount);

  if (async_test)
    run_async_test(cfg, warmup, iters);
  else if (legacy)
    run_legacy(cfg.hidden_dim, warmup, iters);
  else
    run_ffn(cfg, warmup, iters);

  qnn_cleanup();
  return 0;
}
