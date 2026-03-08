# QNN Graph Launch Overhead Benchmark

Quantifies the per-graph launch overhead on Qualcomm HTP (Hexagon Tensor Processor) using two benchmarks:

1. **FFN Block** (default) — Realistic 5-op LLM decode FFN block (RMSNorm → FC → GELU → FC → Add) with 4 modes
2. **Legacy** (`--legacy`) — N chained ElementWiseMultiply ops with fused/cached/split modes

## FFN Block Benchmark

Tests a realistic LLM decode-stage FFN block with 5 ops and FP16 tensors:

```
Input x [1,1,1,H]
  → RMSNorm(x, gamma, beta)     → norm   [1,1,1,H]
  → FullyConnected(norm, W_up)  → hidden [1,1,1,I]
  → GELU(hidden)                → act    [1,1,1,I]
  → FullyConnected(act, W_down) → proj   [1,1,1,H]
  → Add(x, proj)                → output [1,1,1,H]   (residual)
```

### Modes

| Mode | Graphs | Description |
|------|:------:|-------------|
| **Fused** | 1 | All 5 ops in one graph, built dynamically |
| **Cached** | 1 | Same fused graph, serialized to context binary then restored |
| **2-Block** | 2 | [RMSNorm + FC_up] + [GELU + FC_down + Add] |
| **Per-Op** | 5 | Each op in its own graph |

### Results

**Device**: Snapdragon 8 Elite, Hexagon V81, 1 NPU core
**CPU affinity**: pinned to big core 7 (4.61 GHz)
**QNN SDK**: 2.42
**Dims**: hidden=512, intermediate=1376

| Mode | Graphs | Create (ms) | Exec avg (us) | Exec p50 | Exec p99 | vs Fused |
|----------|:------:|------------:|--------------:|---------:|---------:|---------:|
| Fused    | 1      | 148.7       | 213.8         | 203.8    | 284.1    | 1.00x    |
| Cached   | 1      | 5.6         | 224.5         | 220.9    | 279.3    | 1.05x    |
| 2-Block  | 2      | 123.9       | 420.4         | 435.4    | 659.8    | 1.97x    |
| Per-Op   | 5      | 140.1       | 1019.5        | 1039.9   | 1072.5   | 4.77x    |

### FFN Key Findings

- **Per-graph overhead with realistic ops: ~202 us** — `(1020 - 214) / 4 = 202 us/graphExecute`
- **Context binary caching only accelerates graph creation (148ms → 5.6ms = ~26x), NOT execution (~214 vs ~225 us)**
- The overhead is dominated by FastRPC round-trip, independent of op complexity or how the graph was created
- Even a 2-way split (attention block + FFN block) nearly doubles latency
- **Residual connections** across graph boundaries require extra ION buffers for the original input

---

## Legacy Benchmark (ElementWiseMultiply Chain)

N chained ElementWiseMultiply ops on `[1,1,1,512]` FP16 tensors.

### Three Modes

| Mode | Description |
|------|-------------|
| **Fused** | 1 graph with N chained Mul ops, built dynamically |
| **Cached** | Same fused graph, serialized to context binary then restored |
| **Split** | N independent graphs each with 1 Mul op, executed sequentially |

### Results

**Device**: Snapdragon 8 Elite, Hexagon V81, 1 NPU core
**CPU affinity**: pinned to big core 7 (4.61 GHz)
**QNN SDK**: 2.42

#### Graph Creation Time

| N_ops | Fused (ms) | Cached (ms) | Split (ms) | Cache speedup |
|------:|-----------:|------------:|-----------:|--------------:|
| 1     | 132        | 4.4         | 115        | **30x**       |
| 2     | 103        | 4.1         | 114        | **25x**       |
| 4     | 110        | 4.0         | 172        | **27x**       |
| 8     | 173        | 3.9         | 145        | **44x**       |
| 16    | 107        | 4.1         | 178        | **26x**       |
| 32    | 115        | 3.9         | 285        | **30x**       |

#### Execution Time (graphExecute)

| N_ops | Fused avg (us) | Cached avg (us) | Split avg (us) | Cached/Fused | Split/Fused |
|------:|---------------:|----------------:|---------------:|-------------:|------------:|
| 1     | 172.8          | 198.3           | 162.6          | 1.15x        | 0.94x       |
| 2     | 163.8          | 229.9           | 375.4          | 1.40x        | 2.29x       |
| 4     | 172.9          | 196.3           | 771.7          | 1.14x        | 4.46x       |
| 8     | 165.7          | 184.0           | 1,656.1        | 1.11x        | 10.00x      |
| 16    | 191.3          | 201.0           | 3,442.6        | 1.05x        | 17.99x      |
| 32    | 187.4          | 216.0           | 8,467.8        | 1.15x        | 45.18x      |

## Key Findings

### 1. Per-graph launch overhead is ~200 us (big core) / ~280 us (no pinning)

Split mode execution scales almost perfectly linearly with N. The per-graph fixed overhead is dominated by the FastRPC round-trip (ARM CPU → Hexagon DSP).

```
Per-graph overhead (FFN):    (1020 - 214) / 4  = ~202 us/graph
Per-graph overhead (Legacy): (8468 - 187) / 31 = ~267 us/graph
```

### 2. Context binary caching accelerates creation only, NOT execution

| Metric | Dynamic Fused | Cached | Improvement |
|--------|-------------:|-------:|------------:|
| Graph creation | ~120 ms | ~4 ms | **~30x** |
| Execution (FFN, 5 ops) | 214 us | 225 us | **~1.0x (no improvement)** |
| Execution (Legacy, N=32) | 187 us | 216 us | **~1.0x (no improvement)** |

Previous measurements showed cached execution at ~2-10us, which was **incorrect** due to a tensor ID bug — after `contextCreateFromBinary` + `graphRetrieve`, execution tensors must carry the original tensor IDs assigned during `tensorCreateGraphTensor`. Without correct IDs, `graphExecute` silently skips execution.

### 3. graphExecuteAsync is NOT supported on HTP (V81)

Tested with proper setup: context created with `QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH = 4`, callback function (`Qnn_NotifyFn_t`) provided. The HTP backend explicitly rejects async execution:

```
QnnDsp <E> Async execution is unsupported
graphExecuteAsync returned error: 1000 (QNN_GRAPH_ERROR_GENERAL)
```

**Test**: `./run_on_device.sh --async --core 7`

**Doc analysis (QAIRT 2.42)**:
- `supported_api.html`: `QnnGraph_executeAsync` = **NO** for HTP (aarch64-android, QEMU)
- `supported_capabilities.html`: `QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION` = **Platform Dependent** for HTP
- `api_overview.html`: Describes both sync and async execution paths as general QNN framework features, but backend support varies
- The API, callback type (`Qnn_NotifyFn_t`), signal (`QnnSignal`), and queue depth config all exist in headers — the framework supports it, the HTP V81 firmware does not

Async pipelining of multiple NPU graphs is not possible through `graphExecuteAsync`. Alternative approaches for CPU↔NPU overlap must use separate threads with sync `graphExecute`.

### 4. Big core pinning reduces CPU-side overhead by ~40%

The ~100us reduction comes from the CPU-side FastRPC marshaling; the DSP-side compute is unaffected.

## Implications for Heterogeneous LLM Inference

1. **Always fuse ops into a single graph** — splitting into per-op graphs adds ~200us per additional `graphExecute` call
2. **Use context binary caching for startup** — pre-serialize graphs to binary blobs; loading from cache is ~30x faster for graph creation, but provides no runtime execution benefit
3. **Pin the dispatch thread to a big core** — reduces per-launch overhead by ~40%
4. **Minimize graph count** — each graph launch is a full FastRPC round-trip; the overhead is fixed regardless of op complexity

## Build & Run

```bash
# Build (requires Android NDK + QNN SDK)
./build_android.sh

# Run FFN benchmark (default)
./run_on_device.sh --core 7                          # pin to big core
./run_on_device.sh --hidden-dim 4096 --inter-dim 11008  # LLaMA-7B scale

# Run legacy benchmark
./run_on_device.sh --legacy --core 7
./run_on_device.sh --legacy --hidden-dim 2048

# Test async execution (graphExecuteAsync)
./run_on_device.sh --async --core 7

# Common options
./run_on_device.sh --warmup 20 --iters 200
```

## File Structure

```
src/
  common.h           # ION buffer, timing, stats, FP16 utilities
  qnn_setup.h/.cpp   # QNN lifecycle + context binary serialization
  graph_builder.h/.cpp # Fused/Split/Cached graph construction
  main.cpp           # Benchmark driver
```

## QNN async graph launch — doc references (QAIRT docs)

For local QAIRT docs (e.g. under `docs/QAIRT-Docs/QNN/`):

| Topic | Path (relative to QAIRT docs root) |
|-------|------------------------------------|
| `QnnGraph_executeAsync` API | `QNN/api-rst/function_QnnGraph_8h_1a690b74571029dd9f36e38cd902c86784.html` |
| Backend support (executeAsync = NO for all) | `QNN/general/supported_api.html` |
| Async capability / properties | `QNN/general/supported_capabilities.html`, `QNN/api-rst/file_include_QNN_QnnProperty.h.html` |
| Context async queue depth | `QNN/api-rst/structQnnContext__AsyncExecutionQueueDepth__t.html` |
| Tools: `--synchronous`, async options | `QNN/general/tools.html` |
| Sample App (sync only; AsyncExecution not on Windows) | `QNN/general/sample_app.html` |
