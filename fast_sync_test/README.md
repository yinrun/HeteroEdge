# Fast Sync 原型：GPU↔NPU 同步开销测试

## 背景

论文 **HeteroInfer** (SOSP 2025) 提出的 fast sync 机制是 decode 阶段获得 2-4x 加速的关键。

**问题**: 移动 SoC 上 `clFinish()` 同步开销约 **~400 us**，在 decode 阶段（kernel 执行仅几百微秒）占比极大。

**论文方案** (Section 4.3):
> "A flag bit is added alongside the output tensor and is updated once the output tensor
> is completely populated. The CPU core only needs to poll this flag bit for a few
> microseconds and can immediately notify the NPU for subsequent execution."

核心思路是 **数据级同步**：
1. GPU kernel 写完输出后，在共享内存中写一个 **flag bit**
2. CPU 先 `usleep(predicted_time)` 粗等待
3. CPU 直接从 ION buffer 的 CPU 映射地址 **读 flag**（不经过 OpenCL 驱动）
4. Flag 置位 → GPU 完成，通知 NPU

这完全绕过了 `clFinish()` / `cl_event` 的驱动路径，实现 ~1us 的同步开销。

## 测试场景

模拟 LLM decoder layer 的 GPU↔NPU pipeline（batch=1, hidden_dim=4096, FP16）：

```
GPU(RMSNorm) → sync → NPU(RMSNorm) → sync → GPU(RMSNorm) → sync → NPU(RMSNorm) → ...
```

GPU 和 NPU 通过 ION 共享内存实现零拷贝数据传递。

## 四种同步模式

### Mode 1: Sequential Blocking（基线）

```
主线程: clEnqueue + clFinish() → graphExecute() → 重复
```

GPU 和 NPU 串行阻塞执行。`clFinish()` 的开销直接体现在 pipeline 延迟中。

### Mode 2: Threaded + clFinish

```
主线程: clEnqueue + clFinish() → signal(npu_start) → spin_wait(npu_done)
NPU线程: spin_wait(npu_start) → graphExecute() → signal(npu_done)
```

NPU 在独立线程执行，但 GPU 仍用 `clFinish()`。目的：隔离线程机制本身的开销。

### Mode 3: Event Poll（cl_event 轮询，对照组）

```
主线程: clEnqueue + clFlush() → poll(clGetEventInfo) → signal(npu_start) → spin_wait(npu_done)
NPU线程: spin_wait(npu_start) → graphExecute() → signal(npu_done)
```

用 `clGetEventInfo()` 替代 `clFinish()`。但 `clGetEventInfo()` 仍经过 OpenCL 驱动，
与 `clFinish` 开销差异很小。**这不是论文的方案**。保留做对照。

### Mode 4: Fast Sync（论文方案）

```
主线程: clEnqueue + clFlush() → usleep(hint) → poll(共享内存 flag) → signal(npu_start) → spin_wait(npu_done)
NPU线程: spin_wait(npu_start) → graphExecute() → signal(npu_done)
```

- **GPU kernel 修改**: 写完输出数据后，在共享内存 flag buffer 写 `1`
- **CPU 同步**: `usleep(predicted_time)` 粗等待 + 直接读 ION 映射地址的 flag
- **不经过 OpenCL 驱动**：flag 是 GPU kernel 的输出，CPU 直接读共享内存

## 关键设计

### GPU Kernel Flag 写入

```opencl
__kernel void rmsnorm(
    __global scalar_t* output,
    __global const scalar_t* input,
    __global const scalar_t* gamma,
    const int hidden_dim,
    const float epsilon,
    __local float* sdata,
    __global volatile uint* done_flag)   // ← 新增：完成标志
{
  // ... Phase 1-4: 计算 RMSNorm，写入 output ...

  // Phase 5: 写完成标志
  barrier(CLK_GLOBAL_MEM_FENCE);   // 确保所有输出写入已提交
  if (get_local_id(0) == 0)
    *done_flag = 1u;
}
```

关键点：
- `barrier(CLK_GLOBAL_MEM_FENCE)` 确保所有 work-item 的数据写在 flag 之前
- `volatile` 防止 GPU 编译器优化掉写入
- batch=1 时只有一个 work-group，work-group barrier 足够

### CPU 侧 Flag 轮询

```cpp
// Flag buffer: 独立的 4 字节 ION buffer
volatile uint32_t* flag_ptr = (volatile uint32_t*)ion_flag.ptr;
*flag_ptr = 0;  // 重置

gpu_submit();   // clEnqueue + clFlush（非阻塞）

// 直接读共享内存，不经过 OpenCL 驱动
if (usleep_hint > 0) usleep(usleep_hint);
while (*flag_ptr == 0) cpu_pause();
// GPU 完成！
```

### ARM UMA 缓存一致性

SM8850 使用 ACE 协议的 coherent interconnect：
- GPU 和 CPU 在同一个一致性域内
- `CL_MEM_HOST_UNCACHED_QCOM` 确保 CPU 读绕过 CPU cache
- `volatile` 确保 C++ 编译器不优化 CPU 读操作
- `barrier(CLK_GLOBAL_MEM_FENCE)` 确保 GPU 侧内存序

### ION 共享内存（零拷贝）

三个 ION buffer：

```
ion_buf0: GPU 读 (input)  → NPU 写 (output)   [hidden_dim * 2 bytes]
ion_buf1: GPU 写 (output) → NPU 读 (input)    [hidden_dim * 2 bytes]
ion_flag: GPU 写 flag → CPU 读 flag            [4 bytes]
```

- GPU 端: `CL_MEM_ION_HOST_PTR_QCOM` Qualcomm 扩展导入
- NPU 端: `QNN_MEM_TYPE_ION` 注册

### 测量指标

| 指标 | 来源 |
|------|------|
| gpu_compute_us | OpenCL profiling (COMMAND_END - COMMAND_START) |
| gpu_sync_us | 同步机制耗时减去 compute |
| npu_compute_us | graphExecute 返回耗时 |
| npu_sync_us | 等待 NPU 完成的轮询时间 |
| step_total_us | 端到端单步延迟 |

统计量: min / P50 / avg / P99 / max

### GPU 同步诊断

对比四种 GPU 完成检测方式（GPU-only，无 NPU 干扰）：

```
1. clFinish (blocking)        — OpenCL 驱动阻塞等待
2. clFlush + Event Poll    — OpenCL 驱动级轮询（clGetEventInfo）
3. clFlush + WaitForEvents    — OpenCL OS 级等待
4. clFlush + flag poll        — 共享内存直接轮询（论文方案）← ground truth

clFinish 真正开销 = clFinish 总时间 - flag 轮询总时间
```

flag 轮询检测的是 GPU 硬件完成时刻（kernel 写 flag 到共享内存），不经过驱动。
与 clFinish 的差值才是驱动引入的真正开销。

### OpenCL Profiling 时间线

利用 `CL_PROFILING_COMMAND_QUEUED/SUBMIT/START/END` 四个硬件时间戳分解 GPU 命令流水线：

| 时间戳 | 含义 |
|--------|------|
| COMMAND_QUEUED | 命令在 host 入队时刻 |
| COMMAND_SUBMIT | 命令提交到 GPU 硬件时刻 |
| COMMAND_START | GPU 开始执行 kernel 时刻 |
| COMMAND_END | GPU 执行完成时刻 |

注意：profiling 时间戳用的是 GPU 设备时钟，与 CPU host 时钟不同，两者不能直接相减。
但各阶段的**差值**（如 SUBMIT→START）是可靠的设备侧延迟度量。

## 实测结果（SM8850, Adreno 840 + Hexagon V81）

### GPU 同步诊断

单独测试 GPU kernel（hidden=4096, FP16, batch=1, 100 次迭代）：

```
Method                    total_p50   extra
clFinish (blocking)         676.3 us
clFlush+cl_event poll       687.4 us  submit=159.3 poll=527.2 polls=1205
clFlush+WaitForEvents       654.8 us
clFlush+flag poll ★         292.2 us  polls=5956  ← ground truth
```

**Flag 轮询比 clFinish 快 384us** — 这就是 clFinish 的驱动端后处理开销。
cl_event 轮询与 clFinish 开销相近（两者都经过 OpenCL 驱动，~395us 开销），
只有 flag 方案（直接读共享内存）才能绕过驱动。

### OpenCL Profiling 时间线分析

利用 `CL_PROFILING_COMMAND_QUEUED/SUBMIT/START/END` 分解 GPU 命令流水线各阶段延迟：

```
GPU command pipeline (device clock, p50):

  Host (CPU)                        Device (GPU)
  ─────────────                     ───────────────
  clEnqueueNDRange ─┐
                    │   98.4 us     QUEUED → SUBMIT    驱动翻译：命令 → GPU 硬件指令
                    ├──────────────►
  clFlush ──────────┘               │
                                    │  244.7 us         GPU 调度：command stream → 执行单元
                                    ├──────────────────►
                                    │  START
                                    │   18.7 us         kernel 执行
                                    │  END
                                    ├──────────────────►
                                    │                    flag 写入共享内存 ← CPU 可检测 (292us)
                                    │
  clFinish 返回 ◄───────────────────┘  +384 us 驱动后处理
  (总计 676us)
```

**关键发现：**

| 阶段 | 耗时 (p50) | 说明 |
|------|-----------|------|
| QUEUED→SUBMIT | 98.4 us | OpenCL 驱动翻译命令给 GPU 硬件 |
| SUBMIT→START | 244.7 us | GPU 硬件调度延迟（最大瓶颈） |
| START→END | 18.7 us | kernel 真正执行时间 |
| QUEUED→END | 362.3 us | 设备侧总时间 |
| Flag wall time | 292.2 us | Host 视角：提交 → flag 可检测 |
| clFinish 返回 | 676.3 us | Host 视角：提交 → clFinish 返回 |
| **clFinish 开销** | **384.1 us** | **clFinish 返回 - flag 检测 = 纯驱动后处理** |

**292us 的 flag 检测时间分解**：kernel compute 仅 18.7us，剩余 ~274us 是 host 侧提交 + GPU 硬件调度延迟。这部分是不可避免的（命令必须经过驱动到达 GPU），但 flag 方案省掉了 clFinish 返回前的 384us 驱动后处理。

### Pipeline 模式对比（无绑核）

```
Mode                   step_p50   gpu_sync   npu_sync   sync_tot    speedup
------------------------------------------------------------------------
Seq Blocking            935.1 us    632.6 us      0.0 us    632.6 us     1.00x
Thread+clFinish        1101.7 us    824.4 us      1.5 us    825.9 us     0.85x
Event Poll             1169.8 us    860.5 us      1.7 us    862.2 us     0.80x
Fast Sync ★             841.1 us    356.0 us      2.3 us    358.3 us     1.11x
```

不绑核时 Fast Sync 仅 1.11x 加速。线程模式下 spin-loop 与 NPU graphExecute 竞争同一 CPU 核，
导致 NPU compute 从 283us 膨胀到 476us（+68%），严重削弱加速效果。

### Pipeline 模式对比（绑核：main→core 7, npu→core 6）

SM8850 CPU 拓扑：Core 0-5 @ 3.63GHz (performance), Core 6-7 @ 4.61GHz (prime)。
将主线程（flag 轮询）绑到 prime core 7，NPU 线程绑到 prime core 6，消除 CPU 竞争：

```
Mode                   step_p50   gpu_sync   npu_sync   sync_tot    speedup
------------------------------------------------------------------------
Seq Blocking            762.4 us    547.0 us      0.0 us    547.0 us     1.00x
Thread+clFinish         743.4 us    548.8 us      0.4 us    549.1 us     1.03x
Event Poll              671.9 us    492.9 us      0.2 us    493.1 us     1.13x
Fast Sync ★             301.5 us    119.9 us      0.4 us    120.3 us     2.53x
```

**Fast Sync + 绑核达到 2.53x 加速**，完全落在论文预测的 2-4x 范围内。

### 绑核配置综合对比

测试了四种 CPU 绑核配置下 Fast Sync 的表现，每种配置同时跑 Seq Blocking 基线：

SM8850 CPU 拓扑：
- Core 0-5 @ 3.63GHz（performance 核）
- Core 6-7 @ 4.61GHz（prime 核）

```
                         Fast Sync                     Seq Baseline
Config              step_p50  gpu_sync  npu_compute   step_p50   speedup
--------------------------------------------------------------------------
No pinning           841.1 us   356.0 us   476.0 us    935.1 us    1.11x
Little+Little (0,1)  752.0 us   364.1 us   383.6 us    941.1 us    1.25x
Big+Little (7,0)     719.2 us   285.9 us   429.8 us    765.3 us    1.06x
Big+Big (7,6) ★      301.5 us   119.9 us   179.8 us    762.4 us    2.53x
```

**关键发现：**

1. **Big+Big (7,6) 是唯一有效配置**：2.53x 加速，其他配置均 < 1.3x
2. **NPU compute 对 CPU 频率极度敏感**：
   - Prime 核 (core 6): 179.8 us
   - Performance 核 (core 0): 429.8 us（+139%）
   - 无绑核: 476.0 us（+165%，spin-loop 竞争 + 调度不确定）
3. **GPU sync 只对主线程核频率敏感**：
   - Big 核 (core 7): 119.9-285.9 us
   - Little 核 (core 0): 364.1 us
4. **NPU graphExecute() 是 CPU RPC 调度到 DSP**：频率越低 RPC 延迟越大

### 绑核效果分析（Fast Sync: 无绑核 vs Big+Big）

| 指标 | 不绑核 | Big+Big (7,6) | 变化 |
|------|--------|---------------|------|
| step_p50 | 841.1 us | 301.5 us | **-64%** |
| gpu_sync | 356.0 us | 119.9 us | -66% |
| npu_compute | 476.0 us | 179.8 us | **-62%** |
| npu_sync | 2.3 us | 0.4 us | -83% |
| speedup | 1.11x | **2.53x** | |

两个主要改善来源：

1. **NPU compute 大幅下降**（476.0→179.8us, -296us）：之前 spin-loop 和 NPU graphExecute
   竞争同一个 CPU 核，导致 DSP 调度严重受阻。绑核后两个线程各自独占 prime 核，消除竞争。
2. **GPU sync 也下降**（356.0→119.9us, -236us）：主线程绑到 prime 核后，flag 轮询响应更快
   （不被抢占），submit 路径也更快。

### 开销来源分析

```
Sequential Blocking (762 us, Big+Big 绑核):
  gpu: clEnqueue + clFinish (含驱动后处理) = 547us (gpu_sync)
  npu: graphExecute = 197us
  合计: 547 + 197 ≈ 762us (串行)

Fast Sync (302 us, Big+Big 绑核):
  gpu: clEnqueue + clFlush + flag_detect = 120us (gpu_sync)
  npu: graphExecute = 180us (绑核消除 CPU 竞争)
  sync: npu_sync ≈ 0us
  合计: 120 + 180 ≈ 302us (gpu→npu 串行，省掉 clFinish 开销 + 消除 CPU 竞争)
```

**绑核后省掉了 ~460us**：clFinish 驱动开销 ~427us + CPU 竞争消除 ~17us NPU 加速。

## 目录结构

```
fast_sync_test/
├── README.md                     # 本文档
├── CMakeLists.txt
├── build_android.sh
├── run_on_device.sh
├── kernels/
│   └── rmsnorm.cl                # GPU FP16 RMSNorm + 完成 flag 写入
└── src/
    ├── common.h                  # ION/rpcmem + SyncMode/StepTiming/Stats 类型
    ├── gpu_engine.h/.cpp         # GPU OpenCL: blocking + nonblocking + flag-based
    ├── npu_engine.h/.cpp         # NPU QNN: blocking graphExecute
    ├── pipeline.h/.cpp           # 四种同步模式 + GPU 诊断
    └── main.cpp                  # CLI + 结果输出
```

### CPU 绑核

通过 `sched_setaffinity()` 将主线程和 NPU 工作线程分别绑定到不同 CPU 核，
消除 spin-loop 轮询与 NPU graphExecute 的 CPU 竞争。

SM8850 推荐配置：`--main-core 7 --npu-core 6`（两个 prime 核 @ 4.61GHz）。

## 构建与运行

```bash
export ANDROID_NDK=/path/to/android-ndk-r25c
export QNN_SDK_ROOT=/path/to/qualcomm/qairt/2.42.0.251225

bash build_android.sh
bash run_on_device.sh
bash run_on_device.sh --main-core 7 --npu-core 6              # 绑核（推荐）
bash run_on_device.sh --hidden-dim 2048 --steps 200
bash run_on_device.sh --mode fast --main-core 7 --npu-core 6  # 仅测 Fast Sync
```

## 结论

1. **clFinish 开销确认**：SM8850 上 clFinish 的驱动后处理开销约 **384us**，与论文 ~400us 一致
2. **cl_event 无法替代 flag**：clGetEventInfo 轮询与 clFinish 开销相同（~395us），两者共用驱动代码路径
3. **Flag-based 共享内存轮询有效**：绕过 OpenCL 驱动，GPU 完成检测从 ~676us 降到 ~292us
4. **CPU 绑核至关重要**：不绑核时 spin-loop 与 NPU 竞争导致 NPU compute +68%，绑核后消除竞争
5. **必须绑定到 prime 核**：四种绑核配置中只有 Big+Big (7,6) 有效，NPU RPC 调度对 CPU 频率极敏感
6. **Fast Sync + Big+Big 绑核达到 2.53x 加速**：step_p50 从 762us 降到 302us，落在论文预测的 2-4x 范围内
7. **主要瓶颈在 GPU 调度**：SUBMIT→START 的 245us 调度延迟不可避免，占 flag wall time 的 84%

## 参考

- **HeteroInfer** (SOSP 2025), Section 4.3: Fast Synchronization
- Qualcomm QNN SDK 2.42.0 — `graphExecute()` (blocking API)
- OpenCL 2.0 — `clFlush()`, `clGetEventInfo()`, `CL_QUEUE_PROFILING_ENABLE`, `CL_PROFILING_COMMAND_*`
- Qualcomm OpenCL extension — `CL_MEM_ION_HOST_PTR_QCOM` (zero-copy ION import)
