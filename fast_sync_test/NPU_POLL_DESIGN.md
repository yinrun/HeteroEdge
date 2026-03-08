# NPU-Side Flag Polling (NPU_POLL) 设计文档

## Context

当前 fast sync 流程 (301.5us/step, 绑核):
```
GPU submit → GPU执行→写flag → CPU轮询flag(~120us) → 启动NPU → NPU执行(~180us)
                                                      ↑ NPU launch开销在关键路径上
```

**核心问题**: NPU `graphExecute()` 的 179.8us 中，包含了 RPC dispatch 开销 (~80-100us)，这部分开销完全在关键路径上。

**优化思路**: 将 flag 轮询从 CPU 侧移到 NPU 侧——用自定义 HVX 算子实现。GPU 和 NPU 同时 launch，NPU 的 HVX kernel 先自旋等待 flag，检测到后再计算 RMSNorm。这样 NPU 的 RPC dispatch 开销与 GPU 执行并行。

```
优化后:
GPU submit ──── GPU执行 ──── 写flag ─┐
NPU launch ──── RPC dispatch ──── HVX自旋等flag ──→ HVX计算RMSNorm
                (与GPU并行)          (检测到flag)
Total ≈ max(GPU_flag_time, NPU_dispatch) + HVX_compute
```

**预期收益**: 从 ~301us 降至 ~120-180us (NPU dispatch 完全隐藏)。

---

## 实现步骤

### Step 1: 创建自定义 HVX FlagPoll RMSNorm 算子

**新建目录**: `fast_sync_test/custom_op/`

**1.1 HVX Kernel** — `FlagPollRmsNormImpl.cpp`
- 基于 `rmsnorm/custom_op/RmsNormOpPackageRmsNorm.cpp` 的 HVX RMSNorm 实现
- 新增 flag 轮询前导阶段:
  - 输入: data(FP16), gamma(FP16), **flag(UINT32 tensor [1,1,1,1])**
  - 参数: epsilon(float)
  - 输出: output(FP16), **poll_count(UINT32 tensor [1,1,1,1])** (诊断用)
- HVX kernel 逻辑:
  1. 获取 flag tensor 的 raw pointer (`volatile uint32_t*`)
  2. 自旋轮询 flag，每次迭代用 `dcinva` 失效 Hexagon L2 cache line（确保看到 GPU 的写入）
  3. 设置最大轮询次数上限 (防止死锁，例如 1000000 次)
  4. flag 检测到后，对 input data 区域做 cache invalidation
  5. 执行标准 RMSNorm HVX 计算 (复用 `rmsnorm_hvx_row` 逻辑)

**1.2 OpPackage Interface** — `FlagPollRmsNormInterface.cpp`
- 基于 `rmsnorm/custom_op/RmsNormOpPackageInterface.cpp`
- 注册算子名: `FlagPollRmsNorm`, 包名: `flagpoll_rmsnorm.HvxOpPackage`
- Validation: 3 inputs (data, gamma, flag), 2 outputs (result, poll_count), 1 param (epsilon)

**1.3 Makefile**
- 基于 `rmsnorm/custom_op/Makefile`
- 构建目标: hexagon-v81 (.so) + aarch64-android (.so)

### Step 2: 修改 NPU Engine 支持自定义算子

**文件**: `fast_sync_test/src/npu_engine.cpp` / `npu_engine.h`

- 新增 `npu_init_flagpoll()` 函数:
  - 调用 `backendRegisterOpPackage()` 加载自定义算子包
  - 构建包含 FlagPollRmsNorm 节点的 QNN graph
  - 注册 flag ION buffer 为 tensor (memRegister + memHandle 绑定)
  - flag tensor 作为第三个输入传入 graph node

### Step 3: 新增 NPU_POLL Pipeline 模式

**文件**: `fast_sync_test/src/pipeline.cpp`

- 新增 `run_npu_poll()` 函数:
  ```
  for each step:
    *flag_ptr = 0              // 重置 flag
    gpu_submit()               // clEnqueue + clFlush (非阻塞, ~10us)
    npu_execute_blocking()     // graphExecute (内部 HVX 轮询 flag + 计算)
    // 返回时 GPU 和 NPU 都已完成
  ```
- 关键区别: **无 CPU 侧轮询**, 无需 NPU worker 线程, 主线程顺序调用即可

**文件**: `fast_sync_test/src/common.h`
- `SyncMode` 枚举新增 `NPU_POLL`

**文件**: `fast_sync_test/src/main.cpp`
- 新增 `--mode npu_poll` 命令行选项

### Step 4: 构建和部署脚本

- `fast_sync_test/custom_op/build.sh` — 编译自定义算子
- 修改 `fast_sync_test/run_on_device.sh` — push 算子 .so 到设备

---

## 关键风险: Cache Coherency

**最高风险**: Hexagon DSP L2 cache 可能缓存 flag 的旧值，导致 HVX 永远看不到 GPU 写入的 flag=1。

**策略** (按优先级):
1. **`dcinva` 指令**: 在轮询循环中失效 flag 所在 cache line → 强制从 DRAM 读取
2. **Uncached 内存分配**: `rpcmem_alloc` 时使用 uncached flag，使 DSP 侧访问绕过 L2
3. **`Q6_dcfetch_A`**: 如果 `dcinva` 在用户态不可用，用 prefetch hint 作为替代

**Data tensor 的 coherency**: flag 检测到后，GPU output data 也需要 cache invalidation，对 input 数据区域循环调用 `dcinva` (8KB / 32B cache line = 256 次)。

**安全措施**: 设置 `max_poll_iters` 超时，避免 HVX 无限自旋。

---

## 验证方案

1. **Cache coherency 验证**: 先构建最小化测试——CPU 写 flag=1 后调用 graphExecute，验证 HVX 能立即检测到
2. **正确性验证**: 对比 NPU_POLL 模式和 SEQUENTIAL_BLOCKING 模式的 RMSNorm 输出
3. **性能对比**: 5 种模式 (hidden=4096, batch=1, 100 steps, CPU 绑核) 的 step_total 对比
4. **诊断指标**: 自定义算子输出 poll_count 到 debug tensor，量化 HVX 实际等待时间

---

## 关键文件清单

| 操作 | 文件路径 |
|------|----------|
| 参考模板 | `rmsnorm/custom_op/RmsNormOpPackageRmsNorm.cpp` |
| 参考模板 | `rmsnorm/custom_op/RmsNormOpPackageInterface.cpp` |
| 参考模板 | `rmsnorm/custom_op/Makefile` |
| 新建 | `fast_sync_test/custom_op/FlagPollRmsNormImpl.cpp` |
| 新建 | `fast_sync_test/custom_op/FlagPollRmsNormInterface.cpp` |
| 新建 | `fast_sync_test/custom_op/Makefile` |
| 新建 | `fast_sync_test/custom_op/build.sh` |
| 修改 | `fast_sync_test/src/common.h` |
| 修改 | `fast_sync_test/src/npu_engine.h` |
| 修改 | `fast_sync_test/src/npu_engine.cpp` |
| 修改 | `fast_sync_test/src/pipeline.cpp` |
| 修改 | `fast_sync_test/src/main.cpp` |
| 修改 | `fast_sync_test/run_on_device.sh` |
