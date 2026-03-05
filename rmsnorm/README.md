# RMSNorm on Qualcomm HTP: 方案总结与问题记录

## 1. 项目目标

在 Qualcomm SM8850 (Snapdragon 8 Gen 5) 平台上，测试 RMSNorm 算子在 HTP (Hexagon Tensor Processor) 上的可用性和性能，并与 GPU (Adreno 840 OpenCL) 进行对比。

RMSNorm 是 LLM（如 LLaMA、Qwen）中的核心归一化算子，公式为：

```
y[i] = (x[i] / sqrt(mean(x²) + ε)) * γ[i]
```

## 2. 实验环境

| 项目 | 规格 |
|------|------|
| SoC | Snapdragon 8 Gen 5 (SM8850) |
| GPU | Qualcomm Adreno 840, 12 CU, 7.32 GB |
| NPU | Hexagon V81, 1 core, 8 HVX threads |
| 内存 | LPDDR5X-5300, 4ch × 16bit, 理论峰值 84.8 GB/s |
| SDK | QAIRT 2.42.0.251225 (QNN) |
| NDK | android-ndk-r25c |

## 3. 技术方案

### 3.1 整体架构

项目采用 C++ 单文件实现 (`src/main.cpp`)，通过 Android NDK 交叉编译后 adb push 到设备运行。核心包含两个测试引擎：

- **GPU 引擎 (`GpuSession`)**: 基于 OpenCL FP16，使用自定义 kernel (`kernels/rmsnorm.cl`) 实现 RMSNorm。每个 work-group 处理一个 batch 行，local memory 做 tree reduction 求 RMS。
- **NPU 引擎 (`NpuSession`)**: 基于 QNN C API 动态构建计算图，支持三种模式：
  1. **Native**: 使用 `QNN_OP_RMS_NORM` 原生算子
  2. **Decomposed**: 用 Mul → ReduceMean → Add → Rsqrt → Mul → Mul 六步分解
  3. **ElementAdd**: UINT8 加法基线，测量 NPU 调度开销

### 3.2 QNN 图构建流程

```
dlopen(libQnnHtp.so)
  → QnnInterface_getProviders()
  → backendCreate() → deviceCreate() → contextCreate()
  → memRegister() (ION buffers)
  → graphCreate() (with HTP custom config)
  → tensorCreateGraphTensor() (ALL tensors, including param tensors)
  → graphAddNode() → graphFinalize()
  → graphExecute() (warmup + timed)
```

### 3.3 ION 共享内存

NPU 通过 `libcdsprpc.so` 的 `rpcmem_alloc/rpcmem_to_fd` 分配 ION buffer，然后通过 `QnnMem_register` 注册给 HTP 后端。这样 CPU 和 DSP 可以零拷贝共享数据。

### 3.4 性能模式配置

通过 DCVS V3 配置锁定 HTP 到最高频率：
- Bus/Core voltage corner 设为 `MAX_VOLTAGE_CORNER`
- 禁用 DCVS 动态调频
- 禁用 sleep

## 4. 遇到的困难与解决

### 4.1 [关键] axes tensor 未注册为 graph tensor

**现象**: 所有 `graphAddNode(RmsNorm)` 调用均返回错误码 6001 (`QNN_GRAPH_ERROR_INVALID_HANDLE`)，无论 FP16/FP32 数据类型、是否带 beta、是否设置 FP16 precision 均失败。

**排查过程**:
1. 最初怀疑是 FP16 tensor 的 quantization 参数配置问题
2. 尝试了 rank-1 vs rank-4 的 gamma/beta 配置 → 无效
3. 尝试了 FP32 数据类型 → 无效
4. 尝试了每次失败后重建 context 避免状态污染 → 无效
5. **添加 QNN 日志回调** (`QnnLog_create` with `QNN_LOG_LEVEL_DEBUG`)，获得关键错误信息：
   ```
   [QNN-ERR] Failed to create InputDef for Op rmsnorm (qti.aisw::RmsNorm) param axes
   [QNN-ERR] Failed to set the params for the op: RmsNorm
   ```

**根因**: QNN `graphAddNode()` 的文档明确要求：

> All tensor objects in the operation configuration must have been created via `QnnTensor_createGraphTensor()` or `QnnTensor_createContextTensor()`.

这包括 **参数中的 tensor**（如 `QNN_OP_RMS_NORM_PARAM_AXES`）。代码中 axes tensor 作为 `Qnn_Param_t.tensorParam` 传入 `graphAddNode`，但没有先调用 `tensorCreateGraphTensor` 注册，导致后端找不到该 tensor 的定义。

**修复**: 在 `graphAddNode` 之前，对 axes tensor 也调用 `tensorCreateGraphTensor`:

```cpp
// 修复前 — axes tensor 仅作为局部变量，未注册
Qnn_Tensor_t axesTensor = ...;
axesP.tensorParam = axesTensor;
q->graphAddNode(graph, op);  // ← 失败: INVALID_HANDLE

// 修复后 — 先注册 axes tensor
Qnn_Tensor_t axesTensor = ...;
q->tensorCreateGraphTensor(graph, &axesTensor);  // ← 关键一步
axesP.tensorParam = axesTensor;
q->graphAddNode(graph, op);  // ← 成功
```

同样的问题也影响了 Decomposed 模式中 `ReduceMean` 的 axes 参数。

### 4.2 错误码误导性

`QNN_GRAPH_ERROR_INVALID_HANDLE` (6001) 的语义是 "graph handle 无效"，但实际原因是参数 tensor 未注册。这个错误码缺乏区分度，容易让开发者误以为是 graph 创建或 context 的问题，而不是去排查参数 tensor 的注册状态。

### 4.3 FP16 vs FP32 数据类型

**发现**: HTP 上 Native RmsNorm 成功的配置是 **FP32 数据类型 + FP16 精度模式**（`QNN_DATATYPE_FLOAT_32` + `QNN_PRECISION_FLOAT16`），而非直接使用 `QNN_DATATYPE_FLOAT_16`。HTP 后端内部会将 FP32 tensor 转换为 FP16 进行计算。

### 4.4 失败图导致 context 状态污染

当 `graphCreate` 成功但 `graphAddNode` 失败时，需要销毁整个 context 并重建，否则后续在同一 context 下创建新图可能行为异常。方案中对每次策略尝试采用 deregister → contextFree → contextCreate → re-register 的完整重置流程。

### 4.5 Decomposed 模式中的 broadcasting

`gamma` tensor 在 RMSNorm 中需要与 `[batch, 1, 1, hidden]` 的 tensor 做逐元素乘法。使用 rank-1 `[hidden]` 的 gamma 会导致维度不匹配。解决方案是将 gamma 扩展为 rank-4 `[1, 1, 1, hidden]` 以兼容 HTP 的 broadcasting 规则。

## 5. 测试结果

### 5.1 NPU RmsNorm 支持情况

| 模式 | 状态 | 配置 |
|------|------|------|
| Native `QNN_OP_RMS_NORM` | **支持** | FP32 dtype + FP16 precision + 3 inputs (data, gamma, beta) |
| Decomposed (6 node graph) | **支持** | FP16 precision |
| ElementWiseAdd (UINT8) | **支持** | 基线 |

### 5.2 GPU vs NPU 性能对比

```
Scene      batch   hid |   GPU(us) GPU(GB/s) |   NPU-RMS  BW(GB/s) |   Add(us)  BW(GB/s) | GPU/NPU
--------------------------------------------------------------------------------------------------------------
decode         1  2048 |      62.7      0.18 |     264.8      0.04 |     269.1      0.02 |  4.22x
decode         1  3200 |      60.3      0.30 |     272.9      0.07 |     271.7      0.03 |  4.53x
decode         1  4096 |      61.4      0.37 |     274.8      0.08 |     275.2      0.04 |  4.47x
prefill-16    16  4096 |      60.3      4.17 |     373.4      0.67 |     279.8      0.65 |  6.19x
prefill-64    64  4096 |      74.4     13.23 |     658.0      1.50 |     293.6      2.49 |  8.84x
pf-256       256  4096 |     110.5     35.40 |    2526.8      1.55 |     332.5      8.81 | 22.86x
pf-512       512  4096 |     230.7     33.89 |    4085.5      1.91 |     411.5     14.24 | 17.71x
pf-1k       1024  4096 |     483.9     32.31 |    7342.1      2.13 |     553.0     21.19 | 15.17x
```

### 5.3 关键分析

1. **GPU 全面优于 NPU**: GPU 在所有 batch size 下均比 NPU Native RmsNorm 快 4~23 倍。

2. **NPU RmsNorm 延迟随 batch 线性增长**: batch=1 时 ~265us，batch=1024 时 ~7342us，接近线性。说明 HTP 对 RmsNorm 的实现可能没有有效利用并行性。

3. **GPU 延迟在小 batch 时几乎恒定**: batch 1~16 均为 ~60us，这是 GPU kernel launch 和 RMS reduction 的固定开销。batch 增大后 GPU 带宽利用率快速上升至 35 GB/s（理论峰值的 41%）。

4. **NPU RmsNorm 带宽利用率极低**: 仅 0.04~2.5 GB/s (理论峰值的 0.05%~3%)，远低于 NPU 在 UINT8 ElementAdd 上的表现（21 GB/s）。FP16 归一化操作明显不是 HTP 的强项。

5. **NPU Add 基线反而更快**: NPU 执行一次 UINT8 ElementAdd 只需 ~270us，而 FP16 RmsNorm 在相同数据量下需要 ~265us~7342us。这说明 FP16 RmsNorm 的开销主要来自计算而非数据搬运。

### 5.4 Custom HVX Op vs QNN Native Op 对比

使用 Hexagon SDK 6.5.0.0 实现了自定义 HVX FP16 RMSNorm 算子（`custom_op/`），与 QNN 内置 `QNN_OP_RMS_NORM` 在 HTP 上进行对比。

```
Scene        batch   hid | Cust(us) BW(GB/s)       MaxErr | Natv(us) BW(GB/s)       MaxErr | Speedup
------------------------------------------------------------------------------------------------------------------------------
decode-2k        1  2048 |    808.0     0.02 2.071619e-03 |    807.3     0.02 2.071619e-03 | 1.00x
decode-3.2k      1  3200 |    813.8     0.02 2.264500e-03 |    820.2     0.02 2.264500e-03 | 1.01x
decode-4k        1  4096 |    819.2     0.03 2.174854e-03 |    837.5     0.03 2.174854e-03 | 1.02x
prefill-16      16  4096 |   1294.0     0.21 2.145529e-03 |   1079.8     0.25 2.145529e-03 | 0.83x
prefill-64      64  4096 |   2404.1     0.44 2.191782e-03 |   1881.6     0.56 2.191782e-03 | 0.78x
pf-256         256  4096 |   6446.1     0.65 2.099752e-03 |   4450.0     0.94 2.099752e-03 | 0.69x
pf-512         512  4096 |  18152.0     0.46 2.149343e-03 |   7628.7     1.10 2.149343e-03 | 0.42x
pf-1k         1024  4096 |  30840.0     0.54 2.029657e-03 |  12567.8     1.34 2.029657e-03 | 0.41x
```

**正确性验证**: Custom 和 Native 输出逐元素 bit-identical（均为 FP16 精度），max_err ~2e-3 vs FP32 CPU 参考值，符合 FP16 精度预期。

### 5.5 Custom vs Native 分析

1. **batch=1 时性能持平**: Custom ~810us vs Native ~820us，自定义 HVX 内核与 QNN 内置实现效率相当。

2. **大 batch 时 Native 领先 1.5~2.4x**: Custom 缺少 `AUTOSPLIT` tiling 规则，所有行在单个 HVX 线程上串行处理。Native 则由 HTP runtime 自动跨 8 个 HVX 线程做行级并行。HTP 日志确认：`does not have a valid splitting rule`。

3. **数值完全一致**: 两者在 HTP 上走相同的 FP32→FP16→qf32→FP16→FP32 数据路径，计算结果 bit-identical。

4. **优化方向**: 为 custom op 添加 `AUTOSPLIT` 规则可在 batch 维度自动切分，预期能追平 Native 性能。

## 6. 结论

**RMSNorm 更适合在 GPU 上执行**，这一结论从算子层面得到了实证验证：

- HTP 虽然支持 `QNN_OP_RMS_NORM`，但其 FP16 实现效率极低
- GPU OpenCL FP16 RMSNorm 在 decode (batch=1) 场景下仅需 60us，而 NPU 需要 265us
- 在 prefill 场景（大 batch）下差距进一步拉大至 15~23 倍
- 自定义 HVX 算子在 batch=1 时能追平 QNN 内置算子，但大 batch 因缺少多线程 tiling 慢 1.5~2.4x
- HTP 的优势在于量化矩阵运算（如 INT8 MatMul），而非 element-wise 的归一化操作

对于异构 LLM 推理，建议将 RMSNorm/LayerNorm 等归一化算子调度到 GPU，将量化 Attention/FFN 的矩阵乘法调度到 NPU。

## 7. 项目结构

```
rmsnorm/
├── CMakeLists.txt              # CMake 构建配置
├── build_android.sh            # Android 交叉编译脚本
├── run_on_device.sh            # 设备部署与执行脚本
├── kernels/
│   └── rmsnorm.cl              # OpenCL FP16 RMSNorm kernel
├── src/
│   └── main.cpp                # 主程序（GPU + NPU benchmark）
├── custom_op/
│   ├── Makefile                # Hexagon V81 + ARM aarch64 构建
│   ├── build.sh                # 一键构建脚本
│   ├── RmsNormOpPackageRmsNorm.cpp      # HVX FP16 RMSNorm 内核实现
│   ├── RmsNormOpPackageInterface.cpp    # QNN op package 注册接口
│   └── test/
│       ├── main.cpp            # Custom vs Native benchmark + 正确性验证
│       ├── CMakeLists.txt      # 测试程序构建配置
│       ├── build_test.sh       # 编译测试程序
│       └── run_on_device.sh    # 部署并运行测试
└── README.md                   # 本文档
```

## 8. 构建与运行

```bash
# GPU + NPU benchmark
bash build_android.sh
bash run_on_device.sh [--iters 200] [--warmup 20]

# Custom HVX op package
cd custom_op && bash build.sh

# Custom vs Native benchmark
cd custom_op/test && bash build_test.sh && bash run_on_device.sh
```

依赖：
- Android NDK r25c
- Qualcomm QAIRT SDK 2.42.0
- Hexagon SDK 6.5.0.0（custom op 构建）
- 已连接的 SM8850 设备（adb）
