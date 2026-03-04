# GPU + NPU 并发带宽测试（统一缓冲区模式）

## 背景

本测试在 **Snapdragon 8 Gen 5 (SM8850)** 上验证 GPU 和 NPU **共享同一 ION 缓冲区**并发执行的带宽表现。

与 `concurrent_bandwidth_test`（独立缓冲区，6 个 ION buffers）不同，本测试仅分配 **3 个 ION 缓冲区**（A, B, C），GPU 和 NPU 通过子区间访问同一物理内存的不同分区。

## 实测结果

**平台**: Snapdragon 8 Gen 5 (SM8850), Adreno 840 + Hexagon V81, LPDDR5X-5300
**理论峰值**: 84.8 GB/s (10600 MT/s × 64-bit, 4ch × 16bit)
**运算**: Element-wise Add (C = A + B), UFIXED_POINT_8

### Solo 基线

| 处理器 | 256MB | 512MB |
|--------|-------|-------|
| GPU (Adreno 840, OpenCL) | 51.01 GB/s (60.2%) | ~52 GB/s (61%) |
| NPU (Hexagon V81, QNN BURST) | 53.87 GB/s (63.5%) | ~53 GB/s (63%) |

### 并发 Ratio 扫描（统一缓冲区）

**256MB:**

| GPU Ratio | GPU MB | NPU MB | GPU GB/s | NPU GB/s | 聚合 GB/s | 利用率 |
|-----------|--------|--------|----------|----------|-----------|--------|
| 0.10 | 25 | 230 | 35.11 | 36.64 | **71.74** | **84.6%** |
| 0.20 | 51 | 204 | 34.47 | 36.82 | **71.29** | 84.1% |
| 0.50 | 128 | 128 | 33.19 | 36.82 | 70.02 | 82.6% |

**512MB:**

| GPU Ratio | GPU MB | NPU MB | GPU GB/s | NPU GB/s | 聚合 GB/s | 利用率 |
|-----------|--------|--------|----------|----------|-----------|--------|
| 0.10 | 51 | 460 | 34.84 | 36.57 | 71.41 | 84.2% |
| 0.20 | 102 | 409 | 36.05 | 36.49 | **72.54** | **85.5%** |

### 与独立缓冲区模式对比

| 模式 | 缓冲区数 | 最佳配置 | GPU GB/s | NPU GB/s | 聚合 GB/s | 利用率 |
|------|---------|---------|----------|----------|-----------|--------|
| 独立缓冲区 | 6 | 384MB, ratio=0.1 | 27.01 | 49.79 | **76.80** | **90.6%** |
| 统一缓冲区 | 3 | 512MB, ratio=0.2 | 36.05 | 36.49 | **72.54** | **85.5%** |

### 关键发现

1. **峰值聚合带宽: 72.54 GB/s (85.5%)** — 512MB, ratio=0.2
2. **GPU/NPU 竞争均匀分摊** — 统一模式下两者均 ~36 GB/s，独立模式下 GPU ~27 / NPU ~50
3. **比独立缓冲区低约 6%** — 同一 fd 的子区间访问增加了内存控制器争用
4. **ratio 不敏感** — 统一模式下 ratio 0.1-0.5 聚合带宽变化 < 3%，因为竞争分摊均匀
5. **计算结果验证通过** — `--verify` 确认 NPU 分区、GPU 分区结果正确，padding 间隙未被写入

## 内存布局

```
Buffer A/B/C (unified_bytes):
  [0 .............. npu_bytes | pad | npu_bytes+pad .............. total]
  [    NPU 分区 (offset=0)   |     |    GPU 分区 (clCreateSubBuffer)   ]
```

- **NPU**: 通过 QNN HTP Shared Buffer API (`QNN_HTP_MEM_SHARED_BUFFER`) 注册 `[0, npu_bytes)` 子区间
- **GPU**: 通过 `clCreateSubBuffer(parent, CL_BUFFER_CREATE_TYPE_REGION, {offset, size})` 访问 `[gpu_offset, total)` 子区间
- **对齐**: 所有分区按 1MB 对齐（满足 QNN 张量形状和 OpenCL `CL_DEVICE_MEM_BASE_ADDR_ALIGN` 要求）

### 为什么 NPU 在前、GPU 在后

QNN `memRegister` 的标准 ION 模式（`Qnn_MemIonInfo_t`）**无 offset 字段**，NPU 总是从 buffer 的字节 0 开始。OpenCL 的 `clCreateSubBuffer` 支持任意偏移，因此 GPU 在后。

## 目录结构

```
unified_bandwidth_test/
├── README.md
├── CMakeLists.txt
├── build_android.sh
├── run_on_device.sh
├── kernels/
│   └── element_add.cl          # GPU OpenCL 内核（uchar16 向量化加法）
└── src/
    ├── main.cpp                 # 入口：统一缓冲区分配、分区、线程编排
    ├── common.h                 # 共享类型：BandwidthResult, SpinBarrier, rpcmem API
    ├── gpu_bandwidth.h/.cpp     # GPU: ION 导入 + clCreateSubBuffer 子视图
    └── htp_bandwidth.h/.cpp     # NPU: QNN HTP Shared Buffer API 子区间注册
```

## 与 concurrent_bandwidth_test 的代码差异

| 模块 | concurrent（独立） | unified（统一） |
|------|-------------------|----------------|
| `gpu_init()` | `gpu_init(A, B, C, kernel)` | `gpu_init(A, B, C, kernel, offset, partition_size)` |
| GPU 缓冲区 | 直接 import ION → `g_bufA/B/C` | import 全量 → `g_parentA/B/C` → `clCreateSubBuffer` → `g_bufA/B/C` |
| `htp_init()` | `htp_init(A, B, C)` | `htp_init(A, B, C, partition_size, force_cores)` |
| NPU 注册 | `QNN_MEM_TYPE_ION` 全量注册 | `QNN_HTP_MEM_SHARED_BUFFER` 子区间注册 |
| main 分配 | 6 个 ION buffers | 3 个 ION buffers + 子区间分派 |
| 验证 | 无 | `--verify` 校验 NPU/GPU 分区 + padding 间隙 |

## 命令行参数

```
./unified_bandwidth_test [选项]

--mode gpu|npu|concurrent|all    测试模式（默认 all）
--total-size-mb N                总数据大小 MB（默认 256）
--gpu-ratio R                    GPU 分区比例 0.0-1.0（默认 0.5）
--gpu-iters N                    GPU 迭代次数（默认自动）
--npu-iters N                    NPU 迭代次数（默认自动）
--pad-mb N                       分区间 padding MB（默认 0）
--npu-cores N                    强制 NPU 核心数（默认自动）
--verify                         验证计算结果 (C=A+B)
```

## 构建与运行

```bash
# 构建
export ANDROID_SDK_ROOT=/path/to/android-sdk
export QNN_SDK_ROOT=/path/to/qairt/2.42.0.251225
bash build_android.sh

# 运行（推送到设备并执行）
bash run_on_device.sh --total-size-mb 512 --gpu-ratio 0.2

# 带验证
bash run_on_device.sh --total-size-mb 64 --gpu-ratio 0.5 --verify
```

## 输出示例

`--total-size-mb 512 --gpu-ratio 0.2`:

```
=== 异构并发带宽测试 (统一缓冲区模式) ===
理论峰值: 84.8 GB/s (LPDDR5X-5300, 4ch x 16bit)
任务: Element-wise Add (C = A + B)
总大小: 511 MB, GPU: 102 MB (20%), NPU: 409 MB (80%)

--- GPU 设备信息 ---
  设备: QUALCOMM Adreno(TM) 840
  计算单元: 12, 全局内存: 7.32 GB

--- NPU 设备信息 ---
  Hexagon V81, 1 core(s), 8 HVX threads, BURST 模式

=== GPU-Only 基线 (全量 511 MB) ===
  GPU: 51.01 GB/s (7 iters, 0.100s)
  利用率: 60.2%

=== NPU-Only 基线 (全量 511 MB) ===
  NPU: 53.87 GB/s (3 iters, 0.084s)
  利用率: 63.5%

=== GPU + NPU 并发 (统一缓冲区 511MB: GPU 102MB + NPU 409MB) ===
  统一缓冲区: A(fd=7) B(fd=8) C(fd=9), 各 511MB
  GPU: 36.05 GB/s (16 iters, 0.133s)
  NPU: 36.49 GB/s (3 iters, 0.099s)
  Wall clock: 0.210s
  聚合带宽: 72.54 GB/s (85.5%)

=== 总结 ===
+-----------+-----------+-----------+---------+---------+
| 模式      | GPU GB/s  | NPU GB/s  | 聚合    | 利用率  |
+-----------+-----------+-----------+---------+---------+
| GPU only  |     51.01 |     --    |   51.01 |  60.2%  |
| NPU only  |     --    |     53.87 |   53.87 |  63.5%  |
| 并发      |     36.05 |     36.49 |   72.54 |  85.5%  |
+-----------+-----------+-----------+---------+---------+
```

## 参考

- **HeteroInfer** (SOSP 2025): *Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference*
- `concurrent_bandwidth_test/` — 独立缓冲区模式（6 ION buffers，峰值 76.8 GB/s）
- `htp_bandwidth_test/` — NPU 单独带宽测试
- `gpu_bandwidth_test/` — GPU 单独带宽测试
