# HeteroEdge

Snapdragon 8 Gen 5 (SM8850) 异构计算带宽测试与 UMA 验证。

## 平台信息

- **SoC**: Snapdragon 8 Gen 5 (SM8850)
- **NPU**: Hexagon V81 Tensor Processor (1 core, 8 HVX threads)
- **GPU**: Qualcomm Adreno 840 (12 CU)
- **内存**: LPDDR5X-5300, 4ch x 16bit, 理论峰值 84.8 GB/s

## 带宽测试结果

### 单加速器基线

| 加速器 | 测试方法 | 数据量 | 实测带宽 | 峰值利用率 |
|--------|---------|--------|----------|-----------|
| **HTP (NPU)** | ElementWiseAdd (2读+1写) | 128 MB x 3 | **53.70 GB/s** | 63.3% |
| **GPU** | vector_copy_float8 (1读+1写) | 1024 MB x 2 | **54.79 GB/s** | 64.6% |

两者带宽几乎相同 (~54 GB/s)，均受限于 LPDDR5X 物理带宽上限。

### GPU + NPU 并发测试

任务: Element-wise Add (C = A + B, UFIXED_POINT_8)，GPU 和 NPU 各自处理数据的一部分并同时执行。

| 模式 | 缓冲区 | 配置 | GPU GB/s | NPU GB/s | 聚合带宽 | 峰值利用率 |
|------|--------|------|----------|----------|----------|-----------|
| 独立缓冲区 | 6 个 ION buffers | 384MB, ratio=0.1 | 27.01 | 49.79 | **76.80 GB/s** | **90.6%** |
| 统一缓冲区 | 3 个 ION buffers | 512MB, ratio=0.2 | 36.05 | 36.49 | **72.54 GB/s** | **85.5%** |

- **独立缓冲区**: GPU 和 NPU 各分配独立的 ION 缓冲区（共 6 个），NPU 不受竞争影响（~50 GB/s），GPU 承受主要性能下降
- **统一缓冲区**: GPU 和 NPU 共享 3 个 ION 缓冲区，通过子区间访问同一物理内存，两者均匀分摊竞争开销
- 统一缓冲区比独立缓冲区低约 6%，为 UMA 共享内存总线的固有竞争代价
- 两种模式时间重叠度均 > 88%，确认 GPU 和 NPU 真正并行执行

### HTP 带宽关键发现

- 数据类型必须用 `UFIXED_POINT_8`（HTP 原生类型），INT_8 会插入 Cast 节点导致带宽减半
- 张量 C 维度必须 <= 1048576，否则 tiler 无法有效切分（C=2M 带宽腰斩，C=4M 降至 1.4 GB/s）
- DCVS 必须正确配置（锁定最高频率），否则带宽仅 6.4 GB/s
- 详细测试数据见 [`htp_bandwidth_test/README.md`](htp_bandwidth_test/README.md)

### GPU 带宽测试配置

- Kernel: `vector_copy_float8` (float8 SIMD)
- 设备: Adreno 840, 全局内存 7.32 GB, 12 CU
- 3 次 warmup + 10 次计时取平均

## UMA (统一内存架构) 验证

| 路径 | 内存分配 | 零拷贝 |
|------|---------|--------|
| CPU <-> GPU | `clCreateBuffer(CL_MEM_ALLOC_HOST_PTR)` | ✅ |
| CPU <-> NPU | `rpcmem_alloc(heapid=25)` + `QnnMem_register` (ION) | ✅ |
| GPU + NPU 共享 | 同一 ION fd, `clCreateSubBuffer` + `QNN_HTP_MEM_SHARED_BUFFER` | ✅ |

GPU 和 NPU 均支持与 CPU 之间的 UMA 零拷贝数据传输。GPU 和 NPU 可同时操作同一 ION 缓冲区的不同子区间（计算结果验证通过）。详见 [`UMA验证总结.md`](UMA验证总结.md)。

## 目录结构

```
├── htp_bandwidth_test/           # HTP (NPU) 单独带宽测试
├── gpu_bandwidth_test/           # GPU (OpenCL) 单独带宽测试
├── concurrent_bandwidth_test/    # GPU+NPU 并发测试 (独立缓冲区模式, 6 ION buffers)
├── unified_bandwidth_test/       # GPU+NPU 并发测试 (统一缓冲区模式, 3 ION buffers)
├── UMA验证总结.md                # UMA 验证详细报告
└── README.md                     # 本文件
```

### concurrent_bandwidth_test (独立缓冲区)

GPU 和 NPU 各自分配独立的 ION 缓冲区，互不共享物理内存。

```
Buffer A_gpu (gpu_bytes)    Buffer A_npu (npu_bytes)
Buffer B_gpu (gpu_bytes)    Buffer B_npu (npu_bytes)
Buffer C_gpu (gpu_bytes)    Buffer C_npu (npu_bytes)
```

```bash
cd concurrent_bandwidth_test
bash build_android.sh
bash run_on_device.sh --total-size-mb 384 --gpu-ratio 0.1
```

### unified_bandwidth_test (统一缓冲区)

GPU 和 NPU 共享 3 个 ION 缓冲区，通过子区间访问。GPU 使用 `clCreateSubBuffer`，NPU 使用 QNN HTP Shared Buffer API。

```
Buffer A/B/C (unified_bytes):
  [0 ............ npu_bytes | pad | npu_bytes+pad ............ total]
  [    NPU 分区 (offset=0) |     |    GPU 分区 (sub-buffer)       ]
```

```bash
cd unified_bandwidth_test
bash build_android.sh
bash run_on_device.sh --total-size-mb 256 --gpu-ratio 0.5
bash run_on_device.sh --verify  # 开启计算结果验证
```
