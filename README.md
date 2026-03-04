# HeteroEdge

Snapdragon 8 Gen 5 (SM8850) 异构计算带宽测试与 UMA 验证。

## 平台信息

- **SoC**: Snapdragon 8 Gen 5 (SM8850)
- **NPU**: Hexagon V81 Tensor Processor
- **GPU**: Qualcomm Adreno 840 (12 CU)
- **内存**: LPDDR5X-5300, 4ch x 16bit, 理论峰值 84.8 GB/s

## 带宽测试结果

| 加速器 | 测试方法 | 数据量 | 实测带宽 | 峰值利用率 |
|--------|---------|--------|----------|-----------|
| **HTP (NPU)** | ElementWiseAdd (2读+1写) | 128 MB x 3 | **53.70 GB/s** | 63.3% |
| **GPU** | vector_copy_float8 (1读+1写) | 1024 MB x 2 | **54.79 GB/s** | 64.6% |

两者带宽几乎相同 (~54 GB/s)，均受限于 LPDDR5X 物理带宽上限。

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

GPU 和 NPU 均支持与 CPU 之间的 UMA 零拷贝数据传输。详见 [`UMA验证总结.md`](UMA验证总结.md)。

## 目录结构

```
├── htp_bandwidth_test/     # HTP (NPU) 带宽测试
├── gpu_bandwidth_test/     # GPU (OpenCL) 带宽测试
├── UMA验证总结.md          # UMA 验证详细报告
└── README.md               # 本文件
```
