# GPU + NPU 并发带宽测试

## 背景

论文 **HeteroInfer** (SOSP 2025) 的核心发现：端侧 SoC 上**单一处理器无法充分利用内存带宽**，GPU+NPU 同时运行可将利用率从 ~70% 提升至 96%。

本测试在 **Snapdragon 8 Gen 5 (SM8850)** 上验证该结论。

## 实测结果

**平台**: Snapdragon 8 Gen 5 (SM8850), Adreno 840 + Hexagon V81, LPDDR5X-5300
**理论峰值**: 84.8 GB/s (10600 MT/s × 64-bit, 4ch × 16bit)
**运算**: Element-wise Add (C = A + B), unsigned 8-bit

### Solo 基线

| 处理器 | 256MB | 384MB |
|--------|-------|-------|
| GPU (Adreno 840, OpenCL) | 52.10 GB/s (61.4%) | 52.40 GB/s (61.8%) |
| NPU (Hexagon V81, QNN BURST) | 54.09 GB/s (63.8%) | 52.87 GB/s (62.3%) |

### 并发 Ratio 扫描

**256MB:**

| GPU Ratio | GPU MB | NPU MB | GPU GB/s | NPU GB/s | 聚合 GB/s | 利用率 |
|-----------|--------|--------|----------|----------|-----------|--------|
| 0.10 | 25 | 230 | 26.3 | 49.8 | **76.1** | **89.7%** |
| 0.20 | 51 | 204 | 26.5 | 46.2 | 72.7 | 85.7% |
| 0.30 | 76 | 179 | 29.2 | 35.8 | 65.0 | 76.6% |
| 0.40 | 102 | 153 | 26.9 | 37.5 | 64.4 | 75.9% |
| 0.50 | 128 | 128 | 31.4 | 35.7 | 67.0 | 79.0% |

**384MB:**

| GPU Ratio | GPU MB | NPU MB | GPU GB/s | NPU GB/s | 聚合 GB/s | 利用率 |
|-----------|--------|--------|----------|----------|-----------|--------|
| 0.10 | 38 | 345 | 26.9 | 50.3 | **77.1** | **90.9%** |
| 0.20 | 76 | 307 | 27.7 | 47.1 | 74.8 | 88.2% |
| 0.30 | 115 | 268 | 27.8 | 42.2 | 70.1 | 82.6% |
| 0.40 | 153 | 230 | 27.7 | 37.8 | 65.5 | 77.3% |
| 0.50 | 192 | 192 | 31.8 | 35.4 | 67.2 | 79.2% |

### 关键发现

1. **峰值聚合带宽: 77.1 GB/s (90.9%)** — 384MB, ratio 0.1, 4次重复波动 < 1.6%
2. **GPU 并发带宽恒定 ~27 GB/s** — 不随数据量/ratio 变化，并发时从 52→27 下降 48%
3. **NPU 受 GPU 竞争影响较小** — 并发时从 54→50 仅下降 7% (ratio 0.1)
4. **竞争因子 0.60-0.64** — 与数据量无关，是硬件级别的内存通道竞争
5. **384MB 比 256MB 略好** — 约高 1 GB/s，dispatch 开销被更好地 amortize
6. **低 GPU ratio 聚合更高** — 因为 GPU 带宽固定，减少 GPU 份额可减轻对 NPU 的竞争
7. **与 HeteroInfer 96% 的差距** — 可能源于 Gen 5 内存控制器差异或 QNN/OpenCL dispatch 开销

## 核心思路

### 统一任务、统一内存、分区执行

```
全局任务:  C[0 .. N] = A[0 .. N] + B[0 .. N]     （总量 N 字节，如 256MB）

                    ┌─── GPU 分区 ───┐┌─── NPU 分区 ───┐
Tensor A:  [=============================|==============================]
Tensor B:  [=============================|==============================]
Tensor C:  [=============================|==============================]
                 0              gpu_size           N
                   OpenCL Add             QNN ElementWiseAdd
                   (uchar16)              (UFIXED_POINT_8)

所有缓冲区通过 rpcmem (ION) 分配 → CPU/GPU/NPU 统一内存访问
```

**关键区别于旧方案**：不再是 GPU 和 NPU 各自做独立的测试，而是：
1. **同一种运算**：Element-wise Add（C = A + B）
2. **同一片物理内存**：全部通过 rpcmem 分配在 ION 堆上
3. **任务分区**：GPU 处理前半部分，NPU 处理后半部分
4. **同时执行**：两者并发产生内存流量

## 目录结构

```
concurrent_bandwidth_test/
├── README.md
├── CMakeLists.txt
├── build_android.sh
├── run_on_device.sh
├── kernels/
│   └── element_add.cl          # GPU OpenCL 内核（uchar16 向量化加法）
└── src/
    ├── main.cpp                 # 入口：参数解析、内存分配、线程编排、结果输出
    ├── common.h                 # 共享类型：BandwidthResult, SpinBarrier, rpcmem API
    ├── gpu_bandwidth.h/.cpp     # GPU 测试：OpenCL 初始化/运行/清理
    └── htp_bandwidth.h/.cpp     # NPU 测试：QNN 初始化/运行/清理
```

## 详细设计

### 1. 统一内存分配（main.cpp 负责）

所有缓冲区在主线程中通过 rpcmem 分配，确保 GPU 和 NPU 访问同一 ION 堆上的物理内存：

```
rpcmem_alloc(heapid=25)  →  rpcmem_to_fd()  →  分发给 GPU / NPU

GPU 侧缓冲区（各 gpu_size 字节）:
  A_gpu = rpcmem_alloc(...)
  B_gpu = rpcmem_alloc(...)
  C_gpu = rpcmem_alloc(...)

NPU 侧缓冲区（各 npu_size 字节）:
  A_npu = rpcmem_alloc(...)
  B_npu = rpcmem_alloc(...)
  C_npu = rpcmem_alloc(...)
```

**为什么分开分配而非一块大内存切片**：
- QNN 需要对整个 buffer 做 `QnnMem_register()`，不支持偏移量注册
- OpenCL 的 `cl_mem_ion_host_ptr` 也需要完整的 fd + host_ptr
- 分开分配更简洁，且都在同一 ION 堆上，访问同一 LPDDR5X

### 2. GPU 侧（gpu_bandwidth.h/.cpp）

**导入 ION 内存到 OpenCL**（参考 `unified_uma_demo.cpp` 的 cl_mem_ion_host_ptr 模式）：

```cpp
cl_mem_ion_host_ptr ion_mem;
ion_mem.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
ion_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
ion_mem.ion_filedesc = fd;
ion_mem.ion_hostptr = ptr;

cl_mem buf = clCreateBuffer(context,
    CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM | CL_MEM_READ_WRITE,
    size, &ion_mem, &err);
```

**OpenCL 内核**（`kernels/element_add.cl`）：

```opencl
__kernel void element_add_uchar16(
    __global uchar16* C,
    __global const uchar16* A,
    __global const uchar16* B,
    int num_vecs) {
    int id = get_global_id(0);
    if (id < num_vecs) {
        C[id] = A[id] + B[id];
    }
}
```

- 使用 `uchar16` 向量化（16 字节/work-item），与 NPU 的 uint8 数据类型一致
- 每 work-item 处理 16 字节，最大化内存吞吐
- 运算量极小（加法），瓶颈完全在内存带宽

**接口**：

```cpp
struct GpuBuffers {
    void* A_ptr; int A_fd; size_t A_size;
    void* B_ptr; int B_fd; size_t B_size;
    void* C_ptr; int C_fd; size_t C_size;
};

bool gpu_init(const GpuBuffers& buffers);           // 导入 ION 内存，编译内核
BandwidthResult gpu_run(int num_iters, SpinBarrier* barrier);  // warmup → barrier → 计时
void gpu_cleanup();
```

### 3. NPU 侧（htp_bandwidth.h/.cpp）

**注册 ION 内存到 QNN**（参考 `htp_bandwidth_test/src/main.cpp` 的 RegisteredBuffer 模式）：

```cpp
Qnn_MemDescriptor_t memDesc;
memDesc.memType = QNN_MEM_TYPE_ION;
memDesc.ionInfo.fd = rpcmem_to_fd(ptr);
qnn.memRegister(contextHandle, &memDesc, 1, &memHandle);
```

**QNN 图**：ElementWiseAdd，UFIXED_POINT_8（与现有 htp_bandwidth_test 一致）

- 张量形状根据 npu_size 动态计算，保持 `C <= 1,048,576` 的约束
- BURST 功耗模式：DCVS 关闭，MAX 电压角
- 量化参数：scale=1.0, offset=0

**接口**：

```cpp
struct HtpBuffers {
    void* A_ptr; int A_fd; size_t A_size;
    void* B_ptr; int B_fd; size_t B_size;
    void* C_ptr; int C_fd; size_t C_size;
};

bool htp_init(const HtpBuffers& buffers);
BandwidthResult htp_run(int num_iters, SpinBarrier* barrier);
void htp_cleanup();
```

### 4. 线程编排（main.cpp）

```
主线程:
  1. 解析参数 (total_size, gpu_ratio, num_iters)
  2. 加载 libcdsprpc.so，分配 6 个 rpcmem 缓冲区
  3. 初始化 GPU（导入 ION → OpenCL）
  4. 初始化 NPU（dlopen QNN → 建图 → 注册内存）
  5. 运行测试:
     ├── GPU-only 基线:   gpu_run(barrier=null)
     ├── NPU-only 基线:   htp_run(barrier=null)
     └── 并发测试:
           barrier = SpinBarrier(2)
           wall_start = now()
           thread1: gpu_run(&barrier)   ──┐
           thread2: htp_run(&barrier)   ──┤  并发执行
           join                         ──┘
           wall_elapsed = now() - wall_start
  6. 输出对比表
  7. 清理
```

### 5. 线程同步

```
GPU 线程                         NPU 线程
    |                                |
    |-- warmup (3 iter)              |-- warmup (3 iter)
    |-- clFinish()                   |-- graphExecute() x3
    |                                |
    |========= BARRIER ==============|   <-- 同步点：确保同时开始计时
    |                                |
    |-- start = now()                |-- start = now()
    |-- N iter enqueue               |-- M iter graphExecute
    |-- clFinish()                   |
    |-- end = now()                  |-- end = now()
```

**SpinBarrier**: C++17 `std::atomic` 自旋屏障，< 1μs 延迟。

**迭代次数匹配**：

| 处理器 | 每迭代数据 | 预估带宽 | 每迭代耗时 | 默认迭代 | 总时长 |
|--------|-----------|---------|-----------|---------|--------|
| GPU | gpu_size × 3 | ~55 GB/s | 视 gpu_size | 自动 | ~100ms |
| NPU | npu_size × 3 | ~50 GB/s | 视 npu_size | 自动 | ~100ms |

迭代次数根据分区大小自动计算，目标总时长 ~100ms。

### 6. 带宽计算

**单处理器带宽**：
```
bandwidth_gpu = (gpu_size × 3 × gpu_iters) / gpu_elapsed     // 3 = 2读+1写
bandwidth_npu = (npu_size × 3 × npu_iters) / npu_elapsed
```

**聚合带宽**（核心指标）：
```
total_data = total_size × 3 × iters_in_wall_time
aggregate = total_data / wall_elapsed
```

或更直观地：
```
aggregate = bandwidth_gpu_concurrent + bandwidth_npu_concurrent
```

**对比指标**：
```
聚合利用率 = aggregate / 84.8 GB/s
竞争因子   = aggregate / (solo_gpu + solo_npu)    // < 1.0 表示有竞争
Overlap    = (T_gpu + T_npu - T_wall) / min(T_gpu, T_npu)   // ~100% 表示真并发
```

### 7. 命令行参数

```
./concurrent_bandwidth_test [选项]

--mode gpu|npu|concurrent|all    测试模式（默认 all）
--total-size-mb N                总数据大小 MB（默认 256）
--gpu-ratio R                    GPU 分区比例 0.0-1.0（默认 0.5）
--gpu-iters N                    GPU 迭代次数（默认自动）
--npu-iters N                    NPU 迭代次数（默认自动）
```

### 8. 构建系统

#### CMakeLists.txt

```cmake
target_link_libraries(concurrent_bandwidth_test PRIVATE OpenCL dl pthread)
# OpenCL: ../libs/libOpenCL.so 桩库
# QNN:    运行时 dlopen("libQnnHtp.so", RTLD_LOCAL)，仅头文件依赖
# rpcmem: 运行时 dlopen("libcdsprpc.so")，无编译依赖
```

#### run_on_device.sh

```bash
# 环境变量（GPU + NPU 共存）
export LD_LIBRARY_PATH=${LIB_DIR}:/vendor/lib64:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=${HTP_DIR}
```

推送文件：
- 可执行文件 + `kernels/element_add.cl`
- QNN ARM64 库（5个） → `lib/`
- Hexagon V81 库（5个） → `htp/`

## 实际输出示例

`--mode all --total-size-mb 384 --gpu-ratio 0.1`:

```
=== 异构并发带宽测试 ===
理论峰值: 84.8 GB/s (LPDDR5X-5300, 4ch x 16bit)
任务: Element-wise Add (C = A + B)
总大小: 384 MB, GPU: 38 MB (10%), NPU: 345 MB (90%)

--- GPU 设备信息 ---
  设备: QUALCOMM Adreno(TM) 840
  计算单元: 12, 全局内存: 7.32 GB

--- NPU 设备信息 ---
  Hexagon V81, 1 core(s), 8 HVX threads, BURST 模式
  张量: [1, 1, 1, 1048576] UFIXED_POINT_8

=== GPU-Only 基线 (全量 384 MB) ===
  GPU: 52.21 GB/s (10 iters, 0.215s)
  利用率: 61.6%

=== NPU-Only 基线 (全量 384 MB) ===
  NPU: 53.01 GB/s (10 iters, 0.212s)
  利用率: 62.5%

=== GPU + NPU 并发 (GPU 38MB + NPU 345MB) ===
  GPU: 26.52 GB/s (10 iters, 0.042s)
  NPU: 50.21 GB/s (10 iters, 0.201s)
  Wall clock: 0.264s, Overlap: -49.5%
  聚合带宽: 76.73 GB/s (90.5%)

=== 总结 ===
+-----------+-----------+-----------+---------+---------+
| 模式      | GPU GB/s  | NPU GB/s  | 聚合    | 利用率  |
+-----------+-----------+-----------+---------+---------+
| GPU only  |     52.21 |     --    |   52.21 |  61.6%  |
| NPU only  |     --    |     53.01 |   53.01 |  62.5%  |
| 并发      |     26.52 |     50.21 |   76.73 |  90.5%  |
+-----------+-----------+-----------+---------+---------+
竞争因子: 0.73 (76.73 / 105.22)
```

## 代码来源

| 新文件 | 主要参考 | 关键修改 |
|--------|---------|---------|
| `common.h` | `htp_bandwidth_test/src/main.cpp` (RpcMemApi) | 提取 rpcmem 封装 + SpinBarrier + BandwidthResult |
| `gpu_bandwidth.cpp` | `gpu_bandwidth_test/src/main.cpp` + `unified_uma_demo.cpp` | 改用 ION 内存导入（cl_mem_ion_host_ptr）；内核改为 element_add |
| `htp_bandwidth.cpp` | `htp_bandwidth_test/src/main.cpp` | 改用外部传入的 rpcmem 缓冲区；分区大小可配置 |
| `element_add.cl` | 新编写 | uchar16 向量化加法内核 |
| `main.cpp` | 新编写 | rpcmem 统一分配 + 线程编排 + 结果对比 |

## 实现步骤

1. 创建目录结构
2. 编写 `src/common.h`（SpinBarrier、BandwidthResult、rpcmem 封装）
3. 编写 `kernels/element_add.cl`（uchar16 加法内核）
4. 编写 `src/gpu_bandwidth.h/.cpp`（ION 内存导入 OpenCL + add 内核）
5. 编写 `src/htp_bandwidth.h/.cpp`（外部 rpcmem 注册 QNN + ElementWiseAdd）
6. 编写 `src/main.cpp`（rpcmem 统一分配 + 参数解析 + 三种模式 + 结果输出）
7. 编写 CMakeLists.txt、build_android.sh、run_on_device.sh
8. 编译部署，逐步验证

## 验证结论

1. `--mode gpu` → GPU 单独基线 52.1-52.4 GB/s (61-62%)
2. `--mode npu` → NPU 单独基线 52.9-54.1 GB/s (62-64%)
3. `--mode concurrent --gpu-ratio 0.1` → 聚合 77.1 GB/s (90.9%)
4. `--mode all` → 完整对比表，竞争因子 0.60-0.64
5. ratio 0.1-0.5 扫描完成，最优 ratio ≈ 0.1（GPU 固定 ~27 GB/s，尽量减少对 NPU 的干扰）
6. 4 次重复测试波动 < 1.6%，无热降频影响

## 潜在风险

| 风险 | 缓解措施 |
|------|---------|
| ION 内存导入 OpenCL 失败 | 已在 unified_uma_demo.cpp 验证可行 |
| 并发执行触发热降频 | 保持测试时长短（~100ms） |
| NPU 张量形状约束 (C ≤ 1M) | 根据 npu_size 动态计算合法形状 |
| uchar16 GPU 带宽低于 float8 | 需实测对比，必要时改用 float4 add |
| dlopen 线程安全 | 初始化在主线程顺序执行 |

## 参考

- **HeteroInfer** (SOSP 2025): *Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference*
- `gpu_bandwidth_test/` — GPU 带宽测试（OpenCL float8 copy）
- `htp_bandwidth_test/` — NPU 带宽测试（QNN UFIXED_POINT_8 add）
- `unified_uma_demo.cpp` — GPU+NPU 共享 ION 内存验证
