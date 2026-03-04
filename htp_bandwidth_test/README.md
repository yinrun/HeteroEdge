# HTP Memory Bandwidth Test

通过 QNN HTP 后端执行 ElementWiseAdd 测量 Hexagon Tensor Processor 的最大内存带宽。

- **目标平台**: Snapdragon 8 Gen 5 (Hexagon V81 / SM8850)
- **测试方法**: 两个 128MB 输入张量 (UFIXED_POINT_8) 逐元素相加，计算有效带宽 (2读 + 1写 = 3x data)
- **测试结果**: **53.70 GB/s** (单节点, 3次平均, BURST mode, SDK 2.42)

## 测试结果

| 指标 | 值 |
|------|-----|
| 峰值带宽 | **53.70 GB/s** (单节点, 3次平均) |
| 数据量 | 128 MB (单节点) |
| 有效传输 | 384 MB/次 (2读 + 1写) |
| 每次延迟 | ~6.98 ms |
| 旧版基线 | 6.42 GB/s |
| 提升倍数 | **8.4x** |

## 关键优化点

### 1. Power Config 正确配置 (6.4 → 34.5 GB/s, 5.4x)

```cpp
// contextId 必须绑定到 createPowerConfigId 返回的 ID
dcvsConfig.dcvsV3Config.contextId = powerConfigId;  // 不能用 0
// 禁用 DCVS，锁定最高频率
dcvsConfig.dcvsV3Config.dcvsEnable = 0;
// 不要在执行前调用 destroyPowerConfigId
```

**根因**: `contextId=0` 导致 DCVS 配置未绑定到 power client，`destroyPowerConfigId` 立即销毁配置使其失效，HTP 以默认低功耗模式运行。

### 2. 使用 UFIXED_POINT_8 数据类型 (34.5 → 53 GB/s, +45%)

```cpp
// INT_8 会导致 HTP 插入隐式 Cast 节点，增加额外内存读写
// UFIXED_POINT_8 + 显式量化参数是 HTP 原生类型，无需转换
tensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
tensor.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
tensor.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
```

**根因**: INT_8 不是 HTP 的原生量化类型，graph compiler 会自动插入 Cast 节点做类型转换，额外的读写开销拉低带宽。

### 3. 单节点替代多 tile (简化结构, 带宽不变)

原方案用 4 个独立 Add 节点 (4 tiles x 32MB, dims `[32,1,1,1M]`)，需要 8 input + 4 output tensor。
现改为单个 Add 节点 (dims `[128,1,1,1M]`)，只需 2 input + 1 output tensor，带宽从 53.17 → 53.70 GB/s。

**关键发现: C 维度 (channels) 必须 <= 1048576**

| batch | C | 总量 | 带宽 | 说明 |
|-------|---|------|------|------|
| 128 | 1M | 128MB | **54.16 GB/s** | 最优 |
| 64 | 1M | 64MB | **51.86 GB/s** | C=1M 即可 |
| 64 | 2M | 128MB | 29.53 GB/s | C 翻倍，带宽腰斩 |
| 32 | 4M | 128MB | 1.37 GB/s | C 太大，tiler 无法有效切分 |
| 4-tile 基线 | 1M | 128MB | 53.17 GB/s | 旧方案参考 |

**根因**: HTP tiler 在 C 维度上做切分调度。C=1M 时 tiler 能生成高效的执行计划；C>1M 触发 "no valid splitting rule" 后回退到低效策略，严重影响 HVX 并行度。batch 维度本身不是瓶颈，只需保证 C<=1M。

## 目录结构

- `src/main.cpp`: 直接用 QNN C API 构建图并执行带宽测试
- `build_android.sh`: NDK 交叉编译脚本
- `run_on_device.sh`: 推送依赖并在设备上运行

## 前置条件

- Android NDK (通过 ANDROID_SDK_ROOT 自动检测)
- QNN SDK 2.42+ (`QNN_SDK_ROOT` 环境变量)
- 设备已连接，`adb devices` 可见

## 编译与运行

```bash
cd htp_bandwidth_test
export ANDROID_SDK_ROOT=/path/to/Android/Sdk
export QNN_SDK_ROOT=/path/to/qairt/2.42.0.251225
./build_android.sh
./run_on_device.sh
```

输出示例:

```
[QNN] HTP Add demo starting...
[QNN] Using provider API 2.32.0
[QNN] HTP num_cores=1
[QNN] HTP num_hvx_threads=8
[QNN] Single-node dims: [128, 1, 1, 1048576]  totalBytes=134217728
[QNN] graphFinalize succeeded
  run 1: 6.93 ms
  run 2: 7.07 ms
  run 3: 6.94 ms
[QNN] Done. runs=3 max_error=0 sample_out=3 avg_ms=6.98 bandwidth_GBps=53.70 layout=[128,1,1,1048576]
```

## 技术细节

- 张量布局: NHWC `[128, 1, 1, 1048576]`，单节点 128MB (C<=1M 是关键)
- 数据类型: `UFIXED_POINT_8` (scale=1.0, offset=0)，HTP 原生量化类型
- 内存分配: `rpcmem_alloc` (heapid=25) + `QnnMem_register` (ION zero-copy)
- 性能模式: DCVS 禁用, MAX 电压角 (bus+core), HMX V2 MAX, sleep 禁用
- 测量方式: 3 次 warmup + 3 次计时取平均
- 运算类型: `QNN_OP_ELEMENT_WISE_ADD`
