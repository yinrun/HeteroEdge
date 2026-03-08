# Qwen3-0.6B 异构推理可行性分析

基于 SM8850 平台实测数据，分析 HeteroInfer (SOSP 2025) 各类 GPU-NPU 异构方案在 Qwen3-0.6B 上的收益，并提出自定义 HTP 同步算子以突破 graphExecute 开销瓶颈。

---

## 1. 平台参数 (SM8850)

### 1.1 算力与带宽

| 加速器 | 算力 (FP16) | 实测带宽 | 备注 |
|--------|------------|---------|------|
| GPU (Adreno 840) | ~1 TFLOPS | 54.79 GB/s | 12 CU |
| NPU (Hexagon V81) | ~10 TFLOPS | 53.70 GB/s | 1 core, 8 HVX threads |
| GPU+NPU 聚合 | — | 76.80 GB/s | 90.6% of 84.8 GB/s LPDDR5X peak |

### 1.2 关键开销参数 (实测)

| 开销项 | 数值 | 来源 |
|--------|------|------|
| graphExecute 固定开销 (大核) | ~200 us | FFN benchmark: (1020-214)/4 = 202 us/call |
| graphExecute 固定开销 (未绑核) | ~280 us | FFN benchmark |
| Context binary cache: 图创建加速 | 148ms → 5.6ms (~26x) | graph_launch_test |
| Context binary cache: 执行无影响 | 214 vs 225 us | graph_launch_test (差异在误差范围) |
| Fast Sync (CPU flag poll) | ~120 us | clFinish 676us → flag poll 292us |
| GPU→CPU flag 延迟 (usleep+poll) | ~120 us | fast_sync_test |
| graphExecuteAsync | **不支持** | HTP V81: error 1000, "Async execution is unsupported" |
| UMA 零拷贝 | 已验证 | ION shared memory, GPU+NPU 同时操作同一 buffer |

---

## 2. Qwen3-0.6B 模型架构

| 参数 | 值 |
|------|-----|
| hidden_size (H) | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads (GQA) | 8 |
| head_dim | 64 |
| intermediate_size (I) | 3072 |
| vocab_size | 151936 |

### 每层权重 (FP16)

| 模块 | 权重矩阵 | 形状 | 大小 |
|------|---------|------|------|
| Attention | W_q | [1024, 1024] | 2.0 MB |
| | W_k | [1024, 512] | 1.0 MB |
| | W_v | [1024, 512] | 1.0 MB |
| | W_o | [1024, 1024] | 2.0 MB |
| | **小计** | | **6.0 MB** |
| FFN (SwiGLU) | W_gate | [1024, 3072] | 6.0 MB |
| | W_up | [1024, 3072] | 6.0 MB |
| | W_down | [3072, 1024] | 6.0 MB |
| | **小计** | | **18.0 MB** |
| **每层总计** | | | **24.0 MB** |
| **28 层总计** | | | **672 MB** |

---

## 3. 各异构方案分析

所有方案均以 Layer 为单位，GPU 做 element-wise ops，NPU 做 MatMul。下文分析每层的开销分布。

### 3.0 基线方案

#### NPU-only (fused, 1 graph/layer) — 最优基线

| 阶段 | 时间/层 | 28 层 | 吞吐 |
|------|---------|-------|------|
| Prefill (seq=512) | 1.40ms compute + 0.20ms launch = **1.60ms** | 44.8ms | — |
| Decode (seq=1) | 0.44ms bandwidth + 0.20ms launch = **0.64ms** | 18.0ms | **55 tok/s** |

#### GPU-only

| 阶段 | 时间/层 | 28 层 | 吞吐 |
|------|---------|-------|------|
| Prefill | ~14ms (1T FLOPS, 14B FLOPs) | 392ms | — |
| Decode | 0.44ms bandwidth + ~0.21ms kernel launch = **0.65ms** | 18.3ms | **55 tok/s** |

---

### 3.1 方案一: Layer-level

**原理**: GPU 做 element-wise，NPU 做 MatMul，每层交替执行 4 次。

**每层流程** (4 次 GPU↔NPU 切换):
```
GPU(RMSNorm) ──sync──▶ NPU(QKV) ──sync──▶ GPU(RoPE+Softmax) ──sync──▶
NPU(AttnOut+O) ──sync──▶ GPU(Residual+RMSNorm) ──sync──▶
NPU(Gate+Up) ──sync──▶ GPU(SiLU×mul) ──sync──▶ NPU(Down) ──sync──▶ GPU(Residual)
```

**每层同步开销**:
```
4 × graphExecute launch:  4 × 200us = 800us
4 × Fast Sync (GPU→NPU): 4 × 120us = 480us
总同步开销:                         = 1,280us
```

**Prefill (seq=512) NPU MatMul FLOPs**:

| Op | FLOPs |
|----|-------|
| Q/K/V projection | 1.074B + 537M + 537M = 2.148B |
| QK^T + Attn×V (16 heads) | 537M + 537M = 1.074B |
| O projection | 1.074B |
| Gate + Up + Down (FFN) | 3.22B × 3 = 9.66B |
| **NPU 合计** | **13.96B** |

**结果**:

| 阶段 | NPU-only | Layer-level | 倍数 |
|------|---------|------------|------|
| Prefill/层 | 1.60ms | 2.68ms (1.40ms NPU + 1.28ms sync) | **0.60x (慢 67%)** |
| Decode/层 | 0.64ms | 1.59ms (0.31ms BW + 1.28ms sync) | **0.40x (慢 2.5x)** |

**结论**: Layer-level 对任何规模的 Qwen3-0.6B 均无收益。

---

### 3.2 方案二: Tensor-level Weight-centric

**原理**: 每个 MatMul 的权重沿 row 维切分，GPU 和 NPU 各算一部分，结果 concat 合并。

```
W_q = [W_q_gpu (top rows)]   GPU: Q_gpu = input × W_q_gpu ─┐
      [W_q_npu (bottom rows)] NPU: Q_npu = input × W_q_npu ─┴─ concat → Q
```

**关键约束**: 中间结果必须合并后才能进入下一个 op（完整的 Q, K, V 才能算 attention）。因此每层仍需 **4 次 NPU graphExecute**，结构与 Layer-level 相同。

**Decode 分析 (50:50 权重切分)**:

```
Step 1: QKV — GPU+NPU 并行加载各 2MB
  GPU: 2MB @ 54 GB/s = 37us | NPU: 37us + 200us launch = 237us
  等待慢者 + sync: max(37, 237) + 120 = 357us

Step 2: O proj — GPU+NPU 各 1MB + GPU attention compute
  = max(25, 218) + 120 = 338us

Step 3: Gate+Up — GPU+NPU 各 6MB
  = max(116, 311) + 120 = 431us

Step 4: Down — GPU+NPU 各 3MB
  = max(60, 255) + 120 = 375us

每层合计: 1,501us
```

**带宽聚合收益 vs 实际开销**:

| 项目 | 数值 |
|------|------|
| 单设备权重加载/层 | 444us (24MB @ 54 GB/s) |
| 聚合后权重加载/层 | 313us (24MB @ 76.8 GB/s) |
| **带宽节省** | **131us** |
| 同步开销增加 (4×launch + 4×sync) | +1,028us |
| **净影响** | **+897us/层 (恶化)** |

Weight-centric 比单设备慢 **2.3x**。带宽收益被同步开销完全抵消（约 8 倍）。

---

### 3.3 方案三: Activation-centric

**原理**: 针对动态 seq_len。将 activation 沿 seq_len 维切分，标准尺寸部分走 NPU 预编译图，剩余走 GPU。

```
input [seq=300, H] → [256, H] 走 NPU (预编译图) | [44, H] 走 GPU (OpenCL)
```

**对 Qwen3-0.6B 的适用性**:
- 主要解决 prefill 阶段 NPU 静态图与动态 seq_len 的矛盾
- 对 decode (seq_len=1) **完全不适用**
- 对 prefill: 直接使用 NPU-only + padding 到标准尺寸（256/512/1024）更简单
- Qwen3-0.6B prefill 本身已经很快（NPU-only 44.8ms for seq=512），padding 开销微不足道

**结论**: 对小模型意义不大。

---

### 3.4 方案四: Hybrid Partition

**原理**: 组合 activation-centric + weight-centric + padding，由离线 solver 按 tensor shape 自动选择。

**对 Qwen3-0.6B**: 最大权重矩阵为 [1024, 3072]，仅 6MB。论文 solver 的 shape-sensitive 优化针对的是 [4096, 14336]（112MB）这量级的矩阵。小权重矩阵的切分同步开销远超计算收益，Hybrid 方案退化为与 Weight-centric 相同的结论。

---

### 3.5 各方案汇总

| 方案 | Decode/层 | Decode 28层 | tok/s | vs NPU-only |
|------|-----------|------------|-------|-------------|
| **NPU-only fused** | 640 us | 18.0 ms | **55** | 1.00x |
| **GPU-only** | 650 us | 18.3 ms | **55** | 1.00x |
| Weight-centric (现有开销) | 1,501 us | 42.0 ms | 24 | 0.44x |
| Layer-level | 1,593 us | 44.6 ms | 22 | 0.40x |

**根本原因**:

```
带宽聚合收益:  131us/层  (24MB/54 → 24MB/76.8 GB/s)
同步开销下限:  320us/层  (2×200us launch + 2×120us sync, 最少 2 次切换)
收益/开销比:   0.41x     → 永远无法回本
```

---

## 4. 与 HeteroInfer 论文的规模差异

### 论文评测模型 vs Qwen3-0.6B

| 模型 | 参数量 | 每层权重 (FP16) | vs Qwen3-0.6B |
|------|--------|----------------|---------------|
| InternLM-1.8B | 1.8B | ~80 MB | **3.3x** |
| Llama-3B | 3B | ~160 MB | 6.7x |
| Llama-7B | 7B | ~300 MB | 12.5x |
| Llama-8B | 8B | ~300 MB | 12.5x |
| **Qwen3-0.6B** | **0.6B** | **24 MB** | **1x (基准)** |

论文最小评测模型 InternLM-1.8B 每层权重是 Qwen3-0.6B 的 3.3 倍，论文有意避开了小模型。

### 收益拐点估算

Weight-centric 方案的盈亏平衡条件：**带宽聚合节省 > 最小同步开销**

```
W/54 - W/76.8 > 320us
W × (1/54 - 1/76.8) / 1000 > 0.32   (W 单位 MB, 带宽 GB/s)
W × 5.43×10⁻³ > 0.32
W > 59 MB   (2×launch + 2×sync, 最少切换方案)

若 4 次切换 (1,280us overhead):
W > 236 MB   → 对应 hidden_dim ≈ 8192+, LLaMA-13B 级别
```

即使是论文的最小模型 InternLM-1.8B（~80MB/层），也仅在最少切换时勉强过线。

---

## 5. 自定义 HTP 同步算子：突破 graphExecute 瓶颈

### 5.1 问题根源

当前所有异构方案失败的根本原因是 **FastRPC 开销（~200us）**，而非计算或带宽限制：

```
现有方案 (每次 GPU→NPU 切换):
  ARM CPU ──FastRPC──▶ Hexagon DSP   // ~200us 不可压缩
  CPU 轮询 GPU flag                  // ~120us
  共: ~320us per switch point
```

### 5.2 方案：将 flag polling 移至 NPU 图内部

通过自定义 HVX 算子，将 flag 轮询/写入嵌入 QNN 图，使全层 NPU 工作在 **一次 graphExecute** 内完成。

```
当前（4 次 graphExecute）:
  CPU: execute(QKV) → wait → execute(O) → wait → execute(GateUp) → wait → execute(Down)
  ├── 4 × 200us FastRPC = 800us
  └── 4 × 120us CPU sync = 480us

自定义同步算子（1 次 graphExecute）:
  NPU 图: [WaitFlag₁ → MatMul_QKV → SetFlag₁ → WaitFlag₂ → MatMul_O → SetFlag₂
           → WaitFlag₃ → MatMul_GateUp → SetFlag₃ → WaitFlag₄ → MatMul_Down → SetFlag₄]
  ├── 1 × 200us FastRPC = 200us
  └── 4 × ~10us DSP flag poll = ~40us
```

### 5.3 自定义算子设计

```c
// WaitFlag HVX op: 轮询共享 ION buffer 中的 flag
void wait_flag_impl(const volatile uint32_t* flag) {
    // Hexagon L2 cache invalidate + 轮询
    while (1) {
        Q6_dcfetch_A(flag);           // 失效 cache line，强制从 DRAM 读
        if (*flag != 0) break;
    }
    *flag = 0;  // 清除，复用
}

// SetFlag HVX op: 写完成标志
void set_flag_impl(volatile uint32_t* flag) {
    asm volatile("barrier" ::: "memory");  // Hexagon memory barrier
    *flag = 1;
}
```

**已有基础**: 项目中已有 `FlagPollRmsNorm` 自定义 HVX op（`fast_sync_test/custom_op/`），将 flag polling 与 RMSNorm 融合为一个算子，验证了该路径的可行性。

### 5.4 NPU-side flag polling 的性能预期

**`fast_sync_test/NPU_POLL_DESIGN.md` 描述的优化目标**:

```
当前 Fast Sync:
  GPU submit → GPU执行 → 写flag → CPU轮询(~120us) → 启动NPU(200us) → NPU执行
  关键路径: GPU_exec + 120us + 200us

NPU Poll:
  GPU submit ──── GPU执行 ──── 写flag ─┐
  NPU launch ──── RPC dispatch ──────── HVX spin-wait → 计算
  关键路径: max(GPU_exec, NPU_dispatch) + HVX_compute
  预期: 301us → ~120-180us (NPU dispatch 与 GPU 并行)
```

### 5.5 扩展到异构推理：Decode 性能预测

以 weight-centric 50:50 split + 全层 1 graphExecute 为例：

**每层开销**:

| 项目 | 数值 |
|------|------|
| graphExecute launch (一次) | 200us |
| NPU-side flag poll × 4 | 4 × ~10us = 40us |
| GPU-side flag poll × 4 | 4 × ~10us = 40us |
| Weight 加载 (GPU+NPU 并行) | 24MB / 76.8 GB/s = 313us |
| **每层合计** | **~593us** |

| 方案 | 每层 | 28 层 | tok/s | vs 单设备 |
|------|------|-------|-------|---------|
| NPU-only fused (基线) | 640us | 18.0ms | 55 | 1.00x |
| **Weight-centric + 自定义同步** | **593us** | **16.6ms** | **60** | **1.09x** |
| 4 层融合 (4层→1 graphExecute) | 403us | 11.3ms | 88 | 1.60x |
| 全层融合 (28层→1 graphExecute) | 360us | 10.1ms | **99** | **1.80x** |

> 多层融合将 200us launch 摊薄到多层，理论上限 ~100-114 tok/s（受限于 76.8 GB/s 聚合带宽）。

### 5.6 关键技术风险

| 风险 | 严重性 | 分析 |
|------|--------|------|
| **DSP↔GPU 缓存一致性** | **高** | Hexagon L2 与 GPU 可能不在同一 ACE 一致性域。需用 `dcinva` 指令失效 cache line，或使用 uncached ION 映射 |
| HTP watchdog 超时 | 中 | 长时间 spin 可能触发 DSP watchdog（通常数百 ms）。实际等待时间 <1ms，风险低 |
| 多层图规模限制 | 中 | 28 层融合时图需引用 672MB 权重，VTCM 和图元数据可能受限。建议先测 4-8 层 |
| 自定义 op 与 QNN MatMul 混用 | 低 | 已有 FlagPollRmsNorm 验证了 custom HVX + QNN native ops 混用可行 |

### 5.7 验证优先级

1. **DSP cache coherency 验证** (最高优先级): CPU 写 flag=1 → graphExecute → 验证 HVX 能读到 flag
2. **WaitFlag + SetFlag 独立测试**: 测量 flag polling 延迟（目标 < 20us）
3. **单层 pipeline**: GPU element-wise + NPU 全层 MatMuls（含 4 对 flag ops），测量实际 decode 延迟
4. **多层融合**: 测试 4/8/28 层融合图的创建时间和执行性能

---

## 6. 结论

### 对 Qwen3-0.6B 的建议策略

| 策略 | 实现 | 预期收益 |
|------|------|---------|
| **NPU-only fused** (立即可用) | 1 graphExecute/layer，全层融合 | 55 tok/s，最稳定 |
| + Context binary cache | 加速启动 148ms→5.6ms | 首 token 延迟显著降低 |
| + Big core pinning | 200us→170us graphExecute | ~15% decode 加速 |
| **自定义同步算子** (实验) | WaitFlag/SetFlag HVX op，1 graphExecute 覆盖多层 | 60-99 tok/s (+9~80%) |

### 何时 HeteroInfer 的标准方案才有效

在当前 SM8850 平台开销参数下：

| 条件 | 阈值 | 对应模型规模 |
|------|------|------------|
| 每层权重 (最少 2 次切换) | > 59 MB | ~2B+ 参数 |
| 每层权重 (4 次切换) | > 236 MB | ~13B+ 参数 |
| 论文最小评测模型 (InternLM-1.8B) | ~80 MB/层 | 勉强达到 2 次切换阈值 |

---

## 附: 数据来源

| 数据 | 来源文件 |
|------|----------|
| GPU/NPU 带宽 | [`README.md`](../README.md) |
| graphExecute 开销 (~202us) | [`graph_launch_test/README.md`](../graph_launch_test/README.md) |
| Context binary cache 加速 (~26x) | [`graph_launch_test/README.md`](../graph_launch_test/README.md) |
| Fast Sync (676→292us) | [`fast_sync_test/README.md`](../fast_sync_test/README.md) |
| graphExecuteAsync 不支持 | [`graph_launch_test/README.md`](../graph_launch_test/README.md) Section 3 |
| NPU Poll 设计方案 | [`fast_sync_test/NPU_POLL_DESIGN.md`](../fast_sync_test/NPU_POLL_DESIGN.md) |
| HeteroInfer 论文 | Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference (SOSP 2025) |
