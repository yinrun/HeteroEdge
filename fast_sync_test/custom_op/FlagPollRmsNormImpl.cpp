//=============================================================================
//  Custom FlagPoll RMSNorm HTP Op Package - HVX Implementation
//
//  This operator combines flag-based synchronization with RMSNorm computation:
//    1. Spin-poll on a shared memory flag (written by GPU)
//    2. Once flag is detected, perform RMSNorm on the input data
//
//  This allows NPU launch overhead to overlap with GPU execution,
//  eliminating the CPU-side flag polling from the critical path.
//
//  Inputs:  data (FP16 [B,H,W,D]), gamma (FP16 [D]), flag (UINT32 [1,1,1,1])
//  Outputs: result (FP16 [B,H,W,D]), poll_count (UINT32 [1,1,1,1])
//  Params:  epsilon (scalar float)
//
//  RMSNorm: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
//=============================================================================

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_FlagPollRmsNorm);

// Parameter order: epsilon is a scalar float param
DEF_PACKAGE_PARAM_ORDER("FlagPollRmsNorm", "epsilon", false, nullptr)

// Forward declarations
template <typename OutTtype, typename InTtype>
int flagpoll_rmsnorm_fp_impl(OutTtype &out, Tensor &poll_out,
                              const InTtype &in, const InTtype &gamma,
                              const Tensor &flag_tensor, const Tensor &epsilon);

template <typename Ttype>
int flagpoll_rmsnorm_ref_impl(Ttype &out, Tensor &poll_out,
                               const Ttype &in, const Ttype &gamma,
                               const Tensor &flag_tensor, const Tensor &epsilon);

// Register reference (scalar) implementation for generic Tensor type
DEF_PACKAGE_OP((flagpoll_rmsnorm_ref_impl<Tensor>), "FlagPollRmsNorm")

// Register HVX FP16 implementation with FAST cost and HVX resource flag
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (flagpoll_rmsnorm_fp_impl<PlainFloat16Tensor, PlainFloat16Tensor>),
    "FlagPollRmsNorm",
    FAST,
    Flags::RESOURCE_HVX)

// TCM variant
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (flagpoll_rmsnorm_fp_impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>),
    "FlagPollRmsNorm",
    FAST,
    Flags::RESOURCE_HVX)

// Tensor layout: Flat for FP16 tensors
DEF_TENSOR_PROPERTIES(
    Op("FlagPollRmsNorm", "in", "gamma", "flag", "Epsilon"),
    Flat("*", "in", "gamma"))

//=============================================================================
// Cache coherency helpers for Hexagon DSP
//
// dcinva: Data Cache INValidate Address
// Invalidates the cache line containing the given address in Hexagon L2,
// forcing the next read to go to main memory (DRAM/interconnect).
// This is essential for seeing GPU writes to shared ION memory.
//=============================================================================

#if defined(__hexagon__)
// Hexagon DSP: use dcinva to invalidate L2 cache line
static inline void hexagon_dcache_invalidate(volatile void *addr) {
  asm volatile("dcinva(%0)" : : "r"(addr) : "memory");
}

static inline void hexagon_dcache_invalidate_region(volatile void *start, size_t size) {
  const size_t cache_line = 32;  // Hexagon L2 cache line = 32 bytes
  volatile char *p = (volatile char *)start;
  volatile char *end = p + size;
  for (; p < end; p += cache_line) {
    asm volatile("dcinva(%0)" : : "r"(p) : "memory");
  }
}
#else
// Non-Hexagon (aarch64 stub, x86 simulation): just memory barrier
static inline void hexagon_dcache_invalidate(volatile void *addr) {
  (void)addr;
  __sync_synchronize();
}

static inline void hexagon_dcache_invalidate_region(volatile void *start, size_t size) {
  (void)start;
  (void)size;
  __sync_synchronize();
}
#endif

//=============================================================================
// HVX FP16 RMSNorm computation (reused from rmsnorm custom op)
//
// Algorithm per row (d elements):
//   1. Compute sum_sq = sum(x[i]^2) using qf32 accumulation
//   2. Horizontal reduce sum_sq to scalar
//   3. Compute scale = rsqrt(sum_sq / d + epsilon)
//   4. Output: y[i] = x[i] * gamma[i] * scale
//=============================================================================

static void rmsnorm_hvx_row(Float16 *pout, const Float16 *pin, const Float16 *pgamma,
                             float epsilon, int length) {
  union {
    float f;
    int32_t i;
  } ftmp;

  HVX_Vector *iptr = (HVX_Vector *)pin;
  HVX_Vector vzero = Q6_V_vzero();

  // Step 1: Accumulate sum of squares in qf32
  HVX_Vector vsum_lo = Q6_V_vzero();
  HVX_Vector vsum_hi = Q6_V_vzero();

  int d = length;
  HVX_Vector *ptr = iptr;
  for (; d > 63; d -= 64) {
    HVX_Vector x = vmemu(ptr);
    ptr++;
    HVX_VectorPair x_sq = Q6_Wqf32_vmpy_VhfVhf(x, x);
    vsum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_lo, Q6_V_lo_W(x_sq));
    vsum_hi = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_hi, Q6_V_hi_W(x_sq));
  }
  if (d > 0) {
    HVX_Vector x = vmemu(ptr);
    HVX_VectorPred qmask = Q6_Q_vsetq2_R(d * 2);
    x = Q6_V_vmux_QVV(qmask, x, vzero);
    HVX_VectorPair x_sq = Q6_Wqf32_vmpy_VhfVhf(x, x);
    vsum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_lo, Q6_V_lo_W(x_sq));
    vsum_hi = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_hi, Q6_V_hi_W(x_sq));
  }

  HVX_Vector vsum = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_lo, vsum_hi);

  // Step 2: Horizontal reduction
  for (int i = 0, nshift = 4; i < 5; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(vsum, vsum, nshift);
    vsum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }

  HVX_Vector vsf = Q6_Vsf_equals_Vqf32(vsum);
  ftmp.i = Q6_R_vextract_VR(vsf, 0);
  float sum_sq = ftmp.f;

  // Step 3: Compute scale
  float mean_sq = sum_sq / (float)length;
  float scale = 1.0f / sqrtf(mean_sq + epsilon);

  ftmp.f = scale;
  HVX_Vector vscale = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(ftmp.i), vzero);

  // Step 4: Compute y[i] = x[i] * gamma[i] * scale
  HVX_Vector *gptr = (HVX_Vector *)pgamma;
  HVX_Vector *optr = (HVX_Vector *)pout;
  ptr = iptr;
  d = length;

  for (; d > 63; d -= 64) {
    HVX_Vector x = vmemu(ptr);
    ptr++;
    HVX_Vector g = vmemu(gptr);
    gptr++;

    HVX_VectorPair xg = Q6_Wqf32_vmpy_VhfVhf(x, g);
    HVX_Vector lo = Q6_Vqf32_vmpy_Vqf32Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xg), vzero), vscale);
    HVX_Vector hi = Q6_Vqf32_vmpy_Vqf32Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xg), vzero), vscale);

    HVX_VectorPair result = Q6_W_vcombine_VV(hi, lo);
    q6op_vstu_AV(optr, Q6_Vhf_equals_Wqf32(result));
    optr++;
  }

  if (d > 0) {
    HVX_Vector x = vmemu(ptr);
    HVX_Vector g = vmemu(gptr);

    HVX_VectorPair xg = Q6_Wqf32_vmpy_VhfVhf(x, g);
    HVX_Vector lo = Q6_Vqf32_vmpy_Vqf32Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xg), vzero), vscale);
    HVX_Vector hi = Q6_Vqf32_vmpy_Vqf32Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xg), vzero), vscale);

    HVX_VectorPair result = Q6_W_vcombine_VV(hi, lo);
    q6op_vstu_variable_ARV(optr, d * 2, Q6_Vhf_equals_Wqf32(result));
  }
}

//=============================================================================
// HVX FP16 entry point with flag polling
//=============================================================================

// Maximum poll iterations before timeout (safety against deadlock)
static constexpr uint32_t MAX_POLL_ITERS = 10000000;

template <typename OutTtype, typename InTtype>
int flagpoll_rmsnorm_fp_impl(OutTtype &out, Tensor &poll_out,
                              const InTtype &in, const InTtype &gamma,
                              const Tensor &flag_tensor, const Tensor &epsilon) {
  out.set_dims(in);

  auto [b_in, h_in, w_in, d_in] = in.dims();
  float eps = epsilon(0, 0, 0, 0);

  // ── Phase 1: Flag Polling ──
  // Get raw pointer to flag in shared ION memory (written by GPU kernel)
  volatile uint32_t *flag_ptr = (volatile uint32_t *)flag_tensor.raw_data_const();

  uint32_t poll_count = 0;
  while (poll_count < MAX_POLL_ITERS) {
    // Invalidate Hexagon L2 cache line to see GPU's write
    hexagon_dcache_invalidate(flag_ptr);

    // Check if GPU has signaled completion
    if (*flag_ptr != 0) break;

    poll_count++;
  }

  // Record poll count for diagnostics
  poll_out.set_dims(flag_tensor);
  *(uint32_t *)poll_out.raw_data() = poll_count;

  // ── Phase 2: Cache invalidation for input data ──
  // GPU wrote the input data to ION buffer; invalidate our cached copies
  {
    const void *data_start = &in.get_raw(0, 0, 0, 0);
    size_t data_bytes = (size_t)b_in * h_in * w_in * d_in * sizeof(Float16);
    hexagon_dcache_invalidate_region((volatile void *)data_start, data_bytes);
  }

  // ── Phase 3: RMSNorm computation ──
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        using T = typename InTtype::element_type;
        const T *pin = &in.get_raw(b, h, w, 0);
        const T *pgamma = &gamma.get_raw(0, 0, 0, 0);
        typename OutTtype::element_type *pout = &out.get_raw(b, h, w, 0);
        rmsnorm_hvx_row(pout, pin, pgamma, eps, d_in);
      }
    }
  }
  return GraphStatus::Success;
}

//=============================================================================
// Reference (scalar) implementation with flag polling
//=============================================================================

template <typename Ttype>
int flagpoll_rmsnorm_ref_impl(Ttype &out, Tensor &poll_out,
                               const Ttype &in, const Ttype &gamma,
                               const Tensor &flag_tensor, const Tensor &epsilon) {
  out.set_dims(in);

  auto [b_in, h_in, w_in, d_in] = in.dims();
  float eps = epsilon(0, 0, 0, 0);

  // Flag polling (reference: just check the flag, no HVX cache ops)
  volatile uint32_t *flag_ptr = (volatile uint32_t *)flag_tensor.raw_data_const();
  uint32_t poll_count = 0;
  while (*flag_ptr == 0 && poll_count < MAX_POLL_ITERS) {
    poll_count++;
  }

  poll_out.set_dims(flag_tensor);
  poll_out(0, 0, 0, 0) = *(float *)&poll_count;

  // RMSNorm computation
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        float sum_sq = 0.0f;
        for (Idx d = 0; d < d_in; d++) {
          float val = in(b, h, w, d);
          sum_sq += val * val;
        }
        float scale = 1.0f / sqrtf(sum_sq / (float)d_in + eps);
        for (Idx d = 0; d < d_in; d++) {
          out(b, h, w, d) = in(b, h, w, d) * gamma(0, 0, 0, d) * scale;
        }
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_FlagPollRmsNorm);
