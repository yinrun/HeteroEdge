//=============================================================================
//  Custom RMSNorm HTP Op Package - HVX Implementation
//
//  RMSNorm: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
//         = x[i] * rsqrt(sum(x^2)/N + eps) * gamma[i]
//
//  Uses HVX FP16 (qf16/qf32) intrinsics for vectorized computation.
//  Each HVX vector = 128 bytes = 64 FP16 elements.
//=============================================================================

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_RmsNorm);

// Define parameter order: epsilon is a scalar float param
DEF_PACKAGE_PARAM_ORDER("RmsNorm", "epsilon", false, nullptr)

// Forward declarations
template <typename OutTtype, typename InTtype>
int rmsnorm_fp_impl(OutTtype &out, const InTtype &in, const InTtype &gamma, const Tensor &epsilon);

template <typename Ttype>
int rmsnorm_ref_impl(Ttype &out, const Ttype &in, const Ttype &gamma, const Tensor &epsilon);

// Register reference (scalar) implementation for generic Tensor type
DEF_PACKAGE_OP((rmsnorm_ref_impl<Tensor>), "RmsNorm")

// Register HVX FP16 implementation with FAST cost and HVX resource flag
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((rmsnorm_fp_impl<PlainFloat16Tensor, PlainFloat16Tensor>),
                                  "RmsNorm",
                                  FAST,
                                  Flags::RESOURCE_HVX)

// TCM (Tightly Coupled Memory) variant for better performance
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((rmsnorm_fp_impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>),
                                  "RmsNorm",
                                  FAST,
                                  Flags::RESOURCE_HVX)

// Tensor layout: Flat for FP16 tensors
DEF_TENSOR_PROPERTIES(Op("RmsNorm", "in", "gamma", "Epsilon"), Flat("*", "in", "gamma"))

// Optimization: Cast FP32 inputs to FP16 when relaxed precision is enabled
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
    GRAPH_CLEANUP,
    relaxed_precision_flag,
    Op("RmsNorm", "In", "Gamma", "Epsilon"),
    AND(EQ(DTYPE_OF("In"), DType::Float32), EQ(DTYPE_OF("*"), DType::Float32)),
    WITH_OUTPUT_TYPE(
        DType::Float32,
        0,
        1.0f,
        Op(FROM_DEFAULT_PACKAGE("Cast"),
           WITH_SIZE("*",
                     WITH_OUTPUT_TYPE(DType::Float16,
                                      0,
                                      1.0f,
                                      Op("RmsNorm",
                                         WITH_SIZE("In", Op(FROM_DEFAULT_PACKAGE("Cast"), "In")),
                                         WITH_SIZE("Gamma", Op(FROM_DEFAULT_PACKAGE("Cast"), "Gamma")),
                                         "Epsilon"))))))

// QNN default: if epsilon not provided, use 1e-5
DEF_PACKAGE_OPTIMIZATION(QNN,
                         Op("RmsNorm", "In", "Gamma"),
                         OK,
                         Op("RmsNorm", "In", "Gamma", gen_ConstScalar_f32(1e-5f)))

//=============================================================================
// HVX FP16 RMSNorm implementation
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

  // ---- Step 1: Accumulate sum of squares in qf32 ----
  // We accumulate lo and hi halves of each qf32 pair into two accumulators,
  // then combine them. Each iteration processes 64 FP16 elements.
  HVX_Vector vsum_lo = Q6_V_vzero();
  HVX_Vector vsum_hi = Q6_V_vzero();

  int d = length;
  HVX_Vector *ptr = iptr;
  for (; d > 63; d -= 64) {
    HVX_Vector x = vmemu(ptr);
    ptr++;
    // x^2 in qf32: Wqf32 = vmpy(Vhf, Vhf)
    HVX_VectorPair x_sq = Q6_Wqf32_vmpy_VhfVhf(x, x);
    vsum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_lo, Q6_V_lo_W(x_sq));
    vsum_hi = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_hi, Q6_V_hi_W(x_sq));
  }
  // Handle remainder (< 64 elements)
  if (d > 0) {
    HVX_Vector x = vmemu(ptr);
    // Mask out-of-bounds elements to zero
    // For FP16: d elements = d*2 bytes. We need to zero elements beyond d.
    // Use vsetq2 which sets predicate for bytes [0, R) where R = d*2
    HVX_VectorPred qmask = Q6_Q_vsetq2_R(d * 2);
    x = Q6_V_vmux_QVV(qmask, x, vzero);
    HVX_VectorPair x_sq = Q6_Wqf32_vmpy_VhfVhf(x, x);
    vsum_lo = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_lo, Q6_V_lo_W(x_sq));
    vsum_hi = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_hi, Q6_V_hi_W(x_sq));
  }

  // Combine lo and hi accumulators (both are 32-element qf32 vectors)
  HVX_Vector vsum = Q6_Vqf32_vadd_Vqf32Vqf32(vsum_lo, vsum_hi);

  // ---- Step 2: Horizontal reduction of 32-element qf32 vector ----
  // Shuffle-and-add pattern: reduce 32 → 16 → 8 → 4 → 2 → 1
  for (int i = 0, nshift = 4; i < 5; i++) {
    HVX_VectorPair temps = Q6_W_vshuff_VVR(vsum, vsum, nshift);
    vsum = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(temps), Q6_V_hi_W(temps));
    nshift <<= 1;
  }

  // Convert qf32 to sf and extract scalar
  HVX_Vector vsf = Q6_Vsf_equals_Vqf32(vsum);
  ftmp.i = Q6_R_vextract_VR(vsf, 0);
  float sum_sq = ftmp.f;

  // ---- Step 3: Compute scale = rsqrt(sum_sq / length + epsilon) ----
  float mean_sq = sum_sq / (float)length;
  float scale = 1.0f / sqrtf(mean_sq + epsilon);

  // Broadcast scale as qf32 vector
  ftmp.f = scale;
  HVX_Vector vscale = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(ftmp.i), vzero);

  // ---- Step 4: Compute y[i] = x[i] * gamma[i] * scale ----
  HVX_Vector *gptr = (HVX_Vector *)pgamma;
  HVX_Vector *optr = (HVX_Vector *)pout;
  ptr = iptr;
  d = length;

  for (; d > 63; d -= 64) {
    HVX_Vector x = vmemu(ptr);
    ptr++;
    HVX_Vector g = vmemu(gptr);
    gptr++;

    // x * gamma in qf32
    HVX_VectorPair xg = Q6_Wqf32_vmpy_VhfVhf(x, g);

    // Multiply by scale
    HVX_Vector lo = Q6_Vqf32_vmpy_Vqf32Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(xg), vzero), vscale);
    HVX_Vector hi = Q6_Vqf32_vmpy_Vqf32Vqf32(
        Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(xg), vzero), vscale);

    // Convert back to FP16 and store
    HVX_VectorPair result = Q6_W_vcombine_VV(hi, lo);
    q6op_vstu_AV(optr, Q6_Vhf_equals_Wqf32(result));
    optr++;
  }

  // Handle remainder
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
// HVX FP16 entry point - iterates over [batch, height, width] dimensions
//=============================================================================

template <typename OutTtype, typename InTtype>
int rmsnorm_fp_impl(OutTtype &out, const InTtype &in, const InTtype &gamma,
                     const Tensor &epsilon) {
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();
  float eps = epsilon(0, 0, 0, 0);

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
// Reference (scalar) implementation for correctness verification
//=============================================================================

template <typename Ttype>
int rmsnorm_ref_impl(Ttype &out, const Ttype &in, const Ttype &gamma,
                      const Tensor &epsilon) {
  out.set_dims(in);
  auto [b_in, h_in, w_in, d_in] = in.dims();
  float eps = epsilon(0, 0, 0, 0);

  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (Idx d = 0; d < d_in; d++) {
          float val = in(b, h, w, d);
          sum_sq += val * val;
        }
        // Compute scale
        float scale = 1.0f / sqrtf(sum_sq / (float)d_in + eps);
        // Apply RMSNorm
        for (Idx d = 0; d < d_in; d++) {
          out(b, h, w, d) = in(b, h, w, d) * gamma(0, 0, 0, d) * scale;
        }
      }
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_RmsNorm);
