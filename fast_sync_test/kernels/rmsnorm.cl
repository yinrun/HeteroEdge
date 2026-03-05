// RMSNorm kernel: y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
//
// One work-group per row (batch element).
// Each work-item handles hidden_dim / local_size elements.
// Uses float accumulation for numerical stability even in FP16 mode.
//
// Compile with -DUSE_FP16 for half precision.

#ifdef USE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half  scalar_t;
#define TO_FLOAT(x)  convert_float(x)
#define TO_SCALAR(x) convert_half(x)
#else
typedef float scalar_t;
#define TO_FLOAT(x)  (x)
#define TO_SCALAR(x) (x)
#endif

__kernel void rmsnorm(
    __global scalar_t*       output,     // [batch, hidden_dim]
    __global const scalar_t* input,      // [batch, hidden_dim]
    __global const scalar_t* gamma,      // [hidden_dim]
    const int hidden_dim,
    const float epsilon,
    __local float* sdata,                // local memory for reduction
    __global volatile uint*  done_flag)  // completion flag (NULL = skip)
{
  int row = get_group_id(0);    // which batch element
  int lid = get_local_id(0);
  int lsz = get_local_size(0);

  __global const scalar_t* x = input  + row * hidden_dim;
  __global scalar_t*       y = output + row * hidden_dim;

  // Phase 1: Partial sum of squares (float accumulation)
  float partial = 0.0f;
  for (int i = lid; i < hidden_dim; i += lsz) {
    float val = TO_FLOAT(x[i]);
    partial += val * val;
  }
  sdata[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Phase 2: Tree reduction in local memory
  for (int s = lsz >> 1; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] += sdata[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Phase 3: Compute normalization factor
  float rms_inv = rsqrt(sdata[0] / (float)hidden_dim + epsilon);

  // Phase 4: Normalize and scale by gamma
  for (int i = lid; i < hidden_dim; i += lsz) {
    float val = TO_FLOAT(x[i]);
    float g   = TO_FLOAT(gamma[i]);
    y[i] = TO_SCALAR(val * rms_inv * g);
  }

  // Phase 5: Write completion flag (for fast sync — CPU polls this)
  if (done_flag) {
    barrier(CLK_GLOBAL_MEM_FENCE);  // ensure all output writes are committed
    if (lid == 0)
      *done_flag = 1u;
  }
}
