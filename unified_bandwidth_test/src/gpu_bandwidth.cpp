#define CL_TARGET_OPENCL_VERSION 200
#include "gpu_bandwidth.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Qualcomm ION extension for zero-copy buffer import
#ifndef CL_MEM_ION_HOST_PTR_QCOM
#define CL_MEM_ION_HOST_PTR_QCOM 0x40A8
#endif
#ifndef CL_MEM_EXT_HOST_PTR_QCOM
#define CL_MEM_EXT_HOST_PTR_QCOM (1 << 29)
#endif
#ifndef CL_MEM_HOST_UNCACHED_QCOM
#define CL_MEM_HOST_UNCACHED_QCOM 0
#endif

typedef struct {
  cl_uint allocation_type;
  cl_uint host_cache_policy;
} cl_mem_ext_host_ptr;

typedef struct {
  cl_mem_ext_host_ptr ext_host_ptr;
  int   ion_filedesc;
  void* ion_hostptr;
} cl_mem_ion_host_ptr;

// ── File-scope OpenCL state ─────────────────────────────────────────────────
namespace {
cl_platform_id    g_platform = nullptr;
cl_device_id      g_device   = nullptr;
cl_context        g_context  = nullptr;
cl_command_queue   g_queue    = nullptr;
cl_program        g_program  = nullptr;
cl_kernel         g_kernel   = nullptr;
cl_mem            g_parentA  = nullptr;  // full ION-backed cl_mem
cl_mem            g_parentB  = nullptr;
cl_mem            g_parentC  = nullptr;
cl_mem            g_bufA     = nullptr;  // sub-buffer (or retained parent)
cl_mem            g_bufB     = nullptr;
cl_mem            g_bufC     = nullptr;
size_t            g_data_size = 0;  // bytes per tensor (gpu partition)
size_t            g_num_vecs  = 0;  // data_size / 16 (uchar16)

static char* read_file(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "r");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc(sz + 1);
  size_t n = fread(buf, 1, sz, f);
  buf[n] = '\0';
  fclose(f);
  if (out_size) *out_size = n;
  return buf;
}

static cl_mem import_ion_buffer(const IonBuffer& ion, cl_mem_flags flags) {
  cl_mem_ion_host_ptr ion_mem = {};
  ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
  ion_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
  ion_mem.ion_filedesc = ion.fd;
  ion_mem.ion_hostptr  = ion.ptr;

  cl_int err;
  cl_mem buf = clCreateBuffer(g_context,
      flags | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
      ion.size, &ion_mem, &err);
  if (err != CL_SUCCESS) {
    printf("[GPU] clCreateBuffer (ION import) failed: %d\n", err);
    return nullptr;
  }
  return buf;
}
}  // namespace

// ── Public API ──────────────────────────────────────────────────────────────

void gpu_print_info() {
  if (!g_device) return;
  char name[256];
  clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(name), name, nullptr);

  cl_uint cu;
  clGetDeviceInfo(g_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);

  cl_ulong mem;
  clGetDeviceInfo(g_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);

  printf("  设备: %s\n", name);
  printf("  计算单元: %u, 全局内存: %.2f GB\n", cu, mem / (1024.0*1024.0*1024.0));
}

bool gpu_init(const IonBuffer& A, const IonBuffer& B, const IonBuffer& C,
              const char* kernel_path,
              size_t offset, size_t partition_size) {
  cl_int err;

  // Platform & device
  err = clGetPlatformIDs(1, &g_platform, nullptr);
  if (err != CL_SUCCESS) { printf("[GPU] clGetPlatformIDs: %d\n", err); return false; }

  err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, nullptr);
  if (err != CL_SUCCESS) { printf("[GPU] clGetDeviceIDs: %d\n", err); return false; }

  // Context & queue
  g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateContext: %d\n", err); return false; }

  g_queue = clCreateCommandQueueWithProperties(g_context, g_device, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateCommandQueue: %d\n", err); return false; }

  // Compile kernel
  size_t src_size = 0;
  char* src = read_file(kernel_path, &src_size);
  if (!src) { printf("[GPU] Cannot read %s\n", kernel_path); return false; }

  g_program = clCreateProgramWithSource(g_context, 1, (const char**)&src, &src_size, &err);
  free(src);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateProgramWithSource: %d\n", err); return false; }

  err = clBuildProgram(g_program, 1, &g_device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_sz;
    clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
    char* log = (char*)malloc(log_sz);
    clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_sz, log, nullptr);
    printf("[GPU] Build error:\n%s\n", log);
    free(log);
    return false;
  }

  g_kernel = clCreateKernel(g_program, "element_add_uchar16", &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateKernel: %d\n", err); return false; }

  // Import full ION buffers into OpenCL
  g_parentA = import_ion_buffer(A, CL_MEM_READ_ONLY);
  g_parentB = import_ion_buffer(B, CL_MEM_READ_ONLY);
  g_parentC = import_ion_buffer(C, CL_MEM_WRITE_ONLY);
  if (!g_parentA || !g_parentB || !g_parentC) return false;

  size_t actual_size = (partition_size > 0) ? partition_size : A.size;

  if (offset == 0 && actual_size == A.size) {
    // Full buffer — retain parent as working buffer
    g_bufA = g_parentA; clRetainMemObject(g_bufA);
    g_bufB = g_parentB; clRetainMemObject(g_bufB);
    g_bufC = g_parentC; clRetainMemObject(g_bufC);
  } else {
    // Create sub-buffer views for GPU partition
    cl_buffer_region region = { offset, actual_size };
    cl_int sub_err;
    g_bufA = clCreateSubBuffer(g_parentA, CL_MEM_READ_ONLY,
                 CL_BUFFER_CREATE_TYPE_REGION, &region, &sub_err);
    if (sub_err != CL_SUCCESS) {
      printf("[GPU] clCreateSubBuffer(A) failed: %d\n", sub_err); return false;
    }
    g_bufB = clCreateSubBuffer(g_parentB, CL_MEM_READ_ONLY,
                 CL_BUFFER_CREATE_TYPE_REGION, &region, &sub_err);
    if (sub_err != CL_SUCCESS) {
      printf("[GPU] clCreateSubBuffer(B) failed: %d\n", sub_err); return false;
    }
    g_bufC = clCreateSubBuffer(g_parentC, CL_MEM_WRITE_ONLY,
                 CL_BUFFER_CREATE_TYPE_REGION, &region, &sub_err);
    if (sub_err != CL_SUCCESS) {
      printf("[GPU] clCreateSubBuffer(C) failed: %d\n", sub_err); return false;
    }
    printf("[GPU] 统一缓冲区模式: offset=%zu, partition=%zu (fd=%d)\n",
           offset, actual_size, A.fd);
  }

  g_data_size = actual_size;
  g_num_vecs  = g_data_size / 16;  // uchar16 = 16 bytes

  // Set kernel args
  int nv = static_cast<int>(g_num_vecs);
  clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_bufC);
  clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_bufA);
  clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &g_bufB);
  clSetKernelArg(g_kernel, 3, sizeof(int), &nv);

  return true;
}

BandwidthResult gpu_run(int num_warmup, int num_iters, SpinBarrier* barrier) {
  BandwidthResult res;
  res.num_iterations = num_iters;
  res.total_data_bytes = (double)g_data_size * 3.0 * num_iters;  // 2 read + 1 write

  // Work sizes
  size_t global = g_num_vecs;
  size_t local  = 256;
  size_t max_wg;
  clGetDeviceInfo(g_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
  if (local > max_wg) local = max_wg;
  if (global % local != 0)
    global = ((global / local) + 1) * local;

  // Warmup
  for (int i = 0; i < num_warmup; ++i)
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  clFinish(g_queue);

  // Barrier: synchronized start with NPU
  if (barrier) barrier->arrive_and_wait();

  // Timed run
  double t0 = now_seconds();
  for (int i = 0; i < num_iters; ++i)
    clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  clFinish(g_queue);
  double t1 = now_seconds();

  res.elapsed_seconds = t1 - t0;
  res.bandwidth_gbps  = (res.total_data_bytes / (1024.0*1024.0*1024.0)) / res.elapsed_seconds;
  res.success = true;
  return res;
}

void gpu_cleanup() {
  if (g_kernel)  clReleaseKernel(g_kernel);
  if (g_bufA)    clReleaseMemObject(g_bufA);
  if (g_bufB)    clReleaseMemObject(g_bufB);
  if (g_bufC)    clReleaseMemObject(g_bufC);
  if (g_parentA) clReleaseMemObject(g_parentA);
  if (g_parentB) clReleaseMemObject(g_parentB);
  if (g_parentC) clReleaseMemObject(g_parentC);
  if (g_program) clReleaseProgram(g_program);
  if (g_queue)   clReleaseCommandQueue(g_queue);
  if (g_context) clReleaseContext(g_context);
  g_kernel = nullptr;
  g_bufA = nullptr; g_bufB = nullptr; g_bufC = nullptr;
  g_parentA = nullptr; g_parentB = nullptr; g_parentC = nullptr;
  g_program = nullptr; g_queue = nullptr; g_context = nullptr;
}
