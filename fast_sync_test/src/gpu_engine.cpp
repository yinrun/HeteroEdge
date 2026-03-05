#define CL_TARGET_OPENCL_VERSION 200
#include "gpu_engine.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

namespace {

cl_platform_id   g_platform = nullptr;
cl_device_id     g_device   = nullptr;
cl_context       g_context  = nullptr;
cl_command_queue  g_queue    = nullptr;
cl_program       g_program  = nullptr;
cl_kernel        g_kernel   = nullptr;
cl_mem           g_bufInput = nullptr;
cl_mem           g_bufOutput= nullptr;
cl_mem           g_bufGamma = nullptr;
cl_mem           g_bufFlag  = nullptr;
volatile uint32_t* g_flagPtr = nullptr;
int              g_hidden   = 0;
size_t           g_local    = 256;
size_t           g_global   = 0;

char* read_file(const char* path, size_t* out_size) {
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

cl_mem import_ion_buffer(const IonBuffer& ion, cl_mem_flags flags) {
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
    printf("[GPU] ION import failed: %d\n", err);
    return nullptr;
  }
  return buf;
}

}  // namespace

void gpu_print_info() {
  if (!g_device) return;
  char name[256];
  clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  cl_uint cu;
  clGetDeviceInfo(g_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
  cl_ulong mem;
  clGetDeviceInfo(g_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
  printf("  GPU: %s, %u CU, %.2f GB\n", name, cu, mem / (1024.0*1024.0*1024.0));
}

bool gpu_init(int hidden_dim, float epsilon,
              const IonBuffer& ion_input, const IonBuffer& ion_output,
              const char* kernel_path) {
  cl_int err;
  g_hidden = hidden_dim;

  // Platform & device
  err = clGetPlatformIDs(1, &g_platform, nullptr);
  if (err != CL_SUCCESS) { printf("[GPU] clGetPlatformIDs: %d\n", err); return false; }
  err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, nullptr);
  if (err != CL_SUCCESS) { printf("[GPU] clGetDeviceIDs: %d\n", err); return false; }

  // Context
  g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateContext: %d\n", err); return false; }

  // Queue with profiling enabled (to separate compute from sync)
  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  g_queue = clCreateCommandQueueWithProperties(g_context, g_device, props, &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateCommandQueue: %d\n", err); return false; }

  // Compile kernel
  size_t src_size = 0;
  char* src = read_file(kernel_path, &src_size);
  if (!src) { printf("[GPU] Cannot read %s\n", kernel_path); return false; }
  g_program = clCreateProgramWithSource(g_context, 1, (const char**)&src, &src_size, &err);
  free(src);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateProgramWithSource: %d\n", err); return false; }

  err = clBuildProgram(g_program, 1, &g_device, "-DUSE_FP16", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_sz;
    clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
    char* log = (char*)malloc(log_sz);
    clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_sz, log, nullptr);
    printf("[GPU] Build error:\n%s\n", log);
    free(log);
    return false;
  }

  g_kernel = clCreateKernel(g_program, "rmsnorm", &err);
  if (err != CL_SUCCESS) { printf("[GPU] clCreateKernel: %d\n", err); return false; }

  // Import ION buffers for zero-copy sharing with NPU
  g_bufInput  = import_ion_buffer(ion_input,  CL_MEM_READ_ONLY);
  g_bufOutput = import_ion_buffer(ion_output, CL_MEM_WRITE_ONLY);
  if (!g_bufInput || !g_bufOutput) return false;

  // Gamma buffer (local to GPU, not shared)
  size_t gamma_bytes = (size_t)hidden_dim * 2;
  g_bufGamma = clCreateBuffer(g_context, CL_MEM_READ_ONLY, gamma_bytes, nullptr, &err);
  if (err != CL_SUCCESS) { printf("[GPU] gamma buffer: %d\n", err); return false; }
  std::vector<uint16_t> host_gamma(hidden_dim, float_to_half(1.0f));
  clEnqueueWriteBuffer(g_queue, g_bufGamma, CL_TRUE, 0, gamma_bytes, host_gamma.data(), 0, nullptr, nullptr);

  // Work sizes: batch=1, one work-group
  size_t max_wg;
  clGetDeviceInfo(g_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
  g_local = 256;
  if (g_local > max_wg) g_local = max_wg;
  g_global = g_local;  // batch=1

  // Set kernel args
  int hd = hidden_dim;
  clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_bufOutput);
  clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_bufInput);
  clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &g_bufGamma);
  clSetKernelArg(g_kernel, 3, sizeof(int), &hd);
  clSetKernelArg(g_kernel, 4, sizeof(float), &epsilon);
  clSetKernelArg(g_kernel, 5, g_local * sizeof(float), nullptr);
  // Arg 6: done_flag (NULL = disabled, kernel skips flag write)
  cl_mem null_mem = nullptr;
  clSetKernelArg(g_kernel, 6, sizeof(cl_mem), &null_mem);

  return true;
}

bool gpu_enable_flag(const IonBuffer& ion_flag) {
  g_bufFlag = import_ion_buffer(ion_flag, CL_MEM_WRITE_ONLY);
  if (!g_bufFlag) return false;
  g_flagPtr = reinterpret_cast<volatile uint32_t*>(ion_flag.ptr);
  clSetKernelArg(g_kernel, 6, sizeof(cl_mem), &g_bufFlag);
  return true;
}

void gpu_disable_flag() {
  cl_mem null_mem = nullptr;
  clSetKernelArg(g_kernel, 6, sizeof(cl_mem), &null_mem);
  if (g_bufFlag) { clReleaseMemObject(g_bufFlag); g_bufFlag = nullptr; }
  g_flagPtr = nullptr;
}

volatile uint32_t* gpu_get_flag_ptr() {
  return g_flagPtr;
}

void gpu_submit() {
  if (g_flagPtr) *g_flagPtr = 0;  // reset flag before submission
  clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &g_global, &g_local, 0, nullptr, nullptr);
  clFlush(g_queue);
}

double gpu_execute_blocking(double* gpu_compute_us) {
  cl_event evt;
  double t0 = now_us();
  clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &g_global, &g_local, 0, nullptr, &evt);
  clFinish(g_queue);
  double t1 = now_us();

  if (gpu_compute_us) {
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
    *gpu_compute_us = (t_end - t_start) / 1000.0;
  }
  clReleaseEvent(evt);
  return t1 - t0;
}

double gpu_execute_blocking_noprof() {
  double t0 = now_us();
  clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &g_global, &g_local, 0, nullptr, nullptr);
  clFinish(g_queue);
  return now_us() - t0;
}

cl_event gpu_execute_nonblocking() {
  cl_event evt;
  clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &g_global, &g_local, 0, nullptr, &evt);
  clFlush(g_queue);
  return evt;
}

bool gpu_poll_event(cl_event evt) {
  cl_int status;
  clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr);
  return status == CL_COMPLETE;
}

double gpu_event_compute_us(cl_event evt) {
  cl_ulong t_start, t_end;
  clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
  clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
  return (t_end - t_start) / 1000.0;
}

GpuProfilingInfo gpu_event_profiling(cl_event evt) {
  cl_ulong t_queued, t_submit, t_start, t_end;
  clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_QUEUED, sizeof(t_queued), &t_queued, nullptr);
  clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_SUBMIT, sizeof(t_submit), &t_submit, nullptr);
  clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
  clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);

  GpuProfilingInfo info;
  info.queued_us    = t_queued / 1000.0;
  info.submit_us    = t_submit / 1000.0;
  info.start_us     = t_start / 1000.0;
  info.end_us       = t_end / 1000.0;
  info.queue_delay  = (t_submit - t_queued) / 1000.0;
  info.submit_delay = (t_start - t_submit) / 1000.0;
  info.compute      = (t_end - t_start) / 1000.0;
  info.total_device = (t_end - t_queued) / 1000.0;
  return info;
}

void gpu_cleanup() {
  if (g_kernel)    clReleaseKernel(g_kernel);
  if (g_bufInput)  clReleaseMemObject(g_bufInput);
  if (g_bufOutput) clReleaseMemObject(g_bufOutput);
  if (g_bufGamma)  clReleaseMemObject(g_bufGamma);
  if (g_bufFlag)   clReleaseMemObject(g_bufFlag);
  if (g_program)   clReleaseProgram(g_program);
  if (g_queue)     clReleaseCommandQueue(g_queue);
  if (g_context)   clReleaseContext(g_context);
  g_kernel = nullptr; g_bufInput = nullptr; g_bufOutput = nullptr; g_bufGamma = nullptr;
  g_bufFlag = nullptr; g_flagPtr = nullptr;
  g_program = nullptr; g_queue = nullptr; g_context = nullptr;
  g_platform = nullptr; g_device = nullptr;
}
