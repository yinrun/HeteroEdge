//=============================================================================
//  Benchmark: Custom HVX RMSNorm vs QNN Native RMSNorm on HTP
//
//  Usage: ./rmsnorm_custom_test [--iters N] [--warmup N]
//=============================================================================

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnMem.h"
#include "QnnOpDef.h"
#include "QnnProperty.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpMem.h"

//=============================================================================
// ION memory via libcdsprpc.so
//=============================================================================
static void *(*rpc_alloc)(int, uint32_t, int);
static void (*rpc_free)(void *);
static int (*rpc_to_fd)(void *);

static bool init_rpcmem() {
  void *h = dlopen("libcdsprpc.so", RTLD_NOW);
  if (!h) return false;
  rpc_alloc = (decltype(rpc_alloc))dlsym(h, "rpcmem_alloc");
  rpc_free = (decltype(rpc_free))dlsym(h, "rpcmem_free");
  rpc_to_fd = (decltype(rpc_to_fd))dlsym(h, "rpcmem_to_fd");
  return rpc_alloc && rpc_free && rpc_to_fd;
}

struct IonBuf {
  void *ptr = nullptr;
  int fd = -1;
  size_t size = 0;
  bool alloc(size_t sz) {
    size = sz;
    ptr = rpc_alloc(25, 1, (int)sz);
    if (!ptr) return false;
    fd = rpc_to_fd(ptr);
    return fd >= 0;
  }
  void free() { if (ptr) { rpc_free(ptr); ptr = nullptr; } }
};

//=============================================================================
// QNN loading
//=============================================================================
static const QNN_INTERFACE_VER_TYPE *qnn;

static void logCb(const char *fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level <= QNN_LOG_LEVEL_WARN) {
    printf("[QNN-%s] ", level <= QNN_LOG_LEVEL_ERROR ? "ERR" : "WARN");
    vprintf(fmt, args);
    printf("\n");
  }
}

static bool loadQnn(const char *lib) {
  void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen %s: %s\n", lib, dlerror()); return false; }
  auto get = (decltype(&QnnInterface_getProviders))dlsym(h, "QnnInterface_getProviders");
  if (!get) return false;
  const QnnInterface_t **provs = nullptr;
  uint32_t n = 0;
  if (get(&provs, &n) != QNN_SUCCESS || n == 0) return false;
  qnn = &provs[0]->QNN_INTERFACE_VER_NAME;
  return true;
}

//=============================================================================
// CPU reference
//=============================================================================
static void rmsnorm_cpu(float *out, const float *x, const float *g, int H, float eps) {
  float ss = 0;
  for (int i = 0; i < H; i++) ss += x[i] * x[i];
  float s = 1.0f / sqrtf(ss / H + eps);
  for (int i = 0; i < H; i++) out[i] = x[i] * g[i] * s;
}

//=============================================================================
// Tensor helper
//=============================================================================
static Qnn_Tensor_t mkTensor(const char *name, Qnn_TensorType_t type,
                              uint32_t *dims, uint32_t rank) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version = QNN_TENSOR_VERSION_1;
  t.v1 = {.name = name, .type = type,
           .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
           .dataType = QNN_DATATYPE_FLOAT_32,
           .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {}},
           .rank = rank, .dimensions = dims,
           .memType = QNN_TENSORMEMTYPE_RAW, .clientBuf = QNN_CLIENT_BUFFER_INIT};
  return t;
}

//=============================================================================
// Graph session
//=============================================================================
struct Session {
  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device;
  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  Qnn_MemHandle_t memIn = nullptr, memOut = nullptr;
  Qnn_Tensor_t execIn, execOut;
  uint32_t dimsIO[4], dimsG[1];

  bool check(Qnn_ErrorHandle_t rc, const char *msg) {
    if (rc == QNN_SUCCESS) return true;
    fprintf(stderr, "  %s failed: %d\n", msg, (int)rc);
    return false;
  }

  bool init(const char *name) {
    if (!check(qnn->contextCreate(backend, device, nullptr, &ctx), "contextCreate"))
      return false;

    QnnHtpGraph_CustomConfig_t hcfg[2] = {};
    hcfg[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    hcfg[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    hcfg[0].precision = QNN_PRECISION_FLOAT16;
    hcfg[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    hcfg[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    hcfg[1].optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    hcfg[1].optimizationOption.floatValue = 3.0f;

    QnnGraph_Config_t gc[2];
    for (int i = 0; i < 2; i++) {
      gc[i] = QNN_GRAPH_CONFIG_INIT;
      gc[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      gc[i].customConfig = &hcfg[i];
    }
    const QnnGraph_Config_t *cfgs[] = {&gc[0], &gc[1], nullptr};
    return check(qnn->graphCreate(ctx, name, cfgs, &graph), "graphCreate");
  }

  bool registerIon(IonBuf &buf, const uint32_t *dims, uint32_t rank, Qnn_MemHandle_t &handle) {
    Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
    desc.memShape.numDim = rank;
    desc.memShape.dimSize = const_cast<uint32_t *>(dims);
    desc.dataType = QNN_DATATYPE_FLOAT_32;
    desc.memType = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd = buf.fd;
    return check(qnn->memRegister(ctx, &desc, 1, &handle), "memRegister");
  }

  bool buildCustom(int B, int H, IonBuf &ionIn, IonBuf &ionGamma, IonBuf &ionOut, float eps) {
    dimsIO[0] = B; dimsIO[1] = 1; dimsIO[2] = 1; dimsIO[3] = H;
    dimsG[0] = H;

    auto tIn = mkTensor("input", QNN_TENSOR_TYPE_APP_WRITE, dimsIO, 4);
    auto tGamma = mkTensor("gamma", QNN_TENSOR_TYPE_STATIC, dimsG, 1);
    tGamma.v1.clientBuf = {ionGamma.ptr, (uint32_t)ionGamma.size};
    auto tOut = mkTensor("output", QNN_TENSOR_TYPE_APP_READ, dimsIO, 4);

    if (!check(qnn->tensorCreateGraphTensor(graph, &tIn), "tensor:in") ||
        !check(qnn->tensorCreateGraphTensor(graph, &tGamma), "tensor:gamma") ||
        !check(qnn->tensorCreateGraphTensor(graph, &tOut), "tensor:out"))
      return false;

    Qnn_Param_t epsP = QNN_PARAM_INIT;
    epsP.paramType = QNN_PARAMTYPE_SCALAR;
    epsP.name = "epsilon";
    epsP.scalarParam = {QNN_DATATYPE_FLOAT_32, {.floatValue = eps}};

    Qnn_Tensor_t ins[] = {tIn, tGamma}, outs[] = {tOut};
    Qnn_Param_t params[] = {epsP};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1 = {"rmsnorm_custom", "rmsnorm.HvxOpPackage", "RmsNorm",
             1, params, 2, ins, 1, outs};

    if (!check(qnn->graphAddNode(graph, op), "graphAddNode(Custom)")) return false;
    execIn = tIn; execOut = tOut;
    return true;
  }

  bool buildNative(int B, int H, IonBuf &ionIn, IonBuf &ionGamma, IonBuf &ionBeta,
                   IonBuf &ionOut, float eps) {
    dimsIO[0] = B; dimsIO[1] = 1; dimsIO[2] = 1; dimsIO[3] = H;
    dimsG[0] = H;

    auto tIn = mkTensor("input", QNN_TENSOR_TYPE_APP_WRITE, dimsIO, 4);
    auto tGamma = mkTensor("gamma", QNN_TENSOR_TYPE_STATIC, dimsG, 1);
    tGamma.v1.clientBuf = {ionGamma.ptr, (uint32_t)ionGamma.size};
    auto tBeta = mkTensor("beta", QNN_TENSOR_TYPE_STATIC, dimsG, 1);
    tBeta.v1.clientBuf = {ionBeta.ptr, (uint32_t)ionBeta.size};
    auto tOut = mkTensor("output", QNN_TENSOR_TYPE_APP_READ, dimsIO, 4);

    if (!check(qnn->tensorCreateGraphTensor(graph, &tIn), "tensor:in") ||
        !check(qnn->tensorCreateGraphTensor(graph, &tGamma), "tensor:gamma") ||
        !check(qnn->tensorCreateGraphTensor(graph, &tBeta), "tensor:beta") ||
        !check(qnn->tensorCreateGraphTensor(graph, &tOut), "tensor:out"))
      return false;

    // Epsilon
    Qnn_Param_t epsP = QNN_PARAM_INIT;
    epsP.paramType = QNN_PARAMTYPE_SCALAR;
    epsP.name = QNN_OP_RMS_NORM_PARAM_EPSILON;
    epsP.scalarParam = {QNN_DATATYPE_FLOAT_32, {.floatValue = eps}};

    // Axes tensor (must be a graph tensor)
    uint32_t axesData[] = {3}, axesDim[] = {1};
    Qnn_Tensor_t axesT = QNN_TENSOR_INIT;
    axesT.version = QNN_TENSOR_VERSION_1;
    axesT.v1 = {.name = "axes", .type = QNN_TENSOR_TYPE_STATIC,
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = QNN_DATATYPE_UINT_32,
                .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {}},
                .rank = 1, .dimensions = axesDim,
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {axesData, sizeof(axesData)}};
    if (!check(qnn->tensorCreateGraphTensor(graph, &axesT), "tensor:axes")) return false;

    Qnn_Param_t axesP = QNN_PARAM_INIT;
    axesP.paramType = QNN_PARAMTYPE_TENSOR;
    axesP.name = QNN_OP_RMS_NORM_PARAM_AXES;
    axesP.tensorParam = axesT;

    Qnn_Param_t params[] = {epsP, axesP};
    Qnn_Tensor_t ins[] = {tIn, tGamma, tBeta}, outs[] = {tOut};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1 = {"rmsnorm_native", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RMS_NORM,
             2, params, 3, ins, 1, outs};

    if (!check(qnn->graphAddNode(graph, op), "graphAddNode(Native)")) return false;
    execIn = tIn; execOut = tOut;
    return true;
  }

  double run(int warmup, int iters) {
    if (!check(qnn->graphFinalize(graph, nullptr, nullptr), "graphFinalize")) return -1;

    execIn.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    execIn.v1.memHandle = memIn;
    execOut.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    execOut.v1.memHandle = memOut;

    Qnn_Tensor_t in[] = {execIn}, out[] = {execOut};
    for (int i = 0; i < warmup; i++)
      if (qnn->graphExecute(graph, in, 1, out, 1, nullptr, nullptr) != QNN_SUCCESS)
        return -1;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
      qnn->graphExecute(graph, in, 1, out, 1, nullptr, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  }

  void destroy() {
    Qnn_MemHandle_t handles[] = {memIn, memOut};
    for (auto &h : handles) if (h) { qnn->memDeRegister(&h, 1); h = nullptr; }
    if (ctx) { qnn->contextFree(ctx, nullptr); ctx = nullptr; }
    graph = nullptr;
  }
};

//=============================================================================
// Correctness check: max error of row 0 vs CPU reference, print first 8 elems
//=============================================================================
struct ErrorStats { float maxErr, avgErr; };

static ErrorStats computeError(const float *out, const float *ref, int H) {
  float maxE = 0, sumE = 0;
  for (int i = 0; i < H; i++) {
    float e = fabsf(out[i] - ref[i]);
    if (e > maxE) maxE = e;
    sumE += e;
  }
  return {maxE, sumE / H};
}

static void printRow(const char *label, const float *data, int H, int n = 8) {
  printf("  %6s: ", label);
  for (int i = 0; i < n && i < H; i++) printf("%10.6f ", data[i]);
  printf("\n");
}

//=============================================================================
// Run one test: build graph, benchmark, capture output, compute error
//=============================================================================
struct Result { double us; ErrorStats err; std::vector<float> output; };

static Result runCustom(Qnn_BackendHandle_t be, Qnn_DeviceHandle_t dev,
                        const char *name, int B, int H, float eps,
                        IonBuf &ionIn, IonBuf &ionGamma, IonBuf &ionOut,
                        int warmup, int iters) {
  Result r = {-1, {99, 99}, {}};
  Session s{be, dev};
  uint32_t dims[] = {(uint32_t)B, 1, 1, (uint32_t)H};
  if (!s.init(name) || !s.registerIon(ionIn, dims, 4, s.memIn) ||
      !s.registerIon(ionOut, dims, 4, s.memOut))
    { s.destroy(); return r; }
  if (!s.buildCustom(B, H, ionIn, ionGamma, ionOut, eps))
    { s.destroy(); return r; }
  r.us = s.run(warmup, iters);
  if (r.us > 0) {
    float *out = (float *)ionOut.ptr;
    r.output.assign(out, out + H);  // save row 0
    r.err = computeError(out, out, H); // placeholder, caller computes vs ref
  }
  s.destroy();
  return r;
}

static Result runNative(Qnn_BackendHandle_t be, Qnn_DeviceHandle_t dev,
                        const char *name, int B, int H, float eps,
                        IonBuf &ionIn, IonBuf &ionGamma, IonBuf &ionBeta, IonBuf &ionOut,
                        int warmup, int iters) {
  Result r = {-1, {99, 99}, {}};
  Session s{be, dev};
  uint32_t dims[] = {(uint32_t)B, 1, 1, (uint32_t)H};
  if (!s.init(name) || !s.registerIon(ionIn, dims, 4, s.memIn) ||
      !s.registerIon(ionOut, dims, 4, s.memOut))
    { s.destroy(); return r; }
  if (!s.buildNative(B, H, ionIn, ionGamma, ionBeta, ionOut, eps))
    { s.destroy(); return r; }
  r.us = s.run(warmup, iters);
  if (r.us > 0) {
    float *out = (float *)ionOut.ptr;
    r.output.assign(out, out + H);
  }
  s.destroy();
  return r;
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char **argv) {
  int iters = 100, warmup = 10;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
  }

  printf("=== Custom HVX RMSNorm vs QNN Native RMSNorm Benchmark ===\n");
  printf("iters=%d, warmup=%d\n\n", iters, warmup);

  if (!init_rpcmem()) { fprintf(stderr, "Failed to init rpcmem\n"); return 1; }
  if (!loadQnn("libQnnHtp.so")) { fprintf(stderr, "Failed to load QNN HTP\n"); return 1; }

  Qnn_LogHandle_t logH = nullptr;
  qnn->logCreate(logCb, QNN_LOG_LEVEL_WARN, &logH);

  Qnn_BackendHandle_t backend = nullptr;
  if (qnn->backendCreate(logH, nullptr, &backend) != QNN_SUCCESS) return 1;

  Qnn_DeviceHandle_t device = nullptr;
  if (qnn->deviceCreate) {
    auto s = qnn->deviceCreate(nullptr, nullptr, &device);
    if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) device = nullptr;
  }

  // Register custom op package (ARM for prepare + Hexagon for execution)
  bool hasPkg =
    qnn->backendRegisterOpPackage(backend, "./libQnnHtpRmsNormOpPackage.so",
                                  "rmsnormInterfaceProvider", "CPU") == QNN_SUCCESS &&
    qnn->backendRegisterOpPackage(backend, "./htp/libQnnHtpRmsNormOpPackage.so",
                                  "rmsnormInterfaceProvider", "HTP") == QNN_SUCCESS;
  printf("Custom op package: %s\n", hasPkg ? "registered" : "FAILED");

  struct { int B, H; const char *name; } cases[] = {
    {1, 2048, "decode-2k"}, {1, 3200, "decode-3.2k"}, {1, 4096, "decode-4k"},
    {16, 4096, "prefill-16"}, {64, 4096, "prefill-64"},
    {256, 4096, "pf-256"}, {512, 4096, "pf-512"}, {1024, 4096, "pf-1k"},
  };
  float eps = 1e-5f;

  printf("\n%-12s %5s %5s | %8s %8s %12s | %8s %8s %12s | %s\n",
         "Scene", "batch", "hid", "Cust(us)", "BW(GB/s)", "MaxErr",
         "Natv(us)", "BW(GB/s)", "MaxErr", "Speedup");
  printf("--------------------------------------------------------------"
         "----------------------------------------------------------------\n");

  for (auto &tc : cases) {
    int B = tc.B, H = tc.H;
    size_t dataBytes = (size_t)B * H * sizeof(float);
    size_t gammaBytes = (size_t)H * sizeof(float);

    IonBuf ionIn, ionGamma, ionBeta, ionOut;
    if (!ionIn.alloc(dataBytes) || !ionGamma.alloc(gammaBytes) ||
        !ionBeta.alloc(gammaBytes) || !ionOut.alloc(dataBytes)) {
      fprintf(stderr, "ION alloc failed for %s\n", tc.name);
      continue;
    }

    // Fill input data
    float *inF = (float *)ionIn.ptr, *gF = (float *)ionGamma.ptr;
    srand(42);
    for (int i = 0; i < B * H; i++) inF[i] = (rand() % 10000 / 5000.0f - 1.0f);
    for (int i = 0; i < H; i++) gF[i] = rand() % 10000 / 5000.0f;
    memset(ionBeta.ptr, 0, gammaBytes);

    // CPU reference (row 0)
    std::vector<float> ref(H);
    rmsnorm_cpu(ref.data(), inF, gF, H, eps);

    // BW: read input + gamma + write output, all in FP16 on HTP
    double bw_bytes = (double)B * H * 2 * 2 + (double)H * 2;
    char gname[64];

    // Custom
    snprintf(gname, sizeof(gname), "custom_%s", tc.name);
    auto cust = hasPkg ? runCustom(backend, device, gname, B, H, eps,
                                   ionIn, ionGamma, ionOut, warmup, iters)
                       : Result{-1, {99, 99}, {}};
    if (!cust.output.empty()) cust.err = computeError(cust.output.data(), ref.data(), H);

    // Native
    snprintf(gname, sizeof(gname), "native_%s", tc.name);
    auto natv = runNative(backend, device, gname, B, H, eps,
                          ionIn, ionGamma, ionBeta, ionOut, warmup, iters);
    if (!natv.output.empty()) natv.err = computeError(natv.output.data(), ref.data(), H);

    // Print results
    double cBW = cust.us > 0 ? bw_bytes / (cust.us * 1e3) : 0;
    double nBW = natv.us > 0 ? bw_bytes / (natv.us * 1e3) : 0;
    char spd[16] = "N/A";
    if (cust.us > 0 && natv.us > 0) snprintf(spd, sizeof(spd), "%.2fx", natv.us / cust.us);

    printf("%-12s %5d %5d | %8.1f %8.2f %e | %8.1f %8.2f %e | %s\n",
           tc.name, B, H, cust.us, cBW, cust.err.maxErr,
           natv.us, nBW, natv.err.maxErr, spd);

    // Detailed verification
    printf("  [verify row 0, first 8 elems]\n");
    printRow("CPU", ref.data(), H);
    if (!cust.output.empty()) {
      printRow("Custom", cust.output.data(), H);
      printf("  Custom vs CPU: max_err=%e, avg_err=%e\n", cust.err.maxErr, cust.err.avgErr);
    }
    if (!natv.output.empty()) {
      printRow("Native", natv.output.data(), H);
      printf("  Native vs CPU: max_err=%e, avg_err=%e\n", natv.err.maxErr, natv.err.avgErr);
    }
    printf("\n");

    ionIn.free(); ionGamma.free(); ionBeta.free(); ionOut.free();
  }

  if (device) qnn->deviceFree(device);
  qnn->backendFree(backend);
  if (logH) qnn->logFree(logH);
  printf("Done.\n");
  return 0;
}
