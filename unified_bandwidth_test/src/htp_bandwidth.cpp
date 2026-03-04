#include "htp_bandwidth.h"

#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnOpDef.h"
#include "QNN/QnnMem.h"
#include "QNN/QnnTensor.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QNN/HTP/QnnHtpMem.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

// ── File-scope QNN state ────────────────────────────────────────────────────
namespace {

constexpr uint32_t kTensorRank = 4;

void*                       g_libHandle      = nullptr;
const QNN_INTERFACE_VER_TYPE* g_qnn           = nullptr;
Qnn_BackendHandle_t         g_backend        = nullptr;
Qnn_DeviceHandle_t          g_device         = nullptr;
Qnn_ContextHandle_t         g_context        = nullptr;
Qnn_GraphHandle_t           g_graph          = nullptr;
uint32_t                    g_htpCoreCount   = 0;
size_t                      g_data_size      = 0;

// Registered memory handles (for deregistration)
struct RegMem { Qnn_MemHandle_t handle = nullptr; };
RegMem g_regA, g_regB, g_regC;

// Execution tensors (pre-built, reused across iterations)
Qnn_Tensor_t g_execInputs[2];
Qnn_Tensor_t g_execOutputs[1];

// Tensor dimensions (computed from data_size)
uint32_t g_dims[kTensorRank];

bool check(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) { printf("[HTP] %s failed: %lu\n", w, (unsigned long)s); return false; }
  return true;
}

// ── Compute NHWC dims from byte count. C must be <= 1048576 ─────────────────
// Layout: [N, 1, 1, C] where N*C = total_elements
void computeDims(size_t totalBytes, uint32_t dims[4]) {
  uint32_t total = static_cast<uint32_t>(totalBytes);  // 1 byte per element
  uint32_t C = 1048576;  // max channel
  if (total < C) C = total;
  uint32_t N = total / C;
  dims[0] = N; dims[1] = 1; dims[2] = 1; dims[3] = C;
}

// ── Register an externally-allocated ION buffer with QNN ────────────────────
// When partition_size < ion.size, use HTP shared buffer API for precise sub-region mapping.
bool registerBuffer(const IonBuffer& ion, const uint32_t* dims, uint32_t ndims,
                    RegMem& out) {
  Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
  desc.memShape.numDim  = ndims;
  desc.memShape.dimSize = const_cast<uint32_t*>(dims);
  desc.dataType         = QNN_DATATYPE_UFIXED_POINT_8;

  // Compute tensor byte count from dims (1 byte per element for UFIXED_POINT_8)
  size_t tensor_bytes = 1;
  for (uint32_t i = 0; i < ndims; ++i) tensor_bytes *= dims[i];

  QnnMemHtp_Descriptor_t htpDesc;
  if (tensor_bytes < ion.size) {
    // Sub-region: use HTP shared buffer API with explicit offset
    htpDesc.type = QNN_HTP_MEM_SHARED_BUFFER;
    htpDesc.size = ion.size;
    htpDesc.sharedBufferConfig.fd     = ion.fd;
    htpDesc.sharedBufferConfig.offset = 0;  // NPU always at offset 0

    desc.memType    = QNN_MEM_TYPE_CUSTOM;
    desc.customInfo = &htpDesc;
  } else {
    // Full buffer: standard ION registration
    desc.memType   = QNN_MEM_TYPE_ION;
    desc.ionInfo.fd = ion.fd;
  }

  if (QNN_SUCCESS != g_qnn->memRegister(g_context, &desc, 1, &out.handle)) {
    printf("[HTP] memRegister failed (fd=%d, ion_size=%zu, tensor=%zu)\n",
           ion.fd, ion.size, tensor_bytes);
    return false;
  }
  return true;
}

void deregisterAll() {
  if (!g_qnn || !g_qnn->memDeRegister) return;
  std::vector<Qnn_MemHandle_t> handles;
  if (g_regA.handle) handles.push_back(g_regA.handle);
  if (g_regB.handle) handles.push_back(g_regB.handle);
  if (g_regC.handle) handles.push_back(g_regC.handle);
  if (!handles.empty())
    g_qnn->memDeRegister(handles.data(), static_cast<uint32_t>(handles.size()));
  g_regA.handle = g_regB.handle = g_regC.handle = nullptr;
}

// ── Power config: BURST mode, DCVS disabled ─────────────────────────────────
void setHighPerformanceMode() {
  if (!g_qnn->deviceGetInfrastructure) return;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (QNN_SUCCESS != g_qnn->deviceGetInfrastructure(&infra) || !infra) return;
  auto* htpInfra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  auto& perf = htpInfra->perfInfra;
  if (!perf.createPowerConfigId || !perf.setPowerConfig) return;

  uint32_t powerConfigId = 0;
  if (QNN_SUCCESS != perf.createPowerConfigId(0, 0, &powerConfigId)) return;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId            = powerConfigId;
  dcvs.dcvsV3Config.setDcvsEnable        = 1;
  dcvs.dcvsV3Config.dcvsEnable           = 0;
  dcvs.dcvsV3Config.powerMode            = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepDisable      = 1;
  dcvs.dcvsV3Config.sleepDisable         = 1;
  dcvs.dcvsV3Config.setBusParams         = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams        = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t hmx = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  hmx.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
  hmx.hmxV2Config.hmxPickDefault          = 0;
  hmx.hmxV2Config.hmxVoltageCornerMin     = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxVoltageCornerTarget  = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxVoltageCornerMax     = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxPerfMode             = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &hmx, nullptr};
  perf.setPowerConfig(powerConfigId, cfgs);
}

// ── Query HTP core count ────────────────────────────────────────────────────
uint32_t queryCoreCount() {
  if (!g_qnn->deviceGetPlatformInfo) return 0;
  const QnnDevice_PlatformInfo_t* info = nullptr;
  if (QNN_SUCCESS != g_qnn->deviceGetPlatformInfo(nullptr, &info) || !info) return 0;
  uint32_t cores = 0;
  if (info->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    for (uint32_t i = 0; i < info->v1.numHwDevices; ++i) {
      auto& dev = info->v1.hwDevices[i];
      if (dev.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) continue;
      auto* ext = reinterpret_cast<QnnHtpDevice_DeviceInfoExtension_t*>(dev.v1.deviceInfoExtension);
      if (ext && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) { cores = dev.v1.numCores; break; }
    }
  }
  if (g_qnn->deviceFreePlatformInfo) g_qnn->deviceFreePlatformInfo(nullptr, info);
  return cores;
}

Qnn_Tensor_t makeTensor(const char* name, Qnn_TensorType_t type, uint32_t* dims) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version      = QNN_TENSOR_VERSION_1;
  t.v1.id        = 0;
  t.v1.name      = name;
  t.v1.type      = type;
  t.v1.dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType       = QNN_DATATYPE_UFIXED_POINT_8;
  t.v1.quantizeParams.encodingDefinition   = QNN_DEFINITION_DEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  t.v1.quantizeParams.scaleOffsetEncoding.scale  = 1.0f;
  t.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
  t.v1.rank           = kTensorRank;
  t.v1.dimensions     = dims;
  t.v1.memType        = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf      = QNN_CLIENT_BUFFER_INIT;
  return t;
}

}  // namespace

// ── Public API ──────────────────────────────────────────────────────────────

void htp_print_info() {
  printf("  Hexagon V81, %u core(s), 8 HVX threads, BURST 模式\n", g_htpCoreCount);
  printf("  张量: [%u, %u, %u, %u] UFIXED_POINT_8\n",
         g_dims[0], g_dims[1], g_dims[2], g_dims[3]);
}

bool htp_init(const IonBuffer& A, const IonBuffer& B, const IonBuffer& C,
              size_t partition_size, int force_cores) {
  g_data_size = (partition_size > 0) ? partition_size : A.size;
  computeDims(g_data_size, g_dims);

  // dlopen QNN backend
  g_libHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_libHandle) { printf("[HTP] dlopen failed: %s\n", dlerror()); return false; }

  // Load interface
  using GetProvidersFn = decltype(&QnnInterface_getProviders);
  auto getProviders = reinterpret_cast<GetProvidersFn>(
      dlsym(g_libHandle, "QnnInterface_getProviders"));
  if (!getProviders) { printf("[HTP] getProviders not found\n"); return false; }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    printf("[HTP] No providers\n"); return false;
  }

  // Pick best matching API version
  const QnnInterface_t* best = nullptr;
  for (uint32_t i = 0; i < numProviders; ++i) {
    if (!providers[i]) continue;
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
      if (!best || providers[i]->apiVersion.coreApiVersion.minor >
                       best->apiVersion.coreApiVersion.minor)
        best = providers[i];
    }
  }
  if (!best) best = providers[0];
  g_qnn = &best->QNN_INTERFACE_VER_NAME;

  // Backend, device, power, context
  if (!check(g_qnn->backendCreate(nullptr, nullptr, &g_backend), "backendCreate"))
    return false;
  if (g_qnn->deviceCreate) {
    auto s = g_qnn->deviceCreate(nullptr, nullptr, &g_device);
    if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      printf("[HTP] deviceCreate failed: %lu\n", (unsigned long)s); return false;
    }
  }

  setHighPerformanceMode();

  if (!check(g_qnn->contextCreate(g_backend, g_device, nullptr, &g_context), "contextCreate"))
    return false;

  // Register ION buffers
  if (!registerBuffer(A, g_dims, kTensorRank, g_regA) ||
      !registerBuffer(B, g_dims, kTensorRank, g_regB) ||
      !registerBuffer(C, g_dims, kTensorRank, g_regC))
    return false;

  // Build graph
  g_htpCoreCount = queryCoreCount();
  if (force_cores > 0)
    g_htpCoreCount = static_cast<uint32_t>(force_cores);
  QnnHtpGraph_CustomConfig_t htpCfgs[2] = {QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT};
  uint32_t cfgCount = 0;
  if (g_htpCoreCount > 0) {
    htpCfgs[cfgCount].option   = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES;
    htpCfgs[cfgCount].numCores = g_htpCoreCount;
    cfgCount++;
  }
  htpCfgs[cfgCount].option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  htpCfgs[cfgCount].numHvxThreads = 8;
  cfgCount++;

  QnnGraph_Config_t graphCfg = QNN_GRAPH_CONFIG_INIT;
  graphCfg.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphCfg.customConfig = htpCfgs;
  const QnnGraph_Config_t* graphCfgList[] = {&graphCfg, nullptr};

  if (!check(g_qnn->graphCreate(g_context, "add_graph", graphCfgList, &g_graph), "graphCreate"))
    return false;

  // Graph tensors
  Qnn_Tensor_t in0 = makeTensor("input0", QNN_TENSOR_TYPE_APP_WRITE, g_dims);
  Qnn_Tensor_t in1 = makeTensor("input1", QNN_TENSOR_TYPE_APP_WRITE, g_dims);
  Qnn_Tensor_t out = makeTensor("output", QNN_TENSOR_TYPE_APP_READ,  g_dims);

  if (!check(g_qnn->tensorCreateGraphTensor(g_graph, &in0), "tensor in0") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &in1), "tensor in1") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &out), "tensor out"))
    return false;

  // Add node
  Qnn_Tensor_t opIn[2]  = {in0, in1};
  Qnn_Tensor_t opOut[1] = {out};
  Qnn_OpConfig_t addCfg = QNN_OPCONFIG_INIT;
  addCfg.version = QNN_OPCONFIG_VERSION_1;
  addCfg.v1.name         = "elementwise_add";
  addCfg.v1.packageName  = QNN_OP_PACKAGE_NAME_QTI_AISW;
  addCfg.v1.typeName     = QNN_OP_ELEMENT_WISE_ADD;
  addCfg.v1.numOfParams  = 0;
  addCfg.v1.params       = nullptr;
  addCfg.v1.numOfInputs  = 2;
  addCfg.v1.inputTensors = opIn;
  addCfg.v1.numOfOutputs = 1;
  addCfg.v1.outputTensors = opOut;

  if (!check(g_qnn->graphAddNode(g_graph, addCfg), "graphAddNode"))
    return false;
  if (!check(g_qnn->graphFinalize(g_graph, nullptr, nullptr), "graphFinalize"))
    return false;

  // Prepare execution tensors with registered mem handles
  g_execInputs[0] = in0;
  g_execInputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execInputs[0].v1.memHandle = g_regA.handle;

  g_execInputs[1] = in1;
  g_execInputs[1].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execInputs[1].v1.memHandle = g_regB.handle;

  g_execOutputs[0] = out;
  g_execOutputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execOutputs[0].v1.memHandle = g_regC.handle;

  return true;
}

BandwidthResult htp_run(int num_warmup, int num_iters, SpinBarrier* barrier) {
  BandwidthResult res;
  res.num_iterations   = num_iters;
  res.total_data_bytes = (double)g_data_size * 3.0 * num_iters;  // 2 read + 1 write

  // Warmup
  for (int i = 0; i < num_warmup; ++i) {
    if (!check(g_qnn->graphExecute(g_graph, g_execInputs, 2, g_execOutputs, 1,
                                   nullptr, nullptr), "warmup")) {
      res.error = "warmup failed";
      return res;
    }
  }

  // Barrier: synchronized start with GPU
  if (barrier) barrier->arrive_and_wait();

  // Timed run
  double t0 = now_seconds();
  for (int i = 0; i < num_iters; ++i) {
    if (!check(g_qnn->graphExecute(g_graph, g_execInputs, 2, g_execOutputs, 1,
                                   nullptr, nullptr), "exec")) {
      res.error = "exec failed";
      return res;
    }
  }
  double t1 = now_seconds();

  res.elapsed_seconds = t1 - t0;
  res.bandwidth_gbps  = (res.total_data_bytes / (1024.0*1024.0*1024.0)) / res.elapsed_seconds;
  res.success = true;
  return res;
}

void htp_cleanup() {
  deregisterAll();
  if (g_qnn && g_context) g_qnn->contextFree(g_context, nullptr);
  if (g_qnn && g_device && g_qnn->deviceFree) g_qnn->deviceFree(g_device);
  if (g_qnn && g_backend) g_qnn->backendFree(g_backend);
  if (g_libHandle) dlclose(g_libHandle);
  g_context = nullptr; g_device = nullptr; g_backend = nullptr;
  g_graph = nullptr; g_qnn = nullptr; g_libHandle = nullptr;
}
