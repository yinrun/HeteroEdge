#include "npu_engine.h"

#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnOpDef.h"
#include "QNN/QnnMem.h"
#include "QNN/QnnTensor.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

namespace {

constexpr uint32_t kTensorRank = 4;

void*                        g_libHandle  = nullptr;
const QNN_INTERFACE_VER_TYPE* g_qnn       = nullptr;
Qnn_LogHandle_t              g_log        = nullptr;
Qnn_BackendHandle_t          g_backend    = nullptr;
Qnn_DeviceHandle_t           g_device     = nullptr;
Qnn_ContextHandle_t          g_context    = nullptr;
Qnn_GraphHandle_t            g_graph      = nullptr;
uint32_t                     g_coreCount  = 0;

// NPU owns gamma/beta; input/output are external ION buffers
IonBuffer g_ionGamma, g_ionBeta;

struct RegMem { Qnn_MemHandle_t handle = nullptr; };
RegMem g_regInput, g_regOutput;
RegMem g_regFlag, g_regPollCount;  // for NPU_POLL mode

// For native graph: 1 input, 1 output
// For flagpoll graph: 3 inputs (data, gamma_static_unused, flag), 2 outputs (result, poll_count)
// graphExecute uses only APP_WRITE/APP_READ tensors
Qnn_Tensor_t g_execInputs[2];   // [0]=data, [1]=flag (flagpoll mode)
Qnn_Tensor_t g_execOutputs[2];  // [0]=result, [1]=poll_count (flagpoll mode)
uint32_t g_numExecInputs = 1;
uint32_t g_numExecOutputs = 1;

uint32_t g_dimsIO[kTensorRank];
uint32_t g_dimsGamma1D[1];
uint32_t g_dimsAxes[1];
uint32_t g_dimsFlagIO[kTensorRank] = {1, 1, 1, 1};  // for flag/poll_count tensors

int g_hidden = 0;
bool g_flagpollMode = false;
void* g_pollCountPtr = nullptr;  // CPU-mapped pointer to poll_count ION buffer

void qnnLogCallback(const char* fmt, QnnLog_Level_t level,
                     uint64_t /*timestamp*/, va_list args) {
  if (level != QNN_LOG_LEVEL_ERROR) return;
  printf("[QNN-E] ");
  vprintf(fmt, args);
  printf("\n");
}

bool check(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) {
    printf("[NPU] %s failed: %lu\n", w, (unsigned long)s);
    return false;
  }
  return true;
}

bool registerBuffer(const IonBuffer& ion, const uint32_t* dims, uint32_t ndims,
                    Qnn_DataType_t dtype, RegMem& out) {
  Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
  desc.memShape.numDim  = ndims;
  desc.memShape.dimSize = const_cast<uint32_t*>(dims);
  desc.dataType         = dtype;
  desc.memType          = QNN_MEM_TYPE_ION;
  desc.ionInfo.fd       = ion.fd;
  if (QNN_SUCCESS != g_qnn->memRegister(g_context, &desc, 1, &out.handle)) {
    printf("[NPU] memRegister failed (fd=%d)\n", ion.fd);
    return false;
  }
  return true;
}

void deregisterAll() {
  if (!g_qnn || !g_qnn->memDeRegister) return;
  std::vector<Qnn_MemHandle_t> handles;
  if (g_regInput.handle)     handles.push_back(g_regInput.handle);
  if (g_regOutput.handle)    handles.push_back(g_regOutput.handle);
  if (g_regFlag.handle)      handles.push_back(g_regFlag.handle);
  if (g_regPollCount.handle) handles.push_back(g_regPollCount.handle);
  if (!handles.empty())
    g_qnn->memDeRegister(handles.data(), static_cast<uint32_t>(handles.size()));
  g_regInput.handle = g_regOutput.handle = nullptr;
  g_regFlag.handle = g_regPollCount.handle = nullptr;
}

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
  dcvs.dcvsV3Config.contextId              = powerConfigId;
  dcvs.dcvsV3Config.setDcvsEnable          = 1;
  dcvs.dcvsV3Config.dcvsEnable             = 0;
  dcvs.dcvsV3Config.powerMode              = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepDisable        = 1;
  dcvs.dcvsV3Config.sleepDisable           = 1;
  dcvs.dcvsV3Config.setBusParams           = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams          = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin   = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget= DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax   = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t hmx = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  hmx.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
  hmx.hmxV2Config.hmxPickDefault         = 0;
  hmx.hmxV2Config.hmxVoltageCornerMin    = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxVoltageCornerMax    = DCVS_EXP_VCORNER_MAX;
  hmx.hmxV2Config.hmxPerfMode            = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &hmx, nullptr};
  perf.setPowerConfig(powerConfigId, cfgs);
}

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

Qnn_Tensor_t makeFp16Tensor(const char* name, Qnn_TensorType_t type,
                              uint32_t* dims, uint32_t rank = kTensorRank) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version      = QNN_TENSOR_VERSION_1;
  t.v1.id        = 0;
  t.v1.name      = name;
  t.v1.type      = type;
  t.v1.dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType       = QNN_DATATYPE_FLOAT_16;
  t.v1.quantizeParams.encodingDefinition   = QNN_DEFINITION_UNDEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  t.v1.rank           = rank;
  t.v1.dimensions     = dims;
  t.v1.memType        = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf      = QNN_CLIENT_BUFFER_INIT;
  return t;
}

bool createAxesTensor(Qnn_Tensor_t& out, uint32_t* axes_data, uint32_t num_axes) {
  g_dimsAxes[0] = num_axes;
  out = QNN_TENSOR_INIT;
  out.version = QNN_TENSOR_VERSION_1;
  out.v1.name = "axes";
  out.v1.type = QNN_TENSOR_TYPE_STATIC;
  out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  out.v1.dataType = QNN_DATATYPE_UINT_32;
  out.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  out.v1.rank = 1;
  out.v1.dimensions = g_dimsAxes;
  out.v1.memType = QNN_TENSORMEMTYPE_RAW;
  out.v1.clientBuf.data     = axes_data;
  out.v1.clientBuf.dataSize = num_axes * sizeof(uint32_t);
  return check(g_qnn->tensorCreateGraphTensor(g_graph, &out), "tensor axes");
}

bool buildNativeGraph() {
  Qnn_Tensor_t input  = makeFp16Tensor("input",  QNN_TENSOR_TYPE_APP_WRITE, g_dimsIO);
  Qnn_Tensor_t output = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  g_dimsIO);
  Qnn_Tensor_t gamma  = makeFp16Tensor("gamma",  QNN_TENSOR_TYPE_STATIC,    g_dimsGamma1D, 1);
  Qnn_Tensor_t beta   = makeFp16Tensor("beta",   QNN_TENSOR_TYPE_STATIC,    g_dimsGamma1D, 1);
  gamma.v1.clientBuf.data     = g_ionGamma.ptr;
  gamma.v1.clientBuf.dataSize = static_cast<uint32_t>(g_ionGamma.size);
  beta.v1.clientBuf.data      = g_ionBeta.ptr;
  beta.v1.clientBuf.dataSize  = static_cast<uint32_t>(g_ionBeta.size);

  if (!check(g_qnn->tensorCreateGraphTensor(g_graph, &input),  "tensor input") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &gamma),  "tensor gamma") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &beta),   "tensor beta") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &output), "tensor output"))
    return false;

  Qnn_Param_t eps_param = QNN_PARAM_INIT;
  eps_param.paramType    = QNN_PARAMTYPE_SCALAR;
  eps_param.name         = QNN_OP_RMS_NORM_PARAM_EPSILON;
  eps_param.scalarParam.dataType   = QNN_DATATYPE_FLOAT_32;
  eps_param.scalarParam.floatValue = 1e-6f;

  uint32_t axes_data[] = {3};
  Qnn_Tensor_t axes_tensor;
  if (!createAxesTensor(axes_tensor, axes_data, 1))
    return false;

  Qnn_Param_t axes_param = QNN_PARAM_INIT;
  axes_param.paramType   = QNN_PARAMTYPE_TENSOR;
  axes_param.name        = QNN_OP_RMS_NORM_PARAM_AXES;
  axes_param.tensorParam = axes_tensor;

  Qnn_Param_t params[] = {eps_param, axes_param};
  Qnn_Tensor_t opIn[]  = {input, gamma, beta};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "rmsnorm";
  op.v1.packageName  = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName     = QNN_OP_RMS_NORM;
  op.v1.numOfParams  = 2; op.v1.params = params;
  op.v1.numOfInputs  = 3; op.v1.inputTensors  = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;

  if (!check(g_qnn->graphAddNode(g_graph, op), "graphAddNode(RmsNorm)"))
    return false;

  g_execInputs[0] = input;
  g_execOutputs[0] = output;
  return true;
}

Qnn_Tensor_t makeUint32Tensor(const char* name, Qnn_TensorType_t type,
                               uint32_t* dims, uint32_t rank = kTensorRank) {
  Qnn_Tensor_t t = QNN_TENSOR_INIT;
  t.version      = QNN_TENSOR_VERSION_1;
  t.v1.id        = 0;
  t.v1.name      = name;
  t.v1.type      = type;
  t.v1.dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t.v1.dataType       = QNN_DATATYPE_UINT_32;
  t.v1.quantizeParams.encodingDefinition   = QNN_DEFINITION_UNDEFINED;
  t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  t.v1.rank           = rank;
  t.v1.dimensions     = dims;
  t.v1.memType        = QNN_TENSORMEMTYPE_RAW;
  t.v1.clientBuf      = QNN_CLIENT_BUFFER_INIT;
  return t;
}

bool buildFlagPollGraph() {
  // Inputs: data (FP16, APP_WRITE), gamma (FP16, STATIC), flag (UINT32, APP_WRITE)
  Qnn_Tensor_t input  = makeFp16Tensor("input",  QNN_TENSOR_TYPE_APP_WRITE, g_dimsIO);
  Qnn_Tensor_t output = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  g_dimsIO);
  Qnn_Tensor_t gamma  = makeFp16Tensor("gamma",  QNN_TENSOR_TYPE_STATIC,    g_dimsGamma1D, 1);
  gamma.v1.clientBuf.data     = g_ionGamma.ptr;
  gamma.v1.clientBuf.dataSize = static_cast<uint32_t>(g_ionGamma.size);

  Qnn_Tensor_t flag_in    = makeUint32Tensor("flag",       QNN_TENSOR_TYPE_APP_WRITE, g_dimsFlagIO);
  Qnn_Tensor_t poll_out   = makeUint32Tensor("poll_count", QNN_TENSOR_TYPE_APP_READ,  g_dimsFlagIO);

  if (!check(g_qnn->tensorCreateGraphTensor(g_graph, &input),    "tensor input") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &gamma),    "tensor gamma") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &flag_in),  "tensor flag") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &output),   "tensor output") ||
      !check(g_qnn->tensorCreateGraphTensor(g_graph, &poll_out), "tensor poll_count"))
    return false;

  // Epsilon parameter
  Qnn_Param_t eps_param = QNN_PARAM_INIT;
  eps_param.paramType    = QNN_PARAMTYPE_SCALAR;
  eps_param.name         = "epsilon";
  eps_param.scalarParam.dataType   = QNN_DATATYPE_FLOAT_32;
  eps_param.scalarParam.floatValue = 1e-6f;

  Qnn_Param_t params[] = {eps_param};
  Qnn_Tensor_t opIn[]  = {input, gamma, flag_in};
  Qnn_Tensor_t opOut[] = {output, poll_out};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "flagpoll_rmsnorm";
  op.v1.packageName  = "flagpoll_rmsnorm.HvxOpPackage";
  op.v1.typeName     = "FlagPollRmsNorm";
  op.v1.numOfParams  = 1; op.v1.params = params;
  op.v1.numOfInputs  = 3; op.v1.inputTensors  = opIn;
  op.v1.numOfOutputs = 2; op.v1.outputTensors = opOut;

  if (!check(g_qnn->graphAddNode(g_graph, op), "graphAddNode(FlagPollRmsNorm)"))
    return false;

  // Set up execution tensors: APP_WRITE inputs = data + flag, APP_READ outputs = result + poll_count
  g_execInputs[0]  = input;
  g_execInputs[1]  = flag_in;
  g_execOutputs[0] = output;
  g_execOutputs[1] = poll_out;
  g_numExecInputs  = 2;
  g_numExecOutputs = 2;
  return true;
}

bool buildGraph() {
  QnnHtpGraph_CustomConfig_t htpCfgs[4] = {QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                             QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT};
  uint32_t cfgCount = 0;

  htpCfgs[cfgCount].option    = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  htpCfgs[cfgCount].precision = QNN_PRECISION_FLOAT16;
  cfgCount++;
  if (g_coreCount > 0) {
    htpCfgs[cfgCount].option   = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES;
    htpCfgs[cfgCount].numCores = g_coreCount;
    cfgCount++;
  }
  htpCfgs[cfgCount].option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  htpCfgs[cfgCount].numHvxThreads = 8;
  cfgCount++;

  QnnGraph_Config_t graphCfg = QNN_GRAPH_CONFIG_INIT;
  graphCfg.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graphCfg.customConfig = htpCfgs;
  const QnnGraph_Config_t* graphCfgList[] = {&graphCfg, nullptr};

  if (!check(g_qnn->graphCreate(g_context, "rmsnorm_graph", graphCfgList, &g_graph), "graphCreate"))
    return false;

  bool graphOk = g_flagpollMode ? buildFlagPollGraph() : buildNativeGraph();
  if (!graphOk) { g_graph = nullptr; return false; }

  if (!check(g_qnn->graphFinalize(g_graph, nullptr, nullptr), "graphFinalize")) {
    g_graph = nullptr; return false;
  }
  return true;
}

}  // namespace

void npu_print_info() {
  if (g_flagpollMode)
    printf("  NPU: Hexagon V81, %u core(s), FlagPoll RmsNorm (custom HVX, FP16)\n", g_coreCount);
  else
    printf("  NPU: Hexagon V81, %u core(s), Native RmsNorm (FP16)\n", g_coreCount);
}

bool npu_init(int hidden_dim, float epsilon,
              const IonBuffer& ion_input, const IonBuffer& ion_output) {
  g_hidden = hidden_dim;
  size_t gamma_bytes = (size_t)hidden_dim * 2;

  // Tensor dimensions: batch=1
  g_dimsIO[0] = 1; g_dimsIO[1] = 1; g_dimsIO[2] = 1; g_dimsIO[3] = hidden_dim;
  g_dimsGamma1D[0] = hidden_dim;

  // Allocate gamma/beta (internal, not shared)
  if (!allocIonBuffer(gamma_bytes, 0, g_ionGamma) ||
      !allocIonBuffer(gamma_bytes, 0, g_ionBeta)) {
    printf("[NPU] Failed to alloc ION gamma/beta\n"); return false;
  }
  uint16_t one = float_to_half(1.0f);
  uint16_t* gp = reinterpret_cast<uint16_t*>(g_ionGamma.ptr);
  for (int i = 0; i < hidden_dim; ++i) gp[i] = one;

  // dlopen QNN backend
  g_libHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_libHandle) { printf("[NPU] dlopen failed: %s\n", dlerror()); return false; }

  using GetProvidersFn = decltype(&QnnInterface_getProviders);
  auto getProviders = reinterpret_cast<GetProvidersFn>(
      dlsym(g_libHandle, "QnnInterface_getProviders"));
  if (!getProviders) { printf("[NPU] getProviders not found\n"); return false; }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    printf("[NPU] No providers\n"); return false;
  }

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

  if (g_qnn->logCreate)
    g_qnn->logCreate(qnnLogCallback, QNN_LOG_LEVEL_ERROR, &g_log);

  if (!check(g_qnn->backendCreate(g_log, nullptr, &g_backend), "backendCreate"))
    return false;
  if (g_qnn->deviceCreate) {
    auto s = g_qnn->deviceCreate(nullptr, nullptr, &g_device);
    if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      printf("[NPU] deviceCreate failed: %lu\n", (unsigned long)s); return false;
    }
  }

  setHighPerformanceMode();

  if (!check(g_qnn->contextCreate(g_backend, g_device, nullptr, &g_context), "contextCreate"))
    return false;

  g_coreCount = queryCoreCount();

  // Register external ION buffers
  if (!registerBuffer(ion_input,  g_dimsIO, kTensorRank, QNN_DATATYPE_FLOAT_16, g_regInput) ||
      !registerBuffer(ion_output, g_dimsIO, kTensorRank, QNN_DATATYPE_FLOAT_16, g_regOutput))
    return false;

  if (!buildGraph())
    return false;

  // Bind registered memory handles
  g_execInputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execInputs[0].v1.memHandle = g_regInput.handle;
  g_execOutputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execOutputs[0].v1.memHandle = g_regOutput.handle;

  return true;
}

bool npu_init_flagpoll(int hidden_dim, float epsilon,
                       const IonBuffer& ion_input, const IonBuffer& ion_output,
                       const IonBuffer& ion_flag, const IonBuffer& ion_poll_count) {
  g_flagpollMode = true;
  g_pollCountPtr = ion_poll_count.ptr;
  g_hidden = hidden_dim;
  size_t gamma_bytes = (size_t)hidden_dim * 2;

  g_dimsIO[0] = 1; g_dimsIO[1] = 1; g_dimsIO[2] = 1; g_dimsIO[3] = hidden_dim;
  g_dimsGamma1D[0] = hidden_dim;

  // Allocate gamma (internal)
  if (!allocIonBuffer(gamma_bytes, 0, g_ionGamma)) {
    printf("[NPU] Failed to alloc ION gamma\n"); return false;
  }
  uint16_t one = float_to_half(1.0f);
  uint16_t* gp = reinterpret_cast<uint16_t*>(g_ionGamma.ptr);
  for (int i = 0; i < hidden_dim; ++i) gp[i] = one;

  // dlopen QNN backend
  g_libHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_libHandle) { printf("[NPU] dlopen failed: %s\n", dlerror()); return false; }

  using GetProvidersFn = decltype(&QnnInterface_getProviders);
  auto getProviders = reinterpret_cast<GetProvidersFn>(
      dlsym(g_libHandle, "QnnInterface_getProviders"));
  if (!getProviders) { printf("[NPU] getProviders not found\n"); return false; }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    printf("[NPU] No providers\n"); return false;
  }

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

  if (g_qnn->logCreate)
    g_qnn->logCreate(qnnLogCallback, QNN_LOG_LEVEL_ERROR, &g_log);

  if (!check(g_qnn->backendCreate(g_log, nullptr, &g_backend), "backendCreate"))
    return false;

  // Register custom op package
  if (!check(g_qnn->backendRegisterOpPackage(g_backend,
      "libQnnHtpFlagPollRmsNormOpPackage.so",
      "flagpollRmsnormInterfaceProvider", "CPU"),
      "registerOpPackage(CPU)"))
    return false;
  if (!check(g_qnn->backendRegisterOpPackage(g_backend,
      "htp/libQnnHtpFlagPollRmsNormOpPackage.so",
      "flagpollRmsnormInterfaceProvider", "HTP"),
      "registerOpPackage(HTP)"))
    return false;

  if (g_qnn->deviceCreate) {
    auto s = g_qnn->deviceCreate(nullptr, nullptr, &g_device);
    if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      printf("[NPU] deviceCreate failed: %lu\n", (unsigned long)s); return false;
    }
  }

  setHighPerformanceMode();

  if (!check(g_qnn->contextCreate(g_backend, g_device, nullptr, &g_context), "contextCreate"))
    return false;

  g_coreCount = queryCoreCount();

  // Register external ION buffers: input, output, flag, poll_count
  if (!registerBuffer(ion_input,      g_dimsIO,     kTensorRank, QNN_DATATYPE_FLOAT_16, g_regInput) ||
      !registerBuffer(ion_output,     g_dimsIO,     kTensorRank, QNN_DATATYPE_FLOAT_16, g_regOutput) ||
      !registerBuffer(ion_flag,       g_dimsFlagIO, kTensorRank, QNN_DATATYPE_UINT_32,  g_regFlag) ||
      !registerBuffer(ion_poll_count, g_dimsFlagIO, kTensorRank, QNN_DATATYPE_UINT_32,  g_regPollCount))
    return false;

  if (!buildGraph())
    return false;

  // Bind registered memory handles
  // execInputs: [0]=data, [1]=flag
  g_execInputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execInputs[0].v1.memHandle = g_regInput.handle;
  g_execInputs[1].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execInputs[1].v1.memHandle = g_regFlag.handle;
  // execOutputs: [0]=result, [1]=poll_count
  g_execOutputs[0].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execOutputs[0].v1.memHandle = g_regOutput.handle;
  g_execOutputs[1].v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  g_execOutputs[1].v1.memHandle = g_regPollCount.handle;

  return true;
}

double npu_execute_blocking() {
  double t0 = now_us();
  g_qnn->graphExecute(g_graph,
                       g_execInputs,  g_numExecInputs,
                       g_execOutputs, g_numExecOutputs,
                       nullptr, nullptr);
  double t1 = now_us();
  return t1 - t0;
}

uint32_t npu_get_last_poll_count() {
  if (!g_pollCountPtr) return 0;
  uint32_t val;
  memcpy(&val, g_pollCountPtr, sizeof(val));
  return val;
}

void npu_cleanup() {
  deregisterAll();
  if (g_qnn && g_context) g_qnn->contextFree(g_context, nullptr);
  if (g_qnn && g_device && g_qnn->deviceFree) g_qnn->deviceFree(g_device);
  if (g_qnn && g_backend) g_qnn->backendFree(g_backend);
  if (g_qnn && g_log && g_qnn->logFree) g_qnn->logFree(g_log);
  if (g_libHandle) dlclose(g_libHandle);
  g_context = nullptr; g_device = nullptr; g_backend = nullptr;
  g_graph = nullptr; g_qnn = nullptr; g_libHandle = nullptr; g_log = nullptr;

  freeIonBuffer(g_ionGamma);
  freeIonBuffer(g_ionBeta);

  g_flagpollMode = false;
  g_pollCountPtr = nullptr;
  g_numExecInputs = 1;
  g_numExecOutputs = 1;
}
