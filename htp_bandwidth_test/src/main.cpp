// Minimal QNN HTP demo: element-wise add on two 128MB tensors.
#include <dlfcn.h>
#include <stdint.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

namespace {
// Single-node bandwidth test: 2 inputs + 1 output, layout [128, 1, 1, 1048576]
constexpr uint32_t kBatchSize     = 128;
constexpr uint32_t kHeight        = 1;
constexpr uint32_t kWidth         = 1;
constexpr uint32_t kChannels      = 1048576;
constexpr uint32_t kTotalElements = kBatchSize * kHeight * kWidth * kChannels; // 128MB
constexpr uint32_t kTensorRank    = 4;

bool checkStatus(Qnn_ErrorHandle_t status, const char* what) {
  if (status != QNN_SUCCESS) {
    std::cerr << "[QNN] " << what << " failed with status=" << status << std::endl;
    return false;
  }
  return true;
}


struct RegisteredBuffer {
  void* data              = nullptr;
  size_t size             = 0;
  int fd                  = -1;
  Qnn_MemHandle_t memHandle = nullptr;
};

struct RpcMemApi {
  void* libHandle = nullptr;
  void* (*alloc)(int, int, int) = nullptr;
  void (*freeMem)(void*) = nullptr;
  int (*toFd)(void*) = nullptr;
};

RpcMemApi& getRpcMemApi() {
  static RpcMemApi api;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    const char* candidates[] = {
        "libcdsprpc.so",
        "/vendor/lib64/libcdsprpc.so",
        "/system/lib64/libcdsprpc.so",
        nullptr};
    for (const char** path = candidates; *path != nullptr && !api.libHandle; ++path) {
      api.libHandle = dlopen(*path, RTLD_LAZY | RTLD_LOCAL);
    }
    if (api.libHandle) {
      api.alloc = reinterpret_cast<void* (*)(int, int, int)>(dlsym(api.libHandle, "rpcmem_alloc"));
      api.freeMem = reinterpret_cast<void (*)(void*)>(dlsym(api.libHandle, "rpcmem_free"));
      api.toFd = reinterpret_cast<int (*)(void*)>(dlsym(api.libHandle, "rpcmem_to_fd"));
    }
  }
  return api;
}

constexpr int RPCMEM_HEAP_ID_SYSTEM = 25;
constexpr int RPCMEM_DEFAULT_FLAGS  = 0;

bool createRegisteredBuffer(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                            Qnn_ContextHandle_t contextHandle,
                            const uint32_t* dims,
                            uint32_t numDims,
                            size_t sizeBytes,
                            int8_t initValue,
                            RegisteredBuffer& out) {
  RpcMemApi& rpc = getRpcMemApi();
  if (!rpc.alloc || !rpc.freeMem || !rpc.toFd) {
    std::cerr << "[QNN] rpcmem API not available" << std::endl;
    return false;
  }

  out.data = rpc.alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                       static_cast<int>(sizeBytes));
  if (!out.data) {
    std::cerr << "[QNN] rpcmem_alloc failed" << std::endl;
    return false;
  }
  out.size = sizeBytes;
  if (initValue >= 0) {
    std::memset(out.data, initValue, sizeBytes);
  }
  out.fd = rpc.toFd(out.data);
  if (out.fd < 0) {
    std::cerr << "[QNN] rpcmem_to_fd failed" << std::endl;
    rpc.freeMem(out.data);
    out.data = nullptr;
    return false;
  }

  if (qnnInterface.memRegister == nullptr) {
    std::cerr << "[QNN] memRegister not available" << std::endl;
    rpc.freeMem(out.data);
    out.data = nullptr;
    return false;
  }

  Qnn_MemDescriptor_t memDesc = QNN_MEM_DESCRIPTOR_INIT;
  memDesc.memShape.numDim      = numDims;
  memDesc.memShape.dimSize     = const_cast<uint32_t*>(dims);
  memDesc.memShape.shapeConfig = nullptr;
  memDesc.dataType             = QNN_DATATYPE_UFIXED_POINT_8;
  memDesc.memType              = QNN_MEM_TYPE_ION;
  memDesc.ionInfo.fd           = out.fd;

  if (QNN_SUCCESS !=
      qnnInterface.memRegister(contextHandle, &memDesc, 1, &out.memHandle)) {
    std::cerr << "[QNN] memRegister failed" << std::endl;
    rpc.freeMem(out.data);
    out.data = nullptr;
    return false;
  }
  return true;
}

void releaseRegisteredBuffers(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                              std::vector<RegisteredBuffer>& buffers) {
  if (!buffers.empty() && qnnInterface.memDeRegister) {
    std::vector<Qnn_MemHandle_t> handles;
    handles.reserve(buffers.size());
    for (const auto& buf : buffers) {
      if (buf.memHandle != nullptr) {
        handles.push_back(buf.memHandle);
      }
    }
    if (!handles.empty()) {
      qnnInterface.memDeRegister(handles.data(), static_cast<uint32_t>(handles.size()));
    }
  }

  RpcMemApi& rpc = getRpcMemApi();
  for (auto& buf : buffers) {
    if (buf.data && rpc.freeMem) {
      rpc.freeMem(buf.data);
    }
  }
  buffers.clear();
}

int cleanupAndExit(const QNN_INTERFACE_VER_TYPE& qnnInterface,
                   Qnn_BackendHandle_t backendHandle,
                   Qnn_DeviceHandle_t deviceHandle,
                   Qnn_ContextHandle_t contextHandle,
                   void* backendLibHandle,
                   bool hasContext,
                   bool hasDevice) {
  if (hasContext) {
    qnnInterface.contextFree(contextHandle, nullptr);
  }
  if (hasDevice && qnnInterface.deviceFree) {
    qnnInterface.deviceFree(deviceHandle);
  }
  qnnInterface.backendFree(backendHandle);
  dlclose(backendLibHandle);
  return 1;
}

void setHtpHighPerformanceMode(const QNN_INTERFACE_VER_TYPE& qnnInterface) {
  if (qnnInterface.deviceGetInfrastructure == nullptr) {
    return;
  }

  QnnDevice_Infrastructure_t infra = nullptr;
  if (QNN_SUCCESS != qnnInterface.deviceGetInfrastructure(&infra) || infra == nullptr) {
    return;
  }

  auto* htpInfra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
    return;
  }

  auto& perf = htpInfra->perfInfra;
  if (perf.createPowerConfigId == nullptr || perf.setPowerConfig == nullptr ||
      perf.destroyPowerConfigId == nullptr) {
    return;
  }

  uint32_t powerConfigId = 0;
  if (QNN_SUCCESS != perf.createPowerConfigId(0, 0, &powerConfigId)) {
    return;
  }

  QnnHtpPerfInfrastructure_PowerConfig_t dcvsConfig =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvsConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvsConfig.dcvsV3Config.contextId       = powerConfigId;
  dcvsConfig.dcvsV3Config.setDcvsEnable   = 1;
  dcvsConfig.dcvsV3Config.dcvsEnable      = 0;
  dcvsConfig.dcvsV3Config.powerMode       = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvsConfig.dcvsV3Config.setSleepDisable = 1;
  dcvsConfig.dcvsV3Config.sleepDisable    = 1;
  dcvsConfig.dcvsV3Config.setBusParams    = 1;
  dcvsConfig.dcvsV3Config.busVoltageCornerMin =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvsConfig.dcvsV3Config.busVoltageCornerTarget =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvsConfig.dcvsV3Config.busVoltageCornerMax =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvsConfig.dcvsV3Config.setCoreParams = 1;
  dcvsConfig.dcvsV3Config.coreVoltageCornerMin =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvsConfig.dcvsV3Config.coreVoltageCornerTarget =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvsConfig.dcvsV3Config.coreVoltageCornerMax =
      DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t hmxConfig =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  hmxConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
  hmxConfig.hmxV2Config.hmxPickDefault = 0;
  hmxConfig.hmxV2Config.hmxVoltageCornerMin = DCVS_EXP_VCORNER_MAX;
  hmxConfig.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX;
  hmxConfig.hmxV2Config.hmxVoltageCornerMax = DCVS_EXP_VCORNER_MAX;
  hmxConfig.hmxV2Config.hmxPerfMode = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {
      &dcvsConfig, &hmxConfig, nullptr};
  perf.setPowerConfig(powerConfigId, powerConfigs);
  // Do NOT destroy — keep power config alive during execution
}

uint32_t getHtpCoreCount(const QNN_INTERFACE_VER_TYPE& qnnInterface) {
  if (qnnInterface.deviceGetPlatformInfo == nullptr) {
    return 0;
  }
  const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
  if (QNN_SUCCESS != qnnInterface.deviceGetPlatformInfo(nullptr, &platformInfo) ||
      platformInfo == nullptr) {
    return 0;
  }

  uint32_t coreCount = 0;
  if (platformInfo->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    for (uint32_t i = 0; i < platformInfo->v1.numHwDevices; ++i) {
      QnnDevice_HardwareDeviceInfo_t& dev = platformInfo->v1.hwDevices[i];
      if (dev.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) {
        continue;
      }
      auto* ext =
          reinterpret_cast<QnnHtpDevice_DeviceInfoExtension_t*>(dev.v1.deviceInfoExtension);
      if (ext != nullptr && ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
        coreCount = dev.v1.numCores;
        break;
      }
    }
  }

  if (qnnInterface.deviceFreePlatformInfo) {
    qnnInterface.deviceFreePlatformInfo(nullptr, platformInfo);
  }
  return coreCount;
}

const QnnInterface_t* loadQnnInterface(void* backendLibHandle) {
  using GetProvidersFn = decltype(&QnnInterface_getProviders);
  auto getProviders =
      reinterpret_cast<GetProvidersFn>(dlsym(backendLibHandle, "QnnInterface_getProviders"));
  if (!getProviders) {
    std::cerr << "[QNN] Failed to find QnnInterface_getProviders" << std::endl;
    return nullptr;
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders            = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    std::cerr << "[QNN] No QNN interface providers available" << std::endl;
    return nullptr;
  }

  const QnnInterface_t* bestMatch = nullptr;
  for (uint32_t i = 0; i < numProviders; ++i) {
    const auto* provider = providers[i];
    if (provider == nullptr) {
      continue;
    }
    if (provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
      if (!bestMatch ||
          provider->apiVersion.coreApiVersion.minor >
              bestMatch->apiVersion.coreApiVersion.minor) {
        bestMatch = provider;
      }
    }
  }
  if (bestMatch) {
    return bestMatch;
  }
  return providers[0];
}

Qnn_Tensor_t makeTensor(const char* name,
                        Qnn_TensorType_t type,
                        Qnn_DataType_t dataType,
                        uint32_t* dims) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.version      = QNN_TENSOR_VERSION_1;
  tensor.v1.id        = 0;
  tensor.v1.name      = name;
  tensor.v1.type      = type;
  tensor.v1.dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.v1.dataType       = dataType;
  tensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  tensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  tensor.v1.quantizeParams.scaleOffsetEncoding.scale = 1.0f;
  tensor.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
  tensor.v1.rank           = kTensorRank;
  tensor.v1.dimensions     = dims;
  tensor.v1.memType        = QNN_TENSORMEMTYPE_RAW;
  tensor.v1.clientBuf      = QNN_CLIENT_BUFFER_INIT;
  return tensor;
}
}  // namespace

int main() {
  std::cout << "[QNN] HTP Add demo starting..." << std::endl;

  void* backendLibHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!backendLibHandle) {
    std::cerr << "[QNN] dlopen libQnnHtp.so failed: " << dlerror() << std::endl;
    return 1;
  }

  const QnnInterface_t* interfaceProvider = loadQnnInterface(backendLibHandle);
  if (!interfaceProvider) {
    dlclose(backendLibHandle);
    return 1;
  }

  std::cout << "[QNN] Using provider API "
            << interfaceProvider->apiVersion.coreApiVersion.major << "."
            << interfaceProvider->apiVersion.coreApiVersion.minor << "."
            << interfaceProvider->apiVersion.coreApiVersion.patch << std::endl;

  auto& qnn = interfaceProvider->QNN_INTERFACE_VER_NAME;

  Qnn_BackendHandle_t backendHandle = nullptr;
  if (!checkStatus(qnn.backendCreate(nullptr, nullptr, &backendHandle), "backendCreate")) {
    dlclose(backendLibHandle);
    return 1;
  }

  Qnn_DeviceHandle_t deviceHandle = nullptr;
  if (qnn.deviceCreate) {
    auto status = qnn.deviceCreate(nullptr, nullptr, &deviceHandle);
    if (status != QNN_SUCCESS && status != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      std::cerr << "[QNN] deviceCreate failed with status=" << status << std::endl;
      return cleanupAndExit(qnn, backendHandle, deviceHandle, nullptr, backendLibHandle, false, false);
    }
  }

  setHtpHighPerformanceMode(qnn);

  Qnn_ContextHandle_t contextHandle = nullptr;
  if (!checkStatus(qnn.contextCreate(backendHandle, deviceHandle, nullptr, &contextHandle),
                   "contextCreate")) {
    return cleanupAndExit(qnn, backendHandle, deviceHandle, nullptr, backendLibHandle, false, true);
  }

  Qnn_GraphHandle_t graphHandle = nullptr;
  uint32_t htpCoreCount = getHtpCoreCount(qnn);
  const QnnGraph_Config_t* graphConfigs[] = {nullptr, nullptr};
  QnnGraph_Config_t graphConfig           = QNN_GRAPH_CONFIG_INIT;
  QnnHtpGraph_CustomConfig_t htpGraphConfigs[2] = {QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                                                   QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT};
  uint32_t htpConfigCount = 0;
  if (htpCoreCount > 0) {
    htpGraphConfigs[htpConfigCount].option   = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES;
    htpGraphConfigs[htpConfigCount].numCores = htpCoreCount;
    htpConfigCount++;
    std::cout << "[QNN] HTP num_cores=" << htpCoreCount << std::endl;
  }
  htpGraphConfigs[htpConfigCount].option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  htpGraphConfigs[htpConfigCount].numHvxThreads = 8;
  htpConfigCount++;
  std::cout << "[QNN] HTP num_hvx_threads=8" << std::endl;

  if (htpConfigCount > 0) {
    graphConfig.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graphConfig.customConfig = htpGraphConfigs;
    graphConfigs[0]          = &graphConfig;
  }

  const QnnGraph_Config_t** graphConfigList = (htpConfigCount > 0) ? graphConfigs : nullptr;
  if (!checkStatus(qnn.graphCreate(contextHandle, "add_graph", graphConfigList, &graphHandle),
                   "graphCreate")) {
    return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
  }

  // Single-node layout: [N, H, W, C]
  uint32_t dims[kTensorRank] = {kBatchSize, kHeight, kWidth, kChannels};
  const size_t totalBytes = static_cast<size_t>(kTotalElements) * sizeof(int8_t);

  std::cout << "[QNN] Single-node dims: [" << kBatchSize << ", " << kHeight
            << ", " << kWidth << ", " << kChannels << "]"
            << "  totalBytes=" << totalBytes << std::endl;

  // Allocate 3 registered buffers: input0, input1, output
  RegisteredBuffer input0Buf, input1Buf, outputBuf;

  auto cleanup = [&]() {
    std::vector<RegisteredBuffer> bufs;
    if (input0Buf.data) bufs.push_back(input0Buf);
    if (input1Buf.data) bufs.push_back(input1Buf);
    if (outputBuf.data) bufs.push_back(outputBuf);
    releaseRegisteredBuffers(qnn, bufs);
  };

  if (!createRegisteredBuffer(qnn, contextHandle, dims, kTensorRank, totalBytes, 1, input0Buf) ||
      !createRegisteredBuffer(qnn, contextHandle, dims, kTensorRank, totalBytes, 2, input1Buf) ||
      !createRegisteredBuffer(qnn, contextHandle, dims, kTensorRank, totalBytes, 0, outputBuf)) {
    cleanup();
    return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
  }

  // Create graph tensors (1 Add node: 2 inputs + 1 output)
  Qnn_Tensor_t input0Tensor = makeTensor("input0", QNN_TENSOR_TYPE_APP_WRITE,
                                          QNN_DATATYPE_UFIXED_POINT_8, dims);
  Qnn_Tensor_t input1Tensor = makeTensor("input1", QNN_TENSOR_TYPE_APP_WRITE,
                                          QNN_DATATYPE_UFIXED_POINT_8, dims);
  Qnn_Tensor_t outputTensor = makeTensor("output", QNN_TENSOR_TYPE_APP_READ,
                                          QNN_DATATYPE_UFIXED_POINT_8, dims);

  if (!checkStatus(qnn.tensorCreateGraphTensor(graphHandle, &input0Tensor),
                   "createGraphTensor input0") ||
      !checkStatus(qnn.tensorCreateGraphTensor(graphHandle, &input1Tensor),
                   "createGraphTensor input1") ||
      !checkStatus(qnn.tensorCreateGraphTensor(graphHandle, &outputTensor),
                   "createGraphTensor output")) {
    cleanup();
    return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
  }

  // Single Add op
  Qnn_Tensor_t opInputs[2]  = {input0Tensor, input1Tensor};
  Qnn_Tensor_t opOutputs[1] = {outputTensor};

  Qnn_OpConfig_t addConfig = QNN_OPCONFIG_INIT;
  addConfig.version         = QNN_OPCONFIG_VERSION_1;
  addConfig.v1.name         = "elementwise_add";
  addConfig.v1.packageName  = QNN_OP_PACKAGE_NAME_QTI_AISW;
  addConfig.v1.typeName     = QNN_OP_ELEMENT_WISE_ADD;
  addConfig.v1.numOfParams  = 0;
  addConfig.v1.params       = nullptr;
  addConfig.v1.numOfInputs  = 2;
  addConfig.v1.inputTensors = opInputs;
  addConfig.v1.numOfOutputs = 1;
  addConfig.v1.outputTensors = opOutputs;

  if (!checkStatus(qnn.graphAddNode(graphHandle, addConfig), "graphAddNode add")) {
    cleanup();
    return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
  }

  std::cout << "[QNN] graphFinalize starting..." << std::endl;
  if (!checkStatus(qnn.graphFinalize(graphHandle, nullptr, nullptr), "graphFinalize")) {
    cleanup();
    return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
  }
  std::cout << "[QNN] graphFinalize succeeded" << std::endl;

  // Prepare execution tensors with registered mem handles
  Qnn_Tensor_t execIn0 = input0Tensor;
  execIn0.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  execIn0.v1.memHandle = input0Buf.memHandle;

  Qnn_Tensor_t execIn1 = input1Tensor;
  execIn1.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  execIn1.v1.memHandle = input1Buf.memHandle;

  Qnn_Tensor_t execOut = outputTensor;
  execOut.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  execOut.v1.memHandle = outputBuf.memHandle;

  Qnn_Tensor_t execInputs[2]  = {execIn0, execIn1};
  Qnn_Tensor_t execOutputs[1] = {execOut};

  // Warmup (3 runs)
  for (int w = 0; w < 3; ++w) {
    if (!checkStatus(qnn.graphExecute(graphHandle,
                                      execInputs, 2,
                                      execOutputs, 1,
                                      nullptr, nullptr),
                     "graphExecute warmup")) {
      cleanup();
      return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
    }
  }

  // Timed run: 3 iterations, take average
  constexpr int kNumRuns = 3;
  double totalMs = 0.0;
  for (int run = 0; run < kNumRuns; ++run) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto execStatus = qnn.graphExecute(graphHandle,
                                             execInputs, 2,
                                             execOutputs, 1,
                                             nullptr, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    if (!checkStatus(execStatus, "graphExecute")) {
      cleanup();
      return cleanupAndExit(qnn, backendHandle, deviceHandle, contextHandle, backendLibHandle, true, true);
    }
    double runMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    totalMs += runMs;
    std::cout << "  run " << (run + 1) << ": " << runMs << " ms" << std::endl;
  }
  const double avgMs = totalMs / kNumRuns;
  const double totalDataBytes =
      static_cast<double>(kTotalElements) * sizeof(int8_t) * 3.0;  // inA + inB + out
  const double bandwidthGBps =
      (totalDataBytes / (1024.0 * 1024.0 * 1024.0)) / (avgMs / 1000.0);

  // Verify correctness: 1 + 2 = 3
  int maxError = 0;
  auto* outPtr = static_cast<int8_t*>(outputBuf.data);
  const size_t total = static_cast<size_t>(kTotalElements);
  const size_t indices[] = {0, total / 2, total - 1};
  for (size_t idx : indices) {
    int diff = std::abs(static_cast<int>(outPtr[idx]) - 3);
    if (diff > maxError) {
      maxError = diff;
    }
  }
  std::cout << "[QNN] Done. runs=" << kNumRuns
            << " max_error=" << maxError
            << " sample_out=" << static_cast<int>(outPtr[0])
            << " avg_ms=" << avgMs
            << " bandwidth_GBps=" << bandwidthGBps
            << " layout=[" << kBatchSize << "," << kHeight
            << "," << kWidth << "," << kChannels << "]" << std::endl;

  cleanup();
  qnn.contextFree(contextHandle, nullptr);
  if (qnn.deviceFree) {
    qnn.deviceFree(deviceHandle);
  }
  qnn.backendFree(backendHandle);
  dlclose(backendLibHandle);
  return 0;
}
