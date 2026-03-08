#include "qnn_setup.h"
#include "common.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdint>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"

const QNN_INTERFACE_VER_TYPE* g_qnn      = nullptr;
Qnn_ContextHandle_t          g_context   = nullptr;
Qnn_BackendHandle_t          g_backend   = nullptr;
Qnn_DeviceHandle_t           g_device    = nullptr;
uint32_t                     g_coreCount = 0;

namespace {

void*           g_libHandle = nullptr;
Qnn_LogHandle_t g_log       = nullptr;

void qnnLogCallback(const char* fmt, QnnLog_Level_t level,
                     uint64_t /*timestamp*/, va_list args) {
  if (level != QNN_LOG_LEVEL_ERROR) return;
  printf("[QNN-E] ");
  vprintf(fmt, args);
  printf("\n");
}

bool check(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) {
    printf("[QNN] %s failed: %lu\n", w, (unsigned long)s);
    return false;
  }
  return true;
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

}  // namespace

bool qnn_init(uint32_t async_queue_depth) {
  g_libHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_libHandle) { printf("[QNN] dlopen failed: %s\n", dlerror()); return false; }

  using GetProvidersFn = decltype(&QnnInterface_getProviders);
  auto getProviders = reinterpret_cast<GetProvidersFn>(
      dlsym(g_libHandle, "QnnInterface_getProviders"));
  if (!getProviders) { printf("[QNN] getProviders not found\n"); return false; }

  const QnnInterface_t** providers = nullptr;
  uint32_t numProviders = 0;
  if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0) {
    printf("[QNN] No providers\n"); return false;
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
      printf("[QNN] deviceCreate failed: %lu\n", (unsigned long)s); return false;
    }
  }

  setHighPerformanceMode();

  // Build context config (optional async queue depth)
  QnnContext_Config_t asyncCfg = QNN_CONTEXT_CONFIG_INIT;
  const QnnContext_Config_t* ctxConfigs[2] = {nullptr, nullptr};
  if (async_queue_depth > 0) {
    asyncCfg.option = QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH;
    asyncCfg.asyncExeQueueDepth.type = QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_NUMERIC;
    asyncCfg.asyncExeQueueDepth.depth = async_queue_depth;
    ctxConfigs[0] = &asyncCfg;
    printf("[QNN] Async queue depth: %u\n", async_queue_depth);
  }
  const QnnContext_Config_t** cfgPtr = (ctxConfigs[0] ? ctxConfigs : nullptr);
  if (!check(g_qnn->contextCreate(g_backend, g_device, cfgPtr, &g_context), "contextCreate"))
    return false;

  g_coreCount = queryCoreCount();
  return true;
}

void qnn_cleanup() {
  if (g_qnn && g_context) g_qnn->contextFree(g_context, nullptr);
  if (g_qnn && g_device && g_qnn->deviceFree) g_qnn->deviceFree(g_device);
  if (g_qnn && g_backend) g_qnn->backendFree(g_backend);
  if (g_qnn && g_log && g_qnn->logFree) g_qnn->logFree(g_log);
  if (g_libHandle) dlclose(g_libHandle);
  g_context = nullptr; g_device = nullptr; g_backend = nullptr;
  g_qnn = nullptr; g_libHandle = nullptr; g_log = nullptr;
}

bool qnn_context_get_binary(std::vector<uint8_t>& out_binary) {
  Qnn_ContextBinarySize_t binSize = 0;
  if (!check(g_qnn->contextGetBinarySize(g_context, &binSize), "contextGetBinarySize"))
    return false;
  out_binary.resize(binSize);
  Qnn_ContextBinarySize_t written = 0;
  if (!check(g_qnn->contextGetBinary(g_context, out_binary.data(), binSize, &written),
             "contextGetBinary"))
    return false;
  out_binary.resize(written);
  printf("[QNN] Context binary: %.1f KB\n", written / 1024.0);
  return true;
}

bool qnn_context_create_from_binary(const std::vector<uint8_t>& binary, double& out_load_us) {
  // Free old context
  if (g_context) {
    g_qnn->contextFree(g_context, nullptr);
    g_context = nullptr;
  }

  double t0 = now_us();
  if (!check(g_qnn->contextCreateFromBinary(g_backend, g_device, nullptr,
                                              binary.data(), binary.size(),
                                              &g_context, nullptr),
             "contextCreateFromBinary"))
    return false;
  double t1 = now_us();
  out_load_us = t1 - t0;
  return true;
}
