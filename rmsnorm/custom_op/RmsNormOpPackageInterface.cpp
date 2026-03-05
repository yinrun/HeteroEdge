//=============================================================================
//  Custom RMSNorm HTP Op Package - Interface
//  Implements RMSNorm using HVX intrinsics for Qualcomm Hexagon V81
//=============================================================================

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpDef.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()

DECLARE_PKG_OPS_OPTS_LIST(PKG_RmsNorm)

END_PKG_OPS_OPTS_LIST()

// Package info
static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static constexpr auto sg_opNameRmsNorm = "RmsNorm";
static std::array<const char *, 1> sg_opNames{{sg_opNameRmsNorm}};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static Qnn_Version_t sg_opsetVersion = {
    QNN_OPSET_VERSION_MAJOR, QNN_OPSET_VERSION_MINOR, QNN_OPSET_VERSION_PATCH};
static QnnOpPackage_Info_t sg_packageInfo = {sg_packageName,
                                              sg_opNames.data(),
                                              nullptr,
                                              sg_opNames.size(),
                                              nullptr,
                                              0,
                                              QNN_SDK_BUILD_ID,
                                              &sg_sdkApiVersion,
                                              nullptr,
                                              &sg_opsetVersion,
                                              {0}};

static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_packageInitialized = false;
static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel = (QnnLog_Level_t)0;
static bool sg_logInitialized = false;

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()

INIT_PKG_CORE_INIT_FUNC()

Qnn_ErrorHandle_t rmsnormInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  REGISTER_PACKAGE_PARAM_ORDERS()
  REGISTER_PACKAGE_AXIS_PARAMS()
  REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()
  sg_globalInfra = infrastructure;
  sg_packageInitialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t rmsnormGetInfo(const QnnOpPackage_Info_t **info) {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  *info = &sg_packageInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t rmsnormLogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
  if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_logCallback = callback;
  sg_maxLogLevel = maxLogLevel;
  sg_logInitialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t rmsnormLogSetLevel(QnnLog_Level_t maxLogLevel) {
  if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
  sg_maxLogLevel = maxLogLevel;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t rmsnormLogTerminate() {
  sg_logCallback = nullptr;
  sg_maxLogLevel = (QnnLog_Level_t)0;
  sg_logInitialized = false;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t rmsnormValidateOpConfig(Qnn_OpConfig_t opConfig) {
  if (std::string(sg_packageName) != opConfig.v1.packageName) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  if (std::string(opConfig.v1.typeName) == sg_opNameRmsNorm) {
    // RmsNorm: 2 inputs (data, gamma), 1 output, 1 param (epsilon)
    if (opConfig.v1.numOfInputs != 2 || opConfig.v1.numOfOutputs != 1 ||
        opConfig.v1.numOfParams > 1)
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  } else {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t rmsnormCreateOpImpl(QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                       QnnOpPackage_Node_t node,
                                       QnnOpPackage_OpImpl_t *opImpl) {
  (void)graphInfrastructure;
  (void)node;
  (void)opImpl;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t rmsnormFreeOpImpl(QnnOpPackage_OpImpl_t opImpl) {
  (void)opImpl;
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t rmsnormTerminate() {
  if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  sg_globalInfra = nullptr;
  sg_packageInitialized = false;
  return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t rmsnormInterfaceProvider(QnnOpPackage_Interface_t *interface) {
  if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  interface->interfaceVersion = {1, 4, 0};
  interface->v1_4.init = rmsnormInit;
  interface->v1_4.terminate = rmsnormTerminate;
  interface->v1_4.getInfo = rmsnormGetInfo;
  interface->v1_4.validateOpConfig = rmsnormValidateOpConfig;
  interface->v1_4.createOpImpl = rmsnormCreateOpImpl;
  interface->v1_4.freeOpImpl = rmsnormFreeOpImpl;
  interface->v1_4.logInitialize = rmsnormLogInitialize;
  interface->v1_4.logSetLevel = rmsnormLogSetLevel;
  interface->v1_4.logTerminate = rmsnormLogTerminate;
  return QNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
