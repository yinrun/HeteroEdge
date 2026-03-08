/**
 * QnnGraph_executeAsync 做对验证示例
 *
 * 演示：
 * 1) 调用前：通过 QnnGraph_getProperty + QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION 查询是否支持异步
 * 2) 调用后：校验 executeAsync 返回值
 * 3) 回调中：校验 notifyStatus.error，确认本次执行是否成功
 *
 * 类型与常量名与 QAIRT 文档一致，需在真实工程中链接 QNN 头库后使用。
 */

#include <stdint.h>
#include <stddef.h>

/* -------- 与文档一致的符号（示意）-------- */
typedef struct QnnGraph_Opaque*  Qnn_GraphHandle_t;
typedef struct QnnTensor_Opaque  Qnn_Tensor_t;
typedef struct QnnProfile_Opaque* Qnn_ProfileHandle_t;
typedef struct QnnSignal_Opaque*  Qnn_SignalHandle_t;
typedef uint32_t                 Qnn_ErrorHandle_t;

/* 文档：QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION - determines support for QnnGraph_executeAsync
 * 官方定义：(QNN_PROPERTY_GROUP_GRAPH + 3)，见 QnnProperty.h */
#ifndef QNN_PROPERTY_GROUP_GRAPH
#define QNN_PROPERTY_GROUP_GRAPH 0
#endif
#define QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION  (QNN_PROPERTY_GROUP_GRAPH + 3)

#define QNN_SUCCESS 0
#define QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE 1000  /* 常见错误码，以实际 QnnCommon.h 为准 */

typedef struct {
  Qnn_ErrorHandle_t error;
} Qnn_NotifyStatus_t;

typedef void (*Qnn_NotifyFn_t)(void* notifyParam, Qnn_NotifyStatus_t notifyStatus);

/* 假设：通过 QnnInterface 拿到的 graphGetProperty、graphExecuteAsync */
typedef Qnn_ErrorHandle_t (*GraphGetPropertyFn)(Qnn_GraphHandle_t graph, const void* const* propList, uint32_t count);
typedef Qnn_ErrorHandle_t (*GraphExecuteAsyncFn)(
    Qnn_GraphHandle_t   graphHandle,
    const Qnn_Tensor_t* inputs,
    uint32_t            numInputs,
    Qnn_Tensor_t*       outputs,
    uint32_t            numOutputs,
    Qnn_ProfileHandle_t profileHandle,
    Qnn_SignalHandle_t  signalHandle,
    Qnn_NotifyFn_t      notifyFn,
    void*               notifyParam
);

/* -------- 1) 调用前：验证后端是否支持 executeAsync -------- */

/**
 * 查询当前 graph 所在后端是否支持 QnnGraph_executeAsync。
 * 使用 QnnGraph_getProperty + QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION。
 * 返回 1 表示支持，0 表示不支持或查询失败。
 */
static int check_async_supported(GraphGetPropertyFn graphGetProperty, Qnn_GraphHandle_t graph) {
  /* 实际使用时需按 QnnProperty 文档构造 Qnn_PropertyHandle_t 等并传入 propList */
  (void)graphGetProperty;
  (void)graph;
  /* 示意：若 getProperty 返回支持则用异步，否则回退同步 */
  return 0;  /* 占位：在真实工程中根据 getProperty 结果返回 0/1 */
}

/* -------- 2) 调用后：校验 executeAsync 返回值 -------- */

/**
 * 调用 executeAsync 后必须检查返回值。
 * 非 QNN_SUCCESS 时不会调用 notifyFn，应在此处做错误处理或回退到 graphExecute。
 */
static int validate_execute_async_return(Qnn_ErrorHandle_t err) {
  if (err == QNN_SUCCESS)
    return 1;  /* 已入队，等待 notifyFn 报告执行结果 */
  if (err == QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE)
    return 0;  /* 后端不支持异步，应改用 QnnGraph_execute */
  return -1;   /* 其他错误：invalid handle、未 finalize、tensor 错误等 */
}

/* -------- 3) 回调中：校验本次执行是否成功 -------- */

typedef struct {
  int   request_id;
  int   callback_fired;   /* 验证：是否收到回调 */
  int   execution_ok;     /* 验证：notifyStatus.error == QNN_SUCCESS */
} VerifyNotifyParam;

static void verify_callback(void* notifyParam, Qnn_NotifyStatus_t notifyStatus) {
  VerifyNotifyParam* p = (VerifyNotifyParam*)notifyParam;
  p->callback_fired = 1;
  p->execution_ok   = (notifyStatus.error == QNN_SUCCESS);
}

/* -------- 串联：带验证的异步调用示例 -------- */

/**
 * 做对验证流程：
 * - 调用前：若不支持异步则直接返回，建议调用方改用 graphExecute。
 * - 调用：执行 graphExecuteAsync，并校验返回值。
 * - 回调：在 notifyFn 中校验 notifyStatus.error，用于确认该次执行是否成功。
 */
static void execute_async_with_validation(
    GraphGetPropertyFn   graphGetProperty,
    GraphExecuteAsyncFn  graphExecuteAsync,
    Qnn_GraphHandle_t    graph,
    const Qnn_Tensor_t*  inputs,
    uint32_t             numInputs,
    Qnn_Tensor_t*        outputs,
    uint32_t             numOutputs,
    VerifyNotifyParam*   notifyParam)
{
  notifyParam->callback_fired = 0;
  notifyParam->execution_ok   = 0;

  if (!check_async_supported(graphGetProperty, graph)) {
    /* 验证失败：不支持异步，应使用 QnnGraph_execute */
    return;
  }

  Qnn_ErrorHandle_t err = graphExecuteAsync(
      graph,
      inputs,  numInputs,
      outputs, numOutputs,
      NULL, NULL,
      verify_callback,
      notifyParam
  );

  int valid = validate_execute_async_return(err);
  if (valid != 1) {
    /* 验证失败：入队未成功，不依赖回调；若 valid==0 可回退 graphExecute */
    return;
  }

  /* 已入队；最终执行结果在 verify_callback 中通过 notifyParam->execution_ok 确认 */
  (void)valid;
}
