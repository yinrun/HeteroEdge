/**
 * QnnGraph_executeAsync 调用示例（参考用，不依赖完整 QNN 工程）
 *
 * 基于 QAIRT 2.42 API 文档整理，展示：
 * 1) 回调函数签名与实现
 * 2) 调用 executeAsync 的典型写法
 * 3) 与同步 graphExecute 的对比
 *
 * 注意：HTP 等后端在 supported_api 中标注为不支持，实际调用可能返回
 * QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE。需通过 QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL
 * 或 supported_capabilities 查询当前后端是否支持。
 */

#include <stdint.h>

/* 以下类型名与 QAIRT QnnGraph.h / QnnCommon.h 一致，仅作示意 */
typedef struct QnnGraph_Opaque*    Qnn_GraphHandle_t;
typedef struct QnnTensor_Opaque   Qnn_Tensor_t;
typedef struct QnnProfile_Opaque* Qnn_ProfileHandle_t;
typedef struct QnnSignal_Opaque*  Qnn_SignalHandle_t;
typedef uint32_t                  Qnn_ErrorHandle_t;

/* 回调入参：执行状态，含 error 码 */
typedef struct {
  Qnn_ErrorHandle_t error;
} Qnn_NotifyStatus_t;

/* 完成通知回调：在 backend 线程中调用 */
typedef void (*Qnn_NotifyFn_t)(void* notifyParam, Qnn_NotifyStatus_t notifyStatus);

/* 假设通过 QnnInterface 获取到的函数指针 */
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

/* ---------------------------------------------------------------------------
 * 示例 1：定义完成回调
 * --------------------------------------------------------------------------- */

typedef struct {
  int   request_id;
  void* user_context;  /* 例如：输出 buffer、event 等 */
} MyNotifyParam;

static void my_execute_done_callback(void* notifyParam, Qnn_NotifyStatus_t notifyStatus) {
  MyNotifyParam* p = (MyNotifyParam*)notifyParam;
  /* 做对验证：回调中必须校验 notifyStatus.error，仅 QNN_SUCCESS 表示本次执行成功 */
  if (notifyStatus.error != 0) {
    /* 本次异步执行失败，error 即 Qnn_ErrorHandle_t；outputs 可能无效 */
    return;
  }
  /* 成功：可在此处理 outputs、发 signal、入队下一阶段等 */
  (void)p;
}

/* ---------------------------------------------------------------------------
 * 示例 2：调用 QnnGraph_executeAsync（通过 QnnInterface 的 graphExecuteAsync）
 * --------------------------------------------------------------------------- */

static void example_execute_async(
    GraphExecuteAsyncFn graphExecuteAsync,
    Qnn_GraphHandle_t    graph,
    Qnn_Tensor_t*       input_tensor,
    Qnn_Tensor_t*       output_tensor,
    MyNotifyParam*      notify_param)
{
  Qnn_ErrorHandle_t err;

  /* 不采集 profile、不用 signal 控制时，后四参传 NULL 或 0 */
  err = graphExecuteAsync(
      graph,
      input_tensor,  1,
      output_tensor, 1,
      NULL,          /* profileHandle */
      NULL,          /* signalHandle */
      my_execute_done_callback,
      notify_param
  );

  /* 做对验证：必须校验返回值；非 QNN_SUCCESS 时不会触发 notifyFn */
  if (err != 0) {
    /* 常见：后端不支持时返回 QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE，应回退 graphExecute */
    return;
  }
  /* 已入队；实际执行是否成功需在 my_execute_done_callback 中校验 notifyStatus.error */
}

/* ---------------------------------------------------------------------------
 * 示例 3：与同步调用的对比（同一 graph / tensors）
 * --------------------------------------------------------------------------- */

typedef Qnn_ErrorHandle_t (*GraphExecuteFn)(
    Qnn_GraphHandle_t   graphHandle,
    const Qnn_Tensor_t* inputs,
    uint32_t            numInputs,
    Qnn_Tensor_t*       outputs,
    uint32_t            numOutputs,
    Qnn_ProfileHandle_t profileHandle,
    Qnn_SignalHandle_t  signalHandle
);

static Qnn_ErrorHandle_t example_execute_sync(
    GraphExecuteFn      graphExecute,
    Qnn_GraphHandle_t   graph,
    Qnn_Tensor_t*       input_tensor,
    Qnn_Tensor_t*       output_tensor)
{
  /* 同步：函数返回时执行已完成 */
  return graphExecute(graph, input_tensor, 1, output_tensor, 1, NULL, NULL);
}
