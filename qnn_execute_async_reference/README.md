# QnnGraph_executeAsync 调用说明

本目录基于 QAIRT 2.42 官方 API 文档整理，说明 **QnnGraph_executeAsync** 的签名、参数与调用方式。  
实际可用性取决于后端：`supported_api.html` 中当前所有后端（含 HTP）对该 API 均为 **NO**。

## API 声明（QnnGraph.h）

```c
Qnn_ErrorHandle_t QnnGraph_executeAsync(
    Qnn_GraphHandle_t    graphHandle,
    const Qnn_Tensor_t*  inputs,
    uint32_t             numInputs,
    Qnn_Tensor_t*        outputs,
    uint32_t             numOutputs,
    Qnn_ProfileHandle_t  profileHandle,
    Qnn_SignalHandle_t   signalHandle,
    Qnn_NotifyFn_t       notifyFn,
    void*                notifyParam
);
```

## 行为摘要（来自文档）

- **异步执行**：将已 finalize 的 graph 放入执行队列，按 FIFO 顺序执行；完成顺序不一定与入队顺序一致。
- **队列满时阻塞**：若执行队列已满，本函数会阻塞直到有空位。
- **通知**：通过 `notifyFn` 在**后端线程**中回调，通知本次执行结束；调用失败时**不会**调用 `notifyFn`。

## 参数说明

| 参数 | 方向 | 说明 |
|------|------|------|
| `graphHandle` | in | 已 finalize 的 graph 句柄 |
| `inputs` | in | 输入 tensor 数组，用于填充 graph 输入 |
| `numInputs` | in | 输入 tensor 数量 |
| `outputs` | out | 输出 tensor 数组，graph 将写入结果 |
| `numOutputs` | in | 输出 tensor 数量 |
| `profileHandle` | in | 性能数据写入的 profile 句柄；不用 profiling 传 **NULL** |
| `signalHandle` | in | 用于控制本次执行（取消/超时等）；不控制传 **NULL**。同一 graph 多次 executeAsync 需用**不同**的 signal |
| `notifyFn` | in | 执行完成时调用的回调；不需要通知传 **NULL** |
| `notifyParam` | in | 传给 `notifyFn` 的用户数据，可用于区分多次异步调用；可为 NULL |

## 回调类型 Qnn_NotifyFn_t

```c
typedef void (*Qnn_NotifyFn_t)(void* notifyParam, Qnn_NotifyStatus_t notifyStatus);
```

- `notifyParam`：即调用 `QnnGraph_executeAsync` 时传入的 `notifyParam`。
- `notifyStatus`：类型为 `Qnn_NotifyStatus_t`，其中包含 `error`（`Qnn_ErrorHandle_t`），表示该次执行的错误码。

**注意**：回调在**后端线程**中执行，优先级 ≤ 调用线程；回调返回并不保证队列已有空位，下一次 `QnnGraph_executeAsync` 仍可能因队列满而阻塞。

## 返回值（常见）

- `QNN_SUCCESS`：已成功入队执行。
- `QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE`：**当前后端不支持异步 graph 执行**（如 HTP 常见情况）。
- 其他：见 API 文档（invalid handle、graph 未 finalize、invalid tensor、signal 占用等）。

## 与 QnnGraph_execute 的对应关系

- **同步**：`QnnGraph_execute(graph, inputs, numInputs, outputs, numOutputs, profileHandle, signalHandle)`，无 `notifyFn`/`notifyParam`，调用在 graph 执行完成后返回。
- **异步**：`QnnGraph_executeAsync(..., profileHandle, signalHandle, notifyFn, notifyParam)`，调用在**入队后**即可能返回，完成通过 `notifyFn` 通知。
- 若同时使用 `graphExecute` 与 `graphExecuteAsync`，二者共用同一队列，入队/执行优先级相同。

## 做对验证（调用前与调用后）

### 1. 调用前：确认后端是否支持异步执行

- 使用 **QnnGraph_getProperty** 查询 property：**QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION**（文档：determines support for QnnGraph_executeAsync）。
- 若不支持，调用 `QnnGraph_executeAsync` 会返回 **QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE**；可先查属性再决定用异步还是回退到 `QnnGraph_execute`。
- 参考实现见本目录 `validate_async_support.c`。

### 2. 调用时：校验返回值

- `QNN_SUCCESS`（0）：已成功入队，之后在 `notifyFn` 中确认执行是否真正成功。
- 非 0：本次调用失败，**不会**触发 `notifyFn`；应根据错误码处理（如 UNSUPPORTED_FEATURE 则改用同步）。

### 3. 回调中：校验执行结果

- `notifyFn` 的第二个参数为 `Qnn_NotifyStatus_t`，其中 **`notifyStatus.error`** 表示该次**执行**的错误码。
- 仅当 `notifyStatus.error == QNN_SUCCESS` 时，可认为该次异步执行成功，outputs 有效。

### 4. 验证清单（简要）

| 步骤 | 验证项 | 失败时建议 |
|------|--------|------------|
| 1 | 查询 QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION | 不支持则使用 graphExecute |
| 2 | graph 已 finalize、inputs/outputs 与 tensor ID 正确 | 修正构造/反序列化流程 |
| 3 | executeAsync 返回值 == QNN_SUCCESS | 根据错误码处理，不依赖回调 |
| 4 | notifyFn 内 notifyStatus.error == QNN_SUCCESS | 按业务做重试或报错 |

## 文档出处

- API 详细说明：QAIRT 文档 `QNN/api-rst/function_QnnGraph_8h_1a690b74571029dd9f36e38cd902c86784.html`
- API 总览（同步/异步）：`QNN/general/api_overview.html` → “Graph execution”
- 后端支持表：`QNN/general/supported_api.html`（QnnGraph_executeAsync 行）
- 能力属性（ASYNC_EXECUTION）：`QNN/general/supported_capabilities.html`（QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION）
