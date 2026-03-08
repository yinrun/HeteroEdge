#pragma once

#include "QNN/QnnInterface.h"
#include "QNN/QnnContext.h"
#include <cstdint>
#include <vector>

extern const QNN_INTERFACE_VER_TYPE* g_qnn;
extern Qnn_ContextHandle_t g_context;
extern Qnn_BackendHandle_t g_backend;
extern Qnn_DeviceHandle_t  g_device;
extern uint32_t g_coreCount;

bool qnn_init(uint32_t async_queue_depth = 0);
void qnn_cleanup();

// Serialize current context to binary blob
bool qnn_context_get_binary(std::vector<uint8_t>& out_binary);

// Create a new context from binary blob, replacing g_context
// Returns time taken in microseconds via out_load_us
bool qnn_context_create_from_binary(const std::vector<uint8_t>& binary, double& out_load_us);
