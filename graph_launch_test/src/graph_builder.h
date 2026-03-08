#pragma once

#include "common.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnMem.h"
#include "QNN/QnnTensor.h"
#include <vector>
#include <string>

// ─── Legacy (ElementWiseMultiply chain) ───

struct FusedResult {
  Qnn_GraphHandle_t graph = nullptr;
  IonBuffer ion_in, ion_out;
  Qnn_MemHandle_t reg_in  = nullptr;
  Qnn_MemHandle_t reg_out = nullptr;
  Qnn_Tensor_t exec_in;
  Qnn_Tensor_t exec_out;
  double create_us = 0;
  char graph_name[64] = {};
  // Tensor IDs assigned by tensorCreateGraphTensor — needed for cache restore
  uint32_t input_tensor_id = 0;
  uint32_t output_tensor_id = 0;
};

struct SplitResult {
  std::vector<Qnn_GraphHandle_t> graphs;
  std::vector<IonBuffer> ion_bufs;
  std::vector<Qnn_MemHandle_t> reg_handles;
  std::vector<Qnn_Tensor_t> exec_ins;
  std::vector<Qnn_Tensor_t> exec_outs;
  double create_us = 0;
};

bool build_fused_graph(int num_ops, int hidden_dim, FusedResult& out);
bool build_split_graphs(int num_ops, int hidden_dim, SplitResult& out);
void cleanup_fused(FusedResult& r);
void cleanup_split(SplitResult& r);
bool retrieve_fused_from_cache(int num_ops, int hidden_dim,
                               const char* graph_name,
                               uint32_t input_tensor_id, uint32_t output_tensor_id,
                               FusedResult& out);

// ─── FFN block benchmark (RMSNorm → MatMul → GELU → MatMul → Add) ───

enum class SplitMode { FUSED, BLOCK_2, PER_OP };

struct FFNConfig {
  int hidden_dim = 512;
  int intermediate_dim = 1376;
};

struct FFNResult {
  std::vector<Qnn_GraphHandle_t> graphs;
  std::vector<IonBuffer> ion_bufs;           // boundary ION buffers
  std::vector<Qnn_MemHandle_t> reg_handles;
  struct GraphExec {
    std::vector<Qnn_Tensor_t> ins;
    std::vector<Qnn_Tensor_t> outs;
  };
  std::vector<GraphExec> graph_execs;
  double create_us = 0;
  int num_graphs = 0;
  IonBuffer weight_buf;                      // all static weights
  std::vector<std::string> graph_names;
  // Persistent dims arrays — tensor structs point to these, must outlive execution
  uint32_t dims_h4[4] = {1, 1, 1, 0};       // [1,1,1,H]
  uint32_t dims_i4[4] = {1, 1, 1, 0};       // [1,1,1,I]
};

const char* split_mode_name(SplitMode mode);
bool build_ffn_block(SplitMode mode, const FFNConfig& cfg, FFNResult& out);
void cleanup_ffn(FFNResult& r);

// Build FFN fused, serialize to context binary, then restore from cache.
// Returns the cached FFNResult ready for execution, plus cache load time.
bool build_ffn_cached(const FFNConfig& cfg, FFNResult& out, double& out_cache_load_us);
