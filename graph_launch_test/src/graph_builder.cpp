#include "graph_builder.h"
#include "qnn_setup.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "QNN/QnnGraph.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnOpDef.h"
#include "QNN/QnnMem.h"
#include "QNN/QnnTensor.h"
#include "QNN/HTP/QnnHtpGraph.h"

static constexpr uint32_t kTensorRank = 4;
static uint32_t g_graphSeq = 0;  // unique graph name counter

static bool check(Qnn_ErrorHandle_t s, const char* w) {
  if (s != QNN_SUCCESS) {
    printf("[Graph] %s failed: %lu\n", w, (unsigned long)s);
    return false;
  }
  return true;
}

static bool registerIon(const IonBuffer& ion, const uint32_t* dims, uint32_t ndims,
                        Qnn_MemHandle_t& handle) {
  Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
  desc.memShape.numDim  = ndims;
  desc.memShape.dimSize = const_cast<uint32_t*>(dims);
  desc.dataType         = QNN_DATATYPE_FLOAT_16;
  desc.memType          = QNN_MEM_TYPE_ION;
  desc.ionInfo.fd       = ion.fd;
  if (QNN_SUCCESS != g_qnn->memRegister(g_context, &desc, 1, &handle)) {
    printf("[Graph] memRegister failed (fd=%d)\n", ion.fd);
    return false;
  }
  return true;
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

namespace {

bool createGraphWithHtpConfig(const char* name, Qnn_GraphHandle_t& graph) {
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

  return check(g_qnn->graphCreate(g_context, name, graphCfgList, &graph), "graphCreate");
}

bool addMulNode(Qnn_GraphHandle_t graph, const char* nodeName,
                Qnn_Tensor_t& input, Qnn_Tensor_t& output) {
  Qnn_Tensor_t opIn[]  = {input, input};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = nodeName;
  op.v1.packageName  = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName     = QNN_OP_ELEMENT_WISE_MULTIPLY;
  op.v1.numOfParams  = 0; op.v1.params = nullptr;
  op.v1.numOfInputs  = 2; op.v1.inputTensors  = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;

  return check(g_qnn->graphAddNode(graph, op), nodeName);
}

}  // namespace

bool build_fused_graph(int num_ops, int hidden_dim, FusedResult& out) {
  size_t buf_bytes = (size_t)hidden_dim * 2;  // FP16
  uint32_t dims[kTensorRank] = {1, 1, 1, (uint32_t)hidden_dim};

  // Allocate ION buffers
  if (!allocIonBuffer(buf_bytes, 0, out.ion_in) ||
      !allocIonBuffer(buf_bytes, 0, out.ion_out)) {
    printf("[Fused] ION alloc failed\n"); return false;
  }
  // Fill input with 0.5 in FP16
  uint16_t half_val = float_to_half(0.5f);
  uint16_t* inp = reinterpret_cast<uint16_t*>(out.ion_in.ptr);
  for (int i = 0; i < hidden_dim; ++i) inp[i] = half_val;

  // Register ION
  if (!registerIon(out.ion_in, dims, kTensorRank, out.reg_in) ||
      !registerIon(out.ion_out, dims, kTensorRank, out.reg_out))
    return false;

  // --- Timed: graph creation ---
  double t0 = now_us();

  Qnn_GraphHandle_t graph = nullptr;
  snprintf(out.graph_name, sizeof(out.graph_name), "fused_%u", g_graphSeq++);
  if (!createGraphWithHtpConfig(out.graph_name, graph)) return false;

  // Tensor name storage (must outlive graph usage)
  // We use static vectors since we only build one fused graph at a time
  static std::vector<std::string> tensor_names;
  static std::vector<std::string> node_names;
  tensor_names.clear();
  node_names.clear();

  // Create all tensors
  std::vector<Qnn_Tensor_t> tensors;  // t_0 (input), t_1 ... t_{N-1} (intermediates), t_N (output)
  // Need persistent dims array
  static uint32_t s_dims[kTensorRank];
  s_dims[0] = 1; s_dims[1] = 1; s_dims[2] = 1; s_dims[3] = (uint32_t)hidden_dim;

  for (int i = 0; i <= num_ops; ++i) {
    char name[32];
    snprintf(name, sizeof(name), "t_%d", i);
    tensor_names.push_back(name);

    Qnn_TensorType_t type;
    if (i == 0) type = QNN_TENSOR_TYPE_APP_WRITE;
    else if (i == num_ops) type = QNN_TENSOR_TYPE_APP_READ;
    else type = QNN_TENSOR_TYPE_NATIVE;

    Qnn_Tensor_t t = makeFp16Tensor(tensor_names.back().c_str(), type, s_dims);
    if (!check(g_qnn->tensorCreateGraphTensor(graph, &t), tensor_names.back().c_str()))
      return false;
    tensors.push_back(t);
  }

  // Add N Mul nodes
  for (int i = 0; i < num_ops; ++i) {
    char name[32];
    snprintf(name, sizeof(name), "mul_%d", i);
    node_names.push_back(name);
    if (!addMulNode(graph, node_names.back().c_str(), tensors[i], tensors[i + 1]))
      return false;
  }

  if (!check(g_qnn->graphFinalize(graph, nullptr, nullptr), "graphFinalize"))
    return false;

  double t1 = now_us();
  out.create_us = t1 - t0;
  out.graph = graph;

  // Save tensor IDs for cache restore
  out.input_tensor_id  = tensors[0].v1.id;
  out.output_tensor_id = tensors[num_ops].v1.id;

  // Bind ION handles for execution
  out.exec_in  = tensors[0];
  out.exec_out = tensors[num_ops];
  out.exec_in.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  out.exec_in.v1.memHandle = out.reg_in;
  out.exec_out.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  out.exec_out.v1.memHandle = out.reg_out;

  return true;
}

bool build_split_graphs(int num_ops, int hidden_dim, SplitResult& out) {
  size_t buf_bytes = (size_t)hidden_dim * 2;
  uint32_t dims[kTensorRank] = {1, 1, 1, (uint32_t)hidden_dim};

  // Allocate N+1 ION buffers
  out.ion_bufs.resize(num_ops + 1);
  for (int i = 0; i <= num_ops; ++i) {
    if (!allocIonBuffer(buf_bytes, 0, out.ion_bufs[i])) {
      printf("[Split] ION alloc failed for buf %d\n", i); return false;
    }
  }
  // Fill first input
  uint16_t half_val = float_to_half(0.5f);
  uint16_t* inp = reinterpret_cast<uint16_t*>(out.ion_bufs[0].ptr);
  for (int i = 0; i < hidden_dim; ++i) inp[i] = half_val;

  // Register all ION buffers
  static uint32_t s_dims[kTensorRank];
  s_dims[0] = 1; s_dims[1] = 1; s_dims[2] = 1; s_dims[3] = (uint32_t)hidden_dim;

  out.reg_handles.resize(num_ops + 1, nullptr);
  for (int i = 0; i <= num_ops; ++i) {
    if (!registerIon(out.ion_bufs[i], s_dims, kTensorRank, out.reg_handles[i]))
      return false;
  }

  // Persistent name storage
  static std::vector<std::string> graph_names;
  static std::vector<std::string> tensor_in_names;
  static std::vector<std::string> tensor_out_names;
  static std::vector<std::string> node_names;
  graph_names.clear();
  tensor_in_names.clear();
  tensor_out_names.clear();
  node_names.clear();

  out.graphs.resize(num_ops, nullptr);
  out.exec_ins.resize(num_ops);
  out.exec_outs.resize(num_ops);

  // --- Timed: graph creation ---
  double t0 = now_us();

  for (int g = 0; g < num_ops; ++g) {
    char gname[32], iname[32], oname[32], nname[32];
    snprintf(gname, sizeof(gname), "split_%u_%d", g_graphSeq++, g);
    snprintf(iname, sizeof(iname), "in_%d", g);
    snprintf(oname, sizeof(oname), "out_%d", g);
    snprintf(nname, sizeof(nname), "mul_%d", g);
    graph_names.push_back(gname);
    tensor_in_names.push_back(iname);
    tensor_out_names.push_back(oname);
    node_names.push_back(nname);

    if (!createGraphWithHtpConfig(graph_names.back().c_str(), out.graphs[g]))
      return false;

    Qnn_Tensor_t tin  = makeFp16Tensor(tensor_in_names.back().c_str(),
                                        QNN_TENSOR_TYPE_APP_WRITE, s_dims);
    Qnn_Tensor_t tout = makeFp16Tensor(tensor_out_names.back().c_str(),
                                        QNN_TENSOR_TYPE_APP_READ, s_dims);

    if (!check(g_qnn->tensorCreateGraphTensor(out.graphs[g], &tin),  tensor_in_names.back().c_str()) ||
        !check(g_qnn->tensorCreateGraphTensor(out.graphs[g], &tout), tensor_out_names.back().c_str()))
      return false;

    if (!addMulNode(out.graphs[g], node_names.back().c_str(), tin, tout))
      return false;

    if (!check(g_qnn->graphFinalize(out.graphs[g], nullptr, nullptr), "graphFinalize"))
      return false;

    // Bind ION handles
    tin.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
    tin.v1.memHandle = out.reg_handles[g];
    tout.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
    tout.v1.memHandle = out.reg_handles[g + 1];

    out.exec_ins[g]  = tin;
    out.exec_outs[g] = tout;
  }

  double t1 = now_us();
  out.create_us = t1 - t0;

  return true;
}

void cleanup_fused(FusedResult& r) {
  std::vector<Qnn_MemHandle_t> handles;
  if (r.reg_in)  handles.push_back(r.reg_in);
  if (r.reg_out) handles.push_back(r.reg_out);
  if (!handles.empty() && g_qnn && g_qnn->memDeRegister)
    g_qnn->memDeRegister(handles.data(), (uint32_t)handles.size());
  r.reg_in = r.reg_out = nullptr;

  freeIonBuffer(r.ion_in);
  freeIonBuffer(r.ion_out);
  r.graph = nullptr;
}

void cleanup_split(SplitResult& r) {
  std::vector<Qnn_MemHandle_t> handles;
  for (auto h : r.reg_handles)
    if (h) handles.push_back(h);
  if (!handles.empty() && g_qnn && g_qnn->memDeRegister)
    g_qnn->memDeRegister(handles.data(), (uint32_t)handles.size());
  r.reg_handles.clear();

  for (auto& buf : r.ion_bufs)
    freeIonBuffer(buf);
  r.ion_bufs.clear();
  r.graphs.clear();
  r.exec_ins.clear();
  r.exec_outs.clear();
}

// ─── FFN block helpers ──────────────────────────────────────────────────────

static Qnn_Tensor_t makeStaticFp16Tensor(const char* name, uint32_t* dims,
                                          void* data, uint32_t data_bytes) {
  Qnn_Tensor_t t = makeFp16Tensor(name, QNN_TENSOR_TYPE_STATIC, dims);
  t.v1.clientBuf.data     = data;
  t.v1.clientBuf.dataSize = data_bytes;
  return t;
}

// axes tensor for RMSNorm (rank-1, UINT32, STATIC)
static bool addAxesTensor(Qnn_GraphHandle_t graph, const char* name,
                          uint32_t* axes_data, uint32_t num_axes,
                          uint32_t* dims_storage, Qnn_Tensor_t& out) {
  dims_storage[0] = num_axes;
  out = QNN_TENSOR_INIT;
  out.version = QNN_TENSOR_VERSION_1;
  out.v1.name = name;
  out.v1.type = QNN_TENSOR_TYPE_STATIC;
  out.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  out.v1.dataType = QNN_DATATYPE_UINT_32;
  out.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  out.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  out.v1.rank = 1;
  out.v1.dimensions = dims_storage;
  out.v1.memType = QNN_TENSORMEMTYPE_RAW;
  out.v1.clientBuf.data = axes_data;
  out.v1.clientBuf.dataSize = num_axes * sizeof(uint32_t);
  return check(g_qnn->tensorCreateGraphTensor(graph, &out), name);
}

static bool addRmsNormNode(Qnn_GraphHandle_t graph, const char* name,
                           Qnn_Tensor_t& input, Qnn_Tensor_t& gamma,
                           Qnn_Tensor_t& beta, Qnn_Tensor_t& output,
                           uint32_t* axes_dims_storage) {
  // axes tensor (normalize over dim 3 for rank-4 tensors [1,1,1,H])
  static uint32_t axes_data[] = {3};
  Qnn_Tensor_t axes_tensor;
  char axes_name[64];
  snprintf(axes_name, sizeof(axes_name), "%s_axes", name);
  if (!addAxesTensor(graph, axes_name, axes_data, 1, axes_dims_storage, axes_tensor))
    return false;

  Qnn_Param_t params[2];
  // epsilon
  params[0] = QNN_PARAM_INIT;
  params[0].paramType = QNN_PARAMTYPE_SCALAR;
  params[0].name = QNN_OP_RMS_NORM_PARAM_EPSILON;
  params[0].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  params[0].scalarParam.floatValue = 1e-6f;
  // axes
  params[1] = QNN_PARAM_INIT;
  params[1].paramType = QNN_PARAMTYPE_TENSOR;
  params[1].name = QNN_OP_RMS_NORM_PARAM_AXES;
  params[1].tensorParam = axes_tensor;

  Qnn_Tensor_t opIn[] = {input, gamma, beta};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_RMS_NORM;
  op.v1.numOfParams = 2; op.v1.params = params;
  op.v1.numOfInputs = 3; op.v1.inputTensors = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;
  return check(g_qnn->graphAddNode(graph, op), name);
}

static bool addFCNode(Qnn_GraphHandle_t graph, const char* name,
                      Qnn_Tensor_t& input, Qnn_Tensor_t& weight, Qnn_Tensor_t& output) {
  // Note: HTP only supports keep_dims=0 (default), so no params needed
  Qnn_Tensor_t opIn[] = {input, weight};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_FULLY_CONNECTED;
  op.v1.numOfParams = 0; op.v1.params = nullptr;
  op.v1.numOfInputs = 2; op.v1.inputTensors = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;
  return check(g_qnn->graphAddNode(graph, op), name);
}

static bool addGeluNode(Qnn_GraphHandle_t graph, const char* name,
                        Qnn_Tensor_t& input, Qnn_Tensor_t& output) {
  Qnn_Tensor_t opIn[] = {input};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_GELU;
  op.v1.numOfParams = 0; op.v1.params = nullptr;
  op.v1.numOfInputs = 1; op.v1.inputTensors = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;
  return check(g_qnn->graphAddNode(graph, op), name);
}

static bool addAddNode(Qnn_GraphHandle_t graph, const char* name,
                       Qnn_Tensor_t& a, Qnn_Tensor_t& b, Qnn_Tensor_t& output) {
  Qnn_Tensor_t opIn[] = {a, b};
  Qnn_Tensor_t opOut[] = {output};

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = name;
  op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op.v1.typeName = QNN_OP_ELEMENT_WISE_ADD;
  op.v1.numOfParams = 0; op.v1.params = nullptr;
  op.v1.numOfInputs = 2; op.v1.inputTensors = opIn;
  op.v1.numOfOutputs = 1; op.v1.outputTensors = opOut;
  return check(g_qnn->graphAddNode(graph, op), name);
}

const char* split_mode_name(SplitMode mode) {
  switch (mode) {
    case SplitMode::FUSED:   return "Fused";
    case SplitMode::BLOCK_2: return "2-Block";
    case SplitMode::PER_OP:  return "Per-Op";
  }
  return "?";
}

// Build a single graph containing the specified ops.
// ops_begin..ops_end are op indices [0..4].
// Boundary tensors (graph input/output) get ION backing; internal tensors are NATIVE.
//
// FFN block topology:
//   Op 0: RMSNorm(x, gamma)     → norm   [1,1,1,H]
//   Op 1: MatMul(norm, W_up)    → hidden [1,1,1,I]
//   Op 2: GELU(hidden)          → act    [1,1,1,I]
//   Op 3: MatMul(act, W_down)   → proj   [1,1,1,H]
//   Op 4: Add(x, proj)          → output [1,1,1,H]
//
// Tensor names (for the full graph): x, norm, hidden, act, proj, output
// x is also used by op 4 (residual), so in split modes x must be available to both op 0 and op 4.

struct FFNWeights {
  // Pointers into weight_buf
  void* gamma;       // [H] rank-1
  void* beta;        // [H] rank-1 (zeros)
  void* w_up;        // [I,H] rank-2 FC weight
  void* w_down;      // [H,I] rank-2 FC weight
  // Weight dims (only used during graph build for STATIC tensors, safe as local)
  uint32_t dims_wup[2];           // [I,H] rank-2 FC weight
  uint32_t dims_wdn[2];           // [H,I] rank-2 FC weight
  uint32_t dims_gamma1d[1];       // [H] rank-1 for gamma/beta
  uint32_t axes_dims[1];          // for RMSNorm axes tensor
};

static bool allocWeights(const FFNConfig& cfg, IonBuffer& buf, FFNWeights& w) {
  int H = cfg.hidden_dim, I = cfg.intermediate_dim;
  size_t gamma_bytes = (size_t)H * 2;
  size_t beta_bytes  = (size_t)H * 2;
  size_t wup_bytes   = (size_t)H * I * 2;
  size_t wdn_bytes   = (size_t)I * H * 2;
  size_t total = gamma_bytes + beta_bytes + wup_bytes + wdn_bytes;

  if (!allocIonBuffer(total, 0, buf)) {
    printf("[FFN] weight alloc failed (%zu bytes)\n", total);
    return false;
  }

  // Fill with small random FP16 values
  fillRandomFp16(buf.ptr, total / 2);

  // Set pointers into the buffer
  uint8_t* base = reinterpret_cast<uint8_t*>(buf.ptr);
  w.gamma  = base;
  w.beta   = base + gamma_bytes;
  w.w_up   = base + gamma_bytes + beta_bytes;
  w.w_down = base + gamma_bytes + beta_bytes + wup_bytes;

  // Set gamma to 1.0
  uint16_t one = float_to_half(1.0f);
  uint16_t* gp = reinterpret_cast<uint16_t*>(w.gamma);
  for (int i = 0; i < H; ++i) gp[i] = one;

  // Set beta to 0.0
  std::memset(w.beta, 0, beta_bytes);

  // FC weight: [out_features, in_features] (rank-2)
  w.dims_wup[0] = (uint32_t)I; w.dims_wup[1] = (uint32_t)H;
  w.dims_wdn[0] = (uint32_t)H; w.dims_wdn[1] = (uint32_t)I;
  w.dims_gamma1d[0] = (uint32_t)H;
  return true;
}

// Build all ops into a single graph.
// x_tensor and output_tensor are APP_WRITE/APP_READ (rank-4); all others are NATIVE.
static bool buildFFNSingleGraph(Qnn_GraphHandle_t graph, const FFNConfig& cfg,
                                FFNWeights& w, uint32_t* dims_h4, uint32_t* dims_i4,
                                Qnn_Tensor_t& x_tensor, Qnn_Tensor_t& out_tensor) {
  // Internal tensors: rank-4 [1,1,1,H] or [1,1,1,I]
  Qnn_Tensor_t norm   = makeFp16Tensor("norm",   QNN_TENSOR_TYPE_NATIVE, dims_h4);
  Qnn_Tensor_t hidden = makeFp16Tensor("hidden", QNN_TENSOR_TYPE_NATIVE, dims_i4);
  Qnn_Tensor_t act    = makeFp16Tensor("act",    QNN_TENSOR_TYPE_NATIVE, dims_i4);
  Qnn_Tensor_t proj   = makeFp16Tensor("proj",   QNN_TENSOR_TYPE_NATIVE, dims_h4);

  // Static weight tensors (gamma/beta rank-1, FC weights rank-2)
  Qnn_Tensor_t gamma = makeFp16Tensor("gamma", QNN_TENSOR_TYPE_STATIC, w.dims_gamma1d, 1);
  gamma.v1.clientBuf.data     = w.gamma;
  gamma.v1.clientBuf.dataSize = (uint32_t)(cfg.hidden_dim * 2);
  Qnn_Tensor_t beta = makeFp16Tensor("beta", QNN_TENSOR_TYPE_STATIC, w.dims_gamma1d, 1);
  beta.v1.clientBuf.data     = w.beta;
  beta.v1.clientBuf.dataSize = (uint32_t)(cfg.hidden_dim * 2);
  Qnn_Tensor_t wup = makeFp16Tensor("w_up", QNN_TENSOR_TYPE_STATIC, w.dims_wup, 2);
  wup.v1.clientBuf.data     = w.w_up;
  wup.v1.clientBuf.dataSize = (uint32_t)((size_t)cfg.hidden_dim * cfg.intermediate_dim * 2);
  Qnn_Tensor_t wdn = makeFp16Tensor("w_down", QNN_TENSOR_TYPE_STATIC, w.dims_wdn, 2);
  wdn.v1.clientBuf.data     = w.w_down;
  wdn.v1.clientBuf.dataSize = (uint32_t)((size_t)cfg.intermediate_dim * cfg.hidden_dim * 2);

  // Register all tensors with graph
  if (!check(g_qnn->tensorCreateGraphTensor(graph, &x_tensor), "x") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &gamma), "gamma") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &beta), "beta") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &wup), "w_up") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &wdn), "w_down") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &norm), "norm") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &hidden), "hidden") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &act), "act") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &proj), "proj") ||
      !check(g_qnn->tensorCreateGraphTensor(graph, &out_tensor), "output"))
    return false;

  // Op 0: RMSNorm
  if (!addRmsNormNode(graph, "rmsnorm", x_tensor, gamma, beta, norm, w.axes_dims))
    return false;
  // Op 1: FC(norm, W_up) → hidden
  if (!addFCNode(graph, "fc_up", norm, wup, hidden))
    return false;
  // Op 2: GELU
  if (!addGeluNode(graph, "gelu", hidden, act))
    return false;
  // Op 3: FC(act, W_down) → proj
  if (!addFCNode(graph, "fc_down", act, wdn, proj))
    return false;
  // Op 4: Add(x, proj) → output
  if (!addAddNode(graph, "residual_add", x_tensor, proj, out_tensor))
    return false;

  return true;
}

// Helper: register ION buffer with rank-4 dims, store handle in out.reg_handles
static bool registerFFNIon(const IonBuffer& ion, const uint32_t* dims4,
                           Qnn_MemHandle_t& handle) {
  return registerIon(ion, dims4, kTensorRank, handle);
}

bool build_ffn_block(SplitMode mode, const FFNConfig& cfg, FFNResult& out) {
  int H = cfg.hidden_dim;
  int I = cfg.intermediate_dim;
  size_t h_bytes = (size_t)H * 2;
  size_t i_bytes = (size_t)I * 2;

  // Initialize persistent dims in FFNResult (tensor structs will point here)
  out.dims_h4[0] = 1; out.dims_h4[1] = 1; out.dims_h4[2] = 1; out.dims_h4[3] = (uint32_t)H;
  out.dims_i4[0] = 1; out.dims_i4[1] = 1; out.dims_i4[2] = 1; out.dims_i4[3] = (uint32_t)I;

  // Allocate weights
  FFNWeights w;
  if (!allocWeights(cfg, out.weight_buf, w))
    return false;

  if (mode == SplitMode::FUSED) {
    // ── Single graph with all 5 ops ──
    out.num_graphs = 1;

    // ION: input [H] + output [H]
    out.ion_bufs.resize(2);
    if (!allocIonBuffer(h_bytes, 0, out.ion_bufs[0]) ||
        !allocIonBuffer(h_bytes, 0, out.ion_bufs[1]))
      return false;
    fillRandomFp16(out.ion_bufs[0].ptr, H);

    double t0 = now_us();

    Qnn_GraphHandle_t graph = nullptr;
    char gname[64];
    snprintf(gname, sizeof(gname), "ffn_%u", g_graphSeq++);
    out.graph_names.push_back(gname);
    if (!createGraphWithHtpConfig(gname, graph))
      return false;
    out.graphs.push_back(graph);

    // Register ION BEFORE graph build (rank-4 dims)
    out.reg_handles.resize(2, nullptr);
    if (!registerFFNIon(out.ion_bufs[0], out.dims_h4, out.reg_handles[0]) ||
        !registerFFNIon(out.ion_bufs[1], out.dims_h4, out.reg_handles[1]))
      return false;

    // Rank-4 I/O tensors
    Qnn_Tensor_t x_t   = makeFp16Tensor("x",      QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
    Qnn_Tensor_t out_t  = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  out.dims_h4);

    if (!buildFFNSingleGraph(graph, cfg, w, out.dims_h4, out.dims_i4, x_t, out_t))
      return false;

    if (!check(g_qnn->graphFinalize(graph, nullptr, nullptr), "graphFinalize"))
      return false;

    double t1 = now_us();
    out.create_us = t1 - t0;

    // Bind ION via memHandle
    x_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
    x_t.v1.memHandle = out.reg_handles[0];
    out_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
    out_t.v1.memHandle = out.reg_handles[1];

    out.graph_execs.push_back({{x_t}, {out_t}});
    return true;
  }

  if (mode == SplitMode::BLOCK_2) {
    // ── 2 graphs: [RMSNorm + FC_up] and [GELU + FC_down + Add] ──
    out.num_graphs = 2;

    // ION buffers: x [H], hidden [I], output [H]
    // x is also needed by graph 1 for residual add
    out.ion_bufs.resize(3);
    if (!allocIonBuffer(h_bytes, 0, out.ion_bufs[0]) ||  // x
        !allocIonBuffer(i_bytes, 0, out.ion_bufs[1]) ||  // hidden
        !allocIonBuffer(h_bytes, 0, out.ion_bufs[2]))     // output
      return false;
    fillRandomFp16(out.ion_bufs[0].ptr, H);

    // Register all ION with rank-4 dims
    out.reg_handles.resize(3, nullptr);
    if (!registerFFNIon(out.ion_bufs[0], out.dims_h4, out.reg_handles[0]) ||
        !registerFFNIon(out.ion_bufs[1], out.dims_i4, out.reg_handles[1]) ||
        !registerFFNIon(out.ion_bufs[2], out.dims_h4, out.reg_handles[2]))
      return false;

    double t0 = now_us();

    // ── Graph 0: RMSNorm + FC_up ──
    {
      char gname[64];
      snprintf(gname, sizeof(gname), "ffn_b0_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g0 = nullptr;
      if (!createGraphWithHtpConfig(gname, g0)) return false;
      out.graphs.push_back(g0);

      Qnn_Tensor_t x_t      = makeFp16Tensor("x",      QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
      Qnn_Tensor_t norm_t    = makeFp16Tensor("norm",   QNN_TENSOR_TYPE_NATIVE,    out.dims_h4);
      Qnn_Tensor_t hidden_t  = makeFp16Tensor("hidden", QNN_TENSOR_TYPE_APP_READ,  out.dims_i4);
      Qnn_Tensor_t gamma_t   = makeFp16Tensor("gamma",  QNN_TENSOR_TYPE_STATIC, w.dims_gamma1d, 1);
      gamma_t.v1.clientBuf.data = w.gamma; gamma_t.v1.clientBuf.dataSize = (uint32_t)(H * 2);
      Qnn_Tensor_t beta_t    = makeFp16Tensor("beta",   QNN_TENSOR_TYPE_STATIC, w.dims_gamma1d, 1);
      beta_t.v1.clientBuf.data = w.beta; beta_t.v1.clientBuf.dataSize = (uint32_t)(H * 2);
      Qnn_Tensor_t wup_t     = makeFp16Tensor("w_up", QNN_TENSOR_TYPE_STATIC, w.dims_wup, 2);
      wup_t.v1.clientBuf.data = w.w_up; wup_t.v1.clientBuf.dataSize = (uint32_t)((size_t)H * I * 2);

      if (!check(g_qnn->tensorCreateGraphTensor(g0, &x_t), "x") ||
          !check(g_qnn->tensorCreateGraphTensor(g0, &gamma_t), "gamma") ||
          !check(g_qnn->tensorCreateGraphTensor(g0, &beta_t), "beta") ||
          !check(g_qnn->tensorCreateGraphTensor(g0, &wup_t), "w_up") ||
          !check(g_qnn->tensorCreateGraphTensor(g0, &norm_t), "norm") ||
          !check(g_qnn->tensorCreateGraphTensor(g0, &hidden_t), "hidden"))
        return false;

      if (!addRmsNormNode(g0, "rmsnorm", x_t, gamma_t, beta_t, norm_t, w.axes_dims))
        return false;
      if (!addFCNode(g0, "fc_up", norm_t, wup_t, hidden_t))
        return false;

      if (!check(g_qnn->graphFinalize(g0, nullptr, nullptr), "graphFinalize g0"))
        return false;

      // Bind memHandle
      x_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      x_t.v1.memHandle = out.reg_handles[0];
      hidden_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      hidden_t.v1.memHandle = out.reg_handles[1];
      out.graph_execs.push_back({{x_t}, {hidden_t}});
    }

    // ── Graph 1: GELU + FC_down + Add(x, proj) ──
    {
      char gname[64];
      snprintf(gname, sizeof(gname), "ffn_b1_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g1 = nullptr;
      if (!createGraphWithHtpConfig(gname, g1)) return false;
      out.graphs.push_back(g1);

      Qnn_Tensor_t hidden_t = makeFp16Tensor("hidden", QNN_TENSOR_TYPE_APP_WRITE, out.dims_i4);
      Qnn_Tensor_t act_t    = makeFp16Tensor("act",    QNN_TENSOR_TYPE_NATIVE,    out.dims_i4);
      Qnn_Tensor_t proj_t   = makeFp16Tensor("proj",   QNN_TENSOR_TYPE_NATIVE,    out.dims_h4);
      Qnn_Tensor_t x_t      = makeFp16Tensor("x_res",  QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
      Qnn_Tensor_t out_t    = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  out.dims_h4);
      Qnn_Tensor_t wdn_t    = makeFp16Tensor("w_down", QNN_TENSOR_TYPE_STATIC, w.dims_wdn, 2);
      wdn_t.v1.clientBuf.data = w.w_down; wdn_t.v1.clientBuf.dataSize = (uint32_t)((size_t)I * H * 2);

      if (!check(g_qnn->tensorCreateGraphTensor(g1, &hidden_t), "hidden") ||
          !check(g_qnn->tensorCreateGraphTensor(g1, &x_t), "x_res") ||
          !check(g_qnn->tensorCreateGraphTensor(g1, &wdn_t), "w_down") ||
          !check(g_qnn->tensorCreateGraphTensor(g1, &act_t), "act") ||
          !check(g_qnn->tensorCreateGraphTensor(g1, &proj_t), "proj") ||
          !check(g_qnn->tensorCreateGraphTensor(g1, &out_t), "output"))
        return false;

      if (!addGeluNode(g1, "gelu", hidden_t, act_t)) return false;
      if (!addFCNode(g1, "fc_down", act_t, wdn_t, proj_t)) return false;
      if (!addAddNode(g1, "residual_add", x_t, proj_t, out_t)) return false;

      if (!check(g_qnn->graphFinalize(g1, nullptr, nullptr), "graphFinalize g1"))
        return false;

      // Bind memHandle
      hidden_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      hidden_t.v1.memHandle = out.reg_handles[1];
      x_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      x_t.v1.memHandle = out.reg_handles[0];
      out_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      out_t.v1.memHandle = out.reg_handles[2];
      out.graph_execs.push_back({{hidden_t, x_t}, {out_t}});
    }

    double t1 = now_us();
    out.create_us = t1 - t0;
    return true;
  }

  if (mode == SplitMode::PER_OP) {
    // ── 5 graphs, one per op ──
    // ION buffers: x[H], norm[H], hidden[I], act[I], proj[H], output[H]
    out.num_graphs = 5;

    out.ion_bufs.resize(6);
    if (!allocIonBuffer(h_bytes, 0, out.ion_bufs[0]) ||  // x
        !allocIonBuffer(h_bytes, 0, out.ion_bufs[1]) ||  // norm
        !allocIonBuffer(i_bytes, 0, out.ion_bufs[2]) ||  // hidden
        !allocIonBuffer(i_bytes, 0, out.ion_bufs[3]) ||  // act
        !allocIonBuffer(h_bytes, 0, out.ion_bufs[4]) ||  // proj
        !allocIonBuffer(h_bytes, 0, out.ion_bufs[5]))     // output
      return false;
    fillRandomFp16(out.ion_bufs[0].ptr, H);

    // Register all ION with rank-4 dims
    out.reg_handles.resize(6, nullptr);
    if (!registerFFNIon(out.ion_bufs[0], out.dims_h4, out.reg_handles[0]) ||  // x
        !registerFFNIon(out.ion_bufs[1], out.dims_h4, out.reg_handles[1]) ||  // norm
        !registerFFNIon(out.ion_bufs[2], out.dims_i4, out.reg_handles[2]) ||  // hidden
        !registerFFNIon(out.ion_bufs[3], out.dims_i4, out.reg_handles[3]) ||  // act
        !registerFFNIon(out.ion_bufs[4], out.dims_h4, out.reg_handles[4]) ||  // proj
        !registerFFNIon(out.ion_bufs[5], out.dims_h4, out.reg_handles[5]))    // output
      return false;

    double t0 = now_us();

    // Graph 0: RMSNorm(x → norm)
    {
      char gname[64]; snprintf(gname, sizeof(gname), "ffn_p0_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g = nullptr;
      if (!createGraphWithHtpConfig(gname, g)) return false;
      out.graphs.push_back(g);

      Qnn_Tensor_t x_t     = makeFp16Tensor("x",    QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
      Qnn_Tensor_t norm_t   = makeFp16Tensor("norm", QNN_TENSOR_TYPE_APP_READ,  out.dims_h4);
      Qnn_Tensor_t gamma_t  = makeFp16Tensor("gamma", QNN_TENSOR_TYPE_STATIC, w.dims_gamma1d, 1);
      gamma_t.v1.clientBuf.data = w.gamma; gamma_t.v1.clientBuf.dataSize = (uint32_t)(H * 2);
      Qnn_Tensor_t beta_t   = makeFp16Tensor("beta",  QNN_TENSOR_TYPE_STATIC, w.dims_gamma1d, 1);
      beta_t.v1.clientBuf.data = w.beta; beta_t.v1.clientBuf.dataSize = (uint32_t)(H * 2);

      if (!check(g_qnn->tensorCreateGraphTensor(g, &x_t), "x") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &gamma_t), "gamma") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &beta_t), "beta") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &norm_t), "norm"))
        return false;
      if (!addRmsNormNode(g, "rmsnorm", x_t, gamma_t, beta_t, norm_t, w.axes_dims)) return false;
      if (!check(g_qnn->graphFinalize(g, nullptr, nullptr), "fin p0")) return false;

      x_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      x_t.v1.memHandle = out.reg_handles[0];
      norm_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      norm_t.v1.memHandle = out.reg_handles[1];
      out.graph_execs.push_back({{x_t}, {norm_t}});
    }

    // Graph 1: FC(norm → hidden)
    {
      char gname[64]; snprintf(gname, sizeof(gname), "ffn_p1_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g = nullptr;
      if (!createGraphWithHtpConfig(gname, g)) return false;
      out.graphs.push_back(g);

      Qnn_Tensor_t norm_t   = makeFp16Tensor("norm",   QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
      Qnn_Tensor_t hidden_t = makeFp16Tensor("hidden", QNN_TENSOR_TYPE_APP_READ,  out.dims_i4);
      Qnn_Tensor_t wup_t    = makeFp16Tensor("w_up", QNN_TENSOR_TYPE_STATIC, w.dims_wup, 2);
      wup_t.v1.clientBuf.data = w.w_up; wup_t.v1.clientBuf.dataSize = (uint32_t)((size_t)H * I * 2);

      if (!check(g_qnn->tensorCreateGraphTensor(g, &norm_t), "norm") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &wup_t), "w_up") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &hidden_t), "hidden"))
        return false;
      if (!addFCNode(g, "fc_up", norm_t, wup_t, hidden_t)) return false;
      if (!check(g_qnn->graphFinalize(g, nullptr, nullptr), "fin p1")) return false;

      norm_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      norm_t.v1.memHandle = out.reg_handles[1];
      hidden_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      hidden_t.v1.memHandle = out.reg_handles[2];
      out.graph_execs.push_back({{norm_t}, {hidden_t}});
    }

    // Graph 2: GELU(hidden → act)
    {
      char gname[64]; snprintf(gname, sizeof(gname), "ffn_p2_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g = nullptr;
      if (!createGraphWithHtpConfig(gname, g)) return false;
      out.graphs.push_back(g);

      Qnn_Tensor_t hidden_t = makeFp16Tensor("hidden", QNN_TENSOR_TYPE_APP_WRITE, out.dims_i4);
      Qnn_Tensor_t act_t    = makeFp16Tensor("act",    QNN_TENSOR_TYPE_APP_READ,  out.dims_i4);

      if (!check(g_qnn->tensorCreateGraphTensor(g, &hidden_t), "hidden") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &act_t), "act"))
        return false;
      if (!addGeluNode(g, "gelu", hidden_t, act_t)) return false;
      if (!check(g_qnn->graphFinalize(g, nullptr, nullptr), "fin p2")) return false;

      hidden_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      hidden_t.v1.memHandle = out.reg_handles[2];
      act_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      act_t.v1.memHandle = out.reg_handles[3];
      out.graph_execs.push_back({{hidden_t}, {act_t}});
    }

    // Graph 3: FC(act → proj)
    {
      char gname[64]; snprintf(gname, sizeof(gname), "ffn_p3_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g = nullptr;
      if (!createGraphWithHtpConfig(gname, g)) return false;
      out.graphs.push_back(g);

      Qnn_Tensor_t act_t  = makeFp16Tensor("act",  QNN_TENSOR_TYPE_APP_WRITE, out.dims_i4);
      Qnn_Tensor_t proj_t = makeFp16Tensor("proj", QNN_TENSOR_TYPE_APP_READ,  out.dims_h4);
      Qnn_Tensor_t wdn_t  = makeFp16Tensor("w_down", QNN_TENSOR_TYPE_STATIC, w.dims_wdn, 2);
      wdn_t.v1.clientBuf.data = w.w_down; wdn_t.v1.clientBuf.dataSize = (uint32_t)((size_t)I * H * 2);

      if (!check(g_qnn->tensorCreateGraphTensor(g, &act_t), "act") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &wdn_t), "w_down") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &proj_t), "proj"))
        return false;
      if (!addFCNode(g, "fc_down", act_t, wdn_t, proj_t)) return false;
      if (!check(g_qnn->graphFinalize(g, nullptr, nullptr), "fin p3")) return false;

      act_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      act_t.v1.memHandle = out.reg_handles[3];
      proj_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      proj_t.v1.memHandle = out.reg_handles[4];
      out.graph_execs.push_back({{act_t}, {proj_t}});
    }

    // Graph 4: Add(x, proj → output)
    {
      char gname[64]; snprintf(gname, sizeof(gname), "ffn_p4_%u", g_graphSeq++);
      out.graph_names.push_back(gname);
      Qnn_GraphHandle_t g = nullptr;
      if (!createGraphWithHtpConfig(gname, g)) return false;
      out.graphs.push_back(g);

      Qnn_Tensor_t x_t    = makeFp16Tensor("x_res",  QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
      Qnn_Tensor_t proj_t = makeFp16Tensor("proj",   QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
      Qnn_Tensor_t out_t  = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ,  out.dims_h4);

      if (!check(g_qnn->tensorCreateGraphTensor(g, &x_t), "x_res") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &proj_t), "proj") ||
          !check(g_qnn->tensorCreateGraphTensor(g, &out_t), "output"))
        return false;
      if (!addAddNode(g, "residual_add", x_t, proj_t, out_t)) return false;
      if (!check(g_qnn->graphFinalize(g, nullptr, nullptr), "fin p4")) return false;

      x_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      x_t.v1.memHandle = out.reg_handles[0];
      proj_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      proj_t.v1.memHandle = out.reg_handles[4];
      out_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
      out_t.v1.memHandle = out.reg_handles[5];
      out.graph_execs.push_back({{x_t, proj_t}, {out_t}});
    }

    double t1 = now_us();
    out.create_us = t1 - t0;
    return true;
  }

  printf("[FFN] Unknown split mode\n");
  return false;
}

void cleanup_ffn(FFNResult& r) {
  std::vector<Qnn_MemHandle_t> handles;
  for (auto h : r.reg_handles)
    if (h) handles.push_back(h);
  if (!handles.empty() && g_qnn && g_qnn->memDeRegister)
    g_qnn->memDeRegister(handles.data(), (uint32_t)handles.size());
  r.reg_handles.clear();

  for (auto& buf : r.ion_bufs) freeIonBuffer(buf);
  r.ion_bufs.clear();
  freeIonBuffer(r.weight_buf);
  r.graphs.clear();
  r.graph_execs.clear();
  r.graph_names.clear();
}

// ─── FFN cached mode ────────────────────────────────────────────────────────

bool build_ffn_cached(const FFNConfig& cfg, FFNResult& out, double& out_cache_load_us) {
  // Step 1: Build fused graph normally (to get tensor IDs and context binary)
  if (!build_ffn_block(SplitMode::FUSED, cfg, out))
    return false;

  // Save tensor IDs assigned during graph build
  uint32_t input_id  = out.graph_execs[0].ins[0].v1.id;
  uint32_t output_id = out.graph_execs[0].outs[0].v1.id;
  std::string gname  = out.graph_names[0];

  // Step 2: Serialize context to binary
  std::vector<uint8_t> binary;
  if (!qnn_context_get_binary(binary)) {
    cleanup_ffn(out);
    return false;
  }

  // Step 3: Cleanup old context completely
  cleanup_ffn(out);
  qnn_cleanup();

  // Step 4: Re-init backend (without context — we'll create from binary)
  if (!qnn_init()) {
    printf("[FFN-Cache] qnn_init failed\n");
    return false;
  }

  // Step 5: Create context from binary (timed)
  // qnn_context_create_from_binary frees the existing context internally
  double load_us = 0;
  double t0 = now_us();
  if (!qnn_context_create_from_binary(binary, load_us)) {
    printf("[FFN-Cache] contextCreateFromBinary failed\n");
    return false;
  }

  // Step 6: Retrieve graph
  Qnn_GraphHandle_t graph = nullptr;
  if (!check(g_qnn->graphRetrieve(g_context, gname.c_str(), &graph), "graphRetrieve")) {
    return false;
  }

  int H = cfg.hidden_dim;
  size_t h_bytes = (size_t)H * 2;

  // Initialize persistent dims
  out = FFNResult{};  // reset
  out.dims_h4[0] = 1; out.dims_h4[1] = 1; out.dims_h4[2] = 1; out.dims_h4[3] = (uint32_t)H;
  out.dims_i4[0] = 1; out.dims_i4[1] = 1; out.dims_i4[2] = 1; out.dims_i4[3] = (uint32_t)cfg.intermediate_dim;
  out.num_graphs = 1;
  out.graphs.push_back(graph);
  out.graph_names.push_back(gname);

  // Step 7: Allocate and register ION on new context
  out.ion_bufs.resize(2);
  if (!allocIonBuffer(h_bytes, 0, out.ion_bufs[0]) ||
      !allocIonBuffer(h_bytes, 0, out.ion_bufs[1]))
    return false;
  fillRandomFp16(out.ion_bufs[0].ptr, H);

  out.reg_handles.resize(2, nullptr);
  if (!registerFFNIon(out.ion_bufs[0], out.dims_h4, out.reg_handles[0]) ||
      !registerFFNIon(out.ion_bufs[1], out.dims_h4, out.reg_handles[1]))
    return false;

  double t1 = now_us();
  out.create_us = t1 - t0;  // total cache load time (context + graph + ION)
  out_cache_load_us = out.create_us;

  // Step 8: Build execution tensors with saved tensor IDs
  Qnn_Tensor_t x_t   = makeFp16Tensor("x",      QNN_TENSOR_TYPE_APP_WRITE, out.dims_h4);
  x_t.v1.id        = input_id;
  x_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  x_t.v1.memHandle = out.reg_handles[0];

  Qnn_Tensor_t out_t = makeFp16Tensor("output", QNN_TENSOR_TYPE_APP_READ, out.dims_h4);
  out_t.v1.id        = output_id;
  out_t.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  out_t.v1.memHandle = out.reg_handles[1];

  out.graph_execs.push_back({{x_t}, {out_t}});
  return true;
}

// ─── Legacy functions ───────────────────────────────────────────────────────

bool retrieve_fused_from_cache(int num_ops, int hidden_dim,
                               const char* graph_name,
                               uint32_t input_tensor_id, uint32_t output_tensor_id,
                               FusedResult& out) {
  size_t buf_bytes = (size_t)hidden_dim * 2;
  static uint32_t s_dims[kTensorRank];
  s_dims[0] = 1; s_dims[1] = 1; s_dims[2] = 1; s_dims[3] = (uint32_t)hidden_dim;

  // Allocate ION buffers
  if (!allocIonBuffer(buf_bytes, 0, out.ion_in) ||
      !allocIonBuffer(buf_bytes, 0, out.ion_out)) {
    printf("[Cache] ION alloc failed\n"); return false;
  }
  uint16_t half_val = float_to_half(0.5f);
  uint16_t* inp = reinterpret_cast<uint16_t*>(out.ion_in.ptr);
  for (int i = 0; i < hidden_dim; ++i) inp[i] = half_val;

  // Register ION on the new context
  if (!registerIon(out.ion_in, s_dims, kTensorRank, out.reg_in) ||
      !registerIon(out.ion_out, s_dims, kTensorRank, out.reg_out))
    return false;

  // Retrieve the graph from deserialized context
  if (!check(g_qnn->graphRetrieve(g_context, graph_name, &out.graph), "graphRetrieve"))
    return false;

  snprintf(out.graph_name, sizeof(out.graph_name), "%s", graph_name);

  // Build execution tensors with saved tensor IDs from the original graph build.
  // After contextCreateFromBinary + graphRetrieve, the DSP-side tensors keep their
  // original IDs. We must set v1.id to match, otherwise graphExecute silently fails.
  char in_name[] = "t_0";
  char out_name[16];
  snprintf(out_name, sizeof(out_name), "t_%d", num_ops);

  out.exec_in  = makeFp16Tensor(in_name, QNN_TENSOR_TYPE_APP_WRITE, s_dims);
  out.exec_in.v1.id        = input_tensor_id;
  out.exec_in.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  out.exec_in.v1.memHandle = out.reg_in;

  out.exec_out = makeFp16Tensor(out_name, QNN_TENSOR_TYPE_APP_READ, s_dims);
  out.exec_out.v1.id        = output_tensor_id;
  out.exec_out.v1.memType   = QNN_TENSORMEMTYPE_MEMHANDLE;
  out.exec_out.v1.memHandle = out.reg_out;

  out.input_tensor_id  = input_tensor_id;
  out.output_tensor_id = output_tensor_id;

  return true;
}
