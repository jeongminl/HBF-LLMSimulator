
#include "lm_head.h"

#include <algorithm>
#include <cmath>
#include "common/assert.h"
#include "hardware/hardware_config.h"
#include "hardware/layer_impl.h"
// LmHead //

namespace llm_system {

LmHead::LmHead(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;
  // Vocab-parallel sharding across the TP group (Megatron-style column-parallel
  // logits): each TP rank holds n_vocab/tp columns. Divisor is ne_tp_dg, NOT
  // device_list.size() (which spans DP replicas here). The cross-rank argmax /
  // token-id gather this implies is a few bytes per sequence per step --
  // negligible, so no communication op is added.
  int tp = (model_config.ne_tp_dg > 0) ? model_config.ne_tp_dg : 1;
  int vocab_per_rank = (n_vocab + tp - 1) / tp;
  std::vector<int> wgt_shape = {hidden_dimension, vocab_per_rank};
  std::vector<int> act_shape = {1, hidden_dimension};

  Tensor::Ptr LmHead = Tensor::Create("lm_head_wgt", wgt_shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr output = Tensor::Create("Hidden vector", act_shape, "act", device, device->model_config.precision_byte);
  add_tensor(LmHead);
  add_tensor(output);
}

Tensor::Ptr LmHead::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = sequences_metadata->get_process_token();
  int k = model_config.hidden_dim;
  // Vocab-parallel: this rank computes logits for its n_vocab/tp shard only
  // (matches the sharded weight tensor in the constructor).
  int tp = (model_config.ne_tp_dg > 0) ? model_config.ne_tp_dg : 1;
  int n = (model_config.n_vocab + tp - 1) / tp;
  int precision = device->model_config.precision_byte;

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Hidden vector", shape);

  if (m == 0 || k == 0 || n == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  // Memory time: uses flash-aware path for HBF configs (weight on flash,
  // activations on HBM/SRAM), falls back to uniform BW for HBM-only configs.
  hw_metric total_memory_size = ((hw_metric)k * n + (hw_metric)m * k + (hw_metric)m * n) * precision;
  time_ns memory_time = getLinearMemoryDuration(device->config, (double)m, (double)k, (double)n,
                                                 precision, total_memory_size, memory_bandwidth);

  // Compute time: 2 * m * k * n multiply-accumulates.
  time_ns compute_time = (time_ns)(2.0 * m * (double)k * n / compute_peak_flops * 1e9);

  time_ns total_time = std::max(memory_time, compute_time);

  if (input->parallel_execution) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

}  // namespace llm_system