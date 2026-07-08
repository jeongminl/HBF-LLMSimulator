
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
  // I5: derate by effectiveMFU, mirroring every other GEMM's compute_duration
  // (linear_impl.cpp, activation_impl.cpp, attention_*_impl.cpp) -- LmHead was
  // the sole GEMM using raw peak flops. mfu_max defaults to 1.0, so this is a
  // no-op unless simulation.mfu_max is swept below 1.0.
  time_ns compute_time = (time_ns)(2.0 * m * (double)k * n /
                                    (compute_peak_flops * effectiveMFU(device->config, (double)m)) * 1e9);

  time_ns total_time = std::max(memory_time, compute_time);

  if (input->parallel_execution) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;
  // PP_FIX_SPEC.md §3.3: LmHead manipulates device_time directly (never goes
  // through ExecStatus/set_pop_status), so it needs its own device_time_dep
  // mirror. Same refined rule as Linear (linear_impl.cpp): compute_time is
  // always fully batch-dependent (2*m*k*n, linear in m). memory_time is only
  // uniformly one-or-the-other on the flash path (getLinearMemoryDuration's
  // flash branch maxes two terms that are each already purely
  // one-or-the-other); on the non-flash path memory_time is an ADDITIVE sum
  // of a fixed weight-read term and an m-scaled activation term, so it must
  // be split by byte fraction (exact, since duration is linear in bytes) --
  // treating the whole thing as indep undercounts real linear scaling enough
  // to blow the pp>1 reconstruction's +/-1% target (n_vocab/tp is often not
  // overwhelmingly larger than the per-step activation at this op).
  if (compute_time > memory_time) {
    device->status.device_time_dep += total_time;
  } else if (device->config.use_hbf && device->config.hbf_config.num_flash_stacks > 0) {
    const auto& hbf = device->config.hbf_config;
    hw_metric act_read_size = (hw_metric)m * k * precision;
    hw_metric act_write_size = (hw_metric)m * n * precision;
    double act_time = (hbf.num_hbm_stacks > 0)
        ? (act_read_size + act_write_size) / hbf.hbm_read_bandwidth * 1e9
        : (act_read_size + act_write_size) / hbf.logic_sram_bandwidth * 1e9;
    if (act_time >= memory_time) device->status.device_time_dep += total_time;
  } else {
    hw_metric weight_bytes = (hw_metric)k * n * precision;
    hw_metric act_bytes = total_memory_size - weight_bytes;
    if (total_memory_size > 0)
      device->status.device_time_dep += memory_time * (act_bytes / total_memory_size);
  }

  return output;
}

}  // namespace llm_system