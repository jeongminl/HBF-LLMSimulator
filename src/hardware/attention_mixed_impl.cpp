#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus AttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;
  time_ns total_compute_duration = 0;  // track total compute for KV write overlap

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  time_ns scoring_compute_duration = 0;
  time_ns scoring_memory_duration = 0;
  hw_metric scoring_kv_read_size = 0;
  hw_metric scoring_act_size = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    scoring_compute_duration += compute_duration;
    total_compute_duration += compute_duration;

    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      scoring_kv_read_size += k * n * num_kv_heads * input->precision_byte;
      scoring_act_size += m * k * num_heads * input->precision_byte;
    } else {
      scoring_memory_duration += memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    }
  }
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    scoring_memory_duration = getAttentionMemoryDuration(config, scoring_kv_read_size, scoring_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
  }
  total_duration += std::max(scoring_compute_duration, scoring_memory_duration);
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  time_ns context_compute_duration = 0;
  time_ns context_memory_duration = 0;
  hw_metric context_kv_read_size = 0;
  hw_metric context_act_size = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    context_compute_duration += compute_duration;
    total_compute_duration += compute_duration;

    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      context_kv_read_size += k * n * num_kv_heads * input->precision_byte;
      context_act_size += m * k * num_heads * input->precision_byte;
    } else {
      context_memory_duration += memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    }
  }
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    context_memory_duration = getAttentionMemoryDuration(config, context_kv_read_size, context_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
  }
  total_duration += std::max(context_compute_duration, context_memory_duration);

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            ((config.use_hbf && config.hbf_config.num_flash_stacks > 0) ? config.hbf_config.flash_read_bandwidth : memory_bandwidth) / exec_status.total_duration;

  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    time_ns kv_write = getKVWriteDuration(config, num_seq, num_kv_heads, head_dim, input->precision_byte, device->model_config.compressed_kv, layer_info.kv_lora_rank, layer_info.qk_rope_head_dim, device->model_config.input_len, device->model_config.output_len, layer_info.local_attention_window);
    time_ns unhidden_write = std::max((time_ns)0, kv_write - total_compute_duration);
    exec_status.total_duration += unhidden_write;
    exec_status.kv_write_duration = unhidden_write;
  }

  return exec_status;
};

ExecStatus AttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator) {
  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  ExecStatus exec_status;

  return exec_status;
};

ExecStatus AttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

ExecStatus MultiLatentAttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int attention_group_size = layer_info.attention_group_size;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim + qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  return exec_status;
};

// need to fix
ExecStatus MultiLatentAttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator) {
  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  ExecStatus exec_status;

  return exec_status;
};

ExecStatus MultiLatentAttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int attention_group_size = layer_info.attention_group_size;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim + qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }
  // Softmax //

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  ExecStatus exec_status;

  exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};


}  // namespace llm_system