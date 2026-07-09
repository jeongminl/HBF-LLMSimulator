#include <memory>

#include "common/assert.h"
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
  // FA master flag. RULING #5: the Mixed family is DEAD CODE (never instantiated --
  // parallel.cpp builds only Sum/Gen), brought to parity with the live Sum/Gen
  // kernels (I16 score/output memory term + I17 Softmax phase) and made flag-aware
  // for internal consistency only. Not numerically testable.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;
  time_ns total_compute_duration = 0;  // track total compute for KV write overlap

  // paper2 CPU-memory/NVLink-C2C KV offload tier: SelfAttentionMixed::Create
  // (module/attention.h) has zero call sites anywhere in this tree -- this
  // GPU path is dead code today, never reached regardless of decode_mode.
  // Fail loudly rather than silently mis-time it if it's ever wired up with
  // the offload tier on.
  if (cpuKvOffloadActive(config)) {
    fail("cpu_kv_offload: unmodeled attention path AttentionMixedExecutionGPU");
  }

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  double batch_m = sequences_metadata->get_process_token();
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

    // I16: Mixed omitted the score-output write term (m*n*heads) that Gen/Sum charge.
    // Flag-aware: dropped under ON (score never materializes), kept under OFF.
    memory_size = (m * k * num_heads + k * n * num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads)) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    scoring_compute_duration += compute_duration;
    total_compute_duration += compute_duration;

    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      scoring_kv_read_size += k * n * num_kv_heads * input->precision_byte;
      scoring_act_size += (m * k * num_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads)) * input->precision_byte;
    } else {
      scoring_memory_duration += memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    }
  }
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    scoring_memory_duration = getAttentionMemoryDuration(config, scoring_kv_read_size, scoring_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
  }
  total_duration += std::max(scoring_compute_duration, scoring_memory_duration);

  // Softmax // -- I17: Mixed had no Softmax phase (no FLOPs, no KV-write hide budget).
  // Add it at parity with Gen/Sum: compute always charged; memory 2*m*n*heads only OFF.
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads; // scale + mask + softmax
    total_flops += flops;

    double sm_memory_size = use_flash_attention ? 0.0 : 2.0 * (double)m * n * num_heads * input->precision_byte;
    total_memory_size += sm_memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // Materialized score/P priced at the scarce activation tier -- see the Gen kernels.
    hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
        ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                  : config.hbf_config.logic_sram_bandwidth)
        : memory_bandwidth;
    memory_duration = sm_memory_size / softmax_bw * 1000 * 1000 * 1000;
    total_compute_duration += compute_duration;
    total_duration += std::max(compute_duration, memory_duration);
  }

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

    // I16 parity with Sum's Context: P (score-read, m*k*heads) dropped under ON;
    // add the context-output write (m*n*heads, always present) Mixed omitted.
    memory_size = ((use_flash_attention ? 0.0 : (double)m * k * num_heads) + k * n * num_kv_heads + (double)m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    context_compute_duration += compute_duration;
    total_compute_duration += compute_duration;

    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      context_kv_read_size += k * n * num_kv_heads * input->precision_byte;
      context_act_size += ((use_flash_attention ? 0.0 : (double)m * k * num_heads) + (double)m * n * num_heads) * input->precision_byte;
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
    // Program latency amortized across this stage's per-layer write stream --
    // see getKVWriteDuration's program_latency_amortize_calls doc.
    int layers_per_stage = device->model_config.num_layers /
        (device->model_config.pp_dg > 0 ? device->model_config.pp_dg : 1);
    time_ns kv_write = getKVWriteDuration(config, num_seq, num_kv_heads, head_dim, input->precision_byte, device->model_config.compressed_kv, layer_info.kv_lora_rank, layer_info.qk_rope_head_dim, device->model_config.input_len, device->model_config.output_len, layer_info.local_attention_window, layers_per_stage);
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
  // FA master flag (ruling #5, dead-code parity + flag-aware -- see the GPU variant).
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  double batch_m = sequences_metadata->get_process_token();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // I16: add the score-output write term (m*n*heads), flag-aware.
    memory_size = (m * k * num_heads + k * n * num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads)) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax // -- I17: add the Softmax phase Mixed lacked (parity with Gen/Sum).
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads; // scale + mask + softmax
    total_flops += flops;

    memory_size = use_flash_attention ? 0.0 : 2.0 * (double)m * n * num_heads * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // Materialized score/P priced at the scarce activation tier -- see the Gen kernels.
    hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
        ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                  : config.hbf_config.logic_sram_bandwidth)
        : memory_bandwidth;
    memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // I16 parity: P (score-read m*k*heads) dropped under ON; add output write (m*n*heads).
    memory_size = ((use_flash_attention ? 0.0 : (double)m * k * num_heads) + k * n * num_kv_heads + (double)m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
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
  // FA master flag (ruling #5, dead-code parity + flag-aware -- see the GQA variant).
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // paper2 CPU-memory/NVLink-C2C KV offload tier: MultiLatentAttentionMixed::
  // Create has zero call sites anywhere in this tree -- this GPU path is dead
  // code today, never reached regardless of decode_mode. Fail loudly rather
  // than silently mis-time it if it's ever wired up with the offload tier on.
  if (cpuKvOffloadActive(config)) {
    fail("cpu_kv_offload: unmodeled attention path MultiLatentAttentionMixedExecutionGPU");
  }

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  double batch_m = sequences_metadata->get_process_token();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim + qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // I16: add the score-output write term (m*n*heads), flag-aware.
    memory_size = (m * k * num_heads + k * n * num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads)) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax // -- I17: add the Softmax phase Mixed lacked (parity with Gen/Sum).
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads; // scale + mask + softmax
    total_flops += flops;

    memory_size = use_flash_attention ? 0.0 : 2.0 * (double)m * n * num_heads * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // Materialized score/P priced at the scarce activation tier -- see the Gen kernels.
    hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
        ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                  : config.hbf_config.logic_sram_bandwidth)
        : memory_bandwidth;
    memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // I16 parity: P (score-read m*k*heads) dropped under ON; add output write (m*n*heads).
    memory_size = ((use_flash_attention ? 0.0 : (double)m * k * num_heads) + k * n * num_kv_heads + (double)m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
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
  // FA master flag (ruling #5, dead-code parity + flag-aware -- see the GQA variant).
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  // Scoring //
  int num_seq = sequences_metadata->sequence.size();
  double batch_m = sequences_metadata->get_process_token();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = head_dim + qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // I16: add the score-output write term (m*n*heads), flag-aware.
    memory_size = (m * k * num_heads + k * n * num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads)) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax // -- I17: add the Softmax phase Mixed lacked (parity with Gen/Sum).
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads; // scale + mask + softmax
    total_flops += flops;

    memory_size = use_flash_attention ? 0.0 : 2.0 * (double)m * n * num_heads * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // Materialized score/P priced at the scarce activation tier -- see the Gen kernels.
    hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
        ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                  : config.hbf_config.logic_sram_bandwidth)
        : memory_bandwidth;
    memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

    total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  num_seq = sequences_metadata->sequence.size();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    Sequence::Ptr seq = sequences_metadata->get_seq()[seq_idx];

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // I16 parity: P (score-read m*k*heads) dropped under ON; add output write (m*n*heads).
    memory_size = ((use_flash_attention ? 0.0 : (double)m * k * num_heads) + k * n * num_kv_heads + (double)m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
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