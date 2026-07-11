#include <algorithm>
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

ExecStatus AttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;
  // FlashAttention master flag (FA_FLAG_SPEC.md). ON: the QK^T score and the
  // softmax-weight P never materialize -- drop the score term from Scoring, the P
  // term from Context, and charge the softmax loop ZERO memory. OFF (new default):
  // keep the materialized score/P timing terms AND charge the softmax loop a
  // 2*m*n*heads/kv score-read+P-write memory term (unifying GQA's cost model to
  // MLA's existing 2*m*n softmax term).
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  hw_metric total_kv_read_size = 0;
  hw_metric total_act_size = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;
    // Llama-4-style interleaved local/global attention: cap this layer's KV read at
    // its effective window (0 = no cap, every model except llama4_maverick/scout's
    // local layers -- see LayerInfo::local_attention_window / model_config.h).
    if (layer_info.local_attention_window > 0) n = std::min(n, layer_info.local_attention_window);

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = 1.0 * m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      // ON: score (m*n*heads/kv) never materializes -> drop from Scoring traffic.
      // I19: latent at decode's m=1, fixed for consistency with the prefill sites.
      memory_size = 1.0 * ((double)m * k * num_heads / num_kv_heads + (double)k * n + (use_flash_attention ? 0.0 : (double)m * n * num_heads / num_kv_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        total_kv_read_size += k * n * input->precision_byte;
        total_act_size += ((double)m * k * num_heads / num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads / num_kv_heads)) * input->precision_byte;
      } else if (cpuKvOffloadActive(config)) {
        // paper2 CPU-offload tier: accumulate the same aggregate read/act
        // totals the HBF flash branch above accumulates, so the single
        // post-loop hbmKvOffloadReadDuration() call below composes with
        // accumul_compute_duration exactly the way the flash branch's
        // getAttentionMemoryDuration() call does. Also mirrors the flash
        // branch's use_flash_attention gating (FA_FLAG_SPEC.md): when FA is
        // ON, the score never materializes, so it must not be charged here
        // either regardless of which memory tier backs the KV read.
        total_kv_read_size += k * n * input->precision_byte;
        total_act_size += ((double)m * k * num_heads / num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads / num_kv_heads)) * input->precision_byte;
      } else {
        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
    }
    accumul_len += n;
  }

  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    // V's page address has no data dependency on K/softmax, so its fetch can be
    // prefetched under K's transfer window (same cross-op hiding convention as
    // weight_stream_ops_per_iter) -- K and V together expose ~one fill/layer.
    accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size, 2, layer_info.kv_hbm_fraction);
  } else if (cpuKvOffloadActive(config)) {
    accumul_memory_duration = hbmKvOffloadReadDuration(config, total_kv_read_size, total_act_size, sequences_metadata->p2_kv_offload_fraction);
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

  // Softmax //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;
    if (layer_info.local_attention_window > 0) n = std::min(n, layer_info.local_attention_window);

    flops = 7.0 * m * n * num_heads; // scale + mask + softmax
    total_flops += flops;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;

    // OFF (materialized): the softmax reads the resident score and writes P back
    // to memory -- charge 2*m*n*heads, matching MLA's convention (see
    // MultiLatentAttentionGenExecutionGPU:867) and this same kernel's own FLOPs
    // line 3 above (7*m*n*num_heads, not .../num_kv_heads). I18: the score/P sheet
    // is per query head regardless of GQA K/V sharing -- one read + one write per
    // element is 2*m*n*num_heads bytes, not 2*m*n*num_heads/num_kv_heads (this loop
    // is a single pass over seq_idx, not a per-kv-head loop like Context below, so
    // there is nothing to sum the missing num_kv_heads factor back in). Was
    // undercounting softmax materialization memory by num_kv_heads-fold (8x for
    // 64/8 GQA) whenever use_flash_attention is off. ON: on-chip, zero memory
    // (today's behavior). The score/P round-trip is activation traffic on the
    // SCARCE tier, NOT flash: mirror getAttentionMemoryDuration's act-tier
    // bandwidth (on HBF-family the local `memory_bandwidth` is repurposed to
    // flash_read_bandwidth, device.cpp:32-33) -- hbm_read_bandwidth (num_hbm_stacks>0)
    // else logic_sram_bandwidth; plain-HBM keeps memory_bandwidth.
    if (!use_flash_attention) {
      double softmax_mem = 2.0 * (double)m * n * num_heads * input->precision_byte;
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = softmax_mem / softmax_bw * 1000 * 1000 * 1000;
      total_memory_size += softmax_mem;
    }

    time_ns softmax_phase = std::max(compute_duration, memory_duration);
    exec_status.total_duration += softmax_phase;
    exec_status.batch_dependent_duration += softmax_phase;
    // PP_FLAGS_SPEC §4.2: let softmax compute serve as KV-write hiding budget (:227).
    if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
  }

  // Context //
  // Context: accumulate total KV read and activation sizes across all (seq × kv_head) pairs
  // and call getAttentionMemoryDuration once after the loop.  Per-iteration calls would charge
  // one full flash_page_read_latency per (seq, kv_head) pair instead of once per chunk.
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  hw_metric total_context_kv_read_size = 0;
  hw_metric total_context_act_size = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    // Llama-4-style interleaved local/global attention: same cap as the Scoring loop
    // above, applied to k (the KV-position count) here since Context swaps k/n vs.
    // Scoring's roles.
    if (layer_info.local_attention_window > 0) k = std::min(k, layer_info.local_attention_window);
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = 1.0 * m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      // ON: the softmax-weight P (read here as m*k*heads/kv) never materializes
      // -> drop from Context traffic. m*n*heads/kv is the context output write (kept).
      // I19: latent at decode's m=1, fixed for consistency with the prefill sites.
      memory_size = 1.0 * ((use_flash_attention ? 0.0 : (double)m * k * num_heads / num_kv_heads) + (double)k * n + (double)m * n * num_heads / num_kv_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        total_context_kv_read_size += k * n * input->precision_byte;
        total_context_act_size += ((use_flash_attention ? 0.0 : (double)m * k * num_heads / num_kv_heads) + (double)m * n * num_heads / num_kv_heads) * input->precision_byte;
      } else if (cpuKvOffloadActive(config)) {
        // See the Scoring-loop cpuKvOffloadActive branch above: mirror the
        // flash branch's use_flash_attention gating here too.
        total_context_kv_read_size += k * n * input->precision_byte;
        total_context_act_size += ((use_flash_attention ? 0.0 : (double)m * k * num_heads / num_kv_heads) + (double)m * n * num_heads / num_kv_heads) * input->precision_byte;
      } else {
        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
    }
    accumul_len += k;
  }
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    // Other half of the K->V fill amortization above: V's fill hides under K's
    // transfer, so this call also charges only its 1/2 share of the page fill.
    accumul_memory_duration = getAttentionMemoryDuration(config, total_context_kv_read_size, total_context_act_size, layer_info.use_chunked_attention, layer_info.chunk_size, 2, layer_info.kv_hbm_fraction);
  } else if (cpuKvOffloadActive(config)) {
    accumul_memory_duration = hbmKvOffloadReadDuration(config, total_context_kv_read_size, total_context_act_size, sequences_metadata->p2_kv_offload_fraction);
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            ((config.use_hbf && config.hbf_config.num_flash_stacks > 0) ? config.hbf_config.flash_read_bandwidth : memory_bandwidth) / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    // Only the unhidden portion of KV write extends the critical path.
    // The write can overlap with the attention compute phase (both Scoring and
    // Context).  exec_status.compute_duration accumulates across both phases.
    // All of this device's attention layers (num_layers / pp stage share) write
    // their admitted-KV back-to-back each iteration as one flash stream, so the
    // page-program latency is amortized across those per-layer calls (see
    // getKVWriteDuration's program_latency_amortize_calls doc).
    int layers_per_stage = device->model_config.num_layers /
        (device->model_config.pp_dg > 0 ? device->model_config.pp_dg : 1);
    time_ns kv_write = getKVWriteDuration(config, num_seq, num_kv_heads, head_dim, input->precision_byte, false, 0, 0, device->model_config.input_len, device->model_config.output_len, layer_info.local_attention_window, layers_per_stage);
    time_ns unhidden_write = std::max((time_ns)0, kv_write - exec_status.compute_duration);
    exec_status.total_duration += unhidden_write;
    exec_status.batch_dependent_duration += unhidden_write;
    exec_status.kv_write_duration = unhidden_write;
  }

  return exec_status;
};

ExecStatus AttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  // GPU/LOGIC both take total_duration += max(compute, memory) (roofline overlap).
  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

  // Softmax // -- mirrors the GPU/LOGIC softmax compute charge.
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    int softmax_m = seq->num_process_token;
    int softmax_n = seq->current_len + seq->num_process_token;

    double softmax_flops = 7.0 * softmax_m * softmax_n * num_heads; // scale + mask + softmax
    total_flops += softmax_flops;

    time_ns softmax_compute_duration = softmax_flops /
        (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;

    exec_status.total_duration += softmax_compute_duration;
    exec_status.batch_dependent_duration += softmax_compute_duration;
    // PP_FLAGS_SPEC §4.2: let softmax compute serve as KV-write hiding budget.
    if (config.kv_write_softmax_hide) exec_status.compute_duration += softmax_compute_duration;
  }

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else{
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  // Same max(compute, memory) roofline overlap as the Scoring section above.
  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_mla = layer_info.use_flash_mla;
  // FA master flag. RULING #3: only the materialized (use_flash_mla==false) else
  // branch honors it -- ON drops the score/P memory terms + softmax memory/energy;
  // OFF keeps them (today's behavior). The use_flash_mla==true flash branch is
  // untouched. Dormant for paper-1 (no MLA model).
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // paper2 CPU-memory/NVLink-C2C KV offload tier: neither of paper2's Fig5
  // device_HBM cases (llama4_maverick / GQA -> AttentionGenExecutionGPU;
  // deepseekR1 / absorb-MLA -> AbsorbMLAGenExecutionGPU) reaches this
  // non-absorb MLA gen path (deepseekR1's preset sets use_absorb=true), so it
  // was deliberately left un-patched. Fail loudly rather than silently
  // computing wrong timing if this path is ever reached with the offload
  // tier turned on (e.g. a future non-absorb-MLA + cpu_kv_offload config).
  if (cpuKvOffloadActive(config)) {
    fail("cpu_kv_offload: unmodeled attention path MultiLatentAttentionGenExecutionGPU (non-absorb MLA gen)");
  }

  std::vector<int> orig_shape = input->shape;
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    hw_metric total_kv_read_size = 0;
    hw_metric total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;

      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;

      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        total_kv_read_size += n * (2.0 * head_dim + qk_rope_head_dim) * num_heads * input->precision_byte;
        total_act_size += m * (2.0 * head_dim + qk_rope_head_dim) * num_heads * input->precision_byte;
      } else {
        accumul_memory_duration += memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      }
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
    }
    exec_status.memory_duration += accumul_memory_duration;

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

     // Score //
    hw_metric total_kv_read_size = 0;
    hw_metric total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        // ON: score (m*n) never materializes -> drop from Score traffic (ruling #3).
        memory_size = (m * k + k * n + (use_flash_attention ? 0.0 : (double)m * n)) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
          total_kv_read_size += k * n * input->precision_byte;
          total_act_size += (m * k + (use_flash_attention ? 0.0 : (double)m * n)) * input->precision_byte;
        } else {
          memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
          accumul_memory_duration += memory_duration;
        }
      }
      accumul_len += n;
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      // I3: fill_amortize_calls=2, standardizing to GQA-gen's convention
      // (attention_gen_impl.cpp:98,187) -- one staging pool serves both the
      // Score and Context page-fill per layer, so the pipeline-fill latency
      // should be shared across 2 calls, not charged in full to each.
      accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size, 2);
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      // score write into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      }

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      // score write into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      // ON: softmax on-chip -> zero memory (ruling #3). OFF: 2*m*n*heads.
      memory_size = use_flash_attention ? 0.0 : (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P is scarce-tier activation traffic; on HBF-family
      // memory_bandwidth is repurposed to flash_read_bandwidth (device.cpp:32-33),
      // so price the softmax round-trip at the activation tier (matches
      // getAttentionMemoryDuration's act-tier bandwidth).
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      // Softmax score read + P write into the energy counters only when materialized.
      if (!use_flash_attention) {
        if(use_ramulator){
          memory_duration = 0;

          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                  DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;

          // store output
          temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                  DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;
        }
        else{
          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
          exec_status += temp;

          // store output
          temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
          exec_status += temp;
        }
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
      exec_status.batch_dependent_duration += std::max(compute_duration, memory_duration);
      // PP_FLAGS_SPEC §4.3: same softmax->hide-budget contribution as the GQA path.
      if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    total_kv_read_size = 0;
    total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        // ON: the softmax-weight P (read here as m*k) never materializes -> drop.
        memory_size = ((use_flash_attention ? 0.0 : (double)m * k) + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
          total_kv_read_size += k * n * input->precision_byte;
          total_act_size += ((use_flash_attention ? 0.0 : (double)m * k) + m * n) * input->precision_byte;
        } else {
          memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
          accumul_memory_duration += memory_duration;
        }
      }
      accumul_len += k;
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      // I3: fill_amortize_calls=2 -- see the Score loop's I3 comment above.
      accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size, 2);
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      // score (P) read into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // score (P) read into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  ((config.use_hbf && config.hbf_config.num_flash_stacks > 0) ? config.hbf_config.flash_read_bandwidth : memory_bandwidth) / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input

  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    // Program latency amortized across this stage's per-layer write stream --
    // see getKVWriteDuration's program_latency_amortize_calls doc.
    int layers_per_stage = device->model_config.num_layers /
        (device->model_config.pp_dg > 0 ? device->model_config.pp_dg : 1);
    time_ns kv_write = getKVWriteDuration(config, num_seq, num_kv_heads, head_dim, input->precision_byte, compressed_kv, layer_info.kv_lora_rank, qk_rope_head_dim, device->model_config.input_len, device->model_config.output_len, layer_info.local_attention_window, layers_per_stage);
    // Use total compute (Scoring + Context) for overlap; exec_status.compute_duration
    // accumulates across both phases and is correct here.
    time_ns unhidden_write = std::max((time_ns)0, kv_write - exec_status.compute_duration);
    exec_status.total_duration += unhidden_write;
    exec_status.batch_dependent_duration += unhidden_write;
    exec_status.kv_write_duration = unhidden_write;
  }

  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops =
    config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;
  // FA master flag (ruling #3, materialized else-branch only). Dormant for paper-1.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      // Add per-iteration value, not the running sum (running sum causes N²/2 over-count).
      time_ns iter_compute = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      accumul_compute_duration += iter_compute;
      exec_status.compute_duration += iter_compute;

      time_ns iter_memory = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += iter_memory;
      exec_status.memory_duration += iter_memory;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        // ON: score (m*n) never materializes -> drop from Score traffic (ruling #3).
        memory_size = (m * k + k * n + (use_flash_attention ? 0.0 : (double)m * n)) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      // score write into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      }

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      // score write into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      // ON: softmax on-chip -> zero memory (ruling #3). OFF: 2*m*n*heads.
      memory_size = use_flash_attention ? 0.0 : (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P is scarce-tier activation traffic; on HBF-family
      // memory_bandwidth is repurposed to flash_read_bandwidth (device.cpp:32-33),
      // so price the softmax round-trip at the activation tier (matches
      // getAttentionMemoryDuration's act-tier bandwidth).
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      // Softmax score read + P write into the energy counters only when materialized.
      if (!use_flash_attention) {
        if(use_ramulator){
          memory_duration = 0;

          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                  DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;

          // store output
          temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                  DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;
        }
        else{
          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
          exec_status += temp;

          // store output
          temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
          exec_status += temp;
        }
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
      exec_status.batch_dependent_duration += std::max(compute_duration, memory_duration);
      // PP_FLAGS_SPEC §4.3: same softmax->hide-budget contribution as the GQA path.
      if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        // ON: the softmax-weight P (read here as m*k) never materializes -> drop.
        memory_size = ((use_flash_attention ? 0.0 : (double)m * k) + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      // score (P) read into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // score (P) read into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;
  // FA master flag (ruling #3, materialized else-branch only). Dormant for paper-1.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      // Add per-iteration value, not the running sum (running sum causes N²/2 over-count).
      time_ns iter_compute = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      accumul_compute_duration += iter_compute;
      exec_status.compute_duration += iter_compute;

      time_ns iter_memory = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += iter_memory;
      exec_status.memory_duration += iter_memory;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        // ON: score (m*n) never materializes -> drop from Score traffic (ruling #3).
        memory_size = (m * k + k * n + (use_flash_attention ? 0.0 : (double)m * n)) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);

      // score write into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      }

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      // score write into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      // ON: softmax on-chip -> zero memory (ruling #3). OFF: 2*m*n*heads.
      memory_size = use_flash_attention ? 0.0 : (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P is scarce-tier activation traffic; on HBF-family
      // memory_bandwidth is repurposed to flash_read_bandwidth (device.cpp:32-33),
      // so price the softmax round-trip at the activation tier (matches
      // getAttentionMemoryDuration's act-tier bandwidth).
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      // Softmax score read + P write into the energy counters only when materialized.
      if (!use_flash_attention) {
        if(use_ramulator){
          memory_duration = 0;

          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                  DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;

          // store output
          temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                  DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;
        }
        else{
          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
          exec_status += temp;

          // store output
          temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
          exec_status += temp;
        }
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
      exec_status.batch_dependent_duration += std::max(compute_duration, memory_duration);
      // PP_FLAGS_SPEC §4.3: same softmax->hide-budget contribution as the GQA path.
      if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        // ON: the softmax-weight P (read here as m*k) never materializes -> drop.
        memory_size = ((use_flash_attention ? 0.0 : (double)m * k) + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      // score (P) read into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // score (P) read into the energy counters only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;
  // FA master flag (ruling #3, materialized else-branch only). Dormant for paper-1.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    hw_metric total_kv_read_size = 0;
    hw_metric total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;

      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;

      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        total_kv_read_size += n * (kv_lora_rank + qk_rope_head_dim) * input->precision_byte;
        total_act_size += num_heads * (kv_lora_rank + qk_rope_head_dim + kv_lora_rank) * input->precision_byte;
      } else if (cpuKvOffloadActive(config)) {
        // paper2 CPU-offload tier: deepseekR1's Fig5 device_HBM case (absorb
        // MLA, use_flash_mla=true) fires this branch. Mirror the HBF flash
        // branch's aggregate-then-single-post-loop-call structure above.
        total_kv_read_size += n * (kv_lora_rank + qk_rope_head_dim) * input->precision_byte;
        total_act_size += num_heads * (kv_lora_rank + qk_rope_head_dim + kv_lora_rank) * input->precision_byte;
      } else {
        accumul_memory_duration += memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      }
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
    } else if (cpuKvOffloadActive(config)) {
      accumul_memory_duration = hbmKvOffloadReadDuration(config, total_kv_read_size, total_act_size, sequences_metadata->p2_kv_offload_fraction);
    }
    exec_status.memory_duration += accumul_memory_duration;

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // paper2 CPU-memory/NVLink-C2C KV offload tier: deepseekR1's Fig5
    // device_HBM case sets use_flash_mla=true, so this non-flash-mla
    // (separate Scoring/Softmax/Context phases) branch was deliberately left
    // un-patched -- its Scoring-NoPE/Scoring-RoPE/Softmax sub-phases below
    // don't even have an HBF flash-storage variant to mirror. Fail loudly
    // rather than silently mis-time this path if it's ever reached with the
    // offload tier on.
    if (cpuKvOffloadActive(config)) {
      fail("cpu_kv_offload: unmodeled attention path AbsorbMLAGenExecutionGPU (use_flash_mla=false)");
    }
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      // ON: nope-score (m*n*heads) never materializes -> drop (ruling #3).
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      // ON: rope-score (m*n*heads) never materializes -> drop (ruling #3).
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      // ON: softmax on-chip -> zero memory (ruling #3). OFF: 2*m*n*heads.
      memory_size = use_flash_attention ? 0.0 : (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P priced at the scarce activation tier -- see Block-A note.
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      // Softmax score read + P write into the energy counters only when materialized.
      if (!use_flash_attention) {
        if(use_ramulator){
          memory_duration = 0;

          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                  DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;

          // store output
          temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                  DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;
        }
        else{
          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
          exec_status += temp;

          // store output
          temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
          exec_status += temp;
        }
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
      exec_status.batch_dependent_duration += std::max(compute_duration, memory_duration);
      // PP_FLAGS_SPEC §4.3: same softmax->hide-budget contribution as the GQA path.
      if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    hw_metric total_kv_read_size = 0;
    hw_metric total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop.
      memory_size = ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        total_kv_read_size += k * n * input->precision_byte;
        total_act_size += ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + m * n * num_heads) * input->precision_byte;
      } else {
        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;


      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }


  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  ((config.use_hbf && config.hbf_config.num_flash_stacks > 0) ? config.hbf_config.flash_read_bandwidth : memory_bandwidth) / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input

  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    // Program latency amortized across this stage's per-layer write stream --
    // see getKVWriteDuration's program_latency_amortize_calls doc.
    int layers_per_stage = device->model_config.num_layers /
        (device->model_config.pp_dg > 0 ? device->model_config.pp_dg : 1);
    time_ns kv_write = getKVWriteDuration(config, num_seq, num_kv_heads, head_dim, input->precision_byte, true, kv_lora_rank, qk_rope_head_dim, device->model_config.input_len, device->model_config.output_len, layer_info.local_attention_window, layers_per_stage);
    // Use total compute (Scoring + Context) for overlap; exec_status.compute_duration
    // accumulates across both phases and is correct here.
    time_ns unhidden_write = std::max((time_ns)0, kv_write - exec_status.compute_duration);
    exec_status.total_duration += unhidden_write;
    exec_status.batch_dependent_duration += unhidden_write;
    exec_status.kv_write_duration = unhidden_write;
  }

  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;
  // FA master flag (ruling #3, materialized else-branch only). Dormant for paper-1.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output
                    input->precision_byte;
      total_memory_size += memory_size;

      // Add per-iteration value, not the running sum (running sum causes N²/2 over-count).
      time_ns iter_compute = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      accumul_compute_duration += iter_compute;
      exec_status.compute_duration += iter_compute;

      time_ns iter_memory = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += iter_memory;
      exec_status.memory_duration += iter_memory;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      // ON: nope-score (m*n*heads) never materializes -> drop (ruling #3).
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      // ON: rope-score (m*n*heads) never materializes -> drop (ruling #3).
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      // ON: softmax on-chip -> zero memory (ruling #3). OFF: 2*m*n*heads.
      memory_size = use_flash_attention ? 0.0 : (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P priced at the scarce activation tier -- see Block-A note.
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      // Softmax score read + P write into the energy counters only when materialized.
      if (!use_flash_attention) {
        if(use_ramulator){
          memory_duration = 0;

          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                  DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;

          // store output
          temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                  DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;
        }
        else{
          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
          exec_status += temp;

          // store output
          temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
          exec_status += temp;
        }
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
      exec_status.batch_dependent_duration += std::max(compute_duration, memory_duration);
      // PP_FLAGS_SPEC §4.3: same softmax->hide-budget contribution as the GQA path.
      if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop.
      memory_size = ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      // k_cache->setShape(shape);
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;
  // FA master flag (ruling #3, materialized else-branch only). Dormant for paper-1.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_gen_process_token();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output
                    input->precision_byte;
      total_memory_size += memory_size;

      // Add per-iteration value, not the running sum (running sum causes N²/2 over-count).
      time_ns iter_compute = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      accumul_compute_duration += iter_compute;
      exec_status.compute_duration += iter_compute;

      time_ns iter_memory = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += iter_memory;
      exec_status.memory_duration += iter_memory;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{  
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      // ON: nope-score (m*n*heads) never materializes -> drop (ruling #3).
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      // ON: rope-score (m*n*heads) never materializes -> drop (ruling #3).
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value (score) -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      // ON: softmax on-chip -> zero memory (ruling #3). OFF: 2*m*n*heads.
      memory_size = use_flash_attention ? 0.0 : (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P priced at the scarce activation tier -- see Block-A note.
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      // Softmax score read + P write into the energy counters only when materialized.
      if (!use_flash_attention) {
        if(use_ramulator){
          memory_duration = 0;

          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                  DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;

          // store output
          temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                  DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
          exec_status += temp;
          memory_duration += temp.memory_duration;
        }
        else{
          ExecStatus temp;

          // read input
          input->setShape({m, n * num_heads});
          temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
          exec_status += temp;

          // store output
          temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
          exec_status += temp;
        }
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
      exec_status.batch_dependent_duration += std::max(compute_duration, memory_duration);
      // PP_FLAGS_SPEC §4.3: same softmax->hide-budget contribution as the GQA path.
      if (config.kv_write_softmax_hide) exec_status.compute_duration += compute_duration;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop.
      memory_size = ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;

      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    
      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output -- only when materialized (OFF).
      if (!use_flash_attention) {
        input->setShape({num_heads, accumul_len});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.batch_dependent_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

}  // namespace llm_system