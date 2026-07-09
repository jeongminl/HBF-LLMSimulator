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

ExecStatus AttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);
  // Tensor_Ptr k_cache = tensor.at(1);
  //  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  // FlashAttention master flag (FA_FLAG_SPEC.md, item I1/I14). ON: score/P never
  // materialize -> drop the score term from Scoring, the P (score-read) term from
  // Context, charge softmax ZERO memory, and DO NOT issue the score write / softmax
  // read+write / context score-read into the energy/util counters. OFF (new
  // default): keep the materialized score/P terms, charge softmax 2*m*n*heads, and
  // keep those energy counters firing.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  // paper2 CPU-memory/NVLink-C2C KV offload tier: this Sum (prefill) GPU path
  // is unreachable under decode_mode (paper2's Fig5 device_HBM harness runs
  // decode_mode on, injection_rate 0), so it was deliberately left
  // un-patched. Fail loudly rather than silently mis-time it if it's ever
  // reached with the offload tier on (e.g. a mixed prefill+decode run).
  if (cpuKvOffloadActive(config)) {
    fail("cpu_kv_offload: unmodeled attention path AttentionSumExecutionGPU");
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_sum_process_token();
  // For HBF configs: KV (key cache) is on flash, activations (query + score
  // output) are on the scarce tier (HBM for HBF, logic SRAM for HBF+). Sizes are
  // accumulated across all sequences and getAttentionMemoryDuration is called once
  // after the loop (matches AttentionGen/AttentionMixed) so the flash page-read
  // latency is paid once per SRAM-sized chunk, not once per sequence.
  time_ns accumul_compute_duration = 0;
  hw_metric total_kv_read_size = 0;
  hw_metric total_act_size = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;
    // Llama-4-style interleaved local/global attention: cap this layer's scoring
    // read at its effective window (0 = no cap, every model except
    // llama4_maverick/scout's local layers -- see LayerInfo::local_attention_window
    // / model_config.h), mirroring attention_gen_impl.cpp's Scoring-loop cap (:69).
    if (layer_info.local_attention_window > 0) n = std::min(n, layer_info.local_attention_window);

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: score (m*n*heads) never materializes -> drop from Scoring traffic.
    memory_size =
        1.0 * (m * k * num_heads + k * n * num_kv_heads + (use_flash_attention ? 0.0 : (double)m * n * num_heads)) *
        input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // kv_read_size = k*n*kv_heads (the key cache rows read for scoring);
    // act_size = (query + output score) = (m*k*heads + m*n*heads); score dropped under ON.
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      total_kv_read_size += (hw_metric)k * n * num_kv_heads * input->precision_byte;
      total_act_size += ((hw_metric)m * k * num_heads + (use_flash_attention ? (hw_metric)0 : (hw_metric)m * n * num_heads)) * input->precision_byte;
      accumul_compute_duration += compute_duration;
    } else {
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    }

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // read query
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      // read value
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;

      // write score -- only when materialized (OFF); ON keeps it on-chip (I14).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                           DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
      memory_duration = accumul_memory_duration;
    }
    else{
      // read query
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device,ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      // read value
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score -- only when materialized (OFF); ON keeps it on-chip (I14).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }
    // Flash-analytic path (HBF, non-ramulator): memory_duration is deferred to a
    // single post-loop getAttentionMemoryDuration call (see below) so the page-read
    // latency is paid once per SRAM chunk, not once per sequence. The ramulator path
    // still overwrites memory_duration per sequence above (pre-existing behavior,
    // unrelated to the flash analytic model) and is added here as before.
    bool defer_to_chunk_aggregate = config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator;
    if (!defer_to_chunk_aggregate) {
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator) {
    // I3: fill_amortize_calls=2, standardizing this K+V page-fill amortization
    // to match GQA-gen's convention (attention_gen_impl.cpp:98,187) -- one
    // staging pool serves both the K and V page-fill per layer, so the
    // pipeline-fill latency should be shared across 2 calls, not charged in
    // full to each. Was defaulting to 1 (2x over-charge). Prefill-only path
    // (GQA-Sum runs on the prefill/sum leg) -- report whether any published
    // decode cell moves (should be none; this loop never executes for decode).
    time_ns accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size, 2);
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  // Softmax + Scale + Mask//
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;
    // Same cap as the Scoring loop above, mirroring attention_gen_impl.cpp's
    // Softmax-loop cap (:125).
    if (layer_info.local_attention_window > 0) n = std::min(n, layer_info.local_attention_window);

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // ON: softmax runs on-chip -> zero memory (today's behavior). OFF (materialized):
    // read the resident score + write P back = 2*m*n*heads, unifying GQA-Sum to MLA's
    // existing 2*m*n softmax-memory term (I1/I9). This closes I14: today the score
    // read/write below fires into the energy counters even though memory_size==0.
    memory_size = use_flash_attention ? 0.0 : 2.0 * (double)m * n * num_heads * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // The materialized score/P round-trip is activation traffic on the SCARCE tier
    // (HBM stack or logic SRAM), NOT flash. Mirror getAttentionMemoryDuration's
    // act-tier bandwidth: on HBF-family the local `memory_bandwidth` is repurposed to
    // flash_read_bandwidth (device.cpp:32-33 / test.cpp:144), so use hbm_read_bandwidth
    // (num_hbm_stacks>0) else logic_sram_bandwidth; plain-HBM keeps memory_bandwidth.
    // (memory_size==0 under ON, so this is a no-op there.)
    hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
        ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                  : config.hbf_config.logic_sram_bandwidth)
        : memory_bandwidth;
    memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

    // Score read + P write into the energy/util counters only when materialized (OFF).
    if (!use_flash_attention) {
      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                           DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        temp =
            issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                           DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
      else{
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  // Same aggregate-then-call-once pattern as Scoring above.
  time_ns context_accumul_compute_duration = 0;
  hw_metric context_total_kv_read_size = 0;
  hw_metric context_total_act_size = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    // Llama-4-style interleaved local/global attention: same cap as the Scoring
    // loop above, applied to k (the KV-position count) here since Context swaps
    // k/n vs. Scoring's roles -- mirrors attention_gen_impl.cpp's Context-loop
    // cap (:152).
    if (layer_info.local_attention_window > 0) k = std::min(k, layer_info.local_attention_window);
    n = head_dim;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop
    // from Context traffic. m*n*heads is the context output write (kept).
    memory_size =
        1.0 * ((use_flash_attention ? 0.0 : (double)m * k * num_heads) + k * n * num_kv_heads + m * n * num_heads) *
        input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;

    // Context phase uses the flash model on HBF presets (same as Scoring above).
    // kv_read_size = V-cache(k*n*kv_heads); act_size = P read (m*k*heads, dropped
    // under ON) + output write (m*n*heads).
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
      context_total_kv_read_size += (hw_metric)k * n * num_kv_heads * input->precision_byte;
      context_total_act_size += ((use_flash_attention ? (hw_metric)0 : (hw_metric)m * k * num_heads) + (hw_metric)m * n * num_heads) * input->precision_byte;
      context_accumul_compute_duration += compute_duration;
    } else {
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    }

    if (use_ramulator) {
      // Accumulate memory_duration across all three ramulator ops (score-read, kv-read, write).
      // Overwriting would leave only the final write's latency (ramulator path only).
      time_ns accumul_memory_duration = 0;
      auto shape = input->getShape();
      ExecStatus temp;

      // read score -- only when materialized (OFF); ON keeps it on-chip (I1/I14).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                           DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      // read kv
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      auto shape = input->getShape();
      ExecStatus temp;

      // read score -- only when materialized (OFF); ON keeps it on-chip (I1/I14).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      // read kv
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    bool context_defer_to_chunk_aggregate = config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator;
    if (!context_defer_to_chunk_aggregate) {
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator) {
    // I3: fill_amortize_calls=2 -- see the Scoring loop's I3 comment above.
    time_ns context_accumul_memory_duration = getAttentionMemoryDuration(config, context_total_kv_read_size, context_total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size, 2);
    exec_status.total_duration += std::max(context_accumul_compute_duration, context_accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};

// need to fix
ExecStatus AttentionSumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
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

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_sum_process_token();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);

      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;

      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // memory_size = (m * k * num_heads + k * n * num_kv_heads) *
    // input->precision_byte; total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      auto shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n * num_kv_heads;
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionSumExecutionPIM(Device_Ptr device,
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

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_sum_process_token();
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (m * k * num_heads + k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      auto shape = input->getShape();
      shape.at(0) = m;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = m;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // memory_size = (m * k * num_heads + k * n * num_kv_heads) *
    // input->precision_byte; total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    flops = m * k * n * 2.0 * num_heads;
    total_flops += flops;

    memory_size = (k * n * num_kv_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    if (use_ramulator) {
      auto shape = input->getShape();
      shape.at(0) = k;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ATTENTION_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      auto shape = input->getShape();
      shape.at(0) = k;

      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

// Multi Latent Attention // 
ExecStatus MultiLatentAttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);

  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  // paper2 CPU-memory/NVLink-C2C KV offload tier: this Sum (prefill) GPU path
  // is unreachable under decode_mode (paper2's Fig5 device_HBM harness runs
  // decode_mode on, injection_rate 0), so it was deliberately left
  // un-patched. Fail loudly rather than silently mis-time it if it's ever
  // reached with the offload tier on (e.g. a mixed prefill+decode run).
  if (cpuKvOffloadActive(config)) {
    fail("cpu_kv_offload: unmodeled attention path MultiLatentAttentionSumExecutionGPU");
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;
  // seq_list is get_sum() for the whole function (both the flash and non-flash
  // branches below iterate the same list), so the batch-wide M is get_sum_process_token()
  // in both branches -- no per-branch distinction needed.
  double batch_m = sequences_metadata->get_sum_process_token();

  if(use_flash_attention) {
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      hw_metric shared_mem_size = 256 * 1024; // Byte

      int block_size_r = std::min((shared_mem_size / (4 * k * input->precision_byte)), static_cast<double>(k)); // num_rows, B_r
      int block_size_c = shared_mem_size / (4 * k * input->precision_byte);

      int num_tile_r = (m + block_size_r - 1)/ block_size_r;
      int num_tile_c = (m + block_size_c - 1) / block_size_c;

      int q_i = block_size_r * (head_dim + qk_rope_head_dim);
      int k_i = block_size_c * (head_dim + qk_rope_head_dim);

      int v_i = block_size_c * head_dim;
      int o_i = block_size_r * head_dim;

      int m_i = block_size_r;
      int l_i = block_size_r;

      flops = 1.0 * num_tile_r * num_tile_c * 
              (2.0 * block_size_r * (head_dim + qk_rope_head_dim) * block_size_c + 
               2.0 * block_size_r * head_dim * block_size_c) * num_heads; // consider only matmuls
      total_flops += flops;

      memory_size = (1.0 * q_i * num_tile_r * num_tile_c +   // query
                     1.0 * (k_i + v_i) * num_tile_c +        // key + value
                     2.0 * o_i * num_tile_r * num_tile_c +  // output, read/write output 
                     2.0 * (m_i + l_i) * num_tile_r * num_tile_c) *
                     num_heads * input->precision_byte;      // consider only matmuls
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write output
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write norm stat
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write output
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write norm stat
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  else{
    // Scoring //
    int num_seq = seq_list.size();
    // I15: same aggregate-then-call-once flash-read pattern as GQA-Sum's
    // Scoring/Context loops (AttentionSumExecutionGPU above) -- MLA-Sum was
    // reading KV at plain HBM bandwidth unconditionally, even under HBF/HBF+
    // presets. GQA cells are unaffected (MLA-only code path).
    time_ns accumul_compute_duration = 0;
    hw_metric total_kv_read_size = 0;
    hw_metric total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // kv_read_size = k*n*heads (the key/value cache read for scoring);
      // act_size = (query + output score) = (m*k*heads + m*n*heads).
      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        total_kv_read_size += (hw_metric)k * n * num_heads * input->precision_byte;
        total_act_size += ((hw_metric)m * k * num_heads + (hw_metric)m * n * num_heads) * input->precision_byte;
        accumul_compute_duration += compute_duration;
      } else {
        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      }

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      bool defer_to_chunk_aggregate = config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator;
      if (!defer_to_chunk_aggregate) {
        exec_status.total_duration += std::max(compute_duration, memory_duration);
      }
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator) {
      time_ns accumul_memory_duration = getAttentionMemoryDuration(config, total_kv_read_size, total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
      exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
    }

    // Softmax + Scale + Mask //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads;
      total_flops += flops;

      memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
      // memory_size = 0; // can be overlapped
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P priced at the scarce activation tier: on HBF-family
      // memory_bandwidth is repurposed to flash_read_bandwidth (device.cpp:32-33).
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Context //
    // I15: same aggregate-then-call-once flash-read pattern as GQA-Sum's
    // Context loop (see above).
    time_ns context_accumul_compute_duration = 0;
    hw_metric context_total_kv_read_size = 0;
    hw_metric context_total_act_size = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = 1.0 *
                    (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;

      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // kv_read_size = V-cache(k*n*heads); act_size = score read(m*k*heads) + output write(m*n*heads)
      // (mirrors GQA-Sum's Context split: kv_read is the KV-cache term only).
      if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
        context_total_kv_read_size += (hw_metric)k * n * num_heads * input->precision_byte;
        context_total_act_size += ((hw_metric)m * k * num_heads + (hw_metric)m * n * num_heads) * input->precision_byte;
        context_accumul_compute_duration += compute_duration;
      } else {
        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      }

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        // read score
        auto shape = input->getShape();

        ExecStatus temp;
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read kv

        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::GPU,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read score
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      bool context_defer_to_chunk_aggregate = config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator;
      if (!context_defer_to_chunk_aggregate) {
        exec_status.total_duration += std::max(compute_duration, memory_duration);
      }
    }
    if (config.use_hbf && config.hbf_config.num_flash_stacks > 0 && !use_ramulator) {
      time_ns context_accumul_memory_duration = getAttentionMemoryDuration(config, context_total_kv_read_size, context_total_act_size, layer_info.use_chunked_attention, layer_info.chunk_size);
      exec_status.total_duration += std::max(context_accumul_compute_duration, context_accumul_memory_duration);
    }
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore original shape
  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};

// Multi Latent Attention // 
ExecStatus MultiLatentAttentionSumExecutionLogic(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);

  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;
  // seq_list is get_sum() for the whole function (both the flash and non-flash
  // branches below iterate the same list), so the batch-wide M is get_sum_process_token()
  // in both branches -- no per-branch distinction needed.
  double batch_m = sequences_metadata->get_sum_process_token();

  if(use_flash_attention) {
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      hw_metric shared_mem_size = 256 * 1024; // Byte

      int block_size_r = shared_mem_size / (4 * k * input->precision_byte); // num_rows
      int block_size_c = k;  // num_rows

      int num_tile_r = (m + block_size_r - 1) / block_size_r;
      int num_tile_c = (m + block_size_c - 1) / block_size_c;

      int q_i = block_size_r * (head_dim + qk_rope_head_dim);
      int k_i = block_size_c * (head_dim + qk_rope_head_dim);
      int v_i = block_size_c * head_dim;
      int o_i = block_size_r * head_dim;

      int m_i = block_size_r;
      int l_i = block_size_r;

      flops = 1.0 * num_tile_r * num_tile_c * 
              (2.0 * block_size_r * (head_dim + qk_rope_head_dim) * block_size_c + 
               2.0 * block_size_r * head_dim * block_size_c) * num_heads; // consider only matmuls
      total_flops += flops;

      memory_size = (1.0 * q_i * num_tile_r * num_tile_c +        // query
                     1.0 * (k_i + v_i) * num_tile_c +             // key + value
                     2.0 * o_i * num_tile_r * num_tile_c +        // output, read/write output
                     2.0 * (m_i + l_i) * num_tile_r * num_tile_c  // read/write norm_stat
                    ) * num_heads * input->precision_byte;        // consider only matmuls
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write output
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write norm stat
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write output
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write norm stat
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  else{
    // Scoring //
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Softmax + Scale + Mask //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads;
      total_flops += flops;

      memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
      // memory_size = 0; // can be overlapped
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P priced at the scarce activation tier: on HBF-family
      // memory_bandwidth is repurposed to flash_read_bandwidth (device.cpp:32-33).
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        ExecStatus temp;
          // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Context //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = 1.0 *
                    (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        // read score
        auto shape = input->getShape();

        ExecStatus temp;
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read kv

        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::LOGIC,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read score
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  
  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore original shape
  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};

ExecStatus MultiLatentAttentionSumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);

  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;
  // seq_list is get_sum() for the whole function (both the flash and non-flash
  // branches below iterate the same list), so the batch-wide M is get_sum_process_token()
  // in both branches -- no per-branch distinction needed.
  double batch_m = sequences_metadata->get_sum_process_token();

  if(use_flash_attention) {
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      hw_metric shared_mem_size = 256 * 1024; // Byte

      int block_size_r = shared_mem_size / (4 * k * input->precision_byte); // num_rows
      int block_size_c = k;  // num_rows

      int num_tile_r = (m + block_size_r - 1) / block_size_r;
      int num_tile_c = (m + block_size_c - 1) / block_size_c;

      int q_i = block_size_r * (head_dim + qk_rope_head_dim);
      int k_i = block_size_c * (head_dim + qk_rope_head_dim);
      int v_i = block_size_c * head_dim;
      int o_i = block_size_r * head_dim;

      int m_i = block_size_r;
      int l_i = block_size_r;

      flops = 1.0 * num_tile_r * num_tile_c * 
              (2.0 * block_size_r * (head_dim + qk_rope_head_dim) * block_size_c + 
               2.0 * block_size_r * head_dim * block_size_c) * num_heads; // consider only matmuls
      total_flops += flops;

      memory_size = (1.0 * q_i * num_tile_r * num_tile_c +        // query
                     1.0 * (k_i + v_i) * num_tile_c +             // key + value
                     2.0 * o_i * num_tile_r * num_tile_c +        // output, read/write output
                     2.0 * (m_i + l_i) * num_tile_r * num_tile_c  // read/write norm_stat
                    ) * num_heads * input->precision_byte;        // consider only matmuls
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write output
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write norm stat
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = q_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read key/value
        shape.at(0) = (k_i + v_i);
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read output
        shape.at(0) = o_i * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write output
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;

        // read norm stat
        shape.at(0) = (m_i + l_i) * num_tile_r;
        shape.at(1) = num_tile_c * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;
        
        // write norm stat
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }
  else{
    // Scoring //
    int num_seq = seq_list.size();
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read value
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write score
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Softmax + Scale + Mask //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads;
      total_flops += flops;

      memory_size = 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;

      // memory_size = 0; // can be overlapped
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      // Materialized score/P priced at the scarce activation tier -- see above.
      hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
          ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                    : config.hbf_config.logic_sram_bandwidth)
          : memory_bandwidth;
      memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        ExecStatus temp;
          // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        ExecStatus temp;
        // read query
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }

    // Context //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = 1.0 *
                    (1.0 * m * k * num_heads + 1.0 * k * n * num_heads +
                    1.0 * m * n * num_heads) *
                    input->precision_byte;
      
      total_memory_size += memory_size;

      compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;
        // read score
        auto shape = input->getShape();

        ExecStatus temp;
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);

        temp =
            issueRamulator(device, LayerType::MLA_SUM, ProcessorType::PIM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read score
        auto shape = input->getShape();
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // read kv
        shape.at(0) = k;
        shape.at(1) = n * num_kv_heads;
        input->setShape(shape);

        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // write attention output
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
      exec_status.total_duration += std::max(compute_duration, memory_duration);
    }
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore original shape
  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  return exec_status;
};


ExecStatus AbsorbMLASumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr output = tensor.at(1);
  std::vector<int> orig_shape = input->shape;

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  // FA master flag (FA_FLAG_SPEC.md, dormant Absorb-Sum path). ON: score/P never
  // materialize -> drop score(nope+rope) and Context-P memory terms, charge softmax
  // ZERO memory, and skip the score-write / softmax rd+wr / context-score-read energy
  // issues. OFF: materialized (today's behavior).
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  // paper2 CPU-memory/NVLink-C2C KV offload tier: this Sum (prefill) GPU path
  // is unreachable under decode_mode (paper2's Fig5 device_HBM harness runs
  // decode_mode on, injection_rate 0), so it was deliberately left
  // un-patched. Fail loudly rather than silently mis-time it if it's ever
  // reached with the offload tier on (e.g. a mixed prefill+decode run).
  if (cpuKvOffloadActive(config)) {
    fail("cpu_kv_offload: unmodeled attention path AbsorbMLASumExecutionGPU");
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_sum_process_token();
  assertTrue(num_heads == input->shape[0], "input tensor shape is not match in num_head");

  // Scoring for nope //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = kv_lora_rank;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: nope-score (m*n*heads) never materializes -> drop from traffic.
    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // read query x W_UK
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write score for nope -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->shape.resize(0);
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                           DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
      memory_duration = accumul_memory_duration;
    }
    else{
      // read query x W_UK
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score for nope -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;
        input->shape.resize(0);
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Scoring for rope //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: rope-score (m*n*heads) never materializes -> drop from traffic.
    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write score for rope -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                           DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }
      memory_duration = accumul_memory_duration;
    }
    else{
      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score for rope -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // ON: softmax on-chip -> zero memory. OFF: score read + P write = 2*m*n*heads.
    memory_size = use_flash_attention ? 0.0 : 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // Materialized score/P priced at the scarce activation tier -- see MLA-Sum note.
    hw_metric softmax_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
        ? ((config.hbf_config.num_hbm_stacks > 0) ? config.hbf_config.hbm_read_bandwidth
                                                  : config.hbf_config.logic_sram_bandwidth)
        : memory_bandwidth;
    memory_duration = memory_size / softmax_bw * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    // Score read + P write into the energy counters only when materialized (OFF).
    if (!use_flash_attention) {
      if (use_ramulator) {
        time_ns accumul_memory_duration = 0;

        // read input
        std::vector<int> shape = {1,1};
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        ExecStatus temp;
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                           DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        // store output
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                           DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
        memory_duration = accumul_memory_duration;
      }
      else{
        // read input
        std::vector<int> shape = {1,1};
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        ExecStatus temp;
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }

    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = kv_lora_rank;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop.
    memory_size = 1.0 *
                  ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      // read score
      time_ns accumul_memory_duration = 0;

      std::vector<int> shape = {1,1};
      ExecStatus temp;

      // read score -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->shape.resize(0);
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                           DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;
      }

      // read compressed_kv

      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::GPU,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      ExecStatus temp;

      // read score -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->shape.resize(0);
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      // read compressed_kv
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  assertTrue(total_flops > 0, "fail");
  assertTrue(total_memory_size > 0, "fail");

  input->setShape(orig_shape);

  return exec_status;
};

ExecStatus AbsorbMLASumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

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
  // FA master flag (dormant Absorb-Sum Logic path). ON: drop score/P memory terms,
  // zero softmax memory, skip the rope score-write + context score-read energy.
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring //
  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_sum_process_token();
  assertTrue(num_heads == input->shape[0], "input tensor shape is not match in num_head");

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = kv_lora_rank;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: nope-score (m*n*heads) never materializes -> drop from traffic.
    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
      (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);

      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      memory_duration = accumul_memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);

      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      shape = input->getShape();
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Scoring for rope //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: rope-score (m*n*heads) never materializes -> drop from traffic.
    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      time_ns accumul_memory_duration = 0;

      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;

      // write score for rope -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                           DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        accumul_memory_duration += temp.memory_duration;

        accumul_memory_duration += temp.memory_duration;
      }
      memory_duration = accumul_memory_duration;
    }
    else{
      // query_rope
      std::vector<int> shape = {1,1};
      shape.at(0) = m;
      shape.at(1) = k * num_heads;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_PE
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write score for rope -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = n * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // ON: softmax on-chip -> zero memory (this Logic path never issued softmax
    // energy or memory_duration; only total_memory_size reflected it).
    memory_size = use_flash_attention ? 0.0 : 2.0 * (1.0 * m * n * num_heads) * input->precision_byte;
    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;
    total_memory_size += memory_size;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = kv_lora_rank;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop.
    memory_size = 1.0 *
                  ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + 1.0 * k * n +
                   1.0 * m * n * num_heads) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      ExecStatus temp;

      // read score -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = k * num_heads;

        input->shape.resize(0);
        input->setShape(shape);
        temp =
            issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                           DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration = temp.memory_duration;
      }

      // read compressed kv
      shape.at(0) = k;
      shape.at(1) = n;

      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;

      input->shape.resize(0);
      input->setShape(shape);

      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::LOGIC,
                         DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      ExecStatus temp;

      // read score -- only when materialized (OFF).
      if (!use_flash_attention) {
        shape.at(0) = m;
        shape.at(1) = k * num_heads;
        input->shape.resize(0);
        input->setShape(shape);
        temp =
            getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;
      }

      // read compressed_kv
      shape.at(0) = k;
      shape.at(1) = n;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // write attention output
      shape.at(0) = m;
      shape.at(1) = n * num_heads;
      input->shape.resize(0);
      input->setShape(shape);
      temp =
          getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AbsorbMLASumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  // fail("not implemented");
  Tensor_Ptr input = tensor.at(0);
  // Tensor_Ptr k_cache = tensor.at(1);
  // Tensor_Ptr v_cache = tensor.at(2);

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
  // FA master flag (dormant Absorb-Sum PIM path). ON: drop score/P memory terms.
  // (This roofline path issues only GEMV reads -- no separate score read/write
  // energy to gate; softmax already charges no memory here.)
  bool use_flash_attention = layer_info.use_flash_attention;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_sum();
  Sequence::Ptr seq;

  // Scoring for NoPE//
  int num_seq = seq_list.size();
  double batch_m = sequences_metadata->get_sum_process_token();
  assertTrue(num_heads == input->shape[0], "input tensor shape is not match in num_head");

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = kv_lora_rank;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: nope-score (m*n*heads) never materializes -> drop from traffic.
    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
                   (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;
    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Scoring for RoPE //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = qk_rope_head_dim;
    n = seq->current_len + seq->num_process_token;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: rope-score (m*n*heads) never materializes -> drop from traffic.
    memory_size = (1.0 * m * k * num_heads + 1.0 * k * n+
                   (use_flash_attention ? 0.0 : 1.0 * m * n * num_heads)) *
                  input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = m;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  // Softmax + Scale + Mask //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    n = seq->current_len + seq->num_process_token;

    flops = 7.0 * m * n * num_heads;
    total_flops += flops;

    // memory_size = (m * k * num_heads + k * n * num_kv_heads) *
    // input->precision_byte; total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    // memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    exec_status.total_duration += compute_duration;
  }

  // Context //
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    time_ns compute_duration = 0;
    time_ns memory_duration = 0;

    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = kv_lora_rank;

    flops = 1.0 * m * k * n * 2.0 * num_heads;
    total_flops += flops;

    // ON: the softmax-weight P (read here as m*k*heads) never materializes -> drop.
    memory_size = ((use_flash_attention ? 0.0 : 1.0 * m * k * num_heads) + k * n + 1.0 * m * n * num_heads) * input->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / (compute_peak_flops * effectiveMFU(config, batch_m)) * 1000 * 1000 * 1000;
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

    exec_status.compute_duration += compute_duration;

    if (use_ramulator) {
      std::vector<int> shape = {1,1};
      shape.at(0) = k;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          issueRamulator(device, LayerType::ABSORBED_MLA_SUM, ProcessorType::PIM,
                         DRAMRequestType::kGEMV, PIMOperandType::kSrc, input);
      exec_status += temp;
      memory_duration = temp.memory_duration;
    }
    else{
      std::vector<int> shape = {1,1};
      shape.at(0) = k;

      input->shape.resize(0);
      input->setShape(shape);
      ExecStatus temp;
      temp =
          getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(compute_duration, memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};
}  // namespace llm_system