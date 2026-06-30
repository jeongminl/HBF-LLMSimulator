#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <tuple>

#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/base.h"
#include "module/status.h"
#include "model/model_config.h"
#include "scheduler/sequence.h"
#include "hardware/hardware_config.h"

namespace llm_system {

inline time_ns getLinearMemoryDuration(const SystemConfig& config, double m, double k, double n, int precision, hw_metric total_memory_size, hw_metric memory_bandwidth, int num_heads = 1, bool duplicated_input = false) {
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    auto& hbf = config.hbf_config;
    // Weight (k×n) is on flash; batched linear (num_heads>1) has num_heads
    // weight tiles regardless of whether the input is shared (duplicated_input).
    hw_metric weight_size = k * n * precision;
    if (num_heads > 1) {
      weight_size *= num_heads;
    }
    double weight_read_time = (weight_size / hbf.flash_read_bandwidth * 1e9) + hbf.flash_page_read_latency_ns;

    // Activations: input and output are on HBM or logic-die SRAM.
    // When num_hbm_stacks==0 (HBF+/CONV+), activations stage in logic-die SRAM
    // (~320 MB total, fast) → modeled as free (act_time=0).
    // act_read_size carries num_heads when input is not duplicated (to match act_write_size).
    hw_metric act_read_size = m * k * precision * (duplicated_input ? 1 : num_heads);
    hw_metric act_write_size = m * n * precision * num_heads;
    double act_time = 0;
    if (hbf.num_hbm_stacks > 0) {
      act_time = (act_read_size + act_write_size) / hbf.hbm_read_bandwidth * 1e9;
    }
    return std::max(weight_read_time, act_time);
  } else {
    return total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;
  }
}

// chunk_size_override: if > 0, overrides config.chunk_size for this call.
// Pass 0 (or omit) to use config.chunk_size.  Callers that pass layer_info.chunk_size
// should only do so when the layer graph sets a deliberate per-layer chunk; otherwise
// use 0 so the global system.chunk_size setting is honored.
inline time_ns getAttentionMemoryDuration(const SystemConfig& config, hw_metric kv_read_size, hw_metric act_size, bool use_chunked_attention = false, int chunk_size_override = 0) {
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    auto& hbf = config.hbf_config;

    double kv_read_time = 0;
    if (use_chunked_attention) {
      // Chunked attention: double-buffering hides read latency per chunk, so we
      // pay one page-read latency per chunk.  Chunk granularity:
      //   chunk_size_override > 0 → use caller's value (per-layer override).
      //   config.chunk_size > 0   → use global system setting.
      //   both == 0               → auto = full SRAM staging capacity.
      // All values clamped to sram_capacity (double-buffer can't stage more than SRAM).
      double sram_capacity = (double)hbf.sram_per_stack_bytes * hbf.num_flash_stacks;
      int effective_chunk = (chunk_size_override > 0) ? chunk_size_override : config.chunk_size;
      double chunk_bytes = (effective_chunk > 0)
          ? std::min((double)effective_chunk, sram_capacity)
          : sram_capacity;
      int num_chunks = (chunk_bytes > 0)
          ? (int)std::ceil((double)kv_read_size / chunk_bytes)
          : 1;
      if (num_chunks < 1) num_chunks = 1;
      kv_read_time = (kv_read_size / hbf.flash_read_bandwidth * 1e9) + num_chunks * hbf.flash_page_read_latency_ns;
    } else {
      // No chunked attention: if read size exceeds SRAM buffer size, double-buffering fails
      unsigned long long sram_capacity = hbf.sram_per_stack_bytes * hbf.num_flash_stacks;
      if (kv_read_size > sram_capacity) {
        // Pay page read latency for every page (ceil to whole pages, consistent with chunked branch).
        double num_pages = std::ceil((double)kv_read_size / hbf.page_size_bytes);
        kv_read_time = (kv_read_size / hbf.flash_read_bandwidth * 1e9) + num_pages * hbf.flash_page_read_latency_ns;
      } else {
        // Fits in SRAM: pay page read latency only once
        kv_read_time = (kv_read_size / hbf.flash_read_bandwidth * 1e9) + hbf.flash_page_read_latency_ns;
      }
    }

    double act_time = 0;
    if (hbf.num_hbm_stacks > 0) {
      act_time = act_size / hbf.hbm_read_bandwidth * 1e9;
    }
    return std::max(kv_read_time, act_time);
  }
  return 0;
}

inline time_ns getKVWriteDuration(const SystemConfig& config, int num_seq, int num_kv_heads, int head_dim, int precision, bool compressed_kv, int kv_lora_rank, int qk_rope_head_dim, int input_len, int output_len) {
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    auto& hbf = config.hbf_config;
    double num_new_queries = (double)num_seq / (output_len > 0 ? output_len : 1);
    double kv_write_size = 0;
    if (compressed_kv) {
      kv_write_size = num_new_queries * (kv_lora_rank + qk_rope_head_dim) * input_len * precision;
    } else {
      kv_write_size = 2.0 * num_new_queries * num_kv_heads * head_dim * input_len * precision;
    }
    double write_time = (kv_write_size / hbf.flash_write_bandwidth * 1e9) + hbf.flash_page_program_latency_ns;
    return (time_ns)write_time;
  }
  return 0;
}

ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, Tensor_Ptr tensor);

ExecStatus getIdealMemoryStatus(Device_Ptr device, ProcessorType processor_type,
                          DRAMRequestType dram_request_type, Tensor_Ptr tensor);

ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator);

ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr ouptut,
                                bool use_ramulator);

ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator);

ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator, bool duplicated_input);

ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr ouptut,
                                bool use_ramulator, bool duplicated_input);

ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator, bool duplicated_input);

ExecStatus ActivationExecutionGPU(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator);

ExecStatus ActivationExecutionLogic(Device_Ptr device, Tensor_Ptr gate_output,
                                    Tensor_Ptr input, Tensor_Ptr output,
                                    bool use_ramulator);

ExecStatus ActivationExecutionPIM(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator);

ExecStatus AttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionSumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionSumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus AttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor_list,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator);

ExecStatus AttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionSumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionSumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus MultiLatentAttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor_list,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator);

ExecStatus MultiLatentAttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus AbsorbMLAGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AbsorbMLAGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus AbsorbMLAGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AbsorbMLASumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

ExecStatus AbsorbMLASumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

ExecStatus AbsorbMLASumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

}  // namespace llm_system