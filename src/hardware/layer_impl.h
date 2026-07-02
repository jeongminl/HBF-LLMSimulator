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
    // Weight reads stage through the same per-stack SRAM double-buffer as KV reads
    // (the paper's methodology describes prefetch/double-buffering generically for
    // "read data", not scoped to KV specifically -- see getAttentionMemoryDuration's
    // non-chunked branch below, which this mirrors exactly). Double-buffering means
    // chunk N+1's page-read latency overlaps chunk N's transfer and is hidden for
    // every chunk except the first (pipeline fill), which always exposes one full
    // page latency; if a chunk's transfer time is shorter than the page latency (e.g.
    // an unusually small SRAM capacity or high page latency), the residual is exposed
    // per chunk too. Under every current preset's constants, chunk transfer time at
    // full SRAM capacity exceeds the page latency, so this reduces to exactly one
    // exposed page latency regardless of weight size or chunk count -- numerically
    // identical to the old flat one-latency-per-op charge, but now structurally
    // consistent with (and correct if config constants ever change relative to) the
    // KV-read model instead of being a special-cased exemption from it.
    double weight_sram_capacity = (double)hbf.sram_per_stack_bytes * hbf.num_flash_stacks;
    int weight_num_chunks = (weight_sram_capacity > 0)
        ? (int)std::ceil((double)weight_size / weight_sram_capacity) : 1;
    if (weight_num_chunks < 1) weight_num_chunks = 1;
    double weight_chunk_transfer_ns = weight_sram_capacity / hbf.flash_read_bandwidth * 1e9;
    double weight_exposed_latency_ns = (double)hbf.flash_page_read_latency_ns +
        (weight_num_chunks - 1) * std::max(0.0, (double)hbf.flash_page_read_latency_ns - weight_chunk_transfer_ns);
    double weight_read_time = (weight_size / hbf.flash_read_bandwidth * 1e9) + weight_exposed_latency_ns;

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
      // Chunked attention, double-buffered per the paper ("prefetch and double-buffer
      // read data, thereby minimizing the performance impact of us-scale read
      // latency"): while chunk N streams out of one SRAM buffer, chunk N+1's page
      // read is issued into the other buffer, so its latency overlaps chunk N's
      // transfer and is hidden -- for every chunk EXCEPT the first, which has
      // nothing to overlap with (pipeline fill) and always exposes one full page
      // latency. If a chunk's own transfer time is shorter than the page latency
      // (e.g. an explicit small chunk_size), double-buffering can't fully hide it;
      // the residual (latency - transfer) per chunk is still exposed. Chunk
      // granularity:
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
      double chunk_transfer_ns = chunk_bytes / hbf.flash_read_bandwidth * 1e9;
      double exposed_latency_ns = (double)hbf.flash_page_read_latency_ns +
          (num_chunks - 1) * std::max(0.0, (double)hbf.flash_page_read_latency_ns - chunk_transfer_ns);
      kv_read_time = (kv_read_size / hbf.flash_read_bandwidth * 1e9) + exposed_latency_ns;
    } else {
      // No chunked attention: still double-buffer by SRAM-sized chunks (plane-level
      // parallelism means flash_page_read_latency is only fully exposed once, for
      // the pipeline-fill chunk -- see the chunked branch above for the reasoning).
      double sram_capacity = (double)hbf.sram_per_stack_bytes * hbf.num_flash_stacks;
      int num_chunks = (sram_capacity > 0)
          ? (int)std::ceil((double)kv_read_size / sram_capacity)
          : 1;
      if (num_chunks < 1) num_chunks = 1;
      double chunk_transfer_ns = sram_capacity / hbf.flash_read_bandwidth * 1e9;
      double exposed_latency_ns = (double)hbf.flash_page_read_latency_ns +
          (num_chunks - 1) * std::max(0.0, (double)hbf.flash_page_read_latency_ns - chunk_transfer_ns);
      kv_read_time = (kv_read_size / hbf.flash_read_bandwidth * 1e9) + exposed_latency_ns;
    }

    double act_time = 0;
    if (hbf.num_hbm_stacks > 0) {
      act_time = act_size / hbf.hbm_read_bandwidth * 1e9;
    }
    return std::max(kv_read_time, act_time);
  }
  return 0;
}

inline time_ns getKVWriteDuration(const SystemConfig& config, int num_seq, int num_kv_heads, int head_dim, int precision, bool compressed_kv, int kv_lora_rank, int qk_rope_head_dim, int input_len, int output_len, int local_attention_window = 0) {
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    auto& hbf = config.hbf_config;
    double num_new_queries = (double)num_seq / (output_len > 0 ? output_len : 1);
    // Llama-4 iRoPE: a local layer only ever retains/writes a bounded KV window
    // (attn_chunk_size), never the full input_len -- matches the cap already
    // applied to the KV-READ path (see this file's getAttentionMemoryDuration
    // callers, e.g. attention_gen_impl.cpp's local_attention_window usage) and
    // the capacity path's effectiveKvLen() (model/model_config.h). Without this,
    // the write path contradicted the capacity path (claiming to write more KV
    // than the cache is ever sized to hold). window==0 (every non-iRoPE model,
    // and any call site that hasn't threaded the per-layer window through) is a
    // no-op: effective_input_len == input_len, byte-identical to before.
    int effective_input_len = (local_attention_window > 0)
        ? std::min(input_len, local_attention_window) : input_len;
    double kv_write_size = 0;
    if (compressed_kv) {
      kv_write_size = num_new_queries * (kv_lora_rank + qk_rope_head_dim) * effective_input_len * precision;
    } else {
      kv_write_size = 2.0 * num_new_queries * num_kv_heads * head_dim * effective_input_len * precision;
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