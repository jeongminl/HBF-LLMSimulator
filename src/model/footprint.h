#pragma once
// footprint.h — shared capacity-check helpers used by BOTH the parallelism
// optimizer (pre-graph, analytic) and checkMemorySize() in cluster.cpp.
//
// Dependency policy: include ONLY hardware/hardware_config.h (which already
// pulls in dram/hbf_memory_config.h).  Do NOT include cluster.h or any module
// header — that would create an include cycle.
#include <algorithm>
#include <cmath>
#include <string>

#include "hardware/hardware_config.h"
#include "model/model_config.h"

namespace llm_system {

// ---------------------------------------------------------------------------
// Capacity rule
// ---------------------------------------------------------------------------
// HBF with flash stacks present:
//   weight + kv  vs  hbf.total_capacity_bytes               (flash pool)
//   activation   vs  num_hbm_stacks * hbm_per_stack_bytes   (HBM tier, if any)
//                    logic_sram_bytes                         (no-HBM fallback)
// Plain HBM (use_hbf==false OR num_flash_stacks==0):
//   act + weight + kv  vs  memory_capacity                   (lumped)
// ---------------------------------------------------------------------------

struct CapacityResult {
  bool oom;
  std::string reason;
};

// Returns the capacity limit (bytes) for activations on the scarce tier.
// For HBF with HBM stacks  → total HBM capacity across all stacks.
// For HBF+ / CONV+ (no HBM)→ logic SRAM.
// For plain HBM             → full memory_capacity (acts not separately bounded).
inline double scarceTierActivationLimit(const SystemConfig& s) {
  if (s.use_hbf && s.hbf_config.num_flash_stacks > 0) {
    if (s.hbf_config.num_hbm_stacks > 0) {
      return static_cast<double>(s.hbf_config.num_hbm_stacks) *
             static_cast<double>(s.hbf_config.hbm_per_stack_bytes);
    } else {
      return static_cast<double>(s.hbf_config.logic_sram_bytes);
    }
  }
  return s.memory_capacity;
}

// Returns true when activations have a dedicated scarce-tier limit separate
// from the main (flash) pool.  Plain-HBM runs have only one pool.
inline bool hasScarceTier(const SystemConfig& s) {
  return s.use_hbf && s.hbf_config.num_flash_stacks > 0;
}

// Unified capacity check.
//   act    – activation bytes (per GPU, single-layer peak)
//   weight – weight bytes (per GPU, full PP stage)
//   kv     – KV-cache bytes (per GPU, full PP stage)
inline CapacityResult checkCapacity(const SystemConfig& s,
                                     double act,
                                     double weight,
                                     double kv) {
  if (s.use_hbf && s.hbf_config.num_flash_stacks > 0) {
    const auto& hbf = s.hbf_config;

    // Weights + KV live on flash
    double flash_cap = static_cast<double>(hbf.total_capacity_bytes);
    if (weight + kv > flash_cap) {
      return {true,
              "Flash capacity exceeded (" +
              std::to_string((weight + kv) / 1e9) + " GB > " +
              std::to_string(flash_cap / 1e9) + " GB)"};
    }

    // Activations live on the scarce tier (HBM stacks or logic SRAM)
    double scarce_cap = scarceTierActivationLimit(s);
    if (act > scarce_cap) {
      std::string tier = (hbf.num_hbm_stacks > 0) ? "HBM" : "Logic SRAM";
      return {true,
              tier + " activation capacity exceeded (" +
              std::to_string(act / 1e6) + " MB > " +
              std::to_string(scarce_cap / 1e6) + " MB)"};
    }

    return {false, ""};
  } else {
    // Plain HBM: lumped check against memory_capacity
    double total = act + weight + kv;
    if (total > s.memory_capacity) {
      return {true,
              "HBM capacity exceeded (" +
              std::to_string(total / 1e9) + " GB > " +
              std::to_string(s.memory_capacity / 1e9) + " GB)"};
    }
    return {false, ""};
  }
}

// ---------------------------------------------------------------------------
// Peak intermediate-data (activation) footprint for the scarce-tier gate.
// ---------------------------------------------------------------------------
// The scarce tier (36-GB reserved HBM stack for HBF, 320-MB logic-die SRAM for
// HBF+/CONV+) holds "intermediate data" -- the paper (Son et al., IEEE CAL
// 2026) is explicit that this must be sized by its PEAK concurrently-live
// footprint, not the sum of every tensor a layer touches:
//   "much of the intermediate data has a short lifetime, which enables quick
//    release of SRAM-buffer space" ... "a larger batch size ... linearly
//    increases the peak size of intermediate data".
// Tensors are produced and consumed sequentially within a layer (attention
// phase, then FFN phase), so at any instant the live set is the RESIDUAL
// stream plus whichever phase (attention or FFN) is currently executing --
// never both, and never every op's output at once. This function returns
// max(attention-phase total, FFN-phase total) instead of their sum.
//
// It also excludes the sequence-length-scaled Q.K^T attention-score matrix
// (and, for compressed-KV, the decompressed KV) from the resident set: per
// INSTRUCTIONS.md's Read Prefetching section, that data is chunked through a
// SEPARATE per-stack 3.13-MB double-buffer staging pool and is never resident
// in the intermediate-data pool this function's result is gated against.
//
// Mirrors (and replaces, for the scarce-tier gate only) the three decode
// branches previously duplicated in cluster.cpp:108-176 and
// parallelism_optimizer.cpp:243-326. Both call sites must use this single
// definition so the optimizer and simulator gates never drift apart.
//
//   model               - selects MLA-absorb / MLA-compressed-kv / GQA-base,
//                          and dense vs. MoE FFN shape.
//   batch_per_dp        - sequences this GPU's DP replica handles this step
//                          (cluster.cpp's batch_size_per_dp / optimizer's
//                          batch_size_per_gpu).
//   tp                  - non-expert tensor-parallel degree (ne_tp_dg / tp).
//   expert_batch_size, num_routed_expert_per_device
//                       - as already computed identically by both callers
//                          (cluster.cpp:88-95 / parallelism_optimizer.cpp:
//                          247-253); passed through unchanged.
//   has_moe_layer       - true if any of the model's layers use a routed-MoE
//                          FFN (model.num_routed_expert > 0).
//   has_dense_layer     - true if any of the model's layers use a dense FFN
//                          (always true for non-MoE models; also true for
//                          MoE models with first_k_dense>0 or expert_freq>1,
//                          which mix dense and MoE layers).
// Layers of different types never execute concurrently on one GPU, so the
// returned FFN-phase total is the max over whichever layer type is worse.
inline double peakIntermediateBytes(const ModelConfig& model,
                                     double batch_per_dp,
                                     int tp,
                                     double expert_batch_size,
                                     double num_routed_expert_per_device,
                                     bool has_moe_layer,
                                     bool has_dense_layer) {
  double precision = model.precision_byte;
  double hidden_dim = model.hidden_dim;

  double attn_total;
  if (model.use_absorb) {
    // Query's non-rope output width is qk_nope_head_dim, not head_dim (which is V's
    // up-projected width, used correctly below in "v_up out") -- dormant on this
    // model's own preset since deepseekV3 sets qk_nope_head_dim == head_dim == 128,
    // but was silently wrong for any config where they diverge (BUGS_HIDDEN_BY_FLAGS #7).
    double common_prefix =
        (batch_per_dp * hidden_dim) +
        (batch_per_dp * model.q_lora_rank) +
        (batch_per_dp * model.kv_lora_rank) +
        (batch_per_dp * model.qk_rope_head_dim) +
        (batch_per_dp * (3.0 * model.qk_rope_head_dim + model.qk_nope_head_dim) * model.num_heads / tp);
    attn_total = (common_prefix +
                  (batch_per_dp * model.num_heads * model.kv_lora_rank / tp) +  // tr_k up out
                  (batch_per_dp * model.num_heads * model.kv_lora_rank / tp) +  // attn context out
                  (batch_per_dp * model.num_heads * model.head_dim / tp) +      // v_up out
                  (batch_per_dp * hidden_dim)) * precision;                     // out proj out
  } else if (model.compressed_kv) {
    // Same qk_nope_head_dim fix as the absorb branch above.
    double common_prefix =
        (batch_per_dp * hidden_dim) +
        (batch_per_dp * model.q_lora_rank) +
        (batch_per_dp * model.kv_lora_rank) +
        (batch_per_dp * model.qk_rope_head_dim) +
        (batch_per_dp * (3.0 * model.qk_rope_head_dim + model.qk_nope_head_dim) * model.num_heads / tp);
    attn_total = (common_prefix +
                  (batch_per_dp * model.num_heads * model.head_dim / tp) +  // attn context out
                  (batch_per_dp * hidden_dim)) * precision;                 // out proj out
  } else {
    // GQA base (llama3_405B / llama4_maverick / llama4_scout).
    attn_total = ((batch_per_dp * hidden_dim) +                              // input/residual
                  (batch_per_dp * model.q_lora_rank) +                       // 0 for non-MLA
                  (batch_per_dp * model.kv_lora_rank) +                      // 0 for non-MLA
                  (batch_per_dp * model.qk_rope_head_dim) +                  // 0 for non-MLA
                  (batch_per_dp * (3.0 * model.qk_rope_head_dim + model.head_dim) *
                   model.num_heads / tp) +                                   // q proj(+rope) out
                  (batch_per_dp * 2.0 * model.head_dim * model.num_heads / tp) +  // current-token k,v out
                  (batch_per_dp * model.num_heads * model.head_dim / tp) +   // attn context out
                  (batch_per_dp * hidden_dim)) * precision;                  // out proj out
  }

  double ffn_moe = 0.0;
  if (has_moe_layer && model.num_routed_expert > 0) {
    ffn_moe = (num_routed_expert_per_device + model.num_shared_expert) *
        (2.0 * (expert_batch_size * model.expert_intermediate_dim) +  // gate proj + silu out
         expert_batch_size * model.expert_intermediate_dim +          // up proj out
         expert_batch_size * hidden_dim) * precision;                 // down proj out
  }
  double ffn_dense = 0.0;
  if (has_dense_layer) {
    ffn_dense = (batch_per_dp * 2.0 * model.intermediate_dim / tp +  // gate+up concurrently live (TP-sharded)
                 batch_per_dp * hidden_dim) * precision;             // down proj out
  }
  double ffn_total = std::max(ffn_moe, ffn_dense);

  return std::max(attn_total, ffn_total);
}

// True if this model has at least one dense-FFN layer (always true for
// non-MoE models; also true for MoE models that mix in dense layers via
// first_k_dense or expert_freq>1 -- mirrors parallelism_optimizer.cpp:156-174).
inline bool hasDenseFfnLayer(const ModelConfig& model) {
  if (model.num_routed_expert <= 0) return true;
  if (model.first_k_dense > 0) return true;
  if (model.expert_freq > 1) return true;
  return false;
}

}  // namespace llm_system
