#pragma once
#include <algorithm>
#include <map>
#include <string>

#include "common/assert.h"

namespace llm_system {

class ModelConfig {
  // default: mixtral 8x7B
 public:
  ModelConfig(int hidden_dim = 4096, int head_dim = 128, int num_layers = 32,
              int num_heads = 32, int num_kv_heads = 8, int max_seq_len = 32768,
              int intermediate_dim = 14336, int expert_intermediate_dim = 14336,
              int activation_factor = 1, int precision_byte = 2,
              int num_routed_expert = 8, int num_shared_expert = 0,
              int expert_freq = 1, int top_k = 2, int ffn_way = 3,
              int first_k_dense = 0, int q_lora_rank = 0, int kv_lora_rank = 0,
              int qk_nope_head_dim = 0, int qk_rope_head_dim = 0, int n_vocab = 32000, bool compressed_kv = false,
              bool use_absorb = false,
              double skewness = 0.0,
              std::string model_name = "",
              int attn_chunk_size = 0, int attn_global_interval = 1)
      : hidden_dim(hidden_dim),
        head_dim(head_dim),
        num_layers(num_layers),
        num_heads(num_heads),
        num_kv_heads(num_kv_heads),
        max_seq_len(max_seq_len),
        intermediate_dim(intermediate_dim),
        expert_intermediate_dim(expert_intermediate_dim),
        activation_factor(activation_factor),
        precision_byte(precision_byte),
        num_routed_expert(num_routed_expert),
        num_shared_expert(num_shared_expert),
        expert_freq(expert_freq),
        top_k(top_k),
        ffn_way(ffn_way),
        first_k_dense(first_k_dense),
        q_lora_rank(q_lora_rank),
        kv_lora_rank(kv_lora_rank),
        qk_nope_head_dim(qk_nope_head_dim),
        qk_rope_head_dim(qk_rope_head_dim),
        n_vocab(n_vocab),
        compressed_kv(compressed_kv),
        use_absorb(use_absorb),
        skewness(skewness),
        model_name(model_name),
        attn_chunk_size(attn_chunk_size),
        attn_global_interval(attn_global_interval) {
    if(q_lora_rank == 0){
    assertTrue(hidden_dim == head_dim * num_heads,
               "hidden_dim != head_dim * num_heads");
    }
  };

  ModelConfig& operator=(const ModelConfig& rhs) = default;

  int hidden_dim;
  int head_dim;
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int max_seq_len;
  int intermediate_dim;
  int expert_intermediate_dim;
  int activation_factor;
  int precision_byte;
  int num_routed_expert;
  int num_shared_expert;
  int expert_freq;
  int top_k;
  int ffn_way;
  int first_k_dense; // 0 for not use
  int q_lora_rank;     // for MLA
  int kv_lora_rank;    // for MLA
  int qk_nope_head_dim; // for MLA
  int qk_rope_head_dim; // for MLA  
  int n_vocab;
  
  bool compressed_kv;
  bool use_absorb;
  double skewness; // for Zipfian distribution
  std::string model_name;

  // Llama-4-style interleaved local/global ("iRoPE") attention. attn_chunk_size==0 means
  // "every layer does full global attention over the whole context" (today's behavior for
  // every model preset except llama4_maverick/llama4_scout) -- see isGlobalAttentionLayer()/
  // effectiveKvLen() below. attn_global_interval==1 also means "every layer is global"
  // (redundant with attn_chunk_size==0, kept as a second backward-compat guard).
  int attn_chunk_size;      // local-attention window in tokens; 0 = no windowing (full global)
  int attn_global_interval; // every Nth layer (1-indexed) is a full/global ("NoPE") layer

  int ne_tp_dg;  // non-expert tensor parallelism degree
  int e_tp_dg;   // expert tensor parallelism degree
  int pp_dg = 1; // pipeline parallelism degree
  std::string dataset;

  int input_len;
  int output_len;
};

// ---------------------------------------------------------------------------
// Interleaved local/global ("iRoPE") attention helpers -- shared by every call
// site that computes KV-cache size or KV-read cost, so the peak-batch capacity
// gate (parallelism_optimizer.cpp / cluster.cpp) and the live decode-phase KV
// read (attention_gen_impl.cpp) never drift apart. See CHANGES.md item 16.
//
// `layer` is 0-indexed. With the default attn_chunk_size==0 (every model
// preset except llama4_maverick/llama4_scout), every layer is global and
// these all reduce EXACTLY to the "every layer sees the full context"
// formulas.
// ---------------------------------------------------------------------------

// True if `layer` is a full/global ("NoPE") attention layer per Llama-4's
// no_rope_layer_interval convention: 0-indexed layer L is global when
// (L+1) % attn_global_interval == 0 (matches HF transformers'
// configuration_llama4.py default_no_rope_layers computation exactly).
inline bool isGlobalAttentionLayer(const ModelConfig& mc, int layer) {
  if (mc.attn_chunk_size == 0 || mc.attn_global_interval <= 1) return true;
  return (layer + 1) % mc.attn_global_interval == 0;
}

// Effective KV length (tokens) a given layer's attention must read/retain at
// the given context length. Global layers see the full context; local layers
// are capped at attn_chunk_size. Deliberate simplification: Llama-4's real
// local attention is chunked into fixed non-overlapping blocks (a token's
// true local KV length sawtooths between 1 and attn_chunk_size depending on
// position within its block), but this simulator models representative
// per-decode-step costs rather than exact per-token trajectories, and the
// flat cap empirically matches the paper's reported anchor much more closely
// than the sawtooth's ~half-chunk average would (see CHANGES.md).
inline double effectiveKvLen(const ModelConfig& mc, int layer, double context_len) {
  if (isGlobalAttentionLayer(mc, layer)) return context_len;
  return std::min(context_len, (double)mc.attn_chunk_size);
}

// Sum of effectiveKvLen() across ALL of the model's layers -- the quantity
// that replaces "num_layers * context_len" in every KV-cache-size formula.
// Reduces exactly to num_layers * context_len when attn_chunk_size==0.
inline double effectiveKvLenSumAllLayers(const ModelConfig& mc, double context_len) {
  double total = 0.0;
  for (int layer = 0; layer < mc.num_layers; ++layer) {
    total += effectiveKvLen(mc, layer, context_len);
  }
  return total;
}

// Returns true if the given (0-indexed) layer is a MoE (routed-expert) layer.
// Honors both first_k_dense (the first N layers are always dense regardless of
// expert_freq) and expert_freq (every expert_freq-th layer is MoE). Works for
// ALL models, not just deepseekV3: deepseekV3 uses first_k_dense only,
// llama4/mixtral use expert_freq only, dense models have num_routed_expert==0.
// Single shared definition used by both the module-construction path
// (llm.cpp) and the parallelism optimizer (parallelism_optimizer.cpp), so the
// two never disagree on which layers are MoE.
inline bool isMoELayer(const ModelConfig& mc, int layer);

// Number of consecutive weight-read linear ops a (stage-0) device issues per
// decode iteration -- the length of its per-iteration flash weight stream.
// Used to amortize the flash page-read PIPELINE-FILL latency in
// getLinearMemoryDuration (layer_impl.h): weights carry no activation
// dependency, so the staging-SRAM prefetcher double-buffers ACROSS consecutive
// weight tensors just as it does across chunks within one tensor, exposing
// ~one fill latency per stream rather than one per op. Mirrors
// getKVWriteDuration's program_latency_amortize_calls on the write side.
// Counts follow the live module graph (layer.cpp / decoder.cpp / expert.cpp):
//   GQA attention: attn_qkv_proj + attn_o_proj = 2 weight tensors.
//   MLA: q_down/kv_down/kr/q_up/(absorb extras)/o_proj -> use_absorb ? 8 : 7
//        (same convention as the optimizer's former per-op page counts).
//   FFN: dense = ffn_way; MoE = router gate + resident_experts*ffn_way +
//        num_shared_expert*ffn_way. Resident (static) expert count is used;
//        the batch-dependent ACTIVE count can be lower, which under-exposes at
//        most ~one extra page latency total (microseconds -- negligible).
//   lm_head: +1 when pp == 1 (stage 0 is then also the last stage).
inline int weightReadOpsPerIteration(const ModelConfig& mc, int total_num_devices) {
  int pp = (mc.pp_dg > 0) ? mc.pp_dg : 1;
  int layers_per_stage = mc.num_layers / pp;
  if (layers_per_stage < 1) layers_per_stage = 1;
  int devices_per_stage = (total_num_devices > 0) ? total_num_devices / pp : 1;
  if (devices_per_stage < 1) devices_per_stage = 1;
  int e_tp = (mc.e_tp_dg > 0) ? mc.e_tp_dg : 1;

  int experts_per_device = 0;
  if (mc.num_routed_expert > 0) {
    int expert_groups = devices_per_stage / e_tp;
    if (expert_groups < 1) expert_groups = 1;
    experts_per_device = mc.num_routed_expert / expert_groups;
    if (experts_per_device < 1) experts_per_device = 1;
  }

  int attn_ops = (mc.q_lora_rank != 0) ? (mc.use_absorb ? 8 : 7) : 2;

  int ops = 0;
  for (int layer = 0; layer < layers_per_stage; ++layer) {
    ops += attn_ops;
    if (isMoELayer(mc, layer)) {
      ops += 1 + mc.ffn_way * experts_per_device +
             mc.ffn_way * mc.num_shared_expert;
    } else {
      ops += mc.ffn_way;
    }
  }
  if (pp == 1) ops += 1;  // lm_head
  return (ops > 0) ? ops : 1;
}

inline bool isMoELayer(const ModelConfig& mc, int layer) {
  if (mc.num_routed_expert == 0) return false;              // dense model
  if (mc.first_k_dense > 0 && layer < mc.first_k_dense) return false; // forced-dense prefix
  if (mc.expert_freq == 0) return true;                     // all remaining layers are MoE
  return (layer % mc.expert_freq == 0);
}

static ModelConfig mixtral = ModelConfig(4096, 128, 32, 32, 8, 32768, 14336,
                                         14336, 1, 2, 8, 0, 1, 2, 3, 0, 0, 0, 0, 0, 32000, false, false, 0.0, "mixtral");

static ModelConfig openMoE = ModelConfig(
    3072, 128, 32, 24, 24, 2048, 12288, 12288, 2, 2, 32, 0, 4, 2, 3, 0, 0, 0, 0, 0, 32000, false, false, 0.0, "openMoE");

static ModelConfig llama7bMoE =
    ModelConfig(4096, 128, 32, 32, 32, 4096, 11008, 688, 1, 2, 16, 0, 1, 2, 3, 0, 0, 0, 0, 0, 32000, false, false, 0.0,
                "llama7bMoE");

static ModelConfig grok1 = ModelConfig(6144, 128, 64, 48, 8, 8192, 32768, 32768,
                                       1, 2, 8, 0, 1, 2, 3, 0, 0, 0, 0, 0, 131072, false, false, 0.0, "grok1");

static ModelConfig glam = ModelConfig(4096, 128, 32, 32, 32, 8192, 16384, 16384,
                                      1, 2, 64, 0, 2, 2, 2, 0, 0, 0, 0, 0, 256000, false, false, 0.0, "glam");

static ModelConfig deepseekV3 =
    ModelConfig(7168, 128, 60, 128, 128, 131072, 18432, 2048, 1, 1, 256, 1, 1, 8,
                3, 3, 1536, 512, 128, 64, 129280, true, true, 0.0,"deepseekV3"); // n_layer = 60 (not consider MTP module)

// precision_byte=2 (BF16): matches the paper's explicit "no 1-/2-GPU segments in all HBM4
// bars" claim, which requires llama3_405B's real footprint to exceed 2x288GB=576GB -- only
// consistent with native BF16 (~810GB), not FP8 (~405GB, which fits under 576GB and would
// wrongly show 2-GPU HBM4 as feasible). See CHANGES.md item 11.
static ModelConfig llama3_405B =
    ModelConfig(16384, 128, 126, 128, 8, 131072, 53248, 53248, 1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 128256, false, false, 0.0,
                "llama3_405B");

// attn_chunk_size=8192, attn_global_interval=4: Llama 4's "iRoPE" interleaved local/global
// attention (Meta's own architecture description; independently confirmed against Maverick's
// released config.json: attention_chunk_size=8192 explicit, no_rope_layer_interval defaults to
// 4 and isn't overridden). Only every 4th layer ("NoPE") attends over the full context; the
// other 3 use a fixed 8192-token local window. See footprint.h/model_config.h's
// isGlobalAttentionLayer()/effectiveKvLen() and CHANGES.md item 16.
// Scout's exact config reportedly differs from Maverick's in some fields
// (unverified which); applying the same iRoPE constants here for consistency, but note Scout
// isn't one of the paper's own evaluated models so this is lower-confidence/lower-stakes.
static ModelConfig llama4_scout = // 16 Expert
    ModelConfig(5120, 128, 48, 40, 8, 10485760, 16384, 8192, 1, 2, 16, 1, 1, 1,
                3, 0, 0, 0, 0, 0, 202048, false, false, 0.0, "llama4_scout", 8192, 4);

static ModelConfig llama4_maverick = // 128 Expert
                ModelConfig(5120, 128, 48, 40, 8, 1048576, 16384, 8192, 1, 2, 128, 1, 2, 1,
                            3, 0, 0, 0, 0, 0, 202048, false, false, 0.0, "llama4_maverick", 8192, 4);

// if model_config.q_lora_rank != 0 -> MLA로

}  // namespace llm_system