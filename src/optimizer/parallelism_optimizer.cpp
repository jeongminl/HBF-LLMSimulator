#include "optimizer/parallelism_optimizer.h"
#include "model/footprint.h"
#include <cmath>
#include <algorithm>

namespace llm_system {

std::vector<ParallelConfig> ParallelismOptimizer::EnumerateCandidates(
    const ModelConfig& model_config,
    const SystemConfig& system_config,
    int total_gpus,
    int batch_size,
    int sequence_length,
    bool require_batch_divisible) {
  std::vector<ParallelConfig> candidates;

  // Search space
  for (int tp = 1; tp <= total_gpus; tp *= 2) {
    // Require even divisibility (each TP rank gets num_kv_heads/tp KV heads). tp >
    // num_kv_heads is NOT supported by the simulator today -- src/module/parallel.cpp
    // hard-asserts num_kv_heads % parallel_num == 0 (crashes rather than replicating KV
    // heads across TP ranks), so a config with tp > num_kv_heads that this optimizer
    // proposed would make the live simulator abort. Unconditionally excluding it here
    // caps TP at num_kv_heads; DP/PP make up the remainder of the GPU count.
    if (model_config.num_kv_heads % tp != 0) {
      continue;
    }
    for (int pp = 1; pp <= total_gpus; pp *= 2) {
      if (tp * pp > total_gpus) continue;
      if (total_gpus % (tp * pp) != 0) continue;  // prevent stranded GPUs (non-divisible configs)
      if (model_config.num_layers % pp != 0) continue;

      // PP_FLAGS_SPEC §3.1: inter-node-only PP. A pipeline replica occupies tp*pp
      // contiguous ranks (rank = d*(tp*pp) + s*tp + t, LLM::LLM/llm.cpp). With
      // node size N = num_device (GPUs/node) and power-of-2 tp,pp,N the replicas
      // tile node boundaries exactly, so:
      //   tp*pp <= N  <=>  every replica fits inside ONE node
      //                    => all pp stages co-located => every PP hop intra-node
      //   tp*pp >  N  <=>  a replica spans >= 2 nodes => at least one PP hop is
      //                    inter-node (the benefit PP is meant to buy).
      // Intra-node pp>1 never wins a cell under the pipelined model (Hunt B) and
      // is disallowed here so the analytic listing, the pruning seed, AND
      // Optimize() all agree (this function feeds all three). pp==1 always allowed.
      if (system_config.pp_internode_only && pp > 1 &&
          (long long)tp * pp <= system_config.num_device) {
        continue;
      }

      int dp = total_gpus / (tp * pp);
      // Skip configs where batch cannot be evenly divided across DP groups.
      // This is physically correct: uneven DP creates load imbalance.  The
      // always-available dp=1 config ensures feasibility is never lost.
      // Granularity (finding the best batch near a non-divisible point) is
      // handled by the external sweep (run_flash_only.py) rather than relaxing
      // this constraint (ceil(batch/dp) would diverge from the live-sim's
      // integer floor division and reintroduce a gate mismatch).
      // Callers doing a PER-CONFIG batch search (analytic_configs_only mode)
      // pass require_batch_divisible=false and instead probe each config only
      // at multiples of its own dp.
      if (require_batch_divisible && batch_size % dp != 0) continue;

      // ---- Expert (tensor) parallelism sweep ----------------------------------
      // EP (= e_tp_dg) is an independent degree of freedom from non-expert TP:
      // routed-expert weights may be tensor-split across a different number of
      // GPUs than the attention/dense weights.  The expert module spans ONE PP
      // stage (decoder.cpp/llm.cpp: stage_device_list has total_gpus/pp devices),
      // so e_tp_dg is bounded by that, not by tp.  We mirror the validity
      // assertions in expert.cpp:49-53 so the optimizer never proposes a config
      // the simulation would reject.  For dense models max_e_tp==1, so this
      // collapses to a single e_tp_dg==1 iteration (no behavior change).
      int devices_per_stage = total_gpus / pp;  // = parallel_num seen by ExpertFFN
      int max_e_tp = (model_config.num_routed_expert > 0)
          ? std::min(devices_per_stage, model_config.num_routed_expert)
          : 1;
      for (int e_tp_dg = 1; e_tp_dg <= max_e_tp; e_tp_dg *= 2) {
        if (model_config.num_routed_expert > 0) {
          if (model_config.num_routed_expert % e_tp_dg != 0) continue;            // expert.cpp:49
          if (devices_per_stage % e_tp_dg != 0) continue;                         // expert.cpp:51
          if ((devices_per_stage / e_tp_dg) > model_config.num_routed_expert)     // expert.cpp:53
            continue;
        }

      candidates.push_back(EvaluateConfig(model_config, system_config, total_gpus,
                                          tp, pp, dp, e_tp_dg, batch_size,
                                          sequence_length));
      }  // expert-parallelism (e_tp_dg) sweep
    }
  }

  return candidates;
}

ParallelConfig ParallelismOptimizer::EvaluateConfig(const ModelConfig& model_config,
                                                    const SystemConfig& system_config,
                                                    int total_gpus,
                                                    int tp,
                                                    int pp,
                                                    int dp,
                                                    int e_tp_dg,
                                                    int batch_size,
                                                    int sequence_length) {
      int devices_per_stage = total_gpus / pp;  // = parallel_num seen by ExpertFFN

      ParallelConfig config;
      config.tp = tp;
      config.pp = pp;
      config.dp = dp;
      config.ep = e_tp_dg;

      // Memory footprint calculation
      double layers_per_stage = (double)model_config.num_layers / pp;
      double batch_size_per_gpu = (double)batch_size / dp;
      // PP_FIX_SPEC.md §4: the analytic latency mirror of the pp>1 runtime fix
      // (cluster.cpp's reportedIterationTime()) -- the true per-stage decode
      // latency is a function of the MICROBATCH (B/pp), not the full
      // per-replica batch (capacity/KV/PEC accounting stays at full
      // batch_size_per_gpu; ONLY the latency terms below use this). Reduces to
      // batch_size_per_gpu exactly when pp==1 (regression-safe).
      // PP_FLAGS_SPEC §2.2: microbatch latency basis only when pipelined timing ships.
      double batch_for_latency = system_config.pp_pipelined_timing
                                   ? batch_size_per_gpu / pp
                                   : batch_size_per_gpu;
      double precision = model_config.precision_byte;
      double hidden_dim = model_config.hidden_dim;

      // ---- Attention weights per layer (TP-aware, per GPU) -------------------
      // MLA uses separate projection matrices (q_lora_rank != 0).
      // e_tp_dg is the expert-parallelism sweep variable; it may differ from ne_tp.

      double attn_weights_per_gpu;  // bytes, single layer, per GPU
      if (model_config.q_lora_rank != 0) {
        // MLA: q_down, q_up, kv_down, kv_up, o_proj
        // head-dimension terms are TP-split; lora-rank terms are replicated
        double q_down  = (double)hidden_dim * model_config.q_lora_rank;
        double q_up    = (double)model_config.q_lora_rank *
                         (model_config.num_heads / (double)tp) *
                         (model_config.qk_nope_head_dim + model_config.qk_rope_head_dim);
        double kv_down = (double)hidden_dim *
                         (model_config.kv_lora_rank + model_config.qk_rope_head_dim);
        double kv_up   = (double)model_config.kv_lora_rank *
                         (model_config.num_heads / (double)tp) *
                         (model_config.qk_nope_head_dim + model_config.head_dim);
        double o_proj  = (double)(model_config.num_heads / (double)tp) *
                         model_config.head_dim * hidden_dim;
        attn_weights_per_gpu = (q_down + q_up + kv_down + kv_up + o_proj) * precision;
      } else {
        // Standard GQA: QKV projection + output projection, TP-split
        double qkv = hidden_dim *
                     (model_config.num_heads * model_config.head_dim +
                      2.0 * model_config.num_kv_heads * model_config.head_dim);
        double o_proj = hidden_dim * (model_config.num_heads * model_config.head_dim);
        attn_weights_per_gpu = (qkv + o_proj) * precision / tp;
      }

      // Parameter counts for the compute-time estimate (below).
      // Named _params (not _bytes) to prevent the precision byte-factor from
      // creeping back in: FLOPs = 2 * param_count * batch, and compute_peak_flops
      // is in op/s — precision bytes must NOT appear here.
      // Kept separate from attn_weights_per_gpu so the memory-latency estimate is unaffected.
      double attn_weights_params = (hidden_dim *
                                    (model_config.num_heads * model_config.head_dim +
                                     2.0 * model_config.num_kv_heads * model_config.head_dim) +
                                    hidden_dim * (model_config.num_heads * model_config.head_dim));

      // ---- MLP weights per layer (TP-aware, per GPU) -------------------------
      // Include shared experts; use e_tp_dg (not tp) for routed experts.
      double mlp_weights_per_gpu;   // bytes, single layer, per GPU
      double mlp_weights_params;    // parameter count for FLOP estimate (no precision factor, see above)
      if (model_config.num_routed_expert > 0) {
        // Routed experts: weight per device = num_routed / devices_per_stage × per_expert_weight.
        // e_tp_dg cancels: expert.cpp stores (num_routed * e_tp_dg / devices_per_stage) expert-
        // tensors per device, but each tensor is (inter/e_tp_dg)-wide (column-parallel), so
        // the product is num_routed / devices_per_stage × full_expert_weight (independent of EP).
        double routed = (double)model_config.num_routed_expert / devices_per_stage *
                        model_config.ffn_way * hidden_dim *
                        model_config.expert_intermediate_dim * precision;
        // Shared experts: NOT tensor-sharded at the live-simulation level
        // (expert.cpp builds the shared expert with use_dp=true -> a plain
        // Linear layer via layer.cpp's else-branch, not ColumnParallel/
        // RowParallel) -- full weight per GPU, matching the live sim's
        // recorded per-GPU weights exactly: 116.384 GiB @ tp1 == 106.020 GiB
        // @ tp2 (both full-shared arithmetic; /tp would predict 103.207 GiB
        // @ tp2, which does not match). Also mirrors the latency estimate
        // below (weight_for_latency, :266-268), which already treats the
        // shared expert as full-size.
        double shared = (double)model_config.num_shared_expert *
                        model_config.ffn_way * hidden_dim *
                        model_config.expert_intermediate_dim * precision;
        mlp_weights_per_gpu = routed + shared;
        // Use top_k active experts (not num_routed/tp): the outer /tp in total_flops
        // provides the TP split, so no inner /tp here.  No * precision: param count.
        // active experts per token: top_k routed + always-on shared (expert.cpp runs the shared expert over the full batch)
        mlp_weights_params  = 3.0 * hidden_dim * model_config.expert_intermediate_dim *
                              (model_config.top_k + model_config.num_shared_expert);
      } else {
        // Dense FFN: TP-split
        mlp_weights_per_gpu = 3.0 * hidden_dim * model_config.intermediate_dim * precision / tp;
        mlp_weights_params  = 3.0 * hidden_dim * model_config.intermediate_dim; // no * precision: param count
      }

      double weight_per_gpu = layers_per_stage * (attn_weights_per_gpu + mlp_weights_per_gpu);

      // Number of MoE layers in this PP stage, via an exact per-layer count
      // (isMoELayer(), shared with llm.cpp's module-construction path -- model/model_config.h,
      // which honors both first_k_dense and expert_freq for any model). Every stage is evaluated
      // and the HEAVIEST stage's count taken: this optimizer models one representative PP stage,
      // and the pipeline throughput / capacity (OOM) gate are both governed by the heaviest stage,
      // not an average one.
      int moe_layers_in_stage = 0;
      {
        int lps = (int)layers_per_stage;  // num_layers % pp == 0 is enforced above (line 30)
        for (int stage = 0; stage < pp; ++stage) {
          int start_layer = stage * lps;
          int count = 0;
          for (int layer = start_layer; layer < start_layer + lps; ++layer) {
            if (isMoELayer(model_config, layer)) count++;
          }
          moe_layers_in_stage = std::max(moe_layers_in_stage, count);
        }
      }
      int non_moe_in_stage = (int)layers_per_stage - moe_layers_in_stage;
      // Weight of a dense FFN layer inside a MoE model (e.g. first_k_dense layers of deepseekV3,
      // or the non-MoE layers of llama4). This uses intermediate_dim (not expert_intermediate_dim).
      // Note: distinct from mlp_weights_per_gpu which, for MoE models, holds the MoE-expert weight.
      double dense_ffn_per_layer = 3.0 * hidden_dim * model_config.intermediate_dim *
                                   precision / tp;

      if (model_config.num_routed_expert > 0) {
        // Rebuild weight estimate with the exact MoE/dense split.
        weight_per_gpu = layers_per_stage * attn_weights_per_gpu
                       + moe_layers_in_stage * mlp_weights_per_gpu
                       + non_moe_in_stage * dense_ffn_per_layer;
      }

      // ---- Parity terms: weights the live simulator RECORDS that the per-layer
      // sums above omit. Cluster::checkMemorySize gates on device 0's recorded
      // tensors, so any term missing here lets a batch pass this analytic gate
      // and then OOM in the live sim (the batch-4001..4095 "crash band",
      // PAPER_INCONSISTENCIES.md U1 -- the router term alone is that measured
      // ~0.01% drift: 24 layers x 5120 x 128 x 2B = 31.5 MB for maverick).
      //   1) MoE router/gate projection: expert.cpp's gate_fn =
      //      ColumnParallelLinear(hidden, num_routed) on the local device only
      //      -> replicated per GPU, one per MoE layer.
      //   2) LayerNorm gammas: 2 per decoder layer (input + post-attention),
      //      hidden x precision each, replicated (layernorm.cpp).
      //   3) Embedding / LM head: vocab-parallel {ceil(n_vocab/tp), hidden}
      //      tensors, TP-sharded Megatron-style (embedding.cpp / lm_head.cpp).
      //      Embedding lives on pp stage 0 and the LM head on the last stage; the
      //      live gate checks device 0 == stage 0, so mirror stage 0's holdings:
      //      embedding always, LM head only when pp == 1 (stage 0 is then also
      //      the last stage).
      double router_weight_bytes = (model_config.num_routed_expert > 0)
          ? moe_layers_in_stage * hidden_dim * model_config.num_routed_expert * precision
          : 0.0;
      double layernorm_weight_bytes = layers_per_stage * 2.0 * hidden_dim * precision;
      double vocab_rows_per_rank = std::ceil((double)model_config.n_vocab / tp);
      double embed_weight_bytes = vocab_rows_per_rank * hidden_dim * precision;
      double lm_head_weight_bytes = (pp == 1)
          ? (double)hidden_dim * vocab_rows_per_rank * precision
          : 0.0;
      weight_per_gpu += router_weight_bytes + layernorm_weight_bytes +
                        embed_weight_bytes + lm_head_weight_bytes;

      // Memory type determination (used both here and in the latency section below).
      bool use_flash = system_config.use_hbf && system_config.hbf_config.num_flash_stacks > 0;

      // Fix 2a: E_active — number of on-device routed experts that receive >=1 token.
      // The simulator streams the full weight of every active expert (linear_impl.cpp:35
      // short-circuits only experts with zero input tokens).
      //   experts_per_device = num_routed_expert * e_tp_dg / devices_per_stage
      //                      (expert.cpp:57: num_expert_per_device = num_routed / (parallel_num/e_tp))
      //   E_active = min(experts_per_device, ceil(batch * top_k * device_share))
      // For dense/non-MoE models e_active stays 0.
      double e_active = 0.0;
      if (model_config.num_routed_expert > 0 && model_config.top_k > 0) {
        // e_tp_dg cancels (same as weight formula): E_active is in "full expert" units
        // (full weight = ffn_way × hidden × inter, not divided by e_tp_dg).
        double experts_per_device = (double)model_config.num_routed_expert / devices_per_stage;
        double assignments        = (double)batch_size * model_config.top_k;
        double device_share       = experts_per_device / model_config.num_routed_expert;
        // MOE_TAG_FIX_SPEC §7 / PP_FLAGS_SPEC §4.5: microbatch basis, mirroring
        // the runtime cold-at-micro correction and PP_FIX_SPEC §4's
        // batch_for_latency = batch_size_per_gpu/pp. Latency streams only
        // A(B/pp) experts per stage per step; capacity is unaffected
        // (weight_per_gpu above keeps ALL experts resident). Keyed to the SAME
        // pp_pipelined_timing flag as batch_for_latency so the MoE tag's
        // optimizer mirror stays consistent with the runtime's dep-tag
        // (inert whenever pipelined timing is off -- PP_FLAGS_SPEC §4.5 proof).
        // pp==1 => eactive_pp==1 either way => regression-safe.
        double eactive_pp = system_config.pp_pipelined_timing ? pp : 1;
        e_active = std::min(experts_per_device, std::ceil(assignments * device_share / eactive_pp));
        if (e_active < 1.0) e_active = 1.0;
      }

      // Effective weight for LATENCY — E_active-based active-expert weight for
      // BOTH memory tiers.
      // The live simulator dispatches routed experts identically regardless of memory tier
      // (ExpertFFN::forward loops per active expert, charging each full k·n weight via
      // getLinearMemoryDuration, skipped only when a device receives zero tokens for that
      // expert — src/module/expert.cpp / src/hardware/linear_impl.cpp). There is NO
      // grouped-GEMM / active-row HBM-specific optimization anywhere in the dispatch code,
      // so the number of active experts whose weight is streamed is the right bandwidth
      // multiplier for HBM exactly as it is for flash. See CHANGES.md item 29 /
      // PAPER_INCONSISTENCIES.md U1/U2 for the tier-unification rationale.
      double weight_for_latency = weight_per_gpu;  // default: same as capacity weight (non-MoE)
      if (model_config.num_routed_expert > 0 && model_config.top_k > 0) {
        double shared_per_moe_layer = (double)model_config.num_shared_expert *
                                       model_config.ffn_way * hidden_dim *
                                       model_config.expert_intermediate_dim * precision;
        int non_moe_layers = layers_per_stage - moe_layers_in_stage;
        // Non-routed weight: attention + shared experts + dense-FFN layers (always read).
        // Include the parity weights the sim actually STREAMS each step: router
        // gate (gate_fn forward runs every MoE layer), LayerNorms, and the LM head
        // (pp==1). Embedding is recorded for capacity but its forward charges no
        // memory op (embedding.cpp), so it is excluded from the latency stream.
        double non_routed = layers_per_stage * attn_weights_per_gpu +
                            moe_layers_in_stage * shared_per_moe_layer +
                            non_moe_layers * dense_ffn_per_layer +
                            router_weight_bytes + layernorm_weight_bytes +
                            lm_head_weight_bytes;
        // E_active experts' full weight streamed sequentially per dispatch op (both tiers).
        double routed_active = e_active *
                               model_config.ffn_way * hidden_dim *
                               model_config.expert_intermediate_dim * precision;
        weight_for_latency = non_routed + moe_layers_in_stage * routed_active;
      }

      // KV cache size. Uses effectiveKvLenSumAllLayers() (model/model_config.h) instead of
      // "layers_per_stage * sequence_length" so Llama-4-style interleaved local/global
      // attention (attn_chunk_size>0) is accounted for: local layers only read/retain a
      // bounded window, not the full context. Divide the whole-model sum by pp to get this
      // representative stage's share (this optimizer evaluates one representative stage, not
      // a sum across stages -- consistent with weight_per_gpu above). Reduces EXACTLY to
      // "layers_per_stage * sequence_length" when attn_chunk_size==0 (every other model).
      double effective_seqsum_per_stage =
          effectiveKvLenSumAllLayers(model_config, sequence_length) / pp;
      double kv_cache_per_gpu = effective_seqsum_per_stage *
          (2.0 * (model_config.num_kv_heads / (double)tp) * model_config.head_dim * precision) *
          batch_size_per_gpu;
      if (model_config.compressed_kv) {
        // MLA/deepseek: full-global attention only (attn_chunk_size==0) -- whole-context formula.
        kv_cache_per_gpu = layers_per_stage * (model_config.kv_lora_rank + model_config.qk_rope_head_dim) * precision * batch_size_per_gpu * sequence_length;
      }

      // ---- Activation size (peak intermediate data, per GPU) -----------------
      // Gates the scarce tier (36-GB HBM stack for HBF, 320-MB logic SRAM for
      // HBF+/CONV+) against the PEAK concurrently-live footprint, not a sum of
      // every op's output -- see footprint.h::peakIntermediateBytes for why
      // (paper: intermediate data has a short lifetime and is quickly
      // released; the seq-length-scaled attention-score/decompressed-KV terms
      // stream through a separate double-buffer and are excluded entirely).
      // expert_batch_size: matches cluster.cpp:88 (uses total batch, not
      // per-gpu; gated by expert_freq). num_routed_expert_per_device: matches
      // cluster.cpp:84, and divides by devices_per_stage (the devices sharing one
      // pipeline stage's expert allotment), mirroring the weight term above (:123).
      double expert_batch_size = (model_config.expert_freq > 0 && model_config.num_routed_expert > 0)
          ? (double)batch_size * model_config.top_k / model_config.num_routed_expert
          : 0.0;
      double num_routed_expert_per_device = (model_config.num_routed_expert > 0)
          ? (double)model_config.num_routed_expert * e_tp_dg / devices_per_stage
          : 0.0;
      // peakIntermediateBytes/intermediateExtrasBytes divide the routed-expert
      // activation term by model.e_tp_dg internally (footprint.h:231, :367) to
      // undo the ×e_tp_dg baked into num_routed_expert_per_device above. But
      // model_config here is the caller's const config, whose e_tp_dg is the
      // sweep-invariant default (1) -- not this candidate's swept e_tp_dg. Use
      // a local copy with e_tp_dg patched to the swept value so the division
      // actually cancels; otherwise ep>1 candidates over-charge routed
      // activation bytes by a factor of ep.
      ModelConfig mc = model_config;
      mc.e_tp_dg = e_tp_dg;
      double act_size = peakIntermediateBytes(
          mc, batch_size_per_gpu, tp, expert_batch_size,
          num_routed_expert_per_device,
          /*has_moe_layer=*/model_config.num_routed_expert > 0,
          /*has_dense_layer=*/hasDenseFfnLayer(model_config));

      // Full faithful paper-1 intermediate-data accounting: add the complete
      // resident intermediate set (footprint.h::intermediateExtrasBytes --
      // KV-write on-chip staging + score tile + all-reduce scratch + MoE
      // dispatch/GateOut + tiled LM-head logits) on top of
      // peakIntermediateBytes. layers_per_stage is this representative
      // stage's layer count (num_layers/pp), matching the weight/KV terms
      // above. Mirrors the live-sim gate (cluster.cpp) so the two never
      // drift.
      act_size += intermediateExtrasBytes(mc, batch_size_per_gpu, tp,
                                          layers_per_stage);

      // ---- Memory limit verification (via shared checkCapacity) ---------------
      // Uses hbm_per_stack_bytes and the same partitioning rule as cluster.cpp (via footprint.h).
      {
        CapacityResult cap = checkCapacity(system_config, act_size, weight_per_gpu, kv_cache_per_gpu);
        if (cap.oom) {
          config.oom = true;
          config.oom_reason = cap.reason;
        }
      }

      // ---- Latency estimation -------------------------------------------------
      // Part D: two models selectable via system_config.optimizer_latency_model.
      //   "sum" (default/conservative): compute + memory terms are additive.
      //   "max": per-layer max(compute, weight_mem) + kv_access; hides overlap.
      double total_latency_ns = 0.0;

      // Memory bandwidth for weight (flash or HBM); use_flash set above near e_active.
      double weight_bw  = use_flash ? system_config.hbf_config.flash_read_bandwidth
                                    : system_config.memory_bandwidth;

      // Fix 2b: per-op page-read latency.
      // The simulator calls getLinearMemoryDuration once per linear op (layer_impl.h:17),
      // which now double-buffers each weight read through the same per-stack SRAM
      // staging mechanic as KV reads (audit F2 fix): exposed latency per op is
      // page_lat + (num_chunks-1)*max(0, page_lat-chunk_transfer). Under every current
      // preset's constants, chunk-transfer time at full SRAM capacity exceeds the page
      // latency for both HBF/HBF+ (1us) and CONV/CONV+ (3us), so this reduces to exactly
      // ONE exposed page latency per op regardless of the op's weight size or chunk
      // count -- i.e. `ops * page_lat_per_ns` below already matches the simulator's
      // per-op chunked model exactly. This equivalence would break only if a future
      // preset made chunk-transfer-at-full-SRAM-capacity shorter than the page latency
      // (e.g. much higher bandwidth or much higher page latency than any current
      // preset) -- if that ever happens, this formula would need to become genuinely
      // per-op-size-aware (this aggregate model doesn't track individual op weight
      // sizes) to stay in lock-step with layer_impl.h's getLinearMemoryDuration.
      // We count distinct ops per layer to match the cycle model.  HBM path has
      // page_lat_per_ns=0 so page_lat_total==0.
      //
      //   attn_ops: MLA-absorb=8 (q_down, kv_down, kr, q_up, qr, tr_k_up, v_up, o_proj)
      //             MLA-base≈7 (without separate tr_k_up/v_up expansion)
      //             GQA=4 (q/k/v + o_proj)
      //   moe_ffn_ops: 1 (gate_fn) + 3*E_active*e_tp_dg (routed; ffn_way=3) + 3*shared
      //     Note: with EP=e_tp_dg, each "full expert" is split into e_tp_dg weight tensors,
      //     each paying one page-read latency.  E_active is in "full expert" units, so we
      //     multiply by e_tp_dg to count actual LinearImpl calls.  Shared experts use ne_tp_dg
      //     (not e_tp_dg) for their TP split so their count does not scale with e_tp_dg.
      //   dense_ffn_ops: 3 (gate/up/down)
      double page_lat_per_ns = use_flash
          ? (double)system_config.hbf_config.flash_page_read_latency_ns : 0.0;
      // One pipeline-fill latency per per-iteration weight stream, in lock-step
      // with getLinearMemoryDuration's weight_stream_ops_per_iter amortization:
      // consecutive weight tensors double-buffer across ops (no activation
      // dependency), so per-op charging (the former attn_ops/moe_ffn_ops count
      // here) no longer matches the live model.
      double page_lat_total = page_lat_per_ns;

      // Weight read time: active-expert weight bandwidth + per-op page latency.
      double weight_read_time = (weight_for_latency / weight_bw * 1e9) + page_lat_total;

      // Compute time: FLOPs = 2 * param_count * batch (factor-of-2 is MAC→FLOP).
      // attn_weights_params and mlp_weights_params carry NO precision factor; that
      // factor must not appear here because compute_peak_flops is in op/s not byte-op/s.
      double total_flops = layers_per_stage *
                           (2.0 * attn_weights_params + 2.0 * mlp_weights_params) *
                           batch_for_latency / tp;
      double compute_time = total_flops /
          (system_config.compute_peak_flops * effectiveMFU(system_config, batch_for_latency)) * 1e9;

      // KV read time (decode). CONTEXT BASIS: the capacity term above sizes the
      // FULL lifetime (input+output — a sequence must fit at its longest), but
      // the per-step READ volume the live simulator measures is the
      // steady-state average context (input + output/2: continuous batching
      // with arrival matching completion spreads in-flight queries uniformly
      // over their output lifetime, and the live scheduler seeds decode
      // sequences at exactly that age). Using the full lifetime here
      // over-estimated attention-read latency by up to 22% (SHORT), which
      // deflated seed_tps below its supposed upper-bound role and could prune
      // the true winner unverified (run_experiments.py's pruning invariant).
      double steady_ctx = (double)sequence_length;
      if (model_config.input_len > 0 && model_config.output_len > 0 &&
          model_config.input_len + model_config.output_len == sequence_length) {
        steady_ctx = model_config.input_len + model_config.output_len / 2.0;
      }
      double kv_read_size;
      if (model_config.compressed_kv) {
        kv_read_size = layers_per_stage *
            (model_config.kv_lora_rank + model_config.qk_rope_head_dim) *
            precision * batch_for_latency * steady_ctx;
      } else {
        kv_read_size = (effectiveKvLenSumAllLayers(model_config, (int)steady_ctx) / pp) *
            (2.0 * (model_config.num_kv_heads / (double)tp) * model_config.head_dim * precision) *
            batch_for_latency;
      }
      double kv_read_time = 0.0;
      if (use_flash) {
        const auto& hbf = system_config.hbf_config;
        double sram_cap = (double)hbf.sram_per_stack_bytes * hbf.num_flash_stacks;
        // Chunked attention: matches getAttentionMemoryDuration in layer_impl.h.
        // Chunk granularity = configurable system.chunk_size (bytes); 0 = auto
        // (full SRAM staging capacity), explicit value clamped to that capacity.
        double chunk_bytes = (system_config.chunk_size > 0)
            ? std::min((double)system_config.chunk_size, sram_cap)
            : sram_cap;
        int num_chunks = (chunk_bytes > 0)
            ? (int)std::ceil(kv_read_size / chunk_bytes)
            : 1;
        if (num_chunks < 1) num_chunks = 1;
        // Double-buffered: page-read latency for chunk N+1 overlaps chunk N's
        // transfer, so only the first (pipeline-fill) chunk fully exposes it;
        // later chunks expose only the residual if a chunk's own transfer time
        // is shorter than the page latency. Must stay in lock-step with
        // getAttentionMemoryDuration (layer_impl.h) -- see its comment for the
        // full reasoning.
        double chunk_transfer_ns = chunk_bytes / hbf.flash_read_bandwidth * 1e9;
        double exposed_latency_ns = (double)hbf.flash_page_read_latency_ns +
            (num_chunks - 1) * std::max(0.0, (double)hbf.flash_page_read_latency_ns - chunk_transfer_ns);
        kv_read_time = (kv_read_size / hbf.flash_read_bandwidth * 1e9) + exposed_latency_ns;
      } else {
        kv_read_time = kv_read_size / system_config.memory_bandwidth * 1e9;
      }

      // KV write time: hidden behind attention compute (unhidden portion only).
      // Window-aware for Llama-4 iRoPE (attn_chunk_size>0): a local layer only ever
      // retains/writes a bounded KV window (effectiveKvLen caps input_len at
      // attn_chunk_size), matching the KV-READ term above (effectiveKvLenSumAllLayers,
      // line ~246) and the simulator's write cap (hardware/layer_impl.h's
      // getKVWriteDuration, capped via LayerInfo::local_attention_window).
      // unhidden_write = max(0, write - attn_compute) is NONLINEAR in the write size, and
      // attn_compute is identical for every layer (depends on attn_weights_params, not the
      // write length) -- so the nonlinear clamp is evaluated PER LAYER and summed, not
      // approximated by scaling a single representative layer's write by an average length.
      // When attn_chunk_size==0 (every layer global) this reduces to
      // num_layers*unhidden_write/pp == layers_per_stage*unhidden_write.
      double kv_write_time = 0.0;
      if (use_flash) {
        const auto& hbf = system_config.hbf_config;
        double num_new_queries = batch_for_latency /
                                 (model_config.output_len > 0 ? model_config.output_len : 1);

        // Attention-only compute per layer (the hiding budget). Basis matches the
        // simulator exactly: attention_gen_impl.cpp's unhidden_write overlaps only the
        // attention kernel's own compute (FFN is a separate kernel). Identical for every
        // layer, so computed once.
        double attn_flops_per_layer = 2.0 * attn_weights_params * batch_for_latency / tp;
        double single_layer_attn_compute = attn_flops_per_layer /
            (system_config.compute_peak_flops * effectiveMFU(system_config, batch_for_latency)) * 1e9;
        // Hiding budget is ATTENTION-ONLY compute (not attn+FFN), matching the simulator's
        // basis exactly: attention_gen_impl.cpp's unhidden_write = max(0, kv_write -
        // exec_status.compute_duration) only accumulates the attention kernel's own
        // compute time (FFN runs as a separate module/kernel and never contributes to
        // this overlap). Using total per-layer compute here would over-credit hiding and
        // under-count the write penalty relative to the simulator.

        auto unhidden_write_for = [&](double write_len) {
          double kv_write_size = model_config.compressed_kv
              ? num_new_queries * (model_config.kv_lora_rank + model_config.qk_rope_head_dim) *
                    write_len * precision
              : 2.0 * num_new_queries * (model_config.num_kv_heads / (double)tp) *
                    model_config.head_dim * write_len * precision;
          // Page-program latency amortized across the stage's per-layer write
          // stream, mirroring the simulator exactly (getKVWriteDuration's
          // program_latency_amortize_calls, passed as num_layers/pp at every
          // attention call site): the per-iteration stream exposes ONE program
          // tail, not one per layer.
          double single_layer_kv_write = (kv_write_size / hbf.flash_write_bandwidth * 1e9) +
                                         (double)hbf.flash_page_program_latency_ns /
                                             layers_per_stage;
          return std::max(0.0, single_layer_kv_write - single_layer_attn_compute);
        };

        double kv_write_all_layers = 0.0;
        for (int layer = 0; layer < model_config.num_layers; ++layer) {
          double write_len = effectiveKvLen(model_config, layer, (double)model_config.input_len);
          kv_write_all_layers += unhidden_write_for(write_len);
        }
        kv_write_time = kv_write_all_layers / pp;
      }

      if (system_config.optimizer_latency_model == "max") {
        // max model: compute and weight access overlap per layer; KV is sequential.
        double compute_per_layer     = compute_time / layers_per_stage;
        double weight_mem_per_layer  = weight_read_time / layers_per_stage;
        double kv_read_per_layer     = kv_read_time / layers_per_stage;
        double kv_write_per_layer    = kv_write_time / layers_per_stage;  // already unhidden
        double layer_ns = std::max(compute_per_layer, weight_mem_per_layer) +
                          kv_read_per_layer + kv_write_per_layer;
        total_latency_ns = layer_ns * layers_per_stage;
      } else {
        // sum model: for Flash, the weight DMA (from flash chips to HBM staging) can
        // overlap with GPU compute because they use separate controllers.  Page latencies
        // are sequential flash-controller stalls and remain additive.
        // For HBM, weight reads and compute contend over the same memory subsystem, so
        // no overlap credit is applied and the classic additive sum is kept.
        if (use_flash) {
          double weight_bw_only = weight_read_time - page_lat_total;
          double unhidden_bw = std::max(0.0, weight_bw_only - compute_time);
          total_latency_ns = compute_time + unhidden_bw + page_lat_total + kv_read_time + kv_write_time;
        } else {
          total_latency_ns = compute_time + weight_read_time + kv_read_time + kv_write_time;
        }
      }

      // Communication latency
      // TP All-Reduce (2 per layer) and PP send/receive share the same message shape.
      double inter_stage_message_size = batch_for_latency * hidden_dim * precision;
      // All-reduce cost model, mirroring the live AllReduce::forward
      // (src/module/communication.cpp): bandwidth-optimal volume 2(N-1)/N * size per
      // link, plus a LOGARITHMIC latency term 2*ceil(log2(N)) (recursive-doubling on
      // the NVSwitch-connected Rubin node) -- NOT the ring's 2(N-1) sequential hops.
      // Groups that span nodes (rank stride > gpus per node) pay the inter-node link.
      int gpus_per_node = system_config.num_device;
      auto allreduce_ns = [&](int n_ranks, double msg_bytes) -> double {
        if (n_ranks <= 1) return 0.0;
        bool cross_node = n_ranks > gpus_per_node;
        double lat = cross_node ? system_config.node_ict_latency
                                : system_config.device_ict_latency;
        double bw  = cross_node ? system_config.node_ict_bandwidth
                                : system_config.device_ict_bandwidth;
        double latency_hops = 2.0 * std::ceil(std::log2((double)n_ranks));
        return latency_hops * lat +
               (2.0 * (n_ranks - 1) / n_ranks) * (msg_bytes / bw * 1e9);
      };
      // Two all-reduces per dense layer (attention + decoder-FFN AR, decoder.cpp:62-63/:88);
      // ONE per MoE layer here (attention AR only -- layer.cpp:43-45/:58 -- the MoE gather AR
      // is charged separately below as moe_comm_time's ne_tp_ar_ns, expert.cpp:113-115/:197).
      // Recurs identically on EVERY pp stage, so it stays a PER-STAGE term (added into
      // stage_latency_ns). Reduces to the old 2.0*layers_per_stage form when moe_layers_in_stage==0.
      double tp_comm_time = 0.0;
      if (tp > 1) {
        tp_comm_time = allreduce_ns(tp, inter_stage_message_size) *
                       (2.0 * (layers_per_stage - moe_layers_in_stage) + 1.0 * moe_layers_in_stage);
      }

      // MoE communication, mirroring the live modules per MoE layer
      // (expert.cpp: MoEScatter -> e_tp all-reduce -> MoEGather -> ne_tp all-reduce):
      //  - scatter: crossing fraction excludes the ne_tp(=tp) group
      //    (communication.cpp MoEScatter builds its exclusion set from ne_tp_dg) and
      //    the volume is divided by ne_tp_dg (TP-replicated tokens are sent once);
      //  - gather: crossing fraction excludes the e_tp group and the volume is
      //    divided by e_tp_dg * ne_tp_dg (communication.cpp MoEGather);
      //  - link: node-aware max composition, mirroring the fixed live
      //    MoEScatter/MoEGather decode branches (communication.cpp): of the
      //    crossing fraction, the part on the local node rides NVLink
      //    (device_ict), the part on remote node(s) rides InfiniBand
      //    (node_ict), the two links run concurrently (max), and a zero
      //    fraction charges no latency;
      //  - the two all-reduces use the shared allreduce_ns model above (the e_tp
      //    group spans nodes when e_tp_dg > gpus_per_node).
      double moe_comm_time = 0.0;
      if (model_config.num_routed_expert > 0) {
        double token_bytes = batch_for_latency * model_config.top_k *
                             hidden_dim * precision;
        // Node-aware link split, mirroring the fixed live MoEScatter/MoEGather decode
        // branches (communication.cpp): intra-node crossing bytes ride NVLink,
        // inter-node bytes ride InfiniBand, links concurrent (max composition).
        // Stage devices are contiguous, so a stage has max(0, devices_per_stage - gpn)
        // devices on the remote node; the excluded group (contiguous) is local.
        int gpn = system_config.num_device;  // GPUs per node
        auto a2a_ns = [&](double excluded, double per_dev_bytes) -> double {
          double local_devs  = (double)std::min(devices_per_stage, gpn);
          double remote_devs = (double)devices_per_stage - local_devs;
          double intra_frac = std::max(0.0, local_devs - excluded) / devices_per_stage;
          double inter_frac = remote_devs / devices_per_stage;
          double intra_ns = intra_frac > 0 ? system_config.device_ict_latency +
              (intra_frac * per_dev_bytes) / system_config.device_ict_bandwidth * 1e9 : 0.0;
          double inter_ns = inter_frac > 0 ? system_config.node_ict_latency +
              (inter_frac * per_dev_bytes) / system_config.node_ict_bandwidth * 1e9 : 0.0;
          return std::max(intra_ns, inter_ns);
        };
        double scatter_ns = 0.0, gather_ns = 0.0;
        if (e_tp_dg < devices_per_stage) {
          scatter_ns = a2a_ns((double)tp, token_bytes / tp);
          gather_ns  = a2a_ns((double)e_tp_dg, token_bytes / (e_tp_dg * tp));
        }
        // Live e_tp AR (expert.cpp:188, all_reduce_for_e_tp) reduces the (batch, hidden)
        // block -- the un-duplicated input tensor -- not the top_k-duplicated scatter/gather
        // volume above, so it gets its own message size.
        double e_tp_ar_volume = batch_for_latency * hidden_dim * precision;
        double e_tp_ar_ns  = allreduce_ns(e_tp_dg, e_tp_ar_volume);
        double ne_tp_ar_ns = allreduce_ns(tp, inter_stage_message_size);
        moe_comm_time = moe_layers_in_stage *
                        (scatter_ns + gather_ns + e_tp_ar_ns + ne_tp_ar_ns);
      }
      double scatter_time = moe_comm_time;

      // total_latency_ns above (compute/weight/kv terms, "max" or "sum" model) plus
      // tp_comm_time and scatter_time together are ONE PIPELINE STAGE's decode-step
      // time. A decode token must traverse all `pp` stages SEQUENTIALLY before the
      // next token can begin (no micro-batch-level pipeline overlap is modeled -- one
      // simulator iteration is one full forward pass of the whole batch through every
      // stage; the live simulator mirrors this via PipelineStage::forward's
      // time-propagation in communication.cpp and Cluster::maxDeviceTime() in
      // cluster.cpp, which read the fully-propagated per-token latency across stages).
      // So the true per-token latency is `pp` stages' worth of this per-stage total,
      // plus (pp-1) inter-stage send/receive hops.
      double stage_latency_ns = total_latency_ns + tp_comm_time + scatter_time;
      double full_pipeline_latency_ns = stage_latency_ns * pp;

      if (pp > 1) {
        int src_node = 0;
        int dst_node = (total_gpus / pp) / system_config.num_device; // approximate next stage node
        double pp_hop_comm_time = 0.0;
        if (src_node == dst_node) {
          pp_hop_comm_time = system_config.device_ict_latency + (inter_stage_message_size / system_config.device_ict_bandwidth * 1e9);
        } else {
          pp_hop_comm_time = system_config.node_ict_latency + (inter_stage_message_size / system_config.node_ict_bandwidth * 1e9);
        }
        full_pipeline_latency_ns += (double)(pp - 1) * pp_hop_comm_time;
      }

      // NOTE: estimated_latency_ms is used only to RANK capacity-feasible candidates
      // (see selection below) — it must never set config.oom, and no SLO check is
      // performed against it here. The simulator's measured tpot remains the sole SLO
      // arbiter (see run_experiments.py's analytic-search-then-simulator-verify batch
      // search, which always falls through to a real verify() call regardless of what
      // this estimate says). Capacity/SRAM (checked above via checkCapacity) remain
      // the only hard feasibility constraints in this function.
      config.estimated_latency_ms = full_pipeline_latency_ns / 1e6 * system_config.latency_margin;

      // Part E: store predicted footprint for the drift harness
      config.pred_weight_bytes = weight_per_gpu;
      config.pred_kv_bytes     = kv_cache_per_gpu;
      config.pred_act_bytes    = act_size;
      config.pred_total_bytes  = weight_per_gpu + kv_cache_per_gpu + act_size;

      return config;
}

ParallelConfig ParallelismOptimizer::Optimize(const ModelConfig& model_config,
                                             const SystemConfig& system_config,
                                             int total_gpus,
                                             int batch_size,
                                             int sequence_length,
                                             double tpot_slo_ms) {
  (void)tpot_slo_ms;  // SLO is deliberately NOT a gate here -- see ranking comment.
  std::vector<ParallelConfig> candidates = EnumerateCandidates(
      model_config, system_config, total_gpus, batch_size, sequence_length,
      /*require_batch_divisible=*/true);

  // Find the optimal non-OOM config: rank by SYSTEM throughput, which at a fixed
  // total batch reduces to argmin(estimated_latency_ms). Derivation: every decode
  // step, the whole system (all DP replicas together) emits `batch_size` tokens in
  // one step latency L, so system TPS = batch_size / L and per-GPU TPS =
  // batch_size / (L * total_gpus). Within one Optimize() call, batch_size and
  // total_gpus are constants across candidates, so argmax TPS == argmin L. This
  // matches the paper's stated objective (§III: "selects the parallelism
  // configuration that maximizes the achievable system throughput subject to ...
  // SLO requirements"). Note the throughput rank uses SYSTEM batch (not batch/dp):
  // a dp-way config's dp replicas each carry batch/dp and run concurrently, so the
  // per-replica /dp cancels.
  //
  // Deliberately NO SLO veto here based on estimated_latency_ms: this estimate is a
  // heuristic that can diverge from the live simulator, so hard-gating on it risks
  // discarding a candidate the real simulator would accept. Capacity/SRAM (checked
  // above via checkCapacity) remain the ONLY hard vetoes; run_experiments.py's
  // verify() against the live simulator remains the sole SLO arbiter.
  ParallelConfig optimal_config;
  double min_latency_ms = -1.0;
  bool found_valid = false;

  for (const auto& cand : candidates) {
    if (!cand.oom) {
      if (!found_valid || cand.estimated_latency_ms < min_latency_ms) {
        min_latency_ms = cand.estimated_latency_ms;
        optimal_config = cand;
        found_valid = true;
      }
    }
  }

  if (!found_valid) {
    // If all are OOM, return the first one with OOM set
    optimal_config = candidates.empty() ? ParallelConfig() : candidates[0];
    optimal_config.oom = true;
  }

  return optimal_config;
}

} // namespace llm_system
