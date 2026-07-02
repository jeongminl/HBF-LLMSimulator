#include "optimizer/parallelism_optimizer.h"
#include "model/footprint.h"
#include <cmath>
#include <algorithm>

namespace llm_system {

ParallelConfig ParallelismOptimizer::Optimize(const ModelConfig& model_config,
                                             const SystemConfig& system_config,
                                             int total_gpus,
                                             int batch_size,
                                             int sequence_length,
                                             double tpot_slo_ms) {
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

      int dp = total_gpus / (tp * pp);
      // Skip configs where batch cannot be evenly divided across DP groups.
      // This is physically correct: uneven DP creates load imbalance.  The
      // always-available dp=1 config ensures feasibility is never lost.
      // Granularity (finding the best batch near a non-divisible point) is
      // handled by the external sweep (run_flash_only.py) rather than relaxing
      // this constraint (ceil(batch/dp) would diverge from the live-sim's
      // integer floor division and reintroduce a gate mismatch).
      if (batch_size % dp != 0) continue;

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

      ParallelConfig config;
      config.tp = tp;
      config.pp = pp;
      config.dp = dp;
      config.ep = e_tp_dg;

      // Memory footprint calculation
      double layers_per_stage = (double)model_config.num_layers / pp;
      double batch_size_per_gpu = (double)batch_size / dp;
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
        // Shared experts: fully replicated on each device (not TP-split).
        double shared = (double)model_config.num_shared_expert *
                        model_config.ffn_way * hidden_dim *
                        model_config.expert_intermediate_dim * precision;
        mlp_weights_per_gpu = routed + shared;
        // Use top_k active experts (not num_routed/tp): the outer /tp in total_flops
        // provides the TP split, so no inner /tp here.  No * precision: param count.
        mlp_weights_params  = 3.0 * hidden_dim * model_config.expert_intermediate_dim *
                              model_config.top_k;
      } else {
        // Dense FFN: TP-split
        mlp_weights_per_gpu = 3.0 * hidden_dim * model_config.intermediate_dim * precision / tp;
        mlp_weights_params  = 3.0 * hidden_dim * model_config.intermediate_dim; // no * precision: param count
      }

      double weight_per_gpu = layers_per_stage * (attn_weights_per_gpu + mlp_weights_per_gpu);

      // Determine how many MoE layers are in this PP stage, using an EXACT per-layer count
      // (isMoELayer(), shared with llm.cpp's module-construction path -- model/model_config.h)
      // instead of the old two-pattern approximation. This fixes two bugs in one pass:
      //   1) first_k_dense handling was gated to `model_name == "deepseekV3"` only, silently
      //      ignoring it for any other model with first_k_dense>0 (e.g. llama4_maverick/scout).
      //   2) the expert_freq>1 branch rounded an evenly-averaged count
      //      (round(total_moe_layers/pp)) instead of counting each stage exactly, which could
      //      over/under-count a specific stage's MoE-layer weight.
      // isMoELayer() already honors both first_k_dense and expert_freq for any model, so a
      // single exact per-stage count replaces both old patterns. We evaluate every stage and
      // take the HEAVIEST stage's count: this optimizer models one representative PP stage,
      // and the pipeline throughput / capacity (OOM) gate are both governed by the slowest/
      // heaviest stage, not an average one.
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
        e_active = std::min(experts_per_device, std::ceil(assignments * device_share));
        if (e_active < 1.0) e_active = 1.0;
      }

      // Fix 2c: Effective weight for LATENCY.
      // Flash path (use_flash=true): use E_active-based active expert weight — the simulator
      //   streams the full weight of every active expert, and page latency makes each op
      //   sequential, so the active-expert count is the right bandwidth multiplier.
      // HBM path (use_flash=false): keep the historical sparse-ratio formula that was
      //   calibrated against HBM timing; it accounts for the compute-memory overlap that
      //   the "sum" latency model cannot model directly.
      double weight_for_latency = weight_per_gpu;  // default: same as capacity weight (non-MoE)
      if (model_config.num_routed_expert > 0 && model_config.top_k > 0) {
        double shared_per_moe_layer = (double)model_config.num_shared_expert *
                                       model_config.ffn_way * hidden_dim *
                                       model_config.expert_intermediate_dim * precision;
        int non_moe_layers = layers_per_stage - moe_layers_in_stage;
        // Non-routed weight: attention + shared experts + dense-FFN layers (always read)
        double non_routed = layers_per_stage * attn_weights_per_gpu +
                            moe_layers_in_stage * shared_per_moe_layer +
                            non_moe_layers * dense_ffn_per_layer;
        if (use_flash) {
          // Flash: E_active experts' full weight streamed sequentially per page-read op.
          double routed_active = e_active *
                                 model_config.ffn_way * hidden_dim *
                                 model_config.expert_intermediate_dim * precision;
          weight_for_latency = non_routed + moe_layers_in_stage * routed_active;
        } else {
          // HBM: historical sparse-ratio formula (top_k/num_routed applied to per-device
          // routed weight). Use num_routed/devices_per_stage (e_tp_dg cancels, same as
          // mlp_weights formula above).
          double sparse_ratio = (double)model_config.top_k / model_config.num_routed_expert;
          double routed_per_moe_layer = (double)model_config.num_routed_expert / devices_per_stage *
                                         model_config.ffn_way * hidden_dim *
                                         model_config.expert_intermediate_dim * precision;
          weight_for_latency = non_routed + moe_layers_in_stage * routed_per_moe_layer * sparse_ratio;
        }
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
        // MLA/deepseek: full-global attention only (attn_chunk_size==0), unaffected by this
        // change -- kept as the pre-existing whole-context formula.
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
      // cluster.cpp:84, and now divides by devices_per_stage (not total_gpus),
      // mirroring the weight term's already-correct pattern above (:123) --
      // dividing by total_gpus was an 8x undercount for degenerate
      // one-device-per-stage configs (e.g. PP=8/EP=1: devices_per_stage=1).
      double expert_batch_size = (model_config.expert_freq > 0 && model_config.num_routed_expert > 0)
          ? (double)batch_size * model_config.top_k / model_config.num_routed_expert
          : 0.0;
      double num_routed_expert_per_device = (model_config.num_routed_expert > 0)
          ? (double)model_config.num_routed_expert * e_tp_dg / devices_per_stage
          : 0.0;
      double act_size = peakIntermediateBytes(
          model_config, batch_size_per_gpu, tp, expert_batch_size,
          num_routed_expert_per_device,
          /*has_moe_layer=*/model_config.num_routed_expert > 0,
          /*has_dense_layer=*/hasDenseFfnLayer(model_config));

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
      double page_lat_total = 0.0;
      if (page_lat_per_ns > 0.0) {
        int attn_ops = (model_config.q_lora_rank != 0)
            ? (model_config.use_absorb ? 8 : 7)   // MLA-absorb=8, MLA-base~7
            : 4;                                    // GQA: Q+K+V+O
        int non_moe_layers_in_stage = layers_per_stage - moe_layers_in_stage;
        int moe_ffn_ops   = 1                                                     // gate routing op
                          + 3 * (int)std::round(e_active) * e_tp_dg              // page reads scale with e_tp_dg
                          + 3 * model_config.num_shared_expert;                   // shared (ne_tp_dg split)
        int dense_ffn_ops = 3;                                      // gate/up/down for dense FFN
        int ops_per_moe_layer   = attn_ops + (model_config.num_routed_expert > 0
                                              ? moe_ffn_ops : dense_ffn_ops);
        int ops_per_dense_layer = attn_ops + dense_ffn_ops;
        page_lat_total = (moe_layers_in_stage * ops_per_moe_layer +
                          non_moe_layers_in_stage * ops_per_dense_layer) * page_lat_per_ns;
      }

      // Weight read time: active-expert weight bandwidth + per-op page latency.
      double weight_read_time = (weight_for_latency / weight_bw * 1e9) + page_lat_total;

      // Compute time: FLOPs = 2 * param_count * batch (factor-of-2 is MAC→FLOP).
      // attn_weights_params and mlp_weights_params carry NO precision factor; that
      // factor must not appear here because compute_peak_flops is in op/s not byte-op/s.
      double total_flops = layers_per_stage *
                           (2.0 * attn_weights_params + 2.0 * mlp_weights_params) *
                           batch_size_per_gpu / tp;
      double compute_time = total_flops /
          (system_config.compute_peak_flops * effectiveMFU(system_config, batch_size_per_gpu)) * 1e9;

      // KV read time (decode: reads full KV cache from flash/HBM)
      double kv_read_size = kv_cache_per_gpu;
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
      // getKVWriteDuration, capped via LayerInfo::local_attention_window). Without this,
      // the write term contradicted the capacity/read terms (claiming to write more KV
      // than the cache is ever sized to hold/read back).
      // unhidden_write = max(0, write - attn_compute) is NONLINEAR in the write size, and
      // attn_compute is identical for every layer (depends on attn_weights_params, not the
      // write length) -- so the nonlinear clamp must be evaluated PER LAYER and summed, not
      // approximated by scaling a single representative layer's write by an average length.
      // Reduces EXACTLY to the old `unhidden_write * layers_per_stage` when attn_chunk_size==0
      // (every layer global => effectiveKvLen==input_len for all layers => sum/pp ==
      // num_layers*unhidden_write/pp == layers_per_stage*unhidden_write).
      double kv_write_time = 0.0;
      if (use_flash) {
        const auto& hbf = system_config.hbf_config;
        double num_new_queries = batch_size_per_gpu /
                                 (model_config.output_len > 0 ? model_config.output_len : 1);

        // Attention-only compute per layer (the hiding budget). Basis matches the
        // simulator exactly: attention_gen_impl.cpp's unhidden_write overlaps only the
        // attention kernel's own compute (FFN is a separate kernel) -- see the original
        // comment below. Identical for every layer, so computed once.
        double attn_flops_per_layer = 2.0 * attn_weights_params * batch_size_per_gpu / tp;
        double single_layer_attn_compute = attn_flops_per_layer /
            (system_config.compute_peak_flops * effectiveMFU(system_config, batch_size_per_gpu)) * 1e9;
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
          double single_layer_kv_write = (kv_write_size / hbf.flash_write_bandwidth * 1e9) +
                                         (double)hbf.flash_page_program_latency_ns;
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
      double inter_stage_message_size = batch_size_per_gpu * hidden_dim * precision;
      // Ring all-reduce, matching the live simulator's AllReduce::forward exactly
      // (src/module/communication.cpp): for a group of size N, cost = 2*(N-1) hops,
      // each hop paying one device_ict_latency plus 1/N of the message over
      // device_ict_bandwidth. Two all-reduces per layer (attention + FFN). This
      // recurs identically on EVERY pp stage (each stage does its own TP all-reduce
      // over its own layers), so it stays a PER-STAGE term below (added into
      // stage_latency_ns, not total_latency_ns directly).
      double tp_comm_time = 0.0;
      if (tp > 1) {
        double per_allreduce =
            2.0 * (tp - 1) * system_config.device_ict_latency +
            (2.0 * (tp - 1) / tp) * (inter_stage_message_size / system_config.device_ict_bandwidth * 1e9);
        tp_comm_time = 2.0 * layers_per_stage * per_allreduce;
      }

      // MoE scatter/gather overhead: when e_tp_dg < devices_per_stage, each MoE layer
      // requires scattering a fraction of the batch tokens to another device and gathering
      // results back (NVLink hop).  When e_tp_dg == devices_per_stage, all experts are
      // local to each device (EP-TP group) so scatter is eliminated.
      // This term correctly makes EP=devices_per_stage strictly better than EP < devices_per_stage
      // when bandwidth and page-latency are otherwise equivalent (as on HBM). Also a
      // PER-STAGE term (each stage's own MoE layers), same reasoning as TP above.
      double scatter_time = 0.0;
      if (model_config.num_routed_expert > 0 && e_tp_dg < devices_per_stage) {
        double scatter_frac = 1.0 - (double)e_tp_dg / devices_per_stage;
        // Tokens that need to cross a device boundary (scatter + gather = 2×)
        double scatter_msg = scatter_frac * batch_size_per_gpu * model_config.top_k *
                             hidden_dim * precision;
        // 2 (scatter + gather) × moe_layers_in_stage intra-node NVLink round-trips
        scatter_time = 2.0 * moe_layers_in_stage *
                       (system_config.device_ict_latency +
                        scatter_msg / system_config.device_ict_bandwidth * 1e9);
      }

      // total_latency_ns above (compute/weight/kv terms, "max" or "sum" model) plus
      // tp_comm_time and scatter_time together are ONE PIPELINE STAGE's decode-step
      // time. A decode token must traverse all `pp` stages SEQUENTIALLY before the
      // next token can begin (no micro-batch-level pipeline overlap is modeled -- one
      // simulator iteration is one full forward pass of the whole batch through every
      // stage; see PipelineStage::forward's time-propagation fix in communication.cpp
      // for the live-simulator side of this same correction, and Cluster::
      // maxDeviceTime() in cluster.cpp for how the live sim now reads the true,
      // fully-propagated per-token latency instead of stage 0's local time alone).
      // So the true per-token latency is `pp` stages' worth of this per-stage total,
      // plus (pp-1) inter-stage send/receive hops -- not one stage's total alone,
      // which is what this estimate computed before this fix (an ~pp x under-count
      // for pp>1, matching the live simulator's pre-fix bug).
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

      candidates.push_back(config);
      }  // expert-parallelism (e_tp_dg) sweep
    }
  }

  // Find the optimal non-OOM config: rank by THROUGHPUT (batch_size_per_gpu /
  // estimated_latency_ms), not argmin(latency). Matches the paper's own stated
  // objective (§III: "selects the parallelism configuration that maximizes the
  // achievable system throughput subject to ... SLO requirements") rather than
  // minimizing latency for its own sake once a config is already capacity-feasible.
  // Deliberately NO added SLO veto here based on estimated_latency_ms: this estimate
  // is a heuristic that can diverge from the live simulator (this file's own
  // pp-latency fix above is proof it was systematically wrong until now) -- hard-
  // gating on it risks discarding a candidate the real simulator would actually
  // accept. Capacity/SRAM (checked above via checkCapacity) remain the ONLY hard
  // vetoes; run_experiments.py's verify() against the live simulator remains the
  // sole SLO arbiter.
  ParallelConfig optimal_config;
  double max_throughput = -1.0;
  bool found_valid = false;

  for (const auto& cand : candidates) {
    if (!cand.oom) {
      double cand_batch_size_per_gpu = (double)batch_size / cand.dp;
      double cand_throughput = cand_batch_size_per_gpu / cand.estimated_latency_ms;
      if (cand_throughput > max_throughput) {
        max_throughput = cand_throughput;
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
