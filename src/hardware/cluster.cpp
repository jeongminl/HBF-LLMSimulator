#include "hardware/cluster.h"

#include <filesystem>
#include <cmath>

#include "common/assert.h"
#include "hardware/stat.h"
#include "model/footprint.h"
#include "module/module_graph.h"
#include "module/tensor.h"
#include "module/timeboard.h"

namespace llm_system {

Cluster::Cluster(SystemConfig config, Scheduler::Ptr scheduler)
    : config(config), scheduler(scheduler), executor() {
  cluster_ict_latency = config.node_ict_latency;
  cluster_ict_bandwidth = config.node_ict_bandwidth;
  num_device = config.num_device;
  num_node = config.num_node;
  num_total_device = num_device * num_node;
}

Device::Ptr Cluster::get_device(int device_total_rank) {
  int node_id = device_total_rank / num_device;
  return node.at(node_id)->get_device(device_total_rank);
}

time_ns Cluster::maxDeviceTime() {
  time_ns max_time = 0;
  for (int dr = 0; dr < num_total_device; dr++) {
    max_time = std::max(max_time, get_device(dr)->status.device_time);
  }
  return max_time;
}

void Cluster::add_module(int device_rank, std::string name,
                         Module::Ptr module) {
  auto &module_map_ = module_map.at(device_rank);

  if (module_map_.find(name) == module_map_.end()) {
    module_map_.emplace(name, module);
  } else {
    fail("Cluster::add_module, same module name");
  }
}

void Cluster::set_dependency() {
  for (Node::Ptr _node : node) {
    _node->set_dependency();
  }
}

void Cluster::restartModuleGraph() {
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr device = get_device(device_rank);
    device->restartGraph();
    device->reset_status();
    device->reset_timeboard();
  }
}

void Cluster::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr device = get_device(device_rank);
    device->initializeDRAM(ProcessorType, dramEnergy);
  }
}

void Cluster::set_dependency_tensor(std::vector<Tensor::Ptr> &list,
                                    Tensor::Ptr tensor,
                                    const std::vector<int> &device_list) {
  list.resize(0);

  Tensor::Ptr temp;
  Module::Ptr module;
  for (int device_rank :
       device_list) {  // for modules of devices in device_list, check dependency which if they have same name of tensor
    module = module_map.at(device_rank).at(tensor->get_module_map_name());
    temp = module->get_activation(tensor->name, {}, false);
    list.push_back(temp);
  }
}

bool Cluster::checkMemorySize(double pred_weight_bytes,
                               double pred_kv_bytes,
                               double pred_act_bytes,
                               double pred_total_bytes) {
  Device::Ptr device = get_device(0);
  auto module = module_map.at(0).at("::LLM");
  auto size_vector = module->get_size();

  int ne_tp_dg = device->model_config.ne_tp_dg;
  int e_tp_dg = device->model_config.e_tp_dg;

  int num_total_device = device->config.num_device * device->config.num_node;
  // devices_per_stage = devices sharing ONE pipeline stage's expert allotment
  // (total_gpus / pp), matching parallelism_optimizer.cpp's `devices_per_stage`
  // (its weight term at parallelism_optimizer.cpp:123 divides by this, not by
  // num_total_device).
  int devices_per_stage = num_total_device / device->model_config.pp_dg;
  // Use double: Maverick top_k=1/num_routed=128 gives expert_batch_size=0 as int
  // below batch 128, zeroing all MoE-FFN activation terms in the live-sim gate.
  double num_routed_expert_per_device = (double)device->model_config.num_routed_expert * e_tp_dg / devices_per_stage;

  int batch_size_per_dp = scheduler->batch_size_per_dp;
  int total_batch_size = scheduler->total_batch_size;
  double expert_batch_size = device->model_config.expert_freq ? (double)total_batch_size * device->model_config.top_k / device->model_config.num_routed_expert : 0.0;

  int input_len = device->model_config.input_len;

  int hidden_dim = device->model_config.hidden_dim;
  int q_lora_rank = device->model_config.q_lora_rank;
  int kv_lora_rank = device->model_config.kv_lora_rank;
  int qk_rope_head_dim = device->model_config.qk_rope_head_dim;
  int head_dim = device->model_config.head_dim;
  int num_heads = device->model_config.num_heads;
  int expert_intermediate_dim = device->model_config.expert_intermediate_dim;

  long long activation_size = 0;
  if(config.decode_mode){
    // Peak concurrently-live intermediate-data footprint (not a sum of every
    // op's output) -- see footprint.h::peakIntermediateBytes. Must stay the
    // single shared definition with parallelism_optimizer.cpp so the two
    // scarce-tier gates (Part C below / checkCapacity) never drift apart.
    activation_size = (long long)peakIntermediateBytes(
        device->model_config, batch_size_per_dp, ne_tp_dg, expert_batch_size,
        num_routed_expert_per_device,
        /*has_moe_layer=*/device->model_config.num_routed_expert > 0,
        /*has_dense_layer=*/hasDenseFfnLayer(device->model_config));
  }
  else{ // prefill mode & colocated system (mixed)
    if(device->model_config.use_absorb){
      // Every product below leads with a (double) cast on batch_size_per_dp so the
      // whole left-to-right multiply chain evaluates in double -- with realistic
      // input_len (thousands) and num_heads (128), the equivalent all-int expression
      // overflows INT_MAX before ever reaching a double operand.
      activation_size =
        ((double)batch_size_per_dp * input_len * hidden_dim + // input seqeunces (or tokens)
        (double)batch_size_per_dp * input_len * q_lora_rank + // c_q
        (double)batch_size_per_dp * input_len * kv_lora_rank + // c_kv
        (double)batch_size_per_dp * input_len * qk_rope_head_dim + // kr

        (double)batch_size_per_dp * input_len * (3.0 * qk_rope_head_dim + head_dim) * num_heads / ne_tp_dg + // query + rope out + cos/sin
        (double)batch_size_per_dp * input_len * num_heads * kv_lora_rank / ne_tp_dg + // tr_k up out

        2.0 * ((double)batch_size_per_dp * input_len *  num_heads * input_len / ne_tp_dg) + // attn score out
        (double)batch_size_per_dp * input_len * num_heads * kv_lora_rank / ne_tp_dg + // attn context out

        (double)batch_size_per_dp * input_len * num_heads * head_dim / ne_tp_dg + // v_up out

        (double)batch_size_per_dp * input_len * hidden_dim + // out proj out

        // MoE FFN — routed experts at expert_batch_size; the SHARED expert is
        // dense (sees the full per-device batch; expert.cpp passes the whole
        // input) and is TP-sharded on the ne_tp group at runtime.
        // Column-parallel gate/up/silu over e_tp (expert.cpp:84): shard width is
        // expert_intermediate_dim/e_tp_dg; num_routed_expert_per_device already
        // carries the x e_tp_dg expert-count factor, so full width double-counts it.
        // Down-proj stays full width (row-parallel input).
        num_routed_expert_per_device *
        (2.0 * (expert_batch_size * input_len * expert_intermediate_dim / e_tp_dg) + // gate proj out + silu out
        (expert_batch_size * input_len * expert_intermediate_dim / e_tp_dg) + // up proj out
        (expert_batch_size * input_len * hidden_dim)) +  // down proj out
        (double)device->model_config.num_shared_expert *
        (2.0 * ((double)batch_size_per_dp * input_len * expert_intermediate_dim / ne_tp_dg) +
        ((double)batch_size_per_dp * input_len * expert_intermediate_dim / ne_tp_dg) +
        ((double)batch_size_per_dp * input_len * hidden_dim))) *
        device->model_config.precision_byte;
    }
    else{ // base
      // Same double-cast chain as the absorb branch above (avoids INT_MAX overflow).
      activation_size =
        ((double)batch_size_per_dp * input_len * hidden_dim + // input seqeunces (or tokens)
        (double)batch_size_per_dp * input_len * q_lora_rank + // c_q
        (double)batch_size_per_dp * input_len * kv_lora_rank + // c_kv
        (double)batch_size_per_dp * input_len * qk_rope_head_dim + // kr

        (double)batch_size_per_dp * input_len * (3.0 * qk_rope_head_dim + head_dim) * num_heads / ne_tp_dg + // query + rope out + cos/sin
        (double)batch_size_per_dp * input_len * 2.0 * (head_dim) * num_heads / ne_tp_dg + // kv

        2.0 * ((double)batch_size_per_dp * input_len * input_len * num_heads / ne_tp_dg) + // attn score out
        (double)batch_size_per_dp * input_len * num_heads * head_dim / ne_tp_dg + // attn context out

        (double)batch_size_per_dp * input_len * hidden_dim + // out proj out

        // MoE FFN — routed experts at expert_batch_size; the SHARED expert is
        // dense (sees the full per-device batch; expert.cpp passes the whole
        // input) and is TP-sharded on the ne_tp group at runtime.
        // Column-parallel gate/up/silu over e_tp (expert.cpp:84): shard width is
        // expert_intermediate_dim/e_tp_dg; num_routed_expert_per_device already
        // carries the x e_tp_dg expert-count factor, so full width double-counts it.
        // Down-proj stays full width (row-parallel input).
        num_routed_expert_per_device *
        (2.0 * (expert_batch_size * input_len * expert_intermediate_dim / e_tp_dg) + // gate proj out + silu out
        (expert_batch_size * input_len * expert_intermediate_dim / e_tp_dg) + // up proj out
        (expert_batch_size * input_len * hidden_dim)) +  // down proj out
        (double)device->model_config.num_shared_expert *
        (2.0 * ((double)batch_size_per_dp * input_len * expert_intermediate_dim / ne_tp_dg) +
        ((double)batch_size_per_dp * input_len * expert_intermediate_dim / ne_tp_dg) +
        ((double)batch_size_per_dp * input_len * hidden_dim))) *
        device->model_config.precision_byte;
    }
  }

  // Use analytic activation_size for both branches so the printed ACT:, the size total,
  // and the Part C scarce-tier gate all reference the same per-decode-step metric.
  double size = 0;
  std::cout << "ACT: "
            << activation_size / 1024.0 / 1024 / 1024
            << "GB, Weight: " << size_vector.at(1) / 1024.0 / 1024 / 1024
            << "GB, Cache: " << size_vector.at(2) / 1024.0 / 1024 / 1024 << "GB"
            << std::endl;
  size = activation_size + size_vector.at(1) + size_vector.at(2);
                 
  std::cout << "Total: " << size / 1024.0 / 1024 / 1024 << "GB" << std::endl;

  // ---- Part C: generalised activation-scarce-tier check ---------------------
  // Checks activation against the scarce tier -- logic SRAM when num_hbm_stacks==0,
  // else the HBM-stack capacity -- using the scarceTierActivationLimit() helper
  // shared with the optimizer. Side-effect-free: only sets out_of_memory / fail(),
  // never mutates batch size (the mem_cap_limit batch-shrink path at :289 is separate).
  if (hasScarceTier(config)) {
    double act_limit = scarceTierActivationLimit(config);
    // Use the analytic per-decode-step activation peak for both MLA and non-MLA models.
    // The recorded size_vector.at(0)/num_layers is the cumulative activation across the
    // full scheduler working set, not the per-step peak — using it here would falsely OOM
    // HBF+/CONV+ runs even at batch=1 (see Part E comment below for why it is skipped
    // in the drift harness for the same reason).  The analytic activation_size is computed
    // above (peakIntermediateBytes for decode; the older summed formula for prefill/mixed)
    // and is the same metric used by the optimizer's checkCapacity gate, so the two gates
    // are consistent for the decode-mode path that all current sweeps use.
    double act_val = activation_size;

    // Diagnostic (never gates): paper-style score-inclusive footprint and the
    // per-GPU batch ceiling it would imply if it WERE the gate -- logged so
    // sweeps can compare both SRAM accountings without changing any reported
    // metric. See footprint.h::scoreInclusiveIntermediateBytes and
    // PAPER_INCONSISTENCIES.md U7.
    if (config.decode_mode && batch_size_per_dp > 0) {
      double diag_act = scoreInclusiveIntermediateBytes(
          device->model_config, batch_size_per_dp, ne_tp_dg, expert_batch_size,
          num_routed_expert_per_device,
          device->model_config.num_routed_expert > 0,
          hasDenseFfnLayer(device->model_config),
          (double)device->model_config.input_len +
              device->model_config.output_len);
      double per_seq = diag_act / batch_size_per_dp;
      double ceiling_per_gpu = (per_seq > 0)
          ? (act_limit / per_seq) * scheduler->dp_degree / num_total_device
          : 0.0;
      std::cout << "SRAM_DIAG_SCORE_INCLUSIVE_ACT_BYTES: " << diag_act << std::endl;
      std::cout << "SRAM_DIAG_CEILING_BATCH_PER_GPU: " << ceiling_per_gpu << std::endl;
    }

    if (act_val > act_limit) {
      out_of_memory = true;
      std::string tier = (config.hbf_config.num_hbm_stacks > 0) ? "HBM" : "Logic SRAM";
      if (config.exit_out_of_memory) {
        fail("Activations exceed " + tier + " capacity: " +
             std::to_string(act_val / 1e6) + " MB > " +
             std::to_string(act_limit / 1e6) + " MB");
      }
    }
  }

  // ---- Part E: optimizer drift harness (footprint) --------------------------
  // Compare the optimizer's predictions against the simulator's ground truth.
  // Non-circular: weight (at(1)) and KV (at(2)) use recorded tensors; act uses
  // recorded at(0)/num_layers for non-MLA (non-circular) or the analytic formula
  // for MLA (partially circular — still catches formula-vs-formula drift).
  if (pred_weight_bytes >= 0 && config.validate_optimizer != "off") {
    double eps = 1.0;  // 1 byte guard against division by zero
    double thresh = config.validate_optimizer_threshold;

    auto check_field = [&](const std::string& name, double pred, double actual) {
      double div = std::abs(pred - actual) / std::max(actual, eps);
      if (div > thresh) {
        std::string msg = "[OptValidation] " + name +
                          ": pred=" + std::to_string(pred / 1e9) + "GB" +
                          " actual=" + std::to_string(actual / 1e9) + "GB" +
                          " div=" + std::to_string(div * 100.0) + "% WARN";
        if (config.validate_optimizer == "strict") {
          fail(msg);
        } else {
          std::cout << msg << std::endl;
        }
      }
    };

    double actual_weight = size_vector.at(1);
    double actual_kv     = size_vector.at(2);
    double actual_total  = size;

    check_field("weight", pred_weight_bytes, actual_weight);
    check_field("kv",     pred_kv_bytes,     actual_kv);

    if (device->model_config.q_lora_rank != 0) {
      // MLA: compare analytic activation_size on both sides (partially circular —
      // catches formula-vs-formula drift between optimizer and cluster).
      check_field("act", pred_act_bytes, (double)activation_size);
      // Total: include analytic act on both sides (consistent with size = activation_size + wgt + kv).
      check_field("total", pred_total_bytes, actual_total);
    } else {
      // Non-MLA: at(0)/num_layers is the max-batch cumulative activation sum (graph built with
      // num_max_batched_token/batch_size_per_dp tokens/sequence).  The optimizer predicts the
      // per-decode-step peak activation (1 token/sequence) — fundamentally different metrics.
      // Comparing them produces a spurious WARN (~100%), so we skip the act comparison.
      // For total: exclude the act term from both sides so the weight+KV signal isn't masked.
      double pred_wgt_kv  = pred_weight_bytes + pred_kv_bytes;
      double actual_wgt_kv = actual_weight + actual_kv;
      check_field("total(wgt+kv)", pred_wgt_kv, actual_wgt_kv);
    }
  }
  // Capacity gate (F8 fix): for flash systems, activation already has its own scarce-tier
  // gate above (Part C -- HBM stack or logic SRAM). Charging it AGAIN here against the
  // flash weight+KV pool double-counts it and diverges from the optimizer's own
  // checkCapacity() (footprint.h), which never sums activation into the flash-pool
  // comparison. For plain HBM systems (no scarce tier), this lumped act+weight+kv check
  // is the sole legitimate capacity gate, so activation must stay included there.
  // Mirrors footprint.h checkCapacity()'s flash_cap: the reserved HBM stack(s) back the
  // activation tier above, not weight+KV, so they must come out of the flash pool here
  // too (0 for HBF+/CONV+ where num_hbm_stacks==0; also 0 for HBM4, which never reaches
  // this branch since hasScarceTier() is false there).
  double capacity_limit = config.memory_capacity;
  if (hasScarceTier(config)) {
    capacity_limit -= (double)config.hbf_config.num_hbm_stacks *
                      (double)config.hbf_config.hbm_per_stack_bytes;
  }
  double size_for_capacity_gate = hasScarceTier(config) ? (size - activation_size) : size;
  if (size_for_capacity_gate > capacity_limit) {
    out_of_memory = true;
    // Descriptive marker for the sweep's bound_reason classifier
    // (run_experiments.py::classify_failure). Without it, a forced-distribution
    // run that trips THIS gate only ever prints the generic "Out of Memory:"
    // line (test.cpp), which classifies as "unknown" instead of a capacity
    // bound. Wording matches classify_failure's expected substrings.
    std::cout << (hasScarceTier(config) ? "Flash capacity exceeded"
                                        : "HBM capacity exceeded")
              << ": weight+kv " << size_for_capacity_gate / 1073741824.0 << "GiB > "
              << capacity_limit / 1073741824.0 << "GiB" << std::endl;
    if (config.exit_out_of_memory) {
      return true;
    } else if (config.mem_cap_limit == true){
      long long kv_cache_size_per_seq = 0;
      if((device->model_config.qk_rope_head_dim != 0) && (device->model_config.compressed_kv == true)){
        kv_cache_size_per_seq = 1.0 * 
          (device->model_config.input_len + device->model_config.output_len) *
          (device->model_config.kv_lora_rank + device->model_config.qk_rope_head_dim) *
          device->model_config.num_layers * device->model_config.precision_byte;
      }
      else{
        // effectiveKvLenSumAllLayers() replaces the raw "(input+output) * num_layers"
        // factor so iRoPE local layers (attn_chunk_size>0) are capped at their bounded
        // window here too, matching the timing/capacity KV sizing elsewhere
        // (parallelism_optimizer.cpp / footprint.h). Reduces exactly to the old
        // expression when attn_chunk_size==0.
        kv_cache_size_per_seq =
            2.0 *
            effectiveKvLenSumAllLayers(device->model_config,
                device->model_config.input_len + device->model_config.output_len) *
            device->model_config.head_dim *
            device->model_config.num_kv_heads / device->model_config.ne_tp_dg *
            device->model_config.precision_byte;
      }
      hw_metric avail_capacity = 0;
      if(device->model_config.q_lora_rank == 0){
        avail_capacity =
            capacity_limit -
            (size_vector.at(0) / device->model_config.num_layers) -
            size_vector.at(1);
      }
      else{
        avail_capacity =
            capacity_limit - activation_size - size_vector.at(1);
      }

      if (avail_capacity < 0) {
        fail("Memory capacity is smaller than model weight");
      }
      std::cout << "Available capacity for KV cache is "
                << avail_capacity / 1073741824.0 << "GiB" << std::endl;
      std::cout << "KV cache per seq is "
                << kv_cache_size_per_seq / 1073741824.0 << "GiB" << std::endl;
      int max_batch_size =
          (int)(avail_capacity / kv_cache_size_per_seq) * scheduler->dp_degree;
      std::cout << "Modify max_batch_size to " << max_batch_size - 1
                << std::endl;
      scheduler->total_batch_size = max_batch_size - 1;
      scheduler->batch_size_per_dp =
          (max_batch_size - 1) / scheduler->dp_degree;
      scheduler->clear();
      scheduler->initRunningQueue();
      return false;
    }
    else{
      scheduler->clear();
      scheduler->initRunningQueue();
      return false;
    }
  }
  return false;
}

// AUDIT (BUGS.md #6): grep-confirmed zero call sites anywhere in this tree --
// this function is dead code, not merely "unclear if used." Its capacity math
// (hardcoded "3.3 GB Non MoE weight" magic-number subtraction, no activation
// term in the `size` OOM check) diverges from checkMemorySize()'s scarce-tier
// gate and should not be trusted if it's ever wired back up. Left in place
// (not deleted) pending an explicit decision on whether it's still needed --
// see BUGS_FIXES.md.
bool Cluster::checkHeteroMemorySize() {
  Device::Ptr device = get_device(0);
  auto module = module_map.at(0).at("::LLM");
  auto size_vector = module->get_size();

  std::cout << "ACT: "
            << size_vector.at(0) / 1024.0 / 1024 / 1024 /
                   device->model_config.num_layers
            << "GB, Weight: " << size_vector.at(1) / 1024.0 / 1024 / 1024
            << "GB, Cache: " << size_vector.at(2) / 1024.0 / 1024 / 1024 << "GB"
            << std::endl;
  double size = size_vector.at(1) + size_vector.at(2) - 3.3 * 1024.0 * 1024.0 * 1024.0 /
                device->model_config.ne_tp_dg; // Non MoE weight
  std::cout << "Total: " << size / 1024.0 / 1024 / 1024  << "GB" << std::endl;
  if (size > config.memory_capacity) {
    if (config.exit_out_of_memory) {
      return true;
    } else {
      long kv_cache_size_per_seq =
          2 *
          (device->model_config.input_len + device->model_config.output_len) *
          device->model_config.num_layers * device->model_config.head_dim *
          device->model_config.num_kv_heads / device->model_config.ne_tp_dg *
          device->model_config.precision_byte;

      hw_metric avail_capacity = config.memory_capacity - (size_vector.at(0) / device->model_config.num_layers) -
        size_vector.at(1);
      if (avail_capacity < 0) {
        fail("Memory capacity is smaller than model weight");
      }
      std::cout << "Available capacity for KV cache is "
                << avail_capacity / 1024.0 / 1024 / 1024 << "GB" << std::endl;
      std::cout << "KV cache per seq is "
                << kv_cache_size_per_seq / 1024.0 / 1024 / 1024 << "GB" << std::endl;
      int max_batch_size =
          (int)(avail_capacity / kv_cache_size_per_seq) * scheduler->dp_degree;
      std::cout << "Modify max_batch_size to " << max_batch_size - 1
                << std::endl;
      scheduler->total_batch_size = max_batch_size - 1;
      scheduler->batch_size_per_dp =
          (max_batch_size - 1) / scheduler->dp_degree;
      scheduler->clear();
      scheduler->initRunningQueue();
      return false;
    }
  }
  return false;
}

std::vector<energy_nJ> Cluster::getTotalEnergy() {
  std::vector<energy_nJ> total_energy = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
    Device::Ptr device = get_device(device_rank);
    std::vector<energy_nJ> device_energy =
        device->top_module_graph->getDeviceEnergy();
    for (int e_idx = 0; e_idx < total_energy.size(); e_idx++) {
      total_energy[e_idx] += device_energy[e_idx];
    }
  }
  return total_energy;
}

bool Cluster::check_module_graph_remain() {
  for (Node::Ptr _node : node) {
    if (_node->check_module_graph_remain()) {
      return true;
    }
  }
  return false;
}

void Cluster::exportToCSV(std::ofstream &csv, std::vector<Stat> &stat_list) {
  for (auto temp : stat_list) {
    csv << std::to_string(temp.iter_info) << "," << std::to_string(temp.split)
        << "," << temp.type << "," << std::to_string(temp.time) << ","
        << std::to_string(temp.latency) << ","
        << std::to_string(temp.queueing_delay) << ","
        << std::to_string(temp.arrival_time) << ","
        << std::to_string(temp.seq_queue_size) << ","
        << std::to_string(temp.input_len) << ","
        << std::to_string(temp.output_len) << ","
        << std::to_string(temp.num_sum_iter) << ","
        << std::to_string(temp.is_mixed) << ","
        << std::to_string(temp.batchsize) << ","
        << std::to_string(temp.process_token) << ","
        << std::to_string(temp.sum_seq) << "," << std::to_string(temp.gen_seq)
        << "," << std::to_string(temp.average_seq_len) << ","
        << std::to_string(temp.sum_attention_opb) << ","
        << std::to_string(temp.qkv_gen) << "," 
        << std::to_string(temp.q_down_proj) << "," 
        << std::to_string(temp.kv_down_proj) << ","
        << std::to_string(temp.kr_proj) << ","
        << std::to_string(temp.q_up_proj) << ","
        << std::to_string(temp.qr_proj) << ","
        << std::to_string(temp.kv_up_proj) << ","
        << std::to_string(temp.tr_k_up_proj) << ","
        << std::to_string(temp.v_up_proj) << ","
        << std::to_string(temp.atten_sum)
        << "," << std::to_string(temp.atten_gen) << ","
        << std::to_string(temp.o_proj) << "," << std::to_string(temp.ffn) << ","
        << std::to_string(temp.expert_ffn) << ","
        << std::to_string(temp.communication) << ","
        << std::to_string(temp.kv_write) << ","
        << std::to_string(temp.rope) << ","
        << std::to_string(temp.layernorm) << ","
        << std::to_string(temp.residual) << ","
        << std::to_string(temp.lm_head) << ","
        << std::to_string(temp.act_energy) << ","
        << std::to_string(temp.read_energy) << ","
        << std::to_string(temp.write_energy) << ","
        << std::to_string(temp.all_act_energy) << ","
        << std::to_string(temp.all_read_energy) << ","
        << std::to_string(temp.all_write_energy) << ","
        << std::to_string(temp.mac_energy) << ","
        << std::to_string(temp.total_energy) << ","
        << std::to_string(temp.FC_DRAM_energy) << ","
        << std::to_string(temp.FC_COMP_energy) << ","
        << std::to_string(temp.Attn_DRAM_energy) << ","
        << std::to_string(temp.Attn_COMP_energy) << ","
        << std::to_string(temp.MoE_DRAM_energy) << ","
        << std::to_string(temp.MoE_COMP_energy) << ","
        << std::to_string(temp.isOOM) << std::endl;
  }
  stat_list.resize(0);
}

std::vector<Stat> Cluster::runIteration(int iter, std::string file_name) {
  std::ofstream csv;
  csv.open(file_name);

  csv << "iter_info,split,type,time,latency,queueing_delay,arrival_time,seq_queue_"
         "size,"
         "input_len,output_len,num_sum_iter,mixed,batchsize,numtoken,num_sum_"
         "seq,num_gen_seq,seqlen,sum_attention_opb,qkvgen,q_down_proj,kv_down_proj,kr_proj,"
         "q_up_proj,qr_proj,kv_up_proj,tr_k_up_proj,v_up_proj,atten_sum,atten_gen,"
         "o_proj,ffn,expert_ffn,communication,kv_write,rope,layernorm,residual,lm_head,act_energy,read_energy,write_"
         "energy,all_act_energy,all_read_energy,all_write_energy,mac_energy,"
         "total_energy,fc_dram,fc_comp,attn_dram,attn_comp,moe_dram,moe_comp,OOM"
      << std::endl;

  std::vector<Stat> stat_list;

  scheduler->fillSequenceQueue();
  scheduler->fillRunningQueue();

  // Initial dummy population is done: subsequent pushDummySeq calls (steady-state
  // refills as sequences complete) must seed at start-of-generation, not the
  // staggered initial-fill offset -- see pushDummySeq's decode_mode branch.
  scheduler->initial_fill_done = true;

  // hitting
  scheduler->hittingQueue(10000);

  if (config.disagg_system) {
    stat_list = runIterationSumGenSplit(iter, csv);
  } else {
    stat_list = runIterationMixed(iter, csv);
  }

  std::cout << "Total: " << std::to_string(scheduler->total_time) << std::endl;
  std::cout << file_name << std::endl;
  csv.close();

  return stat_list;
}

std::vector<Stat> Cluster::runIterationMixed(int iter, std::ofstream &csv) {
  time_ns total_time = 0;

  std::vector<Stat> stat_list;
  bool is_after_sum = false;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iter; i++) {
    // export to csv, you can modify the frequency of export_to_csv by changing the number. now it is 1
    if (i % 1 == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - start);
      start = std::chrono::high_resolution_clock::now();
      exportToCSV(csv, stat_list);
    }

    auto metadata = scheduler->setMetadata();
    run(metadata);
    // maxDeviceTime(), not get_device(0): with pipeline parallelism, only the
    // slowest-finishing device's status.device_time reflects the true, fully
    // propagated per-token latency across all pp stages -- see maxDeviceTime()'s
    // doc comment (cluster.h) and PipelineStage::forward (communication.cpp).
    time_ns time = maxDeviceTime();

    // if no reqeusts, add time
    if (scheduler->getNumProcessToken() == 0) {
      time = 20 * 1000 * 1000;
      continue;
    }

    // NOTE: unconditionally added regardless of whether this iteration processed
    // "sum" (prefill) or "gen" (decode) sequences -- unlike runIterationSumGenSplit,
    // which isolates prefill time into a separate sum_machine_time accumulator. This
    // is safe ONLY because decode_mode=on (config.yaml) guarantees every sequence is
    // synthesized already fully prefilled (current_len == input_len at creation, see
    // scheduler.cpp's pushDummySeq), so hasSumSeq() is always false here and no
    // prefill time is ever actually accumulated. If decode_mode were ever disabled
    // (without disagg_system=on), this leaks prefill compute time into the reported
    // decode-only TPOT (BUGS.md #2) -- warn once so the contamination isn't silent.
    if (!config.disagg_system && scheduler->hasSumSeq()) {
      static bool warned_prefill_contamination = false;
      if (!warned_prefill_contamination) {
        std::cerr << "WARNING: runIterationMixed is accumulating a \"sum\" "
                     "(prefill) iteration into the decode-only total_time -- "
                     "decode_mode should be on unless disagg_system is also on "
                     "(see BUGS.md #2)."
                  << std::endl;
        warned_prefill_contamination = true;
      }
    }
    total_time += time;

    Stat stat;
    stat.iter_info = 1;
    stat.type = "t2t";
    stat.time = total_time;
    scheduler->total_time = total_time;
    if (config.disagg_system) {
      stat.split = 1;
    }

    // power
    std::vector<energy_nJ> total_energy = getTotalEnergy();
    stat.act_energy = total_energy[0];
    stat.read_energy = total_energy[1];
    stat.write_energy = total_energy[2];
    stat.all_act_energy = total_energy[3];
    stat.all_read_energy = total_energy[4];
    stat.all_write_energy = total_energy[5];
    stat.mac_energy = total_energy[6];
    stat.total_energy = total_energy[7];
    stat.seq_queue_size = scheduler->sequence_queue.size();

    setStat(stat);
    setTimeBreakDown(stat);

    // tokens which generated first token or eos token
    std::vector<Sequence::Ptr> token_list;

    stat_list.push_back(stat);

    token_list = scheduler->updateScheduler(time);
    addLatency(stat_list, token_list, total_time);

    scheduler->fillSequenceQueue(time, total_time);
    scheduler->fillRunningQueue();
  }

  return stat_list;
}

std::vector<Stat> Cluster::runIterationSumGenSplit(int iter,
                                                   std::ofstream &csv) {
  time_ns total_time = 0;

  time_ns sum_machine_time = 0;

  std::vector<Stat> stat_list;
  std::vector<Sequence::Ptr> token_list;

  time_ns gen_start_time = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iter; i++) {
    // export to csv
    if (i % 25 == 24) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::high_resolution_clock::now() - start);
      start = std::chrono::high_resolution_clock::now();
      exportToCSV(csv, stat_list);
    }

    auto metadata = scheduler->setMetadata();
    run(metadata);
    // maxDeviceTime(), not get_device(0) -- see the identical comment in
    // runIterationMixed above / maxDeviceTime()'s doc comment in cluster.h.
    time_ns time = maxDeviceTime();

    // if no reqeusts, add time
    if (scheduler->getNumProcessToken() == 0) {
      time = 20 * 1000 * 1000;
      continue;
    }

    // gen machine
    if (!scheduler->hasSumSeq()) {
      total_time += time;

      Stat stat;
      stat.iter_info = 1;
      stat.type = "t2t";
      stat.time = total_time;
      scheduler->total_time = total_time;

      // power
      std::vector<energy_nJ> total_energy = getTotalEnergy();
      stat.act_energy = total_energy[0];
      stat.read_energy = total_energy[1];
      stat.write_energy = total_energy[2];
      stat.all_act_energy = total_energy[3];
      stat.all_read_energy = total_energy[4];
      stat.all_write_energy = total_energy[5];
      stat.mac_energy = total_energy[6];
      stat.total_energy = total_energy[7];
      stat.seq_queue_size = scheduler->sequence_queue.size();

      setStat(stat);
      setTimeBreakDown(stat);

      stat_list.push_back(stat);
      token_list = scheduler->updateScheduler(time);
      addLatency(stat_list, token_list, total_time);

      scheduler->fillSequenceQueue(time, total_time);
      scheduler->fillRunningQueue(sum_machine_time);
    }
    // sum machine
    else {
      Stat stat;
      stat.iter_info = 1;
      stat.type = "sum";
      stat.time = std::max(total_time, sum_machine_time) + time;
      stat.latency = time;

      // Populate energy/batch/breakdown fields like the gen branch above.
      // setStat() leaves stat.latency untouched on the disagg+hasSumSeq() path
      // (see its own branching), so this does not override the value set above.
      std::vector<energy_nJ> total_energy = getTotalEnergy();
      stat.act_energy = total_energy[0];
      stat.read_energy = total_energy[1];
      stat.write_energy = total_energy[2];
      stat.all_act_energy = total_energy[3];
      stat.all_read_energy = total_energy[4];
      stat.all_write_energy = total_energy[5];
      stat.mac_energy = total_energy[6];
      stat.total_energy = total_energy[7];
      stat.seq_queue_size = scheduler->sequence_queue.size();

      setStat(stat);
      setTimeBreakDown(stat);

      stat_list.push_back(stat);

      sum_machine_time = stat.time;
      // tokens which generated first token or eos token
      token_list = scheduler->updateSchedulerSumGenSplit(time);
      addLatency(stat_list, token_list, stat.time);
    }
  }

  return stat_list;
}

void Cluster::addLatency(std::vector<Stat> &stat_list,
                         const std::vector<Sequence::Ptr> &seq_list,
                         time_ns time) {
  for (auto &seq : seq_list) {
    seq->gen_start_time = time;
    if (seq->first_token_time == 0.0 || seq->arrival_time == 0.0) {
      continue;
    }
    Stat stat;
    stat.iter_info = 0;
    stat.time = time;
    // end token
    if (seq->current_len == seq->total_len) {
      stat.type = "e2e";
      stat.latency = seq->end_token_time;
      stat.input_len = seq->input_len;
      stat.output_len = seq->output_len;
      stat.num_sum_iter = seq->num_sum_iter;
      stat.queueing_delay = seq->queueing_delay;
      stat.arrival_time = seq->arrival_time;
    } else if (seq->current_len == seq->input_len) {
      stat.type = "t2ft";
      stat.latency = seq->first_token_time;
      stat.input_len = seq->input_len;
      stat.num_sum_iter = seq->num_sum_iter;
      stat.queueing_delay = seq->queueing_delay;
      stat.arrival_time = seq->arrival_time;
    }
    stat_list.push_back(stat);
  }
}

void Cluster::exportGantt(std::string gantt_file_path) {
  std::filesystem::path dir = gantt_file_path;
  std::filesystem::create_directories(dir);

  if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir)) {
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
      std::filesystem::remove_all(entry);
    }
  } else {
    std::cerr << "Error: Directory does not exist.\n";
  }

  for (int i = 0; i < num_total_device; i++) {
    TopModuleGraph::Ptr top = get_device(i)->top_module_graph;
    top->exportGantt(gantt_file_path, i);
  }
}
void Cluster::setStat(Stat &stat) {
  // maxDeviceTime(), not get_device(0) -- see cluster.h's doc comment. Kept as
  // an independent read (mirroring the caller's own maxDeviceTime() call) since
  // this function only ever reads device_time right after the same run(), so
  // the two calls agree exactly.
  time_ns time = maxDeviceTime();

  stat.batchsize = scheduler->getBatchSize();
  stat.average_seq_len = scheduler->getAverageSeqlen();
  stat.process_token = scheduler->getNumProcessToken();
  stat.sum_seq = scheduler->getSumSize();
  stat.gen_seq = scheduler->getGenSize();

  if (!config.disagg_system) {
    if (scheduler->hasSumSeq()) {
      stat.latency = time;
      stat.is_mixed = 1;

    } else {
      stat.latency = time;
      stat.is_mixed = 0;
    }
  } else {
    if (!scheduler->hasSumSeq()) {
      stat.latency = time;
      stat.is_mixed = 0;
    }
  }
}

void Cluster::setTimeBreakDown(Stat &stat) {
  TimeBoard &timeboard = get_device(0)->top_module_graph->timeboard;

  if(scheduler->model_config.qk_nope_head_dim == 0){
    std::vector<TimeStamp *> QKV_gen;    // GPU
    std::vector<TimeStamp *> AttnSum;    // GPU
    std::vector<TimeStamp *> AttnGen;    // PIM or Logic
    std::vector<TimeStamp *> O_proj;     // GPU
    std::vector<TimeStamp *> FFN;        // PIM or Logic
    std::vector<TimeStamp *> ExpertFFN;  // PIM or Logic
    std::vector<TimeStamp *> Comm;       // PIM or Logic
    std::vector<TimeStamp *> CommInExpertFFN;

    std::vector<TimeStamp *> RoPE;
    std::vector<TimeStamp *> LayerNorm;
    std::vector<TimeStamp *> Residual;
    std::vector<TimeStamp *> LmHead;

    timeboard.find_stamp("attn_qkv_proj", QKV_gen);
    timeboard.find_stamp("AttentionSum", AttnSum);
    timeboard.find_stamp("AttentionGen", AttnGen);
    timeboard.find_stamp("attn_o_proj", O_proj);
    timeboard.find_stamp("feedforward", FFN);
    timeboard.find_stamp("expertFFN", ExpertFFN);
    timeboard.find_stamp("moe_scatter", CommInExpertFFN);
    timeboard.find_stamp("moe_all_reduce_for_e_tp", CommInExpertFFN);
    timeboard.find_stamp("moe_all_reduce_for_gather", CommInExpertFFN);
    timeboard.find_stamp("moe_gather", CommInExpertFFN);
    timeboard.find_stamp("all_reduce", Comm);
    timeboard.find_stamp("moe_scatter", Comm);
    timeboard.find_stamp("moe_gather", Comm);

    timeboard.find_stamp("k_rope", RoPE);
    timeboard.find_stamp("q_rope", RoPE);

    timeboard.find_stamp("input_layer_norm", LayerNorm);
    timeboard.find_stamp("post_attn_layer_norm", LayerNorm);
    
    timeboard.find_stamp("residual_1", Residual);
    timeboard.find_stamp("residual_2", Residual);
    timeboard.find_stamp("lm_head", LmHead);

    time_ns qkv_gen = 0;
    time_ns atten_sum = 0;
    time_ns atten_gen = 0;
    time_ns o_proj = 0;
    time_ns ffn = 0;
    time_ns expert_ffn = 0;
    time_ns comm_in_expert_ffn = 0;
    time_ns communication = 0;

    time_ns rope = 0;
    time_ns layernorm = 0;
    time_ns residual = 0;
    time_ns lm_head_time = 0;

    energy_nJ FC_DRAM = 0;
    energy_nJ FC_COMP = 0;
    energy_nJ MoE_DRAM = 0;
    energy_nJ MoE_COMP = 0;
    energy_nJ Attn_DRAM = 0;
    energy_nJ Attn_COMP = 0;

    for (auto stamp : QKV_gen) {
      qkv_gen += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * num_total_device;
      FC_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.qkv_gen = qkv_gen;

    time_ns kv_write = 0;
    for (auto stamp : AttnSum) {
      time_ns stamp_dur = stamp->get_duration();
      time_ns stamp_kv = stamp->get_kv_write();
      atten_sum += (stamp_dur - stamp_kv);
      kv_write += stamp_kv;
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.atten_sum = atten_sum;

    for (auto stamp : AttnGen) {
      time_ns stamp_dur = stamp->get_duration();
      time_ns stamp_kv = stamp->get_kv_write();
      atten_gen += (stamp_dur - stamp_kv);
      kv_write += stamp_kv;
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.atten_gen = atten_gen;
    stat.kv_write = kv_write;

    for (auto stamp : O_proj) {
      o_proj += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * num_total_device;
      FC_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.o_proj = o_proj;

    for (auto stamp : FFN) {
      ffn += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * num_total_device;
      FC_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.ffn = ffn;

    for (auto stamp : ExpertFFN) {
      expert_ffn += stamp->get_duration();
    }

    // expertFFN may have different energy by device
    for(int device_id = 0; device_id < num_total_device; device_id ++){
      TimeBoard &timeboard_temp = get_device(device_id)->top_module_graph->timeboard;
      std::vector<TimeStamp *> ExpertFFN_temp;  // PIM or Logic
      timeboard_temp.find_stamp("expertFFN", ExpertFFN_temp);
      for (auto stamp : ExpertFFN_temp) {
        MoE_DRAM += stamp->getDramEnergy();
        MoE_COMP += stamp->getCompEnergy();
      }
    }

    for (auto stamp : CommInExpertFFN) {
      comm_in_expert_ffn += stamp->get_duration();
    }

    stat.expert_ffn = expert_ffn - comm_in_expert_ffn;

    for (auto stamp : Comm) {
      communication += stamp->get_duration();
    }
    stat.communication = communication;

    for (auto stamp : RoPE) {
      rope += stamp->get_duration();
    }
    stat.rope = rope;

    for (auto stamp : LayerNorm) {
      layernorm += stamp->get_duration();
    }
    stat.layernorm = layernorm;

    for (auto stamp : Residual) {
      residual += stamp->get_duration();
    }
    stat.residual = residual;

    for (auto stamp : LmHead) {
      lm_head_time += stamp->get_duration();
    }
    stat.lm_head = lm_head_time;

    stat.FC_DRAM_energy = FC_DRAM;
    stat.FC_COMP_energy = FC_COMP;
    stat.Attn_DRAM_energy = Attn_DRAM;
    stat.Attn_COMP_energy = Attn_COMP;
    stat.MoE_DRAM_energy = MoE_DRAM;
    stat.MoE_COMP_energy = MoE_COMP;
    stat.isOOM = out_of_memory;

    double opb = 0;
    for (auto stamp : AttnSum) {
      opb += stamp->getOpb();
    }

    if (AttnSum.size()) {
      opb /= AttnSum.size();
    }
    stat.sum_attention_opb = opb;
  }
  else{ // if Use MLA
    std::vector<TimeStamp *> Decoders;    
    std::vector<TimeStamp *> Q_down;    
    std::vector<TimeStamp *> KV_down;    
    std::vector<TimeStamp *> KR_proj;    

    std::vector<TimeStamp *> RoPE;
    std::vector<TimeStamp *> LayerNorm;
    std::vector<TimeStamp *> Residual;

    std::vector<TimeStamp *> Q_up;    
    std::vector<TimeStamp *> QR_proj;    
    std::vector<TimeStamp *> KV_up;

    // for Absorb Impl //
    std::vector<TimeStamp *> tr_K_up;
    std::vector<TimeStamp *> V_up;
    
    std::vector<TimeStamp *> AttnSum;    
    std::vector<TimeStamp *> AttnGen;    

    std::vector<TimeStamp *> O_proj;     

    std::vector<TimeStamp *> FFN;        
    std::vector<TimeStamp *> ExpertFFN;  
    std::vector<TimeStamp *> Comm;
    std::vector<TimeStamp *> CommInExpertFFN;
    std::vector<TimeStamp *> Test;
    std::vector<TimeStamp *> LmHead;

    timeboard.find_stamp("attn_q_down_proj", Q_down);
    timeboard.find_stamp("attn_kv_down_proj", KV_down);
    timeboard.find_stamp("attn_kr_proj", KR_proj);

    timeboard.find_stamp("attn_q_up_proj", Q_up);
    timeboard.find_stamp("attn_qr_proj", QR_proj);
    timeboard.find_stamp("attn_kv_up_proj", KV_up);

    // for MLA absorb //
    timeboard.find_stamp("attn_tr_k_up_proj", tr_K_up);
    timeboard.find_stamp("attn_v_up_proj", V_up);

    timeboard.find_stamp("AttentionSum", AttnSum);
    timeboard.find_stamp("AttentionGen", AttnGen);

    timeboard.find_stamp("attn_o_proj", O_proj);

    timeboard.find_stamp("feedforward", FFN);
    timeboard.find_stamp("expertFFN", ExpertFFN);
    timeboard.find_stamp("moe_scatter", CommInExpertFFN);
    timeboard.find_stamp("moe_all_reduce_for_e_tp", CommInExpertFFN);
    timeboard.find_stamp("moe_all_reduce_for_gather", CommInExpertFFN);
    timeboard.find_stamp("moe_gather", CommInExpertFFN);
    timeboard.find_stamp("all_reduce", Comm);
    timeboard.find_stamp("moe_scatter", Comm);
    timeboard.find_stamp("moe_gather", Comm);

    timeboard.find_stamp("k_rope", RoPE);
    timeboard.find_stamp("q_rope", RoPE);

    timeboard.find_stamp("input_layer_norm", LayerNorm);
    timeboard.find_stamp("latent_q_layer_norm", LayerNorm);
    timeboard.find_stamp("latent_kv_layer_norm", LayerNorm);
    timeboard.find_stamp("post_attn_layer_norm", LayerNorm);
    
    timeboard.find_stamp("residual_1", Residual);
    timeboard.find_stamp("residual_2", Residual);
    timeboard.find_stamp("lm_head", LmHead);

    timeboard.find_stamp("decoder_", Decoders);

    time_ns q_down_proj = 0;
    time_ns kv_down_proj = 0;
    time_ns kr_proj = 0;

    time_ns q_up_proj = 0;
    time_ns qr_proj = 0;
    time_ns kv_up_proj = 0;

    // for MLA absorb //
    time_ns tr_k_up_proj = 0;
    time_ns v_up_proj = 0;
    // 

    time_ns atten_sum = 0;
    time_ns atten_gen = 0;
    time_ns o_proj = 0;
    time_ns ffn = 0;
    time_ns expert_ffn = 0;
    time_ns comm_in_expert_ffn = 0;
    time_ns communication = 0;
    
    time_ns rope = 0;
    time_ns layernorm = 0;
    time_ns residual = 0;
    time_ns lm_head_time = 0;

    energy_nJ FC_DRAM = 0;
    energy_nJ FC_COMP = 0;
    energy_nJ MoE_DRAM = 0;
    energy_nJ MoE_COMP = 0;
    energy_nJ Attn_DRAM = 0;
    energy_nJ Attn_COMP = 0;

    for (auto stamp : Q_down){
      q_down_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.q_down_proj = q_down_proj;

    for (auto stamp : KV_down){
      kv_down_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.kv_down_proj = kv_down_proj;

    for (auto stamp : KR_proj){
      kr_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.kr_proj = kr_proj;

    for (auto stamp : Q_up){
      q_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.q_up_proj = q_up_proj;

    for (auto stamp : QR_proj){
      qr_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.qr_proj = qr_proj;

    for (auto stamp : KV_up){
      kv_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.kv_up_proj = kv_up_proj;

    for (auto stamp : tr_K_up){
      tr_k_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.tr_k_up_proj = tr_k_up_proj;

    for (auto stamp : V_up){
      v_up_proj += stamp->get_duration();
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.v_up_proj = v_up_proj;

    time_ns kv_write = 0;
    for (auto stamp : AttnSum) {
      time_ns stamp_dur = stamp->get_duration();
      time_ns stamp_kv = stamp->get_kv_write();
      atten_sum += (stamp_dur - stamp_kv);
      kv_write += stamp_kv;
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.atten_sum = atten_sum;

    for (auto stamp : AttnGen) {
      time_ns stamp_dur = stamp->get_duration();
      time_ns stamp_kv = stamp->get_kv_write();
      atten_gen += (stamp_dur - stamp_kv);
      kv_write += stamp_kv;
      Attn_DRAM += stamp->getDramEnergy() * num_total_device;
      Attn_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.atten_gen = atten_gen;
    stat.kv_write = kv_write;

    for (auto stamp : O_proj) {
      o_proj += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * num_total_device;
      FC_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.o_proj = o_proj;

    for (auto stamp : FFN) {
      ffn += stamp->get_duration();
      FC_DRAM += stamp->getDramEnergy() * num_total_device;
      FC_COMP += stamp->getCompEnergy() * num_total_device;
    }
    stat.ffn = ffn;

    for (auto stamp : ExpertFFN) {
      expert_ffn += stamp->get_duration();
    }

    // expertFFN may have different energy by device
    for(int device_id = 0; device_id < num_total_device; device_id ++){
      TimeBoard &timeboard_temp = get_device(device_id)->top_module_graph->timeboard;
      std::vector<TimeStamp *> ExpertFFN_temp;  // PIM or Logic
      timeboard_temp.find_stamp("expertFFN", ExpertFFN_temp);
      for (auto stamp : ExpertFFN_temp) {
        MoE_DRAM += stamp->getDramEnergy();
        MoE_COMP += stamp->getCompEnergy();
      }
    }

    for (auto stamp : CommInExpertFFN) {
      comm_in_expert_ffn += stamp->get_duration();
    }

    stat.expert_ffn = expert_ffn - comm_in_expert_ffn;

    for (auto stamp : Comm) {
      communication += stamp->get_duration();
    }
    stat.communication = communication;

    for (auto stamp : RoPE) {
      rope += stamp->get_duration();
    }
    stat.rope = rope;

    for (auto stamp : LayerNorm) {
      layernorm += stamp->get_duration();
    }
    stat.layernorm = layernorm;

    for (auto stamp : Residual) {
      residual += stamp->get_duration();
    }
    stat.residual = residual;

    for (auto stamp : LmHead) {
      lm_head_time += stamp->get_duration();
    }
    stat.lm_head = lm_head_time;

    stat.FC_DRAM_energy = FC_DRAM;
    stat.FC_COMP_energy = FC_COMP;
    stat.Attn_DRAM_energy = Attn_DRAM;
    stat.Attn_COMP_energy = Attn_COMP;
    stat.MoE_DRAM_energy = MoE_DRAM;
    stat.MoE_COMP_energy = MoE_COMP;
    stat.isOOM = out_of_memory;

    double opb = 0;
    for (auto stamp : AttnSum) {
      opb += stamp->getOpb();
    }

    if (AttnSum.size()) {
      opb /= AttnSum.size();
    }
    stat.sum_attention_opb = opb;
  }
}

void Cluster::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  setPerformExecution(true);
  restartModuleGraph();
  while (check_module_graph_remain()) {
    for (Node::Ptr _node : node) {
      _node->run(sequences_metadata_list);
    }
  }
}

void Cluster::setPerformExecution(bool perform) {
  for (Node::Ptr _node : node) {
    _node->setPerformExecution(perform);
  }
};

void Cluster::set(SystemConfig config) {
  CreateNode(config);
  module_map.resize(num_total_device);
}

void Cluster::CreateNode(SystemConfig config) {
  for (int node_rank = 0; node_rank < config.num_node; node_rank++) {
    node.push_back(Node::Create(config, node_rank, getptr()));
  }
}

};  // namespace llm_system