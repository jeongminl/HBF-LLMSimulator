#include "module/expert.h"

#include "model/util.h"
#include "module/base.h"
#include "module/communication.h"
#include "module/layer.h"
#include "module/route.h"

namespace llm_system {

// ExpertFFN

ExpertFFN::ExpertFFN(std::string& prefix, std::string& name,
                     const ModelConfig& model_config, Scheduler::Ptr scheduler,
                     std::vector<int>& device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int ne_tp_dg = model_config.ne_tp_dg;
  int e_tp_dg = model_config.e_tp_dg;
  int ne_tp_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;

  std::vector<int> non_moe_device_list;
  set_device_list(non_moe_device_list, ne_tp_offset, ne_tp_dg);

  assertTrue(model_config.num_routed_expert % non_moe_device_list.size() == 0,
             "Non-expert tensor parallel degree is not supported");
  assertTrue(model_config.num_routed_expert >= non_moe_device_list.size(),
             "Num_Expert is smaller than num_device");

  auto gate_fn = ColumnParallelLinear::Create(
      module_map_name, "gate_fn", model_config.hidden_dim,
      model_config.num_routed_expert, {device->device_total_rank}, device);
  add_module(gate_fn);

  auto gate_update =
      GateUpdate::Create(module_map_name, "gate_update", device_list, device);
  add_module(gate_update);

  auto moe_scatter =
      MoEScatter::Create(module_map_name, "moe_scatter", device_list, device);
  add_module(moe_scatter);

  int parallel_num = device_list.size();
  assertTrue(model_config.num_routed_expert % e_tp_dg == 0,
             "Expert tensor parallel degree is not supported");
  assertTrue(parallel_num % e_tp_dg == 0 && parallel_num >= e_tp_dg,
             "Expert tensor parallel degree is not supported");
  assertTrue((parallel_num / e_tp_dg) <= model_config.num_routed_expert,
             "Expert tensor parallel degree is not supported");

  int num_expert_tp_gr_rank = parallel_num / e_tp_dg;
  num_expert_per_device = model_config.num_routed_expert / (num_expert_tp_gr_rank);
  int device_rank = device->device_total_rank;
  int local_rank = 0;
  for (int idx = 0; idx < device_list.size(); idx++) {
    if (device_list[idx] == device_rank) {
      local_rank = idx;
      break;
    }
  }

  int device_offset = (local_rank / e_tp_dg) * e_tp_dg;
  expert_offset = num_expert_per_device * (local_rank / e_tp_dg);
  std::vector<int> expert_device_list;
  for (int idx = device_offset; idx < device_offset + e_tp_dg; idx++) {
    expert_device_list.push_back(device_list.at(idx));
  }

  // paper2 §IV first-activated-expert page-latency exposure: built BEFORE
  // moe_route (below) specifically so their Module::Ptrs exist to pass into
  // Route::Create -- Route::forward() is the module that actually gets
  // re-invoked every decode step (ExpertFFN itself is a plain, non-leaf
  // Module: its forward() runs exactly once, at module-graph CONSTRUCTION
  // time, to record the op sequence -- see module/module_graph.cpp's
  // TopModuleGraph::run(), which only re-calls forward() on
  // graph_execution==true LEAF modules like Route/Linear/Activation on every
  // replay). Route already computes this step's LIVE per-local-expert token
  // counts (expert_token_list, route.cpp) on every such replay, so it is the
  // correct (and only) place that can pick "this step's first activated
  // expert" with real per-step data -- see route.cpp's forward() for the arm
  // logic. local_expert_ffns is built in the SAME local order (index i ==
  // global expert expert_offset+i) that Route's expert_token_list uses, so
  // Route can index directly into it. Pure additive plumbing: this reorders
  // WHEN these Module objects get constructed, not the CALL order recorded
  // during the actual forward() graph-build pass below (gate_fn -> ...
  // -> route -> expert loop, unchanged) -- module_list is a std::map keyed by
  // name, so construction order has no effect on paper1 behavior.
  std::vector<Module::Ptr> local_expert_ffns;
  if (model_config.ffn_way == 2) {
    for (int expert_id = expert_offset;
         expert_id < expert_offset + num_expert_per_device; expert_id++) {
      auto expert_ffn = FeedForward2Way::Create(
          module_map_name, "expert_FFN_" + std::to_string(expert_id),
          model_config, scheduler, expert_device_list, device, false, true);
      add_module(expert_ffn);
      local_expert_ffns.push_back(expert_ffn);
    }
  } else if (model_config.ffn_way == 3) {
    for (int expert_id = expert_offset;
         expert_id < expert_offset + num_expert_per_device; expert_id++) {
      auto expert_ffn = FeedForward3Way::Create(
          module_map_name, "expert_FFN_" + std::to_string(expert_id),
          model_config, scheduler, expert_device_list, device, false, true);
      add_module(expert_ffn);
      local_expert_ffns.push_back(expert_ffn);
    }
  }

  auto moe_route =
      Route::Create(module_map_name, "moe_route", num_expert_per_device,
                    expert_offset, device_list, device, local_expert_ffns);
  add_module(moe_route);

  num_shared_expert = model_config.num_shared_expert;
  for(int shared_expert_idx = 0 ; shared_expert_idx < num_shared_expert; shared_expert_idx ++){
    if (model_config.ffn_way == 2) {
        auto shared_expert_ffn = FeedForward2Way::Create(
            module_map_name, "shared_expert_FFN_" + std::to_string(shared_expert_idx),
            model_config, scheduler, non_moe_device_list, device, false, true, true); // Shared Expert use TP degree of non-moe
        add_module(shared_expert_ffn);
    } else if (model_config.ffn_way == 3) {
        auto shared_expert_ffn = FeedForward3Way::Create(
            module_map_name, "shared_expert_FFN_" + std::to_string(shared_expert_idx),
            model_config, scheduler, non_moe_device_list, device, false, true, true); // Shared Expert use TP degree of non-moe
        add_module(shared_expert_ffn);
    }
  }

  auto all_reduce_for_e_tp = AllReduce::Create(module_map_name, "moe_all_reduce_for_e_tp",
                                      expert_device_list, device);               
  add_module(all_reduce_for_e_tp);

  auto moe_gather = MoEGather::Create(module_map_name, "moe_gather", device_list, device);
  add_module(moe_gather);

  auto all_reduce_for_gather = AllReduce::Create(module_map_name, "moe_all_reduce_for_gather",
    non_moe_device_list, device);      
  add_module(all_reduce_for_gather);

  auto sync_0 = Sync::Create(module_map_name, "sync_0", device_list, device);
  add_module(sync_0);

  auto sync_for_moe_scatter = Sync::Create(module_map_name, "sync_for_moe_scatter", device_list, device);
  add_module(sync_for_moe_scatter);

  auto sync_2 = Sync::Create(module_map_name, "sync_2", device_list, device);
  add_module(sync_2);

  auto sync_3 = Sync::Create(module_map_name, "sync_3", device_list, device);
  add_module(sync_3);

  auto sync_for_moe_gather = Sync::Create(module_map_name, "sync_for_moe_gather", device_list, device);
  add_module(sync_for_moe_gather);

  auto sync_5 = Sync::Create(module_map_name, "sync_5", device_list, device);
  add_module(sync_5);

  auto sync_6 = Sync::Create(module_map_name, "sync_6", device_list, device);
  add_module(sync_6);
}

Tensor::Ptr ExpertFFN::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr sync_0 = get_module("sync_0");
  Module::Ptr sync_for_moe_scatter = get_module("sync_for_moe_scatter");
  (*sync_0)(input, sequences_metadata);

  Module::Ptr gate_fn = get_module("gate_fn");
  Module::Ptr gate_update = get_module("gate_update");

  (*gate_fn)(input, sequences_metadata);  // gate scoring op (result unused; timed for scheduling)
  Tensor::Ptr gate_update_out = (*gate_update)(input, sequences_metadata);

  // MoE Scatter //
  Module::Ptr moe_scatter = get_module("moe_scatter");
  Module::Ptr route = get_module("moe_route");

  Tensor::Ptr scatter_out = (*moe_scatter)(gate_update_out, sequences_metadata);
  (*sync_for_moe_scatter)(input, sequences_metadata);

  TensorVec input_vec;
  input_vec.push_back(scatter_out);
  TensorVec route_out = (*route)(input_vec, sequences_metadata);

  // ExpertFFN //

  Tensor::Ptr result;

  Module::Ptr sync_2 = get_module("sync_2");
  Module::Ptr sync_3 = get_module("sync_3");
  Module::Ptr sync_for_moe_gather = get_module("sync_for_moe_gather");
  Module::Ptr sync_5 = get_module("sync_5");
  Module::Ptr sync_6 = get_module("sync_6");

  Module::Ptr all_reduce_for_e_tp = get_module("moe_all_reduce_for_e_tp");
  Module::Ptr all_reduce_for_gather = get_module("moe_all_reduce_for_gather");

  TensorVec expert_out;

  // paper2 §IV first-activated-expert page-latency exposure is armed inside
  // Route::forward() (module/route.cpp), NOT here -- see this file's ctor
  // comment on local_expert_ffns for why: this ExpertFFN::forward() method
  // only ever runs ONCE (at module-graph construction), while Route is a
  // graph_execution leaf that genuinely re-runs every decode step with live
  // per-expert token counts, which is what "first ACTIVATED expert" needs.

  for (int expert_id = expert_offset;
       expert_id < expert_offset + num_expert_per_device; expert_id++) {
    Module::Ptr expert_ffn =
        get_module("expert_FFN_" + std::to_string(expert_id));
    result = (*expert_ffn)(route_out.at(expert_id), sequences_metadata);

    expert_out.push_back(result);
  }

  (*sync_2)(input, sequences_metadata);

  result = (*all_reduce_for_e_tp)(input, sequences_metadata);

  (*sync_3)(input, sequences_metadata);

  Module::Ptr moe_gather = get_module("moe_gather");
  TensorVec moe_gather_out = (*moe_gather)(expert_out, sequences_metadata);

  (*sync_for_moe_gather)(input, sequences_metadata);

  result = (*all_reduce_for_gather)(input, sequences_metadata);

  (*sync_5)(input, sequences_metadata);
  
  // Shared Expert //
  for(int shared_expert_idx = 0 ; shared_expert_idx < num_shared_expert; shared_expert_idx ++){
    Module::Ptr shared_expert_ffn = get_module("shared_expert_FFN_" + std::to_string(shared_expert_idx));
    result = (*shared_expert_ffn)(input, sequences_metadata);
  }

  (*sync_6)(input, sequences_metadata);

  return result;
}

}  // namespace llm_system