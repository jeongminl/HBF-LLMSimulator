#include "module/communication.h"
#include "scheduler/scheduler.h"
#include "hardware/cluster.h"

#include <cmath>
#include <algorithm>
#include <tuple>
#include <utility>

#include "common/assert.h"
#include "model/util.h"
// AllReduce //

namespace llm_system {

namespace {
// PP_FIX_SPEC.md §3.3: MoEScatter/MoEGather (like AllReduce/PipelineStage)
// manipulate device_time directly and never go through ExecStatus/
// set_pop_status, so they need their own device_time_dep tracking. Shared by
// both modules' send/receive legs: computes {total_time, batch_dependent_time}
// for one directional a2a transfer given its intra-/inter-node byte volumes.
// Each link's cost is latency (fixed, batch-independent) + bandwidth (volume-
// proportional, batch-dependent since the token/expert counts scale with the
// batch); the two links are combined via the SAME max() the live modules use,
// and the winning link's own bandwidth term is what gets returned as the
// batch-dependent contribution (mirroring AllReduce::forward's split).
std::pair<time_ns, time_ns> moeLinkTime(hw_metric intra_size, hw_metric inter_size,
                                        bool is_prefill, bool is_decode,
                                        hw_metric device_bw, time_ns device_lat,
                                        hw_metric node_bw, time_ns node_lat) {
  time_ns intra_bw = 0, intra_lat = 0, inter_bw = 0, inter_lat = 0;
  if (is_prefill) {
    // I6: gate each link's latency on whether it actually carries any volume,
    // mirroring the decode branch below -- a link with zero bytes to move
    // shouldn't still incur its fixed hop latency. (Prefill is off the paper-1
    // decode-steady-state path, so this is a latent-consistency fix only.)
    intra_bw = intra_size > 0 ? intra_size / device_bw * 1000 * 1000 * 1000 : 0.0;
    intra_lat = intra_size > 0 ? device_lat : 0.0;
    inter_bw = inter_size > 0 ? inter_size / node_bw * 1000 * 1000 * 1000 : 0.0;
    inter_lat = inter_size > 0 ? node_lat : 0.0;
  } else if (is_decode) {
    intra_bw = intra_size > 0 ? intra_size / device_bw * 1000 * 1000 * 1000 : 0.0;
    intra_lat = intra_size > 0 ? device_lat : 0.0;
    inter_bw = inter_size > 0 ? inter_size / node_bw * 1000 * 1000 * 1000 : 0.0;
    inter_lat = inter_size > 0 ? node_lat : 0.0;
  }
  time_ns intra_total = intra_lat + intra_bw;
  time_ns inter_total = inter_lat + inter_bw;
  return (intra_total >= inter_total) ? std::make_pair(intra_total, intra_bw)
                                      : std::make_pair(inter_total, inter_bw);
}
}  // namespace

AllReduce::AllReduce(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) { // sync == true, thererfore need to be synced
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output = Tensor::Create("allreduce_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr AllReduce::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];

  Tensor::Ptr output = get_activation("allreduce_output", input->shape);

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  // All-reduce time = latency term + bandwidth term, modeled SEPARATELY.
  //
  // Bandwidth term: the bandwidth-optimal all-reduce volume 2(N-1)/N * size per
  // link (ring / reduce-scatter + all-gather lower bound) -- unchanged.
  //
  // Latency term: 2*ceil(log2(N)) link latencies (recursive-doubling schedule),
  // NOT the ring's 2(N-1) sequential hops. The paper's reference system is a
  // DGX Rubin NVL8 whose GPUs are fully connected through NVSwitch, where the
  // latency-optimal schedule is logarithmic; chaining 2(N-1) fixed hop latencies
  // produced a batch-independent floor (e.g. 2.8 ms/step for llama3 TP=8) that
  // contradicts SS-III's "the inter-GPU communication increases almost linearly
  // with batch size" and Fig. 5's communication shares.
  int n_ranks = (int)device_list.size();
  int bw_hops = (n_ranks - 1) * 2;
  int latency_hops =
      (n_ranks > 1) ? 2 * (int)std::ceil(std::log2((double)n_ranks)) : 0;

  // Link selection: a group confined to one node runs on NVLink
  // (device_ict); a group spanning nodes is bottlenecked by the
  // inter-node link (node_ict), same rule as MoEScatter/MoEGather's
  // decode path and PipelineStage.
  int num_device_per_node = device->config.num_device;
  int min_node = device_list.front() / num_device_per_node;
  int max_node = min_node;
  for (int rank : device_list) {
    int node = rank / num_device_per_node;
    min_node = std::min(min_node, node);
    max_node = std::max(max_node, node);
  }
  bool cross_node = (max_node != min_node);
  hw_metric link_latency = cross_node ? device->config.node_ict_latency
                                      : device->config.device_ict_latency;
  hw_metric link_bandwidth = cross_node ? device->config.node_ict_bandwidth
                                        : device->config.device_ict_bandwidth;

  // PP_FIX_SPEC.md §3.3: like PipelineStage, AllReduce::forward manipulates
  // device_time directly (never goes through ExecStatus/set_pop_status), so
  // the mechanical batch_dependent_duration tagging applied to the Linear/
  // Activation/AttentionGen impl files never reaches it. Split total_time
  // into its batch-INDEPENDENT latency term (fixed hop count, independent of
  // message size) and its batch-DEPENDENT bandwidth term (size/n_ranks scales
  // with the batch-sized message) so the pp>1 reconstruction's device_time -
  // device_time_dep split (cluster.cpp) correctly treats only the volume
  // term as something to divide by pp, not the whole AR cost.
  time_ns latency_term = (time_ns)(latency_hops * link_latency);
  time_ns bandwidth_term = (time_ns)((double)bw_hops * ((double)size / n_ranks) /
                                     link_bandwidth * 1e9);
  time_ns total_time = latency_term + bandwidth_term;

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;
  device->status.device_time_dep += bandwidth_term;

  return output;
}

MoEScatter::MoEScatter(std::string& prefix, std::string& name,
                       std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output =
      Tensor::Create("moe_scatter_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr MoEScatter::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];
  Tensor::Ptr output = get_activation("moe_scatter_output", input->shape);

  
  // assume using disaggregated system

  if (!device->perform_execution) {
    return output;
  }

  time_ns total_time = 0;
  time_ns send_time = 0;
  time_ns receive_time = 0;

  // Send time // 
  int intra_node_comm_token = 0;
  int inter_node_comm_token = 0;

  int src = device->device_total_rank; // current device
  int src_node = src / device->config.num_device;

  int ne_tp_dg = device->model_config.ne_tp_dg;
  int e_tp_dg = device->model_config.e_tp_dg;

  std::vector<int> tp_sharing_device_list = {};
  int device_list_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;

  for(int device_idx = device_list_offset; device_idx < device_list_offset + ne_tp_dg; device_idx ++){
    tp_sharing_device_list.push_back(device_idx);
  }

  std::unordered_set<int> set_tp_devices(tp_sharing_device_list.begin(), tp_sharing_device_list.end());

  // PP_FIX (MoE a2a pp>1 scoping): iterate this STAGE's device_list (member,
  // set at Create-time to the stage-scoped devices -- see expert.cpp:39/110),
  // not the whole cluster. At pp==1 the stage device_list == the whole
  // cluster in rank order, so stage_n == total_num_device and
  // device_list[di] == di for every di -- byte-identical to the prior
  // total_num_device-based loop.
  int stage_n = (int)device_list.size();

  for(int di = 0; di < stage_n; di ++){ // di: LOCAL index within this stage's device_list
    int dst = device_list[di]; // dst: destination device (GLOBAL rank)
    if(set_tp_devices.count(dst) == 0){ // outer tp space
      // I7: multiply-first (matches cluster.cpp:154's num_routed_expert*e_tp_dg/devices_per_stage
      // capacity-model convention) instead of divide-first -- identical today since
      // num_routed_expert is evenly divisible by stage_n for every swept config.
      int expert_id_offset = device->model_config.num_routed_expert * e_tp_dg / stage_n * (di / e_tp_dg);
      // I7: multiply-first, matching cluster.cpp's capacity-model convention (see comment above).
      int num_expert_per_device = device->model_config.num_routed_expert * e_tp_dg / stage_n;

      if((int)(dst / device->config.num_device) == src_node){
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id]; // to the experts in a dst device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg (send only 1/ne_tp_dg tokens)

  hw_metric intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  hw_metric inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return output;
  }

  // PP_FIX_SPEC.md §3.3: moeLinkTime splits the fixed link latency
  // (batch-independent) from the volume-proportional bandwidth term
  // (batch-dependent) and propagates the winning (max) link's dep value --
  // see its doc comment above. Numerically identical to the prior inline
  // send_time computation (same formulas, same max composition).
  time_ns send_dep = 0, receive_dep = 0;
  bool is_prefill = sequences_metadata->get_sum_process_token() > 0;
  bool is_decode = sequences_metadata->get_gen_process_token() > 0;
  std::tie(send_time, send_dep) = moeLinkTime(
      intra_node_comm_size, inter_node_comm_size, is_prefill, is_decode,
      device->config.device_ict_bandwidth, device->config.device_ict_latency,
      device->config.node_ict_bandwidth, device->config.node_ict_latency);

  // Receive time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int dst_device_rank = device->device_total_rank; // dst: destination device, current device
  int dst_node = dst_device_rank / device->config.num_device;

  int pp_dg = device->model_config.pp_dg;
  int dst_dp_rank = (device->device_total_rank / ne_tp_dg) / pp_dg; // data parallel index

  // PP_FIX: LOCAL rank of this device within its stage's device_list, not a
  // global-rank-derived offset -- see get_local_rank_in_list (model/util.h).
  int dst_local_rank = get_local_rank_in_list(device_list, dst_device_rank);
  // I7: multiply-first, matching cluster.cpp's capacity-model convention (see comment above).
  int expert_id_offset = device->model_config.num_routed_expert * e_tp_dg / stage_n * (dst_local_rank / e_tp_dg); // experts in a dst_device
  // I7: multiply-first, matching cluster.cpp's capacity-model convention.
  int num_expert_per_device = device->model_config.num_routed_expert * e_tp_dg / stage_n;

  // stage_n/ne_tp_dg == dp (the stage device_list already excludes other pp
  // stages, so no separate /pp_dg is needed here -- equals the prior
  // (total_num_device/ne_tp_dg)/pp_dg at every pp, byte-identical at pp==1).
  for(int src_dp_idx = 0; src_dp_idx < stage_n / ne_tp_dg; src_dp_idx ++){
    if(src_dp_idx != dst_dp_rank){ // from other dp space

      int src_device_rank = device_list[src_dp_idx * ne_tp_dg]; // GLOBAL rank, for the node check
      int src_node = src_device_rank / device->config.num_device;

      if(dst_node == src_node){ 
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->scheduler->running_queue[src_dp_idx]->local_num_token_in_expert[e_id]; // from the experts in a src device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->scheduler->running_queue[src_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg (send only 1/ne_tp_dg tokens)
  
  intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return output;
  }

  std::tie(receive_time, receive_dep) = moeLinkTime(
      intra_node_comm_size, inter_node_comm_size, is_prefill, is_decode,
      device->config.device_ict_bandwidth, device->config.device_ict_latency,
      device->config.node_ict_bandwidth, device->config.node_ict_latency);

  total_time = std::max(send_time, receive_time);
  time_ns total_dep = (send_time >= receive_time) ? send_dep : receive_dep;

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
      // device->status.device_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;
  device->status.device_time_dep += total_dep;

  return output;
}

MoEGather::MoEGather(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output =
      Tensor::Create("moe_gather_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

TensorVec MoEGather::forward(const TensorVec input_vec,
                             BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr input = input_vec.at(0);
  int m = input->shape[0];
  int k = input->shape[1];
  Tensor::Ptr output = get_activation("moe_gather_output", input->shape);
  
  // assume using disaggregated system
  if (!device->perform_execution) {
    return input_vec;
  }

  time_ns total_time = 0;
  time_ns send_time = 0;
  time_ns receive_time = 0;

  // Receive time // 
  int intra_node_comm_token = 0;
  int inter_node_comm_token = 0;

  int dst = device->device_total_rank; // current device
  int dst_node = dst / device->config.num_device;

  int e_tp_dg = device->model_config.e_tp_dg;
  int ne_tp_dg = device->model_config.ne_tp_dg;

  std::vector<int> e_tp_sharing_device_list = {};
  int device_list_offset = device->device_total_rank / e_tp_dg * e_tp_dg;

  for(int device_idx = device_list_offset; device_idx < device_list_offset + e_tp_dg; device_idx ++){
    e_tp_sharing_device_list.push_back(device_idx);
  }

  std::unordered_set<int> set_e_tp_devices(e_tp_sharing_device_list.begin(), e_tp_sharing_device_list.end());

  // PP_FIX (MoE a2a pp>1 scoping): iterate this STAGE's device_list (member),
  // not the whole cluster -- see the matching comment in MoEScatter::forward
  // above. At pp==1 the stage device_list == the whole cluster in rank
  // order, so stage_n == total_num_device and device_list[di] == di for
  // every di -- byte-identical to the prior total_num_device-based loop.
  //
  // NOTE: this leg excludes via set_e_tp_devices (the e_tp peer set), not the
  // tp set used in MoEScatter's send leg. e_tp peers are contiguous only when
  // e_tp_dg <= ne_tp_dg; under pp>1 with e_tp_dg>ne_tp_dg the e_tp block can
  // span non-contiguous groups of this stage's device_list -- pp=1-safe,
  // latent for e_tp_dg>ne_tp_dg (not reached by paper-1 configs).
  int stage_n = (int)device_list.size();

  for(int di = 0; di < stage_n; di ++){ // di: LOCAL index within this stage's device_list
    int src = device_list[di]; // src: source device (GLOBAL rank)
    if(set_e_tp_devices.count(src) == 0){ // outer tp space
      // I7: multiply-first (matches cluster.cpp:154's num_routed_expert*e_tp_dg/devices_per_stage
      // capacity-model convention) instead of divide-first -- identical today since
      // num_routed_expert is evenly divisible by stage_n for every swept config.
      int expert_id_offset = device->model_config.num_routed_expert * e_tp_dg / stage_n * (di / e_tp_dg);
      // I7: multiply-first, matching cluster.cpp's capacity-model convention (see comment above).
      int num_expert_per_device = device->model_config.num_routed_expert * e_tp_dg / stage_n;

      if((int)(src / device->config.num_device) == dst_node){
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id]; // to the experts in a src device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  hw_metric intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  hw_metric inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= device->model_config.e_tp_dg;
  inter_node_comm_size /= device->model_config.e_tp_dg;

  intra_node_comm_size /= device->model_config.ne_tp_dg; // receive only (1 / tp_degree) tokens, and then all reduce
  inter_node_comm_size /= device->model_config.ne_tp_dg; // receive only (1 / tp_degree) tokens, and then all reduce

  // FP8 dispatch && BF16 combine
  if((device->model_config.model_name == "deepseekV3") && device->model_config.precision_byte == 1){
    intra_node_comm_size *= 2;
    inter_node_comm_size *= 2;
  }

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return input_vec;
  }

  // PP_FIX_SPEC.md §3.3: see moeLinkTime's doc comment above (shared with
  // MoEScatter::forward) -- splits fixed link latency (indep) from the
  // volume-proportional bandwidth term (dep), numerically identical to the
  // prior inline receive_time computation.
  time_ns send_dep = 0, receive_dep = 0;
  bool is_prefill = sequences_metadata->get_sum_process_token() > 0;
  bool is_decode = sequences_metadata->get_gen_process_token() > 0;
  std::tie(receive_time, receive_dep) = moeLinkTime(
      intra_node_comm_size, inter_node_comm_size, is_prefill, is_decode,
      device->config.device_ict_bandwidth, device->config.device_ict_latency,
      device->config.node_ict_bandwidth, device->config.node_ict_latency);

  // Send time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int src_device_rank = device->device_total_rank; // src: source device, current device
  int src_node = src_device_rank / device->config.num_device;

  int pp_dg = device->model_config.pp_dg;
  int src_dp_rank = (device->device_total_rank / ne_tp_dg) / pp_dg; // data parallel index

  // PP_FIX: LOCAL rank of this device within its stage's device_list, not a
  // global-rank-derived offset -- see get_local_rank_in_list (model/util.h).
  int src_local_rank = get_local_rank_in_list(device_list, src_device_rank);
  // I7: multiply-first, matching cluster.cpp's capacity-model convention (see comment above).
  int expert_id_offset = device->model_config.num_routed_expert * e_tp_dg / stage_n * (src_local_rank / e_tp_dg); // experts in a src_device
  // I7: multiply-first, matching cluster.cpp's capacity-model convention.
  int num_expert_per_device = device->model_config.num_routed_expert * e_tp_dg / stage_n;

  // stage_n/ne_tp_dg == dp (the stage device_list already excludes other pp
  // stages, so no separate /pp_dg is needed here -- equals the prior
  // (total_num_device/ne_tp_dg)/pp_dg at every pp, byte-identical at pp==1).
  for(int dst_dp_idx = 0; dst_dp_idx < stage_n / ne_tp_dg; dst_dp_idx ++){
    if(dst_dp_idx != src_dp_rank){ // to other dp space

      int dst_device_rank = device_list[dst_dp_idx * ne_tp_dg]; // GLOBAL rank, for the node check
      int dst_node = dst_device_rank / device->config.num_device;

      if(dst_node == src_node){ 
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->scheduler->running_queue[dst_dp_idx]->local_num_token_in_expert[e_id]; // to orig device from the experts in a src device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->scheduler->running_queue[dst_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg (send only 1/ne_tp_dg tokens)
  
  intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return input_vec;
  }

  std::tie(send_time, send_dep) = moeLinkTime(
      intra_node_comm_size, inter_node_comm_size, is_prefill, is_decode,
      device->config.device_ict_bandwidth, device->config.device_ict_latency,
      device->config.node_ict_bandwidth, device->config.node_ict_latency);

  total_time = std::max(send_time, receive_time);
  time_ns total_dep = (send_time >= receive_time) ? send_dep : receive_dep;

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;
  device->status.device_time_dep += total_dep;

  return input_vec;
}

Sync::Sync(std::string& prefix, std::string& name, std::vector<int> device_list,
           Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  std::vector<int> shape = {1, 1};

  auto sync__set =
      Sync__Set::Create(module_map_name, "sync__set", device_list, device);
  add_module(sync__set);

  auto sync__ = Sync__::Create(module_map_name, "sync__", device_list, device);
  add_module(sync__);
}

Tensor::Ptr Sync::forward(const Tensor::Ptr input,
                          BatchedSequence::Ptr sequences_metadata) {
  auto sync__set = get_module("sync__set");
  auto sync__ = get_module("sync__");

  Tensor::Ptr temp = (*sync__set)(input, sequences_metadata);
  Tensor::Ptr output = (*sync__)(temp, sequences_metadata);
  return output;
}

Sync__Set::Sync__Set(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {1, 1};
  Tensor::Ptr output = Tensor::Create("sync", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr Sync__Set::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output = get_activation("sync");
  device->status.device_time =
      std::max(device->status.device_time,
               std::max(device->status.low_time, device->status.high_time));
  device->status.high_time = device->status.device_time;
  device->status.low_time = device->status.device_time;
  return output;
}

Sync__::Sync__(std::string& prefix, std::string& name,
               std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {}

Tensor::Ptr Sync__::forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) {
  return input;
}

PipelineStage::PipelineStage(std::string& prefix, std::string& name,
                             int src_rank, int dst_rank, Device::Ptr device)
    : Module(prefix, name, device, {src_rank}, true),
      src_rank(src_rank),
      dst_rank(dst_rank) {
  std::vector<int> shape = {1, 1};
  Tensor::Ptr output = Tensor::Create("pipeline_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

Tensor::Ptr PipelineStage::forward(const Tensor::Ptr input,
                                   BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output = get_activation("pipeline_output", input->shape);
  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  int src_node = src_rank / device->config.num_device;
  int dst_node = dst_rank / device->config.num_device;

  time_ns comm_time = 0;
  if (src_node == dst_node) {
    comm_time = device->config.device_ict_latency +
                (double)size / device->config.device_ict_bandwidth * 1e9;
  } else {
    comm_time = device->config.node_ict_latency +
                (double)size / device->config.node_ict_bandwidth * 1e9;
  }

  device->status.device_time += comm_time;
  // PP_FIX_SPEC.md §3.5: also record this stage's own hop cost separately from
  // device_time, so the pp>1 reconstruction (cluster.cpp) can read back
  // exactly what this stage's PipelineStage contributed and account for it
  // once (as the explicit (pp-1)*hop term) instead of leaving it embedded in
  // this stage's own device_time (which would double-count it, since the
  // reconstruction already adds one hop per non-last stage explicitly).
  device->status.pipeline_hop_time += comm_time;

  // PP_FIX_SPEC.md §3.4 (the double-count fix): this used to also do
  // `dst_device->status.device_time += device->status.device_time;` --
  // propagating THIS device's entire cumulative clock (this stage's own work
  // PLUS the hop) onto the next stage's device. That models strict
  // serialization across pipeline stages at the FULL per-replica batch, which
  // is exactly the "3rd stage" double-count/over-serialization this fix
  // removes (see PP_FIX_SPEC.md §1 root-cause trace): the non-atomic
  // round-robin device scheduler lets a stage-1 group AllReduce's
  // sync_devices max-broadcast (module_graph.cpp) land between the 8 per-tp-
  // peer PipelineStage adds, re-stamping an already-added clock onto the two
  // devices that had not yet been added to, which then get added to a SECOND
  // time. Stages are no longer serialized through this propagation; each
  // stage's device_time now holds only its own locally-accumulated work (plus
  // its own one-hop send cost above). Cluster::runIterationMixed /
  // runIterationSumGenSplit reconstruct the correct pp>1 reported time as
  // full_model(B/pp) + (pp-1)*hop directly from each stage's own clock
  // (device_time/device_time_dep/pipeline_hop_time), rather than relying on
  // clocks chained stage-to-stage through this module. dst_rank is retained
  // as a PipelineStage member (constructor) for module-graph wiring, but is
  // no longer dereferenced here.

  return output;
}

}  // namespace llm_system

