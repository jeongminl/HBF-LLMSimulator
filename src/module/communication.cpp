#include "module/communication.h"
#include "scheduler/scheduler.h"
#include "hardware/cluster.h"

#include <cmath>
#include <algorithm>

#include "common/assert.h"
// AllReduce //

namespace llm_system {

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

  time_ns total_time =
      (time_ns)(latency_hops * link_latency +
                (double)bw_hops * ((double)size / n_ranks) /
                    link_bandwidth * 1e9);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

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

  int total_num_device = device->config.num_device * device->config.num_node;

  for(int dst = 0; dst < total_num_device; dst ++){ // dst: destination device
    if(set_tp_devices.count(dst) == 0){ // outer tp space
      int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (dst / e_tp_dg) * e_tp_dg;
      int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

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

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    send_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  // Receive time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int dst_device_rank = device->device_total_rank; // dst: destination device, current device
  int dst_node = dst_device_rank / device->config.num_device;

  int pp_dg = device->model_config.pp_dg;
  int dst_dp_rank = (device->device_total_rank / ne_tp_dg) / pp_dg; // data parallel index
  
  int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (dst_device_rank / e_tp_dg) * e_tp_dg; // experts in a dst_device
  int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

  for(int src_dp_idx = 0; src_dp_idx < (total_num_device / ne_tp_dg) / pp_dg; src_dp_idx ++){ 
    if(src_dp_idx != dst_dp_rank){ // from other dp space

      int src_device_rank = src_dp_idx * ne_tp_dg;
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

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    receive_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  total_time = std::max(send_time, receive_time);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
      // device->status.device_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

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

  int total_num_device = device->config.num_device * device->config.num_node;
  
  for(int src = 0; src < total_num_device; src ++){ // dst: destination device
    if(set_e_tp_devices.count(src) == 0){ // outer tp space
      int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (src / e_tp_dg) * e_tp_dg;
      int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

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

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    receive_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  // Send time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int src_device_rank = device->device_total_rank; // src: source device, current device
  int src_node = src_device_rank / device->config.num_device;

  int pp_dg = device->model_config.pp_dg;
  int src_dp_rank = (device->device_total_rank / ne_tp_dg) / pp_dg; // data parallel index
  
  int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (src_device_rank / e_tp_dg) * e_tp_dg; // experts in a src_device
  int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

  for(int dst_dp_idx = 0; dst_dp_idx < (total_num_device / ne_tp_dg) / pp_dg; dst_dp_idx ++){ 
    if(dst_dp_idx != src_dp_rank){ // to other dp space

      int dst_device_rank = dst_dp_idx * ne_tp_dg;
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

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    send_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  total_time = std::max(send_time, receive_time);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

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

  // Propagate elapsed time to the destination stage's device. A token cannot
  // begin stage (k+1)'s compute before stage k's output has physically arrived,
  // and in autoregressive decode the stages of one token pass are strictly
  // sequential (a request's token t+1 cannot enter stage 0 before token t
  // leaves the last stage, and even micro-batch pipelining leaves the
  // steady-state rate at batch / sum-of-stages) -- so the destination's clock
  // must end at src's send-complete time PLUS the destination's own stage work.
  //
  // This is an ADDITION of the upstream cumulative time, not a max() bump.
  // Every device's clock is reset to 0 at the start of each iteration
  // (Cluster::run -> restartModuleGraph -> reset_status), so at this moment
  // dst's device_time holds exactly its own locally-accumulated stage work
  // (plus any upstream time already chained in by an earlier stage's
  // PipelineStage -- devices execute in rank order = pipeline-stage order, so
  // upstream bumps land before this stage's own PipelineStage runs).
  //
  // Addition is exact for both device-execution orderings the round-robin
  // scheduler (Cluster::run) produces: when dst has not yet run (tp==1, no
  // intra-stage sync blocks it, so devices drain in rank = pipeline order) it
  // contributes 0 and stages serialize; when dst has already accumulated its
  // stage work (tp>=2, per-layer AllReduce advances all stages in lock-step) it
  // contributes that work on top of the upstream time. Either way the destination
  // clock ends at src's send-complete time plus its own stage work. Serialization
  // is the correct decode accounting (autoregressive dependency; steady-state rate
  // is batch / sum-of-stages regardless of micro-batch pipelining). See CHANGES.md
  // items 22/35.
  Device::Ptr dst_device = device->cluster->get_device(dst_rank);
  dst_device->status.device_time += device->status.device_time;

  return output;
}

}  // namespace llm_system

