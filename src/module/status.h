#pragma once
#include <memory>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"
#include "module/base.h"

namespace llm_system {

class Tensor;

struct ExecStatus {
  ProcessorType processor_type;
  time_ns time = 0.0;
  time_ns total_duration = 0.0;
  time_ns compute_duration = 0.0;
  time_ns memory_duration = 0.0;
  time_ns kv_write_duration = 0.0;  // track unhidden KV write duration separately
  // PP_FIX_SPEC.md §3.1/3.3: portion of total_duration that scales with the
  // op's num_process_token (batch). Tagged per-op in linear_impl.cpp,
  // activation_impl.cpp, attention_gen_impl.cpp (see each file's comment for
  // its rule); comm modules (AllReduce/MoEScatter-Gather/PipelineStage) leave
  // this at its default 0 -- their volume-proportional time is handled
  // separately by the pp>1 reconstruction's explicit hop term (cluster.cpp).
  // Untagged ops default to 0 (treated as batch-independent), which is safe:
  // pp==1 never reads this field (cluster.cpp gates the reconstruction on
  // pp>1), so mis-tagging cannot regress the pp==1 path.
  time_ns batch_dependent_duration = 0.0;

  hw_metric flops = 0.0;
  hw_metric memory_size = 0.0;

  hw_metric opb = 0.0;
  util compute_util = 0.0;
  util memory_util = 0.0;
  // time_ns pim_duration = 0.0;

  counter_t generic_read_cmd = 0.0;
  counter_t generic_write_cmd = 0.0;
  counter_t compute_pim_cmd = 0.0;
  counter_t move_pim_cmd = 0.0;

  counter_t act_count = 0;
  counter_t read_count = 0;
  counter_t write_count = 0;
  counter_t all_act_count = 0;
  counter_t all_read_count = 0;
  counter_t all_write_count = 0;
  counter_t ref_count = 0;

  bool parallel_execution = false;

  ExecStatus& operator+=(const ExecStatus& rhs) {
    memory_duration += rhs.memory_duration;
    kv_write_duration += rhs.kv_write_duration;
    batch_dependent_duration += rhs.batch_dependent_duration;
    generic_read_cmd += rhs.generic_read_cmd;
    generic_write_cmd += rhs.generic_write_cmd;
    compute_pim_cmd += rhs.compute_pim_cmd;
    move_pim_cmd += rhs.move_pim_cmd;

    act_count += rhs.act_count;
    read_count += rhs.read_count;
    write_count += rhs.write_count;
    all_act_count += rhs.all_act_count;
    all_read_count += rhs.all_read_count;
    all_write_count += rhs.all_write_count;
    ref_count += rhs.ref_count;

    return *this;
  }
};

class StatusBoard {
 public:
  using Tensor_Ptr = std::shared_ptr<Tensor>;
  StatusBoard() = default;

  // accumulate
  time_ns device_time = 0;
  // PP_FIX_SPEC.md §3.1/3.2: batch-dependent portion of device_time, mirrored
  // in lock-step with device_time by TopModuleGraph::set_pop_status
  // (module_graph.cpp). Read by the pp>1 reconstruction (cluster.cpp) to
  // split each pipeline stage's clock into W (batch-independent, summed once
  // per stage) and K*B (batch-dependent, summed then divided by pp) --
  // pp==1 never reads this field.
  time_ns device_time_dep = 0;
  // PP_FIX_SPEC.md §3.4/§3.5: the actual per-iteration comm_time a
  // PipelineStage module added to ITS OWN device (communication.cpp:558),
  // separate from device_time so the pp>1 reconstruction can net it back out
  // of that stage's clock before splitting into indep/dep (it is neither
  // batch-independent stage work nor batch-dependent stage work -- it is the
  // physical inter-stage hop, added back exactly once per non-last stage by
  // the reconstruction). 0 for the last stage (no PipelineStage module) and
  // always 0 when pp==1 (module is never constructed, see llm.cpp:68).
  time_ns pipeline_hop_time = 0;

  time_ns high_time = 0;  // time of GPU
  time_ns low_time = 0;   // time of PIM
  time_ns kv_write_time = 0; // accumulated KV write time

  Tensor_Ptr tensor;

  TensorVec tensor_vec;
  // TensorVec output_tensor_vec;

  // energy_nJ act;
  // energy_nJ read;
  // energy_nJ write;
  // energy_nJ tsv;
  // energy_nJ interposer_io;
  energy_nJ act_energy;
  energy_nJ read_energy;
  energy_nJ write_energy;

  energy_nJ all_act_energy;
  energy_nJ all_read_energy;
  energy_nJ all_write_energy;

  energy_nJ act_energy_load;
  energy_nJ read_energy_load;
  energy_nJ write_energy_load;

  energy_nJ all_act_energy_load;
  energy_nJ all_read_energy_load;
  energy_nJ all_write_energy_load;

  energy_nJ mac_energy;

  util compute_util;
  util memory_util;

  // record
  time_ns start_time = 0;
  time_ns end_time = 0;

  hw_metric flops = 0.0;
  hw_metric memory_size = 0.0;
  hw_metric opb = 0.0;
  bool isTensorVec;
  std::vector<int> input_tensor_shape;
  std::vector<int> output_tensor_shape;
  std::vector<std::vector<int>> input_tensor_vec_shape;
  std::vector<std::vector<int>> output_tensor_vec_shape;
  ProcessorType processor_type;
  bool parallel_execution = false;
};

}  // namespace llm_system