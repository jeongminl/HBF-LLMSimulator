#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/hardware_config.h"
#include "hardware/node.h"
#include "hardware/stat.h"
#include "module/module.h"
#include "scheduler/scheduler.h"
#include "dram/power.h"

namespace llm_system {

class Cluster : public std::enable_shared_from_this<Cluster> {
  friend class Device;

 public:
  using Ptr = std::shared_ptr<Cluster>;
  [[nodiscard]] static Ptr Create(SystemConfig config,
                                  Scheduler::Ptr scheduler) {
    Ptr cluster = Ptr(new Cluster(config, scheduler));
    cluster->set(config);
    cluster->initializeDRAM((int)(ProcessorType::GPU), gpuEnergy);
    cluster->initializeDRAM((int)(ProcessorType::LOGIC), logicEnergy);
    cluster->initializeDRAM((int)(ProcessorType::PIM), pimEnergy);
    return cluster;
  }

  hw_metric cluster_ict_latency;
  hw_metric cluster_ict_bandwidth;

  std::vector<Node::Ptr> node;

  Ptr getptr() { return shared_from_this(); }

  Cluster(Cluster &&) = default;
  Cluster &operator=(Cluster &&) = default;

  SystemConfig config;
  int num_device;
  int num_node;
  int num_total_device;
  bool out_of_memory = false;

  Device::Ptr get_device(int device_total_rank);

  // The per-iteration decode step time: max(status.device_time) across every
  // device in the cluster. With pipeline parallelism (pp>1), a token's true
  // per-step latency is the FULL pipeline traversal, not any single stage's
  // local time -- PipelineStage::forward (communication.cpp) propagates each
  // stage's cumulative elapsed time forward to the next stage's device, so
  // the LAST stage of whichever pipeline finishes last already holds the true
  // cumulative critical-path time; taking the max across all devices is a
  // topology-agnostic way to read it back out (correct for pp==1 too, where
  // it's a no-op equal to get_device(0)'s own value).
  time_ns maxDeviceTime();

  void set_dependency();
  void add_module(int device_rank, std::string name, Module::Ptr module);

  void set_dependency_tensor(std::vector<Tensor::Ptr> &list, Tensor::Ptr tensor,
                             const std::vector<int> &device_list);

  std::vector<Stat> runIteration(int iter, std::string file_name = "stat");

  std::vector<Stat> runIterationMixed(int iter, std::ofstream &csv);
  std::vector<Stat> runIterationSumGenSplit(int iter, std::ofstream &csv);

  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);
  void restartModuleGraph();

  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);

  void setPerformExecution(bool perform);

  void updateTimestamp();
  void calibrateMemPoolLoadTime(Stat &stat);

  std::vector<std::map<std::string, Module::Ptr>> module_map;

  // pred_* args (all default -1) carry the optimizer's footprint prediction for
  // the drift harness (Part E).  Pass negative values to skip comparison.
  bool checkMemorySize(double pred_weight_bytes = -1,
                       double pred_kv_bytes     = -1,
                       double pred_act_bytes    = -1,
                       double pred_total_bytes  = -1);
  bool checkHeteroMemorySize();
  std::vector<energy_nJ> getTotalEnergy();

  void setTimeBreakDown(Stat &stat);
  void setStat(Stat &stat);

  void addLatency(std::vector<Stat> &stat_list,
                  const std::vector<Sequence::Ptr> &seq_list, time_ns time);

  void exportGantt(std::string gantt_file_path);

  std::map<CacheKey, ExecStatus> execution_time_cache;

 private:
  bool check_module_graph_remain();
  void exportToCSV(std::ofstream &csv, std::vector<Stat> &stat_list);
  void set(SystemConfig config);
  Cluster(SystemConfig config, Scheduler::Ptr scheduler);
  Cluster() = default;
  Executor executor;

  Scheduler::Ptr scheduler;

  void CreateNode(SystemConfig config);
};

}  // namespace llm_system