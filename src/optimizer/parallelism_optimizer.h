#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "model/model_config.h"
#include "hardware/hardware_config.h"

namespace llm_system {

struct ParallelConfig {
  int tp = 1;
  int pp = 1;
  int dp = 1;
  int ep = 1;  // expert (tensor) parallelism degree = e_tp_dg; 1 for dense models
  double estimated_latency_ms = 0.0;
  bool oom = false;
  std::string oom_reason = "";

  // Predicted memory footprint (bytes, per GPU, for the chosen PP stage).
  // Populated by Optimize(); consumed by the drift harness in test.cpp / cluster.cpp.
  // Negative value means "not available" (e.g. optimize_parallelism was off).
  double pred_weight_bytes = -1.0;
  double pred_kv_bytes    = -1.0;
  double pred_act_bytes   = -1.0;
  double pred_total_bytes = -1.0;
};

class ParallelismOptimizer {
 public:
  static ParallelConfig Optimize(const ModelConfig& model_config,
                                 const SystemConfig& system_config,
                                 int total_gpus,
                                 int batch_size,
                                 int sequence_length,
                                 double tpot_slo_ms = 100.0);
};

} // namespace llm_system
