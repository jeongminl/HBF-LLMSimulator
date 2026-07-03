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

  // Enumerate every structurally-valid (tp, pp, dp, ep) candidate and evaluate
  // each at batch_size. require_batch_divisible=true additionally applies the
  // batch % dp == 0 gate (Optimize()'s historical behavior); per-config batch
  // searches pass false and probe each config only at multiples of its own dp.
  static std::vector<ParallelConfig> EnumerateCandidates(
      const ModelConfig& model_config,
      const SystemConfig& system_config,
      int total_gpus,
      int batch_size,
      int sequence_length,
      bool require_batch_divisible);

  // Evaluate ONE fixed (tp, pp, dp, ep) candidate at the given batch: fills
  // oom/oom_reason (capacity/SRAM only), estimated_latency_ms (ranking
  // heuristic), and the pred_* footprint fields. Callers must pass a
  // structurally-valid tuple (as produced by EnumerateCandidates' gates).
  static ParallelConfig EvaluateConfig(const ModelConfig& model_config,
                                       const SystemConfig& system_config,
                                       int total_gpus,
                                       int tp,
                                       int pp,
                                       int dp,
                                       int e_tp_dg,
                                       int batch_size,
                                       int sequence_length);
};

} // namespace llm_system
