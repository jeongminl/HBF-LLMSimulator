#pragma once
// footprint.h — shared capacity-check helpers used by BOTH the parallelism
// optimizer (pre-graph, analytic) and checkMemorySize() in cluster.cpp.
//
// Dependency policy: include ONLY hardware/hardware_config.h (which already
// pulls in dram/hbf_memory_config.h).  Do NOT include cluster.h or any module
// header — that would create an include cycle.
#include <algorithm>
#include <cmath>
#include <string>

#include "hardware/hardware_config.h"

namespace llm_system {

// ---------------------------------------------------------------------------
// Capacity rule
// ---------------------------------------------------------------------------
// HBF with flash stacks present:
//   weight + kv  vs  hbf.total_capacity_bytes               (flash pool)
//   activation   vs  num_hbm_stacks * hbm_per_stack_bytes   (HBM tier, if any)
//                    logic_sram_bytes                         (no-HBM fallback)
// Plain HBM (use_hbf==false OR num_flash_stacks==0):
//   act + weight + kv  vs  memory_capacity                   (lumped)
// ---------------------------------------------------------------------------

struct CapacityResult {
  bool oom;
  std::string reason;
};

// Returns the capacity limit (bytes) for activations on the scarce tier.
// For HBF with HBM stacks  → total HBM capacity across all stacks.
// For HBF+ / CONV+ (no HBM)→ logic SRAM.
// For plain HBM             → full memory_capacity (acts not separately bounded).
inline double scarceTierActivationLimit(const SystemConfig& s) {
  if (s.use_hbf && s.hbf_config.num_flash_stacks > 0) {
    if (s.hbf_config.num_hbm_stacks > 0) {
      return static_cast<double>(s.hbf_config.num_hbm_stacks) *
             static_cast<double>(s.hbf_config.hbm_per_stack_bytes);
    } else {
      return static_cast<double>(s.hbf_config.logic_sram_bytes);
    }
  }
  return s.memory_capacity;
}

// Returns true when activations have a dedicated scarce-tier limit separate
// from the main (flash) pool.  Plain-HBM runs have only one pool.
inline bool hasScarceTier(const SystemConfig& s) {
  return s.use_hbf && s.hbf_config.num_flash_stacks > 0;
}

// Unified capacity check.
//   act    – activation bytes (per GPU, single-layer peak)
//   weight – weight bytes (per GPU, full PP stage)
//   kv     – KV-cache bytes (per GPU, full PP stage)
inline CapacityResult checkCapacity(const SystemConfig& s,
                                     double act,
                                     double weight,
                                     double kv) {
  if (s.use_hbf && s.hbf_config.num_flash_stacks > 0) {
    const auto& hbf = s.hbf_config;

    // Weights + KV live on flash
    double flash_cap = static_cast<double>(hbf.total_capacity_bytes);
    if (weight + kv > flash_cap) {
      return {true,
              "Flash capacity exceeded (" +
              std::to_string((weight + kv) / 1e9) + " GB > " +
              std::to_string(flash_cap / 1e9) + " GB)"};
    }

    // Activations live on the scarce tier (HBM stacks or logic SRAM)
    double scarce_cap = scarceTierActivationLimit(s);
    if (act > scarce_cap) {
      std::string tier = (hbf.num_hbm_stacks > 0) ? "HBM" : "Logic SRAM";
      return {true,
              tier + " activation capacity exceeded (" +
              std::to_string(act / 1e6) + " MB > " +
              std::to_string(scarce_cap / 1e6) + " MB)"};
    }

    return {false, ""};
  } else {
    // Plain HBM: lumped check against memory_capacity
    double total = act + weight + kv;
    if (total > s.memory_capacity) {
      return {true,
              "HBM capacity exceeded (" +
              std::to_string(total / 1e9) + " GB > " +
              std::to_string(s.memory_capacity / 1e9) + " GB)"};
    }
    return {false, ""};
  }
}

}  // namespace llm_system
