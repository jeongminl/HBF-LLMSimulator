#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "module/base.h"
#include "scheduler/sequence.h"

namespace llm_system {

class GateUpdate : public Module {
  // Route tensor in to tensor vec
 public:
  using Ptr = std::shared_ptr<GateUpdate>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new GateUpdate(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };
  void aggregate_expert(BatchedSequence::Ptr sequences_metadata);
 private:
  GateUpdate(std::string& prefix, std::string& name,
        std::vector<int> device_list, Device::Ptr device);
  GateUpdate() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class Route : public Module {
  // Route tensor in to tensor vec
 public:
  using Ptr = std::shared_ptr<Route>;

  // local_expert_ffns: this device's locally-owned expert_FFN_X Module::Ptrs,
  // in the SAME local order forward() builds expert_token_list below (index i
  // == global expert expert_offset+i). Used ONLY for the paper2 §IV
  // first-activated-expert page-latency exposure (see forward()'s arm logic);
  // defaults to empty, which makes that logic a no-op -- any future caller
  // that doesn't pass it gets today's behavior unchanged.
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int num_expert_per_device,
                                  int expert_offset,
                                  std::vector<int> device_list,
                                  Device::Ptr device,
                                  std::vector<Module::Ptr> local_expert_ffns = {}) {
    Ptr ptr = Ptr(
        new Route(prefix, name, num_expert_per_device, expert_offset, device_list, device, local_expert_ffns));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Route(std::string& prefix, std::string& name, int num_expert_per_deice, int expert_offset,
        std::vector<int> device_list, Device::Ptr device,
        std::vector<Module::Ptr> local_expert_ffns);
  Route() = default;

  int expert_offset;
  int num_expert_per_device;
  std::vector<Module::Ptr> local_expert_ffns;

  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system