#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/layernorm.h"
#include "module/residual.h"
#include "module/parallel.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class Decoder : public Module {
 public:
  using Ptr = std::shared_ptr<Decoder>;
  // layer_idx (0-indexed, model-global): threaded through to Attention::Create so it
  // can determine local vs. global attention (Llama-4-style iRoPE) -- see layer.h.
  // Defaults to 0 (a global layer), so any existing caller not passing it is unaffected.
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device, int layer_idx = 0) {
    Ptr ptr = Ptr(new Decoder(prefix, name, model_config, scheduler,
                              device_list, device, layer_idx));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Decoder(std::string& prefix, std::string& name,
          const ModelConfig& model_config, Scheduler::Ptr scheduler,
          std::vector<int>& device_list, Device::Ptr device, int layer_idx);
  Decoder() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class MoEDecoder : public Module {
 public:
  using Ptr = std::shared_ptr<MoEDecoder>;
  // layer_idx: see Decoder::Create's doc comment above.
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device, int layer_idx = 0) {
    Ptr ptr = Ptr(new MoEDecoder(prefix, name, model_config, scheduler,
                                 device_list, device, layer_idx));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  MoEDecoder(std::string& prefix, std::string& name,
             const ModelConfig& model_config, Scheduler::Ptr scheduler,
             std::vector<int>& device_list, Device::Ptr device, int layer_idx);
  MoEDecoder() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system