#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class AllReduce : public Module {
 public:
  using Ptr = std::shared_ptr<AllReduce>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllReduce(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AllReduce(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  AllReduce() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class MoEScatter : public Module {
 public:
  using Ptr = std::shared_ptr<MoEScatter>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEScatter(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  MoEScatter(std::string& prefix, std::string& name,
             std::vector<int> device_list, Device::Ptr device);
  MoEScatter() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class MoEGather : public Module {
 public:
  using Ptr = std::shared_ptr<MoEGather>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEGather(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  MoEGather(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  MoEGather() = default;

  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;
};

class Sync : public Module {
 public:
  using Ptr = std::shared_ptr<Sync>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Sync(std::string& prefix, std::string& name, std::vector<int> device_list,
       Device::Ptr device);
  Sync() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class Sync__Set : public Module {
 public:
  using Ptr = std::shared_ptr<Sync__Set>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync__Set(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Sync__Set(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  Sync__Set() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class Sync__ : public Module {
 public:
  using Ptr = std::shared_ptr<Sync__>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync__(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Sync__(std::string& prefix, std::string& name, std::vector<int> device_list,
         Device::Ptr device);
  Sync__() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

class PipelineStage : public Module {
 public:
  using Ptr = std::shared_ptr<PipelineStage>;

  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int src_rank, int dst_rank,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new PipelineStage(prefix, name, src_rank, dst_rank, device));
    ptr->set_tensor_module();
    return ptr;
  }

 private:
  PipelineStage(std::string& prefix, std::string& name,
                int src_rank, int dst_rank, Device::Ptr device);
  PipelineStage() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int src_rank;
  int dst_rank;
};

}  // namespace llm_system