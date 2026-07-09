#pragma once

#include <map>
#include <memory>
#include <vector>

#include "hardware/device.h"
#include "module/base.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class ModuleGraph;

class Module : public std::enable_shared_from_this<Module> {
  friend class ModuleGraph;

 public:
  using Ptr = std::shared_ptr<Module>;

  Module(std::string prefix = "", std::string name = "",
         Device::Ptr device = nullptr, std::vector<int> device_list = {},
         bool graph_execution = false, bool sync = false)
      : name(name),
        device(device),
        device_list(device_list),
        graph_execution(graph_execution),
        sync(sync) {
    module_map_name = prefix + "::" + name;
    if (sync) {
      sync_master = device_list.at(0);
    }
    need_to_aggregate = false;
  }

  virtual ~Module(){};

  Module::Ptr getptr() { return shared_from_this(); }

  Tensor::Ptr operator()(const Tensor::Ptr input,
                         BatchedSequence::Ptr sequences_metadata);

  TensorVec operator()(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata);

  void set_device(Device::Ptr device);

  long size(std::vector<std::string> tag = {"weight", "act", "cache"});

  bool execution() { return graph_execution; }
  void add_tensor(Tensor::Ptr tensor);

  void add_tensor(std::string name, Tensor::Ptr tensor);

  Module::Ptr get_module(std::string module_name) {
    if (module_list.find(module_name) == module_list.end()) {
      std::cerr << "Module Lookup Error: Parent module '" << this->name 
                << "' cannot find child module '" << module_name << "'" << std::endl;
    }
    return module_list.at(module_name);
  }
  Tensor::Ptr get_activation(std::string name, std::vector<int> shape = {1, 1},
                             bool ready = true);

  Tensor::Ptr get_cache(std::string name, int seq_idx, int kv_idx, bool compressed_kv,
                        std::vector<int> shape = {1, 1});

  template <class T>
  void add_module(std::shared_ptr<T> module) {
    Module::Ptr _module = std::static_pointer_cast<Module>(module);
    add_module_with_name(module->name, _module);
  }

  void set_dependency_tensor(std::vector<Tensor::Ptr>& dependency_tensor_list,
                             Tensor::Ptr tensor);

  void unset_tensor();

  void add_module_with_name(std::string name, Module::Ptr module);

  void set_module_map_name(std::string name) { module_map_name = name; }

  void set_tensor_module();

  // paper2 §IV first-activated-expert page-latency exposure (see tensor.h's
  // exposeFirstExpertPageLatency comment for the full mechanism). Default
  // no-op so every module except the specific chain that actually leads to
  // an expert's first weight-bearing Linear (Linear itself, and the
  // FeedForward2Way/3Way + ColumnParallelLinear wrappers that forward to it)
  // does nothing -- callers can invoke this on any Module::Ptr without
  // knowing its concrete type.
  virtual void armExposeFirstExpertPageLatency() {}

  std::string name;
  std::string module_map_name;
  bool sync;
  bool need_to_aggregate;

  std::vector<long> get_size();

 protected:
  std::vector<int> device_list;

  Device::Ptr device;
  int sync_master;
  std::map<std::string, Tensor::Ptr> tensor_list;
  std::map<std::string, Module::Ptr> module_list;

  const bool graph_execution;

 private:
  virtual Tensor::Ptr forward(const Tensor::Ptr input,
                              BatchedSequence::Ptr sequences_metadata) {
    return input;
  }
  virtual TensorVec forward(const TensorVec input,
                            BatchedSequence::Ptr sequences_metadata) {
    return input;
  }
};

}  // namespace llm_system