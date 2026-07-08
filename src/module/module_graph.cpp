#include "module/module_graph.h"

namespace llm_system {

ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         Tensor::Ptr input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = false;
  stamped = false;
};

ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         TensorVec input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input_vec(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = true;
  stamped = false;
};

bool ModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  if (module == nullptr || !module->execution()) {
    return true;
  } else if (check_ready() == true) {
    if (isTensorVec) {
      module->forward(input_vec, sequences_metadata);
    } else {
      module->forward(input, sequences_metadata);
    }
    return true;
  } else {
    return false;
  }
};

void ModuleGraph::set_dependency() { set_dependency_tensor(); }

bool ModuleGraph::checkListReady(TensorVec tensor_list) {
  for (Tensor::Ptr tensor : dependency_tensor_list) {
    if (tensor->ready == false) {
      return false;
    }
  }
  return true;
}

bool ModuleGraph::check_ready() {
  if (module->sync) {
    if (!checkListReady(dependency_tensor_list)) {
      return false;
    }
    // all operations are doned, we have to sync the devices
    sync_devices();
    return true;
  } else {
    if (input && input->ready) {
      return true;
    } else if (input_vec.size() != 0) {
      if (checkListReady(input_vec)) {
        return true;
      }
    }
  }
  return false;
}

void ModuleGraph::sync_devices() {
  // not yet synced
  if (input) {
    if (!input->timeboard_synced) {
      Device::Ptr device;
      time_ns time = 0;
      // PP_FIX_SPEC.md §3.2: device_time_dep must be broadcast in lockstep
      // with device_time, or a TP-peer whose device_time gets bumped up to
      // another peer's (because the round-robin device scheduler ran it
      // behind) keeps its OWN stale device_time_dep -- corrupting the
      // (device_time - device_time_dep) indep/dep split the pp>1
      // reconstruction (cluster.cpp) reads back out. Track and broadcast the
      // (time, time_dep) PAIR from whichever device carries the max time
      // (not an independent max of each field), since dep is only meaningful
      // as a component of ITS OWN device's time.
      time_ns time_dep = 0;
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        device = tensor->get_device();
        time_ns device_time = device->get_time();
        if (device_time >= time) {
          time = device_time;
          time_dep = device->get_time_dep();
        }
      }
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        tensor->timeboard_synced = true;
        device = tensor->get_device();
        device->set_time(time);
        device->set_time_dep(time_dep);
      }
      input->timeboard_synced = false;
    } else {
      input->timeboard_synced = false;
    }
  } else {
    fail("Module cannot be synced when it's inputs are TensorVector");
  }
}

void ModuleGraph::set_dependency_tensor() {
  // only when inputs are one Tensor pointer
  if (input && module && module->sync) {
    module->set_dependency_tensor(dependency_tensor_list, input);
  }
}

void ModuleGraph::print_graph() {
  if (module != nullptr) {
    for (int i = 0; i < module_level; i++) {
      std::cout << "\t";
    }
    std::cout << module->name << std::endl;
  }
}

TopModuleGraph::TopModuleGraph(StatusBoard& status)
    : status(status), module_graph(){};

void TopModuleGraph::push_module_graph(Module::Ptr module, Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

void TopModuleGraph::pop_module_graph(Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

void TopModuleGraph::push_module_graph(Module::Ptr module, TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

void TopModuleGraph::pop_module_graph(TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

void TopModuleGraph::set_dependency() {
  for (auto module : module_graph) {
    module->set_dependency();
  }
  restart_graph();
}

void TopModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  for (; current_module != module_graph.end(); current_module++) {
    set_stamp();
    // if execution is blocked because of sync
    if (!(*current_module)->run(sequences_metadata)) {
      break;
    }
  }
}

void TopModuleGraph::set_stamp() {
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_stamped()) {
    if (module_graph->is_pop()) {
      set_pop_status();
      timeboard.pop_timestamp(status);
    } else {
      set_push_status();
      timeboard.push_timestamp(status, module_graph->get_name());
    }
    module_graph->set_stamped();
  }
}

void TopModuleGraph::set_push_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;

    status.device_time = std::max(status.device_time,
                                  std::max(status.low_time, status.high_time));
    status.low_time = status.device_time;
    status.high_time = status.device_time;
    status.parallel_execution = false;
    //

  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
      if (status.tensor->isPerformHigh()) {
        status.device_time = status.high_time;
      } else {
        status.device_time = status.low_time;
      }
    } else {
      status.device_time = std::max(
          status.device_time, std::max(status.low_time, status.high_time));
      status.low_time = status.device_time;
      status.high_time = status.device_time;
      status.parallel_execution = false;
    }
  }
}

void TopModuleGraph::set_pop_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;
  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
    } else {
      status.parallel_execution = false;
    }
  }

  ExecStatus exec_status = device->getExecStatus();
  status.kv_write_time += exec_status.kv_write_duration;

  if (status.parallel_execution) {
    // PP_FIX_SPEC.md §3.2: no device_time_dep mirror on this branch --
    // device->config.parallel_execution is hardcoded false in every
    // SystemConfig hardware preset (hardware_config.h: A100/H100/B100/B200/
    // Rubin), so status.parallel_execution can never become true (it only
    // derives from tensor->parallel_execution, itself only ever set by
    // Tensor::setPerformHigh/Low, all of which are gated behind
    // device->config.parallel_execution or an already-true predecessor
    // tensor). This branch is unreachable in the current build; if the
    // feature is ever enabled, low_time/high_time would need matching
    // low_time_dep/high_time_dep accumulators added here.
    if (exec_status.processor_type == ProcessorType::LOGIC ||
        exec_status.processor_type == ProcessorType::PIM) {
      status.low_time += exec_status.total_duration;
      status.device_time = status.low_time;
    } else if (exec_status.processor_type == ProcessorType::GPU) {
      status.high_time += exec_status.total_duration;
      status.device_time = status.high_time;
    }
  } else {
    status.device_time += exec_status.total_duration;
    // PP_FIX_SPEC.md §3.2: mirror the batch-dependent portion in lock-step.
    status.device_time_dep += exec_status.batch_dependent_duration;
  }

  if (exec_status.processor_type == ProcessorType::PIM ||
      exec_status.processor_type == ProcessorType::LOGIC ||
      exec_status.processor_type == ProcessorType::GPU) {
    int processor_type = (int)exec_status.processor_type;
    status.act_energy +=
        exec_status.act_count * dram_powers[processor_type].kACT_energy_j_;
    status.read_energy +=
        exec_status.read_count * dram_powers[processor_type].kREAD_energy_j_;
    status.write_energy +=
        exec_status.write_count * dram_powers[processor_type].kWRITE_energy_j_;

    status.all_act_energy += exec_status.all_act_count *
                             dram_powers[processor_type].kALL_ACT_energy_j_;
    status.all_read_energy += exec_status.all_read_count *
                              dram_powers[processor_type].kALL_READ_energy_j_;
    status.all_write_energy += exec_status.all_write_count *
                               dram_powers[processor_type].kALL_WRITE_energy_j_;

    status.mac_energy +=
        exec_status.flops * dram_powers[processor_type].kMAC_energy_j_;
  }

  status.compute_util = exec_status.compute_util;
  status.memory_util = exec_status.memory_util;
  status.processor_type = exec_status.processor_type;

  status.flops += exec_status.flops;
  status.memory_size += exec_status.memory_size;

  status.opb = exec_status.opb;
}

void TopModuleGraph::print_graph() {
  std::cout << "Print graph" << std::endl;
  for (auto module_graph_ : module_graph) {
    module_graph_->print_graph();
  }
}

void TopModuleGraph::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  if(dram_powers.size() == 0){
    for (int i = 0; i < (int)ProcessorType::MAX; i++) {
      DramEnergy temp;
      dram_powers.push_back(temp);
    }
  }
  dram_powers[ProcessorType] = dramEnergy;
}

std::vector<energy_nJ> TopModuleGraph::getDeviceEnergy(){
  std::vector<energy_nJ> device_energy {status.act_energy, status.read_energy, status.write_energy, 
                            status.all_act_energy, status.all_read_energy, status.all_write_energy,
                            status.mac_energy, status.act_energy + status.read_energy + status.write_energy + 
                            status.all_act_energy + status.all_read_energy + status.all_write_energy + status.mac_energy}; 
  return device_energy;
}

void TopModuleGraph::restart_graph() {
  current_module = module_graph.begin();
  assertTrue(current_module != module_graph.end(),
             "No module in TopModuleGraph");
  (*current_module)->set_ready();

  for (auto module_graph_ : module_graph) {
    module_graph_->unset_tensor();
  }
};

}  // namespace llm_system