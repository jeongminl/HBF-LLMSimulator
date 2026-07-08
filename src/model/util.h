#pragma once
#include <string>
#include <vector>

namespace llm_system {
void inline set_device_list(std::vector<int> &device_list, int device_offset,
                            int num_device) {
  device_list.resize(0);

  for (int device = device_offset; device < device_offset + num_device;
       device++) {
    device_list.push_back(device);
  }
}

// Linear scan for `rank`'s index within `device_list` (the stage-scoped
// device set a Module already holds, e.g. Module::device_list). Returns -1
// if `rank` is not a member. Used by MoEScatter/MoEGather (communication.cpp)
// to translate a global device rank into its LOCAL index within the current
// pipeline stage's device list, so per-device expert-range math scopes to
// the stage (size == device_list.size()) instead of the whole cluster.
int inline get_local_rank_in_list(const std::vector<int> &device_list,
                                  int rank) {
  for (int idx = 0; idx < (int)device_list.size(); idx++) {
    if (device_list[idx] == rank) {
      return idx;
    }
  }
  return -1;
}
}  // namespace llm_system