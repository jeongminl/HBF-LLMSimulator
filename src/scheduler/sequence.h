#pragma once
#include <chrono>  // for timing of loading sequence
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/type.h"

namespace llm_system {

struct SequenceInfo {
  using Ptr = std::shared_ptr<SequenceInfo>;
  int Lin;
  int Lout;
  int cur;
  std::vector<std::vector<int>> expert_list;
};

class Sequence {
 public:
  using Ptr = std::shared_ptr<Sequence>;

  inline static int count_id = 0;

  [[nodiscard]] static Ptr Create(int expert_seq_id = 0, int input_len = 256,
                                  int output_len = 256) {
    return Ptr(new Sequence(expert_seq_id, input_len, output_len));
  }

  // Sequence operator=(Sequence seq) {
  //   Sequence temp;
  //   temp.input_len = seq.input_len;
  //   temp.max_len = seq.max_len;
  //   temp.num_process_token = seq.num_process_token;
  //   temp.current_len = seq.current_len;
  //   return temp;
  // }

  int id;
  int expert_seq_id;
  int input_len;
  int output_len;
  int total_len;
  int num_process_token;
  int current_len;

  int num_sum_iter;
  time_ns gen_start_time;
  time_ns arrival_time;
  time_ns first_token_time;
  time_ns end_token_time;
  time_ns queueing_delay;

  void update(time_ns time);
  void setGenStartTime(time_ns time);

  bool record;
  bool sum_stage;
  bool get_expert_from_list;

 private:
  Sequence(int expert_seq_id, int input_len, int output_len);
  std::vector<std::vector<int>> expert_list;  // 2차원 벡터 type,sequence안의
                                              // token 수만큼 expert들이 존재.
};

// TODO: count_id 초기화를 scheduler에서하든 해야함
// int Sequence::count_id = 0;

class Scheduler;
class BatchedSequence {
 public:
  // static int cur;
  using Ptr = std::shared_ptr<BatchedSequence>;
  using Scheduler_ptr = std::shared_ptr<Scheduler>;

  [[nodiscard]] static Ptr Create(int num_expert = 8, int top_k = 2,
                                  Scheduler_ptr scheduler = nullptr) {
    return Ptr(new BatchedSequence(num_expert, top_k, scheduler));
  }

  BatchedSequence(int num_expert, int top_k, Scheduler_ptr scheduler)
      : num_expert(num_expert), top_k(top_k), scheduler(scheduler) {
    for (int i = 0; i < num_expert; i++) {  // initialize
      local_num_token_in_expert.push_back(0);
      num_token_in_expert.push_back(0);
      num_token_in_expert_micro.push_back(0);  // MOE_TAG_FIX_SPEC
    }
    cur_layer = 0;
  }

  void add(Sequence::Ptr seq);

  std::vector<Sequence::Ptr> get_seq();
  std::vector<Sequence::Ptr> get_sum();
  std::vector<Sequence::Ptr> get_gen();
  void update_expert(int num_expert, int top_k, bool need_new_expert);
  void clear_expert();
  void update(time_ns time);
  void pop(Sequence::Ptr seq);
  int get_num_seq();
  int get_process_token();
  // BUGS #30 (CB1): widened to int64_t -- these sum num_process_token/
  // current_len (each individually bounded by max_seq_len, safe as int)
  // across an entire batch, which at the large batch sizes the offline/
  // capacity-search paths can reach (tens of thousands of sequences x
  // up to ~131072 current_len) can exceed INT32_MAX and silently wrap.
  int64_t get_gen_process_token();
  int64_t get_sum_process_token();
  int get_average_sequence_length();
  int64_t get_total_sequence_length();

  void add_dummy_sequence(int num_seq, int input_len, int output_len);

  void add_sequence(std::vector<int> seq_ids);

  int cur_layer;  // 이 batch의 현재 layer와 stage
  int num_expert;
  int top_k;
  std::vector<int> local_num_token_in_expert;
  std::vector<int> num_token_in_expert;
  std::vector<int> num_token_in_expert_micro;  // MOE_TAG_FIX_SPEC: first ceil(tokens/pp) tokens' activation
  std::vector<Sequence::Ptr> sequence;
  std::vector<int> seq_ids;
  Scheduler_ptr scheduler;

  // paper2 CPU-memory/NVLink-C2C KV offload tier: fraction of this dp-shard's
  // reservation-based total KV bytes that live in CPU memory (beyond the
  // local HBM-KV budget), recomputed once per scheduler step in
  // Scheduler::setMetadata() -- see that function's doc comment. 0.0 (the
  // default) whenever cpuKvOffloadActive() is false, i.e. a complete no-op
  // for paper1 and every non-offload paper2 config. Consumed by the GPU
  // attention gen paths (attention_gen_impl.cpp) via
  // hbmKvOffloadReadDuration() (hardware/layer_impl.h).
  double p2_kv_offload_fraction = 0.0;
};

}  // namespace llm_system