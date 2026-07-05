#pragma once
#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>


#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "scheduler/sequence.h"

namespace llm_system {

class Scheduler : public std::enable_shared_from_this<Scheduler> {
 public:
  using Ptr = std::shared_ptr<Scheduler>;

  [[nodiscard]] static Ptr Create(SystemConfig system_config,
                                  ModelConfig& model_config,
                                  std::string expert_file_path = "",
                                  int batch_size = 4096,
                                  int num_max_batched_token = 4096 * 64,
                                  int max_process_token = 8192 * 4) {
    Ptr ptr = Ptr(new Scheduler(system_config, model_config, expert_file_path,
                                batch_size, num_max_batched_token,
                                max_process_token));
    ptr->initRunningQueue();
    return ptr;
  };

  Ptr getPtr() { return shared_from_this(); }

  void clear();

  void pushSeq(int num_seq);
  void pushDummySeq(int input_len = 256, int max_len = 1024, int idx = 0, int count = 1);
  void pushRealSeq(int num_seq);

  // paper2 stochastic workload sampler (system_config.paper2_workload): draws
  // one sequence's (input_len, output_len) pair from a truncated-normal total
  // context length crossed with a Beta-distributed output/context ratio. See
  // scheduler.cpp for the full derivation. size_biased selects the
  // length-biased sampling used ONLY for the initial-fill population (the
  // in-flight population at a random instant is length-biased, per the
  // renewal/inspection paradox -- see scheduler.cpp); steady-state refills use
  // size_biased=false. Never called when paper2_workload is false, so paper1
  // behavior/RNG streams are entirely unaffected by this method's existence.
  void sampleWorkloadLengths(int& input_len, int& output_len, bool size_biased);

  void initializeDummyInput(int num_seq, int input_len, int output_len);

  void initializeRealInput(int num_seq);

  void hittingQueue(int iter);

  bool hasSumSeq();

  std::vector<BatchedSequence::Ptr> getAllMetadata();
  BatchedSequence::Ptr getMetadata(int dp_rank);
  BatchedSequence::Ptr getMaxMetadata(int num_expert, int top_k, Ptr scheduler = nullptr);

  std::vector<BatchedSequence::Ptr> setMetadata();
  std::vector<Sequence::Ptr> updateScheduler(time_ns time = 0);
  std::vector<Sequence::Ptr> updateSchedulerSumGenSplit(time_ns time = 0);

  void printStatus();

  // random
  std::set<int> getRandomExpert(int top_k);
  std::set<int> getZipfianRandomExpert(std::vector<double> skewness_weight, int top_k);
  std::set<int> getEquallyDistributedExpert(int token_id, int top_k);
  int getRandomExpertSeqId();
  static double getNormaldistribution();
  static int getPoissondistribution(int request_per_second);
  static int getNumInjection();

  void getActualArrivalTime(int num_iter);

  void fillSequenceQueue(time_ns iter_time = 0, time_ns total_time = 0);

  // time to fill running queue
  void fillRunningQueue(time_ns time = 0);

  int getBatchSize();
  int getSumSize();
  int getGenSize();

  int getAverageSeqlen();
  int getNumProcessToken();

  int total_seq_num;

  time_ns total_time;

  int total_batch_size;
  int batch_size_per_dp;  // per dp
  int num_max_batched_token;
  int max_process_token;
  int dp_degree;

  bool real_data;
  bool real_expert_data;

  // Set once by Cluster::runIteration after the initial fillSequenceQueue()/
  // fillRunningQueue() population (before hittingQueue(10000)): distinguishes the
  // initial-fill dummy population (which staggers current_len across the generation
  // lifetime, see pushDummySeq) from steady-state refills (which enter at
  // start-of-generation). See pushDummySeq's decode_mode branch for the full model.
  bool initial_fill_done = false;

  // paper2 admission-token counter: sum of input_len over sequences admitted
  // into a running batch (BatchedSequence::add, called from fillRunningQueue)
  // since the counter was last reset. Reset at the top of setMetadata() --
  // the cleanest "step" boundary, since setMetadata() is what snapshots
  // running_queue for the hardware-timing pass that follows (run(metadata)),
  // and fillRunningQueue() is the sole place sequences transition into
  // running_queue. Exists so a LATER change can replace
  // getKVWriteDuration's num_seq/output_len amortization (layer_impl.h:162,
  // which is only exact for uniform-length batches) with an exact per-step
  // admitted-token count; nothing reads this counter yet. Maintained
  // unconditionally (not guarded behind paper2_workload) because upkeep is a
  // single integer add per admitted sequence and, since nothing consumes it
  // yet, it cannot perturb any existing timing output for paper1 -- verified
  // by the byte-identity regression check.
  long long admitted_prefill_tokens_this_step = 0;
  long long getAdmittedPrefillTokensThisStep() const {
    return admitted_prefill_tokens_this_step;
  }

  std::vector<Sequence::Ptr> sequence_queue;
  std::vector<BatchedSequence::Ptr> running_queue;

  std::vector<time_ns> actual_arrival_time;
  int cur_arrival_time_idx;
  std::vector<int> total_token_in_expert;

  SystemConfig system_config;

  // paper2 stochastic workload sampler's dedicated RNG, seeded from
  // system_config.workload_seed. Kept entirely separate from the getRandom*/
  // getNormaldistribution/getPoissondistribution static-local generators
  // (all seed=777, but independent streams/state) so that turning on
  // paper2_workload cannot perturb expert-routing or any other existing
  // random draw's sequence -- paper1 (paper2_workload=false) never calls
  // sampleWorkloadLengths, so workload_rng is constructed but never advanced,
  // making its mere presence a no-op for paper1 bit-identity. Declared after
  // system_config (member init order follows declaration order) so its
  // initializer can read system_config.workload_seed.
  std::mt19937 workload_rng;

  ModelConfig& model_config;
  std::vector<SequenceInfo::Ptr> sequences_info;
  std::string expert_file_path;
  
  void initRunningQueue(); // juhwan
  
 private:
  Scheduler(SystemConfig system_config, ModelConfig& model_config,
            std::string expert_file, int batch_size, int num_max_batched_token,
            int max_process_token);

  void initExpertList(std::string expert_file_path);
  // void initRunningQueue();

  bool disagg_system = false;
};

}  // namespace llm_system