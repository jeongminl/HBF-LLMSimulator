#include "scheduler/scheduler.h"
#include <numeric>

namespace llm_system {
Scheduler::Scheduler(SystemConfig system_config, ModelConfig& model_config,
                     std::string expert_file_path, int total_batch_size,
                     int num_max_batched_token,  // per dp, total
                     int max_process_token)      // per dp, in sum
    : system_config(system_config),
      model_config(model_config),
      expert_file_path(expert_file_path),
      total_batch_size(total_batch_size),
      num_max_batched_token(num_max_batched_token),
      max_process_token(max_process_token) {
  dp_degree =
      system_config.num_device * system_config.num_node / (model_config.ne_tp_dg * model_config.pp_dg);
  batch_size_per_dp = total_batch_size / dp_degree;
  total_seq_num = 0;
  total_time = 0;
  real_data = false;
  disagg_system = system_config.disagg_system;
  initExpertList(expert_file_path);
};

bool Scheduler::hasSumSeq() {
  for (auto batchseq : running_queue) {
    if (batchseq->get_sum().size() != 0) {
      return true;
    }
  }
  return false;
}

void Scheduler::initRunningQueue() {
  running_queue.resize(0);
  for (int batch_idx = 0; batch_idx < dp_degree; batch_idx++) {
    BatchedSequence::Ptr sequences_metadata = BatchedSequence::Create(
        model_config.num_routed_expert, model_config.top_k, getPtr());
    running_queue.push_back(sequences_metadata);
  }
}

void Scheduler::pushDummySeq(int input_len, int output_len, int idx, int count) {
  // QUIRK (guide §16 -- not tracked in BUGS.md, this is a deliberate documentation-only
  // item, not an open bug): this whole block -- norm_dist_value, the rejection loop
  // below, and delta -- computes a per-sequence length-jitter value that is NEVER
  // applied: the three lines that would consume it (input_len = input_len - delta;
  // output_len = output_len + delta;) are commented out just below. Every call pays the
  // RNG + rejection-loop cost for no effect on input_len/output_len. Left as dead
  // computation intentionally, not removed: every reported sweep's results were
  // generated with jitter off, so deleting or enabling this now would both be behavior
  // changes that need a separate, deliberate decision (documented, not fixed, per
  // explicit instruction -- see ledgers/BUGS_FIXES.md, item G2).
  double norm_dist_value = getNormaldistribution();

  int delta = std::min(256, input_len);
  delta = std::min(delta, output_len) - 1;

  while (norm_dist_value <= -0.95 || norm_dist_value >= 0.95) {
    norm_dist_value = getNormaldistribution();
  }

  delta = delta * norm_dist_value;
  if (delta < 0) {
    delta += 1;
  } else {
    delta -= 1;
  }

  // to give sequence some randomness, you can insert delat value in to length by uncommenting below
  // delta = 0;
  // input_len = input_len - delta;
  // output_len = output_len + delta;

  if (output_len == 0) {
    return;
  }

  Sequence::Ptr new_seq = Sequence::Create(0, input_len, output_len);
  if(total_time > 0){
    try {
      new_seq->arrival_time = actual_arrival_time.at(cur_arrival_time_idx++);
    } catch (...) {
      new_seq->arrival_time = 1;
    }
  }
  if(system_config.reuse_kv_cache){
    static unsigned int seed = 777;
    static std::mt19937 generator(seed);
    static std::uniform_real_distribution<> distribution(0, system_config.kv_cache_reuse_rate);

    new_seq->current_len = input_len * system_config.kv_cache_reuse_rate;
  }

  if(system_config.prefill_mode){
    new_seq->output_len = input_len; // for prefill mode
    new_seq->total_len = input_len;
  }
  else if(system_config.decode_mode){
    // Steady-state decode: under continuous batching with arrival rate matching
    // completion rate (paper SS-III), in-flight queries are uniformly distributed
    // over their output lifetime, so the POPULATION's mean context is input +
    // output/2 at every tick (matching the optimizer's steady_ctx,
    // parallelism_optimizer.cpp:404-408) -- not start-of-generation (input), which
    // under-charged the context-linear attention cost by output/2 tokens (~10% of
    // context on SHORT, ~4% MID, ~0.5% LONG).
    //
    // The INITIAL fill must realize that population mean as a uniform SNAPSHOT
    // across the generation lifetime, not a single cohort all seeded at the
    // midpoint: seeding every dummy at exactly input+output/2 makes them all
    // advance in lockstep, so hittingQueue(N)'s periodic sampling always lands on
    // the same phase of the cycle instead of the intended steady-state average
    // (measured drift: SHORT's sampled context read 1988-1996 instead of the
    // intended ~1846). Staggering idx/count sequences evenly across [0, output_len)
    // reproduces the uniform-snapshot population at t=0 and keeps it uniform
    // forever after (steady-state refills below enter at start-of-generation,
    // maintaining the same age distribution as sequences complete and are replaced).
    //
    // The output_len-2 clamp keeps the stagger strictly inside the sequence's
    // lifetime: updateScheduler's completion check (current_len == total_len) runs
    // AFTER advancing current_len by one step, so a dummy seeded at exactly
    // total_len-1 would complete on its very first step, but one seeded at
    // total_len would never satisfy the equality test post-increment and would
    // become a zombie that never gets collected.
    if (!initial_fill_done) {
      long long offset = (long long)idx * output_len / std::max(count, 1);
      offset = std::min(offset, (long long)(output_len - 2));
      if (offset < 0) offset = 0;
      new_seq->current_len = input_len + offset;
    } else {
      new_seq->current_len = input_len; // steady-state refill: enters at start-of-generation
    }
  }
  new_seq->get_expert_from_list = false;
  sequence_queue.push_back(new_seq);
}

void Scheduler::pushSeq(int num_seq) {
  if (!real_data) {
    assertTrue(model_config.input_len < model_config.max_seq_len, "Invalid input_len (= " 
               + std::to_string(model_config.input_len) + ")" );
    if(model_config.input_len + model_config.output_len > model_config.max_seq_len){
      model_config.output_len = model_config.max_seq_len - model_config.input_len;
      std::cout << "output_len is modfied to " << model_config.output_len << std::endl;
    }
    for(int i = 0; i < num_seq; i ++){
      pushDummySeq(model_config.input_len, model_config.output_len, i, num_seq);
    }
  } else {
    pushRealSeq(num_seq);
  }
}

void Scheduler::clear() {
  sequence_queue.clear();
  running_queue.clear();
  total_token_in_expert.resize(0);
}

// num_sequences_metadata is the same as the data parallelism degree
void Scheduler::initializeDummyInput(int num_seq, int input_len,
                                     int output_len) {

  for (int batch_idx = 0; batch_idx < dp_degree; batch_idx++) {
    int each_num_seq = (num_seq + dp_degree - 1) / dp_degree;

    BatchedSequence::Ptr sequences_metadata = running_queue.at(batch_idx);

    sequences_metadata->add_dummy_sequence(each_num_seq, input_len, output_len);
  }
}

int Scheduler::getRandomExpertSeqId() {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, 4096 - 1);

  int seq_id = distribution(generator);
  seq_id %= total_seq_num;

  return seq_id;
}

std::set<int> Scheduler::getRandomExpert(int top_k) {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, model_config.num_routed_expert - 1);

  std::set<int> route;

  while (route.size() < top_k) {
    int seq_id = (distribution(generator) % model_config.num_routed_expert);
    route.insert(seq_id);
  }
  return route;
}

std::set<int> Scheduler::getEquallyDistributedExpert(int token_id, int top_k) {
  std::set<int> route;

  while (route.size() < top_k) {
    int seq_id = token_id++ % model_config.num_routed_expert;
    route.insert(seq_id);
  }
  return route;
}

std::set<int> Scheduler::getZipfianRandomExpert(std::vector<double> weight, int top_k) {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::discrete_distribution<int> distribution(weight.begin(), weight.end());

  // The Zipf weights are index-monotone (expert 0 hottest) and expert->device
  // placement is contiguous (expert.cpp: experts [i*16, i*16+16) -> device i),
  // so drawing the raw index would colocate ALL hot experts on device 0 —
  // an accidental artifact, not a modeled placement policy. At low batch
  // (llama4 LONG, ~31 active experts) that inflates the bottleneck device's
  // expert weight-stream count ~3x vs any balanced placement, since
  // per-iteration time is max-over-devices and expert weight-read cost is
  // token-count-independent. Decorrelate hotness from placement with a fixed
  // coprime-stride permutation of the drawn hotness rank: hot ranks 0..7 land
  // on 8 distinct devices (round-robin), preserving the skew distribution
  // itself and per-seed determinism.
  int n = model_config.num_routed_expert;
  static const int kStrideCandidates[] = {17, 19, 23, 29, 31, 1};
  int stride = 1;
  for (int s : kStrideCandidates) {
    // Coprimality alone admits s == identity whenever s ≡ 1 (mod n) (e.g. stride 17
    // with n in {8, 16}): that leaves hot experts contiguous, defeating the whole
    // point of this permutation. Reject those candidates too.
    if (std::gcd(s, n) == 1 && (s % n) != 1) { stride = s; break; }
  }

  std::set<int> route;

  while (route.size() < top_k) {
    int hotness_rank = (distribution(generator) % n);
    int expert_id = (int)(((long)hotness_rank * stride) % n);
    route.insert(expert_id);
  }
  return route;
}

double Scheduler::getNormaldistribution() {
  static unsigned int seed = 777;
  static std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0, 0.4);
  double number = distribution(generator);
  return number;
}

int Scheduler::getPoissondistribution(int request_per_second) {
  static unsigned int seed = 777;
  static std::default_random_engine generator(seed);
  std::poisson_distribution<> distribution(request_per_second);
  int number = distribution(generator);
  return number;
}

int Scheduler::getNumInjection() {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, 4);
  static std::uniform_int_distribution<int> get_seq(0, 4);

  int random = distribution(generator);
  int num_injection = 0;
  if (random == 0) {
    num_injection = get_seq(generator);
  }
  return num_injection;
}

void Scheduler::initializeRealInput(int num_seq) {
  for (int batch_idx = 0; batch_idx < dp_degree; batch_idx++) {
    int each_num_seq = (num_seq + dp_degree - 1) / dp_degree;

    BatchedSequence::Ptr sequences_metadata = running_queue.at(batch_idx);
    std::vector<int> seq_ids;
    for (int seq_id = 0; seq_id < each_num_seq; seq_id++) {
      seq_ids.push_back(getRandomExpertSeqId());
    }
    sequences_metadata->add_sequence(seq_ids);
  }
}

void Scheduler::pushRealSeq(int num_seq) {
  assertTrue(sequences_info.size() != 0, "Input trace is not set");
  for (int temp = 0; temp < num_seq; temp++) {
    int seq_id = 0;
    int input_len = 0;
    int output_len = 0;
    do {
      seq_id = getRandomExpertSeqId();
      input_len = sequences_info.at(seq_id)->Lin;
      output_len = sequences_info.at(seq_id)->Lout;
    } while (output_len == 512);

    Sequence::Ptr new_seq = Sequence::Create(seq_id, input_len, output_len);
    sequence_queue.push_back(new_seq);
  }
}

std::vector<BatchedSequence::Ptr> Scheduler::getAllMetadata() {
  return running_queue;
}

BatchedSequence::Ptr Scheduler::getMetadata(int dp_rank) {
  return running_queue.at(dp_rank);
}

BatchedSequence::Ptr Scheduler::getMaxMetadata(int num_expert, int top_k,
                                               Ptr scheduler) {
  BatchedSequence::Ptr sequences_metadata =
      BatchedSequence::Create(num_expert, top_k, scheduler);
  int seq_len = num_max_batched_token / batch_size_per_dp;
  // Same bug class as the committed fix in CHANGES.md item 68: integer division
  // floors to 0 once batch_size_per_dp > num_max_batched_token (8192, hardcoded at
  // eval/test.cpp's Scheduler::Create call), which makes Sequence::Create(0, 0)
  // trip sequence.cpp's assertTrue(input_len > 0, ...) -> process exit. run_experiments.py
  // classifies that crash as "OOM/Crash" (indistinguishable from a real capacity OOM),
  // so the batch search reads batch_size_per_dp=8192 as a spurious hard ceiling for any
  // cell whose true ceiling is actually higher. No-op below the wall (seq_len>=1 already).
  if (seq_len < 1) seq_len = 1;
  for (int seq_idx = 0; seq_idx < batch_size_per_dp; seq_idx++) {
    Sequence::Ptr seq = Sequence::Create(seq_len, seq_len);
    seq->num_process_token = seq_len;
    sequences_metadata->add(seq);
  }
  return sequences_metadata;
}

std::vector<BatchedSequence::Ptr> Scheduler::setMetadata() {
  bool process_gen = true;
  bool process_sum = true;

  std::vector<Sequence::Ptr> execution_queue;
  execution_queue.resize(running_queue.size());

  if (disagg_system == false) {
    if (hasSumSeq()) {
      process_gen = false;
      process_sum = true;
    } else {
      process_gen = true;
      process_sum = false;
    }
  }

  for (int batch_idx = 0; batch_idx < running_queue.size(); batch_idx++) {
    BatchedSequence::Ptr batch = running_queue.at(batch_idx);

    int num_gen_seq = batch->get_gen().size();
    std::vector<Sequence::Ptr> gen_seq = batch->get_gen();
    for (auto& seq : gen_seq) {
      if (process_gen) {
        if (seq->gen_start_time == 0) {
          seq->num_process_token = 1;
        } else if (seq->gen_start_time <= total_time) {
          seq->num_process_token = 1;
        }
      }
    }

    int process_token = num_max_batched_token - batch->get_gen_process_token();

    std::vector<Sequence::Ptr> sum_seq = batch->get_sum();
    if (sum_seq.size() != 0) {
      int num_sum_seq = sum_seq.size();
      // max_process_token <= 0 (config.yaml's default) means "no per-step cap" --
      // integer-dividing by num_sum_seq used to floor to 0, permanently stalling
      // every prefill sequence's current_len -- fixed 2026-07-02 (CHANGES.md item 68).
      // INT_MAX is then clamped per-sequence by the std::min below, same as an
      // explicit cap.
      int num_process = (max_process_token > 0)
          ? std::max(1, max_process_token / num_sum_seq)
          : INT_MAX;
      for (int seq_idx = 0; seq_idx < num_sum_seq; seq_idx++) {
        if (process_sum) {
          sum_seq[seq_idx]->num_process_token =
              std::min(num_process, sum_seq[seq_idx]->input_len -
                                        sum_seq[seq_idx]->current_len);
        }
      }
    }
  }
  return running_queue;
}

std::vector<Sequence::Ptr> Scheduler::updateScheduler(time_ns time) {
  std::vector<Sequence::Ptr> token_list;

  for (int batch_idx = 0; batch_idx < running_queue.size(); batch_idx++) {
    BatchedSequence::Ptr batch = running_queue.at(batch_idx);
    batch->update(time);

    // remove seq of which generation is done
    for (auto seq : batch->get_seq()) {
      if (seq->current_len == seq->total_len) {
        batch->pop(seq);
        token_list.push_back(seq);
      } else if (seq->current_len == seq->input_len) {
        if (!seq->record) {
          token_list.push_back(seq);
          seq->record = true;
        }
      }
    }
  }
  return token_list;
}

std::vector<Sequence::Ptr> Scheduler::updateSchedulerSumGenSplit(time_ns time) {
  std::vector<Sequence::Ptr> token_list;

  for (int batch_idx = 0; batch_idx < running_queue.size(); batch_idx++) {
    BatchedSequence::Ptr batch = running_queue.at(batch_idx);
    // remove seq of which generation is done
    for (auto seq : batch->get_sum()) {
      seq->update(time);
      seq->setGenStartTime(total_time + time);
      if (seq->current_len == seq->total_len) {
        batch->pop(seq);
        token_list.push_back(seq);
      } else if (seq->current_len == seq->input_len) {
        if (!seq->record) {
          token_list.push_back(seq);
          seq->record = true;
        }
      }
    }
  }
  return token_list;
}

void Scheduler::fillRunningQueue(time_ns time) {

  static int rotate = 0;
  int num_empty_seq = total_batch_size - getBatchSize();
  num_empty_seq = std::min(num_empty_seq, int(sequence_queue.size()));
  for (int i = 0; i < num_empty_seq; i++) {
    auto seq = sequence_queue.begin();

    Sequence::Ptr seqPtr = *seq;

    for (int i = 0; i < running_queue.size(); i++) {
      BatchedSequence::Ptr batch =
          running_queue[rotate++ % running_queue.size()];

      if (batch->get_num_seq() < batch_size_per_dp) {
        sequence_queue.erase(seq);
        if (system_config.use_inject_rate) {
          seqPtr->queueing_delay = total_time - seqPtr->arrival_time;
        }
        batch->add(seqPtr);
        break;
      }
    }
  }
}

void Scheduler::printStatus() {
  BatchedSequence::Ptr batseq = running_queue.at(0);
  std::cout << "Current status: " << std::to_string(batseq->get_process_token())
            << " | Sum: " << std::to_string(batseq->get_sum().size()) << ", "
            << std::to_string(batseq->get_sum_process_token())
            << " | Gen : " << std::to_string(batseq->get_gen().size()) << ", "
            << std::to_string(batseq->get_gen_process_token()) << ", average: "
            << std::to_string(batseq->get_average_sequence_length())
            << std::endl;
}

int Scheduler::getBatchSize() {
  int size = 0;
  for (auto batch_seq : running_queue) {
    size += batch_seq->get_num_seq();
  }
  return size;
}

int Scheduler::getSumSize() {
  int size = 0;
  for (auto batch_seq : running_queue) {
    auto seq_list = batch_seq->get_sum();
    for (auto seq : seq_list) {
      if (seq->num_process_token != 0) {
        size++;
      }
    }
  }
  return size;
}

int Scheduler::getGenSize() {
  int size = 0;
  for (auto batch_seq : running_queue) {
    auto seq_list = batch_seq->get_gen();
    for (auto seq : seq_list) {
      if (seq->num_process_token != 0) {
        size++;
      }
    }
  }
  return size;
}

  int Scheduler::getAverageSeqlen() {
    int len = 0;
    for (auto batch_seq : running_queue) {
      len += batch_seq->get_total_sequence_length();
    }

    int total_batch = getBatchSize();

    if (total_batch != 0) {
      len /= total_batch;
    } else {
      len = 0;
    }

    return len;
  }

  int Scheduler::getNumProcessToken() {
    int num_token = 0;
    for (auto batch_seq : running_queue) {
      num_token += batch_seq->get_process_token();
    }
    return num_token;
  };

  void Scheduler::fillSequenceQueue(time_ns iter_time, time_ns total_time) {
    // add reqeust
    static int iter = 0;
    int num_request_to_inject = 0;
    if (system_config.use_inject_rate) {
      if (iter_time != 0) {
        time_ns start_time = total_time - iter_time;
        time_ns end_time = total_time;
        for (int i = 0; i < actual_arrival_time.size(); i++) {
          if (actual_arrival_time[i] >= start_time &&
              actual_arrival_time[i] < end_time) {
            num_request_to_inject++;
          }
        }
      } else {
        num_request_to_inject = total_batch_size * 0.7 - getBatchSize();
        num_request_to_inject = std::max(num_request_to_inject, 0);
      }
    } else {
      num_request_to_inject = total_batch_size - getBatchSize();
    }
    pushSeq(num_request_to_inject);
  }

  void Scheduler::hittingQueue(int iter) {
    for (int i = 0; i < iter; i++) {
      setMetadata();
      // run
      updateScheduler(0);

      // fill
      fillSequenceQueue();
      fillRunningQueue();
    }
  }

  void Scheduler::getActualArrivalTime(int num_iter) {
    // QUIRK (guide §16 -- not tracked in BUGS.md, this is a deliberate documentation-only
    // item, not an open bug): a Poisson *process*'s inter-arrival times are
    // exponentially distributed, not Poisson-distributed -- std::poisson_distribution
    // below draws integer counts with the right mean (average_ns_per_request) but the
    // wrong shape/variance for inter-arrival gaps. Also note request_per_second == 0
    // divides by zero at average_ns_per_request just below. Not exercised by any
    // run_experiments.py sweep (injection_rate is always 0 there); live only on a bare
    // `./run config.yaml` with injection_rate > 0. Documented, not fixed, per explicit
    // instruction -- see ledgers/BUGS_FIXES.md, item G1, for the intended
    // exponential_distribution(1.0 / average_ns_per_request) replacement.
    static unsigned int seed = 777;
    static std::default_random_engine generator(seed);
    double request_per_second = system_config.request_per_second;
    int average_ns_per_request = 1e+9 / request_per_second;
    std::poisson_distribution<> distribution(average_ns_per_request);

    std::vector<time_ns> inter_arrival_times(num_iter * 100);  // temp *10
    for (time_ns& time : inter_arrival_times) {
      time = distribution(generator);
    }

    std::vector<time_ns> actual_arrival_times(num_iter * 100);
    std::partial_sum(inter_arrival_times.begin(), inter_arrival_times.end(),
                     actual_arrival_times.begin());

    for (const time_ns& time : actual_arrival_times) {
      actual_arrival_time.push_back(time);  // second -> ns
    }
    cur_arrival_time_idx = 0;
  }

  void Scheduler::initExpertList(std::string expert_file_path) {
    std::ifstream openFile;

    if (!expert_file_path.compare("none")) {
      std::cout << "Using synthesis trace" << std::endl;
      real_data = false;
      real_expert_data = false;
      return;
    }

    real_data = true;
    real_expert_data = true;

    openFile.open(expert_file_path);
    if (!openFile.is_open()) {
      std::cout << "Using trace of mixtral, use random expert selection"
                << std::endl;
      real_expert_data = false;
      std::string file_path;
      file_path =
          "../expert_data/experts_mixtral_" + model_config.dataset + ".csv";

      openFile.open(file_path);
      if (!openFile.is_open()) {
        std::cout << file_path << std::endl;
        std::cout << "Cannot open mixtral, use random dataset" << std::endl;
        real_data = false;
        return;
      }
    }

    SequenceInfo::Ptr cur_seq = SequenceInfo::Ptr(new SequenceInfo);
    std::string line;

    int num_layer = model_config.expert_freq
        ? model_config.num_layers / model_config.expert_freq
        : model_config.num_layers;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Loading sequences Start" << std::endl;

    bool process_sum = false;
    std::vector<std::vector<int>> sum_expert_list;

    int idx = 0;

    while (getline(openFile, line)) {
      std::istringstream iss(line);
      std::string key;

      int top_1, top_2;

      iss >> key;

      if (key == "Lin") {
        iss >> cur_seq->Lin;
      } else if (key == "Lout") {
        iss >> cur_seq->Lout;
        sequences_info.push_back(cur_seq);
        cur_seq = SequenceInfo::Ptr(new SequenceInfo);
      } else if (real_expert_data && key == "Sum") {
        cur_seq->expert_list.resize(cur_seq->Lin * num_layer);
        process_sum = true;
        idx = 0;
      } else if (real_expert_data && key == "Gen") {
        process_sum = false;
      } else if (real_expert_data && !(iss >> top_2).fail()) {
        top_1 = std::stoi(key);
        if (process_sum) {
          int token_id = idx % cur_seq->Lin;
          int layer_id = idx / cur_seq->Lin;
          cur_seq->expert_list.at(token_id * num_layer + layer_id) = {top_1,
                                                                      top_2};
          idx++;

        } else {
          cur_seq->expert_list.push_back({top_1, top_2});
        }
      } else if (key.substr(0, 4) == "seq_") {
        total_seq_num += 1;
      }
    }

    if (!real_expert_data) {
      for (auto& seq_info : sequences_info) {
        for (int token_id = 0; token_id < seq_info->Lin + seq_info->Lout;
             token_id++) {
          for (int layer_id = 0; layer_id < model_config.num_layers;
               layer_id++) {
            std::set<int> expert_list = getRandomExpert(model_config.top_k);
            std::vector<int> expert_vec;
            for (auto idx : expert_list) {
              expert_vec.push_back(idx);
            }
            seq_info->expert_list.push_back(expert_vec);
          }
        }
      }
    }
  };

}  // namespace llm_system