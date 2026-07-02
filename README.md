# HBF-LLMSimulator

## Overview

HBF-LLMSimulator is a C++ cycle-accurate simulator for Large Language Model inference on heterogeneous memory systems. It extends the original LLMSimulator (MICRO 2024) with **High-Bandwidth Flash (HBF)** memory modeling — flash-backed stacks as a high-capacity, lower-bandwidth alternative to HBM — and a constraint-aware parallelism optimizer.

The central research question: can flash-backed memory (NAND stacks on the HBM die) substitute for HBM to serve very large LLMs at scale, and under what SLO constraints does it remain competitive?

Supported attention mechanisms: MHA, GQA, MQA, MLA (DeepSeek-style with optional KV compression and absorbed projection).
Supported models: Llama 3/4, DeepSeek V3, Mixtral, and others configurable via `config.yaml`.
Integrated with patched [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) for detailed DRAM timing.

## Memory Presets

Set `system.memory_type` in `config.yaml` to one of five presets per GPU:

| Preset | Stacks | Total Capacity | Flash Read BW | Notes |
|--------|--------|---------------|---------------|-------|
| `HBM4`  | 8 HBM  | 288 GB  | 12.8 TB/s     | Baseline; symmetric read/write |
| `HBF`   | 1 HBM + 7 HBF | 3,620 GB | 11.2 TB/s | 1 µs 4 KiB page-read latency |
| `HBF+`  | 8 HBF  | 4,096 GB | 12.8 TB/s    | No HBM; 40 MB logic-die SRAM per stack for activations |
| `CONV`  | 1 HBM + 7 flash | 3,620 GB | 2.45 TB/s | 3 µs page-read latency |
| `CONV+` | 8 flash | 4,096 GB | 2.80 TB/s   | No HBM; 40 MB logic-die SRAM per stack |

Data placement rules:
- **Model weights and KV-cache** reside on flash (large capacity, read-dominant).
- **Intermediate activations** use the 1 reserved HBM stack (HBF/CONV) or logic-die SRAM (HBF+/CONV+).
- **SRAM per GPU**: 8 stacks × 40 MB = 320 MB total, used as the activation buffer. This is a hard batch-size constraint for HBF+/CONV+.
- **Staging buffer**: each flash stack has a 3.13 MB SRAM prefetch buffer (25 MB total per GPU). KV reads use double-buffered chunked prefetch — only one page-read latency per chunk.
- **Flash writes** are never used for intermediate data; only KV-cache from newly admitted prefill queries is written to flash.

## Key Features

- **Parallelism optimizer**: sweeps TP × PP × DP × EP, hard-gated on memory capacity and SRAM limits, ranking capacity-feasible candidates by estimated throughput; the live discrete-event simulator's measured TPOT remains the sole arbiter of TPOT-SLO pass/fail.
- **Disaggregated prefill-decode**: decode-focused simulation with KV-cache write penalty modeling (write time minus overlap with attention computation).
- **Continuous batching**: two-phase batch-size search (analytic capacity/SLO estimate, then simulator-verified exponential probe + bisect) for maximum batch size under each SLO.
- **Pipeline parallelism** (with cross-stage latency correctly propagated to the true per-token critical path) and **chunked attention** with configurable chunk size.
- **Expert parallelism** (EP) independent of tensor parallelism; valid EP degrees verified against `expert.cpp` constraints.

## Workloads and SLOs

Three production-representative workload profiles (⟨input len, output len⟩):

| Name | Dataset | Input / Output |
|------|---------|----------------|
| SHORT | ShareGPT | ⟨1660, 373⟩ |
| MID   | LongBench | ⟨5900, 499⟩ |
| LONG  | English summarization | ⟨103500, 1100⟩ |

SLO sweep: 0.05 s, 0.1 s, 0.2 s, Offline (86400 s).

Baseline for normalized metrics: 8-GPU HBM4 under the same SLO.

## Metrics

1. **Max Per-GPU Batch Size** — two-phase search (exponential then binary).
2. **System Throughput** — per-GPU tokens per second at max batch in steady-state.
3. **Runtime Breakdown** (under 0.1 s SLO): attention, FFN, KV write, communication, others.
4. **SLO Sensitivity** (LONG workload): TPS and batch size vs SLO, normalized to HBM4 baseline.
5. **Write Endurance (PEC)** — 3-year program/erase count = (bytes/s × 3 years) / flash capacity; target < 100K SLC cycles.

## Prerequisites

- g++ 11.4+ with C++17
- cmake 3.16+
- Python 3.8+ with PyYAML

## Building

```bash
git clone <this-repo>
cd HBF-LLMSimulator
git submodule update --init --recursive

# Apply the Ramulator 2.0 PIM patch
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch
cd ../../..

mkdir build && cd build
cmake ..
make -j
cd ..
```

## Running a Single Simulation

Edit `config.yaml`, then:

```bash
./run config.yaml > output.log
```

Key `config.yaml` fields:

```yaml
system:
  memory_type: HBF          # HBM (plain, upstream-compatible) | HBM4 | HBF | HBF+ | CONV | CONV+
  num_device: 8             # GPUs per node
  num_node: 1
  chunk_size: 0             # 0 = auto (full SRAM staging capacity per chunk)
  tpot_slo: 0.1             # target TPOT in seconds; large value (e.g. 86400) = offline

model:
  model_name: llama3_405B   # model preset key
```

Exit code 0 = success; 1 = OOM (capacity or SRAM limit exceeded).

## Running Sweeps

**Full experiment sweep** — all GPU counts (1/2/4/8/16), all presets, all workloads and SLOs:
```bash
python3 run_experiments.py
```

Simulation CSV outputs go to `data/`. The sweep script invokes the optimizer, binary-searches for max batch, then runs the final simulation and extracts metrics.

See the source paper (`Exploring_High-Bandwidth_Flash_for_Modern_LLM_Inference_Opportunities_and_Challenges.pdf`, in this repo) for the complete hardware and simulation specification — the ground truth this simulator implements.

## Contact
Of the original (LLMSimulator)[https://github.com/scale-snu/LLMSimulator]:

Sungmin Yun — sungmin.yun@snu.ac.kr

Kwanhee Kyung — kwanhee.kyung@scale.snu.ac.kr

Juhwan Cho — juhwan.cho@scale.snu.ac.kr

Of this modified version, based on Son et. al. "Exploring High-Bandwidth Flash for Modern LLM Inference:  Opportunities and Challenges" published on CAL'26:

Jeongmin Lee — jeongminl@kaist.ac.kr

## Based On

MICRO 2024: "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching."
