# Paper Ground Truth — Hardware, Methodology, and Prose Claims

"Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges" — IEEE CAL,
DOI 10.1109/LCA.2026.3705817. Son et al. (POSTECH/ETH). 4-page letter.

Non-figure ground truth extracted from the paper (fresh extraction from the PDF alone, no repo
analysis docs consulted). For figure-reading data (Figs. 3-7 tables), see
`paper_figure_readings.md` — that file's numbers were cross-validated against this extraction and,
where the two conflicted, corrected in favor of this document's independent re-derivation (see its
Figure 6 note). This document previously existed as `PAPER_ANCHOR_SHEET.md`; renamed/split
2026-07-06 so figure-reading data lives in one file and non-figure ground truth in another.

Preset naming (as printed): HBM4, CONV, CONV+, HBF, HBF+. Table I column sub-header reads
"(HBM/HBF)" — "HBM", not "HBM4".

---

## 1. Hardware constants

### Table I — "GPU-memory configurations of evaluated systems" (transcribed exactly)
Columns: GPU | # Stacks (HBM/HBF) | Capacity [GB] | RD BW [TB/s] (HBM/HBF) | WR BW [TB/s] (HBM/HBF)

| GPU | # Stacks (HBM/HBF) | Capacity [GB] | RD BW [TB/s] (HBM/HBF) | WR BW [TB/s] (HBM/HBF) |
|------|------|------|------|------|
| HBM4  | 8/0 | 288   | 12.8 / –  | 12.8 / –  |
| CONV  | 1/7 | 3,620 | 1.6 / 2.45 | 1.6 / 0.073 |
| CONV+ | 0/8 | 4,096 | – / 2.80  | – / 0.084 |
| HBF   | 1/7 | 3,620 | 1.6 / 11.2 | 1.6 / 0.112 |
| HBF+  | 0/8 | 4,096 | – / 12.8  | – / 0.128 |

Unit flag: capacities printed as [GB]; GB vs GiB never disambiguated. Text uses 36-GB, 512 GB,
288 GB, 746 GB. Checks: 8×36=288; 1×36+7×512=3,620; 8×512=4,096.

### Per-stack values (§II/§III text)
- HBM4 stack: 36-GB, 1.6-TB/s read. "eight HBM4 stacks per GPU, each providing 36-GB capacity and
  1.6-TB/s read bandwidth." → 288 GB, 12.8 TB/s aggregate.
- HBF stack: 512 GB, 1.6 TB/s read per stack (Sandisk). "HBF and HBF+ replace seven and all eight
  HBM4 stacks … with HBF stacks offering 512 GB of capacity and 1.6 TB/s of read bandwidth per stack."
- HBF read microarch: "we assume 25 planes per HBF core die and a 1-μs 4-KiB page-read latency."
- HBF write: "we assume a 100-μs page-program latency for all HBF configurations based on [9]."
  Per-stack write ≈ 0.112/7 = 0.128/8 = 0.016 TB/s.
- CONV/CONV+ derivation: "We assume 16 planes per die [10] and a 3-μs page-read latency [9],
  achieving per-stack read bandwidth of 0.35 TB/s." (7×0.35=2.45; 8×0.35=2.80). Same 100-μs program
  latency; lower write BW from 16 vs 25 planes (ratio 16/25≈0.64).
- CONV/CONV+ "identical to HBF and HBF+, respectively, except for each stack's read and write
  bandwidth."

### Reference NAND cited
- Samsung Z-NAND [9]: 87.4-GB/s read, 16 dies. Kioxia XL-Flash [10]: 262-GB/s read, 16 dies.

### SRAM (exact sentences)
- 40-MB logic-die SRAM: "HBF reserves one HBM stack to store intermediate data, whereas HBF+
  leverages 40-MB SRAM on the logic die instead." Aggregate: "…despite the limited SRAM capacity
  (320 MB total)." → 40 MB/stack × 8 = 320 MB total (HBF+). Also named per-stack constraint: "the
  limited SRAM-buffer capacity (e.g., 40 MB) bottlenecks batch-size scaling in HBF+."
- 3.13-MB staging buffer (per stack): "each HBF stack uses a dedicated 3.13-MB SRAM staging buffer
  on its logic die to prefetch and double-buffer read data."

### Power / compute / interconnect / endurance
- HBF stack < 80 W; HBM stack 40 W.
- No explicit FLOPS printed. All systems "share the same state-of-the-art GPU-core architecture
  [5]" = NVIDIA DGX Rubin NVL8. Rubin-class, no FLOPS number → simulator compute constant is
  unanchored.
- NVLink 1,800 GB/s intra-node (≤8 GPUs/node); InfiniBand 100 GB/s inter-node. No interconnect
  latencies stated.
- SLC endurance 100K P/E cycles; eval window 3 years (GPU-warranty); retention-relaxed KV writes
  [16]. HBF density "more than 14×… (e.g., 512 GB) compared to HBM4 (36 GB)."

---

## 2. Models evaluated (§I–§III)

- Llama 3 405B — dense. Single FFN over all tokens. No GB footprint, no head/kv-head counts, no
  precision stated in this paper.
- Llama 4 Maverick — MoE. Model size 746 GB vs GPU 288 GB (§I). MoE routing only generic (Fig 1c
  "Top-k Gating", "Expert FFN 1…E"). No numeric experts/top_k/kv-heads/GQA/context/iRoPE/precision
  in this paper — must come from refs [2]/[3]. Quantization mentioned only as a general
  optimization.

---

## 3. Methodology (§III, quoted)

### Workloads ⟨L_IN, L_OUT⟩
"we vary each query's ⟨L_IN, L_OUT⟩ values across ⟨1,660, 373⟩ (SHORT), ⟨5.9K, 499⟩ (MID), and
⟨103.5K, 1.1K⟩ (LONG) based on the average values in shareGPT [13] (SHORT), LongBench [14] (MID),
and English summarization [15] (LONG)."
- SHORT 1,660/373 · MID 5.9K/499 · LONG 103.5K/1.1K
- "We model a constant query arrival rate matching the completion rate… steady-state performance
  analysis under continuous batching."

### SLO / throughput / batch
- TPOT ≤ 0.1 s default. Fig 6 sweeps {0.05, 0.1, 0.2 s, offline}. Offline = "complete within 24
  hours (i.e., no TPOT SLO)."
- Throughput = per-GPU TPS (decode token generation). Batch = max per-GPU batch meeting SLO + all
  constraints.

### Parallelism selection (exact)
"each evaluated system selects the parallelism configuration that maximizes the achievable system
throughput subject to all constraints, including SLO requirements (e.g., TPOT ≤ 0.1 second),
GPU-memory capacity, and on-die SRAM limits (in HBF+ and CONV+), by combining data, tensor,
pipeline, and expert parallelism." → search DP×TP×PP×EP, maximize throughput s.t. {SLO, capacity,
SRAM}.

### MoE parallelism semantics (confirmed vs CB1 reference, 2026-07-08)

The authors' model uses **no orthogonal expert-parallel degree**. Expert parallelism =
`(ne_tp·dp)/e_tp` **by construction**; `dp` is derived from the device count, never specified.
Routed-expert weight per device = `num_routed·e_tp/(ne_tp·dp)` — **`dp` reduces per-device expert
weight** (so e.g. `tp1/dp8` is an 8-way expert-parallel placement, 16 of 128 experts/device). The
expert all-to-all **pools tokens across `dp` replicas** onto disjoint expert owners. Asymmetry:
**routed** experts absorb `dp` (above), but **shared** experts and the `e_tp` tensor-shard stay
**within a replica** (`non_moe_device_list`, `use_dp=true`). This fork matches the authors' reference
tool **CB1 (`LLMSimulator_HBF`) exactly** — verified `expert.cpp:42,50-51`, `decoder.cpp:137-139`,
`parallelism_optimizer.cpp:174` (all unmodified EP-across-DP). The alternative "dp = pure
replication" model (`MOE_DP_FIX_SPEC.md`, Model A) was rejected as a *divergence* from the reference,
not applied.

### Four assumptions
1. "we focus on decode nodes in disaggregated prefill-decode execution, as the prefill phase is
   inherently compute-bound and thus benefits little from HBF's large capacity."
2. "continuous batching that maintains a full batch by admitting new queries as in-flight ones
   complete."
3. Parallelism-selection sentence; SRAM limits apply in HBF+ and CONV+.
4. NVLink 1,800 GB/s (≤8 GPUs/node), InfiniBand 100 GB/s.

### Data placement / staging / KV writes
- Weights + KV cache → NAND flash: "Both HBF and HBF+ store model weights and KV cache in NAND
  flash… read-dominant access patterns."
- Intermediate data: HBF reserves 1 HBM stack; HBF+ uses 40-MB SRAM. Footnote 1: "writing
  intermediate data increases write traffic by up to 100× for every decode iteration… We use the
  SRAM capacity from prior work [8]."
- "much of the intermediate data has a short lifetime… allows HBF+ to support a large batch size
  (e.g., 327 per Llama 4 Maverick) despite the limited SRAM capacity (320 MB total)."
- KV-write modeling (sim extension #2): "accounting for the performance overhead of KV-cache
  writes when transferring outputs from prefill to decode nodes." Fig 5 "KV Write" = "writing the
  KV cache for newly admitted queries under continuous batching." Footnote 2: KV writes during
  decode "can be overlapped with computation in the attention layer."
- Chunked attention + pipeline parallelism = sim extension #3.

### Simulator
Base LLMSimulator [12] (Duplex, MICRO'24), extended: (1) HBF-aware read/write-asymmetric timing,
(2) KV-write / disaggregated prefill-decode, (3) chunked attention + pipeline parallelism.

### Communication model
Only qualitative: "inter-GPU communication increases almost linearly with batch size"; HBF+
"reduces communication overheads by leveraging less tightly coupled inter-GPU parallelism." No
closed-form comm equation beyond NVLink/IB bandwidths.

---

## 4. Every other quantitative prose claim (with section)

§I: 14× per-stack capacity (512 vs 36 GB); Maverick 746 GB vs GPU 288 GB.
§II: μs-order 4-KiB reads, 1.6 TB/s; HBF <80 W vs HBM 40 W; HBF introduced early 2025 [4].
§III: Z-NAND 87.4-GB/s, XL-Flash 262-GB/s; CONV 0.35 TB/s/stack (16 planes, 3-μs); HBF 1.6 TB/s
(25 planes, 1-μs); 100-μs program latency; 3.13-MB staging; 40-MB SRAM; 320 MB total; batch 327;
intermediate writes up to 100×/decode iter; workloads 1660/373, 5.9K/499, 103.5K/1.1K; NVLink 1800
GB/s, IB 100 GB/s.
§IV Batch (Fig 3): 8-GPU gains — HBF 1.3–4.5×, HBF+ 1.7–5.3×; CONV underutilization up to 95%
(Llama3 SHORT); HBF+ 24% larger batch than HBF at 8 GPUs; HBF+ gains over HBM4 124% higher in MID
/ 200% higher in LONG than SHORT.
§IV Throughput (Fig 4): Llama4 LONG 4-GPU HBF+ 15% higher TPS than 8-GPU HBM4; 8 GPUs = two
independent 4-GPU HBF+ instances w/ higher aggregate throughput.
§IV Breakdown (Fig 5): KV-cache writes 5–13.9% of exec time in Llama4 (HBF+).
§IV SLO (Fig 6): HBF+ beats HBM4 even at 0.05-s SLO in LONG; offline gain 4.1%→14.8% (Llama4 16
GPUs); offline = 24 h.
§IV Write Traffic (Fig 7): SLC 100K P/E limit; 3-year window; most cases exceed 100K PEC;
retention-relaxed writes [16].

---

## Ambiguities to flag for downstream auditors

1. GB vs GiB never disambiguated — check capacity constants (36, 512, 288, 746).
2. No GPU FLOPS printed — only "Rubin NVL8 [5]"; any FLOPS constant is unanchored.
3. No interconnect latencies — only 1,800 GB/s NVLink, 100 GB/s IB.
4. Model arch numbers (experts, top_k, kv-heads, GQA, context, iRoPE, precision) NOT in paper —
   only dense vs MoE + 746 GB.
5. CONV/CONV+ write BW (0.073/0.084) don't divide as cleanly per-stack as HBF's 0.016 — verify
   simulator uses plane-count scaling (16/25) not a flat value.
6. Table header is "(HBM/HBF)" not "(HBM4/HBF)" — the HBM column in CONV/HBF rows is the single
   reserved HBM4 stack (1.6 TB/s, 36 GB).
7. Figure visual reads ±0.1–0.15 (norm) / ±5% (Fig 5) / ±15×10³ (Fig 7) — red-label absolute
   anchors (194/61.5/3.75/460/151.5/31 and 3.3K/1.8K/147/19K/6.3K/1.3K) are exact; bar heights
   approximate. (Figure-reading data itself now lives in `paper_figure_readings.md`.)
8. ✕ SRAM-bound marker appears ONLY on Llama4-SHORT HBF+ — the canonical SRAM-bound cell.
