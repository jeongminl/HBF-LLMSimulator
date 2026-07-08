# Codebase Modeling-Inconsistency Audit (2026-07-08)

Cross-site *internal* inconsistencies: two code sites that make **contradictory implicit
assumptions about the same physical quantity**. NOT paper-mismatches, NOT single-site bugs.
Archetype: the attention **score matrix** — charged as materialized memory traffic in the
timing model, but treated as an on-chip 256-tile in the capacity model.

Paper anchor: base = Duplex sim (MICRO'24) + 3 extensions (HBF read/write timing;
KV-write/disaggregation; chunked-attention + PP). **Paper never mentions FlashAttention.**
SRAM (320 MB) sized by *peak occupancy of short-lived intermediate data* (footnote 1).
KV writes overlap attention compute (footnote 2). Comm is qualitative only (NVLink 1800 / IB 100).

Method: 5 parallel subsystem hunters + lead verification of every load-bearing claim against
both code sites. 17 confirmed inconsistencies.

---

## THE FLASHATTENTION INCONSISTENCY (one question, five faces) — HELD for user ruling

The single question: **does the simulator model FlashAttention (score/softmax never
materialize, live on-chip) or not?** The code currently answers *both*, incoherently.
These five items must all resolve the **same** direction or a new inconsistency is created.

- **I1 — Attention score matrix.** Timing (`attention_gen_impl.cpp:76` HBM4; `:85/:176` HBF)
  charges full `m·n·G` = O(context) score/P as materialized traffic; capacity
  (`footprint.h:121-125` excludes it, `:375` counts a fixed `(num_heads/tp)·256` tile).
  Softmax pass (`:122-141`) charges **zero** memory — already assuming on-chip. LIVE on HBM4;
  hidden under `max()` on flash. **Driver of U7.** This is *the* FlashAttention decision.
- **I9 — MLA/Absorb softmax materialized** (`2·m·n·num_heads` on the critical path,
  `attention_gen_impl.cpp:789,1885`; `attention_sum_impl.cpp:1010`) vs GQA softmax free.
  Same score-materialization question, MLA path. (MLA not in paper-1 → dormant for numbers.)
- **I10 — `use_flash_attention` flag honored for MLA, ignored for GQA.** Read only in MLA
  impls + `module/attention.cpp`; never in the GQA impls (which always materialize the score).
  The code half-implements FlashAttention. `config.yaml` sets the flag true globally.
- **I11 — MLA softmax modeled memory-bound yet its compute donated to KV-write hiding**
  (`attention_gen_impl.cpp:825` + `:828`). Not a numeric double-count (verified via
  `status.h operator+=`); a coherence defect that dissolves once I9 is resolved on-chip.
- **I14 — GQA-Sum softmax reporting.** Issues an ideal DRAM read+write of the full score into
  the energy/util counters (`attention_sum_impl.cpp:219-233`) while latency stays compute-only.
  The reporting-side face of I1.

Recommendation (conditional): **if the paper's explicit peak-occupancy statement governs →
standardize on-chip** (drop materialized score traffic everywhere; make GQA/MLA consistent;
remove the score from energy counters). **If the "no compute-side scratchpad" axiom governs →
materialized** (keep I1; but then GQA's now-free softmax must be *charged* memory for
coherence). The current state is the incoherent middle. Both directions require edits.

---

## PROCEEDING — paper-clear, independent of the FlashAttention question

| ID | Site A (implied model) | Site B (contradictory) | Paper side / recommendation | Impact |
|----|------------------------|------------------------|-----------------------------|--------|
| **I2** | LM-head timing `lm_head.cpp:58,92` charges full `m·(n_vocab/tp)` write | capacity `footprint.h:390` = 2048 argmax tile | Tile (argmax provably needs only a tile; `PAPER_INCONSISTENCIES.md:302-308`). Charge tile only. | LOW (weight read dominates) |
| **I3** | GQA-gen `attention_gen_impl.cpp:98,187` amortizes K+V page-fill (=2 → 1 fill/layer) | Sum `attention_sum_impl.cpp:170,352` & MLA-gen `:737,866` default 1 → 2 fills/layer | One staging pool (§III) → standardize on `=2`. | <0.4%; prefill/MLA only |
| **I4** | timing double-buffers a full-pool chunk (`layer_impl.h:112-116`) | capacity comment `footprint.h:354` "no double buffering"; physical staging 3.13 MB/stack | Paper: 3.13MB/stack *is* the double-buffer. Reconcile (comment, or half-pool chunk if inert). | likely inert |
| **I5** | every GEMM derates by `effectiveMFU` (`linear_impl.cpp:75`) | LM-head `lm_head.cpp:63` raw peak | Consistency → route LM-head through effectiveMFU. | zero (MFU=1.0) |
| **I6** | decode gates link latency on volume (`communication.cpp:37-41`) | prefill adds latency unconditionally (`:32-36`) | Volume-gated (decode form). | zero (prefill off decode path) |
| **I7** | comm expert-count divide-first (`communication.cpp:191,247,369,431`) | capacity mult-first (`cluster.cpp:154`) | Normalize to mult-first. | zero (divisible today) |
| **I8** | AllReduce whole-group single link (`communication.cpp:88-104`) | MoE a2a per-flow split (`moeLinkTime`) | Hierarchical AllReduce (intra=NVLink, inter=IB). **Moves tp>8 16-GPU cells — needs ruling.** | some 16-GPU cells |
| **I12** | MLA KV read includes `qk_rope` (`attention_gen_impl.cpp:643,727`) | KV write omits it (`layer_impl.h:181`) | Add qk_rope to write (+ staging, to keep N1 match). | zero for GQA (qk_rope=0) |
| **I13** | timing SRAM link = `logic_sram_bandwidth` (12.8e12) | memory_util diagnostic = `1e13` (`activation_impl.cpp:95`) | Use logic_sram_bandwidth. | zero (util display only) |
| **I15** | GQA-Sum & MLA-gen read KV via HBF flash model | MLA-Sum reads at plain HBM bw (`attention_sum_impl.cpp:710-1169`) | Extend flash-read branch to MLA-Sum. | zero for GQA; live for MLA |

## DEAD CODE — held (Mixed family never instantiated; verified `parallel.cpp:163,173`)

- **I16** — Mixed variants omit the `m·n` score/output write term Gen+Sum charge
  (`attention_mixed_impl.cpp:60,96`).
- **I17** — Mixed variants have no Softmax phase (no FLOPs, no KV-write hide budget).
- Recommendation: **delete the Mixed family** (dead code) or bring to parity. Not auto-touched.

## CLEARED / verified consistent (do not re-hunt)

- **REFUTED — HBM4 KV-read double-count.** `dram_interface.cpp:54` sets only `memory_duration`;
  `status.h:54-72 operator+=` never touches `total_duration`. KV read reaches latency once.
- Verified consistent: MoE a2a stage-scoping (prior fix complete, all 4 legs), `checkCapacity`
  branch symmetry, per-stack vs total SRAM, cluster.cpp↔optimizer non-drift, KV-write
  timing-vs-staging byte formula, read/write bandwidth asymmetry, no double-hiding of KV write,
  program-latency amortization, rank layout, TP-AllReduce scoping, memory-preset internal consistency.
- Watch-item (not an inconsistency): HBF intermediate traffic @1.6 TB/s (1 reserved HBM stack)
  vs HBF+ @12.8 TB/s (logic SRAM) — 8× jump, but two different physical media; defensible.
