# `use_flash_attention` master-flag — implementation blueprint (2026-07-08)

Goal: one flag toggling two **internally coherent** attention models across every variant
(timing + capacity + energy). Default **OFF** (coherent materialized; moves numbers vs today's
incoherent hybrid — accepted). Neither branch reproduces today's published numbers; today's live
GQA is the *incoherent middle* (score in timing, softmax mem=0, capacity excludes resident score).

## Two branches
- **ON (FlashAttention / on-chip):** drop the Scoring score term (`m*n*heads/kv`) and Context P
  term (`m*k*heads/kv`) from timing; softmax charges 0 memory; MLA/Absorb softmax `2*m*n` dropped;
  capacity = current 256-tile; energy counters exclude score I/O.
- **OFF (materialized / no scratchpad):** keep Scoring/Context score+P timing terms; **ADD** a
  `2*m*n*heads/kv` softmax-memory charge (score-read + P-write) that GQA omits today — this is
  exactly MLA's existing `2*m*n` softmax term, so OFF unifies GQA to MLA; capacity = full resident
  score via `scoreInclusiveIntermediateBytes`; energy counters keep score I/O.

Note: `use_chunked_attention` (flash KV-read streaming) is ORTHOGONAL and stays ON in both branches
— it is the HBF read scheduler (sim ext #3), not FlashAttention. The FA flag only toggles
score/softmax **materialization**.

## RULINGS (settled)
1. **master + sub.** `use_flash_attention` = master (materialization, all variants incl. the
   materialized `else` of MLA/Absorb). `use_flash_mla` = sub (absorb/latent kernel selector; FLOPs
   axis). Keep independent — do NOT collapse (non-MLA override forces `use_flash_mla=false` at
   test.cpp:629 and would wrongly toggle GQA).
3. **MLA/Absorb-Gen `else`-branch:** materialize S **iff `!use_flash_attention`**. Never fires for paper-1.
4. **Capacity 256-tile:** drop `intermediateExtrasBytes` footprint.h:375 term under OFF (resident
   score dominates; avoid double-count).
5. **Mixed variants:** bring to parity + flag-aware (per user). Dead code, no runtime effect.
2. **PIM/LOGIC GQA (G2,G3,S2,S3): LEAVE AS-IS / always-flash + DOCUMENT** (per user). These roofline
   paths never modeled the score; OFF is NOT modeled there (documented limitation). They MUST stay
   byte-identical across ON/OFF — a regression tripwire.

## Per-site edits (from fa-flag-spec inventory)
- **GQA-Gen-GPU (attention_gen_impl.cpp:17 / G1):** Scoring :76 & HBF act :85 drop `+m*n*heads/kv`
  under ON; softmax :121-141 add `memory_size=2*m*n*heads/kv`, `total_duration+=max(compute,mem)`
  under OFF; Context :167 & :176 drop P term under ON. (Live decode flagship.)
- **GQA-Sum-GPU (attention_sum_impl.cpp:16 / S1):** Scoring :75-77/:85 gate score; softmax :192
  set `2*m*n*heads` under OFF; Context :261-263/:272 gate P; **ideal/ramulator score write
  (:117-125,:155-157) + softmax rd/wr (:207-215,:227-231) + ctx read (~:284):** gate `if(!FA)`
  (the I14 energy counters — ON must skip, today they fire even with mem=0 = incoherent).
- **MLA-Sum-GPU (S4, attention_sum_impl.cpp:710):** already `if(FA){flash}else{materialize}` — the
  reference template. No structural change; confirm Absorb-Sum S7-S9 (hard-materialized today) get
  the same `if(!FA)` wrapper.
- **MLA/Absorb-Gen (G4-G9):** in the `use_flash_mla=false` else-branch, gate the `2*m*n` softmax
  (:1885/2321/2729…), score/P terms (:1736/1810/1942…), and ideal score I/O (:1913/1917…) behind
  `if(!use_flash_attention)`.
- **PIM/LOGIC GQA (G2,G3,S2,S3):** no change (ruling #2). Document.
- **Mixed:** apply Pattern A/B shapes + flag read (dead-code parity, ruling #5).

## Capacity switch (footprint.h)
OFF formula already exists: `scoreInclusiveIntermediateBytes` (footprint.h:292-309) =
`peakIntermediateBytes(...) + 2.0*batch*(num_heads/tp)*min(seq_len,chunk?)*precision`. Matches CB1
(`LLMSimulator_HBF/cluster.cpp:183-184`, resident full-score summed). At BOTH callers:
- `cluster.cpp:177/190-193` (uses member `config.use_flash_attention`)
- `parallelism_optimizer.cpp:372/387` (uses `system_config.use_flash_attention`)
select `use_flash_attention ? peakIntermediateBytes : scoreInclusiveIntermediateBytes`; add
`intermediateExtrasBytes` in both (drop its :375 tile term under OFF, ruling #4). **Apply in BOTH
or neither** or the optimizer's admitted batch won't match the live gate.

## Flag plumbing (the GQA gap)
`use_flash_attention` lives on SystemConfig (hardware_config.h:143-144), parsed test.cpp:182-185,
config.yaml:51-52. It reaches MLA (attention.cpp:439) but NOT GQA. Thread it:
- add `bool use_flash_attention` member+ctor-param to `SelfAttentionGen`/`SelfAttentionSum`
  (attention.h GQA classes) and set `layer_info.use_flash_attention` in their `forward`
  (attention.cpp:57-84 / 109-144).
- add member+param to `SelfAttentionParallel` (parallel.h/.cpp:136-188), pass into the two
  `Create` calls (parallel.cpp:163,173), and pass `system_config.use_flash_attention` from
  `layer.cpp` (compare :186 which already passes flags to the MLA parallels).

## Verification plan
- **A/B#1 default-OFF vs today's published:** OFF adds softmax-mem + resident-score capacity →
  batch *reduces* on scarce tiers (most on HBF+/CONV+ = the U7 bar ab-score reproduced; least on
  HBF/CONV/HBM4); TPS ≥ today where memory-bound. Report per-family deltas.
- **A/B#2 ON vs OFF** (both coherent).
- **Tripwire (I1 anchor):** ON must show the earlier I1 ~3-HBM4-cell movement (score removed); if ON
  == today, the timing gate didn't take effect in G1/S1.
- **Controls that MUST stay byte-identical:** PIM/LOGIC GQA (G2,G3,S2,S3) across ON/OFF.
- MLA/Absorb/Mixed: build + spot-check only (no paper-1 MLA; Mixed dead).
