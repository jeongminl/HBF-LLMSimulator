# Findings register — third-pass bug hunt (2026-07-03)

## Phase-4 convergence summary (vs prior docs, read only after all verdicts fixed)
- CONTRADICTS-DOC: C1 (CHANGES #20/#37 claim optimizer/live "lock-step" — false, optimizer mirror
  never updated); C12 (#32 "parity <0.01%" verified only at pp=1/TP=1); C10 ("routing skew ruled
  out" was scoped to U1's capacity-bound point only — real ~3× effect at llama4 LONG).
- GENUINELY-NEW: C3, C5, C6, C7, C8, C11, L2-L5, L7, I5, I6.
- ALREADY-KNOWN: C2 (+precision added), L1, L6, I1-I4.
- SUPERSEDES-DEFERRAL: C4 — applied alone with user approval; the docs' paired
  MFU/score-removal item remains OUTSTANDING → expect residual ~2-4% fast SHORT/MID and slightly
  worse llama4 SHORT until that pair is decided. Seeding is physically grounded (not the fitted
  half of the pair), so applying it alone violates no rule.
- Track A: outcome matches docs ("no fix", batch drifts higher) but REPLACES the diagnosis —
  capacity/KV-headroom, not the comm-model suspects listed in PAPER_INCONSISTENCIES.
- Track B/C9: delivers U7's own open directive (context-independent per-seq accounting);
  corrected ceilings bracket the docs' predicted 374-430 KB/seq band (EP4 297, EP8 534;
  paper bar ⇒ 441). Which EP wins = answered by the post-fix regression sweep.
- Doc corrections queued (do NOT edit main-tree docs from this worktree; list for the user):
  CHANGES #20/#32/#37 parity language; PAPER_INC Residual-1 framing, U7 disposition, deferred-#2
  status, ruled-out-skew scoping; PEC windowing gap belongs in the iRoPE-sweep doc.

Status legend: CANDIDATE (awaiting Phase-2 refutation) / VERIFIED / REFUTED / LATENT (real but
unreached by paper sweeps) / INFO (no action). Independence rule applied: no finder read the
repo's prior analysis docs; paper ground truth = PAPER_ANCHOR_SHEET.md.

## Actionable candidates (Phase-2 refutation queue)

### C1 [F7-1, DANGEROUS prune risk] — VERIFIED (R1: CONFIRMED, magnitude 1.6128 ms/step recomputed; optimizer comment falsely claims parity with live AllReduce) — Optimizer TP all-reduce hop model: ring vs recursive-doubling
- Optimizer parallelism_optimizer.cpp:515 charges 2(tp−1) latency hops; live communication.cpp:47-48
  charges 2⌈log2 tp⌉. BW term matches. Overcharge at tp=8: 8 hops × 800ns × 2AR × 126 layers ≈
  1.61 ms/step spurious (llama3, pp=1) → seed_tps deflated ~16% more than tp=4 → true TP=8 winner
  can be pruned unverified (run_experiments.py:640).
- Affected: llama3 8/16-GPU cells (all presets). Fix direction: 2⌈log2 tp⌉ in per_allreduce.
- Discriminator: HBF_DISABLE_CONFIG_PRUNING=1 rerun of llama3_405B/HBM4/8-GPU; winner flip or TPS
  rise confirms.

### C2 [F7-2+3, MoE prune risk] — VERIFIED (R1: CONFIRMED; two independent errors: wrong exclusion-group variable for scatter direction + missing /tp and /e_tp divisions; overcharge ≥×tp for MoE tp>1) — Optimizer scatter volume + grouping drift
- scatter_msg (opt :531) missing live's /ne_tp_dg divisor (communication.cpp:140-141, 327-328);
  single scatter_frac=1−e_tp/devices (opt :529) used for both directions while live scatter
  excludes ne_tp group (:105) and gather excludes e_tp group (:291).
- Affected: llama4 tp>1, e_tp<devices_per_stage configs — overcharge → prune risk (marginal).
- Discriminator: analytic_configs_only dump diff across e_tp_dg for llama4 8-GPU tp>1.

### C3 [F7-5, LIVE bug] — VERIFIED (R2: CONFIRMED; reachable at 16 GPUs via valid candidate e.g. tp=8/pp=1/dp=2/e_tp=16, which the optimizer's EP-max bias prefers; moe_all_reduce_for_e_tp spans both nodes at NVLink pricing) — AllReduce not node-aware
- communication.cpp:51-53 uses device_ict (NVLink) unconditionally — no src/dst node check, unlike
  MoEScatter/Gather and PipelineStage. A 16-GPU 2-node e_tp=16 MoE all-reduce group is priced at
  1800 GB/s instead of IB 100 GB/s (~18× undercharge) → inflates reported 16-GPU llama4 TPS
  (Fig 4 16-GPU cell) if the live winner uses a cross-node all-reduce group.
- Fix direction: node-aware link selection in AllReduce (mirror PipelineStage).
- Discriminator: 16-GPU llama4 cell, force e_tp_dg=16; node-aware variant should lower TPS.

### C4 [F4-1] — VERIFIED (R3: CONFIRMED; 10 iters all at ctx≈input, no churn; input+out/2 correct steady-state basis; normalization does NOT cancel — HBF-family gains slightly inflated; TPOT-bias = deficit% × attention-share, cell-dependent) — Decode context seeded at input_len vs steady-state input+output/2
- scheduler.cpp:98 pushDummySeq seeds current_len=input_len; measured 10 iterations sit at
  ctx≈input. Paper §3 steady-state (arrival=completion) implies representative avg context
  input+output/2. TPOT underestimated ~5% SHORT / ~3% MID / ~0 LONG; Fig 3/4 SHORT/MID cells
  biased high (incl. the exact red-label anchors).
- Note: analytic side uses input+output (a THIRD convention). Neither matches steady-state.
- Discriminator: reseed current_len=input+output/2, rerun SHORT+MID 8-GPU both models.

### C5 [F4-2] — PARTIAL (R3: mechanism CONFIRMED incl. the code's own [OptValidation] OVERESTIMATE warning; risk ranking INVERTED — worst cell is SHORT (+22% KV-read overestimate), not LONG; empirical prune-flip unverified → needs HBF_DISABLE_CONFIG_PRUNING=1 rerun in Phase 3) — Batch-search pruning invariant violated by context-basis mismatch
- Analytic KV/attention latency uses full input+output (test.cpp:323/447/556, optimizer :290-297);
  live sim uses ≈input (C4). Analytic overestimates attention latency → seed_tps NOT an upper
  bound → attention-bound configs (LONG, low-TP) can be pruned unsimulated (run_experiments.py:640;
  "[OptValidation] OVERESTIMATE" print is warning-only, test.cpp:816).
- Affected: LONG cells; direction = reported batch/TPS too LOW where winner pruned.
- Discriminator: HBF_DISABLE_CONFIG_PRUNING=1 diff on LONG cells.
- Interaction: fixing C4 (live → input+out/2) shrinks but does not eliminate the gap (analytic
  still input+output). Full fix = same context basis both sides.

### C6 [F6-1] — VERIFIED (R4: CONFIRMED as internal inconsistency, not ambiguity — the sim's own timing/capacity already commit to windowed semantics; ratio re-derived 3.158×, windowed PEC ≈71.2×10³ < 100K SLC line; fix diverges from paper's Fig-7 bar which matches unwindowed → surface to user at fix time) — Fig-7 PEC uses unwindowed KV volume; timing/capacity use iRoPE window
- run_experiments.py::compute_pec (:772) × PEC_KV_BYTES_PER_TOKEN (test.cpp:772-776) charges
  2·num_layers·kv_heads·hd·prec × (in+out) — no effectiveKvLen. Timing (layer_impl.h:163-169) and
  capacity (optimizer :472) window local layers at attn_chunk_size=8192 (36/48 layers local).
- llama4 LONG only (in+out=104600 ≫ 8192): 3.16× overcount. Windowed PEC ≈ 71×10³ vs current
  ≈225×10³ — which MATCHES the paper's Fig 7 anchor. Physical correctness (windowed) vs paper
  conformance (unwindowed) point in OPPOSITE directions → user decision required at fix time.
- Discriminator: llama4 LONG HBF+ offline 8-GPU; ratio of PEC-implied volume vs Σ getKVWriteDuration
  writes ≈ 3.16 (MID control = 1.0).

### C7 [F1-3] — VERIFIED (R5: CONFIRMED ~1%; paper's 3620 is HBM+flash combined; PEC denominator shares the constant; impact likely sub-rounding on headline bars) — HBF/CONV flash capacity gate double-counts reserved HBM stack
- hbf_memory_config.h:75,109 total_capacity_bytes=3620 GiB gates weights+KV (footprint.h:67), but
  flash pool is 7×512=3584 GiB; the 36 GiB HBM stack is separately the activation tier
  (footprint.h:40). ~1% capacity inflation → HBF/CONV capacity-bound batch cells ~1% high, PEC
  denominator ~1% high.
- Discriminator: llama3 HBF 8-GPU LONG max_batch with 3584 vs 3620.

### C8 [F1-1] — VERIFIED (R2: CONFIRMED; unphysical inversion, not a direct paper contradiction since paper anchors no latencies; compounds C3 on cross-node paths) — Fabricated interconnect latencies, physically inverted
- eval/test.cpp:83,95: device_ict_latency=800ns (NVLink), node_ict_latency=130ns (IB). Paper gives
  NO latencies; values inherited from H100-era guesses; NVLink latency 6× HIGHER than IB is
  physically inverted (IB end-to-end is typically ≥ NVLink). Effect <1% (comm share), but a
  non-paper assumption feeding every all-reduce.
- Discriminator: 16-GPU llama3 MID, latencies both 0 vs current — Fig-5 comm share delta.
- NOTE: F7's audit quotes the same values from eval/test.cpp:68-107 — consistent reads.

### C9 [F5-1, HIGH confidence — U7-relevant] — VERIFIED (R6: CONFIRMED; ratio 1.568 re-derived, ceiling 855→546; corrected formula supplied: shared term × batch_per_dp instead of expert_batch_size; DP global-average convention confirmed correct/orthogonal; llama3 & deepseek fix-direction safe) — Shared expert undersized in SRAM footprint gate
- footprint.h:195-198 charges (num_routed_expert_per_device + num_shared_expert) blocks all at
  expert_batch_size (= total·top_k/128, avg per ROUTED expert; cluster.cpp:108). Shared expert is
  dense → processes batch_size_per_dp (16× larger for maverick 8-GPU). Corrected ffn_moe binds the
  peak: true SRAM peak ×~1.57 → llama4-SHORT-HBF+ SRAM ceiling ~855 → ~545/GPU (toward paper 327).
  Runtime shared-expert TIME is correct (expert.cpp:204) — capacity-gate-only bug. Optimizer and
  live gate share the function, so both move together.
- Discriminator: charge shared at batch_size_per_dp; llama4 SHORT HBF+ 8-GPU SRAM ceiling falls
  ~1.5×.

### C10 [F5-2, MEDIUM — llama4 LONG cells] — VERIFIED (R7: CONFIRMED; device0 carries 49.3% routing mass, E[active] 8.61 vs balanced ~2.9 at LONG draws≈31 → ~2.97× bottleneck inflation; live path for paper figures confirmed; fix = round-robin placement, preserves skew; NOTE fix will RAISE llama4 LONG TPS, possibly above paper anchor — mechanism-explained) — Zipf-hot experts colocated on device 0
- sequence.cpp:168-170 skew weight 1/(i+1)^0.8 favors low expert indices; expert.cpp:62 maps
  experts contiguously (0-15→dev0 …) → hot experts colocate. At low batch (LONG, ~31 active
  experts) dev0 streams ~13 token-independent expert weight reads while dev7 idles; iteration
  time = maxDeviceTime (cluster.cpp:655) → up to ~3× expert-FFN inflation, depressing llama4 LONG
  TPS. SHORT/MID unaffected (all resident experts active). Bug-vs-modeling-choice debatable —
  paper gives no basis for hot-expert colocation; real systems balance placement.
- Discriminator: per-device expert_ffn times (dev0≫dev7), or skewness=0 → LONG TPS jumps,
  SHORT/MID flat.

### C11 [R5-discovered via X1, LIVE bug] Residual op prices activation traffic at device scalar
- residual.cpp:31,38 uses device->config.memory_bandwidth unconditionally — no use_hbf branch. On
  HBF that is flash_read_bandwidth (11.2 TB/s) for pure residual-stream activation traffic that
  belongs on the 1.6 TB/s HBM stack → ~7× too fast on that op. llama3 8-GPU MID: ~0.9-1 ms/step
  understated (~1-3% of decode step). Same pattern in MLA softmax (~gen_impl:768, DeepSeek-only,
  latent). All other reached ops verified correct (linear/activation/layernorm/attention/KV-write
  table in R5 verdict).
- Fix direction: route Residual (and MLA softmax) memory time through the same act-tier selection
  as activationCore (hbm_read_bandwidth when use_hbf, 0-cost SRAM when num_hbm_stacks==0 per
  HBF+ convention).
- Status: VERIFIED (R5 code trace; F1-4's "all activation ops" version REFUTED, F3's linear-path
  check confirmed).

### C12 [R8-discovered] Optimizer shared-expert capacity weight not TP-sharded
- parallelism_optimizer.cpp:151-154 counts the shared expert as "fully replicated, not TP-split";
  runtime genuinely TP-shards it (expert.cpp:92-104, non_moe_device_list of size ne_tp_dg —
  confirmed by measured 116.384→106.020 GiB weight drop at TP=2). Optimizer overestimates tp>1 MoE
  weight/GPU → can reject or misrank capacity-edge candidates the live sim would accept. The live
  gate (cluster.cpp, actual tensor bytes) is correct — analytic-side drift only.
- Fix direction: shard shared-expert bytes by tp in the optimizer capacity formula (mirror
  runtime).
- Status: VERIFIED (R8 trace + measured weight delta).

## Cross-agent contradictions — RESOLVED

### X1 → resolved into C11 (see above). F1-4 overbroad; F3 correct on linear path.

### X2 [F4-flag vs F3/F2 + Track A data] KV TP-sharding — RESOLVED: correctly sharded
- R5 trace: parallel.cpp passes num_kv_heads/parallel_num into Split/Sum/Gen/Merge;
  attention.cpp:461-463 stores the sharded value with an explicit don't-divide-again comment.
  Consistent with Track A's measured TP-invariant atten_gen. F4's flag REFUTED.

### X3 [F4 vs F7] Optimizer comm terms existence
- RESOLVED by F7: comm terms exist (:512-537); F4's "no comm term" misread the :333-497 stage
  block. F4's C5 stands independently of this.

## Latent findings (real divergences, unreached by paper sweeps — hygiene fixes only)
- L1 [F2-1] iRoPE window cap only in Gen-GPU base; missing in Sum/Mixed/Logic/PIM variants
  (LONG llama4 prefill would over-read 12.6×).
- L2 [F2-2] AbsorbMLA gen score phase lacks use_hbf branch (attention_gen_impl.cpp:1699,1772) —
  DeepSeek HBF undercharged.
- L3 [F2-3] Naive MLA gen reads full per-head KV vs Absorb's compressed latent — ~70× divergence
  (deepseek use_absorb=off only).
- L4 [F2-4] Mixed-GPU memory_size drops score-write/output-write terms (mixed_impl:59,95).
- L5 [F2-5] flash-mla flops omit m factor + effectiveMFU derating (gen_impl:1615 vs :617,985).
- L6 [F6-note] cluster.cpp:316-330 mem_cap_limit fallback uses unwindowed KV — dead under
  exit_out_of_memory=True.
- L7 [F1-5] Stale comment eval/test.cpp:262 (precision claim contradicts code).

## Info / no action
- I1 [F3-1] Page-latency exposure once-per-iteration vs per-burst — ≤1-3% optimistic on flash
  presets, uniform, optimizer-consistent; modeling-aggressive but defensible under the paper's
  double-buffer framing.
- I2 [F3-2] Decode 1-token KV appends not charged — MATCHES paper methodology; recorded to
  prevent rediscovery.
- I3 [F1-2] Rubin FLOPS 8.75 PF unanchored (paper prints none) — provenance note only.
- I4 [F7] Optimizer omits MoE all-reduces + prices scatter NVLink-only at 16 GPU — SAFE
  (undercharge) for pruning; ranking noise only. Bundle with C2 if fixing.
- I5 [Track A] Sweep understated TP=2 ceiling by ~1% (3868 reported vs 3908 measured) — search
  tightness, no wrong-direction risk. Check bisection gap logic if time permits.
- I6 [R8] expert_ffn +1.7% at TP=2 vs TP=1 at equal total batch — nominally TP-invariant; small
  unexplained drift, low priority.

## U7 FINAL (post-fix 13-cell regression, 2026-07-03)
C9 fixed but the cell does NOT flip: the throughput-max search escapes the SRAM ceiling via EP=1
(corrected ceiling 2824/GPU, the highest of any EP) and lands slo-bound at 2245/GPU (+196% vs the
paper's 759 ✕ bar); winner tp=1/pp=1/ep=1/dp=8 batch=17960. Track B's flip prediction assumed the
EP4 winner held — it did not (exactly the uncertainty it flagged). Notable: the score-inclusive
DIAGNOSTIC ceiling with C9 applied now reads 755.6/GPU (434 KB/seq — inside the docs' predicted
374-430 band, ≈ the paper's 760 bar), but the impossibility proof still bars adopting it as the
gate (it would sink MID to ~254 vs the paper's own 742). FINAL DISPOSITION: C9 was a real bug,
fixed on correctness grounds; the paper's flat ✕ bar remains unreproducible under any single
consistent accounting + throughput-max EP search — a paper-internal inconsistency (score-inclusive
text/marker vs score-exclusive MID/LONG bars), now with a sharper mechanism map than the prior
docs had.

## Track B REVISED verdict (U7) — REAL FOOTPRINT BUG (C9) PARTIALLY RESPONSIBLE
After integrating C9: the shared-expert correction adds a context-INDEPENDENT ~58KB/seq, making
ffn_moe binding at ceiling 2825/1883/1130/628 per-GPU for EP=1/2/4/8 (tp1/dp8). Verified winner
EP=4 → ceiling 1130 < slo-bound 2170 → llama4-SHORT-HBF+ FLIPS TO SRAM-BOUND; paper's 760±70 sits
inside the 628-1130 band (dead-on if winner moves to EP8 at the ceiling). MID stays slo@742
(<1130) ✓, llama3 LONG unchanged (dense) ✓ — no regression. Context-independence sidesteps the
impossibility proof (which only killed O(ctx) charges). Score-inclusion and 40MiB-per-stack remain
REFUTED (each breaks MID/LONG). "327" remains a paper-text artifact (score-inclusive MID-ctx
ceiling = 327.7, inconsistent with the paper's own score-exclusive MID/LONG bars).
Post-fix sweep must confirm the winning EP and sram/slo boundary. Validation anchor: current
EP4 ceiling formula reproduces the analytic dump exactly (11120/8 = 1390.6 = 320MiB/241,280B).

## Superseded original verdict (kept for the impossibility proof, which stands)
Impossibility proof: SHORT binding at 759 while MID unbound at 742 requires an O(ctx) coefficient
k ≤ 2.3 B/token; smallest physical score coefficient ≥ 20 B/token (tp8). No context-scaled SRAM
charge fits both. "327" = 320MiB/(2·40·6399·2B) = 327.7 — the paper's TEXT used a score-inclusive
MID-context ceiling while its MID/LONG BARS are score-exclusive. Current gate (aggregate 320 MiB,
max-phase, score-excluded) matches 5/6 bars incl. MID and l3 LONG exactly (742 slo ✓, ~19 slo ✓);
score-inclusive variant wins only SHORT's ✕ and regresses MID (265) and l3 LONG (6.2) 2.8-3×.
Optimizer winners verified low-TP (TP1/DP8/EP4 for l4 SHORT/MID), matching the paper's "less
tightly coupled" language — tp choice not a bug either. RECOMMENDATION: keep current model,
document the paper's inconsistency. C9's fix stands on correctness grounds but does NOT change the
8-GPU SHORT bar (corrected ceiling ~2824 > slo-bound ~2170); it lowers 16-GPU SRAM-bound cells.
NOTE: F5/R6's "855→546" ceiling framing used a stale 855 baseline; actual current ceiling 4428
(Track B analytic dump, exact). Corrected-formula ceiling ≈ 2824/GPU at tp1.
Drivers left at worktree root: trackb_dump.py, trackb_pick.py.

## Track A verdict — REFUTATION PASSED (R8: SAFE)
Full 9-component decomposition sums to reported latency to the ns in both layouts; KV/GPU
invariance, shared-expert runtime TP-sharding, and search-space reading all confirmed. Verdict
stands: Residual-1 = paper restricted its config space to DP-pure; no simulator fix.

## Track A verdict (Residual-1) — no bug, pending refutation pass
Cell capacity-bound (tpot≈26ms ≪ 100ms SLO). TP=2 crowning correct: TP shards non-expert weights
(−10.36 GiB/GPU) → +6% KV headroom → batch ceiling 3908 vs 3680 → +1.7% TPS despite +408µs comm.
DP-pure ceiling = 3680 = paper's 460/GPU exactly → paper's tool likely restricted to DP-pure.
Matching it would be calibration — recommended NO fix. Residual TPS gap at matched batch:
25.77ms vs paper-implied 24.2ms (−6%) — separate compute-speed question, goes to convergence check.
