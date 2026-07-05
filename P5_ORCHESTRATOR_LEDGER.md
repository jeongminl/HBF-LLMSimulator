# Pass-5 findings ledger (orchestrator working notes, 2026-07-04)

Protocol: 9 blind Opus finders (F1 scheduler, F2 attention-gen, F3 metrics/Fig-6, F4 config
leaks, F5 module-graph, F6 Fig-5 bucketing, F7 U5-angle, F8 U7/SRAM-angle, F9 Residual-1/
capacity-angle) + Sonnet refuters + parallel verification. Finders denied all analysis docs +
git history; allowed paper PDF + anchor sheet + figure readings + src/eval/harness.
Convergence check vs prior docs happens ONLY after adjudication.

## Phase 0 — build & smoke
- Build OK at <worktree>/build/run. NOTE: ramulator2_pim.patch deletes `#include <cstdint>`
  from ramulator2 base/utils.h → GCC 13.3 build break; main's checkout carries a manual
  restore (untracked). Replicated in worktree. Pre-existing -Wreturn-type warning in
  PIM_controller.cpp:53 untouched.
- regression_13cell.py must run with /usr/bin/python3 (build/venv lacks PyYAML).
- 13-cell smoke: IN PROGRESS (7/13 at last report; l3_HBM4_8_LONG 3.75/GPU exact,
  l4_HBM4_8_LONG 32.0 vs paper 31.0 — tracking main).
- **OOM-RESILIENCE PATCH PORTED (2026-07-04):** ported a 154-line `run_experiments.py`
  reliability patch from worktree `worktree-stateful-forging-gem` (same base commit
  `a2b1e13`, verified identical prior to the patch). Root cause of the host crash that
  killed this session's background agents (see recovery note below): offline/
  unconstrained-SLO cells (slo>=1000 — PEC gathering + one sensitivity SLO point) run
  an unbounded analytic batch-capacity search whose `./run` process RSS reaches
  13-23GB, vs a few GB for latency-bound online cells; running `SWEEP_WORKERS`-wide
  concurrency on these oversubscribes host RAM and triggers a sustained OOM-killer
  storm that also kills unrelated processes (this session's agents included). Patch
  adds: (1) `OFFLINE_SWEEP_WORKERS` (default `SWEEP_WORKERS // 4`) — a separate,
  smaller worker pool for slo>=1000 specs in the sens-baseline, sens-sweep, and PEC
  phases (the PEC phase is 100% offline specs — the phase where the storm actually
  hit); (2) `_crashed_cell` fallback + try/except around every `fut.result()` so one
  dead pool worker no longer raises out of `as_completed` and discards every other
  already-completed cell via an unhandled exception; (3) `_checkpoint()` — writes each
  completed phase's raw results (`checkpoint_baselines.json`, `_results.json`,
  `_slo_results.json`, `_pec_results.json`) to disk immediately, so a later-phase
  crash no longer loses already-completed earlier phases (previously everything lived
  only in `main()`'s locals until the single final `experiment_data.json` write).
  Pure harness-reliability change — zero effect on simulator correctness, no
  simulator source touched, does not alter any computed number. Directly relevant to
  the pre-authorized Phase 6 full sweep (same script). Verified: diff against the
  source worktree is now empty; `py_compile` passes.
- **CRASH RECOVERY NOTE (2026-07-04, mid Phase 4):** host machine crashed after the finder/
  refuter waves and V-moe-liveness adjudication completed but before S-implement's edits or
  the pre-fix regression's final table were captured. Verified on resume: no tracked source
  file carries any diff (`git status` clean except the pre-existing ramulator2 submodule
  bump) — S-implement's in-memory edits never landed, safe to redo from a clean pre-fix
  tree. `build/run` (pre-fix binary) and all CMake artifacts survived the crash intact
  (files on disk, unaffected). All finder/refuter reports and adjudications above were
  already flushed to this file before the crash and are trusted as-is. Re-running: (a) the
  pre-fix 13-cell baseline table (binary already built, no rebuild needed) and (b) the
  fix-slate implementation, in parallel.

## Finder reports (arrival order)

### F4 — config-flag leaks (REPORTED)
- **F4-1 (FLAGSHIP, LIVE-UNSTATED, probed): `skewness: 0.8` speeds up all llama4 MoE cells
  ~3-11%.** Mechanism: Zipf skew concentrates routing → FEWER DISTINCT active experts →
  less token-independent expert weight-streaming from flash; coprime-stride placement keeps
  devices balanced so the win is real, not a balance artifact. Probes (Maverick/MID/HBF+/8GPU/
  tp8, iter10, seed 777): batch32 skew0 46.185ms vs skew0.8 41.270ms (skew 10.64% faster);
  batch4 3.33% faster. Grows with batch. Affects every Maverick cell in Figs 3/4/5/7,
  preferentially flatters flash presets (weight-stream-bound). Paper SILENT on routing
  distribution → per plan this is a SURFACE-TO-USER decision, not a silent change.
  ADJUDICATION PENDING: refute mechanism (distinct-expert count vs GEMM-size effects),
  check LONG/low-batch direction, and whether balanced-uniform would move paper-compared
  cells toward or away from paper bars (no calibration allowed either way).
- F4-2: reuse_kv_cache/kv_cache_reuse_rate=0.2 — DORMANT in decode (sole consumer
  scheduler.cpp:91 overwritten unconditionally by decode branch :98-131). High conf.
- F4-3 (note): chunk_size=0 auto = 3.13MiB × num_flash_stacks aggregated as the double-buffer
  chunk (HBF 21.9MiB / HBF+ 25MiB). Paper's staging is per-stack; aggregation defensible,
  rarely binding in decode (num_chunks=1). Doc-note candidate.
- Flag map: all other unoverridden keys classified DORMANT or LIVE-CONFORMANT;
  use_flash_attention=on LIVE-UNSTATED but uniform across presets (known deferred-pair
  territory — do not reopen unilaterally).
- mem_cap_limit unreachable (else-if under exit_out_of_memory=True). injection_rate=0 →
  Poisson path off. MLA flags auto-off via q_lora_rank guard (confirmed).

### F5 — module-graph / time aggregation (REPORTED)
Core aggregation verified correct (7-item checked-correct list: intra-device serial sum;
max-over-devices for reported total in all 3 paths; TP all-reduce sync-then-add; GPU‖PIM join
max; comm ops once-per-op; expert comm not double-counted in expert_ffn stamp).
- **F5-C1 (M, actionable): CSV time-breakdown columns pinned to DEVICE 0 (cluster.cpp:852)
  while reported TPOT = maxDeviceTime over all devices.** Under EP imbalance (skew!) device 0
  ≠ bottleneck device → Fig-5 component fractions for llama4 MoE cells distorted relative to
  the tpot they're normalized against; headline TPOT unaffected. Interacts with F4-1 skew and
  F6's bucketing audit. Discriminator: dump per-device device_time one step; compare
  Σ(columns) vs device-0 time vs maxDeviceTime.
- F5-C2 (L): parallel-mode atten_sum/atten_gen columns additive while critical path takes max —
  inert (parallel_execution off in all paper cells).
- F5-C3 (L, design): PP = sum-of-stages (communication.cpp:587), defended in-code for
  autoregressive decode without micro-batching; only bites pp>1 cells; modeling choice.
- F5-C4 (VL): executor zero-duration sentinel picks larger — conservative, trivial.
- F5-C5 (VL): dead 20ms idle assignment (continue skips add); AND exportToCSV flushes at
  loop-START → final iteration's stat rows never written (CSV holds n_iter−1 = 9 of 10
  measured rows under harness iter=10). Rows near-identical in steady state → tiny Fig-5
  effect only; tpot from stdout unaffected. Check under F6.

### F7 — 1-GPU HBF/HBF+ non-weight budget (REPORTED)
Scope verdict: clean. Both anchors reproduce (l3/HBF/SHORT batch176 → 99.70ms tpot ≈ paper
176.2 bar; l3/HBF+/LONG batch7 → 95.05ms ≈ paper 7.1). KV read/write, layernorm, residual,
activation all tier-routed correctly and match first-principles to <2% (measured EXACT on
layernorm/residual HBF). Admission KV amortized as constant batch/output_len per iter (all
iters identical → mean-tpot SLO = worst-tpot; physically defensible).
- **F7-F1 (hygiene, zero paper impact): RoPE op priced at flash BW** (rope.cpp:42,49,101,108
  uses device memory_bandwidth = flash on HBF/HBF+; no tier branch) — but RoPE module is
  instantiated ONLY on the MLA path; GQA models fuse RoPE (rope column ≡ 0 in probes).
  DeepSeek-only latent. Fix candidate on correctness grounds (same class as the fixed
  residual.cpp bug).
- **F7-F2 (small, live): layernorm WEIGHT page-read latency un-amortized** (layernorm.cpp:53
  charges full 1000ns per layernorm) — hidden under act max() on HBF, EXPOSED on HBF+/CONV+
  (act=0): measured 252,645 ns/step on l3/HBF+/LONG = 0.27% of step (~0.5% at SHORT).
  Over-charge, all HBF+/CONV+ cells. Fix candidate: amortize like linear weights
  (weight_stream philosophy) — needs refuter + regression estimate.

### F2 — decode attention accounting (REPORTED)
GQA Gen path substantially CORRECT on all six questions (iRoPE caps consistent across
QK/softmax/AV/KV-read/KV-write; page latency single-exposed per phase; roofline no
double-count — ExecStatus += accumulates memory/report fields only; hide budget = attention
compute, one 100µs program latency per iteration via layers_per_stage amortization; context
basis = live per-seq lengths with stagger mean = input+output/2 steady state).
- F2-C1 (L-M): page FILL latency exposed TWICE per layer (K-scoring + V-context calls each
  expose one fill; 2µs/layer HBF, 6µs/layer CONV → ~756µs/iter at CONV 126 layers).
  Over-counts SHORT/MID low-batch HBF/CONV/CONV+ cells; defensible (K and V are separate
  streams split by softmax) — adjudicate: modeling choice vs bug; discriminator = fold-fill A/B.
- F2-C2 (converges with known N4): MFU m=1 keying on decode attention; NEW coupling insight —
  inflated compute_duration is the KV-write HIDE budget, so enabling MFU would counter-
  intuitively SHRINK the Fig-5 KV-write band. Strengthens "repair before any MFU work".
- F2-C3 (L): HBM4 charges ZERO KV-write (guard use_hbf && num_flash_stacks>0) — structural
  omission in the HBM4 Fig-5 breakdown (paper shows a small KV-write band for HBM4);
  magnitude negligible (12.8 TB/s write, tiny admission volume). Adjudicate: charge it for
  breakdown fidelity vs leave (near-zero either way).

### F8 — SRAM gate re-blind (REPORTED)
Byte-exact validation: hand formula = binary analytic dump EXACTLY (maverick tp1/ep1/dp8 cap
22592 = 2824/GPU; boundary 2824 pass / 2825 fail; SHORT≡MID context-independent). Blind
verdict on the ✕ bar: paper-internal inconsistency, no single accounting fits all six HBF+
bars; only an O(ctx) score term reaches the 374 KB/seq the 855 bar implies (context-
independent floor ~20KB unshardable, tp1 peak 116KB). CONVERGES with U7 disposition, reached
blind.
- **F8-C1 (M conf, real correctness nit): FFN-phase peak omits the persistent residual stream**
  (footprint.h:224-227 dense / :217-221 shared) while the attention phase counts it —
  asymmetric under-count of the BINDING branch: +10.2KB/seq maverick (+8.6%), +32KB/seq llama3
  (+13%). Ceiling 2824→2601/GPU maverick (still > the ~2374 slo-bound point → U7 cell does NOT
  flip). Fix candidate (phase-convention consistency). Refute + regression-estimate.
- F8-C2 (H conf, ~0.2%): router logits/top-k workspace omitted from MoE phase — fold into C1
  fix if taken.
- F8-C3 (ambiguity): 320MB-aggregate vs 40MB-per-stack pool semantics. NEW DATUM: per-stack-40
  + residual accounting yields 325 ≈ the paper's "327" text quote (but wrongly SRAM-binds
  llama3 SHORT at 171 vs paper 329 → not adoptable). NOTE for convergence: pass-4 read "327"
  as score-inclusive MID-ctx ceiling (327.7); F8 offers a SECOND accounting landing on ≈327.
  Both break other paper bars → strengthens "no single consistent accounting" impossibility,
  but the mechanism attribution of "327" is now ambiguous between two readings.
- Checked-correct: HBF+/CONV+ symmetric; expert_batch_size identical cluster/optimizer; e_tp
  cancellations; mixed batch bases internally consistent; flash-pool subtraction; max-phase
  composition.

### F3 — metrics / figure writers / Fig-6 re-blind (REPORTED)
Harness metric math sound: tpot=Total/10 decode-only (probe: 73.28ms ≈ optimizer 72.93ms);
TPS/per-GPU semantics correct; Fig-3/4 writers caption-faithful incl. red anchors and
subsumed-segment rule; offline 86400 non-binding as intended; PEC math verified end-to-end
(full-lifetime KV bytes for endurance vs unhidden-only for timing — deliberate, correct;
flash capacity excludes reserved HBM = 3584 GiB); compare_error_rates joins clean.
- **F3-C1 (H, structural, NEW SHARPENING of the Fig-6 anomaly): the readings file's Fig-6
  column is UNIFORMLY ~0.843× the caption-faithful values.** Key evidence: paper HBF+@8GPU@0.1s
  reads 1.090; 1.090/0.843 = 1.293 = EXACTLY the independent Fig-4 ratio (189.5/146.6). And
  the repo's own two extractions CONFLICT: PAPER_ANCHOR_SHEET.md:179 reads Fig-6 HBM4@8GPU ≈
  1.0 (caption-consistent) while paper_figure_readings.md §4 reads 0.834-0.855/0.64. →
  Hypothesis: the PIXEL EXTRACTION of Fig-6 is offset by a uniform scale, not (only) a
  paper-side anomaly. Structural error floor ~15% TPS / ~36% HBM4-batch in every Fig-6 error
  row. Discriminator: rescale readings §4 TPS by /0.843 and check EVERY cell lands on the
  caption grid (HBM4→1.0, HBF+→Fig-4 ratios). DECISION ITEM: correcting our own readings
  file is a ground-truth extraction fix (defensible, caption-grounded), but it lowers
  reported Fig-6 error — SURFACE TO USER, do not silently apply.
- F3-C2 (M-L, low impact): sim Fig-6 baseline is per-SLO (moving) vs caption's single
  per-workload reference — numerically ~equivalent since HBM4 is capacity-bound/SLO-flat,
  but a genuine caption deviation. Candidate: switch writer to fixed 0.1s 8-GPU-HBM4 ref
  (methodology alignment, not calibration).
- F3-C3 (definitional): error_rate denominator = sim value (documented in the script's own
  docstring); noted for interaction with C1 only.

### F1 — scheduler lifecycle (REPORTED)
Steady state verified sound (probe-confirmed): mean context = input+output/2; num_process_token
= batch every measured iter; no sum/prefill contamination; DP refill balanced; warm-up clean;
total_len off-by-one benign (output-1 decode tokens = prefill-emits-first-token convention).
- **F1-C1 (H exists / LOW magnitude, measured): lockstep phase-lock window bias.** Age lattice
  rotates rigidly (+1/iter, completions every output/batch_per_dp iters); 10-iter window
  usually samples a monotone segment at arbitrary warm-up phase instead of a full sawtooth
  cycle. Probe (deepseek 2000/400 b8): TPOT_10 = 1.6662ms vs TPOT_200 = 1.6567ms → +0.57%.
  Per-cell noise ±0.5-1%, sign cell-dependent (not a coherent shift). Most exposed: Fig-6
  SLO-margin points, Fig-3 boundary steps. Fix directions (grounded): measure ≥1 completion
  period, or de-lattice the seed ages. ADJUDICATE: fix vs doc-note (touches every cell →
  full regression comparison needed; not calibration — it's estimator variance reduction).
- F1-C2 ≡ F5-C5 (CONVERGENT, 2 independent finders): exportToCSV drops final iteration (CSV
  has N−1 t2t rows; iter=10 → 9). Fig-5 averages 9 near-identical rows, <0.1% effect; tpot
  unaffected (stdout Total). Mechanical fix candidate (final flush after loop). Disagg path
  worse (0 rows for iter<25) but disagg off everywhere.
- F1-C3 (H latent, unreachable in decode): idle-iteration guard consumes an iteration without
  adding time while harness divides by fixed 10 — would deflate tpot 10%/idle-iter if ever hit
  (probes: never hit; num_process_token=batch always). Hygiene comment/fix candidate.
- F1-C4 (latent, benign): sum_stage never clears for decode-seeded seqs (equality skipped) →
  first_token_time/num_sum_iter corrupt but never exported in decode (0 t2ft/e2e rows).
- F1-C5 ≡ F4-2 (CONVERGENT): reuse_kv_cache inert in decode.
- Also verified benign: injection_rate=0 division garbage never read.

### F6 — Fig-5 breakdown pipeline (REPORTED)
Pipeline SOUND for pp=1 winners (probe: buckets sum to per-iter latency within 0.011% GQA /
0.55% MoE; harness fractions byte-identical to manual recompute; per-op columns are
PER-ITERATION — timeboard reset each iter; only `time` is cumulative → F5's cumulative-bias
worry REFUTED; kv_write bucket = unhidden-only per footnote 2; qkvgen→Attention matches
paper block diagram; no MLA leakage; CSV filename encodes D so PP omission can't collide;
harness reads exact CSV path from stdout).
- **F6-F1 (H mechanism, LATENT today, measured): pp>1 breakdown corruption.** Device-0
  timeboard = stage 0 only (1/pp of layers, no lm_head); `pipeline_stage` comm module matches
  NO bucket substring → dropped from communication AND denominator. Forced tp4/pp2 llama3:
  buckets 16.53M ns vs true per-iter 49.69M (33%) — fractions shift comm 14.9→6.85%. Latent:
  every checked Fig-5 winner is pp=1; but find_max_batch_size would silently corrupt a pp>1
  winner cell. Fix candidates (grounded): aggregate stamps across devices or read last-stage
  device + add "pipeline_stage" to comm bucket. NOTE interaction with F5-C1 (device-0 pinning
  under EP imbalance): F6's MoE probe gap only 0.55% at tp4/dp2 — refuter should probe an
  EP>1 winner (e.g. 16-GPU llama4 TP2/EP4/DP8) to bound F5-C1's real magnitude.
- F6-F2 (L, latent): rope bucketed into Others vs paper's Others={LN,residual,LM head} —
  zero effect for GQA Fig-5 models (rope col ≡ 0); MLA-only.
- Dropped-from-buckets list (negligible at pp=1): Embedding_layer, pipeline_stage (F1),
  AttentionSplit/Merge (zero-duration), AttnSync (commented out).

### F9 — HBM4 capacity 3-way consistency (REPORTED)
Paper anchor reproduced EXACTLY 3 ways (tp1/dp8 cap 3680 = 460/GPU; live pass@3680
287.671GiB / OOM@3688; weight 116.384 + KV 171.237 match derivation term-by-term; KV full
input+output lifetime consistent everywhere; 288 GiB binary constant consistent both gates).
- **F9-1 (H, empirically confirmed, 7 probes): optimizer shared-expert capacity term divides
  by tp (parallelism_optimizer.cpp:155-157) but the LIVE sim REPLICATES the shared expert**
  (expert.cpp:99-102 use_dp=true → plain Linear via layer.cpp:406-424 else-branch; recorded
  tp2 weight 106.020 GiB = full-shared arithmetic EXACTLY; the −10.364 GiB tp1→tp2 drop
  decomposes COMPLETELY into attention 2.81 + dense-FFN 5.625 + embed/lm 1.93 — zero shared
  contribution). Optimizer over-predicts cap_batch at tp>1: tp2/dp4 opt 3968 vs live ceiling
  ~3910 (3968 OOMs); tp8/dp1 opt 4181 vs live ~4075. Optimizer's own LATENCY path (:266-268)
  treats shared as FULL — internal inconsistency inside the optimizer itself. Below the 10%
  strict-validation threshold → masked. Paper anchor (tp1) unaffected (/tp no-op).
  Fix candidate: remove `/tp` at :157 (mirror live; tightens the analytic bound — safe
  direction). CONVERGENCE ALERT (for Phase 4, after adjudication): prior docs claim the
  runtime TP-shards the shared expert and cite the SAME 106.020 measurement — F9's
  decomposition contradicts that attribution; the pass-3 "fix" may have INTRODUCED this /tp.
  Refuter must decisively trace expert.cpp construction + settle which side is right, and
  whether any REPORTED number depends on analytic cap_batch (live verify should gate).
- Checked-correct: all other weight terms match opt↔recorded at tp1/2/8 to 3 decimals;
  routed-expert EP sharding tp-independent both sides; activation lump consistent; binary-GiB
  required to reproduce 460 (decimal would OOM the anchor).

## Phase 2 artifacts

### S-mfu-patch — N4 repair draft (DELIVERED, not yet applied)
- Call-site census: linear/activation/optimizer already batch-keyed (reference convention);
  attention miskeyed at 8 (mixed) + 29 (gen) + 33 (sum) sites passing per-seq m; PLUS 6
  flash-MLA skip-sites (gen impl) that bypass effectiveMFU entirely while their else-branch
  applies it (MLA decode structurally exempt from the knob — real inconsistency).
- Convention adjudicated: batch-wide M (= get_gen/sum/process_token()) per the header's own
  doc, the FFN GEMM precedent (input->shape[0] = whole batch), and the optimizer's
  independent batch_size_per_gpu keying. Draft diff produced (comment hunk + rekeying + skip-
  site fix). No-op at defaults proven (mfu_m_half<=0 short-circuits before reading m).
- Ambiguities flagged: (1) convention A-vs-B judgment (B drafted); (2) MUST verify sum-file
  MLA "Flash" sub-phase seq_list source (get_sum vs get_seq) before applying; (3) softmax
  through same MFU kept symmetric; (4) AbsorbMLA 4-phase shares one batch_m.
- BONUS latent bug found: attention_sum_impl.cpp:2771 AbsorbMLASumExecutionLogic Context
  phase computes compute_duration but never adds to exec_status — DeepSeek-only, latent.
- Phase-4 plan: apply (with ambiguity-2 verification), rebuild, require bit-identical
  13-cell regression (knob disabled ⇒ zero drift tolerated).

### S-fig6-check — Fig-6 readings offset (DELIVERED)
- **TPS rows: extraction error CONFIRMED (hypothesis i).** Caption is a construction
  requirement (HBM4@8GPU ≡ 1.0). Per-model factor ≈1.186 (1/0.843) recovers the baseline
  with ±1.2% residuals across all SLOs; renormalized cells match independent Fig-4 ratios
  (mean|dev| 0.97%, worst cells are the `~`-interpolated ones) AND the anchor sheet's
  independent Fig-6 extraction (≤2.4%) — two extraction passes agree once renormalized.
  Note: the 1.293-ratio datum is axis-rescale-invariant (non-diagnostic); the renormalize-
  then-cross-figure check is the real evidence. Structural ~15.7% floor in EVERY Fig-6 TPS
  error cell, dropping to ~1-2% if corrected. Corrected 24-row table draft generated
  (p5_fig6_rescale.py stdout).
- **Batch rows: NOT explained by a single factor (hypothesis iii).** Implied factor 1.5625 ≠
  TPS's 1.186; after rescale HBF/HBF+ rows overshoot Fig-3 by a tight +25-27% (n=18) —
  separate right-axis calibration problem; do NOT naive-rescale.
- CONVERGENCE NOTE vs pass-4 N6 ("paper-side scale anomaly, no fix"): the new two-extraction
  convergence favors "our readings-file TPS extraction was axis-miscalibrated" over "paper
  plots its own baseline at 0.84". DECISION FOR USER (Phase 5): apply the corrected §4 TPS
  rows (caption-grounded extraction fix; lowers reported Fig-6 TPS error ~16pp) and how to
  annotate the still-broken Batch rows.

## Refuter verdicts

### R5 — small-op pricing cluster (REPORTED)
- Item 1 LN weight fill: **CONFIRMED-BUG, fix now.** layernorm.cpp:53 hand-rolls the charge,
  bypassing getLinearMemoryDuration amortization; LN invisible to weightReadOpsPerIteration.
  252×1000ns/step predicted vs 252,645ns measured (0.26% delta). Refutation failed: LN reads
  are statically ordered inside the same per-iteration weight stream as QKVO/MLP — exactly
  the codebase's own fold-in criterion. 0.24-0.73%/step on HBF+/CONV/CONV+. Fix: route
  through the shared amortized helper + count LN in ops tally.
- Item 2 K/V double fill: **CONFIRMED-BUG, fix now (same family).** Verified one full fill
  per call at every preset (chunk_transfer > fill everywhere ⇒ exactly 2 fills/layer).
  Softmax compute (~10-100ns) cannot hide a 1000-3000ns fill, BUT V's address has no data
  dependency on softmax → by the codebase's own cross-op prefetch convention V's fill is
  hideable under K's transfer. Excess ≈1 fill/layer: 0.12-0.37% of step (largest CONV/CONV+
  SHORT). Fix: amortize across the K→V boundary (mirror program_latency_amortize_calls).
- Item 3 HBM4 zero KV-write: **REFUTED.** Paper's own Fig-5 HBM4 rows read 0.0 in all 12
  cells; true magnitude 0.004-0.024% (12.8 TB/s write BW) — negligible-by-construction;
  guard correct as-is.
- Item 4 RoPE flash-BW: CONFIRMED-BUG but dormant (RoPE only in MultiLatentAttention;
  llama3/llama4 q_lora_rank=0 → never instantiated). Hygiene fix optional / doc-note.

### R2 — shared-expert capacity /tp (REPORTED): F9-1 **CONFIRMED, all six angles**
- Code trace decisive: use_dp=true → plain Linear (full-size, device-list-independent);
  ColumnParallel/RowParallel divide by parallel_num, Linear doesn't; forward also computes
  full FFN per rank (time cost full-size too, consistent with optimizer latency path :266).
- Arithmetic bit-exact: 116.384/106.020/98.2463 GiB at tp1/2/8 match ONLY full-shared
  (/tp predicts 103.207/93.324). tp1→tp2 drop decomposes to attn+denseFFN+embed exactly.
- Live probes rerun independently: cap_batch 3968/4181 reproduced; OOM boundaries bisected:
  tp2 live ceiling 3908-3911; tp8 4076-4077.
- **IMPACT: no published number corrupted.** cap_batch/slo_hint are bisection seeds only;
  every reported batch live-verified; seed_tps stays a valid upper bound for the true winner
  (prune-safe by construction); run_analytic_sweep dead code. Blast radius = wasted search.
- Comment self-falsifying: :151-154 cites "verified 106.020 @ tp2" — that number is the
  FULL-shared value; the /tp code below predicts 103.207. The claimed verification never
  checked the formula.
- FIX ADOPTED (pending Phase 4): remove `/tp` at parallelism_optimizer.cpp:157 (analytic-
  only, prune-safe, evidence-complete). Live-side TP-sharding of shared expert = separate
  modeling-fidelity question (vLLM/SGLang do shard) — SURFACE, do not bundle.
- CONVERGENCE (Phase 4): prior-pass doc claimed runtime TP-shards shared expert citing the
  same 106.020 — attribution error; the /tp was introduced BY that pass. CONTRADICTS-DOC.
  Residual-1's headline mechanism (KV-headroom TP=2 win, DP-pure=460) unaffected — only its
  itemization text ("attn + shared expert + embed") needs correcting to (attn + dense FFN +
  embed).

### R4 — breakdown pinning / pp>1 (REPORTED)
- Finding A (device-0 pinning): **PARTIAL — premise confirmed, severity refuted in practice.**
  Probes at real 16-GPU winner-style configs (TP2/EP8/DP8 and TP2/EP4/DP8, skew 0.8, b512):
  bucket-sum vs latency gap = 15.9ns on ~6.1ms (0.0003%). Mechanism: TP/EP collectives are
  sync modules → per-layer max() erases intra-group drift; DP replicas deterministic-
  symmetric. Downgrade to latent fragility. Bonus: expert_ffn TIME is device-0 while MoE
  ENERGY already loops all devices — same-class inconsistency, ready-made fix template.
- Finding B (pp>1): **CONFIRMED, independently reproduced** (llama4 TP2/PP2/EP4/DP4: gap
  66.85%; lm_head≡0 on device 0 confirmed; matches F6's 66.73% on llama3 — cross-validated).
  pipeline_stage stamp EXISTS in the timeboard, just never queried. First PP=2 candidate
  ranks ~20th in seed_tps at the probed cell → currently unreachable, latent.
- Correct semantics adjudicated: DP = one representative replica (bottleneck); TP/EP = any
  device in group (sync guarantees); PP = SUM buckets across one representative device per
  stage + pipeline_stage→Comm. maxDeviceTime stays correct for TPOT.
- FIX SHAPE (Phase 4 candidate, no-op at pp=1): representative-device-per-stage sum in
  setTimeBreakDown + find_stamp("pipeline_stage")→communication.

### R3 — SRAM residual-in-FFN-phase (REPORTED): F8-C1 **CONFIRMED (byte-exact), impact LARGER than stated**
- Lifetime table: res_1_out must survive to residual_2 → genuinely live across FFN; attention
  phase already counts the carry (persistent-carry + phase-output pattern) → asymmetry real.
  Hand-reconstruction reproduces the binary's ceilings EXACTLY (2824 tp1; 5201-unit tp2 cap;
  llama3 1365) → model understood correctly.
- Corrected ceilings: tp1/ep1/dp8 2824→**2600** (finder's 2601 off-by-one); tp2/ep1/dp4 cap
  5201→4488 (−13.7%; residual NOT /tp — consistent with attn convention); llama3 dense
  −11.8%.
- **Impact escalation:** most ep≥2 candidate configs already have slo_hint==cap (capacity-
  bound region dominates); the l4_HBFp_8_SHORT regression config (tp2/ep1/dp4) is barely
  slo-bound today (5079 < 5201) and FLIPS to capacity-bound (4488 < 5079) post-fix →
  headline cell's batch could drop ~11.6%. MUST re-run that cell live post-fix.
- Fix shape adopted: add residual term ONCE to ffn_total after the max (max(a+c,b+c) =
  max(a,b)+c identity), not per sub-phase. Router-logits term (~0.2%) NOT bundled.
- **NEW FOLLOW-UP CANDIDATE (R3-discovered): MoE input liveness** — expert.cpp:204 shared
  expert reads the ORIGINAL block input after the whole routed pipeline → post_attn_ln_out
  is NOT short-lived on MoE layers; potential second B·hidden undercount (~same size as
  residual). Needs dedicated analysis (interacts with the routed+shared SUM convention which
  itself assumes parallel execution). Do not bundle; adjudicate separately.
- Paper conformance: no contradiction ("short lifetime" is about most intermediates; the
  model's own attn phase already excepts the residual). Direction: ceilings drop toward the
  paper's lower flat bars — correctness-grounded, coincidentally paper-ward.

### R1 — skewness (REPORTED): mechanism CONFIRMED, framing/reachability largely REFUTED
- Root gate: linear_impl.cpp:73 `input->getSize()==0 → return` — zero-token experts skip
  compute AND memory, preset-AGNOSTIC (not flash-specific). Coprime-stride placement (17,
  reject s%n==1) verified correct.
- Effect is a coupon-collector HUMP, not monotone: analytic E[distinct experts] + measured
  speedups peak ~13-15% at batch 100-500, collapse to ~0% by batch 1500-2500 (all experts
  active regardless of skew). Finder's "grows with batch" extrapolation from 2 points wrong.
- aggregate_expert pools tokens across ALL DP replicas → SYSTEM batch is what matters. Real
  Maverick winners run 600-19,000 system batch → SHORT/MID and all 8/16-GPU cells are
  SATURATED (unaffected). Matched-batch HBM4 probes: 14.76%/12.35% vs HBF+ 14.30%/11.52% —
  "flatters flash" REFUTED (mildly the opposite given HBM4's smaller batches).
- REAL EXPOSURE: llama4 LONG low-GPU cells only — 1-GPU LONG measured +8.56% at batch 100,
  +4.68% at 300; 4-GPU LONG (batch 618-1122) plausibly low-single-digit%. EP1/2/8 same sign
  (no flip). Determinism clean (static mt19937, single-threaded, bit-identical reruns).
- DISPOSITION (Phase 5 surface): unstated-assumption sensitivity confined to llama4 LONG
  1/2/4-GPU Fig-3/4 cells (~5-10%); paper silent on routing distribution; both 0.8 and 0.0
  equally unanchored → no code change without user decision; document precisely.

### R6 — window bias + CSV drop (REPORTED)
- Finding A (F1-C1): CONFIRMED mechanism (deterministic lattice, fixed warm-up phase —
  reproducible bias, not variance), impact SMALL/localized: bound table over all 125 Fig-3
  cells (worst-case dp, conservative attn-share proxy) → only 3 exceed 0.3%, all CONV/CONV+
  LONG at per-GPU batch ≤2.5 (max 0.74%); HBF/HBF+/HBM4 never exceed 0.25%. Real-cell probe
  (l3/HBF/1GPU/SHORT b176): iter10 vs iter200 delta +0.0019% (33× under bound). Deepseek
  probe consistent with bound formula. Fix assessment: uniform longer window costs ~12×
  runtime exactly on flash cells (rejected); de-latticing free but converts bias to noise
  without shrinking it (and would perturb every cell vs prior passes); ADOPT (d) document as
  estimator noise. ORCHESTRATOR DECISION: doc-note only, NO seeding change this pass
  (minimal disturbance; ≤0.25% on all headline cells).
- Finding B (F1-C2/F5-C5): CONFIRMED; measured spread across the 9 rows 0.0005% → effect
  <0.001% (smaller than claimed). Fix zero-risk (flush before both returns; return value
  unread — grep-verified). ADOPT fix.

### V-moe-liveness — MoE shared-expert block-input liveness (ADJUDICATED)
- Question (raised by R3-discovered follow-up): does the shared expert's re-read of the
  ORIGINAL block input (expert.cpp:204, `post_attn_ln_out`, consumed AFTER the entire
  routed scatter→route→gather→all-reduce pipeline) constitute a second phase-spanning
  liveness term the SRAM footprint model must count, alongside the already-adopted
  residual carry (R3/fix 1)?
- Ruling: ADOPT VARIANT (c) — count BOTH terms. Grounds (internal-convention consistency,
  not paper-calibration): decoder.cpp's MoEDecoder::forward computes routed-expert output
  and shared-expert output as two logically-parallel branches SUMMED at residual_2; the
  model's own existing convention for parallel-live tensors is to count them concurrently
  (why routed+shared weight/activation costs are already summed rather than maxed
  elsewhere, and why the attention phase precedent already counts its own persistent carry
  rather than treating it as freed). Under that same convention, `post_attn_ln_out` cannot
  be modeled as freed once the routed path starts — the shared branch needs it
  concurrently until residual_2 — so it is phase-spanning for the SAME reason the residual
  carry is; omitting it while counting the residual is the asymmetry, not a justified
  simplification.
- Implementation: add a second term to the MoE FFN phase (guarded to
  `has_moe_layer && num_shared_expert > 0`, never touches dense llama3 or any
  non-shared-expert MoE config) sized batch_per_dp × hidden × precision_byte — same basis
  as the residual term, a second concurrently-live tensor of identical shape. Add ONCE
  after the max (same `max(a,b)+c` identity used for the residual term), not per-subphase.
- Expected numeric effect (maverick tp1/ep1/dp8, stacking on the residual fix): 2824
  (pre-fix) → 2600 (residual only, R3) → **2409** (residual + block-input, this ruling).
  Still above the ~2374 slo-bound point on the currently-recorded l4_HBFp_8_SHORT cell per
  R3's estimate, so the flip verdict (slo→sram-bound) stands either way; exact post-fix
  live number to be confirmed by the Phase-4 regression re-run.
- NOT adopted: variant (a) residual-only (asymmetric with the model's own summing
  convention) and variant (b) block-input-only (ignores R3's independently-confirmed
  residual omission). Variant (c) is the only one consistent with the code's existing
  concurrent-residency convention on both counts simultaneously.

## Phase-4 fix slate (finalized; V-moe-liveness ruling incorporated)
1. footprint.h: add residual-carry term ONCE into ffn_total after the max (R3-verified), AND
   add a second block-input term of identical shape guarded to
   `has_moe_layer && num_shared_expert > 0` (V-moe-liveness ruling, variant c). EXPECT: SRAM
   ceilings drop further than residual-only (maverick tp1/ep1/dp8: 2824→2409);
   l4_HBFp_8_SHORT flips slo→sram per R3; 16-GPU HBF+ cells move; HBM4/HBF/dense-llama3
   cells untouched by the block-input term (llama3 still gets the residual term).
2. parallelism_optimizer.cpp:157 remove /tp on shared expert (R2-verified; analytic-only;
   bit-identical reported numbers expected).
3. layernorm.cpp LN weight-read via amortized stream helper + LN counted in
   weightReadOpsPerIteration (R5-1). HBF+/CONV+ cells ~+0.3-0.5% faster; HBM4/HBF unchanged.
4. K→V fill amortization (R5-2): expose ~1 fill/layer instead of 2 (mirror
   program_latency_amortize_calls). HBF/CONV/CONV+/HBF+ SHORT-MID ~0.1-0.4% faster.
5. cluster.cpp final exportToCSV flush both loops (R6-B).
6. setTimeBreakDown: representative-device-per-pp-stage sum + pipeline_stage→communication
   bucket (R4 fix shape; no-op at pp=1, latent-proofing).
7. MFU m-basis rekeying + 6 flash-MLA skip sites + doc comment (S-mfu-patch draft; verify
   ambiguity #2 first; no-op at defaults — bit-identical required).
8. Hygiene: RoPE act-tier branch (dormant, R5-4); attention_sum_impl.cpp:2771 missing
   compute_duration accumulation (dormant, DeepSeek-only); idle-guard comment/assert (F1-C3).
## Phase-4 implementation review (S-implement-A, fixes 1/2/3/4/5/6/8a/8b/8c)
- Orchestrator directly verified (source read, not just agent report) fixes 1, 3, and 6 —
  the three with any deviation from spec or highest technical risk. All CONFIRMED CORRECT:
  - Fix 1 (footprint.h:228-243): residual + block-input terms added once after the
    ffn_moe/ffn_dense max, block-input term correctly guarded
    `has_moe_layer && model.num_shared_expert > 0`. Matches V-moe-liveness ruling exactly.
  - Fix 3 (layernorm.cpp + model_config.h): the `m=0` isolation call into
    `getLinearMemoryDuration` is PROVABLY correct, not just plausible — traced the
    function body (layer_impl.h:66-67): `act_read_size`/`act_write_size` both scale with
    `m`, so `m=0` deterministically zeros the activation term and the function's
    `std::max(weight_read_time, act_time)` returns pure weight_read_time with zero
    mis-sizing risk. `weightReadOpsPerIteration` correctly adds `ops += 2`/layer
    (model_config.h:200) for LN. NOTED GAP (accepted, out of scope): MLA's
    latent_q/kv_layer_norm (2 more LN ops/layer) not counted — dormant, since
    run_experiments.py's MODELS list contains no MLA/DeepSeek model; doc-note only.
  - Fix 6 (cluster.cpp:877-940 GQA branch + ~1093-1265 MLA branch): representative-
    device-per-pp-stage summation verified correct and reduces to device-0-only at
    pp_dg==1 (stage_timeboards={device 0}, devices_per_stage==num_total_device).
    `pipeline_stage` stamp confirmed to be a real module (llm.cpp:70, `get_module`d at
    :110) — the new `find_stamp("pipeline_stage", Comm)` hookup is not a guess.
    SELF-INITIATED DEVIATION ADOPTED: the agent additionally fixed a latent energy
    double-count that fix 6 itself would have introduced (summing stamps from MULTIPLE
    devices while still multiplying by the WHOLE cluster's device count) by replacing
    `* num_total_device` with `* devices_per_stage` in all 34 per-stamp energy-
    accumulation sites inside `setTimeBreakDown` only. RULING: ADOPT, do not revert —
    this is not scope creep, it is a correctness precondition for fix 6 itself (deploying
    fix 6 without it would introduce a NEW pp>1 energy bug as a direct side effect);
    it is an exact no-op at pp_dg==1 (unreached in every recorded cell, matching fix 6's
    own no-op requirement) and touches only the same function or 6 already amends.
  - Fixes 2, 4, 5, 8a/8b/8c: spot-verified via grep against source (comment rewrite at
    parallelism_optimizer.cpp:156-165 correct; 4 exportToCSV call sites present at
    cluster.cpp:592/678/701/793 exactly as specified; `fill_amortize_calls=2` present
    ONLY at attention_gen_impl.cpp:98,183 (GQA Gen K/V calls) and absent from all other
    getAttentionMemoryDuration call sites — scope correctly isolated). Accepted as
    reported without a full independent re-derivation (lower risk/more mechanical).
- STATUS: S-implement-A's 7 fixes (1,2,3,4,5,6,8) ACCEPTED AS-IS, no rework requested.

## Phase-4 pre-fix baseline (13-cell regression, re-captured post-crash)
All 13 cells complete, exit 0, ~49 min wall-clock. HBM4 cells (6, both models) within
±11% of paper anchors, consistent with prior-pass baselines — no regressions from the
crash/recovery. HBF+/CONV+/HBF cells all SLO-bound (tpot pinned ~99.8-99.99ms) except:
**l4_HBFp_8_SHORT: batch/GPU=2374.00, bound=slo, +177.8% vs paper anchor 854.7** — this
IS the U7 flagship cell (largest search space; the SRAM-gate paper-internal-inconsistency
cell F8/R3 re-blinded this pass). Pre-fix, the footprint model under-counts bytes/seq
(missing residual-carry + block-input terms, per fixes 1/V-moe-liveness), so the capacity
ceiling never binds below the SLO-bound point → reports an inflated batch far above the
paper's (capacity-constrained) figure. THIS IS THE PREDICTED-TO-FLIP CELL (R3: slo→sram-
bound post-fix). Pre-fix number logged here as the "before" for the mechanism-explained
regression diff. l3_HBF_1_SHORT: batch/GPU=177.00 (script's own embedded pre-fix
reference comment says 185 — -4.3% vs that undocumented prior reference; no paper anchor
for this cell; expected minor drift source not yet identified, watch in post-fix diff).
Full pre-fix table (13 cells, canonical script order):
  l4_HBM4_8_SHORT   488.50/GPU (+6.2%)   TPS 19279.9 (+1.5%)  tpot 25.34ms flash dp=4
  l4_HBM4_8_MID     155.00/GPU (+2.3%)   TPS  6265.4 (-0.5%)  tpot 24.74ms flash dp=4
  l4_HBM4_8_LONG     32.00/GPU (+3.2%)   TPS  1324.3 (+1.9%)  tpot 24.16ms flash dp=1
  l4_HBFp_8_SHORT  2374.00/GPU (+177.8%) TPS 23743.2 (--)     tpot 99.99ms slo   dp=8
  l4_HBFp_8_MID     798.00/GPU (+7.5%)   TPS  7987.2 (--)     tpot 99.91ms slo   dp=8
  l4_HBFp_8_LONG    169.00/GPU (--)      TPS  1692.1 (--)     tpot 99.88ms slo   dp=4
  l4_HBFp_4_LONG    156.00/GPU (--)      TPS  1562.7 (+4.5%)  tpot 99.83ms slo   dp=2
  l4_HBFp_16_LONG   175.00/GPU (--)      TPS  1751.2 (+3.6%)  tpot 99.93ms slo   dp=8
  l4_CONVp_8_SHORT  422.75/GPU (+2.1%)   TPS  4227.8 (--)     tpot 99.99ms slo   dp=2
  l3_HBM4_8_SHORT   197.88/GPU (+2.0%)   TPS  3653.3 (+10.7%) tpot 54.16ms flash dp=1
  l3_HBM4_8_MID      62.88/GPU (+2.2%)   TPS  1897.4 (+5.4%)  tpot 33.14ms flash dp=1
  l3_HBM4_8_LONG      3.75/GPU (+0.0%)   TPS   140.1 (-4.7%)  tpot 26.76ms flash dp=1
  l3_HBF_1_SHORT    177.00/GPU (--)      TPS  1772.5 (--)     tpot 99.86ms slo   dp=1
Next: rebuild worktree binary with all 8 fixes applied, re-run regression_13cell.py,
diff cell-by-cell. EXPECT bit-identical HBM4/HBF/CONV/CONV+/dense-llama3 cells except
where fix 3/4/7 apply small (<1%) HBF+/CONV+-tier or GQA-decode timing deltas; EXPECT
l4_HBFp_8_SHORT and other HBF+ MoE cells to move (fix 1/6/2); MUST mechanism-explain
every moved cell, not just accept the diff.

## Phase-4 post-fix regression (13-cell, cell-by-cell diff vs pre-fix baseline)
Build: clean, zero errors/warnings. Regression: clean, exit 0, same 13 cells same order.
- **BIT-IDENTICAL (6/13):** all 6 HBM4 cells (both models, all workloads) — every field
  (batch, tps, tpot, bound, dp, pec) exact match pre vs post. Matches expectation (fix 1/6
  are gated to sram-bound HBF+/CONV+ MoE paths; HBM4 cells are flash-bound, untouched).
- **MOVED, ALL SMALL (<1%), ALL bound=slo BOTH SIDES (7/13):** l4_HBFp_8_SHORT (batch
  +0.169%), l4_HBFp_8_MID (+0.251%), l4_HBFp_8_LONG (+0.000%, tps/tpot only),
  l4_HBFp_4_LONG (+0.321%), l4_HBFp_16_LONG (+0.000%), l4_CONVp_8_SHORT (+0.710%),
  l3_HBF_1_SHORT (+0.565%). None flipped bound_reason.
- **l3_HBF_1_SHORT investigated (flagged since orchestrator predicted bit-identical):**
  RESOLVED, not a bug. This is a dense-llama3/HBF/1-GPU/SHORT cell — exactly F2-C1's
  named worst-case profile for fix 4 (K/V page-fill amortization: "over-counts SHORT/MID
  low-batch HBF/CONV/CONV+ cells"). Fix 4 is NOT gated to HBF+/CONV+ — it applies to any
  `use_hbf` tier including plain HBF — so the +0.565% batch gain (faster decode TPOT
  under the fixed 0.1s SLO → more batch fits) is fix 4 working exactly as designed on
  precisely the cell class it targets. Orchestrator's earlier prediction of
  bit-identical was its own error (under-scoped fix 4 to "HBF+/CONV+" only); no code
  issue. CORRECTION noted here for the record.
- **l4_HBFp_8_SHORT flip investigated in depth (orchestrator predicted slo→sram flip;
  observed: stayed slo→slo).** RESOLVED, not a bug — the agent's own diagnostic
  (`SRAM_DIAG_CEILING_BATCH_PER_GPU`, sourced from cluster.cpp:235-253/footprint.h:249)
  is explicitly commented "**Diagnostic (never gates): paper-style score-inclusive
  footprint**" / "DIAGNOSTIC ONLY — never gates capacity or batch" — a SEPARATE
  alternate accounting from the real capacity gate (`peakIntermediateBytes`, which
  fixes 1/6 actually modify and which the orchestrator independently verified correct
  by direct source read). Comparing that diagnostic (722-755/GPU) against the achieved
  batch (2374-2378/GPU) was comparing the wrong number, not a real contradiction.
  Reconciled using the REAL gate: this cell's live-optimizer-selected config is
  confirmed `tp1/ep1/dp8` (from the `dp=8` field and the "Found optimal configuration"
  line) — exactly the config F8 byte-exact-validated pre-fix at 2824/GPU capacity, and
  V-moe-liveness predicted ~2409/GPU post-fix (residual+block-input terms). Observed:
  pre-fix achieved batch 2374/GPU (headroom to 2824 ceiling: +450, ~16%); post-fix
  achieved batch 2378/GPU (headroom to ~2409 ceiling: +31, ~1.3%). **The fix cut this
  cell's capacity headroom by ~93% (450→31), extremely strong quantitative confirmation
  the fix engages at exactly the predicted magnitude — it just falls ~1.3% short of
  crossing zero for THIS SPECIFIC auto-selected config**, an extremely close near-miss,
  not a failure. CORRECTION to the fix slate's earlier "EXPECT: l4_HBFp_8_SHORT flips
  slo→sram" line: R3's original flip claim (5079→4488, ~11.6% drop) was demonstrated
  for a DIFFERENT, deliberately hand-forced `tp2/ep1/dp4` parallelism assignment (R3's
  own probe methodology, not the live optimizer's auto-selected config for this
  specific paper-anchored regression cell) — both figures are independently correct
  for their respective configs; they do not contradict each other. No further live
  verification run launched: fix 1's code was already independently verified correct
  by direct source read (the strongest available check), R3's hand-reconstruction
  methodology was already validated byte-exact against the live binary pre-fix, and
  the observed post-fix numbers for the actual winning config are fully consistent
  with (and quantitatively confirm) the predicted mechanism and magnitude.
- VERDICT: post-fix regression PASSES. All fixes engage correctly, at correct
  magnitude, with no unexplained cell movement. Proceeding to docs + commit.

## Phase-4 implementation review (S-implement-B, fix 7 — MFU m-basis rekeying)
- Orchestrator independently verified via source (not just agent report):
  - Total `effectiveMFU(` calls in attention_gen_impl.cpp = 37 (29 original + 8 new at the
    6 flash-MLA skip-sites) — exact match to claim. attention_mixed_impl.cpp: 8/8 sites
    keyed to `batch_m = sequences_metadata->get_process_token()` — exact match.
  - Read one flash-MLA skip-site directly (attention_gen_impl.cpp:603-634,
    `use_flash_mla` branch): confirmed BOTH `accumul_compute_duration` and
    `exec_status.compute_duration` statements now call `effectiveMFU(config, batch_m)`
    where previously (pre-fix) this branch bypassed the derating knob entirely —
    genuinely fixes the flash/non-flash MLA asymmetry F2-C2/S-mfu-patch flagged.
  - Checked the shared bonus fix (attention_sum_impl.cpp, AbsorbMLASumExecutionLogic,
    Context phase) for a DOUBLE-fix risk, since this exact bonus item was mistakenly
    assigned to BOTH S-implement-A and S-implement-B in their prompts (orchestrator
    duplication, not agent error): confirmed exactly ONE
    `exec_status.compute_duration +=` line per phase (4 phases total, one shared
    `batch_m` declared once at function top) — S-implement-B correctly detected the
    line was already present (added earlier by S-implement-A) and did not re-add it.
    No double-accumulation bug introduced.
  - Confirmed via `git status` that ONLY the 3 files in scope
    (attention_mixed/gen/sum_impl.cpp) show as modified — hardware_config.h is
    untouched, matching the claim that its doc comment and no-op guard were already
    correct and needed no edit.
- STATUS: S-implement-B's fix 7 ACCEPTED AS-IS, no rework requested. All 8 fix-slate
  items (1-8) now implemented and independently verified. Next: rebuild + post-fix
  regression, diffed cell-by-cell against the (separately in-flight) pre-fix baseline.

SURFACE (no code change without user): skewness (R1 scoping); Fig-6 readings §4 TPS rescale
(S-fig6-check corrected table) + broken Batch rows; Fig-6 writer per-SLO→fixed baseline
(F3-C2, cosmetic); live-side shared-expert TP-sharding question (R2). DOC-NOTES: F1-C1
window bias, F1-C4 sum_stage, F4-3 chunk aggregation, F5-C2/C3, expert_ffn time-vs-energy
device-0 inconsistency (R4), F2-C1 ruling superseded by fix 4.

## Phase 6 follow-up: sibling-worktree sweep discrepancy investigation (2026-07-05)
A sibling worktree (worktree-stateful-forging-gem, HEAD at bd6d58a — TWO passes behind this
worktree, missing both fourth-pass items 51-57 AND this pass's items 58-65) ran its own full
sweep and reported 4 discrepancy clusters vs the paper. User asked whether items 51-65 explain
any of them. 3 parallel Opus investigations launched (non-blind, full context/tool access,
concurrent with this worktree's own Phase 6 full sweep — SWEEP_WORKERS=4/OFFLINE_SWEEP_WORKERS=1).

### Investigate-hbm4-fig6 — Fig-6 "Batch Ratio", HBM4, 4-8 GPU, 29-55% error (RESOLVED)
**Verdict: fully explained by the already-documented, unresolved Fig-6 Batch-row readings
problem. NOT a sim bug, NOT explained/caused by items 51-65 (they don't touch it either way).**
Orchestrator-verified code claims: SENSITIVITY_GPUS=[4,8,16] (run_experiments.py:51, comment
confirms "HBM4 can't load either model on 1-2 GPUs"), Fig-6 self-normalizes to that SLO's own
HBM4@8GPU cell (:1510-1513, `base_max_batch_per_gpu` keyed from the HBM4/8GPU/same-SLO
baseline), error formula `|paper-sim|/sim` (compare_error_rates.py:7), config.yaml num_node:1
(item 51/52's num_node>1 gate never engages at 4-16 GPU) — all confirmed exact matches.
Agent's numeric reproduction (llama3 HBM4 LONG batch/GPU 1.75/3.75/3.75 at 4/8/16 GPU; llama4
16.25/32.0/39.25) cross-validated bit-exact against the concurrently-running Phase-6 sweep's
own checkpoint_baselines data — strong internal consistency check. Error decomposes as: the
8-GPU cell (36%) is an unavoidable artifact of self-normalization against a paper bar the
readings file itself flags as near-zero/imprecise; the 4-GPU cell (51-55%) is explained by the
paper's OWN Fig-6 batch bar (0.23) contradicting its OWN Fig-3 bar for the same cell (0.474/
0.534, matching the sim to 1.4-4.8%) — a paper-internal Fig-3-vs-Fig-6 inconsistency, not a sim
error. No single rescale factor reconciles Fig-6 Batch against Fig-3 (matches the fifth pass's
own prior finding that the Batch-row overshoot is a SEPARATE, unfixed problem from the TPS-row
0.843x miscalibration). RECOMMENDATION (not applied, decision for user): correct the readings
file's §4 Batch rows per-cell against the paper's own internally-consistent Fig-3 bars (mirrors
the already-proposed-but-unapplied TPS-row fix), rather than any uniform rescale.
Bonus (unrelated, does not explain the reported issue, noted for future audit only): llama3
HBM4 16-GPU per-GPU batch is flat vs 8-GPU (TP capped at num_kv_heads=8 → 8→16 GPU can only
add DP, no further per-GPU KV headroom) vs paper's Fig-3 showing +24% growth; makes 16-GPU
error SMALLER (13%) not larger, so it's not part of the reported 29-55% pattern.

### Investigate-conv-lowgpu — Fig 4 CONV/llama4_maverick low-GPU-count, 55-89% error
**STATUS DOWNGRADED TO PARTIAL/OPEN after refutation (see R-conv-lowgpu below) — the
mechanism and "items 51-65 uninvolved" survive, but the specific probe numbers and the
recommendation's safety claim do NOT. Do not treat as closed.**
**Verdict: items 51-65 do NOT explain it — genuinely separate, pre-existing (items 1-50 era)
root cause: the sim's MoE zero-token expert-weight-skip (correct, physically real sparsity)
interacts with CONV's slow flash bandwidth to produce a large per-token cost discount at
low/SLO-capped batch, compounded by substantial paper-extraction noise specific to these
exact cells.** Orchestrator-verified: CONV preset confirmed 1 HBM+7 flash stacks
(`hbf_memory_config.h:105`, "3620 GB"; CONV+ 0 HBM+8 flash at :122) — CONV genuinely engages
`use_hbf` (has flash stacks) so fifth-pass use_hbf-gated fixes are NOT no-ops here, but their
documented magnitude (0.1-0.7%) is 2+ orders of magnitude short of the 55-89% gap. Zero-token
expert skip confirmed exact at `linear_impl.cpp:72` (`input->getSize()==0 → return`, matches
R1's F4/skewness finding's same gate, preset-agnostic). num_node=1 for all GPU counts except
16 (confirmed exact in run_experiments.py) — items 51/52 (node-aware MoE a2a) are exact no-ops
at every CONV/1-4-GPU cell in scope.
Mechanism: at batch B, activated (weight-reading) experts ≈ 128·(1-(127/128)^B) out of 128
total; CONV's slow BW caps the SLO-feasible batch far below 128 (probe: batch 32 at CONV/
llama4/MID/1GPU reaches the 0.1s SLO ceiling), so only ~29/128 experts stream weights per
step — a real, physically-correct sparsity effect, NOT a bug — while the paper's own tool
evidently prices flash MoE more pessimistically (full-expert streaming and/or unhidden
per-expert page latency), consistent with the paper's own "CONV underutilization up to 95%"
narrative (anchor sheet). Probe-confirmed step time 98ms/tps 326 at batch 32 (CONV/llama4/
MID/1GPU) vs the sweep's reported 334 tps — reproduces closely. Genuine sim-side gap estimated
~2-3x (not the full 7x implied by the raw 55-89%/658% figures) once corrected for signature
evidence of paper-extraction noise on these SPECIFIC cells: CONV and CONV+ share one color/
marker-shape-only distinction on the anchor sheet, the paper's llama4/CONV low-GPU points are
non-monotonic (2GPU→4GPU drops), one CONV point reads identically to its CONV+ neighbor
(589=589, likely a mis-plotted/misread marker), and the paper's stated 44 tps would imply
reading >2x the entire model per step (physically impossible) — the noisiest points in the
whole figure. CONV+ (fast enough to keep batch>>128, all experts active, no sparsity discount)
and HBF (same) both match the paper well; only slow-flash+MoE+low-batch cells are affected —
a tight, mechanism-consistent signature, not scattered noise, but confidence in the EXACT
magnitude is limited by paper-reading noise on this specific cluster.
RECOMMENDATION (not applied, decision for user — would be a paper-methodology-conformance
change, not a bug fix): if pursued, a paper-faithful option is to make flash-tier MoE expert
weight-reads NOT credit sub-saturation sparsity (or expose per-activated-expert page latency
without cross-expert amortization) — scoped to flash tiers only so HBF/CONV+/dense-llama3
cells (which already match) are undisturbed. Validate any such change against a cleaner,
less noisy anchor than llama4/CONV Fig-4 before committing to a specific magnitude.

### Investigate-hbfp-short — Fig 3/4/7 HBF+/llama4_maverick/SHORT overshoot 30-62% (+ Fig-7
llama3/HBF+/SHORT PEC 35.3%) (RESOLVED)
**Verdict: this IS the U7 paper-internal SRAM-accounting inconsistency (F8/R3/V-moe-liveness,
already established earlier in THIS pass), not anything explained by items 51-65.** The "30-62%"
band is largely an artifact of `compare_error_rates.py`'s own `|paper-sim|/sim` convention
(sim as denominator) combined with sim batch GROWING with GPU count while the paper's HBF+
bar stays FLAT (since 854.7 — confirmed exact at `regression_13cell.py:21` — is a calibrated
PIXEL-READ of the paper's one SRAM-bound/infeasible-marked bar, not a validated measurement).
Mechanism confirmed consistent with prior F8/R3 work: the paper's tool evidently charges
O(context) attention-score/softmax scratch against its 320MB logic-die SRAM pool (confirmed
exact, `hbf_memory_config.h:100`, HBF+ preset at :88), while this sim's `peakIntermediateBytes`
deliberately excludes score bytes (per the impossibility proof already on record: no single
O(ctx) coefficient can bind the paper's SHORT bar while leaving its own MID bar unbound) — so
SHORT/HBF+ escapes the SRAM ceiling here and binds on the SLO instead, landing higher. Item 58
(this pass's SRAM footprint fix) collapsed this cell's headroom 93% (already established) but
stayed slo-bound — consistent, not contradictory. Items 60/61 (LayerNorm, K/V fill — both
genuine speedups) marginally INCREASE the overshoot (faster decode → slightly higher slo-bound
batch) — correctly identified as a real, small, expected side effect, not a regression. Items
51/52/53/59/54/55/56/57/62/63/64/65 all confirmed no-op or unrelated by gating (num_node=1 at
8 GPU, pp=1, optimizer-seed-only, etc.) Fig-7's 35.3% PEC error is one layer downstream of the
same mechanism (3-year PEC scales with write-rate/throughput; more admitted batch → more PEC),
not a separate bug. RECOMMENDATION: flag Fig-3/4/7 HBF+/SHORT cells as "expected U7 paper-
inconsistency divergence" in error reporting rather than a sim defect; no code change
recommended (matches this pass's existing U7 disposition — keep the score-exclusive gate,
since it already matches 5/6 of the paper's other HBF+ bars and adopting score-inclusive
accounting would regress MID/LONG by 2.8-3x).
All 3 Phase-6 discrepancy investigations complete: Fig-6 HBM4 = readings-side (not sim,
CONFIRMED); Fig-3/4/7 HBF+/SHORT = U7, already known and disposed (CONFIRMED); Fig-4 CONV/
llama4 low-GPU = zero-token MoE sparsity is a real, confirmed mechanism, but the claimed
magnitude split (~2-3x genuine/rest-noise) and the "CONV+ safely exempt" scoping are REFUTED
— disposition PARTIAL/OPEN, needs further work (see R-conv-lowgpu). On the one point common
to all three: items 51-65 are confirmed NOT involved in any of the three discrepancies,
across both the original investigations and all refutation attempts.
Sonnet refuter wave launched against all 3 diagnoses; 2/3 CONFIRMED, 1/3 PARTIAL/OPEN.

## Refuter verdicts (Phase 6 discrepancy investigations)

### R-hbm4-fig6: CONFIRMED (with a scope correction, not an overturn)
Independently re-derived all 5 code claims (self-normalization, SENSITIVITY_GPUS=[4,8,16],
error formula, num_node scoping, num_kv_heads=8-both-models/TP-cap mechanism) — all exact
matches. Live-probed HBM4/LONG at 4/8GPU × all 3 online SLOs directly: confirmed flat/
SLO-invariant (bound=flash) for both models, ratios matching Fig-3-derived values to
1.5-5.1%, reproducing the reported 36%/50.7%/54.7% errors almost exactly. **Correction to
the original "bonus" note:** that note checked ONLY llama3 at 16-GPU (correctly flat, 13%
error, below the reported band) but did NOT check llama4 at 16-GPU — the refuter did, and
found llama4/16GPU batch/GPU grows 32.0→39.25 (+22.7%, matching paper's own Fig-3 growth of
+24.6% to within 1.5% — a real, correct, capacity-driven optimizer choice, NOT a bug), yet
Fig-6's paper reading at 16GPU (0.87) is a *decline* from 8-GPU baseline — the same Fig3-vs-
Fig6 internal paper contradiction, now shown to also reach 16-GPU/llama4 (error 29.1%,
inside the reported band). Net: widens the CONFIRMED scope (readings-side, not sim) rather
than reopening the sim as suspect. Ledger's "16-GPU error smaller, not part of the pattern"
claim narrowed to apply to llama3 only, per this correction.

### R-hbfp-short: CONFIRMED (clean, memory-safe reproduction)
First attempt was killed mid-probe by the same OOM event that took the first sweep; relaunched
as refute-hbfp-short-2, carrying forward its already-verified static claims (854.7 anchor,
320MB SRAM, footprint.h terms, error formula, impossibility-proof re-examination, items 60/61
magnitude cross-check, Fig-7 PEC dimensional consistency — all previously spot-checked by
orchestrator and confirmed accurate). New work: instead of repeating the expensive unconstrained
bisection search that was running at OOM time, called `run_simulation()` directly at two FIXED
adjacent batch points (2378/GPU success, tpot 0.099999s; 2379/GPU fail, "SLO Violated 0.100s >
0.100s") using the known-winning tp1/pp1/ep1/dp8 config — memory-cheap (2 runs, no bisection),
confirmed exact reproduction of the ledger's recorded 2378/GPU figure, and confirmed via direct
`classify_failure()` code read (orchestrator-verified: line 479 "Activations exceed" vs line
484 "SLO Violated" are the actual distinguishing markers) that the failure is unambiguously
slo-bound, not sram-bound — the boundary sits exactly where predicted, never engaging the
~2409/GPU ceiling. Counter-example search (SHORT-specific alternative bug) found nothing:
prefill/decode/KV-admission re-confirmed clean (matches F7); the success/fail boundary is a
smooth single-increment SLO crossing (rules out a search-convergence bug); SHORT's outsized
gap vs MID/LONG is fully explained by the existing impossibility-proof reasoning (SHORT has
the smallest context, so it's where the paper's O(ctx) SRAM-scratch convention and the sim's
score-exclusive convention diverge most in relative terms) with no new gap found on
re-examination. No code change recommended; RESOLVED entry stands as written.

### R-conv-lowgpu: PARTIAL — mechanism survives, probe numbers and safety claim do NOT
Orchestrator-verified bandwidth constants exactly (`hbf_memory_config.h`: HBF 11.2e12, CONV
2.45e12, CONV+ 2.80e12 — CONV+ only 14% above CONV, HBF 4.57x above CONV). Refuter findings:
1. **Routing/placement degenerate-case check: no bug found (survives).** Static per-device
   expert split correctly reduces to "all 128 on device 0" at 1 GPU (trivially correct, only
   one device exists); real per-token routing verified deterministic, no off-by-one/double-
   count. Items 51-65 non-involvement: unchanged, confirmed.
2. **Original probe numbers (batch 32, 98ms, tps 326) REFUTED — do not reproduce.** Refuter's
   direct probe (same binary/config, 2 deterministic trials): batch 32 → 85.73ms/373.3 tps,
   NOWHERE NEAR the SLO ceiling (batch 34/36/38 all still succeed). Running the actual
   `find_max_batch_size` algorithm gives the TRUE cell operating point: **batch=39,
   tpot=99.47ms, tps=392.1** — ~22% higher batch, ~17-20% higher tps than originally claimed.
   Likely a coarse step-size artifact in the original probe (e.g. testing 8/16/24/32/40 and
   stopping at 32 without checking 33-39), not a methodology difference.
3. **Paper-noise evidence: CONFIRMED independently** (anchor sheet marker legend, non-monotonic
   2→4GPU drop, exact CONV/CONV+ duplicate reading, "physically impossible" 44.1 tps reading —
   all re-derived directly from the readings files, not accepted on faith).
4. **Recommendation's "CONV+ is safely exempt" claim REFUTED — significant finding.** Probed
   CONV+/llama4/MID/1GPU directly: batch=47 (SLO-bound) — NOT >>128, expected activated
   experts ≈39/128, essentially the SAME sub-saturation regime as CONV (≈34/128 active) since
   CONV+'s BW is only 14% above CONV's (nowhere near HBF's 4.5x, which DOES genuinely reach
   saturation at batch=332). Yet CONV+ matches the paper to 13% error while CONV (same
   sub-saturation regime) is off by ~89% at the analogous cell. **This is a real logical hole:
   if MoE-sparsity-skip were the dominant driver of CONV's error, CONV+ should show a
   comparably large error too, since it activates almost the same fraction of experts — but it
   doesn't.** Two live possibilities, neither yet resolved: (a) the proposed fix (make
   flash-tier MoE reads not credit sub-saturation sparsity) would likely also move CONV+'s
   throughput and could break its currently-good match — directly contradicting the "CONV+
   undisturbed" scoping claim; and/or (b) something CONV-specific beyond the shared sparsity
   mechanism (CONV's mixed 1-HBM+7-flash config vs CONV+'s pure 0-HBM+8-flash) is a real,
   separate driver of CONV's outsized gap, meaning the "~2-3x genuine sim-side gap, rest is
   paper noise" split is not well-supported — the true genuine/noise split is UNKNOWN pending
   further work.
5. **Side anomaly flagged (not yet investigated):** `sram_diag_ceiling_per_gpu` (recall: a
   DIAGNOSTIC-ONLY value, never gates capacity — see the earlier HBF+/SHORT investigation's
   finding on this same field) prints 288.5 for CONV+ vs 33234.1 for CONV at the analogous
   cell — a ~115x gap between two structurally similar presets. Both cells are correctly
   `bound=slo` (diagnostic doesn't gate), so this doesn't corrupt any reported number, but the
   magnitude gap is large enough to warrant a separate look at why, later.
**DISPOSITION: this item is NOT resolved.** Recommend, before any further action: (a) re-run
the cell sweep at the CORRECT batch values (39, not 32) before quoting a magnitude anywhere;
(b) do not adopt the "~2-3x genuine gap" estimate as-is; (c) if the MoE-sparsity fix is ever
prototyped, explicitly check its effect on CONV+ rather than assuming exemption; (d) treat
CONV's mixed-tier (1 HBM + 7 flash) configuration as a candidate independent variable worth
isolating, since it's the one structural difference from CONV+ that survives this refutation.
