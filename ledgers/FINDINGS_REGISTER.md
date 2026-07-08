# Findings register — Paper-1 bug-hunt campaign (2026-07-07)

Protocol: 5 parallel Opus hunt tracks (A fig4-short-hbf/MFU, B 16-GPU PP serialization, C fig6-slo +
CONV-llama3 + U7 refresh, D fig7-pec chain, E fresh-bugs sweep) + Sonnet refuters (Refuter A on the
MFU grid, Refuter B on PP) + orchestrator adjudication, all confined to worktree
`bughunt-paper1-campaign` @ c2afcf8 (post intermediate-data-gate merge). Full findings + every
reproduction script: `FINDINGS_REPORT.md`; hunt-by-hunt narrative: `CAMPAIGN_NOTES.md`. **No source
files were modified by this pass** — the two proposed code fixes below are tracked as `BUGS.md` items
19-20 pending review/fix; doc updates landed in `PAPER_INCONSISTENCIES.md` and `BUGS.md`.

## Executive summary (verdicts)
- **PP runtime 3× stage over-count** — NEW BUG, CONFIRMED byte/ratio-exact (measured/stage
  2.999/3.005/3.009/3.014/3.017 at batch 16/64/128/256/400, `huntB_disc.py`). Root cause: item-35's
  `max()`→`+=` flip compounds through `sync_devices()`'s max-broadcast on every tp≥2 stage. FIX IN
  PROGRESS (`BUGS.md` #19).
- **PP decode model (serialized vs pipelined)** — Methodology deviation, CONFIRMED: the faithful
  pipelined model (period `W+K·B/pp`) reproduces the paper's entire 16-GPU Fig-3/Fig-4 growth (MID
  80.3/GPU, 2011 TPS vs paper 74.9/1906.8; LONG 4.75/GPU vs paper 4.6; SHORT near-tie, flat like the
  paper). Correct-serialized (2S) alone reproduces none of the growth. Pairs with the item above
  (`huntB_wk.py`, `huntB_verdict.py`).
- **Optimizer stale `e_tp_dg`** — NEW BUG, CONFIRMED byte-exact (43.51 MB over-charge at ep2, analytic
  cap 3027 vs true 3476, `huntE_analytic.py`/`huntE_pin.py`). Also the dominant PEC offline-sweep cost
  driver (~82% of maverick multi-GPU ep>1 probe-walk cost). FIX IN PROGRESS (`BUGS.md` #20).
- **Fig-4 plain-HBF/llama3/SHORT +25.3%** — Explained: MFU=1.0 crossover mechanism (crossover-MFU
  0.23 @1-GPU vs 0.86 @multi-GPU; mfu 0.45 → 253.5/GPU vs paper 253.0). Deferred/report-only — the
  refuter pass surfaced a genuine dp1→dp2 winner flip plus an MFU/Rubin-FLOPs degeneracy
  (`huntA_mfu_probe.py`, `refA_probe.py`).
- **Fig-6 0.05s SLO ordering flip** — Explained: paper-internal. HBF SLO-bound (KV-read 71% of
  tpot), HBM4 capacity-bound with 46% SLO headroom (immune to latency levers); MFU refuted as a lever
  (widens HBF's lead, 1.0485→1.0712); sole lever = the paper's own 11.2 TB/s flash-BW spec, ~6-9%
  below which flips the ordering (`huntC_boundary.py`, `huntC_mfu.py`).
- **Fig-7 PEC errors** — Explained: chain clean, errors fully inherited. Zero-free-parameter identity
  closes to 0.0% on 11/12 cells; the paper's own PEC = K_sim × the paper's own TPS to within
  −2.3%…+1.0%. Opposite GPU trends (llama3 grows 19.5%→60.5%, llama4 shrinks 16.7%→10.5%) solved by
  the paper's own Fig-4 TPS-saturation-vs-scaling shape (`huntD2_decomp2.py`).
- **Fig-5 16-GPU comm-share drop** — Still OPEN (narrowed). The PP fix alone keeps ~14.5% share
  (TP-AllReduce sits on each stage's own critical path); tp4/pp2/dp2 has the right comm share
  (~6-7%) but is capacity-flat/never selected; MFU 0.5 is too weak (needs tpot ×1.93, delivers
  ×1.08). Candidate: microbatch comm/compute overlap (~50% AR hiding ⇒ ~7.2% ≈ paper's 7.5%) —
  untestable until the pipelined-PP fix lands (`huntB_mfumatrix.py`, `huntB_rank.py`).
- **U7 HBF+ absolute offset** — Explained: config-dependent scatter, now sign-inconsistent (llama3
  +14.7/+6.4/+6.4%, llama4 +10.1/+10.1/−4.6%; ratios 2.26/2.43/2.11 bracketing the paper's 2.35); no
  additive per-seq term closes all six cells. Variant sweep: KV double-buffer over-corrects ~30%
  below; LM full-row is physically wrong and breaks the ratio; a fit-chosen ~32k vocab-chunk lands
  both but still fails llama4-16g (`huntC_u7_task1.py`, `huntC_u7_gate.py`).
- **llama3 SHORT CONV/CONV+** — Explained: paper-side. The paper's own Fig3÷Fig4 arithmetic implies
  tpot over its own 0.1s SLO in every cell (up to 0.1278s); CONV batch matches ±2% so the TPS gap is
  exactly the paper's own overshoot; not a U7-class issue (SRAM ceilings 673/434 ≫ operating batch).
  Folded into the existing Fig-4 CONV item (`huntC_conv_task1.py`).
- **PEC offline-sweep cost** — 4 accepted measures: dedup 8 LONG-offline cells + 6 more
  `sens_baseline` duplicates; the `e_tp_dg` fix (same bug as above); on-disk memoization of
  `run_simulation`; 2 risk-flagged C++ opts (deferred, byte-identical-only). `iter=10` reduction was
  rejected (not provably result-preserving).
- **Misc small findings** — `classify_failure` "flash"-label conflation (cosmetic); CHANGES#61 doc
  gap (sum/MLA paths still effectively `fill_amortize=1`, dormant for paper1); `attention_sum_impl.cpp`
  memory_util divisor drift (print-only). Tracked as `BUGS.md` items 21-23.
- **Repo hazard** — the `ramulator2` submodule's HBF/PIM integration exists only as uncommitted local
  modifications in the main checkout (~40 modified + 4 untracked files); a fresh clone cannot build.
  Commit/vendor decision needed from the user (process note, not a paper-comparison finding).

## Refuter outcomes
- **Refuter A (MFU grid attack on Fig-4/Residual-1) — PARTIALLY SURVIVED.** The orchestrator first
  corrected an input-contamination bug in its own feed to the refuter: wrong paper TPS/batch values
  had been supplied for cells c2/c4/c5/c6 (true vector readings: llama4/HBM4/SHORT TPS=19000, sim
  error 1.4% not the originally-claimed 77%; llama3/HBF/MID batch 146.2/TPS 1494.1, sim 11.0%/9.0%
  not 58%/35%; llama3/HBF/LONG 15.9/159.1; llama4/HBF+/SHORT TPS=18592.8, sim error 11.5% not 65%) —
  the refuter's aggregate "62-77% pre-existing TPS bugs" claim was entirely an artifact of these bad
  inputs, not a real finding. On the corrected grid, global `mfu_max=0.5` improves 6/9 probe metrics
  (including landing llama4/HBM4/SHORT batch exactly on the paper's 460 anchor) but genuinely
  WORSENS c1 (llama3/HBM4/SHORT batch error 1.9%→92.1%, winner flips dp1→dp2: forced dp1's true
  ceiling TPS 2735 < dp2's 3149 at mfu 0.5) — confirmed a real, verified property of the lever (not a
  search artifact) via a full-compute-seed_tps-as-upper-bound pruning audit. Surviving refutation
  points: the c1 winner-flip (genuine global-application hazard), the mfu_max/Rubin-peak-FLOPs
  degeneracy (same free parameter, two names), and the 253/267 exact-match being one sub-problem
  dp-replicated rather than an independent confirmation. Disposition: the MFU=1.0 mechanism is
  CONFIRMED; the fix stays DEFERRED/report-only (`refA_probe.py`, `refA_mfu045.py`).
- **Refuter B (PP serialization attack) — REFUTED, and its own key measurement traced to the same
  artifact it was attacking.** Refuter B first verified Hunt B's anchors exactly, then attacked with a
  half-batch-trick emulation of faithful pp2, measuring tpot 59.05ms @B'=625.5 (vs Hunt B's algebraic
  prediction of 39.1ms) — claiming the fix changes nothing, since fixed-pp2 still loses to dp2
  everywhere and MID's 62.88-vs-74.9 gap stays unexplained. The orchestrator then showed that ALL
  serialized pp2 measurements — including the refuter's own 59.05ms — decompose exactly as
  `3×stage(b) + ~2·hop` (31.814/42.426/59.05 ≈ 3×10.55/14.06/19.55 + ε): the refuter's "18.1ms real
  hop" reading was itself the 3× propagation artifact (`BUGS.md` #19), not genuine communication cost.
  Refuter B's separate point that the paper is silent on PP scheduling (verified by PDF grep) was
  ACCEPTED and reframed the fix's justification from paper-conformance to physical realism; its point
  that an optimizer-only fix would be incoherent was also ACCEPTED (both the optimizer and the runtime
  must change together) (`refB_verify1.py`; discriminator re-run in `huntB_disc.py`).

## Convergence vs prior docs
| Item | This pass's verdict | vs. prior-pass disposition |
|---|---|---|
| Fig-4 plain-HBF/llama3/SHORT | MFU=1.0 crossover mechanism explains the whole pattern (+6% @1-GPU vs +34% @multi-GPU) | Was "Still open, None confirmed" — now EXPLAINED (deferred) |
| Fig-5 comm-share drop | PP fix alone, a tp4-class winner, and MFU all tested and refuted as sole closers; microbatch-overlap candidate documented | Was "Still open, not investigated" — now NARROWED, stays OPEN |
| Fig-6 ordering flip | Single lever = the paper's own 11.2 TB/s flash-BW spec; report-only | CONVERGES with the sibling hunt that first traced this; tightened here |
| Fig-7 PEC errors | Chain clean, 0.0% identity closure on 11/12 cells; opposite GPU trends solved by the paper's own Fig-4 TPS shape | Was "Still open, escape-dynamics guess" — now EXPLAINED, guess REFUTED |
| U7 HBF+ offset | Sign-inconsistent ±15% scatter; no additive fix exists; numbers refreshed to the current build | CONVERGES with the prior "report-only" disposition; the A/B-era table is SUPERSEDED |
| llama3 SHORT CONV/CONV+ | The paper's own SLO arithmetic explains the gap | Folded into the existing Fig-4 CONV item, no new mechanism needed |
| PP runtime + decode model | NEW bug (3× over-count) + a paired methodology deviation (serialized vs pipelined) | NEW this pass; retroactively invalidates CHANGES-26's PP measurements |
| Optimizer `e_tp_dg` | NEW bug, byte-exact, also the dominant PEC-sweep cost driver | NEW this pass |

---

# Findings register — FIFTH-pass bug hunt (2026-07-04)

Protocol: 9 blind Opus finder tracks (F1 scheduler lifecycle, F2 decode attention accounting, F3
metrics/figure writers/Fig-6 re-blind, F4 config-flag leaks, F5 module-graph/time aggregation, F6
Fig-5 bucketing pipeline, F7 U5-angle/1-GPU HBF budget, F8 U7-angle/SRAM-gate re-blind, F9
Residual-1-angle/HBM4 capacity 3-way consistency) + 6 Sonnet adversarial refuters (R1-R6) +
targeted verification runs. Finders were denied all analysis docs and git history; allowed only
the paper PDF, `PAPER_ANCHOR_SHEET.md`, `paper_figure_readings.md`, and `src/eval` harness code.
Fixes = CHANGES.md items 58-65. Gate: 13-cell regression — 6/13 (all HBM4) bit-identical, 7/13
(all HBF/HBF+/CONV+) moved <1% each, fully mechanism-explained, zero unexplained deltas.

## Finder headline summary
- **F1 (scheduler lifecycle):** sound overall (steady-state context basis, DP refill balance,
  warm-up all verified correct). One real-but-small finding: a bounded phase-lock-window bias
  from the rigid age lattice sampling a non-representative segment of the completion sawtooth at
  an arbitrary warm-up phase (+0.57% measured worst-case on a probe; bound-tabled ≤0.74% across
  all 125 Fig-3 cells, most ≤0.25%). Doc-note only, no seeding change this pass. Also
  independently re-found the CSV-drop bug (convergent with F5).
- **F2 (decode attention accounting):** sound on all six audited questions (iRoPE caps, page
  latency, roofline non-double-counting, hide budget, context basis). Found the K/V double-fill
  bug (fixed, item 61) and a small structural HBM4 zero-KV-write nit (refuted as
  negligible-by-construction — the paper's own Fig-5 HBM4 rows read 0.0 too).
- **F3 (metrics/figure writers, Fig-6 re-blind):** harness metric math sound end-to-end (tpot,
  TPS, PEC, Fig-3/4 writers). **NEW discovery:** the readings file's Fig-6 TPS column is
  uniformly ~0.843× the caption-faithful values — a readings-file extraction error, not the
  paper-side scale anomaly the fourth pass had concluded (see `PAPER_INCONSISTENCIES.md`). Fig-6
  Batch rows remain separately, unexplainedly broken (~25% overshoot after any single rescale) —
  flagged unresolved, do not naively rescale.
- **F4 (config-flag leaks):** surfaced the unstated `skewness: 0.8` Zipf-routing assumption (the
  paper is silent on routing distribution) and confirmed `reuse_kv_cache`/`chunk_size` dormancy
  already known from prior passes. All other flags classified dormant or conformant.
- **F5 (module-graph/time aggregation):** core aggregation verified correct (7-item checklist:
  intra-device serial sum, max-over-devices reporting, TP sync-then-add, comm once-per-op,
  etc.). Found the device-0 CSV-pinning nit (fixed, item 63) and independently re-confirmed the
  CSV-drop bug F1 also found (fixed, item 62).
- **F6 (Fig-5 bucketing pipeline):** sound at pp=1 (buckets sum to per-iteration latency within
  0.011%/0.55%). **Discovered LATENT pp>1 corruption:** device-0's timeboard misses later
  stages' work entirely and `pipeline_stage` matches no bucket — a 33% bucket-sum gap on a
  forced tp4/pp2 probe (fixed, item 63); unreached by any current paper-compared cell (all pp=1).
- **F7 (U5-angle, 1-GPU HBF/HBF+ non-weight budget):** clean re-derivation of the
  weight-streaming-floor mechanism (measured 72.1ms vs 72.3ms first-principles; both anchors
  reproduce). Found the LN weight-fill nit (fixed, item 60) and the dormant RoPE flash-BW nit
  (hygiene, item 65).
- **F8 (U7-angle, SRAM-gate re-blind):** byte-exact validation of the analytic ceiling (hand
  formula = binary dump exactly). Reconfirmed the paper-internal SRAM inconsistency blind (no
  single accounting fits all six HBF+ bars), and found the residual/block-input under-count
  (fixed, item 58 + V-moe-liveness adjudication).
- **F9 (Residual-1-angle, HBM4 capacity 3-way consistency):** paper anchor reproduced exactly 3
  independent ways (live pass/OOM boundary, weight derivation, KV-lifetime consistency). Found
  the shared-expert `/tp` bug (fixed, item 59) — which directly CONTRADICTS a prior-pass
  documentation attribution (see below).

## Refuter verdicts
- **R1 (skewness):** mechanism CONFIRMED, framing/reachability largely REFUTED. The effect is a
  coupon-collector hump (peaks ~13-15% at batch 100-500, saturates to ~0% by batch 1500-2500),
  not the finder's originally-claimed monotone/preferentially-flatters-flash effect
  (matched-batch probes: HBM4 14.76%/12.35% vs HBF+ 14.30%/11.52% — mildly the opposite). Real
  exposure confined to llama4 LONG at 1/2/4-GPU cells (~5-10%); SHORT/MID and 8/16-GPU cells are
  batch-saturated and unaffected. Disposition: SURFACE, no code change (paper anchors neither 0.0
  nor 0.8).
- **R2 (shared-expert `/tp`):** CONFIRMED, all six angles (code trace, bit-exact arithmetic,
  independent live-probe rebisection). No published number corrupted (`cap_batch` is a bisection
  seed only, every batch live-verified). FIX ADOPTED (item 59).
- **R3 (SRAM residual-in-FFN-phase):** CONFIRMED byte-exact, impact LARGER than the finder's own
  estimate (corrected ceiling tp1/ep1/dp8 2824→2600, not the finder's 2601; tp2/ep1/dp4 cap
  5201→4488, −13.7%). Discovered the follow-on MoE block-input liveness question, referred to the
  V-moe-liveness adjudication rather than bundled unilaterally.
- **R4 (breakdown pinning/pp>1):** device-0 pinning (Finding A) PARTIAL — premise confirmed,
  severity refuted in practice (0.0003% gap at real 16-GPU TP/EP winner-style configs; TP/EP sync
  collectives erase the drift the finder worried about). pp>1 corruption (Finding B) CONFIRMED,
  independently reproduced (66.85% gap on llama4 TP2/PP2/EP4/DP4, cross-validated against F6's
  66.73% on llama3). Both folded into item 63's fix shape.
- **R5 (small-op pricing cluster):** LN weight-fill and K/V double-fill both CONFIRMED-BUG, fixed
  now (items 60, 61). HBM4 zero-KV-write REFUTED (paper's own Fig-5 reads 0.0 too; true magnitude
  0.004-0.024%). RoPE flash-BW CONFIRMED-BUG but dormant (MLA-only path, never instantiated by
  llama3/llama4) — hygiene fix only (item 65).
- **R6 (window bias + CSV drop):** window bias (F1-C1) CONFIRMED mechanism, impact SMALL/localized
  (bound-tabled ≤0.74% worst-case over all 125 Fig-3 cells, most ≤0.25%) — doc-note only, no
  seeding change this pass (minimal-disturbance decision). CSV drop (F1-C2/F5-C5) CONFIRMED,
  zero-risk fix ADOPTED (item 62).

## V-moe-liveness adjudication — MoE shared-expert block-input liveness
Question: does the shared expert's re-read of the original block input (`expert.cpp:204`,
`post_attn_ln_out`, consumed AFTER the entire routed scatter→route→gather→all-reduce pipeline)
constitute a second phase-spanning liveness term the SRAM footprint model must count, alongside
the already-adopted residual carry (R3)? **Ruling: ADOPT variant (c) — count BOTH terms.**
Grounds are internal-convention consistency, not paper-calibration: `decoder.cpp`'s
`MoEDecoder::forward` computes the routed-expert and shared-expert outputs as two
logically-parallel branches summed at `residual_2` — the model's own existing convention for
parallel-live tensors is to count them concurrently (the same reason routed+shared weight/
activation costs are already summed rather than maxed elsewhere, and the same reason the
attention phase already counts its own persistent carry rather than treating it as freed). Under
that same convention `post_attn_ln_out` cannot be modeled as freed once the routed path starts —
the shared branch needs it concurrently until `residual_2` — so omitting it while counting the
residual carry is the asymmetry, not a justified simplification. Rejected: variant (a)
residual-only (asymmetric with the model's own summing convention) and variant (b)
block-input-only (ignores R3's independently-confirmed residual omission). Implemented as item
58; expected effect (maverick tp1/ep1/dp8): 2824 (pre-fix) → 2600 (residual only) → **2409**
(residual + block-input, this ruling) — confirmed by the post-fix regression.

## CONTRADICTS-DOC: shared-expert TP-sharding attribution
A prior pass's documentation (`CHANGES.md`'s former item 43 / third-pass `FINDINGS_REGISTER.md`
C12) attributed the runtime as TP-sharding the shared expert, citing the measured 106.020 GiB
weight at TP=2 as its verification. **That attribution was factually wrong** — 106.020 GiB
matches ONLY the full, undivided shared-expert computation (the TP=1→TP=2 weight delta decomposes
completely into attention + dense-FFN + embed/lm_head, with zero shared-expert contribution). The
erroneous `/tp` this pass removes from `parallelism_optimizer.cpp:157` (item 59) was very likely
introduced by that same prior pass under the mistaken belief its own attribution was correct.
**No published or reported number was corrupted by the error** — `cap_batch` only seeds the
live-verified bisection search, and every reported batch is confirmed by the actual simulator
run, not the analytic estimate.

## Convergence table
| Item | This pass's blind verdict | vs. prior-pass disposition |
|---|---|---|
| U5 (1-GPU HBF/HBF+ non-weight budget) | F7: weight-stream floor 72.1ms ≈ 72.3ms first-principles; sim matches the paper's own 1-GPU bars to ~1% | **CONVERGED** with fourth-pass F8a |
| U7 (SRAM-gate paper-internal inconsistency) | F8: byte-exact ceiling reproduction; no single accounting fits all six HBF+ bars, reached blind | **CONVERGED**, and **SHARPENED** by the residual/block-input fix (item 58), which narrows but does not close the gap |
| Residual-1 (HBM4 capacity 3-way consistency) | F9: DP-pure anchor reproduces the paper's 460/GPU exactly, 3 independent ways | **CONVERGED**, and found a genuinely NEW bug (the shared-expert `/tp`, item 59) that partially DIVERGES from a prior pass's documented mechanism (see CONTRADICTS-DOC above) |
| Deferred score+MFU pair | Not re-derived this pass (out of scope); item 64 fixes the unrelated MFU M-basis miskeying flagged as a prerequisite for any future MFU-sensitivity work | **UNCHANGED**, still deferred; the M-basis prerequisite fix is now done |

---

# Findings register — FOURTH-pass bug hunt (2026-07-04)

Protocol: 10 blind Opus finder tracks (F1 energy/power, F2 ramulator trace, F3 GB-vs-GiB,
F4 roofline/MFU/flops, F5 FlashAttention score-traffic, F6 harness mechanics, F7 LmHead/node/
CONV-presets, F8a/b/c re-blind U5/U7/Residual-1) + 6 Sonnet adversarial refuters (R1-R6) +
targeted verification runs. Finders were denied all analysis docs AND git history; allowed only
the paper PDF, PAPER_ANCHOR_SHEET.md, paper_figure_readings.md, src/eval/harness code. Fixes =
CHANGES.md items 51-57. Gate: 13-cell regression, all ≤8-GPU cells bit-identical post-fix.

## Convergence summary (blind verdicts vs prior docs)
- CONVERGED + SHARPENED: U5 (F8a: weight-stream floor 72.1ms measured = 72.3ms first-principles;
  paper-claim ledger 10/11 cells exceed the 8-GPU HBM4 anchor, the single miss IS the paper's
  "in most cases" hedge; sim matches the paper's OWN 1-GPU bars to ~1%). U7 (F8b: the 854.7
  flat ✕ bar and the paper-text "327" are ONE score-inclusive accounting at TWO contexts —
  SHORT→755.6, MID→294-327 — which contradicts the paper's own unmarked MID=745 bar; sim gate
  correct, no fix; NOTE: footprint.h's own comments reference the U7 diagnostic, a partial
  context leak via allowed code — the convergence rests on F8b's fresh measurements).
  Residual-1 (F8c: DP-pure ceiling = 460.00/GPU EXACT; TP=2 argmax legitimate under §III; NEW:
  460 is only reproducible with 288 GiB — decimal GB gives ~428 — so the paper's tool used
  binary GiB, settling F3's question too).
- CORRECTED vs docs: the deferred score+MFU pair's "cancellation" rationale. R3 proved no
  structural cancellation exists (score bytes ≡ 0 on HBF+/CONV+, < KV max on HBF; live only on
  the HBM4 sum path); R1 established use_flash_attention is STRUCTURALLY MLA-only (GQA classes
  lack the member) and the paper never mentions FlashAttention → keeping score bytes is
  plausibly paper-conformant, a stronger keep-deferred ground than cancellation. V-MFU measured
  mfu_max=0.5 → SHORT +33.6% / MID +8.0% tpot (M-basis = per-DP-replica tokens; R3's per-GPU
  threshold arithmetic was wrong — orchestrator caveat, run-confirmed).
- RESOLVED empirically: C5/I5 prune-risk — HBF_DISABLE_CONFIG_PRUNING=1 at the worst-case LONG
  cell (30 configs, 0 pruned) reproduces the pruned winner exactly.
- CLOSES anchor-sheet ambiguity #5: CONV/CONV+ write-BW = plane-count derivation (16/25 planes,
  16 dies × 4 KiB / 100 µs) reconstructs Table I exactly (F7 checked-correct).

## Fixed this pass (see CHANGES.md 51-57 for full records)
- P4-1 [F7-1, R4: SURVIVES all 4 angles; run-verified] MoE a2a decode branches priced intra-node
  bytes on IB, serialized (4 sites) → node-aware max composition. 16-GPU llama4 MID: comm share
  19.1%→8.7% (paper ~3.9%), tpot −11.4%. ≤8-GPU bit-identical (guarded reduction).
- P4-2 [orchestrator] Optimizer a2a mirror of P4-1 (prune-safe: new cost ≤ old).
- P4-3 [F4-3, R5: real but impact-refuted → hygiene] Optimizer MoE compute + shared-expert flops.
- P4-4 [F6-1, R6: SURVIVES, strengthened] CSV filename + _EP{e}_MEM{preset} → Fig-5 clobber gone.
- P4-5 [F1-3 ≡ F2-2, convergent] device.cpp run_ideal pCH_1 %==1→==0 (energy counters only).
- P4-6 [F3-1] OOM messages GiB labeling (display-only).
- P4-7 [F7-2] node.cpp dead node_ict_* fields warning comment.

## Doc-notes (real, deliberately not fixed)
- N1 [F1+F2, independent convergence] Entire energy/ramulator path DECORATIVE: use_ramulator off
  everywhere; timing analytic; energy CSV columns written, never read; paper reports no energy
  figure. (Corrects the pre-pass exploration claim that FC/Attn/MoE energy columns are never
  assigned — they ARE assigned, cluster.cpp:1003-1008/1263-1268, just unread.)
- N2 [F2-1] Ramulator YAML chosen by gpu_gen only — all presets map to HBM3-class configs;
  enabling use_ramulator on a flash preset would silently simulate HBM3E. Landmine.
- N3 [F1-1/F1-2] power.h ×8/×32 factors vs "X4"/"X16" comments (2× self-inconsistency); kMAC
  uncited with unexplained /2. Unread columns.
- N4 [F4-1, R2: survives, dormant] effectiveMFU M-basis miskeyed: decode-attention sites pass
  per-op m=1, contradicting hardware_config.h's own "batch*tokens" doc; dormant (knob never
  enabled). MUST be fixed before any MFU-sensitivity study.
- N5 [F4-4] Optimizer KV-write hiding budget uses QKVO-projection flops as the compute basis vs
  the sim's score+context compute — comment claims exact match, doesn't hold at LONG. Empirically
  no prune flip (C5-B). Hygiene candidate for a future optimizer-parity pass.
- N6 [F6-2, R6: survives; orchestrator adjudicated vs the actual PDF figure] Fig-6 normalization:
  sim per-SLO self-baseline ≡ caption's fixed per-workload baseline (HBM4 is SLO-invariant), but
  the paper's own 8-GPU HBM4 points plot at ~0.84 TPS / 0.64 batch — not 1.0 under any reading
  of its caption — while its text's RELATIVE claims match the readings. Paper-side scale anomaly
  (or pixel-offset artifact on near-zero bars). Structural ~16-19% component in Fig-6 TPS error
  rates; batch rows worse. No sim change.
- N7 [F6-3] classify_failure buckets "HBM capacity exceeded" under "flash" — semantically
  "capacity"; harmless to every current consumer.
- N8 [F5-3/F5-4, latent] attention_mixed_impl omits score-write AND output-write (unreconciled
  with Gen); GQA prefill sum kernel charges score round-trip like Gen (consistent, non-flash);
  both off the paper's decode-only path.
- N9 [F1-5, latent] StatusBoard energy fields lack initializers (safe today via value-init).
- N10 [F1/F2] Dead code: data_object.cpp (no callers), getDramEnergyForLoad + *_energy_load.
- N11 [F8c cosmetic] cluster.cpp HBM-gate OOM message says "weight+kv" but the gated sum
  includes the small activation term.

## Verification-run record
- V-MFU (forced llama3/HBM4/8 tp8/dp1): SHORT 54.16→72.33ms, MID 33.14→35.79ms at mfu 0.5.
- V-comm16 (llama4/HBM4/16 MID 2432, winner TP2/EP4/DP8): pre-fix comm 4.80ms/19.1% share;
  post-fix 1.95ms/8.7%; tpot 25.20→22.32ms; atten_gen unchanged.
- V-C5 (llama4/HBF+/8 LONG, pruning off): identical winner (tp2/dp4, 169/GPU, 1692.07 TPS/GPU).
- V-classify: forced 60000-batch HBF+ SHORT → "sram" ✓.
- F8a probes: batch sweep + boundary 177 pass/178 fail; component decomposition (FFN weight-read
  ~58.9ms of 72.1ms floor).
- F8b probes: analytic ceilings (tp1/ep1/dp8 cap 22,592 total = 2,824/GPU, SHORT≡MID
  context-independent; LONG flash-bound 5,415); score-inclusive diag 755.6 (SHORT) / 293.7 (MID).
- F8c probes: TP scan (460.00/488.5/502.5/509.5 per GPU at tp1/2/4/8, argmax tp2), OOM boundary
  probes, matched-batch comm deficit −0.7%.
- Post-fix 13-cell regression: all ≤8-GPU cells bit-identical; l4_HBFp_16_LONG 174→175/GPU
  (+3.6% vs paper TPS anchor); l4_HBFp_8_SHORT completed at 2374/GPU slo (the U7 cell, vs ✕854.7).

---

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
Drivers (trackb_dump.py, trackb_pick.py) were scratch, since removed 2026-07-06.

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
