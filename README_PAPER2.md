# Paper2 (Kyung et al., IEEE CAL 2026) Fig4/Fig5 harness

`run_paper2.py` drives `./build/run <config>` to replicate Fig4 (HBF lifetime
heatmaps) and Fig5 (throughput vs. NVLink5.0/6.0/HBF/1-2-HBF) from
`Fig4_Fig5_extracted_numbers.md`. Run from the repo root (worktree root).

## Usage

```
python3 run_paper2.py probe                  # V_weight_node + KV bytes/token per model (cached in paper2_results/probe_cache.json)
python3 run_paper2.py smoke                  # 1 cell @ iter=100, decides FINAL_ITER (paper2_results/iter_decision.json)
python3 run_paper2.py fig4 --dry-run         # print the 60-cell batch table, no simulation
python3 run_paper2.py fig4                   # run the 60-cell sweep (checkpointed to paper2_results/fig4_results.jsonl)
python3 run_paper2.py fig4 --cells "llama4_maverick|cv0.1_k90|ctx8192|ratio1:3"
python3 run_paper2.py fig5 --dry-run         # print the 4 baselines + 32-bar batch table
python3 run_paper2.py fig5                   # run baselines + bars (checkpointed to paper2_results/fig5_results.jsonl)
python3 run_paper2.py tables                 # (re)generate fig4_table.md / fig5_table.md / summary.md from the jsonl checkpoints
python3 run_paper2.py all                    # probe + smoke only; does NOT auto-launch the full sweeps (safety gate)
```

Restarting `fig4`/`fig5` skips cells already present in the corresponding
`.jsonl` (keyed by cell id + a hash of that cell's defining config fields), so
a killed sweep can simply be re-invoked.

`PAPER2_WORKERS` env var (default 2) controls how many `./run` subprocesses
run concurrently — keep this low, host RAM is tight (~8-11GB free typical).

## Outputs (`paper2_results/`)
- `probe_cache.json`, `iter_decision.json` -- cached probe/smoke state.
- `fig4_results.jsonl`, `fig5_results.jsonl` -- one JSON record per completed cell (checkpoint log).
- `fig4_table.md`, `fig5_table.md` -- sim-vs-paper tables, laid out like `Fig4_Fig5_extracted_numbers.md`.
- `summary.md` -- headline deviations.

See the module docstring at the top of `run_paper2.py` for the batch policy,
lifetime/TPOT formulas, and SLO-search algorithm.
