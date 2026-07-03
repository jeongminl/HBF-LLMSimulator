import sys, os, json
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass")
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_experiments as R

MODEL = "llama4_maverick"
MEM = "HBM4"
GPU = 8
IN_LEN, OUT_LEN = 1660, 373
SLO = 0.1

# (tag, tp, total_batch)
CASES = [
    ("tp2_b3868", 2, 3868),
    ("tp1_b3680", 1, 3680),
    ("tp1_b3868", 1, 3868),
    ("tp2_b3680", 2, 3680),
]

def work(case):
    tag, tp, batch = case
    dist = {"tp": tp, "pp": 1, "ep": 1}
    res = R.run_simulation(MODEL, MEM, GPU, batch, IN_LEN, OUT_LEN,
                           optimize_parallelism=False, tpot_slo=SLO, distribution=dist,
                           temp_cfg_name=f"config_{tag}.yaml", worker_tag=f"e1_{tag}")
    out = {"tag": tag, "tp": tp, "batch": batch,
           "success": res.get("success"), "reason": res.get("reason"),
           "tpot": res.get("tpot"), "csv_file": res.get("csv_file"), "dp": res.get("dp")}
    return out

if __name__ == "__main__":
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(work, c): c for c in CASES}
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: r["tag"])
    for r in results:
        if r["success"]:
            tps_per_gpu = r["batch"] / (r["tpot"] * GPU)
            r["tps_per_gpu"] = tps_per_gpu
            print(f"{r['tag']}: OK tpot={r['tpot']*1000:.3f}ms tps/gpu={tps_per_gpu:.1f} dp={r['dp']} csv={r['csv_file']}")
        else:
            print(f"{r['tag']}: FAIL reason={r['reason']} tpot={r.get('tpot')}")
    with open("/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass/exp1_out.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")
