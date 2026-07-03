import sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-ab-score")
import run_experiments as re_mod

# 8-cell MID/LONG B-side sweep, HBF+ only, tpot_slo=0.1.
MODELS = ["llama4_maverick", "llama3_405B"]
WORKLOADS = [("MID", 5900, 499), ("LONG", 103500, 1100)]
GPUS = [8, 16]

CASES = []
idx = 0
for model in MODELS:
    for (wl, il, ol) in WORKLOADS:
        for gpus in GPUS:
            CASES.append((idx, model, wl, il, ol, gpus))
            idx += 1


def cell(args):
    i, model, wl, il, ol, gpus = args
    tag = f"ml{i:02d}"
    try:
        res = re_mod.find_max_batch_size(
            model, "HBF+", gpus, il, ol, tpot_slo=0.1,
            temp_cfg_name=f"config_ab{tag}.yaml", worker_tag=f"ab{tag}")
        max_b, tpot = res[0], res[1]
        bound = res[6]
        bpg = (max_b / gpus) if gpus else 0.0
        line = (f"ABSCORE {model}/HBF+/{wl}/{gpus}gpu: "
                f"batch_per_gpu={bpg:.1f} bound={bound} "
                f"(total={max_b} tpot={tpot:.4f})")
    except Exception as e:
        line = f"ABSCORE {model}/HBF+/{wl}/{gpus}gpu: ERROR {type(e).__name__}: {e}"
    print(line, flush=True)
    return (i, line)


if __name__ == "__main__":
    print(f"Launching {len(CASES)} MID/LONG cells, max_workers=8", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(cell, CASES):
            results.append(r)
    print("\n===== SORTED RESULTS =====", flush=True)
    for i, line in sorted(results, key=lambda x: x[0]):
        print(line, flush=True)
    print("===== DONE =====", flush=True)
