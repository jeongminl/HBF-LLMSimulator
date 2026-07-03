import sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-ab-score")
import run_experiments as re_mod

# 15-cell B-side sweep: score-accounting variant.
# (model, mem) x gpus, workload SHORT (input 1660, output 373), tpot_slo=0.1.
MODELS_MEMS = [
    ("llama4_maverick", "HBF+"),
    ("llama4_maverick", "CONV+"),
    ("llama3_405B", "HBF+"),
]
GPUS = [1, 2, 4, 8, 16]

CASES = []
idx = 0
for (model, mem) in MODELS_MEMS:
    for gpus in GPUS:
        CASES.append((idx, model, mem, gpus))
        idx += 1


def cell(args):
    i, model, mem, gpus = args
    tag = f"{i:02d}"
    try:
        res = re_mod.find_max_batch_size(
            model, mem, gpus, 1660, 373, tpot_slo=0.1,
            temp_cfg_name=f"config_ab{tag}.yaml", worker_tag=f"ab{tag}")
        max_b, tpot = res[0], res[1]
        bound = res[6]
        bpg = (max_b / gpus) if gpus else 0.0
        line = (f"ABSCORE {model}/{mem}/SHORT/{gpus}gpu: "
                f"batch_per_gpu={bpg:.1f} bound={bound} "
                f"(total={max_b} tpot={tpot:.4f})")
    except Exception as e:
        line = f"ABSCORE {model}/{mem}/SHORT/{gpus}gpu: ERROR {type(e).__name__}: {e}"
    print(line, flush=True)
    return (i, line)


if __name__ == "__main__":
    print(f"Launching {len(CASES)} cells, max_workers=8", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(cell, CASES):
            results.append(r)
    print("\n===== SORTED RESULTS =====", flush=True)
    for i, line in sorted(results, key=lambda x: x[0]):
        print(line, flush=True)
    print("===== DONE =====", flush=True)
