#!/usr/bin/env python3
"""13-cell paper-anchor regression sweep on the fixed binary (third-pass bug hunt).

Each cell runs run_experiments.find_max_batch_size (full per-config
simulator-verified search) with a unique worker tag. Results printed as a
table with per-GPU batch, TPS/GPU, tpot, bound_reason, and the paper anchor
where the anchor sheet provides an exact red-label value.
"""
import concurrent.futures as cf
import json
import sys

import run_experiments as rx

# (label, model, mem, gpus, workload, paper_batch_per_gpu, paper_tps_per_gpu)
# Anchors: PAPER_ANCHOR_SHEET.md exact red labels; None = no exact anchor.
CELLS = [
    ("l4_HBM4_8_SHORT", "llama4_maverick", "HBM4", 8, "SHORT", 460.0, 19000.0),
    ("l4_HBM4_8_MID",   "llama4_maverick", "HBM4", 8, "MID",   151.5, 6300.0),
    ("l4_HBM4_8_LONG",  "llama4_maverick", "HBM4", 8, "LONG",  31.0,  1300.0),
    ("l4_HBFp_8_SHORT", "llama4_maverick", "HBF+", 8, "SHORT", 759.0, None),   # 1.65x460 visual, SRAM-marked
    ("l4_HBFp_8_MID",   "llama4_maverick", "HBF+", 8, "MID",   742.0, None),   # 4.9x151.5 visual
    ("l4_HBFp_8_LONG",  "llama4_maverick", "HBF+", 8, "LONG",  None,  None),
    ("l4_HBFp_4_LONG",  "llama4_maverick", "HBF+", 4, "LONG",  None,  1495.0), # 1.15x1300 visual
    ("l4_HBFp_16_LONG", "llama4_maverick", "HBF+", 16, "LONG", None,  1690.0), # 1.3x1300 visual
    ("l4_CONVp_8_SHORT","llama4_maverick", "CONV+", 8, "SHORT", 414.0, None),  # 0.9x460 visual
    ("l3_HBM4_8_SHORT", "llama3_405B",     "HBM4", 8, "SHORT", 194.0, 3300.0),
    ("l3_HBM4_8_MID",   "llama3_405B",     "HBM4", 8, "MID",   61.5,  1800.0),
    ("l3_HBM4_8_LONG",  "llama3_405B",     "HBM4", 8, "LONG",  3.75,  147.0),
    ("l3_HBF_1_SHORT",  "llama3_405B",     "HBF",  1, "SHORT", None,  None),   # vs pre-fix 185
]

WORKLOADS = {"SHORT": (1660, 373), "MID": (5900, 499), "LONG": (103500, 1100)}


def run_cell(args):
    label, model, mem, gpus, wl, pb, pt = args
    in_len, out_len = WORKLOADS[wl]
    tag = "reg_" + label
    try:
        max_b, tpot, csv, pec_kv, pec_cap, dp, bound = rx.find_max_batch_size(
            model, mem, gpus, in_len, out_len, tpot_slo=0.1,
            temp_cfg_name=f"config_{tag}.yaml", worker_tag=tag)
        per_gpu = max_b / gpus if max_b else 0.0
        tps_gpu = (max_b / (tpot * gpus)) if (max_b and tpot) else 0.0
        pec = None
        if max_b and pec_kv and pec_cap:
            row = {"workload": wl, "max_batch": max_b, "tpot": tpot,
                   "gpus": gpus, "pec_kv_bytes": pec_kv, "pec_capacity": pec_cap}
            p = rx.compute_pec(row)
            pec = p["pec"] if p else None
        return {"label": label, "batch_per_gpu": per_gpu, "tps_per_gpu": tps_gpu,
                "tpot": tpot, "bound": bound, "dp": dp, "pec": pec,
                "paper_batch": pb, "paper_tps": pt}
    except Exception as e:
        return {"label": label, "error": repr(e)}


def main():
    results = []
    with cf.ProcessPoolExecutor(max_workers=7) as pool:
        futs = {pool.submit(run_cell, c): c[0] for c in CELLS}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            print("DONE:", json.dumps(r), flush=True)

    order = {c[0]: i for i, c in enumerate(CELLS)}
    results.sort(key=lambda r: order[r["label"]])
    print("\n==== 13-CELL REGRESSION TABLE ====")
    for r in results:
        if "error" in r:
            print(f"{r['label']:<18} ERROR {r['error']}")
            continue
        pb = r["paper_batch"]; pt = r["paper_tps"]
        db = f"{(r['batch_per_gpu']/pb-1)*100:+.1f}%" if pb else "  --"
        dt = f"{(r['tps_per_gpu']/pt-1)*100:+.1f}%" if pt else "  --"
        pec_s = f" pec={r['pec']/1e3:.0f}K" if r.get("pec") else ""
        print(f"{r['label']:<18} batch/GPU={r['batch_per_gpu']:8.2f} ({db} vs paper)"
              f" TPS/GPU={r['tps_per_gpu']:8.1f} ({dt}) tpot={r['tpot']*1e3:6.2f}ms"
              f" bound={r['bound']} dp={r['dp']}{pec_s}")


if __name__ == "__main__":
    sys.exit(main())
