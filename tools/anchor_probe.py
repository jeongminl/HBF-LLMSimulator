"""
Anchor probe: measure a single (model, memory, GPU-count, workload, SLO) cell against the
CURRENT binary, reusing run_experiments.py's own primitives (find_max_batch_size,
run_simulation, parse_csv_breakdown, compute_breakdown_fractions, compute_pec) so this never
duplicates or drifts from the actual sweep's logic.

Built for PAPER_INCONSISTENCIES.md re-verification (see the approved plan) -- every
anchor cited in that doc must be re-measured against the current (post-4182c54, post
bug-fix-merge) binary before drawing any conclusion. Never edit run_experiments.py itself
to probe one cell; import it as a module instead.

Usage:
    python tools/anchor_probe.py <model> <mem_type> <gpus> <workload> [--slo SLO]

    model:     llama3_405B | llama4_maverick
    mem_type:  HBM4 | HBF | HBF+ | CONV | CONV+
    gpus:      1 | 2 | 4 | 8 | 16
    workload:  SHORT | MID | LONG
    --slo:     0.05 | 0.1 | 0.2 | 86400 (default 0.1)

Prints: chosen TP/PP/DP/EP, total & per-GPU batch, TPOT, bound_reason, and the Fig-5
component breakdown (attention/FFN/KV-write/comm/others).
"""
import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_experiments import (  # noqa: E402
    WORKLOADS,
    find_max_batch_size,
    run_simulation,
    parse_csv_breakdown,
    compute_breakdown_fractions,
    compute_pec,
)

CONFIG_RE = re.compile(
    r"\[Parallelism Optimizer\] Found optimal configuration: "
    r"TP=(\d+), PP=(\d+), DP=(\d+), EP=(\d+)"
)


def probe(model, mem_type, gpus, workload, tpot_slo=0.1):
    in_len, out_len = WORKLOADS[workload]

    max_b, tpot, csv_file, pec_kv, pec_cap, dp, bound_reason = find_max_batch_size(
        model, mem_type, gpus, in_len, out_len, tpot_slo
    )

    result = {
        "model": model,
        "memory": mem_type,
        "gpus": gpus,
        "workload": workload,
        "slo": tpot_slo,
        "max_batch_total": max_b,
        "max_batch_per_gpu": (max_b / gpus) if gpus else 0.0,
        "tpot": tpot,
        "tps_per_gpu": (max_b / (tpot * gpus)) if tpot > 0 else 0.0,
        "bound_reason": bound_reason,
        "tp": None,
        "pp": None,
        "dp": dp,
        "ep": None,
        "breakdown": None,
    }

    if max_b > 0:
        # Re-run at the discovered max batch (optimizer re-derives its own config since we
        # pass no explicit distribution) purely to capture stdout for TP/PP/EP parsing and a
        # fresh CSV breakdown -- find_max_batch_size's internal search doesn't expose these.
        res = run_simulation(
            model, mem_type, gpus, max_b, in_len, out_len,
            optimize_parallelism=True, tpot_slo=tpot_slo,
        )
        if res["success"]:
            m = CONFIG_RE.search(res.get("stdout", ""))
            if m:
                result["tp"], result["pp"], result["dp"], result["ep"] = (
                    int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                )
            bd = parse_csv_breakdown(res.get("csv_file"))
            fracs = compute_breakdown_fractions(model, bd) if bd else None
            result["breakdown"] = fracs

            pec_row = {
                "workload": workload, "max_batch": max_b, "tpot": tpot,
                "gpus": gpus, "pec_kv_bytes": pec_kv, "pec_capacity": pec_cap,
            }
            result["pec"] = compute_pec(pec_row)

    return result


def format_result(r):
    lines = [
        f"{r['model']} | {r['workload']} | {r['memory']} | {r['gpus']} GPU | SLO={r['slo']}s",
        f"  config: TP={r['tp']} PP={r['pp']} DP={r['dp']} EP={r['ep']}",
        f"  batch: total={r['max_batch_total']} per_gpu={r['max_batch_per_gpu']:.2f} "
        f"bound_reason={r['bound_reason']}",
        f"  tpot={r['tpot']:.6f}s  tps_per_gpu={r['tps_per_gpu']:.2f}",
    ]
    if r["breakdown"]:
        b = r["breakdown"]
        lines.append(
            "  breakdown: attn={:.1%} ffn={:.1%} kv_write={:.1%} comm={:.1%} others={:.1%}".format(
                b["attention"], b["ffn"], b["kv_write"], b["communication"], b["others"]
            )
        )
    if r.get("pec"):
        lines.append(
            f"  pec: write_rate={r['pec']['write_rate_mbps']:.1f} MB/s  3yr_pec={r['pec']['pec']:.1f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["llama3_405B", "llama4_maverick"])
    ap.add_argument("mem_type", choices=["HBM4", "HBF", "HBF+", "CONV", "CONV+"])
    ap.add_argument("gpus", type=int, choices=[1, 2, 4, 8, 16])
    ap.add_argument("workload", choices=list(WORKLOADS.keys()))
    ap.add_argument("--slo", type=float, default=0.1)
    args = ap.parse_args()

    r = probe(args.model, args.mem_type, args.gpus, args.workload, args.slo)
    print(format_result(r))


if __name__ == "__main__":
    main()
