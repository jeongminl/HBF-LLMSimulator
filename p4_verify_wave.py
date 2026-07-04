#!/usr/bin/env python3
"""Pass-4 verification wave A (orchestrator): MFU m-basis discriminator,
16-GPU MoE comm share (F7-1), SRAM classify check. Unique p4_v_* tags per cell."""
import concurrent.futures as cf
import csv
import json
import sys

import run_experiments as rx

SHORT = (1660, 373)
MID = (5900, 499)


def comm_share_from_csv(path):
    """Average per-iteration component shares from the t2t rows of a breakdown CSV."""
    try:
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r.get("type") == "t2t"]
        if not rows:
            return None
        keys = [k for k in rows[0]
                if k not in ("type", "seq", "num_gen_seq", "seqlen", "latency",
                             "sum_attention_opb") and not k.endswith("_energy")]
        sums = {}
        for r in rows:
            for k in keys:
                try:
                    sums[k] = sums.get(k, 0.0) + float(r[k])
                except (ValueError, TypeError):
                    pass
        total = sum(sums.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in sorted(sums.items(), key=lambda kv: -kv[1]) if v > 0}
    except Exception as e:
        return {"error": repr(e)}


def cell(label, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
    except Exception as e:
        return {"label": label, "error": repr(e)}
    out = {"label": label}
    if isinstance(res, dict):
        out["success"] = res.get("success")
        out["tpot"] = res.get("tpot")
        out["dp"] = res.get("dp")
        out["reason"] = res.get("reason")
        if res.get("csv_file"):
            out["csv"] = res["csv_file"]
            out["shares"] = comm_share_from_csv(res["csv_file"])
        if not res.get("success"):
            # keep only the tail of stdout for failure diagnosis
            out["stdout_tail"] = (res.get("stdout") or "")[-600:]
            try:
                out["classified"] = rx.classify_failure(res)
            except Exception as e:
                out["classified"] = f"classify_error: {e!r}"
    else:
        out["raw"] = str(res)
    return out


JOBS = [
    ("mfu_short_base", dict(model="llama3_405B", mem="HBM4", n=8, batch=1583,
                            wl=SHORT, mfu=None)),
    ("mfu_short_half", dict(model="llama3_405B", mem="HBM4", n=8, batch=1583,
                            wl=SHORT, mfu=0.5)),
    ("mfu_mid_base", dict(model="llama3_405B", mem="HBM4", n=8, batch=503,
                          wl=MID, mfu=None)),
    ("mfu_mid_half", dict(model="llama3_405B", mem="HBM4", n=8, batch=503,
                          wl=MID, mfu=0.5)),
]


def run_mfu(label, spec):
    return cell(label, rx.run_simulation, spec["model"], spec["mem"], spec["n"],
                spec["batch"], spec["wl"][0], spec["wl"][1],
                optimize_parallelism=False, tpot_slo=86400.0,
                distribution={"tp": 8, "pp": 1, "ep": 1},
                mfu_max=spec["mfu"], mfu_m_half=0 if spec["mfu"] else None,
                temp_cfg_name=f"config_p4_v_{label}.yaml", worker_tag=f"p4_v_{label}")


def run_comm16(label):
    return cell(label, rx.run_simulation, "llama4_maverick", "HBM4", 16, 2432,
                MID[0], MID[1], optimize_parallelism=True, tpot_slo=0.1,
                temp_cfg_name=f"config_p4_v_{label}.yaml", worker_tag=f"p4_v_{label}")


def run_sramfail(label):
    return cell(label, rx.run_simulation, "llama4_maverick", "HBF+", 8, 60000,
                SHORT[0], SHORT[1], optimize_parallelism=True, tpot_slo=0.1,
                temp_cfg_name=f"config_p4_v_{label}.yaml", worker_tag=f"p4_v_{label}")


def main():
    results = []
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        futs = {}
        for label, spec in JOBS:
            futs[pool.submit(run_mfu, label, spec)] = label
        futs[pool.submit(run_comm16, "comm16_l4_mid")] = "comm16_l4_mid"
        futs[pool.submit(run_sramfail, "sram_classify")] = "sram_classify"
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            print("DONE:", json.dumps(r), flush=True)
    print("\n==== WAVE A SUMMARY ====")
    for r in sorted(results, key=lambda x: x["label"]):
        print(json.dumps(r))


if __name__ == "__main__":
    sys.exit(main())
