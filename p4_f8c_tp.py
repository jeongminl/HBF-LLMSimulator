#!/usr/bin/env python3
"""F8c: scan TP in {1,2,4,8} for llama4_maverick/HBM4/8gpu/SHORT to prove why
the search picks its winner. Each config capacity-ceiling + operating TPS.
Also run the full find_max_batch_size once to capture the winner line."""
import re, sys
import run_experiments as rx

MODEL, MEM, G, IN, OUT, SLO = "llama4_maverick", "HBM4", 8, 1660, 373, 0.1


def parse_mem(s):
    m = re.search(r"ACT:\s*([\d.eE+-]+)GB,\s*Weight:\s*([\d.eE+-]+)GB,\s*Cache:\s*([\d.eE+-]+)GB", s or "")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None, None, None)


def one(dist, b, tag):
    r = rx.run_simulation(MODEL, MEM, G, b, IN, OUT, optimize_parallelism=False, tpot_slo=SLO,
                          distribution=dist, temp_cfg_name=f"config_p4_f8c_{tag}.yaml",
                          worker_tag=f"p4_f8c_{tag}")
    return {"b": b, "ok": r.get("success"), "tpot": r.get("tpot"),
            "reason": r.get("reason"), "mem": parse_mem(r.get("stdout"))}


def ceiling(tp, tag):
    dp = 8 // tp
    dist = {"tp": tp, "pp": 1, "ep": 1}
    k, last, fail = 1, None, None
    while k <= 8192:
        o = one(dist, k * dp, f"{tag}_{k}")
        if o["ok"]:
            last, k = o, k * 2
        else:
            fail = o; break
    if last is None:
        return None
    lo, hi = last["b"] // dp, (fail["b"] // dp if fail else k)
    best = last
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        o = one(dist, mid * dp, f"{tag}_m{mid}")
        if o["ok"]:
            best, lo = o, mid
        else:
            hi = mid
    return best, dp


def main():
    print("=== TP scan (capacity ceilings) ===")
    for tp in (1, 2, 4, 8):
        res = ceiling(tp, f"tp{tp}")
        if not res:
            print(f"TP={tp}: infeasible"); continue
        best, dp = res
        b = best["b"]; pg = b / G; tps = b / (best["tpot"] * G)
        act, w, kv = best["mem"]
        print(f"TP={tp} DP={dp}: max_total={b} per_gpu={pg:.2f} tpot={best['tpot']*1e3:.3f}ms "
              f"tps/gpu={tps:.1f} | W={w}GB KV={kv}GB sum={ (act+w+kv):.2f}GiB")

    print("\n=== full find_max_batch_size (winner line) ===")
    out = rx.find_max_batch_size(MODEL, MEM, G, IN, OUT, tpot_slo=SLO,
                                 temp_cfg_name="config_p4_f8c_full.yaml", worker_tag="p4_f8c_full")
    max_b, tpot, csv, kv, cap, dp, bound = out
    print(f"RESULT max_b={max_b} per_gpu={max_b/G:.2f} tpot={tpot*1e3:.3f}ms "
          f"tps/gpu={max_b/(tpot*G):.1f} dp={dp} bound={bound}")


if __name__ == "__main__":
    sys.exit(main())
