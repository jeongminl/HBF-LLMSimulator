#!/usr/bin/env python3
"""F8c decomposition: llama4_maverick / HBM4 / 8 GPU / SHORT (1660,373), SLO 0.1s.
Force winner (TP=2/PP=1/DP=4) vs pure-DP (TP=1/PP=1/DP=8), find each capacity
ceiling by bisection, capture ACT/Weight/Cache per-GPU and the OOM boundary msg.
"""
import re
import sys
import run_experiments as rx

MODEL = "llama4_maverick"
MEM = "HBM4"
G = 8
IN, OUT = 1660, 373
SLO = 0.1


def parse_mem(stdout):
    """Return (act_gb, weight_gb, cache_gb) from the ACT: line (device 0)."""
    m = re.search(r"ACT:\s*([\d.eE+-]+)GB,\s*Weight:\s*([\d.eE+-]+)GB,\s*Cache:\s*([\d.eE+-]+)GB", stdout or "")
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def cap_line(stdout):
    for ln in (stdout or "").split("\n"):
        if "capacity exceeded" in ln or "Out of Memory" in ln or "Available capacity" in ln:
            return ln.strip()
    return ""


def one(dist, b, tag):
    res = rx.run_simulation(MODEL, MEM, G, b, IN, OUT, optimize_parallelism=False,
                            tpot_slo=SLO, distribution=dist,
                            temp_cfg_name=f"config_p4_f8c_{tag}.yaml",
                            worker_tag=f"p4_f8c_{tag}")
    out = {"b": b, "success": res.get("success"), "tpot": res.get("tpot"),
           "reason": res.get("reason"), "mem": parse_mem(res.get("stdout")),
           "capline": cap_line(res.get("stdout")),
           "pec_cap": res.get("pec_capacity")}
    return out


def find_ceiling(dist, dp, tag, lo_k=1, hi_k=4096):
    """Bisect max feasible k (batch = k*dp). Returns (best_out, fail_out)."""
    best = None
    fail = None
    # exponential up first
    k = lo_k
    last_ok = None
    while k <= hi_k:
        o = one(dist, k * dp, f"{tag}_{k}")
        print(f"    probe {tag} k={k} b={k*dp} ok={o['success']} tpot={o['tpot']} reason={o['reason']} cap='{o['capline']}'", flush=True)
        if o["success"]:
            last_ok = o
            k *= 2
        else:
            fail = o
            break
    if last_ok is None:
        return None, fail
    # bisect between last_ok.k and failing k
    lo = last_ok["b"] // dp
    hi = (fail["b"] // dp) if fail else hi_k
    best = last_ok
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        o = one(dist, mid * dp, f"{tag}_m{mid}")
        print(f"    bisect {tag} k={mid} b={mid*dp} ok={o['success']} tpot={o['tpot']} reason={o['reason']} cap='{o['capline']}'", flush=True)
        if o["success"]:
            best = o
            lo = mid
        else:
            fail = o
            hi = mid
    return best, fail


def report(name, dist, dp, best, fail):
    print(f"\n===== {name}  dist={dist} dp={dp} =====")
    if best:
        b = best["b"]
        pg = b / G
        tps = b / (best["tpot"] * G) if best["tpot"] else 0
        act, w, kv = best["mem"] if best["mem"] else (None, None, None)
        print(f"  MAX feasible total_batch={b}  per_gpu={pg:.2f}  tpot={best['tpot']*1e3:.3f}ms  tps/gpu={tps:.1f}")
        print(f"  device0 mem: ACT={act}GB Weight={w}GB Cache(KV)={kv}GB  sum={ (act+w+kv) if act else None}GB")
    if fail:
        print(f"  boundary FAIL at total_batch={fail['b']} reason={fail['reason']}")
        print(f"    capline: {fail['capline']}")


def main():
    winner = {"tp": 2, "pp": 1, "ep": 1}
    pure = {"tp": 1, "pp": 1, "ep": 1}

    bw, fw = find_ceiling(winner, 4, "win")
    report("WINNER TP=2/PP=1/DP=4", winner, 4, bw, fw)

    bp, fp = find_ceiling(pure, 8, "pure")
    report("PURE-DP TP=1/PP=1/DP=8", pure, 8, bp, fp)

    # forced matched batch on winner: total 3680 (=460*8) per prompt
    print("\n===== MATCHED-BATCH probe on winner =====")
    o = one(winner, 3680, "match3680")
    print(f"  winner @ total=3680 (460/gpu): success={o['success']} tpot={o['tpot']} reason={o['reason']}")
    if o["mem"]:
        print(f"  device0 mem: ACT={o['mem'][0]}GB Weight={o['mem'][1]}GB Cache={o['mem'][2]}GB")
    tps = 3680 / (o["tpot"] * G) if o.get("tpot") else None
    print(f"  implied tps/gpu at 3680 = {tps}")


if __name__ == "__main__":
    sys.exit(main())
