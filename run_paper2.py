#!/usr/bin/env python3
"""run_paper2.py -- Fig4/Fig5 replication harness for paper2:
Kyung, Moon, Cho, Ahn, "High-Bandwidth Flash for KV Caches: Endurance and
Performance Implications," IEEE Computer Architecture Letters, 2026.

Ground truth: ./Fig4_Fig5_extracted_numbers.md (read that file for the full
paper table contents -- PAPER_FIG4/PAPER_FIG5 below are transcribed from it).

This is a NEW driver, deliberately NOT built on top of run_experiments.py's
run_simulation()/find_max_batch_size(): paper2 cells use a completely
different config shape (workload_mode: paper2's stochastic context/ratio
sampler instead of fixed input_len/output_len, optimize_parallelism always
off with a FIXED distribution, decode_mode-only, no PEC/SRAM-diag markers to
parse, no per-config analytic-listing search -- see run_experiments.py's
run_simulation()/find_max_batch_size() docstrings). Reused BY IMPORT: none of
run_experiments.py's functions match this shape closely enough to import
directly. COPY-ADAPTED patterns (see comments at each site below):
  - temp-config-under-BUILD_DIR + `["./run", temp_cfg_name]` subprocess
    invocation with cwd=BUILD_DIR (run_simulation).
  - per-worker output_directory isolation via a worker tag, to avoid
    concurrent callers clobbering each other's CSV/output directory
    (run_simulation's worker_tag comment).
  - the "Total: " stdout parsing rule -- skip any line whose value contains
    an alpha character (cluster.cpp also prints "Total: X GB" memory-size
    lines; only the last, alpha-free "Total: <ns>" line is the wall-clock
    total) -- identical rule to run_simulation's own parsing loop.
  - ProcessPoolExecutor + picklable top-level worker functions + a
    "_crashed_cell"-style fallback so one dead worker doesn't lose the whole
    sweep (run_experiments.py's _run_cell/_crashed_cell/_checkpoint).

BATCH POLICY (decimal GB, paper convention -- see the codebase's own GB-vs-
GiB note: presets are sized in GiB (e.g. paper2_hbm_preset: 256*2^30 B/device
exactly), but this script's budget/TBW arithmetic uses decimal GB per the
paper's own Eq. 2 convention. This creates a documented ~7% sensitivity
between "what capacity we planned the batch against" and "what the simulator
actually enforces" -- never resolved because the paper's own units are
decimal and the simulator's are binary; see PAPER2_NOTES.md item 1):
  budget_node(mult) = mult * 8 * 256e9                  # "512GB"/device @ mult=2, "768GB"/device @ mult=3
  reserved_kv        = budget_node - V_weight_node       # V_weight_node = P2_WEIGHT_BYTES_NODE (probed)
  batch              = floor(reserved_kv / (context_mean * KV_bytes_per_token_full))
  batch              = (batch // 8) * 8                  # node batch splits across dp=8 shards;
                                                          # Scheduler::batch_size_per_dp = total/dp
                                                          # (integer division, scheduler.cpp:20) --
                                                          # a non-multiple-of-8 batch silently drops
                                                          # up to 7 slots' worth of the intended batch.

FIG4 LIFETIME/TPOT (per cell):
  write_rate_bps = (P2_KV_BYTES_WRITTEN_TOTAL / (Total_ns/1e9)) * 1.02        # WAF
  TBW_bytes      = (5*512e9*8 - V_weight_node) * 100000                       # SLC P/E, PHYSICAL flash, decimal
  lifetime_years = TBW_bytes / write_rate_bps / 31_557_600
  tpot_ms        = Total_ns / P2_TIMED_ITERS / 1e6

FIG5 (per model/ctx/budget, 4 bars: NVLink5.0, NVLink6.0, HBF, 1/2-HBF):
  All bars run at the SAME budget-derived batch first; a bar whose measured
  TPOT > 200ms SLO is binary-searched downward (multiples of 8) for the
  largest batch clearing the SLO, then re-measured at FINAL_ITER.
  throughput_tok_s = served_batch / (Total_ns/P2_TIMED_ITERS/1e9)
  CPU/HBM ratio (NVLink bars only) = max(0, reserved_kv_served - hbm_resident) / hbm_resident
    where reserved_kv_served = served_batch * ctx * KV_tok, hbm_resident = 8*256e9 - V_weight_node.

CLI: python3 run_paper2.py probe|smoke|fig4|fig5|tables|all [--cells k1,k2] [--dry-run]
"""

import os
import sys
import json
import copy
import time
import shutil
import hashlib
import argparse
import subprocess
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "paper2_results")
TMP_OUT_DIR = os.path.join(RESULTS_DIR, "tmp_out")
TMP_CFG_DIR = os.path.join(BUILD_DIR, "paper2_tmp")
PROBE_CACHE_PATH = os.path.join(RESULTS_DIR, "probe_cache.json")
ITER_DECISION_PATH = os.path.join(RESULTS_DIR, "iter_decision.json")
FIG4_JSONL = os.path.join(RESULTS_DIR, "fig4_results.jsonl")
FIG5_JSONL = os.path.join(RESULTS_DIR, "fig5_results.jsonl")

for _d in (RESULTS_DIR, TMP_OUT_DIR, TMP_CFG_DIR):
    os.makedirs(_d, exist_ok=True)

# PAPER2_WORKERS: host RAM is tight (~8-11GB free observed) -- default to 2
# concurrent ./run subprocesses. Each cell's own SLO binary search (Fig5) is
# strictly sequential internally, same design as run_experiments.py's
# SWEEP_WORKERS (only the outer loop over independent cells is parallelized).
MAX_WORKERS = int(os.environ.get("PAPER2_WORKERS", "2"))

MODELS = ["llama4_maverick", "deepseekR1"]
# Pure function of model geometry (heads/head_dim/precision/num_layers, or
# MLA lora ranks), independent of batch/context/ratio -- see eval/test.cpp's
# P2_KV_BYTES_PER_TOKEN_FULL block and this repo's model_config.h presets.
# Cross-checked by hand: Maverick 2*8*128*2*48=196608; R1 (512+64)*2*61=70272.
EXPECTED_KV_BYTES_PER_TOKEN = {"llama4_maverick": 196608.0, "deepseekR1": 70272.0}
EXPECTED_V_WEIGHT_NODE = {"llama4_maverick": 999.7e9, "deepseekR1": 1581.7e9}  # approx, PAPER2_NOTES.md

DISPERSIONS = [("cv0.1_k90", 0.1, 90.0), ("cv0.3_k30", 0.3, 30.0)]
CONTEXTS = [8192, 16384, 32768]
RATIOS = [("1:3", 0.75), ("1:1", 0.5), ("3:1", 0.25), ("7:1", 0.125), ("15:1", 0.0625)]

FIG5_CONTEXTS = [8192, 32768]
FIG5_BUDGETS = [2, 3]  # budget_node multiplier x -> "512GB"/device (x=2), "768GB"/device (x=3)
FIG5_DEVICES = ["NVLink5.0", "NVLink6.0", "HBF", "1/2-HBF"]
FIG5_RATIO_LABEL = "1:3"
FIG5_LOUT_RATIO = 0.75
FIG5_CV, FIG5_KAPPA = 0.1, 90.0  # Fig5 doesn't state dispersion; low-dispersion assumed (documented)

SEARCH_ITER = 150
DEFAULT_ITER = 1000  # fallback FINAL_ITER before smoke's decision is persisted
SLO_MS = 200.0
WAF = 1.02
PE_CYCLES = 100000
SECONDS_PER_YEAR = 31_557_600
SMOKE_ITER = 100

MARKER_KEYS = [
    "P2_KV_BYTES_PER_TOKEN_FULL", "P2_WEIGHT_BYTES_NODE", "P2_PHYSICAL_FLASH_BYTES_PER_DEV",
    "P2_KV_BYTES_WRITTEN_TOTAL", "P2_KV_ADMISSION_BYTES", "P2_KV_DECODE_BYTES", "P2_TIMED_ITERS",
    "P2_KV_OFFLOAD_FRACTION",
    # Not yet emitted by the binary as of this writing (a concurrent C++ change
    # adds them) -- parsed as None until present; forward-compatible.
    "P2_SRAM_ACT_BYTES", "P2_SRAM_DBUF_BYTES", "P2_SRAM_KVWRITE_BYTES",
    "P2_REQUIRED_SRAM_BYTES_PER_DEVICE",
]

# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
_BASE_YAML_CACHE = None


def load_base_yaml():
    global _BASE_YAML_CACHE
    if _BASE_YAML_CACHE is None:
        with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
            _BASE_YAML_CACHE = yaml.safe_load(f)
    return copy.deepcopy(_BASE_YAML_CACHE)


def build_config(model_name, memory_type, context_mean, cv, kappa, lout_ratio, batch_size, iters,
                  expose_first_expert_latency=False, cpu_kv_offload=False, c2c_nvlink_gen=None):
    """Build one paper2 run's full config dict, per the CONFIG TEMPLATE (see
    module docstring / the orchestrator's spec). All deltas are per-cell
    parameters; everything else is the fixed paper2 template."""
    cfg = load_base_yaml()
    cfg["model"]["model_name"] = model_name

    s = cfg["system"]
    s["gpu_gen"] = "B200"
    s["memory_type"] = memory_type
    s["nvlink_gen"] = 5  # ALWAYS 5 -- inter-device NVL72; only c2c_nvlink_gen varies for Fig5
    s["infiniband_gen"] = 800
    s["num_node"] = 1
    s["num_device"] = 8
    s["processor_type"] = "GPU"
    s["optimize_parallelism"] = False
    s["tpot_slo"] = 0.2
    s["chunk_size"] = 0
    s["distribution"] = {"expert_tensor_degree": 1, "none_expert_tensor_degree": 1, "pipeline_degree": 1}
    s["expose_first_expert_latency"] = bool(expose_first_expert_latency)
    s["cpu_kv_offload"] = bool(cpu_kv_offload)
    if c2c_nvlink_gen is not None:
        s["c2c_nvlink_gen"] = int(c2c_nvlink_gen)

    opt = s["optimization"]
    opt["parallel_execution"] = False
    opt["hetero_subbatch"] = False
    opt["disagg_system"] = False
    opt["use_low_unit_moe_only"] = False
    opt["use_ramulator"] = False
    opt["compressed_kv"] = True
    opt["use_absorb"] = True
    opt["use_flash_mla"] = True
    opt["use_flash_attention"] = True
    opt["reuse_kv_cache"] = False
    opt["kv_cache_reuse_rate"] = 0.0
    opt["prefill_mode"] = False
    opt["decode_mode"] = True

    cfg["serving"] = {"max_batch_size": int(batch_size), "max_process_token": 0}

    sim = cfg["simulation"]
    sim["data"] = "synthesis"
    sim["precision_byte"] = 0
    sim["skewness"] = 0.8
    sim["iter"] = int(iters)
    sim["injection_rate"] = 0  # MANDATORY: nonzero starves the batch below max_batch_size
    sim["exit_out_of_memory"] = False
    sim["mem_cap_limit"] = False
    sim["validate_optimizer"] = "off"
    sim["workload_mode"] = "paper2"
    sim["context_mean"] = float(context_mean)
    sim["context_cv"] = float(cv)
    sim["context_trunc_sigmas"] = 2.0
    sim["lout_mean_ratio"] = float(lout_ratio)
    sim["lout_beta_kappa"] = float(kappa)
    sim["workload_seed"] = 777
    sim["disable_irope"] = (model_name == "llama4_maverick")
    # Placeholders only: eval/test.cpp reads simulation.input_len/output_len
    # unconditionally as plain ints before workload_mode is consulted, but
    # paper2_workload's pushSeq branch (scheduler.cpp) samples its own
    # (input_len, output_len) per sequence and never reads these; the one
    # place they DO still flow through (the pre-Scheduler::Create weight
    # probe, test.cpp ~1002-1019) is documented not to depend on batch/seqlen.
    sim["input_len"] = max(1, int(context_mean * (1.0 - lout_ratio)))
    sim["output_len"] = max(2, int(context_mean * lout_ratio))

    log = cfg["log"]
    log["print_log"] = False
    log["export_gantt"] = False

    return cfg


def parse_stdout(stdout):
    """Parse `Total: <ns>` (skip the two 'Total: X GB' memory-size lines --
    copy-adapted from run_simulation's identical rule) plus every P2_* marker
    line, tolerating markers that are absent (None)."""
    out = {"total_ns": None}
    for k in MARKER_KEYS:
        out[k] = None
    marker_prefixes = {k + ": ": k for k in MARKER_KEYS}
    for line in stdout.split("\n"):
        if line.startswith("Total: "):
            raw = line[len("Total: "):].strip()
            if raw and not any(c.isalpha() for c in raw):
                try:
                    out["total_ns"] = float(raw)
                except ValueError:
                    pass
            continue
        for prefix, key in marker_prefixes.items():
            if line.startswith(prefix):
                try:
                    out[key] = float(line[len(prefix):].strip())
                except ValueError:
                    pass
                break
    return out


FAIL_MARKERS = ["Out of Memory", "Activations exceed", "Flash capacity exceeded",
                "capacity exceeded", "Assertion", "terminate called", "fail(", "[FAIL]"]


def run_binary(cfg, tag, timeout=5400, keep_on_failure=True):
    """Write a per-tag temp config + isolated output dir under BUILD_DIR,
    invoke ./run (cwd=BUILD_DIR), parse markers. Mirrors run_simulation's
    temp-config + worker-tag isolation pattern (see module docstring)."""
    cfg = copy.deepcopy(cfg)
    safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag)
    out_rel = f"../paper2_results/tmp_out/{safe_tag}/"
    cfg["log"]["output_directory"] = out_rel
    out_abs = os.path.join(TMP_OUT_DIR, safe_tag)
    os.makedirs(out_abs, exist_ok=True)

    temp_cfg_name = f"paper2_tmp/cfg_{safe_tag}.yaml"
    temp_cfg_path = os.path.join(BUILD_DIR, temp_cfg_name)
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    result = {"success": False, "tag": tag}
    try:
        res = subprocess.run(["./run", temp_cfg_name], cwd=BUILD_DIR,
                              capture_output=True, text=True, timeout=timeout)
        stdout = (res.stdout or "") + "\n" + (res.stderr or "")
        rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if res.returncode != 0 or any(m in stdout for m in FAIL_MARKERS):
            result["reason"] = f"returncode={res.returncode}"
            result["stdout_tail"] = stdout[-4000:]
        else:
            parsed = parse_stdout(stdout)
            if parsed["total_ns"] is None:
                result["reason"] = "no Total: line in stdout"
                result["stdout_tail"] = stdout[-4000:]
            else:
                result.update(parsed)
                result["success"] = True
        result["child_maxrss_kb"] = rss_kb
    except subprocess.TimeoutExpired:
        result["reason"] = "timeout"
    except Exception as e:
        result["reason"] = repr(e)
    finally:
        try:
            os.remove(temp_cfg_path)
        except OSError:
            pass
        if result["success"] or not keep_on_failure:
            shutil.rmtree(out_abs, ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# Probe: static per-model markers (V_weight_node, KV bytes/token)
# ---------------------------------------------------------------------------
def load_probe_cache():
    if os.path.exists(PROBE_CACHE_PATH):
        with open(PROBE_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_probe_cache(cache):
    with open(PROBE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def run_probe(model_name, force=False):
    cache = load_probe_cache()
    if not force and model_name in cache:
        return cache[model_name]
    # iter=2, max_batch=8, P2_HBF -- cheap, and P2_WEIGHT_BYTES_NODE/
    # P2_KV_BYTES_PER_TOKEN_FULL don't depend on batch/context/ratio.
    cfg = build_config(model_name, "P2_HBF", context_mean=8192, cv=0.1, kappa=90.0,
                        lout_ratio=0.75, batch_size=8, iters=2, expose_first_expert_latency=True)
    res = run_binary(cfg, f"probe_{model_name}")
    if not res["success"]:
        raise RuntimeError(f"probe failed for {model_name}: {res.get('reason')}\n"
                            f"{res.get('stdout_tail', '')}")
    entry = {
        "kv_bytes_per_token_full": res["P2_KV_BYTES_PER_TOKEN_FULL"],
        "weight_bytes_node": res["P2_WEIGHT_BYTES_NODE"],
        "physical_flash_bytes_per_dev": res["P2_PHYSICAL_FLASH_BYTES_PER_DEV"],
    }
    cache[model_name] = entry
    save_probe_cache(cache)
    return entry


def probe_all(force=False):
    out = {}
    for m in MODELS:
        out[m] = run_probe(m, force=force)
    return out


# ---------------------------------------------------------------------------
# Batch policy
# ---------------------------------------------------------------------------
def compute_batch(v_weight_node, budget_mult, context_mean, kv_bytes_per_token):
    """floor(reserved_kv / (ctx*KV_tok)) rounded DOWN to a multiple of 8 (dp
    shards). Guards against a computed batch < 8 (would make >=1 dp shard get
    zero sequences) by clamping up to 8 and flagging it -- this slightly
    exceeds the nominal budget, a documented corner case, not expected to
    trigger for any Fig4/Fig5 cell in-range."""
    budget_node = budget_mult * 8 * 256e9
    reserved_kv = budget_node - v_weight_node
    raw = reserved_kv / (context_mean * kv_bytes_per_token)
    batch = (int(raw) // 8) * 8
    clamped = False
    if batch < 8:
        batch = 8
        clamped = True
    return batch, reserved_kv, raw, clamped


# ---------------------------------------------------------------------------
# Fig4 metrics
# ---------------------------------------------------------------------------
def fig4_metrics(result, v_weight_node):
    total_ns = result["total_ns"]
    timed_iters = result["P2_TIMED_ITERS"]
    kv_written_total = result["P2_KV_BYTES_WRITTEN_TOTAL"]
    total_s = total_ns / 1e9
    write_rate_bps = (kv_written_total / total_s) * WAF
    tbw_bytes = (5 * 512e9 * 8 - v_weight_node) * PE_CYCLES
    lifetime_years = tbw_bytes / write_rate_bps / SECONDS_PER_YEAR
    tpot_ms = total_ns / timed_iters / 1e6
    return {"lifetime_years": lifetime_years, "tpot_ms": tpot_ms,
            "write_rate_bps": write_rate_bps, "tbw_bytes": tbw_bytes}


def fig5_throughput(result, served_batch):
    total_ns = result["total_ns"]
    timed_iters = result["P2_TIMED_ITERS"]
    tpot_s = total_ns / timed_iters / 1e9
    return served_batch / tpot_s, tpot_s * 1000.0


# ---------------------------------------------------------------------------
# Iteration-count decision (persisted after `smoke`)
# ---------------------------------------------------------------------------
def load_iter_decision():
    if os.path.exists(ITER_DECISION_PATH):
        with open(ITER_DECISION_PATH, "r") as f:
            return json.load(f)
    return {}


def save_iter_decision(d):
    with open(ITER_DECISION_PATH, "w") as f:
        json.dump(d, f, indent=2)


def final_iter():
    d = load_iter_decision()
    return int(d.get("final_iter", DEFAULT_ITER))


# ---------------------------------------------------------------------------
# Cell id / checkpoint helpers
# ---------------------------------------------------------------------------
def config_hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def load_done_ids(jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec.get("cell_id"), rec.get("config_hash")))
                except json.JSONDecodeError:
                    continue
    return done


def append_jsonl(jsonl_path, rec):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Fig4 cell generation + worker
# ---------------------------------------------------------------------------
def fig4_cell_id(model, disp_key, ctx, ratio_label):
    return f"{model}|{disp_key}|ctx{ctx}|ratio{ratio_label}"


def gen_fig4_specs(probes, iters):
    specs = []
    for model in MODELS:
        v_weight = probes[model]["weight_bytes_node"]
        kv_tok = probes[model]["kv_bytes_per_token_full"]
        for disp_key, cv, kappa in DISPERSIONS:
            for ctx in CONTEXTS:
                for ratio_label, ratio in RATIOS:
                    cid = fig4_cell_id(model, disp_key, ctx, ratio_label)
                    batch, reserved_kv, raw, clamped = compute_batch(v_weight, 2, ctx, kv_tok)
                    spec = {
                        "cell_id": cid, "model": model, "disp_key": disp_key, "cv": cv, "kappa": kappa,
                        "ctx": ctx, "ratio_label": ratio_label, "ratio": ratio, "iters": iters,
                        "batch": batch, "reserved_kv": reserved_kv, "raw_batch": raw, "clamped": clamped,
                        "v_weight_node": v_weight, "kv_bytes_per_token": kv_tok,
                    }
                    spec["config_hash"] = config_hash({k: spec[k] for k in
                                                        ("model", "disp_key", "ctx", "ratio_label", "batch", "iters")})
                    specs.append(spec)
    return specs


def _fig4_worker(spec):
    cfg = build_config(spec["model"], "P2_HBF", spec["ctx"], spec["cv"], spec["kappa"], spec["ratio"],
                        spec["batch"], spec["iters"], expose_first_expert_latency=True)
    tag = f"fig4_{spec['config_hash']}_w{os.getpid()}"
    res = run_binary(cfg, tag)
    out = {"cell_id": spec["cell_id"], "config_hash": spec["config_hash"], "model": spec["model"],
           "disp_key": spec["disp_key"], "ctx": spec["ctx"], "ratio_label": spec["ratio_label"],
           "batch": spec["batch"], "reserved_kv": spec["reserved_kv"], "clamped_batch": spec["clamped"],
           "iters": spec["iters"], "success": res["success"]}
    if not res["success"]:
        out["reason"] = res.get("reason")
        return out
    out.update(fig4_metrics(res, spec["v_weight_node"]))
    out["markers"] = {k: res.get(k) for k in MARKER_KEYS}
    return out


def _crashed(spec):
    return {"cell_id": spec["cell_id"], "config_hash": spec["config_hash"], "success": False,
            "reason": "worker_crashed"}


def run_fig4(iters, cells_filter=None, dry_run=False):
    probes = probe_all()
    specs = gen_fig4_specs(probes, iters)
    if cells_filter:
        wanted = set(cells_filter)
        specs = [s for s in specs if s["cell_id"] in wanted]
        missing = wanted - {s["cell_id"] for s in specs}
        if missing:
            print(f"[fig4] WARNING: requested cells not found: {sorted(missing)}")

    if dry_run:
        print(f"{'cell_id':45s} {'batch':>7s} {'reserved_kv(GB)':>16s} {'clamped':>8s}")
        for s in specs:
            print(f"{s['cell_id']:45s} {s['batch']:7d} {s['reserved_kv']/1e9:16.1f} {str(s['clamped']):>8s}")
        return []

    done = load_done_ids(FIG4_JSONL)
    todo = [s for s in specs if (s["cell_id"], s["config_hash"]) not in done]
    print(f"[fig4] {len(specs)} cells total, {len(specs) - len(todo)} already checkpointed, "
          f"{len(todo)} to run (iters={iters}, workers={MAX_WORKERS})")

    results = []
    if not todo:
        return results
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_fig4_worker, s): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[fig4] worker crashed for {s['cell_id']}: {e!r}")
                r = _crashed(s)
            append_jsonl(FIG4_JSONL, r)
            results.append(r)
            status = "OK" if r["success"] else f"FAIL({r.get('reason')})"
            extra = f" lifetime={r.get('lifetime_years'):.2f}yr tpot={r.get('tpot_ms'):.1f}ms" if r["success"] else ""
            print(f"[fig4] {r['cell_id']:45s} {status}{extra}")
    return results


# ---------------------------------------------------------------------------
# Fig5 cell generation + workers
# ---------------------------------------------------------------------------
def fig5_baseline_id(model, ctx):
    return f"baseline|{model}|ctx{ctx}"


def fig5_bar_id(model, ctx, budget, device):
    return f"{model}|ctx{ctx}|budget{budget}x|{device}"


def _fig5_baseline_worker(spec):
    cfg = build_config(spec["model"], "P2_HBM", spec["ctx"], FIG5_CV, FIG5_KAPPA, FIG5_LOUT_RATIO,
                        spec["batch"], spec["iters"], expose_first_expert_latency=False,
                        cpu_kv_offload=False)
    tag = f"fig5base_{spec['config_hash']}_w{os.getpid()}"
    res = run_binary(cfg, tag)
    out = {"cell_id": spec["cell_id"], "config_hash": spec["config_hash"], "model": spec["model"],
           "ctx": spec["ctx"], "batch": spec["batch"], "success": res["success"]}
    if not res["success"]:
        out["reason"] = res.get("reason")
        return out
    throughput, tpot_ms = fig5_throughput(res, spec["batch"])
    out["throughput_tok_s"] = throughput
    out["tpot_ms"] = tpot_ms
    return out


def _device_cfg_kwargs(device):
    if device == "NVLink5.0":
        return dict(memory_type="P2_HBM", cpu_kv_offload=True, c2c_nvlink_gen=5, expose_first_expert_latency=False)
    if device == "NVLink6.0":
        return dict(memory_type="P2_HBM", cpu_kv_offload=True, c2c_nvlink_gen=6, expose_first_expert_latency=False)
    if device == "HBF":
        return dict(memory_type="P2_HBF", cpu_kv_offload=False, expose_first_expert_latency=True)
    if device == "1/2-HBF":
        return dict(memory_type="P2_HBF_HALF", cpu_kv_offload=False, expose_first_expert_latency=True)
    raise ValueError(device)


def _fig5_bar_worker(spec):
    kwargs = _device_cfg_kwargs(spec["device"])
    model, ctx = spec["model"], spec["ctx"]

    def build(batch, iters):
        return build_config(model, kwargs["memory_type"], ctx, FIG5_CV, FIG5_KAPPA, FIG5_LOUT_RATIO,
                             batch, iters, expose_first_expert_latency=kwargs["expose_first_expert_latency"],
                             cpu_kv_offload=kwargs["cpu_kv_offload"], c2c_nvlink_gen=kwargs.get("c2c_nvlink_gen"))

    budget_batch = spec["batch"]
    tag_base = f"fig5_{spec['config_hash']}_w{os.getpid()}"

    probe_res = run_binary(build(budget_batch, SEARCH_ITER), f"{tag_base}_probe_b{budget_batch}")
    out = {"cell_id": spec["cell_id"], "config_hash": spec["config_hash"], "model": model, "ctx": ctx,
           "budget": spec["budget"], "device": spec["device"], "budget_batch": budget_batch}
    if not probe_res["success"]:
        out["success"] = False
        out["reason"] = probe_res.get("reason")
        return out

    probe_tpot_ms = probe_res["total_ns"] / probe_res["P2_TIMED_ITERS"] / 1e6
    if probe_tpot_ms <= SLO_MS:
        served_batch, slo_bound = budget_batch, False
    else:
        slo_bound = True
        lo, hi = 1, max(1, (budget_batch // 8) - 1)
        served_batch = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            b = mid * 8
            r = run_binary(build(b, SEARCH_ITER), f"{tag_base}_probe_b{b}")
            if r["success"] and (r["total_ns"] / r["P2_TIMED_ITERS"] / 1e6) <= SLO_MS:
                served_batch = b
                lo = mid + 1
            else:
                hi = mid - 1
        if served_batch == 0:
            served_batch = 8  # floor guard; document as a corner case

    final_res = run_binary(build(served_batch, spec["iters"]), f"{tag_base}_final_b{served_batch}")
    if not final_res["success"]:
        out["success"] = False
        out["reason"] = final_res.get("reason")
        out["served_batch"] = served_batch
        out["slo_bound"] = slo_bound
        return out

    throughput, tpot_ms = fig5_throughput(final_res, served_batch)
    kv_tok = spec["kv_bytes_per_token"]
    reserved_kv_served = served_batch * ctx * kv_tok
    hbm_resident = 8 * 256e9 - spec["v_weight_node"]
    cpu_hbm_ratio = max(0.0, reserved_kv_served - hbm_resident) / hbm_resident if hbm_resident > 0 else None
    out.update({
        "success": True, "served_batch": served_batch, "slo_bound": slo_bound,
        "tpot_ms": tpot_ms, "throughput_tok_s": throughput,
        "cpu_hbm_ratio": cpu_hbm_ratio if spec["device"].startswith("NVLink") else None,
        "sim_avg_offload_fraction": final_res.get("P2_KV_OFFLOAD_FRACTION"),
        "sram_required_bytes_per_device": final_res.get("P2_REQUIRED_SRAM_BYTES_PER_DEVICE"),
    })
    return out


def gen_fig5_specs(probes, iters):
    # `iters` (the served/final measurement iter count -- SEARCH_ITER is used
    # internally for the SLO probes regardless) is folded into config_hash
    # BEFORE hashing, same as gen_fig4_specs, so a later FINAL_ITER change
    # correctly invalidates old checkpointed cells on resume rather than
    # silently reusing stale-iteration results.
    baseline_specs, bar_specs = [], []
    for model in MODELS:
        v_weight = probes[model]["weight_bytes_node"]
        kv_tok = probes[model]["kv_bytes_per_token_full"]
        for ctx in FIG5_CONTEXTS:
            base_batch, _, _, _ = compute_batch(v_weight, 1, ctx, kv_tok)
            bcid = fig5_baseline_id(model, ctx)
            bspec = {"cell_id": bcid, "model": model, "ctx": ctx, "batch": base_batch, "iters": iters}
            bspec["config_hash"] = config_hash({k: bspec[k] for k in ("model", "ctx", "batch", "iters")})
            baseline_specs.append(bspec)
            for budget in FIG5_BUDGETS:
                batch, reserved_kv, raw, clamped = compute_batch(v_weight, budget, ctx, kv_tok)
                for device in FIG5_DEVICES:
                    cid = fig5_bar_id(model, ctx, budget, device)
                    spec = {"cell_id": cid, "model": model, "ctx": ctx, "budget": budget, "device": device,
                            "batch": batch, "reserved_kv": reserved_kv, "clamped": clamped,
                            "v_weight_node": v_weight, "kv_bytes_per_token": kv_tok, "iters": iters}
                    spec["config_hash"] = config_hash({k: spec[k] for k in
                                                        ("model", "ctx", "budget", "device", "batch", "iters")})
                    bar_specs.append(spec)
    return baseline_specs, bar_specs


def run_fig5(iters, cells_filter=None, dry_run=False):
    probes = probe_all()
    baseline_specs, bar_specs = gen_fig5_specs(probes, iters)

    if cells_filter:
        wanted = set(cells_filter)
        baseline_specs = [s for s in baseline_specs if s["cell_id"] in wanted]
        bar_specs = [s for s in bar_specs if s["cell_id"] in wanted]

    if dry_run:
        print("-- baselines --")
        for s in baseline_specs:
            print(f"{s['cell_id']:35s} batch={s['batch']:6d}")
        print("-- bars --")
        for s in bar_specs:
            print(f"{s['cell_id']:45s} budget_batch={s['batch']:6d} reserved_kv(GB)={s['reserved_kv']/1e9:8.1f} "
                  f"clamped={s['clamped']}")
        return [], []

    done_base = load_done_ids(FIG5_JSONL)
    todo_base = [s for s in baseline_specs if (s["cell_id"], s["config_hash"]) not in done_base]
    todo_bar = [s for s in bar_specs if (s["cell_id"], s["config_hash"]) not in done_base]
    print(f"[fig5] {len(baseline_specs)} baselines ({len(todo_base)} to run), "
          f"{len(bar_specs)} bars ({len(todo_bar)} to run), iters={iters}, workers={MAX_WORKERS}")

    base_results, bar_results = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_fig5_baseline_worker, s): ("base", s) for s in todo_base}
        futs.update({pool.submit(_fig5_bar_worker, s): ("bar", s) for s in todo_bar})
        for fut in as_completed(futs):
            kind, s = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[fig5] worker crashed for {s['cell_id']}: {e!r}")
                r = {"cell_id": s["cell_id"], "config_hash": s["config_hash"], "success": False,
                     "reason": "worker_crashed"}
            append_jsonl(FIG5_JSONL, r)
            status = "OK" if r.get("success") else f"FAIL({r.get('reason')})"
            print(f"[fig5] {r['cell_id']:45s} {status}")
            if kind == "base":
                base_results.append(r)
            else:
                bar_results.append(r)
    return base_results, bar_results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def run_smoke():
    probes = probe_all()
    model = "llama4_maverick"
    v_weight = probes[model]["weight_bytes_node"]
    kv_tok = probes[model]["kv_bytes_per_token_full"]
    ctx, ratio_label, ratio = 8192, "1:3", 0.75
    batch, reserved_kv, raw, clamped = compute_batch(v_weight, 2, ctx, kv_tok)
    print(f"[smoke] {model} ctx={ctx} ratio={ratio_label} batch={batch} "
          f"(raw={raw:.1f}, reserved_kv={reserved_kv/1e9:.1f}GB, clamped={clamped})")

    cfg = build_config(model, "P2_HBF", ctx, 0.1, 90.0, ratio, batch, SMOKE_ITER,
                        expose_first_expert_latency=True)

    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    res = run_binary(cfg, "smoke_maverick_8k_13", timeout=1800)
    wall_s = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    if not res["success"]:
        print(f"[smoke] FAILED: {res.get('reason')}\n{res.get('stdout_tail', '')}")
        return None

    metrics = fig4_metrics(res, v_weight)
    print(f"[smoke] wall_time={wall_s:.1f}s child_maxrss={rss_after/1024:.0f}MB "
          f"(delta vs pre-smoke ru_maxrss: {(rss_after-rss_before)/1024:.0f}MB; "
          f"ru_maxrss is cumulative-max over all children of this process, see run_binary)")
    print(f"[smoke] iter={SMOKE_ITER}: tpot={metrics['tpot_ms']:.1f}ms lifetime={metrics['lifetime_years']:.2f}yr "
          f"(paper: tpot~52ms lifetime~6.2yr) -- report only, no tuning")

    # FINAL_ITER decision: extrapolate wall time linearly in iter count.
    per_iter_s = wall_s / SMOKE_ITER
    projected_1000_min = per_iter_s * 1000 / 60.0
    final_iter_choice = 1000 if projected_1000_min <= 25.0 else 500
    print(f"[smoke] projected iter=1000 wall time ~= {projected_1000_min:.1f} min/cell -> "
          f"FINAL_ITER = {final_iter_choice} "
          f"({'<=25min threshold met' if final_iter_choice == 1000 else '>25min, falling back to 500 (steady-state stable, defensible)'})")

    decision = {
        "final_iter": final_iter_choice, "search_iter": SEARCH_ITER,
        "smoke_wall_s": wall_s, "smoke_child_maxrss_kb": rss_after,
        "smoke_tpot_ms": metrics["tpot_ms"], "smoke_lifetime_years": metrics["lifetime_years"],
        "projected_1000_iter_min": projected_1000_min,
    }
    save_iter_decision(decision)
    return decision


# ---------------------------------------------------------------------------
# Paper ground-truth tables (transcribed from Fig4_Fig5_extracted_numbers.md)
# ---------------------------------------------------------------------------
PAPER_FIG4 = {
    ("llama4_maverick", "cv0.1_k90"): {
        "norm": (1.00, 2.04, 4.14),
        "rows": {
            "1:3": {8192: (6.2, 52), 16384: (12.8, 52), 32768: (26.3, 53)},
            "1:1": {8192: (4.7, 60), 16384: (9.8, 59), 32768: (19.1, 59)},
            "3:1": {8192: (2.7, 68), 16384: (5.3, 67), 32768: (11.4, 65)},
            "7:1": {8192: (1.4, 75), 16384: (2.9, 73), 32768: (5.6, 71)},
            "15:1": {8192: (0.8, 80), 16384: (1.5, 79), 32768: (3.0, 75)},
        },
    },
    ("llama4_maverick", "cv0.3_k30"): {
        "norm": (1.00, 1.92, 4.22),
        "rows": {
            "1:3": {8192: (6.6, 54), 16384: (13.1, 54), 32768: (26.7, 55)},
            "1:1": {8192: (5.1, 61), 16384: (9.7, 61), 32768: (22.3, 61)},
            "3:1": {8192: (2.9, 72), 16384: (5.2, 70), 32768: (13.1, 69)},
            "7:1": {8192: (1.5, 79), 16384: (3.0, 77), 32768: (6.0, 74)},
            "15:1": {8192: (0.8, 84), 16384: (1.6, 82), 32768: (3.4, 79)},
        },
    },
    ("deepseekR1", "cv0.1_k90"): {
        "norm": (1.00, 1.70, 3.37),
        "rows": {
            "1:3": {8192: (10.0, 70), 16384: (17.5, 59), 32768: (34.3, 57)},
            "1:1": {8192: (7.4, 78), 16384: (12.0, 65), 32768: (24.2, 61)},
            "3:1": {8192: (4.0, 82), 16384: (6.9, 76), 32768: (13.7, 72)},
            "7:1": {8192: (2.0, 87), 16384: (3.5, 76), 32768: (7.2, 72)},
            "15:1": {8192: (1.1, 91), 16384: (1.8, 80), 32768: (3.5, 75)},
        },
    },
    ("deepseekR1", "cv0.3_k30"): {
        "norm": (1.00, 1.76, 3.31),
        "rows": {
            "1:3": {8192: (10.1, 72), 16384: (17.9, 62), 32768: (34.1, 58)},
            "1:1": {8192: (7.4, 78), 16384: (13.4, 67), 32768: (25.5, 64)},
            "3:1": {8192: (4.2, 85), 16384: (7.0, 74), 32768: (12.3, 70)},
            "7:1": {8192: (2.1, 89), 16384: (3.8, 79), 32768: (7.0, 75)},
            "15:1": {8192: (1.1, 94), 16384: (1.9, 83), 32768: (3.8, 78)},
        },
    },
}

PAPER_FIG5_BASELINE = {
    ("llama4_maverick", 8192): 9843, ("llama4_maverick", 32768): 2467,
    ("deepseekR1", 8192): 20483, ("deepseekR1", 32768): 5566,
}

PAPER_FIG5_SRAM_MB = {
    ("llama4_maverick", 2, 8192): 206, ("llama4_maverick", 3, 8192): 308,
    ("llama4_maverick", 2, 32768): 89, ("llama4_maverick", 3, 32768): 114,
    ("deepseekR1", 2, 8192): 265, ("deepseekR1", 3, 8192): 431,
    ("deepseekR1", 2, 32768): 104, ("deepseekR1", 3, 32768): 145,
}

# (model, budget, ctx) -> {"cpu_hbm_ratio": {"NVLink5.0":.., "NVLink6.0":..},
#                          "bars": {device: {"slo_bound":bool, "tpot_ms":val|None}}}
PAPER_FIG5_BARS = {
    ("llama4_maverick", 2, 8192): {"ratio": {"NVLink5.0": 1.6, "NVLink6.0": 1.8}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (False, 124), "HBF": (False, 51), "1/2-HBF": (False, 100)}},
    ("llama4_maverick", 3, 8192): {"ratio": {"NVLink5.0": 1.6, "NVLink6.0": 3.2}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (True, None), "HBF": (False, 75), "1/2-HBF": (False, 144)}},
    ("llama4_maverick", 2, 32768): {"ratio": {"NVLink5.0": 1.7, "NVLink6.0": 1.8}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (False, 122), "HBF": (False, 49), "1/2-HBF": (False, 97)}},
    ("llama4_maverick", 3, 32768): {"ratio": {"NVLink5.0": 1.7, "NVLink6.0": 3.3}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (True, None), "HBF": (False, 72), "1/2-HBF": (False, 142)}},
    ("deepseekR1", 2, 8192): {"ratio": {"NVLink5.0": 2.7, "NVLink6.0": 3.4}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (False, 153), "HBF": (False, 81), "1/2-HBF": (False, 130)}},
    ("deepseekR1", 3, 8192): {"ratio": {"NVLink5.0": 2.7, "NVLink6.0": 4.8}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (True, None), "HBF": (False, 129), "1/2-HBF": (True, None)}},
    ("deepseekR1", 2, 32768): {"ratio": {"NVLink5.0": 3.1, "NVLink6.0": 3.5}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (False, 130), "HBF": (False, 57), "1/2-HBF": (False, 108)}},
    ("deepseekR1", 3, 32768): {"ratio": {"NVLink5.0": 3.1, "NVLink6.0": 6.0}, "bars": {
        "NVLink5.0": (True, None), "NVLink6.0": (True, None), "HBF": (False, 83), "1/2-HBF": (False, 155)}},
}


def _pct_err(sim, paper):
    if paper is None or paper == 0 or sim is None:
        return None
    return (sim - paper) / paper * 100.0


def _sim_baseline_rec(bar_recs, model, ctx):
    """The paper's Fig5 'baseline throughput' (Fig4_Fig5_extracted_numbers.md
    'Baseline throughputs' table) is the DENOMINATOR its normalized bars are
    plotted against -- i.e. the leftmost, weakest bar of each group, which sits
    at 1.0x by construction. That bar is the device_HBM + NVLink5.0 CPU-offload
    config at the 512GB (budget=2x) reservation, run at/near the 200ms SLO --
    NOT the synthetic all-in-HBM batch=648 no-offload probe the harness also
    computes (that is a different, higher operating point: it fits entirely in
    HBM so it never pays the slow C2C read, giving ~2.5x the paper baseline).
    Cross-checked: sim NVLink5.0-512GB = 9,687 tok/s vs paper 9,843 for
    Maverick 8K (-1.6%); the paper-implied served batch at the 200ms SLO for
    all four (model,ctx) baselines matches compute_batch's NVLink5.0-512GB
    batch to a few percent (see PAPER2_NOTES.md). One baseline per (model,ctx),
    budget-independent -- matches the paper's single per-(model,ctx) number, so
    the 768GB group's bars normalize against this same 512GB reference."""
    return bar_recs.get(fig5_bar_id(model, ctx, 2, "NVLink5.0"))


def write_fig4_table():
    recs = {}
    if os.path.exists(FIG4_JSONL):
        with open(FIG4_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("success"):
                    recs[r["cell_id"]] = r

    lines = ["# Fig4 replication: sim vs paper\n",
             "Cell = lifetime_years (tpot_ms). Sim | Paper | %err on lifetime.\n"]
    for model in MODELS:
        for disp_key, cv, kappa in DISPERSIONS:
            paper = PAPER_FIG4[(model, disp_key)]
            lines.append(f"\n## {model} -- {disp_key}\n")
            lines.append(f"Paper normalized lifetime (8K/16K/32K): {paper['norm']}\n")
            lines.append("| L_in:L_out | 8K | 16K | 32K |")
            lines.append("|---|---|---|---|")
            for ratio_label, _ in RATIOS:
                cells = []
                for ctx in CONTEXTS:
                    cid = fig4_cell_id(model, disp_key, ctx, ratio_label)
                    r = recs.get(cid)
                    p_life, p_tpot = paper["rows"][ratio_label][ctx]
                    if r:
                        err = _pct_err(r["lifetime_years"], p_life)
                        cells.append(f"{r['lifetime_years']:.1f} ({r['tpot_ms']:.0f}ms) vs {p_life} ({p_tpot}ms) "
                                     f"[{err:+.0f}%]")
                    else:
                        cells.append(f"NOT RUN vs {p_life} ({p_tpot}ms)")
                lines.append(f"| {ratio_label} | " + " | ".join(cells) + " |")

            # ratio-averaged normalized lifetime (mean over 5 ratios of lifetime_ctx/lifetime_8K)
            norm_by_ctx = {}
            for ctx in CONTEXTS:
                vals = []
                for ratio_label, _ in RATIOS:
                    r8k = recs.get(fig4_cell_id(model, disp_key, 8192, ratio_label))
                    rctx = recs.get(fig4_cell_id(model, disp_key, ctx, ratio_label))
                    if r8k and rctx and r8k["lifetime_years"] > 0:
                        vals.append(rctx["lifetime_years"] / r8k["lifetime_years"])
                norm_by_ctx[ctx] = sum(vals) / len(vals) if vals else None
            sim_norm = tuple(norm_by_ctx[c] for c in CONTEXTS)
            lines.append(f"\nSim ratio-averaged normalized lifetime (8K/16K/32K): {sim_norm} "
                         f"(paper: {paper['norm']})\n")

    with open(os.path.join(RESULTS_DIR, "fig4_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tables] wrote {os.path.join(RESULTS_DIR, 'fig4_table.md')}")


def write_fig5_table():
    base_recs, bar_recs = {}, {}
    if os.path.exists(FIG5_JSONL):
        with open(FIG5_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("success"):
                    continue
                if r["cell_id"].startswith("baseline|"):
                    base_recs[r["cell_id"]] = r
                else:
                    bar_recs[r["cell_id"]] = r

    lines = ["# Fig5 replication: sim vs paper\n"]
    lines.append("## Baseline throughputs\n")
    lines.append("Paper baseline denominator = sim's **NVLink5.0 device_HBM @ 512GB** bar (at/near the "
                 "200ms SLO), NOT the synthetic all-in-HBM no-offload probe (last column, a different "
                 "operating point kept only as a diagnostic). See `_sim_baseline_rec` / PAPER2_NOTES.md.\n")
    lines.append("| Model | Context | Sim NVLink5.0-512GB tok/s | Paper tok/s | %err | "
                 "(diag) all-in-HBM probe tok/s |")
    lines.append("|---|---|---|---|---|---|")
    for model in MODELS:
        for ctx in FIG5_CONTEXTS:
            r = _sim_baseline_rec(bar_recs, model, ctx)
            probe = base_recs.get(fig5_baseline_id(model, ctx))
            probe_s = f"{probe['throughput_tok_s']:.0f}" if probe else "NOT RUN"
            paper = PAPER_FIG5_BASELINE[(model, ctx)]
            if r:
                err = _pct_err(r["throughput_tok_s"], paper)
                lines.append(f"| {model} | {ctx} | {r['throughput_tok_s']:.0f} | {paper} | "
                             f"{err:+.1f}% | {probe_s} |")
            else:
                lines.append(f"| {model} | {ctx} | NOT RUN | {paper} | - | {probe_s} |")

    lines.append("\n## Per-group bars\n")
    for model in MODELS:
        for ctx in FIG5_CONTEXTS:
            for budget in FIG5_BUDGETS:
                paper = PAPER_FIG5_BARS[(model, budget, ctx)]
                gb_label = budget * 256
                lines.append(f"\n### {model}, {gb_label}GB, ctx={ctx}\n")
                lines.append("| Device | Sim SLO-bound | Sim TPOT(ms) | Sim tok/s | Sim norm-tok/s | "
                              "Sim CPU/HBM ratio | Paper SLO-bound | Paper TPOT(ms) | Paper ratio | SRAM MB (sim) |")
                lines.append("|---|---|---|---|---|---|---|---|---|---|")
                # Self-consistent normalization: divide by the SIM's own
                # NVLink5.0-512GB baseline (so NVLink5.0-512 reads ~1.0x, exactly
                # as the paper's own axis is built), NOT the paper's absolute
                # number -- keeps "sim normalized bars" comparable to "paper
                # normalized bars" without importing the paper's value.
                sim_base_rec = _sim_baseline_rec(bar_recs, model, ctx)
                sim_base = sim_base_rec["throughput_tok_s"] if sim_base_rec else None
                for device in FIG5_DEVICES:
                    r = bar_recs.get(fig5_bar_id(model, ctx, budget, device))
                    p_slo, p_tpot = paper["bars"][device]
                    p_ratio = paper["ratio"].get(device)
                    if r:
                        norm = r["throughput_tok_s"] / sim_base if sim_base else None
                        norm_s = f"{norm:.2f}" if norm is not None else "n/a"
                        sram_mb = (r.get("sram_required_bytes_per_device") / 1e6
                                   if r.get("sram_required_bytes_per_device") else None)
                        lines.append(
                            f"| {device} | {r['slo_bound']} | {r['tpot_ms']:.1f} | {r['throughput_tok_s']:.0f} | "
                            f"{norm_s} | {r.get('cpu_hbm_ratio')} | {p_slo} | {p_tpot} | {p_ratio} | {sram_mb} |")
                    else:
                        lines.append(f"| {device} | NOT RUN | - | - | - | - | {p_slo} | {p_tpot} | {p_ratio} | - |")
                paper_sram = PAPER_FIG5_SRAM_MB.get((model, budget, ctx))
                lines.append(f"\nPaper SRAM/device for this group: {paper_sram} MB\n")

    with open(os.path.join(RESULTS_DIR, "fig5_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tables] wrote {os.path.join(RESULTS_DIR, 'fig5_table.md')}")


def write_summary():
    lines = ["# Paper2 replication -- headline summary\n",
              "Auto-generated from fig4_results.jsonl / fig5_results.jsonl. See fig4_table.md / "
              "fig5_table.md for the full breakdown.\n"]
    probes = load_probe_cache()
    lines.append("## Probe values\n")
    for m, v in probes.items():
        lines.append(f"- {m}: kv_bytes_per_token_full={v['kv_bytes_per_token_full']}, "
                     f"weight_bytes_node={v['weight_bytes_node']:.3e}")
    decision = load_iter_decision()
    if decision:
        lines.append(f"\n## Iteration decision\n- FINAL_ITER={decision.get('final_iter')}, "
                     f"smoke wall={decision.get('smoke_wall_s'):.1f}s, "
                     f"projected 1000-iter={decision.get('projected_1000_iter_min'):.1f}min\n")

    if os.path.exists(FIG4_JSONL):
        worst = []
        with open(FIG4_JSONL) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("success"):
                    continue
                try:
                    disp_key = r["cell_id"].split("|")[1]
                    paper_life = PAPER_FIG4[(r["model"], disp_key)]["rows"][r["ratio_label"]][r["ctx"]][0]
                    err = abs(_pct_err(r["lifetime_years"], paper_life))
                    worst.append((err, r["cell_id"]))
                except (KeyError, IndexError):
                    continue
        worst.sort(reverse=True)
        if worst:
            lines.append("\n## Largest Fig4 lifetime deviations (top 10)\n")
            for err, cid in worst[:10]:
                lines.append(f"- {cid}: {err:.0f}% off paper")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tables] wrote {os.path.join(RESULTS_DIR, 'summary.md')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probe", "smoke", "fig4", "fig5", "tables", "all"])
    ap.add_argument("--cells", default=None, help="comma-separated cell ids to restrict to")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells_filter = args.cells.split(",") if args.cells else None

    if args.command == "probe":
        probes = probe_all()
        for m, v in probes.items():
            expected_kv = EXPECTED_KV_BYTES_PER_TOKEN[m]
            ok = v["kv_bytes_per_token_full"] == expected_kv
            print(f"{m}: kv_bytes_per_token_full={v['kv_bytes_per_token_full']} "
                  f"(expected {expected_kv}, {'OK' if ok else 'MISMATCH'}) "
                  f"weight_bytes_node={v['weight_bytes_node']:.4e} "
                  f"(expected ~{EXPECTED_V_WEIGHT_NODE[m]:.4e})")
        return

    if args.command == "smoke":
        run_smoke()
        return

    if args.command == "fig4":
        run_fig4(final_iter(), cells_filter=cells_filter, dry_run=args.dry_run)
        return

    if args.command == "fig5":
        run_fig5(final_iter(), cells_filter=cells_filter, dry_run=args.dry_run)
        return

    if args.command == "tables":
        write_fig4_table()
        write_fig5_table()
        write_summary()
        return

    if args.command == "all":
        probe_all()
        run_smoke()
        print("[all] NOTE: the full 60-cell Fig4 + 32-cell Fig5 sweeps are NOT auto-launched here "
              "as a safety gate -- run `fig4`/`fig5` explicitly once smoke looks reasonable.")
        return


if __name__ == "__main__":
    main()
