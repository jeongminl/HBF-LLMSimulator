import os
import sys
import json
import yaml
import subprocess
import glob
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Must match model names with compressed_kv=true (MLA) in model_config.h.
MLA_MODELS = {"deepseekV3", "deepseekR1"}

# ---------------------------------------------------------------------------
# Shared sweep-scope constants. Single source of truth for both main() (the
# sweep driver) and generate_figures() (the paper-figure renderer), so the two
# never drift apart (e.g. generate_figures running standalone via
# --figures-only, without main()'s local variables in scope).
# ---------------------------------------------------------------------------
MODELS = ["llama3_405B", "llama4_maverick"]
WORKLOADS = {
    "SHORT": (1660, 373),
    "MID": (5900, 499),
    "LONG": (103500, 1100)
}
MEM_TYPES = ["HBM4", "HBF", "HBF+", "CONV", "CONV+"]
GPUS = [1, 2, 4, 8, 16]
SENSITIVITY_GPUS = [4, 8, 16]  # Fig. 6 scope (P2: matches paper's Fig. 6 GPU-count scope)
BREAKDOWN_GPUS = [4, 8, 16]    # Fig. 5 scope (P2: matches paper's Fig. 5 GPU-count scope)
SLOS = [0.05, 0.1, 0.2, 86400.0]  # Offline represents 24 hours
# Fig. 7 scope: the paper's Fig. 7 shows online (0.1s SLO) vs. offline (24h SLO)
# PEC for {SHORT,MID,LONG} at gpu {1,8,16} for {HBF,HBF+} only (flash-backed
# configs -- HBM4/CONV/CONV+ aren't part of the paper's PEC comparison).
PEC_WORKLOADS = ["SHORT", "MID", "LONG"]
PEC_GPUS = [1, 8, 16]
PEC_MEM_TYPES = ["HBF", "HBF+"]
# Fig. 3/4 x-axis ordering: pairs each flash config next to its conservative
# counterpart (HBM4, then CONV/HBF, then CONV+/HBF+), matching the paper's own
# Fig. 3 bar ordering. Distinct from MEM_TYPES (which drives the sweep loop and
# the Markdown report's table order, left unchanged).
FIG_MEM_ORDER = ["HBM4", "CONV", "HBF", "CONV+", "HBF+"]

# ---------------------------------------------------------------------------
# Optional plotting deps. Imported once at module load so a missing install
# fails fast with a clear message, but never blocks the (expensive) sweep
# itself from completing and writing its Markdown report + JSON dump.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_PLOTTING = True
except ImportError:
    HAVE_PLOTTING = False

if HAVE_PLOTTING:
    GPU_COLORS = {1: "#c6dbef", 2: "#9ecae1", 4: "#6baed6", 8: "#3182bd", 16: "#08519c"}
    FIG_MEM_STYLE = {
        "HBM4":  {"color": "#1f77b4", "marker": "D"},
        "HBF":   {"color": "#2ca02c", "marker": "^"},
        "HBF+":  {"color": "#d62728", "marker": "o"},
        "CONV":  {"color": "#7f7f7f", "marker": "v"},
        "CONV+": {"color": "#ff7f0e", "marker": "s"},
    }

def apply_mla_flags(cfg, model):
    # F6 defense-in-depth: config.yaml's system.optimization block defaults
    # compressed_kv/use_absorb/use_flash_mla to "on" (tuned for MLA models like
    # deepseekV3). eval/test.cpp now derives these from the model preset's
    # q_lora_rank as the primary fix, but set them explicitly here too so this
    # script never depends on the C++ binary's model-preset-aware correction.
    is_mla = model in MLA_MODELS
    cfg["system"]["optimization"]["compressed_kv"] = is_mla
    cfg["system"]["optimization"]["use_absorb"] = is_mla
    cfg["system"]["optimization"]["use_flash_mla"] = is_mla

def run_simulation(model, mem_type, num_device, batch_size, input_len, output_len, optimize_parallelism=True, tpot_slo=0.1, distribution=None, mfu_max=None, mfu_m_half=None):
    # mfu_max/mfu_m_half: optional compute-utilization (MFU) override, forwarded to
    # config.yaml's simulation.mfu_max/mfu_m_half (see SystemConfig::mfu_max/mfu_m_half,
    # hardware_config.h). None (default) omits the key entirely, so the C++ binary keeps
    # its in-class defaults (mfu_max=1.0, mfu_m_half=0.0) -- an exact no-op.
    # distribution: optional {"tp":.., "pp":.., "ep":..} to force a SPECIFIC parallelism
    # config (skips the in-process Optimize() re-derivation the binary would otherwise do
    # -- used when the analytic sweep already discovered the config for this exact batch).
    # When provided, optimize_parallelism is forced False regardless of the parameter above.
    # dp is derived here (total_gpus/(tp*pp)) since with an explicit distribution we already
    # know it; when distribution is None, dp/tp/pp/ep are parsed from the optimizer's own
    # "[Parallelism Optimizer] Found optimal configuration" stdout line instead.
    # Load default config
    with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # Update config values
    cfg["model"]["model_name"] = model
    cfg["system"]["memory_type"] = mem_type
    apply_mla_flags(cfg, model)

    # Handle num_device and num_node partitioning
    if num_device == 16:
        cfg["system"]["num_node"] = 2
        cfg["system"]["num_device"] = 8
    else:
        cfg["system"]["num_node"] = 1
        cfg["system"]["num_device"] = num_device
    total_gpus = num_device  # matches num_node*num_device regardless of the 16-GPU split above

    cfg["serving"]["max_batch_size"] = batch_size
    cfg["simulation"]["input_len"] = input_len
    cfg["simulation"]["output_len"] = output_len
    n_iter = 10
    cfg["simulation"]["iter"] = n_iter
    cfg["simulation"]["injection_rate"] = 0  # Continuous batching

    known_dp = None
    if distribution is not None:
        cfg["system"]["optimize_parallelism"] = False
        cfg["system"]["distribution"]["none_expert_tensor_degree"] = distribution["tp"]
        cfg["system"]["distribution"]["expert_tensor_degree"] = distribution["ep"]
        cfg["system"]["distribution"]["pipeline_degree"] = distribution["pp"]
        known_dp = total_gpus // (distribution["tp"] * distribution["pp"])
    else:
        cfg["system"]["optimize_parallelism"] = optimize_parallelism

    cfg["system"]["tpot_slo"] = tpot_slo
    # Make capacity / SRAM violations a hard failure so the batch-size sweep's
    # stop condition enforces all three constraints (capacity, SRAM, SLO), not
    # just the SLO.  The optimizer already exits non-zero when no parallelism
    # config fits; this also catches any over-capacity the simulation itself sees.
    cfg["simulation"]["exit_out_of_memory"] = True

    if mfu_max is not None:
        cfg["simulation"]["mfu_max"] = mfu_max
    if mfu_m_half is not None:
        cfg["simulation"]["mfu_m_half"] = mfu_m_half

    # Save temp config inside build
    temp_cfg_path = os.path.join(BUILD_DIR, "config_temp.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    cmd = ["./run", "config_temp.yaml"]
    try:
        res = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True, timeout=1800)
        stdout = res.stdout + "\n" + res.stderr

        # returncode != 0 catches optimizer OOM (exits with EXIT_FAILURE).
        # String markers catch the simulation OOM path which does "return 0".
        fail_markers = ["Out of Memory", "Activations exceed",
                        "Flash capacity exceeded", "capacity exceeded"]
        if res.returncode != 0 or any(m in stdout for m in fail_markers):
            return {"success": False, "reason": "OOM/Crash", "stdout": stdout}

        # Parse total simulation time.  cluster.cpp also emits "Total: X.XXXeGB"
        # memory-size lines; skip those by rejecting values containing letters.
        total_time_ns = None
        for line in stdout.split("\n"):
            if line.startswith("Total: "):
                raw = line[len("Total: "):].strip()
                if raw and not any(c.isalpha() for c in raw):
                    try:
                        total_time_ns = float(raw)
                    except ValueError:
                        pass
        if total_time_ns is None:
            return {"success": False, "reason": "No total time printed", "stdout": stdout}

        tpot = total_time_ns / (n_iter * 1e9)
        if tpot > tpot_slo:
            return {"success": False, "reason": f"SLO Violated ({tpot:.3f}s > {tpot_slo:.3f}s)", "tpot": tpot, "stdout": stdout}

        # Parse PEC geometry emitted by the C++ binary (single source of truth for
        # model dimensions, precision, and flash capacity).
        pec_kv_bytes = None
        pec_capacity = None
        dp = known_dp
        for line in stdout.split("\n"):
            if line.startswith("PEC_KV_BYTES_PER_TOKEN: "):
                try:
                    pec_kv_bytes = float(line.split(": ", 1)[1])
                except Exception:
                    pass
            elif line.startswith("PEC_FLASH_CAPACITY_BYTES: "):
                try:
                    pec_capacity = float(line.split(": ", 1)[1])
                except Exception:
                    pass
            elif dp is None and line.startswith("[Parallelism Optimizer] Found optimal configuration:"):
                # e.g. "...configuration: TP=1, PP=1, DP=8, EP=1 (Estimated Latency: 99.9 ms)"
                try:
                    dp_field = [p for p in line.split(",") if "DP=" in p][0]
                    dp = int(dp_field.split("DP=")[1].strip())
                except Exception:
                    pass

        # Find the newly generated or modified CSV file
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        csv_file = None
        newest_time = 0
        current_time = time.time()
        for f in csv_files:
            try:
                mtime = os.path.getmtime(f)
                if mtime > newest_time and (current_time - mtime) < 15:
                    newest_time = mtime
                    csv_file = f
            except:
                pass

        # Fallback: if dp still unknown (e.g. optimize_parallelism was False without an
        # explicit distribution -- shouldn't happen in this script, but guard anyway),
        # assume dp == total_gpus (pure data parallelism) rather than leaving it None.
        if dp is None:
            dp = total_gpus

        return {"success": True, "tpot": tpot, "csv_file": csv_file, "stdout": stdout,
                "pec_kv_bytes": pec_kv_bytes, "pec_capacity": pec_capacity, "dp": dp}
    except Exception as e:
        return {"success": False, "reason": str(e), "stdout": ""}

def run_analytic_sweep(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1):
    """Fast, in-process (no discrete-event simulation) batch-size search using ONLY
    the analytic max()-model, via eval/test.cpp's analytic_sweep_only mode (which
    calls ParallelismOptimizer::Optimize() repeatedly -- capacity/SRAM are hard
    constraints, analytic latency is a ranking/bounding heuristic).

    Returns a dict: {"max_batch": B*_analytic (0 if none found under the analytic SLO
    estimate), "cap_feasible_at_1": whether capacity/SRAM alone (ignoring latency) is
    satisfiable at B=1 -- distinguishes a genuine capacity infeasibility (max_batch==0 AND
    cap_feasible_at_1==False, no simulator run can rescue it) from a pure analytic-latency
    rejection (max_batch==0 AND cap_feasible_at_1==True, the simulator must still be asked
    per the audit's F1 finding), "tp"/"pp"/"ep"/"dp": the winning config, if any.

    IMPORTANT: this is a SEARCH HEURISTIC ONLY. The caller (find_max_batch_size) MUST
    verify the result with run_simulation() before reporting any batch size or metric --
    the simulator's measured tpot is the sole SLO arbiter (F1)."""
    with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["model_name"] = model
    cfg["system"]["memory_type"] = mem_type
    apply_mla_flags(cfg, model)
    if num_device == 16:
        cfg["system"]["num_node"] = 2
        cfg["system"]["num_device"] = 8
    else:
        cfg["system"]["num_node"] = 1
        cfg["system"]["num_device"] = num_device
    cfg["simulation"]["input_len"] = input_len
    cfg["simulation"]["output_len"] = output_len
    cfg["system"]["tpot_slo"] = tpot_slo
    cfg["system"]["analytic_sweep_only"] = True

    temp_cfg_path = os.path.join(BUILD_DIR, "config_temp.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    result = {"max_batch": 0, "cap_batch": 0, "cap_feasible_at_1": False,
              "tp": None, "pp": None, "ep": None, "dp": None}
    cmd = ["./run", "config_temp.yaml"]
    try:
        res = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True, timeout=120)
        stdout = res.stdout + "\n" + res.stderr
        for line in stdout.split("\n"):
            if line.startswith("ANALYTIC_CAP_FEASIBLE_AT_1: "):
                result["cap_feasible_at_1"] = line.split(": ", 1)[1].strip() == "1"
            elif line.startswith("ANALYTIC_CAP_BATCH: "):
                try:
                    result["cap_batch"] = int(line.split(": ", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("ANALYTIC_MAX_BATCH: "):
                try:
                    result["max_batch"] = int(line.split(": ", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("ANALYTIC_TP: "):
                result["tp"] = int(line.split(": ", 1)[1].strip())
            elif line.startswith("ANALYTIC_PP: "):
                result["pp"] = int(line.split(": ", 1)[1].strip())
            elif line.startswith("ANALYTIC_EP: "):
                result["ep"] = int(line.split(": ", 1)[1].strip())
            elif line.startswith("ANALYTIC_DP: "):
                result["dp"] = int(line.split(": ", 1)[1].strip())
        return result
    except Exception:
        return result

def classify_failure(fail_info):
    """Classify why the tightest known failing batch probe failed, for Fig. 3's
    SRAM-bound hatching (the paper hatches Fig. 3 bars whose batch size is capped
    by on-die SRAM, not by the flash bulk pool or the SLO latency estimate).
    `fail_info` is the {"reason", "stdout"} of the last verify() call that failed
    during find_max_batch_size's search for one (model, mem, gpu, workload) point --
    relies on monotonicity (once a capacity check fails it stays failed for all
    larger batches, already an existing assumption of the search algorithm itself),
    so any failing probe reasonably close to the true boundary classifies it
    correctly. Returns "sram" | "flash" | "slo" | "unknown".
    """
    reason = fail_info.get("reason")
    stdout = fail_info.get("stdout") or ""
    if reason == "OOM/Crash":
        # These substrings are emitted by src/model/footprint.h::checkCapacity
        # (optimizer, pre-run) and src/hardware/cluster.cpp's Part C scarce-tier
        # gate (simulator, post-run) -- both call into the shared
        # peakIntermediateBytes() gate (see CHANGES.md item 14).
        if "Activations exceed" in stdout or "activation capacity exceeded" in stdout:
            return "sram"
        if "Flash capacity exceeded" in stdout or "HBM capacity exceeded" in stdout:
            return "flash"
        return "unknown"
    if reason and reason.startswith("SLO Violated"):
        return "slo"
    return "unknown"

def find_max_batch_size(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1, mfu_max=None, mfu_m_half=None):
    # mfu_max/mfu_m_half: forwarded only to the simulator-verification calls (verify()
    # below), not to run_analytic_sweep -- the analytic phase is a heuristic seed only
    # (F1) and find_max_batch_size already reconciles analytic/simulator disagreement
    # via its fallback search branches, so an MFU-unaware analytic seed just costs a
    # few extra simulator calls to converge, never an incorrect result.
    # F1: the batch-size search is split into a cheap ANALYTIC phase (one fast
    # in-process sweep inside the C++ binary, no simulation spawned per probed
    # batch -- see run_analytic_sweep) that bounds the search, followed by a
    # SIMULATOR VERIFICATION phase that is the sole source of truth for every
    # reported batch size and metric. Preserves the spec's two-phase shape
    # (Phase 1 bounds / Phase 2 exact-integer binary search) -- Phase 1 now runs
    # analytically; Phase 2 is a small number of simulator calls near the
    # boundary, with graceful fallback to a full simulator-driven search if the
    # analytic estimate turns out to diverge from the simulator's measurement.
    #
    # Returns (max_b, max_tpot, csv, pec_kv, pec_cap, dp, bound_reason). max_b is the
    # RAW TOTAL/global batch (across all dp replicas) -- exactly what the C++/sim
    # layer works with internally (scheduler.cpp: batch_size_per_dp =
    # total_batch_size/dp_degree; parallelism_optimizer.cpp: batch_size_per_gpu =
    # batch_size/dp) and what the TPS/PEC formulas in main() already correctly
    # treat as a total (dividing by GPU count). Callers computing a PER-GPU batch
    # size (Metric 1/4's "Max Per-GPU Batch Size") must divide by the TOTAL GPU
    # COUNT used in the experiment, NOT the returned dp -- matches this project's
    # TPS definition ("... / Number of GPUs") and the paper's own published
    # anchors (e.g. 1555/8=194.4 vs. the paper's stated 194 for llama3
    # HBM4/8-GPU/SHORT). Dividing by dp instead penalizes a memory system for using DP
    # *more* effectively (more independent replicas = more total capacity from the same
    # hardware) -- see CHANGES.md for the investigation that corrected this (previously
    # "P1b"). Returned dp is still needed by callers for other purposes (e.g. distribution
    # passthrough), just not for this division.
    # bound_reason (new): classify_failure()'s verdict on why the batch just above
    # max_b failed -- "sram"/"flash"/"slo"/"unknown". Used only for Fig. 3's
    # SRAM-bound hatching (paper convention); callers not plotting Fig. 3 can
    # discard it.
    analytic = run_analytic_sweep(model, mem_type, num_device, input_len, output_len, tpot_slo)
    b_analytic = analytic["max_batch"]

    # Tracks the tightest known failing verify() call across this whole search,
    # for classify_failure() at every return site below.
    last_fail = {"reason": None, "stdout": ""}

    def verify(b, distribution=None):
        res = run_simulation(model, mem_type, num_device, b, input_len, output_len, True, tpot_slo,
                              distribution=distribution, mfu_max=mfu_max, mfu_m_half=mfu_m_half)
        if not res["success"]:
            last_fail["reason"] = res.get("reason")
            last_fail["stdout"] = res.get("stdout", "")
        return res

    def unpack(b, res):
        dp = res.get("dp") or 1
        return b, res["tpot"], res.get("csv_file"), res.get("pec_kv_bytes"), res.get("pec_capacity"), dp

    # BOUNDARY_WINDOW/probe_window: guards every search step below (exponential
    # doubling, and both binary searches) against a NON-MONOTONICITY artifact where
    # one probed batch happens to be the single integer in a divisibility cycle
    # (batch_size % dp == 0, parallelism_optimizer.cpp) that routes the live
    # optimizer to a WORSE parallelism config than its neighbors -- confirmed
    # empirically while investigating PAPER_INCONSISTENCIES.md's U8 (llama4_
    # maverick/HBF+/16-GPU/offline): b=9292 is divisible by 4 (TP=1/PP=4/DP=4
    # selected, feasible up to ~10947); b=9293 is divisible by neither 4 nor 2, so
    # only TP=4/PP=4/DP=1 (ceiling ~8192) remains divisibility-eligible and fails
    # there, even though b=9296 (divisible by 4 again) succeeds with TP=1 same as
    # b=9292. A single probe cannot distinguish this from a genuine ceiling, so
    # every step scans a window before concluding infeasibility.
    # Window size: a fixed 8 (sized to the TP degrees (<=8) this codebase's model
    # presets actually use) only guarantees catching the next value divisible by
    # some dp <= 8. Now that the optimizer ranks by throughput rather than
    # argmin(latency) (parallelism_optimizer.cpp), dp is no longer implicitly
    # biased toward small dp/large pp by a latency-minimizing objective and can be
    # chosen up to the full GPU count -- the worst-case gap between consecutive
    # multiples of dp is dp-1, so the window must cover up to num_device-1 to
    # guarantee catching a hit for ANY legal dp at this GPU count.
    BOUNDARY_WINDOW = max(num_device, 8)

    def probe_window(b_start):
        """Scan [b_start, b_start+BOUNDARY_WINDOW] for the first successful verify().
        Returns (b_found, res) on success, or (None, last res) if the whole window fails."""
        res_local = None
        for offset in range(0, BOUNDARY_WINDOW + 1):
            b_try = b_start + offset
            res_local = verify(b_try)
            if res_local["success"]:
                return b_try, res_local
        return None, res_local

    def analytic_distribution():
        if analytic["tp"] is None or analytic["pp"] is None or analytic["ep"] is None:
            return None
        return {"tp": analytic["tp"], "pp": analytic["pp"], "ep": analytic["ep"]}

    if b_analytic <= 0:
        if not analytic["cap_feasible_at_1"]:
            # Genuine capacity/SRAM infeasibility even at B=1 -- exact, analytic-only
            # constraint; no simulator run can change this (audit F1: this branch is the
            # legitimate analytic rejection, unlike a pure-latency-estimate one).
            return 0, 0.0, None, None, None, 1, "unknown"
        # Audit F1 fix: capacity was fine at B=1, only the analytic LATENCY ESTIMATE
        # rejected it. The estimate must never itself be the final word (the optimizer/
        # simulator separation-of-concerns principle) -- ask the real simulator before
        # declaring this combo infeasible.
        res1 = verify(1)
        if not res1["success"]:
            return 0, 0.0, None, None, None, 1, classify_failure(last_fail)
        b, tpot, csv, kv, cap, dp = unpack(1, res1)
        return b, tpot, csv, kv, cap, dp, classify_failure(last_fail)

    # Skip the redundant in-process Optimize() re-derivation for the analytically-known
    # batch/boundary checks by passing the discovered config through explicitly.
    dist = analytic_distribution()

    res = verify(b_analytic, distribution=dist)
    if res["success"]:
        max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp = unpack(b_analytic, res)

        # Boundary safety check: confirm batches just above b_analytic truly fail
        # (both the analytic model under-estimating latency, and the divisibility-
        # cycle non-monotonicity described above at probe_window's definition).
        b_probe, boundary_res = probe_window(b_analytic + 1)
        if b_probe is None:
            return max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp, classify_failure(last_fail)

        b_success = b_probe
        max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp = unpack(b_success, boundary_res)

        b = b_success * 2
        while True:
            b_next, res2 = probe_window(b)
            if b_next is not None:
                b_success = b_next
                max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp = unpack(b_next, res2)
                b = b_next * 2
            else:
                b_fail = b
                break

        low, high = b_success + 1, b_fail - 1
        while low <= high:
            mid = (low + high) // 2
            mid_found, res2 = probe_window(mid)
            if mid_found is not None:
                max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp = unpack(mid_found, res2)
                low = mid_found + 1
            else:
                high = mid - 1

        return max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp, classify_failure(last_fail)

    # Analytic estimate under-shot reality: the simulator found b_analytic itself
    # infeasible. Binary-search DOWNWARD with the simulator to pinpoint the true
    # max. b=1 is the base case: if even that fails, this combo is infeasible.
    # No pre-known config for these smaller candidates, so let the optimizer re-derive.
    res1 = verify(1)
    if not res1["success"]:
        return 0, 0.0, None, None, None, 1, classify_failure(last_fail)

    max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp = unpack(1, res1)

    low, high = 2, b_analytic - 1
    while low <= high:
        mid = (low + high) // 2
        mid_found, res2 = probe_window(mid)
        if mid_found is not None:
            max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp = unpack(mid_found, res2)
            low = mid_found + 1
        else:
            high = mid - 1

    return max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap, dp, classify_failure(last_fail)

def parse_csv_breakdown(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # We find rows with type = t2t or rows with actual duration info
        t2t_rows = [r for r in rows if r.get("type") == "t2t"]
        if not t2t_rows:
            return None

        # Parse average values
        avg_row = {}
        for col in t2t_rows[0].keys():
            try:
                avg_row[col] = sum(float(r[col]) for r in t2t_rows) / len(t2t_rows)
            except:
                avg_row[col] = 0.0
        return avg_row
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None

def compute_breakdown_fractions(model, bd):
    """Fractions of decode time spent on {attention, ffn, kv_write, communication,
    others} from one CSV-derived breakdown row `bd` -- the same grouping used by
    report Section 3 and Fig. 5, factored out here so the table and figure never
    drift apart. Returns None if `bd` is None (OOM/failed point)."""
    if bd is None:
        return None
    if model in MLA_MODELS:
        # MLA
        attn = (bd.get("q_down_proj", 0.0) + bd.get("kv_down_proj", 0.0) + bd.get("kr_proj", 0.0) +
                bd.get("q_up_proj", 0.0) + bd.get("qr_proj", 0.0) + bd.get("kv_up_proj", 0.0) +
                bd.get("tr_k_up_proj", 0.0) + bd.get("v_up_proj", 0.0) + bd.get("atten_sum", 0.0) +
                bd.get("atten_gen", 0.0) + bd.get("o_proj", 0.0))
    else:
        # Standard GQA
        attn = (bd.get("atten_sum", 0.0) + bd.get("atten_gen", 0.0) +
                bd.get("o_proj", 0.0) + bd.get("qkvgen", 0.0))
    ffn = bd.get("ffn", 0.0) + bd.get("expert_ffn", 0.0)
    comm = bd.get("communication", 0.0)
    others = bd.get("layernorm", 0.0) + bd.get("residual", 0.0) + bd.get("rope", 0.0) + bd.get("lm_head", 0.0)
    kv_write = bd.get("kv_write", 0.0)

    tot = attn + ffn + comm + kv_write + others
    if tot == 0:
        tot = 1.0
    return {
        "attention": attn / tot,
        "ffn": ffn / tot,
        "kv_write": kv_write / tot,
        "communication": comm / tot,
        "others": others / tot,
    }

def compute_pec(row):
    """3-year write-endurance PEC (program/erase cycle count) and per-GPU write
    rate for one result row -- the same formula used by report Section 5 and
    Fig. 7, factored out here so the table and figure never drift apart.
    `row` needs: workload, max_batch, tpot, gpus, pec_kv_bytes, pec_capacity.
    Returns None if max_batch is 0 or the PEC geometry wasn't emitted by the
    binary (kv_bytes_per_token/capacity sourced from the C++ binary's own
    model-dimension/precision/flash-capacity computation, so this never
    duplicates or risks desyncing from that)."""
    if row is None or row.get("max_batch", 0) == 0:
        return None
    kv_bytes_per_token = row.get("pec_kv_bytes")
    capacity = row.get("pec_capacity")
    if kv_bytes_per_token is None or capacity is None or capacity == 0:
        return None

    in_len, out_len = WORKLOADS[row["workload"]]
    # kv_bytes_per_token is the system-total KV bytes per token (all GPUs, all
    # layers).  Dividing by row["gpus"] gives per-GPU write rate, which is
    # correct for any TP/PP split (uniform sharding).
    kv_size = kv_bytes_per_token * (in_len + out_len)
    num_gpus = row["gpus"]

    # Completion rate ≈ max_batch / out_len per decode step. NOTE: uses the RAW
    # TOTAL batch (row["max_batch"]), not a per-GPU value -- this formula's
    # num_new_queries/write-rate already treats the batch as a system-wide
    # total (matching getKVWriteDuration's own num_new_queries =
    # num_seq/output_len), divided by num_gpus at the end to get a per-GPU rate.
    write_rate_bps = (row["max_batch"] / out_len) * kv_size / (row["tpot"] * num_gpus)
    write_rate_mbps = write_rate_bps / 1e6

    # 3-year PEC: total bytes written / flash capacity per GPU.
    pec = (write_rate_bps * 3600 * 24 * 365 * 3) / capacity
    return {"write_rate_mbps": write_rate_mbps, "pec": pec}

def _fmt_compact(v):
    """Compact numeric label for figure annotations, e.g. 1834 -> '1.8K'."""
    if v >= 1000:
        return f"{v/1000:.1f}K"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"

def _fig3_per_gpu_batch(results, baselines, out_path):
    """Fig. 3 replication: per-GPU batch size under 0.1s TPOT SLO. One panel per
    (model, workload); within a panel, one overlaid bar per memory config
    encoding GPU count as nested rectangles (largest GPU count drawn first/
    behind, so each GPU count's top edge stays visible -- the paper's "top of
    each colored segment = max value at that GPU count" convention). A GPU
    count whose value doesn't exceed the running max for smaller GPU counts is
    omitted entirely (paper: "subsumed by the preceding segment"). HBF+/CONV+
    bars are hatched where the batch is SRAM-bound (bound_reason=="sram")."""
    fig, axes = plt.subplots(len(MODELS), len(WORKLOADS), figsize=(18, 9), squeeze=False)
    x = np.arange(len(FIG_MEM_ORDER))
    for mi, model in enumerate(MODELS):
        for wi, wl in enumerate(WORKLOADS.keys()):
            ax = axes[mi][wi]
            base = baselines[model][wl]["max_batch_per_gpu"]
            for xi, mem in enumerate(FIG_MEM_ORDER):
                rows_by_gpu = {}
                for gpu in GPUS:
                    row = next((r for r in results if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    if row is not None and row["max_batch"] > 0:
                        rows_by_gpu[gpu] = row
                visible = []
                running_max = -1.0
                for gpu in sorted(rows_by_gpu.keys()):
                    row = rows_by_gpu[gpu]
                    v = row["max_batch_per_gpu"]
                    if v > running_max:
                        visible.append((gpu, row))
                        running_max = v
                top_val = None
                for gpu, row in reversed(visible):
                    norm_v = row["max_batch_per_gpu"] / base if base > 0 else 0.0
                    hatch = "///" if (mem in ("HBF+", "CONV+") and row.get("bound_reason") == "sram") else None
                    ax.bar(xi, norm_v, color=GPU_COLORS[gpu], hatch=hatch,
                           edgecolor="black", linewidth=0.4, zorder=20 - gpu)
                    if top_val is None:
                        top_val = (norm_v, row["max_batch_per_gpu"])
                if top_val is not None:
                    ax.text(xi, top_val[0] + 0.03, _fmt_compact(top_val[1]),
                            ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(FIG_MEM_ORDER, rotation=0, fontsize=8)
            ax.set_title(f"{model} / {wl}", fontsize=9)
            if wi == 0:
                ax.set_ylabel("Per-GPU batch size\n(norm. to 8-GPU HBM4)")
    gpu_handles = [plt.Rectangle((0, 0), 1, 1, color=GPU_COLORS[g]) for g in GPUS]
    sram_handle = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="///")
    fig.legend(gpu_handles + [sram_handle], [f"{g} GPU" for g in GPUS] + ["SRAM bound"],
               loc="upper center", ncol=len(GPUS) + 1, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Fig. 3 replication: Per-GPU batch size under 0.1s TPOT SLO", y=1.07)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fig4_throughput(results, baselines, out_path):
    """Fig. 4 replication: per-GPU TPS across GPU counts, one line per memory
    config, one panel per (model, workload), normalized to 8-GPU HBM4."""
    fig, axes = plt.subplots(len(MODELS), len(WORKLOADS), figsize=(18, 9), squeeze=False)
    for mi, model in enumerate(MODELS):
        for wi, wl in enumerate(WORKLOADS.keys()):
            ax = axes[mi][wi]
            anchor = baselines[model][wl]["tps"]
            for mem in FIG_MEM_ORDER:
                xs, ys = [], []
                for gpu in GPUS:
                    row = next((r for r in results if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    if row is not None and row["max_batch"] > 0:
                        xs.append(gpu)
                        ys.append(row["norm_tps"])
                style = FIG_MEM_STYLE[mem]
                ax.plot(xs, ys, marker=style["marker"], color=style["color"], label=mem,
                        linewidth=1.5, markersize=5)
            ax.set_xscale("log", base=2)
            ax.set_xticks(GPUS)
            ax.set_xticklabels(GPUS)
            ax.set_title(f"{model} / {wl}\n(w/ 8 GPUs HBM4: {_fmt_compact(anchor)} tok/s/GPU)", fontsize=8)
            if wi == 0:
                ax.set_ylabel("Per-GPU TPS\n(norm. to 8-GPU HBM4)")
    axes[0][0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Fig. 4 replication: System throughput", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fig5_breakdown(results, out_path):
    """Fig. 5 replication: stacked runtime-fraction bars for {MID,LONG} x
    {HBM4,HBF+} x GPU in {4,8,16}, one panel per model -- the paper's exact
    Fig. 5 scope."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 6), squeeze=False)
    components = ["attention", "ffn", "kv_write", "communication", "others"]
    comp_labels = ["Attention", "FFN", "KV Write", "Communication", "Others"]
    comp_colors = ["#4c72b0", "#dd8452", "#c44e52", "#8172b2", "#ccb974"]
    fig5_wls = ["MID", "LONG"]
    fig5_mems = ["HBM4", "HBF+"]
    for mi, model in enumerate(MODELS):
        ax = axes[0][mi]
        xpos = 0.0
        xticks, xlabels = [], []
        group_bounds = []
        for wl in fig5_wls:
            group_start = xpos
            for mem in fig5_mems:
                for gpu in BREAKDOWN_GPUS:
                    row = next((r for r in results if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    fracs = compute_breakdown_fractions(model, row["breakdown"]) if (row and row["max_batch"] > 0) else None
                    bottom = 0.0
                    if fracs is not None:
                        for comp, color in zip(components, comp_colors):
                            ax.bar(xpos, fracs[comp] * 100, bottom=bottom, color=color, width=0.8)
                            bottom += fracs[comp] * 100
                    xticks.append(xpos)
                    xlabels.append(str(gpu))
                    xpos += 1.0
                xpos += 0.6  # gap between mem sub-groups
            group_bounds.append((group_start, xpos - 0.6, wl))
            xpos += 1.0  # gap between workload groups
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7)
        for start, end, wl in group_bounds:
            ax.text((start + end) / 2, -12, wl, ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(model, fontsize=9)
        if mi == 0:
            ax.set_ylabel("TPOT breakdown [%]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in comp_colors]
    fig.legend(handles, comp_labels, loc="upper center", ncol=len(components), fontsize=8,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Fig. 5 replication: Runtime breakdown under 0.1s TPOT SLO", y=1.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fig6_slo_sensitivity(slo_results, out_path):
    """Fig. 6 replication: per-GPU batch size (bars, right axis) and per-GPU
    TPS (lines, left axis) across TPOT SLOs, for {HBM4,HBF,HBF+} within GPU
    groups {4,8,16}, one panel per model, LONG workload only (matches the
    sweep's own scope)."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 6), squeeze=False)
    fig6_mems = ["HBM4", "HBF", "HBF+"]
    width = 0.25
    for mi, model in enumerate(MODELS):
        ax = axes[0][mi]
        ax2 = ax.twinx()
        xpos = 0.0
        xticks, xlabels = [], []
        for gpu in SENSITIVITY_GPUS:
            base_x = xpos
            line_pts = {mem: ([], []) for mem in fig6_mems}
            for si, slo in enumerate(SLOS):
                for bi, mem in enumerate(fig6_mems):
                    row = next((r for r in slo_results if r["model"] == model and r["memory"] == mem
                                and r["gpus"] == gpu and r["slo"] == slo), None)
                    if row is None:
                        continue
                    bar_x = base_x + si * (len(fig6_mems) * width + 0.3) + bi * width
                    ax2.bar(bar_x, row["norm_batch"], width=width,
                            color=FIG_MEM_STYLE[mem]["color"], alpha=0.5)
                    line_pts[mem][0].append(bar_x)
                    line_pts[mem][1].append(row["norm_tps"])
            for mem in fig6_mems:
                xs, ys = line_pts[mem]
                if xs:
                    ax.plot(xs, ys, marker=FIG_MEM_STYLE[mem]["marker"],
                            color=FIG_MEM_STYLE[mem]["color"], linewidth=1.2, markersize=5)
            group_width = len(SLOS) * (len(fig6_mems) * width + 0.3)
            xticks.append(base_x + group_width / 2)
            xlabels.append(f"{gpu} GPU")
            xpos += group_width + 0.8
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_title(model, fontsize=9)
        ax.set_ylabel("Per-GPU TPS (norm.)")
        ax2.set_ylabel("Per-GPU batch size (norm.)")
    handles = [plt.Line2D([0], [0], color=FIG_MEM_STYLE[m]["color"], marker=FIG_MEM_STYLE[m]["marker"],
                          label=f"{m} TPS") for m in fig6_mems]
    handles += [plt.Rectangle((0, 0), 1, 1, color=FIG_MEM_STYLE[m]["color"], alpha=0.5,
                              label=f"{m} Batch") for m in fig6_mems]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.1))
    fig.suptitle("Fig. 6 replication: SLO sensitivity (LONG workload)", y=1.16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fig7_pec(results, pec_results, out_path):
    """Fig. 7 replication: required 3-year PEC for {HBF,HBF+} x {online,offline}
    across workloads {SHORT,MID,LONG} and GPU counts {1,8,16}, one panel per
    model, log-scale Y with the SLC PEC limit (1e5) marked. Online PEC comes
    from the main sweep's `results` (0.1s SLO); offline comes from the
    dedicated `pec_results` gathering pass (24h/no-limit SLO)."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 6), squeeze=False)
    series = [("HBF", "online"), ("HBF+", "online"), ("HBF", "offline"), ("HBF+", "offline")]
    series_colors = {
        ("HBF", "online"): "#2ca02c", ("HBF+", "online"): "#d62728",
        ("HBF", "offline"): "#98df8a", ("HBF+", "offline"): "#ff9896",
    }
    for mi, model in enumerate(MODELS):
        ax = axes[0][mi]
        xpos = 0.0
        xticks, xlabels = [], []
        for wl in PEC_WORKLOADS:
            for gpu in PEC_GPUS:
                group_start = xpos
                for mem, mode in series:
                    src = results if mode == "online" else pec_results
                    row = next((r for r in src if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    pec_info = compute_pec(row)
                    val = pec_info["pec"] if pec_info else 0.0
                    ax.bar(xpos, max(val, 1.0), color=series_colors[(mem, mode)], width=0.8)
                    xpos += 1.0
                xticks.append((group_start + xpos - 1.0) / 2.0)
                xlabels.append(str(gpu))
                xpos += 1.0  # gap between GPU-count groups
            xpos += 1.5  # gap between workload groups
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.axhline(1e5, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_title(model, fontsize=9)
        ax.set_ylabel("3-year PEC (log scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in series_colors.values()]
    labels = [f"{m} ({mode})" for (m, mode) in series_colors.keys()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
    labels.append("SLC PEC limit")
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fig. 7 replication: Required PEC under different operating scenarios", y=1.14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def generate_figures(data):
    """Render Figures 3-7 (Son et al., "Exploring High-Bandwidth Flash for Modern
    LLM Inference", IEEE CAL 2026) as PNGs in the project root, from data already
    computed by main()'s sweep (and persisted to experiment_data.json -- see
    --figures-only for standalone re-rendering without re-running the sweep).

    Known fidelity gaps vs. the paper's own figures (not chased): exact
    color/marker/font/spacing match to the paper's plotting style.
    """
    if not HAVE_PLOTTING:
        print("generate_figures: matplotlib/numpy not importable; skipping figure "
              "generation. Install them into this venv, then rerun with "
              "--figures-only to render figures from the already-saved "
              "experiment_data.json (no simulator re-run needed).")
        return

    results = data["results"]
    slo_results = data["slo_results"]
    pec_results = data["pec_results"]
    baselines = data["baselines"]

    _fig3_per_gpu_batch(results, baselines, os.path.join(SCRIPT_DIR, "fig3_per_gpu_batch.png"))
    _fig4_throughput(results, baselines, os.path.join(SCRIPT_DIR, "fig4_throughput.png"))
    _fig5_breakdown(results, os.path.join(SCRIPT_DIR, "fig5_breakdown.png"))
    _fig6_slo_sensitivity(slo_results, os.path.join(SCRIPT_DIR, "fig6_slo_sensitivity.png"))
    _fig7_pec(results, pec_results, os.path.join(SCRIPT_DIR, "fig7_pec.png"))
    print("Wrote fig3_per_gpu_batch.png, fig4_throughput.png, fig5_breakdown.png, "
          "fig6_slo_sensitivity.png, fig7_pec.png to the project root.")

def main():
    # Make sure output data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # We will record results for:
    # Models: llama3_405B, llama4_maverick
    # GPU counts: 1, 2, 4, 8, 16
    # Memory types: HBM4, HBF, HBF+, CONV, CONV+
    # Workloads: SHORT, MID, LONG
    models = MODELS
    workloads = WORKLOADS
    mem_types = MEM_TYPES
    gpus = GPUS

    # Gather baselines first (8 GPU, HBM4, slo=0.1); cached and reused in the
    # main sweep to avoid re-running the same simulation.
    baselines = {}
    print("--- GATHERING BASELINES (8 GPU HBM4) ---")
    for model in models:
        baselines[model] = {}
        for wl_name, (in_len, out_len) in workloads.items():
            max_b, tpot, csv_file, pec_kv, pec_cap, dp, bound_reason = find_max_batch_size(
                model, "HBM4", 8, in_len, out_len, 0.1)
            # Per-GPU metrics divide by the TOTAL GPU count, not dp -- matches
            # this project's TPS definition ("... / Number of GPUs") and
            # the paper's cross-GPU-count normalization (a DP replica still consumes
            # real GPU hardware; dividing by dp instead of GPU count penalizes a
            # config for using DP *more* effectively). See find_max_batch_size's
            # docstring and CHANGES.md for the investigation that corrected this.
            tps = max_b / (tpot * 8) if tpot > 0 else 0.0
            max_b_per_gpu = max_b / 8
            baselines[model][wl_name] = {
                "max_batch": max_b,
                "max_batch_per_gpu": max_b_per_gpu,
                "tps": tps,
                "tpot": tpot,
                "csv_file": csv_file,
                "pec_kv_bytes": pec_kv,
                "pec_capacity": pec_cap,
                "bound_reason": bound_reason,
            }
            print(f"Baseline {model} {wl_name}: Max Batch/GPU = {max_b_per_gpu:.1f} (total {max_b}, dp={dp}), TPS/GPU = {tps:.2f}")

    results = []

    print("\n--- RUNNING ALL SWEEPS ---")
    for model in models:
        for wl_name, (in_len, out_len) in workloads.items():
            for mem in mem_types:
                for gpu in gpus:
                    print(f"Running {model} | {wl_name} | {mem} | {gpu} GPUs...")

                    # HBM4/8-GPU result was already gathered as the baseline;
                    # pull from cache to avoid a duplicate simulation.
                    if mem == "HBM4" and gpu == 8:
                        bl = baselines[model][wl_name]
                        max_b      = bl["max_batch"]
                        max_b_per_gpu = bl["max_batch_per_gpu"]
                        tpot       = bl["tpot"]
                        csv_path   = bl["csv_file"]
                        last_pec_kv = bl["pec_kv_bytes"]
                        last_pec_cap = bl["pec_capacity"]
                        bound_reason = bl["bound_reason"]
                    else:
                        max_b, tpot, csv_path, last_pec_kv, last_pec_cap, dp, bound_reason = find_max_batch_size(
                            model, mem, gpu, in_len, out_len, 0.1)
                        max_b_per_gpu = max_b / gpu

                    tps = max_b / (tpot * gpu) if tpot > 0 else 0.0

                    # Normalization: batch-size normalizes per-GPU values (total/GPU-count) --
                    # GPU count can differ between the baseline (fixed at 8) and this config,
                    # so normalizing raw totals would be wrong whenever it does.
                    base_max_batch_per_gpu = baselines[model][wl_name]["max_batch_per_gpu"]
                    base_tps = baselines[model][wl_name]["tps"]

                    norm_batch = max_b_per_gpu / base_max_batch_per_gpu if base_max_batch_per_gpu > 0 else 0.0
                    norm_tps = tps / base_tps if base_tps > 0 else 0.0

                    # Breakdown parse
                    breakdown = parse_csv_breakdown(csv_path) if csv_path else None

                    results.append({
                        "model": model,
                        "workload": wl_name,
                        "memory": mem,
                        "gpus": gpu,
                        "max_batch": max_b,
                        "max_batch_per_gpu": max_b_per_gpu,
                        "tps": tps,
                        "tpot": tpot,
                        "norm_batch": norm_batch,
                        "norm_tps": norm_tps,
                        "breakdown": breakdown,
                        "csv_path": csv_path,
                        "pec_kv_bytes": last_pec_kv,
                        "pec_capacity": last_pec_cap,
                        "bound_reason": bound_reason,
                    })
                    print(f"  -> Max Batch/GPU: {max_b_per_gpu:.1f} (Norm: {norm_batch:.2f}), TPS/GPU: {tps:.2f} (Norm: {norm_tps:.2f})")

    # SLO Sensitivity Sweeps (LONG Workload only)
    print("\n--- RUNNING SLO SENSITIVITY SWEEPS (LONG Workload) ---")
    slo_results = []
    long_in, long_out = workloads["LONG"]
    slos = SLOS
    # P2: match the paper's Fig. 6 GPU-count scope. HBM4 can't even load either model
    # on 1-2 GPUs (paper states this explicitly), and this metric normalizes to HBM4,
    # so 1-2 GPU points would have an infeasible/undefined baseline.
    sensitivity_gpus = SENSITIVITY_GPUS

    # Gather sensitivity baselines (HBM4, 8 GPU) per SLO.
    # slo=0.1 / LONG workload was already gathered as the main baseline.
    sens_baselines = {}
    for model in models:
        sens_baselines[model] = {}
        for slo in slos:
            if slo == 0.1:
                bl = baselines[model]["LONG"]
                max_b_per_gpu = bl["max_batch_per_gpu"]
                tpot  = bl["tpot"]
                tps   = bl["tps"]
            else:
                max_b, tpot, _, _, _, dp, _ = find_max_batch_size(
                    model, "HBM4", 8, long_in, long_out, slo)
                max_b_per_gpu = max_b / 8
                tps = max_b / (tpot * 8) if tpot > 0 else 0.0
            sens_baselines[model][slo] = {
                "max_batch_per_gpu": max_b_per_gpu,
                "tps": tps,
                "tpot": tpot
            }
            print(f"SLO Sensitivity Baseline {model} SLO {slo}: Max Batch/GPU = {max_b_per_gpu:.1f}, TPS/GPU = {tps:.2f}")

    for model in models:
        for mem in mem_types:
            for gpu in sensitivity_gpus:
                for slo in slos:
                    # HBM4/8-GPU/slo=0.1 is the main LONG baseline — reuse it.
                    if mem == "HBM4" and gpu == 8 and slo == 0.1:
                        bl = baselines[model]["LONG"]
                        max_b_per_gpu = bl["max_batch_per_gpu"]
                        tps = bl["tps"]
                    else:
                        print(f"Sensitivity Sweep: {model} | {mem} | {gpu} GPUs | SLO {slo}s...")
                        max_b, tpot, _, _, _, dp, _ = find_max_batch_size(
                            model, mem, gpu, long_in, long_out, slo)
                        max_b_per_gpu = max_b / gpu
                        tps = max_b / (tpot * gpu) if tpot > 0 else 0.0

                    base_max_batch_per_gpu = sens_baselines[model][slo]["max_batch_per_gpu"]
                    base_tps = sens_baselines[model][slo]["tps"]

                    norm_batch = max_b_per_gpu / base_max_batch_per_gpu if base_max_batch_per_gpu > 0 else 0.0
                    norm_tps = tps / base_tps if base_tps > 0 else 0.0

                    slo_results.append({
                        "model": model,
                        "memory": mem,
                        "gpus": gpu,
                        "slo": slo,
                        "max_batch_per_gpu": max_b_per_gpu,
                        "tps": tps,
                        "norm_batch": norm_batch,
                        "norm_tps": norm_tps
                    })
                    if not (mem == "HBM4" and gpu == 8 and slo == 0.1):
                        print(f"  -> Max Batch/GPU: {max_b_per_gpu:.1f} (Norm: {norm_batch:.2f}), TPS/GPU: {tps:.2f} (Norm: {norm_tps:.2f})")

    # Offline-PEC data gathering (Fig. 7's "offline" bars). Online PEC for the
    # same scope is already derivable from `results` (0.1s SLO), but offline
    # (SLO=86400, i.e. no rate limit) isn't probed anywhere else -- the SLO
    # sensitivity sweep above only covers the LONG workload, and Fig. 7 needs
    # {SHORT,MID,LONG} x gpu{1,8,16} x {HBF,HBF+}.
    print("\n--- GATHERING OFFLINE PEC DATA (Fig. 7) ---")
    pec_results = []
    for model in models:
        for wl_name in PEC_WORKLOADS:
            in_len, out_len = workloads[wl_name]
            for gpu in PEC_GPUS:
                for mem in PEC_MEM_TYPES:
                    print(f"Offline PEC: {model} | {wl_name} | {mem} | {gpu} GPUs...")
                    max_b, tpot, _, pec_kv, pec_cap, dp, _ = find_max_batch_size(
                        model, mem, gpu, in_len, out_len, 86400.0)
                    row = {
                        "model": model,
                        "workload": wl_name,
                        "memory": mem,
                        "gpus": gpu,
                        "max_batch": max_b,
                        "tpot": tpot,
                        "pec_kv_bytes": pec_kv,
                        "pec_capacity": pec_cap,
                    }
                    pec_results.append(row)
                    pec_info = compute_pec(row)
                    pec_str = f"{pec_info['pec']:.1f}" if pec_info else "N/A"
                    print(f"  -> Offline Max Batch: {max_b}, 3yr PEC: {pec_str}")

    # Generate Markdown Report — written to the project root alongside this script.
    report_path = os.path.join(SCRIPT_DIR, "experiment_results.md")
    print(f"\nWriting final report to {report_path}...")

    with open(report_path, "w") as f:
        f.write("# HBF memory model simulation results\n\n")
        f.write("This report presents the findings from the automated sweeps evaluating HBF memory models.\n\n")

        # 1. Maximum Per-GPU Batch Size (Figure 3)
        f.write("## 1. Maximum Per-GPU Batch Size (Figure 3 Replication)\n\n")
        f.write("| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in mem_types:
                    gpu_vals = []
                    for gpu in gpus:
                        val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == gpu)
                        gpu_vals.append(f"{val['max_batch_per_gpu']:.1f} ({val['norm_batch']:.2f}x)")
                    f.write(f"| {model} | {wl} | {mem} | " + " | ".join(gpu_vals) + " |\n")

        f.write("\n")

        # 2. System Throughput (Figure 4)
        f.write("## 2. System Throughput (Figure 4 Replication)\n\n")
        f.write("| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in mem_types:
                    gpu_vals = []
                    for gpu in gpus:
                        val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == gpu)
                        gpu_vals.append(f"{val['tps']:.2f} ({val['norm_tps']:.2f}x)")
                    f.write(f"| {model} | {wl} | {mem} | " + " | ".join(gpu_vals) + " |\n")

        f.write("\n")

        # 3. Performance Breakdown (Figure 5)
        # P2: matches the paper's Fig. 5 GPU-count scope {4, 8, 16} (previously only 8),
        # since the paper uses multiple GPU counts to show how the breakdown shifts
        # (e.g. communication/attention fraction) as GPU count increases.
        breakdown_gpus = BREAKDOWN_GPUS
        f.write("## 3. Runtime Performance Breakdown (Figure 5 Replication)\n\n")
        f.write("Fractions of decode execution time spent on key components under a 0.1s TPOT SLO:\n\n")
        f.write("| Model | Workload | Memory | GPUs | Attention | FFN | KV Write | Communication | Others |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in mem_types:
                    for bgpu in breakdown_gpus:
                        val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == bgpu)

                        # If OOM/failed, write N/A
                        if val["max_batch"] == 0 or not val["breakdown"]:
                            f.write(f"| {model} | {wl} | {mem} | {bgpu} | N/A | N/A | N/A | N/A | N/A |\n")
                            continue

                        fracs = compute_breakdown_fractions(model, val["breakdown"])
                        f.write(f"| {model} | {wl} | {mem} | {bgpu} | {fracs['attention']:.1%} | {fracs['ffn']:.1%} | {fracs['kv_write']:.1%} | {fracs['communication']:.1%} | {fracs['others']:.1%} |\n")

        f.write("\n")

        # 4. SLO Sensitivity Analysis (Figure 6) — P2: matches the paper's GPU-count
        # scope {4, 8, 16} (previously swept/reported the full 1-16 range, which
        # included GPU counts where the HBM4 normalization baseline is infeasible).
        f.write("## 4. SLO Sensitivity Analysis (Figure 6 Replication)\n\n")
        gpu_headers = " | ".join(f"{g} GPU" for g in sensitivity_gpus)
        f.write(f"| Model | Memory | SLO | Metric | {gpu_headers} |\n")
        f.write("|" + "---|" * (4 + len(sensitivity_gpus)) + "\n")
        for model in models:
            for mem in mem_types:
                for slo in slos:
                    slo_label = f"{slo}s" if slo < 1000 else "Offline (24h)"
                    # Batch size row
                    b_vals = []
                    for gpu in sensitivity_gpus:
                        val = next(r for r in slo_results if r["model"] == model and r["memory"] == mem and r["gpus"] == gpu and r["slo"] == slo)
                        b_vals.append(f"{val['max_batch_per_gpu']:.1f} ({val['norm_batch']:.2f}x)")
                    f.write(f"| {model} | {mem} | {slo_label} | Batch Size | " + " | ".join(b_vals) + " |\n")

                    # TPS row
                    tps_vals = []
                    for gpu in sensitivity_gpus:
                        val = next(r for r in slo_results if r["model"] == model and r["memory"] == mem and r["gpus"] == gpu and r["slo"] == slo)
                        tps_vals.append(f"{val['tps']:.2f} ({val['norm_tps']:.2f}x)")
                    f.write(f"| {model} | {mem} | {slo_label} | TPS/GPU | " + " | ".join(tps_vals) + " |\n")

        f.write("\n")

        # 5. Write Traffic and Endurance Assessment (Figure 7)
        f.write("## 5. Write Traffic and Endurance Assessment (Figure 7 Replication)\n\n")
        f.write("Assuming continuous batching in steady state over a 3-year warranty lifespan:\n\n")
        f.write("| Model | Workload | Memory | Batch Size | TPS/GPU | Write rate/GPU (MB/s) | 3-Year PEC | Lifetime Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in ["HBF", "HBF+", "CONV", "CONV+"]:
                    val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == 8)
                    if val["max_batch"] == 0:
                        f.write(f"| {model} | {wl} | {mem} | 0 | 0.00 | 0.0 | N/A | N/A |\n")
                        continue

                    pec_info = compute_pec(val)
                    if pec_info is None:
                        f.write(f"| {model} | {wl} | {mem} | {val['max_batch_per_gpu']:.1f} | {val['tps']:.2f} | N/A | N/A | N/A |\n")
                        continue

                    status = "PASS (Safe)" if pec_info["pec"] < 100000 else "FAIL (Wear-out)"
                    f.write(f"| {model} | {wl} | {mem} | {val['max_batch_per_gpu']:.1f} | {val['tps']:.2f} | {pec_info['write_rate_mbps']:.1f} | {pec_info['pec']:.1f} | {status} |\n")

    # Persist raw data so figures (and any future analysis) can be regenerated
    # without re-running the (expensive) sweep -- see --figures-only.
    data = {
        "results": results,
        "slo_results": slo_results,
        "pec_results": pec_results,
        "baselines": baselines,
    }
    data_path = os.path.join(SCRIPT_DIR, "experiment_data.json")
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote raw experiment data to {data_path}")

    generate_figures(data)

if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        data_path = os.path.join(SCRIPT_DIR, "experiment_data.json")
        with open(data_path, "r") as f:
            _data = json.load(f)
        generate_figures(_data)
    else:
        main()
