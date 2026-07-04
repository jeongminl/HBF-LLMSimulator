import os
import sys
import json
import yaml
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Default pool width for parallelizing independent sweep cells (main() below). Each cell's
# OWN batch-size search is strictly sequential internally (find_max_batch_size's
# exponential/binary search issues one ./run at a time) -- only the OUTER loop over
# independent (model, workload, mem, gpu) cells is parallelized, so N workers means at most
# N concurrent ./run subprocesses. Leaves headroom (the -2) for the driver process itself;
# override via the SWEEP_WORKERS env var if this box's core count/memory profile differs.
DEFAULT_WORKERS = max(1, min((os.cpu_count() or 4) - 2, 18))
SWEEP_WORKERS = int(os.environ.get("SWEEP_WORKERS", DEFAULT_WORKERS))

# Offline/unconstrained-SLO cells (slo >= 1000, i.e. the 86400s "24h, no rate limit"
# probes used by the PEC-gathering phase and one SLOS entry in the sensitivity sweep)
# search for the ANALYTIC memory-capacity ceiling batch size, whose host-process RSS
# scales with that batch -- observed to reach 13-23GB per ./run, vs. a few GB for
# latency-bound online cells. Running SWEEP_WORKERS of these concurrently can
# oversubscribe host RAM well before hitting SWEEP_WORKERS (e.g. 6 x ~20GB > 63GB),
# triggering a sustained kernel OOM-killer storm that also kills unrelated processes.
# Use a much smaller, independently-tunable pool for offline cells specifically.
OFFLINE_SWEEP_WORKERS = int(os.environ.get("OFFLINE_SWEEP_WORKERS", max(1, SWEEP_WORKERS // 4)))

# Must match model names with compressed_kv=true (MLA) in model_config.h.
MLA_MODELS = {"deepseekV3", "deepseekR1"}

# ---------------------------------------------------------------------------
# Shared sweep-scope constants. Single source of truth for both main() (the
# sweep driver) and generate_figures() (the paper-figure renderer), so the two
# never drift apart (e.g. generate_figures running standalone via
# --figures-only, without main()'s local variables in scope).
# ---------------------------------------------------------------------------
MODELS = ["llama3_405B", "llama4_maverick"]
# Short display names used only in figure group labels, matching the paper's
# own "Llama3"/"Llama4" bold model-group labels in Figs. 3-7.
MODEL_DISPLAY = {"llama3_405B": "Llama3", "llama4_maverick": "Llama4"}
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
# Fig. 3/4 x-axis ordering: HBM4, then both conservative configs (CONV,
# CONV+), then both HBF configs (HBF, HBF+) -- matches the paper's own Fig. 3
# bar ordering exactly (verified against the paper's figure). Distinct from
# MEM_TYPES (which drives the sweep loop and the Markdown report's table
# order, left unchanged).
FIG_MEM_ORDER = ["HBM4", "CONV", "CONV+", "HBF", "HBF+"]
# Fig. 4's own legend/line order is different from Fig. 3's bar order in the
# paper (HBM4, HBF, HBF+, CONV, CONV+ -- flash configs grouped together
# before the conservative baselines).
FIG4_MEM_ORDER = ["HBM4", "HBF", "HBF+", "CONV", "CONV+"]

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
    # Colors below are sampled directly from the paper's own Figs. 3-7 (400 DPI
    # render of the PDF, legend-swatch pixel sampling) so the replications use
    # the paper's own palette rather than matplotlib defaults.
    GPU_COLORS = {1: "#595959", 2: "#5b7ab0", 4: "#99b487", 8: "#eabfa7", 16: "#f2e9d3"}
    SRAM_BOUND_COLOR = "#e48e37"
    FIG_MEM_STYLE = {
        "HBM4":  {"color": "#ed7d31", "edge": "#843c0c", "marker": "D"},
        "HBF":   {"color": "#9cc680", "edge": "#385723", "marker": "^"},
        "HBF+":  {"color": "#548235", "edge": "#243816", "marker": "o"},
        "CONV":  {"color": "#8faadc", "edge": "#203864", "marker": "^"},
        "CONV+": {"color": "#2f5597", "edge": "#1b3157", "marker": "o"},
    }
    FIG5_COLORS = {
        "attention": "#4472c4", "ffn": "#f4b183", "kv_write": "#7c7c7c",
        "communication": "#ffe699", "others": "#6fad46",
    }
    FIG6_BATCH_COLORS = {"HBM4": "#f4b183", "HBF": "#e2f0d9", "HBF+": "#92ae7d"}
    FIG7_COLORS = {
        ("HBF", "online"): "#e2f1da", ("HBF+", "online"): "#a9d28e",
        ("HBF", "offline"): "#6fad46", ("HBF+", "offline"): "#385723",
    }
    FIG7_EDGE = "#172c51"

    # ------------------------------------------------------------------
    # Shared layout helpers for the paper's "nested categorical axis"
    # style: one combined panel per figure (not one subplot per model),
    # with leaf ticks for the innermost category and 1-3 rows of centered
    # group labels (workload/memory/model) below the axis, separated by
    # vertical divider lines that run from the top of the plot down
    # through the label rows they group.
    # ------------------------------------------------------------------
    def _label_row(ax, spans, depth, bold=False, fontsize=9, rowlabel=None, base=-0.09, step=0.11):
        trans = ax.get_xaxis_transform()
        y = base - depth * step
        for start, end, label in spans:
            ax.text((start + end) / 2, y, label, transform=trans, ha="center", va="top",
                     fontsize=fontsize, fontweight="bold" if bold else "normal", clip_on=False)
        if rowlabel:
            ax.text(-0.008, y, rowlabel, transform=ax.transAxes, ha="right", va="top",
                     fontsize=fontsize, fontweight="bold", clip_on=False)

    def _dividers(ax, xs, label_depth, linewidth=0.8, base=-0.09, step=0.11):
        trans = ax.get_xaxis_transform()
        y_bottom = base - label_depth * step - 0.03
        for x in xs:
            ax.plot([x, x], [1.0, y_bottom], transform=trans, color="black",
                     linewidth=linewidth, clip_on=False, zorder=15)

def apply_mla_flags(cfg, model):
    # config.yaml's system.optimization block defaults compressed_kv/use_absorb/
    # use_flash_mla to "on" (tuned for MLA models like deepseekV3). eval/test.cpp
    # also derives these from the model preset's q_lora_rank, but set them explicitly
    # here too so this script never depends on the C++ binary's model-preset-aware
    # derivation.
    is_mla = model in MLA_MODELS
    cfg["system"]["optimization"]["compressed_kv"] = is_mla
    cfg["system"]["optimization"]["use_absorb"] = is_mla
    cfg["system"]["optimization"]["use_flash_mla"] = is_mla

def run_simulation(model, mem_type, num_device, batch_size, input_len, output_len, optimize_parallelism=True, tpot_slo=0.1, distribution=None, mfu_max=None, mfu_m_half=None, temp_cfg_name="config_temp.yaml", worker_tag=None):
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
    # temp_cfg_name/worker_tag: I/O-isolation for parallel callers. temp_cfg_name is the
    # config filename written under BUILD_DIR (defaults to the original shared name, so
    # sequential/single-process callers are unaffected). worker_tag, when set, additionally
    # isolates the CSV output directory (see below) so concurrent calls sharing the same
    # (model, gpu, batch) but different mem_type -- whose CSV filename does NOT encode
    # mem_type (only processor_type, which this script never varies) -- cannot clobber each
    # other's CSV file.
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

    # Suppress the per-op timeboard dump (config.yaml's log.print_log default: on).
    # Verified cosmetic-only: gates exactly one std::cout-only call
    # (top_module_graph->print_timeboard(), eval/test.cpp) which is the LAST statement
    # before main() returns -- strictly after the CSV write and every stdout marker this
    # function parses below (Total:/PEC_*/optimizer config/OOM markers). No overlap with
    # anything read here; this only removes the dominant per-op wall-clock cost. The
    # on-disk config.yaml itself is untouched (only this in-memory copy, dumped to the
    # temp config), so a bare `./run config.yaml` is unaffected.
    cfg["log"]["print_log"] = False
    if worker_tag is not None:
        # Isolate this worker's CSV output directory. Required for safe parallelism: the
        # CSV filename (eval/test.cpp) encodes processor_type, NOT mem_type, so two
        # concurrent calls differing only in mem_type at the same batch would otherwise
        # emit the identical path and clobber each other's file.
        out_dir = f"../data/{worker_tag}/"
        cfg["log"]["output_directory"] = out_dir
        os.makedirs(os.path.join(DATA_DIR, worker_tag), exist_ok=True)

    # Save temp config inside build
    temp_cfg_path = os.path.join(BUILD_DIR, temp_cfg_name)
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    cmd = ["./run", temp_cfg_name]
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
        # cluster.cpp additionally prints the exact CSV output path as the line
        # immediately following the numeric "Total: " line (unconditionally on the
        # success path) -- capture it here instead of globbing data/ afterward (see
        # csv_file below): safer (globbing an entire directory for "the newest file"
        # is a race under concurrent callers) and strictly equivalent (the CSV name is
        # a pure deterministic function of config params, no randomness/timestamp).
        total_time_ns = None
        raw_csv_line = None
        lines = stdout.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Total: "):
                raw = line[len("Total: "):].strip()
                if raw and not any(c.isalpha() for c in raw):
                    try:
                        total_time_ns = float(raw)
                        if i + 1 < len(lines):
                            raw_csv_line = lines[i + 1].strip()
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
        sram_diag_ceiling = None  # score-inclusive SRAM ceiling, diagnostic only (see cluster.cpp)
        dp = known_dp
        for line in stdout.split("\n"):
            if line.startswith("PEC_KV_BYTES_PER_SEQ: "):
                # Full per-sequence lifetime KV write volume (all layers, iRoPE
                # window applied per layer) — replaces the old uniform
                # PEC_KV_BYTES_PER_TOKEN marker; see eval/test.cpp emission.
                try:
                    pec_kv_bytes = float(line.split(": ", 1)[1])
                except Exception:
                    pass
            elif line.startswith("PEC_FLASH_CAPACITY_BYTES: "):
                try:
                    pec_capacity = float(line.split(": ", 1)[1])
                except Exception:
                    pass
            elif line.startswith("SRAM_DIAG_CEILING_BATCH_PER_GPU: "):
                try:
                    sram_diag_ceiling = float(line.split(": ", 1)[1])
                except Exception:
                    pass
            elif dp is None and line.startswith("[Parallelism Optimizer] Found optimal configuration:"):
                # e.g. "...configuration: TP=1, PP=1, DP=8, EP=1 (Estimated Latency: 99.9 ms)"
                try:
                    dp_field = [p for p in line.split(",") if "DP=" in p][0]
                    dp = int(dp_field.split("DP=")[1].strip())
                except Exception:
                    pass

        # Resolve the CSV path printed by the binary (see comment above). It's printed
        # relative to BUILD_DIR (the binary runs with cwd=BUILD_DIR), e.g. "../data/....csv"
        # -- normalize against BUILD_DIR, which is algebraically identical to DATA_DIR/<name>
        # (BUILD_DIR/../data == DATA_DIR). No fallback to a directory scan: the printed line
        # is unconditionally guaranteed once a numeric "Total: " was already found (both are
        # only reachable after cluster.cpp's export code has run); a silent glob fallback
        # would reintroduce the exact shared-directory race this replaces. If the expected
        # line is somehow missing, leave csv_file as None -- parse_csv_breakdown already
        # degrades gracefully for a missing/None path, same as today's "no CSV found" case.
        csv_file = None
        if raw_csv_line:
            csv_file = os.path.normpath(os.path.join(BUILD_DIR, raw_csv_line))

        # Fallback: if dp still unknown (e.g. optimize_parallelism was False without an
        # explicit distribution -- shouldn't happen in this script, but guard anyway),
        # assume dp == total_gpus (pure data parallelism) rather than leaving it None.
        if dp is None:
            dp = total_gpus

        return {"success": True, "tpot": tpot, "csv_file": csv_file, "stdout": stdout,
                "pec_kv_bytes": pec_kv_bytes, "pec_capacity": pec_capacity, "dp": dp,
                "sram_diag_ceiling": sram_diag_ceiling}
    except Exception as e:
        return {"success": False, "reason": str(e), "stdout": ""}

def run_analytic_sweep(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1, temp_cfg_name="config_temp.yaml"):
    """Fast, in-process (no discrete-event simulation) batch-size search using ONLY
    the analytic max()-model, via eval/test.cpp's analytic_sweep_only mode (which
    calls ParallelismOptimizer::Optimize() repeatedly -- capacity/SRAM are hard
    constraints, analytic latency is a ranking/bounding heuristic).

    Returns a dict: {"max_batch": B*_analytic (0 if none found under the analytic SLO
    estimate), "cap_feasible_at_1": whether capacity/SRAM alone (ignoring latency) is
    satisfiable at B=1 -- distinguishes a genuine capacity infeasibility (max_batch==0 AND
    cap_feasible_at_1==False, no simulator run can rescue it) from a pure analytic-latency
    rejection (max_batch==0 AND cap_feasible_at_1==True, the simulator must still be asked),
    "tp"/"pp"/"ep"/"dp": the winning config, if any.

    IMPORTANT: this is a SEARCH HEURISTIC ONLY. The caller (find_max_batch_size) MUST
    verify the result with run_simulation() before reporting any batch size or metric --
    the simulator's measured tpot is the sole SLO arbiter."""
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
    # Cosmetic-only suppression, same justification as run_simulation's identical line
    # (the analytic-sweep-only path returns before ever reaching the gated print call
    # anyway, so this is a no-op here in practice; kept for uniformity/future-proofing).
    cfg["log"]["print_log"] = False

    temp_cfg_path = os.path.join(BUILD_DIR, temp_cfg_name)
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    result = {"max_batch": 0, "cap_batch": 0, "cap_feasible_at_1": False,
              "tp": None, "pp": None, "ep": None, "dp": None}
    cmd = ["./run", temp_cfg_name]
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

def run_analytic_configs(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1, temp_cfg_name="config_temp.yaml"):
    """Per-config analytic listing via eval/test.cpp's analytic_configs_only mode,
    supporting the paper's SS-III objective ("each evaluated system selects the
    parallelism configuration that maximizes the achievable system throughput
    subject to all constraints"): for EVERY capacity-feasible (tp, pp, ep, dp)
    parallelism config, returns that config's own analytic capacity ceiling and
    SLO-latency hint so find_max_batch_size can run a simulator-verified
    max-batch search per config and report the argmax-TPS winner.

    Returns {"cap_feasible_at_1": bool, "configs": [ {tp, pp, ep, dp, cap_batch,
    slo_hint_batch, est_lat_min_ms, est_lat_hint_ms}, ... ]}. Batch values are
    TOTAL batches (multiples of that config's dp). All analytic values are
    SEARCH SEEDS/BOUNDS ONLY -- the simulator remains the sole SLO arbiter.
    """
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
    cfg["system"]["analytic_configs_only"] = True
    cfg["log"]["print_log"] = False

    temp_cfg_path = os.path.join(BUILD_DIR, temp_cfg_name)
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    result = {"cap_feasible_at_1": False, "configs": []}
    cmd = ["./run", temp_cfg_name]
    try:
        res = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True, timeout=300)
        stdout = res.stdout + "\n" + res.stderr
        for line in stdout.split("\n"):
            if line.startswith("ANALYTIC_CAP_FEASIBLE_AT_1: "):
                result["cap_feasible_at_1"] = line.split(": ", 1)[1].strip() == "1"
            elif line.startswith("ANALYTIC_CONFIG: "):
                fields = {}
                try:
                    for tok in line[len("ANALYTIC_CONFIG: "):].split():
                        k, v = tok.split("=", 1)
                        fields[k] = float(v) if "ms" in k else int(v)
                    result["configs"].append(fields)
                except Exception:
                    pass
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

def find_max_batch_size(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1, mfu_max=None, mfu_m_half=None, temp_cfg_name="config_temp.yaml", worker_tag=None):
    # Paper SS-III objective: "each evaluated system selects the parallelism
    # configuration that maximizes the achievable system throughput subject to all
    # constraints" (SLO, GPU-memory capacity, on-die SRAM). This function therefore
    # searches PER parallelism config -- every capacity-feasible (tp, pp, ep, dp)
    # gets its own simulator-verified max-batch search (over multiples of its own
    # dp; within a FIXED config feasibility is monotone in batch: capacity footprint
    # and measured tpot are both nondecreasing, and there is no batch%dp
    # config-switching) -- and the cell reports the config with the highest verified
    # TPS = batch / (tpot * total_gpus): its batch AND tpot become the cell's
    # operating point. See CHANGES.md item 31.
    #
    # mfu_max/mfu_m_half: forwarded only to the simulator-verification calls, not
    # to the analytic listing -- the analytic phase is a heuristic seed/bound only;
    # the simulator's measured tpot is the sole SLO arbiter.
    #
    # Returns (max_b, max_tpot, csv, pec_kv, pec_cap, dp, bound_reason); every field
    # describes the throughput-winning config's operating point. max_b is the RAW
    # TOTAL/global batch (across all dp replicas); per-GPU batch = max_b / TOTAL GPU
    # COUNT (not / dp), matching the TPS definition.
    # bound_reason: classify_failure()'s verdict on the WINNING config's own
    # boundary (the failing probe just above its max batch) -- "sram"/"flash"/
    # "slo"/"unknown". Used for Fig. 3's SRAM-bound hatching.
    #
    # Pruning: configs are simulated in descending analytic-TPS order; once one
    # config's verified TPS is in hand, any config whose analytic-TPS upper bound
    # cannot beat it is skipped without simulation. The bound is seed_tps =
    # slo_hint_batch / (est_lat_hint * G): under the estimate<=measured invariant
    # (audited by test.cpp's OVERESTIMATE warning), any sim-feasible batch b satisfies
    # est_lat(b) <= tpot(b) <= SLO, hence b <= slo_hint_batch, and TPS(b) =
    # b/(tpot(b)*G) <= b/(est_lat(b)*G), which is nondecreasing in b (est_lat has
    # constant terms), so it is maximized at the hint. Set HBF_DISABLE_CONFIG_PRUNING=1
    # to simulate every feasible config (validation / provable-argmax mode).
    listing = run_analytic_configs(model, mem_type, num_device, input_len, output_len,
                                   tpot_slo, temp_cfg_name=temp_cfg_name)

    total_gpus = num_device

    # Tracks the tightest known failing verify() call across the whole cell, for
    # classify_failure() at the no-winner return sites below.
    last_fail = {"reason": None, "stdout": ""}

    def verify(b, distribution=None):
        res = run_simulation(model, mem_type, num_device, b, input_len, output_len, True, tpot_slo,
                              distribution=distribution, mfu_max=mfu_max, mfu_m_half=mfu_m_half,
                              temp_cfg_name=temp_cfg_name, worker_tag=worker_tag)
        if not res["success"]:
            last_fail["reason"] = res.get("reason")
            last_fail["stdout"] = res.get("stdout", "")
        return res

    def unpack(b, res):
        dp = res.get("dp") or 1
        return b, res["tpot"], res.get("csv_file"), res.get("pec_kv_bytes"), res.get("pec_capacity"), dp

    def per_config_max_batch(c):
        """Simulator-verified max batch for ONE forced config.
        Returns (best_b, best_res, cfg_fail): best_b == 0 if even the minimum
        batch (dp) fails. cfg_fail is the last failing probe of THIS config (its
        own boundary), for bound_reason classification. Feasibility is monotone
        in batch within a fixed config, so plain bisection over multiples of dp."""
        dist = {"tp": c["tp"], "pp": c["pp"], "ep": c["ep"]}
        dp = max(c["dp"], 1)
        cfg_fail = {"reason": None, "stdout": ""}

        def v(b):
            res = verify(b, distribution=dist)
            if not res["success"]:
                cfg_fail["reason"] = res.get("reason")
                cfg_fail["stdout"] = res.get("stdout", "")
            return res

        k_cap = max(c["cap_batch"] // dp, 1)
        k_hint = min(max(c["slo_hint_batch"] // dp, 1), k_cap)

        res_hint = v(k_hint * dp)
        if not res_hint["success"]:
            # Analytic hint over-shot the simulator: bisect downward in [1, k_hint-1].
            best_k, best_res = 0, None
            lo, hi = 1, k_hint - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                r = v(mid * dp)
                if r["success"]:
                    best_k, best_res = mid, r
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best_k * dp, best_res, cfg_fail

        best_k, best_res = k_hint, res_hint

        # Jump straight to the analytic capacity ceiling (post weight-parity it
        # tracks the simulator's recorded footprint to <0.01%). If the ceiling
        # fails in the simulator, the true boundary lies inside (k_hint, k_cap):
        # bisect and return (a failure above best_k is then established).
        if k_cap > k_hint:
            res_cap = v(k_cap * dp)
            if res_cap["success"]:
                best_k, best_res = k_cap, res_cap
            else:
                lo, hi = k_hint + 1, k_cap - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    r = v(mid * dp)
                    if r["success"]:
                        best_k, best_res = mid, r
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best_k * dp, best_res, cfg_fail

        # Upward probe past the verified point: the analytic bounds are heuristics
        # in BOTH directions (the latency estimate can over-estimate -- the
        # est<=measured invariant is audited, not guaranteed), so exponentially
        # widen the step until a probe fails, then bisect the final gap.
        gap = 1
        while True:
            k_try = best_k + gap
            r = v(k_try * dp)
            if r["success"]:
                best_k, best_res = k_try, r
                gap *= 2
            else:
                lo, hi = best_k + 1, k_try - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    r2 = v(mid * dp)
                    if r2["success"]:
                        best_k, best_res = mid, r2
                        lo = mid + 1
                    else:
                        hi = mid - 1
                break
        return best_k * dp, best_res, cfg_fail

    configs = listing["configs"]
    if not configs:
        if not listing["cap_feasible_at_1"]:
            # Genuine capacity/SRAM infeasibility at every config's minimum batch --
            # exact, analytic-only constraint; no simulator run can change it.
            return 0, 0.0, None, None, None, 1, "unknown"
        # Defensive: marker says batch 1 fits but no config line was parsed. Ask
        # the simulator directly (optimizer picks the config) rather than declare
        # infeasibility from a parse gap.
        res1 = verify(1)
        if not res1["success"]:
            return 0, 0.0, None, None, None, 1, classify_failure(last_fail)
        b, tpot, csv, kv, cap, dp = unpack(1, res1)
        return b, tpot, csv, kv, cap, dp, classify_failure(last_fail)

    # Simulate likely winners first (descending analytic TPS at the SLO hint) so
    # the pruning bound bites early. seed_tps doubles as the config's analytic
    # TPS upper bound (see the pruning derivation in the docstring above).
    for c in configs:
        est_hint_s = max(c["est_lat_hint_ms"] / 1000.0, 1e-9)
        c["seed_tps"] = c["slo_hint_batch"] / est_hint_s / total_gpus
    configs.sort(key=lambda c: c["seed_tps"], reverse=True)

    prune = os.environ.get("HBF_DISABLE_CONFIG_PRUNING") != "1"

    best = None  # {"tps", "b", "res", "bound_reason", "config"}
    legacy_global_max_b = 0  # global max batch across all configs, logged for comparison only
    pruned = 0
    for c in configs:
        if prune and best is not None and c["seed_tps"] <= best["tps"]:
            pruned += 1
            continue
        b_star, res_star, cfg_fail = per_config_max_batch(c)
        if b_star <= 0 or res_star is None:
            continue
        legacy_global_max_b = max(legacy_global_max_b, b_star)
        tps = b_star / (res_star["tpot"] * total_gpus)
        if best is None or tps > best["tps"]:
            best = {"tps": tps, "b": b_star, "res": res_star,
                    "bound_reason": classify_failure(cfg_fail), "config": c}

    if best is None:
        # Every analytically-feasible config failed even its minimum batch in the
        # simulator. Final fallback: one optimizer-picked run at batch 1.
        res1 = verify(1)
        if not res1["success"]:
            return 0, 0.0, None, None, None, 1, classify_failure(last_fail)
        b, tpot, csv, kv, cap, dp = unpack(1, res1)
        return b, tpot, csv, kv, cap, dp, classify_failure(last_fail)

    w = best["config"]
    diag = best["res"].get("sram_diag_ceiling")
    diag_str = f", sram_diag_ceiling_per_gpu={diag:.1f}" if diag else ""
    print(f"[config-search] {model}/{mem_type}/{num_device}gpu slo={tpot_slo}: winner "
          f"tp={w['tp']} pp={w['pp']} ep={w['ep']} dp={w['dp']} batch={best['b']} "
          f"tps_per_gpu={best['tps']:.1f} (configs={len(configs)}, pruned={pruned}, "
          f"legacy_global_max_batch={max(legacy_global_max_b, best['b'])}{diag_str})")
    b, tpot, csv, kv, cap, dp = unpack(best["b"], best["res"])
    return b, tpot, csv, kv, cap, dp, best["bound_reason"]

def _run_cell(spec):
    """Picklable top-level worker for ProcessPoolExecutor (main(), below): runs ONE
    independent find_max_batch_size call for the cell described by `spec` (a plain dict --
    picklable). Isolates its own temp config filename and CSV output directory via a
    PID-derived tag (see run_simulation's temp_cfg_name/worker_tag) so concurrent workers
    can never collide on config_temp.yaml or on a CSV filename (which encodes
    processor_type, not mem_type -- see run_simulation's worker_tag comment).

    Deliberately does NOT do any of the per-cell post-processing (tps/norm_batch/norm_tps/
    breakdown parsing, result-dict construction, or printing) that main() does for a
    sequential call -- that all happens in the main process after collecting every future's
    result, in canonical spec order, so parallel vs. sequential execution is provably
    behavior-identical: this function computes exactly what a direct find_max_batch_size
    call would, nothing more, just relocated to a worker process.
    """
    worker_tag = f"w{os.getpid()}"
    temp_cfg_name = f"config_temp_{worker_tag}.yaml"
    max_b, tpot, csv_path, pec_kv, pec_cap, dp, bound_reason = find_max_batch_size(
        spec["model"], spec["mem"], spec["gpu"], spec["in_len"], spec["out_len"],
        spec.get("slo", 0.1), temp_cfg_name=temp_cfg_name, worker_tag=worker_tag)
    return {**spec, "max_b": max_b, "tpot": tpot, "csv_path": csv_path,
            "pec_kv": pec_kv, "pec_cap": pec_cap, "dp": dp, "bound_reason": bound_reason}

def _crashed_cell(spec):
    """Fallback result for a spec whose pool worker died before returning anything
    (e.g. the worker process itself -- not just its ./run child -- got killed by a
    system-wide OOM storm, or ProcessPoolExecutor reports BrokenProcessPool). Keeps
    the phase's canonical-order assembly loop working (same dict shape as _run_cell)
    instead of an unhandled exception from fut.result() crashing the whole script and
    discarding every other already-completed cell in this and all later phases."""
    return {**spec, "max_b": 0, "tpot": 0.0, "csv_path": None,
            "pec_kv": None, "pec_cap": None, "dp": 1, "bound_reason": "worker_crashed"}

def _checkpoint(name, obj):
    """Best-effort incremental checkpoint. This script has no resume logic, but
    writing each completed phase's raw results to disk as soon as it's done means a
    later-phase failure (timeout, worker crash, host OOM storm, ...) no longer
    discards prior phases -- previously, everything lived only in main()'s local
    variables until the single experiment_data.json write at the very end, so e.g. a
    fully-completed 144-cell main sweep + 118-cell sensitivity sweep were lost in
    their entirety when the PEC phase later died."""
    path = os.path.join(SCRIPT_DIR, f"checkpoint_{name}.json")
    try:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"[checkpoint] wrote {path}")
    except Exception as e:
        print(f"[checkpoint] failed to write {path}: {e!r}")

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
    kv_bytes_per_seq = row.get("pec_kv_bytes")
    capacity = row.get("pec_capacity")
    if kv_bytes_per_seq is None or capacity is None or capacity == 0:
        return None

    in_len, out_len = WORKLOADS[row["workload"]]
    # kv_bytes_per_seq is the system-total lifetime KV write volume per
    # sequence (all GPUs, all layers, per-layer iRoPE window applied — emitted
    # by the binary so it can never desync from the write-timing model).
    # Dividing by row["gpus"] gives per-GPU write rate, which is correct for
    # any TP/PP split (uniform sharding).
    kv_size = kv_bytes_per_seq
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
    """Fig. 3 replication: per-GPU batch size under 0.1s TPOT SLO. Single
    combined panel matching the paper's own layout: x-axis nests Model >
    Workload > MemConfig (order HBM4, CONV, CONV+, HBF, HBF+), with GPU count
    encoded as nested bar segments (largest GPU count drawn first/behind, so
    each GPU count's top edge stays visible -- the paper's "top of each
    colored segment = max value at that GPU count" convention). A GPU count
    whose value doesn't exceed the running max for smaller GPU counts is
    omitted entirely ("subsumed by the preceding segment"). HBF+/CONV+ bars
    that are SRAM-bound get the paper's orange X marker. Each workload
    group's own 8-GPU HBM4 batch size (the normalization base) is called out
    with a red arrow, exactly as in the paper."""
    fig, ax = plt.subplots(figsize=(16, 5.2))
    bar_width = 0.85
    wl_spans, model_spans = [], []
    wl_dividers, model_dividers = [], []
    xticks, xlabels = [], []
    xpos = 0.0
    for mi, model in enumerate(MODELS):
        if mi > 0:
            xpos += 2.0
            model_dividers.append(xpos - 1.0)
        model_start = xpos
        for wi, wl in enumerate(WORKLOADS.keys()):
            if wi > 0:
                xpos += 1.0
                wl_dividers.append(xpos - 0.5)
            wl_start = xpos
            base = baselines[model][wl]["max_batch_per_gpu"]
            hbm4_x, hbm4_top = None, 0.0
            for mem in FIG_MEM_ORDER:
                xi = xpos
                xticks.append(xi)
                xlabels.append(mem)
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
                top_norm, sram_bound = 0.0, False
                for gpu, row in reversed(visible):
                    norm_v = row["max_batch_per_gpu"] / base if base > 0 else 0.0
                    ax.bar(xi, norm_v, width=bar_width, color=GPU_COLORS[gpu],
                           edgecolor="black", linewidth=0.4, zorder=20 - gpu)
                    top_norm = max(top_norm, norm_v)
                    if mem in ("HBF+", "CONV+") and row.get("bound_reason") == "sram":
                        sram_bound = True
                if sram_bound:
                    ax.plot(xi, top_norm + 0.15, marker="X", color=SRAM_BOUND_COLOR,
                            markeredgecolor="black", markeredgewidth=0.7, markersize=12, zorder=40)
                if mem == "HBM4":
                    hbm4_x, hbm4_top = xi, top_norm
                xpos += 1.0
            ax.annotate(_fmt_compact(base), xy=(hbm4_x, hbm4_top),
                        xytext=(hbm4_x - 0.9, hbm4_top + 1.3),
                        color="#c00000", fontsize=8.5, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.1))
            wl_spans.append((wl_start, xpos - 1.0, wl))
        model_spans.append((model_start, xpos - 1.0, MODEL_DISPLAY.get(model, model)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7.5)
    ax.set_xlim(-0.7, xpos - 0.3)
    ax.set_ylabel("Per-GPU batch size\n(norm. to 8-GPU HBM4)")
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    _label_row(ax, wl_spans, depth=1, fontsize=9, base=-0.20, step=0.11)
    _label_row(ax, model_spans, depth=2, bold=True, fontsize=10, base=-0.20, step=0.11)
    _dividers(ax, wl_dividers, label_depth=1, linewidth=0.7, base=-0.20, step=0.11)
    _dividers(ax, model_dividers, label_depth=2, linewidth=1.3, base=-0.20, step=0.11)
    gpu_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GPU_COLORS[g], edgecolor="black", linewidth=0.4)
                   for g in GPUS]
    sram_handle = plt.Line2D([0], [0], marker="X", color="none", markerfacecolor=SRAM_BOUND_COLOR,
                              markeredgecolor="black", markersize=10, linewidth=0)
    fig.legend(gpu_handles + [sram_handle], [f"{g} GPU" for g in GPUS] + ["SRAM bound"],
               loc="upper center", ncol=len(GPUS) + 1, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig. 3 replication: Per-GPU batch size under 0.1s TPOT SLO", y=1.14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def _fig4_throughput(results, baselines, out_path):
    """Fig. 4 replication: per-GPU TPS, single combined panel matching the
    paper's layout: x-axis nests Model > Workload > GPU count (1,2,4,8,16),
    with one line per memory config crossing each workload group's five
    GPU-count positions. Each workload group's own 8-GPU HBM4 TPS (the
    normalization base) is called out with a red arrow, exactly as in the
    paper."""
    fig, ax = plt.subplots(figsize=(16, 5.2))
    wl_spans, model_spans = [], []
    wl_dividers, model_dividers = [], []
    xticks, xlabels = [], []
    xpos = 0.0
    for mi, model in enumerate(MODELS):
        if mi > 0:
            xpos += 2.0
            model_dividers.append(xpos - 1.0)
        model_start = xpos
        for wi, wl in enumerate(WORKLOADS.keys()):
            if wi > 0:
                xpos += 1.0
                wl_dividers.append(xpos - 0.5)
            wl_start = xpos
            anchor = baselines[model][wl]["tps"]
            gpu_x = {}
            for gi, gpu in enumerate(GPUS):
                gpu_x[gpu] = xpos + gi
                xticks.append(xpos + gi)
                xlabels.append(str(gpu))
            hbm4_pt = None
            for mem in FIG4_MEM_ORDER:
                xs, ys = [], []
                for gpu in GPUS:
                    row = next((r for r in results if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    if row is not None and row["max_batch"] > 0:
                        xs.append(gpu_x[gpu])
                        ys.append(row["norm_tps"])
                        if mem == "HBM4" and gpu == 8:
                            hbm4_pt = (gpu_x[gpu], row["norm_tps"])
                style = FIG_MEM_STYLE[mem]
                ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                        markeredgecolor=style["edge"], markeredgewidth=0.6,
                        linewidth=1.6, markersize=6, zorder=10)
            if hbm4_pt is not None:
                ax.annotate(_fmt_compact(anchor), xy=hbm4_pt,
                            xytext=(hbm4_pt[0] - 1.2, hbm4_pt[1] + 0.32),
                            color="#c00000", fontsize=8.5, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.1))
            wl_spans.append((wl_start, xpos + len(GPUS) - 1, wl))
            xpos += len(GPUS)
        model_spans.append((model_start, xpos - 1.0, MODEL_DISPLAY.get(model, model)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_xlim(-0.7, xpos - 0.3)
    ax.set_ylabel("Per-GPU TPS\n(norm. to 8-GPU HBM4)")
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.008, -0.03, "# GPUs", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", clip_on=False)
    _label_row(ax, wl_spans, depth=1, fontsize=9, base=-0.09, step=0.11)
    _label_row(ax, model_spans, depth=2, bold=True, fontsize=10, base=-0.09, step=0.11)
    _dividers(ax, wl_dividers, label_depth=1, linewidth=0.7, base=-0.09, step=0.11)
    _dividers(ax, model_dividers, label_depth=2, linewidth=1.3, base=-0.09, step=0.11)
    handles = [plt.Line2D([0], [0], color=FIG_MEM_STYLE[m]["color"], marker=FIG_MEM_STYLE[m]["marker"],
                          markeredgecolor=FIG_MEM_STYLE[m]["edge"], linewidth=1.6, markersize=7, label=m)
               for m in FIG4_MEM_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(FIG4_MEM_ORDER), fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig. 4 replication: System throughput", y=1.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def _fig5_breakdown(results, out_path):
    """Fig. 5 replication: stacked runtime-fraction bars, single combined
    panel matching the paper's layout: x-axis nests Model > Memory
    {HBM4,HBF+} > Workload {MID,LONG} > GPU {4,8,16} -- the paper's exact
    Fig. 5 scope and nesting order."""
    fig, ax = plt.subplots(figsize=(15, 5.4))
    components = ["attention", "ffn", "kv_write", "communication", "others"]
    comp_labels = ["Attention", "FFN", "KV Write", "Communication", "Others"]
    comp_colors = [FIG5_COLORS[c] for c in components]
    fig5_wls = ["MID", "LONG"]
    fig5_mems = ["HBM4", "HBF+"]
    xticks, xlabels = [], []
    wl_spans, mem_spans, model_spans = [], [], []
    wl_dividers, mem_dividers, model_dividers = [], [], []
    xpos = 0.0
    for mi, model in enumerate(MODELS):
        if mi > 0:
            xpos += 2.0
            model_dividers.append(xpos - 1.0)
        model_start = xpos
        for mj, mem in enumerate(fig5_mems):
            if mj > 0:
                xpos += 1.0
                mem_dividers.append(xpos - 0.5)
            mem_start = xpos
            for wi, wl in enumerate(fig5_wls):
                if wi > 0:
                    xpos += 1.0
                    wl_dividers.append(xpos - 0.5)
                wl_start = xpos
                for gpu in BREAKDOWN_GPUS:
                    row = next((r for r in results if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    fracs = compute_breakdown_fractions(model, row["breakdown"]) if (row and row["max_batch"] > 0) else None
                    bottom = 0.0
                    if fracs is not None:
                        for comp, color in zip(components, comp_colors):
                            ax.bar(xpos, fracs[comp] * 100, bottom=bottom, color=color,
                                   width=0.85, edgecolor="black", linewidth=0.3, zorder=10)
                            bottom += fracs[comp] * 100
                    xticks.append(xpos)
                    xlabels.append(str(gpu))
                    xpos += 1.0
                wl_spans.append((wl_start, xpos - 1.0, wl))
            mem_spans.append((mem_start, xpos - 1.0, mem))
        model_spans.append((model_start, xpos - 1.0, MODEL_DISPLAY.get(model, model)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_xlim(-0.7, xpos - 0.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("TPOT breakdown [%]")
    ax.text(-0.008, -0.03, "# GPUs", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", clip_on=False)
    _label_row(ax, wl_spans, depth=1, fontsize=8.5, base=-0.09, step=0.10)
    _label_row(ax, mem_spans, depth=2, fontsize=9, base=-0.09, step=0.10)
    _label_row(ax, model_spans, depth=3, bold=True, fontsize=10, base=-0.09, step=0.10)
    _dividers(ax, wl_dividers, label_depth=1, linewidth=0.6, base=-0.09, step=0.10)
    _dividers(ax, mem_dividers, label_depth=2, linewidth=0.9, base=-0.09, step=0.10)
    _dividers(ax, model_dividers, label_depth=3, linewidth=1.3, base=-0.09, step=0.10)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.3) for c in comp_colors]
    fig.legend(handles, comp_labels, loc="upper center", ncol=len(components), fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig. 5 replication: Runtime breakdown under 0.1s TPOT SLO", y=1.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def _fig6_slo_sensitivity(slo_results, out_path):
    """Fig. 6 replication: per-GPU TPS (lines, left axis) and per-GPU batch
    size (bars, right axis), single combined panel matching the paper's
    layout: x-axis nests Model > GPU {4,8,16} > TPOT SLO {0.05,0.1,0.2,
    offline}, for {HBM4,HBF,HBF+}, LONG workload only (matches the sweep's
    own scope)."""
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    fig6_mems = ["HBM4", "HBF", "HBF+"]
    bar_width = 0.26
    offsets = [-0.28, 0.0, 0.28]
    gpu_spans, model_spans = [], []
    gpu_dividers, model_dividers = [], []
    xticks, xlabels = [], []
    line_pts = {model: {mem: ([], []) for mem in fig6_mems} for model in MODELS}
    xpos = 0.0
    for mi, model in enumerate(MODELS):
        if mi > 0:
            xpos += 2.0
            model_dividers.append(xpos - 1.0)
        model_start = xpos
        for gi, gpu in enumerate(SENSITIVITY_GPUS):
            if gi > 0:
                xpos += 1.0
                gpu_dividers.append(xpos - 0.5)
            gpu_start = xpos
            for slo in SLOS:
                leaf_x = xpos
                xticks.append(leaf_x)
                xlabels.append("offline" if slo > 1000 else str(slo))
                for bi, mem in enumerate(fig6_mems):
                    row = next((r for r in slo_results if r["model"] == model and r["memory"] == mem
                                and r["gpus"] == gpu and r["slo"] == slo), None)
                    if row is None:
                        continue
                    bar_x = leaf_x + offsets[bi]
                    ax2.bar(bar_x, row["norm_batch"], width=bar_width,
                            color=FIG6_BATCH_COLORS[mem], edgecolor="black", linewidth=0.3, zorder=5)
                    line_pts[model][mem][0].append(leaf_x)
                    line_pts[model][mem][1].append(row["norm_tps"])
                xpos += 1.0
            gpu_spans.append((gpu_start, xpos - 1.0, f"{gpu}"))
        model_spans.append((model_start, xpos - 1.0, MODEL_DISPLAY.get(model, model)))
    for model in MODELS:
        for mem in fig6_mems:
            xs, ys = line_pts[model][mem]
            if xs:
                style = FIG_MEM_STYLE[mem]
                ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                        markeredgecolor=style["edge"], markeredgewidth=0.6,
                        linewidth=1.6, markersize=6, zorder=20)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_xlim(-0.7, xpos - 0.3)
    ax.set_ylabel("Per-GPU TPS (norm.)")
    ax2.set_ylabel("Per-GPU batch size (norm.)")
    ax.text(-0.008, -0.24, "SLO", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", clip_on=False)
    _label_row(ax, gpu_spans, depth=1, fontsize=9, rowlabel="# GPUs", base=-0.22, step=0.12)
    _label_row(ax, model_spans, depth=2, bold=True, fontsize=10, base=-0.22, step=0.12)
    _dividers(ax, gpu_dividers, label_depth=1, linewidth=0.7, base=-0.22, step=0.12)
    _dividers(ax, model_dividers, label_depth=2, linewidth=1.3, base=-0.22, step=0.12)
    tps_handles = [plt.Line2D([0], [0], color=FIG_MEM_STYLE[m]["color"], marker=FIG_MEM_STYLE[m]["marker"],
                              markeredgecolor=FIG_MEM_STYLE[m]["edge"], linewidth=1.6, markersize=7, label=m)
                   for m in fig6_mems]
    batch_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FIG6_BATCH_COLORS[m], edgecolor="black",
                                   linewidth=0.3, label=m) for m in fig6_mems]
    leg1 = fig.legend(handles=tps_handles, loc="upper center", ncol=3, fontsize=8,
                       bbox_to_anchor=(0.3, 1.02), title="TPS", title_fontsize=8)
    fig.add_artist(leg1)
    fig.legend(handles=batch_handles, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.72, 1.02), title="Batch", title_fontsize=8)
    fig.suptitle("Fig. 6 replication: SLO sensitivity (LONG workload)", y=1.16)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def _fig7_pec(results, pec_results, out_path):
    """Fig. 7 replication: required 3-year PEC, single combined panel
    matching the paper's layout: x-axis nests Model > Workload {SHORT,MID,
    LONG} > GPU {1,8,16}, with 4 clustered bars per GPU count ({HBF,HBF+} x
    {online,offline}). Linear Y in units of 10^3 PEC (matching the paper's
    own axis), with the SLC PEC limit (100 == 1e5) marked in red. Online PEC
    comes from the main sweep's `results` (0.1s SLO); offline comes from the
    dedicated `pec_results` gathering pass (24h/no-limit SLO)."""
    fig, ax = plt.subplots(figsize=(15, 5.4))
    series = [("HBF", "online"), ("HBF+", "online"), ("HBF", "offline"), ("HBF+", "offline")]
    bar_width = 0.20
    offsets = [-0.33, -0.11, 0.11, 0.33]
    wl_spans, model_spans = [], []
    wl_dividers, model_dividers = [], []
    xticks, xlabels = [], []
    xpos = 0.0
    for mi, model in enumerate(MODELS):
        if mi > 0:
            xpos += 2.0
            model_dividers.append(xpos - 1.0)
        model_start = xpos
        for wi, wl in enumerate(PEC_WORKLOADS):
            if wi > 0:
                xpos += 1.0
                wl_dividers.append(xpos - 0.5)
            wl_start = xpos
            for gpu in PEC_GPUS:
                leaf_x = xpos
                xticks.append(leaf_x)
                xlabels.append(str(gpu))
                for si, (mem, mode) in enumerate(series):
                    src = results if mode == "online" else pec_results
                    row = next((r for r in src if r["model"] == model and r["workload"] == wl
                                and r["memory"] == mem and r["gpus"] == gpu), None)
                    pec_info = compute_pec(row)
                    val = (pec_info["pec"] if pec_info else 0.0) / 1000.0
                    ax.bar(leaf_x + offsets[si], val, width=bar_width,
                           color=FIG7_COLORS[(mem, mode)], edgecolor=FIG7_EDGE, linewidth=0.4, zorder=10)
                xpos += 1.0
            wl_spans.append((wl_start, xpos - 1.0, wl))
        model_spans.append((model_start, xpos - 1.0, MODEL_DISPLAY.get(model, model)))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlim(-0.7, xpos - 0.3)
    ax.set_ylabel("3-year PEC ($10^3$)")
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(100, color="#c00000", linewidth=1.4, zorder=25)
    ax.annotate("SLC PEC limit", xy=(0.4, 100), xycoords="data",
                xytext=(0.02, 0.82), textcoords="axes fraction",
                color="#c00000", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.3))
    ax.text(-0.008, -0.03, "# GPUs", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", clip_on=False)
    _label_row(ax, wl_spans, depth=1, fontsize=9, base=-0.09, step=0.11)
    _label_row(ax, model_spans, depth=2, bold=True, fontsize=10, base=-0.09, step=0.11)
    _dividers(ax, wl_dividers, label_depth=1, linewidth=0.7, base=-0.09, step=0.11)
    _dividers(ax, model_dividers, label_depth=2, linewidth=1.3, base=-0.09, step=0.11)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FIG7_COLORS[k], edgecolor=FIG7_EDGE, linewidth=0.4)
               for k in series]
    labels = [f"{m} ({mode})" for (m, mode) in series]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig. 7 replication: Required PEC under different operating scenarios", y=1.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def generate_figures(data):
    """Render Figures 3-7 (Son et al., "Exploring High-Bandwidth Flash for Modern
    LLM Inference", IEEE CAL 2026) as PNGs in the project root, from data already
    computed by main()'s sweep (and persisted to experiment_data.json -- see
    --figures-only for standalone re-rendering without re-running the sweep).

    Each figure matches the paper's own layout: a single combined panel (not
    one subplot per model) with a nested categorical x-axis (Model > Workload
    > ... > leaf category) and colors sampled directly from the paper's own
    figures (400 DPI PDF render, legend-swatch pixel sampling) -- see the
    GPU_COLORS/FIG_MEM_STYLE/FIG5_COLORS/FIG6_BATCH_COLORS/FIG7_COLORS tables
    above. Remaining fidelity gaps: exact font and fine-grained spacing.
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
    #
    # Parallelized across a process pool: each (model, workload) baseline cell is fully
    # independent (find_max_batch_size depends only on its own scalar arguments -- no
    # shared mutable state read/written between cells other than this dict, which each
    # worker doesn't touch at all). This must fully complete (pool `with`-block exit joins
    # every submitted future) before the main sweep starts, since the main sweep reads
    # `baselines`. Provably behavior-identical to a sequential loop: _run_cell computes
    # exactly what a direct find_max_batch_size call would; all post-processing (tps/
    # max_b_per_gpu/dict construction/printing) happens here in the main process,
    # reading each cell's result from a future instead of a direct call.
    baselines = {model: {} for model in models}
    print("--- GATHERING BASELINES (8 GPU HBM4) ---")
    baseline_specs = [
        {"model": model, "wl_name": wl_name, "mem": "HBM4", "gpu": 8,
         "in_len": in_len, "out_len": out_len, "slo": 0.1}
        for model in models
        for wl_name, (in_len, out_len) in workloads.items()
    ]
    with ProcessPoolExecutor(max_workers=min(SWEEP_WORKERS, len(baseline_specs))) as pool:
        fut_to_spec = {pool.submit(_run_cell, spec): spec for spec in baseline_specs}
        for fut in as_completed(fut_to_spec):
            spec = fut_to_spec[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[baseline] worker crashed for {spec}: {e!r} -- recording as failed cell")
                r = _crashed_cell(spec)
            model, wl_name = r["model"], r["wl_name"]
            max_b, tpot, dp = r["max_b"], r["tpot"], r["dp"]
            # Per-GPU metrics divide by the TOTAL GPU count, not dp -- matches
            # this project's TPS definition ("... / Number of GPUs") and
            # the paper's cross-GPU-count normalization (a DP replica still consumes
            # real GPU hardware; dividing by dp instead of GPU count penalizes a
            # config for using DP *more* effectively). See CHANGES.md item 15.
            tps = max_b / (tpot * 8) if tpot > 0 else 0.0
            max_b_per_gpu = max_b / 8
            baselines[model][wl_name] = {
                "max_batch": max_b,
                "max_batch_per_gpu": max_b_per_gpu,
                "tps": tps,
                "tpot": tpot,
                "csv_file": r["csv_path"],
                "pec_kv_bytes": r["pec_kv"],
                "pec_capacity": r["pec_cap"],
                "bound_reason": r["bound_reason"],
            }
            print(f"Baseline {model} {wl_name}: Max Batch/GPU = {max_b_per_gpu:.1f} (total {max_b}, dp={dp}), TPS/GPU = {tps:.2f}")
    # Pool joined (all baseline futures resolved) -- safe to read `baselines` below.
    _checkpoint("baselines", baselines)

    results = []

    # Main sweep, same parallelization pattern. Canonical (model, workload, mem, gpu)
    # iteration order is built up front as `sweep_specs`; HBM4/8-GPU cells are pulled from
    # the baseline cache (never resubmitted to the pool -- avoids recomputing what
    # gather-baselines already computed). The remaining cells are submitted to the pool;
    # a future's completion (out-of-order, printed as it happens) is separate from
    # `results` assembly, which iterates `sweep_specs` in canonical order afterward -- so
    # `results`' order and content are identical to a sequential run regardless of which
    # cell's subprocess happened to finish first.
    print("\n--- RUNNING ALL SWEEPS ---")
    sweep_specs = [
        {"model": model, "wl_name": wl_name, "in_len": in_len, "out_len": out_len,
         "mem": mem, "gpu": gpu, "slo": 0.1}
        for model in models
        for wl_name, (in_len, out_len) in workloads.items()
        for mem in mem_types
        for gpu in gpus
    ]

    def _spec_key(s):
        return (s["model"], s["wl_name"], s["mem"], s["gpu"])

    to_submit = [s for s in sweep_specs if not (s["mem"] == "HBM4" and s["gpu"] == 8)]
    cell_results = {}  # spec key -> _run_cell's returned dict
    with ProcessPoolExecutor(max_workers=min(SWEEP_WORKERS, max(1, len(to_submit)))) as pool:
        fut_to_spec = {pool.submit(_run_cell, spec): spec for spec in to_submit}
        for fut in as_completed(fut_to_spec):
            spec = fut_to_spec[fut]
            key = _spec_key(spec)
            try:
                r = fut.result()
            except Exception as e:
                print(f"[sweep] worker crashed for {spec}: {e!r} -- recording as failed cell")
                r = _crashed_cell(spec)
            cell_results[key] = r
            print(f"Completed {r['model']} | {r['wl_name']} | {r['mem']} | {r['gpu']} GPUs "
                  f"-> Max Batch/GPU: {(r['max_b'] / r['gpu']):.1f}, TPOT: {r['tpot']:.4f}s")
    # Pool joined -- assemble `results` in canonical order (identical to sequential output).

    for spec in sweep_specs:
        model, wl_name, mem, gpu = spec["model"], spec["wl_name"], spec["mem"], spec["gpu"]
        in_len, out_len = spec["in_len"], spec["out_len"]

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
            r = cell_results[_spec_key(spec)]
            max_b, tpot, csv_path, last_pec_kv, last_pec_cap, bound_reason = (
                r["max_b"], r["tpot"], r["csv_path"], r["pec_kv"], r["pec_cap"], r["bound_reason"])
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
        print(f"{model} | {wl_name} | {mem} | {gpu} GPUs -> Max Batch/GPU: {max_b_per_gpu:.1f} "
              f"(Norm: {norm_batch:.2f}), TPS/GPU: {tps:.2f} (Norm: {norm_tps:.2f})")

    _checkpoint("results", results)

    # SLO Sensitivity Sweeps (LONG Workload only)
    print("\n--- RUNNING SLO SENSITIVITY SWEEPS (LONG Workload) ---")
    slo_results = []
    long_in, long_out = workloads["LONG"]
    slos = SLOS
    # P2: match the paper's Fig. 6 GPU-count scope. HBM4 can't even load either model
    # on 1-2 GPUs (paper states this explicitly), and this metric normalizes to HBM4,
    # so 1-2 GPU points would have an infeasible/undefined baseline.
    sensitivity_gpus = SENSITIVITY_GPUS

    # Gather sensitivity baselines (HBM4, 8 GPU) per SLO. slo=0.1/LONG was already
    # gathered as the main baseline; the remaining (model, slo!=0.1) cells are
    # independent -- same parallelization pattern as the main sweep above.
    sens_baseline_specs = [
        {"model": model, "slo": slo, "mem": "HBM4", "gpu": 8,
         "in_len": long_in, "out_len": long_out}
        for model in models for slo in slos if slo != 0.1
    ]
    sens_baseline_results = {}
    # Split offline (slo=86400, unconstrained-batch-search) specs into their own,
    # smaller pool -- see OFFLINE_SWEEP_WORKERS above; these can need 10x+ the host
    # RAM per process of a latency-bound (online) probe.
    sens_baseline_offline = [s for s in sens_baseline_specs if s["slo"] >= 1000]
    sens_baseline_normal = [s for s in sens_baseline_specs if s["slo"] < 1000]
    for group, workers, label in ((sens_baseline_normal, SWEEP_WORKERS, "sens-baseline"),
                                   (sens_baseline_offline, OFFLINE_SWEEP_WORKERS, "sens-baseline-offline")):
        if not group:
            continue
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(group)))) as pool:
            fut_to_spec = {pool.submit(_run_cell, spec): spec for spec in group}
            for fut in as_completed(fut_to_spec):
                spec = fut_to_spec[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"[{label}] worker crashed for {spec}: {e!r} -- recording as failed cell")
                    r = _crashed_cell(spec)
                sens_baseline_results[(spec["model"], spec["slo"])] = r

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
                r = sens_baseline_results[(model, slo)]
                max_b, tpot = r["max_b"], r["tpot"]
                max_b_per_gpu = max_b / 8
                tps = max_b / (tpot * 8) if tpot > 0 else 0.0
            sens_baselines[model][slo] = {
                "max_batch_per_gpu": max_b_per_gpu,
                "tps": tps,
                "tpot": tpot
            }
            print(f"SLO Sensitivity Baseline {model} SLO {slo}: Max Batch/GPU = {max_b_per_gpu:.1f}, TPS/GPU = {tps:.2f}")

    # Main sensitivity sweep, same parallelization pattern as the main sweep: canonical
    # spec list built up front, HBM4/8-GPU/slo=0.1 cache-pulled (never resubmitted), the
    # rest submitted to the pool, then assembled in canonical order afterward.
    sens_specs = [
        {"model": model, "mem": mem, "gpu": gpu, "slo": slo,
         "in_len": long_in, "out_len": long_out}
        for model in models for mem in mem_types for gpu in sensitivity_gpus for slo in slos
    ]

    def _sens_key(s):
        return (s["model"], s["mem"], s["gpu"], s["slo"])

    sens_to_submit = [s for s in sens_specs if not (s["mem"] == "HBM4" and s["gpu"] == 8 and s["slo"] == 0.1)]
    sens_cell_results = {}
    # Same offline/normal pool split as the sensitivity baselines above.
    sens_offline = [s for s in sens_to_submit if s["slo"] >= 1000]
    sens_normal = [s for s in sens_to_submit if s["slo"] < 1000]
    for group, workers, label in ((sens_normal, SWEEP_WORKERS, "sensitivity"),
                                   (sens_offline, OFFLINE_SWEEP_WORKERS, "sensitivity-offline")):
        if not group:
            continue
        with ProcessPoolExecutor(max_workers=min(workers, max(1, len(group)))) as pool:
            fut_to_spec = {pool.submit(_run_cell, spec): spec for spec in group}
            for fut in as_completed(fut_to_spec):
                spec = fut_to_spec[fut]
                key = _sens_key(spec)
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"[{label}] worker crashed for {spec}: {e!r} -- recording as failed cell")
                    r = _crashed_cell(spec)
                sens_cell_results[key] = r
                print(f"Completed Sensitivity Sweep: {r['model']} | {r['mem']} | {r['gpu']} GPUs | SLO {r['slo']}s "
                      f"-> Max Batch/GPU: {(r['max_b'] / r['gpu']):.1f}")

    for spec in sens_specs:
        model, mem, gpu, slo = spec["model"], spec["mem"], spec["gpu"], spec["slo"]
        # HBM4/8-GPU/slo=0.1 is the main LONG baseline — reuse it.
        if mem == "HBM4" and gpu == 8 and slo == 0.1:
            bl = baselines[model]["LONG"]
            max_b_per_gpu = bl["max_batch_per_gpu"]
            tps = bl["tps"]
        else:
            r = sens_cell_results[_sens_key(spec)]
            max_b, tpot = r["max_b"], r["tpot"]
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
            print(f"{model} | {mem} | {gpu} GPUs | SLO {slo}s -> Max Batch/GPU: {max_b_per_gpu:.1f} "
                  f"(Norm: {norm_batch:.2f}), TPS/GPU: {tps:.2f} (Norm: {norm_tps:.2f})")

    _checkpoint("slo_results", slo_results)

    # Offline-PEC data gathering (Fig. 7's "offline" bars). Online PEC for the
    # same scope is already derivable from `results` (0.1s SLO), but offline
    # (SLO=86400, i.e. no rate limit) isn't probed anywhere else -- the SLO
    # sensitivity sweep above only covers the LONG workload, and Fig. 7 needs
    # {SHORT,MID,LONG} x gpu{1,8,16} x {HBF,HBF+}. Fully independent cells, no
    # cache-hit branch needed -- simplest of the three parallelized loops.
    print("\n--- GATHERING OFFLINE PEC DATA (Fig. 7) ---")
    pec_specs = [
        {"model": model, "wl_name": wl_name, "in_len": workloads[wl_name][0],
         "out_len": workloads[wl_name][1], "gpu": gpu, "mem": mem, "slo": 86400.0}
        for model in models
        for wl_name in PEC_WORKLOADS
        for gpu in PEC_GPUS
        for mem in PEC_MEM_TYPES
    ]
    pec_cell_results = {}
    # All PEC specs are offline (slo=86400, unconstrained-batch-search) -- see
    # OFFLINE_SWEEP_WORKERS above. This phase is where the host OOM-killer storm
    # actually occurred (individual ./run RSS of 13-23GB, SWEEP_WORKERS-wide
    # concurrency oversubscribing this box's RAM for hours).
    with ProcessPoolExecutor(max_workers=min(OFFLINE_SWEEP_WORKERS, max(1, len(pec_specs)))) as pool:
        fut_to_spec = {pool.submit(_run_cell, spec): spec for spec in pec_specs}
        for fut in as_completed(fut_to_spec):
            spec = fut_to_spec[fut]
            key = (spec["model"], spec["wl_name"], spec["gpu"], spec["mem"])
            try:
                r = fut.result()
            except Exception as e:
                print(f"[pec] worker crashed for {spec}: {e!r} -- recording as failed cell")
                r = _crashed_cell(spec)
            pec_cell_results[key] = r
            print(f"Completed Offline PEC: {r['model']} | {r['wl_name']} | {r['mem']} | {r['gpu']} GPUs "
                  f"-> Max Batch: {r['max_b']}")

    pec_results = []
    for spec in pec_specs:
        key = (spec["model"], spec["wl_name"], spec["gpu"], spec["mem"])
        r = pec_cell_results[key]
        row = {
            "model": spec["model"],
            "workload": spec["wl_name"],
            "memory": spec["mem"],
            "gpus": spec["gpu"],
            "max_batch": r["max_b"],
            "tpot": r["tpot"],
            "pec_kv_bytes": r["pec_kv"],
            "pec_capacity": r["pec_cap"],
        }
        pec_results.append(row)
        pec_info = compute_pec(row)
        pec_str = f"{pec_info['pec']:.1f}" if pec_info else "N/A"
        print(f"{spec['model']} | {spec['wl_name']} | {spec['mem']} | {spec['gpu']} GPUs "
              f"-> Offline Max Batch: {r['max_b']}, 3yr PEC: {pec_str}")

    _checkpoint("pec_results", pec_results)

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
        # Matches the paper's Fig. 5 GPU-count scope {4, 8, 16}, since the paper uses
        # multiple GPU counts to show how the breakdown shifts (e.g. communication/
        # attention fraction) as GPU count increases.
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

        # 4. SLO Sensitivity Analysis (Figure 6) — matches the paper's GPU-count
        # scope {4, 8, 16}; GPU counts outside this range include ones where the
        # HBM4 normalization baseline is infeasible.
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
