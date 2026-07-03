import sys, json
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass")
import run_experiments as R

runs = json.load(open("/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass/exp1_out.json"))
COMPS = ["qkvgen","atten_sum","atten_gen","o_proj","ffn","expert_ffn",
         "communication","kv_write","layernorm","residual","rope","lm_head","latency"]
for r in runs:
    if not r.get("success"):
        print(f"\n== {r['tag']} FAILED =="); continue
    bd = R.parse_csv_breakdown(r["csv_file"])
    print(f"\n== {r['tag']} tp={r['tp']} batch={r['batch']} tpot={r['tpot']*1000:.3f}ms dp={r['dp']} ==")
    if bd is None:
        print("  no csv"); continue
    lat = bd.get("latency", 0) or sum(bd.get(c,0) for c in COMPS if c!="latency")
    for c in COMPS:
        v = bd.get(c,0)
        if c=="latency": continue
        print(f"  {c:14s} {v/1000:8.3f} us  {100*v/lat if lat else 0:5.1f}%")
    print(f"  {'latency(tpot)':14s} {bd.get('latency',0)/1000:8.3f} us")
