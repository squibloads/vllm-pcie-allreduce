#!/usr/bin/env bash
# verify_car.sh - the correctness gate for 1Cat's custom all-reduce over PCIe P2P. Run it with the GPUs idle,
# on exactly the four GPUs you will serve on, BEFORE serving with VLLM_CAR_P2P_MESH=1. It exits 0 only if:
#   negctl          NEGATIVE CONTROL: C++ is told "not fully connected", so the kernels launch nothing.
#                   Every size must come back WRONG, or the checks below are blind.
#   perturb         NEGATIVE CONTROL: rank 1 adds 1.0 to one input element behind the reference's back.
#                   Exactly that element must be wrong, on all 4 ranks, in every collective.
#   car_int         custom AR is OK at all 31 sizes (16 B to 8 MiB) for exact-integer, random and
#   car_randn       special-value data (signed zeros, infinities, NaN, the push kernel's sentinel bits):
#   car_special     eager calls and CUDA-graph chains with rank skew, all ranks bitwise identical, a skipped
#                   replay caught as stale, captured results up to 80 KiB (the push kernel) bitwise equal to
#                   the FP32 rank-order reference;
#   car_load_soak   still OK through a >=1e5-collective soak at 2-80 KiB while car_load.py moves copy-engine
#                   traffic across the same links;
#   car_ref, digest the same seeds on REF_GPUS (default: the same GPUs in reverse order, so every rank uses
#                   other links) give bit-identical results;
#   nccl            native NCCL passes within its own rounding bound (the baseline; its bits differ by design).
# Every case must also run to completion: a crash, a timeout (CAP) or a missing load fails it.
#
# From the repo root:  IMG=your-1cat-image GPUS=0,1,2,3 verify/verify_car.sh
#   ONLY=<regex>  run and check only the matching cases (names above)
#   OUT=<dir>     per-case logs (default ./car-results). GPUS, REF_GPUS, LOAD_GPUS, CAP: see car_gate.sh.
#   CHECK_ONLY=1  skip the runs and re-check the logs already in OUT
# About 30 minutes on 4x V100. Follow it from another shell with verify/watch_gate.sh.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OUT=${OUT:-car-results}
# car_gate.py reads these from the environment; an inherited SIZES would silently shrink the gate
unset DATA SIZES EAGER_ITERS CHAIN CHAIN_REPLAYS SKEW SOAK SOAK_SIZES TIME_K TIME_REPLAYS PROFILE SET LOAD
EAGER=20 REPLAYS=20   # quick per-size depth; the soak goes deep
SOAK_SET=2048,5120,10240,20480,40960,81920
fail=0
want() { [[ -z "${ONLY:-}" || "$1" =~ $ONLY ]]; }
run_case() {  # run_case TAG MODE [VAR=value...]: one car_gate.sh run, then a one-line summary
  local tag=$1 mode=$2
  shift 2
  echo "$(date +%H:%M:%S) start $tag"
  env EAGER_ITERS=$EAGER CHAIN_REPLAYS=$REPLAYS "$@" "$HERE/car_gate.sh" "$mode" "$tag" >/dev/null
  echo "$(date +%H:%M:%S) done  $tag: $(grep -c '^{' "$OUT/car.$tag.log" 2>/dev/null) rows;" \
    "$(tail -1 "$OUT/car.$tag.log" 2>/dev/null)"
}

if [ "${CHECK_ONLY:-0}" != 1 ]; then
  : "${IMG:?set IMG to your 1Cat-vLLM image}"
  want negctl && run_case negctl negctl DATA=int
  want perturb && run_case perturb perturb DATA=randn
  for d in int randn special; do
    want "car_$d" && run_case "car_$d" car DATA=$d
  done
  want car_load_soak && run_case car_load_soak car DATA=randn LOAD=1 SOAK=100000 SIZES=$SOAK_SET
  want car_ref && run_case car_ref car DATA=randn SET=ref
  want nccl && run_case nccl nccl DATA=randn
fi

python3 - "${ONLY:-}" "$OUT" "$EAGER" "$SOAK_SET" <<'EOF' || fail=1
import json, os, re, sys

only, out, eager, soak_set = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
EXPECT = {t: 31 for t in ("negctl", "perturb", "car_int", "car_randn", "car_special", "car_ref", "nccl")}
EXPECT["car_load_soak"] = len(soak_set.split(","))
bad = False


def load(tag):
    """(rows, summary) from car_gate.sh's log; rows None when the case never ran."""
    path = os.path.join(out, f"car.{tag}.log")
    if not os.path.exists(path):
        return None, None
    rows, meta = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"unparsable": line[:80]})
        elif line.startswith("car_gate.sh: "):
            meta = dict(kv.split("=", 1) for kv in line[len("car_gate.sh: "):].rstrip("\n").split(" ", 2))
    return rows, meta


def check(name, ok, detail):
    global bad
    if only and not re.search(only, name):
        return
    print(f"{'pass' if ok else 'FAIL'}  {name:<14} {detail}")
    bad |= not ok


def ran(name, rows, meta):
    """True when the case ran to completion; otherwise reports why and returns False."""
    if only and not re.search(only, name):
        return False
    if rows is None:
        check(name, False, f"no log in {out}/ (case not run)")
        return False
    problems = []
    if meta is None:
        problems.append("no car_gate.sh summary line")
    else:
        if meta.get("exit") != "0":
            problems.append(f"container exit {meta.get('exit')}")
        if meta.get("timeout") != "0":
            problems.append("TIMEOUT")
    if any("unparsable" in x for x in rows):
        problems.append("unparsable result line")
    if len(rows) != EXPECT[name]:
        problems.append(f"{len(rows)} of {EXPECT[name]} sizes reported")
    if problems:
        check(name, False, "; ".join(problems) + f" (see {out}/car.{name}.log)")
        return False
    return True


rows, meta = load("negctl")
if ran("negctl", rows, meta):
    live = [x for x in rows if "skip" not in x]
    n = sum(x["verdict"] == "WRONG" for x in live)
    check("negctl", bool(live) and n == len(live), f"{n}/{len(live)} sizes WRONG (must be all)")
rows, meta = load("perturb")
if ran("perturb", rows, meta):
    live = [x for x in rows if "skip" not in x]
    exact = [x for x in live if x["eager_wrong"] == 4 * eager
             and x["graph_wrong"] == 4 * x["graph_collectives"] and x["rank_hash_equal"]]
    check("perturb", bool(live) and len(exact) == len(live),
          f"{len(exact)}/{len(live)} sizes wrong in exactly the perturbed element on all 4 ranks")
for tag in ("car_int", "car_randn", "car_special", "car_load_soak", "car_ref", "nccl"):
    rows, meta = load(tag)
    if not ran(tag, rows, meta):
        continue
    live = [x for x in rows if "skip" not in x]
    okn = sum(x["verdict"] == "OK" for x in live)
    soak = max((x["graph_collectives"] for x in live), default=0)
    pushbits = sum(x["graph_bitdiff"] for x in live if x["bytes"] <= 81920)  # captured: push, rank order
    detail = f"{okn}/{len(live)} sizes OK, graph collectives up to {soak}, graph bitdiff <=80 KiB {pushbits}"
    ok = bool(live) and okn == len(live)
    if tag.startswith("car") and tag != "car_special":
        ok = ok and pushbits == 0
    if tag == "car_load_soak":
        ok = ok and soak >= 100000 and meta.get("load") not in ("none", "MISSING")
        detail += f", load: {meta.get('load')}"
    check(tag, ok, detail)
(a, _), (b, _) = load("car_randn"), load("car_ref")
if a and b and not (only and not re.search(only, "digest")):
    da = {x["bytes"]: x.get("digest") for x in a if "skip" not in x}
    db = {x["bytes"]: x.get("digest") for x in b if "skip" not in x}
    same = [s for s in da if s in db and da[s] == db[s]]
    check("digest", len(same) == len(da) == len(db) > 0,
          f"{len(same)}/{len(da)} sizes bit-identical between GPUS and REF_GPUS")
sys.exit(1 if bad else 0)
EOF
exit $fail
