#!/usr/bin/env python3
"""Teacher forcing, step 3: how far a candidate engine's next-token distribution is from the reference's,
position by position, on identical prefixes (tf_run.py output for both).

Per position, with R = the reference run and X = the candidate:
  |dlogprob|  |log p_R(t) - log p_X(t)| for the sequence's own next token t (both runs always return it)
  KL20        KL(P_R || P_X) in nats over R's top-20 tokens plus one bucket for the rest of the mass: a
              coarsening of the full distributions, so a lower bound on the full-vocabulary KL. An R top-20
              token missing from X's top-20 takes X's 20th logprob, an upper bound on its true value; such
              substitutions are counted.
  top-1       R's and X's argmax agree
Positions are split into the teacher-forced completion tokens (the reference's own output) and the prompt
tokens before them (the same measurement on more positions).

"Bit-identical" means every position has the same target logprob and the same top-20 IDs and logprobs, which
is what two runs of one engine configuration give. --require-identical turns that into the exit status.

Stdlib only.
  tf_compare.py REF.json.gz [LABEL=]CAND.json.gz ... [--json OUT.json] [--require-identical]
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path


def load(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def q(xs: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def kl20(a: dict, x: dict) -> tuple[float, int, bool]:
    """KL(P_a || P_x) on a's top-20 plus a rest bucket; returns (nats, substituted tokens, rest clamped)."""
    xmap = dict(zip(x["ids"], x["lps"]))
    floor = min(x["lps"])
    subs, kl, sa, sx = 0, 0.0, 0.0, 0.0
    for t, la in zip(a["ids"], a["lps"]):
        lx = xmap.get(t)
        if lx is None:
            subs += 1
            lx = floor
        pa, px = math.exp(la), math.exp(lx)
        kl += pa * (la - lx)
        sa += pa
        sx += px
    ra, rx = 1.0 - sa, 1.0 - sx
    clamped = False
    if ra > 1e-9:
        if rx < 1e-12:
            rx, clamped = 1e-12, True
        kl += ra * math.log(ra / rx)
    return kl, subs, clamped


def summarize(vals: dict) -> dict:
    d, k, t = vals["dlp"], vals["kl"], vals["top1"]
    return {"positions": len(d),
            "dlogprob": {"mean": sum(d) / len(d), "p95": q(d, 0.95), "p99": q(d, 0.99), "max": max(d)},
            "kl20": {"mean": sum(k) / len(k), "p95": q(k, 0.95), "p99": q(k, 0.99), "max": max(k)},
            "top1_agreement": sum(t) / len(t), "top1_disagreements": len(t) - sum(t),
            "bit_identical_target_logprob": vals["same"], "identical_top20": vals["same20"],
            "missing_top20_substitutions": vals["subs"], "rest_bucket_clamped": vals["clamped"],
            "kl_max_at": vals["kl_max_at"]}


def compare(ref: dict, cand: dict) -> dict:
    """Per-region statistics; the two runs must have scored the same token sequences."""
    if ref["ids_sha256"] != cand["ids_sha256"]:
        raise SystemExit("reference and candidate scored different token sequences")
    if len(ref["rows"]) != len(cand["rows"]):
        raise SystemExit("reference and candidate scored different numbers of sequences")
    out: dict[str, dict] = {}
    for region in ("completion", "prompt"):
        vals = {"dlp": [], "kl": [], "top1": [], "same": 0, "same20": 0, "subs": 0, "clamped": 0,
                "kl_max_at": None}
        best = -1.0
        for ra, rx in zip(ref["rows"], cand["rows"]):
            if ra["idx"] != rx["idx"] or len(ra["positions"]) != len(rx["positions"]):
                raise SystemExit(f"row {ra['idx']}: runs do not line up")
            n0 = ra["n_prompt"] - 1  # positions[j-1] scores sequence token j; the completion starts at n_prompt
            sl = slice(n0, n0 + ra["n_completion"]) if region == "completion" else slice(0, n0)
            for j, (a, x) in enumerate(zip(ra["positions"][sl], rx["positions"][sl]), start=sl.start):
                k, subs, clamped = kl20(a, x)
                vals["dlp"].append(abs(a["lp"] - x["lp"]))
                vals["kl"].append(k)
                vals["top1"].append(a["ids"][0] == x["ids"][0])
                vals["same"] += a["lp"] == x["lp"]
                vals["same20"] += a["ids"] == x["ids"] and a["lps"] == x["lps"]
                vals["subs"] += subs
                vals["clamped"] += clamped
                if k > best:
                    best = k
                    vals["kl_max_at"] = {"row": ra["idx"], "position": j + 1 - ra["n_prompt"]
                                         if region == "completion" else j + 1}
        if vals["dlp"]:
            out[region] = summarize(vals)
    return out


def is_identical(result: dict) -> bool:
    return all(s["bit_identical_target_logprob"] == s["positions"] == s["identical_top20"]
               for s in result.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare teacher-forced runs position by position.")
    ap.add_argument("reference", help="tf_run.py output of the reference engine")
    ap.add_argument("candidates", nargs="+", help="tf_run.py output to compare, optionally LABEL=PATH")
    ap.add_argument("--json", dest="out_json", help="also write the statistics here")
    ap.add_argument("--require-identical", action="store_true",
                    help="exit 1 unless every candidate is bit-identical to the reference")
    args = ap.parse_args(argv)

    ref = load(args.reference)
    result: dict = {"reference": args.reference, "candidates": {}}
    all_identical = True
    print(f"reference {args.reference}; KL20 = KL(P_ref || P_cand), nats, ref top-20 + rest bucket")
    print("label | region | positions | |dlogprob| mean / p95 / max | KL20 mean / p95 / max | top-1 agree | "
          "bit-identical target lp | identical top-20")
    for spec in args.candidates:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).name.split(".")[0], spec
        r = compare(ref, load(path))
        result["candidates"][label] = {"path": path, **r}
        for region, s in r.items():
            d, k = s["dlogprob"], s["kl20"]
            print(f"{label} | {region} | {s['positions']} | {d['mean']:.3e} / {d['p95']:.3e} / {d['max']:.3e} | "
                  f"{k['mean']:.3e} / {k['p95']:.3e} / {k['max']:.3e} | {100 * s['top1_agreement']:.2f}% "
                  f"({s['top1_disagreements']} differ) | {s['bit_identical_target_logprob']} | "
                  f"{s['identical_top20']}")
        same = is_identical(r)
        all_identical &= same
        total = sum(s["positions"] for s in r.values())
        print(f"{label}: {'bit-identical to the reference on all ' + str(total) + ' positions' if same else 'differs from the reference'}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0 if all_identical or not args.require_identical else 1


if __name__ == "__main__":
    sys.exit(main())
