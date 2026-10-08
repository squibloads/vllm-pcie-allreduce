#!/usr/bin/env python3
"""Teacher forcing, step 2: score the recorded sequences on one engine.

Each request is prompt + completion as ONE token-ID prompt with prompt_logprobs=20 and max_tokens=1, so the
engine reports, at every position, its own top-20 next-token logprobs under the SAME prefix (teacher forcing)
instead of following its own argmax. Two engines that disagree therefore disagree about one position at a
time, and a single early flip cannot drag the rest of the sequence away. Run it once per engine, including
the reference engine, and compare the files with tf_compare.py.

What this path exercises: prefill. The engine computes all positions in its normal prefill chunks, where an
all-reduce is megabytes and goes to NCCL; a decode-only custom all-reduce (CUDA-graph captured, a few KiB)
is not on this path. vLLM ignores the prefix cache for prompt_logprobs requests, so every request is a cold
prefill. One request at a time. With --metrics the engine's own counters are read around each request, so
traffic from someone else sharing the engine shows up in the record.

Output: gzip JSON; per position the target token's logprob and rank and the top-20 IDs and logprobs.

Stdlib only.
  tf_run.py --base-url http://HOST:8000 --model NAME --inputs inputs.json --out run.json.gz \
      [--metrics http://HOST:8000/metrics] [--n N]
"""
import argparse
import gzip
import json
import time
import urllib.error
import urllib.request

TOP = 20
COUNTERS = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_success_total",
            "vllm:prompt_tokens_total")


def post(base: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {e.read().decode(errors='replace')[:500]}") from None


def counters(url: str | None) -> dict[str, float]:
    """Running/waiting requests and finished-request totals from the engine's /metrics."""
    if not url:
        return {}
    out: dict[str, float] = {}
    with urllib.request.urlopen(url, timeout=10) as r:
        for line in r.read().decode().splitlines():
            for key in COUNTERS:
                if line.startswith(key + "{") or line.startswith(key + " "):
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def entries(pos: dict, target: int) -> dict:
    """One position: the target's logprob and rank, and the top-20 by rank (the target may be a 21st entry)."""
    tv = pos.get(str(target))
    if tv is None:
        raise SystemExit(f"the server's prompt_logprobs omit the scored token {target}")
    items = sorted(((int(t), v) for t, v in pos.items()), key=lambda kv: kv[1]["rank"])
    top = [(t, v["logprob"]) for t, v in items if v["rank"] <= TOP][:TOP]
    return {"lp": tv["logprob"], "rank": tv["rank"], "ids": [t for t, _ in top], "lps": [x for _, x in top]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Teacher-forced top-20 logprobs for recorded token sequences.")
    ap.add_argument("--base-url", required=True, help="server root, e.g. http://localhost:8000")
    ap.add_argument("--model", required=True, help="model name as the server serves it")
    ap.add_argument("--inputs", required=True, help="inputs JSON from tf_prep.py")
    ap.add_argument("--out", required=True, help="gzip JSON to write")
    ap.add_argument("--metrics", help="the engine's /metrics URL, to record other traffic around each request")
    ap.add_argument("--n", type=int, help="score only the first N sequences")
    args = ap.parse_args(argv)

    with open(args.inputs, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"][:args.n] if args.n else data["rows"]
    res = []
    for r in rows:
        seq = r["prompt_ids"] + r["completion_ids"]
        before = counters(args.metrics)
        t0 = time.monotonic()
        resp = post(args.base_url, {"model": args.model, "prompt": seq, "max_tokens": 1, "temperature": 0.0,
                                    "seed": 0, "prompt_logprobs": TOP})
        wall = time.monotonic() - t0
        after = counters(args.metrics)
        plp = resp["choices"][0]["prompt_logprobs"]
        if len(plp) != len(seq) or plp[0] is not None:
            raise SystemExit(f"#{r['idx']}: {len(plp)} prompt_logprobs entries for {len(seq)} tokens")
        pos = [entries(plp[j], seq[j]) for j in range(1, len(seq))]
        others = None
        if before and after:
            others = {"running_before": before.get("vllm:num_requests_running"),
                      "running_after": after.get("vllm:num_requests_running"),
                      "finished_delta": after.get("vllm:request_success_total", 0)
                      - before.get("vllm:request_success_total", 0),
                      "prompt_tokens_delta": after.get("vllm:prompt_tokens_total", 0)
                      - before.get("vllm:prompt_tokens_total", 0)}
        res.append({"idx": r["idx"], "n_prompt": len(r["prompt_ids"]), "n_completion": len(r["completion_ids"]),
                    "usage": resp.get("usage"), "wall_s": round(wall, 3), "metrics": others, "positions": pos})
        print(f"#{r['idx']:2d} {len(seq)} tokens scored in {wall:.1f}s; usage {resp.get('usage')}; "
              f"others {others}", flush=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump({"model": args.model, "ids_sha256": data["ids_sha256"], "top": TOP, "rows": res}, f)
    print(f"tf_run: {len(res)} sequences, {sum(len(x['positions']) for x in res)} positions -> {args.out}")


if __name__ == "__main__":
    main()
