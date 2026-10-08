#!/usr/bin/env python3
"""Teacher forcing, step 1: record a reference run as token IDs.

Why token IDs: the candidate engine has to score exactly the sequence the reference produced. Re-tokenizing
the reference's text does not always give the same sequence (a model can emit a non-canonical split), so the
IDs are taken from the server instead of from decoded strings.

For every prompt this tokenizes it with the server's own /tokenize, generates a greedy completion on the
REFERENCE engine, and keeps both as ID lists. Greedy and seed 0, one request at a time: concurrency changes
batch shapes and with them the numerics.

The prompts decide what the later comparison can see. Use prompts of a few thousand tokens: the prefill
all-reduces are then megabytes, as they are in serving, and the completion is long enough to matter.

Stdlib only. The server must be vLLM's OpenAI-compatible one (/tokenize, /v1/completions with
return_tokens_as_token_ids).

  tf_prep.py --base-url http://HOST:8000 --model NAME --prompts prompts.json --out inputs.json \
      [--max-tokens 192] [--ignore-eos]

--prompts is a JSON list of strings, or a text file with one prompt per non-empty line.
"""
import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

TOKEN_ID = re.compile(r"token_id:(\d+)")


def post(base: str, path: str, body: dict, timeout: float = 900) -> dict:
    """POST JSON and return the JSON reply; any HTTP error ends the run with the server's message."""
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path}: HTTP {e.code} {e.read().decode(errors='replace')[:500]}") from None


def read_prompts(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [line for line in raw.splitlines() if line.strip()]
    if not (isinstance(data, list) and data and all(isinstance(p, str) and p for p in data)):
        raise SystemExit(f"{path}: expected a JSON list of non-empty strings or one prompt per line")
    return data


def record(base: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool) -> dict:
    """One prompt: its token IDs and the reference engine's greedy completion IDs."""
    prompt_ids = post(base, "/tokenize", {"model": model, "prompt": prompt, "add_special_tokens": True})["tokens"]
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0,
            "logprobs": 1, "return_tokens_as_token_ids": True}
    if ignore_eos:
        body["ignore_eos"] = True
    resp = post(base, "/v1/completions", body)
    choice = resp["choices"][0]
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    matches = [TOKEN_ID.fullmatch(t) for t in tokens]
    if not tokens or not all(matches):
        raise SystemExit("the server did not return tokens as 'token_id:N' strings "
                         "(return_tokens_as_token_ids unsupported?)")
    completion_ids = [int(m.group(1)) for m in matches if m]
    usage = resp.get("usage") or {}
    if usage.get("completion_tokens") not in (None, len(completion_ids)):
        raise SystemExit(f"usage says {usage['completion_tokens']} completion tokens, got {len(completion_ids)} IDs")
    if usage.get("prompt_tokens") not in (None, len(prompt_ids)):
        raise SystemExit(f"/tokenize gave {len(prompt_ids)} IDs, the server counted {usage['prompt_tokens']}")
    return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "finish_reason": choice.get("finish_reason")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Record a reference run's prompt and greedy completion token IDs.")
    ap.add_argument("--base-url", required=True, help="server root, e.g. http://localhost:8000")
    ap.add_argument("--model", required=True, help="model name as the server serves it")
    ap.add_argument("--prompts", required=True, help="JSON list of prompt strings, or one prompt per line")
    ap.add_argument("--out", required=True, help="inputs JSON to write")
    ap.add_argument("--max-tokens", type=int, default=192, help="completion length (default 192)")
    ap.add_argument("--ignore-eos", action="store_true", help="always generate --max-tokens tokens")
    args = ap.parse_args(argv)

    rows = []
    for idx, prompt in enumerate(read_prompts(args.prompts)):
        row = {"idx": idx, **record(args.base_url, args.model, prompt, args.max_tokens, args.ignore_eos)}
        rows.append(row)
        print(f"#{idx:2d} prompt {len(row['prompt_ids'])} + completion {len(row['completion_ids'])} tokens "
              f"({row['finish_reason']})", flush=True)
    blob = json.dumps([[r["prompt_ids"], r["completion_ids"]] for r in rows]).encode()
    out = {"model": args.model, "max_tokens": args.max_tokens, "ids_sha256": hashlib.sha256(blob).hexdigest(),
           "rows": rows}
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"tf_prep: {len(rows)} sequences, {sum(len(r['completion_ids']) for r in rows)} completion tokens, "
          f"ids sha256 {out['ids_sha256'][:16]} -> {args.out}")


if __name__ == "__main__":
    main()
