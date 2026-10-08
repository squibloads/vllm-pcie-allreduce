"""Tests for verify/teacher_forcing that need neither a GPU nor a model.

tf_compare is checked on small synthetic runs whose answers can be worked out by hand: identical runs give
zeros, a perturbed run gives the perturbation, and the KL formula matches a hand computation. tf_prep and
tf_run are checked end to end against a toy HTTP server that speaks the three vLLM endpoints they use, so
the request shapes and the stored schema are exercised once, from prompts to a clean comparison.

Run from the repo root: python3 -m unittest discover -s tests -v   (or: python -m pytest tests -q)
"""

import contextlib
import gzip
import io
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TF_DIR = Path(__file__).resolve().parents[1] / "verify" / "teacher_forcing"
sys.path.insert(0, str(TF_DIR))

import tf_compare  # noqa: E402
import tf_prep  # noqa: E402
import tf_run  # noqa: E402

SHA = "0" * 64


def position(target: int, lp: float, others: list[tuple[int, float]]) -> dict:
    """One scored position whose argmax is `target`, in tf_run's stored form."""
    items = [(target, lp)] + others
    return {"lp": lp, "rank": 1, "ids": [t for t, _ in items], "lps": [v for _, v in items]}


def synthetic_run(sha: str = SHA) -> dict:
    """Two sequences of 3 prompt + 4 completion tokens; positions[j-1] scores sequence token j."""
    rows = []
    for idx in range(2):
        positions = [position(10 + idx, math.log(0.6), [(20, math.log(0.3)), (30, math.log(0.05))])
                     for _ in range(3 + 4 - 1)]
        rows.append({"idx": idx, "n_prompt": 3, "n_completion": 4, "positions": positions})
    return {"ids_sha256": sha, "top": 20, "rows": rows}


def write_gz(path: Path, run: dict) -> str:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(run, f)
    return str(path)


class CompareTest(unittest.TestCase):
    def test_identical_runs_are_bit_identical(self) -> None:
        res = tf_compare.compare(synthetic_run(), synthetic_run())
        self.assertEqual(res["completion"]["positions"], 8)
        self.assertEqual(res["prompt"]["positions"], 4)
        for s in res.values():
            self.assertEqual(s["dlogprob"]["max"], 0.0)
            self.assertEqual(s["kl20"]["max"], 0.0)
            self.assertEqual(s["top1_disagreements"], 0)
        self.assertTrue(tf_compare.is_identical(res))

    def test_perturbation_is_reported_where_it_is(self) -> None:
        cand = synthetic_run()
        # row 1, first completion position (index n_prompt - 1 = 2): target logprob moves by 0.25
        cand["rows"][1]["positions"][2]["lp"] += 0.25
        # row 0, second completion position: the candidate's argmax is another token
        flip = cand["rows"][0]["positions"][3]
        flip["ids"] = [20, 11, 30]
        flip["lps"] = [math.log(0.6), math.log(0.3), math.log(0.05)]
        # row 1, second completion position: same top-20 IDs, one logprob moved
        cand["rows"][1]["positions"][3]["lps"][1] += 0.01
        res = tf_compare.compare(synthetic_run(), cand)["completion"]
        self.assertAlmostEqual(res["dlogprob"]["max"], 0.25)
        self.assertEqual(res["top1_disagreements"], 1)
        self.assertEqual(res["bit_identical_target_logprob"], 7)
        self.assertEqual(res["identical_top20"], 6)
        self.assertEqual(res["kl_max_at"], {"row": 0, "position": 1})
        self.assertFalse(tf_compare.is_identical({"completion": res}))
        self.assertEqual(tf_compare.compare(synthetic_run(), cand)["prompt"]["dlogprob"]["max"], 0.0)

    def test_kl_matches_a_hand_computation(self) -> None:
        ref = position(1, math.log(0.6), [(2, math.log(0.3))])
        cand = position(1, math.log(0.5), [(2, math.log(0.4))])
        kl, subs, clamped = tf_compare.kl20(ref, cand)
        # the rest bucket is 0.1 on both sides and contributes nothing
        want = 0.6 * math.log(0.6 / 0.5) + 0.3 * math.log(0.3 / 0.4)
        self.assertAlmostEqual(kl, want)
        self.assertEqual((subs, clamped), (0, False))

    def test_missing_top20_token_takes_the_candidates_floor(self) -> None:
        ref = position(1, math.log(0.5), [(2, math.log(0.4))])
        cand = position(1, math.log(0.5), [(3, math.log(0.4))])
        _, subs, _ = tf_compare.kl20(ref, cand)
        self.assertEqual(subs, 1)

    def test_different_token_sequences_are_refused(self) -> None:
        with self.assertRaises(SystemExit):
            tf_compare.compare(synthetic_run(), synthetic_run(sha="1" * 64))

    def test_cli_on_two_gz_files(self) -> None:
        cand = synthetic_run()
        cand["rows"][0]["positions"][4]["lp"] -= 0.5
        with tempfile.TemporaryDirectory() as tmp:
            ref_p = write_gz(Path(tmp) / "ref.json.gz", synthetic_run())
            same_p = write_gz(Path(tmp) / "same.json.gz", synthetic_run())
            diff_p = write_gz(Path(tmp) / "diff.json.gz", cand)
            out_json = Path(tmp) / "out.json"
            script = str(TF_DIR / "tf_compare.py")
            ok = subprocess.run([sys.executable, script, ref_p, f"same={same_p}", "--require-identical"],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("same: bit-identical to the reference on all 12 positions", ok.stdout)
            bad = subprocess.run([sys.executable, script, ref_p, f"same={same_p}", f"diff={diff_p}",
                                  "--json", str(out_json), "--require-identical"], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("diff: differs from the reference", bad.stdout)
            stats = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(stats["candidates"]["diff"]["completion"]["dlogprob"]["max"], 0.5)
            self.assertEqual(stats["candidates"]["same"]["completion"]["dlogprob"]["max"], 0.0)


class ToyServer(BaseHTTPRequestHandler):
    """/tokenize, /v1/completions (greedy with token_id strings, and prompt_logprobs scoring), /metrics."""

    def log_message(self, *args: object) -> None:
        pass

    def reply(self, obj: dict) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        raw = b"vllm:num_requests_running{model=\"toy\"} 0\nvllm:request_success_total{model=\"toy\"} 3\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            self.reply({"tokens": [ord(c) % 100 for c in body["prompt"]]})
        elif "prompt_logprobs" in body:
            seq = body["prompt"]
            plp = [None] + [{str(t): {"logprob": -0.1, "rank": 1}, str(t + 1): {"logprob": -2.5, "rank": 2},
                             str(t + 2): {"logprob": -4.0, "rank": 3}} for t in seq[1:]]
            self.reply({"choices": [{"text": "x", "prompt_logprobs": plp}], "usage": {"prompt_tokens": len(seq)}})
        else:
            seq = body["prompt"]
            out = [(seq[-1] + 1 + i) % 100 for i in range(body["max_tokens"])]
            self.reply({"choices": [{"finish_reason": "length", "logprobs": {
                "tokens": [f"token_id:{t}" for t in out]}}],
                "usage": {"prompt_tokens": len(seq), "completion_tokens": len(out)}})


@contextlib.contextmanager
def toy_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ToyServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


class PipelineTest(unittest.TestCase):
    def test_prep_run_compare_end_to_end(self) -> None:
        with toy_server() as base, tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            prompts = tmp_p / "prompts.json"
            prompts.write_text(json.dumps(["hello there", "a second prompt"]), encoding="utf-8")
            inputs = tmp_p / "inputs.json"
            with contextlib.redirect_stdout(io.StringIO()):
                tf_prep.main(["--base-url", base, "--model", "toy", "--prompts", str(prompts),
                              "--out", str(inputs), "--max-tokens", "5"])
                for name in ("ref", "cand"):
                    tf_run.main(["--base-url", base, "--model", "toy", "--inputs", str(inputs),
                                 "--out", str(tmp_p / f"{name}.json.gz"), "--metrics", base + "/metrics"])
            data = json.loads(inputs.read_text(encoding="utf-8"))
            self.assertEqual([len(r["completion_ids"]) for r in data["rows"]], [5, 5])
            self.assertEqual(len(data["rows"][0]["prompt_ids"]), len("hello there"))
            run = tf_compare.load(str(tmp_p / "ref.json.gz"))
            self.assertEqual(len(run["rows"][0]["positions"]), 11 + 5 - 1)
            self.assertEqual(run["rows"][0]["positions"][0]["ids"][0], data["rows"][0]["prompt_ids"][1])
            self.assertEqual(run["rows"][0]["metrics"]["finished_delta"], 0)
            res = tf_compare.compare(run, tf_compare.load(str(tmp_p / "cand.json.gz")))
            self.assertTrue(tf_compare.is_identical(res))
            self.assertEqual(res["completion"]["positions"], 10)

    def test_plain_text_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prompts.txt"
            p.write_text("first\n\nsecond one\n", encoding="utf-8")
            self.assertEqual(tf_prep.read_prompts(str(p)), ["first", "second one"])

    def test_completion_ids_must_come_back_as_token_ids(self) -> None:
        saved = tf_prep.post
        try:
            def fake(base: str, path: str, body: dict, timeout: float = 0) -> dict:
                if path == "/tokenize":
                    return {"tokens": [1, 2]}
                return {"choices": [{"logprobs": {"tokens": ["hello"]}}]}
            tf_prep.post = fake
            with self.assertRaisesRegex(SystemExit, re.escape("token_id:N")):
                tf_prep.record("http://x", "m", "p", 4, False)
        finally:
            tf_prep.post = saved


if __name__ == "__main__":
    unittest.main()
