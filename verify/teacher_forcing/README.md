# Teacher-forced logprob comparison

`verify_car.sh` tells you the all-reduce is exact. This tells you what the whole model does with it: it
scores the same token sequence on two serving setups and compares their next-token distributions at every
position. Greedy text comparisons stop being informative at the first flipped token; this does not, because
both engines are always asked about the same prefix.

## Method

1. **Record a reference run** (`tf_prep.py`). Tokenize your prompts with the server's own `/tokenize`, let
   the reference engine generate a greedy completion for each, and keep prompt and completion as token IDs.
2. **Re-score the same token IDs on every engine** (`tf_run.py`), the reference included. Each sequence goes
   in as one prompt with `prompt_logprobs: 20` and `max_tokens: 1`, so the engine returns its own top-20
   logprobs at every position instead of following its own argmax.
3. **Compare per position** (`tf_compare.py`): `|Δlogprob|` of the sequence's own next token, KL over the
   reference's top-20 plus a rest bucket (a lower bound on the full-vocabulary KL), top-1 agreement, and
   how many positions are bit-identical. Positions are reported separately for the completion tokens
   (the reference's own output) and the prompt tokens before them.

The tools are stdlib-only and talk to vLLM's OpenAI-compatible server (`/tokenize`, `/v1/completions`).

## Commands

Prompts: a JSON list of strings, or a text file with one prompt per line. Use a few thousand tokens each, so
the prefill all-reduces are megabytes as they are in serving. We used 16 prompts of 2.5k-6k tokens and 192
completion tokens each: 70,273 scored positions, under a minute per engine.

```bash
BASE=http://localhost:8000     # the engine you are scoring on; MODEL is the name it serves
MODEL=your-model-name

# 1. serve the REFERENCE layout (a setup you already trust), then:
python3 tf_prep.py --base-url $BASE --model $MODEL --prompts prompts.json --out inputs.json \
    --max-tokens 192 --ignore-eos
python3 tf_run.py  --base-url $BASE --model $MODEL --inputs inputs.json --out ref.json.gz
python3 tf_run.py  --base-url $BASE --model $MODEL --inputs inputs.json --out ref-repeat.json.gz

# 2. stop it, serve each CANDIDATE setup in turn (same model, same flags), and for each:
python3 tf_run.py  --base-url $BASE --model $MODEL --inputs inputs.json --out cand.json.gz

# 3. compare
python3 tf_compare.py ref.json.gz repeat=ref-repeat.json.gz cand=cand.json.gz --json compare.json
```

`tf_run.py --n 1` scores only the first sequence, to check that the engine accepts a long
`prompt_logprobs` request before you commit to the whole set. `--metrics http://HOST:8000/metrics` records
the engine's own request counters around each request, so traffic from someone else shows up in the file.
`tf_compare.py --require-identical` exits 1 unless every candidate is bit-identical to the reference.

## Reading it

- **Run the reference twice and compare it with itself** (`repeat=` above). It must be zero everywhere. If
  not, the engine is not deterministic at one request at a time and every other number is noise.
- **Bit-identical** means every position has the same target logprob and the same top-20 IDs and logprobs.
  That is what the same engine configuration gives, and what a change that does not touch the prefill path
  must give.
- **A different reduction order is not zero, and not a bug.** A layout whose all-reduces add in another
  order or precision moves logprobs a little everywhere. Judge the size: the mean and p95 of `|Δlogprob|`
  and KL, and the top-1 flips, which in practice are near-ties.
- **What it covers.** Scoring a prompt with `prompt_logprobs` is a prefill, so it covers the all-reduces
  prefill uses and nothing that only runs in captured decode graphs. For a decode-only kernel, pair it with
  a free-running greedy comparison and with `verify_car.sh`.

Results for the PCIe custom all-reduce: [RESULTS.md](../../RESULTS.md), "Quality gate (2026-10-08)".
