# Changelog

## 0.2.0 - 2026-10-08

Quality evidence for the patch on 1Cat-vLLM c4f6245f8, and the tool behind it.

- `verify/teacher_forcing/`: `tf_prep.py`, `tf_run.py` and `tf_compare.py` record a reference run as token
  IDs, re-score the same IDs on any vLLM engine with `prompt_logprobs: 20`, and compare per position
  (abs(delta logprob), KL over the top 20, top-1 agreement, bit-identity). Stdlib only.
  `tests/test_teacher_forcing.py` covers it without a GPU, including an end-to-end run against a toy server.
- RESULTS.md section 7, "Quality gate (2026-10-08)": the 2+2 group with the patch is bit-identical to the
  same group on native NCCL over PCIe on all 70,273 teacher-forced positions; both differ from four NVLink
  GPUs by the same prefill shift (abs(delta logprob) mean 1.12e-2, KL mean 1.02e-3, top-1 98.93%);
  1Cat's three-seed quality suite passes 108/108 on the NVLink reference, with the patch, and on native NCCL.
- Serving: `VLLM_SM70_TOP1_CUSTOM_AR=0` is recommended on c4f6245f8 and later. Its top-1 kernel is a pull
  kernel, which loses over PCIe; switching it off gave +0.8% decode at 6k and +1.0% at 160k with bit-identical
  greedy output. The README's serving table and Known limits say so, and RESULTS.md section 5 now
  points to the measurement.
- No change to `car_patch.py`, `patches/`, the gate, `ops/` or the example.

## 0.1.0 - 2026-10-08

First public release.

- `car_patch.py` writes the patched `custom_all_reduce.py` with three env switches: `VLLM_CAR_P2P_MESH`,
  `VLLM_CAR_MAX_BYTES`, `VLLM_CAR_GRAPH_ONLY`. Its output has the same Python AST as the files behind
  RESULTS.md; only the comments it inserts were reworded. `tests/test_car_patch.py` covers it without a GPU.
- `patches/`: the resulting diffs for 1Cat-vLLM 357d07bcb and c4f6245f8. Both apply with `git apply` and
  reproduce `car_patch.py`'s output byte for byte.
- `verify/`: the correctness gate (negative controls, 31 sizes from 16 B to 8 MiB, eager and CUDA-graph
  chains, a 1e5-collective soak under copy-engine load, bit-identity across GPU sets). Changes from the
  version behind RESULTS.md:
  - the bit-identity reference is a second GPU set or order (`REF_GPUS`, default the same GPUs reversed)
    instead of an all-NVLink board;
  - a case also fails when its container crashed or timed out, when it reported fewer sizes than requested,
    or when the soak's load generator did not run;
  - `CHECK_ONLY=1` re-checks existing logs;
  - `watch_gate.sh` follows this gate (it used to follow a different one).
- `ops/`: clears ACS P2P redirect at boot, with the switch ports configurable.
- `examples/docker-run.sh`: the serving launch reduced to what the all-reduce needs.
