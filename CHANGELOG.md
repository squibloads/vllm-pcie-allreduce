# Changelog

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
