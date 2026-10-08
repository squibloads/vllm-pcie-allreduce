# vllm-pcie-allreduce

Run [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)'s SM70 push custom all-reduce at TP=4 over PCIe P2P, on
V100s without an NVLink mesh.

vLLM, and 1Cat-vLLM with it, enables its custom all-reduce for more than two GPUs only when NVML reports NVLink
between every pair. On a 4x V100 PCIe box every TP=4 all-reduce therefore goes through NCCL. 1Cat's SM70 push
kernel needs nothing NVLink-specific: it needs CUDA peer-to-peer writes between every pair. This repo holds:

- `car_patch.py`, which adds three env switches to 1Cat's `custom_all_reduce.py`. With them, the group counts
  as fully connected when every pair has P2P, and only the all-reduces the push kernel covers leave NCCL;
- `verify/`, the correctness gate we ran before trusting it, and a teacher-forced logprob comparison that
  tells you what the whole model does with it (`verify/teacher_forcing/`);
- `ops/`, the PCIe ACS fix the P2P path needs;
- the measurements, in [RESULTS.md](RESULTS.md).

On our box, four V100s in two NVLink pairs joined only by PCIe through the CPU root complex, single-request
decode of Qwen3.8-Flash-Next NVFP4 went from **75.9 to 91.5 tok/s**. Four all-NVLink V100s on the same box do
**93.5**. Prefill did not change.

## Who this is for

- You run 4x (or 8x) V100 on PCIe, without NVLink or with NVLink only in pairs, and serve Qwen3.8-class models
  with 1Cat-vLLM at TP=4.
- You want to know whether 1Cat's push kernel beats NCCL/PyNCCL on your PCIe topology.
  [`verify/verify_car.sh`](verify/verify_car.sh) tells you whether it is exact on your box and how fast it is
  per message size. Your own serving benchmark, with and without the switches, tells you what it is worth.
- Not covered: GPUs other than SM 7.0 (the push kernel is V100-only), TP sizes other than 4 (see the
  [FAQ](#faq)), and prefill-bound workloads.

## Results

Qwen3.8-Flash-Next NVFP4, 262k context, 1Cat-vLLM 357d07bcb. "2+2" is four GPUs, one NVLink pair on each of
two boards, so 4 of the 6 GPU pairs talk over PCIe only. Measured 2026-10-07; details in [RESULTS.md](RESULTS.md).

| | 2+2, NCCL | **2+2, this patch** | 4x NVLink, NCCL | 4x NVLink, custom AR |
|---|---|---|---|---|
| decode at 6k context (tok/s) | 75.9 | **91.5** | 76.7 | 93.5 |
| decode at 160k (tok/s) | 66.9 | **78.9** | 67.7 | 80.8 |
| TTFT at 160k (s) | 54.2 | 54.2 | 45.3 | 45.0 |
| KV cache (tokens) | 467,896 | 470,944 | 458,752 | 461,800 |

One all-reduce, 2+2 group across the CPU root complex, captured in a CUDA graph (`verify/car_gate.py`):

| | 5 KiB | 10 KiB | 32 KiB | 80 KiB | 128 KiB |
|---|---|---|---|---|---|
| NCCL | 16.0 us | 16.6 us | 18.7 us | 33.5 us | **44.8 us** |
| 1Cat push kernel | **5.1 us** | **6.7 us** | **13.5 us** | **30.7 us** | not admitted |
| 1Cat pull kernel | 14.6 us | 18.8 us | 27.9 us | 48.6 us | 67.8 us |

- On 1Cat-vLLM c4f6245f8 (2026-10-08): 90.4 and 90.9 tok/s at 6k across PCIe, against 95.1 and 95.5 on four
  NVLink GPUs.
- **Quality, teacher-forced** (c4f6245f8, [RESULTS.md](RESULTS.md) section 7): scored on identical token
  sequences, the 2+2 group with this patch is bit-identical to the same group on native NCCL over PCIe on all
  70,273 positions. Against four NVLink GPUs both arms show the same shift: abs(delta logprob) mean 1.12e-2,
  p95 5.63e-2, max 0.409; KL(top-20) mean 1.02e-3; top-1 agreement 98.93% on completion positions. It comes
  from NCCL's fp16 hops against the NVLink group's fp32-accumulating custom AR in prefill, not from the
  kernel. Teacher forcing scores in prefill, where the push kernel does not run, so it shows the patch leaves
  prefill untouched. The decode kernel itself is covered by the gate, by free-running greedy text, and by
  1Cat's three-seed quality suite: 108/108 on the NVLink reference, with this patch and on native NCCL.
- Earlier, on 357d07bcb, greedy outputs drifted from NCCL's as much as custom AR on NVLink already drifts from
  NCCL. GSM8K: 194/200 with NCCL, 194/200 with this patch.

## How it works

vLLM's custom all-reduce maps every peer's buffers into each rank with CUDA IPC and reduces in its own kernel
instead of NCCL. That matters at decode sizes, where NCCL's per-call cost dominates. 1Cat-vLLM adds an SM70
"push" variant for TP=4, adapted from SGLang-V100's one-shot push collective. Each rank stores its fp16 input
straight into a slot in every peer's buffer. It then reads only its own memory, until no element of any slot
still holds the "empty" sentinel, and sums the four slots in rank order in FP32. PCIe writes are posted, so the
sender never waits for them. Polling local memory costs no PCIe round trips. Every element signals its own
arrival, so the kernel needs no flags, fences, write ordering or peer atomics, and nothing that only NVLink
provides. vLLM keeps it off on PCIe because its gate asks NVML about NVLink. `VLLM_CAR_P2P_MESH=1` answers that
question with P2P reachability instead. `VLLM_CAR_MAX_BYTES=81921` and `VLLM_CAR_GRAPH_ONLY=1` keep everything
the push kernel does not cover on NCCL: eager calls and anything over 80 KiB. 1Cat would run those through its
pull kernel, which reads peer memory and loses to NCCL across PCIe.

```
one TP=4 decode all-reduce: 5 KiB = 2560 fp16 values, batch 1

 rank 0 --+   each rank: 16-byte st.volatile of its values into slot[e][rank]
 rank 1 --+   of EVERY rank's push buffer. PCIe posted writes: no round trip,
 rank 2 --+   no flag, no fence, no atomic.
 rank 3 --+
              each rank's own push buffer, epoch e (every fp16 starts as 0x7f7f = "empty"):
              +---------+---------+---------+---------+
              | slot 0  | slot 1  | slot 2  | slot 3  |
              +---------+---------+---------+---------+
 then each rank:  spin on LOCAL loads until no element in the 4 slots is 0x7f7f
                  out = fp16(((s0 + s1) + s2) + s3), summed in fp32: same bits on every rank
                  refill its own slots with 0x7f7f; the next call uses epoch e^1

 pull kernel (eager calls; anything over 80 KiB): flag barrier, then each rank LOADS
 every peer's input across PCIe, one round trip per load. The patch keeps these on NCCL.
```

An input element that happens to be 0x7f7f (a NaN) is sent as 0x7e00, still a NaN, so the sentinel cannot be
forged.

## Prerequisites

- **GPUs:** SM 7.0 (V100), four per tensor-parallel group, with working CUDA P2P between every pair of the four.
  ```bash
  nvidia-smi topo -m        # PIX/PXB: same switch; PHB: same CPU root complex; NODE/SYS: crosses NUMA nodes or sockets
  nvidia-smi topo -p2p w    # P2P writes, every pair: OK (the push kernel writes)
  nvidia-smi topo -p2p r    # P2P reads, every pair: OK (pull kernels, NCCL and vLLM's checks read)
  ```
  Peer atomics are not needed. Pairs across CPU sockets usually have no P2P; if any pair lacks it, the patch
  leaves custom AR off on every rank.
- **ACS:** P2P redirect cleared on every PCIe switch downstream port between the GPUs (`ops/`, under
  Install). With an IOMMU present, Linux turns redirect on even under `iommu=pt`, and peer traffic then goes up
  to the root port and back. On our box that made switch-to-switch P2P slower than going through the CPU.
- **IOMMU:** no translation on the GPU path: disabled, or passthrough (`iommu=pt`).
  `dmesg | grep -i 'default domain'` must not say `Translated`. NVIDIA's
  [GPU troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
  states that CUDA does not support PCIe P2P under a translating IOMMU on bare metal and warns of silent
  memory corruption. Passthrough does not clear ACS redirect by itself.
- **NVIDIA driver:** the 580 branch, the last with Volta support, with the proprietary kernel modules (the open
  modules start at Turing).
- **CUDA:** build 1Cat with a 12.x toolkit; our 357d07bcb wheel used 12.6. CUDA 13 drops offline sm_70
  compilation, and 1Cat's `CMakeLists.txt` only lists 7.0 below 13. Our images: Python 3.12,
  torch 2.10.0+cu128, NCCL 2.27.5.
- **1Cat-vLLM:** 357d07bcb or c4f6245f8 (`patches/`), or any revision whose anchors `car_patch.py` finds. It
  refuses the rest.
- **NCCL:** `NCCL_P2P_LEVEL=SYS`, so NCCL's own all-reduces (everything over 80 KiB) use P2P across root ports
  too. Without it NCCL staged our cross-board traffic through host shared memory at 3.5 GB/s.
- **Tools:** Docker with the NVIDIA container toolkit (gate and example), `python3` on the host (generator and
  gate checker), root and `pciutils` for `ops/`.

## Install

1. **Clear ACS redirect**, now and at every boot:
   ```bash
   sudo install -m 755 ops/pex-acs-p2p.sh /usr/local/sbin/
   sudo install -m 644 ops/pex-acs-p2p.service /etc/systemd/system/
   # name YOUR switches (lspci -nn) or their downstream ports; see the script's header
   echo 'ACS_VENDEVS="VVVV:DDDD"' | sudo tee /etc/default/pex-acs-p2p
   sudo systemctl daemon-reload && sudo systemctl enable --now pex-acs-p2p
   sudo /usr/local/sbin/pex-acs-p2p.sh status     # every port "direct", exit 0
   ```
2. **Generate the patched file** from the exact image you serve:
   ```bash
   IMG=your-1cat-image
   PKG=$(docker run --rm --entrypoint python3 "$IMG" -c \
     'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
   mkdir -p patched
   docker run --rm --entrypoint cat "$IMG" "$PKG/distributed/device_communicators/custom_all_reduce.py" \
     > patched/upstream.py
   python3 car_patch.py patched/upstream.py patched/custom_all_reduce.py
   ```
   A copy cut from another 1Cat revision would silently revert that revision's own changes to the file
   (c4f6245f8 changed it after 357d07bcb), so always cut it from the image you mount it into. If
   `car_patch.py` reports a missing anchor, your revision moved code it was not checked against; compare with
   `patches/` before touching the anchors.
3. **Run the gate** on the four GPUs you will serve on (next section).
4. **Serve** with the patched file mounted read-only over the image's copy (`examples/docker-run.sh`,
   under Serving).

Without containers:

- Source checkout at one of the two revisions: `git -C 1Cat-vLLM apply /path/to/patches/custom_all_reduce.c4f6245f8.diff`.
  An editable install picks it up; a wheel install needs a reinstall.
- Installed package, patched in place with the original kept:
  ```bash
  F=$(python3 -c 'import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin), "distributed/device_communicators/custom_all_reduce.py"))')
  cp "$F" "$F.orig" && python3 car_patch.py "$F.orig" "$F"
  ```

With none of its switches set, the patched file behaves exactly like the original.
`python3 -m unittest discover -s tests` checks the generator without a GPU.

## The gate

Run `verify/verify_car.sh` before serving: GPUs idle, the same four GPUs, and again after any driver, BIOS,
kernel, ACS, slot or 1Cat change. It takes about 30 minutes.

```bash
IMG=your-1cat-image GPUS=0,1,2,3 verify/verify_car.sh      # exit 0 = every check passed
verify/watch_gate.sh gate.log                               # optional progress view; see its header
```

It runs `verify/car_gate.py` in your image with exactly four GPUs visible. The harness forces the custom
all-reduce on by monkeypatch, so it needs no patched file, and checks every result against an FP32 rank-order
reference that each rank computes itself from seeds. The check never relies on the fabric under test.

| Case | What it shows | FAIL means |
|---|---|---|
| `negctl` | The checker sees a broken all-reduce: the C++ side is told "not fully connected" and launches nothing, so every size must come back WRONG | a size came back OK: the checker is blind, so ignore every other result |
| `perturb` | One bad element is visible: rank 1 changes one input element behind the reference's back | the error was missed, or appeared elsewhere |
| `car_int`, `car_randn`, `car_special` | 31 sizes from 16 B to 8 MiB; integer, random and special values (signed zeros, infinities, NaN, the sentinel bits); eager calls and CUDA-graph chains with rank skew. All ranks bitwise identical, a skipped replay caught as stale, every captured result up to 80 KiB bitwise equal to the reference | a wrong element, ranks disagreeing, a stale result missed, or push-size bits off the reference |
| `car_load_soak` | 1e5 collectives per size at 2-80 KiB while `car_load.py` keeps copy engines busy on the same links | an ordering failure under load, or the load never ran |
| `car_ref`, `digest` | The same seeds on `REF_GPUS` (default: your GPUs in reverse order, so every rank uses other links) give the same bits | results depend on which GPU or link carries which rank |
| `nccl` | The harness and NCCL work on your box | fix NCCL and P2P first |

Every case must also finish. A crash, a timeout (`CAP`, default 1200 s per case), fewer sizes than requested,
or a missing load fails it. `ONLY=<regex>` runs a subset, `CHECK_ONLY=1` re-checks old logs, and the per-case
logs go to `./car-results/`. `REF_GPUS` must be the same GPU model as `GPUS`: the seeded inputs depend on it.

A failing size, from `car-results/car.car_randn.log`, and its summary line:

```
{"mode": "car", "data": "randn", "bytes": 5120, "eager_wrong": 0, "eager_bitdiff": 0, "graph_collectives": 160,
 "graph_wrong": 37, "graph_bitdiff": 41, "stale_caught": true, "rank_hash_equal": false, ..., "verdict": "WRONG"}
FAIL  car_randn      30/31 sizes OK, graph collectives up to 160, graph bitdiff <=80 KiB 41
```

Each row also carries `us`, the time per all-reduce on the slowest rank, and `kernels`, the kernels that ran.
Compare `us` with the `nccl` case for your per-size speed table.

In serving, a wrong all-reduce looks like garbage or repeating tokens, NaN logits, or a sudden drop in eval
scores. A broken P2P path more often hangs CUDA-graph capture or the first request.

Our 2+2 group passed every check, before and after we added a PCIe switch ([RESULTS.md](RESULTS.md) section 2).

### Teacher-forced logprob check

The gate proves the all-reduce exact; [`verify/teacher_forcing/`](verify/teacher_forcing/README.md) measures
the model. It records a reference run as token IDs (`tf_prep.py`), re-scores the same IDs on each engine with
`prompt_logprobs: 20` (`tf_run.py`), and compares per position (`tf_compare.py`): abs(delta logprob), KL over
the top 20, top-1 agreement, and how many positions are bit-identical. Serve the reference layout, then the
patched and unpatched candidates on the same GPUs, and compare them. Stdlib only, against vLLM's
OpenAI-compatible server. It scores in prefill, so it checks that the patch leaves prefill alone and measures
the layout's own shift; it does not run the push kernel.

## Serving

[`examples/docker-run.sh`](examples/docker-run.sh) is the launch, with `IMG`, `GPUS`, `MODEL_DIR` and `PATCH`
as placeholders.

| Setting | Value | Why |
|---|---|---|
| `VLLM_CAR_P2P_MESH` | `1` | count the group as fully connected when every pair has P2P; every rank checks, and all must agree |
| `VLLM_CAR_MAX_BYTES` | `81921` | custom AR only below 81921 bytes; the push kernel stops at 81920 |
| `VLLM_CAR_GRAPH_ONLY` | `1` | custom AR only inside CUDA graphs; 1Cat's eager path is the pull kernel |
| `NCCL_P2P_LEVEL` | `SYS` | NCCL keeps the large all-reduces; let it use P2P across root ports |
| `VLLM_SM70_TOP1_CUSTOM_AR` | `0` (c4f6245f8 and later) | greedy top-1 is a pull kernel, which loses over PCIe; this keeps it on NCCL (`TOP1_CUSTOM_AR=0` in the example) |
| `VLLM_TP_ALLREDUCE_TRACE` | `1` (optional) | log the backend of each all-reduce shape, once per shape |
| `--tensor-parallel-size` | `4` | the push kernel is TP=4 only |

Do not pass `--enforce-eager`: without CUDA graphs there is no push kernel, and with `VLLM_CAR_GRAPH_ONLY=1`
no custom AR at all. Do not pass `--disable-custom-all-reduce` either.

After startup the log should show:

```
VLLM_CAR_P2P_MESH: custom allreduce treats GPUs [0, 1, 2, 3] as fully connected: True    (once per worker)
SM70 TP4 SGLang-style push all-reduce enabled for the FP16 80-KiB verifier, 8-KiB decode, ...
Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' ...
TP all-reduce trace backend=custom group=tp:0 shape=(1, 16, 160) dtype=torch.float16 bytes=5120 ...
TP all-reduce trace backend=pynccl group=tp:0 shape=(2048, 2560) dtype=torch.float16 bytes=10485760 ...
```

The decode shape also shows up once with `backend=pynccl`: those are eager warm-up calls, which
`VLLM_CAR_GRAPH_ONLY=1` sends to NCCL. `Unknown vLLM environment variable detected: VLLM_CAR_P2P_MESH` is
expected, because vLLM's registry does not know the patch's switches.

## Known limits

- **TP=4 only.** The push kernel is compiled for exactly four ranks. For TP=8, see the [FAQ](#faq).
- **Decode-sized all-reduces only (80 KiB cap).** Prefill all-reduces (120 KiB to 10 MiB here) stay on NCCL
  and are bandwidth-bound across PCIe, so TTFT does not change. For Qwen3.8-Flash-Next (2560 fp16 values per
  token) 80 KiB is 16 token rows per step; bigger decode batches, including speculative tokens, fall back to
  NCCL.
- **CUDA graphs only.** Eager all-reduces go to NCCL (`VLLM_CAR_GRAPH_ONLY=1`); outside capture 1Cat only has
  its pull kernel.
- **The pull kernel loses across PCIe.** It is level with NCCL at 5 KiB and slower from 10 KiB up
  ([RESULTS.md](RESULTS.md) section 1), because every pull load waits for a PCIe round trip. The patch keeps
  all of its work on NCCL.
- **Read/write asymmetry on AMD root complexes.** Through our Zen 2 root complex: SM P2P stores 10.5 GB/s,
  loads 8.5 GB/s, copy engine 13.2 GB/s, 1.58 us one way. The push kernel only stores to peers. Intel root
  complexes and dual-socket boxes are untested.
- **What the gate does not cover.** Once the group counts as fully connected, 1Cat runs more collectives
  through the same IPC buffers, outside `should_custom_ar` and therefore outside `VLLM_CAR_MAX_BYTES` and
  `VLLM_CAR_GRAPH_ONLY`:
  - Qwen3.8-Flash-Next's fused HC all-gathers (`VLLM_SM70_QWEN38_FUSED_HC_FP16=1`), in eager and captured
    calls. They are push-style like the all-reduce: sentinel-marked or tagged values, local polling.
  - Greedy top-1 (`VLLM_SM70_TOP1_CUSTOM_AR`), a pull kernel: a flag barrier, then peer loads. It has been on
    by default since [1Cat-vLLM #821](https://github.com/1CatAI/1Cat-vLLM/pull/821), so it runs on c4f6245f8
    and not on 357d07bcb.
  - Through `should_custom_ar`, so capped and graph-only: the MoE `all_reduce_sum2` push variant and, for
    models with hidden size 5120, the fused all-reduce + RMSNorm push.

  `verify_car.sh` tests the plain all-reduce only. For the rest our evidence is end-to-end
  ([RESULTS.md](RESULTS.md) sections 6 and 7, with `VLLM_SM70_QWEN38_FUSED_HC_FP16=1` in every run): outputs
  within the drift custom AR already has on NVLink, GSM8K unchanged, 1Cat's quality suite 108/108.
  `VLLM_SM70_TOP1_CUSTOM_AR=0` (`TOP1_CUSTOM_AR=0` in the example) keeps top-1 on NCCL. It gave +0.8% decode
  at 6k and +1.0% at 160k with TTFT unchanged, and greedy tokens and logprobs bit-identical to leaving it on.
- **Correctness is per box.** We gated V100-SXM2 on PCIe carrier boards behind PLX/Broadcom switches and a
  Zen 2 root complex. Other boards, switches and root complexes need their own gate run.
- **1Cat revisions move.** `car_patch.py` was checked against 357d07bcb and c4f6245f8 and refuses files whose
  anchors moved. Admission rules (sizes, defaults such as top-1) change between revisions too: run the gate
  on every new image.

## FAQ

### Does the custom all-reduce need NVLink inside the group? Does it run on 4 PCIe-only cards? On 8?

It needs no NVLink. Four PCIe-only cards should work, though a box with no NVLink at all is not what we
tested. Eight cards get it per group of four, not at TP=8.

What 1Cat's code requires. Line numbers are for c4f6245f8. 357d07bcb has the same logic, at nearby lines in
the Python files and up to about 250 lines earlier in `custom_all_reduce.cuh`.

- **The only NVLink check is the gate in front of the kernel.** `is_fully_connected` asks NVML for
  `NVML_P2P_CAPS_INDEX_NVLINK` on every pair (`vllm/platforms/cuda.py:756-778`). When that fails,
  `vllm/distributed/device_communicators/custom_all_reduce.py:216-233` turns custom AR off for more than two
  GPUs, except in 1Cat's opt-in TP=8 mode. `VLLM_CAR_P2P_MESH=1` replaces the answer with
  `torch.cuda.can_device_access_peer` for every pair, agreed by all ranks.
- **The C++ side does what it is told.** It keeps the flag it is given (`csrc/custom_all_reduce.cuh:1921-1922`,
  commented "Full NVLink or xGMI connection"). For more than two GPUs not marked fully connected it launches no
  kernel at all (`custom_all_reduce.cuh:2315-2324`: an `if` / `else if` with no `else`). So forcing only the
  Python side returns garbage; the gate's `negctl` case checks exactly that.
- **It needs CUDA P2P between every pair.** Every rank opens every peer's buffers through CUDA IPC
  (`custom_all_reduce.py:1034-1056`; `cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)` at
  `custom_all_reduce.cuh:1983-1985`). vLLM also runs a real cross-process IPC write test first
  (`custom_all_reduce.py:266-276`).
- **The kernel only stores to peers.** `sm70_cross_device_reduce_1stage_push` (`custom_all_reduce.cuh:816-891`)
  issues 16-byte `st.volatile.global` stores into every peer's slot (`:843-850`), polls only its own buffer for
  the sentinel (`:853-868`), and sums in rank order in FP32 (`:870-871`). It uses no flags, fences or peer
  atomics.
- **Admission:** SM 7.0, world size exactly 4, fully connected (`custom_all_reduce.py:332-348`,
  `custom_all_reduce.cuh:2033-2038`), inside CUDA-graph capture, FP16, at most 81920 bytes
  (`kSm70Tp4PushAllreduceM8Bytes`, `custom_all_reduce.cuh:90-91`, `:194-248`, `:2208-2223`). Everything else
  goes to vLLM's pull kernels.

Four PCIe-only cards:

- **Verified:** four V100s with 4 of their 6 pairs PCIe-only (two NVLink pairs joined through the CPU root
  complex, later through a PCIe switch). Every gate check passed, and decode went from 75.9 to 91.5 tok/s.
- **Inferred, not tested:** a box with no NVLink at all. Each rank's stores then cross PCIe to three peers
  instead of two. At decode sizes (5-20 KiB) that costs latency, not bandwidth, so we expect it to work with a
  similar gain. Measure it: `verify_car.sh` reports per-size times next to NCCL's.

Eight cards:

- **Not at TP=8.** The push kernel is built for four ranks: `kSm70Tp4PushAllreduceWorldSize = 4`
  (`custom_all_reduce.cuh:84`), `static_assert(ngpus == 4)` (`:822`), and `world_size == 4` checks in Python
  (`custom_all_reduce.py:334`) and C++ (`:2034`, `:2214`). At TP=8 the patch would only unlock vLLM's generic
  pull kernels (`custom_all_reduce.cuh:2305-2340`: one-stage below 256 KiB, two-stage above), which lost to
  NCCL from 10 KiB up at TP=4 on our box. We have not measured TP=8; leave `VLLM_CAR_P2P_MESH` unset there.
- **1Cat's own TP=8 mode does not fit either.** `VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR=1`, off by default, is
  built for two 4-GPU NVLink cliques. It requires the group not to be fully connected
  (`custom_all_reduce.cuh:294-299`) and admits only FP16 payloads of exactly 4096 or 32768 elements
  (`custom_all_reduce.py:32`, `:434-438`), sized for a hidden size of 4096; Qwen3.8-Flash-Next's decode
  all-reduces are 2560 values per token. Its topology check is plain P2P (`custom_all_reduce.py:246-252`), so
  a PCIe box would pass it. We have not tried it.
- **What works: groups of four.** Two TP=4 replicas, or TP=4 x PP=2 when the model needs all eight cards'
  memory. Each TP group gets the push kernel if its six pairs have P2P; run the gate on each group. We have not
  tested this, and some 1Cat model profiles pin PP=1.
  [vLLM #50941](https://github.com/vllm-project/vllm/issues/50941) proposes an island-aware custom all-reduce
  for 2x4 PCIe boxes; also untested here.

What else a PCIe-only box needs: ACS redirect cleared on every switch downstream port (`ops/`), no IOMMU
translation (disabled or `iommu=pt`), and `NCCL_P2P_LEVEL=SYS` for what stays on NCCL. Expect loads to be the
slow direction. Through our Zen 2 root complex, P2P stores ran at 10.5 GB/s and loads at 8.5 GB/s, and every
load waits for a full round trip (1.58 us each way). The push kernel only stores to peers, which is why it
wins. The pull kernel loads from peers and loses, which is why the patch keeps its work on NCCL.

### Will a PCIe Gen4 switch help, and does ACS matter?

We put a Broadcom PEX880xx Gen4 switch above both GPU boards, so board-to-board traffic no longer went
through the CPU root complex.

| Board to board, GPU to GPU | Through the root complex | Switch, ACS redirect on | Switch, ACS cleared |
|---|---|---|---|
| Latency, one way | 1.58 us | 1.79 us | **1.25 us** |
| SM loads | 8.5 GB/s | 7.12 GB/s | 9.44 GB/s |
| Copy engine, one way | 13.2 GB/s | 13.18 GB/s | 13.19 GB/s |

- With ACS redirect left on, which is the Linux default whenever an IOMMU is present, the switch made things
  worse: peer traffic still went up to the root port and back, with one more hop.
- With redirect cleared, latency dropped from 1.58 to 1.25 us and loads got faster, but bandwidth did not move,
  because each board's Gen3 x16 uplink was already at line rate. A Gen4 switch adds bandwidth only if the links
  below it are faster than the ones it replaces.
- Serving did not move: decode with custom AR went 91.5 to 91.4 tok/s, TTFT at 160k 54.2 to 54.1 s, and the
  push kernel 5.1 to 5.3 us at 5 KiB. The custom AR needs P2P, not a switch.

### Does it help prefill?

No. Prefill all-reduces here are 120 KiB to 10 MiB, above the push kernel's 80 KiB and bandwidth-bound across
PCIe, so they stay on NCCL. TTFT at 160k was 54.2 s with and without the patch, and 45.0 s on four NVLink GPUs.

### TP=2, or GPUs other than V100?

- **TP=2:** vLLM already enables its custom all-reduce for two GPUs without NVLink (`custom_all_reduce.py:219`,
  `:441`), with its one-stage pull kernel. The patch changes nothing there. We have not measured TP=2 over PCIe.
- **Other GPUs:** the push kernel is gated to SM 7.0. Elsewhere the patch would only enable vLLM's pull kernels
  for more than two PCIe GPUs, the case upstream keeps off because "for 4 or more non NVLink-capable GPUs,
  custom allreduce provides little performance improvement over NCCL" (`custom_all_reduce.py:439-440`). Our
  pull-kernel numbers agree. Don't use it there.

## Troubleshooting

- **`Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs`.** The patched
  file is not mounted (compare the mount target with the image's `vllm` path), `VLLM_CAR_P2P_MESH=1` did not
  reach the workers, or a rank cannot reach a peer: look for `fully connected: False` in the
  `VLLM_CAR_P2P_MESH` line and run `nvidia-smi topo -p2p w`.
- **`Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed`.** vLLM's
  own cross-process IPC write test failed. Fix ACS or the IOMMU, then delete the cached verdict,
  `gpu_p2p_access_cache_for_*.json` under `VLLM_CACHE_ROOT` (default `~/.cache/vllm`), which later runs reuse.
- **No `SM70 TP4 SGLang-style push all-reduce enabled` line.** The push path registers only for world size 4
  on SM 7.0, with the group fully connected and `VLLM_SM70_TP4_PUSH_ALLREDUCE` not set to 0.
- **The trace shows decode shapes on `pynccl` only.** CUDA graphs are off (`--enforce-eager`), or the decode
  all-reduce is above 81920 bytes (rows x hidden size x 2).
- **Hang in CUDA-graph capture or on the first request.** Suspect P2P in one direction. Run
  `IMG=your-1cat-image SIZES=5120 verify/car_gate.sh car` for a quick look, then the full gate. Separately,
  NCCL 2.27.5 hung once in five engine starts on our box in its lazy Tree connect, with custom AR not
  involved; `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,PROXY` captures a recurrence.
- **`pex-acs-p2p.sh status` exits 1 some time after boot.** A port went through AER recovery or runtime power
  management and got the kernel default back. Run `apply` again and check `dmesg` for AER messages.
- **Docker created a directory named `custom_all_reduce.py`.** The mount source did not exist when docker
  started. `examples/docker-run.sh` checks for it first.

## Credits

- The [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) authors, for the SM70 custom all-reduce kernels,
  including the push kernel this repo enables.
- [SGLang-V100](https://github.com/haohervchb/sglang-V100), whose one-shot push collective 1Cat's kernel is
  adapted from, per 1Cat's source.
- [vLLM](https://github.com/vllm-project/vllm), for the custom all-reduce framework both build on.

## License

Apache-2.0, see [LICENSE](LICENSE). The diffs in `patches/` and the anchors in `car_patch.py` are derived from
1Cat-vLLM, which is also Apache-2.0; see [NOTICE](NOTICE).
