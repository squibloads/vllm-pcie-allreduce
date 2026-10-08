# Results

Everything here was measured on one box between 2026-10-06 and 2026-10-08. Times are UTC.

## The box

- **GPUs:** six V100-SXM2-32GB on two PCIe carrier boards. Board A holds four in an NVLink (NV2) mesh behind
  a PLX PEX8796 switch; board B holds two, an NVLink pair, behind another PEX8796. Each board has one PCIe
  Gen3 x16 uplink.
- **Host:** AMD Threadripper PRO 3975WX (Zen 2) on WRX80. 300 W power cap per GPU in every run.
- **Two fabric states:**
  - until 2026-10-07: each board on its own CPU root port, so board-to-board P2P crosses the CPU root complex;
  - from 2026-10-07: both boards under one Broadcom PEX880xx Gen4 switch, ACS redirect cleared
    (`ops/pex-acs-p2p.sh`), so board-to-board P2P stays inside the switches.
- **The TP=4 group under test ("2+2"):** two GPUs from each board, i.e. one NVLink pair per board. Every pair
  that crosses the boards, 4 of the group's 6 pairs, has PCIe P2P only.
- **Reference ("NVLink board"):** the four GPUs of board A, the layout the box served from before this work.
- **Software:** 1Cat-vLLM images at 357d07bcb (2026-09-28) and c4f6245f8 (2026-10-08). Python 3.12,
  torch 2.10.0+cu128, NCCL 2.27.5 (torch's), NVIDIA driver 580; the 357d07bcb wheel was built with CUDA 12.6.
- **Model:** an NVFP4 checkpoint of Qwen3.8-Flash-Next (125B total, 6B active parameters; a fine-tune with
  the stock architecture). Its TP=4 decode all-reduce is 2560 fp16 values per token: 5 KiB at batch 1.
- **Serving:** 262,144-token context, `--max-num-seqs 2`, CUDA graphs on, prefix caching on.

## 1. All-reduce per call (`verify/car_gate.py`)

A CUDA graph of 50 independent all-reduces, time per all-reduce, slowest rank. 2+2 group, board-to-board P2P
through the CPU root complex, 357d07bcb image, 2026-10-06/07.

| Size | NCCL (P2P transport) | Custom AR, push | Custom AR, pull only |
|---|---|---|---|
| 5 KiB (decode, batch 1) | 16.0 us | **5.1 us** | 14.6 us |
| 10 KiB (batch 2) | 16.6 us | **6.7 us** | 18.8 us |
| 32 KiB | 18.7 us | **13.5 us** | 27.9 us |
| 80 KiB | 33.5 us | **30.7 us** | 48.6 us |
| 128 KiB | **44.8 us** | 67.6 us (pull: push stops at 80 KiB) | 67.8 us |
| 1 MiB | **186 us** | 239 us (pull) | 240 us |

- 1Cat runs the push kernel only for all-reduces captured into a CUDA graph, up to 81920 bytes. Above that, and
  for every eager call, it runs the pull kernel. "Pull only" is `VLLM_SM70_TP4_PUSH_ALLREDUCE=0`.
- Pull is level with NCCL at 5 KiB and loses from 10 KiB up: every pull load is a round trip across the root
  complex. NCCL's transport writes.
- The same push kernel on the NVLink board: 3.4 us at 5 KiB. Crossing the root complex costs about 2 us.
- Under copy-engine load across the root complex (`car_load.py`), push at 5 KiB slows to 9.8 us and stays exact.
- After the Gen4 switch (2026-10-07): push 5.3 us at 5 KiB; at 128 KiB pull 66 us and NCCL 43.6 us.

## 2. Correctness gate (`verify/verify_car.sh`)

Passed on the 2+2 group through the root complex, and again after the Gen4 switch:

- `negctl` WRONG at all 31 sizes, so the checker can see a broken all-reduce;
- `perturb` wrong in exactly the perturbed element, on all 4 ranks, in every collective;
- custom AR OK at all 31 sizes (16 B to 8 MiB) for integer, random and special-value data, eager and graph
  chains with rank skew, stale replay caught, every result up to 80 KiB bitwise equal to the FP32 rank-order
  reference;
- OK through a 1e5-collective soak at 2-80 KiB under copy-engine load across the root complex;
- the same seeds on the NVLink board gave bit-identical results. The published gate compares against a second
  GPU set or order (`REF_GPUS`) instead, so that it also runs on boxes without NVLink.

## 3. Serving A/B, 357d07bcb, through the root complex

One window, 2026-10-07 06:07-06:48, same image, harness and model for every arm. "decode at 6k" is one greedy
request with a ~6,000-token prompt and 32 output tokens; "160k" has a 160,008-token prompt and 128 output
tokens; "24k x2" is two concurrent ~24,000-token requests with 256 output tokens each. Decode rate is
(tokens - 1) / (last token time - first token time).

| | 2+2, NCCL | **2+2, custom AR (this repo)** | NVLink board, NCCL | NVLink board, custom AR |
|---|---|---|---|---|
| decode at 6k (tok/s) | 75.9 | **91.5** (+21%) | 76.7 | 93.5 |
| decode, 24k x2 concurrent (tok/s) | 44.7 / 49.0 | **50.6 / 56.1** | 47.7 / 49.4 | 52.3 / 57.3 |
| decode at 160k (tok/s) | 66.9 | **78.9** (+18%) | 67.7 | 80.8 |
| TTFT, 24k x2 concurrent (s) | 13.83 / 14.34 | 13.82 / 14.33 | 11.21 / 11.39 | 11.01 / 11.45 |
| TTFT at 160k (s) | 54.2 | 54.2 | 45.3 | 45.0 |
| KV cache (tokens) | 467,896 | 470,944 | 458,752 | 461,800 |

- Across the boards, custom AR brings decode within 2-3% of the NVLink board (91.5 vs 93.5, 78.9 vs 80.8),
  up from 18-20% behind.
- Without custom AR, the 2+2 group decodes as fast as the NVLink board does (75.9 vs 76.7). The PCIe hop was
  not the decode bottleneck; NCCL's per-call cost was.
- Prefill is unchanged by custom AR: TTFT at 160k is 54.2 s against 45.0 s on the NVLink board. Prefill
  all-reduces (120 KiB to 10 MiB) are bandwidth-bound across PCIe and stay on NCCL.
- `VLLM_TP_ALLREDUCE_TRACE=1` showed decode's 5, 10 and 20 KiB all-reduces on the custom path and every prefill
  chunk on NCCL.
- An earlier A/B (cap 96 KiB, eager calls allowed) gave the same decode: 91.2 and 79.1 tok/s.

## 4. Same harness after the Gen4 switch (357d07bcb, 2026-10-07)

| | TTFT at 160k (s) | decode at 6k (tok/s) | decode at 160k (tok/s) |
|---|---|---|---|
| 2+2, NCCL: root complex, then switch | 54.2, 54.1 | 75.9, 75.9 | 66.9, 66.8 |
| 2+2, custom AR: root complex, then switch | 54.2, 54.1 | 91.5, 91.4 | 78.9, 79.4 |
| NVLink board, custom AR: before, after | 45.0, 45.0 | 93.5, 93.3 | 80.8, 80.4 |

PCIe P2P between the boards, GPU to GPU (GB/s = 1e9):

| | Through the root complex | Switch, ACS redirect on | Switch, ACS cleared |
|---|---|---|---|
| Copy engine, one way | 13.2 | 13.18 | 13.19 |
| Copy engine, both ways | not measured | 25.0 | 25.1 |
| SM stores | 10.5 | 10.49 | 10.49 |
| SM loads | 8.5 | 7.12 | 9.44 |
| Latency, one way | 1.58 us | 1.79 us | **1.25 us** (2.75 under 25 GB/s load) |

Each board's Gen3 x16 uplink was already at line rate through the root complex, so the switch changed latency
and loads, not bandwidth, and serving did not move.

## 5. Serving, c4f6245f8, after the Gen4 switch (2026-10-08)

Both arms used the same c4f6245f8 serving configuration, and the patch was regenerated from the c4f6245f8
file.

| | NVLink board, custom AR | **2+2, custom AR (this repo)** |
|---|---|---|
| decode at 6k, two requests (tok/s) | 95.1 / 95.5 | **90.4 / 90.9** |
| decode, ~800-token prompt (tok/s) | 97.1 | 92.0 |
| decode at 160k (tok/s) | 81.6 | 78.5 |
| TTFT at 160k (s) | 36.8 | 46.4 |
| KV cache (tokens) | 455,703 | 466,372 |

- No 2+2 NCCL arm was timed on this image; the NCCL arm in section 7 was run for quality only.
- The decode gap to the NVLink board grew from 2.1% (357d07bcb: 91.5 vs 93.5) to 4.9%. 1Cat-vLLM #821, which
  landed between the two images, made `VLLM_SM70_TOP1_CUSTOM_AR=1` the default. With the group counted as
  fully connected, greedy top-1 then goes through a custom-AR pull kernel across PCIe. That was the first
  suspect; section 7 isolated it, and switching it off recovers about 1%.
- TTFT at 160k is 26% longer than on the NVLink board (357d07bcb: 20%). c4f6245f8 prefills in larger steps,
  and the NVLink board sends their 8 MB all-reduces through custom AR, while across PCIe they stay on NCCL;
  whether that is the whole gap is open.

## 6. Output quality

Greedy parity, 357d07bcb, 2026-10-07: 32 prompts x 256 tokens, temperature 0, one request at a time.

| Pair | Identical | First divergence, median (q1-q3) | Mean abs(delta logprob) before it |
|---|---|---|---|
| NVLink board: custom AR vs NCCL (the drift that layout already had) | 8/32 | token 106 (47-188) | 8.0e-3 |
| 2+2: custom AR vs NCCL | 7/32 | token 91 (48-179) | 7.1e-3 |
| NCCL: NVLink board vs 2+2 (no custom AR on either side) | 9/32 | token 154 (64-228) | 8.0e-3 |

Custom AR across PCIe drifts from NCCL about as much as custom AR on NVLink drifts from NCCL. The custom path
adds in FP32 and rounds once, NCCL rounds at every step, so bitwise equality with NCCL is impossible by
construction.

GSM8K, 357d07bcb, 2026-10-07 07:01-09:27: 100 questions x 2 seeds, temperature 1, max_tokens 65536.

| | 2+2, NCCL | **2+2, custom AR** | NVLink board, custom AR |
|---|---|---|---|
| GSM8K, both seeds | 194/200 | **194/200** | 193/200 |
| Decode median, same runs (tok/s) | 74.4 | **89.0** (+20%) | 92.1 |

McNemar per seed: p = 0.5 for custom AR vs NCCL on the 2+2 group. The misses are the same hard questions
moving between arms under temperature-1 sampling. These runs had no reasoning parser enabled, which changes
output parsing only, not speed or grading.

c4f6245f8 against the NVLink board: 3 of 4 greedy texts identical, the fourth a coherent near-tie flip. On 16
long prompts (2.5k-6k tokens, 192 output tokens), 1 of 16 identical, first divergence median at token 48,
mean abs(delta logprob) before it 9.2e-3. At every divergence the two candidate tokens are within 0.19 nats
of each other.

1Cat's own quality suite and a teacher-forced (per-token KL) comparison are in section 7.

## 7. Quality gate (2026-10-08)

c4f6245f8, after the Gen4 switch, one engine launch per arm, the same serving configuration in all three:

- **NVLink reference:** the four NVLink GPUs of board A, custom AR as 1Cat ships it.
- **2+2, custom AR:** this repo's patch, the 2+2 group across the boards.
- **2+2, NCCL:** the same 2+2 group, unpatched: every all-reduce on NCCL over PCIe.

Four checks: 1Cat's own quality suite, teacher-forced logprobs, free-running greedy text, and the greedy top-1
switch.

### 1Cat's quality suite

1Cat-vLLM's Qwen3.8 quality gate (`benchmarks/benchmark_sm70_qwen38_quality.py`), with its own prompts, scoring
and health checks: three seeds, temperature 1.0, top-p 0.95, top-k 20, up to 4096 tokens, natural EOS. The
script loads its own engine and strips every `VLLM_*` variable, which would remove this patch's switches, so the
generations went over HTTP to the served arms and the scoring used the gate's functions unchanged.

| Generations passed | NVLink reference | 2+2, custom AR | 2+2, NCCL |
|---|---|---|---|
| MBPP (36) | 36/36 | 36/36 | 36/36 |
| GSM8K (36) | 36/36 | 36/36 | 36/36 |
| Chinese (24) | 24/24 | 24/24 | 24/24 |
| Needle, 8k / 32k / 131k / 258k (12) | 12/12 | 12/12 | 12/12 |
| **Total (108)** | **108/108** | **108/108** | **108/108** |
| Unhealthy outputs (no natural EOS, empty answer, replacement characters, repeated answer line) | 0 | 0 | 0 |
| `compare_quality` against the NVLink reference | - | pass | pass |

### Teacher-forced logprobs

16 prompts of 2.5k-6k tokens, each followed by the NVLink reference's own 192 greedy tokens: 70,273 scored
positions, 3,072 of them completion tokens and 67,201 prompt tokens. Every arm scored the same token IDs with
`prompt_logprobs: 20`, one request at a time (`verify/teacher_forcing/`). "NVLink, scored again" is the
reference engine a second time; "NVLink, relaunched" is the same layout in a fresh engine launch.

| Arm | Positions | abs(delta logprob) mean / p95 / max | KL(top-20) mean / p95 / max | Top-1 agreement |
|---|---|---|---|---|
| NVLink, scored again | completion | 0 / 0 / 0 | 0 / 0 / 0 | 100% |
| NVLink, relaunched | completion | 0 / 0 / 0 | 0 / 0 / 0 | 100% |
| **2+2, custom AR** | completion | 1.12e-2 / 5.63e-2 / 0.409 | 1.02e-3 / 4.21e-3 / 8.72e-2 | 98.93% (33 of 3,072 differ) |
| **2+2, NCCL** | completion | 1.12e-2 / 5.63e-2 / 0.409 | 1.02e-3 / 4.21e-3 / 8.72e-2 | 98.93% (33 of 3,072 differ) |
| NVLink, scored again and relaunched | prompt | 0 / 0 / 0 | 0 / 0 / 0 | 100% |
| **2+2, custom AR** | prompt | 1.22e-1 / 0.599 / 13.4 | 3.10e-2 / 7.81e-2 / 10.7 | 95.83% (2,801 of 67,201 differ) |
| **2+2, NCCL** | prompt | 1.22e-1 / 0.599 / 13.4 | 3.10e-2 / 7.81e-2 / 10.7 | 95.83% (2,801 of 67,201 differ) |

KL is taken over the reference's top-20 distribution plus one bucket for the rest of the mass, in nats: a lower
bound on the full-vocabulary KL.

- **The patch changes nothing the model can see in prefill.** The 2+2 custom AR arm is bit-identical to the 2+2
  NCCL arm on all 70,273 positions: the same figures to the last digit, the same position of the largest KL.
  Scoring a prompt is a prefill. Its all-reduces are 120 KiB to 10 MiB, above the push kernel's 80 KiB, and
  `VLLM_CAR_GRAPH_ONLY=1` keeps custom AR to captured decode graphs, so every one of them runs on NCCL with or
  without the patch. The push kernel does not run in this measurement; it is covered by the gate
  (`verify_car.sh`), by the free-running comparison below, and by the quality suite above.
- **The shift against the NVLink reference is the layout's.** Both 2+2 arms sit 1.12e-2 mean and 5.63e-2 p95
  from the reference in abs(delta logprob) (max 0.409), at KL 1.02e-3 mean, 4.21e-3 p95 and 8.72e-2 max, with
  top-1 agreement of 98.93% on completion positions. We attribute it to prefill rounding. On the NVLink group
  the multi-MiB prefill all-reduces go through custom AR, which accumulates in fp32 and rounds once. Across
  PCIe they stay on NCCL, which rounds to fp16 at each hop. The arm without the patch has the same shift, so
  the push kernel is not its source.
- **The control is exact.** Scoring the reference engine again, and the same layout in a fresh launch, gives
  zero on every figure, so these differences are not run-to-run noise.

### Free-running greedy text

16 prompts of 2.5k-6k tokens, 192 greedy tokens each. Decode runs in captured graphs here, so the push kernel
does run. First divergence is the first token that differs; an identical completion counts as 192.

| Pair | Identical | First divergence, median (q1-q3) | Mean abs(delta logprob) before it |
|---|---|---|---|
| NVLink reference vs the same layout relaunched | 16/16 | token 192 | 0 |
| NVLink reference vs 2+2, custom AR | 1/16 | token 48 (8-92) | 9.21e-3 |
| NVLink reference vs 2+2, NCCL | 1/16 | token 48 (8-124) | 1.03e-2 |
| 2+2, NCCL vs 2+2, custom AR | 4/16 | token 168 (64-191) | 4.75e-3 |

Both 2+2 arms leave the NVLink reference at the same median, token 48. The teacher-forced table points at the
prefill: both start from the same shifted state. The two 2+2 arms stay together more than three times as long,
median token 168. They share that prefill, so what separates them is decode: the push kernel's fp32 sum against
NCCL's fp16 hops. At every divergence from the reference, the two candidate tokens are within 0.2 nats of each
other (largest 0.191).

A decode probe with one-token prompts, meant to isolate the decode path, left the NVLink reference at token 0
or 1 on every 2+2 arm, the NCCL arm included, so it does not separate the kernel from the layout and is not
used here.

### Greedy top-1 on c4f6245f8

1Cat-vLLM's greedy top-1 custom all-reduce (`VLLM_SM70_TOP1_CUSTOM_AR`, on by default since #821) is a pull
kernel, which loses over PCIe. Switching it off on the 2+2 group, custom AR otherwise as before:

| 2+2, custom AR | top-1 on | top-1 off | Change |
|---|---|---|---|
| decode at 6k, two requests (tok/s) | 90.96 / 91.51 | 91.54 / 92.38 | +0.8% |
| decode at 160k (tok/s) | 78.51 | 79.26 | +1.0% |
| TTFT at 160k (s) | 43.00 | 43.07 | unchanged |

Greedy tokens and logprobs are bit-identical with it on and off (16 of 16 completions, mean abs(delta logprob)
0). This pair ran on a configuration whose cross-board NCCL transport differs from the rest of this section,
which is why its TTFT is lower than section 5's 46.4 s; read the change within the pair, not the absolute
figures. The pull kernel costs about 1% of decode here. The gap to the NVLink board widened from 2.1% to
4.9% between the two images (section 5), 2.8 points, of which it explains about one.
