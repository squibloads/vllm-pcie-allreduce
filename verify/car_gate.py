"""car_gate - does 1Cat-vLLM's custom all-reduce run TP=4 over PCIe P2P correctly, and how fast?

vLLM disables its IPC custom all-reduce for more than two GPUs that are not NVLink-fully-connected, and
1Cat's C++ dispatch launches *nothing* for world_size > 2 when told the GPUs are not fully connected, so
forcing only the Python gate would return garbage. This harness forces fully_connected=True by
monkeypatch - no file in the image changes - on four GPUs some or all of whose pairs talk only over PCIe
P2P. 1Cat's push kernel needs peer stores; its pull kernels and their flag barriers need peer loads and
plain stores; neither uses peer atomics, which PCIe P2P between GPUs often lacks. Per message size it checks:

  1. eager: poisoned caller-owned output, fresh inputs every iteration;
  2. CUDA-graph chains of K back-to-back all-reduces (each input is x + 0*previous output, so every step
     waits for the last and reuses its signalling state; the value stays x), replayed with fresh inputs,
     poisoned outputs and rank skew - the path vLLM decode takes, where 1Cat's push kernels live; a SOAK
     of many more replays at chosen sizes;
  3. a STALE self-test: one poisoned replay is skipped and must verify as WRONG everywhere;
  4. latency of K independent all-reduces in one graph, and which kernels ran (torch.profiler).

Every rank regenerates all four ranks' inputs from seeds and computes the reference itself - FP32 sums in
rank order, rounded once - so the check never uses the fabric under test. `wrong` counts elements outside
fp16 rounding of that reference (NaN must meet NaN, inf the same inf); `bitdiff` counts elements whose bits
differ from it (push and one-stage pull add in rank order, so expect 0 there); `rank_hash` must agree.

MODE  nccl        torch.distributed NCCL (P2P transport with NCCL_P2P_LEVEL=SYS) - the baseline
      car         custom AR forced on (1Cat SM70 TP4 push path on by its own default)
      car-nopush  custom AR forced on, VLLM_SM70_TP4_PUSH_ALLREDUCE=0 (pull one-/two-stage only)
      negctl      NEGATIVE CONTROL: Python believes fully connected, C++ is told it is not, so the kernels
                  launch nothing; every size must come back WRONG or the verifier is blind
      perturb     NEGATIVE CONTROL: custom AR, but rank 1 adds 1.0 to element 0 of its input behind the
                  reference's back; exactly that element must come back WRONG, on every rank
env   DATA=int|randn|special  SIZES=b1,b2,..  EAGER_ITERS CHAIN CHAIN_REPLAYS SKEW=1 SOAK=<collectives>
      SOAK_SIZES=b1,b2  TIME_K TIME_REPLAYS PROFILE=1

Run inside a 1Cat-vLLM image with exactly four GPUs visible (verify/car_gate.sh does that). One JSON line
per size on rank 0's stdout; `digest` must match across GPU sets of the same model (verify_car.sh).
"""

import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
DEFAULT_SIZES = [16, 32, 48, 256, 1024, 2048, 2064, 2688, 4096, 5120, 5136, 8192, 10240, 10752, 16384,
                 20480, 25600, 32768, 40960, 65536, 81920, 81936, 131072, 262144, 393216, 524288, 786432,
                 1 << 20, 2 << 20, 4 << 20, (8 << 20) - 16]


def env_int(k: str, d: int) -> int:
    return int(float(os.environ.get(k, d)))


def env_sizes(k: str, d: list[int]) -> list[int]:
    v = os.environ.get(k)
    return [int(float(s)) for s in v.split(",")] if v else d


def inputs(n: int, salt: int, data: str, dev: torch.device) -> list[torch.Tensor]:
    """All four ranks' inputs for one (size, salt), identical on every rank. `int` keeps sums exact; `randn`
    exercises rounding; `special` adds signed zeros, infinities, NaN and the push kernel's sentinel bits."""
    i = torch.arange(n, device=dev, dtype=torch.int64)
    xs = []
    for r in range(WORLD):
        if data == "int":
            xs.append(((r + 1) * 3 + (i + salt) % 7).to(torch.float16))
            continue
        g = torch.Generator(device=dev)
        g.manual_seed((n * 1_000_003 + salt * 7919 + r * 104_729) % (1 << 62))
        x = (torch.randn(n, generator=g, device=dev) * 0.03).to(torch.float16)
        if data == "special" and n >= 8:
            k = salt % (n - 7)
            vals = [0.0, -0.0, float("inf"), float("-inf"), 65504.0, -65504.0, 6e-8]
            x[k:k + 7] = torch.tensor(vals, dtype=torch.float16, device=dev).roll(r)
            if r == 2:
                x.view(torch.int16)[(k + 3) % n] = 0x7F7F  # the push kernel's "not arrived" sentinel
        xs.append(x)
    return xs


def reference(xs: list[torch.Tensor]) -> torch.Tensor:
    acc = xs[0].float()
    for x in xs[1:]:
        acc = acc + x.float()
    return acc.to(torch.float16)


def absum(xs: list[torch.Tensor]) -> torch.Tensor:
    return sum(x.float().abs() for x in xs)


def compare(out: torch.Tensor, ref: torch.Tensor, absum: torch.Tensor | None = None) -> tuple[int, int]:
    """(wrong, bitdiff): wrong = outside fp16 rounding of the reference, or a NaN/inf mismatch. With absum
    (sum of |inputs|, for NCCL, which rounds to fp16 after every ring step), the bound is four fp16
    roundings of partial sums no larger than absum."""
    o, r = out.float(), ref.float()
    both_nan = torch.isnan(o) & torch.isnan(r)
    finite = torch.isfinite(r)
    tol = r.abs() * 2.0 ** -9 + 6e-8 if absum is None else absum * 4 * 2.0 ** -11 + 4 * 6e-8
    ok = both_nan | (finite & ((o - r).abs() <= tol)) | (~finite & ~torch.isnan(r) & (o == r))
    bits = (out.view(torch.int16) != ref.view(torch.int16)) & ~both_nan
    return int((~ok).sum().item()), int(bits.sum().item())


def make_ca(mode: str, cpu_group: dist.ProcessGroup, dev: torch.device):
    """Build vLLM's CustomAllreduce with the topology check forced, or return None for the NCCL baseline."""
    if mode == "nccl":
        return None
    if mode in ("car-nopush", "negctl"):  # negctl: push registration itself refuses a not-fully-connected C++ side
        os.environ["VLLM_SM70_TP4_PUSH_ALLREDUCE"] = "0"
    from vllm import _custom_ops as ops
    from vllm.distributed.device_communicators import custom_all_reduce as car
    from vllm.platforms import current_platform

    current_platform.is_fully_connected = lambda ids: True
    if mode == "negctl":
        real_init = ops.init_custom_ar
        car.ops.init_custom_ar = lambda meta, rank_data, rank, fc: real_init(meta, rank_data, rank, False)
    ca = car.CustomAllreduce(group=cpu_group, device=dev)
    if ca.disabled:
        raise RuntimeError("CustomAllreduce stayed disabled despite the forced topology")
    return ca


def ar(ca, x: torch.Tensor) -> torch.Tensor:
    """One out-of-place all-reduce the way vLLM issues it."""
    if ca is None:
        y = x.clone()
        dist.all_reduce(y)
        return y
    return ca.custom_all_reduce(x)


def worker(rank: int, mode: str, port: int) -> None:
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            device_id=dev, timeout=timedelta(seconds=300))
    cpu_group = dist.new_group(backend="gloo")
    ca = make_ca(mode, cpu_group, dev)
    stream = torch.cuda.Stream()
    data = os.environ.get("DATA", "randn")
    eager_iters, chain = env_int("EAGER_ITERS", 50), env_int("CHAIN", 8)
    chain_replays, skew = env_int("CHAIN_REPLAYS", 30), os.environ.get("SKEW", "1") == "1"
    soak, soak_sizes = env_int("SOAK", 0), set(env_sizes("SOAK_SIZES", [5120, 10240]))
    time_k, time_replays = env_int("TIME_K", 50), env_int("TIME_REPLAYS", 20)
    perturb = mode == "perturb"

    def mine(xs: list[torch.Tensor]) -> torch.Tensor:
        x = xs[rank].clone()
        if perturb and rank == 1:
            x[0] += 1.0
        return x

    def gather_sum(v: int) -> int:
        t = torch.tensor([v], dtype=torch.int64)
        dist.all_reduce(t, group=cpu_group)
        return int(t.item())

    def same_everywhere(h: int) -> bool:
        hs: list[int | None] = [None] * WORLD
        dist.all_gather_object(hs, h, group=cpu_group)
        return len(set(hs)) == 1

    def digest(t: torch.Tensor, h: int) -> int:
        w = torch.arange(t.numel(), device=t.device, dtype=torch.int64) % 65521 + 1
        return (h * 1_000_003 + int((t.view(torch.int16).to(torch.int64) * w).sum().item())) % (1 << 61)

    for nbytes in env_sizes("SIZES", DEFAULT_SIZES):
        n = nbytes // 2
        row: dict = {"mode": mode, "data": data, "bytes": nbytes}
        probe = inputs(n, 0, data, dev)[rank]
        if ca is not None and not ca.should_custom_ar(probe):
            row["skip"] = "custom AR declines this size"
            if rank == 0:
                print(json.dumps(row), flush=True)
            continue
        h = 0

        # 1. eager: a caller-owned poisoned output (a fresh empty_like could reuse a block that still holds
        # last iteration's correct sums)
        wrong = bitdiff = 0
        out = torch.empty(n, dtype=torch.float16, device=dev)
        for it in range(eager_iters):
            xs = inputs(n, it, data, dev)
            out.fill_(float("nan"))
            if ca is None:
                out.copy_(mine(xs))
                dist.all_reduce(out)
            else:
                ca.all_reduce(mine(xs), out=out, registered=False)
            torch.cuda.synchronize()
            w, b = compare(out, reference(xs), absum(xs) if ca is None else None)
            wrong, bitdiff, h = wrong + w, bitdiff + b, digest(out, h)
        row["eager_wrong"], row["eager_bitdiff"] = gather_sum(wrong), gather_sum(bitdiff)

        # 2. graph chain, with rank skew before each replay, then the soak and the stale self-test
        xin = torch.empty(n, dtype=torch.float16, device=dev)
        xin.copy_(mine(inputs(n, 0, data, dev)))
        k = 50 if nbytes in soak_sizes and soak else chain
        g = torch.cuda.CUDAGraph()
        outs: list[torch.Tensor] = []
        with ca.capture() if ca is not None else torch.no_grad():
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=stream):
                y = ar(ca, xin)
                outs.append(y)
                for _ in range(k - 1):
                    y = ar(ca, xin + 0 * y)
                    outs.append(y)

        def replay(salt: int, run: bool = True) -> tuple[int, int, int, int]:
            """(wrong, bitdiff, digest, positions checked that a skipped replay must fail)."""
            xs = inputs(n, salt, data, dev)
            xin.copy_(mine(xs))
            for o in outs:
                o.fill_(float("nan"))
            torch.cuda.synchronize()
            if run:
                if skew:  # ranks enter the collective up to ~0.1 ms apart, a different order every replay
                    torch.cuda._sleep(((salt * 7 + rank * 3) % 5) * 40_000)
                g.replay()
            torch.cuda.synchronize()
            ref, ab = reference(xs), absum(xs) if ca is None else None
            w = b = 0
            checked = outs[:1] if data == "special" else outs  # 0*inf is NaN: specials stop at the first AR
            for o in checked:
                ww, bb = compare(o, ref, ab)
                w, b = w + ww, b + bb
            return w, b, digest(outs[0], 0), len(checked) * int((~torch.isnan(ref.float())).sum().item())

        replays = chain_replays
        if nbytes in soak_sizes and soak:
            replays = max(chain_replays, soak // k)
        wrong = bitdiff = 0
        for rep in range(replays):
            w, b, d, _ = replay(1000 + rep)
            wrong, bitdiff, h = wrong + w, bitdiff + b, (h * 31 + d) % (1 << 61)
        row["graph_collectives"] = replays * k
        row["graph_wrong"], row["graph_bitdiff"] = gather_sum(wrong), gather_sum(bitdiff)
        stale_w, _, _, must_fail = replay(999_999, run=False)
        row["stale_caught"] = gather_sum(int(stale_w == must_fail)) == WORLD
        row["rank_hash_equal"] = same_everywhere(h)
        del g, outs

        # 3. latency: K independent all-reduces captured back to back, and the kernels that ran
        xt = torch.empty(n, dtype=torch.float16, device=dev)
        xt.copy_(mine(inputs(n, 0, data, dev)))
        g = torch.cuda.CUDAGraph()
        with ca.capture() if ca is not None else torch.no_grad():
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=stream):
                keep = [ar(ca, xt) for _ in range(time_k)]
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(time_replays):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        t = torch.tensor([t0.elapsed_time(t1) * 1000.0 / (time_replays * time_k)], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=cpu_group)
        row["us"] = round(float(t.item()), 2)
        if os.environ.get("PROFILE", "1") == "1":  # every rank replays: a collective waits for all four
            dist.barrier(group=cpu_group)
            if rank == 0:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    g.replay()
                    torch.cuda.synchronize()
                row["kernels"] = sorted({e.key.split("<")[0].split("(")[0][:60] for e in prof.key_averages()
                                         if any(s in e.key.lower() for s in ("reduce", "nccl", "push"))})
            else:
                g.replay()
                torch.cuda.synchronize()
        del g, keep

        if rank == 0:
            bad = row["eager_wrong"] or row["graph_wrong"] or not row["rank_hash_equal"] or not row["stale_caught"]
            row["verdict"] = "WRONG" if bad else "OK"
            row["digest"] = h  # same seeds on another GPU set of the same model must give the same bits
            print(json.dumps(row), flush=True)

    dist.barrier(group=cpu_group)
    if ca is not None:
        ca.close()
    dist.destroy_process_group()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "car"
    if mode not in ("nccl", "car", "car-nopush", "negctl", "perturb"):
        sys.exit(f"unknown MODE {mode}")
    if torch.cuda.device_count() != WORLD:
        sys.exit(f"need exactly {WORLD} visible GPUs, got {torch.cuda.device_count()}")
    port = 29500 + int(time.time()) % 1000
    mp.spawn(worker, args=(mode, port), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
