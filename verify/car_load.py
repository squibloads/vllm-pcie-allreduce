"""car_load - keep PCIe busy while car_gate runs, so ordering is tested under load.

Failures that appear only when the fabric is loaded are a known pattern for multi-GPU collectives over
PCIe (vLLM issues 47979 and 50941). Two copy-engine streams cross the links the all-reduce uses, one in
each direction: visible GPU 0 -> GPU 2 and GPU 3 -> GPU 1. car_gate.sh runs this on LOAD_GPUS, by default
the four GPUs under test; on a larger box, idle GPUs behind the same switches or root ports also work.

Usage: python3 car_load.py <seconds>
"""

import signal
import sys
import time

import torch

MB = 64


def main() -> None:
    seconds = float(sys.argv[1])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # docker stop ends the load: still report it
    if torch.cuda.device_count() != 4:
        sys.exit("need exactly four visible GPUs")
    buf = [torch.empty(MB << 20, dtype=torch.uint8, device=f"cuda:{i}") for i in range(4)]
    pairs = [(0, 2), (3, 1)]  # visible 0 -> 2 and 3 -> 1: both directions across the same links
    streams = {s: torch.cuda.Stream(device=f"cuda:{s}") for s, _ in pairs}
    end, n, t0 = time.time() + seconds, 0, time.time()
    try:
        while time.time() < end:
            for s, d in pairs:
                with torch.cuda.stream(streams[s]):
                    buf[d].copy_(buf[s], non_blocking=True)
            n += 1
            if n % 16 == 0:
                for st in streams.values():
                    st.synchronize()
    finally:
        for st in streams.values():
            st.synchronize()
        gbs = n * len(pairs) * MB / 1024 / (time.time() - t0)
        print(f"car_load: {n * len(pairs)} copies of {MB} MiB, {gbs:.1f} GB/s total over PCIe",
              flush=True)


if __name__ == "__main__":
    main()
