"""car_patch - write a copy of 1Cat-vLLM's custom_all_reduce.py whose custom all-reduce runs TP=4 over PCIe P2P.

vLLM enables its IPC custom all-reduce for more than two GPUs only if NVML reports NVLink between every
pair (vllm/platforms/cuda.py is_fully_connected), and 1Cat's C++ dispatch launches nothing for
world_size > 2 when told the GPUs are not fully connected, so lifting the Python gate alone returns
garbage. 1Cat's SM70 TP4 push kernel needs only peer stores, which PCIe P2P carries: verify/car_gate.py
measured it exact across a PCIe root complex and 3x faster than NCCL at decode sizes, while 1Cat's pull
path (peer reads across the root complex) loses to NCCL. The copy adds three env switches and changes
nothing else, so the image stays untouched and a read-only bind mount puts the copy over the original:

  VLLM_CAR_P2P_MESH=1    treat the group as fully connected when every rank can reach every peer by P2P
                         (all ranks must agree, or they would take different all-reduce paths)
  VLLM_CAR_MAX_BYTES=N   send only all-reduces smaller than N bytes through custom AR; NCCL takes the rest
                         (1Cat's push kernel covers up to 81920 bytes: N=81921)
  VLLM_CAR_GRAPH_ONLY=1  only all-reduces being captured into a CUDA graph; 1Cat runs push only there, and
                         its eager path is the pull kernel, which loses to NCCL across the root complex

With none of them set, the patched file behaves exactly like the original.

Each insertion sits next to an anchor copied from 1Cat's file. An anchor that is missing or repeated means
a 1Cat revision this was not written against, so the script refuses rather than guess. Checked against
1Cat-vLLM 357d07bcb and c4f6245f8; patches/ holds the resulting diffs.

Usage: python3 car_patch.py <custom_all_reduce.py> <out.py>
"""

import sys

GATE = "        fully_connected = current_platform.is_fully_connected(physical_device_ids)\n"
GATE_ADD = '''        if not fully_connected and os.environ.get("VLLM_CAR_P2P_MESH") == "1":
            # vllm-pcie-allreduce: 1Cat's push kernels need only peer stores, which PCIe P2P
            # carries (verify/car_gate.py). Every rank must agree on the path it takes.
            local_ok = all(
                torch.cuda.can_device_access_peer(device.index, device_ids.index(p))
                for p in physical_device_ids
                if p != physical_device_id
            )
            p2p_ok: list[bool | None] = [None] * world_size
            dist.all_gather_object(p2p_ok, local_ok, group=self.group)
            fully_connected = all(ok is True for ok in p2p_ok)
            logger.info(
                "VLLM_CAR_P2P_MESH: custom allreduce treats GPUs %s as fully connected: %s",
                physical_device_ids,
                fully_connected,
            )
'''
CAP = """        self.dispatch_max_size = (
            min(max_size, 8192 * 1024) if long_prefill_fusion_enabled else max_size
        )
"""
CAP_ADD = """        if os.environ.get("VLLM_CAR_MAX_BYTES"):
            # vllm-pcie-allreduce: above the push kernel's sizes the cross-root pull path loses to NCCL
            self.dispatch_max_size = min(
                self.dispatch_max_size, int(os.environ["VLLM_CAR_MAX_BYTES"])
            )
"""


ADMIT = """        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.dispatch_max_size
"""
ADMIT_ADD = """        if (
            os.environ.get("VLLM_CAR_GRAPH_ONLY") == "1"
            and not torch.cuda.is_current_stream_capturing()
        ):
            # vllm-pcie-allreduce: outside capture 1Cat runs pull, which loses to NCCL across PCIe
            return False
"""


def patch(src: str) -> str:
    """Insert each addition after its anchor (ADMIT_ADD before its own); an anchor that is missing or
    repeated means a different vLLM version, so refuse rather than guess."""
    for anchor, add, before in ((GATE, GATE_ADD, False), (CAP, CAP_ADD, False), (ADMIT, ADMIT_ADD, True)):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor found {src.count(anchor)} times, expected once:\n{anchor}")
        src = src.replace(anchor, add + anchor if before else anchor + add)
    return src


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src_path, out_path = sys.argv[1], sys.argv[2]
    with open(src_path, encoding="utf-8") as f:
        out = patch(f.read())
    compile(out, out_path, "exec")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
