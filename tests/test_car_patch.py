"""Tests for car_patch.py that need neither 1Cat's file nor a GPU.

The anchors are copied from 1Cat-vLLM, so the tests embed car_patch's own anchor strings in a small
skeleton of CustomAllreduce, patch it, and execute the result against stubs. That checks the inserted
code itself: every switch is inert unless set, the P2P mesh needs every rank to agree, and the size cap
and graph-only rule compose with 1Cat's admission. The real files are covered by patches/*.diff, which
are car_patch.py's output against 1Cat-vLLM 357d07bcb and c4f6245f8.

Run from the repo root: python3 -m unittest discover -s tests -v
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import car_patch  # noqa: E402

SKELETON_HEAD = "import os\n\n\nclass CustomAllreduce:\n"
SKELETON_INIT = (
    "    def init_for_test(self, world_size, physical_device_ids, physical_device_id, device,\n"
    "                      device_ids, max_size, long_prefill_fusion_enabled):\n"
)
SKELETON_ADMIT = "    def should_custom_ar(self, inp_size):\n"


def skeleton() -> str:
    """CustomAllreduce reduced to the three anchors, in 1Cat's order."""
    return (
        SKELETON_HEAD
        + SKELETON_INIT
        + car_patch.GATE
        + "        self.fully_connected = fully_connected\n"
        + car_patch.CAP
        + "        self.world_size = world_size\n\n"
        + SKELETON_ADMIT
        + car_patch.ADMIT
        + "        return False\n"
    )


class FakeDist:
    """all_gather_object stand-in: every rank reports `others`, except this rank's own slot."""

    def __init__(self, rank: int, others: bool = True) -> None:
        self.rank, self.others = rank, others

    def all_gather_object(self, out: list, obj: object, group: object = None) -> None:
        for i in range(len(out)):
            out[i] = obj if i == self.rank else self.others


def load(peer_ok: bool = True, others: bool = True, capturing: bool = False) -> type:
    """Patch the skeleton and exec it with stubs for current_platform, torch, dist and logger."""
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            can_device_access_peer=lambda a, b: peer_ok,
            is_current_stream_capturing=lambda: capturing,
        )
    )
    namespace = {
        "current_platform": types.SimpleNamespace(is_fully_connected=lambda ids: False),
        "torch": torch,
        "dist": FakeDist(rank=0, others=others),
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    }
    exec(compile(car_patch.patch(skeleton()), "patched", "exec"), namespace)
    return namespace["CustomAllreduce"]


def build(cls: type, max_size: int = 8192 * 1024) -> object:
    ca = cls()
    ca.group = None
    ca.init_for_test(4, [0, 1, 4, 5], 0, types.SimpleNamespace(index=0), [0, 1, 4, 5], max_size, False)
    return ca


ENV_KEYS = ("VLLM_CAR_P2P_MESH", "VLLM_CAR_MAX_BYTES", "VLLM_CAR_GRAPH_ONLY")


def clean_env(**kv: str) -> mock._patch:
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(kv)
    return mock.patch.dict(os.environ, env, clear=True)


class PatchText(unittest.TestCase):
    def test_each_insertion_lands_once_next_to_its_anchor(self) -> None:
        out = car_patch.patch(skeleton())
        self.assertEqual(out.count(car_patch.GATE + car_patch.GATE_ADD), 1)
        self.assertEqual(out.count(car_patch.CAP + car_patch.CAP_ADD), 1)
        self.assertEqual(out.count(car_patch.ADMIT_ADD + car_patch.ADMIT), 1)

    def test_only_insertions_no_original_line_changes(self) -> None:
        src = skeleton()
        out = car_patch.patch(src)
        for add in (car_patch.GATE_ADD, car_patch.CAP_ADD, car_patch.ADMIT_ADD):
            out = out.replace(add, "", 1)
        self.assertEqual(out, src)

    def test_refuses_a_missing_anchor(self) -> None:
        with self.assertRaises(SystemExit):
            car_patch.patch(skeleton().replace(car_patch.CAP, ""))

    def test_refuses_a_repeated_anchor(self) -> None:
        with self.assertRaises(SystemExit):
            car_patch.patch(skeleton() + "\n    def again(self, inp_size):\n" + car_patch.ADMIT)


class Behaviour(unittest.TestCase):
    def test_no_switch_set_keeps_1cat_behaviour(self) -> None:
        with clean_env():
            ca = build(load())
            self.assertFalse(ca.fully_connected)
            self.assertEqual(ca.dispatch_max_size, 8192 * 1024)
            self.assertFalse(ca.should_custom_ar(5120))

    def test_mesh_counts_as_fully_connected_when_every_rank_reaches_every_peer(self) -> None:
        with clean_env(VLLM_CAR_P2P_MESH="1"):
            self.assertTrue(build(load(peer_ok=True, others=True)).fully_connected)

    def test_mesh_refused_when_this_rank_lacks_a_peer(self) -> None:
        with clean_env(VLLM_CAR_P2P_MESH="1"):
            self.assertFalse(build(load(peer_ok=False, others=True)).fully_connected)

    def test_mesh_refused_when_another_rank_disagrees(self) -> None:
        with clean_env(VLLM_CAR_P2P_MESH="1"):
            self.assertFalse(build(load(peer_ok=True, others=False)).fully_connected)

    def test_size_cap_admits_the_push_kernels_80_kib_and_nothing_larger(self) -> None:
        with clean_env(VLLM_CAR_P2P_MESH="1", VLLM_CAR_MAX_BYTES="81921"):
            ca = build(load(capturing=True))
            self.assertEqual(ca.dispatch_max_size, 81921)
            self.assertTrue(ca.should_custom_ar(81920))
            self.assertFalse(ca.should_custom_ar(81936))

    def test_graph_only_declines_eager_calls(self) -> None:
        with clean_env(VLLM_CAR_P2P_MESH="1", VLLM_CAR_GRAPH_ONLY="1"):
            self.assertFalse(build(load(capturing=False)).should_custom_ar(5120))
            self.assertTrue(build(load(capturing=True)).should_custom_ar(5120))


if __name__ == "__main__":
    unittest.main()
