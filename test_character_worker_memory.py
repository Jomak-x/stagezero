"""CPU-only checks for the isolated character worker's memory boundaries."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from character_worker_trellis import (
    GIB, GPU_ALLOCATOR_BUDGET_BYTES, _configure_gpu_budget, _StagedModelOffload,
)


class Model:
    def __init__(self, name, events):
        self.name, self.events, self.hooks = name, events, []

    def cpu(self):
        self.events.append((self.name, "cpu"))

    def to(self, device):
        self.events.append((self.name, device))

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def forward(self):
        for hook in self.hooks:
            hook(self, ())


class Pipeline:
    device = "cpu"


class WorkerMemoryTests(unittest.TestCase):
    def test_phase_transition_offloads_before_upload_and_reuses_active_model(self):
        events = []
        pipeline = Pipeline()
        pipeline.models = {name: Model(name, events) for name in ("flow", "mesh")}
        torch = SimpleNamespace(device=lambda name: name, cuda=MagicMock())
        with patch("character_worker_trellis._gpu_memory", return_value={}), patch("character_worker_trellis._stage"):
            offload = _StagedModelOffload(pipeline, torch)
            self.assertEqual(pipeline.device, "cuda")
            pipeline.models["flow"].forward()
            pipeline.models["flow"].forward()
            pipeline.models["mesh"].forward()
            offload.close()
        self.assertEqual(events, [("flow", "cpu"), ("mesh", "cpu"),
                                 ("flow", "cuda"), ("flow", "cpu"),
                                 ("mesh", "cuda"), ("mesh", "cpu")])
        self.assertIs(type(pipeline), Pipeline)
        self.assertTrue(all(not model.hooks for model in pipeline.models.values()))

    def test_insufficient_capacity_never_sets_allocator_budget(self):
        torch = SimpleNamespace(cuda=MagicMock())
        torch.cuda.mem_get_info.return_value = (13 * GIB, 48 * GIB)
        with patch("character_worker_trellis._stage") as stage:
            with self.assertRaisesRegex(RuntimeError, "Waiting for GPU capacity"):
                _configure_gpu_budget(torch, wait_seconds=0)
        self.assertEqual(stage.call_args.args[0], "waiting_gpu_memory")
        torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_capacity_recovery_applies_absolute_budget(self):
        torch = SimpleNamespace(cuda=MagicMock())
        torch.cuda.mem_get_info.side_effect = [(13 * GIB, 48 * GIB), (15 * GIB, 48 * GIB)]
        with patch("character_worker_trellis.time.sleep"), patch("character_worker_trellis._stage"), patch("character_worker_trellis._gpu_memory", return_value={}):
            _configure_gpu_budget(torch)
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(GPU_ALLOCATOR_BUDGET_BYTES / (48 * GIB))


if __name__ == "__main__":
    unittest.main()
