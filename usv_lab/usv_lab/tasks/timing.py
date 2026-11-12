"""Where a step's time goes: wall time per named part, per policy step.

Off by default, and then every timed call costs one `if`. When on, each part is
bracketed by GPU synchronisation (the CPU waits for queued GPU work), so its time
includes the GPU work it launched. Those waits slow the step down a little, so
throughput is measured with the timer off and the breakdown with it on.

The renderer's own GPU work does not always finish inside `sim.render()`: some of
it may be waited for later, when camera images are read. Count render + sensor
reading together as the camera's cost.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class StepTimer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.names: list[str] = []                       # every part seen, in first-seen order
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._step: dict[str, float] = defaultdict(float)

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self._step[name] += time.perf_counter() - t0
            if name not in self.names:
                self.names.append(name)

    def wrap(self, fn, name: str):
        """Time someone else's function (e.g. the engine's `sim.step`) without editing it."""

        def timed(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            with self(name):
                return fn(*args, **kwargs)

        return timed

    def end_step(self) -> None:
        """Close one policy step. Parts that did not run this step (e.g. lidar, every 3rd step) count as 0."""
        for name in self.names:
            self.samples[name].append(self._step.get(name, 0.0))
        self._step.clear()

    def clear(self) -> None:
        self.names.clear()
        self.samples.clear()
        self._step.clear()

    def series(self, name: str) -> torch.Tensor:
        """[steps] ms per policy step for one part; steps before it first ran count as 0."""
        n = max((len(v) for v in self.samples.values()), default=0)
        v = self.samples.get(name, [])
        return torch.tensor([0.0] * (n - len(v)) + list(v), dtype=torch.float64) * 1e3

    def summary(self) -> dict[str, tuple[float, float]]:
        """{part: (mean ms per policy step, 95th percentile ms)}."""
        out = {}
        for name in self.names:
            v = self.series(name)
            out[name] = (float(v.mean()), float(torch.quantile(v, 0.95))) if v.numel() else (0.0, 0.0)
        return out
