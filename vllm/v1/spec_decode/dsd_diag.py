# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribution diagnostics for dynamic speculative decoding.

Enabled with ``VLLM_DSD_DIAG=1``. The target model's cudagraph dispatch is
already covered by ``--cudagraph-metrics``; this covers the drafter side:

- which cudagraph runtime mode the drafter forward dispatches to, per K;
- how much GPU time the drafter forward costs, per K;
- how often K resolves to 0, and how long those runs are.

Timing uses a lazily harvested CUDA event queue, so no step blocks on the
device.
"""

import atexit
import os
from collections import Counter, deque
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_PENDING = 512


class _KStat:
    def __init__(self) -> None:
        self.forwards = 0
        self.timed = 0
        self.total_ms = 0.0
        self.modes: Counter[str] = Counter()


class DSDDiagnostics:
    """Per-worker collector. Every method is a no-op unless enabled."""

    def __init__(self) -> None:
        self.enabled = os.environ.get("VLLM_DSD_DIAG", "0") == "1"
        self.interval = int(os.environ.get("VLLM_DSD_DIAG_INTERVAL", "1000"))
        self._per_k: dict[int, _KStat] = {}
        self._pending: deque[tuple[torch.cuda.Event, torch.cuda.Event, int]] = deque()
        self._zero_runs: Counter[int] = Counter()
        self._run = 0
        self._steps = 0
        if self.enabled:
            atexit.register(self.log)

    def _stat(self, k: int) -> _KStat:
        if (stat := self._per_k.get(k)) is None:
            stat = self._per_k[k] = _KStat()
        return stat

    def observe_dispatch(self, k: int, runtime_mode: object) -> None:
        if not self.enabled:
            return
        stat = self._stat(k)
        stat.forwards += 1
        stat.modes[str(runtime_mode)] += 1

        if k == 0:
            self._run += 1
        elif self._run:
            self._zero_runs[self._run] += 1
            self._run = 0

        self._steps += 1
        if self.interval and self._steps % self.interval == 0:
            self.log()

    @contextmanager
    def time_forward(self, k: int):
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        self._harvest()
        if len(self._pending) >= _MAX_PENDING:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((start, end, k))

    def _harvest(self, drain: bool = False) -> None:
        while self._pending:
            start, end, k = self._pending[0]
            if not drain and not end.query():
                return
            self._pending.popleft()
            if drain:
                end.synchronize()
            stat = self._stat(k)
            stat.timed += 1
            stat.total_ms += start.elapsed_time(end)

    def log(self) -> None:
        if not self.enabled:
            return
        self._harvest(drain=True)
        if not self._per_k:
            return

        lines = [
            "",
            "**Dynamic SD drafter diagnostics:**",
            "",
            "| K | Forwards | Timed | Mean ms | Total ms | Cudagraph modes |",
            "|---|---|---|---|---|---|",
        ]
        for k in sorted(self._per_k):
            s = self._per_k[k]
            mean = f"{s.total_ms / s.timed:.3f}" if s.timed else "n/a"
            modes = ", ".join(f"{m}x{c}" for m, c in s.modes.most_common())
            lines.append(
                f"| {k} | {s.forwards} | {s.timed} | {mean} | "
                f"{s.total_ms:.1f} | {modes} |"
            )

        total = sum(s.forwards for s in self._per_k.values())
        if (zero := self._per_k.get(0)) is not None and total:
            lines += [
                "",
                f"K=0: {zero.forwards}/{total} drafter steps "
                f"({100.0 * zero.forwards / total:.1f}%), "
                f"{zero.total_ms / 1000.0:.1f}s of drafter GPU time.",
            ]

        runs = Counter(self._zero_runs)
        if self._run:
            runs[self._run] += 1
        if runs:
            spans = ", ".join(f"{n}x{c}" for n, c in sorted(runs.items()))
            mean_run = sum(n * c for n, c in runs.items()) / sum(runs.values())
            lines += [
                f"K=0 run lengths: {spans}",
                f"Mean K=0 run: {mean_run:.2f} steps (the fold factor a "
                f"deferred KV catch-up would buy).",
            ]

        logger.info("\n".join(lines))


_diag: DSDDiagnostics | None = None


def get_dsd_diagnostics() -> DSDDiagnostics:
    global _diag
    if _diag is None:
        _diag = DSDDiagnostics()
    return _diag
