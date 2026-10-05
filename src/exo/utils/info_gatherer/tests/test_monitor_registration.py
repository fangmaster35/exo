"""Wiring test: run() must actually REGISTER the periodic re-announce tasks.

The monitor tests in test_node_config.py / test_node_backends_monitor.py call
``_monitor_node_config`` / ``_monitor_node_backends`` directly, so they prove
the loop bodies behave correctly but are blind to whether run() ever STARTS
them. Deleting the two ``tg.start_soon(...)`` lines in run() therefore left
the whole suite green while the production fix was completely inert — the
node would still advertise only once at startup, which is the exact bug
9299909c/dead4e88 fix.

This test closes that gap by observing the task group run() populates.

NOTE: ``_tg`` is a dataclass field with ``default_factory``, so it is assigned
per-instance in ``__init__``. Patching it on the CLASS is silently overridden
by that assignment (and the real TaskGroup then runs the monitors forever, so
the test HANGS rather than fails). It must be replaced on the instance.
"""

import pytest

import exo.utils.info_gatherer.info_gatherer as ig
from exo.utils.channels import channel


class RecordingTaskGroup:
    """Minimal stand-in capturing start_soon(fn, ...) registrations.

    Nothing is actually started, so run() returns promptly.
    """

    def __init__(self) -> None:
        self.started: list[object] = []

    async def __aenter__(self) -> "RecordingTaskGroup":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def start_soon(self, fn: object, *args: object) -> None:
        self.started.append(fn)

    def cancel_tasks(self) -> None:
        pass


class TestMonitorRegistration:
    async def test_run_registers_periodic_node_monitors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() starts both re-announce monitors.

        A negative control that removes either start_soon line fails this
        test, which the loop-body tests cannot do.
        """
        sender, _receiver = channel[ig.GatheredInfo]()

        async def fake_config_gather() -> ig.NodeConfig | None:
            return None

        async def fake_backends_gather() -> ig.NodeBackends:
            return ig.NodeBackends(backends=[ig.Backend.MlxCpu])

        monkeypatch.setattr(ig.NodeConfig, "gather", fake_config_gather)
        monkeypatch.setattr(ig.NodeBackends, "gather", fake_backends_gather)

        gatherer = ig.InfoGatherer(info_sender=sender)
        tg = RecordingTaskGroup()
        gatherer._tg = tg  # type: ignore[assignment]  # instance attr: survives the dataclass __init__

        await gatherer.run()

        names = {getattr(fn, "__name__", repr(fn)) for fn in tg.started}
        assert "_monitor_node_config" in names, (
            f"run() never registered _monitor_node_config; registered: {sorted(names)}"
        )
        assert "_monitor_node_backends" in names, (
            f"run() never registered _monitor_node_backends; registered: {sorted(names)}"
        )
