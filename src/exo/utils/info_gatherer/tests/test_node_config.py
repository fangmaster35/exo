"""Tests for periodic NodeConfig (identity) re-announcement.

NodeConfig is gathered exactly once at startup; if that first message is
lost (e.g. the router is still connecting when the gatherer starts), the
node joins elections and is "discovered" by peers but never appears in any
node's topology. ``_monitor_node_config`` re-sends it on an interval so the
identity announcement self-heals within one poll interval (fix dead4e88,
same class as the NodeBackends re-announce 9299909c).

The loop assertions below require MORE THAN ONE iteration. An earlier
revision asserted ``calls >= 1``, which a loop that returns after a single
iteration also satisfies — so the "periodic" behaviour was never actually
verified (proved by a negative control that mutated each loop to
``return`` immediately and still passed). The ``calls >= 3`` assertions are
what make this a periodicity test.

TIMING: the loop is stopped by CANCELLING IT once enough iterations have
happened, not by a wall-clock budget. An earlier revision ran the loop
under ``anyio.move_on_after(0.1)`` and asserted 3 iterations of a 10ms tick.
That couples the assertion to machine speed: under CPU contention a tick
can exceed its budget, the deadline fires first, and a perfectly correct
loop fails (observed once in ~40 runs on a loaded 10-core host, and it is
inherent to shared-CI-runner timing). The deadline is now only a safety net
that should never fire; the assertions themselves are load-independent.
"""

import anyio
import pytest

import exo.utils.info_gatherer.info_gatherer as ig
from exo.utils.channels import channel

TICK_SECONDS = 0.0
MIN_ITERATIONS = 3
# Safety net only: the cancel below ends the loop long before this.
DEADLINE_SECONDS = 30.0


class TestMonitorNodeConfig:
    async def test_monitor_node_config_reannounces_periodically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each loop iteration re-gathers and re-sends NodeConfig, so a lost
        startup announcement is repaired within one poll interval."""
        sender, receiver = channel[ig.GatheredInfo]()

        calls = 0

        with anyio.CancelScope() as scope:

            async def fake_gather() -> ig.NodeConfig | None:
                nonlocal calls
                calls += 1
                if calls > MIN_ITERATIONS:
                    scope.cancel()  # enough full iterations are done
                return ig.NodeConfig()

            monkeypatch.setattr(ig.NodeConfig, "gather", fake_gather)

            gatherer = ig.InfoGatherer(info_sender=sender)
            with anyio.move_on_after(DEADLINE_SECONDS):
                await gatherer._monitor_node_config(TICK_SECONDS)  # pyright: ignore[reportPrivateUsage]

        # >1 iteration: proves the announcement repeats, not just that it fired.
        assert calls >= MIN_ITERATIONS, (
            f"expected >= {MIN_ITERATIONS} re-announces, got {calls}"
        )
        assert sender.statistics().current_buffer_used >= MIN_ITERATIONS
        for _ in range(MIN_ITERATIONS):
            sent = receiver.receive_nowait()
            assert isinstance(sent, ig.NodeConfig)

    async def test_monitor_node_config_skips_when_gather_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed gather (None) sends nothing but keeps the loop alive."""
        sender, _receiver = channel[ig.GatheredInfo]()

        calls = 0

        with anyio.CancelScope() as scope:

            async def fake_gather() -> ig.NodeConfig | None:
                nonlocal calls
                calls += 1
                if calls > MIN_ITERATIONS:
                    scope.cancel()
                return None

            monkeypatch.setattr(ig.NodeConfig, "gather", fake_gather)

            gatherer = ig.InfoGatherer(info_sender=sender)
            with anyio.move_on_after(DEADLINE_SECONDS):
                await gatherer._monitor_node_config(TICK_SECONDS)  # pyright: ignore[reportPrivateUsage]

        # The loop must keep re-gathering even though every gather failed —
        # a one-shot loop would stop after the first failure and never recover.
        assert calls >= MIN_ITERATIONS, (
            f"expected >= {MIN_ITERATIONS} attempts, got {calls}"
        )
        assert sender.statistics().current_buffer_used == 0
