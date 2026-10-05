"""Tests for periodic NodeBackends re-announcement.

NodeBackends is gathered exactly once at startup; if that first message is
lost (e.g. the router is still connecting when the gatherer starts), the
node advertises no backends and placement rejects every cycle containing
it — permanently, until the app restarts. ``_monitor_node_backends``
re-sends it on an interval so the advertisement self-heals within one poll
interval (fix 9299909c, same class as the NodeConfig re-announce dead4e88
and the NodeApiInfo re-announce 36d58099).

These tests cover the periodic re-announcement loop, which had no direct
coverage anywhere in history (verified: `git log --all -S
'_monitor_node_backends' -- '*test*'` is empty).

The loop assertions require MORE THAN ONE iteration. Asserting ``calls >= 1``
would also pass for a loop that returns after a single send, so it does not
test periodicity at all (proved by a negative control).

TIMING: the loop is stopped by CANCELLING IT once enough iterations have
happened, not by a wall-clock budget. An earlier revision ran the loop under
``anyio.move_on_after(0.2)`` and asserted 3 iterations of a 10ms tick. That
couples the assertion to machine speed: under CPU contention a tick can
exceed its budget, the deadline fires first, and a perfectly correct loop
fails (observed once in ~40 runs on a loaded 10-core host, and it is inherent
to shared-CI-runner timing). The deadline is now only a safety net that
should never fire; the assertions themselves are load-independent.
"""

import anyio
import pytest

import exo.utils.info_gatherer.info_gatherer as ig
from exo.utils.channels import channel

TICK_SECONDS = 0.0
MIN_ITERATIONS = 3
# Safety net only: the cancel below ends the loop long before this.
DEADLINE_SECONDS = 30.0


class TestMonitorNodeBackends:
    async def test_monitor_node_backends_reannounces_periodically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each loop iteration re-gathers and re-sends NodeBackends, so a lost
        startup announcement is repaired within one poll interval."""
        sender, receiver = channel[ig.GatheredInfo]()

        calls = 0

        with anyio.CancelScope() as scope:

            async def fake_gather() -> ig.NodeBackends:
                nonlocal calls
                calls += 1
                if calls > MIN_ITERATIONS:
                    scope.cancel()  # enough full iterations are done
                return ig.NodeBackends(backends=[ig.Backend.MlxCpu])

            monkeypatch.setattr(ig.NodeBackends, "gather", fake_gather)

            gatherer = ig.InfoGatherer(info_sender=sender)
            with anyio.move_on_after(DEADLINE_SECONDS):
                await gatherer._monitor_node_backends(TICK_SECONDS)  # pyright: ignore[reportPrivateUsage]

        assert calls >= MIN_ITERATIONS, (
            f"expected >= {MIN_ITERATIONS} re-announces, got {calls}"
        )
        assert sender.statistics().current_buffer_used >= MIN_ITERATIONS
        for _ in range(MIN_ITERATIONS):
            sent = receiver.receive_nowait()
            assert isinstance(sent, ig.NodeBackends)
            assert ig.Backend.MlxCpu in sent.backends

    async def test_monitor_node_backends_survives_gather_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gather failure logs and retries on the next tick instead of
        killing the monitor task (which would leave the node advertising no
        backends for the rest of the process lifetime)."""
        sender, receiver = channel[ig.GatheredInfo]()

        calls = 0

        with anyio.CancelScope() as scope:

            async def flaky_gather() -> ig.NodeBackends:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("nvml probe failed")
                if calls > MIN_ITERATIONS:
                    scope.cancel()
                return ig.NodeBackends(backends=[ig.Backend.MlxCpu])

            monkeypatch.setattr(ig.NodeBackends, "gather", flaky_gather)

            gatherer = ig.InfoGatherer(info_sender=sender)
            with anyio.move_on_after(DEADLINE_SECONDS):
                await gatherer._monitor_node_backends(TICK_SECONDS)  # pyright: ignore[reportPrivateUsage]

        # Must get past the first (failing) gather and keep retrying.
        assert calls >= MIN_ITERATIONS, (
            f"expected >= {MIN_ITERATIONS} attempts, got {calls}"
        )
        assert sender.statistics().current_buffer_used >= MIN_ITERATIONS - 1
        sent = receiver.receive_nowait()
        assert isinstance(sent, ig.NodeBackends)
