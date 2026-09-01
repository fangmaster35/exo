"""Unit tests for bootstrap-peers election gating (t_2809b765).

Verifies that when --bootstrap-peers is configured, the initial election
campaign is deferred until at least one peer connects or the grace period
expires, preventing split-brain on simultaneous restart.
"""

import anyio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.anyio
async def test_bootstrap_signals_on_first_connection():
    """Bootstrap should signal done on FIRST successful peer connection,
    not wait for all peers."""
    from exo.main import Node

    mock_router = AsyncMock()
    # Round 1: peer1 fails, peer2 succeeds (signals event), peer3 succeeds.
    # Round 2: peer1 retry succeeds -> loop exits
    mock_router.connect_peer = AsyncMock(
        side_effect=[False, True, True, True]
    )

    node = MagicMock(spec=Node)
    node.bootstrap_peers = ["10.0.0.1:52414", "10.0.0.2:52414", "10.0.0.3:52414"]
    node.router = mock_router
    node._bootstrap_peers_loop = Node._bootstrap_peers_loop.__get__(node, Node)

    with patch("exo.main.anyio.sleep", new=AsyncMock()):
        await node._bootstrap_peers_loop()

    assert node._bootstrap_done.is_set(), (
        "Bootstrap should signal done after first successful connection"
    )
    assert mock_router.connect_peer.call_count == 4


@pytest.mark.anyio
async def test_bootstrap_exponential_backoff():
    """Bootstrap retry delay should follow exponential backoff with jitter."""
    from exo.main import Node

    mock_router = AsyncMock()
    # Fail 4 times, then succeed on 5th call so the loop exits naturally
    call_count = 0

    async def connect_side_effect(host, port):
        nonlocal call_count
        call_count += 1
        return call_count >= 5  # succeed on 5th call

    mock_router.connect_peer = AsyncMock(side_effect=connect_side_effect)

    node = MagicMock(spec=Node)
    node.bootstrap_peers = ["10.0.0.1:52414"]
    node.router = mock_router
    node._bootstrap_peers_loop = Node._bootstrap_peers_loop.__get__(node, Node)

    sleep_times = []
    original_sleep = anyio.sleep

    async def capture_sleep(delay):
        sleep_times.append(delay)
        await original_sleep(0)  # don't actually wait

    with patch("exo.main.anyio.sleep", side_effect=capture_sleep):
        await node._bootstrap_peers_loop()

    # 4 failures -> 4 sleeps before the 5th call succeeds
    assert len(sleep_times) == 4, f"Expected 4 sleeps, got {len(sleep_times)}: {sleep_times}"
    assert 1.5 <= sleep_times[0] <= 2.5, f"First retry ~2s, got {sleep_times[0]}"
    assert 3.0 <= sleep_times[1] <= 5.0, f"Second retry ~4s, got {sleep_times[1]}"
    assert 6.0 <= sleep_times[2] <= 10.0, f"Third retry ~8s, got {sleep_times[2]}"
    assert 12.0 <= sleep_times[3] <= 20.0, f"Fourth retry ~16s, got {sleep_times[3]}"
    assert node._bootstrap_done.is_set()


@pytest.mark.anyio
async def test_deferred_election_starts_after_bootstrap():
    """Election should start immediately once bootstrap signals done."""
    from exo.main import Node

    node = MagicMock(spec=Node)
    node._bootstrap_done = anyio.Event()
    node.election = AsyncMock()
    node.election.run = AsyncMock()
    node._deferred_election_start = Node._deferred_election_start.__get__(node, Node)

    node._bootstrap_done.set()

    await node._deferred_election_start()

    node.election.run.assert_called_once()


@pytest.mark.anyio
async def test_deferred_election_grace_period_timeout():
    """Election should start after grace period even if no peer connects."""
    from exo.main import Node

    node = MagicMock(spec=Node)
    node._bootstrap_done = anyio.Event()
    node.election = AsyncMock()
    node.election.run = AsyncMock()
    node._deferred_election_start = Node._deferred_election_start.__get__(node, Node)

    import exo.main as main_mod
    original_move_on = anyio.move_on_after

    with patch.object(main_mod.anyio, "move_on_after", lambda t: original_move_on(0.1)):
        await node._deferred_election_start()

    node.election.run.assert_called_once()
    assert not node._bootstrap_done.is_set()


@pytest.mark.anyio
async def test_no_bootstrap_peers_no_gate():
    """Without bootstrap peers, no gating event is created."""
    from exo.main import Node

    node = MagicMock(spec=Node)
    node.bootstrap_peers = []
    node._bootstrap_peers_loop = Node._bootstrap_peers_loop.__get__(node, Node)

    await node._bootstrap_peers_loop()

    assert not hasattr(node, "_bootstrap_done"), (
        "No bootstrap_done event when no peers configured"
    )
