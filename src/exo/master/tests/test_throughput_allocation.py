# pyright: reportAny=false, reportUnknownVariableType=false
"""Tests for throughput-based layer allocation and the chip bandwidth table.

Ported from PR #2229 (https://github.com/exo-explore/exo/pull/2229) by
@jgawronek — bandwidth-aware placement portion.
"""

import pytest

from exo.master.placement_utils import (
    allocate_layers_by_throughput,
    estimate_memory_bandwidth_gigabytes_per_second,
    find_ip_prioritised,
)
from exo.master.tests.conftest import (
    create_socket_connection,
)
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection


def test_allocate_layers_by_throughput_fills_fastest_first() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=12,
        node_throughputs=[819.0, 273.0, 800.0],
        max_layers_per_node=[4, 4, 8],
    )
    assert allocations == [4, 1, 7]
    assert sum(allocations) == 12


def test_allocate_layers_by_throughput_keeps_one_layer_per_node() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=3,
        node_throughputs=[1000.0, 1.0, 1.0],
        max_layers_per_node=[3, 3, 3],
    )
    assert allocations == [1, 1, 1]


def test_allocate_layers_by_throughput_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _ = allocate_layers_by_throughput(
            total_layers=10,
            node_throughputs=[100.0, 100.0],
            max_layers_per_node=[4, 4],
        )


def test_estimate_memory_bandwidth_matches_known_chips() -> None:
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Apple M3 Ultra")
        )
        == 819.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GB10")
        )
        == 273.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GeForce RTX 3090")
        )
        == 936.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GeForce RTX 5060 Ti")
        )
        == 448.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Unknown Chip")
        )
        is None
    )


def _two_node_topology_with_two_links(
    thunderbolt_ip: str, ethernet_ip: str
) -> tuple[Topology, NodeId, NodeId]:
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    for ip in (thunderbolt_ip, ethernet_ip):
        last_octet = int(ip.rsplit(".", 1)[1])
        topology.add_connection(
            Connection(
                source=node_a,
                sink=node_b,
                edge=create_socket_connection(last_octet),
            )
        )
    return topology, node_a, node_b


def test_find_ip_prioritised_prefers_measured_link_speed_for_ring() -> None:
    thunderbolt_ip, ethernet_ip = "169.254.0.8", "169.254.0.9"
    topology, node_a, node_b = _two_node_topology_with_two_links(
        thunderbolt_ip, ethernet_ip
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en5",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="enp1s0f0",
                    ip_address=ethernet_ip,
                    interface_type="ethernet",
                    link_speed_megabits=200_000,
                ),
            ]
        )
    }

    selected_ip = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)

    # 200 GbE with a measured speed beats thunderbolt's nominal 40 Gb/s.
    assert selected_ip == ethernet_ip


def test_find_ip_prioritised_falls_back_to_nominal_speeds_for_ring() -> None:
    thunderbolt_ip, ethernet_ip = "169.254.0.8", "169.254.0.9"
    topology, node_a, node_b = _two_node_topology_with_two_links(
        thunderbolt_ip, ethernet_ip
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en5",
                    ip_address=thunderbolt_ip,
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address=ethernet_ip,
                    interface_type="ethernet",
                ),
            ]
        )
    }

    selected_ip = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)

    # Without measured speeds the previous type preference is preserved.
    assert selected_ip == thunderbolt_ip
