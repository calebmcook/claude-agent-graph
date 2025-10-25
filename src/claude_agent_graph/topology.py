"""
Topology detection and validation utilities.

This module provides functions for detecting and validating graph topologies
using NetworkX algorithms. It supports detection of trees, DAGs, chains, stars,
and cycle graphs.
"""

from enum import Enum

import networkx as nx


class GraphTopology(str, Enum):
    """Enumeration of supported graph topology types."""

    EMPTY = "empty"
    SINGLE_NODE = "single_node"
    TREE = "tree"
    DAG = "dag"
    CHAIN = "chain"
    STAR = "star"
    CYCLE = "cycle"
    UNKNOWN = "unknown"


def has_cycles(graph: nx.DiGraph) -> bool:
    """
    Check if a directed graph has cycles.

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph has cycles, False otherwise
    """
    return not nx.is_directed_acyclic_graph(graph)


def is_connected(graph: nx.DiGraph) -> bool:
    """
    Check if a graph is weakly connected.

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph is weakly connected, False otherwise
    """
    if graph.number_of_nodes() == 0:
        return False
    return nx.is_weakly_connected(graph)


def is_tree(graph: nx.DiGraph) -> bool:
    """
    Check if a graph is a tree.

    A tree is a DAG with exactly one root node (in-degree 0) and all
    other nodes have in-degree 1 (exactly one parent).

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph is a tree, False otherwise
    """
    if graph.number_of_nodes() <= 1:
        return True

    # Must be a DAG
    if has_cycles(graph):
        return False

    # Count nodes with in-degree 0 (roots)
    root_count = sum(1 for node in graph.nodes() if graph.in_degree(node) == 0)

    # Must have exactly one root
    if root_count != 1:
        return False

    # All non-root nodes must have exactly in-degree 1 (one parent)
    for node in graph.nodes():
        if graph.in_degree(node) == 0:
            # This is the root, skip
            continue
        if graph.in_degree(node) != 1:
            return False

    # Must be weakly connected
    return is_connected(graph)


def is_dag(graph: nx.DiGraph) -> bool:
    """
    Check if a graph is a directed acyclic graph (DAG).

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph is a DAG, False otherwise
    """
    return nx.is_directed_acyclic_graph(graph)


def is_chain(graph: nx.DiGraph) -> bool:
    """
    Check if a graph is a linear chain.

    A chain is a DAG where each node has at most 1 incoming and 1 outgoing edge,
    and forms a single path (not multiple disconnected paths).

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph is a chain, False otherwise
    """
    if graph.number_of_nodes() == 0:
        return False

    if graph.number_of_nodes() == 1:
        return True

    # Must be a DAG (no cycles)
    if has_cycles(graph):
        return False

    if graph.number_of_nodes() == 2:
        # Two nodes: check if connected in ONE direction only (not both)
        nodes = list(graph.nodes())
        edge1 = graph.has_edge(nodes[0], nodes[1])
        edge2 = graph.has_edge(nodes[1], nodes[0])
        return edge1 != edge2  # XOR: exactly one edge, not both

    # Every node can have at most 1 incoming and 1 outgoing
    for node in graph.nodes():
        out_degree = graph.out_degree(node)
        in_degree = graph.in_degree(node)

        if in_degree > 1 or out_degree > 1:
            return False

    # Must form a single connected path
    # Check that there's exactly one root and one leaf
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

    return len(roots) == 1 and len(leaves) == 1


def is_star(graph: nx.DiGraph) -> bool:
    """
    Check if a graph is a star topology.

    A star has one central node (hub) with high fanout or fanin (>1),
    connected to all other nodes (spokes), with no connections between spokes.

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph is a star, False otherwise
    """
    if graph.number_of_nodes() <= 2:
        return False

    # For a star: there should be exactly one node that connects to all others
    # and no edges between non-center nodes
    center_candidates = []
    for potential_center in graph.nodes():
        # Count how many other nodes this node connects to (either direction)
        connected_count = 0
        for other_node in graph.nodes():
            if other_node == potential_center:
                continue
            if graph.has_edge(potential_center, other_node) or graph.has_edge(
                other_node, potential_center
            ):
                connected_count += 1

        if connected_count == graph.number_of_nodes() - 1:
            center_candidates.append(potential_center)

    # Should have exactly one center
    if len(center_candidates) != 1:
        return False

    center = center_candidates[0]

    # Center must have multiple connections (fanout > 1 OR fanin > 1)
    # This distinguishes stars from simple chains
    if graph.out_degree(center) < 2 and graph.in_degree(center) < 2:
        return False

    # Verify no edges between spokes (non-center nodes)
    for node1 in graph.nodes():
        if node1 == center:
            continue
        for node2 in graph.nodes():
            if node2 == center or node2 == node1:
                continue
            if graph.has_edge(node1, node2):
                return False  # Found edge between spokes

    return True


def is_cycle_graph(graph: nx.DiGraph) -> bool:
    """
    Check if a graph is a simple cycle.

    A cycle graph is where all nodes have in-degree 1 and out-degree 1.

    Args:
        graph: NetworkX DiGraph to check

    Returns:
        True if graph is a cycle, False otherwise
    """
    if graph.number_of_nodes() < 3:
        return False

    # All nodes must have exactly in-degree and out-degree of 1
    for node in graph.nodes():
        if graph.in_degree(node) != 1 or graph.out_degree(node) != 1:
            return False

    return True


def detect_topology(graph: nx.DiGraph) -> GraphTopology:
    """
    Detect the topology type of a graph.

    Checks topologies in order of specificity (most specific first).

    Args:
        graph: NetworkX DiGraph to analyze

    Returns:
        Detected GraphTopology enum value
    """
    if graph.number_of_nodes() == 0:
        return GraphTopology.EMPTY

    if graph.number_of_nodes() == 1:
        return GraphTopology.SINGLE_NODE

    # Check for cycles first
    if has_cycles(graph):
        if is_cycle_graph(graph):
            return GraphTopology.CYCLE
        return GraphTopology.UNKNOWN

    # For acyclic graphs, check from most specific to least specific
    # Order: chain, tree, star, dag
    if is_chain(graph):
        return GraphTopology.CHAIN

    if is_tree(graph):
        return GraphTopology.TREE

    if is_star(graph):
        return GraphTopology.STAR

    # Default to DAG for acyclic graphs
    return GraphTopology.DAG


def validate_topology(
    graph: nx.DiGraph,
    required_topology: GraphTopology,
) -> bool:
    """
    Validate that a graph matches a required topology.

    Args:
        graph: NetworkX DiGraph to validate
        required_topology: Required GraphTopology type

    Returns:
        True if graph matches the required topology

    Raises:
        ValueError: If topology doesn't match
    """
    current_topology = detect_topology(graph)

    if current_topology == required_topology:
        return True

    raise ValueError(
        f"Graph topology is '{current_topology.value}', "
        f"but '{required_topology.value}' was required"
    )


def get_root_nodes(graph: nx.DiGraph) -> list:
    """
    Get all root nodes (nodes with in-degree 0) in a graph.

    Args:
        graph: NetworkX DiGraph to analyze

    Returns:
        List of root node identifiers
    """
    return [node for node in graph.nodes() if graph.in_degree(node) == 0]


def get_leaf_nodes(graph: nx.DiGraph) -> list:
    """
    Get all leaf nodes (nodes with out-degree 0) in a graph.

    Args:
        graph: NetworkX DiGraph to analyze

    Returns:
        List of leaf node identifiers
    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]


def get_isolated_nodes(graph: nx.DiGraph) -> list:
    """
    Get all isolated nodes (nodes with no connections).

    Args:
        graph: NetworkX DiGraph to analyze

    Returns:
        List of isolated node identifiers
    """
    return [node for node in graph.nodes() if graph.degree(node) == 0]
