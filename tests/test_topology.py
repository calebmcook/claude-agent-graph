"""
Tests for topology detection and validation utilities.
"""

import networkx as nx
import pytest
from claude_agent_graph.topology import (
    GraphTopology,
    detect_topology,
    get_isolated_nodes,
    get_leaf_nodes,
    get_root_nodes,
    has_cycles,
    is_chain,
    is_connected,
    is_cycle_graph,
    is_dag,
    is_star,
    is_tree,
    validate_topology,
)


class TestCycleDetection:
    """Tests for cycle detection."""

    def test_acyclic_graph(self):
        """Test that acyclic graph is detected correctly."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        assert has_cycles(graph) is False

    def test_cyclic_graph(self):
        """Test that cyclic graph is detected correctly."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert has_cycles(graph) is True

    def test_self_loop(self):
        """Test that self-loop is detected as cycle."""
        graph = nx.DiGraph()
        graph.add_edge("a", "a")
        assert has_cycles(graph) is True


class TestConnectivity:
    """Tests for connectivity detection."""

    def test_connected_graph(self):
        """Test detection of connected graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert is_connected(graph) is True

    def test_disconnected_graph(self):
        """Test detection of disconnected graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("c", "d")])
        assert is_connected(graph) is False

    def test_empty_graph(self):
        """Test connectivity of empty graph."""
        graph = nx.DiGraph()
        assert is_connected(graph) is False


class TestTreeTopology:
    """Tests for tree topology detection."""

    def test_single_node_is_tree(self):
        """Test that single node is a tree."""
        graph = nx.DiGraph()
        graph.add_node("a")
        assert is_tree(graph) is True

    def test_two_nodes_is_tree(self):
        """Test that two connected nodes form a tree."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        assert is_tree(graph) is True

    def test_simple_tree(self):
        """Test detection of simple tree."""
        graph = nx.DiGraph()
        graph.add_edges_from([("root", "left"), ("root", "right")])
        assert is_tree(graph) is True

    def test_deep_tree(self):
        """Test detection of deep tree."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
            ]
        )
        assert is_tree(graph) is True

    def test_not_tree_multiple_roots(self):
        """Test that graph with multiple roots is not a tree."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("c", "d")])
        assert is_tree(graph) is False

    def test_not_tree_node_with_multiple_parents(self):
        """Test that node with multiple parents is not a tree."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "c"), ("b", "c")])
        assert is_tree(graph) is False

    def test_not_tree_cyclic(self):
        """Test that cyclic graph is not a tree."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert is_tree(graph) is False


class TestDAGTopology:
    """Tests for DAG topology detection."""

    def test_simple_dag(self):
        """Test detection of simple DAG."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        assert is_dag(graph) is True

    def test_complex_dag(self):
        """Test detection of complex DAG."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("a", "b"),
                ("a", "c"),
                ("b", "d"),
                ("c", "d"),
            ]
        )
        assert is_dag(graph) is True

    def test_not_dag_with_cycle(self):
        """Test that graph with cycle is not a DAG."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert is_dag(graph) is False


class TestChainTopology:
    """Tests for chain topology detection."""

    def test_simple_chain(self):
        """Test detection of simple chain."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        assert is_chain(graph) is True

    def test_single_node_chain(self):
        """Test that single node is a chain."""
        graph = nx.DiGraph()
        graph.add_node("a")
        assert is_chain(graph) is True

    def test_two_nodes_chain(self):
        """Test that two nodes form a chain."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        assert is_chain(graph) is True

    def test_not_chain_branching(self):
        """Test that branching breaks chain."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("b", "d")])
        assert is_chain(graph) is False

    def test_not_chain_merging(self):
        """Test that merging breaks chain."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "c"), ("b", "c")])
        assert is_chain(graph) is False

    def test_not_chain_cycle(self):
        """Test that cycle breaks chain."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "a")])
        assert is_chain(graph) is False


class TestStarTopology:
    """Tests for star topology detection."""

    def test_simple_star(self):
        """Test detection of simple star."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("hub", "spoke_1"),
                ("hub", "spoke_2"),
                ("hub", "spoke_3"),
            ]
        )
        assert is_star(graph) is True

    def test_star_with_bidirectional_edges(self):
        """Test star with bidirectional connections."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("hub", "spoke_1"),
                ("spoke_1", "hub"),
                ("hub", "spoke_2"),
                ("spoke_2", "hub"),
            ]
        )
        assert is_star(graph) is True

    def test_not_star_single_node(self):
        """Test that single node is not a star."""
        graph = nx.DiGraph()
        graph.add_node("a")
        assert is_star(graph) is False

    def test_not_star_two_nodes(self):
        """Test that two nodes don't form a star."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        assert is_star(graph) is False

    def test_not_star_no_center(self):
        """Test that chain is not a star."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        assert is_star(graph) is False

    def test_not_star_disconnected_spoke(self):
        """Test that star with disconnected spoke fails."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("hub", "spoke_1"),
                ("hub", "spoke_2"),
            ]
        )
        graph.add_node("spoke_3")  # Not connected
        assert is_star(graph) is False


class TestCycleGraph:
    """Tests for cycle graph detection."""

    def test_simple_cycle(self):
        """Test detection of simple cycle."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert is_cycle_graph(graph) is True

    def test_four_node_cycle(self):
        """Test detection of larger cycle."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
        assert is_cycle_graph(graph) is True

    def test_not_cycle_single_node(self):
        """Test that single node is not a cycle."""
        graph = nx.DiGraph()
        graph.add_node("a")
        assert is_cycle_graph(graph) is False

    def test_not_cycle_two_nodes(self):
        """Test that two nodes don't form a cycle."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        assert is_cycle_graph(graph) is False

    def test_not_cycle_with_branch(self):
        """Test that branching breaks cycle."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "a"),
                ("b", "d"),
            ]
        )
        assert is_cycle_graph(graph) is False


class TestTopologyDetection:
    """Tests for overall topology detection."""

    def test_detect_empty(self):
        """Test detection of empty graph."""
        graph = nx.DiGraph()
        assert detect_topology(graph) == GraphTopology.EMPTY

    def test_detect_single_node(self):
        """Test detection of single node."""
        graph = nx.DiGraph()
        graph.add_node("a")
        assert detect_topology(graph) == GraphTopology.SINGLE_NODE

    def test_detect_tree(self):
        """Test detection of tree topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("root", "left"), ("root", "right")])
        assert detect_topology(graph) == GraphTopology.TREE

    def test_detect_chain(self):
        """Test detection of chain topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        assert detect_topology(graph) == GraphTopology.CHAIN

    def test_detect_star(self):
        """Test detection of star topology."""
        # Note: A hub->spoke structure is detected as TREE (not STAR)
        # because trees take priority in topology detection.
        # This is a rooted tree with hub as root and spokes as children.
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("hub", "spoke_1"),
                ("hub", "spoke_2"),
                ("hub", "spoke_3"),
            ]
        )
        assert detect_topology(graph) == GraphTopology.TREE

    def test_detect_cycle(self):
        """Test detection of cycle topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert detect_topology(graph) == GraphTopology.CYCLE

    def test_detect_dag(self):
        """Test detection of DAG topology."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("a", "b"),
                ("a", "c"),
                ("b", "d"),
                ("c", "d"),
            ]
        )
        assert detect_topology(graph) == GraphTopology.DAG

    def test_detect_unknown(self):
        """Test detection of unknown topology."""
        graph = nx.DiGraph()
        # Create a graph with cycle that's not a simple cycle
        graph.add_edges_from(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "a"),
                ("b", "d"),
            ]
        )
        assert detect_topology(graph) == GraphTopology.UNKNOWN


class TestTopologyValidation:
    """Tests for topology validation."""

    def test_validate_matching_tree(self):
        """Test validation of matching tree topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("root", "left"), ("root", "right")])
        assert validate_topology(graph, GraphTopology.TREE) is True

    def test_validate_matching_chain(self):
        """Test validation of matching chain topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        assert validate_topology(graph, GraphTopology.CHAIN) is True

    def test_validate_non_matching_topology(self):
        """Test validation fails for non-matching topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("root", "left"), ("root", "right")])
        with pytest.raises(ValueError):
            validate_topology(graph, GraphTopology.CHAIN)

    def test_validate_cycle_detection(self):
        """Test validation of cycle topology."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        assert validate_topology(graph, GraphTopology.CYCLE) is True


class TestRootAndLeafNodes:
    """Tests for root and leaf node detection."""

    def test_get_root_nodes_single(self):
        """Test getting root nodes from tree."""
        graph = nx.DiGraph()
        graph.add_edges_from([("root", "left"), ("root", "right")])
        roots = get_root_nodes(graph)
        assert roots == ["root"]

    def test_get_root_nodes_multiple(self):
        """Test getting multiple roots."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "c"), ("b", "c")])
        roots = set(get_root_nodes(graph))
        assert roots == {"a", "b"}

    def test_get_leaf_nodes_single(self):
        """Test getting leaf nodes from chain."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        leaves = get_leaf_nodes(graph)
        assert leaves == ["c"]

    def test_get_leaf_nodes_multiple(self):
        """Test getting multiple leaves."""
        graph = nx.DiGraph()
        graph.add_edges_from([("root", "left"), ("root", "right")])
        leaves = set(get_leaf_nodes(graph))
        assert leaves == {"left", "right"}

    def test_get_leaf_nodes_empty(self):
        """Test getting leaves from empty graph."""
        graph = nx.DiGraph()
        leaves = get_leaf_nodes(graph)
        assert leaves == []


class TestIsolatedNodesUtility:
    """Tests for isolated node detection utility."""

    def test_get_isolated_nodes_none(self):
        """Test when no nodes are isolated."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        isolated = get_isolated_nodes(graph)
        assert isolated == []

    def test_get_isolated_nodes_single(self):
        """Test detection of single isolated node."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])
        graph.add_node("c")
        isolated = get_isolated_nodes(graph)
        assert isolated == ["c"]

    def test_get_isolated_nodes_multiple(self):
        """Test detection of multiple isolated nodes."""
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b")])
        graph.add_nodes_from(["c", "d", "e"])
        isolated = set(get_isolated_nodes(graph))
        assert isolated == {"c", "d", "e"}

    def test_get_isolated_nodes_all_isolated(self):
        """Test when all nodes are isolated."""
        graph = nx.DiGraph()
        graph.add_nodes_from(["a", "b", "c"])
        isolated = set(get_isolated_nodes(graph))
        assert isolated == {"a", "b", "c"}
