"""
Tests for AgentGraph class and core graph operations.
"""

from pathlib import Path

import pytest

from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.graph import (
    AgentGraph,
    DuplicateEdgeError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
    TopologyViolationError,
)
from claude_agent_graph.models import Node


class TestAgentGraphInitialization:
    """Tests for AgentGraph initialization."""

    def test_init_with_defaults(self):
        """Test graph initialization with default parameters."""
        graph = AgentGraph(name="test_graph")
        assert graph.name == "test_graph"
        assert graph.max_nodes == 10000
        assert graph.persistence_enabled is True
        assert graph.topology_constraint is None
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_init_with_custom_params(self):
        """Test graph initialization with custom parameters."""
        graph = AgentGraph(
            name="custom_graph",
            max_nodes=100,
            persistence_enabled=False,
            topology_constraint="tree",
        )
        assert graph.name == "custom_graph"
        assert graph.max_nodes == 100
        assert graph.persistence_enabled is False
        assert graph.topology_constraint == "tree"

    def test_repr(self):
        """Test string representation of graph."""
        graph = AgentGraph(name="test_graph")
        assert "AgentGraph" in repr(graph)
        assert "test_graph" in repr(graph)
        assert "nodes=0" in repr(graph)
        assert "edges=0" in repr(graph)


class TestNodeOperations:
    """Tests for node management operations."""

    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = AgentGraph(name="test_graph")
        node = graph.add_node(
            node_id="agent_1",
            system_prompt="You are a helpful assistant",
            model="claude-sonnet-4-20250514",
        )

        assert isinstance(node, Node)
        assert node.node_id == "agent_1"
        assert node.system_prompt == "You are a helpful assistant"
        assert graph.node_count == 1

    def test_add_node_with_metadata(self):
        """Test adding a node with metadata."""
        graph = AgentGraph(name="test_graph")
        node = graph.add_node(
            node_id="agent_1",
            system_prompt="You are a helpful assistant",
            role="supervisor",
            priority="high",
        )

        assert node.metadata == {"role": "supervisor", "priority": "high"}

    def test_add_duplicate_node_raises(self):
        """Test that adding duplicate node raises error."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="First agent")

        with pytest.raises(DuplicateNodeError):
            graph.add_node(node_id="agent_1", system_prompt="Different prompt")

    def test_add_node_max_nodes_exceeded(self):
        """Test that max_nodes constraint is enforced."""
        graph = AgentGraph(name="test_graph", max_nodes=2)
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        graph.add_node(node_id="agent_2", system_prompt="Agent 2")

        with pytest.raises(Exception):  # AgentGraphError
            graph.add_node(node_id="agent_3", system_prompt="Agent 3")

    def test_get_node(self):
        """Test retrieving a node."""
        graph = AgentGraph(name="test_graph")
        added_node = graph.add_node(node_id="agent_1", system_prompt="Test agent")
        retrieved_node = graph.get_node("agent_1")

        assert retrieved_node.node_id == added_node.node_id
        assert retrieved_node.system_prompt == added_node.system_prompt

    def test_get_node_not_found(self):
        """Test that getting non-existent node raises error."""
        graph = AgentGraph(name="test_graph")

        with pytest.raises(NodeNotFoundError):
            graph.get_node("nonexistent")

    def test_get_nodes(self):
        """Test retrieving all nodes."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        graph.add_node(node_id="agent_3", system_prompt="Agent 3")

        nodes = graph.get_nodes()
        assert len(nodes) == 3
        node_ids = {node.node_id for node in nodes}
        assert node_ids == {"agent_1", "agent_2", "agent_3"}

    def test_node_exists(self):
        """Test node existence check."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")

        assert graph.node_exists("agent_1") is True
        assert graph.node_exists("agent_2") is False

    def test_node_count_property(self):
        """Test node_count property."""
        graph = AgentGraph(name="test_graph")
        assert graph.node_count == 0

        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        assert graph.node_count == 1

        graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        assert graph.node_count == 2


class TestEdgeOperations:
    """Tests for edge management operations."""

    @pytest.fixture
    def graph_with_nodes(self):
        """Create a graph with test nodes."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        graph.add_node(node_id="agent_3", system_prompt="Agent 3")
        return graph

    def test_add_directed_edge(self, graph_with_nodes):
        """Test adding a directed edge."""
        edge = graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2", directed=True)

        assert edge.from_node == "agent_1"
        assert edge.to_node == "agent_2"
        assert edge.directed is True
        assert graph_with_nodes.edge_count == 1

    def test_add_undirected_edge(self, graph_with_nodes):
        """Test adding an undirected edge."""
        edge = graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2", directed=False)

        assert edge.directed is False
        assert graph_with_nodes.edge_count == 1

    def test_add_edge_with_properties(self, graph_with_nodes):
        """Test adding an edge with custom properties."""
        edge = graph_with_nodes.add_edge(
            from_node="agent_1",
            to_node="agent_2",
            weight=0.8,
            description="test connection",
        )

        assert "weight" in edge.properties
        assert edge.properties["weight"] == 0.8
        assert "description" in edge.properties

    def test_add_edge_node_not_found(self, graph_with_nodes):
        """Test that adding edge with missing node raises error."""
        with pytest.raises(NodeNotFoundError):
            graph_with_nodes.add_edge(from_node="agent_1", to_node="nonexistent")

        with pytest.raises(NodeNotFoundError):
            graph_with_nodes.add_edge(from_node="nonexistent", to_node="agent_1")

    def test_add_duplicate_directed_edge(self, graph_with_nodes):
        """Test that duplicate directed edge raises error."""
        graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")

        with pytest.raises(DuplicateEdgeError):
            graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")

    def test_add_duplicate_undirected_edge(self, graph_with_nodes):
        """Test that duplicate undirected edge raises error."""
        graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2", directed=False)

        with pytest.raises(DuplicateEdgeError):
            graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2", directed=False)

    def test_get_edge(self, graph_with_nodes):
        """Test retrieving an edge."""
        added_edge = graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")
        retrieved_edge = graph_with_nodes.get_edge(from_node="agent_1", to_node="agent_2")

        assert retrieved_edge.edge_id == added_edge.edge_id

    def test_get_edge_not_found(self, graph_with_nodes):
        """Test that getting non-existent edge raises error."""
        with pytest.raises(EdgeNotFoundError):
            graph_with_nodes.get_edge(from_node="agent_1", to_node="agent_2")

    def test_get_edges(self, graph_with_nodes):
        """Test retrieving all edges."""
        graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")
        graph_with_nodes.add_edge(from_node="agent_2", to_node="agent_3")
        graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_3")

        edges = graph_with_nodes.get_edges()
        assert len(edges) == 3

    def test_edge_exists(self, graph_with_nodes):
        """Test edge existence check."""
        graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")

        assert graph_with_nodes.edge_exists("agent_1", "agent_2") is True
        assert graph_with_nodes.edge_exists("agent_2", "agent_1") is False

    def test_edge_count_property(self, graph_with_nodes):
        """Test edge_count property."""
        assert graph_with_nodes.edge_count == 0

        graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")
        assert graph_with_nodes.edge_count == 1

        graph_with_nodes.add_edge(from_node="agent_2", to_node="agent_3")
        assert graph_with_nodes.edge_count == 2


class TestNeighborQueries:
    """Tests for neighbor query operations."""

    @pytest.fixture
    def graph_with_edges(self):
        """Create a graph with edges."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        graph.add_node(node_id="agent_3", system_prompt="Agent 3")
        graph.add_node(node_id="agent_4", system_prompt="Agent 4")

        # Create: 1 -> 2, 1 -> 3, 2 -> 4
        graph.add_edge(from_node="agent_1", to_node="agent_2")
        graph.add_edge(from_node="agent_1", to_node="agent_3")
        graph.add_edge(from_node="agent_2", to_node="agent_4")

        return graph

    def test_get_outgoing_neighbors(self, graph_with_edges):
        """Test getting outgoing neighbors."""
        neighbors = graph_with_edges.get_neighbors("agent_1", direction="outgoing")
        assert set(neighbors) == {"agent_2", "agent_3"}

    def test_get_incoming_neighbors(self, graph_with_edges):
        """Test getting incoming neighbors."""
        neighbors = graph_with_edges.get_neighbors("agent_2", direction="incoming")
        assert neighbors == ["agent_1"]

    def test_get_both_neighbors(self, graph_with_edges):
        """Test getting neighbors in both directions."""
        neighbors = graph_with_edges.get_neighbors("agent_2", direction="both")
        assert set(neighbors) == {"agent_1", "agent_4"}

    def test_get_neighbors_isolated_node(self, graph_with_edges):
        """Test getting neighbors of isolated node."""
        graph_with_edges.add_node(node_id="agent_5", system_prompt="Agent 5")
        neighbors = graph_with_edges.get_neighbors("agent_5", direction="both")
        assert neighbors == []

    def test_get_neighbors_invalid_direction(self, graph_with_edges):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError):
            graph_with_edges.get_neighbors("agent_1", direction="invalid")

    def test_get_neighbors_node_not_found(self, graph_with_edges):
        """Test that querying neighbors of non-existent node raises error."""
        with pytest.raises(NodeNotFoundError):
            graph_with_edges.get_neighbors("nonexistent", direction="both")


class TestTopologyDetection:
    """Tests for topology detection."""

    def test_empty_graph_topology(self):
        """Test topology of empty graph."""
        graph = AgentGraph(name="test_graph")
        assert graph.get_topology() == "empty"

    def test_single_node_topology(self):
        """Test topology of single-node graph."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        assert graph.get_topology() == "single_node"

    def test_tree_topology(self):
        """Test detection of tree topology."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="root", system_prompt="Root")
        graph.add_node(node_id="child_1", system_prompt="Child 1")
        graph.add_node(node_id="child_2", system_prompt="Child 2")
        graph.add_node(node_id="grandchild", system_prompt="Grandchild")

        graph.add_edge(from_node="root", to_node="child_1")
        graph.add_edge(from_node="root", to_node="child_2")
        graph.add_edge(from_node="child_1", to_node="grandchild")

        assert graph.get_topology() == "tree"

    def test_chain_topology(self):
        """Test detection of chain topology."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        graph.add_node(node_id="agent_3", system_prompt="Agent 3")

        graph.add_edge(from_node="agent_1", to_node="agent_2")
        graph.add_edge(from_node="agent_2", to_node="agent_3")

        assert graph.get_topology() == "chain"

    def test_star_topology(self):
        """Test detection of star topology (tree with hub as root)."""
        # Note: A hub->spoke structure is a tree, not a star in our classification
        # because trees take priority. This is a rooted tree with one root and multiple children.
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="hub", system_prompt="Hub")
        graph.add_node(node_id="spoke_1", system_prompt="Spoke 1")
        graph.add_node(node_id="spoke_2", system_prompt="Spoke 2")
        graph.add_node(node_id="spoke_3", system_prompt="Spoke 3")

        graph.add_edge(from_node="hub", to_node="spoke_1")
        graph.add_edge(from_node="hub", to_node="spoke_2")
        graph.add_edge(from_node="hub", to_node="spoke_3")

        # This is a tree: hub is root, spokes are children
        assert graph.get_topology() == "tree"

    def test_dag_topology(self):
        """Test detection of DAG topology."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")
        graph.add_node(node_id="c", system_prompt="C")
        graph.add_node(node_id="d", system_prompt="D")

        graph.add_edge(from_node="a", to_node="b")
        graph.add_edge(from_node="a", to_node="c")
        graph.add_edge(from_node="b", to_node="d")
        graph.add_edge(from_node="c", to_node="d")

        assert graph.get_topology() == "dag"

    def test_validate_topology_match(self):
        """Test topology validation when topology matches."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="root", system_prompt="Root")
        graph.add_node(node_id="child", system_prompt="Child")
        graph.add_edge(from_node="root", to_node="child")

        # A 2-node graph with one edge is a chain, not a tree
        assert graph.validate_topology("chain") is True

    def test_validate_topology_mismatch(self):
        """Test topology validation when topology doesn't match."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="root", system_prompt="Root")
        graph.add_node(node_id="child", system_prompt="Child")
        graph.add_edge(from_node="root", to_node="child")

        with pytest.raises(TopologyViolationError):
            graph.validate_topology("star")


class TestTopologyConstraint:
    """Tests for topology constraint enforcement."""

    def test_tree_constraint_prevents_cycle(self):
        """Test that tree constraint prevents cycles."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="tree",
        )
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")

        graph.add_edge(from_node="a", to_node="b")

        # Adding edge that creates cycle should fail
        with pytest.raises(TopologyViolationError):
            graph.add_edge(from_node="b", to_node="a")

    def test_tree_constraint_prevents_multiple_parents(self):
        """Test that tree constraint prevents multiple parents."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="tree",
        )
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")
        graph.add_node(node_id="c", system_prompt="C")

        graph.add_edge(from_node="a", to_node="c")

        # Adding second parent to c should fail
        with pytest.raises(TopologyViolationError):
            graph.add_edge(from_node="b", to_node="c")

    def test_dag_constraint_prevents_cycle(self):
        """Test that DAG constraint prevents cycles."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="dag",
        )
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")

        graph.add_edge(from_node="a", to_node="b")

        # Adding edge that creates cycle should fail
        with pytest.raises(TopologyViolationError):
            graph.add_edge(from_node="b", to_node="a")

    def test_chain_constraint(self):
        """Test that chain constraint enforces linear structure."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="chain",
        )
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")
        graph.add_node(node_id="c", system_prompt="C")

        graph.add_edge(from_node="a", to_node="b")
        graph.add_edge(from_node="b", to_node="c")

        # Adding another edge should fail
        with pytest.raises(TopologyViolationError):
            graph.add_edge(from_node="a", to_node="c")


class TestIsolatedNodes:
    """Tests for isolated node detection."""

    def test_get_isolated_nodes_empty(self):
        """Test isolated nodes in empty graph."""
        graph = AgentGraph(name="test_graph")
        isolated = graph.get_isolated_nodes()
        assert isolated == []

    def test_get_isolated_nodes_no_isolated(self):
        """Test when no nodes are isolated."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")
        graph.add_edge(from_node="a", to_node="b")

        isolated = graph.get_isolated_nodes()
        assert isolated == []

    def test_get_isolated_nodes_with_isolated(self):
        """Test detection of isolated nodes."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")
        graph.add_node(node_id="c", system_prompt="C")

        graph.add_edge(from_node="a", to_node="b")

        isolated = graph.get_isolated_nodes()
        assert isolated == ["c"]

    def test_get_isolated_nodes_multiple(self):
        """Test detection of multiple isolated nodes."""
        graph = AgentGraph(name="test_graph")
        graph.add_node(node_id="a", system_prompt="A")
        graph.add_node(node_id="b", system_prompt="B")
        graph.add_node(node_id="c", system_prompt="C")
        graph.add_node(node_id="d", system_prompt="D")

        graph.add_edge(from_node="a", to_node="b")

        isolated = set(graph.get_isolated_nodes())
        assert isolated == {"c", "d"}


class TestGraphMessageRouting:
    """Tests for AgentGraph message routing methods."""

    @pytest.mark.asyncio
    async def test_send_message_basic(self, tmp_path: Path) -> None:
        """Test sending a basic message between nodes."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "Agent A")
        graph.add_node("b", "Agent B")
        graph.add_edge("a", "b")

        msg = await graph.send_message("a", "b", "Hello")

        assert msg.from_node == "a"
        assert msg.to_node == "b"
        assert msg.content == "Hello"
        assert msg.message_id.startswith("msg_")

    @pytest.mark.asyncio
    async def test_send_message_with_metadata(self, tmp_path: Path) -> None:
        """Test sending message with metadata."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")
        graph.add_edge("a", "b")

        msg = await graph.send_message("a", "b", "Test", priority="high", task_id=123)

        assert msg.metadata["priority"] == "high"
        assert msg.metadata["task_id"] == 123

    @pytest.mark.asyncio
    async def test_send_message_nonexistent_sender(self, tmp_path: Path) -> None:
        """Test sending from non-existent node raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("b", "B")

        with pytest.raises(NodeNotFoundError):
            await graph.send_message("a", "b", "Test")

    @pytest.mark.asyncio
    async def test_send_message_nonexistent_receiver(self, tmp_path: Path) -> None:
        """Test sending to non-existent node raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")

        with pytest.raises(NodeNotFoundError):
            await graph.send_message("a", "b", "Test")

    @pytest.mark.asyncio
    async def test_send_message_no_edge(self, tmp_path: Path) -> None:
        """Test sending without edge raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")

        with pytest.raises(EdgeNotFoundError):
            await graph.send_message("a", "b", "Test")

    @pytest.mark.asyncio
    async def test_send_multiple_messages(self, tmp_path: Path) -> None:
        """Test sending multiple messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")
        graph.add_edge("a", "b")

        msg1 = await graph.send_message("a", "b", "First")
        msg2 = await graph.send_message("a", "b", "Second")
        msg3 = await graph.send_message("a", "b", "Third")

        assert msg1.message_id != msg2.message_id
        assert msg2.message_id != msg3.message_id

    @pytest.mark.asyncio
    async def test_get_conversation_all_messages(self, tmp_path: Path) -> None:
        """Test retrieving all conversation messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")
        graph.add_edge("a", "b")

        await graph.send_message("a", "b", "Message 1")
        await graph.send_message("a", "b", "Message 2")
        await graph.send_message("a", "b", "Message 3")

        messages = await graph.get_conversation("a", "b")

        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[2].content == "Message 3"

    @pytest.mark.asyncio
    async def test_get_conversation_with_limit(self, tmp_path: Path) -> None:
        """Test retrieving limited messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")
        graph.add_edge("a", "b")

        for i in range(10):
            await graph.send_message("a", "b", f"Message {i}")

        messages = await graph.get_conversation("a", "b", limit=3)

        assert len(messages) == 3
        assert messages[0].content == "Message 7"
        assert messages[2].content == "Message 9"

    @pytest.mark.asyncio
    async def test_get_conversation_nonexistent_nodes(self, tmp_path: Path) -> None:
        """Test get_conversation with non-existent nodes."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")

        with pytest.raises(NodeNotFoundError):
            await graph.get_conversation("a", "b")

    @pytest.mark.asyncio
    async def test_get_conversation_no_edge(self, tmp_path: Path) -> None:
        """Test get_conversation without edge."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")

        with pytest.raises(EdgeNotFoundError):
            await graph.get_conversation("a", "b")

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, tmp_path: Path) -> None:
        """Test get_recent_messages convenience method."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")
        graph.add_edge("a", "b")

        for i in range(20):
            await graph.send_message("a", "b", f"Msg {i}")

        recent = await graph.get_recent_messages("a", "b", count=5)

        assert len(recent) == 5
        assert recent[0].content == "Msg 15"
        assert recent[4].content == "Msg 19"

    @pytest.mark.asyncio
    async def test_undirected_edge_messaging(self, tmp_path: Path) -> None:
        """Test messaging works with undirected edges."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        graph.add_node("a", "A")
        graph.add_node("b", "B")
        graph.add_edge("a", "b", directed=False)

        # Should work both directions
        await graph.send_message("a", "b", "A to B")
        await graph.send_message("b", "a", "B to A")

        messages = await graph.get_conversation("a", "b")
        assert len(messages) == 2
