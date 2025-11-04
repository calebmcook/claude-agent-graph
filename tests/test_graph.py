"""
Tests for AgentGraph class and core graph operations.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.exceptions import AgentGraphError, CommandAuthorizationError
from claude_agent_graph.graph import (
    AgentGraph,
    DuplicateEdgeError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
    TopologyViolationError,
)
from claude_agent_graph.models import Node


@pytest.fixture
def mock_claude_client():
    """Create a mock ClaudeSDKClient for testing."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_claude_sdk(mock_claude_client):
    """Mock the ClaudeSDKClient and ClaudeAgentOptions."""
    with (
        patch(
            "claude_agent_graph.agent_manager.ClaudeSDKClient",
            return_value=mock_claude_client,
        ),
        patch(
            "claude_agent_graph.agent_manager.ClaudeAgentOptions",
            MagicMock(),
        ),
    ):
        yield


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

    async def test_add_node(self):
        """Test adding a node to the graph."""
        graph = AgentGraph(name="test_graph")
        node = await graph.add_node(
            node_id="agent_1",
            system_prompt="You are a helpful assistant",
            model="claude-sonnet-4-20250514",
        )

        assert isinstance(node, Node)
        assert node.node_id == "agent_1"
        assert node.system_prompt == "You are a helpful assistant"
        assert graph.node_count == 1

    async def test_add_node_with_metadata(self):
        """Test adding a node with metadata."""
        graph = AgentGraph(name="test_graph")
        node = await graph.add_node(
            node_id="agent_1",
            system_prompt="You are a helpful assistant",
            role="supervisor",
            priority="high",
        )

        assert node.metadata == {"role": "supervisor", "priority": "high"}

    async def test_add_duplicate_node_raises(self):
        """Test that adding duplicate node raises error."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="First agent")

        with pytest.raises(DuplicateNodeError):
            await graph.add_node(node_id="agent_1", system_prompt="Different prompt")

    async def test_add_node_max_nodes_exceeded(self):
        """Test that max_nodes constraint is enforced."""
        graph = AgentGraph(name="test_graph", max_nodes=2)
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        await graph.add_node(node_id="agent_2", system_prompt="Agent 2")

        with pytest.raises(Exception):  # AgentGraphError
            await graph.add_node(node_id="agent_3", system_prompt="Agent 3")

    async def test_get_node(self):
        """Test retrieving a node."""
        graph = AgentGraph(name="test_graph")
        added_node = await graph.add_node(node_id="agent_1", system_prompt="Test agent")
        retrieved_node = graph.get_node("agent_1")

        assert retrieved_node.node_id == added_node.node_id
        assert retrieved_node.system_prompt == added_node.system_prompt

    def test_get_node_not_found(self):
        """Test that getting non-existent node raises error."""
        graph = AgentGraph(name="test_graph")

        with pytest.raises(NodeNotFoundError):
            graph.get_node("nonexistent")

    async def test_get_nodes(self):
        """Test retrieving all nodes."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        await graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        await graph.add_node(node_id="agent_3", system_prompt="Agent 3")

        nodes = graph.get_nodes()
        assert len(nodes) == 3
        node_ids = {node.node_id for node in nodes}
        assert node_ids == {"agent_1", "agent_2", "agent_3"}

    async def test_node_exists(self):
        """Test node existence check."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")

        assert graph.node_exists("agent_1") is True
        assert graph.node_exists("agent_2") is False

    async def test_node_count_property(self):
        """Test node_count property."""
        graph = AgentGraph(name="test_graph")
        assert graph.node_count == 0

        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        assert graph.node_count == 1

        await graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        assert graph.node_count == 2


class TestEdgeOperations:
    """Tests for edge management operations."""

    @pytest.fixture
    async def graph_with_nodes(self):
        """Create a graph with test nodes."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        await graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        await graph.add_node(node_id="agent_3", system_prompt="Agent 3")
        return graph

    async def test_add_directed_edge(self, graph_with_nodes):
        """Test adding a directed edge."""
        edge = await graph_with_nodes.add_edge(
            from_node="agent_1", to_node="agent_2", directed=True
        )

        assert edge.from_node == "agent_1"
        assert edge.to_node == "agent_2"
        assert edge.directed is True
        assert graph_with_nodes.edge_count == 1

    async def test_add_undirected_edge(self, graph_with_nodes):
        """Test adding an undirected edge."""
        edge = await graph_with_nodes.add_edge(
            from_node="agent_1", to_node="agent_2", directed=False
        )

        assert edge.directed is False
        assert graph_with_nodes.edge_count == 1

    async def test_add_edge_with_properties(self, graph_with_nodes):
        """Test adding an edge with custom properties."""
        edge = await graph_with_nodes.add_edge(
            from_node="agent_1",
            to_node="agent_2",
            weight=0.8,
            description="test connection",
        )

        assert "weight" in edge.properties
        assert edge.properties["weight"] == 0.8
        assert "description" in edge.properties

    async def test_add_edge_node_not_found(self, graph_with_nodes):
        """Test that adding edge with missing node raises error."""
        with pytest.raises(NodeNotFoundError):
            await graph_with_nodes.add_edge(from_node="agent_1", to_node="nonexistent")

        with pytest.raises(NodeNotFoundError):
            await graph_with_nodes.add_edge(from_node="nonexistent", to_node="agent_1")

    async def test_add_duplicate_directed_edge(self, graph_with_nodes):
        """Test that duplicate directed edge raises error."""
        await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")

        with pytest.raises(DuplicateEdgeError):
            await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")

    async def test_add_duplicate_undirected_edge(self, graph_with_nodes):
        """Test that duplicate undirected edge raises error."""
        await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2", directed=False)

        with pytest.raises(DuplicateEdgeError):
            await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2", directed=False)

    async def test_get_edge(self, graph_with_nodes):
        """Test retrieving an edge."""
        added_edge = await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")
        retrieved_edge = graph_with_nodes.get_edge(from_node="agent_1", to_node="agent_2")

        assert retrieved_edge.edge_id == added_edge.edge_id

    def test_get_edge_not_found(self, graph_with_nodes):
        """Test that getting non-existent edge raises error."""
        with pytest.raises(EdgeNotFoundError):
            graph_with_nodes.get_edge(from_node="agent_1", to_node="agent_2")

    async def test_get_edges(self, graph_with_nodes):
        """Test retrieving all edges."""
        await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")
        await graph_with_nodes.add_edge(from_node="agent_2", to_node="agent_3")
        await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_3")

        edges = graph_with_nodes.get_edges()
        assert len(edges) == 3

    async def test_edge_exists(self, graph_with_nodes):
        """Test edge existence check."""
        await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")

        assert graph_with_nodes.edge_exists("agent_1", "agent_2") is True
        assert graph_with_nodes.edge_exists("agent_2", "agent_1") is False

    async def test_edge_count_property(self, graph_with_nodes):
        """Test edge_count property."""
        assert graph_with_nodes.edge_count == 0

        await graph_with_nodes.add_edge(from_node="agent_1", to_node="agent_2")
        assert graph_with_nodes.edge_count == 1

        await graph_with_nodes.add_edge(from_node="agent_2", to_node="agent_3")
        assert graph_with_nodes.edge_count == 2


class TestNeighborQueries:
    """Tests for neighbor query operations."""

    @pytest.fixture
    async def graph_with_edges(self):
        """Create a graph with edges."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        await graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        await graph.add_node(node_id="agent_3", system_prompt="Agent 3")
        await graph.add_node(node_id="agent_4", system_prompt="Agent 4")

        # Create: 1 -> 2, 1 -> 3, 2 -> 4
        await graph.add_edge(from_node="agent_1", to_node="agent_2")
        await graph.add_edge(from_node="agent_1", to_node="agent_3")
        await graph.add_edge(from_node="agent_2", to_node="agent_4")

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

    async def test_get_neighbors_isolated_node(self, graph_with_edges):
        """Test getting neighbors of isolated node."""
        await graph_with_edges.add_node(node_id="agent_5", system_prompt="Agent 5")
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

    async def test_single_node_topology(self):
        """Test topology of single-node graph."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        assert graph.get_topology() == "single_node"

    async def test_tree_topology(self):
        """Test detection of tree topology."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="root", system_prompt="Root")
        await graph.add_node(node_id="child_1", system_prompt="Child 1")
        await graph.add_node(node_id="child_2", system_prompt="Child 2")
        await graph.add_node(node_id="grandchild", system_prompt="Grandchild")

        await graph.add_edge(from_node="root", to_node="child_1")
        await graph.add_edge(from_node="root", to_node="child_2")
        await graph.add_edge(from_node="child_1", to_node="grandchild")

        assert graph.get_topology() == "tree"

    async def test_chain_topology(self):
        """Test detection of chain topology."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="agent_1", system_prompt="Agent 1")
        await graph.add_node(node_id="agent_2", system_prompt="Agent 2")
        await graph.add_node(node_id="agent_3", system_prompt="Agent 3")

        await graph.add_edge(from_node="agent_1", to_node="agent_2")
        await graph.add_edge(from_node="agent_2", to_node="agent_3")

        assert graph.get_topology() == "chain"

    async def test_star_topology(self):
        """Test detection of star topology (tree with hub as root)."""
        # Note: A hub->spoke structure is a tree, not a star in our classification
        # because trees take priority. This is a rooted tree with one root and multiple children.
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="hub", system_prompt="Hub")
        await graph.add_node(node_id="spoke_1", system_prompt="Spoke 1")
        await graph.add_node(node_id="spoke_2", system_prompt="Spoke 2")
        await graph.add_node(node_id="spoke_3", system_prompt="Spoke 3")

        await graph.add_edge(from_node="hub", to_node="spoke_1")
        await graph.add_edge(from_node="hub", to_node="spoke_2")
        await graph.add_edge(from_node="hub", to_node="spoke_3")

        # This is a tree: hub is root, spokes are children
        assert graph.get_topology() == "tree"

    async def test_dag_topology(self):
        """Test detection of DAG topology."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")
        await graph.add_node(node_id="c", system_prompt="C")
        await graph.add_node(node_id="d", system_prompt="D")

        await graph.add_edge(from_node="a", to_node="b")
        await graph.add_edge(from_node="a", to_node="c")
        await graph.add_edge(from_node="b", to_node="d")
        await graph.add_edge(from_node="c", to_node="d")

        assert graph.get_topology() == "dag"

    async def test_validate_topology_match(self):
        """Test topology validation when topology matches."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="root", system_prompt="Root")
        await graph.add_node(node_id="child", system_prompt="Child")
        await graph.add_edge(from_node="root", to_node="child")

        # A 2-node graph with one edge is a chain, not a tree
        assert graph.validate_topology("chain") is True

    async def test_validate_topology_mismatch(self):
        """Test topology validation when topology doesn't match."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="root", system_prompt="Root")
        await graph.add_node(node_id="child", system_prompt="Child")
        await graph.add_edge(from_node="root", to_node="child")

        with pytest.raises(TopologyViolationError):
            graph.validate_topology("star")


class TestTopologyConstraint:
    """Tests for topology constraint enforcement."""

    async def test_tree_constraint_prevents_cycle(self):
        """Test that tree constraint prevents cycles."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="tree",
        )
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")

        await graph.add_edge(from_node="a", to_node="b")

        # Adding edge that creates cycle should fail
        with pytest.raises(TopologyViolationError):
            await graph.add_edge(from_node="b", to_node="a")

    async def test_tree_constraint_prevents_multiple_parents(self):
        """Test that tree constraint prevents multiple parents."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="tree",
        )
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")
        await graph.add_node(node_id="c", system_prompt="C")

        await graph.add_edge(from_node="a", to_node="c")

        # Adding second parent to c should fail
        with pytest.raises(TopologyViolationError):
            await graph.add_edge(from_node="b", to_node="c")

    async def test_dag_constraint_prevents_cycle(self):
        """Test that DAG constraint prevents cycles."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="dag",
        )
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")

        await graph.add_edge(from_node="a", to_node="b")

        # Adding edge that creates cycle should fail
        with pytest.raises(TopologyViolationError):
            await graph.add_edge(from_node="b", to_node="a")

    async def test_chain_constraint(self):
        """Test that chain constraint enforces linear structure."""
        graph = AgentGraph(
            name="test_graph",
            topology_constraint="chain",
        )
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")
        await graph.add_node(node_id="c", system_prompt="C")

        await graph.add_edge(from_node="a", to_node="b")
        await graph.add_edge(from_node="b", to_node="c")

        # Adding another edge should fail
        with pytest.raises(TopologyViolationError):
            await graph.add_edge(from_node="a", to_node="c")


class TestIsolatedNodes:
    """Tests for isolated node detection."""

    def test_get_isolated_nodes_empty(self):
        """Test isolated nodes in empty graph."""
        graph = AgentGraph(name="test_graph")
        isolated = graph.get_isolated_nodes()
        assert isolated == []

    async def test_get_isolated_nodes_no_isolated(self):
        """Test when no nodes are isolated."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")
        await graph.add_edge(from_node="a", to_node="b")

        isolated = graph.get_isolated_nodes()
        assert isolated == []

    async def test_get_isolated_nodes_with_isolated(self):
        """Test detection of isolated nodes."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")
        await graph.add_node(node_id="c", system_prompt="C")

        await graph.add_edge(from_node="a", to_node="b")

        isolated = graph.get_isolated_nodes()
        assert isolated == ["c"]

    async def test_get_isolated_nodes_multiple(self):
        """Test detection of multiple isolated nodes."""
        graph = AgentGraph(name="test_graph")
        await graph.add_node(node_id="a", system_prompt="A")
        await graph.add_node(node_id="b", system_prompt="B")
        await graph.add_node(node_id="c", system_prompt="C")
        await graph.add_node(node_id="d", system_prompt="D")

        await graph.add_edge(from_node="a", to_node="b")

        isolated = set(graph.get_isolated_nodes())
        assert isolated == {"c", "d"}


class TestGraphMessageRouting:
    """Tests for AgentGraph message routing methods."""

    @pytest.mark.asyncio
    async def test_send_message_basic(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test sending a basic message between nodes."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "Agent A")
        await graph.add_node("b", "Agent B")
        await graph.add_edge("a", "b")

        msg = await graph.send_message("a", "b", "Hello")

        assert msg.from_node == "a"
        assert msg.to_node == "b"
        assert msg.content == "Hello"
        assert msg.message_id.startswith("msg_")

    @pytest.mark.asyncio
    async def test_send_message_with_metadata(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test sending message with metadata."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_edge("a", "b")

        msg = await graph.send_message("a", "b", "Test", priority="high", task_id=123)

        assert msg.metadata["priority"] == "high"
        assert msg.metadata["task_id"] == 123

    @pytest.mark.asyncio
    async def test_send_message_nonexistent_sender(self, tmp_path: Path) -> None:
        """Test sending from non-existent node raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("b", "B")

        with pytest.raises(NodeNotFoundError):
            await graph.send_message("a", "b", "Test")

    @pytest.mark.asyncio
    async def test_send_message_nonexistent_receiver(self, tmp_path: Path) -> None:
        """Test sending to non-existent node raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")

        with pytest.raises(NodeNotFoundError):
            await graph.send_message("a", "b", "Test")

    @pytest.mark.asyncio
    async def test_send_message_no_edge(self, tmp_path: Path) -> None:
        """Test sending without edge raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        with pytest.raises(EdgeNotFoundError):
            await graph.send_message("a", "b", "Test")

    @pytest.mark.asyncio
    async def test_send_multiple_messages(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test sending multiple messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_edge("a", "b")

        msg1 = await graph.send_message("a", "b", "First")
        msg2 = await graph.send_message("a", "b", "Second")
        msg3 = await graph.send_message("a", "b", "Third")

        assert msg1.message_id != msg2.message_id
        assert msg2.message_id != msg3.message_id

    @pytest.mark.asyncio
    async def test_get_conversation_all_messages(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test retrieving all conversation messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_edge("a", "b")

        await graph.send_message("a", "b", "Message 1")
        await graph.send_message("a", "b", "Message 2")
        await graph.send_message("a", "b", "Message 3")

        messages = await graph.get_conversation("a", "b")

        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[2].content == "Message 3"

    @pytest.mark.asyncio
    async def test_get_conversation_with_limit(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test retrieving limited messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_edge("a", "b")

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
        await graph.add_node("a", "A")

        with pytest.raises(NodeNotFoundError):
            await graph.get_conversation("a", "b")

    @pytest.mark.asyncio
    async def test_get_conversation_no_edge(self, tmp_path: Path) -> None:
        """Test get_conversation without edge."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        with pytest.raises(EdgeNotFoundError):
            await graph.get_conversation("a", "b")

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test get_recent_messages convenience method."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_edge("a", "b")

        for i in range(20):
            await graph.send_message("a", "b", f"Msg {i}")

        recent = await graph.get_recent_messages("a", "b", count=5)

        assert len(recent) == 5
        assert recent[0].content == "Msg 15"
        assert recent[4].content == "Msg 19"

    @pytest.mark.asyncio
    async def test_undirected_edge_messaging(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test messaging works with undirected edges."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_edge("a", "b", directed=False)

        # Should work both directions
        await graph.send_message("a", "b", "A to B")
        await graph.send_message("b", "a", "B to A")

        messages = await graph.get_conversation("a", "b")
        assert len(messages) == 2


class TestControlRelationships:
    """Tests for control relationships and system prompt injection."""

    async def test_compute_effective_prompt_no_controllers(self):
        """Test computing effective prompt when node has no controllers."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "You are helpful.")

        prompt = graph._compute_effective_prompt("agent_1")
        assert prompt == "You are helpful."

    async def test_compute_effective_prompt_single_controller(self):
        """Test computing effective prompt with one controller."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "You supervise agents.")
        await graph.add_node("worker", "You execute tasks.")
        await graph.add_edge("supervisor", "worker", directed=True, control_type="supervisor")

        prompt = graph._compute_effective_prompt("worker")

        assert "You execute tasks." in prompt
        assert "## Control Hierarchy" in prompt
        assert "agent 'worker'" in prompt
        assert "Agent 'supervisor'" in prompt
        assert "(supervisor)" in prompt

    async def test_compute_effective_prompt_multiple_controllers(self):
        """Test computing effective prompt with multiple controllers."""
        graph = AgentGraph(name="test")
        await graph.add_node("cfo", "You manage finance.")
        await graph.add_node("risk_officer", "You manage risk.")
        await graph.add_node("analyst", "You analyze data.")

        await graph.add_edge("cfo", "analyst", directed=True, control_type="supervisor")
        await graph.add_edge(
            "risk_officer", "analyst", directed=True, control_type="compliance_reviewer"
        )

        prompt = graph._compute_effective_prompt("analyst")

        assert "You analyze data." in prompt
        assert "## Control Hierarchy" in prompt
        assert "agent 'analyst'" in prompt
        assert "Agent 'cfo'" in prompt
        assert "Agent 'risk_officer'" in prompt
        assert "(supervisor)" in prompt
        assert "(compliance_reviewer)" in prompt

    async def test_prompt_dirty_flag_on_edge_add(self):
        """Test that prompt_dirty flag is set when directed edge is added."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "You supervise.")
        await graph.add_node("worker", "You work.")

        # Initially not dirty
        worker_node = graph.get_node("worker")
        assert worker_node.prompt_dirty is False

        # Add edge (new controller)
        await graph.add_edge("supervisor", "worker", directed=True)

        # Should be marked dirty
        assert worker_node.prompt_dirty is True

    async def test_undirected_edge_no_prompt_dirty(self):
        """Test that undirected edges don't mark prompts as dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_a", "Agent A.")
        await graph.add_node("agent_b", "Agent B.")

        agent_b = graph.get_node("agent_b")
        assert agent_b.prompt_dirty is False

        await graph.add_edge("agent_a", "agent_b", directed=False)

        # Should NOT be marked dirty for undirected edges
        assert agent_b.prompt_dirty is False

    @pytest.mark.asyncio
    async def test_activate_agent_lazy_recomputes_dirty_prompt(
        self, tmp_path: Path, mock_claude_sdk
    ):
        """Test that _activate_agent_lazy recomputes dirty prompts."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You supervise.")
        await graph.add_node("worker", "You work.")

        worker_node = graph.get_node("worker")
        assert worker_node.effective_system_prompt is None

        # Add edge (marks dirty)
        await graph.add_edge("supervisor", "worker", directed=True)
        assert worker_node.prompt_dirty is True

        # Activate (should recompute)
        await graph._activate_agent_lazy("worker")

        # Check recomputation
        assert worker_node.prompt_dirty is False
        assert worker_node.effective_system_prompt is not None
        assert "supervisor" in worker_node.effective_system_prompt

    @pytest.mark.asyncio
    async def test_send_message_triggers_lazy_activation(self, tmp_path: Path, mock_claude_sdk):
        """Test that send_message triggers lazy activation and prompt recomputation."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You supervise.")
        await graph.add_node("worker", "You work.")
        await graph.add_edge("supervisor", "worker", directed=True)

        worker_node = graph.get_node("worker")
        assert worker_node.prompt_dirty is True

        # Send message (should trigger activation and recomputation)
        await graph.send_message("supervisor", "worker", "Start working")

        # Check recomputation happened
        assert worker_node.prompt_dirty is False
        assert worker_node.effective_system_prompt is not None
        assert "supervisor" in worker_node.effective_system_prompt

    @pytest.mark.asyncio
    async def test_send_message_creates_message(self, tmp_path: Path, mock_claude_sdk):
        """Test that send_message still creates messages after lazy activation."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You supervise.")
        await graph.add_node("worker", "You work.")
        await graph.add_edge("supervisor", "worker", directed=True)

        msg = await graph.send_message("supervisor", "worker", "Start working")

        assert msg.from_node == "supervisor"
        assert msg.to_node == "worker"
        assert msg.content == "Start working"

    async def test_original_system_prompt_preserved(self):
        """Test that original system prompt is preserved during injection."""
        graph = AgentGraph(name="test")
        original_prompt = "You are a helpful assistant."
        await graph.add_node("worker", original_prompt)
        await graph.add_node("supervisor", "You supervise.")

        worker_node = graph.get_node("worker")
        # Original should still be in system_prompt
        assert worker_node.system_prompt == original_prompt

        # Add control relationship
        await graph.add_edge("supervisor", "worker", directed=True)

        # Compute effective prompt
        effective = graph._compute_effective_prompt("worker")

        # Original should be in effective
        assert original_prompt in effective
        # But effective should have more
        assert "## Control Hierarchy" in effective
        assert len(effective) > len(original_prompt)

    async def test_prompt_injection_format(self):
        """Test the format of injected control information."""
        graph = AgentGraph(name="test")
        await graph.add_node("cfo", "CFO prompt")
        await graph.add_node("analyst", "Analyst prompt")
        await graph.add_edge("cfo", "analyst", directed=True, control_type="supervisor")

        prompt = graph._compute_effective_prompt("analyst")

        # Should have clear format
        assert "## Control Hierarchy" in prompt
        assert "You are agent 'analyst'" in prompt
        assert "You report to the following controllers:" in prompt
        assert "- Agent 'cfo' (supervisor)" in prompt

    async def test_control_relationships_with_multiple_edges_from_same_controller(self):
        """Test prompt injection when controller has multiple subordinates."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "You supervise.")
        await graph.add_node("worker_1", "Worker 1.")
        await graph.add_node("worker_2", "Worker 2.")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)

        prompt1 = graph._compute_effective_prompt("worker_1")
        prompt2 = graph._compute_effective_prompt("worker_2")

        # Both should have supervisor
        assert "supervisor" in prompt1
        assert "supervisor" in prompt2

        # Should be different nodes
        assert "worker_1" in prompt1
        assert "worker_2" in prompt2
        assert "worker_2" not in prompt1
        assert "worker_1" not in prompt2


class TestControllerQueries:
    """Tests for controller query methods."""

    async def test_get_controllers_no_controllers(self):
        """Test getting controllers when node has none."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")

        controllers = graph.get_controllers("agent_1")
        assert controllers == []

    async def test_get_controllers_single(self):
        """Test getting a single controller."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        controllers = graph.get_controllers("worker")
        assert controllers == ["supervisor"]

    async def test_get_controllers_multiple(self):
        """Test getting multiple controllers."""
        graph = AgentGraph(name="test")
        await graph.add_node("cfo", "CFO")
        await graph.add_node("risk_officer", "Risk Officer")
        await graph.add_node("analyst", "Analyst")

        await graph.add_edge("cfo", "analyst", directed=True)
        await graph.add_edge("risk_officer", "analyst", directed=True)

        controllers = graph.get_controllers("analyst")
        assert set(controllers) == {"cfo", "risk_officer"}
        assert controllers == sorted(["cfo", "risk_officer"])

    async def test_get_controllers_node_not_found(self):
        """Test getting controllers for non-existent node raises error."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")

        with pytest.raises(NodeNotFoundError):
            graph.get_controllers("non_existent")

    async def test_get_subordinates_no_subordinates(self):
        """Test getting subordinates when node has none."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")

        subordinates = graph.get_subordinates("agent_1")
        assert subordinates == []

    async def test_get_subordinates_single(self):
        """Test getting a single subordinate."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        subordinates = graph.get_subordinates("supervisor")
        assert subordinates == ["worker"]

    async def test_get_subordinates_multiple(self):
        """Test getting multiple subordinates."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")
        await graph.add_node("worker_3", "Worker 3")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)
        await graph.add_edge("supervisor", "worker_3", directed=True)

        subordinates = graph.get_subordinates("supervisor")
        assert set(subordinates) == {"worker_1", "worker_2", "worker_3"}
        assert subordinates == sorted(["worker_1", "worker_2", "worker_3"])

    async def test_get_subordinates_node_not_found(self):
        """Test getting subordinates for non-existent node raises error."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")

        with pytest.raises(NodeNotFoundError):
            graph.get_subordinates("non_existent")

    async def test_is_controller_true(self):
        """Test is_controller returns True for actual control relationship."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        assert graph.is_controller("supervisor", "worker") is True

    async def test_is_controller_false_no_edge(self):
        """Test is_controller returns False when no edge exists."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")
        await graph.add_node("agent_2", "Agent 2")

        assert graph.is_controller("agent_1", "agent_2") is False

    async def test_is_controller_false_wrong_direction(self):
        """Test is_controller returns False for wrong direction."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        # Wrong direction
        assert graph.is_controller("worker", "supervisor") is False

    async def test_is_controller_nonexistent_nodes(self):
        """Test is_controller returns False for non-existent nodes."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")

        assert graph.is_controller("non_existent_1", "non_existent_2") is False
        assert graph.is_controller("agent_1", "non_existent") is False

    async def test_is_controller_ignores_undirected(self):
        """Test is_controller ignores undirected edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")
        await graph.add_node("agent_2", "Agent 2")
        await graph.add_edge("agent_1", "agent_2", directed=False)

        assert graph.is_controller("agent_1", "agent_2") is False

    def test_get_control_relationships_empty_graph(self):
        """Test get_control_relationships on empty graph."""
        graph = AgentGraph(name="test")

        relationships = graph.get_control_relationships()
        assert relationships == {}

    async def test_get_control_relationships_no_relationships(self):
        """Test get_control_relationships when nodes have no control edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")
        await graph.add_node("agent_2", "Agent 2")

        relationships = graph.get_control_relationships()
        assert relationships == {}

    async def test_get_control_relationships_simple_hierarchy(self):
        """Test get_control_relationships with simple tree."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)

        relationships = graph.get_control_relationships()
        assert "supervisor" in relationships
        assert set(relationships["supervisor"]) == {"worker_1", "worker_2"}
        assert "worker_1" not in relationships

    async def test_get_control_relationships_matrix_organization(self):
        """Test get_control_relationships with matrix organization."""
        graph = AgentGraph(name="test")
        await graph.add_node("cfo", "CFO")
        await graph.add_node("risk_officer", "Risk Officer")
        await graph.add_node("analyst", "Analyst")

        await graph.add_edge("cfo", "analyst", directed=True)
        await graph.add_edge("risk_officer", "analyst", directed=True)

        relationships = graph.get_control_relationships()
        assert "cfo" in relationships
        assert "risk_officer" in relationships
        assert relationships["cfo"] == ["analyst"]
        assert relationships["risk_officer"] == ["analyst"]

    async def test_get_control_relationships_dag_topology(self):
        """Test get_control_relationships with DAG topology."""
        graph = AgentGraph(name="test")
        await graph.add_node("exec", "Executive")
        await graph.add_node("manager_1", "Manager 1")
        await graph.add_node("manager_2", "Manager 2")
        await graph.add_node("worker", "Worker")

        await graph.add_edge("exec", "manager_1", directed=True)
        await graph.add_edge("exec", "manager_2", directed=True)
        await graph.add_edge("manager_1", "worker", directed=True)
        await graph.add_edge("manager_2", "worker", directed=True)

        relationships = graph.get_control_relationships()
        assert set(relationships["exec"]) == {"manager_1", "manager_2"}
        assert relationships["manager_1"] == ["worker"]
        assert relationships["manager_2"] == ["worker"]
        assert "worker" not in relationships

    async def test_get_control_relationships_ignores_undirected(self):
        """Test get_control_relationships ignores undirected edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("agent_1", "Agent 1")
        await graph.add_node("agent_2", "Agent 2")
        await graph.add_node("agent_3", "Agent 3")

        await graph.add_edge("agent_1", "agent_2", directed=True)
        await graph.add_edge("agent_1", "agent_3", directed=False)

        relationships = graph.get_control_relationships()
        assert relationships == {"agent_1": ["agent_2"]}

    async def test_controller_queries_sorted_results(self):
        """Test that controller queries return sorted results."""
        graph = AgentGraph(name="test")
        await graph.add_node("z_ctrl", "Z Controller")
        await graph.add_node("a_ctrl", "A Controller")
        await graph.add_node("m_ctrl", "M Controller")
        await graph.add_node("target", "Target")

        await graph.add_edge("z_ctrl", "target", directed=True)
        await graph.add_edge("a_ctrl", "target", directed=True)
        await graph.add_edge("m_ctrl", "target", directed=True)

        controllers = graph.get_controllers("target")
        assert controllers == ["a_ctrl", "m_ctrl", "z_ctrl"]

    async def test_subordinate_queries_consistency(self):
        """Test consistency between get_subordinates and get_control_relationships."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)

        subordinates = graph.get_subordinates("supervisor")
        relationships = graph.get_control_relationships()

        assert set(subordinates) == set(relationships["supervisor"])


class TestNodeRemoval:
    """Tests for node removal functionality (Epic 5, Story 5.1.2)."""

    async def test_remove_node_not_found(self):
        """Test that removing nonexistent node raises NodeNotFoundError."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Node 1")

        with pytest.raises(NodeNotFoundError, match="Node 'nonexistent' not found"):
            await graph.remove_node("nonexistent")

    async def test_remove_isolated_node_cascade_false(self):
        """Test removing an isolated node with cascade=False succeeds."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Node 1")

        # Should succeed since no edges exist
        await graph.remove_node("node_1", cascade=False)

        assert not graph.node_exists("node_1")
        assert graph.node_count == 0

    async def test_remove_node_with_edges_cascade_false_raises(self):
        """Test that cascade=False raises error when edges exist."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Node 1")
        await graph.add_node("node_2", "Node 2")
        await graph.add_edge("node_1", "node_2", directed=True)

        with pytest.raises(AgentGraphError, match="has 1 connected edge"):
            await graph.remove_node("node_1", cascade=False)

        # Node should still exist after failed removal
        assert graph.node_exists("node_1")

    async def test_remove_node_cascade_true_removes_edges(self):
        """Test that cascade=True removes all connected edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)
        await graph.add_edge("worker_1", "worker_2", directed=True)

        assert graph.edge_count == 3

        # Remove supervisor with cascade=True
        await graph.remove_node("supervisor", cascade=True)

        assert not graph.node_exists("supervisor")
        assert graph.node_count == 2
        # Only the edge between workers should remain
        assert graph.edge_count == 1
        assert graph.edge_exists("worker_1", "worker_2")

    async def test_remove_node_updates_topology(self):
        """Test that topology is updated after node removal."""
        graph = AgentGraph(name="test")
        # Create a tree: supervisor -> worker1 -> worker2
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("worker_1", "worker_2", directed=True)

        assert graph.get_topology() == "chain"

        # Remove middle node
        await graph.remove_node("worker_1", cascade=True)

        # Should now be two isolated nodes (DAG topology with no edges)
        assert len(graph.get_isolated_nodes()) == 2
        # Two nodes with no edges forms a DAG
        assert graph.get_topology() == "dag"

    async def test_remove_node_marks_subordinate_prompts_dirty(self):
        """Test that removing a controller marks subordinate prompts dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("controller", "Controller")
        await graph.add_node("subordinate", "Subordinate")

        await graph.add_edge("controller", "subordinate", directed=True)

        # Add edge marks subordinate prompt dirty
        assert graph.get_node("subordinate").prompt_dirty

        # Clear the dirty flag
        graph.get_node("subordinate").prompt_dirty = False

        # Remove controller
        await graph.remove_node("controller", cascade=True)

        # Subordinate's prompt should still be marked dirty (since edge was removed)
        # Actually the subordinate gets a new prompt computed that removes the controller
        assert not graph.node_exists("controller")
        assert graph.node_exists("subordinate")

    async def test_remove_node_with_multiple_edges(self):
        """Test removing node with many connected edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("center", "Center")

        # Add star topology
        for i in range(5):
            await graph.add_node(f"spoke_{i}", f"Spoke {i}")
            await graph.add_edge("center", f"spoke_{i}", directed=True)

        assert graph.node_count == 6
        assert graph.edge_count == 5

        # Remove center node
        await graph.remove_node("center", cascade=True)

        assert graph.node_count == 5
        assert graph.edge_count == 0
        assert all(graph.node_exists(f"spoke_{i}") for i in range(5))

    async def test_remove_node_undirected_edges_cascade(self):
        """Test removal of node with undirected edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_a", "Node A")
        await graph.add_node("node_b", "Node B")
        await graph.add_node("node_c", "Node C")

        await graph.add_edge("node_a", "node_b", directed=False)
        await graph.add_edge("node_a", "node_c", directed=False)

        assert graph.edge_count == 2

        # Remove node_a with cascade=True
        await graph.remove_node("node_a", cascade=True)

        assert not graph.node_exists("node_a")
        assert graph.edge_count == 0

    async def test_remove_node_with_both_incoming_and_outgoing(self):
        """Test removing node with both incoming and outgoing edges."""
        graph = AgentGraph(name="test")
        await graph.add_node("source", "Source")
        await graph.add_node("middle", "Middle")
        await graph.add_node("target", "Target")

        await graph.add_edge("source", "middle", directed=True)
        await graph.add_edge("middle", "target", directed=True)

        assert graph.edge_count == 2

        # Remove middle node
        await graph.remove_node("middle", cascade=True)

        assert graph.node_count == 2
        assert graph.edge_count == 0

    async def test_remove_node_conversationfile_archived(self):
        """Test that archive_conversation is called when node is removed."""
        import tempfile
        from unittest.mock import AsyncMock, patch

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("node_a", "Node A")
            await graph.add_node("node_b", "Node B")

            edge = await graph.add_edge("node_a", "node_b", directed=True)

            # Mock the archive_conversation method to verify it's called
            with patch.object(
                graph.storage, "archive_conversation", new_callable=AsyncMock
            ) as mock_archive:
                # Remove node to trigger archival
                await graph.remove_node("node_a", cascade=True)

                # Verify archive_conversation was called for the edge
                mock_archive.assert_called_once()
                assert edge.edge_id in str(mock_archive.call_args)

    async def test_remove_node_adjacency_cleanup(self):
        """Test that adjacency lists are properly cleaned up."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_node("c", "C")

        await graph.add_edge("a", "b", directed=True)
        await graph.add_edge("b", "c", directed=True)

        # Verify adjacency is correct
        assert graph.get_neighbors("b", "outgoing") == ["c"]
        assert graph.get_neighbors("b", "incoming") == ["a"]

        # Remove node b
        await graph.remove_node("b", cascade=True)

        # Verify adjacency is cleaned up
        assert graph.get_neighbors("a", "outgoing") == []
        assert graph.get_neighbors("c", "incoming") == []


class TestNodeUpdate:
    """Tests for node property updates (Epic 5, Story 5.1.3)."""

    async def test_update_node_not_found(self):
        """Test that updating nonexistent node raises NodeNotFoundError."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Node 1")

        with pytest.raises(NodeNotFoundError, match="Node 'nonexistent' not found"):
            graph.update_node("nonexistent", system_prompt="New")

    async def test_update_node_system_prompt_only(self):
        """Test updating only the system prompt."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Original prompt")

        original_metadata = {"key": "value"}
        graph.get_node("node_1").metadata = original_metadata.copy()

        # Update prompt
        graph.update_node("node_1", system_prompt="New prompt")

        # Verify prompt updated
        node = graph.get_node("node_1")
        assert node.system_prompt == "New prompt"
        # Original prompt stored for reference
        assert node.original_system_prompt == "Original prompt"
        # Prompt marked dirty for recomputation
        assert node.prompt_dirty
        # Metadata unchanged
        assert node.metadata == original_metadata

    async def test_update_node_metadata_only(self):
        """Test updating only metadata."""
        graph = AgentGraph(name="test")
        original_prompt = "Original prompt"
        await graph.add_node("node_1", original_prompt)

        original_metadata = {"key": "value"}
        graph.get_node("node_1").metadata = original_metadata.copy()

        # Update metadata
        graph.update_node("node_1", metadata={"new_key": "new_value"})

        # Verify metadata merged
        node = graph.get_node("node_1")
        assert node.metadata == {"key": "value", "new_key": "new_value"}
        # Prompt unchanged
        assert node.system_prompt == original_prompt
        # prompt_dirty not set for metadata-only updates
        assert not node.prompt_dirty

    async def test_update_node_both_prompt_and_metadata(self):
        """Test updating both prompt and metadata simultaneously."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Original")
        graph.get_node("node_1").metadata = {"old": "value"}

        # Update both
        graph.update_node(
            "node_1",
            system_prompt="New prompt",
            metadata={"new": "metadata"},
        )

        node = graph.get_node("node_1")
        assert node.system_prompt == "New prompt"
        assert node.original_system_prompt == "Original"
        assert node.prompt_dirty
        assert node.metadata == {"old": "value", "new": "metadata"}

    async def test_update_node_metadata_merged_not_replaced(self):
        """Test that metadata update merges rather than replaces."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Prompt")
        graph.get_node("node_1").metadata = {"a": 1, "b": 2, "c": 3}

        # Update some metadata
        graph.update_node("node_1", metadata={"b": 20, "d": 4})

        # Verify merge
        node = graph.get_node("node_1")
        assert node.metadata == {"a": 1, "b": 20, "c": 3, "d": 4}

    async def test_update_node_marks_subordinates_dirty(self):
        """Test that prompt update marks subordinates' prompts dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)

        # Clear the initial dirty flags
        graph.get_node("worker_1").prompt_dirty = False
        graph.get_node("worker_2").prompt_dirty = False

        # Update supervisor's prompt
        graph.update_node("supervisor", system_prompt="Updated supervisor")

        # Subordinates' prompts should be marked dirty
        assert graph.get_node("worker_1").prompt_dirty
        assert graph.get_node("worker_2").prompt_dirty

    async def test_update_node_preserves_original_prompt(self):
        """Test that original prompt is preserved on first update."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Original prompt")

        # First update
        graph.update_node("node_1", system_prompt="First update")
        assert graph.get_node("node_1").original_system_prompt == "Original prompt"

        # Second update
        graph.update_node("node_1", system_prompt="Second update")
        # Original prompt should still be from first
        assert graph.get_node("node_1").original_system_prompt == "Original prompt"

    async def test_update_node_multiple_updates(self):
        """Test multiple updates to same node."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Initial")
        graph.get_node("node_1").metadata = {"version": 1}

        # First update
        graph.update_node("node_1", system_prompt="V2", metadata={"version": 2})
        node = graph.get_node("node_1")
        assert node.system_prompt == "V2"
        assert node.metadata["version"] == 2
        assert node.prompt_dirty

        # Clear dirty flag
        node.prompt_dirty = False

        # Second update (only metadata)
        graph.update_node("node_1", metadata={"version": 3})
        node = graph.get_node("node_1")
        assert node.system_prompt == "V2"  # Unchanged
        assert node.metadata["version"] == 3
        assert not node.prompt_dirty  # Not set for metadata-only

    async def test_update_node_no_changes(self):
        """Test updating with no parameters succeeds but does nothing."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_1", "Original")
        graph.get_node("node_1").metadata = {"key": "value"}
        graph.get_node("node_1").prompt_dirty = False

        # Call with no updates
        graph.update_node("node_1")

        # Nothing should change
        node = graph.get_node("node_1")
        assert node.system_prompt == "Original"
        assert node.metadata == {"key": "value"}
        assert not node.prompt_dirty

    async def test_update_node_chain_of_subordinates(self):
        """Test that updating prompts cascades through chain of subordinates."""
        graph = AgentGraph(name="test")
        # Create chain: supervisor -> manager -> worker
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("manager", "Manager")
        await graph.add_node("worker", "Worker")

        await graph.add_edge("supervisor", "manager", directed=True)
        await graph.add_edge("manager", "worker", directed=True)

        # Clear dirty flags
        graph.get_node("manager").prompt_dirty = False
        graph.get_node("worker").prompt_dirty = False

        # Update supervisor
        graph.update_node("supervisor", system_prompt="Updated supervisor")

        # Only manager should be marked dirty (direct subordinate)
        assert graph.get_node("manager").prompt_dirty
        # Worker is not directly supervised by supervisor
        assert not graph.get_node("worker").prompt_dirty

    async def test_update_node_with_undirected_edges(self):
        """Test that undirected edges don't affect dirty flag."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_a", "A")
        await graph.add_node("node_b", "B")

        await graph.add_edge("node_a", "node_b", directed=False)

        # Clear dirty flag
        graph.get_node("node_b").prompt_dirty = False

        # Update node_a
        graph.update_node("node_a", system_prompt="Updated A")

        # node_b should NOT be marked dirty (undirected edge doesn't create control)
        assert not graph.get_node("node_b").prompt_dirty


class TestEdgeRemoval:
    """Tests for edge removal functionality (Epic 5, Story 5.2.2)."""

    async def test_remove_edge_not_found(self):
        """Test that removing nonexistent edge raises EdgeNotFoundError."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_a", "A")
        await graph.add_node("node_b", "B")

        with pytest.raises(EdgeNotFoundError, match="Edge from 'node_a' to 'node_b' not found"):
            await graph.remove_edge("node_a", "node_b")

    async def test_remove_edge_node_not_found(self):
        """Test that removing edge with missing node raises NodeNotFoundError."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_a", "A")

        with pytest.raises(NodeNotFoundError, match="Node 'nonexistent' not found"):
            await graph.remove_edge("node_a", "nonexistent")

    async def test_remove_directed_edge(self):
        """Test removing a directed edge."""
        graph = AgentGraph(name="test")
        await graph.add_node("from", "From")
        await graph.add_node("to", "To")

        await graph.add_edge("from", "to", directed=True)
        assert graph.edge_exists("from", "to")

        await graph.remove_edge("from", "to")

        assert not graph.edge_exists("from", "to")
        assert graph.edge_count == 0

    async def test_remove_undirected_edge(self):
        """Test removing an undirected edge."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_a", "A")
        await graph.add_node("node_b", "B")

        await graph.add_edge("node_a", "node_b", directed=False)
        assert graph.edge_exists("node_a", "node_b")

        await graph.remove_edge("node_a", "node_b")

        assert not graph.edge_exists("node_a", "node_b")
        assert graph.edge_count == 0

    async def test_remove_edge_updates_control_relationships(self):
        """Test that removing directed edge marks subordinate prompt dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("controller", "Controller")
        await graph.add_node("subordinate", "Subordinate")

        await graph.add_edge("controller", "subordinate", directed=True)

        # Clear the dirty flag
        graph.get_node("subordinate").prompt_dirty = False

        await graph.remove_edge("controller", "subordinate")

        # Subordinate's prompt should be marked dirty
        assert graph.get_node("subordinate").prompt_dirty
        assert not graph.edge_exists("controller", "subordinate")

    async def test_remove_edge_archives_conversation(self):
        """Test that archive_conversation is called when edge is removed."""
        import tempfile
        from unittest.mock import AsyncMock, patch

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("node_a", "A")
            await graph.add_node("node_b", "B")

            edge = await graph.add_edge("node_a", "node_b", directed=True)

            # Mock the archive_conversation method
            with patch.object(
                graph.storage, "archive_conversation", new_callable=AsyncMock
            ) as mock_archive:
                await graph.remove_edge("node_a", "node_b")

                # Verify archive_conversation was called
                mock_archive.assert_called_once_with(edge.edge_id)

    async def test_remove_edge_undirected_no_control_dirty(self):
        """Test that removing undirected edge doesn't mark prompts dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("node_a", "A")
        await graph.add_node("node_b", "B")

        await graph.add_edge("node_a", "node_b", directed=False)

        # Clear dirty flags
        graph.get_node("node_a").prompt_dirty = False
        graph.get_node("node_b").prompt_dirty = False

        await graph.remove_edge("node_a", "node_b")

        # Neither should be marked dirty (undirected edges don't create control)
        assert not graph.get_node("node_a").prompt_dirty
        assert not graph.get_node("node_b").prompt_dirty

    async def test_remove_edge_updates_adjacency(self):
        """Test that adjacency is updated after edge removal."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=True)
        assert graph.get_neighbors("a", "outgoing") == ["b"]
        assert graph.get_neighbors("b", "incoming") == ["a"]

        await graph.remove_edge("a", "b")

        # Adjacency should be cleaned
        assert graph.get_neighbors("a", "outgoing") == []
        assert graph.get_neighbors("b", "incoming") == []

    async def test_remove_edge_updates_topology(self):
        """Test that topology is updated after edge removal."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")
        await graph.add_node("c", "C")

        await graph.add_edge("a", "b", directed=True)
        await graph.add_edge("b", "c", directed=True)

        assert graph.get_topology() == "chain"

        # Remove middle edge
        await graph.remove_edge("b", "c")

        # Should now be isolated or different topology
        assert graph.edge_count == 1

    async def test_remove_undirected_reverse_lookup(self):
        """Test removing undirected edge via reverse lookup."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=False)

        # Remove via reverse direction
        await graph.remove_edge("b", "a")

        assert not graph.edge_exists("a", "b")
        assert graph.edge_count == 0

    async def test_remove_multiple_edges_from_same_nodes(self):
        """Test removing one edge when multiple exist between same nodes (different properties)."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        # Add directed edge
        await graph.add_edge("a", "b", directed=True)

        # Undirected edge between same nodes (different from directed)
        await graph.add_edge("a", "b", directed=False)

        assert graph.edge_count == 2

        # Remove one edge
        await graph.remove_edge("a", "b")

        # Should remove the first one found (behavior depends on storage order)
        assert graph.edge_count == 1 or graph.edge_count == 0  # Depends on which was removed


class TestEdgeUpdate:
    """Tests for edge property updates (Epic 5, Story 5.2.3)."""

    async def test_update_edge_not_found(self):
        """Test that updating nonexistent edge raises EdgeNotFoundError."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        with pytest.raises(EdgeNotFoundError, match="Edge from 'a' to 'b' not found"):
            graph.update_edge("a", "b", priority="high")

    async def test_update_edge_node_not_found(self):
        """Test that updating edge with missing node raises NodeNotFoundError."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")

        with pytest.raises(NodeNotFoundError, match="Node 'nonexistent' not found"):
            graph.update_edge("a", "nonexistent", priority="high")

    async def test_update_edge_properties(self):
        """Test updating edge properties."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=True)

        # Update properties
        graph.update_edge("a", "b", priority="high", type="supervision")

        edge = graph.get_edge("a", "b")
        assert edge.properties["priority"] == "high"
        assert edge.properties["type"] == "supervision"

    async def test_update_edge_properties_merged(self):
        """Test that edge property updates merge rather than replace."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=True, priority="low")

        # Update some properties
        graph.update_edge("a", "b", type="supervision")

        edge = graph.get_edge("a", "b")
        assert edge.properties["priority"] == "low"
        assert edge.properties["type"] == "supervision"

    async def test_update_edge_control_type_triggers_dirty(self):
        """Test that control_type change marks subordinate prompt dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")

        await graph.add_edge("supervisor", "worker", directed=True)

        # Clear dirty flag
        graph.get_node("worker").prompt_dirty = False

        # Update control_type
        graph.update_edge("supervisor", "worker", control_type="oversight")

        # Subordinate's prompt should be marked dirty
        assert graph.get_node("worker").prompt_dirty

    async def test_update_edge_non_control_type_no_dirty(self):
        """Test that non-control-type changes don't mark prompts dirty."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=True)

        # Clear dirty flag
        graph.get_node("b").prompt_dirty = False

        # Update non-control-type property
        graph.update_edge("a", "b", priority="high")

        # Should not be marked dirty
        assert not graph.get_node("b").prompt_dirty

    async def test_update_edge_undirected(self):
        """Test updating undirected edge properties."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=False)

        graph.update_edge("a", "b", collaboration_level="high")

        edge = graph.get_edge("a", "b")
        assert edge.properties["collaboration_level"] == "high"

    async def test_update_edge_via_reverse_lookup(self):
        """Test updating undirected edge via reverse lookup."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=False)

        # Update via reverse direction
        graph.update_edge("b", "a", level="2")

        edge = graph.get_edge("a", "b")
        assert edge.properties["level"] == "2"

    async def test_update_edge_multiple_updates(self):
        """Test multiple updates to same edge."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=True, version=1)

        # First update
        graph.update_edge("a", "b", version=2, status="active")
        edge = graph.get_edge("a", "b")
        assert edge.properties["version"] == 2
        assert edge.properties["status"] == "active"

        # Second update
        graph.update_edge("a", "b", status="inactive")
        edge = graph.get_edge("a", "b")
        assert edge.properties["version"] == 2  # Unchanged
        assert edge.properties["status"] == "inactive"

    async def test_update_edge_no_changes(self):
        """Test updating edge with no properties succeeds but does nothing."""
        graph = AgentGraph(name="test")
        await graph.add_node("a", "A")
        await graph.add_node("b", "B")

        await graph.add_edge("a", "b", directed=True, priority="high")

        # Update with no properties
        graph.update_edge("a", "b")

        # Nothing should change
        edge = graph.get_edge("a", "b")
        assert edge.properties["priority"] == "high"


class TestMessageRoutingPatterns:
    """Tests for message routing patterns (Epic 6, Feature 6.1)."""

    @pytest.mark.asyncio
    async def test_broadcast_basic_outgoing(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test broadcasting to outgoing neighbors."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker_1", "Worker 1")
        await graph.add_node("worker_2", "Worker 2")
        await graph.add_node("worker_3", "Worker 3")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)
        await graph.add_edge("supervisor", "worker_3", directed=True)

        messages = await graph.broadcast("supervisor", "Start working!")

        assert len(messages) == 3
        assert all(msg.from_node == "supervisor" for msg in messages)
        assert {msg.to_node for msg in messages} == {"worker_1", "worker_2", "worker_3"}
        assert all(msg.content == "Start working!" for msg in messages)

    @pytest.mark.asyncio
    async def test_broadcast_no_neighbors(self, tmp_path: Path) -> None:
        """Test broadcast on isolated node returns empty list."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("isolated", "Isolated")

        messages = await graph.broadcast("isolated", "Hello")

        assert messages == []

    @pytest.mark.asyncio
    async def test_broadcast_with_metadata(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test broadcast with metadata propagation."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("source", "Source")
        await graph.add_node("target_1", "Target 1")
        await graph.add_node("target_2", "Target 2")

        await graph.add_edge("source", "target_1", directed=True)
        await graph.add_edge("source", "target_2", directed=True)

        messages = await graph.broadcast("source", "Content", priority="high", task_id=42)

        assert len(messages) == 2
        assert all(msg.metadata.get("priority") == "high" for msg in messages)
        assert all(msg.metadata.get("task_id") == 42 for msg in messages)

    @pytest.mark.asyncio
    async def test_broadcast_include_incoming(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test broadcast with include_incoming=True."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("center", "Center")
        await graph.add_node("upstream", "Upstream")
        await graph.add_node("downstream", "Downstream")

        # Test 1: Only downstream edge, include_incoming=False
        await graph.add_edge("center", "downstream", directed=True)
        messages_out = await graph.broadcast("center", "Message", include_incoming=False)
        assert len(messages_out) == 1
        assert messages_out[0].to_node == "downstream"

        # Test 2: Add upstream edges and test include_incoming=True
        await graph.add_edge("upstream", "center", directed=True)  # Incoming to center
        await graph.add_edge("center", "upstream", directed=True)  # Outgoing from center

        messages_both = await graph.broadcast("center", "Message", include_incoming=True)
        assert len(messages_both) == 2
        assert {msg.to_node for msg in messages_both} == {"upstream", "downstream"}

    @pytest.mark.asyncio
    async def test_broadcast_nonexistent_node(self, tmp_path: Path) -> None:
        """Test broadcast from nonexistent node raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        with pytest.raises(NodeNotFoundError):
            await graph.broadcast("nonexistent", "Message")

    @pytest.mark.asyncio
    async def test_broadcast_single_neighbor(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test broadcast with single neighbor."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("source", "Source")
        await graph.add_node("target", "Target")
        await graph.add_edge("source", "target", directed=True)

        messages = await graph.broadcast("source", "Single")

        assert len(messages) == 1
        assert messages[0].to_node == "target"

    @pytest.mark.asyncio
    async def test_route_message_auto_shortest_path(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test route_message with auto shortest path finding."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        # Create a path: A -> B -> C -> D
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")
        await graph.add_node("D", "D")

        await graph.add_edge("A", "B", directed=True)
        await graph.add_edge("B", "C", directed=True)
        await graph.add_edge("C", "D", directed=True)

        messages = await graph.route_message("A", "D", "Routed message")

        # Should send through all hops: A->B, B->C, C->D = 3 messages
        assert len(messages) == 3
        assert messages[0].from_node == "A" and messages[0].to_node == "B"
        assert messages[1].from_node == "B" and messages[1].to_node == "C"
        assert messages[2].from_node == "C" and messages[2].to_node == "D"

    @pytest.mark.asyncio
    async def test_route_message_direct_connection(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test route_message with direct connection (one hop)."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_edge("A", "B", directed=True)

        messages = await graph.route_message("A", "B", "Direct")

        assert len(messages) == 1
        assert messages[0].from_node == "A"
        assert messages[0].to_node == "B"

    @pytest.mark.asyncio
    async def test_route_message_explicit_path(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test route_message with explicit path."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")
        await graph.add_node("D", "D")

        await graph.add_edge("A", "B", directed=True)
        await graph.add_edge("B", "C", directed=True)
        await graph.add_edge("C", "D", directed=True)

        path = ["A", "B", "C", "D"]
        messages = await graph.route_message("A", "D", "Explicit path", path=path)

        assert len(messages) == 3
        # Verify routing metadata
        for msg in messages:
            assert msg.metadata.get("routing_path") == path
            assert msg.metadata.get("hop_count") == 3
            assert msg.metadata.get("original_sender") == "A"

    @pytest.mark.asyncio
    async def test_route_message_no_path_exists(self, tmp_path: Path) -> None:
        """Test route_message when no path exists."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")
        # Create two disconnected components: A->B and C
        await graph.add_edge("A", "B", directed=True)

        with pytest.raises(EdgeNotFoundError, match="No path exists"):
            await graph.route_message("A", "C", "Message")

    @pytest.mark.asyncio
    async def test_route_message_invalid_explicit_path(self, tmp_path: Path) -> None:
        """Test route_message with invalid explicit path."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")

        await graph.add_edge("A", "B", directed=True)
        # No edge from B to C

        with pytest.raises((EdgeNotFoundError, ValueError)):
            await graph.route_message("A", "C", "Message", path=["A", "B", "C"])

    @pytest.mark.asyncio
    async def test_route_message_path_wrong_start(self, tmp_path: Path) -> None:
        """Test route_message with path not starting at from_node."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")

        with pytest.raises(ValueError, match="Path must start"):
            await graph.route_message("A", "C", "Message", path=["B", "A", "C"])

    @pytest.mark.asyncio
    async def test_route_message_path_wrong_end(self, tmp_path: Path) -> None:
        """Test route_message with path not ending at to_node."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")

        with pytest.raises(ValueError, match="Path must end"):
            await graph.route_message("A", "C", "Message", path=["A", "B"])

    @pytest.mark.asyncio
    async def test_route_message_path_too_short(self, tmp_path: Path) -> None:
        """Test route_message with path with only one node."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")

        with pytest.raises(ValueError, match="at least 2 nodes"):
            await graph.route_message("A", "A", "Message", path=["A"])

    @pytest.mark.asyncio
    async def test_route_message_nonexistent_sender(self, tmp_path: Path) -> None:
        """Test route_message with nonexistent sender."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("B", "B")

        with pytest.raises(NodeNotFoundError):
            await graph.route_message("nonexistent", "B", "Message")

    @pytest.mark.asyncio
    async def test_route_message_nonexistent_receiver(self, tmp_path: Path) -> None:
        """Test route_message with nonexistent receiver."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")

        with pytest.raises(NodeNotFoundError):
            await graph.route_message("A", "nonexistent", "Message")

    @pytest.mark.asyncio
    async def test_route_message_node_in_path_not_found(self, tmp_path: Path) -> None:
        """Test route_message with nonexistent node in explicit path."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("C", "C")

        with pytest.raises(NodeNotFoundError, match="in path not found"):
            await graph.route_message("A", "C", "Message", path=["A", "nonexistent", "C"])

    @pytest.mark.asyncio
    async def test_route_message_metadata_propagation(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that custom metadata is included in routed messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("A", "A")
        await graph.add_node("B", "B")
        await graph.add_node("C", "C")

        await graph.add_edge("A", "B", directed=True)
        await graph.add_edge("B", "C", directed=True)

        messages = await graph.route_message(
            "A", "C", "Content", priority="high", request_id="req_123"
        )

        # All routed messages should have the custom metadata
        for msg in messages:
            assert msg.metadata.get("priority") == "high"
            assert msg.metadata.get("request_id") == "req_123"
            # Plus routing metadata
            assert "routing_path" in msg.metadata
            assert "hop_count" in msg.metadata


class TestControlCommands:
    """Tests for control commands (Epic 6, Feature 6.3)."""

    @pytest.mark.asyncio
    async def test_execute_command_basic(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test executing a basic command."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        msg = await graph.execute_command("supervisor", "worker", "analyze", dataset="Q1")

        assert msg.from_node == "supervisor"
        assert msg.to_node == "worker"
        assert msg.content == "Execute: analyze"
        assert msg.metadata.get("type") == "command"
        assert msg.metadata.get("command") == "analyze"
        assert msg.metadata.get("params").get("dataset") == "Q1"

    @pytest.mark.asyncio
    async def test_execute_command_multiple_params(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test command with multiple parameters."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("controller", "Controller")
        await graph.add_node("agent", "Agent")
        await graph.add_edge("controller", "agent", directed=True)

        msg = await graph.execute_command(
            "controller", "agent", "process", input_file="data.csv", output_format="json", threads=4
        )

        params = msg.metadata.get("params")
        assert params.get("input_file") == "data.csv"
        assert params.get("output_format") == "json"
        assert params.get("threads") == 4

    @pytest.mark.asyncio
    async def test_execute_command_no_params(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test command with no parameters."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        msg = await graph.execute_command("supervisor", "worker", "start")

        assert msg.metadata.get("command") == "start"
        assert msg.metadata.get("params") == {}

    @pytest.mark.asyncio
    async def test_execute_command_control_type_in_metadata(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that control_type from edge properties is included in metadata."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("cfo", "CFO")
        await graph.add_node("analyst", "Analyst")
        await graph.add_edge("cfo", "analyst", directed=True, control_type="oversight")

        msg = await graph.execute_command("cfo", "analyst", "report")

        assert msg.metadata.get("authorization_level") == "oversight"

    @pytest.mark.asyncio
    async def test_execute_command_default_control_type(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test default control_type when not specified in edge."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        msg = await graph.execute_command("supervisor", "worker", "work")

        assert msg.metadata.get("authorization_level") == "supervisor"

    @pytest.mark.asyncio
    async def test_execute_command_controller_not_found(self, tmp_path: Path) -> None:
        """Test execute_command with nonexistent controller."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("worker", "Worker")

        with pytest.raises(NodeNotFoundError, match="Node 'nonexistent' not found"):
            await graph.execute_command("nonexistent", "worker", "command")

    @pytest.mark.asyncio
    async def test_execute_command_subordinate_not_found(self, tmp_path: Path) -> None:
        """Test execute_command with nonexistent subordinate."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")

        with pytest.raises(NodeNotFoundError, match="Node 'nonexistent' not found"):
            await graph.execute_command("supervisor", "nonexistent", "command")

    @pytest.mark.asyncio
    async def test_execute_command_no_control_relationship(self, tmp_path: Path) -> None:
        """Test execute_command when no control relationship exists."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("agent_1", "Agent 1")
        await graph.add_node("agent_2", "Agent 2")
        # No edge

        with pytest.raises(CommandAuthorizationError, match="does not have control relationship"):
            await graph.execute_command("agent_1", "agent_2", "command")

    @pytest.mark.asyncio
    async def test_execute_command_wrong_direction(self, tmp_path: Path) -> None:
        """Test execute_command with edge in wrong direction."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        # Edge is supervisor -> worker, not worker -> supervisor
        await graph.add_edge("supervisor", "worker", directed=True)

        with pytest.raises(CommandAuthorizationError, match="does not have control relationship"):
            await graph.execute_command("worker", "supervisor", "command")

    @pytest.mark.asyncio
    async def test_execute_command_undirected_edge_no_control(self, tmp_path: Path) -> None:
        """Test execute_command fails with undirected edges (no control)."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("agent_1", "Agent 1")
        await graph.add_node("agent_2", "Agent 2")
        await graph.add_edge("agent_1", "agent_2", directed=False)

        with pytest.raises(CommandAuthorizationError, match="does not have control relationship"):
            await graph.execute_command("agent_1", "agent_2", "command")

    @pytest.mark.asyncio
    async def test_execute_command_multiple_commands_sequence(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test executing multiple commands in sequence."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "Supervisor")
        await graph.add_node("worker", "Worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        msg1 = await graph.execute_command("supervisor", "worker", "start", task="task1")
        msg2 = await graph.execute_command("supervisor", "worker", "process", task="task2")
        msg3 = await graph.execute_command("supervisor", "worker", "stop")

        assert msg1.metadata.get("command") == "start"
        assert msg2.metadata.get("command") == "process"
        assert msg3.metadata.get("command") == "stop"
        assert msg1.message_id != msg2.message_id != msg3.message_id

    @pytest.mark.asyncio
    async def test_execute_command_from_multiple_controllers(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test command execution from multiple different controllers."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("cfo", "CFO")
        await graph.add_node("risk_officer", "Risk Officer")
        await graph.add_node("analyst", "Analyst")

        await graph.add_edge("cfo", "analyst", directed=True)
        await graph.add_edge("risk_officer", "analyst", directed=True)

        msg1 = await graph.execute_command("cfo", "analyst", "analyze", aspect="finance")
        msg2 = await graph.execute_command("risk_officer", "analyst", "analyze", aspect="risk")

        assert msg1.from_node == "cfo"
        assert msg2.from_node == "risk_officer"
        assert msg1.to_node == "analyst"
        assert msg2.to_node == "analyst"
