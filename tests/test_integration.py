"""
Integration tests for claude-agent-graph.

These tests verify that different components work together correctly,
including graph construction, messaging, agent lifecycle, and persistence.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.exceptions import (
    DuplicateEdgeError,
    NodeNotFoundError,
    TopologyViolationError,
)


class TestBasicGraphIntegration:
    """Test basic graph construction and operation integration."""

    async def test_create_graph_add_nodes_and_edges(self):
        """Test creating a graph and adding nodes and edges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="integration_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Add nodes
            await graph.add_node("supervisor", "You are a supervisor agent.")
            await graph.add_node("worker1", "You are worker 1.")
            await graph.add_node("worker2", "You are worker 2.")

            assert graph.node_count == 3
            assert graph.node_exists("supervisor")
            assert graph.node_exists("worker1")
            assert graph.node_exists("worker2")

            # Add edges
            await graph.add_edge("supervisor", "worker1", directed=True)
            await graph.add_edge("supervisor", "worker2", directed=True)

            assert graph.edge_count == 2
            assert graph.edge_exists("supervisor", "worker1")
            assert graph.edge_exists("supervisor", "worker2")

    async def test_message_sending_between_nodes(self):
        """Test sending messages between connected nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="messaging_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("alice", "You are Alice.")
            await graph.add_node("bob", "You are Bob.")
            await graph.add_edge("alice", "bob", directed=False)

            # Send message from Alice to Bob
            msg = await graph.send_message(
                from_node="alice",
                to_node="bob",
                content="Hello Bob!",
            )

            assert msg.from_node == "alice"
            assert msg.to_node == "bob"
            assert msg.content == "Hello Bob!"

            # Verify message is in conversation
            messages = await graph.get_conversation("alice", "bob")
            assert len(messages) == 1
            assert messages[0].content == "Hello Bob!"

    async def test_control_relationships(self):
        """Test that control relationships update system prompts correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="control_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("controller", "You are a controller.")
            await graph.add_node("subordinate", "You are a subordinate.")
            await graph.add_edge("controller", "subordinate", directed=True)

            # Get effective prompt
            effective_prompt = graph._compute_effective_prompt("subordinate")

            # Should contain controller information
            assert "controller" in effective_prompt.lower()
            assert "You are a subordinate" in effective_prompt


class TestMultiNodeIntegration:
    """Test integration scenarios with multiple nodes."""

    async def test_hierarchical_graph_messaging(self):
        """Test message flow in a hierarchical graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="hierarchy_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Create hierarchy: manager -> team_lead -> worker
            await graph.add_node("manager", "You are the manager.")
            await graph.add_node("team_lead", "You are the team lead.")
            await graph.add_node("worker", "You are a worker.")

            await graph.add_edge("manager", "team_lead", directed=True)
            await graph.add_edge("team_lead", "worker", directed=True)

            # Send message from manager to team lead
            await graph.send_message("manager", "team_lead", "Delegate task X")

            # Send message from team lead to worker
            await graph.send_message("team_lead", "worker", "Execute task X")

            # Verify conversations exist
            conv1 = await graph.get_conversation("manager", "team_lead")
            conv2 = await graph.get_conversation("team_lead", "worker")

            assert len(conv1) == 1
            assert len(conv2) == 1
            assert conv1[0].content == "Delegate task X"
            assert conv2[0].content == "Execute task X"

    async def test_mesh_network_broadcasting(self):
        """Test broadcasting in a fully connected mesh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="mesh_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Create mesh network
            nodes = ["node1", "node2", "node3", "node4"]
            for node in nodes:
                await graph.add_node(node, f"You are {node}.")

            # Connect all nodes (mesh)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1 :]:
                    await graph.add_edge(n1, n2, directed=False)

            # Broadcast from node1
            await graph.broadcast("node1", "Hello everyone!", direction="outgoing")

            # Verify all neighbors received the message
            for node in ["node2", "node3", "node4"]:
                conv = await graph.get_conversation("node1", node)
                assert len(conv) == 1
                assert conv[0].content == "Hello everyone!"


class TestDynamicGraphModification:
    """Test dynamic graph modification scenarios."""

    async def test_add_node_to_running_graph(self):
        """Test adding a node to an existing graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="dynamic_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Initial setup
            await graph.add_node("existing", "Existing node")
            assert graph.node_count == 1

            # Add new node
            await graph.add_node("new", "New node")
            assert graph.node_count == 2

            # Connect them
            await graph.add_edge("existing", "new", directed=True)
            assert graph.edge_count == 1

    async def test_remove_node_with_cascade(self):
        """Test removing a node and cascading edge removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="removal_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("central", "Central node")
            await graph.add_node("peripheral1", "Peripheral 1")
            await graph.add_node("peripheral2", "Peripheral 2")

            await graph.add_edge("central", "peripheral1", directed=True)
            await graph.add_edge("central", "peripheral2", directed=True)

            # Remove central node with cascade
            await graph.remove_node("central", cascade=True)

            assert graph.node_count == 2
            assert graph.edge_count == 0
            assert not graph.node_exists("central")

    async def test_update_node_system_prompt(self):
        """Test updating a node's system prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="update_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("agent", "Original prompt")
            node = graph.get_node("agent")
            assert node.system_prompt == "Original prompt"

            # Update prompt
            graph.update_node("agent", system_prompt="Updated prompt")
            node = graph.get_node("agent")
            assert node.system_prompt == "Updated prompt"


class TestTopologyConstraints:
    """Test topology constraint enforcement."""

    async def test_tree_topology_prevents_cycles(self):
        """Test that tree topology prevents cycle creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="tree_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                topology_constraint="tree",
            )

            await graph.add_node("root", "Root")
            await graph.add_node("child", "Child")
            await graph.add_edge("root", "child", directed=True)

            # Try to create a cycle (should fail)
            with pytest.raises(TopologyViolationError):
                await graph.add_edge("child", "root", directed=True)

    async def test_dag_topology_prevents_cycles(self):
        """Test that DAG topology prevents cycle creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="dag_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
                topology_constraint="dag",
            )

            await graph.add_node("a", "Node A")
            await graph.add_node("b", "Node B")
            await graph.add_node("c", "Node C")

            await graph.add_edge("a", "b", directed=True)
            await graph.add_edge("b", "c", directed=True)

            # Try to create a cycle (should fail)
            with pytest.raises(TopologyViolationError):
                await graph.add_edge("c", "a", directed=True)


class TestCheckpointIntegration:
    """Test checkpoint and persistence integration."""

    async def test_save_and_load_checkpoint(self):
        """Test saving and loading a graph checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.msgpack"

            # Create and save graph
            graph1 = AgentGraph(
                name="checkpoint_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph1.add_node("node1", "Prompt 1")
            await graph1.add_node("node2", "Prompt 2")
            await graph1.add_edge("node1", "node2", directed=True)

            graph1.save_checkpoint(str(checkpoint_path))

            # Load into new graph
            graph2 = AgentGraph.load_checkpoint(str(checkpoint_path))

            assert graph2.name == "checkpoint_test"
            assert graph2.node_count == 2
            assert graph2.edge_count == 1
            assert graph2.node_exists("node1")
            assert graph2.node_exists("node2")

    async def test_checkpoint_preserves_messages(self):
        """Test that checkpoints preserve conversation history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint_msg.msgpack"

            # Create graph and send messages
            graph1 = AgentGraph(
                name="msg_checkpoint_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph1.add_node("alice", "Alice")
            await graph1.add_node("bob", "Bob")
            await graph1.add_edge("alice", "bob", directed=False)

            await graph1.send_message("alice", "bob", "Hello")
            await graph1.send_message("bob", "alice", "Hi there")

            graph1.save_checkpoint(str(checkpoint_path))

            # Load and verify messages
            graph2 = AgentGraph.load_checkpoint(str(checkpoint_path))
            messages = await graph2.get_conversation("alice", "bob")

            assert len(messages) == 2
            assert messages[0].content == "Hello"
            assert messages[1].content == "Hi there"


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    async def test_send_message_without_edge(self):
        """Test that sending messages without an edge raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="error_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("node1", "Node 1")
            await graph.add_node("node2", "Node 2")

            # Try to send message without edge
            with pytest.raises(Exception):  # Should raise appropriate error
                await graph.send_message("node1", "node2", "Hello")

    async def test_add_duplicate_edge(self):
        """Test that adding duplicate edges raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="duplicate_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            await graph.add_node("node1", "Node 1")
            await graph.add_node("node2", "Node 2")
            await graph.add_edge("node1", "node2", directed=True)

            # Try to add duplicate edge
            with pytest.raises(DuplicateEdgeError):
                await graph.add_edge("node1", "node2", directed=True)

    async def test_operations_on_nonexistent_node(self):
        """Test that operations on nonexistent nodes raise errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="nonexistent_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            with pytest.raises(NodeNotFoundError):
                graph.get_node("nonexistent")

            with pytest.raises(NodeNotFoundError):
                await graph.update_node("nonexistent", system_prompt="New")


class TestConcurrency:
    """Test concurrent operations."""

    async def test_concurrent_message_sending(self):
        """Test sending messages concurrently from multiple nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="concurrent_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Create star topology
            await graph.add_node("center", "Center")
            nodes = ["node1", "node2", "node3", "node4", "node5"]
            for node in nodes:
                await graph.add_node(node, f"Node {node}")
                await graph.add_edge("center", node, directed=False)

            # Send messages concurrently
            tasks = []
            for i, node in enumerate(nodes):
                tasks.append(graph.send_message("center", node, f"Message {i}"))

            results = await asyncio.gather(*tasks)

            # Verify all messages sent
            assert len(results) == 5
            for i, node in enumerate(nodes):
                conv = await graph.get_conversation("center", node)
                assert len(conv) == 1
                assert f"Message {i}" in conv[0].content

    async def test_concurrent_node_addition(self):
        """Test adding nodes concurrently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="concurrent_add_test",
                storage_backend=FilesystemBackend(base_dir=tmpdir),
            )

            # Add nodes concurrently
            tasks = []
            for i in range(10):
                tasks.append(graph.add_node(f"node{i}", f"Node {i}"))

            await asyncio.gather(*tasks)

            assert graph.node_count == 10
            for i in range(10):
                assert graph.node_exists(f"node{i}")
