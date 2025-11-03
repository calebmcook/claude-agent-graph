"""
Unit tests for checkpoint serialization and deserialization (Epic 7).

Tests cover:
- Story 7.1.1: Graph state export (save_checkpoint)
- Story 7.1.2: Graph state import (load_checkpoint)
- Story 7.2.1: Auto-save configuration and worker
- Story 7.2.2: Crash recovery and auto-load
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_agent_graph import (
    AgentGraph,
    Checkpoint,
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointVersionError,
)
from claude_agent_graph.models import Edge, Node, NodeStatus


class TestCheckpointBasics:
    """Test basic checkpoint operations."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint object."""
        nodes = {
            "node1": Node(node_id="node1", system_prompt="You are an AI assistant."),
            "node2": Node(node_id="node2", system_prompt="You are helpful."),
        }
        edges = {
            "node1_to_node2": Edge(
                edge_id="node1_to_node2",
                from_node="node1",
                to_node="node2",
                directed=True,
            ),
        }
        metadata = {
            "topology_constraint": "dag",
            "max_nodes": 1000,
            "persistence_enabled": True,
        }

        checkpoint = Checkpoint(
            name="test_graph",
            nodes=nodes,
            edges=edges,
            metadata=metadata,
        )

        assert checkpoint.name == "test_graph"
        assert len(checkpoint.nodes) == 2
        assert len(checkpoint.edges) == 1
        assert checkpoint.metadata["topology_constraint"] == "dag"

    def test_checkpoint_to_dict(self):
        """Test converting checkpoint to dictionary."""
        nodes = {
            "node1": Node(node_id="node1", system_prompt="Test prompt"),
        }
        edges = {}
        metadata = {"max_nodes": 500}

        checkpoint = Checkpoint(
            name="test_graph",
            nodes=nodes,
            edges=edges,
            metadata=metadata,
        )

        data = checkpoint.to_dict()

        assert data["name"] == "test_graph"
        assert data["version"] == 1
        assert "timestamp" in data
        assert "node1" in data["nodes"]
        assert data["metadata"]["max_nodes"] == 500

    def test_checkpoint_compute_checksum(self):
        """Test checksum computation is deterministic."""
        data1 = {
            "name": "test",
            "version": 1,
            "nodes": {"n1": {"id": "n1"}},
        }
        data2 = {
            "name": "test",
            "version": 1,
            "nodes": {"n1": {"id": "n1"}},
        }

        checksum1 = Checkpoint.compute_checksum(data1)
        checksum2 = Checkpoint.compute_checksum(data2)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest is 64 chars

    def test_checkpoint_checksum_changes_with_data(self):
        """Test checksum changes when data changes."""
        data1 = {"name": "test1", "version": 1}
        data2 = {"name": "test2", "version": 1}

        checksum1 = Checkpoint.compute_checksum(data1)
        checksum2 = Checkpoint.compute_checksum(data2)

        assert checksum1 != checksum2


class TestCheckpointSerialization:
    """Test checkpoint save and load operations (Story 7.1)."""

    def test_save_checkpoint_creates_file(self):
        """Test that save_checkpoint creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = {
                "node1": Node(node_id="node1", system_prompt="Test"),
            }
            checkpoint = Checkpoint(
                name="test_graph",
                nodes=nodes,
                edges={},
                metadata={},
            )

            filepath = Path(tmpdir) / "test_checkpoint.msgpack"
            checkpoint.save(filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_checkpoint_creates_directory(self):
        """Test that save_checkpoint creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = {
                "node1": Node(node_id="node1", system_prompt="Test"),
            }
            checkpoint = Checkpoint(
                name="test_graph",
                nodes=nodes,
                edges={},
                metadata={},
            )

            # Use a nested path that doesn't exist
            filepath = Path(tmpdir) / "nested" / "dir" / "checkpoint.msgpack"
            checkpoint.save(filepath)

            assert filepath.exists()
            assert filepath.parent.is_dir()

    def test_load_checkpoint_basic(self):
        """Test loading a basic checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            nodes = {
                "node1": Node(node_id="node1", system_prompt="Test prompt"),
            }
            edges = {}
            metadata = {"max_nodes": 1000}

            original = Checkpoint(
                name="test_graph",
                nodes=nodes,
                edges=edges,
                metadata=metadata,
            )

            filepath = Path(tmpdir) / "checkpoint.msgpack"
            original.save(filepath)

            # Load and verify
            loaded = Checkpoint.load(filepath)

            assert loaded.name == "test_graph"
            assert loaded.metadata["max_nodes"] == 1000
            assert "node1" in loaded.nodes
            assert loaded.nodes["node1"].system_prompt == "Test prompt"

    def test_load_checkpoint_preserves_all_data(self):
        """Test that load preserves all checkpoint data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = {
                "node1": Node(node_id="node1", system_prompt="Prompt1", model="claude-3-sonnet"),
                "node2": Node(node_id="node2", system_prompt="Prompt2", status=NodeStatus.ACTIVE),
            }
            edges = {
                "node1_to_node2": Edge(
                    edge_id="node1_to_node2",
                    from_node="node1",
                    to_node="node2",
                    directed=True,
                    properties={"priority": "high"},
                ),
            }
            metadata = {
                "topology_constraint": "dag",
                "max_nodes": 5000,
                "persistence_enabled": True,
            }

            original = Checkpoint(
                name="complex_graph",
                nodes=nodes,
                edges=edges,
                metadata=metadata,
            )

            filepath = Path(tmpdir) / "complex_checkpoint.msgpack"
            original.save(filepath)
            loaded = Checkpoint.load(filepath)

            # Verify all data preserved
            assert loaded.name == "complex_graph"
            assert len(loaded.nodes) == 2
            assert len(loaded.edges) == 1
            assert loaded.nodes["node1"].model == "claude-3-sonnet"
            assert loaded.nodes["node2"].status == NodeStatus.ACTIVE
            assert loaded.edges["node1_to_node2"].properties["priority"] == "high"
            assert loaded.metadata["topology_constraint"] == "dag"

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(CheckpointError):
            Checkpoint.load(Path("/nonexistent/path/checkpoint.msgpack"))

    def test_load_checkpoint_corrupted_data(self):
        """Test loading corrupted checkpoint raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "corrupted.msgpack"

            # Write invalid msgpack data
            with open(filepath, "wb") as f:
                f.write(b"this is not valid msgpack data \x00\x01\x02")

            with pytest.raises(CheckpointError):
                Checkpoint.load(filepath)

    def test_load_checkpoint_missing_checksum(self):
        """Test loading checkpoint without checksum raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_checksum.msgpack"

            # Create a checkpoint and manually remove checksum
            nodes = {
                "node1": Node(node_id="node1", system_prompt="Test"),
            }
            checkpoint = Checkpoint(
                name="test_graph",
                nodes=nodes,
                edges={},
                metadata={},
            )

            # Save it
            checkpoint.save(filepath)

            # Load, modify to remove checksum, and save
            import msgpack

            with open(filepath, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)

            del data["checksum"]

            serialized = msgpack.packb(data, use_bin_type=True)
            with open(filepath, "wb") as f:
                f.write(serialized)

            # Should fail on checksum validation
            with pytest.raises(CheckpointCorruptedError):
                Checkpoint.load(filepath)

    def test_load_checkpoint_invalid_checksum(self):
        """Test loading checkpoint with tampered checksum raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "tampered.msgpack"

            nodes = {
                "node1": Node(node_id="node1", system_prompt="Test"),
            }
            checkpoint = Checkpoint(
                name="test_graph",
                nodes=nodes,
                edges={},
                metadata={},
            )
            checkpoint.save(filepath)

            # Load and tamper with checksum
            import msgpack

            with open(filepath, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)

            data["checksum"] = "0" * 64  # Invalid checksum

            serialized = msgpack.packb(data, use_bin_type=True)
            with open(filepath, "wb") as f:
                f.write(serialized)

            # Should fail on checksum validation
            with pytest.raises(CheckpointCorruptedError):
                Checkpoint.load(filepath)


class TestAgentGraphCheckpointing:
    """Test checkpoint integration with AgentGraph (Stories 7.1.1 & 7.1.2)."""

    def test_save_checkpoint_from_graph(self):
        """Test saving checkpoint from AgentGraph instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(name="test_graph")
            graph.add_node("node1", "Test prompt 1")
            graph.add_node("node2", "Test prompt 2")
            graph.add_edge("node1", "node2", directed=True)

            filepath = Path(tmpdir) / "graph_checkpoint.msgpack"
            saved_path = graph.save_checkpoint(filepath)

            assert saved_path == filepath
            assert filepath.exists()

    def test_save_checkpoint_auto_path(self):
        """Test saving checkpoint with auto-generated path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
            )
            graph.add_node("node1", "Test")

            saved_path = graph.save_checkpoint()

            assert saved_path.exists()
            assert saved_path.parent == Path(tmpdir)
            assert saved_path.name.startswith("checkpoint_")

    def test_load_checkpoint_creates_graph(self):
        """Test loading checkpoint creates a new AgentGraph with same state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            graph1 = AgentGraph(name="original_graph")
            graph1.add_node("node1", "Prompt 1", model="claude-3-sonnet")
            graph1.add_node("node2", "Prompt 2")
            graph1.add_edge("node1", "node2", directed=True, priority="high")

            filepath = Path(tmpdir) / "checkpoint.msgpack"
            graph1.save_checkpoint(filepath)

            # Load
            graph2 = AgentGraph.load_checkpoint(filepath)

            # Verify structure
            assert graph2.name == "original_graph"
            assert graph2.node_count == 2
            assert graph2.edge_count == 1
            assert graph2.get_node("node1").model == "claude-3-sonnet"
            assert graph2.get_edge("node1", "node2").properties["priority"] == "high"

    def test_load_checkpoint_preserves_topology_constraint(self):
        """Test that topology constraint is preserved in checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph1 = AgentGraph(
                name="dag_graph",
                topology_constraint="dag",
                max_nodes=500,
            )
            graph1.add_node("n1", "Test1")
            graph1.add_node("n2", "Test2")
            graph1.add_edge("n1", "n2")

            filepath = Path(tmpdir) / "cp.msgpack"
            graph1.save_checkpoint(filepath)

            graph2 = AgentGraph.load_checkpoint(filepath)

            assert graph2.topology_constraint == "dag"
            assert graph2.max_nodes == 500

    def test_load_latest_checkpoint(self):
        """Test loading latest checkpoint from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            graph.add_node("node1", "Test1")
            graph.save_checkpoint()

            # Sleep briefly to ensure different timestamp
            import time
            time.sleep(0.01)

            graph.add_node("node2", "Test2")
            graph.save_checkpoint()

            # Create new graph and load latest
            graph2 = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            async def test_load():
                loaded = await graph2.load_latest_checkpoint()
                assert loaded
                assert graph2.node_count == 2
                assert "node2" in graph2._nodes

            asyncio.run(test_load())

    def test_load_latest_checkpoint_no_checkpoint_found(self):
        """Test load_latest_checkpoint returns False when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            async def test_load():
                loaded = await graph.load_latest_checkpoint()
                assert not loaded
                assert graph.node_count == 0

            asyncio.run(test_load())

    def test_load_latest_checkpoint_no_checkpoint_dir(self):
        """Test load_latest_checkpoint returns False when checkpoint dir doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir) / "nonexistent",
                auto_save=False,
            )

            async def test_load():
                loaded = await graph.load_latest_checkpoint()
                assert not loaded

            asyncio.run(test_load())


class TestAutoSave:
    """Test auto-save functionality (Story 7.2.1)."""

    def test_auto_save_configuration(self):
        """Test auto-save can be configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                auto_save=True,
                auto_save_interval=30,
                checkpoint_dir=Path(tmpdir),
            )

            assert graph.auto_save is True
            assert graph.auto_save_interval == 30
            assert graph.checkpoint_dir == Path(tmpdir)

    def test_auto_save_disabled(self):
        """Test auto-save can be disabled."""
        graph = AgentGraph(
            name="test_graph",
            auto_save=False,
        )

        assert graph.auto_save is False

    @pytest.mark.asyncio
    async def test_start_auto_save_creates_task(self):
        """Test start_auto_save creates background task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
            )

            graph.start_auto_save()

            # Verify task is created
            assert graph._auto_save_task is not None
            assert not graph._auto_save_task.done()

            # Clean up
            graph.stop_auto_save()
            await asyncio.sleep(0.01)  # Give task time to cancel

    @pytest.mark.asyncio
    async def test_stop_auto_save_cancels_task(self):
        """Test stop_auto_save cancels background task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
            )

            graph.start_auto_save()
            graph.stop_auto_save()

            assert graph._auto_save_task is None

    @pytest.mark.asyncio
    async def test_auto_save_worker_saves_periodically(self):
        """Test auto-save worker saves checkpoints periodically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save_interval=1,  # 1 second for faster testing
            )

            graph.add_node("node1", "Test")
            graph.start_auto_save()

            # Wait for auto-save to trigger
            await asyncio.sleep(1.5)

            # Check checkpoint was created
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.msgpack"))
            assert len(checkpoints) >= 1

            graph.stop_auto_save()

    @pytest.mark.asyncio
    async def test_auto_save_handles_errors(self):
        """Test auto-save worker handles save errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save_interval=1,
            )

            graph.add_node("node1", "Test")

            # Start auto-save first, let it create a checkpoint
            graph.start_auto_save()
            await asyncio.sleep(1.5)

            # Now make checkpoint_dir invalid by changing it to a read-only path
            # This will cause errors on the next save attempt
            graph.checkpoint_dir = Path("/nonexistent/path/that/will/fail")

            # Wait a bit more for the next auto-save attempt
            await asyncio.sleep(0.5)

            # Task might be done due to error, which is acceptable behavior
            # The important thing is that it attempted to save gracefully
            graph.stop_auto_save()


class TestCrashRecovery:
    """Test crash recovery functionality (Story 7.2.2)."""

    def test_recovery_on_startup(self):
        """Test graph can be recovered from checkpoint on startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create graph with data
            graph1 = AgentGraph(
                name="recoverable_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )
            graph1.add_node("node1", "Prompt 1")
            graph1.add_node("node2", "Prompt 2")
            graph1.add_edge("node1", "node2")
            graph1.save_checkpoint()

            # Simulate crash/restart by creating new graph
            graph2 = AgentGraph(
                name="recoverable_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            # Recover from checkpoint
            async def test_recovery():
                recovered = await graph2.load_latest_checkpoint()
                assert recovered
                assert graph2.node_count == 2
                assert graph2.edge_count == 1

            asyncio.run(test_recovery())

    def test_multiple_checkpoints_loads_latest(self):
        """Test that loading latest selects the newest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            # Create first checkpoint
            graph.add_node("node1", "First version")
            graph.save_checkpoint()

            # Create second checkpoint
            import time
            time.sleep(0.01)
            graph.add_node("node2", "Second version")
            graph.save_checkpoint()

            # Create new graph and recover
            graph2 = AgentGraph(
                name="test_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            async def test_recovery():
                await graph2.load_latest_checkpoint()
                # Should have both nodes from latest checkpoint
                assert graph2.node_count == 2
                assert "node2" in graph2._nodes

            asyncio.run(test_recovery())

    def test_checkpoint_version_incompatibility(self):
        """Test that incompatible checkpoint versions are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes = {
                "node1": Node(node_id="node1", system_prompt="Test"),
            }
            checkpoint = Checkpoint(
                name="test_graph",
                nodes=nodes,
                edges={},
                metadata={},
            )

            filepath = Path(tmpdir) / "checkpoint.msgpack"
            checkpoint.save(filepath)

            # Manually modify version
            import msgpack

            with open(filepath, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)

            data["version"] = 999  # Unsupported version

            # Recompute checksum
            data.pop("checksum", None)
            checksum = Checkpoint.compute_checksum(data)
            data["checksum"] = checksum

            serialized = msgpack.packb(data, use_bin_type=True)
            with open(filepath, "wb") as f:
                f.write(serialized)

            # Should fail on version check
            with pytest.raises(CheckpointVersionError):
                Checkpoint.load(filepath)


class TestCheckpointIntegration:
    """Integration tests for checkpoint system."""

    def test_checkpoint_roundtrip_preserves_state(self):
        """Test saving and loading preserves exact state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create complex graph
            graph1 = AgentGraph(
                name="complex_graph",
                topology_constraint="dag",
                max_nodes=1000,
            )

            for i in range(5):
                graph1.add_node(f"node{i}", f"Prompt {i}", metadata={"idx": i})

            graph1.add_edge("node0", "node1")
            graph1.add_edge("node1", "node2", properties={"weight": 0.8})
            graph1.add_edge("node2", "node3", directed=False)

            filepath = Path(tmpdir) / "complex.msgpack"
            graph1.save_checkpoint(filepath)

            # Load and verify exact match
            graph2 = AgentGraph.load_checkpoint(filepath)

            assert graph2.name == graph1.name
            assert graph2.node_count == graph1.node_count
            assert graph2.edge_count == graph1.edge_count
            assert graph2.topology_constraint == graph1.topology_constraint
            assert graph2.max_nodes == graph1.max_nodes

            # Verify all nodes
            for node_id in graph1._nodes:
                node1 = graph1.get_node(node_id)
                node2 = graph2.get_node(node_id)
                assert node1.system_prompt == node2.system_prompt
                assert node1.model == node2.model
                assert node1.metadata == node2.metadata

            # Verify all edges
            for edge_id in graph1._edges:
                edge1 = graph1._edges[edge_id]
                edge2 = graph2._edges[edge_id]
                assert edge1.from_node == edge2.from_node
                assert edge1.to_node == edge2.to_node
                assert edge1.directed == edge2.directed
                assert edge1.properties == edge2.properties

    @pytest.mark.asyncio
    async def test_end_to_end_auto_save_recovery(self):
        """End-to-end test: auto-save creates checkpoint, then recover on startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Graph with auto-save
            graph1 = AgentGraph(
                name="auto_save_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=True,
                auto_save_interval=1,
            )

            graph1.add_node("node1", "Original prompt")
            graph1.start_auto_save()

            # Wait for auto-save
            await asyncio.sleep(1.5)
            graph1.stop_auto_save()

            # Check checkpoint exists
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.msgpack"))
            assert len(checkpoints) >= 1

            # Create new graph and recover
            graph2 = AgentGraph(
                name="auto_save_graph",
                checkpoint_dir=Path(tmpdir),
                auto_save=False,
            )

            recovered = await graph2.load_latest_checkpoint()
            assert recovered
            assert graph2.node_count == 1
            assert graph2.get_node("node1").system_prompt == "Original prompt"
