"""
Tests for execution modes (Epic 6, Feature 6.2).

Tests for ManualController, ReactiveExecutor, and ProactiveExecutor.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.execution import ManualController, ProactiveExecutor, ReactiveExecutor
from claude_agent_graph.graph import AgentGraph
from claude_agent_graph.models import Message


@pytest.fixture
def mock_claude_sdk(monkeypatch):
    """Mock the claude-agent-sdk to avoid API calls."""
    class MockClaudeSDKClient:
        def __init__(self, options):
            self.options = options

    # Mock the import
    import sys
    mock_module = type(sys)("mock_claude_agent_sdk")
    mock_module.ClaudeSDKClient = MockClaudeSDKClient
    mock_module.ClaudeAgentOptions = dict
    sys.modules["claude_agent_sdk"] = mock_module


class TestManualController:
    """Tests for ManualController execution mode."""

    @pytest.mark.asyncio
    async def test_manual_controller_initialization(self, tmp_path: Path) -> None:
        """Test manual controller initialization."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        controller = ManualController(graph)

        assert controller._graph is graph
        assert not controller._running
        assert controller._task is None

    @pytest.mark.asyncio
    async def test_manual_controller_start_stop(self, tmp_path: Path) -> None:
        """Test starting and stopping manual controller."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        controller = ManualController(graph)

        await controller.start()
        assert controller._running
        assert controller._task is not None

        await controller.stop()
        assert not controller._running

    @pytest.mark.asyncio
    async def test_manual_controller_cannot_start_twice(self, tmp_path: Path) -> None:
        """Test that starting twice raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        controller = ManualController(graph)

        await controller.start()

        with pytest.raises(RuntimeError, match="already running"):
            await controller.start()

        await controller.stop()

    @pytest.mark.asyncio
    async def test_manual_controller_step_no_queue(self, tmp_path: Path) -> None:
        """Test stepping when no messages exist."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        controller = ManualController(graph)

        # Should not raise - just does nothing
        await controller.step("node_1")

    @pytest.mark.asyncio
    async def test_manual_controller_step_nonexistent_node(self, tmp_path: Path) -> None:
        """Test stepping nonexistent node raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))

        controller = ManualController(graph)

        with pytest.raises(ValueError, match="not found"):
            await controller.step("nonexistent")

    @pytest.mark.asyncio
    async def test_manual_controller_step_all_no_messages(self, tmp_path: Path) -> None:
        """Test step_all with no pending messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")
        await graph.add_node("node_2", "Node 2")

        controller = ManualController(graph)

        count = await controller.step_all()

        assert count == 0

    @pytest.mark.asyncio
    async def test_manual_controller_step_processes_message(self, tmp_path: Path) -> None:
        """Test that step processes a message from queue."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        controller = ManualController(graph)

        # Manually add a message to the queue
        message = Message(
            message_id="msg_123",
            timestamp=datetime.now(timezone.utc),
            from_node="other",
            to_node="node_1",
            content="Test"
        )

        # Create queue and add message
        queue = asyncio.Queue()
        await queue.put(message)
        graph._message_queues["node_1"] = queue

        # Step should dequeue the message
        await controller.step("node_1")

        # Queue should now be empty
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_manual_controller_step_all_multiple_nodes(self, tmp_path: Path) -> None:
        """Test step_all executes all nodes with pending messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")
        await graph.add_node("node_2", "Node 2")
        await graph.add_node("node_3", "Node 3")

        controller = ManualController(graph)

        # Add messages to some nodes
        ts = datetime.now(timezone.utc)
        message1 = Message(message_id="msg_1", timestamp=ts, from_node="x", to_node="node_1", content="M1")
        message2 = Message(message_id="msg_2", timestamp=ts, from_node="x", to_node="node_2", content="M2")

        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()
        await queue1.put(message1)
        await queue2.put(message2)

        graph._message_queues["node_1"] = queue1
        graph._message_queues["node_2"] = queue2

        # step_all should process both
        count = await controller.step_all()

        assert count == 2
        assert queue1.empty()
        assert queue2.empty()


class TestReactiveExecutor:
    """Tests for ReactiveExecutor execution mode."""

    @pytest.mark.asyncio
    async def test_reactive_executor_initialization(self, tmp_path: Path) -> None:
        """Test reactive executor initialization."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ReactiveExecutor(graph)

        assert executor._graph is graph
        assert not executor._running

    @pytest.mark.asyncio
    async def test_reactive_executor_start_stop(self, tmp_path: Path) -> None:
        """Test starting and stopping reactive executor."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ReactiveExecutor(graph)

        await executor.start()
        assert executor._running
        assert executor._task is not None

        await executor.stop()
        assert not executor._running

    @pytest.mark.asyncio
    async def test_reactive_executor_cannot_start_twice(self, tmp_path: Path) -> None:
        """Test that starting twice raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ReactiveExecutor(graph)

        await executor.start()

        with pytest.raises(RuntimeError, match="already running"):
            await executor.start()

        await executor.stop()

    @pytest.mark.asyncio
    async def test_reactive_executor_monitors_queues(self, tmp_path: Path) -> None:
        """Test that reactive executor monitors message queues."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ReactiveExecutor(graph)

        # Start executor
        await executor.start()

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Add a message to a queue
        message = Message(message_id="msg_1", timestamp=datetime.now(timezone.utc), from_node="x", to_node="node_1", content="Test")
        queue = asyncio.Queue()
        await queue.put(message)
        graph._message_queues["node_1"] = queue

        # Give executor time to process
        await asyncio.sleep(0.2)

        # Stop executor
        await executor.stop()

        # Queue should have been processed
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_reactive_executor_handles_empty_graph(self, tmp_path: Path) -> None:
        """Test reactive executor with no nodes/queues."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))

        executor = ReactiveExecutor(graph)

        await executor.start()
        await asyncio.sleep(0.2)
        await executor.stop()

        # Should complete without error

    @pytest.mark.asyncio
    async def test_reactive_executor_multiple_queues(self, tmp_path: Path) -> None:
        """Test reactive executor with multiple message queues."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")
        await graph.add_node("node_2", "Node 2")

        executor = ReactiveExecutor(graph)

        await executor.start()
        await asyncio.sleep(0.1)

        # Add messages to multiple queues
        ts = datetime.now(timezone.utc)
        msg1 = Message(message_id="msg_1", timestamp=ts, from_node="x", to_node="node_1", content="M1")
        msg2 = Message(message_id="msg_2", timestamp=ts, from_node="x", to_node="node_2", content="M2")

        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()
        await queue1.put(msg1)
        await queue2.put(msg2)

        graph._message_queues["node_1"] = queue1
        graph._message_queues["node_2"] = queue2

        # Give executor time to process both
        await asyncio.sleep(0.3)

        await executor.stop()

        # Both queues should be empty
        assert queue1.empty()
        assert queue2.empty()


class TestProactiveExecutor:
    """Tests for ProactiveExecutor execution mode."""

    @pytest.mark.asyncio
    async def test_proactive_executor_initialization(self, tmp_path: Path) -> None:
        """Test proactive executor initialization."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ProactiveExecutor(graph, interval=1.0, start_delay=0.1)

        assert executor._graph is graph
        assert executor._interval == 1.0
        assert executor._start_delay == 0.1

    @pytest.mark.asyncio
    async def test_proactive_executor_default_interval(self, tmp_path: Path) -> None:
        """Test proactive executor with default interval."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ProactiveExecutor(graph)

        assert executor._interval == 60.0
        assert executor._start_delay == 0.0

    @pytest.mark.asyncio
    async def test_proactive_executor_start_stop(self, tmp_path: Path) -> None:
        """Test starting and stopping proactive executor."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ProactiveExecutor(graph, interval=0.1, start_delay=0.0)

        await executor.start()
        assert executor._running

        await executor.stop()
        assert not executor._running

    @pytest.mark.asyncio
    async def test_proactive_executor_respects_start_delay(self, tmp_path: Path) -> None:
        """Test that proactive executor respects start delay."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ProactiveExecutor(graph, interval=10.0, start_delay=0.2)

        import time
        start = time.time()

        await executor.start()
        await asyncio.sleep(0.1)  # Stop before activation should occur

        await executor.stop()

        elapsed = time.time() - start

        # Should have waited at least the start_delay
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_proactive_executor_periodic_activation(self, tmp_path: Path) -> None:
        """Test that proactive executor activates periodically."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        # Use very short interval for testing
        executor = ProactiveExecutor(graph, interval=0.05, start_delay=0.0)

        await executor.start()

        # Let it run for a bit (should have multiple activation cycles)
        await asyncio.sleep(0.15)

        await executor.stop()

        # Should complete without error

    @pytest.mark.asyncio
    async def test_proactive_executor_cannot_start_twice(self, tmp_path: Path) -> None:
        """Test that starting twice raises error."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")

        executor = ProactiveExecutor(graph)

        await executor.start()

        with pytest.raises(RuntimeError, match="already running"):
            await executor.start()

        await executor.stop()

    @pytest.mark.asyncio
    async def test_proactive_executor_empty_graph(self, tmp_path: Path) -> None:
        """Test proactive executor with empty graph."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))

        executor = ProactiveExecutor(graph, interval=0.1)

        await executor.start()
        await asyncio.sleep(0.15)
        await executor.stop()

        # Should complete without error

    @pytest.mark.asyncio
    async def test_proactive_executor_multiple_nodes(self, tmp_path: Path) -> None:
        """Test proactive executor activation with multiple nodes."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("node_1", "Node 1")
        await graph.add_node("node_2", "Node 2")
        await graph.add_node("node_3", "Node 3")

        executor = ProactiveExecutor(graph, interval=0.1)

        await executor.start()
        await asyncio.sleep(0.15)
        await executor.stop()

        # Should have attempted to activate all nodes
