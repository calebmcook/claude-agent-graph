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


@pytest.mark.skip(reason="Manual message processing not yet implemented")
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
            content="Test",
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
        message1 = Message(
            message_id="msg_1", timestamp=ts, from_node="x", to_node="node_1", content="M1"
        )
        message2 = Message(
            message_id="msg_2", timestamp=ts, from_node="x", to_node="node_2", content="M2"
        )

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
        message = Message(
            message_id="msg_1",
            timestamp=datetime.now(timezone.utc),
            from_node="x",
            to_node="node_1",
            content="Test",
        )
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
        msg1 = Message(
            message_id="msg_1", timestamp=ts, from_node="x", to_node="node_1", content="M1"
        )
        msg2 = Message(
            message_id="msg_2", timestamp=ts, from_node="x", to_node="node_2", content="M2"
        )

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


@pytest.mark.skip(reason="Execution mode integration not yet fully implemented")
class TestIntegrationEndToEnd:
    """Integration tests for end-to-end message processing with execution modes."""

    @pytest.mark.asyncio
    async def test_manual_controller_message_enqueuing(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that messages are enqueued when manual controller is active."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You are a supervisor")
        await graph.add_node("worker", "You are a worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        # Start manual controller
        controller = ManualController(graph)
        await controller.start()
        graph._execution_mode = controller

        # Send message - should enqueue it
        msg = await graph.send_message("supervisor", "worker", "Do some work")

        # Verify message is in queue
        assert "worker" in graph._message_queues
        queue = graph._message_queues["worker"]
        assert not queue.empty()

        # Verify the queued message is the same one we sent
        queued_msg = queue.get_nowait()
        assert queued_msg.message_id == msg.message_id
        assert queued_msg.content == "Do some work"

        await controller.stop()

    @pytest.mark.asyncio
    async def test_manual_controller_step_dequeues_message(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that manual step dequeues messages from the queue."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You are a supervisor")
        await graph.add_node("worker", "You are a worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        # Start manual controller
        controller = ManualController(graph)
        await controller.start()
        graph._execution_mode = controller

        # Send message
        await graph.send_message("supervisor", "worker", "Process this")

        # Verify message is queued
        queue = graph._message_queues["worker"]
        assert not queue.empty()

        # Step the worker - should dequeue and process
        await controller.step("worker")

        # Queue should be empty after step
        assert queue.empty()

        await controller.stop()

    @pytest.mark.asyncio
    async def test_reactive_executor_auto_processes_messages(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that reactive executor automatically processes queued messages."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You are a supervisor")
        await graph.add_node("worker", "You are a worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        # Start reactive executor
        executor = ReactiveExecutor(graph)
        await executor.start()
        graph._execution_mode = executor

        # Send message
        await graph.send_message("supervisor", "worker", "Work on this")

        # Give executor time to process
        await asyncio.sleep(0.2)

        # Queue should be empty (message was processed)
        if "worker" in graph._message_queues:
            queue = graph._message_queues["worker"]
            assert queue.empty()

        await executor.stop()

    @pytest.mark.asyncio
    async def test_message_enqueuing_without_execution_mode(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that messages are NOT enqueued when no execution mode is active."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You are a supervisor")
        await graph.add_node("worker", "You are a worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        # No execution mode started
        assert graph._execution_mode is None

        # Send message
        await graph.send_message("supervisor", "worker", "Do work")

        # Queue should not be created when no execution mode is active
        assert "worker" not in graph._message_queues

    @pytest.mark.asyncio
    async def test_multiple_messages_preserve_order(self, tmp_path: Path, mock_claude_sdk) -> None:
        """Test that multiple messages preserve FIFO order in the queue."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You are a supervisor")
        await graph.add_node("worker", "You are a worker")
        await graph.add_edge("supervisor", "worker", directed=True)

        # Start manual controller
        controller = ManualController(graph)
        await controller.start()
        graph._execution_mode = controller

        # Send multiple messages
        msg1 = await graph.send_message("supervisor", "worker", "First task")
        msg2 = await graph.send_message("supervisor", "worker", "Second task")
        msg3 = await graph.send_message("supervisor", "worker", "Third task")

        # Verify messages are in queue in order
        queue = graph._message_queues["worker"]

        queued_msg1 = queue.get_nowait()
        assert queued_msg1.message_id == msg1.message_id
        assert queued_msg1.content == "First task"

        queued_msg2 = queue.get_nowait()
        assert queued_msg2.message_id == msg2.message_id
        assert queued_msg2.content == "Second task"

        queued_msg3 = queue.get_nowait()
        assert queued_msg3.message_id == msg3.message_id
        assert queued_msg3.content == "Third task"

        await controller.stop()

    @pytest.mark.asyncio
    async def test_broadcast_enqueues_to_multiple_recipients(
        self, tmp_path: Path, mock_claude_sdk
    ) -> None:
        """Test that broadcast sends messages to all recipients and enqueues them."""
        graph = AgentGraph(name="test", storage_backend=FilesystemBackend(base_dir=str(tmp_path)))
        await graph.add_node("supervisor", "You are a supervisor")
        await graph.add_node("worker_1", "You are worker 1")
        await graph.add_node("worker_2", "You are worker 2")
        await graph.add_node("worker_3", "You are worker 3")

        await graph.add_edge("supervisor", "worker_1", directed=True)
        await graph.add_edge("supervisor", "worker_2", directed=True)
        await graph.add_edge("supervisor", "worker_3", directed=True)

        # Start manual controller
        controller = ManualController(graph)
        await controller.start()
        graph._execution_mode = controller

        # Broadcast message
        messages = await graph.broadcast("supervisor", "Attention all workers!")

        # Verify all workers have messages in queue
        assert len(messages) == 3
        assert "worker_1" in graph._message_queues
        assert "worker_2" in graph._message_queues
        assert "worker_3" in graph._message_queues

        # Each queue should have exactly one message
        assert not graph._message_queues["worker_1"].empty()
        assert not graph._message_queues["worker_2"].empty()
        assert not graph._message_queues["worker_3"].empty()

        await controller.stop()
