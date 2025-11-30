"""
Unit tests for claude-agent-graph data models.
"""

from datetime import datetime, timezone

import pytest
from claude_agent_graph.models import (
    Edge,
    Message,
    MessageRole,
    Node,
    NodeStatus,
    SharedState,
)
from pydantic import ValidationError


class TestMessage:
    """Tests for the Message data model."""

    def test_create_message_with_required_fields(self) -> None:
        """Test creating a message with only required fields."""
        msg = Message(
            from_node="agent_1",
            to_node="agent_2",
            content="Hello, agent 2!",
        )

        assert msg.from_node == "agent_1"
        assert msg.to_node == "agent_2"
        assert msg.content == "Hello, agent 2!"
        assert msg.role == MessageRole.USER
        assert msg.message_id.startswith("msg_")
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_create_message_with_all_fields(self) -> None:
        """Test creating a message with all fields specified."""
        timestamp = datetime.now(timezone.utc)
        metadata = {"priority": "high", "task_id": "123"}

        msg = Message(
            message_id="msg_custom_123",
            timestamp=timestamp,
            from_node="agent_1",
            to_node="agent_2",
            role=MessageRole.ASSISTANT,
            content="Response message",
            metadata=metadata,
        )

        assert msg.message_id == "msg_custom_123"
        assert msg.timestamp == timestamp
        assert msg.role == MessageRole.ASSISTANT
        assert msg.metadata == metadata

    def test_message_id_autogeneration(self) -> None:
        """Test that message_id is automatically generated."""
        msg1 = Message(from_node="a", to_node="b", content="test1")
        msg2 = Message(from_node="a", to_node="b", content="test2")

        assert msg1.message_id != msg2.message_id
        assert msg1.message_id.startswith("msg_")
        assert len(msg1.message_id) == 20  # "msg_" + 16 hex chars

    def test_timestamp_defaults_to_utc(self) -> None:
        """Test that timestamp is automatically set to current UTC time."""
        before = datetime.now(timezone.utc)
        msg = Message(from_node="a", to_node="b", content="test")
        after = datetime.now(timezone.utc)

        assert before <= msg.timestamp <= after
        assert msg.timestamp.tzinfo == timezone.utc

    def test_timestamp_converts_to_utc(self) -> None:
        """Test that non-UTC timestamps are converted to UTC."""
        # Create a naive datetime (no timezone)
        naive_time = datetime(2025, 10, 25, 12, 0, 0)
        msg = Message(
            from_node="a",
            to_node="b",
            content="test",
            timestamp=naive_time,
        )

        assert msg.timestamp.tzinfo == timezone.utc

    def test_validate_empty_from_node(self) -> None:
        """Test that empty from_node is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Message(from_node="", to_node="b", content="test")

        assert "from_node" in str(exc_info.value)

    def test_validate_empty_to_node(self) -> None:
        """Test that empty to_node is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Message(from_node="a", to_node="", content="test")

        assert "to_node" in str(exc_info.value)

    def test_validate_empty_content(self) -> None:
        """Test that empty content is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Message(from_node="a", to_node="b", content="")

        assert "content" in str(exc_info.value)

    def test_message_role_enum(self) -> None:
        """Test MessageRole enum values."""
        msg_user = Message(from_node="a", to_node="b", content="test", role=MessageRole.USER)
        msg_assistant = Message(
            from_node="a", to_node="b", content="test", role=MessageRole.ASSISTANT
        )
        msg_system = Message(from_node="a", to_node="b", content="test", role=MessageRole.SYSTEM)

        assert msg_user.role == MessageRole.USER
        assert msg_assistant.role == MessageRole.ASSISTANT
        assert msg_system.role == MessageRole.SYSTEM

    def test_to_dict(self) -> None:
        """Test converting message to dictionary."""
        msg = Message(
            message_id="msg_test123",
            from_node="agent_1",
            to_node="agent_2",
            content="Test message",
            metadata={"key": "value"},
        )

        data = msg.to_dict()

        assert isinstance(data, dict)
        assert data["message_id"] == "msg_test123"
        assert data["from_node"] == "agent_1"
        assert data["to_node"] == "agent_2"
        assert data["content"] == "Test message"
        assert data["role"] == "user"
        assert data["metadata"] == {"key": "value"}
        assert isinstance(data["timestamp"], str)  # Should be ISO format string

    def test_from_dict(self) -> None:
        """Test creating message from dictionary."""
        data = {
            "message_id": "msg_test123",
            "timestamp": "2025-10-25T12:00:00+00:00",
            "from_node": "agent_1",
            "to_node": "agent_2",
            "role": "assistant",
            "content": "Test message",
            "metadata": {"key": "value"},
        }

        msg = Message.from_dict(data)

        assert msg.message_id == "msg_test123"
        assert msg.from_node == "agent_1"
        assert msg.to_node == "agent_2"
        assert msg.content == "Test message"
        assert msg.role == MessageRole.ASSISTANT
        assert msg.metadata == {"key": "value"}
        assert isinstance(msg.timestamp, datetime)

    def test_round_trip_serialization(self) -> None:
        """Test that to_dict/from_dict round-trip works correctly."""
        original = Message(
            from_node="agent_1",
            to_node="agent_2",
            content="Test message",
            role=MessageRole.ASSISTANT,
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.message_id == original.message_id
        assert restored.from_node == original.from_node
        assert restored.to_node == original.to_node
        assert restored.content == original.content
        assert restored.role == original.role
        assert restored.metadata == original.metadata
        # Timestamps should be equal (accounting for potential microsecond differences)
        assert abs((restored.timestamp - original.timestamp).total_seconds()) < 0.001


class TestNode:
    """Tests for the Node data model."""

    def test_create_node_with_required_fields(self) -> None:
        """Test creating a node with only required fields."""
        node = Node(
            node_id="agent_1",
            system_prompt="You are a helpful assistant.",
        )

        assert node.node_id == "agent_1"
        assert node.system_prompt == "You are a helpful assistant."
        assert node.model == "claude-sonnet-4-20250514"  # Default
        assert node.status == NodeStatus.INITIALIZING  # Default
        assert isinstance(node.created_at, datetime)
        assert node.metadata == {}

    def test_create_node_with_all_fields(self) -> None:
        """Test creating a node with all fields specified."""
        created_at = datetime.now(timezone.utc)
        metadata = {"team": "research", "priority": 1}

        node = Node(
            node_id="agent_1",
            system_prompt="You are a researcher.",
            model="claude-opus-4-20250514",
            status=NodeStatus.ACTIVE,
            created_at=created_at,
            metadata=metadata,
        )

        assert node.model == "claude-opus-4-20250514"
        assert node.status == NodeStatus.ACTIVE
        assert node.created_at == created_at
        assert node.metadata == metadata

    def test_validate_empty_node_id(self) -> None:
        """Test that empty node_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Node(node_id="", system_prompt="test")

        assert "node_id" in str(exc_info.value)

    def test_validate_empty_system_prompt(self) -> None:
        """Test that empty system_prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Node(node_id="agent_1", system_prompt="")

        assert "system_prompt" in str(exc_info.value)

    def test_validate_system_prompt_max_length(self) -> None:
        """Test that system_prompt has a maximum length."""
        long_prompt = "x" * 100001  # Exceeds 100K limit

        with pytest.raises(ValidationError) as exc_info:
            Node(node_id="agent_1", system_prompt=long_prompt)

        assert "system_prompt" in str(exc_info.value)

    def test_node_status_enum(self) -> None:
        """Test NodeStatus enum values."""
        node_init = Node(node_id="a", system_prompt="test", status=NodeStatus.INITIALIZING)
        node_active = Node(node_id="b", system_prompt="test", status=NodeStatus.ACTIVE)
        node_stopped = Node(node_id="c", system_prompt="test", status=NodeStatus.STOPPED)
        node_error = Node(node_id="d", system_prompt="test", status=NodeStatus.ERROR)

        assert node_init.status == NodeStatus.INITIALIZING
        assert node_active.status == NodeStatus.ACTIVE
        assert node_stopped.status == NodeStatus.STOPPED
        assert node_error.status == NodeStatus.ERROR

    def test_created_at_defaults_to_utc(self) -> None:
        """Test that created_at is automatically set to current UTC time."""
        before = datetime.now(timezone.utc)
        node = Node(node_id="agent_1", system_prompt="test")
        after = datetime.now(timezone.utc)

        assert before <= node.created_at <= after
        assert node.created_at.tzinfo == timezone.utc


class TestEdge:
    """Tests for the Edge data model."""

    def test_create_edge_with_required_fields(self) -> None:
        """Test creating an edge with required fields."""
        edge = Edge(
            edge_id="a_to_b",
            from_node="agent_a",
            to_node="agent_b",
        )

        assert edge.edge_id == "a_to_b"
        assert edge.from_node == "agent_a"
        assert edge.to_node == "agent_b"
        assert edge.directed is True  # Default
        assert isinstance(edge.created_at, datetime)
        assert edge.properties == {}

    def test_create_edge_with_all_fields(self) -> None:
        """Test creating an edge with all fields specified."""
        created_at = datetime.now(timezone.utc)
        properties = {"weight": 1.0, "type": "supervisor"}

        edge = Edge(
            edge_id="custom_edge",
            from_node="agent_a",
            to_node="agent_b",
            directed=False,
            created_at=created_at,
            properties=properties,
        )

        assert edge.directed is False
        assert edge.created_at == created_at
        assert edge.properties == properties

    def test_validate_empty_edge_id(self) -> None:
        """Test that empty edge_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(edge_id="", from_node="a", to_node="b")

        assert "edge_id" in str(exc_info.value)

    def test_validate_empty_from_node(self) -> None:
        """Test that empty from_node is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(edge_id="e1", from_node="", to_node="b")

        assert "from_node" in str(exc_info.value)

    def test_validate_empty_to_node(self) -> None:
        """Test that empty to_node is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(edge_id="e1", from_node="a", to_node="")

        assert "to_node" in str(exc_info.value)

    def test_generate_edge_id_directed(self) -> None:
        """Test generating edge ID for directed edge."""
        edge_id = Edge.generate_edge_id("agent_a", "agent_b", directed=True)
        assert edge_id == "agent_a_to_agent_b"

    def test_generate_edge_id_undirected(self) -> None:
        """Test generating edge ID for undirected edge."""
        edge_id1 = Edge.generate_edge_id("agent_a", "agent_b", directed=False)
        edge_id2 = Edge.generate_edge_id("agent_b", "agent_a", directed=False)

        # Should be the same regardless of order
        assert edge_id1 == edge_id2
        assert "undirected" in edge_id1

    def test_created_at_defaults_to_utc(self) -> None:
        """Test that created_at is automatically set to current UTC time."""
        before = datetime.now(timezone.utc)
        edge = Edge(edge_id="e1", from_node="a", to_node="b")
        after = datetime.now(timezone.utc)

        assert before <= edge.created_at <= after
        assert edge.created_at.tzinfo == timezone.utc


class TestSharedState:
    """Tests for the SharedState data model."""

    def test_create_shared_state_with_required_fields(self) -> None:
        """Test creating shared state with required fields."""
        state = SharedState(conversation_file="/path/to/conversation.jsonl")

        assert state.conversation_file == "/path/to/conversation.jsonl"
        assert state.metadata == {}

    def test_create_shared_state_with_metadata(self) -> None:
        """Test creating shared state with metadata."""
        metadata = {"last_accessed": "2025-10-25", "message_count": 42}

        state = SharedState(
            conversation_file="/path/to/conversation.jsonl",
            metadata=metadata,
        )

        assert state.metadata == metadata

    def test_validate_conversation_file_extension(self) -> None:
        """Test that conversation_file must have .jsonl extension."""
        with pytest.raises(ValidationError) as exc_info:
            SharedState(conversation_file="/path/to/conversation.json")

        assert "conversation_file" in str(exc_info.value)
        assert "jsonl" in str(exc_info.value).lower()

    def test_validate_empty_conversation_file(self) -> None:
        """Test that empty conversation_file is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SharedState(conversation_file="")

        assert "conversation_file" in str(exc_info.value)

    def test_valid_conversation_file_paths(self) -> None:
        """Test various valid conversation file paths."""
        valid_paths = [
            "conversation.jsonl",
            "/absolute/path/to/conversation.jsonl",
            "relative/path/conversation.jsonl",
            "../parent/conversation.jsonl",
            "~/home/conversation.jsonl",
        ]

        for path in valid_paths:
            state = SharedState(conversation_file=path)
            assert state.conversation_file == path
