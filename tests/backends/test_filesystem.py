"""
Tests for FilesystemBackend storage implementation.
"""

from pathlib import Path

import pytest
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.models import Message


class TestFilesystemBackend:
    """Tests for FilesystemBackend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> FilesystemBackend:
        """Create a FilesystemBackend for testing."""
        return FilesystemBackend(base_dir=str(tmp_path))

    @pytest.mark.asyncio
    async def test_directory_created_on_first_write(self, tmp_path: Path) -> None:
        """Test that base directory is created on first message write."""
        test_dir = tmp_path / "test_dir"
        backend = FilesystemBackend(base_dir=str(test_dir))

        # Directory shouldn't exist yet
        assert not test_dir.exists()

        # Write a message
        msg = Message(from_node="a", to_node="b", content="Test")
        await backend.append_message("edge_1", msg)

        # Now directory should exist
        assert test_dir.exists()

    @pytest.mark.asyncio
    async def test_append_message(self, backend: FilesystemBackend) -> None:
        """Test appending a message."""
        msg = Message(from_node="a", to_node="b", content="Test message")
        await backend.append_message("edge_1", msg)

        # Verify message was written
        messages = await backend.read_messages("edge_1")
        assert len(messages) == 1
        assert messages[0].content == "Test message"

    @pytest.mark.asyncio
    async def test_append_multiple_messages(self, backend: FilesystemBackend) -> None:
        """Test appending multiple messages."""
        for i in range(5):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await backend.append_message("edge_1", msg)

        messages = await backend.read_messages("edge_1")
        assert len(messages) == 5
        assert messages[0].content == "Message 0"
        assert messages[4].content == "Message 4"

    @pytest.mark.asyncio
    async def test_read_messages_with_limit(self, backend: FilesystemBackend) -> None:
        """Test reading messages with limit."""
        for i in range(10):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await backend.append_message("edge_1", msg)

        messages = await backend.read_messages("edge_1", limit=3)
        assert len(messages) == 3
        assert messages[0].content == "Message 7"
        assert messages[2].content == "Message 9"

    @pytest.mark.asyncio
    async def test_read_messages_since(self, backend: FilesystemBackend) -> None:
        """Test reading messages since a timestamp."""
        messages_sent = []
        for i in range(5):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await backend.append_message("edge_1", msg)
            messages_sent.append(msg)

        # Read messages since the 3rd message
        since_time = messages_sent[2].timestamp
        messages = await backend.read_messages("edge_1", since=since_time)

        # Should get messages 3 and 4 (after message 2)
        assert len(messages) == 2
        assert messages[0].content == "Message 3"
        assert messages[1].content == "Message 4"

    @pytest.mark.asyncio
    async def test_read_messages_empty_conversation(self, backend: FilesystemBackend) -> None:
        """Test reading from non-existent conversation."""
        messages = await backend.read_messages("nonexistent_edge")
        assert messages == []

    @pytest.mark.asyncio
    async def test_conversation_exists(self, backend: FilesystemBackend) -> None:
        """Test checking if conversation exists."""
        assert not backend.conversation_exists("edge_1")

        msg = Message(from_node="a", to_node="b", content="Test")
        await backend.append_message("edge_1", msg)

        assert backend.conversation_exists("edge_1")

    @pytest.mark.asyncio
    async def test_get_conversation_size(self, backend: FilesystemBackend) -> None:
        """Test getting conversation file size."""
        # Non-existent conversation
        size = await backend.get_conversation_size("edge_1")
        assert size == 0

        # Add some messages
        for i in range(3):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await backend.append_message("edge_1", msg)

        size = await backend.get_conversation_size("edge_1")
        assert size > 0

    @pytest.mark.asyncio
    async def test_multiple_edges(self, backend: FilesystemBackend) -> None:
        """Test handling multiple edges independently."""
        # Add messages to edge_1
        for i in range(3):
            msg = Message(from_node="a", to_node="b", content=f"Edge1 Msg {i}")
            await backend.append_message("edge_1", msg)

        # Add messages to edge_2
        for i in range(2):
            msg = Message(from_node="c", to_node="d", content=f"Edge2 Msg {i}")
            await backend.append_message("edge_2", msg)

        # Verify each edge has correct messages
        edge1_msgs = await backend.read_messages("edge_1")
        edge2_msgs = await backend.read_messages("edge_2")

        assert len(edge1_msgs) == 3
        assert len(edge2_msgs) == 2
        assert edge1_msgs[0].content == "Edge1 Msg 0"
        assert edge2_msgs[0].content == "Edge2 Msg 0"

    @pytest.mark.asyncio
    async def test_custom_max_size(self, tmp_path: Path) -> None:
        """Test creating backend with custom max size."""
        backend = FilesystemBackend(base_dir=str(tmp_path), max_size_mb=50)

        msg = Message(from_node="a", to_node="b", content="Test")
        await backend.append_message("edge_1", msg)

        messages = await backend.read_messages("edge_1")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, backend: FilesystemBackend) -> None:
        """Test concurrent writes to same edge."""
        import asyncio

        async def write_message(i: int) -> None:
            msg = Message(from_node="a", to_node="b", content=f"Concurrent {i}")
            await backend.append_message("edge_1", msg)

        # Write 10 messages concurrently
        await asyncio.gather(*[write_message(i) for i in range(10)])

        messages = await backend.read_messages("edge_1")
        assert len(messages) == 10

    @pytest.mark.asyncio
    async def test_message_order_preserved(self, backend: FilesystemBackend) -> None:
        """Test that message order is preserved."""
        for i in range(100):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await backend.append_message("edge_1", msg)

        messages = await backend.read_messages("edge_1")
        assert len(messages) == 100

        # Verify chronological order
        for i in range(len(messages) - 1):
            assert messages[i].timestamp <= messages[i + 1].timestamp
