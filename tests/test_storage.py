"""
Unit tests for claude-agent-graph storage layer.
"""

import asyncio
import json
from pathlib import Path

import aiofiles
import pytest
from claude_agent_graph.models import Message, MessageRole
from claude_agent_graph.storage import ConversationFile


class TestConversationFileInit:
    """Tests for ConversationFile initialization."""

    def test_init_with_valid_path(self) -> None:
        """Test creating ConversationFile with valid .jsonl path."""
        conv = ConversationFile("test.jsonl")
        assert conv.file_path == Path("test.jsonl")
        assert conv.max_size_mb == 100  # Default
        assert conv._lock is not None

    def test_init_with_custom_max_size(self) -> None:
        """Test creating ConversationFile with custom max size."""
        conv = ConversationFile("test.jsonl", max_size_mb=50)
        assert conv.max_size_mb == 50

    def test_init_with_invalid_extension(self) -> None:
        """Test that non-.jsonl extension raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConversationFile("test.json")
        assert "jsonl" in str(exc_info.value).lower()

    def test_init_with_nested_path(self) -> None:
        """Test creating ConversationFile with nested directory path."""
        conv = ConversationFile("conversations/agent1/agent2.jsonl")
        assert conv.file_path == Path("conversations/agent1/agent2.jsonl")


class TestConversationFileAppend:
    """Tests for ConversationFile.append() method."""

    @pytest.mark.asyncio
    async def test_append_single_message(self, tmp_path: Path) -> None:
        """Test appending a single message to conversation file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(from_node="agent_a", to_node="agent_b", content="Hello")
        await conv.append(msg)

        # Verify file was created
        assert file_path.exists()

        # Verify content
        with open(file_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        # Parse and verify message
        parsed = json.loads(lines[0])
        assert parsed["from_node"] == "agent_a"
        assert parsed["to_node"] == "agent_b"
        assert parsed["content"] == "Hello"
        assert parsed["message_id"] == msg.message_id

    @pytest.mark.asyncio
    async def test_append_multiple_messages(self, tmp_path: Path) -> None:
        """Test appending multiple messages in sequence."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        messages = [
            Message(from_node="a", to_node="b", content="Message 1"),
            Message(from_node="b", to_node="a", content="Message 2"),
            Message(from_node="a", to_node="b", content="Message 3"),
        ]

        for msg in messages:
            await conv.append(msg)

        # Verify all messages were written
        with open(file_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        # Verify order and content
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["content"] == f"Message {i + 1}"

    @pytest.mark.asyncio
    async def test_append_creates_directory(self, tmp_path: Path) -> None:
        """Test that append creates parent directories if they don't exist."""
        file_path = tmp_path / "nested" / "dirs" / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(from_node="a", to_node="b", content="Test")
        await conv.append(msg)

        # Verify directory was created
        assert file_path.parent.exists()
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_append_with_unicode_content(self, tmp_path: Path) -> None:
        """Test appending messages with Unicode characters."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(
            from_node="a",
            to_node="b",
            content="Hello 世界 🌍 émojis",
        )
        await conv.append(msg)

        # Verify Unicode is preserved
        with open(file_path, encoding="utf-8") as f:
            parsed = json.loads(f.readline())
        assert parsed["content"] == "Hello 世界 🌍 émojis"

    @pytest.mark.asyncio
    async def test_append_with_metadata(self, tmp_path: Path) -> None:
        """Test appending messages with metadata."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(
            from_node="a",
            to_node="b",
            content="Test",
            metadata={"priority": "high", "task_id": 123},
        )
        await conv.append(msg)

        # Verify metadata is preserved
        with open(file_path) as f:
            parsed = json.loads(f.readline())
        assert parsed["metadata"]["priority"] == "high"
        assert parsed["metadata"]["task_id"] == 123

    @pytest.mark.asyncio
    async def test_append_with_different_roles(self, tmp_path: Path) -> None:
        """Test appending messages with different roles."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        messages = [
            Message(from_node="a", to_node="b", content="User msg", role=MessageRole.USER),
            Message(
                from_node="b", to_node="a", content="Assistant msg", role=MessageRole.ASSISTANT
            ),
            Message(from_node="a", to_node="b", content="System msg", role=MessageRole.SYSTEM),
        ]

        for msg in messages:
            await conv.append(msg)

        # Verify roles are preserved
        with open(file_path) as f:
            lines = f.readlines()

        assert json.loads(lines[0])["role"] == "user"
        assert json.loads(lines[1])["role"] == "assistant"
        assert json.loads(lines[2])["role"] == "system"


class TestConversationFileConcurrency:
    """Tests for thread-safety and concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, tmp_path: Path) -> None:
        """Test that concurrent appends are thread-safe."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Create 20 messages from different "agents"
        async def append_message(i: int) -> None:
            msg = Message(
                from_node=f"agent_{i % 5}",
                to_node=f"agent_{(i + 1) % 5}",
                content=f"Concurrent message {i}",
            )
            await conv.append(msg)

        # Run 20 concurrent append operations
        tasks = [append_message(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all messages were written
        with open(file_path) as f:
            lines = f.readlines()
        assert len(lines) == 20

        # Verify all lines are valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "from_node" in parsed
            assert "to_node" in parsed
            assert "content" in parsed
            assert "Concurrent message" in parsed["content"]

    @pytest.mark.asyncio
    async def test_concurrent_appends_preserve_order(self, tmp_path: Path) -> None:
        """Test that messages appended concurrently are all persisted."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        message_ids = set()

        async def append_with_id(i: int) -> None:
            msg = Message(
                from_node="a",
                to_node="b",
                content=f"Message {i}",
            )
            message_ids.add(msg.message_id)
            await conv.append(msg)

        # Run concurrent operations
        tasks = [append_with_id(i) for i in range(50)]
        await asyncio.gather(*tasks)

        # Verify all unique message IDs were written
        with open(file_path) as f:
            written_ids = {json.loads(line)["message_id"] for line in f}

        assert len(written_ids) == 50
        assert written_ids == message_ids


class TestConversationFileSize:
    """Tests for file size tracking methods."""

    def test_get_size_nonexistent_file(self, tmp_path: Path) -> None:
        """Test get_size returns 0 for non-existent file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))
        assert conv.get_size() == 0
        assert conv.get_size_mb() == 0.0

    @pytest.mark.asyncio
    async def test_get_size_after_append(self, tmp_path: Path) -> None:
        """Test get_size returns correct size after appending."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(from_node="a", to_node="b", content="Test message")
        await conv.append(msg)

        size = conv.get_size()
        assert size > 0  # File has content

        # Append another message
        await conv.append(msg)
        new_size = conv.get_size()
        assert new_size > size  # Size increased

    @pytest.mark.asyncio
    async def test_get_size_mb_conversion(self, tmp_path: Path) -> None:
        """Test get_size_mb correctly converts to megabytes."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append multiple messages to get measurable size
        for i in range(100):
            msg = Message(
                from_node="a",
                to_node="b",
                content=f"Test message {i} with some content to increase file size",
            )
            await conv.append(msg)

        size_bytes = conv.get_size()
        size_mb = conv.get_size_mb()

        # Verify conversion
        expected_mb = size_bytes / (1024 * 1024)
        assert abs(size_mb - expected_mb) < 0.001  # Account for floating point

    def test_needs_rotation_small_file(self, tmp_path: Path) -> None:
        """Test needs_rotation returns False for small files."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path), max_size_mb=100)

        # Non-existent file
        assert not conv.needs_rotation()

    @pytest.mark.asyncio
    async def test_needs_rotation_below_threshold(self, tmp_path: Path) -> None:
        """Test needs_rotation returns False when below threshold."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path), max_size_mb=1)  # 1 MB threshold

        # Append a small message
        msg = Message(from_node="a", to_node="b", content="Small message")
        await conv.append(msg)

        assert not conv.needs_rotation()

    @pytest.mark.asyncio
    async def test_needs_rotation_above_threshold(self, tmp_path: Path) -> None:
        """Test needs_rotation returns True when exceeding threshold."""
        file_path = tmp_path / "conversation.jsonl"
        # Very small threshold for testing (0.001 MB = 1KB)
        conv = ConversationFile(str(file_path), max_size_mb=0.001)

        # Append enough messages to exceed 1KB
        for i in range(50):
            msg = Message(
                from_node="a",
                to_node="b",
                content=f"Message {i} with enough content to exceed threshold",
            )
            await conv.append(msg)

        # Should need rotation now
        assert conv.needs_rotation()


class TestConversationFileUtility:
    """Tests for utility methods."""

    def test_exists_false_for_nonexistent(self, tmp_path: Path) -> None:
        """Test exists returns False for non-existent file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))
        assert not conv.exists()

    @pytest.mark.asyncio
    async def test_exists_true_after_creation(self, tmp_path: Path) -> None:
        """Test exists returns True after file creation."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(from_node="a", to_node="b", content="Test")
        await conv.append(msg)

        assert conv.exists()


class TestConversationFileEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_append_to_same_file_multiple_instances(self, tmp_path: Path) -> None:
        """Test that multiple ConversationFile instances can write to same file."""
        file_path = tmp_path / "conversation.jsonl"

        conv1 = ConversationFile(str(file_path))
        conv2 = ConversationFile(str(file_path))

        msg1 = Message(from_node="a", to_node="b", content="From instance 1")
        msg2 = Message(from_node="c", to_node="d", content="From instance 2")

        await conv1.append(msg1)
        await conv2.append(msg2)

        # Both messages should be in the file
        with open(file_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_append_preserves_newlines_in_jsonl(self, tmp_path: Path) -> None:
        """Test that each message is on its own line (JSONL format)."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        for i in range(5):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await conv.append(msg)

        # Read raw file and verify newlines
        with open(file_path) as f:
            content = f.read()

        # Should have exactly 5 newlines (one per message)
        assert content.count("\n") == 5

        # Each line should be valid JSON
        lines = content.strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            json.loads(line)  # Should not raise


class TestConversationFileRead:
    """Tests for ConversationFile.read() method."""

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test read returns empty list for non-existent file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        messages = await conv.read()
        assert messages == []

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path: Path) -> None:
        """Test read returns empty list for empty file."""
        file_path = tmp_path / "conversation.jsonl"
        file_path.touch()  # Create empty file
        conv = ConversationFile(str(file_path))

        messages = await conv.read()
        assert messages == []

    @pytest.mark.asyncio
    async def test_read_all_messages(self, tmp_path: Path) -> None:
        """Test reading all messages from file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append multiple messages
        test_messages = [
            Message(from_node="a", to_node="b", content="Message 1"),
            Message(from_node="b", to_node="a", content="Message 2"),
            Message(from_node="a", to_node="b", content="Message 3"),
        ]

        for msg in test_messages:
            await conv.append(msg)

        # Read all messages
        read_messages = await conv.read()

        assert len(read_messages) == 3
        assert read_messages[0].content == "Message 1"
        assert read_messages[1].content == "Message 2"
        assert read_messages[2].content == "Message 3"

    @pytest.mark.asyncio
    async def test_read_preserves_message_properties(self, tmp_path: Path) -> None:
        """Test that read preserves all message properties."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Create message with all properties
        original = Message(
            from_node="agent_a",
            to_node="agent_b",
            content="Test message",
            role=MessageRole.ASSISTANT,
            metadata={"priority": "high", "task_id": 42},
        )

        await conv.append(original)

        # Read back
        messages = await conv.read()
        assert len(messages) == 1

        restored = messages[0]
        assert restored.message_id == original.message_id
        assert restored.from_node == original.from_node
        assert restored.to_node == original.to_node
        assert restored.content == original.content
        assert restored.role == original.role
        assert restored.metadata == original.metadata
        # Timestamps should be very close
        assert abs((restored.timestamp - original.timestamp).total_seconds()) < 0.1

    @pytest.mark.asyncio
    async def test_read_with_since_filter(self, tmp_path: Path) -> None:
        """Test filtering messages by timestamp."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append messages with small delays to ensure different timestamps
        msg1 = Message(from_node="a", to_node="b", content="Message 1")
        await conv.append(msg1)

        await asyncio.sleep(0.01)  # Small delay

        msg2 = Message(from_node="a", to_node="b", content="Message 2")
        await conv.append(msg2)

        await asyncio.sleep(0.01)  # Small delay

        msg3 = Message(from_node="a", to_node="b", content="Message 3")
        await conv.append(msg3)

        # Read messages since msg1's timestamp
        messages = await conv.read(since=msg1.timestamp)

        # Should only get msg2 and msg3 (since is exclusive)
        assert len(messages) == 2
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tmp_path: Path) -> None:
        """Test limiting number of messages returned."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append 10 messages
        for i in range(10):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await conv.append(msg)

        # Read with limit
        messages = await conv.read(limit=3)

        # Should get last 3 messages
        assert len(messages) == 3
        assert messages[0].content == "Message 7"
        assert messages[1].content == "Message 8"
        assert messages[2].content == "Message 9"

    @pytest.mark.asyncio
    async def test_read_with_limit_greater_than_total(self, tmp_path: Path) -> None:
        """Test limit greater than total messages returns all."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append 3 messages
        for i in range(3):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await conv.append(msg)

        # Read with limit greater than total
        messages = await conv.read(limit=10)

        # Should get all 3 messages
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_read_with_since_and_limit(self, tmp_path: Path) -> None:
        """Test combining since and limit filters."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append 10 messages with delays
        messages_sent = []
        for i in range(10):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await conv.append(msg)
            messages_sent.append(msg)
            if i < 9:
                await asyncio.sleep(0.01)

        # Read messages since message 3, limit to 3
        since_timestamp = messages_sent[3].timestamp
        messages = await conv.read(since=since_timestamp, limit=3)

        # Should get messages 7, 8, 9 (last 3 of the 6 messages after msg3)
        assert len(messages) == 3
        assert messages[0].content == "Message 7"
        assert messages[1].content == "Message 8"
        assert messages[2].content == "Message 9"

    @pytest.mark.asyncio
    async def test_read_with_malformed_json(self, tmp_path: Path) -> None:
        """Test that malformed JSON lines are skipped gracefully."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Write valid message
        msg1 = Message(from_node="a", to_node="b", content="Valid 1")
        await conv.append(msg1)

        # Manually append malformed JSON
        async with aiofiles.open(file_path, mode="a") as f:
            await f.write("{ this is not valid json }\n")

        # Write another valid message
        msg2 = Message(from_node="a", to_node="b", content="Valid 2")
        await conv.append(msg2)

        # Read should skip malformed line
        messages = await conv.read()

        assert len(messages) == 2
        assert messages[0].content == "Valid 1"
        assert messages[1].content == "Valid 2"

    @pytest.mark.asyncio
    async def test_read_with_invalid_message_data(self, tmp_path: Path) -> None:
        """Test that invalid message data is skipped gracefully."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Write valid message
        msg1 = Message(from_node="a", to_node="b", content="Valid 1")
        await conv.append(msg1)

        # Manually append valid JSON but invalid Message data
        async with aiofiles.open(file_path, mode="a") as f:
            # Missing required fields
            await f.write('{"invalid": "data", "missing": "required_fields"}\n')

        # Write another valid message
        msg2 = Message(from_node="a", to_node="b", content="Valid 2")
        await conv.append(msg2)

        # Read should skip invalid message data
        messages = await conv.read()

        assert len(messages) == 2
        assert messages[0].content == "Valid 1"
        assert messages[1].content == "Valid 2"

    @pytest.mark.asyncio
    async def test_read_with_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Write message
        msg1 = Message(from_node="a", to_node="b", content="Message 1")
        await conv.append(msg1)

        # Add empty lines
        async with aiofiles.open(file_path, mode="a") as f:
            await f.write("\n")
            await f.write("   \n")  # Whitespace only

        # Write another message
        msg2 = Message(from_node="a", to_node="b", content="Message 2")
        await conv.append(msg2)

        # Read should skip empty lines
        messages = await conv.read()

        assert len(messages) == 2
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Message 2"

    @pytest.mark.asyncio
    async def test_read_with_unicode_content(self, tmp_path: Path) -> None:
        """Test reading messages with Unicode content."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Write message with Unicode
        msg = Message(
            from_node="a",
            to_node="b",
            content="Hello 世界 🌍 émojis",
        )
        await conv.append(msg)

        # Read and verify Unicode is preserved
        messages = await conv.read()
        assert len(messages) == 1
        assert messages[0].content == "Hello 世界 🌍 émojis"

    @pytest.mark.asyncio
    async def test_read_preserves_message_order(self, tmp_path: Path) -> None:
        """Test that messages are returned in chronological order."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append messages
        for i in range(5):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await conv.append(msg)
            await asyncio.sleep(0.01)  # Ensure different timestamps

        # Read all
        messages = await conv.read()

        # Verify order
        for i in range(5):
            assert messages[i].content == f"Message {i}"

        # Verify timestamps are in order
        for i in range(4):
            assert messages[i].timestamp < messages[i + 1].timestamp


class TestConversationFileRotation:
    """Tests for ConversationFile rotation functionality."""

    @pytest.mark.asyncio
    async def test_rotate_nonexistent_file(self, tmp_path: Path) -> None:
        """Test rotating non-existent file returns empty string."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        archive_path = await conv.rotate()
        assert archive_path == ""

    @pytest.mark.asyncio
    async def test_rotate_creates_archive(self, tmp_path: Path) -> None:
        """Test that rotate creates an archive file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append a message
        msg = Message(from_node="a", to_node="b", content="Test")
        await conv.append(msg)

        # Rotate
        archive_path = await conv.rotate()

        # Verify archive was created
        assert archive_path != ""
        assert Path(archive_path).exists()
        assert not file_path.exists()  # Original file should be gone

        # Verify archive filename format
        archive_name = Path(archive_path).name
        assert archive_name.startswith("conversation.")
        assert archive_name.endswith(".jsonl")
        assert ".jsonl" in archive_name  # Has timestamp

    @pytest.mark.asyncio
    async def test_rotate_preserves_content(self, tmp_path: Path) -> None:
        """Test that rotation preserves file content."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append messages
        messages = [
            Message(from_node="a", to_node="b", content="Message 1"),
            Message(from_node="b", to_node="a", content="Message 2"),
        ]
        for msg in messages:
            await conv.append(msg)

        # Rotate
        archive_path = await conv.rotate()

        # Read from archive
        archive_conv = ConversationFile(archive_path)
        archived_messages = await archive_conv.read()

        # Verify content preserved
        assert len(archived_messages) == 2
        assert archived_messages[0].content == "Message 1"
        assert archived_messages[1].content == "Message 2"

    @pytest.mark.asyncio
    async def test_get_archive_files_empty(self, tmp_path: Path) -> None:
        """Test get_archive_files returns empty list when no archives."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        archives = conv.get_archive_files()
        assert archives == []

    @pytest.mark.asyncio
    async def test_get_archive_files_single_archive(self, tmp_path: Path) -> None:
        """Test get_archive_files finds a single archive."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Create and rotate
        msg = Message(from_node="a", to_node="b", content="Test")
        await conv.append(msg)
        archive_path = await conv.rotate()

        # Get archives
        archives = conv.get_archive_files()

        assert len(archives) == 1
        assert str(archives[0]) == archive_path

    @pytest.mark.asyncio
    async def test_get_archive_files_multiple_archives(self, tmp_path: Path) -> None:
        """Test get_archive_files finds multiple archives in order."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Create multiple rotations
        archive_paths = []
        for i in range(3):
            msg = Message(from_node="a", to_node="b", content=f"Msg {i}")
            await conv.append(msg)
            archive_path = await conv.rotate()
            archive_paths.append(archive_path)
            await asyncio.sleep(0.01)  # Ensure different timestamps

        # Get archives
        archives = conv.get_archive_files()

        # Should find all 3, sorted chronologically
        assert len(archives) == 3
        for i, archive in enumerate(archives):
            assert str(archive) == archive_paths[i]

    @pytest.mark.asyncio
    async def test_automatic_rotation_on_append(self, tmp_path: Path) -> None:
        """Test that append automatically rotates when size threshold exceeded."""
        file_path = tmp_path / "conversation.jsonl"
        # Very small threshold (0.001 MB = 1KB)
        conv = ConversationFile(str(file_path), max_size_mb=0.001)

        # Append enough messages to trigger rotation
        for i in range(30):
            msg = Message(
                from_node="a",
                to_node="b",
                content=f"Message {i} with enough content to exceed 1KB threshold eventually",
            )
            await conv.append(msg)

        # Should have created at least one archive
        archives = conv.get_archive_files()
        assert len(archives) >= 1

        # Current file should exist and have recent messages
        assert conv.exists()

    @pytest.mark.asyncio
    async def test_read_with_archives_no_archives(self, tmp_path: Path) -> None:
        """Test read_with_archives works when no archives exist."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append messages
        for i in range(3):
            msg = Message(from_node="a", to_node="b", content=f"Message {i}")
            await conv.append(msg)

        # Read with archives
        messages = await conv.read_with_archives()

        assert len(messages) == 3
        assert messages[0].content == "Message 0"

    @pytest.mark.asyncio
    async def test_read_with_archives_single_rotation(self, tmp_path: Path) -> None:
        """Test read_with_archives reads from archive and current file."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Append messages
        for i in range(3):
            msg = Message(from_node="a", to_node="b", content=f"Old message {i}")
            await conv.append(msg)

        # Rotate
        await conv.rotate()

        # Append new messages
        for i in range(2):
            msg = Message(from_node="a", to_node="b", content=f"New message {i}")
            await conv.append(msg)

        # Read all with archives
        all_messages = await conv.read_with_archives()

        # Should get all 5 messages
        assert len(all_messages) == 5
        assert all_messages[0].content == "Old message 0"
        assert all_messages[3].content == "New message 0"

    @pytest.mark.asyncio
    async def test_read_with_archives_multiple_rotations(self, tmp_path: Path) -> None:
        """Test read_with_archives across multiple rotated files."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Create 3 rotations with messages
        for rotation in range(3):
            for i in range(2):
                msg = Message(from_node="a", to_node="b", content=f"R{rotation}-M{i}")
                await conv.append(msg)
            await conv.rotate()
            await asyncio.sleep(0.01)  # Ensure different timestamps

        # Add current messages
        for i in range(2):
            msg = Message(from_node="a", to_node="b", content=f"Current-M{i}")
            await conv.append(msg)

        # Read all
        all_messages = await conv.read_with_archives()

        # Should get all 8 messages (3 rotations * 2 + 2 current)
        assert len(all_messages) == 8
        assert all_messages[0].content == "R0-M0"
        assert all_messages[1].content == "R0-M1"
        assert all_messages[6].content == "Current-M0"

    @pytest.mark.asyncio
    async def test_read_with_archives_with_limit(self, tmp_path: Path) -> None:
        """Test read_with_archives respects limit parameter."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        # Add 10 messages, rotate, add 10 more
        for i in range(10):
            msg = Message(from_node="a", to_node="b", content=f"Old {i}")
            await conv.append(msg)

        await conv.rotate()

        for i in range(10):
            msg = Message(from_node="a", to_node="b", content=f"New {i}")
            await conv.append(msg)

        # Read with limit
        messages = await conv.read_with_archives(limit=5)

        # Should get last 5 messages
        assert len(messages) == 5
        assert all("New" in msg.content for msg in messages)
        assert messages[0].content == "New 5"
        assert messages[4].content == "New 9"

    @pytest.mark.asyncio
    async def test_read_with_archives_chronological_order(self, tmp_path: Path) -> None:
        """Test read_with_archives returns messages in chronological order."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        all_timestamps = []

        # Create multiple rotations
        for rotation in range(3):
            for i in range(3):
                msg = Message(from_node="a", to_node="b", content=f"R{rotation}-M{i}")
                await conv.append(msg)
                all_timestamps.append(msg.timestamp)
                await asyncio.sleep(0.01)  # Ensure different timestamps
            await conv.rotate()

        # Read all
        messages = await conv.read_with_archives()

        # Verify chronological order
        for i in range(len(messages) - 1):
            assert messages[i].timestamp <= messages[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_rotation_filename_format(self, tmp_path: Path) -> None:
        """Test that rotated filenames follow expected format."""
        file_path = tmp_path / "conversation.jsonl"
        conv = ConversationFile(str(file_path))

        msg = Message(from_node="a", to_node="b", content="Test")
        await conv.append(msg)

        archive_path = await conv.rotate()
        archive_name = Path(archive_path).name

        # Format: conversation.YYYY-MM-DDTHH-MM-SS-mmmmmm.jsonl
        parts = archive_name.split(".")
        assert len(parts) == 3  # name, timestamp, jsonl
        assert parts[0] == "conversation"
        assert parts[2] == "jsonl"

        # Timestamp should match pattern
        timestamp_part = parts[1]
        assert len(timestamp_part) == 26  # YYYY-MM-DDTHH-MM-SS-mmmmmm
        assert timestamp_part[4] == "-"  # Year separator
        assert timestamp_part[10] == "T"  # Date/time separator
