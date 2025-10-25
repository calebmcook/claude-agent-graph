"""
Storage layer for claude-agent-graph package.

This module provides conversation file management with thread-safe async I/O,
JSONL format persistence, and automatic log rotation.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from claude_agent_graph.models import Message

logger = logging.getLogger(__name__)


class ConversationFile:
    """
    Thread-safe async manager for conversation files in JSONL format.

    Each conversation file stores messages exchanged between agents as
    newline-delimited JSON (JSONL). This format enables:
    - Append-only writes for consistency
    - Efficient streaming and filtering
    - Human-readable conversation logs
    - Line-by-line parsing without loading entire file

    Thread safety is achieved using asyncio.Lock for all file operations.
    """

    def __init__(self, file_path: str, max_size_mb: int = 100) -> None:
        """
        Initialize conversation file manager.

        Args:
            file_path: Path to the conversation JSONL file
            max_size_mb: Maximum file size in MB before rotation (default: 100)

        Example:
            >>> conv = ConversationFile("conversations/agent1_agent2.jsonl")
            >>> await conv.append(message)
        """
        self.file_path = Path(file_path)
        self.max_size_mb = max_size_mb
        self._lock = asyncio.Lock()
        self._initialized = False

        # Validate file extension
        if self.file_path.suffix != ".jsonl":
            raise ValueError(f"File path must have .jsonl extension, got: {file_path}")

    async def _ensure_directory(self) -> None:
        """
        Ensure the parent directory exists for the conversation file.

        Creates all necessary parent directories if they don't exist.
        This is called automatically before the first write operation.
        """
        if not self._initialized:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.debug(f"Initialized conversation file directory: {self.file_path.parent}")

    async def append(self, message: Message) -> None:
        """
        Append a message to the conversation file (thread-safe).

        The message is serialized to JSON and appended as a new line in JSONL format.
        This operation is atomic and thread-safe using asyncio.Lock.

        Args:
            message: Message object to append to the conversation

        Raises:
            IOError: If file write operation fails
            ValueError: If message serialization fails

        Example:
            >>> msg = Message(from_node="a", to_node="b", content="Hello")
            >>> await conv.append(msg)
        """
        async with self._lock:
            try:
                # Ensure directory exists before first write
                await self._ensure_directory()

                # Serialize message to dict and then to JSON
                message_dict = message.to_dict()
                json_line = json.dumps(message_dict, ensure_ascii=False)

                # Append to file with newline
                async with aiofiles.open(self.file_path, mode="a", encoding="utf-8") as f:
                    await f.write(json_line + "\n")

                logger.debug(
                    f"Appended message {message.message_id} to {self.file_path} "
                    f"({message.from_node} -> {message.to_node})"
                )

            except OSError as e:
                logger.error(f"Failed to write message to {self.file_path}: {e}")
                raise OSError(f"Failed to append message to conversation file: {e}") from e
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize message {message.message_id}: {e}")
                raise ValueError(f"Failed to serialize message: {e}") from e

    def get_size(self) -> int:
        """
        Get the current size of the conversation file in bytes.

        Returns:
            File size in bytes, or 0 if file doesn't exist

        Example:
            >>> size = conv.get_size()
            >>> print(f"File size: {size / 1024 / 1024:.2f} MB")
        """
        if self.file_path.exists():
            return self.file_path.stat().st_size
        return 0

    def get_size_mb(self) -> float:
        """
        Get the current size of the conversation file in megabytes.

        Returns:
            File size in MB, or 0.0 if file doesn't exist

        Example:
            >>> size_mb = conv.get_size_mb()
            >>> print(f"File size: {size_mb:.2f} MB")
        """
        return self.get_size() / (1024 * 1024)

    def needs_rotation(self) -> bool:
        """
        Check if the conversation file needs rotation based on size.

        Returns:
            True if file size exceeds max_size_mb, False otherwise

        Example:
            >>> if conv.needs_rotation():
            ...     await conv.rotate()
        """
        return self.get_size_mb() >= self.max_size_mb

    def exists(self) -> bool:
        """
        Check if the conversation file exists.

        Returns:
            True if file exists, False otherwise
        """
        return self.file_path.exists()

    async def read(
        self,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        Read messages from the conversation file with optional filtering.

        Messages are parsed from JSONL format and returned as Message objects.
        Malformed lines are logged as warnings and skipped gracefully.

        Args:
            since: Only return messages after this timestamp (exclusive)
            limit: Maximum number of messages to return (most recent if specified)

        Returns:
            List of Message objects, ordered by timestamp (oldest first)
            Returns empty list if file doesn't exist

        Example:
            >>> # Read all messages
            >>> messages = await conv.read()
            >>>
            >>> # Read messages since a specific time
            >>> since_time = datetime(2025, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
            >>> recent = await conv.read(since=since_time)
            >>>
            >>> # Read last 10 messages
            >>> last_ten = await conv.read(limit=10)

        Raises:
            OSError: If file read operation fails
        """
        # Return empty list if file doesn't exist
        if not self.file_path.exists():
            logger.debug(f"Conversation file does not exist: {self.file_path}")
            return []

        messages: list[Message] = []
        line_number = 0

        try:
            async with aiofiles.open(self.file_path, encoding="utf-8") as f:
                async for line in f:
                    line_number += 1
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue

                    try:
                        # Parse JSON line
                        data = json.loads(line)

                        # Convert to Message object
                        message = Message.from_dict(data)

                        # Filter by timestamp if specified
                        if since is not None:
                            # Ensure since has timezone info for comparison
                            if since.tzinfo is None:
                                since = since.replace(tzinfo=timezone.utc)

                            if message.timestamp <= since:
                                continue  # Skip messages at or before 'since'

                        # Add to results
                        messages.append(message)

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Malformed JSON at line {line_number} in {self.file_path}: {e}"
                        )
                        continue  # Skip malformed line
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(
                            f"Invalid message data at line {line_number} in {self.file_path}: {e}"
                        )
                        continue  # Skip invalid message data

            logger.debug(
                f"Read {len(messages)} messages from {self.file_path} "
                f"(since={since}, limit={limit})"
            )

        except OSError as e:
            logger.error(f"Failed to read from {self.file_path}: {e}")
            raise OSError(f"Failed to read conversation file: {e}") from e

        # Apply limit if specified (return most recent N messages)
        if limit is not None and limit > 0:
            messages = messages[-limit:]

        return messages

    async def rotate(self) -> str:
        """
        Rotate the conversation file (to be implemented in Story 3.1.3).

        Returns:
            Path to the archived file

        Note:
            This is a placeholder for Story 3.1.3 implementation.
        """
        raise NotImplementedError("rotate() will be implemented in Story 3.1.3")
