"""
Storage layer for claude-agent-graph package.

This module provides conversation file management with thread-safe async I/O,
JSONL format persistence, and automatic log rotation.
"""

import asyncio
import json
import logging
from datetime import datetime
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
        Read messages from the conversation file (to be implemented in Story 3.1.2).

        Args:
            since: Only return messages after this timestamp
            limit: Maximum number of messages to return

        Returns:
            List of Message objects

        Note:
            This is a placeholder for Story 3.1.2 implementation.
        """
        raise NotImplementedError("read() will be implemented in Story 3.1.2")

    async def rotate(self) -> str:
        """
        Rotate the conversation file (to be implemented in Story 3.1.3).

        Returns:
            Path to the archived file

        Note:
            This is a placeholder for Story 3.1.3 implementation.
        """
        raise NotImplementedError("rotate() will be implemented in Story 3.1.3")
