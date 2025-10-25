"""
Filesystem-based storage backend implementation.

Uses JSONL files for conversation storage with automatic log rotation.
"""

from datetime import datetime
from pathlib import Path

from ..models import Message
from ..storage import ConversationFile
from .base import StorageBackend


class FilesystemBackend(StorageBackend):
    """
    Filesystem storage backend using JSONL files.

    Stores conversations as JSONL files in a configurable directory structure.
    Each edge gets its own conversation file with automatic rotation.
    """

    def __init__(self, base_dir: str = "./conversations", max_size_mb: int = 100):
        """
        Initialize filesystem backend.

        Args:
            base_dir: Base directory for all conversation files
            max_size_mb: Maximum size per conversation file before rotation
        """
        self.base_dir = Path(base_dir)
        self.max_size_mb = max_size_mb
        self._conversation_files: dict[str, ConversationFile] = {}

    def _get_conversation_file(self, edge_id: str) -> ConversationFile:
        """Get or create ConversationFile for an edge."""
        if edge_id not in self._conversation_files:
            file_path = self.base_dir / f"{edge_id}.jsonl"
            self._conversation_files[edge_id] = ConversationFile(
                str(file_path), max_size_mb=self.max_size_mb
            )
        return self._conversation_files[edge_id]

    async def append_message(self, edge_id: str, message: Message) -> None:
        """Append message to edge's conversation file."""
        conv_file = self._get_conversation_file(edge_id)
        await conv_file.append(message)

    async def read_messages(
        self,
        edge_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Read messages from edge's conversation file."""
        conv_file = self._get_conversation_file(edge_id)
        return await conv_file.read(since=since, limit=limit)

    def conversation_exists(self, edge_id: str) -> bool:
        """Check if conversation file exists."""
        conv_file = self._get_conversation_file(edge_id)
        return conv_file.exists()

    async def get_conversation_size(self, edge_id: str) -> int:
        """Get conversation file size in bytes."""
        conv_file = self._get_conversation_file(edge_id)
        return conv_file.get_size()
