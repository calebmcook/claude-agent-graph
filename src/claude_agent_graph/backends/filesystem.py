"""
Filesystem-based storage backend implementation.

Uses JSONL files for conversation storage with automatic log rotation.
"""

import shutil
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

    async def archive_conversation(self, edge_id: str) -> None:
        """
        Archive a conversation file to the archived/ subdirectory.

        Moves the conversation file and any rotated archives to an archived location
        with a timestamp suffix. This preserves the conversation history.

        Args:
            edge_id: The edge identifier

        Raises:
            OSError: If archival operation fails
        """
        conv_file = self._get_conversation_file(edge_id)

        # Check if conversation file exists
        if not conv_file.exists():
            return  # Nothing to archive

        # Create archived directory
        archived_dir = self.base_dir / "archived"
        archived_dir.mkdir(parents=True, exist_ok=True)

        # Generate archive path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archived_dir / f"{edge_id}_{timestamp}.jsonl"

        # Move conversation file to archive
        source_path = conv_file.file_path
        shutil.move(str(source_path), str(archive_path))

        # Also move any rotated archives
        archive_files = conv_file.get_archive_files()
        for archive_file in archive_files:
            archive_name = f"{edge_id}_{archive_file.stem}_{timestamp}.jsonl"
            dest = archived_dir / archive_name
            shutil.move(str(archive_file), str(dest))

        # Remove from cache so it won't be reused
        self._conversation_files.pop(edge_id, None)
