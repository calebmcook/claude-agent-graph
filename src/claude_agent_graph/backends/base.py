"""
Abstract base class for storage backends.

Defines the interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Message


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    All storage backends must implement these methods to provide
    conversation persistence for agent graphs.
    """

    @abstractmethod
    async def append_message(self, edge_id: str, message: Message) -> None:
        """
        Append a message to an edge's conversation.

        Args:
            edge_id: The edge identifier
            message: Message to append

        Raises:
            OSError: If storage operation fails
        """
        pass

    @abstractmethod
    async def read_messages(
        self,
        edge_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        Read messages from an edge's conversation.

        Args:
            edge_id: The edge identifier
            since: Only return messages after this timestamp
            limit: Maximum number of messages to return

        Returns:
            List of Message objects

        Raises:
            OSError: If storage operation fails
        """
        pass

    @abstractmethod
    def conversation_exists(self, edge_id: str) -> bool:
        """
        Check if a conversation file exists for an edge.

        Args:
            edge_id: The edge identifier

        Returns:
            True if conversation exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_conversation_size(self, edge_id: str) -> int:
        """
        Get the size of a conversation in bytes.

        Args:
            edge_id: The edge identifier

        Returns:
            Size in bytes, or 0 if conversation doesn't exist
        """
        pass

    @abstractmethod
    async def archive_conversation(self, edge_id: str) -> None:
        """
        Archive a conversation file.

        Moves the conversation file to an archived location with timestamp.
        This preserves conversation history for audit and debugging.

        Args:
            edge_id: The edge identifier

        Raises:
            OSError: If archival operation fails
        """
        pass
