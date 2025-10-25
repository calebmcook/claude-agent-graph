"""
Storage backend interfaces and implementations for claude-agent-graph.

This module provides an abstraction layer for different storage backends,
allowing users to choose between filesystem, database, or in-memory storage.
"""

from .base import StorageBackend
from .filesystem import FilesystemBackend

__all__ = ["StorageBackend", "FilesystemBackend"]
