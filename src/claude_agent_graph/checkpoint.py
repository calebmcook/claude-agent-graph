"""
Checkpoint management for graph persistence and recovery.

This module provides checkpoint saving and loading functionality for AgentGraph instances,
enabling graph state persistence and crash recovery.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack

from .models import Edge, Node

logger = logging.getLogger(__name__)

# Checkpoint format version for future migrations
CHECKPOINT_VERSION = 1


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint data is corrupted or invalid."""

    pass


class CheckpointVersionError(CheckpointError):
    """Raised when checkpoint version is not supported."""

    pass


class Checkpoint:
    """
    Manages checkpoint serialization and deserialization.

    Checkpoints store:
    - Graph structure (nodes and edges)
    - Graph metadata (name, topology_constraint, etc.)
    - Conversation file references
    - Integrity checksum
    """

    def __init__(
        self,
        name: str,
        nodes: dict[str, Node],
        edges: dict[str, Edge],
        metadata: dict[str, Any],
        timestamp: datetime | None = None,
    ):
        """
        Initialize a checkpoint.

        Args:
            name: Name of the graph
            nodes: Dictionary of nodes by node_id
            edges: Dictionary of edges by edge_id
            metadata: Graph metadata (topology_constraint, max_nodes, etc.)
            timestamp: When checkpoint was created (default: now)
        """
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.metadata = metadata
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.version = CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        Convert checkpoint to dictionary format.

        Returns:
            Dictionary representation of checkpoint
        """
        # Convert nodes to dict, ensuring datetime fields are serialized to ISO format
        nodes_dict = {}
        for node_id, node in self.nodes.items():
            node_data = node.model_dump(exclude={"agent_session"})
            # Convert datetime fields to ISO format for msgpack serialization
            if "created_at" in node_data and isinstance(node_data["created_at"], datetime):
                node_data["created_at"] = node_data["created_at"].isoformat()
            nodes_dict[node_id] = node_data

        # Convert edges to dict, ensuring datetime fields are serialized to ISO format
        edges_dict = {}
        for edge_id, edge in self.edges.items():
            edge_data = edge.model_dump()
            if "created_at" in edge_data and isinstance(edge_data["created_at"], datetime):
                edge_data["created_at"] = edge_data["created_at"].isoformat()
            edges_dict[edge_id] = edge_data

        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "metadata": self.metadata,
            "nodes": nodes_dict,
            "edges": edges_dict,
        }

    @staticmethod
    def compute_checksum(data: dict[str, Any]) -> str:
        """
        Compute SHA256 checksum of checkpoint data.

        Args:
            data: Dictionary to checksum (excluding checksum field itself)

        Returns:
            Hex string of SHA256 hash
        """
        # Serialize data deterministically for consistent checksums
        serialized = msgpack.packb(data, use_bin_type=True)
        return hashlib.sha256(serialized).hexdigest()

    def save(self, filepath: Path) -> None:
        """
        Save checkpoint to file with integrity checksum.

        Args:
            filepath: Path where checkpoint should be saved

        Raises:
            CheckpointError: If save fails
        """
        try:
            # Prepare data
            data = self.to_dict()

            # Compute checksum
            checksum = self.compute_checksum(data)
            data["checksum"] = checksum

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write checkpoint using msgpack
            serialized = msgpack.packb(data, use_bin_type=True)
            with open(filepath, "wb") as f:
                f.write(serialized)

            logger.debug(f"Saved checkpoint to {filepath}")
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    @staticmethod
    def load(filepath: Path) -> "Checkpoint":
        """
        Load checkpoint from file and validate integrity.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Loaded Checkpoint instance

        Raises:
            CheckpointError: If load or validation fails
            CheckpointVersionError: If checkpoint version not supported
            CheckpointCorruptedError: If checksum validation fails
        """
        try:
            if not filepath.exists():
                raise CheckpointError(f"Checkpoint file not found: {filepath}")

            # Read checkpoint
            with open(filepath, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)

            # Validate version
            version = data.get("version")
            if version is None:
                raise CheckpointVersionError("Checkpoint missing version field")
            if version != CHECKPOINT_VERSION:
                raise CheckpointVersionError(
                    f"Unsupported checkpoint version {version}. " f"Expected {CHECKPOINT_VERSION}"
                )

            # Validate checksum
            stored_checksum = data.pop("checksum", None)
            if stored_checksum is None:
                raise CheckpointCorruptedError("Checkpoint missing integrity checksum")

            computed_checksum = Checkpoint.compute_checksum(data)
            if stored_checksum != computed_checksum:
                raise CheckpointCorruptedError(
                    "Checkpoint integrity check failed: checksums don't match"
                )

            # Deserialize nodes and edges, converting ISO timestamp strings back to datetime
            timestamp = datetime.fromisoformat(data["timestamp"])

            # Restore nodes with datetime parsing
            nodes = {}
            for node_id, node_data in data.get("nodes", {}).items():
                # Parse created_at back to datetime if it's a string
                if "created_at" in node_data and isinstance(node_data["created_at"], str):
                    node_data["created_at"] = datetime.fromisoformat(node_data["created_at"])
                nodes[node_id] = Node(**node_data)

            # Restore edges with datetime parsing
            edges = {}
            for edge_id, edge_data in data.get("edges", {}).items():
                if "created_at" in edge_data and isinstance(edge_data["created_at"], str):
                    edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
                edges[edge_id] = Edge(**edge_data)

            checkpoint = Checkpoint(
                name=data["name"],
                nodes=nodes,
                edges=edges,
                metadata=data.get("metadata", {}),
                timestamp=timestamp,
            )

            logger.debug(f"Loaded checkpoint from {filepath}")
            return checkpoint

        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e
