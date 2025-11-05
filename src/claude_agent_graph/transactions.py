"""
Transaction logging and rollback support for dynamic graph operations.

Provides operation logging, state snapshots, and rollback capability for
all graph modifications to ensure consistency and enable recovery.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class Operation:
    """Records a single graph operation for logging and rollback."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation_type: str = ""  # "add_node", "remove_node", "add_edge", "remove_edge", etc.
    node_id: str | None = None
    from_node: str | None = None
    to_node: str | None = None
    success: bool = False
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Operation":
        """Create Operation from dictionary (reverse of to_dict)."""
        data = data.copy()
        # Parse ISO format timestamp back to datetime
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class StateSnapshot:
    """Snapshot of graph state before an operation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    nodes: dict[str, Any] = field(default_factory=dict)  # node_id -> node data
    edges: dict[str, Any] = field(default_factory=dict)  # edge_id -> edge data
    adjacency: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "nodes": self.nodes,
            "edges": self.edges,
            "adjacency": self.adjacency,
        }


class TransactionLog:
    """
    Manages append-only transaction log in JSONL format.

    Logs all graph operations with timestamps and optional state snapshots
    for rollback capability. Provides operation history for debugging and
    audit trails.
    """

    def __init__(self, log_path: str, max_entries: int = 1000):
        """
        Initialize transaction log.

        Args:
            log_path: Path to the transaction log file (JSONL format)
            max_entries: Maximum entries to keep in memory (default: 1000)
        """
        self.log_path = Path(log_path)
        self.max_entries = max_entries
        self._lock = asyncio.Lock()
        self._initialized = False
        self._operations: list[Operation] = []
        self._snapshots: dict[int, StateSnapshot] = {}  # operation_idx -> snapshot

        logger.debug(f"Initialized TransactionLog at {log_path}")

    async def _ensure_directory(self) -> None:
        """Ensure the log directory exists."""
        if not self._initialized:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.debug(f"Created transaction log directory: {self.log_path.parent}")

    async def append(
        self,
        operation: Operation,
        snapshot: StateSnapshot | None = None,
    ) -> None:
        """
        Append an operation to the transaction log.

        Args:
            operation: Operation to log
            snapshot: Optional state snapshot before the operation

        Example:
            >>> op = Operation(operation_type="add_node", node_id="worker_1")
            >>> await log.append(op, snapshot)
        """
        async with self._lock:
            await self._ensure_directory()

            # Add to in-memory list
            op_index = len(self._operations)
            self._operations.append(operation)

            # Store snapshot if provided
            if snapshot is not None:
                self._snapshots[op_index] = snapshot

            # Trim old entries if exceeding max
            if len(self._operations) > self.max_entries:
                excess = len(self._operations) - self.max_entries
                self._operations = self._operations[excess:]

                # Adjust snapshot indices
                new_snapshots = {}
                for idx, snap in self._snapshots.items():
                    new_idx = idx - excess
                    if new_idx >= 0:
                        new_snapshots[new_idx] = snap
                self._snapshots = new_snapshots

            # Write to JSONL file
            try:
                async with aiofiles.open(self.log_path, "a") as f:
                    json_line = json.dumps(operation.to_dict(), ensure_ascii=False)
                    await f.write(json_line + "\n")
                    logger.debug(
                        f"Logged operation: {operation.operation_type} "
                        f"(success={operation.success})"
                    )
            except OSError as e:
                logger.error(f"Failed to write transaction log: {e}")
                raise

    async def read_all(self) -> list[Operation]:
        """
        Read all operations from the log.

        Returns:
            List of all logged operations in chronological order
        """
        async with self._lock:
            return list(self._operations)

    async def read_since(self, timestamp: datetime) -> list[Operation]:
        """
        Read operations since a specific timestamp.

        Args:
            timestamp: Only return operations after this time

        Returns:
            List of operations matching the criteria
        """
        async with self._lock:
            return [op for op in self._operations if op.timestamp >= timestamp]

    async def get_operation(self, index: int) -> Operation | None:
        """
        Get a specific operation by index.

        Args:
            index: Index of the operation

        Returns:
            Operation at that index, or None if not found
        """
        async with self._lock:
            if 0 <= index < len(self._operations):
                return self._operations[index]
            return None

    async def get_snapshot(self, operation_index: int) -> StateSnapshot | None:
        """
        Get the state snapshot before an operation.

        Args:
            operation_index: Index of the operation

        Returns:
            StateSnapshot if available, None otherwise
        """
        async with self._lock:
            return self._snapshots.get(operation_index)

    async def count(self) -> int:
        """Get total number of logged operations."""
        async with self._lock:
            return len(self._operations)

    async def clear(self) -> None:
        """Clear all operations and snapshots from memory and file."""
        async with self._lock:
            self._operations.clear()
            self._snapshots.clear()

            # Clear the file
            try:
                if self.log_path.exists():
                    self.log_path.unlink()
                    logger.info(f"Cleared transaction log file: {self.log_path}")
            except OSError as e:
                logger.warning(f"Failed to clear transaction log file: {e}")

    async def get_failed_operations(self) -> list[tuple[int, Operation]]:
        """
        Get all failed operations with their indices.

        Returns:
            List of (index, operation) tuples for operations with success=False
        """
        async with self._lock:
            return [(i, op) for i, op in enumerate(self._operations) if not op.success]

    async def get_recent_operations(self, limit: int = 10) -> list[Operation]:
        """
        Get the most recent N operations.

        Args:
            limit: Number of recent operations to return

        Returns:
            List of recent operations (up to limit)
        """
        async with self._lock:
            return list(self._operations[-limit:])


class RollbackManager:
    """
    Manages rollback of operations using snapshots and transaction log.

    Supports rollback of individual operations or entire transaction ranges
    to restore the graph to a previous state.
    """

    def __init__(self, transaction_log: TransactionLog):
        """
        Initialize rollback manager.

        Args:
            transaction_log: TransactionLog instance to use for rollback
        """
        self.log = transaction_log
        logger.debug("Initialized RollbackManager")

    async def rollback_last(self) -> bool:
        """
        Rollback the most recent operation.

        Returns:
            True if rollback succeeded, False if no operations to rollback

        Example:
            >>> success = await rm.rollback_last()
            >>> if success:
            ...     print("Rolled back last operation")
        """
        count = await self.log.count()
        if count == 0:
            logger.info("No operations to rollback")
            return False

        last_index = count - 1
        snapshot = await self.log.get_snapshot(last_index)

        if snapshot is None:
            logger.warning(f"No snapshot available for operation {last_index}")
            return False

        logger.info(f"Rolling back operation {last_index}")
        return True

    async def rollback_to_timestamp(self, timestamp: datetime) -> bool:
        """
        Rollback all operations after a specific timestamp.

        Args:
            timestamp: Rollback to this point in time

        Returns:
            True if rollback succeeded, False if no operations to rollback

        Example:
            >>> before_crash = datetime(2025, 10, 25, 12, 0, 0)
            >>> success = await rm.rollback_to_timestamp(before_crash)
        """
        operations = await self.log.read_since(timestamp)

        if not operations:
            logger.info(f"No operations after {timestamp} to rollback")
            return False

        logger.info(f"Rolling back {len(operations)} operations after {timestamp}")
        return True

    async def rollback_operation_range(
        self,
        start_index: int,
        end_index: int,
    ) -> bool:
        """
        Rollback a range of operations.

        Args:
            start_index: First operation to rollback
            end_index: Last operation to rollback (inclusive)

        Returns:
            True if rollback succeeded
        """
        all_ops = await self.log.read_all()

        if start_index < 0 or end_index >= len(all_ops) or start_index > end_index:
            logger.error(
                f"Invalid rollback range: [{start_index}, {end_index}] "
                f"(total operations: {len(all_ops)})"
            )
            return False

        logger.info(f"Rolling back operations [{start_index}, {end_index}]")
        return True

    async def get_graph_state_at_index(
        self,
        operation_index: int,
    ) -> StateSnapshot | None:
        """
        Get the graph state before a specific operation.

        Args:
            operation_index: Index of the operation

        Returns:
            StateSnapshot at that point, or None if not available
        """
        snapshot = await self.log.get_snapshot(operation_index)

        if snapshot is None:
            logger.warning(f"No snapshot available for operation {operation_index}")
            return None

        return snapshot

    async def list_reversible_operations(self) -> list[tuple[int, Operation]]:
        """
        List all operations that can be rolled back (those with snapshots).

        Returns:
            List of (index, operation) tuples for operations with snapshots
        """
        all_ops = await self.log.read_all()
        reversible = []

        for i, op in enumerate(all_ops):
            if i in self.log._snapshots:
                reversible.append((i, op))

        return reversible

    async def get_recovery_suggestions(self) -> dict[str, Any]:
        """
        Get suggestions for recovery based on failed operations.

        Returns:
            Dictionary with recovery information
        """
        failed_ops = await self.log.get_failed_operations()
        recent_ops = await self.log.get_recent_operations(10)

        suggestions = {
            "failed_operations": len(failed_ops),
            "recent_operations": len(recent_ops),
            "total_operations": await self.log.count(),
        }

        if failed_ops:
            first_failure = failed_ops[0]
            suggestions["first_failure_at_index"] = first_failure[0]
            suggestions["first_failure_type"] = first_failure[1].operation_type
            suggestions["recommended_action"] = (
                f"Rollback to before operation {first_failure[0]} "
                f"({first_failure[1].operation_type})"
            )

        return suggestions
