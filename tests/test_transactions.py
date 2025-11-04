"""
Tests for transaction logging and rollback functionality (Epic 5, Stories 5.3.1 & 5.3.2).
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from claude_agent_graph.transactions import (
    Operation,
    RollbackManager,
    StateSnapshot,
    TransactionLog,
)


class TestOperation:
    """Tests for Operation class."""

    def test_operation_creation_with_defaults(self):
        """Test creating operation with default values."""
        op = Operation(operation_type="add_node", node_id="node_1")

        assert op.operation_type == "add_node"
        assert op.node_id == "node_1"
        assert op.timestamp is not None
        assert not op.success
        assert op.error is None
        assert op.data == {}

    def test_operation_creation_with_all_fields(self):
        """Test creating operation with all fields."""
        ts = datetime.now(timezone.utc)
        op = Operation(
            timestamp=ts,
            operation_type="add_edge",
            from_node="a",
            to_node="b",
            success=True,
            data={"directed": True},
        )

        assert op.operation_type == "add_edge"
        assert op.from_node == "a"
        assert op.to_node == "b"
        assert op.success
        assert op.error is None
        assert op.data["directed"] is True

    def test_operation_to_dict(self):
        """Test converting operation to dictionary."""
        op = Operation(operation_type="remove_node", node_id="node_1", success=True)

        op_dict = op.to_dict()

        assert op_dict["operation_type"] == "remove_node"
        assert op_dict["node_id"] == "node_1"
        assert op_dict["success"] is True
        # Timestamp should be ISO format string
        assert isinstance(op_dict["timestamp"], str)
        assert "T" in op_dict["timestamp"]  # ISO format includes T

    def test_operation_from_dict(self):
        """Test creating operation from dictionary."""
        op_dict = {
            "timestamp": "2025-10-25T12:34:56.789000+00:00",
            "operation_type": "add_node",
            "node_id": "node_1",
            "from_node": None,
            "to_node": None,
            "success": True,
            "error": None,
            "data": {"model": "claude-sonnet"},
        }

        op = Operation.from_dict(op_dict)

        assert op.operation_type == "add_node"
        assert op.node_id == "node_1"
        assert op.success
        assert isinstance(op.timestamp, datetime)

    def test_operation_round_trip(self):
        """Test serialization and deserialization round trip."""
        original = Operation(
            operation_type="update_node",
            node_id="node_1",
            success=True,
            data={"prompt": "New prompt"},
        )

        op_dict = original.to_dict()
        restored = Operation.from_dict(op_dict)

        assert restored.operation_type == original.operation_type
        assert restored.node_id == original.node_id
        assert restored.success == original.success
        assert restored.data == original.data


class TestStateSnapshot:
    """Tests for StateSnapshot class."""

    def test_snapshot_creation(self):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            nodes={"node_1": {"id": "node_1", "prompt": "Initial"}},
            edges={"edge_1": {"from": "a", "to": "b"}},
            adjacency={"node_1": ["node_2"]},
        )

        assert len(snapshot.nodes) == 1
        assert len(snapshot.edges) == 1
        assert "node_1" in snapshot.adjacency

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = StateSnapshot(
            nodes={"node_1": {"id": "node_1"}},
            edges={"edge_1": {"from": "a", "to": "b"}},
            adjacency={"a": ["b"]},
        )

        snap_dict = snapshot.to_dict()

        assert "timestamp" in snap_dict
        assert isinstance(snap_dict["timestamp"], str)
        assert snap_dict["nodes"]["node_1"]["id"] == "node_1"
        assert snap_dict["edges"]["edge_1"]["from"] == "a"
        assert snap_dict["adjacency"]["a"] == ["b"]


class TestTransactionLog:
    """Tests for TransactionLog class."""

    async def test_transaction_log_creation(self):
        """Test creating a transaction log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            assert log.max_entries == 1000
            assert await log.count() == 0

    async def test_append_operation(self):
        """Test appending a single operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            op = Operation(operation_type="add_node", node_id="node_1", success=True)
            await log.append(op)

            assert await log.count() == 1
            all_ops = await log.read_all()
            assert all_ops[0].operation_type == "add_node"

    async def test_append_multiple_operations(self):
        """Test appending multiple operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            for i in range(5):
                op = Operation(
                    operation_type="add_node",
                    node_id=f"node_{i}",
                    success=True,
                )
                await log.append(op)

            assert await log.count() == 5

    async def test_append_with_snapshot(self):
        """Test appending operation with state snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            snapshot = StateSnapshot(nodes={"node_1": {}})
            op = Operation(operation_type="add_node", node_id="node_1")

            await log.append(op, snapshot)

            assert await log.count() == 1
            retrieved_snapshot = await log.get_snapshot(0)
            assert retrieved_snapshot is not None
            assert "node_1" in retrieved_snapshot.nodes

    async def test_read_all_operations(self):
        """Test reading all operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            for i in range(3):
                op = Operation(operation_type="add_node", node_id=f"node_{i}")
                await log.append(op)

            all_ops = await log.read_all()

            assert len(all_ops) == 3
            assert all_ops[0].node_id == "node_0"
            assert all_ops[2].node_id == "node_2"

    async def test_read_since_timestamp(self):
        """Test reading operations since a timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            base_time = datetime.now(timezone.utc)

            # Add operation
            op1 = Operation(operation_type="add_node", node_id="node_1")
            op1.timestamp = base_time
            await log.append(op1)

            # Add operation after 1 second
            op2 = Operation(operation_type="add_node", node_id="node_2")
            op2.timestamp = base_time + timedelta(seconds=1)
            await log.append(op2)

            # Read since middle point
            mid_time = base_time + timedelta(milliseconds=500)
            recent = await log.read_since(mid_time)

            assert len(recent) == 1
            assert recent[0].node_id == "node_2"

    async def test_get_operation_by_index(self):
        """Test retrieving operation by index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            op1 = Operation(operation_type="add_node", node_id="node_1")
            op2 = Operation(operation_type="add_node", node_id="node_2")
            await log.append(op1)
            await log.append(op2)

            retrieved = await log.get_operation(1)

            assert retrieved is not None
            assert retrieved.node_id == "node_2"

    async def test_get_operation_out_of_range(self):
        """Test retrieving operation with invalid index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            op = Operation(operation_type="add_node")
            await log.append(op)

            result = await log.get_operation(100)

            assert result is None

    async def test_jsonl_format(self):
        """Test that log file is valid JSONL format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            await log.append(Operation(operation_type="add_node", node_id="node_1"))
            await log.append(Operation(operation_type="add_edge", from_node="a", to_node="b"))

            # Read file and verify JSONL format
            content = log_path.read_text()
            lines = content.strip().split("\n")

            assert len(lines) == 2
            # Each line should be valid JSON
            import json

            for line in lines:
                data = json.loads(line)
                assert "operation_type" in data
                assert "timestamp" in data

    async def test_max_entries_limit(self):
        """Test that log respects max_entries limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path), max_entries=3)

            for i in range(5):
                op = Operation(operation_type="add_node", node_id=f"node_{i}")
                await log.append(op)

            # Should only keep last 3
            assert await log.count() == 3
            all_ops = await log.read_all()
            assert all_ops[0].node_id == "node_2"
            assert all_ops[-1].node_id == "node_4"

    async def test_get_failed_operations(self):
        """Test retrieving failed operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            op1 = Operation(operation_type="add_node", node_id="node_1", success=True)
            op2 = Operation(
                operation_type="add_node", node_id="node_2", success=False, error="Failed"
            )
            op3 = Operation(operation_type="add_node", node_id="node_3", success=True)

            await log.append(op1)
            await log.append(op2)
            await log.append(op3)

            failed = await log.get_failed_operations()

            assert len(failed) == 1
            assert failed[0][1].node_id == "node_2"

    async def test_get_recent_operations(self):
        """Test getting recent N operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            for i in range(10):
                op = Operation(operation_type="add_node", node_id=f"node_{i}")
                await log.append(op)

            recent = await log.get_recent_operations(3)

            assert len(recent) == 3
            assert recent[0].node_id == "node_7"
            assert recent[-1].node_id == "node_9"

    async def test_clear_log(self):
        """Test clearing the transaction log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))

            op = Operation(operation_type="add_node", node_id="node_1")
            await log.append(op)

            assert await log.count() == 1
            assert log_path.exists()

            await log.clear()

            assert await log.count() == 0
            assert not log_path.exists()


class TestRollbackManager:
    """Tests for RollbackManager class."""

    async def test_rollback_manager_creation(self):
        """Test creating a rollback manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            assert rm.log is log

    async def test_rollback_last_with_operations(self):
        """Test rolling back last operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            snapshot = StateSnapshot(nodes={"node_1": {}})
            op = Operation(operation_type="add_node", node_id="node_1")

            await log.append(op, snapshot)

            success = await rm.rollback_last()

            assert success

    async def test_rollback_last_no_operations(self):
        """Test rolling back when no operations exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            success = await rm.rollback_last()

            assert not success

    async def test_rollback_last_no_snapshot(self):
        """Test rolling back operation without snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            op = Operation(operation_type="add_node", node_id="node_1")
            await log.append(op)  # No snapshot

            success = await rm.rollback_last()

            assert not success

    async def test_rollback_to_timestamp(self):
        """Test rolling back to a specific timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            base_time = datetime.now(timezone.utc)

            op1 = Operation(operation_type="add_node", node_id="node_1")
            op1.timestamp = base_time
            await log.append(op1)

            op2 = Operation(operation_type="add_node", node_id="node_2")
            op2.timestamp = base_time + timedelta(seconds=1)
            await log.append(op2)

            mid_time = base_time + timedelta(milliseconds=500)
            success = await rm.rollback_to_timestamp(mid_time)

            assert success

    async def test_rollback_operation_range(self):
        """Test rolling back a range of operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            for i in range(5):
                op = Operation(operation_type="add_node", node_id=f"node_{i}")
                await log.append(op)

            success = await rm.rollback_operation_range(1, 3)

            assert success

    async def test_rollback_operation_range_invalid(self):
        """Test rollback with invalid range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            op = Operation(operation_type="add_node")
            await log.append(op)

            success = await rm.rollback_operation_range(5, 10)

            assert not success

    async def test_get_graph_state_at_index(self):
        """Test retrieving graph state at specific index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            snapshot = StateSnapshot(nodes={"node_1": {"id": "node_1"}})
            op = Operation(operation_type="add_node")

            await log.append(op, snapshot)

            state = await rm.get_graph_state_at_index(0)

            assert state is not None
            assert "node_1" in state.nodes

    async def test_get_graph_state_no_snapshot(self):
        """Test retrieving state when no snapshot available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            op = Operation(operation_type="add_node")
            await log.append(op)

            state = await rm.get_graph_state_at_index(0)

            assert state is None

    async def test_list_reversible_operations(self):
        """Test listing operations that can be reversed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            # Add with snapshot
            snapshot1 = StateSnapshot(nodes={"node_1": {}})
            await log.append(Operation(operation_type="add_node"), snapshot1)

            # Add without snapshot
            await log.append(Operation(operation_type="add_node"))

            # Add with snapshot
            snapshot3 = StateSnapshot(nodes={"node_3": {}})
            await log.append(Operation(operation_type="add_node"), snapshot3)

            reversible = await rm.list_reversible_operations()

            assert len(reversible) == 2
            assert reversible[0][0] == 0
            assert reversible[1][0] == 2

    async def test_get_recovery_suggestions_with_failures(self):
        """Test getting recovery suggestions when failures exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            op1 = Operation(operation_type="add_node", success=True)
            op2 = Operation(operation_type="add_edge", success=False, error="Connection failed")

            await log.append(op1)
            await log.append(op2)

            suggestions = await rm.get_recovery_suggestions()

            assert suggestions["failed_operations"] == 1
            assert suggestions["first_failure_at_index"] == 1
            assert suggestions["first_failure_type"] == "add_edge"
            assert "recommended_action" in suggestions

    async def test_get_recovery_suggestions_no_failures(self):
        """Test recovery suggestions when no failures exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "transactions.jsonl"
            log = TransactionLog(str(log_path))
            rm = RollbackManager(log)

            for _i in range(3):
                op = Operation(operation_type="add_node", success=True)
                await log.append(op)

            suggestions = await rm.get_recovery_suggestions()

            assert suggestions["failed_operations"] == 0
            assert "recommended_action" not in suggestions
