# Epic 5 Implementation Plan: Dynamic Graph Operations

**Status:** Ready for Implementation
**Date:** October 2025
**Estimated Effort:** 5-6 days
**Dependencies:** Epic 4 (Agent Integration)

## Overview

Epic 5 enables runtime modification of graph structure—adding, removing, and updating nodes and edges after the graph has been created. This transforms the graph from a static structure into a dynamic system that can adapt to changing requirements.

**Key Capabilities:**
- Add/remove nodes at runtime with proper cleanup
- Add/remove edges at runtime with conversation file archival
- Update node and edge properties without rebuilding
- Transaction logging for all modifications
- Rollback capability for failed operations
- Thread-safe concurrent modifications

## Architecture Decisions

### 1. Cascade Deletion Pattern
**Decision:** Support both cascade and non-cascade removal via parameter.

**Rationale:**
- `cascade=True` (default): Automatically remove associated edges, more convenient
- `cascade=False`: Raise error if edges exist, safer for critical systems
- Aligns with database deletion patterns

### 2. Conversation File Archival
**Decision:** Archive conversation files instead of deleting when edges are removed.

**Rationale:**
- Preserves audit trail and conversation history
- Enables recovery and investigation of past interactions
- Aligns with data retention policies
- Archive location: `archived/{edge_id}_{timestamp}.jsonl`

### 3. Transaction Logging Strategy
**Decision:** Append-only transaction log in JSONL format with state snapshots.

**Rationale:**
- JSONL format matches conversation file approach
- Append-only ensures no data loss
- State snapshots enable efficient rollback
- Enables replay and debugging

### 4. Prompt Update Lifecycle
**Decision:** Use lazy update pattern (mark dirty, recompute on next activation) inherited from Epic 4.

**Rationale:**
- No disruptive mid-session restarts
- Consistent with Epic 4 design
- Allows batch prompt updates
- Cleaner API without side-effects

### 5. Concurrency Model
**Decision:** Use asyncio.Lock for serialized graph modifications.

**Rationale:**
- Simple, proven pattern in async Python
- Prevents race conditions in graph structure
- Matches async patterns already in codebase
- Easy to understand and maintain

## Feature 5.1: Runtime Node Operations

### Story 5.1.1: Runtime Node Addition (ALREADY IMPLEMENTED)
Note: This is implemented as part of `add_node()` in Epic 2. No additional work needed.

### Story 5.1.2: Node Removal

**Goal:** Enable safe node removal with cascade option and conversation file archival.

**Files to Create:**
- No new files (modifications only)

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~80 lines)
- `tests/test_graph.py` (~150 lines)

**Implementation Details:**

```python
async def remove_node(
    self,
    node_id: str,
    cascade: bool = True,
) -> None:
    """
    Remove a node from the graph.

    Args:
        node_id: ID of the node to remove
        cascade: If True, remove associated edges automatically.
                If False, raise error if edges exist.

    Raises:
        NodeNotFoundError: If node doesn't exist
        AgentGraphError: If edges exist and cascade=False

    Example:
        >>> await graph.remove_node("worker_1", cascade=True)
    """
```

**Implementation Steps:**

1. **Validate node exists**
   - Raise `NodeNotFoundError` if not found

2. **Check for edges if cascade=False**
   - Get all connected edges (incoming and outgoing)
   - If any exist and cascade=False, raise error with edge list

3. **Stop agent session**
   - Call `await self._agent_manager.stop_agent(node_id)`
   - Handle already-stopped gracefully

4. **Handle cascade deletion of edges**
   - If cascade=True, get all connected edges
   - For each edge, archive conversation file
   - Remove from internal structures

5. **Update control relationships**
   - For edges TO this node (incoming): mark superiors' prompts dirty
   - For edges FROM this node (outgoing): mark subordinates' prompts dirty
   - Both will recompute on next activation

6. **Remove node from graph**
   - Delete from `self._nodes` dict
   - Delete from `self._nx_graph`
   - Clean up `self._adjacency` entries

7. **Logging and events**
   - Log removal with cascade status
   - Note: Event system placeholder for future

**Edge Cases:**
- Node is running: Stop gracefully before removal
- Node has multiple controllers/subordinates: Update all affected prompts
- Node is last node in graph: Removal should succeed
- Removal during message send: Lock ensures atomicity

**Acceptance Criteria:**
- ✅ Can remove node with cascade=True (edges auto-removed)
- ✅ Raises error on cascade=False if edges exist
- ✅ Agent session stopped gracefully
- ✅ Conversation files archived to `archived/` with timestamp
- ✅ Affected prompts marked dirty
- ✅ Node removed from all internal structures
- ✅ NetworkX graph updated
- ✅ Tests cover all scenarios

---

### Story 5.1.3: Node Property Updates

**Goal:** Update node properties at runtime without rebuilding.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~60 lines)
- `tests/test_graph.py` (~120 lines)

**Implementation Details:**

```python
def update_node(
    self,
    node_id: str,
    system_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Update node properties.

    Args:
        node_id: ID of the node to update
        system_prompt: New system prompt (optional)
        metadata: New metadata dict (optional, merged with existing)

    Raises:
        NodeNotFoundError: If node doesn't exist

    Example:
        >>> graph.update_node("worker_1",
        ...     system_prompt="New role description",
        ...     metadata={"priority": "high"})
    """
```

**Implementation Steps:**

1. **Validate node exists**
   - Raise `NodeNotFoundError` if not found

2. **Update system_prompt if provided**
   - Store original prompt if not already stored
   - Set new prompt
   - Mark `prompt_dirty = True` if running
   - Will recompute effective prompt on next activation

3. **Update metadata if provided**
   - Merge new metadata with existing (not replace)
   - Preserve existing keys not in new metadata

4. **Mark subordinates dirty if prompt changed**
   - Call `self._mark_subordinates_dirty(node_id)`
   - Ensures subordinates get updated control info on reactivation

5. **Logging**
   - Log what was updated

**Design Notes:**
- Synchronous operation (not async) - just updates metadata
- No agent restart required (lazy update pattern)
- Prompt changes only take effect on next agent activation
- Metadata updates are immediate (no side-effects)

**Acceptance Criteria:**
- ✅ Can update system_prompt only
- ✅ Can update metadata only
- ✅ Can update both in one call
- ✅ Metadata is merged, not replaced
- ✅ Prompt updates trigger dirty flag
- ✅ Subordinate prompts marked dirty
- ✅ Works for stopped and running agents
- ✅ NodeNotFoundError on invalid node

---

## Feature 5.2: Runtime Edge Operations

### Story 5.2.1: Runtime Edge Addition (ALREADY IMPLEMENTED)
Note: Implemented as part of `add_edge()` in Epic 2. No additional work needed.

### Story 5.2.2: Edge Removal

**Goal:** Remove edges with conversation file archival and control relationship updates.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~70 lines)
- `tests/test_graph.py` (~140 lines)

**Implementation Details:**

```python
async def remove_edge(
    self,
    from_node: str,
    to_node: str,
) -> None:
    """
    Remove an edge between two nodes.

    Args:
        from_node: Source node ID
        to_node: Target node ID

    Raises:
        NodeNotFoundError: If either node doesn't exist
        EdgeNotFoundError: If edge doesn't exist

    Example:
        >>> await graph.remove_edge("supervisor", "worker_1")
    """
```

**Implementation Steps:**

1. **Find edge (directed or undirected)**
   - Try directed edge first: `Edge.generate_edge_id(from_node, to_node, True)`
   - Try undirected: `Edge.generate_edge_id(from_node, to_node, False)`
   - If not found, try reverse undirected
   - Raise `EdgeNotFoundError` if not found

2. **Archive conversation file**
   - Get edge's conversation file path
   - Generate archive path with timestamp: `archived/{edge_id}_{timestamp}.jsonl`
   - Asynchronously move/copy file to archive location
   - Create `archived/` directory if needed

3. **Update control relationships**
   - If directed edge:
     - Mark to_node's prompt dirty (control relationship changed)
     - This subordinate will recompute on next activation
     - They'll no longer see the removed controller in their prompt

4. **Remove edge from structures**
   - Delete from `self._edges[edge_id]`
   - Remove from `self._adjacency[from_node]`
   - If undirected, also remove from `self._adjacency[to_node]`
   - Remove from `self._nx_graph`

5. **Logging**
   - Log edge removal with archive location

**Edge Cases:**
- Edge doesn't exist: Raise EdgeNotFoundError
- Conversation file doesn't exist: Still succeed in removing edge
- Undirected edge: Handle both directions in adjacency cleanup
- Multiple edges involved: Each removal is independent

**Acceptance Criteria:**
- ✅ Can remove directed edges
- ✅ Can remove undirected edges
- ✅ Conversation file archived with timestamp
- ✅ Control relationships updated (prompts marked dirty)
- ✅ Edge removed from all internal structures
- ✅ NetworkX graph updated
- ✅ EdgeNotFoundError on invalid edge
- ✅ Tests verify archive and cleanup

---

### Story 5.2.3: Edge Property Updates

**Goal:** Update edge properties (custom metadata).

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~50 lines)
- `tests/test_graph.py` (~100 lines)

**Implementation Details:**

```python
def update_edge(
    self,
    from_node: str,
    to_node: str,
    **properties: Any,
) -> None:
    """
    Update edge properties.

    Args:
        from_node: Source node ID
        to_node: Target node ID
        **properties: Properties to merge into edge.properties

    Raises:
        NodeNotFoundError: If either node doesn't exist
        EdgeNotFoundError: If edge doesn't exist

    Example:
        >>> graph.update_edge("cfo", "analyst",
        ...     control_type="oversight",
        ...     priority="high")
    """
```

**Implementation Steps:**

1. **Find edge**
   - Use same lookup as remove_edge
   - Raise EdgeNotFoundError if not found

2. **Check for control_type changes**
   - If `control_type` in properties and it differs from current
   - Mark to_node's prompt dirty (they need to recompute with new type)

3. **Merge properties**
   - Update `edge.properties` dict with new properties
   - Use dict.update() to merge, not replace

4. **Logging**
   - Log property updates

**Design Notes:**
- Synchronous operation
- Merges, doesn't replace properties
- control_type changes trigger prompt updates
- Other properties are just metadata (no special handling)

**Acceptance Criteria:**
- ✅ Can update edge properties
- ✅ Properties are merged, not replaced
- ✅ control_type changes trigger prompt dirty
- ✅ Works with both directed and undirected edges
- ✅ EdgeNotFoundError on invalid edge

---

## Feature 5.3: Transaction Safety

### Story 5.3.1: Operation Logging

**Goal:** Log all graph modifications for replay and debugging.

**Files to Create:**
- `src/claude_agent_graph/transactions.py` (~200 lines)
- `tests/test_transactions.py` (~180 lines)

**Implementation Details:**

```python
# src/claude_agent_graph/transactions.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json
from pathlib import Path

@dataclass
class Operation:
    """Records a single graph operation."""
    timestamp: datetime
    operation_type: str  # "add_node", "remove_node", "add_edge", etc.
    data: dict[str, Any]  # Operation-specific data
    success: bool = False
    error: str | None = None

class TransactionLog:
    """Manages append-only transaction log in JSONL format."""

    def __init__(self, log_path: str):
        """Initialize transaction log."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, operation: Operation) -> None:
        """Append operation to log."""
        # Write to JSONL file

    async def read_all(self) -> list[Operation]:
        """Read all operations from log."""
        # Read and parse JSONL

    async def read_since(self, timestamp: datetime) -> list[Operation]:
        """Read operations since a timestamp."""
        # Filtered read
```

**Integration with AgentGraph:**
```python
class AgentGraph:
    def __init__(self, ...):
        self._transaction_log = TransactionLog(
            f"./transactions/{self.name}/transactions.jsonl"
        )
```

**Usage Pattern:**
```python
async def remove_node(self, node_id: str, cascade: bool = True):
    operation = Operation(
        timestamp=datetime.now(timezone.utc),
        operation_type="remove_node",
        data={
            "node_id": node_id,
            "cascade": cascade,
            "edges_removed": [...]
        }
    )

    try:
        # Perform removal
        await self._transaction_log.append(operation)
        operation.success = True
    except Exception as e:
        operation.error = str(e)
        operation.success = False
        await self._transaction_log.append(operation)
        raise
```

**Log Format (JSONL):**
```json
{"timestamp": "2025-10-25T12:34:56.789Z", "operation_type": "add_node", "data": {"node_id": "worker_1", "model": "claude-sonnet-4-20250514"}, "success": true, "error": null}
{"timestamp": "2025-10-25T12:34:57.123Z", "operation_type": "add_edge", "data": {"from_node": "supervisor", "to_node": "worker_1"}, "success": true, "error": null}
{"timestamp": "2025-10-25T12:34:58.456Z", "operation_type": "remove_node", "data": {"node_id": "worker_1"}, "success": true, "error": null}
```

**Acceptance Criteria:**
- ✅ TransactionLog class implemented
- ✅ Operations logged as JSONL with timestamp
- ✅ All graph modifications logged
- ✅ Log is append-only (no overwrites)
- ✅ Can read all operations
- ✅ Can read operations since timestamp
- ✅ Success/error status tracked
- ✅ Handles concurrent writes safely

---

### Story 5.3.2: Rollback Support

**Goal:** Enable undoing failed operations via transaction log.

**Files to Modify:**
- `src/claude_agent_graph/transactions.py` (~100 lines)
- `src/claude_agent_graph/graph.py` (integration points)
- `tests/test_transactions.py` (~150 lines)

**Implementation Details:**

```python
@dataclass
class StateSnapshot:
    """Snapshot of graph state before an operation."""
    timestamp: datetime
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    adjacency: dict[str, list[str]]

    async def restore(self, graph: "AgentGraph") -> None:
        """Restore graph to this snapshot state."""

class TransactionLog:

    async def rollback_last_operation(self) -> None:
        """Undo the last operation."""

    async def rollback_to_timestamp(self, timestamp: datetime) -> None:
        """Rollback all operations after a timestamp."""

    async def rollback_operation(self, operation_id: int) -> None:
        """Rollback a specific operation (and all dependent ops)."""
```

**Usage Pattern:**
```python
async def remove_node_with_rollback(graph: AgentGraph, node_id: str):
    """Remove node with automatic rollback on error."""
    try:
        await graph.remove_node(node_id)
    except Exception as e:
        logger.error(f"Removal failed: {e}")
        await graph._transaction_log.rollback_last_operation()
        logger.info("Rolled back removal")
        raise
```

**Rollback Mechanism:**

1. **State Snapshots:** Before each modification, save state
2. **Operation Tracking:** Log what was changed
3. **Reverse Operations:** For each operation in reverse:
   - Remove newly added nodes/edges
   - Restore deleted nodes/edges from snapshot
   - Restore edge properties
   - Restore node properties

**Performance Consideration:**
- Snapshots stored in memory (not persisted)
- Limit history to last N operations (e.g., 100)
- Allow optional persistence for crash recovery (Phase 4+)

**Acceptance Criteria:**
- ✅ StateSnapshot captures full graph state
- ✅ Rollback reverses operations in correct order
- ✅ Restored nodes/edges have correct properties
- ✅ Control relationships restored correctly
- ✅ Can rollback to any point
- ✅ Automatic rollback on exception works
- ✅ Tests verify state correctness after rollback

---

## Testing Strategy

### Unit Tests for Removal Operations

**test_graph.py additions (34 new tests):**

**Node Removal (10 tests):**
```python
async def test_remove_node_basic()
async def test_remove_node_with_cascade_true()
async def test_remove_node_with_cascade_false_no_edges()
async def test_remove_node_with_cascade_false_has_edges_raises()
async def test_remove_node_stops_running_agent()
async def test_remove_node_not_found_raises()
async def test_remove_node_updates_prompts()
async def test_remove_isolated_node()
async def test_remove_node_updates_topology()
async def test_remove_node_with_multiple_edges()
```

**Node Updates (8 tests):**
```python
async def test_update_node_system_prompt()
async def test_update_node_metadata()
async def test_update_node_both()
async def test_update_node_metadata_merged()
async def test_update_node_not_found_raises()
async def test_update_node_marks_dirty()
async def test_update_node_marks_subordinates_dirty()
async def test_update_node_running_agent()
```

**Edge Removal (9 tests):**
```python
async def test_remove_edge_directed()
async def test_remove_edge_undirected()
async def test_remove_edge_archives_conversation()
async def test_remove_edge_updates_control_relationships()
async def test_remove_edge_not_found_raises()
async def test_remove_edge_updates_topology()
async def test_remove_edge_with_multiple_edges()
async def test_remove_last_edge_to_node()
async def test_remove_edge_marks_subordinate_dirty()
```

**Edge Updates (7 tests):**
```python
async def test_update_edge_properties()
async def test_update_edge_properties_merged()
async def test_update_edge_control_type_triggers_dirty()
async def test_update_edge_not_found_raises()
async def test_update_edge_directed_vs_undirected()
async def test_update_edge_preserves_other_properties()
async def test_update_edge_with_multiple_edges()
```

### Unit Tests for Transactions

**test_transactions.py (38 new tests):**

**Operation Logging (8 tests):**
```python
async def test_append_operation()
async def test_append_multiple_operations()
async def test_read_all_operations()
async def test_read_operations_since()
async def test_operation_timestamp_order()
async def test_jsonl_format_valid()
async def test_concurrent_appends()
async def test_large_operation_data()
```

**Rollback (8 tests):**
```python
async def test_rollback_last_operation()
async def test_rollback_to_timestamp()
async def test_rollback_multiple_operations()
async def test_state_snapshot_restore()
async def test_rollback_with_concurrent_ops()
async def test_rollback_preserves_other_state()
async def test_rollback_updates_nx_graph()
async def test_rollback_restores_control_relationships()
```

### Integration Tests

**Multi-operation workflows (8 tests):**
```python
async def test_add_remove_add_cycle()
async def test_dynamic_hierarchy_changes()
async def test_concurrent_removals_no_interference()
async def test_removal_during_messaging()
async def test_update_then_verify_new_properties()
async def test_cascade_removal_updates_all_affected()
async def test_rollback_partial_operation()
async def test_transaction_log_replay()
```

### Concurrency Tests (10 tests)

```python
async def test_concurrent_node_removals()
async def test_concurrent_edge_operations()
async def test_concurrent_updates_no_conflicts()
async def test_add_remove_same_node_sequence()
async def test_high_concurrency_stress_test()
async def test_lock_prevents_races()
async def test_concurrent_rollbacks()
async def test_transaction_log_under_concurrent_load()
async def test_no_deadlocks()
async def test_operation_order_preserved()
```

## Files Summary

### New Files
1. **EPIC_5_IMPLEMENTATION.md** - This document (~800 lines)
2. **src/claude_agent_graph/transactions.py** (~300 lines)
   - Operation dataclass
   - TransactionLog class
   - StateSnapshot class
   - Rollback logic

3. **tests/test_transactions.py** (~330 lines)
   - Operation logging tests (8 tests)
   - Rollback tests (8 tests)
   - Concurrency tests (12+ tests)

### Modified Files
1. **src/claude_agent_graph/graph.py** (+350 lines)
   - Add asyncio.Lock for concurrency
   - `remove_node()` method (~80 lines)
   - `update_node()` method (~60 lines)
   - `remove_edge()` method (~70 lines)
   - `update_edge()` method (~50 lines)
   - Integration with transaction logging

2. **src/claude_agent_graph/storage.py** (+50 lines)
   - Add `archive_conversation()` method
   - Support for archived/ directory creation

3. **tests/test_graph.py** (+510 lines, 34 new tests)
   - Node removal tests (10)
   - Node update tests (8)
   - Edge removal tests (9)
   - Edge update tests (7)

## Success Criteria

### Code Quality
- ✅ >85% test coverage for Epic 5 code
- ✅ All 261 existing tests still pass
- ✅ 72+ new tests pass
- ✅ Type hints on all methods
- ✅ Comprehensive docstrings
- ✅ No ruff/black/mypy issues

### Functionality
- ✅ All 6 user stories fully implemented
- ✅ Nodes removable with/without cascade
- ✅ Edges removable with file archival
- ✅ Properties updateable for nodes and edges
- ✅ Conversation files preserved in archive
- ✅ Control relationships updated correctly
- ✅ Transaction logging captures all operations
- ✅ Rollback successfully undoes changes

### Performance
- ✅ remove_node() < 100ms for <10 edges
- ✅ update_node() < 50ms
- ✅ remove_edge() < 100ms
- ✅ Concurrent operations supported (10+)
- ✅ No memory leaks on repeated ops
- ✅ Transaction log writes < 10ms

### Safety
- ✅ Thread-safe with asyncio.Lock
- ✅ No race conditions in tests
- ✅ Rollback verified in tests
- ✅ Conversation files archived, not deleted
- ✅ Control relationships maintained invariants

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Race conditions in concurrent ops | Medium | High | asyncio.Lock serializes modifications, extensive testing |
| Partial failure during cascade | Medium | High | Transaction logging + rollback for recovery |
| File archiving fails | Low | Medium | Graceful failure, log error, continue edge removal |
| Prompt injection bugs on update | Low | Medium | Mark dirty instead of immediate update, Epic 4 pattern proven |
| Transaction log becomes large | Medium | Low | Rotation strategy, archive old logs (Phase 4+) |
| Rollback undoes too much | Low | High | Careful state snapshot testing, operation-by-operation rollback |

## References

- **IMPLEMENTATION_PLAN.md** - Main project plan (Epic 5 section: lines 570-716)
- **EPIC_4_IMPLEMENTATION.md** - Related architecture decisions
- **CLAUDE.md** - Project guidelines and patterns

## Implementation Timeline

**Phase 1: Core Removal Operations (Days 1-2)**
- Story 5.1.2: Node Removal
- Story 5.2.2: Edge Removal
- Testing and integration

**Phase 2: Update Operations (Day 3)**
- Story 5.1.3: Node Updates
- Story 5.2.3: Edge Updates
- Testing

**Phase 3: Transaction Safety (Days 4-5)**
- Story 5.3.1: Operation Logging
- Story 5.3.2: Rollback Support
- Comprehensive testing

**Phase 4: Polish & Final Testing (Day 5-6)**
- Concurrency stress testing
- Documentation review
- Performance optimization

**Estimated Total:** 5-6 days of implementation

---

**Document Status:** Ready for Implementation
**Last Updated:** October 2025
