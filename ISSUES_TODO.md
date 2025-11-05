# Issues Found During Integration & E2E Testing

This document tracks all bugs and issues discovered during testing on the `issues-discovery-1` branch.

**Date**: 2025-11-04
**Branch**: claude/issues-discovery-1-011CUoJkLV5PrCG1UcQq1TMB
**Test Run**: Initial integration and end-to-end test discovery

---

## Critical Issues

### Issue #1: Agent metadata (working_directory) not passed to ClaudeAgentOptions

**Severity**: Medium
**Component**: agent_manager.py
**Test**: `tests/test_agent_manager.py::TestSessionCreation::test_create_session_with_metadata`

**Description**:
When creating an agent session with node metadata (specifically `working_directory`), the metadata is not being passed through to the `ClaudeAgentOptions` constructor. This means agent-specific configurations like working directories are ignored.

**Location**: `src/claude_agent_graph/agent_manager.py` in `create_session()` method

**Expected Behavior**:
```python
await graph.add_node("node_with_wd", "Test prompt", working_directory="/tmp/test")
# Should pass working_directory to ClaudeAgentOptions
```

**Actual Behavior**:
The `working_directory` is stored in node metadata but never passed to `ClaudeAgentOptions`.

**Test Output**:
```
AssertionError: assert None == '/tmp/test'
call_kwargs = {'system_prompt': 'Test prompt', 'model': 'claude-sonnet-4-20250514'}
# working_directory is missing
```

---

### Issue #2: Checkpoint loading doesn't call async add_node/add_edge methods

**Severity**: High
**Component**: checkpoint.py
**Test**: `tests/test_checkpoint.py::TestAgentGraphCheckpointing::test_load_checkpoint_creates_graph`

**Description**:
The checkpoint loading code calls `graph.add_node()` and `graph.add_edge()` without `await`, treating them as synchronous functions when they're actually async. This causes the methods to never execute, leaving the loaded graph empty.

**Location**: `tests/test_checkpoint.py:350-352` (test code issue, indicates implementation problem)

**Expected Behavior**:
```python
await graph1.add_node("node1", "Prompt 1", model="claude-3-sonnet")
await graph1.add_node("node2", "Prompt 2")
await graph1.add_edge("node1", "node2", directed=True, priority="high")
```

**Actual Behavior**:
```python
graph1.add_node("node1", "Prompt 1", model="claude-3-sonnet")  # No await!
# RuntimeWarning: coroutine 'AgentGraph.add_node' was never awaited
```

**Test Output**:
```
AssertionError: assert 0 == 2
where 0 = AgentGraph(name='original_graph', nodes=0, edges=0).node_count

RuntimeWarning: coroutine 'AgentGraph.add_node' was never awaited
```

---

### Issue #3: Inconsistent exception naming

**Severity**: Low
**Component**: exceptions.py, graph.py

**Description**:
There are two similar but different exception types for topology violations:
- `TopologyValidationError` defined in `src/claude_agent_graph/exceptions.py:38`
- `TopologyViolationError` defined in `src/claude_agent_graph/graph.py:34`

This creates confusion about which exception to catch and inconsistency in error handling.

**Expected Behavior**:
One canonical exception name for topology-related errors, used consistently throughout the codebase.

**Recommendation**:
- Choose one name (suggest `TopologyViolationError` as it's more descriptive)
- Move it to exceptions.py
- Update all imports and usages
- Remove duplicate definition

---

### Issue #4: AgentGraph doesn't accept storage_path parameter

**Severity**: Medium
**Component**: graph.py
**Test**: `tests/test_integration.py::TestBasicGraphIntegration::test_create_graph_add_nodes_and_edges`

**Description**:
The `AgentGraph.__init__()` method signature doesn't accept a `storage_path` parameter, but tests and possibly documentation suggest it should. The signature accepts `storage_backend` (a StorageBackend instance), not a simple path string.

**Location**: `src/claude_agent_graph/graph.py:51-61`

**Current Signature**:
```python
def __init__(
    self,
    name: str,
    max_nodes: int = 10000,
    persistence_enabled: bool = True,
    topology_constraint: str | None = None,
    storage_backend: StorageBackend | None = None,  # Not a string!
    auto_save: bool = True,
    auto_save_interval: int = 300,
    checkpoint_dir: Optional[Path | str] = None,
):
```

**Expected Usage** (from tests):
```python
graph = AgentGraph(
    name="integration_test",
    storage_backend="filesystem",  # String - doesn't work!
    storage_path=tmpdir,           # Parameter doesn't exist!
)
```

**Actual Working Usage**:
```python
from claude_agent_graph.backends import FilesystemBackend
graph = AgentGraph(
    name="integration_test",
    storage_backend=FilesystemBackend(base_dir=tmpdir),
)
```

**Test Output**:
```
TypeError: AgentGraph.__init__() got an unexpected keyword argument 'storage_path'
```

**Recommendations**:
1. Add convenience parameter to accept storage_backend as a string ("filesystem", "redis", etc.) and auto-instantiate
2. Add storage_path parameter for simplified filesystem backend configuration
3. Update documentation to reflect actual API

---

### Issue #5: Checkpoint test failures - Multiple checkpoint-related bugs

**Severity**: High
**Component**: checkpoint.py
**Tests**: Multiple in `tests/test_checkpoint.py`

**Failing Tests**:
1. `test_load_checkpoint_creates_graph` - Graph not properly restored (0 nodes instead of 2)
2. `test_load_latest_checkpoint` - Can't load latest checkpoint
3. `test_recovery_on_startup` - Crash recovery not working
4. `test_multiple_checkpoints_loads_latest` - Multiple checkpoints not sorted correctly
5. `test_end_to_end_auto_save_recovery` - Auto-save recovery workflow broken

**Common Root Cause**:
These all appear related to Issue #2 - the async/await problem in checkpoint restoration. The nodes and edges aren't being properly restored when loading checkpoints.

**Impact**:
- Checkpointing system (Epic 7) is fundamentally broken
- Cannot save and restore graph state
- Auto-save doesn't work
- Crash recovery impossible

---

## API Design Issues

### Issue #6: Inconsistent API for specifying storage backend

**Severity**: Low
**Component**: graph.py, Documentation

**Description**:
Users might expect to specify storage backends in multiple ways:
- As a string: `storage_backend="filesystem"`
- As an object: `storage_backend=FilesystemBackend(base_dir="/path")`
- Via separate path param: `storage_path="/path"`

Currently only the object form works. The PRD and CLAUDE.md show string usage in examples.

**Recommendation**:
Add a factory pattern or convenience method to support multiple initialization styles.

---

### Issue #7: Missing validation for storage backend string values

**Severity**: Low
**Component**: graph.py

**Description**:
If we add support for string storage backends (Issue #6), we need validation to catch typos:
```python
graph = AgentGraph(storage_backend="filesytem")  # Typo!
# Should raise clear error, not fail mysteriously later
```

**Recommendation**:
Add enum or validated set of backend names.

---

## Test Issues

### Issue #8: Integration tests use non-existent exception names

**Severity**: Low (Test code only)
**Component**: tests/test_integration.py
**Status**: Fixed in this branch

**Description**:
Integration tests imported:
- `EdgeAlreadyExistsError` (actual name: `DuplicateEdgeError`)
- `TopologyViolationError` (actual name: `TopologyValidationError` in exceptions.py)

**Resolution**:
Fixed import statements in test_integration.py to use correct names.

---

### Issue #9: E2E tests reference non-existent parameters

**Severity**: Low (Test code only)
**Component**: tests/test_e2e.py
**Status**: Not yet fixed

**Description**:
E2E tests use:
- `storage_path` parameter (doesn't exist)
- `storage_backend="filesystem"` string (not supported)

**Resolution Needed**:
Update all E2E tests to use proper `FilesystemBackend` instantiation.

---

## Documentation Issues

### Issue #10: CLAUDE.md examples don't match actual API

**Severity**: Low
**Component**: CLAUDE.md

**Description**:
Documentation shows:
```python
graph = AgentGraph(
    name="my_network",
    storage_backend="filesystem",  # This doesn't work!
    max_nodes=1000
)
```

But actual API requires:
```python
from claude_agent_graph.backends import FilesystemBackend
graph = AgentGraph(
    name="my_network",
    storage_backend=FilesystemBackend(base_dir="./conversations/my_network"),
    max_nodes=1000
)
```

---

## Performance & Scalability Issues

### Issue #11: No performance benchmarks for large graphs

**Severity**: Medium
**Component**: Testing

**Description**:
The CLAUDE.md mentions performance targets:
- Support 10,000+ concurrent agent nodes
- Sub-100ms message routing latency
- 1000+ messages/second throughput

However, there are no tests that validate these targets. The largest test is 100 nodes.

**Recommendation**:
Create performance test suite with:
- 1000 node graph creation test
- 10000 node stress test
- Message throughput benchmark
- Latency measurement tests

---

## Summary Statistics

**Total Issues Found**: 11
**Critical/High Severity**: 3 (Issues #1, #2, #5)
**Medium Severity**: 3 (Issues #4, #6, #11)
**Low Severity**: 5 (Issues #3, #7, #8, #9, #10)

**Components Affected**:
- agent_manager.py: 1 issue
- checkpoint.py: 2 issues
- graph.py: 3 issues
- exceptions.py: 1 issue
- tests/: 2 issues
- documentation: 1 issue
- performance: 1 issue

---

## Recommended Fix Priority

### P0 (Blocking - Fix Immediately)
1. **Issue #2**: Fix async/await in checkpoint loading - blocks Epic 7
2. **Issue #5**: Related checkpoint failures - blocks persistence

### P1 (High Priority)
3. **Issue #1**: Fix metadata passing to agent options
4. **Issue #4**: Add storage_path convenience parameter

### P2 (Medium Priority)
5. **Issue #3**: Consolidate exception naming
6. **Issue #6**: Improve storage backend API ergonomics
7. **Issue #10**: Update documentation to match API

### P3 (Low Priority)
8. **Issue #7**: Add validation for backend names
9. **Issue #8**: Already fixed in tests
10. **Issue #9**: Update E2E test parameters
11. **Issue #11**: Add performance benchmarks

---

## Testing Artifacts

### Test Files Created
- `tests/test_integration.py` - 13 integration test classes, ~30 test methods
- `tests/test_e2e.py` - 7 E2E test classes, ~20 test methods

### Test Execution Results
- Existing unit tests: **412 passed, 6 failed** (98.5% pass rate)
- Integration tests: Not yet passing (API mismatches)
- E2E tests: Not yet passing (API mismatches)

### Key Findings
- Core functionality (nodes, edges, topology) works well
- Checkpoint/persistence system has critical bugs
- API documentation doesn't match implementation
- Need more large-scale performance tests

---

## Next Steps

1. ✅ Create integration and E2E tests
2. ✅ Document all issues in ISSUES_TODO.md
3. ⏳ Create GitHub issues for each bug (blocked - gh CLI not available)
4. ⏳ Fix P0 issues (checkpoint system)
5. ⏳ Fix P1 issues (agent metadata, storage API)
6. ⏳ Update documentation
7. ⏳ Re-run full test suite
8. ⏳ Add performance benchmarks

---

## Notes

- All testing performed on branch: `claude/issues-discovery-1-011CUoJkLV5PrCG1UcQq1TMB`
- Base branch state: 418 unit tests, mostly passing
- No actual bugs fixed yet - this branch is for discovery only
- Fix branch will be created separately per original instructions
