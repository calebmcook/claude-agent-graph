# Epic 4 Implementation Plan: Agent Management & Lifecycle

**Status:** Planning
**Date:** October 2025
**Estimated Effort:** 4-5 days
**Dependencies:** Epic 3 (State Management)

## Overview

Epic 4 transforms the graph from a data structure simulator into a living, breathing network of Claude AI agents. This epic integrates the claude-agent-sdk to create and manage actual agent sessions, implement lifecycle control, and establish hierarchical control relationships through system prompt injection.

**Key Transformation:**
- Abstract graph structure (Epics 1-3) → Executable agent network (Epic 4)
- Data models (Nodes, Edges) → Live ClaudeSDKClient sessions
- Static graph → Dynamic agent coordination

## Architecture Decisions

### 1. Lazy Agent Initialization
**Decision:** Agents activate on first use, not on node creation.

**Rationale:**
- Large graphs (1000+ nodes) can be built without immediate API calls
- Balances simplicity with resource efficiency
- Users control when agents "wake up"

**Implementation:**
- Nodes start in `INITIALIZING` status
- First `send_message()` or explicit `start_agent()` call triggers activation
- Session created and stored in AgentSessionManager

### 2. SDK Dependency
**Decision:** Add `claude-agent-sdk>=1.0.0` to pyproject.toml

**Rationale:**
- Verified SDK exists: anthropics/claude-agent-sdk-python
- Available on PyPI (or will be)
- No fallback mocking needed; real SDK in dependencies

**Version Strategy:**
```toml
dependencies = [
    "claude-agent-sdk>=1.0.0",
    ...existing dependencies...
]
```

### 3. Multiple Controllers per Node
**Decision:** Subordinates can have multiple incoming directed edges.

**Rationale:**
- Supports complex authority structures (matrix organizations, multi-supervisor)
- More flexible than single-parent constraint
- System prompt lists all controllers clearly

**Example:** Financial analyst node reports to both CFO and Chief Risk Officer

### 4. Lazy Prompt Updates
**Decision:** When edges change, mark prompt dirty; recompute on next activation.

**Rationale:**
- No disruptive mid-session restarts
- Avoids cascading failures
- Cleaner API (no restart side-effects)

**Trade-off:** Brief period where running agent has stale control info (acceptable)

## Feature 4.1: Agent Session Management

### Story 4.1.1: Initialize ClaudeSDKClient for Nodes

**Goal:** Create and configure agent sessions from Node specifications.

**Files to Create:**
- `src/claude_agent_graph/agent_manager.py` (~350 lines)

**Files to Modify:**
- `pyproject.toml` - Add claude-agent-sdk dependency
- `src/claude_agent_graph/models.py` - Add session fields to Node
- `src/claude_agent_graph/graph.py` - Integrate AgentSessionManager

**Implementation Details:**

```python
# src/claude_agent_graph/agent_manager.py

from typing import Optional
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from .models import Node

class AgentSessionManager:
    """Manages ClaudeSDKClient instances for agent nodes."""

    def __init__(self, graph):
        self._graph = graph
        self._sessions: dict[str, ClaudeSDKClient] = {}
        self._contexts: dict[str, Any] = {}  # active async contexts

    async def create_session(self, node_id: str) -> ClaudeSDKClient:
        """
        Create a ClaudeSDKClient for a node.

        Args:
            node_id: ID of the node

        Returns:
            Initialized ClaudeSDKClient

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        node = self._graph.get_node(node_id)

        # Use effective prompt if available, otherwise original
        prompt = node.effective_system_prompt or node.system_prompt

        options = ClaudeAgentOptions(
            system_prompt=prompt,
            model=node.model,
            # Allow configuration via node metadata
            working_directory=node.metadata.get('working_directory'),
        )

        client = ClaudeSDKClient(options=options)
        self._sessions[node_id] = client

        logger.info(f"Created session for agent '{node_id}'")
        return client

    async def get_session(self, node_id: str) -> Optional[ClaudeSDKClient]:
        """Get existing or create new session for a node."""
        if node_id not in self._sessions:
            await self.create_session(node_id)
        return self._sessions.get(node_id)
```

**Node Model Updates:**
```python
class Node(BaseModel):
    # ... existing fields ...

    # Agent session management
    original_system_prompt: Optional[str] = None  # Backup of original
    effective_system_prompt: Optional[str] = None  # With injected info
    prompt_dirty: bool = False  # Needs recomputation
    agent_session: Optional[Any] = None  # Reference to ClaudeSDKClient
```

**Acceptance Criteria:**
- ✅ claude-agent-sdk added to dependencies
- ✅ AgentSessionManager class implemented
- ✅ create_session() creates ClaudeSDKClient with node config
- ✅ Sessions cached in _sessions dict
- ✅ Unit tests pass with mocked SDK

---

### Story 4.1.2: Implement Agent Session Lifecycle

**Goal:** Manage agent startup, shutdown, and restart with proper state transitions.

**Files to Modify:**
- `src/claude_agent_graph/agent_manager.py` - Add lifecycle methods
- `src/claude_agent_graph/graph.py` - Expose methods, integrate manager

**Implementation Details:**

```python
# In AgentSessionManager

async def start_agent(self, node_id: str) -> None:
    """
    Start an agent session.

    Enters the async context manager for the ClaudeSDKClient.
    Updates Node.status to ACTIVE.
    """
    if node_id in self._contexts:
        raise AgentGraphError(f"Agent '{node_id}' is already running")

    node = self._graph.get_node(node_id)
    session = await self.get_session(node_id)

    try:
        context = await session.__aenter__()
        self._contexts[node_id] = context
        node.status = NodeStatus.ACTIVE
        logger.info(f"Started agent '{node_id}'")
    except Exception as e:
        node.status = NodeStatus.ERROR
        node.metadata['startup_error'] = str(e)
        logger.error(f"Failed to start agent '{node_id}': {e}")
        raise

async def stop_agent(self, node_id: str) -> None:
    """
    Stop an agent session gracefully.

    Exits the async context and updates status.
    """
    if node_id not in self._contexts:
        logger.warning(f"Agent '{node_id}' is not running")
        return

    context = self._contexts.pop(node_id)
    node = self._graph.get_node(node_id)

    try:
        await context.__aexit__(None, None, None)
        node.status = NodeStatus.STOPPED
        logger.info(f"Stopped agent '{node_id}'")
    except Exception as e:
        logger.error(f"Error stopping agent '{node_id}': {e}")
        raise

async def restart_agent(self, node_id: str) -> None:
    """Stop and restart an agent with fresh context."""
    await self.stop_agent(node_id)
    await asyncio.sleep(0.1)  # Brief pause
    await self.start_agent(node_id)

async def stop_all(self) -> None:
    """Stop all running agents (for cleanup)."""
    node_ids = list(self._contexts.keys())
    for node_id in node_ids:
        try:
            await self.stop_agent(node_id)
        except Exception as e:
            logger.error(f"Error stopping agent '{node_id}': {e}")
```

**Node Status Transitions:**
```
INITIALIZING → ACTIVE (on start_agent or first send_message)
            ↓
          STOPPED (on stop_agent)

INITIALIZING → ERROR (on initialization failure)
            ↓
        ACTIVE (on successful restart_agent)
```

**AgentGraph Integration:**
```python
class AgentGraph:
    def __init__(self, ...):
        # ... existing init ...
        self._agent_manager = AgentSessionManager(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all agents on graph exit."""
        if hasattr(self, '_agent_manager'):
            await self._agent_manager.stop_all()

    async def start_agent(self, node_id: str) -> None:
        """Start an agent session."""
        await self._agent_manager.start_agent(node_id)

    async def stop_agent(self, node_id: str) -> None:
        """Stop an agent session."""
        await self._agent_manager.stop_agent(node_id)

    async def restart_agent(self, node_id: str) -> None:
        """Restart an agent session."""
        await self._agent_manager.restart_agent(node_id)

    def get_agent_status(self, node_id: str) -> dict[str, Any]:
        """Get agent status including error details."""
        node = self.get_node(node_id)
        return {
            'node_id': node_id,
            'status': node.status.value,
            'model': node.model,
            'is_running': node_id in self._agent_manager._contexts,
            'last_error': node.metadata.get('startup_error'),
            'error_count': node.metadata.get('error_count', 0),
        }
```

**Acceptance Criteria:**
- ✅ start_agent() enters async context, status → ACTIVE
- ✅ stop_agent() exits async context, status → STOPPED
- ✅ restart_agent() performs stop + start sequence
- ✅ AgentGraph supports async context manager
- ✅ Cleanup occurs on graph exit
- ✅ Status transitions correct (INITIALIZING → ACTIVE → STOPPED)
- ✅ Integration tests pass

---

### Story 4.1.3: Handle Agent Errors and Recovery

**Goal:** Implement robust error handling with retry logic and failure state management.

**Files to Modify:**
- `src/claude_agent_graph/agent_manager.py` - Add error recovery

**Implementation Details:**

```python
# In AgentSessionManager

class AgentErrorRecovery:
    """Encapsulates retry and recovery logic."""

    MAX_RETRIES = 3
    RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff

    async def with_retry(self, func, node_id: str, *args, **kwargs):
        """
        Execute a function with retry logic.

        Updates Node.status to ERROR if all retries exhausted.
        """
        node = self._graph.get_node(node_id)

        for attempt in range(self.MAX_RETRIES):
            try:
                result = await func(*args, **kwargs)
                # Success - clear error state
                node.metadata.pop('error_count', None)
                node.metadata.pop('last_error', None)
                return result
            except Exception as e:
                logger.warning(
                    f"Agent '{node_id}' error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}"
                )

                if attempt == self.MAX_RETRIES - 1:
                    # Final failure
                    node.status = NodeStatus.ERROR
                    node.metadata['last_error'] = str(e)
                    node.metadata['error_count'] = node.metadata.get('error_count', 0) + 1
                    logger.error(
                        f"Agent '{node_id}' failed after {self.MAX_RETRIES} retries. "
                        f"Total failures: {node.metadata['error_count']}"
                    )
                    raise AgentGraphError(f"Agent '{node_id}' failed: {e}") from e

                # Wait before retry with exponential backoff
                delay = self.RETRY_DELAYS[attempt]
                logger.info(f"Retrying agent '{node_id}' in {delay}s...")
                await asyncio.sleep(delay)

async def start_agent_with_recovery(self, node_id: str) -> None:
    """Start agent with error recovery."""
    async def _start():
        await self.start_agent(node_id)

    await self.with_retry(_start, node_id)
```

**Error Scenarios Handled:**
1. Network errors (transient) → Retry with backoff
2. API authentication failures (permanent) → ERROR status
3. Rate limiting (transient) → Retry with backoff
4. Agent initialization timeout (transient) → Retry
5. Invalid system prompt (permanent) → ERROR status

**get_agent_status() Enhancement:**
```python
def get_agent_status(self, node_id: str) -> dict[str, Any]:
    """Get comprehensive agent status."""
    node = self.get_node(node_id)
    return {
        'node_id': node_id,
        'status': node.status.value,
        'model': node.model,
        'is_running': node_id in self._agent_manager._contexts,
        'last_error': node.metadata.get('last_error'),
        'error_count': node.metadata.get('error_count', 0),
        'created_at': node.created_at.isoformat(),
    }
```

**Acceptance Criteria:**
- ✅ SDK errors caught and logged with context
- ✅ Retry logic with exponential backoff (1s, 2s, 4s)
- ✅ Node.status = ERROR on final failure
- ✅ Error details stored in node metadata
- ✅ get_agent_status() returns complete status info
- ✅ Tests simulate failures and verify recovery
- ✅ >90% of error scenarios handled

---

## Feature 4.2: Control Relationships

### Story 4.2.1: Implement System Prompt Injection for Control

**Goal:** Establish authority hierarchies by injecting controller information into subordinate prompts.

**Files to Modify:**
- `src/claude_agent_graph/models.py` - Add prompt fields to Node
- `src/claude_agent_graph/graph.py` - Implement prompt computation and injection

**Implementation Details:**

```python
# In graph.py

def _compute_effective_prompt(self, node_id: str) -> str:
    """
    Compute effective system prompt for a node.

    Combines original prompt with injected controller information.
    Handles multiple controllers.
    """
    node = self.get_node(node_id)
    original = node.original_system_prompt or node.system_prompt

    # Find all controllers (incoming directed edges)
    controllers: list[tuple[str, str]] = []
    for edge in self._edges.values():
        if edge.to_node == node_id and edge.directed:
            control_type = edge.properties.get('control_type', 'supervisor')
            controllers.append((edge.from_node, control_type))

    # If no controllers, return original prompt
    if not controllers:
        return original

    # Build controller list
    controller_lines = '\n'.join(
        f"  - Agent '{ctrl_id}' ({ctrl_type})"
        for ctrl_id, ctrl_type in sorted(controllers)
    )

    # Inject control information
    injected = f"""{original}

## Control Hierarchy
You are agent '{node_id}'. You report to the following controllers:
{controller_lines}

Follow directives from your controllers while maintaining your specialized role."""

    return injected

def _mark_subordinates_dirty(self, node_id: str) -> None:
    """Mark all subordinates' prompts as dirty (need recomputation)."""
    for edge in self._edges.values():
        if edge.from_node == node_id and edge.directed:
            subordinate = self.get_node(edge.to_node)
            subordinate.prompt_dirty = True
            logger.debug(f"Marked prompt dirty for subordinate '{edge.to_node}'")

def add_edge(self, from_node: str, to_node: str, directed: bool = True, **properties):
    """Enhanced add_edge with control relationship setup."""
    # ... existing edge creation ...

    # If directed edge, handle control relationship
    if directed:
        self._mark_subordinates_dirty(to_node)
        logger.info(
            f"Control relationship: '{from_node}' → '{to_node}' "
            f"({properties.get('control_type', 'supervisor')})"
        )
```

**Prompt Update on Next Activation:**
```python
async def _activate_agent_lazy(self, node_id: str) -> None:
    """
    Activate agent, recomputing prompt if dirty.

    Called on first send_message() or explicit start_agent().
    """
    node = self.get_node(node_id)

    # Recompute prompt if dirty
    if node.prompt_dirty:
        new_prompt = self._compute_effective_prompt(node_id)
        node.effective_system_prompt = new_prompt
        node.prompt_dirty = False
        logger.info(f"Updated effective prompt for agent '{node_id}'")

    # Start agent if not running
    if node_id not in self._agent_manager._contexts:
        await self._agent_manager.start_agent(node_id)
```

**Integration with send_message():**
```python
async def send_message(self, from_node: str, to_node: str, content: str, **metadata):
    """Enhanced send_message with lazy agent activation."""
    # ... existing validation ...

    # Lazy activate both nodes
    await self._activate_agent_lazy(from_node)
    await self._activate_agent_lazy(to_node)

    # ... existing message creation and storage ...
```

**Example Control Hierarchy:**
```
Original system prompt (Agent A):
"You are a financial analyst. Analyze quarterly reports."

Control edges added:
- CFO → Agent A (control_type="supervisor")
- Chief Risk Officer → Agent A (control_type="compliance_reviewer")

Effective prompt after injection:
"You are a financial analyst. Analyze quarterly reports.

## Control Hierarchy
You are agent 'financial_analyst'. You report to the following controllers:
  - Agent 'CFO' (supervisor)
  - Agent 'chief_risk_officer' (compliance_reviewer)

Follow directives from your controllers while maintaining your specialized role."
```

**Acceptance Criteria:**
- ✅ original_system_prompt stored separately
- ✅ effective_system_prompt computed from original + controllers
- ✅ Multiple controllers supported and listed
- ✅ Prompt marked dirty on edge changes
- ✅ Recomputed on next agent activation (lazy)
- ✅ No mid-session restarts
- ✅ Clear injection format with delimiters
- ✅ Tests verify prompt injection correctness

---

### Story 4.2.2: Implement Controller Query Methods

**Goal:** Enable inspection of control relationships in the graph.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` - Add query methods

**Implementation Details:**

```python
# In AgentGraph

def get_controllers(self, node_id: str) -> list[str]:
    """
    Get all controllers for a node.

    Returns IDs of nodes with directed edges pointing TO this node.

    Args:
        node_id: Node to query

    Returns:
        List of controller node IDs (may be empty)

    Raises:
        NodeNotFoundError: If node doesn't exist
    """
    if not self.node_exists(node_id):
        raise NodeNotFoundError(f"Node '{node_id}' not found")

    controllers = []
    for edge in self._edges.values():
        if edge.to_node == node_id and edge.directed:
            controllers.append(edge.from_node)

    return sorted(controllers)  # Consistent ordering

def get_subordinates(self, node_id: str) -> list[str]:
    """
    Get all subordinates for a node.

    Returns IDs of nodes with directed edges FROM this node.

    Args:
        node_id: Node to query

    Returns:
        List of subordinate node IDs (may be empty)

    Raises:
        NodeNotFoundError: If node doesn't exist
    """
    if not self.node_exists(node_id):
        raise NodeNotFoundError(f"Node '{node_id}' not found")

    subordinates = []
    for edge in self._edges.values():
        if edge.from_node == node_id and edge.directed:
            subordinates.append(edge.to_node)

    return sorted(subordinates)

def is_controller(self, controller_id: str, subordinate_id: str) -> bool:
    """
    Check if one node controls another.

    Args:
        controller_id: Potential controller node ID
        subordinate_id: Potential subordinate node ID

    Returns:
        True if controller_id has a directed edge to subordinate_id
    """
    edge_id = Edge.generate_edge_id(controller_id, subordinate_id, directed=True)
    return edge_id in self._edges and self._edges[edge_id].directed

def get_control_relationships(self) -> dict[str, list[str]]:
    """
    Get all control relationships in the graph.

    Returns:
        Dict mapping node_id → list of subordinate node IDs
    """
    relationships = {}
    for node_id in self._nodes:
        subordinates = self.get_subordinates(node_id)
        if subordinates:
            relationships[node_id] = subordinates

    return relationships
```

**Usage Examples:**
```python
# Build a hierarchy
graph = AgentGraph(name="org")
await graph.add_node("cfo", "You are CFO")
await graph.add_node("analyst", "You analyze finance")
await graph.add_edge("cfo", "analyst", directed=True)

# Query relationships
controllers = graph.get_controllers("analyst")  # ["cfo"]
subordinates = graph.get_subordinates("cfo")   # ["analyst"]
is_controlled = graph.is_controller("cfo", "analyst")  # True

# Get all relationships
all_rels = graph.get_control_relationships()
# {"cfo": ["analyst"]}
```

**Acceptance Criteria:**
- ✅ get_controllers(node_id) returns list of controller IDs
- ✅ get_subordinates(node_id) returns list of subordinate IDs
- ✅ is_controller(a, b) checks control relationship
- ✅ get_control_relationships() returns all hierarchies
- ✅ Works with multiple controllers per node
- ✅ Consistent ordering (sorted results)
- ✅ Tests cover trees, DAGs, and meshes

---

## Testing Strategy

### Unit Tests (test_agent_manager.py)

**Test Categories:**

1. **Session Creation (5 tests)**
   - test_create_session_basic
   - test_create_session_with_metadata
   - test_session_caching
   - test_create_session_node_not_found
   - test_create_session_nonexistent_model

2. **Lifecycle Management (8 tests)**
   - test_start_agent
   - test_stop_agent
   - test_stop_agent_not_running (idempotent)
   - test_restart_agent
   - test_start_already_running_raises
   - test_status_transitions
   - test_multiple_concurrent_agents
   - test_stop_all_cleanup

3. **Error Recovery (10 tests)**
   - test_retry_on_transient_error
   - test_exponential_backoff
   - test_max_retries_exceeded
   - test_permanent_error_status
   - test_error_metadata_stored
   - test_recovery_after_error
   - test_error_logging
   - test_concurrent_errors
   - test_error_state_isolation
   - test_get_agent_status_with_error

4. **Mocking Strategy:**
```python
@pytest.fixture
def mock_claude_client():
    """Create a mock ClaudeSDKClient."""
    client = AsyncMock(spec=ClaudeSDKClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client

@pytest.fixture
def mock_claude_options():
    """Mock ClaudeAgentOptions."""
    return MagicMock(spec=ClaudeAgentOptions)
```

### Integration Tests (test_graph.py additions)

**Test Categories:**

1. **Agent Initialization (6 tests)**
   - test_add_node_lazy_initialization
   - test_send_message_triggers_activation
   - test_explicit_start_agent
   - test_agent_status_after_start
   - test_concurrent_agent_activation
   - test_activate_with_dirty_prompt

2. **Control Relationships (8 tests)**
   - test_prompt_injection_single_controller
   - test_prompt_injection_multiple_controllers
   - test_prompt_update_on_edge_add
   - test_get_controllers
   - test_get_subordinates
   - test_is_controller
   - test_control_relationships_dict
   - test_prompt_injection_format

3. **End-to-End Workflows (5 tests)**
   - test_build_hierarchy_and_send_messages
   - test_dynamic_control_relationship_change
   - test_agent_lifecycle_with_messages
   - test_error_recovery_in_graph
   - test_async_context_manager_cleanup

### Error Scenario Tests (8 tests)

- test_sdk_initialization_failure
- test_agent_timeout
- test_rate_limit_handling
- test_api_key_error
- test_network_error_retry
- test_cascading_failure_isolation
- test_recovery_from_error_state
- test_concurrent_failures_dont_block

---

## Files Summary

### New Files
1. **EPIC_4_IMPLEMENTATION.md** - This document
2. **src/claude_agent_graph/agent_manager.py** (~350 lines)
   - AgentSessionManager class
   - AgentErrorRecovery helper
   - Session lifecycle management

3. **tests/test_agent_manager.py** (~550 lines)
   - Comprehensive unit tests
   - Mock SDK integration
   - Error scenario coverage

### Modified Files
1. **pyproject.toml** (+1 line)
   - Add claude-agent-sdk>=1.0.0

2. **src/claude_agent_graph/models.py** (+8 lines)
   - Add original_system_prompt: Optional[str]
   - Add effective_system_prompt: Optional[str]
   - Add prompt_dirty: bool
   - Add agent_session: Optional[Any]

3. **src/claude_agent_graph/graph.py** (~180 lines)
   - Import and initialize AgentSessionManager
   - Add _compute_effective_prompt()
   - Add _mark_subordinates_dirty()
   - Add _activate_agent_lazy()
   - Add lifecycle methods (start_agent, stop_agent, restart_agent)
   - Add query methods (get_controllers, get_subordinates, is_controller)
   - Add get_agent_status()
   - Add async context manager support
   - Modify add_edge() for control relationships
   - Modify send_message() for lazy activation

4. **tests/test_graph.py** (~280 lines)
   - Integration tests for agent lifecycle
   - Control relationship tests
   - Prompt injection tests
   - End-to-end workflow tests

---

## Success Criteria

### Code Quality
- ✅ >85% test coverage for new code
- ✅ All tests pass with mocked SDK
- ✅ Type hints on all functions
- ✅ Comprehensive docstrings

### Functionality
- ✅ Agents initialize lazily (no immediate API calls)
- ✅ Lifecycle transitions correct (INITIALIZING → ACTIVE → STOPPED)
- ✅ Error recovery works (retry + status update)
- ✅ Multiple controllers supported in prompts
- ✅ Prompt updates deferred to next activation
- ✅ Control query methods work correctly

### Performance
- ✅ Session creation <200ms per agent
- ✅ Status queries <10ms (O(1))
- ✅ Concurrent agent operations supported
- ✅ No resource leaks (proper cleanup)

### Testing
- ✅ 99+ tests total (manager + graph)
- ✅ Error scenarios covered
- ✅ Edge cases tested
- ✅ Concurrent operations tested

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| SDK API changes | Medium | High | Pin version, comprehensive integration tests, adapter pattern |
| Resource exhaustion | High | High | Lazy initialization, agent pooling, resource monitoring |
| Prompt injection failures | Medium | Medium | Clear delimiters, extensive testing, fallback formats |
| Async context bugs | Medium | High | Extensive testing, proper cleanup, asyncio best practices |
| API rate limits | Medium | Medium | Exponential backoff, request queuing, error recovery |

---

## References

- **Claude Agent SDK:** https://github.com/anthropics/claude-agent-sdk-python
- **Related Epics:** Epic 1 (Foundation), Epic 2 (Graph), Epic 3 (State)
- **Depends On:** IMPLEMENTATION_PLAN.md (Epic 4 section)

---

## Implementation Timeline

**Week 1: Foundation & Lifecycle**
- Story 4.1.1: Session initialization (~2 days)
- Story 4.1.2: Lifecycle management (~2 days)
- Integration & testing (~1 day)

**Week 2: Error Recovery & Control**
- Story 4.1.3: Error recovery (~1.5 days)
- Story 4.2.1: Prompt injection (~1.5 days)
- Story 4.2.2: Control queries (~1 day)
- Final testing & polish (~1 day)

**Estimated Total:** 4-5 days of development

---

**Document Status:** Ready for Implementation
**Last Updated:** October 2025
