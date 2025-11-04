# Epic 6 Implementation Plan: Agent Execution & Control

**Status:** Ready for Implementation
**Date:** October 2025
**Estimated Effort:** 5-7 days
**Dependencies:** Epic 4 (Agent Integration), Epic 5 (Dynamic Operations)

## Overview

Epic 6 adds message routing patterns, execution modes, and control commands to enable sophisticated agent orchestration. This epic builds on existing message delivery capabilities to add broadcast, multi-hop routing, and three execution modes (reactive, manual, proactive).

**Key Capabilities:**
- Broadcast messages to multiple neighbors
- Multi-hop message routing through agent networks
- Execute commands on subordinate agents with authorization
- Three execution modes: Reactive (message-driven), Manual (step-by-step), Proactive (periodic)
- Message queue infrastructure for async agent execution

## Current State

✅ **Already Implemented:**
- Direct message routing (`send_message()` in graph.py:809-882)
- Control relationships with prompt injection
- Agent lifecycle management (start/stop/restart)
- Message persistence to conversation files
- Command authorization infrastructure

## Architecture Decisions

### 1. Message Queue Infrastructure
**Decision:** Use asyncio.Queue per node for execution modes.

**Rationale:**
- Async-native, matches existing codebase patterns
- Decouples message sending from agent processing
- Supports all three execution modes
- Per-node queues prevent bottlenecks

### 2. Command Message Format
**Decision:** Special message metadata with `type: "command"` marker.

```python
metadata = {
    "type": "command",
    "command": "process_data",
    "params": {"dataset": "Q1_2025"},
    "authorization_level": "supervisor"
}
```

**Rationale:**
- Unified message infrastructure (no separate command channel)
- Easy to log and audit
- Backward compatible with regular messages

### 3. Multi-hop Routing Path Tracking
**Decision:** Include path metadata in routed messages.

```python
metadata = {
    "routing_path": ["A", "B", "C", "D"],
    "hop_count": 3,
    "original_sender": "A"
}
```

**Rationale:**
- Enables loop detection
- Tracks message origin through network
- Useful for debugging

### 4. Execution Mode Lifecycle
**Decision:** Only one execution mode active at a time. Must stop current before starting new.

**Rationale:**
- Prevents competing message handlers
- Clear lifecycle semantics
- Easy to reason about
- Clean shutdown on graph exit

### 5. Broadcast vs Direct vs Multi-hop
**Decision:** Three distinct routing patterns with different semantics.

- **Direct:** Point-to-point, requires edge
- **Broadcast:** One-to-many, sends to all neighbors
- **Multi-hop:** Path-based routing through intermediate nodes

**Rationale:**
- Direct: Most common, most efficient
- Broadcast: Leader/coordinator patterns
- Multi-hop: Complex workflows, indirect communication

## Feature 6.1: Message Routing Patterns

### Story 6.1.1: Direct Message Routing
**Status:** Already implemented in Epic 3.2.1

No changes needed. Existing `send_message()` handles point-to-point routing.

---

### Story 6.1.2: Broadcast Routing

**Goal:** Enable one agent to send a message to all neighbors.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~80 lines)
- `tests/test_graph.py` (~60 lines)

**Implementation Details:**

```python
async def broadcast(
    self,
    from_node: str,
    content: str,
    include_incoming: bool = False,
    **metadata: Any,
) -> list[Message]:
    """
    Broadcast a message to all neighbors of a node.

    By default, sends only to outgoing edges (nodes this one sends to).
    With include_incoming=True, also sends to incoming edges (nodes that send to this one).

    Args:
        from_node: ID of the sending node
        content: Message content
        include_incoming: If True, include incoming edges (default: False)
        **metadata: Additional metadata for messages

    Returns:
        List of Message objects (one per recipient)

    Raises:
        NodeNotFoundError: If from_node doesn't exist

    Example:
        >>> # Send to all outgoing neighbors
        >>> messages = await graph.broadcast("supervisor", "Status update")
        >>>
        >>> # Send to both incoming and outgoing neighbors
        >>> messages = await graph.broadcast(
        ...     "hub", "Alert", include_incoming=True
        ... )
    """
```

**Implementation Steps:**

1. **Validate node exists**
   - Raise `NodeNotFoundError` if from_node not found

2. **Get target nodes**
   - Get outgoing neighbors via `get_neighbors(from_node, "outgoing")`
   - If include_incoming, also get incoming neighbors
   - Remove duplicates (in case of bidirectional edges)

3. **Send to each neighbor**
   - For each target, call `await send_message(from_node, target, content, **metadata)`
   - Collect all Message objects

4. **Return messages**
   - Return list of sent messages

5. **Logging**
   - Log broadcast count and recipients

**Edge Cases:**
- No neighbors: Return empty list (not an error)
- Isolated node: Still returns empty list
- Bidirectional edges: Don't double-send if include_incoming=True

**Acceptance Criteria:**
- ✅ Sends to all outgoing neighbors
- ✅ include_incoming=True adds incoming neighbors
- ✅ Returns list of Message objects
- ✅ No double-sending
- ✅ Tests cover all routing patterns

---

### Story 6.1.3: Multi-hop Routing

**Goal:** Route messages through a path of intermediate nodes.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~120 lines)
- `tests/test_graph.py` (~80 lines)

**Implementation Details:**

```python
async def route_message(
    self,
    from_node: str,
    to_node: str,
    content: str,
    path: list[str] | None = None,
    **metadata: Any,
) -> list[Message]:
    """
    Route a message through a path of intermediate nodes.

    If path is not provided, finds shortest path using NetworkX.
    Path must start with from_node and end with to_node.

    Args:
        from_node: Starting node ID
        to_node: Ending node ID
        content: Message content
        path: Explicit path as list of node IDs (optional)
        **metadata: Additional metadata

    Returns:
        List of Message objects for each hop

    Raises:
        NodeNotFoundError: If any node in path doesn't exist
        EdgeNotFoundError: If path doesn't connect
        ValueError: If path format invalid

    Example:
        >>> # Auto-find shortest path
        >>> messages = await graph.route_message("A", "D", "Request")
        >>>
        >>> # Use explicit path
        >>> messages = await graph.route_message(
        ...     "A", "D", "Request", path=["A", "B", "C", "D"]
        ... )
    """
```

**Implementation Steps:**

1. **Validate nodes**
   - Check from_node and to_node exist
   - If path provided, validate all nodes in path exist

2. **Find or validate path**
   - If path not provided: use `nx.shortest_path(self._nx_graph, from_node, to_node)`
   - If path provided: validate it connects (each consecutive pair has edge)
   - Raise `EdgeNotFoundError` if path breaks

3. **Route message through path**
   - For each consecutive pair in path, send message
   - Track routing metadata:
     ```python
     metadata["routing_path"] = path
     metadata["hop_count"] = len(path) - 1
     metadata["original_sender"] = from_node
     ```

4. **Collect and return messages**
   - Return list of all Message objects sent

5. **Logging**
   - Log path and hops

**Edge Cases:**
- Direct connection: path is [from_node, to_node]
- Long paths: Handle 10+ hops correctly
- No path exists: Raise error with context
- from_node == to_node: Should raise error or return empty list?
  - Decision: Raise ValueError (can't route to self)

**Acceptance Criteria:**
- ✅ Auto-finds shortest path
- ✅ Validates explicit paths
- ✅ Sends through all hops
- ✅ Tracks routing metadata
- ✅ Raises errors for invalid paths
- ✅ Tests cover direct and indirect paths

---

## Feature 6.3: Control Commands

### Story 6.3.1: Command Execution

**Goal:** Enable controllers to issue commands to subordinates with proper authorization.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (~90 lines)
- `src/claude_agent_graph/exceptions.py` (~5 lines)
- `tests/test_graph.py` (~70 lines)

**Implementation Details:**

```python
async def execute_command(
    self,
    controller: str,
    subordinate: str,
    command: str,
    **params: Any,
) -> Message:
    """
    Execute a command on a subordinate agent.

    Validates that controller has a directed edge to subordinate,
    then sends a specially formatted command message.

    Args:
        controller: ID of the controlling agent
        subordinate: ID of the subordinate agent
        command: Command name (e.g., "process_data")
        **params: Command parameters

    Returns:
        The command Message object

    Raises:
        NodeNotFoundError: If either node doesn't exist
        CommandAuthorizationError: If controller doesn't control subordinate

    Example:
        >>> msg = await graph.execute_command(
        ...     "supervisor", "worker", "analyze",
        ...     dataset="Q1", output_format="json"
        ... )
    """
```

**Implementation Steps:**

1. **Validate nodes exist**
   - Raise `NodeNotFoundError` if either doesn't exist

2. **Check control relationship**
   - Call `self.is_controller(controller, subordinate)`
   - If False, raise `CommandAuthorizationError`

3. **Build command message**
   - Create metadata with:
     ```python
     {
         "type": "command",
         "command": command,
         "params": params,
         "authorization_level": "supervisor"  # from edge properties
     }
     ```

4. **Send command message**
   - Call `await send_message(controller, subordinate, content, **metadata)`
   - Use generic command format for content: `f"Execute: {command}"`

5. **Log command**
   - Log with full parameters for audit trail

6. **Return message**
   - Return the Message object

**Edge Cases:**
- Controller not directly connected: Still requires directed edge (no transitive control)
- Missing params: Allow empty params
- Command name validation: Accept any string

**Acceptance Criteria:**
- ✅ Validates control relationship
- ✅ Creates command metadata
- ✅ Sends via normal message infrastructure
- ✅ Raises CommandAuthorizationError when unauthorized
- ✅ Logs all commands
- ✅ Tests cover authorization scenarios

---

### Story 6.3.2: Command Authorization

**Goal:** Enforce authorization checks for command execution.

**Files to Modify:**
- `src/claude_agent_graph/exceptions.py` (~8 lines)
- `src/claude_agent_graph/graph.py` (already in 6.3.1)
- `tests/test_graph.py` (~50 lines)

**Implementation Details:**

**New Exception:**
```python
class CommandAuthorizationError(AgentGraphError):
    """Raised when command execution is unauthorized."""
    pass
```

**Authorization Check (in execute_command):**
1. Validate control relationship exists
2. Check edge properties for permission levels (if configured)
3. Log unauthorized attempts with context
4. Raise `CommandAuthorizationError` with descriptive message

**Authorization Levels (Future-proof):**
- Store in edge properties: `control_type` field
- Examples: "supervisor", "manager", "peer"
- For now: Only check relationship exists, not type

**Logging:**
```python
logger.warning(
    f"Unauthorized command attempt: {controller} -> {subordinate} "
    f"(not a control relationship)"
)
```

**Acceptance Criteria:**
- ✅ Rejects commands without control relationship
- ✅ Logs unauthorized attempts
- ✅ Includes context in error messages
- ✅ Tests cover all authorization scenarios

---

## Feature 6.2: Execution Modes

### Architecture: Base Execution Class

```python
from abc import ABC, abstractmethod

class ExecutionMode(ABC):
    """Base class for execution modes."""

    def __init__(self, graph: AgentGraph) -> None:
        self._graph = graph
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the execution mode."""
        if self._running:
            raise RuntimeError("Execution mode already running")
        self._running = True
        self._task = asyncio.create_task(self._execute_loop())

    async def stop(self) -> None:
        """Stop the execution mode."""
        self._running = False
        if self._task:
            await self._task

    @abstractmethod
    async def _execute_loop(self) -> None:
        """Main execution loop (implemented by subclasses)."""
        pass
```

### Story 6.2.2: Manual Mode

**Goal:** Step-by-step control over agent execution.

**Files to Create:**
- `src/claude_agent_graph/execution.py` (~150 lines)

**Implementation Details:**

```python
class ManualController(ExecutionMode):
    """
    Manual execution mode for step-by-step control.

    The graph doesn't automatically process messages.
    User explicitly calls step() to execute one agent turn.
    """

    async def step(self, node_id: str) -> None:
        """
        Execute one turn for a specific agent.

        Processes one pending message from the agent's queue,
        if any. Otherwise does nothing.
        """

    async def step_all(self) -> int:
        """
        Execute one turn for all agents with pending messages.

        Returns the number of agents that executed.
        """
```

**Implementation Steps:**

1. **Add message queues to AgentGraph**
   - `self._message_queues: dict[str, asyncio.Queue]` (lazily created)
   - Modified `send_message()` to enqueue when execution active

2. **Implement ManualController**
   - Store reference to graph
   - Implement `step(node_id)`:
     - Get message queue for node
     - Pop one message from queue (non-blocking)
     - Process message (call agent with message content)
     - Return

   - Implement `step_all()`:
     - Get all queues with pending messages
     - Call step() for each
     - Return count

3. **Integration with AgentGraph**
   - Add `start(mode="manual")` method
   - Add `stop_execution()` method
   - Modify `send_message()` to check if execution active

**Edge Cases:**
- Step on empty queue: Return silently (no error)
- Node not in graph: Raise NodeNotFoundError
- Multiple steps: Queue preserves order

**Acceptance Criteria:**
- ✅ step(node_id) processes one message
- ✅ step_all() executes all pending
- ✅ Returns execution counts
- ✅ Queue order preserved
- ✅ Tests cover stepping logic

---

### Story 6.2.1: Reactive Mode

**Goal:** Automatic message-driven execution.

**Files to Modify:**
- `src/claude_agent_graph/execution.py` (~200 lines)
- `tests/test_execution.py` (~200 lines)

**Implementation Details:**

```python
class ReactiveExecutor(ExecutionMode):
    """
    Reactive execution mode (event-driven).

    Automatically processes messages from agent queues.
    When a message arrives, the receiving agent processes it.
    """

    async def _execute_loop(self) -> None:
        """
        Main reactive loop.

        Monitors all message queues and processes messages
        as they arrive.
        """
```

**Implementation Steps:**

1. **Message Queue System**
   - Create queue per node on first message
   - Bounded queues (max 1000 messages, configurable)
   - Non-blocking sends (drop oldest if full)

2. **Reactive Loop (_execute_loop)**
   - Use `asyncio.gather()` to monitor all queues concurrently
   - For each queue with a message:
     - Pop message
     - Activate agent and send message
     - Wait for response (if applicable)
   - Loop until stopped

3. **Concurrency Model**
   - Process all available messages concurrently
   - One task per node
   - Each task handles its queue

4. **Error Handling**
   - Catch agent errors without stopping loop
   - Log errors but continue
   - Mark node status as ERROR

**Edge Cases:**
- Queue fills up: Drop oldest messages or reject?
  - Decision: Use `Queue.put_nowait()` with exception handling
  - Log when queue is full
- Agent fails: Continue with next message
- Graph modified during execution: Safe due to locking

**Acceptance Criteria:**
- ✅ Processes messages automatically
- ✅ Handles concurrent agents
- ✅ Stops cleanly on stop()
- ✅ Handles agent errors gracefully
- ✅ Tests cover happy path and errors

---

### Story 6.2.3: Proactive Mode

**Goal:** Agents initiate conversations periodically.

**Files to Modify:**
- `src/claude_agent_graph/execution.py` (~200 lines)
- `tests/test_execution.py` (~150 lines)

**Implementation Details:**

```python
class ProactiveExecutor(ExecutionMode):
    """
    Proactive execution mode (periodic agent activation).

    Agents wake up periodically and can initiate conversations
    without waiting for incoming messages.
    """

    def __init__(
        self,
        graph: AgentGraph,
        interval: float = 60.0,
        start_delay: float = 0.0,
    ) -> None:
        """
        Args:
            graph: The AgentGraph
            interval: Seconds between agent activations
            start_delay: Seconds before first activation
        """

    async def _execute_loop(self) -> None:
        """Periodic agent activation loop."""
```

**Implementation Steps:**

1. **Configuration**
   - Store interval (default 60 seconds)
   - Store start_delay (stagger first execution)
   - Per-node activation tracking (to allow per-agent intervals in future)

2. **Proactive Loop**
   - Sleep for start_delay
   - Loop while running:
     - Sleep for interval
     - For each node:
       - Activate agent (wake it up)
       - Agent can then initiate messages

3. **Agent Activation**
   - Call `await self._agent_manager.start_agent(node_id)` if not running
   - For stateful agents, could send "wake up" signal

4. **Concurrency**
   - Use `asyncio.sleep()` for intervals
   - Activate agents sequentially or concurrently?
   - Decision: Sequentially to avoid thundering herd

**Edge Cases:**
- Very short interval: No sleep if computation takes time
- Agent activation takes long: Still respects interval
- New nodes added during execution: Picked up next cycle

**Acceptance Criteria:**
- ✅ Activates agents periodically
- ✅ Respects interval timing
- ✅ Handles start_delay
- ✅ Stops cleanly
- ✅ Tests cover timing

---

## Integration with AgentGraph

### Message Queue Infrastructure

**Add to AgentGraph.__init__:**
```python
self._execution_mode: ExecutionMode | None = None
self._message_queues: dict[str, asyncio.Queue] = {}
```

### Modify send_message():

```python
async def send_message(
    self,
    from_node: str,
    to_node: str,
    content: str,
    **metadata: Any,
) -> Message:
    """
    [existing docstring]

    If an execution mode is active, enqueues the message
    for processing by that mode.
    """
    # ... existing validation code ...

    # Create and persist message
    message = Message(...)
    await self.storage.append_message(edge_id, message)

    # Enqueue for execution mode if active
    if self._execution_mode:
        await self._enqueue_message(to_node, message)

    return message
```

### Add execution mode methods:

```python
async def start(
    self,
    mode: str = "reactive",
    **kwargs: Any,
) -> None:
    """
    Start an execution mode.

    Args:
        mode: "reactive", "manual", or "proactive"
        **kwargs: Mode-specific options
            - reactive: no additional options
            - manual: no additional options
            - proactive: interval (float), start_delay (float)
    """
    if self._execution_mode:
        raise RuntimeError("Execution mode already running")

    if mode == "manual":
        self._execution_mode = ManualController(self)
    elif mode == "reactive":
        self._execution_mode = ReactiveExecutor(self)
    elif mode == "proactive":
        self._execution_mode = ProactiveExecutor(self, **kwargs)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")

    await self._execution_mode.start()

async def stop_execution(self) -> None:
    """Stop the current execution mode."""
    if self._execution_mode:
        await self._execution_mode.stop()
        self._execution_mode = None

async def _enqueue_message(self, node_id: str, message: Message) -> None:
    """Enqueue a message for execution mode processing."""
    if node_id not in self._message_queues:
        self._message_queues[node_id] = asyncio.Queue(maxsize=1000)

    try:
        self._message_queues[node_id].put_nowait(message)
    except asyncio.QueueFull:
        logger.warning(f"Message queue full for node '{node_id}', dropping oldest")
        try:
            self._message_queues[node_id].get_nowait()
            self._message_queues[node_id].put_nowait(message)
        except asyncio.QueueEmpty:
            pass
```

### Update __aexit__:

```python
async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Clean up all agents and execution mode."""
    if self._execution_mode:
        await self.stop_execution()

    if hasattr(self, "_agent_manager"):
        await self._agent_manager.stop_all()
```

---

## File Structure

### New Files:
1. **`src/claude_agent_graph/execution.py`** (~450 lines)
   - ExecutionMode base class
   - ManualController
   - ReactiveExecutor
   - ProactiveExecutor

### Modified Files:
1. **`src/claude_agent_graph/graph.py`** (~270 new lines)
   - broadcast() method
   - route_message() method
   - execute_command() method
   - Message queue infrastructure
   - start() / stop_execution() methods
   - _enqueue_message() helper

2. **`src/claude_agent_graph/exceptions.py`** (~8 lines)
   - CommandAuthorizationError exception

3. **`src/claude_agent_graph/__init__.py`** (~10 lines)
   - Export execution mode classes

4. **`tests/test_execution.py`** (~400 lines)
   - Tests for all execution modes
   - Integration tests

5. **`tests/test_graph.py`** (~250 new lines)
   - Tests for routing patterns
   - Tests for control commands

---

## Implementation Sequence

### Phase 1: Message Routing (Days 1-2)
1. Implement broadcast() method
2. Implement route_message() method
3. Write comprehensive routing tests
4. Test with existing graph structures

### Phase 2: Control Commands (Day 3)
1. Add CommandAuthorizationError exception
2. Implement execute_command() method
3. Add authorization checks
4. Write command tests

### Phase 3: Execution Infrastructure (Day 4-5)
1. Create execution.py with base class
2. Add message queues to AgentGraph
3. Implement ManualController (simplest mode)
4. Write manual mode tests
5. Integration with start()/stop_execution()

### Phase 4: Reactive Mode (Day 6)
1. Implement ReactiveExecutor
2. Integrate with message queues
3. Write reactive mode tests
4. Full integration testing

### Phase 5: Proactive Mode (Day 7)
1. Implement ProactiveExecutor
2. Add periodic triggering
3. Write proactive mode tests
4. Full integration testing
5. Documentation

---

## Testing Strategy

### Unit Tests
- Each routing method tested independently
- Each execution mode tested in isolation
- Mock agent responses
- Edge cases: empty graphs, isolated nodes, missing connections

### Integration Tests
- Message flow through execution modes
- Command execution with real relationships
- Multi-hop through complex topologies
- Concurrent agent execution
- Mode switching

### Coverage Target
- **Target:** 85%+ for new code
- Existing code should maintain 87%+ coverage

---

## Success Criteria

- ✅ broadcast() sends to all neighbors correctly
- ✅ route_message() handles explicit and auto paths
- ✅ execute_command() enforces authorization
- ✅ All 3 execution modes work independently
- ✅ Can switch between execution modes
- ✅ Message queues work correctly
- ✅ Clean shutdown on graph exit
- ✅ 85%+ test coverage
- ✅ All tests pass (pytest -v)
- ✅ Integration with existing features works
- ✅ No breaking changes to public API

---

## Estimated Effort

- **Total new code:** ~1,090 lines
- **Total test code:** ~650 lines
- **Total effort:** 5-7 days
- **Complexity:** Medium-High

### Breakdown:
- Message routing: 200 LOC + 140 test LOC (2 days)
- Control commands: 90 LOC + 120 test LOC (1 day)
- Execution modes: 600 LOC + 400 test LOC (4 days)

---

## Risk Mitigation

### Risk 1: Complex Execution Mode Interactions
**Mitigation:** Start with ManualController (simplest), then add Reactive, then Proactive. Each mode tested independently before integration.

### Risk 2: Message Queue Overflow
**Mitigation:** Bounded queues with overflow handling. Monitor and log when queue fills.

### Risk 3: Concurrent Modification During Execution
**Mitigation:** Use existing asyncio.Lock and test concurrent scenarios thoroughly.

### Risk 4: Agent Initialization During Execution
**Mitigation:** Lazy activation already implemented; execution modes rely on it.

---

## Future Enhancements (Not in Scope)

- Per-agent activation intervals for ProactiveExecutor
- Message priority queues
- Execution mode composition (chain multiple modes)
- Monitoring and metrics for execution modes
- Agent response callbacks in command execution
- Conditional routing based on message content
- Rate limiting per agent

---

**Document Status:** Ready for Implementation
**Last Updated:** October 2025
**Author:** Implementation Team
