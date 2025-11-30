# claude-agent-graph Features

This document describes the implemented features and recent enhancements to the claude-agent-graph system.

## Table of Contents

1. [Agent Response Routing for Directed Edges](#agent-response-routing-for-directed-edges)
2. [Per-Node Max Tokens Configuration](#per-node-max-tokens-configuration)
3. [Graceful Agent Shutdown](#graceful-agent-shutdown)
4. [Message Queuing System](#message-queuing-system)

---

## Agent Response Routing for Directed Edges

### Feature Overview

When agents receive messages through directed edges (supervisor → subordinate), their responses are automatically routed back through the same conversation channel, preserving the supervisory relationship while enabling full bidirectional communication.

### How It Works

1. **Supervisor sends message** to worker via directed edge
2. **Worker processes message** and generates response
3. **Response automatically routed back** and stored in the same conversation thread
4. **Supervisor can see response** in conversation history with supervisory context preserved

### Example

```python
import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend
from claude_agent_graph.execution import ManualController

async def main():
    async with AgentGraph(
        name="supervisor_worker",
        storage_backend=FilesystemBackend(base_dir="./conversations")
    ) as graph:
        # Create supervisor and worker nodes
        await graph.add_node(
            "supervisor",
            "You are a supervisor. Delegate tasks to your subordinates."
        )

        await graph.add_node(
            "worker",
            "You are a worker. Execute tasks assigned by your supervisor."
        )

        # Create DIRECTED edge (supervisor → worker)
        # This creates a supervisory relationship
        await graph.add_edge("supervisor", "worker", directed=True)

        # Start executor
        executor = ManualController(graph)
        await executor.start()

        # Supervisor sends task to worker
        await graph.send_message(
            "supervisor",
            "worker",
            "Please analyze the Q4 sales report and provide a summary."
        )

        # Worker processes the task
        await executor.step("worker")

        # Retrieve conversation (both request and response visible)
        messages = await graph.get_conversation("supervisor", "worker")

        for msg in messages:
            role = "👔 Supervisor" if msg.from_node == "supervisor" else "👷 Worker"
            print(f"{role}: {msg.content[:100]}...")

        await executor.stop()

asyncio.run(main())
```

### Output Example

```
👔 Supervisor: Please analyze the Q4 sales report and provide a summary.
👷 Worker: Analysis complete. Q4 sales increased by 15% YoY...
```

### Implementation Details

**Location:** `src/claude_agent_graph/graph.py` - `_process_message_with_agent()` method

**Algorithm:**
1. Agent receives message from queue
2. Agent processes message via Claude API
3. Response collected from agent session
4. Check for direct edge back to sender (bidirectional case)
5. If no direct edge exists, check if original edge was directed
6. If directed, append response to original edge's conversation file
7. If neither, log warning (edge relationship broken)

**Key Benefits:**
- ✅ Preserves hierarchical control relationships
- ✅ No need for bidirectional edges in supervisor-worker patterns
- ✅ Conversation stays in one thread for better UX
- ✅ Automatic - no configuration needed

---

## Per-Node Max Tokens Configuration

### Feature Overview

Configure maximum output tokens per agent node to control response length, reduce costs, and manage resource usage. The configuration is automatically enforced via the `CLAUDE_CODE_MAX_OUTPUT_TOKENS` environment variable.

### How It Works

1. **Specify max_tokens** when creating a node
2. **Session creation** automatically sets `CLAUDE_CODE_MAX_OUTPUT_TOKENS` environment variable
3. **Claude Agent SDK** reads the environment variable and enforces the limit
4. **Responses truncated** if they exceed the configured limit

### Example

```python
import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend

async def main():
    async with AgentGraph(
        name="cost_controlled_system",
        storage_backend=FilesystemBackend(base_dir="./conversations")
    ) as graph:
        # Coordinator with unlimited tokens
        await graph.add_node(
            "coordinator",
            "You coordinate tasks between teams.",
            model="claude-sonnet-4-20250514"
            # No max_tokens = unlimited
        )

        # Writer with generous limit
        await graph.add_node(
            "content_writer",
            "You write detailed content.",
            model="claude-sonnet-4-20250514",
            max_tokens=500  # Can write longer pieces
        )

        # Summarizer with strict limit
        await graph.add_node(
            "summarizer",
            "You provide brief summaries.",
            model="claude-sonnet-4-20250514",
            max_tokens=50  # Forces brevity
        )

        # Verify configurations
        for node_id in ["coordinator", "content_writer", "summarizer"]:
            node = graph.get_node(node_id)
            max_tokens_str = node.max_tokens if node.max_tokens else "Unlimited"
            print(f"{node_id}: max_tokens={max_tokens_str}")

asyncio.run(main())
```

### Output

```
coordinator: max_tokens=Unlimited
content_writer: max_tokens=500
summarizer: max_tokens=50
```

### Environment Variable Details

**Variable Name:** `CLAUDE_CODE_MAX_OUTPUT_TOKENS`

**When Set:** During agent session creation in `agent_manager.py:create_session()`

**Value:** String representation of the max_tokens integer (e.g., "100")

**Enforcement:** Claude Agent SDK reads this environment variable and truncates responses

### Use Cases

| Use Case | max_tokens | Reason |
|----------|-----------|--------|
| Coordinator/Manager | None (unlimited) | Needs flexibility for complex decisions |
| Summarizer | 50-100 | Forces conciseness, reduces costs |
| Code Generator | 1000-2000 | Need enough tokens for code |
| QA/Validator | 100-200 | Brief yes/no or short feedback |
| Report Writer | 500-1000 | Longer responses for detailed analysis |

### Cost Calculation Example

**Scenario:** 100 agent nodes, 1000 requests/day

| Configuration | Tokens/Response | Daily Tokens | Est. Cost |
|---------------|-----------------|--------------|-----------|
| No limit (avg 500) | 500 | 50M | $1.50 |
| max_tokens=200 | ~200 | 20M | $0.60 |
| max_tokens=100 | ~100 | 10M | $0.30 |

**Savings:** 80% cost reduction with max_tokens=100

---

## Graceful Agent Shutdown

### Feature Overview

Agents are shut down gracefully during cleanup, with proper error handling for the Claude Agent SDK's async context issues.

### How It Works

1. **Timeout Protection** - 5-second timeout prevents infinite hangs
2. **Exception Handling** - Catches and suppresses SDK-specific errors
3. **Graceful Degradation** - Errors don't prevent other agents from stopping
4. **Status Tracking** - All agents marked STOPPED despite cleanup errors

### Implementation Details

**Location:** `src/claude_agent_graph/agent_manager.py` - `stop_agent()` method

**Code:**
```python
async def stop_agent(self, node_id: str) -> None:
    if node_id not in self._contexts:
        logger.warning(f"Agent '{node_id}' is not running - nothing to stop")
        return

    context = self._contexts.pop(node_id)
    node = self._graph.get_node(node_id)

    try:
        # Use timeout to prevent hanging on SDK disconnect issues
        try:
            await asyncio.wait_for(context.__aexit__(None, None, None), timeout=5.0)
        except asyncio.TimeoutError:
            logger.debug(f"Agent '{node_id}' disconnect timed out after 5s")
        except (asyncio.CancelledError, RuntimeError):
            # Ignore cancel scope errors - common with SDK
            logger.debug(f"Agent '{node_id}' disconnect encountered scope error")

        node.status = NodeStatus.STOPPED
        logger.info(f"Stopped agent '{node_id}' (status: {node.status.value})")

    except Exception as e:
        logger.debug(f"Error stopping agent '{node_id}': {e}")
        node.status = NodeStatus.STOPPED
```

### Known SDK Issues

**Issue:** `RuntimeError: Attempted to exit a cancel scope that isn't the current task's current cancel scope`

**Cause:** Claude Agent SDK's disconnect process uses anyio cancel scopes that may be in a different context than the query execution

**Solution:** Catch `RuntimeError` and `CancelledError` during cleanup; log as debug, not error

### Benefits

- ✅ Clean application exits without tracebacks
- ✅ All agents stop even if some have cleanup errors
- ✅ Prevents hanging on SDK disconnect issues
- ✅ Proper status tracking for debugging

---

## Message Queuing System

### Feature Overview

Messages are automatically queued for receiving agents and processed via execution modes (Manual, Reactive, Proactive).

### How It Works

1. **Message Created** via `send_message()`
2. **Message Stored** in persistent JSONL conversation file
3. **Message Queued** in asyncio.Queue for receiving node
4. **Executor Processes** queue according to execution mode

### Queue Flow

```
send_message()
    ↓
Create Message object
    ↓
Append to storage (JSONL)
    ↓
Queue message to receiving node
    ↓
Executor.step(node) or automatic processing
    ↓
_process_message_with_agent()
    ↓
Agent responds
    ↓
Response auto-routed back
    ↓
Conversation persisted
```

### Example

```python
from claude_agent_graph.execution import ManualController

# Queue messages
await graph.send_message("a", "b", "Message 1")
await graph.send_message("a", "b", "Message 2")

# Process queue manually
executor = ManualController(graph)
await executor.step("b")  # Process Message 1
await executor.step("b")  # Process Message 2

# Or process automatically
from claude_agent_graph.execution import ReactiveExecutor
reactive = ReactiveExecutor(graph)
await reactive.start()  # Agents auto-process all queued messages
```

### Implementation Details

**Location:** `src/claude_agent_graph/graph.py` - `send_message()` method

**Queue Storage:** `AgentGraph._message_queues` - Dict[str, asyncio.Queue]

**Queue Operations:**
- `send_message()`: Enqueues message to receiving node
- `executor.step(node)`: Dequeues and processes next message
- `executor.step_all()`: Processes all queued messages across all nodes

---

## Feature Interactions

These features work together to create a complete agent communication system:

```
┌─────────────────────────────────────────────────────────┐
│        Multi-Agent Communication System                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Message Queuing ─→ Processing ─→ Response Routing      │
│                         │                                 │
│                    Max Tokens                            │
│                  (cost control)                          │
│                         │                                 │
│                    Graceful Shutdown                     │
│                  (clean termination)                     │
│                                                           │
│  Directed Edges + Response Routing = Supervisory Model   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Testing These Features

### Example Test Script

See `/examples/my-testing.py` for a working example that demonstrates:
- Creating nodes with max_tokens configuration
- Directed edge response routing
- Manual execution stepping
- Graceful shutdown

**Run it:**
```bash
cd /path/to/claude-agent-graph-testing
python3 my-testing.py
```

### Unit Tests

- `tests/test_graph.py` - Core graph operations
- `tests/test_agent_manager.py` - Agent lifecycle (including shutdown)
- `tests/test_execution.py` - Execution modes and message processing
- `tests/test_models.py` - Data validation including max_tokens

---

## Performance Considerations

### Max Tokens Impact

- **Lower limits** → Faster responses, lower cost
- **Response truncation** occurs at token boundary
- **No performance penalty** for setting max_tokens

### Message Queue Impact

- **Async queues** - Non-blocking, scales to 10,000+ nodes
- **JSONL storage** - Append-only, thread-safe persistence
- **Memory** - Queues hold messages in RAM until processed

### Shutdown Impact

- **5-second timeout** prevents hanging
- **No performance impact** during normal operation
- **Only affects cleanup phase** (application exit)

---

## Version History

### v0.1.0 (Current)

**New Features:**
- ✅ Agent response routing for directed edges (preserves supervisory relationships)
- ✅ Per-node max_tokens configuration (cost and response control)
- ✅ Graceful agent shutdown with SDK error handling
- ✅ Message queuing system for execution modes

**Bug Fixes:**
- ✅ Fixed agent shutdown cancel scope errors
- ✅ Fixed response routing in directed edge conversations
- ✅ Improved error handling in stop_all()

---

## Future Enhancements

Potential improvements being considered:

- [ ] Timeout configuration per agent (not just shutdown)
- [ ] Response truncation strategies (middle vs. end truncation)
- [ ] Message priority queues (urgent messages processed first)
- [ ] Queue persistence (survive application restart)
- [ ] Queue metrics (size, depth, latency)
- [ ] Adaptive max_tokens (adjust based on response type)

---

For more information, see:
- [README.md](README.md) - Overview and examples
- [CLAUDE.md](CLAUDE.md) - Technical reference for developers
- [examples/my-testing.py](/examples/my-testing.py) - Working example
