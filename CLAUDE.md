# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**claude-agent-graph** is a Python package enabling creation and orchestration of large-scale graphs where each node represents an independent Claude agent session. The system leverages the claude-agent-sdk to create complex, interconnected networks of AI agents that collaborate and maintain shared state through structured conversation channels.

**Current Status**: v0.1.0-alpha - Core functionality implemented (~85% complete), with 412/418 unit tests passing.

## Quick Reference

### Installation & Setup
```bash
# Install in development mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest                                    # All tests
pytest tests/test_graph.py               # Specific file
pytest tests/test_graph.py::test_add_node # Specific test

# Code quality
black src/ tests/        # Format
ruff check src/ tests/   # Lint
mypy src/                # Type check
```

### Basic Usage Pattern
```python
from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend

# Initialize graph
async with AgentGraph(
    name="my_network",
    storage_backend=FilesystemBackend(base_dir="./conversations")
) as graph:
    # Add nodes
    await graph.add_node("agent1", "You are a coordinator.", model="claude-sonnet-4-20250514")
    await graph.add_node("agent2", "You are a worker.")

    # Add directed edge (agent1 controls agent2)
    await graph.add_edge("agent1", "agent2", directed=True)

    # Send message
    await graph.send_message("agent1", "agent2", "Analyze this data...")

    # Get conversation history
    messages = await graph.get_conversation("agent1", "agent2", limit=10)
```

## Architecture Deep Dive

### Core Components (4,389 LOC)

#### 1. AgentGraph (graph.py - 1,955 lines)
**Main orchestration class managing the entire agent network.**

**Node Operations:**
- `add_node(node_id, system_prompt, model="claude-sonnet-4-20250514", **metadata)` → Node - Async
- `get_node(node_id)` → Node - Raises NodeNotFoundError if missing
- `get_nodes()` → list[Node] - Returns all nodes
- `node_exists(node_id)` → bool
- `update_node(node_id, system_prompt=None, **metadata)` - Updates existing node
- `remove_node(node_id, cascade=True)` - Async, removes node and optionally connected edges
- `node_count` → int - Property

**Edge Operations:**
- `add_edge(from_node, to_node, directed=True, **properties)` - Async
- `get_edge(from_node, to_node)` → Edge
- `get_edges()` → list[Edge]
- `edge_exists(from_node, to_node)` → bool
- `update_edge(from_node, to_node, **properties)`
- `remove_edge(from_node, to_node)` - Async
- `get_neighbors(node_id, direction="outgoing")` → list[str] - direction: "outgoing"/"incoming"/"both"
- `edge_count` → int - Property

**Topology:**
- `get_topology()` → GraphTopology - Detects current topology (TREE, DAG, CHAIN, STAR, CYCLE, UNKNOWN)
- `validate_topology(required_topology)` - Raises TopologyViolationError if constraint violated
- `get_isolated_nodes()` → list[str] - Returns disconnected nodes

**Control Relationships (Epic 4 - Controller Hierarchy):**
- `get_controllers(node_id)` → list[str] - Nodes that control this one (incoming directed edges)
- `get_subordinates(node_id)` → list[str] - Nodes controlled by this one (outgoing directed edges)
- `is_controller(controller_id, subordinate_id)` → bool
- `get_control_relationships()` → dict - Full hierarchy mapping
- `_compute_effective_prompt(node_id)` → str - Injects controller info into agent prompt
- `_mark_subordinates_dirty(controller_id)` - Flags subordinates for prompt recomputation

**Message Routing (Epic 6):**
- `send_message(from_node, to_node, content, **metadata)` - Async, sends user message
- `get_conversation(from_node, to_node, since=None, limit=None)` → list[Message] - Async
- `get_recent_messages(from_node, to_node, count=10)` → list[Message] - Async
- `broadcast(from_node, content, include_incoming=False)` - Async, sends to all neighbors
- `route_message(from_node, to_node, content, path=None)` - Async, multi-hop routing

**Control Commands:**
- `execute_command(controller, subordinate, command, **params)` - Async, authorized commands only

**Agent Lifecycle:**
- `start_agent(node_id)` - Async, activates agent session
- `stop_agent(node_id)` - Async, gracefully stops agent
- `restart_agent(node_id)` - Async, stop then start
- `get_agent_status(node_id)` → dict - Returns status, uptime, message_count, etc.
- `_activate_agent_lazy(node_id)` - Internal lazy activation on first message

**Persistence (Epic 7 - ⚠️ HAS BUGS):**
- `save_checkpoint(filepath)` - Async, saves to msgpack with SHA256 checksum
- `load_checkpoint(filepath)` - Class method, restores graph (⚠️ async/await bug - Issue #2)
- `start_auto_save()` - Background task for periodic saves
- `stop_auto_save()` - Stops auto-save worker
- `load_latest_checkpoint()` - Async, loads most recent checkpoint

**Context Manager:**
- `async with AgentGraph(...) as graph:` - Ensures graceful cleanup

#### 2. Data Models (models.py - 300 lines)

**Message Class:**
```python
@dataclass
class Message:
    from_node: str
    to_node: str
    content: str
    role: MessageRole = MessageRole.USER  # USER, ASSISTANT, SYSTEM
    message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> Message
```

**Node Class:**
```python
@dataclass
class Node:
    node_id: str
    system_prompt: str
    model: str = "claude-sonnet-4-20250514"
    status: NodeStatus = NodeStatus.INITIALIZING  # INITIALIZING, ACTIVE, STOPPED, ERROR
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    # Epic 4 fields for controller injection
    original_system_prompt: str = ""  # Backup before modification
    effective_system_prompt: str = ""  # Prompt with controller info
    prompt_dirty: bool = False  # Flag for recomputation
    agent_session: Any = None  # ClaudeSDKClient reference (not serialized)
```

**Edge Class:**
```python
@dataclass
class Edge:
    from_node: str
    to_node: str
    directed: bool = True  # True = directed, False = bidirectional
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    properties: dict = field(default_factory=dict)

    @property
    def edge_id(self) -> str:
        # Generates deterministic ID from node IDs and direction
```

**Enums:**
- `MessageRole`: USER, ASSISTANT, SYSTEM
- `NodeStatus`: INITIALIZING, ACTIVE, STOPPED, ERROR

#### 3. Storage Layer (storage.py - 408 lines)

**ConversationFile Class:**
Thread-safe JSONL file management for conversation history.
```python
class ConversationFile:
    def __init__(self, filepath: Path, max_size_mb: float = 100.0)

    async def append(self, message: Message) -> None  # Thread-safe append
    async def read(self, since: datetime | None = None, limit: int | None = None) -> list[Message]
    async def rotate(self) -> Path  # Log rotation with timestamp
    async def read_with_archives(self, since: datetime | None = None) -> list[Message]
    def get_size_mb(self) -> float
    def needs_rotation(self) -> bool
```

**StorageBackend (Abstract Base - backends/base.py):**
```python
class StorageBackend(ABC):
    @abstractmethod
    async def append_message(self, edge_id: str, message: Message) -> None

    @abstractmethod
    async def read_messages(self, edge_id: str, since: datetime | None = None, limit: int | None = None) -> list[Message]

    @abstractmethod
    def conversation_exists(self, edge_id: str) -> bool

    @abstractmethod
    async def get_conversation_size(self, edge_id: str) -> float

    @abstractmethod
    async def archive_conversation(self, edge_id: str) -> None
```

**FilesystemBackend (backends/filesystem.py - ~120 lines):**
Concrete implementation storing conversations as JSONL files with auto-rotation.

#### 4. Agent Session Management (agent_manager.py - ~250 lines)

**AgentSessionManager Class:**
Manages ClaudeSDKClient instances with error recovery.
```python
class AgentSessionManager:
    def __init__(self, graph: AgentGraph)

    async def create_session(self, node_id: str) -> ClaudeSDKClient
        # Creates session with effective_system_prompt (controller-injected)
        # ⚠️ Bug: doesn't pass node.metadata to ClaudeAgentOptions (Issue #1)

    async def get_session(self, node_id: str) -> ClaudeSDKClient
        # Gets existing or creates new session

    async def start_agent(self, node_id: str) -> None
        # Enters context manager, updates node status to ACTIVE

    async def stop_agent(self, node_id: str) -> None
        # Graceful shutdown, idempotent

    async def restart_agent(self, node_id: str) -> None
        # Stop then start

    def is_running(self, node_id: str) -> bool

    async def stop_all() -> None
```

**Error Recovery:** Exponential backoff with 3 retries (1s, 2s, 4s delays).

#### 5. Execution Modes (execution.py - ~250 lines)

**ManualController:**
Step-by-step orchestration for explicit control.
```python
class ManualController(ExecutionMode):
    async def step(self, node_id: str) -> Any
        # Execute one turn for specific agent

    async def step_all(self) -> dict[str, Any]
        # Execute all agents with pending messages
```

**ReactiveExecutor:**
Message-driven automatic execution.
```python
class ReactiveExecutor(ExecutionMode):
    # Agents automatically respond to incoming messages
    # Processes message queue continuously
```

**ProactiveExecutor:**
Periodic agent activation.
```python
class ProactiveExecutor(ExecutionMode):
    def __init__(self, graph: AgentGraph, activation_interval: float = 60.0)
    # Agents initiate conversations on schedule
```

#### 6. Checkpoint/Persistence (checkpoint.py - ~250 lines) ⚠️ BROKEN

**Checkpoint Class:**
```python
@dataclass
class Checkpoint:
    name: str
    nodes: list[dict]
    edges: list[dict]
    metadata: dict
    timestamp: datetime
    version: str = "1.0"

    def save(self, filepath: Path) -> None
        # Saves to msgpack with SHA256 checksum

    @classmethod
    def load(cls, filepath: Path) -> Checkpoint
        # Loads and validates checksum
        # ⚠️ Bug: Doesn't await async add_node/add_edge (Issue #2)
```

**Known Issues (Issue #2, #5):** Checkpoint loading doesn't properly restore nodes/edges due to missing `await` on async methods. This blocks entire Epic 7.

#### 7. Transaction Logging (transactions.py - ~150 lines)

**Operation Dataclass:**
Records single graph operation for audit/rollback.
```python
@dataclass
class Operation:
    timestamp: datetime
    operation_type: str  # "add_node", "remove_node", "add_edge", etc.
    node_id: str | None
    from_node: str | None
    to_node: str | None
    success: bool
    error: str | None
    data: dict
```

**TransactionLog Class:**
Append-only JSONL transaction log for operation history and state recovery.

#### 8. Topology Validation (topology.py - 337 lines)

**GraphTopology Enum:**
EMPTY, SINGLE_NODE, TREE, DAG, CHAIN, STAR, CYCLE, UNKNOWN

**Validation Functions:**
```python
has_cycles(graph: nx.DiGraph) -> bool
is_connected(graph: nx.DiGraph) -> bool
is_tree(graph: nx.DiGraph) -> bool
is_dag(graph: nx.DiGraph) -> bool
is_chain(graph: nx.DiGraph) -> bool
is_star(graph: nx.DiGraph) -> bool
is_cycle_graph(graph: nx.DiGraph) -> bool
detect_topology(graph: nx.DiGraph) -> GraphTopology
validate_topology(graph: nx.DiGraph, required: GraphTopology) -> None
get_root_nodes(graph: nx.DiGraph) -> list[str]
get_leaf_nodes(graph: nx.DiGraph) -> list[str]
get_isolated_nodes(graph: nx.DiGraph) -> list[str]
```

#### 9. Exception Hierarchy (exceptions.py - 54 lines)

```python
AgentGraphError (base)
├── NodeNotFoundError
├── EdgeNotFoundError
├── DuplicateNodeError
├── DuplicateEdgeError
├── TopologyValidationError
├── AgentSessionError
└── CommandAuthorizationError
```

⚠️ **Issue #3:** Duplicate exception `TopologyViolationError` also exists in graph.py.

### Key Architectural Patterns

**1. Control Relationships (Epic 4)**
In directed edges, the source node becomes a "controller" of the target. The controlled node's system prompt is automatically injected with controller information to establish authority hierarchy.

Example:
```python
await graph.add_edge("supervisor", "worker", directed=True)
# worker's effective_system_prompt now includes:
# "Note: You are controlled by the following nodes: ['supervisor']"
```

**2. Shared State via JSONL Conversation Files**
Each edge maintains a `convo.jsonl` file with timestamped messages:
```jsonl
{"timestamp": "2025-11-04T12:00:00.000Z", "from_node": "a", "to_node": "b", "message_id": "msg_abc123", "role": "user", "content": "Hello", "metadata": {}}
{"timestamp": "2025-11-04T12:00:01.500Z", "from_node": "b", "to_node": "a", "message_id": "msg_def456", "role": "assistant", "content": "Hi there", "metadata": {}}
```

**Benefits:**
- Append-only for consistency
- Efficient timestamp filtering
- Human-readable
- Thread-safe concurrent access
- Compatible with stream processing tools

**3. Lazy Agent Activation**
Agents only start when needed (on first message or explicit `start_agent()`), reducing resource usage.

**4. Async/Await Throughout**
All I/O operations use async patterns for concurrent agent management. Critical for 1000+ agent scalability.

**5. Thread-Safe Concurrency**
Uses `asyncio.Lock` for graph modifications and file operations.

## Implementation Status

### ✅ Fully Implemented
- **Epic 1: Foundation** - Package structure, dependencies, data models
- **Epic 2: Graph Construction** - Node/edge CRUD, topology detection
- **Epic 3: State Management** - JSONL conversations, log rotation
- **Epic 4: Agent Integration** - ClaudeSDKClient lifecycle, controller injection
- **Epic 5: Dynamic Operations** - Runtime node/edge modification
- **Epic 6: Execution & Control** - Manual/reactive/proactive modes

### ⚠️ Partially Implemented
- **Epic 7: Persistence** - Checkpointing implemented but broken (Issues #2, #5)

### ❌ Not Implemented
- **Epic 8: Monitoring** - Metrics, telemetry, health checks
- **Epic 9: Documentation** - API docs, tutorials, examples/
- Performance benchmarks for 10K+ node graphs (Issue #11)

## Known Critical Issues

### Priority 0 (Blocking)
**Issue #2:** Checkpoint loading doesn't await async methods
- **Impact:** Nodes/edges not restored from checkpoints
- **Location:** checkpoint.py
- **Blocks:** Epic 7 (persistence)

**Issue #5:** Multiple checkpoint test failures
- **Impact:** Checkpointing system fundamentally broken
- **Tests:** 6 failing in test_checkpoint.py

### Priority 1 (High)
**Issue #1:** Agent metadata not passed to ClaudeAgentOptions
- **Impact:** working_directory and other configs ignored
- **Location:** agent_manager.py:create_session()

**Issue #4:** AgentGraph doesn't accept storage_path parameter
- **Impact:** Requires verbose FilesystemBackend instantiation
- **Location:** graph.py:__init__()

### Priority 2 (Medium)
**Issue #3:** Duplicate exception names (TopologyValidationError vs TopologyViolationError)

**Issue #10:** CLAUDE.md examples don't match actual API

**Issue #11:** No performance benchmarks for large graphs

## Testing

### Test Coverage
- **Total Test Code:** 7,385 lines (11 test files)
- **Source Code:** 4,389 lines
- **Test Pass Rate:** 98.5% (412 passed / 418 total)
- **Failed Tests:** 6 (all in test_checkpoint.py - Epic 7 bugs)

### Test Structure
```
tests/
├── test_graph.py (94 KB) - Graph operations, topology, messaging
├── test_storage.py (35 KB) - ConversationFile, rotation, archives
├── test_checkpoint.py (26 KB) - Persistence (6 FAILING)
├── test_agent_manager.py (20 KB) - Lifecycle, recovery
├── test_transactions.py (20 KB) - Transaction logging
├── test_execution.py (15 KB) - Execution modes
├── test_models.py (15 KB) - Data model validation
├── test_topology.py (15 KB) - Topology detection
├── test_e2e.py (21 KB) - End-to-end workflows
├── test_integration.py (17 KB) - Multi-component integration
└── backends/test_filesystem.py - Backend tests
```

### Running Tests
```bash
# All tests
pytest

# Specific module
pytest tests/test_graph.py -v

# Exclude failing checkpoint tests
pytest -k "not checkpoint"

# Integration tests only
pytest -m integration

# With coverage report
pytest --cov=src/claude_agent_graph --cov-report=html
```

## Dependencies

**Core:**
- `anthropic>=0.40.0` - Claude API
- `claude-agent-sdk>=0.1.5` - Agent sessions
- `aiofiles>=23.0.0` - Async file I/O
- `networkx>=3.0` - Graph algorithms
- `pydantic>=2.0` - Data validation
- `msgpack>=1.0.0` - Binary serialization

**Dev:**
- `pytest>=7.0.0`, `pytest-asyncio>=0.21.0`, `pytest-cov>=4.0.0`
- `black>=23.0.0`, `ruff>=0.1.0`, `mypy>=1.0.0`

## Design Decisions & Rationale

**Q: Why JSONL for conversations?**
A: Append-only format ensures atomic writes, easy parsing, human readability, and stream processing compatibility.

**Q: Why NetworkX?**
A: Mature graph algorithms library with topology validation, path finding, and structure analysis out of the box.

**Q: Why async/await?**
A: claude-agent-sdk uses async patterns, and managing 1000+ concurrent agents requires non-blocking I/O.

**Q: Why msgpack for checkpoints?**
A: Binary format provides compact serialization with schema flexibility and faster load times than JSON.

**Q: Why storage backend abstraction?**
A: Allows users to choose filesystem (simple), database (queryable), or Redis (fast) based on scale.

## Performance Targets

- **10,000+ concurrent agent nodes**
- **Sub-100ms message routing latency**
- **1,000+ messages/second throughput**
- **<100MB memory overhead per 100 agents**
- **99.9% message delivery reliability**

⚠️ **Status:** Not yet validated (Issue #11 - no performance benchmarks).

## Common Patterns & Recipes

### Creating Different Topologies

**Tree/Hierarchy:**
```python
await graph.add_node("ceo", "You are the CEO.")
await graph.add_node("vp_eng", "You are VP Engineering.")
await graph.add_node("vp_sales", "You are VP Sales.")
await graph.add_edge("ceo", "vp_eng", directed=True)
await graph.add_edge("ceo", "vp_sales", directed=True)
```

**DAG Pipeline:**
```python
await graph.add_node("ingest", "You ingest data.")
await graph.add_node("process", "You process data.")
await graph.add_node("validate", "You validate data.")
await graph.add_edge("ingest", "process", directed=True)
await graph.add_edge("process", "validate", directed=True)
```

**Star (Hub-and-Spoke):**
```python
await graph.add_node("dispatcher", "You coordinate tasks.")
for i in range(5):
    await graph.add_node(f"worker_{i}", f"You are worker {i}.")
    await graph.add_edge("dispatcher", f"worker_{i}", directed=True)
```

### Message Patterns

**Point-to-Point:**
```python
await graph.send_message("agent1", "agent2", "Process this task")
```

**Broadcast:**
```python
await graph.broadcast("coordinator", "All hands meeting at 3pm")
```

**Multi-Hop Routing:**
```python
await graph.route_message("a", "c", "Hello", path=["a", "b", "c"])
```

### Execution Modes

**Manual Control:**
```python
controller = ManualController(graph)
await controller.step("agent1")  # Execute one agent
result = await controller.step_all()  # Execute all with pending messages
```

**Reactive (Auto-Respond):**
```python
executor = ReactiveExecutor(graph)
await executor.start()  # Agents auto-respond to messages
# ... do work ...
await executor.stop()
```

**Proactive (Periodic):**
```python
executor = ProactiveExecutor(graph, activation_interval=60.0)
await executor.start()  # Agents initiate conversations every 60s
```

## File Locations Quick Reference

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Main graph class | src/claude_agent_graph/graph.py | 1,955 | ✅ Complete |
| Data models | src/claude_agent_graph/models.py | 300 | ✅ Complete |
| Storage layer | src/claude_agent_graph/storage.py | 408 | ✅ Complete |
| Agent manager | src/claude_agent_graph/agent_manager.py | ~250 | ⚠️ Issue #1 |
| Execution modes | src/claude_agent_graph/execution.py | ~250 | ✅ Complete |
| Checkpointing | src/claude_agent_graph/checkpoint.py | ~250 | ❌ Broken (#2) |
| Topology | src/claude_agent_graph/topology.py | 337 | ✅ Complete |
| Transactions | src/claude_agent_graph/transactions.py | ~150 | ✅ Complete |
| Exceptions | src/claude_agent_graph/exceptions.py | 54 | ⚠️ Issue #3 |
| Filesystem backend | src/claude_agent_graph/backends/filesystem.py | ~120 | ✅ Complete |

## Development Workflow

1. **Make changes** to source files
2. **Run tests:** `pytest tests/test_<module>.py`
3. **Format:** `black src/ tests/`
4. **Lint:** `ruff check src/ tests/`
5. **Type check:** `mypy src/`
6. **Commit** with descriptive message

## Next Steps for v1.0.0

**Blockers:**
1. Fix checkpoint async/await bug (Issue #2)
2. Fix metadata passing to agent options (Issue #1)

**High Priority:**
3. Add storage_path convenience parameter (Issue #4)
4. Consolidate exception naming (Issue #3)
5. Update documentation to match API (Issue #10)

**Before Release:**
6. Add performance benchmarks (Issue #11)
7. Create examples/ directory with working demos
8. Epic 8: Monitoring & telemetry
9. Epic 9: Complete API documentation
10. Achieve >80% test coverage

## External Resources

- **PRD.md** - Complete product requirements document
- **IMPLEMENTATION_PLAN.md** - 9 epics broken into features and stories
- **ISSUES_TODO.md** - Detailed bug reports from testing
- **Epic-specific docs:** EPIC_4_IMPLEMENTATION.md, EPIC_5_IMPLEMENTATION.md, EPIC_6_IMPLEMENTATION.md
