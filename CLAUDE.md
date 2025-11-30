# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**claude-agent-graph** is a Python package that enables creation and orchestration of large-scale graphs where each node represents an independent Claude agent session. The system leverages the claude-agent-sdk to create complex, interconnected networks of AI agents that collaborate and maintain shared state through structured conversation channels.

## Development Commands

### Setup
```bash
# Install package in development mode (when pyproject.toml is created)
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/claude_agent_graph --cov-report=html

# Run specific test file
pytest tests/test_graph.py

# Run specific test
pytest tests/test_graph.py::test_add_node
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

### Core Components

1. **AgentGraph** (`src/claude_agent_graph/graph.py`)
   - Main orchestration class managing the entire agent network
   - Handles graph construction, topology validation, and dynamic modifications
   - Coordinates node/edge lifecycle and message routing

2. **Data Models** (`src/claude_agent_graph/models.py`)
   - **Node**: Represents an individual agent with system prompt, model config, and ClaudeSDKClient session
   - **Edge**: Represents connections between agents with directed/undirected relationships
   - **SharedState**: Manages conversation files (convo.jsonl) and inter-agent state
   - **Message**: Structured message format with timestamp, sender, receiver, and metadata

3. **Storage Layer** (`src/claude_agent_graph/storage.py`, `src/claude_agent_graph/backends/`)
   - **ConversationFile**: Thread-safe JSONL file management for agent conversations
   - **StorageBackend**: Abstract interface for multiple storage backends (filesystem, database, Redis)
   - Handles persistence, log rotation, and state synchronization

4. **Agent Management** (`src/claude_agent_graph/agent_manager.py`)
   - Creates and manages ClaudeSDKClient instances for each node
   - Handles agent lifecycle (start, stop, restart)
   - Implements error recovery and status tracking

5. **Topology** (`src/claude_agent_graph/topology.py`)
   - Graph structure validation using NetworkX
   - Supports: trees, DAGs, meshes, chains, stars, cycle graphs
   - Detects cycles, validates connectivity, ensures structural constraints

6. **Execution** (`src/claude_agent_graph/execution.py`)
   - **ReactiveExecutor**: Agents respond to incoming messages
   - **ProactiveExecutor**: Agents initiate conversations periodically
   - **ManualController**: External orchestration with step-by-step control

### Key Architectural Patterns

**Control Relationships**: In directed edges, the source node becomes a "controller" of the target node. The controlled node's system prompt is automatically injected with controller information to establish authority hierarchy.

**Shared State via Conversation Files**: Each edge maintains a `convo.jsonl` file containing timestamped messages. This JSONL format enables:
- Append-only writes for consistency
- Efficient filtering by timestamp
- Durable conversation history
- Thread-safe concurrent access

**Dynamic Graph Modification**: The system supports runtime addition/removal of nodes and edges with:
- Transaction logging for rollback capability
- Graceful agent session termination
- Automatic conversation file archiving
- System prompt updates for affected nodes

## Implementation Status

**Current Version:** v0.1.0-alpha

The project has completed core implementation with comprehensive testing:
- ✅ Complete PRD defining all requirements
- ✅ Detailed implementation plan with 9 epics (largely complete)
- ✅ Core implementation complete (Epics 1-6)
- ✅ Comprehensive test suite: **467/483 tests passing (96.7%)**
- ⚠️ Persistence layer (Epic 7): Core working, edge cases remaining
- ❌ Monitoring/telemetry (Epic 8): Not implemented

**Test Coverage (Issue #24):**
- All basic functionality from README validated
- All 8 topologies tested and working
- All execution modes (manual, reactive, proactive) tested
- Message routing (point-to-point, broadcast, multi-hop) validated
- Dynamic graph operations fully tested
- Concurrency support verified

Reference `IMPLEMENTATION_PLAN.md` and `TEST_REPORT.md` for detailed status:
1. Phase 1 (MVP): ✅ COMPLETE
2. Phase 2 (Dynamic ops): ✅ COMPLETE
3. Phase 3 (Advanced execution): ✅ COMPLETE
4. Phase 4 (Production): ⚠️ IN PROGRESS (persistence working, needs polish)

## Key Dependencies

- **claude-agent-sdk** (>=1.0.0): Core agent session functionality
- **anthropic** (>=0.40.0): Claude API access
- **aiofiles** (>=23.0.0): Async file I/O for conversation files
- **networkx** (>=3.0): Graph algorithms and topology validation
- **pydantic** (>=2.0): Data validation and serialization

## Design Decisions

**Why JSONL for conversation files?**: Append-only format provides atomic writes, easy parsing, human readability, and compatibility with stream processing tools.

**Why NetworkX?**: Mature graph algorithms library that provides topology validation, path finding, and structure analysis out of the box.

**Why async/await throughout?**: The claude-agent-sdk uses async patterns, and managing hundreds/thousands of concurrent agents requires non-blocking I/O.

**Storage backend abstraction**: Allows users to choose between filesystem (simple), database (queryable), or Redis (fast) based on their scale and requirements.

## Testing Strategy

### Test Categories (Issue #24 - Complete)

**Unit Tests** ✅ (420+ tests)
- Individual components (Node, Edge, Message, ConversationFile)
- Data model validation and serialization
- Storage backend operations

**Integration Tests** ✅ (26 tests)
- Multi-agent interactions (22 passing, 4 checkpoint-related failures)
- Graph modifications and topology changes
- Message routing and conversation state
- Control relationships and prompt injection

**Topology Tests** ✅ (8+ topologies validated)
- Tree/Hierarchy: ✅ PASSING
- DAG Pipeline: ⚠️ 1 edge case failing
- Star (Hub-and-Spoke): ✅ PASSING
- Chain: ✅ PASSING
- Mesh (Fully Connected): ✅ PASSING
- Cycle: ✅ PASSING
- Bipartite: ✅ PASSING
- Dynamic Workflow: ✅ PASSING

**Execution Mode Tests** ✅ (All working)
- Manual Controller: ✅ PASSING
- Reactive Executor: ✅ PASSING
- Proactive Executor: ✅ PASSING

**End-to-End Tests** ✅ (12 tests)
- Complete workflows (supervisor-worker, research teams)
- Dynamic graph evolution
- Large-scale operations

### Current Coverage
- **Total Tests:** 483
- **Passing:** 467 (96.7%) ✅
- **Failing:** 16 (3.3%) - mostly checkpoint edge cases
- **Target:** >80% (EXCEEDED ✅)

See `TEST_REPORT.md` for comprehensive testing details, methodology, and recommendations.

## Directory Structure (planned)

```
src/claude_agent_graph/
├── __init__.py              # Package exports
├── graph.py                 # Main AgentGraph class
├── models.py                # Data models (Node, Edge, Message, SharedState)
├── topology.py              # Graph topology validation
├── storage.py               # Conversation file management
├── agent_manager.py         # ClaudeSDKClient lifecycle
├── execution.py             # Execution modes (reactive, proactive, manual)
├── transactions.py          # Transaction logging and rollback
├── visualization.py         # Graph export (GraphViz, JSON)
└── backends/
    ├── __init__.py
    ├── base.py              # StorageBackend interface
    ├── filesystem.py        # Filesystem backend
    ├── database.py          # SQL/NoSQL backend (optional)
    └── redis.py             # Redis backend (optional)

tests/
├── test_models.py
├── test_graph.py
├── test_topology.py
├── test_storage.py
├── test_agent_manager.py
├── test_execution.py
└── backends/
    └── test_filesystem.py

examples/
├── simple_hierarchy.py      # Supervisor + workers
├── collaborative_network.py # Research team mesh
└── dynamic_workflow.py      # Runtime node creation

docs/
├── getting_started.md
├── user_guide.md
├── concepts.md
└── api/
```

## Common Patterns

### Creating a Graph
```python
from claude_agent_graph import AgentGraph

graph = AgentGraph(
    name="my_network",
    storage_backend="filesystem",
    max_nodes=1000
)

# Add nodes
supervisor = await graph.add_node(
    node_id="supervisor",
    system_prompt="You coordinate worker agents.",
    model="claude-sonnet-4-20250514"
)

worker = await graph.add_node(
    node_id="worker_1",
    system_prompt="You execute tasks.",
    model="claude-sonnet-4-20250514"
)

# Add directed edge (supervisor -> worker)
await graph.add_edge(
    from_node="supervisor",
    to_node="worker_1",
    directed=True
)

# Send message
await graph.send_message(
    from_node="supervisor",
    to_node="worker_1",
    content="Analyze the data"
)
```

### Message Structure
All inter-agent messages follow this format in conversation files:
```json
{
  "timestamp": "2025-10-25T12:34:56.789Z",
  "from_node": "agent_1",
  "to_node": "agent_2",
  "message_id": "msg_abc123",
  "role": "user",
  "content": "Message content",
  "metadata": {}
}
```

## Performance Targets

- Support 10,000+ concurrent agent nodes
- Sub-100ms message routing latency
- 1000+ messages/second throughput
- <100MB memory overhead per 100 agents
- 99.9% message delivery reliability
- always allow python and bash
- Always clean up old background Flask server processes when not in use.