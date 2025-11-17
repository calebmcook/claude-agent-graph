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

The project uses a **three-tier hybrid testing strategy**:

#### Tier 1: Unit & Integration Tests (Fast - ~1-2s per test)
```bash
# Run Flask API unit tests
pytest tests/test_flask_api.py -v

# Run with coverage
pytest tests/test_flask_api.py --cov=app --cov-report=html

# Run all unit tests
pytest tests/ -k "not e2e" -v
```

#### Tier 2: End-to-End Tests (Playwright - ~15-30s per test)
```bash
# Install Playwright browsers (one-time)
playwright install

# Start Flask server
python3 app.py &

# Run E2E tests in headless mode
pytest tests/test_flask_e2e_playwright.py -v

# Run with visible browser
pytest tests/test_flask_e2e_playwright.py --headed -v
```

#### Tier 3: Manual Testing (Real Claude API)
```bash
# Start Flask development server with hot reload
FLASK_ENV=development python3 app.py

# Open http://localhost:5001 in browser for interactive testing
```

**📖 See [HYBRID_TESTING_STRATEGY.md](HYBRID_TESTING_STRATEGY.md) and [TESTING_SETUP.md](TESTING_SETUP.md) for comprehensive testing guides.**

### Code Quality
```bash
# Format code
black src/ tests/ app.py

# Lint code
ruff check src/ tests/ app.py

# Type checking
mypy src/ --ignore-missing-imports
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup git hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
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

The project is currently in planning phase with:
- ✅ Complete PRD defining all requirements
- ✅ Detailed implementation plan with 9 epics broken into features and user stories
- ⏳ Implementation not yet started

Reference `IMPLEMENTATION_PLAN.md` for the full development roadmap organized into phases:
1. Phase 1 (MVP): Project foundation, graph construction, state management, agent integration
2. Phase 2: Dynamic operations
3. Phase 3: Advanced execution modes
4. Phase 4: Production readiness (persistence, monitoring, documentation)

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

- **Unit tests**: Individual components (Node, Edge, Message, ConversationFile)
- **Integration tests**: Multi-agent interactions, graph modifications, persistence
- **Performance tests**: Scalability to 1000+ nodes, message throughput, latency
- **End-to-end tests**: Complete workflows, crash recovery scenarios

Target: >80% test coverage before v1.0.0 release.

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