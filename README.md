# claude-agent-graph

<div align="center">

**Build complex multi-agent AI systems with graph-structured Claude agents**

[![Tests](https://github.com/calebmcook/claude-agent-graph/actions/workflows/test.yml/badge.svg)](https://github.com/calebmcook/claude-agent-graph/actions/workflows/test.yml)
[![Documentation](https://github.com/calebmcook/claude-agent-graph/actions/workflows/docs.yml/badge.svg)](https://github.com/calebmcook/claude-agent-graph/actions/workflows/docs.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Create and orchestrate large-scale graphs of Claude AI agents using the [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk).

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Topology Examples](#graph-topologies--use-cases) • [Documentation](#documentation)

</div>

---

## Overview

**claude-agent-graph** is a Python package that enables creation and orchestration of large-scale graphs where each node represents an independent Claude agent session. Build complex multi-agent systems with hierarchical control, shared state, and flexible topologies.

### Key Capabilities

- 🕸️ **Graph-Based Agent Networks** - Create complex topologies (trees, DAGs, meshes, chains, stars, cycles)
- 🔄 **Shared Conversation State** - Agents communicate via persistent JSONL conversation files
- 🎯 **Control Relationships** - Hierarchical controller-subordinate patterns with automatic prompt injection
- ⚡ **Dynamic Topology** - Add/remove nodes and edges at runtime
- 🚀 **Multiple Execution Modes** - Reactive, proactive, and manual control patterns
- 💾 **Persistence & Recovery** - Checkpoint graph state for crash recovery
- 🔒 **Type-Safe** - Full type hints and Pydantic validation
- ⏱️ **Async-First** - Built on async/await for high concurrency (1000+ agents)

---

## Installation

### From Source (Development)

```bash
git clone https://github.com/calebmcook/claude-agent-graph.git
cd claude-agent-graph
pip install -e .
```

### Requirements

- Python 3.10+
- `anthropic>=0.40.0`
- `claude-agent-sdk>=0.1.5`
- `networkx>=3.0`
- `pydantic>=2.0`
- `aiofiles>=23.0.0`

---

## Quick Start

```python
import asyncio
from claude_agent_graph import AgentGraph
from claude_agent_graph.backends import FilesystemBackend

async def main():
    # Create graph with filesystem storage
    async with AgentGraph(
        name="my_network",
        storage_backend=FilesystemBackend(base_dir="./conversations")
    ) as graph:
        # Add agent nodes
        await graph.add_node(
            "coordinator",
            "You coordinate tasks and delegate to workers.",
            model="claude-sonnet-4-20250514"
        )

        await graph.add_node(
            "worker",
            "You execute tasks assigned to you.",
            model="claude-sonnet-4-20250514"
        )

        # Create directed edge (coordinator controls worker)
        await graph.add_edge("coordinator", "worker", directed=True)

        # Send message
        await graph.send_message(
            "coordinator",
            "worker",
            "Please analyze the latest sales data and summarize key trends."
        )

        # Get conversation history
        messages = await graph.get_conversation("coordinator", "worker", limit=10)
        for msg in messages:
            print(f"{msg.from_node} → {msg.to_node}: {msg.content}")

asyncio.run(main())
```

---

## Graph Topologies & Use Cases

### 1. Tree/Hierarchy - Organizational Structure

**Use Case:** Corporate org chart, project management, code review workflows

```python
# Create CEO → VPs → Managers → Workers hierarchy
async with AgentGraph(name="org_chart") as graph:
    # Top level
    await graph.add_node("ceo", "You are the CEO. Coordinate VPs.")

    # VP level
    await graph.add_node("vp_engineering", "You lead the engineering team.")
    await graph.add_node("vp_sales", "You lead the sales team.")

    # Manager level
    await graph.add_node("eng_manager", "You manage software engineers.")
    await graph.add_node("sales_manager", "You manage sales reps.")

    # Worker level
    await graph.add_node("engineer_1", "You write code.")
    await graph.add_node("sales_rep_1", "You handle customer accounts.")

    # Build hierarchy
    await graph.add_edge("ceo", "vp_engineering", directed=True)
    await graph.add_edge("ceo", "vp_sales", directed=True)
    await graph.add_edge("vp_engineering", "eng_manager", directed=True)
    await graph.add_edge("vp_sales", "sales_manager", directed=True)
    await graph.add_edge("eng_manager", "engineer_1", directed=True)
    await graph.add_edge("sales_manager", "sales_rep_1", directed=True)

    # Verify topology
    assert graph.get_topology() == GraphTopology.TREE
```

**Benefits:**
- Clear chain of command
- Automatic prompt injection ("You are controlled by: ['vp_engineering']")
- Scalable delegation patterns

---

### 2. DAG Pipeline - Sequential Data Processing

**Use Case:** ETL pipelines, content workflows, CI/CD orchestration

```python
# Data processing pipeline: Ingest → Transform → Validate → Store
async with AgentGraph(name="data_pipeline") as graph:
    await graph.add_node("ingester", "You ingest raw data from sources.")
    await graph.add_node("transformer", "You clean and transform data.")
    await graph.add_node("validator", "You validate data quality.")
    await graph.add_node("storage", "You store validated data.")

    # Create DAG (no cycles)
    await graph.add_edge("ingester", "transformer", directed=True)
    await graph.add_edge("transformer", "validator", directed=True)
    await graph.add_edge("validator", "storage", directed=True)

    # Verify topology (linear pipeline detected as CHAIN, which is a type of DAG)
    topology = graph.get_topology()
    assert topology in (GraphTopology.DAG, GraphTopology.CHAIN)

    # Start pipeline
    await graph.send_message("ingester", "transformer", "New batch: batch_123.csv")
```

**Benefits:**
- Clear dependencies (no cycles)
- Easy to reason about data flow
- Supports parallel branches (fork-join)

---

### 3. Star (Hub-and-Spoke) - Task Distribution

**Use Case:** Load balancing, task dispatching, microservices orchestration

```python
# Central dispatcher coordinates multiple workers
async with AgentGraph(name="task_dispatcher") as graph:
    # Central hub
    await graph.add_node(
        "dispatcher",
        "You distribute tasks to available workers based on load."
    )

    # Create 10 worker spokes
    for i in range(10):
        await graph.add_node(f"worker_{i}", f"You are worker {i}. Execute assigned tasks.")
        # Star topology: edges FROM spokes TO hub (workers report to dispatcher)
        # Note: edges FROM hub would create a tree topology instead
        await graph.add_edge(f"worker_{i}", "dispatcher", directed=True)

    assert graph.get_topology() == GraphTopology.STAR

    # Workers report status to dispatcher
    await graph.send_message("worker_0", "dispatcher", "Task completed successfully")
```

**Benefits:**
- Centralized coordination
- Easy to add/remove workers
- Single point of control

---

### 4. Chain - Sequential Processing

**Use Case:** Assembly line, approval workflows, sequential transformations

```python
# Document approval chain: Draft → Review → Edit → Approve → Publish
async with AgentGraph(name="approval_chain") as graph:
    await graph.add_node("drafter", "You write initial drafts.")
    await graph.add_node("reviewer", "You review for accuracy.")
    await graph.add_node("editor", "You polish language and style.")
    await graph.add_node("approver", "You give final approval.")
    await graph.add_node("publisher", "You publish approved content.")

    # Create linear chain
    await graph.add_edge("drafter", "reviewer", directed=True)
    await graph.add_edge("reviewer", "editor", directed=True)
    await graph.add_edge("editor", "approver", directed=True)
    await graph.add_edge("approver", "publisher", directed=True)

    assert graph.get_topology() == GraphTopology.CHAIN

    # Start document flow
    await graph.send_message("drafter", "reviewer", "Draft of Q4 report attached.")
```

**Benefits:**
- Simple, predictable flow
- Easy to audit (clear sequence)
- Natural for approval processes

---

### 5. Mesh (Fully Connected) - Collaborative Team

**Use Case:** Brainstorming, peer review, consensus building, research teams

```python
# Research team where everyone can talk to everyone
async with AgentGraph(name="research_team") as graph:
    researchers = ["alice", "bob", "carol", "dave"]

    # Add researchers
    for name in researchers:
        await graph.add_node(name, f"You are researcher {name}. Collaborate on papers.")

    # Create bidirectional edges between all pairs
    for i, r1 in enumerate(researchers):
        for r2 in researchers[i+1:]:
            await graph.add_edge(r1, r2, directed=False)  # Bidirectional

    # Collaborative discussion
    await graph.send_message("alice", "bob", "What do you think of approach X?")
    await graph.send_message("carol", "dave", "I found interesting data on Y.")
```

**Benefits:**
- Maximum collaboration
- No hierarchy constraints
- Democratic decision-making

---

### 6. Cycle - Iterative Refinement

**Use Case:** Quality improvement loops, game AI, iterative design

```python
# Iterative refinement: Generate → Critique → Refine → Generate...
async with AgentGraph(name="iterative_refinement") as graph:
    await graph.add_node("generator", "You generate creative content.")
    await graph.add_node("critic", "You provide constructive criticism.")
    await graph.add_node("refiner", "You refine based on feedback.")

    # Create cycle
    await graph.add_edge("generator", "critic", directed=True)
    await graph.add_edge("critic", "refiner", directed=True)
    await graph.add_edge("refiner", "generator", directed=True)

    assert graph.get_topology() == GraphTopology.CYCLE

    # Start iterative loop
    await graph.send_message("generator", "critic", "Here's draft v1...")
```

**Benefits:**
- Continuous improvement
- Iterative refinement
- Self-correcting systems

---

### 7. Bipartite - Two-Layer Architecture

**Use Case:** Review systems, quality control, map-reduce patterns

```python
# Writers create content, reviewers critique it
async with AgentGraph(name="content_review") as graph:
    # Layer 1: Writers
    writers = ["writer_1", "writer_2", "writer_3"]
    for w in writers:
        await graph.add_node(w, f"You are {w}. Create content.")

    # Layer 2: Reviewers
    reviewers = ["reviewer_a", "reviewer_b"]
    for r in reviewers:
        await graph.add_node(r, f"You are {r}. Review content quality.")

    # Connect all writers to all reviewers (bidirectional)
    for w in writers:
        for r in reviewers:
            await graph.add_edge(w, r, directed=False)

    # Submit content for review
    await graph.send_message("writer_1", "reviewer_a", "Article draft attached")
```

**Benefits:**
- Clear separation of concerns
- Multiple reviewers per item
- Scalable review process

---

### 8. Dynamic Workflow - Runtime Modification

**Use Case:** Adaptive systems, on-demand scaling, dynamic task allocation

```python
# Start small, grow based on load
async with AgentGraph(name="adaptive_system") as graph:
    await graph.add_node("manager", "You manage worker allocation.")
    await graph.add_node("worker_1", "You execute tasks.")
    await graph.add_edge("manager", "worker_1", directed=True)

    # Simulate load increase - add workers dynamically
    for i in range(2, 6):
        if should_scale_up():  # Your scaling logic
            await graph.add_node(f"worker_{i}", f"You are worker {i}.")
            await graph.add_edge("manager", f"worker_{i}", directed=True)

    print(f"Scaled to {graph.node_count} nodes")

    # Remove workers when load decreases
    if should_scale_down():
        await graph.remove_node("worker_5", cascade=True)
```

**Benefits:**
- Adapt to changing requirements
- Resource optimization
- Runtime flexibility

---

## Core Features

### Control Relationships

In directed edges, the source node automatically becomes a "controller" of the target node. The system injects controller information into subordinate prompts:

```python
await graph.add_edge("supervisor", "worker", directed=True)

# Worker's effective prompt now includes:
# "Note: You are controlled by the following nodes: ['supervisor']"
# This establishes authority hierarchy automatically.

# Query control relationships
controllers = graph.get_controllers("worker")  # ['supervisor']
subordinates = graph.get_subordinates("supervisor")  # ['worker']
```

### Shared Conversation State

Each edge maintains a persistent JSONL file with timestamped messages:

```python
# Send message
await graph.send_message("agent_a", "agent_b", "Hello")

# Read conversation
messages = await graph.get_conversation("agent_a", "agent_b", limit=10)

# Read recent messages since timestamp
from datetime import datetime, timezone
since = datetime(2025, 11, 4, 12, 0, 0, tzinfo=timezone.utc)
recent = await graph.get_conversation("agent_a", "agent_b", since=since)
```

### Message Routing Patterns

**Point-to-Point:**
```python
await graph.send_message("a", "b", "Direct message")
```

**Broadcast to neighbors:**
```python
await graph.broadcast("coordinator", "Meeting at 3pm", include_incoming=False)
```

**Multi-hop routing:**
```python
# Route message through intermediate nodes
await graph.route_message("a", "c", "Hello", path=["a", "b", "c"])
```

### Execution Modes

**Manual Control** - Step-by-step execution:
```python
from claude_agent_graph.execution import ManualController

controller = ManualController(graph)
await controller.step("agent_1")  # Execute one agent
result = await controller.step_all()  # Execute all with pending messages
```

**Reactive** - Agents auto-respond to messages:
```python
from claude_agent_graph.execution import ReactiveExecutor

executor = ReactiveExecutor(graph)
await executor.start()
# Agents automatically process incoming messages
await executor.stop()
```

**Proactive** - Periodic activation:
```python
from claude_agent_graph.execution import ProactiveExecutor

executor = ProactiveExecutor(graph, activation_interval=60.0)
await executor.start()
# Agents initiate conversations every 60 seconds
```

### Persistence & Recovery

```python
# Save checkpoint
await graph.save_checkpoint("./checkpoints/my_graph.msgpack")

# Load checkpoint (restore graph state)
graph = await AgentGraph.load_checkpoint("./checkpoints/my_graph.msgpack")

# Auto-save every 5 minutes
graph = AgentGraph(name="autosave_example", auto_save=True, auto_save_interval=300)
```

### Topology Validation

```python
from claude_agent_graph.topology import GraphTopology

# Detect current topology
topology = graph.get_topology()
print(topology)  # GraphTopology.TREE

# Enforce topology constraints
graph = AgentGraph(name="strict_dag", topology_constraint="dag")
await graph.add_node("a", "Node A")
await graph.add_node("b", "Node B")
await graph.add_edge("a", "b", directed=True)
await graph.add_edge("b", "a", directed=True)  # Raises TopologyViolationError (creates cycle)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AgentGraph                              │
│  (Orchestration, Topology, Control Relationships)               │
└────────────┬────────────────────────────────────────────────────┘
             │
      ┌──────┴──────┬──────────────┬──────────────┬──────────────┐
      │             │              │              │              │
┌─────▼─────┐ ┌────▼────┐  ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼────┐
│  Node     │ │  Edge   │  │  Storage    │ │  Agent     │ │ Topology│
│  (Agent)  │ │ (Conn)  │  │  Backend    │ │  Manager   │ │ Validator│
└───────────┘ └─────────┘  └─────────────┘ └────────────┘ └─────────┘
      │             │              │              │
      │             │              │              │
      └─────────────┴──────────────┴──────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         ┌────▼─────┐        ┌─────▼──────┐
         │ JSONL    │        │ ClaudeSDK  │
         │ Convo    │        │ Client     │
         │ Files    │        │ (Anthropic)│
         └──────────┘        └────────────┘
```

**Key Components:**
- **AgentGraph** - Main orchestration class (1,955 LOC)
- **Node** - Individual agent with system prompt and ClaudeSDKClient session
- **Edge** - Connection between agents (directed/undirected)
- **StorageBackend** - Abstract interface for conversation persistence (JSONL/DB/Redis)
- **AgentSessionManager** - ClaudeSDKClient lifecycle management
- **Topology** - Graph structure validation (NetworkX)

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/claude_agent_graph --cov-report=html

# Run specific test file
pytest tests/test_graph.py -v

# Exclude failing checkpoint tests
pytest -k "not checkpoint"
```

### Test Coverage

- **412/418 tests passing** (98.5%)
- **7,385 lines** of test code
- **~85%** estimated coverage
- 6 failing tests in Epic 7 (checkpoint persistence - known bugs)

---

## Performance Targets

- ✅ Support **10,000+ concurrent agent nodes**
- ✅ Sub-**100ms message routing latency**
- ✅ **1,000+ messages/second** throughput
- ✅ **<100MB memory** overhead per 100 agents
- ✅ **99.9%** message delivery reliability

*Note: Benchmarks pending (see [Issue #11](ISSUES_TODO.md#issue-11))*

---

## Project Status

**Current Version:** v0.1.0-alpha

**Implementation Status:**
- ✅ Core graph operations (nodes, edges, topology)
- ✅ Agent lifecycle management
- ✅ Message routing and conversation state
- ✅ Control relationships and prompt injection
- ✅ Execution modes (manual, reactive, proactive)
- ⚠️ Persistence/checkpointing (has bugs - see [ISSUES_TODO.md](ISSUES_TODO.md))
- ❌ Monitoring and telemetry (not implemented)
- ❌ API documentation (in progress)

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the complete roadmap.

---

## Documentation

- [**CLAUDE.md**](CLAUDE.md) - Comprehensive technical reference
- [**IMPLEMENTATION_PLAN.md**](IMPLEMENTATION_PLAN.md) - 9 epics with detailed features
- [**PRD.md**](PRD.md) - Product requirements document
- [**ISSUES_TODO.md**](ISSUES_TODO.md) - Known bugs and issues

---

## Contributing

This project is in active development. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black src/ tests/`)
6. Submit a pull request

See [ISSUES_TODO.md](ISSUES_TODO.md) for known issues and priorities.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Links

- **Repository:** https://github.com/calebmcook/claude-agent-graph
- **Issues:** https://github.com/calebmcook/claude-agent-graph/issues
- **claude-agent-sdk:** https://github.com/anthropics/claude-agent-sdk
- **Anthropic Claude:** https://www.anthropic.com/claude

---

## Citation

If you use claude-agent-graph in your research, please cite:

```bibtex
@software{claude_agent_graph,
  title = {claude-agent-graph: Graph-Structured Multi-Agent AI Systems},
  author = {Cook, Caleb},
  year = {2025},
  url = {https://github.com/calebmcook/claude-agent-graph}
}
```

---

<div align="center">

**Built with ❤️ using [Claude](https://www.anthropic.com/claude) and [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk)**

</div>
