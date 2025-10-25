# claude-agent-graph

Create and orchestrate large-scale graphs of Claude AI agents using the [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk).

## Overview

**claude-agent-graph** is a Python package that enables creation and orchestration of large-scale graphs where each node represents an independent Claude agent session. The system leverages the claude-agent-sdk to create complex, interconnected networks of AI agents that collaborate and maintain shared state through structured conversation channels.

## Features

- **Graph-based Agent Networks**: Create complex agent topologies (trees, DAGs, meshes, etc.)
- **Shared State Management**: Agents communicate via persistent conversation files (JSONL format)
- **Dynamic Topology**: Add/remove nodes and edges at runtime
- **Multiple Execution Modes**: Reactive, proactive, and manual control patterns
- **Type-safe**: Full type hints and Pydantic validation
- **Async-first**: Built on async/await for high concurrency

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/calebmcook/claude-agent-graph.git
cd claude-agent-graph

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### From PyPI (Coming Soon)

```bash
pip install claude-agent-graph
```

## Quick Start

```python
from claude_agent_graph import AgentGraph

# Create a graph
graph = AgentGraph(name="my_network")

# Add nodes (agents)
await graph.add_node(
    node_id="supervisor",
    system_prompt="You coordinate worker agents.",
    model="claude-sonnet-4-20250514"
)

await graph.add_node(
    node_id="worker_1",
    system_prompt="You execute tasks.",
    model="claude-sonnet-4-20250514"
)

# Add edge (connection)
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

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/claude_agent_graph --cov-report=html

# Run specific test file
pytest tests/test_graph.py
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

## Project Status

⚠️ **This project is currently in early development (v0.1.0-alpha)**

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the detailed development roadmap.

## Documentation

- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed development roadmap
- [Product Requirements](PRD.md) - Complete product specification
- [Claude Code Guide](CLAUDE.md) - Development guidelines for AI assistance

## Contributing

This project is in active development. Contributions welcome! Please see the implementation plan for current priorities.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- **Repository**: https://github.com/calebmcook/claude-agent-graph
- **Issues**: https://github.com/calebmcook/claude-agent-graph/issues
