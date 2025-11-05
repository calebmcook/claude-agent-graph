# claude-agent-graph Examples

This directory contains practical, runnable examples demonstrating various features and topologies of claude-agent-graph.

## Running Examples

All examples require:
- Python 3.10+
- `claude-agent-graph` installed (`pip install -e .` from repo root)
- `ANTHROPIC_API_KEY` environment variable set

```bash
export ANTHROPIC_API_KEY=your_api_key_here
python examples/tree_hierarchy.py
```

## Available Examples

### Basic Examples

#### 1. Tree Hierarchy (`tree_hierarchy.py`)
**Complexity:** Basic | **Lines:** ~110 | **Topology:** Tree

Demonstrates a corporate organizational hierarchy with CEO, VPs, managers, and workers.

**Key Concepts:**
- Tree topology creation
- Directed edges for control relationships
- System prompt injection for controllers
- Message flow down hierarchy
- Control relationship queries

**Use Cases:**
- Organizational charts
- Project management hierarchies
- Approval workflows
- Code review chains

**Run:**
```bash
python examples/tree_hierarchy.py
```

---

#### 2. DAG Pipeline (`dag_pipeline.py`)
**Complexity:** Basic | **Lines:** ~150 | **Topology:** DAG

Data processing pipeline with stages: Ingest → Process → Validate → Store

**Key Concepts:**
- DAG (directed acyclic graph) topology
- Sequential data flow
- Stage-by-stage processing
- Message passing through pipeline

**Use Cases:**
- ETL pipelines
- Content workflows
- CI/CD orchestration
- Multi-stage data processing

**Run:**
```bash
python examples/dag_pipeline.py
```

---

#### 3. Star Dispatcher (`star_dispatcher.py`)
**Complexity:** Basic | **Lines:** ~130 | **Topology:** Star

Hub-and-spoke pattern with central dispatcher coordinating multiple workers.

**Key Concepts:**
- Star topology (central hub with spokes)
- Task distribution
- Worker coordination
- Status reporting

**Use Cases:**
- Load balancing
- Task dispatching
- Microservices orchestration
- Central coordination patterns

**Run:**
```bash
python examples/star_dispatcher.py
```

---

### Intermediate Examples

#### 4. Collaborative Network (`collaborative_network.py`)
**Complexity:** Intermediate | **Lines:** ~200 | **Topology:** Mesh

Research team collaboration with multiple agent types working together.

**Key Concepts:**
- Mesh topology (partially connected)
- Broadcast messaging to multiple agents
- Undirected edges for peer relationships
- Multi-agent collaboration
- Message metadata tracking

**Use Cases:**
- Research teams
- Brainstorming sessions
- Peer review systems
- Collaborative decision-making

**Run:**
```bash
python examples/collaborative_network.py
```

---

#### 5. Execution Modes Demo (`execution_modes_demo.py`)
**Complexity:** Intermediate | **Lines:** ~300 | **Modes:** All three

Comprehensive demonstration of all execution modes: Manual, Reactive, and Proactive.

**Key Concepts:**
- Manual mode (step-by-step control)
- Reactive mode (message-driven)
- Proactive mode (periodic activation)
- Mode switching
- Message queue behavior
- Execution patterns

**Use Cases:**
- Understanding execution modes
- Choosing the right orchestration pattern
- Debugging agent workflows
- Performance testing

**Run:**
```bash
python examples/execution_modes_demo.py
```

---

#### 6. Control Commands (`control_commands.py`)
**Complexity:** Intermediate | **Lines:** ~180 | **Topology:** Tree

Hierarchical task delegation with command authorization.

**Key Concepts:**
- `execute_command()` for authorized commands
- Command authorization enforcement
- Control relationship validation
- System prompt injection mechanics
- Command metadata tracking
- Authorization error handling

**Use Cases:**
- Hierarchical task systems
- Authorized command execution
- Delegation patterns
- Audit logging

**Run:**
```bash
python examples/control_commands.py
```

---

## Example Comparison Matrix

| Example | Topology | Complexity | LOC | Key Features |
|---------|----------|------------|-----|--------------|
| `tree_hierarchy.py` | Tree | Basic | 110 | Control relationships, hierarchy |
| `dag_pipeline.py` | DAG | Basic | 150 | Sequential workflow, stages |
| `star_dispatcher.py` | Star | Basic | 130 | Hub-and-spoke, coordination |
| `collaborative_network.py` | Mesh | Intermediate | 200 | Broadcast, collaboration |
| `execution_modes_demo.py` | Chain | Intermediate | 300 | All execution modes |
| `control_commands.py` | Tree | Intermediate | 180 | Commands, authorization |

## Learning Path

**Beginner:** Start here
1. `tree_hierarchy.py` - Understand basic graph construction
2. `dag_pipeline.py` - Learn about workflows and message passing
3. `star_dispatcher.py` - See coordination patterns

**Intermediate:** Progress to
4. `collaborative_network.py` - Explore broadcast and mesh topologies
5. `execution_modes_demo.py` - Master execution modes
6. `control_commands.py` - Understand command authorization

**Advanced:** See the main documentation for:
- Dynamic graph modification
- Custom storage backends
- Checkpointing and recovery
- Performance optimization

## Common Patterns

### Creating a Graph
```python
from claude_agent_graph import AgentGraph

async with AgentGraph(name="my_graph") as graph:
    # Your code here
    pass
```

### Adding Nodes
```python
await graph.add_node(
    "node_id",
    "System prompt describing agent role",
    model="claude-sonnet-4-20250514",
    metadata={"key": "value"}
)
```

### Adding Edges
```python
# Directed edge (creates control relationship)
await graph.add_edge("controller", "subordinate", directed=True)

# Undirected edge (peer relationship)
await graph.add_edge("peer1", "peer2", directed=False)
```

### Sending Messages
```python
await graph.send_message(
    "from_node",
    "to_node",
    "Message content",
    metadata={"priority": "high"}
)
```

### Reading Conversations
```python
messages = await graph.get_conversation("node1", "node2", limit=10)
for msg in messages:
    print(f"{msg.from_node} → {msg.to_node}: {msg.content}")
```

## Troubleshooting

### API Key Not Set
```
Error: ANTHROPIC_API_KEY environment variable not set
```

**Solution:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Import Error
```
ModuleNotFoundError: No module named 'claude_agent_graph'
```

**Solution:**
```bash
cd /path/to/claude-agent-graph
pip install -e .
```

### Permission Denied (Conversations Directory)
```
PermissionError: [Errno 13] Permission denied: './conversations'
```

**Solution:**
```bash
mkdir -p ./conversations
chmod 755 ./conversations
```

## Additional Resources

- **API Documentation:** See `docs/` directory
- **User Guide:** `docs/user_guide/`
- **Tutorial Notebooks:** `examples/notebooks/`
- **Project README:** `../README.md`
- **Technical Reference:** `../CLAUDE.md`
- **Known Issues:** `../ISSUES_TODO.md`

## Contributing Examples

Have a useful example to share? Contributions welcome!

1. Create a new Python file in this directory
2. Follow the structure of existing examples:
   - Comprehensive docstring at top
   - Section comments with `====` separators
   - Print statements showing progress
   - Assertions to verify behavior
   - Clear variable names
3. Add entry to this README
4. Test your example runs correctly
5. Submit a pull request

## Questions?

- **GitHub Issues:** https://github.com/calebmcook/claude-agent-graph/issues
- **Documentation:** https://calebmcook.github.io/claude-agent-graph/
- **Main README:** ../README.md
