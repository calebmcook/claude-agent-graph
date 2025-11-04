"""
claude-agent-graph: A Python package for creating and orchestrating large-scale graphs
of Claude AI agents using the claude-agent-sdk.

This package enables complex, interconnected networks of AI agents that collaborate
and maintain shared state through structured conversation channels.
"""

from claude_agent_graph.agent_manager import AgentSessionManager
from claude_agent_graph.checkpoint import (
    Checkpoint,
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointVersionError,
)
from claude_agent_graph.exceptions import (
    AgentGraphError,
    AgentSessionError,
    CommandAuthorizationError,
    DuplicateEdgeError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
    TopologyViolationError,
)
from claude_agent_graph.execution import (
    ExecutionMode,
    ManualController,
    ProactiveExecutor,
    ReactiveExecutor,
)
from claude_agent_graph.graph import AgentGraph
from claude_agent_graph.models import (
    Edge,
    Message,
    MessageRole,
    Node,
    NodeStatus,
    SharedState,
)
from claude_agent_graph.transactions import (
    Operation,
    RollbackManager,
    StateSnapshot,
    TransactionLog,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core classes
    "AgentGraph",
    "AgentSessionManager",
    # Execution modes (Epic 6)
    "ExecutionMode",
    "ManualController",
    "ReactiveExecutor",
    "ProactiveExecutor",
    # Checkpointing (Epic 7)
    "Checkpoint",
    "CheckpointError",
    "CheckpointVersionError",
    "CheckpointCorruptedError",
    # Data models
    "Message",
    "MessageRole",
    "Node",
    "NodeStatus",
    "Edge",
    "SharedState",
    # Transactions (Epic 5)
    "TransactionLog",
    "Operation",
    "StateSnapshot",
    "RollbackManager",
    # Exceptions
    "AgentGraphError",
    "AgentSessionError",
    "CommandAuthorizationError",
    "NodeNotFoundError",
    "EdgeNotFoundError",
    "DuplicateNodeError",
    "DuplicateEdgeError",
    "TopologyViolationError",
]
