"""
claude-agent-graph: A Python package for creating and orchestrating large-scale graphs
of Claude AI agents using the claude-agent-sdk.

This package enables complex, interconnected networks of AI agents that collaborate
and maintain shared state through structured conversation channels.
"""

from claude_agent_graph.models import (
    Edge,
    Message,
    MessageRole,
    Node,
    NodeStatus,
    SharedState,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Data models
    "Message",
    "MessageRole",
    "Node",
    "NodeStatus",
    "Edge",
    "SharedState",
]
