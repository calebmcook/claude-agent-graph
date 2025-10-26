"""
Exceptions for claude-agent-graph package.

Defines all custom exceptions used throughout the package.
"""


class AgentGraphError(Exception):
    """Base exception for AgentGraph errors."""

    pass


class NodeNotFoundError(AgentGraphError):
    """Raised when a node is not found in the graph."""

    pass


class EdgeNotFoundError(AgentGraphError):
    """Raised when an edge is not found in the graph."""

    pass


class DuplicateNodeError(AgentGraphError):
    """Raised when attempting to add a node with an existing ID."""

    pass


class DuplicateEdgeError(AgentGraphError):
    """Raised when attempting to add a duplicate edge."""

    pass


class TopologyValidationError(AgentGraphError):
    """Raised when graph topology violates constraints."""

    pass


class AgentSessionError(AgentGraphError):
    """Raised when agent session operations fail."""

    pass


class CommandAuthorizationError(AgentGraphError):
    """Raised when command execution is unauthorized."""

    pass
