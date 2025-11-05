"""
Data models for claude-agent-graph package.

This module defines the core data structures:
- Message: Inter-agent communication messages
- Node: Individual agent representation
- Edge: Connection between agents
- SharedState: Conversation state for edges
"""

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """
    Represents a message exchanged between agents.

    Messages are the fundamental unit of inter-agent communication and are
    persisted to conversation files in JSONL format.
    """

    message_id: str = Field(
        default_factory=lambda: f"msg_{uuid4().hex[:16]}",
        description="Unique identifier for the message",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when message was created",
    )
    from_node: str = Field(
        ...,
        description="ID of the sending agent node",
        min_length=1,
    )
    to_node: str = Field(
        ...,
        description="ID of the receiving agent node",
        min_length=1,
    )
    role: MessageRole = Field(
        default=MessageRole.USER,
        description="Role of the message (user, assistant, system)",
    )
    content: str = Field(
        ...,
        description="The message content/text",
        min_length=1,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the message",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware and in UTC."""
        if v.tzinfo is None:
            # Assume naive datetime is UTC
            return v.replace(tzinfo=timezone.utc)
        # Convert to UTC if not already
        return v.astimezone(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert message to dictionary format.

        Returns:
            Dictionary representation of the message with ISO format timestamp.
        """
        data: dict[str, Any] = self.model_dump()
        # Convert datetime to ISO format string for JSON serialization
        data["timestamp"] = self.timestamp.isoformat()
        data["role"] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """
        Create a Message from a dictionary.

        Args:
            data: Dictionary containing message data

        Returns:
            Message instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Parse timestamp if it's a string
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)

    model_config = ConfigDict(
        use_enum_values=False,
    )


class NodeStatus(str, Enum):
    """Status of an agent node."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPED = "stopped"
    ERROR = "error"


class Node(BaseModel):
    """
    Represents an agent node in the graph.

    Each node corresponds to a ClaudeSDKClient session with its own
    system prompt and configuration.
    """

    node_id: str = Field(
        ...,
        description="Unique identifier for the node",
        min_length=1,
    )
    system_prompt: str = Field(
        ...,
        description="System prompt for the agent",
        min_length=1,
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model to use for this agent",
    )
    status: NodeStatus = Field(
        default=NodeStatus.INITIALIZING,
        description="Current status of the agent",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when node was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the node",
    )

    # Agent session management fields (Epic 4)
    original_system_prompt: str | None = Field(
        default=None,
        description="Backup of original system prompt before control injection",
    )
    effective_system_prompt: str | None = Field(
        default=None,
        description="System prompt with injected controller information",
    )
    prompt_dirty: bool = Field(
        default=False,
        description="Whether the prompt needs recomputation due to edge changes",
    )
    agent_session: Any | None = Field(
        default=None,
        description="Reference to ClaudeSDKClient instance (not serialized)",
        exclude=True,
    )

    @field_validator("created_at")
    @classmethod
    def ensure_utc_created_at(cls, v: datetime) -> datetime:
        """Ensure created_at is timezone-aware and in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt_length(cls, v: str) -> str:
        """Validate system prompt is not too long."""
        max_length = 100000  # 100K characters
        if len(v) > max_length:
            raise ValueError(f"system_prompt exceeds maximum length of {max_length} characters")
        return v

    model_config = ConfigDict()


class Edge(BaseModel):
    """
    Represents a connection between two agent nodes.

    Edges define communication channels and maintain shared conversation state.
    """

    edge_id: str = Field(
        ...,
        description="Unique identifier for the edge (generated from node IDs)",
    )
    from_node: str = Field(
        ...,
        description="ID of the source node",
        min_length=1,
    )
    to_node: str = Field(
        ...,
        description="ID of the target node",
        min_length=1,
    )
    directed: bool = Field(
        default=True,
        description="Whether the edge is directed (True) or bidirectional (False)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when edge was created",
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom properties and metadata for the edge",
    )
    # SharedState reference will be added in later epic
    # shared_state: Optional["SharedState"] = None

    @field_validator("created_at")
    @classmethod
    def ensure_utc_created_at(cls, v: datetime) -> datetime:
        """Ensure created_at is timezone-aware and in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("edge_id")
    @classmethod
    def validate_edge_id_format(cls, v: str, info: Any) -> str:
        """Validate edge_id follows expected format."""
        if not v:
            raise ValueError("edge_id cannot be empty")
        return v

    @classmethod
    def generate_edge_id(cls, from_node: str, to_node: str, directed: bool = True) -> str:
        """
        Generate a unique edge ID from node IDs.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            directed: Whether the edge is directed

        Returns:
            Generated edge ID
        """
        if directed:
            return f"{from_node}_to_{to_node}"
        else:
            # For undirected edges, use alphabetical order for consistency
            nodes = sorted([from_node, to_node])
            return f"{nodes[0]}_undirected_{nodes[1]}"

    model_config = ConfigDict()


class SharedState(BaseModel):
    """
    Manages shared state and conversation history for an edge.

    Each edge maintains a conversation file (JSONL format) that stores
    all messages exchanged between the connected agents.
    """

    conversation_file: str = Field(
        ...,
        description="Path to the JSONL conversation file",
        min_length=1,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the shared state",
    )

    @field_validator("conversation_file")
    @classmethod
    def validate_conversation_file_path(cls, v: str) -> str:
        """Validate conversation file path format."""
        if not v.endswith(".jsonl"):
            raise ValueError("conversation_file must have .jsonl extension")
        return v

    model_config = ConfigDict()


@dataclass
class CachedMetric:
    """
    Represents a cached metric with time-to-live (TTL) validation.

    Metrics can be expensive to compute (e.g., message counting from files),
    so this dataclass provides a simple caching mechanism with TTL support.
    """

    value: Any
    timestamp: datetime = dataclass_field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ttl_seconds: int = 300  # Default TTL: 5 minutes

    def is_valid(self) -> bool:
        """
        Check if the cached metric is still valid based on TTL.

        Returns:
            True if metric age is less than TTL, False otherwise.
        """
        age_seconds = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age_seconds < self.ttl_seconds


@dataclass
class GraphMetrics:
    """
    Comprehensive metrics about a graph's structure and activity.

    These metrics provide insights into graph topology, agent utilization,
    and error rates. They can be computed on-demand or cached based on TTL.
    """

    node_count: int = 0
    """Total number of nodes in the graph."""

    edge_count: int = 0
    """Total number of edges in the graph."""

    message_count: int = 0
    """Total number of messages across all edges."""

    active_conversations: int = 0
    """Number of edges with recent message activity."""

    avg_node_degree: float = 0.0
    """Average number of connections per node."""

    isolated_nodes: int = 0
    """Number of nodes with no edges."""

    agent_utilization: dict[str, float] = dataclass_field(default_factory=dict)
    """
    Per-agent message throughput (messages per hour).
    Maps node_id to utilization value.
    """

    error_rate: float = 0.0
    """
    Fraction of operations that resulted in errors (0.0 to 1.0).
    Computed from failed operations in time window.
    """

    timestamp: datetime = dataclass_field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    """UTC timestamp when metrics were computed."""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics to dictionary format.

        Returns:
            Dictionary representation of metrics with ISO format timestamp.
        """
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "message_count": self.message_count,
            "active_conversations": self.active_conversations,
            "avg_node_degree": self.avg_node_degree,
            "isolated_nodes": self.isolated_nodes,
            "agent_utilization": self.agent_utilization,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphMetrics":
        """
        Create GraphMetrics from a dictionary.

        Args:
            data: Dictionary containing metrics data

        Returns:
            GraphMetrics instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
