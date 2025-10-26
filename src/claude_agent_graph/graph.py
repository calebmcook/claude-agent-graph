"""
Main AgentGraph orchestration class.

The AgentGraph class manages the entire agent network including graph construction,
topology validation, and dynamic modifications. It coordinates node/edge lifecycle
and message routing.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import networkx as nx

from .agent_manager import AgentSessionManager
from .backends import FilesystemBackend, StorageBackend
from .exceptions import (
    AgentGraphError,
    DuplicateEdgeError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
)
from .models import Edge, Message, Node

logger = logging.getLogger(__name__)


class TopologyViolationError(AgentGraphError):
    """Raised when a graph operation violates topology constraints."""

    pass


class AgentGraph:
    """
    Main orchestration class for managing agent networks.

    This class handles:
    - Graph construction (adding/removing nodes and edges)
    - Topology validation
    - Message routing and delivery
    - Agent lifecycle management
    """

    def __init__(
        self,
        name: str,
        max_nodes: int = 10000,
        persistence_enabled: bool = True,
        topology_constraint: str | None = None,
        storage_backend: StorageBackend | None = None,
    ):
        """
        Initialize an AgentGraph.

        Args:
            name: Name of the graph
            max_nodes: Maximum number of nodes allowed
            persistence_enabled: Whether to enable persistence
            topology_constraint: Optional topology constraint
                (e.g., "tree", "dag", "mesh", "chain", "star")
            storage_backend: Storage backend for conversations (default: FilesystemBackend)
        """
        self.name = name
        self.max_nodes = max_nodes
        self.persistence_enabled = persistence_enabled
        self.topology_constraint = topology_constraint

        # Set up storage backend (default to FilesystemBackend)
        if storage_backend is None:
            storage_backend = FilesystemBackend(base_dir=f"./conversations/{name}")
        self.storage = storage_backend

        # Internal data structures
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._adjacency: dict[str, list[str]] = defaultdict(list)

        # NetworkX graph for topology operations
        self._nx_graph = nx.DiGraph()

        # Agent session manager (Epic 4)
        self._agent_manager = AgentSessionManager(self)

        # Lock for thread-safe concurrent modifications (Epic 5)
        self._modification_lock = asyncio.Lock()

        logger.debug(f"Created AgentGraph '{name}' with storage backend {type(self.storage).__name__}")

    def __repr__(self) -> str:
        """Return string representation of the graph."""
        return (
            f"AgentGraph(name='{self.name}', "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)})"
        )

    # ==================== Properties ====================

    @property
    def node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        return len(self._edges)

    # ==================== Node Operations ====================

    def add_node(
        self,
        node_id: str,
        system_prompt: str,
        model: str = "claude-sonnet-4-20250514",
        **metadata: Any,
    ) -> Node:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            system_prompt: System prompt for the agent
            model: Claude model to use
            **metadata: Additional metadata for the node

        Returns:
            The created Node object

        Raises:
            DuplicateNodeError: If a node with this ID already exists
            ValueError: If validation fails
            AgentGraphError: If max_nodes exceeded
        """
        # Check if node already exists
        if node_id in self._nodes:
            raise DuplicateNodeError(f"Node with ID '{node_id}' already exists")

        # Check max_nodes constraint
        if self.node_count >= self.max_nodes:
            raise AgentGraphError(
                f"Cannot add node: graph has reached maximum of {self.max_nodes} nodes"
            )

        # Create and validate node
        try:
            node = Node(
                node_id=node_id,
                system_prompt=system_prompt,
                model=model,
                metadata=metadata,
            )
        except ValueError as e:
            raise ValueError(f"Node validation failed: {e}")

        # Store node
        self._nodes[node_id] = node
        self._nx_graph.add_node(node_id)

        logger.debug(f"Added node '{node_id}' to graph '{self.name}'")
        return node

    def get_node(self, node_id: str) -> Node:
        """
        Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            The Node object

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found in graph")
        return self._nodes[node_id]

    def get_nodes(self) -> list[Node]:
        """
        Get all nodes in the graph.

        Returns:
            List of all Node objects
        """
        return list(self._nodes.values())

    def node_exists(self, node_id: str) -> bool:
        """
        Check if a node exists.

        Args:
            node_id: Node identifier

        Returns:
            True if node exists, False otherwise
        """
        return node_id in self._nodes

    # ==================== Edge Operations ====================

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        directed: bool = True,
        **properties: Any,
    ) -> Edge:
        """
        Add an edge between two nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            directed: Whether the edge is directed
            **properties: Custom properties for the edge

        Returns:
            The created Edge object

        Raises:
            NodeNotFoundError: If either node doesn't exist
            DuplicateEdgeError: If edge already exists
            TopologyViolationError: If violates topology constraint
        """
        # Validate nodes exist
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"Target node '{to_node}' not found")

        # Generate edge ID
        edge_id = Edge.generate_edge_id(from_node, to_node, directed)

        # Check for duplicate edge
        if edge_id in self._edges:
            raise DuplicateEdgeError(f"Edge from '{from_node}' to '{to_node}' already exists")

        # Check reverse edge for undirected graphs
        if not directed:
            reverse_id = Edge.generate_edge_id(to_node, from_node, directed)
            if reverse_id in self._edges:
                raise DuplicateEdgeError(
                    f"Undirected edge between '{from_node}' and '{to_node}' already exists"
                )

        # Validate topology constraint if set (before adding the edge)
        if self.topology_constraint:
            self._validate_topology_constraint(from_node, to_node, directed)

        # Create edge
        try:
            edge = Edge(
                edge_id=edge_id,
                from_node=from_node,
                to_node=to_node,
                directed=directed,
                properties=properties,
            )
        except ValueError as e:
            raise ValueError(f"Edge validation failed: {e}")

        # Store edge
        self._edges[edge_id] = edge
        self._adjacency[from_node].append(to_node)

        # Add to networkx graph
        self._nx_graph.add_edge(from_node, to_node)

        if not directed:
            self._adjacency[to_node].append(from_node)
            self._nx_graph.add_edge(to_node, from_node)

        # If directed edge, handle control relationship
        if directed:
            # Mark subordinate's prompt as dirty (new controller added)
            to_node_obj = self.get_node(to_node)
            to_node_obj.prompt_dirty = True
            logger.info(
                f"Control relationship: '{from_node}' → '{to_node}' "
                f"({properties.get('control_type', 'supervisor')})"
            )

        logger.debug(
            f"Added {'directed' if directed else 'undirected'} edge "
            f"'{from_node}' -> '{to_node}' in graph '{self.name}'"
        )
        return edge

    def get_edge(self, from_node: str, to_node: str) -> Edge:
        """
        Get an edge between two nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            The Edge object

        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        edge_id = Edge.generate_edge_id(from_node, to_node, directed=True)
        if edge_id not in self._edges:
            # Try undirected
            edge_id = Edge.generate_edge_id(from_node, to_node, directed=False)

        if edge_id not in self._edges:
            raise EdgeNotFoundError(f"Edge from '{from_node}' to '{to_node}' not found")

        return self._edges[edge_id]

    def get_edges(self) -> list[Edge]:
        """
        Get all edges in the graph.

        Returns:
            List of all Edge objects
        """
        return list(self._edges.values())

    def edge_exists(self, from_node: str, to_node: str) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Returns:
            True if edge exists, False otherwise
        """
        try:
            self.get_edge(from_node, to_node)
            return True
        except EdgeNotFoundError:
            return False

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
    ) -> list[str]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node identifier
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of neighboring node IDs

        Raises:
            NodeNotFoundError: If node doesn't exist
            ValueError: If direction is invalid
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found")

        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError(f"Invalid direction: {direction}")

        neighbors = set()

        if direction in ("outgoing", "both"):
            neighbors.update(self._adjacency.get(node_id, []))

        if direction in ("incoming", "both"):
            for other_node, outgoing in self._adjacency.items():
                if node_id in outgoing:
                    neighbors.add(other_node)

        return list(neighbors)

    # ==================== Topology Operations ====================

    def get_topology(self) -> str:
        """
        Detect the current graph topology type.

        Returns:
            Topology type string: "tree", "dag", "chain", "star",
                                  "cycle", or "unknown"
        """
        if self.node_count == 0:
            return "empty"

        if self.node_count == 1:
            return "single_node"

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self._nx_graph):
            if self._is_cycle_graph():
                return "cycle"
            return "unknown"

        # For acyclic graphs, check from most specific to least specific
        # Order: chain, tree, star, dag

        # Check for chain (linear sequence)
        if self._is_chain():
            return "chain"

        # Check for tree (DAG with one root)
        if self._is_tree():
            return "tree"

        # Check for star (one hub connected to all spokes)
        if self._is_star():
            return "star"

        # Otherwise it's a general DAG
        return "dag"

    def validate_topology(self, required_topology: str) -> bool:
        """
        Validate that graph matches a required topology.

        Args:
            required_topology: Required topology type
                (e.g., "tree", "dag", "mesh", "chain", "star")

        Returns:
            True if topology matches

        Raises:
            TopologyViolationError: If topology doesn't match
        """
        current_topology = self.get_topology()

        if current_topology == required_topology:
            return True

        raise TopologyViolationError(
            f"Graph topology is '{current_topology}', " f"but '{required_topology}' was required"
        )

    def get_isolated_nodes(self) -> list[str]:
        """
        Get list of nodes with no connections.

        Returns:
            List of isolated node IDs
        """
        isolated = []
        for node_id in self._nodes:
            if not self.get_neighbors(node_id, direction="both"):
                isolated.append(node_id)
        return isolated

    # ==================== Private Topology Methods ====================

    def _is_tree(self) -> bool:
        """Check if graph is a tree structure."""
        # A tree should have exactly one node with in-degree 0 (root)
        # and all other nodes have in-degree 1 (exactly one parent)
        if self.node_count <= 1:
            return True

        root_count = 0
        for node in self._nodes:
            in_degree = self._nx_graph.in_degree(node)
            if in_degree == 0:
                root_count += 1

        # Should be exactly one root
        if root_count != 1:
            return False

        # All non-root nodes must have exactly in-degree 1 (one parent)
        for node in self._nodes:
            in_degree = self._nx_graph.in_degree(node)
            if in_degree == 0:
                # This is the root, skip
                continue
            if in_degree != 1:
                return False

        # Should be weakly connected
        return nx.is_weakly_connected(self._nx_graph)

    def _is_chain(self) -> bool:
        """Check if graph is a linear chain."""
        if self.node_count == 0:
            return False

        if self.node_count == 1:
            return True

        # Must be a DAG (no cycles)
        if not nx.is_directed_acyclic_graph(self._nx_graph):
            return False

        if self.node_count == 2:
            # Two nodes: check if connected in ONE direction only (not both)
            nodes = list(self._nodes.keys())
            edge1 = self._nx_graph.has_edge(nodes[0], nodes[1])
            edge2 = self._nx_graph.has_edge(nodes[1], nodes[0])
            return edge1 != edge2  # XOR: exactly one edge, not both

        # Every node can have at most 1 incoming and 1 outgoing
        for node in self._nodes:
            out_degree = self._nx_graph.out_degree(node)
            in_degree = self._nx_graph.in_degree(node)

            # Node can have at most 1 incoming and 1 outgoing
            if in_degree > 1 or out_degree > 1:
                return False

        # Must form a single connected path
        # Check that there's exactly one root and one leaf
        roots = [n for n in self._nodes if self._nx_graph.in_degree(n) == 0]
        leaves = [n for n in self._nodes if self._nx_graph.out_degree(n) == 0]

        return len(roots) == 1 and len(leaves) == 1

    def _is_star(self) -> bool:
        """Check if graph is a star topology (one central node)."""
        if self.node_count <= 2:
            return False

        # For a star: there should be exactly one node that connects to all others
        # and no edges between spokes (non-center nodes)
        center_candidates = []
        for potential_center in self._nodes:
            # Count how many other nodes this node connects to (either direction)
            connected_count = 0
            for other_node in self._nodes:
                if other_node == potential_center:
                    continue
                if self._nx_graph.has_edge(potential_center, other_node) or self._nx_graph.has_edge(
                    other_node, potential_center
                ):
                    connected_count += 1

            if connected_count == self.node_count - 1:
                center_candidates.append(potential_center)

        # Should have exactly one center
        if len(center_candidates) != 1:
            return False

        center = center_candidates[0]

        # Center must have multiple connections (fanout > 1 OR fanin > 1)
        # This distinguishes stars from simple chains
        if self._nx_graph.out_degree(center) < 2 and self._nx_graph.in_degree(center) < 2:
            return False

        # Verify no edges between spokes (non-center nodes)
        for node1 in self._nodes:
            if node1 == center:
                continue
            for node2 in self._nodes:
                if node2 == center or node2 == node1:
                    continue
                if self._nx_graph.has_edge(node1, node2):
                    return False  # Found edge between spokes

        return True

    def _is_cycle_graph(self) -> bool:
        """Check if graph is a simple cycle."""
        if self.node_count < 3:
            return False

        # All nodes should have in-degree and out-degree of 1
        for node in self._nodes:
            if self._nx_graph.in_degree(node) != 1 or self._nx_graph.out_degree(node) != 1:
                return False

        return True

    def _validate_topology_constraint(
        self,
        from_node: str,
        to_node: str,
        directed: bool,
    ) -> None:
        """
        Validate that adding an edge doesn't violate topology constraint.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            directed: Whether edge is directed

        Raises:
            TopologyViolationError: If constraint violated
        """
        constraint = self.topology_constraint.lower()

        if constraint == "tree":
            # Tree: no cycles, each node can have at most one parent (in-degree <= 1)

            # Check if adding this edge creates a cycle
            # This would happen if there's a path from to_node back to from_node
            try:
                nx.shortest_path(self._nx_graph, to_node, from_node)
                raise TopologyViolationError(
                    f"Adding edge {from_node} -> {to_node} would create a cycle"
                )
            except (nx.NetworkXNoPath, nx.NetworkXError):
                # No path exists, good
                pass

            # Check if to_node already has a parent
            in_degree = self._nx_graph.in_degree(to_node)
            if in_degree > 0 and directed:
                raise TopologyViolationError(
                    f"Adding edge {from_node} -> {to_node} violates tree constraint: "
                    f"{to_node} already has a parent"
                )

        elif constraint == "dag":
            # DAG: no cycles
            try:
                nx.shortest_path(self._nx_graph, to_node, from_node)
                raise TopologyViolationError(
                    f"Adding edge {from_node} -> {to_node} would create a cycle"
                )
            except (nx.NetworkXNoPath, nx.NetworkXError):
                pass

        elif constraint == "chain":
            # Chain: each node has at most 1 outgoing and 1 incoming
            if self._nx_graph.out_degree(from_node) > 0 or self._nx_graph.in_degree(to_node) > 0:
                raise TopologyViolationError(
                    f"Adding edge {from_node} -> {to_node} violates chain constraint"
                )

    # ==================== Control Relationships & Prompt Injection ====================

    def _compute_effective_prompt(self, node_id: str) -> str:
        """
        Compute effective system prompt for a node.

        Combines original prompt with injected controller information.
        Handles multiple controllers.

        Args:
            node_id: Node to compute prompt for

        Returns:
            Effective system prompt with controller information injected
        """
        node = self.get_node(node_id)
        original = node.original_system_prompt or node.system_prompt

        # Find all controllers (incoming directed edges)
        controllers: list[tuple[str, str]] = []
        for edge in self._edges.values():
            if edge.to_node == node_id and edge.directed:
                control_type = edge.properties.get("control_type", "supervisor")
                controllers.append((edge.from_node, control_type))

        # If no controllers, return original prompt
        if not controllers:
            return original

        # Build controller list
        controller_lines = "\n".join(
            f"  - Agent '{ctrl_id}' ({ctrl_type})"
            for ctrl_id, ctrl_type in sorted(controllers)
        )

        # Inject control information
        injected = f"""{original}

## Control Hierarchy
You are agent '{node_id}'. You report to the following controllers:
{controller_lines}

Follow directives from your controllers while maintaining your specialized role."""

        return injected

    def _mark_subordinates_dirty(self, controller_id: str) -> None:
        """
        Mark all subordinates' prompts as dirty (need recomputation).

        Called when edges are added from a controller node.

        Args:
            controller_id: Controller node ID
        """
        for edge in self._edges.values():
            if edge.from_node == controller_id and edge.directed:
                subordinate = self.get_node(edge.to_node)
                subordinate.prompt_dirty = True
                logger.debug(f"Marked prompt dirty for subordinate '{edge.to_node}'")

    async def _activate_agent_lazy(self, node_id: str) -> None:
        """
        Activate agent, recomputing prompt if dirty.

        Called on first send_message() or explicit start_agent().
        Ensures agent is running with current control relationships.

        Args:
            node_id: Node to activate
        """
        node = self.get_node(node_id)

        # Recompute prompt if dirty
        if node.prompt_dirty:
            new_prompt = self._compute_effective_prompt(node_id)
            node.effective_system_prompt = new_prompt
            node.prompt_dirty = False
            logger.info(f"Updated effective prompt for agent '{node_id}'")

        # Start agent if not running
        if node_id not in self._agent_manager._contexts:
            await self._agent_manager.start_agent(node_id)

    # ==================== Controller Query Methods ====================

    def get_controllers(self, node_id: str) -> list[str]:
        """
        Get all controllers for a node.

        Returns IDs of nodes with directed edges pointing TO this node.

        Args:
            node_id: Node to query

        Returns:
            List of controller node IDs (sorted, may be empty)

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if not self.node_exists(node_id):
            raise NodeNotFoundError(f"Node '{node_id}' not found")

        controllers = []
        for edge in self._edges.values():
            if edge.to_node == node_id and edge.directed:
                controllers.append(edge.from_node)

        return sorted(controllers)

    def get_subordinates(self, node_id: str) -> list[str]:
        """
        Get all subordinates for a node.

        Returns IDs of nodes with directed edges FROM this node.

        Args:
            node_id: Node to query

        Returns:
            List of subordinate node IDs (sorted, may be empty)

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if not self.node_exists(node_id):
            raise NodeNotFoundError(f"Node '{node_id}' not found")

        subordinates = []
        for edge in self._edges.values():
            if edge.from_node == node_id and edge.directed:
                subordinates.append(edge.to_node)

        return sorted(subordinates)

    def is_controller(self, controller_id: str, subordinate_id: str) -> bool:
        """
        Check if one node controls another.

        Args:
            controller_id: Potential controller node ID
            subordinate_id: Potential subordinate node ID

        Returns:
            True if controller_id has a directed edge to subordinate_id, False otherwise
        """
        edge_id = Edge.generate_edge_id(controller_id, subordinate_id, directed=True)
        return edge_id in self._edges and self._edges[edge_id].directed

    def get_control_relationships(self) -> dict[str, list[str]]:
        """
        Get all control relationships in the graph.

        Returns a dictionary mapping node IDs to their subordinates.
        Only includes nodes that have at least one subordinate.

        Returns:
            Dictionary: node_id → list of subordinate node IDs (sorted)
        """
        relationships = {}
        for node_id in self._nodes:
            subordinates = self.get_subordinates(node_id)
            if subordinates:
                relationships[node_id] = subordinates

        return relationships

    # Message Routing & Delivery Methods

    async def send_message(
        self,
        from_node: str,
        to_node: str,
        content: str,
        **metadata: Any,
    ) -> Message:
        """
        Send a message between two nodes.

        Validates that both nodes exist and an edge connects them, then
        appends the message to the conversation file for that edge.

        Args:
            from_node: ID of the sending node
            to_node: ID of the receiving node
            content: Message content
            **metadata: Additional metadata for the message

        Returns:
            The created Message object

        Raises:
            NodeNotFoundError: If either node doesn't exist
            EdgeNotFoundError: If no edge connects the nodes

        Example:
            >>> msg = await graph.send_message(
            ...     from_node="supervisor",
            ...     to_node="worker_1",
            ...     content="Start processing",
            ...     priority="high"
            ... )
        """
        # Validate nodes exist
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{from_node}' not found")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{to_node}' not found")

        # Validate edge exists (directed or undirected)
        edge_id = None
        directed_edge_id = Edge.generate_edge_id(from_node, to_node, directed=True)
        undirected_edge_id = Edge.generate_edge_id(from_node, to_node, directed=False)

        if directed_edge_id in self._edges:
            edge_id = directed_edge_id
        elif undirected_edge_id in self._edges:
            edge_id = undirected_edge_id
        else:
            # Check reverse direction for undirected
            reverse_undirected = Edge.generate_edge_id(to_node, from_node, directed=False)
            if reverse_undirected in self._edges:
                edge_id = reverse_undirected
            else:
                raise EdgeNotFoundError(f"No edge found between '{from_node}' and '{to_node}'")

        # Lazy activate both nodes (recompute prompts if dirty)
        await self._activate_agent_lazy(from_node)
        await self._activate_agent_lazy(to_node)

        # Create message
        message = Message(
            from_node=from_node,
            to_node=to_node,
            content=content,
            metadata=metadata if metadata else {},
        )

        # Append message to storage backend
        await self.storage.append_message(edge_id, message)

        logger.debug(f"Sent message {message.message_id}: {from_node} -> {to_node}")
        return message

    async def get_conversation(
        self,
        from_node: str,
        to_node: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        Get conversation history between two nodes.

        Args:
            from_node: First node ID
            to_node: Second node ID
            since: Only return messages after this timestamp
            limit: Maximum number of messages to return

        Returns:
            List of Message objects, ordered chronologically

        Raises:
            NodeNotFoundError: If either node doesn't exist
            EdgeNotFoundError: If no edge connects the nodes

        Example:
            >>> # Get all messages
            >>> messages = await graph.get_conversation("a", "b")
            >>>
            >>> # Get recent messages
            >>> recent = await graph.get_conversation("a", "b", limit=10)
        """
        # Validate nodes exist
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{from_node}' not found")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{to_node}' not found")

        # Find edge ID (try both directions and directed/undirected)
        edge_id = None
        for eid, edge in self._edges.items():
            if (
                (edge.from_node == from_node and edge.to_node == to_node)
                or (edge.from_node == to_node and edge.to_node == from_node)
                or (
                    not edge.directed
                    and (
                        (edge.from_node == from_node and edge.to_node == to_node)
                        or (edge.from_node == to_node and edge.to_node == from_node)
                    )
                )
            ):
                edge_id = eid
                break

        if edge_id is None:
            raise EdgeNotFoundError(f"No edge found between '{from_node}' and '{to_node}'")

        # Read messages from storage backend
        messages = await self.storage.read_messages(edge_id, since=since, limit=limit)

        logger.debug(f"Retrieved {len(messages)} messages between {from_node} and {to_node}")
        return messages

    async def get_recent_messages(
        self,
        from_node: str,
        to_node: str,
        count: int = 10,
    ) -> list[Message]:
        """
        Get the most recent N messages between two nodes.

        This is a convenience method that calls get_conversation with a limit.

        Args:
            from_node: First node ID
            to_node: Second node ID
            count: Number of recent messages to return (default: 10)

        Returns:
            List of the most recent Message objects

        Raises:
            NodeNotFoundError: If either node doesn't exist
            EdgeNotFoundError: If no edge connects the nodes

        Example:
            >>> # Get last 5 messages
            >>> recent = await graph.get_recent_messages("a", "b", count=5)
        """
        return await self.get_conversation(from_node, to_node, limit=count)

    # ==================== Async Context Manager (Epic 4) ====================

    async def __aenter__(self) -> "AgentGraph":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """
        Exit async context manager and cleanup all agents.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        if hasattr(self, "_agent_manager"):
            await self._agent_manager.stop_all()
            logger.info(f"AgentGraph '{self.name}' cleaned up all agents")

    # ==================== Agent Lifecycle Management (Epic 4) ====================

    async def start_agent(self, node_id: str) -> None:
        """
        Start an agent session.

        Creates and starts a ClaudeSDKClient for the specified node.
        Updates node status to ACTIVE.

        Args:
            node_id: ID of the node to start

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If agent is already running or start fails

        Example:
            >>> await graph.start_agent("worker_1")
        """
        await self._agent_manager.start_agent(node_id)

    async def stop_agent(self, node_id: str) -> None:
        """
        Stop an agent session gracefully.

        Stops the ClaudeSDKClient for the specified node.
        Updates node status to STOPPED.

        Args:
            node_id: ID of the node to stop

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If stop operation fails

        Example:
            >>> await graph.stop_agent("worker_1")
        """
        await self._agent_manager.stop_agent(node_id)

    async def restart_agent(self, node_id: str) -> None:
        """
        Restart an agent session.

        Stops and then starts the agent with a fresh context.

        Args:
            node_id: ID of the node to restart

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If restart operation fails

        Example:
            >>> await graph.restart_agent("worker_1")
        """
        await self._agent_manager.restart_agent(node_id)

    def get_agent_status(self, node_id: str) -> dict[str, Any]:
        """
        Get comprehensive status information for an agent.

        Args:
            node_id: ID of the node

        Returns:
            Dictionary containing:
                - node_id: The node ID
                - status: Current status (initializing, active, stopped, error)
                - model: Model being used
                - is_running: Whether agent session is active
                - last_error: Last error message (if any)
                - error_count: Number of errors encountered
                - created_at: When the node was created

        Raises:
            NodeNotFoundError: If node doesn't exist

        Example:
            >>> status = graph.get_agent_status("worker_1")
            >>> print(status['status'])  # 'active'
        """
        node = self.get_node(node_id)
        return {
            "node_id": node_id,
            "status": node.status.value,
            "model": node.model,
            "is_running": self._agent_manager.is_running(node_id),
            "last_error": node.metadata.get("last_error"),
            "error_count": node.metadata.get("error_count", 0),
            "created_at": node.created_at.isoformat(),
        }

    # ==================== Dynamic Node Operations (Epic 5) ====================

    async def remove_node(
        self,
        node_id: str,
        cascade: bool = True,
    ) -> None:
        """
        Remove a node from the graph.

        Removes a node and optionally all connected edges. Stops the agent session
        gracefully and archives conversation files. Updates control relationships
        for affected nodes by marking their prompts as dirty.

        Args:
            node_id: ID of the node to remove
            cascade: If True, remove all associated edges automatically.
                    If False, raise error if edges exist (default: True)

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If edges exist and cascade=False

        Example:
            >>> # Remove node and all connected edges
            >>> await graph.remove_node("worker_1", cascade=True)
            >>>
            >>> # Remove only if isolated (no edges)
            >>> await graph.remove_node("worker_1", cascade=False)
        """
        async with self._modification_lock:
            # Validate node exists
            if node_id not in self._nodes:
                raise NodeNotFoundError(f"Node '{node_id}' not found")

            # Check for edges if cascade=False
            if not cascade:
                # Get all connected edges
                connected_edges = [
                    edge for edge in self._edges.values()
                    if edge.from_node == node_id or edge.to_node == node_id
                ]
                if connected_edges:
                    edge_ids = [edge.edge_id for edge in connected_edges]
                    raise AgentGraphError(
                        f"Cannot remove node '{node_id}': has {len(connected_edges)} connected edge(s). "
                        f"Set cascade=True to remove edges, or remove them manually. "
                        f"Edges: {', '.join(edge_ids)}"
                    )

            # Collect edges to remove if cascade=True
            edges_to_remove = []
            if cascade:
                edges_to_remove = [
                    edge for edge in list(self._edges.values())
                    if edge.from_node == node_id or edge.to_node == node_id
                ]

            # Stop agent session gracefully
            try:
                if self._agent_manager.is_running(node_id):
                    await self._agent_manager.stop_agent(node_id)
                    logger.info(f"Stopped agent session for node '{node_id}'")
            except Exception as e:
                logger.warning(f"Error stopping agent '{node_id}': {e}")
                # Continue with removal even if stop fails

            # Archive conversation files and remove edges
            for edge in edges_to_remove:
                try:
                    await self._archive_edge_conversation(edge.edge_id)
                except Exception as e:
                    logger.warning(f"Error archiving conversation for edge '{edge.edge_id}': {e}")

                # Update control relationships
                if edge.from_node == node_id:
                    # This node was a controller; mark subordinate's prompt dirty
                    subordinate = self._nodes.get(edge.to_node)
                    if subordinate:
                        subordinate.prompt_dirty = True
                        logger.debug(
                            f"Marked subordinate '{edge.to_node}' prompt dirty "
                            f"(lost controller '{node_id}')"
                        )
                elif edge.to_node == node_id:
                    # This node was controlled; mark controller affected
                    # (but they don't need prompt updates; the subordinate does)
                    pass

                # Remove edge from all structures
                self._edges.pop(edge.edge_id, None)

                # Clean up adjacency
                if edge.from_node in self._adjacency:
                    try:
                        self._adjacency[edge.from_node].remove(edge.to_node)
                    except ValueError:
                        pass

                if not edge.directed:
                    # Undirected: clean reverse adjacency
                    if edge.to_node in self._adjacency:
                        try:
                            self._adjacency[edge.to_node].remove(edge.from_node)
                        except ValueError:
                            pass

                # Remove from NetworkX graph
                try:
                    self._nx_graph.remove_edge(edge.from_node, edge.to_node)
                except nx.NetworkXError:
                    pass  # Edge might have been removed already

            # Remove node from all structures
            self._nodes.pop(node_id, None)
            self._adjacency.pop(node_id, None)

            # Remove from NetworkX graph
            try:
                self._nx_graph.remove_node(node_id)
            except nx.NetworkXError:
                pass  # Node might have been removed already

            logger.info(
                f"Removed node '{node_id}' from graph '{self.name}' "
                f"(cascade={cascade}, removed {len(edges_to_remove)} edges)"
            )

    async def _archive_edge_conversation(self, edge_id: str) -> None:
        """
        Archive the conversation file for an edge.

        Moves the conversation file to the archived/ directory with a timestamp.
        This preserves the conversation history for audit and debugging.

        Args:
            edge_id: ID of the edge whose conversation to archive
        """
        try:
            await self.storage.archive_conversation(edge_id)
            logger.debug(f"Archived conversation for edge '{edge_id}'")
        except Exception as e:
            logger.warning(f"Could not archive conversation for edge '{edge_id}': {e}")
            raise
