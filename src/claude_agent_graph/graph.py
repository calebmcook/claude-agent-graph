"""
Main AgentGraph orchestration class.

The AgentGraph class manages the entire agent network including graph construction,
topology validation, and dynamic modifications. It coordinates node/edge lifecycle
and message routing.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from .agent_manager import AgentSessionManager
from .backends import FilesystemBackend, StorageBackend
from .checkpoint import Checkpoint, CheckpointError
from .exceptions import (
    AgentGraphError,
    CommandAuthorizationError,
    DuplicateEdgeError,
    DuplicateNodeError,
    EdgeNotFoundError,
    NodeNotFoundError,
    TopologyViolationError,
)
from .models import Edge, Message, Node

logger = logging.getLogger(__name__)


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
        storage_backend: StorageBackend | str | None = None,
        storage_path: Optional[Path | str] = None,
        auto_save: bool = True,
        auto_save_interval: int = 300,
        checkpoint_dir: Optional[Path | str] = None,
    ):
        """
        Initialize an AgentGraph.

        Args:
            name: Name of the graph
            max_nodes: Maximum number of nodes allowed
            persistence_enabled: Whether to enable persistence
            topology_constraint: Optional topology constraint
                (e.g., "tree", "dag", "mesh", "chain", "star")
            storage_backend: Storage backend for conversations. Can be:
                - A StorageBackend instance
                - A string: "filesystem" (default if None)
                - None: uses FilesystemBackend with default path
            storage_path: Path for storage backend (only used with string storage_backend)
            auto_save: Whether to enable automatic checkpointing (default: True)
            auto_save_interval: Seconds between auto-save checkpoints (default: 300)
            checkpoint_dir: Directory for checkpoint files (default: ./checkpoints/{name})

        Raises:
            ValueError: If storage_backend string is invalid or if storage_path is
                provided with a StorageBackend instance
        """
        self.name = name
        self.max_nodes = max_nodes
        self.persistence_enabled = persistence_enabled
        self.topology_constraint = topology_constraint

        # Set up storage backend with enhanced convenience parameters
        self.storage = self._init_storage_backend(storage_backend, storage_path)

        # Checkpoint configuration (Epic 7)
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        if checkpoint_dir is None:
            self.checkpoint_dir = Path(f"./checkpoints/{name}")
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self._auto_save_task: Optional[asyncio.Task] = None

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

        # Message queues for execution modes (Epic 6)
        self._message_queues: dict[str, asyncio.Queue] = {}
        self._execution_mode = None

        logger.debug(f"Created AgentGraph '{name}' with storage backend {type(self.storage).__name__}")

    def _init_storage_backend(
        self,
        storage_backend: StorageBackend | str | None,
        storage_path: Optional[Path | str],
    ) -> StorageBackend:
        """
        Initialize storage backend from various input formats.

        Args:
            storage_backend: Backend instance, string identifier, or None
            storage_path: Optional path for filesystem backend

        Returns:
            Initialized StorageBackend instance

        Raises:
            ValueError: If invalid backend type or conflicting parameters
        """
        # Case 1: Already a StorageBackend instance
        if isinstance(storage_backend, StorageBackend):
            if storage_path is not None:
                raise ValueError(
                    "storage_path cannot be specified when storage_backend is a "
                    "StorageBackend instance. Use the backend's constructor instead."
                )
            return storage_backend

        # Case 2: String identifier (e.g., "filesystem")
        if isinstance(storage_backend, str):
            backend_type = storage_backend.lower()

            # Validate backend type
            valid_backends = ["filesystem"]
            if backend_type not in valid_backends:
                raise ValueError(
                    f"Invalid storage backend: '{backend_type}'. "
                    f"Valid options: {', '.join(valid_backends)}"
                )

            if backend_type == "filesystem":
                base_dir = storage_path if storage_path else f"./conversations/{self.name}"
                return FilesystemBackend(base_dir=base_dir)

        # Case 3: None - use default
        if storage_backend is None:
            base_dir = storage_path if storage_path else f"./conversations/{self.name}"
            return FilesystemBackend(base_dir=base_dir)

        raise ValueError(
            f"storage_backend must be a StorageBackend instance, string, or None. "
            f"Got: {type(storage_backend)}"
        )

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

    async def add_node(
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
        async with self._modification_lock:
            # Ensure this is truly async (force coroutine creation)
            await asyncio.sleep(0)

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

    async def add_edge(
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
        async with self._modification_lock:
            # Ensure this is truly async (force coroutine creation)
            await asyncio.sleep(0)

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

    # ==================== Control Commands (Epic 6) ====================

    async def execute_command(
        self,
        controller: str,
        subordinate: str,
        command: str,
        **params: Any,
    ) -> Message:
        """
        Execute a command on a subordinate agent.

        Validates that controller has a directed edge to subordinate,
        then sends a specially formatted command message.

        Args:
            controller: ID of the controlling agent
            subordinate: ID of the subordinate agent
            command: Command name (e.g., "process_data")
            **params: Command parameters

        Returns:
            The command Message object

        Raises:
            NodeNotFoundError: If either node doesn't exist
            CommandAuthorizationError: If controller doesn't control subordinate

        Example:
            >>> msg = await graph.execute_command(
            ...     "supervisor", "worker", "analyze",
            ...     dataset="Q1", output_format="json"
            ... )
        """
        # Validate nodes exist
        if controller not in self._nodes:
            raise NodeNotFoundError(f"Node '{controller}' not found")
        if subordinate not in self._nodes:
            raise NodeNotFoundError(f"Node '{subordinate}' not found")

        # Check control relationship
        if not self.is_controller(controller, subordinate):
            raise CommandAuthorizationError(
                f"'{controller}' does not have control relationship with '{subordinate}'"
            )

        # Build command metadata
        edge = self.get_edge(controller, subordinate)
        command_metadata = {
            "type": "command",
            "command": command,
            "params": params,
            "authorization_level": edge.properties.get("control_type", "supervisor"),
        }

        # Build command content
        content = f"Execute: {command}"

        # Send command message
        message = await self.send_message(controller, subordinate, content, **command_metadata)

        logger.info(
            f"Command executed: '{controller}' -> '{subordinate}' "
            f"(command='{command}', params={params})"
        )
        return message

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

    # ==================== Message Routing Patterns (Epic 6) ====================

    async def broadcast(
        self,
        from_node: str,
        content: str,
        include_incoming: bool = False,
        **metadata: Any,
    ) -> list[Message]:
        """
        Broadcast a message to all neighbors of a node.

        By default, sends only to outgoing edges (nodes this one sends to).
        With include_incoming=True, also sends to incoming edges (nodes that send to this one).

        Args:
            from_node: ID of the sending node
            content: Message content
            include_incoming: If True, include incoming edges (default: False)
            **metadata: Additional metadata for messages

        Returns:
            List of Message objects (one per recipient)

        Raises:
            NodeNotFoundError: If from_node doesn't exist

        Example:
            >>> # Send to all outgoing neighbors
            >>> messages = await graph.broadcast("supervisor", "Status update")
            >>>
            >>> # Send to both incoming and outgoing neighbors
            >>> messages = await graph.broadcast(
            ...     "hub", "Alert", include_incoming=True
            ... )
        """
        # Validate node exists
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{from_node}' not found")

        # Get target nodes
        targets = set()

        # Add outgoing neighbors
        targets.update(self.get_neighbors(from_node, direction="outgoing"))

        # Add incoming neighbors if requested
        if include_incoming:
            targets.update(self.get_neighbors(from_node, direction="incoming"))

        # Send to each neighbor and collect messages
        messages = []
        for to_node in sorted(targets):  # Sort for deterministic ordering
            try:
                msg = await self.send_message(from_node, to_node, content, **metadata)
                messages.append(msg)
            except (EdgeNotFoundError, NodeNotFoundError) as e:
                logger.warning(f"Failed to send broadcast message to '{to_node}': {e}")
                # Continue with other recipients

        logger.info(
            f"Broadcast from '{from_node}': sent to {len(messages)} recipients "
            f"(include_incoming={include_incoming})"
        )
        return messages

    async def route_message(
        self,
        from_node: str,
        to_node: str,
        content: str,
        path: list[str] | None = None,
        **metadata: Any,
    ) -> list[Message]:
        """
        Route a message through a path of intermediate nodes.

        If path is not provided, finds shortest path using NetworkX.
        Path must start with from_node and end with to_node.

        Args:
            from_node: Starting node ID
            to_node: Ending node ID
            content: Message content
            path: Explicit path as list of node IDs (optional)
            **metadata: Additional metadata

        Returns:
            List of Message objects for each hop

        Raises:
            NodeNotFoundError: If any node in path doesn't exist
            EdgeNotFoundError: If path doesn't connect
            ValueError: If path format invalid

        Example:
            >>> # Auto-find shortest path
            >>> messages = await graph.route_message("A", "D", "Request")
            >>>
            >>> # Use explicit path
            >>> messages = await graph.route_message(
            ...     "A", "D", "Request", path=["A", "B", "C", "D"]
            ... )
        """
        # Validate nodes exist
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{from_node}' not found")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{to_node}' not found")

        # Find or validate path
        if path is None:
            # Auto-find shortest path
            try:
                path = nx.shortest_path(self._nx_graph, from_node, to_node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raise EdgeNotFoundError(
                    f"No path exists between '{from_node}' and '{to_node}'"
                )
        else:
            # Validate provided path
            if len(path) < 2:
                raise ValueError(f"Path must have at least 2 nodes, got {len(path)}")
            if path[0] != from_node:
                raise ValueError(f"Path must start with '{from_node}', got '{path[0]}'")
            if path[-1] != to_node:
                raise ValueError(f"Path must end with '{to_node}', got '{path[-1]}'")

            # Validate all nodes exist and path connects
            for node_id in path:
                if node_id not in self._nodes:
                    raise NodeNotFoundError(f"Node '{node_id}' in path not found")

            # Validate consecutive pairs are connected
            for i in range(len(path) - 1):
                curr_node = path[i]
                next_node = path[i + 1]
                if not self.edge_exists(curr_node, next_node):
                    raise EdgeNotFoundError(
                        f"No edge from '{curr_node}' to '{next_node}' in provided path"
                    )

        # Add routing metadata
        routing_metadata = metadata.copy()
        routing_metadata["routing_path"] = path
        routing_metadata["hop_count"] = len(path) - 1
        routing_metadata["original_sender"] = from_node

        # Send through path (hop by hop)
        messages = []
        for i in range(len(path) - 1):
            curr = path[i]
            next_node = path[i + 1]
            try:
                msg = await self.send_message(curr, next_node, content, **routing_metadata)
                messages.append(msg)
            except (EdgeNotFoundError, NodeNotFoundError) as e:
                logger.error(f"Failed to send message from '{curr}' to '{next_node}': {e}")
                raise

        logger.info(
            f"Routed message from '{from_node}' to '{to_node}': "
            f"path={path}, hops={len(path) - 1}"
        )
        return messages

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

    def update_node(
        self,
        node_id: str,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update node properties at runtime.

        Updates one or more properties of a node. Metadata is merged with existing
        values (not replaced). If system prompt changes, the prompt_dirty flag is
        set to trigger recomputation on next agent activation.

        Args:
            node_id: ID of the node to update
            system_prompt: New system prompt (optional)
            metadata: New metadata to merge (optional)

        Raises:
            NodeNotFoundError: If node doesn't exist

        Example:
            >>> # Update only system prompt
            >>> graph.update_node("worker_1",
            ...     system_prompt="New role description")
            >>>
            >>> # Update only metadata
            >>> graph.update_node("worker_1",
            ...     metadata={"priority": "high", "team": "alpha"})
            >>>
            >>> # Update both
            >>> graph.update_node("worker_1",
            ...     system_prompt="New role",
            ...     metadata={"priority": "high"})
        """
        # Validate node exists
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node '{node_id}' not found")

        node = self._nodes[node_id]

        # Update system prompt if provided
        if system_prompt is not None:
            # Store original prompt if not already stored
            if node.original_system_prompt is None:
                node.original_system_prompt = node.system_prompt

            # Update prompt
            node.system_prompt = system_prompt
            node.prompt_dirty = True

            # Mark subordinates' prompts as dirty (they may need to recompute)
            self._mark_subordinates_dirty(node_id)

            logger.info(f"Updated system prompt for node '{node_id}'")

        # Update metadata if provided (merge, don't replace)
        if metadata is not None:
            node.metadata.update(metadata)
            logger.info(f"Updated metadata for node '{node_id}'")

        logger.debug(f"Updated node '{node_id}'")

    # ==================== Dynamic Edge Operations (Epic 5) ====================

    async def remove_edge(
        self,
        from_node: str,
        to_node: str,
    ) -> None:
        """
        Remove an edge between two nodes.

        Removes an edge (directed or undirected) and archives the associated
        conversation file. Updates control relationships by marking subordinate
        prompts as dirty if the edge was a directed control relationship.

        Args:
            from_node: Source node ID
            to_node: Target node ID

        Raises:
            NodeNotFoundError: If either node doesn't exist
            EdgeNotFoundError: If edge doesn't exist

        Example:
            >>> # Remove directed edge
            >>> await graph.remove_edge("supervisor", "worker_1")
            >>>
            >>> # Remove undirected edge
            >>> await graph.remove_edge("peer_a", "peer_b")
        """
        async with self._modification_lock:
            # Validate nodes exist
            if from_node not in self._nodes:
                raise NodeNotFoundError(f"Node '{from_node}' not found")
            if to_node not in self._nodes:
                raise NodeNotFoundError(f"Node '{to_node}' not found")

            # Try to find the edge (directed or undirected)
            directed_edge_id = Edge.generate_edge_id(from_node, to_node, directed=True)
            undirected_edge_id = Edge.generate_edge_id(from_node, to_node, directed=False)
            reverse_undirected_id = Edge.generate_edge_id(to_node, from_node, directed=False)

            edge_id = None
            edge = None

            if directed_edge_id in self._edges:
                edge_id = directed_edge_id
                edge = self._edges[edge_id]
            elif undirected_edge_id in self._edges:
                edge_id = undirected_edge_id
                edge = self._edges[edge_id]
            elif reverse_undirected_id in self._edges:
                edge_id = reverse_undirected_id
                edge = self._edges[edge_id]
            else:
                raise EdgeNotFoundError(f"Edge from '{from_node}' to '{to_node}' not found")

            # Archive conversation file
            try:
                await self._archive_edge_conversation(edge_id)
            except Exception as e:
                logger.warning(f"Error archiving conversation for edge '{edge_id}': {e}")

            # Update control relationships if directed edge
            if edge.directed and edge.to_node == to_node:
                # Mark subordinate's prompt dirty (controller removed)
                subordinate = self._nodes.get(to_node)
                if subordinate:
                    subordinate.prompt_dirty = True
                    logger.debug(
                        f"Marked subordinate '{to_node}' prompt dirty "
                        f"(lost controller '{from_node}')"
                    )

            # Remove edge from all structures
            self._edges.pop(edge_id, None)

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
                pass

            if not edge.directed:
                # Try removing reverse direction
                try:
                    self._nx_graph.remove_edge(edge.to_node, edge.from_node)
                except nx.NetworkXError:
                    pass

            logger.info(f"Removed edge from '{from_node}' to '{to_node}' (archived conversation)")

    def update_edge(
        self,
        from_node: str,
        to_node: str,
        **properties: Any,
    ) -> None:
        """
        Update edge properties.

        Merges new properties with existing edge properties. If control_type
        changes, marks subordinate's prompt as dirty to trigger recomputation.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            **properties: Properties to merge into edge.properties

        Raises:
            NodeNotFoundError: If either node doesn't exist
            EdgeNotFoundError: If edge doesn't exist

        Example:
            >>> # Update control type
            >>> graph.update_edge("cfo", "analyst",
            ...     control_type="oversight", priority="high")
            >>>
            >>> # Add custom metadata
            >>> graph.update_edge("peer_a", "peer_b",
            ...     collaboration_level="high")
        """
        # Validate nodes exist
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{from_node}' not found")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"Node '{to_node}' not found")

        # Find edge
        directed_edge_id = Edge.generate_edge_id(from_node, to_node, directed=True)
        undirected_edge_id = Edge.generate_edge_id(from_node, to_node, directed=False)
        reverse_undirected_id = Edge.generate_edge_id(to_node, from_node, directed=False)

        edge_id = None
        if directed_edge_id in self._edges:
            edge_id = directed_edge_id
        elif undirected_edge_id in self._edges:
            edge_id = undirected_edge_id
        elif reverse_undirected_id in self._edges:
            edge_id = reverse_undirected_id
        else:
            raise EdgeNotFoundError(f"Edge from '{from_node}' to '{to_node}' not found")

        edge = self._edges[edge_id]

        # Check if control_type is changing (only matters for directed edges)
        old_control_type = edge.properties.get("control_type")
        new_control_type = properties.get("control_type")
        control_type_changed = (
            edge.directed and
            new_control_type is not None and
            old_control_type != new_control_type
        )

        # Merge properties
        edge.properties.update(properties)

        # Mark subordinate's prompt dirty if control type changed
        if control_type_changed:
            subordinate = self._nodes.get(edge.to_node)
            if subordinate:
                subordinate.prompt_dirty = True
                logger.debug(
                    f"Marked subordinate '{edge.to_node}' prompt dirty "
                    f"(control_type changed from '{old_control_type}' to '{new_control_type}')"
                )

        logger.info(f"Updated edge from '{from_node}' to '{to_node}' properties")
        logger.debug(f"Edge properties: {edge.properties}")

    # ==================== Checkpoint Operations (Epic 7) ====================

    def save_checkpoint(self, filepath: Optional[Path | str] = None) -> Path:
        """
        Save graph state to a checkpoint file.

        Creates a checkpoint containing all nodes, edges, and metadata. Checkpoint
        includes an integrity checksum for validation.

        Args:
            filepath: Path where checkpoint should be saved. If None, uses
                     checkpoint_dir/{timestamp}.msgpack

        Returns:
            Path to saved checkpoint file

        Raises:
            CheckpointError: If save fails
        """
        if filepath is None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            filepath = self.checkpoint_dir / f"checkpoint_{timestamp}.msgpack"
        else:
            filepath = Path(filepath)

        # Create checkpoint
        metadata = {
            "topology_constraint": self.topology_constraint,
            "max_nodes": self.max_nodes,
            "persistence_enabled": self.persistence_enabled,
        }

        checkpoint = Checkpoint(
            name=self.name,
            nodes=self._nodes,
            edges=self._edges,
            metadata=metadata,
        )

        # Save checkpoint
        checkpoint.save(filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        return filepath

    @classmethod
    def load_checkpoint(cls, filepath: Path | str) -> "AgentGraph":
        """
        Load graph from a checkpoint file.

        Recreates a graph instance from a checkpoint, including all nodes and edges.
        Restores agent sessions through AgentSessionManager.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Reconstructed AgentGraph instance with all state restored

        Raises:
            CheckpointError: If load or validation fails
        """
        filepath = Path(filepath)

        # Load checkpoint
        checkpoint = Checkpoint.load(filepath)

        # Create graph with metadata
        graph = cls(
            name=checkpoint.name,
            max_nodes=checkpoint.metadata.get("max_nodes", 10000),
            persistence_enabled=checkpoint.metadata.get("persistence_enabled", True),
            topology_constraint=checkpoint.metadata.get("topology_constraint"),
        )

        # Restore nodes
        for node_id, node in checkpoint.nodes.items():
            graph._nodes[node_id] = node
            graph._nx_graph.add_node(node_id)
            logger.debug(f"Restored node '{node_id}'")

        # Restore edges
        for edge_id, edge in checkpoint.edges.items():
            graph._edges[edge_id] = edge
            graph._adjacency[edge.from_node].append(edge.to_node)
            graph._nx_graph.add_edge(edge.from_node, edge.to_node)

            if not edge.directed:
                graph._adjacency[edge.to_node].append(edge.from_node)
                graph._nx_graph.add_edge(edge.to_node, edge.from_node)

            logger.debug(f"Restored edge '{edge_id}'")

        logger.info(f"Loaded checkpoint from {filepath} - "
                   f"restored {len(graph._nodes)} nodes and {len(graph._edges)} edges")
        return graph

    async def _auto_save_worker(self) -> None:
        """
        Background task that periodically saves checkpoints.

        Runs periodically based on auto_save_interval and saves the current
        graph state. Intended to run in background indefinitely until cancelled.

        Raises:
            asyncio.CancelledError: When task is cancelled
        """
        try:
            while True:
                await asyncio.sleep(self.auto_save_interval)
                try:
                    self.save_checkpoint()
                except CheckpointError as e:
                    logger.error(f"Auto-save failed: {e}")
        except asyncio.CancelledError:
            logger.debug("Auto-save worker cancelled")
            raise

    def start_auto_save(self) -> None:
        """
        Start the background auto-save task.

        Creates and starts an asyncio task that periodically saves checkpoints.
        Safe to call multiple times - existing task will be cancelled first.

        Raises:
            RuntimeError: If no event loop is running
        """
        # Cancel existing task if any
        if self._auto_save_task is not None:
            self._auto_save_task.cancel()

        try:
            loop = asyncio.get_running_loop()
            self._auto_save_task = loop.create_task(self._auto_save_worker())
            logger.info(f"Started auto-save worker (interval: {self.auto_save_interval}s)")
        except RuntimeError as e:
            logger.error(f"Cannot start auto-save: {e}")
            raise

    def stop_auto_save(self) -> None:
        """
        Stop the background auto-save task.

        Safely cancels the auto-save task if it's running. Safe to call
        even if auto-save was never started.
        """
        if self._auto_save_task is not None:
            self._auto_save_task.cancel()
            self._auto_save_task = None
            logger.info("Stopped auto-save worker")

    async def load_latest_checkpoint(self) -> bool:
        """
        Load the most recent checkpoint if one exists.

        Searches checkpoint_dir for the newest checkpoint file and loads it.
        Useful for recovery on startup.

        Returns:
            True if checkpoint was loaded, False if none found

        Raises:
            CheckpointError: If loading fails
        """
        if not self.checkpoint_dir.exists():
            return False

        # Find newest checkpoint
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.msgpack"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not checkpoint_files:
            return False

        latest = checkpoint_files[0]
        logger.info(f"Found checkpoint: {latest}")

        # Note: load_checkpoint is a classmethod, so we can't use it directly
        # in recovery context. Instead, we'll manually restore state.
        checkpoint = Checkpoint.load(latest)

        # Restore nodes and edges into current instance
        self._nodes = checkpoint.nodes
        self._edges = checkpoint.edges

        # Rebuild NetworkX graph and adjacency
        self._nx_graph = nx.DiGraph()
        self._adjacency.clear()

        for node_id in self._nodes:
            self._nx_graph.add_node(node_id)

        for edge in self._edges.values():
            self._adjacency[edge.from_node].append(edge.to_node)
            self._nx_graph.add_edge(edge.from_node, edge.to_node)

            if not edge.directed:
                self._adjacency[edge.to_node].append(edge.from_node)
                self._nx_graph.add_edge(edge.to_node, edge.from_node)

        logger.info(f"Recovered graph state from {latest} - "
                   f"restored {len(self._nodes)} nodes and {len(self._edges)} edges")
        return True

