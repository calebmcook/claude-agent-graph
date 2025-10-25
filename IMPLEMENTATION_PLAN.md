# Implementation Plan: claude-agent-graph

**Version:** 1.0.0
**Date:** October 2025
**Based on:** PRD v1.0.0

## Overview

This document breaks down the Product Requirements Document (PRD) into actionable epics, features, and user stories for implementing the claude-agent-graph package. The plan is organized to deliver value incrementally while building a solid foundation.

## Implementation Strategy

### Principles
1. **Foundation First:** Build core infrastructure before advanced features
2. **Incremental Delivery:** Each epic delivers usable functionality
3. **Test-Driven:** Write tests alongside implementation
4. **Documentation Continuous:** Document APIs as they're built

### Dependency Order
```
Epic 1 (Foundation) → Epic 2 (Graph Construction) → Epic 3 (State Management)
→ Epic 4 (Dynamic Operations) → Epic 5 (Execution & Control)
→ Epic 6 (Persistence) → Epic 7 (Monitoring)
```

---

## Epic 1: Project Foundation & Core Infrastructure

**Goal:** Establish project structure, tooling, and basic data models

**Dependencies:** None

**Estimated Effort:** 2-3 days

### Feature 1.1: Project Setup & Tooling

#### Story 1.1.1: Initialize Python Package Structure
**As a** developer
**I want** a properly structured Python package
**So that** the codebase is organized and maintainable

**Acceptance Criteria:**
- [ ] Create package directory structure (`src/claude_agent_graph/`)
- [ ] Set up `pyproject.toml` with project metadata
- [ ] Configure build system (setuptools/hatchling)
- [ ] Create `README.md` with installation instructions
- [ ] Set up `.gitignore` for Python projects
- [ ] Initialize `__init__.py` files

**Files Created:**
- `pyproject.toml`
- `src/claude_agent_graph/__init__.py`
- `src/claude_agent_graph/py.typed`

#### Story 1.1.2: Configure Development Tools
**As a** developer
**I want** development tools configured
**So that** code quality is consistent

**Acceptance Criteria:**
- [ ] Set up `black` for code formatting
- [ ] Configure `ruff` for linting
- [ ] Set up `mypy` for type checking
- [ ] Configure `pytest` for testing
- [ ] Create `pytest.ini` or `pyproject.toml` test config
- [ ] Set up pre-commit hooks (optional)

**Files Created:**
- `pyproject.toml` (tool configurations)
- `.pre-commit-config.yaml` (optional)

#### Story 1.1.3: Set Up Dependency Management
**As a** developer
**I want** dependencies properly managed
**So that** installation is reliable

**Acceptance Criteria:**
- [ ] Add `claude-agent-sdk` to dependencies
- [ ] Add `anthropic` to dependencies
- [ ] Add `aiofiles` for async file I/O
- [ ] Add `networkx` for graph algorithms
- [ ] Add `pydantic` for data validation
- [ ] Specify version constraints
- [ ] Create `requirements-dev.txt` for dev dependencies

**Files Modified:**
- `pyproject.toml`

### Feature 1.2: Core Data Models

#### Story 1.2.1: Define Message Data Model
**As a** developer
**I want** a well-defined Message class
**So that** inter-agent communication is structured

**Acceptance Criteria:**
- [ ] Create `Message` dataclass with pydantic
- [ ] Include fields: message_id, timestamp, from_node, to_node, role, content, metadata
- [ ] Add validation for required fields
- [ ] Implement `to_dict()` and `from_dict()` methods
- [ ] Add type hints for all fields
- [ ] Write unit tests for Message class

**Files Created:**
- `src/claude_agent_graph/models.py`
- `tests/test_models.py`

#### Story 1.2.2: Define Node Data Model
**As a** developer
**I want** a Node class to represent agents
**So that** agent properties are well-structured

**Acceptance Criteria:**
- [ ] Create `Node` class with node_id, system_prompt, model, metadata
- [ ] Add `agent_session` placeholder (ClaudeSDKClient)
- [ ] Include `created_at` timestamp
- [ ] Add `status` enum (INITIALIZING, ACTIVE, STOPPED, ERROR)
- [ ] Implement validation for node_id uniqueness
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/models.py`
- `tests/test_models.py`

#### Story 1.2.3: Define Edge Data Model
**As a** developer
**I want** an Edge class to represent connections
**So that** agent relationships are well-structured

**Acceptance Criteria:**
- [ ] Create `Edge` class with edge_id, from_node, to_node, directed
- [ ] Add `properties` dict for custom metadata
- [ ] Include reference to SharedState
- [ ] Add `created_at` timestamp
- [ ] Implement edge_id generation (e.g., "nodeA_nodeB")
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/models.py`
- `tests/test_models.py`

#### Story 1.2.4: Define SharedState Data Model
**As a** developer
**I want** a SharedState class for edge state
**So that** conversation state is managed consistently

**Acceptance Criteria:**
- [ ] Create `SharedState` class with conversation_file path
- [ ] Add `metadata` dict for custom data
- [ ] Include initialization logic
- [ ] Add validation for file path
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/models.py`
- `tests/test_models.py`

---

## Epic 2: Graph Construction & Topology

**Goal:** Enable creation and management of graph structure

**Dependencies:** Epic 1

**Estimated Effort:** 5-7 days

### Feature 2.1: AgentGraph Core Class

#### Story 2.1.1: Create AgentGraph Class Skeleton
**As a** developer
**I want** a main AgentGraph class
**So that** users can create and manage agent graphs

**Acceptance Criteria:**
- [ ] Create `AgentGraph` class in `src/claude_agent_graph/graph.py`
- [ ] Add `__init__` with name, max_nodes, persistence_enabled parameters
- [ ] Initialize internal data structures (nodes dict, edges dict)
- [ ] Add `name` property
- [ ] Implement `__repr__` for debugging
- [ ] Write basic unit tests

**Files Created:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 2.1.2: Implement Node Addition
**As a** user
**I want** to add nodes to the graph
**So that** I can create agents

**Acceptance Criteria:**
- [ ] Implement `add_node(node_id, system_prompt, model, **metadata)` method
- [ ] Validate node_id uniqueness
- [ ] Validate system_prompt (non-empty, max length)
- [ ] Validate model name
- [ ] Create Node instance and store in graph
- [ ] Return Node object
- [ ] Raise appropriate exceptions for validation failures
- [ ] Write comprehensive unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 2.1.3: Implement Edge Addition
**As a** user
**I want** to add edges between nodes
**So that** agents can communicate

**Acceptance Criteria:**
- [ ] Implement `add_edge(from_node, to_node, directed, **properties)` method
- [ ] Validate both nodes exist
- [ ] Check for duplicate edges (configurable)
- [ ] Create Edge instance with unique edge_id
- [ ] Initialize SharedState for edge
- [ ] Store edge in graph
- [ ] Update node adjacency information
- [ ] Return Edge object
- [ ] Write comprehensive unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 2.1.4: Implement Graph Query Methods
**As a** user
**I want** to query graph structure
**So that** I can inspect nodes and edges

**Acceptance Criteria:**
- [ ] Implement `get_node(node_id)` method
- [ ] Implement `get_edge(from_node, to_node)` method
- [ ] Implement `get_nodes()` to list all nodes
- [ ] Implement `get_edges()` to list all edges
- [ ] Implement `get_neighbors(node_id, direction="both")` method
- [ ] Add `node_count` and `edge_count` properties
- [ ] Write unit tests for all methods

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

### Feature 2.2: Topology Support & Validation

#### Story 2.2.1: Implement Topology Detection
**As a** developer
**I want** to detect graph topology type
**So that** validation rules can be applied

**Acceptance Criteria:**
- [ ] Create `topology.py` module
- [ ] Implement `is_tree()` function using networkx
- [ ] Implement `is_dag()` function
- [ ] Implement `has_cycles()` function
- [ ] Implement `is_connected()` function
- [ ] Create `GraphTopology` enum (TREE, DAG, MESH, CHAIN, STAR, CYCLE, UNKNOWN)
- [ ] Write unit tests with various graph structures

**Files Created:**
- `src/claude_agent_graph/topology.py`
- `tests/test_topology.py`

#### Story 2.2.2: Add Topology Validation to Graph
**As a** user
**I want** topology validation
**So that** invalid graph structures are prevented

**Acceptance Criteria:**
- [ ] Add `get_topology()` method to AgentGraph
- [ ] Implement `validate_topology(required_type)` method
- [ ] Add optional topology constraints to `__init__`
- [ ] Validate on edge addition if constraints are set
- [ ] Provide helpful error messages for violations
- [ ] Write unit tests for validation

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 2.2.3: Implement Isolated Node Detection
**As a** user
**I want** warnings for isolated nodes
**So that** I can ensure proper graph connectivity

**Acceptance Criteria:**
- [ ] Implement `get_isolated_nodes()` method
- [ ] Add configurable warning on node isolation
- [ ] Implement logging for warnings
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

---

## Epic 3: State Management & Conversation Files

**Goal:** Implement shared state and conversation logging between agents

**Dependencies:** Epic 2

**Estimated Effort:** 5-6 days

### Feature 3.1: Conversation File Management

#### Story 3.1.1: Implement JSONL Conversation Writer
**As a** developer
**I want** thread-safe JSONL file writing
**So that** conversations are persisted reliably

**Acceptance Criteria:**
- [ ] Create `storage.py` module
- [ ] Implement `ConversationFile` class
- [ ] Add async `append()` method using aiofiles
- [ ] Implement file locking for thread safety
- [ ] Generate unique message IDs
- [ ] Add timestamp with microsecond precision
- [ ] Handle file creation and directory initialization
- [ ] Write unit tests with concurrent writes

**Files Created:**
- `src/claude_agent_graph/storage.py`
- `tests/test_storage.py`

#### Story 3.1.2: Implement Conversation Reader
**As a** developer
**I want** to read conversation history
**So that** agents can access shared context

**Acceptance Criteria:**
- [ ] Implement async `read()` method in ConversationFile
- [ ] Support filtering by timestamp (`since`)
- [ ] Support limiting results (`limit`)
- [ ] Parse JSONL and convert to Message objects
- [ ] Handle malformed lines gracefully
- [ ] Implement efficient reading (don't load entire file if limit set)
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/storage.py`
- `tests/test_storage.py`

#### Story 3.1.3: Implement Log Rotation
**As a** developer
**I want** automatic log rotation
**So that** conversation files don't grow unbounded

**Acceptance Criteria:**
- [ ] Add configurable max file size
- [ ] Implement rotation when size exceeded
- [ ] Archive old files with timestamp suffix
- [ ] Update SharedState to track current active file
- [ ] Add method to read across rotated files
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/storage.py`
- `tests/test_storage.py`

### Feature 3.2: Message Routing & Delivery

#### Story 3.2.1: Implement Message Sending
**As a** user
**I want** to send messages between agents
**So that** they can communicate

**Acceptance Criteria:**
- [ ] Implement `send_message(from_node, to_node, content, **metadata)` in AgentGraph
- [ ] Validate both nodes exist
- [ ] Validate edge exists (or create if undirected graph allows)
- [ ] Create Message object
- [ ] Append to conversation file
- [ ] Emit message sent event (for future monitoring)
- [ ] Return Message object
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 3.2.2: Implement Conversation Retrieval
**As a** user
**I want** to retrieve conversation history
**So that** I can review agent interactions

**Acceptance Criteria:**
- [ ] Implement `get_conversation(edge_id, since=None, limit=None)` method
- [ ] Support edge_id or (from_node, to_node) parameters
- [ ] Return list of Message objects
- [ ] Handle non-existent edges gracefully
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 3.2.3: Implement Recent Messages Helper
**As a** user
**I want** to get recent messages easily
**So that** I can quickly check latest interactions

**Acceptance Criteria:**
- [ ] Implement `get_recent_messages(edge_id, count=10)` method
- [ ] Return most recent N messages
- [ ] Order by timestamp descending
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

### Feature 3.3: Storage Backend Abstraction

#### Story 3.3.1: Define Storage Backend Interface
**As a** developer
**I want** a storage backend interface
**So that** multiple backends can be supported

**Acceptance Criteria:**
- [ ] Create `StorageBackend` abstract base class
- [ ] Define abstract methods: save_graph, load_graph, append_message, read_messages
- [ ] Add type hints for all methods
- [ ] Document interface in docstrings
- [ ] Write interface compliance tests

**Files Created:**
- `src/claude_agent_graph/backends/__init__.py`
- `src/claude_agent_graph/backends/base.py`

#### Story 3.3.2: Implement Filesystem Backend
**As a** developer
**I want** a filesystem storage backend
**So that** graphs can be persisted to disk

**Acceptance Criteria:**
- [ ] Create `FilesystemBackend` class
- [ ] Implement all StorageBackend methods
- [ ] Store graph metadata as JSON
- [ ] Store conversations as JSONL files
- [ ] Create directory structure (e.g., `graphs/{graph_name}/`)
- [ ] Handle file I/O errors gracefully
- [ ] Write unit tests

**Files Created:**
- `src/claude_agent_graph/backends/filesystem.py`
- `tests/backends/test_filesystem.py`

#### Story 3.3.3: Integrate Storage Backend with AgentGraph
**As a** user
**I want** to specify storage backend
**So that** I can choose where data is persisted

**Acceptance Criteria:**
- [ ] Add `storage_backend` parameter to AgentGraph.__init__
- [ ] Default to FilesystemBackend
- [ ] Use backend for all persistence operations
- [ ] Add `storage` property to AgentGraph
- [ ] Write integration tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

---

## Epic 4: claude-agent-sdk Integration

**Goal:** Integrate with claude-agent-sdk to create actual agent sessions

**Dependencies:** Epic 3

**Estimated Effort:** 4-5 days

### Feature 4.1: Agent Session Management

#### Story 4.1.1: Initialize ClaudeSDKClient for Nodes
**As a** developer
**I want** to create agent sessions when nodes are added
**So that** agents can execute

**Acceptance Criteria:**
- [ ] Create `agent_manager.py` module
- [ ] Implement `AgentSessionManager` class
- [ ] Create ClaudeSDKClient instance in `add_node()`
- [ ] Configure with system_prompt and model
- [ ] Store session in Node object
- [ ] Handle initialization errors
- [ ] Add async context manager support
- [ ] Write unit tests (may need mocking)

**Files Created:**
- `src/claude_agent_graph/agent_manager.py`
- `tests/test_agent_manager.py`

#### Story 4.1.2: Implement Agent Session Lifecycle
**As a** developer
**I want** to manage agent lifecycle
**So that** resources are properly managed

**Acceptance Criteria:**
- [ ] Implement `start_agent(node_id)` method
- [ ] Implement `stop_agent(node_id)` method
- [ ] Add `restart_agent(node_id)` method
- [ ] Update Node status appropriately
- [ ] Handle cleanup on agent stop
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/agent_manager.py`
- `tests/test_agent_manager.py`

#### Story 4.1.3: Handle Agent Errors and Recovery
**As a** developer
**I want** error handling for agent failures
**So that** one agent failure doesn't crash the system

**Acceptance Criteria:**
- [ ] Wrap agent operations in try/except
- [ ] Set Node status to ERROR on failure
- [ ] Log errors with context
- [ ] Implement retry logic (configurable)
- [ ] Add `get_agent_status(node_id)` method
- [ ] Write unit tests with failure scenarios

**Files Modified:**
- `src/claude_agent_graph/agent_manager.py`
- `tests/test_agent_manager.py`

### Feature 4.2: Control Relationships

#### Story 4.2.1: Implement System Prompt Injection for Control
**As a** developer
**I want** controller node_id injected into subordinate prompts
**So that** control relationships are established

**Acceptance Criteria:**
- [ ] Detect directed edges when adding
- [ ] Modify subordinate's system prompt to include controller info
- [ ] Format: "You are agent_{node_id}. You report to agent_{controller_id}."
- [ ] Store original prompt separately
- [ ] Update prompt when edges are added/removed
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 4.2.2: Implement Controller Query Methods
**As a** user
**I want** to query control relationships
**So that** I can understand agent hierarchy

**Acceptance Criteria:**
- [ ] Implement `get_controllers(node_id)` method
- [ ] Implement `get_subordinates(node_id)` method
- [ ] Implement `is_controller(node_id, subordinate_id)` method
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

---

## Epic 5: Dynamic Graph Operations

**Goal:** Enable runtime modification of graph structure

**Dependencies:** Epic 4

**Estimated Effort:** 4-5 days

### Feature 5.1: Runtime Node Operations

#### Story 5.1.1: Implement Runtime Node Addition
**As a** user
**I want** to add nodes at runtime
**So that** the graph can grow dynamically

**Acceptance Criteria:**
- [ ] Implement `add_node_runtime()` method (alias or same as add_node)
- [ ] Ensure thread-safety for concurrent additions
- [ ] Emit node_added event
- [ ] Update topology metadata
- [ ] Write unit tests with concurrent operations

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 5.1.2: Implement Node Removal
**As a** user
**I want** to remove nodes
**So that** I can clean up unused agents

**Acceptance Criteria:**
- [ ] Implement `remove_node(node_id, cascade=True)` method
- [ ] Stop agent session gracefully
- [ ] Remove associated edges if cascade=True
- [ ] Archive conversation files (don't delete)
- [ ] Update affected nodes' system prompts
- [ ] Emit node_removed event
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 5.1.3: Implement Node Property Updates
**As a** user
**I want** to update node properties
**So that** I can modify agent behavior

**Acceptance Criteria:**
- [ ] Implement `update_node(node_id, system_prompt=None, metadata=None)` method
- [ ] Support hot-reload of system prompt
- [ ] Preserve existing properties if not specified
- [ ] Validate changes before applying
- [ ] Support rollback on failure
- [ ] Emit node_updated event
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

### Feature 5.2: Runtime Edge Operations

#### Story 5.2.1: Implement Runtime Edge Addition
**As a** user
**I want** to add edges at runtime
**So that** agent connections can be created dynamically

**Acceptance Criteria:**
- [ ] Implement `add_edge_runtime()` method (alias or same as add_edge)
- [ ] Initialize conversation file immediately
- [ ] Update control relationships if directed
- [ ] Emit edge_added event
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 5.2.2: Implement Edge Removal
**As a** user
**I want** to remove edges
**So that** I can sever agent connections

**Acceptance Criteria:**
- [ ] Implement `remove_edge(from_node, to_node)` method
- [ ] Archive conversation file
- [ ] Update control relationships
- [ ] Remove from graph edge list
- [ ] Emit edge_removed event
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 5.2.3: Implement Edge Property Updates
**As a** user
**I want** to update edge properties
**So that** I can modify connection metadata

**Acceptance Criteria:**
- [ ] Implement `update_edge(edge_id, properties)` method
- [ ] Merge with existing properties
- [ ] Validate edge exists
- [ ] Emit edge_updated event
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

### Feature 5.3: Transaction Safety

#### Story 5.3.1: Implement Operation Logging
**As a** developer
**I want** transaction logs for graph operations
**So that** changes can be replayed or rolled back

**Acceptance Criteria:**
- [ ] Create transaction log file
- [ ] Log all graph modification operations
- [ ] Include operation type, timestamp, parameters
- [ ] Write in append-only format
- [ ] Write unit tests

**Files Created:**
- `src/claude_agent_graph/transactions.py`
- `tests/test_transactions.py`

#### Story 5.3.2: Implement Rollback Support
**As a** developer
**I want** rollback capability
**So that** failed operations can be undone

**Acceptance Criteria:**
- [ ] Track operation state before modification
- [ ] Implement `rollback()` method
- [ ] Restore previous state on failure
- [ ] Write unit tests with failure scenarios

**Files Modified:**
- `src/claude_agent_graph/transactions.py`
- `tests/test_transactions.py`

---

## Epic 6: Agent Execution & Control

**Goal:** Implement execution modes and message routing patterns

**Dependencies:** Epic 4, Epic 5

**Estimated Effort:** 5-6 days

### Feature 6.1: Message Routing Patterns

#### Story 6.1.1: Implement Direct Message Routing
**As a** user
**I want** direct message delivery
**So that** agents can communicate point-to-point

**Acceptance Criteria:**
- [ ] Already implemented in Epic 3.2.1
- [ ] Ensure async delivery
- [ ] Add message queue per node
- [ ] Write additional tests for edge cases

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 6.1.2: Implement Broadcast Routing
**As a** user
**I want** to broadcast messages to neighbors
**So that** one agent can notify multiple agents

**Acceptance Criteria:**
- [ ] Implement `broadcast(from_node, content, include_incoming=False)` method
- [ ] Send to all outgoing edges
- [ ] Optionally include incoming edges
- [ ] Return list of sent messages
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 6.1.3: Implement Multi-hop Routing
**As a** user
**I want** to route messages through a path
**So that** indirect communication is possible

**Acceptance Criteria:**
- [ ] Implement `route_message(from_node, to_node, path)` method
- [ ] Validate path exists and is valid
- [ ] Send message through each hop
- [ ] Use networkx for path finding if path not specified
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

### Feature 6.2: Execution Modes

#### Story 6.2.1: Implement Reactive Mode
**As a** user
**I want** reactive execution mode
**So that** agents respond to incoming messages

**Acceptance Criteria:**
- [ ] Create `execution.py` module
- [ ] Implement `ReactiveExecutor` class
- [ ] Start event loop to process message queues
- [ ] Trigger agent responses to incoming messages
- [ ] Add `start(mode="reactive")` to AgentGraph
- [ ] Write unit tests

**Files Created:**
- `src/claude_agent_graph/execution.py`
- `tests/test_execution.py`

#### Story 6.2.2: Implement Manual Mode
**As a** user
**I want** manual execution control
**So that** I can orchestrate agents externally

**Acceptance Criteria:**
- [ ] Implement `ManualController` class
- [ ] Add `manual_control()` context manager to AgentGraph
- [ ] Implement `step(node_id)` method to execute one agent turn
- [ ] Add `step_all()` to execute all agents once
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/execution.py`
- `tests/test_execution.py`

#### Story 6.2.3: Implement Proactive Mode
**As a** user
**I want** proactive execution mode
**So that** agents can initiate conversations

**Acceptance Criteria:**
- [ ] Implement `ProactiveExecutor` class
- [ ] Allow agents to trigger periodically (configurable interval)
- [ ] Support agent-initiated messages
- [ ] Add `start(mode="proactive", interval=60)` support
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/execution.py`
- `tests/test_execution.py`

### Feature 6.3: Control Commands

#### Story 6.3.1: Implement Command Execution
**As a** user
**I want** to issue commands to subordinates
**So that** controllers can direct worker agents

**Acceptance Criteria:**
- [ ] Implement `execute_command(controller, subordinate, command, **params)` method
- [ ] Validate controller-subordinate relationship
- [ ] Format command as special message
- [ ] Log command execution
- [ ] Support command rejection (configurable)
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 6.3.2: Implement Command Authorization
**As a** developer
**I want** command authorization checks
**So that** only authorized agents can issue commands

**Acceptance Criteria:**
- [ ] Validate control relationship before command execution
- [ ] Add configurable authorization rules
- [ ] Log unauthorized attempts
- [ ] Raise exception for unauthorized commands
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

---

## Epic 7: Persistence & Recovery

**Goal:** Implement graph serialization and crash recovery

**Dependencies:** Epic 5

**Estimated Effort:** 3-4 days

### Feature 7.1: Graph Serialization

#### Story 7.1.1: Implement Graph State Export
**As a** user
**I want** to save graph state
**So that** I can persist and restore graphs

**Acceptance Criteria:**
- [ ] Implement `save_checkpoint(filepath)` method
- [ ] Serialize graph structure (nodes, edges)
- [ ] Include all metadata
- [ ] Store conversation file references
- [ ] Use versioned format
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 7.1.2: Implement Graph State Import
**As a** user
**I want** to load saved graphs
**So that** I can restore previous state

**Acceptance Criteria:**
- [ ] Implement `load_checkpoint(filepath)` class method
- [ ] Deserialize graph structure
- [ ] Recreate nodes and edges
- [ ] Restore agent sessions
- [ ] Validate checkpoint version
- [ ] Handle migration for old versions
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

### Feature 7.2: Crash Recovery

#### Story 7.2.1: Implement Auto-save
**As a** developer
**I want** automatic checkpointing
**So that** state is preserved regularly

**Acceptance Criteria:**
- [ ] Add auto-save configuration to AgentGraph.__init__
- [ ] Implement periodic checkpoint saves
- [ ] Trigger save after N operations
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 7.2.2: Implement Recovery on Startup
**As a** developer
**I want** automatic recovery on restart
**So that** graphs can resume after crashes

**Acceptance Criteria:**
- [ ] Detect existing checkpoint on init
- [ ] Optionally auto-load latest checkpoint
- [ ] Validate checkpoint integrity
- [ ] Log recovery actions
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

---

## Epic 8: Monitoring & Observability

**Goal:** Implement metrics, logging, and visualization

**Dependencies:** Epic 6

**Estimated Effort:** 3-4 days

### Feature 8.1: Metrics Collection

#### Story 8.1.1: Implement Graph Metrics
**As a** user
**I want** to view graph metrics
**So that** I can monitor system health

**Acceptance Criteria:**
- [ ] Implement `get_metrics()` method
- [ ] Return dict with: node_count, edge_count, message_count, active_conversations
- [ ] Add agent_utilization metrics
- [ ] Include error_rate metrics
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/graph.py`
- `tests/test_graph.py`

#### Story 8.1.2: Implement Event Logging
**As a** developer
**I want** comprehensive event logging
**So that** system behavior can be audited

**Acceptance Criteria:**
- [ ] Set up Python logging in all modules
- [ ] Log graph structure changes
- [ ] Log message flow
- [ ] Log errors with full context
- [ ] Use structured logging format
- [ ] Write unit tests

**Files Modified:**
- All modules (add logging)
- `tests/test_logging.py`

### Feature 8.2: Visualization Export

#### Story 8.2.1: Implement GraphViz Export
**As a** user
**I want** to export graph visualization
**So that** I can visualize agent networks

**Acceptance Criteria:**
- [ ] Implement `export_visualization(format="graphviz", output_path)` method
- [ ] Generate DOT format output
- [ ] Include node labels with metadata
- [ ] Show edge direction
- [ ] Optionally include message counts
- [ ] Write unit tests

**Files Created:**
- `src/claude_agent_graph/visualization.py`
- `tests/test_visualization.py`

#### Story 8.2.2: Implement JSON Export
**As a** user
**I want** to export graph as JSON
**So that** I can use custom visualization tools

**Acceptance Criteria:**
- [ ] Support `format="json"` in export_visualization
- [ ] Generate standardized JSON format
- [ ] Include all graph data
- [ ] Make it compatible with common graph viz libraries
- [ ] Write unit tests

**Files Modified:**
- `src/claude_agent_graph/visualization.py`
- `tests/test_visualization.py`

---

## Epic 9: Documentation & Examples

**Goal:** Provide comprehensive documentation and examples

**Dependencies:** All previous epics

**Estimated Effort:** 3-4 days

### Feature 9.1: API Documentation

#### Story 9.1.1: Write API Reference Documentation
**As a** user
**I want** complete API documentation
**So that** I know how to use the package

**Acceptance Criteria:**
- [ ] Add comprehensive docstrings to all public methods
- [ ] Use Google or NumPy docstring style
- [ ] Include parameter types and return types
- [ ] Add usage examples in docstrings
- [ ] Set up Sphinx or mkdocs
- [ ] Generate HTML documentation

**Files Created:**
- `docs/` directory with all documentation
- `docs/api/` for API reference

#### Story 9.1.2: Write User Guide
**As a** user
**I want** a user guide
**So that** I can learn the package quickly

**Acceptance Criteria:**
- [ ] Write getting started guide
- [ ] Document common use cases
- [ ] Explain core concepts
- [ ] Add troubleshooting section
- [ ] Include best practices

**Files Created:**
- `docs/user_guide.md`
- `docs/getting_started.md`
- `docs/concepts.md`

### Feature 9.2: Code Examples

#### Story 9.2.1: Create Example Scripts
**As a** user
**I want** example scripts
**So that** I can see practical usage

**Acceptance Criteria:**
- [ ] Create examples/ directory
- [ ] Write simple hierarchy example (from PRD)
- [ ] Write collaborative network example (from PRD)
- [ ] Write dynamic workflow example (from PRD)
- [ ] Add comments explaining each step
- [ ] Ensure all examples run correctly

**Files Created:**
- `examples/simple_hierarchy.py`
- `examples/collaborative_network.py`
- `examples/dynamic_workflow.py`
- `examples/README.md`

#### Story 9.2.2: Create Tutorial Notebooks
**As a** user
**I want** interactive tutorials
**So that** I can learn by doing

**Acceptance Criteria:**
- [ ] Create Jupyter notebook tutorials
- [ ] Cover basic graph creation
- [ ] Cover agent communication
- [ ] Cover dynamic modifications
- [ ] Add to examples/ directory

**Files Created:**
- `examples/tutorial_01_basics.ipynb`
- `examples/tutorial_02_communication.ipynb`
- `examples/tutorial_03_advanced.ipynb`

### Feature 9.3: Testing & Quality

#### Story 9.3.1: Achieve 80%+ Test Coverage
**As a** developer
**I want** comprehensive test coverage
**So that** the package is reliable

**Acceptance Criteria:**
- [ ] Run coverage report
- [ ] Identify uncovered code
- [ ] Write tests for uncovered areas
- [ ] Achieve >80% coverage
- [ ] Set up coverage reporting in CI

**Files Modified:**
- Various test files

#### Story 9.3.2: Set Up CI/CD Pipeline
**As a** developer
**I want** automated testing and deployment
**So that** quality is maintained

**Acceptance Criteria:**
- [ ] Create GitHub Actions workflow
- [ ] Run tests on push/PR
- [ ] Run linting (ruff, black, mypy)
- [ ] Generate coverage report
- [ ] Set up PyPI publishing workflow

**Files Created:**
- `.github/workflows/test.yml`
- `.github/workflows/publish.yml`

---

## Implementation Priority & Sequencing

### Phase 1: MVP (Minimum Viable Product)
**Goal:** Basic working graph with agents

**Epics:** 1, 2, 3, 4

**Delivers:**
- Project structure and tooling
- Graph creation (add nodes, add edges)
- Conversation logging
- Basic agent sessions
- Simple examples work

**Timeline:** ~2-3 weeks

### Phase 2: Dynamic Operations
**Goal:** Runtime graph modifications

**Epics:** 5

**Delivers:**
- Add/remove nodes and edges at runtime
- Update node/edge properties
- Transaction safety

**Timeline:** ~1 week

### Phase 3: Advanced Execution
**Goal:** Multiple execution modes and routing

**Epics:** 6

**Delivers:**
- Reactive, proactive, manual modes
- Broadcast and multi-hop routing
- Control commands

**Timeline:** ~1 week

### Phase 4: Production Readiness
**Goal:** Persistence, monitoring, documentation

**Epics:** 7, 8, 9

**Delivers:**
- Save/load checkpoints
- Crash recovery
- Metrics and monitoring
- Complete documentation
- CI/CD pipeline

**Timeline:** ~1.5 weeks

---

## Total Estimated Timeline

**Total Effort:** 6-8 weeks for full implementation

**Phases:**
1. MVP: Weeks 1-3
2. Dynamic Operations: Week 4
3. Advanced Execution: Week 5
4. Production Readiness: Weeks 6-7.5

---

## Success Criteria per Epic

### Epic 1: Foundation
- [x] Project builds successfully
- [x] All dev tools configured
- [x] Core data models defined
- [x] Unit tests pass

### Epic 2: Graph Construction
- [x] Can create graph with nodes and edges
- [x] Topology detection works
- [x] Validation prevents invalid structures

### Epic 3: State Management
- [x] Messages persist to JSONL files
- [x] Conversation history can be retrieved
- [x] Thread-safe concurrent writes

### Epic 4: Agent Integration
- [x] Agent sessions created successfully
- [x] Control relationships established
- [x] Error handling works

### Epic 5: Dynamic Operations
- [x] Nodes/edges can be added/removed at runtime
- [x] Properties can be updated
- [x] Rollback works on failures

### Epic 6: Execution
- [x] All execution modes work
- [x] Message routing patterns implemented
- [x] Commands execute successfully

### Epic 7: Persistence
- [x] Checkpoints save/load correctly
- [x] Crash recovery works

### Epic 8: Monitoring
- [x] Metrics are accurate
- [x] Visualization exports work
- [x] Logging is comprehensive

### Epic 9: Documentation
- [x] API docs complete
- [x] Examples run successfully
- [x] >80% test coverage
- [x] CI/CD pipeline working

---

## Risk & Mitigation

### Risk 1: claude-agent-sdk API Changes
**Impact:** High
**Probability:** Medium
**Mitigation:** Pin specific SDK version, monitor for updates, maintain compatibility layer

### Risk 2: Performance at Scale
**Impact:** High
**Probability:** Medium
**Mitigation:** Early performance testing, profiling, optimization as needed

### Risk 3: Concurrency Issues
**Impact:** High
**Probability:** Medium
**Mitigation:** Extensive testing with concurrent operations, proper locking mechanisms

### Risk 4: File I/O Bottlenecks
**Impact:** Medium
**Probability:** Medium
**Mitigation:** Async I/O, buffering, optional database backend

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Set up project structure** (Epic 1, Story 1.1.1)
3. **Begin implementation** following epic order
4. **Regular check-ins** after each epic completion
5. **Iterate** based on feedback and learnings

---

**Document Status:** Draft
**Last Updated:** October 2025
**Author:** Implementation Team
